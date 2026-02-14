#!/usr/bin/env python3
"""
ECVT Strategy — Real Data Evaluation
=====================================
Downloads real EURUSD data and runs the Entropy Collapse Volatility Timing
strategy with walk-forward validation.

Data sources:
  - Hourly: yfinance EURUSD=X (2 years, ~12k bars)
  - Tick:   Dukascopy free tick feed (selected periods)
"""

import sys
import os
import json
import struct
import lzma
import urllib.request
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

# Add parent for ecvt_fast (optimized version)
sys.path.insert(0, str(Path(__file__).parent))
from ecvt_fast import (
    ECVTParams, generate_signals, run_backtest, 
    print_results, BacktestResult, 
    states_from_ohlcv, compute_markov_entropy_fast as compute_markov_entropy
)

DATA_DIR = Path(__file__).parent.parent / "curupira-backtests" / "data"
RESULTS_FILE = Path(__file__).parent / "ecvt_real_data_results.md"


# =============================================================================
# DATA ACQUISITION
# =============================================================================

def load_hourly_data() -> pd.DataFrame:
    """Load hourly EURUSD data from CSV."""
    path = DATA_DIR / "eurusd_hourly.csv"
    if not path.exists():
        raise FileNotFoundError(f"Hourly data not found at {path}")
    
    df = pd.read_csv(path, parse_dates=['timestamp'])
    print(f"Loaded hourly data: {len(df)} bars, {df['timestamp'].min()} to {df['timestamp'].max()}")
    return df


def load_5min_data() -> pd.DataFrame:
    """Load 5-minute EURUSD data from CSV."""
    path = DATA_DIR / "eurusd_5min.csv"
    if not path.exists():
        raise FileNotFoundError(f"5-min data not found at {path}")
    
    df = pd.read_csv(path, parse_dates=['timestamp'])
    print(f"Loaded 5-min data: {len(df)} bars, {df['timestamp'].min()} to {df['timestamp'].max()}")
    return df


def download_dukascopy_hour(symbol: str, year: int, month: int, day: int, hour: int) -> List[dict]:
    """Download one hour of tick data from Dukascopy."""
    month_idx = month - 1  # 0-indexed
    url = f"https://datafeed.dukascopy.com/datafeed/{symbol}/{year}/{month_idx:02d}/{day:02d}/{hour:02d}h_ticks.bi5"
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req, timeout=15)
        data = response.read()
        if len(data) == 0:
            return []
        
        decompressed = lzma.decompress(data)
        ticks = []
        n_ticks = len(decompressed) // 20
        base_time = datetime(year, month, day, hour)
        
        pipette_divisor = 100000.0  # EURUSD is 5-digit
        
        for i in range(n_ticks):
            offset = i * 20
            time_ms, ask_raw, bid_raw, ask_vol, bid_vol = struct.unpack(
                '>IIIff', decompressed[offset:offset+20]
            )
            tick_time = base_time + timedelta(milliseconds=time_ms)
            ticks.append({
                'timestamp': tick_time,
                'bid': bid_raw / pipette_divisor,
                'ask': ask_raw / pipette_divisor,
                'volume': ask_vol + bid_vol
            })
        return ticks
    except Exception as e:
        return []


def download_tick_data_range(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    max_hours: int = 2000
) -> pd.DataFrame:
    """Download tick data for a date range from Dukascopy using concurrent requests."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    # Build list of (year, month, day, hour) to download
    tasks = []
    current = start_date.replace(minute=0, second=0, microsecond=0)
    while current < end_date and len(tasks) < max_hours:
        if current.weekday() < 5:  # Skip weekends
            tasks.append((current.year, current.month, current.day, current.hour))
        current += timedelta(hours=1)
    
    print(f"Downloading {symbol} ticks: {start_date.date()} to {end_date.date()} ({len(tasks)} hours)")
    
    all_ticks = []
    hours_with_data = 0
    done = 0
    
    def fetch(args):
        y, m, d, h = args
        return download_dukascopy_hour(symbol, y, m, d, h)
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch, t): t for t in tasks}
        for future in as_completed(futures):
            ticks = future.result()
            done += 1
            if ticks:
                all_ticks.extend(ticks)
                hours_with_data += 1
            if done % 100 == 0:
                print(f"  ... {done}/{len(tasks)} hours, {hours_with_data} with data, {len(all_ticks)} ticks")
    
    print(f"Downloaded {len(all_ticks)} ticks from {hours_with_data} hours")
    
    if not all_ticks:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_ticks)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def aggregate_ticks_to_bars(tick_df: pd.DataFrame, freq: str = '1min') -> pd.DataFrame:
    """Aggregate tick data to OHLCV bars."""
    tick_df = tick_df.copy()
    tick_df['mid'] = (tick_df['bid'] + tick_df['ask']) / 2
    
    agg = tick_df.resample(freq, on='timestamp').agg({
        'mid': ['first', 'max', 'min', 'last'],
        'volume': 'sum'
    })
    agg.columns = ['open', 'high', 'low', 'close', 'volume']
    agg = agg.dropna(subset=['open'])
    agg = agg.reset_index()
    
    # Remove zero-volume bars (no ticks in that period)
    agg = agg[agg['volume'] > 0].reset_index(drop=True)
    
    return agg


def synthesize_volume_from_volatility(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    When real volume is unavailable, synthesize a proxy from:
    1. Bar range (high - low) as a fraction of close
    2. Absolute return
    This gives a volatility-based "activity" measure.
    """
    df = df.copy()
    
    if df['volume'].sum() == 0 or df['volume'].isna().all():
        print("  [INFO] Volume is zero — synthesizing from price activity")
        bar_range = (df['high'] - df['low']) / df['close']
        abs_return = df['close'].pct_change().abs().fillna(0)
        
        # Combine range and return, normalize to ~1000 mean
        activity = (bar_range + abs_return) / 2
        activity = activity.fillna(activity.median())
        
        if activity.mean() > 0:
            df['volume'] = (activity / activity.mean() * 1000).clip(lower=1)
        else:
            df['volume'] = 1000  # Uniform fallback
    
    return df


# =============================================================================
# WALK-FORWARD VALIDATION
# =============================================================================

@dataclass
class WalkForwardWindow:
    """Results for one walk-forward window."""
    window_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_bars: int
    test_bars: int
    # Test results
    num_trades: int = 0
    win_rate: float = 0.0
    total_pnl_bps: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_bps: float = 0.0
    sharpe_ratio: float = 0.0
    avg_win_bps: float = 0.0
    avg_loss_bps: float = 0.0
    exit_reasons: Dict[str, int] = None
    # Train results (for comparison)
    train_pnl_bps: float = 0.0
    train_win_rate: float = 0.0
    train_num_trades: int = 0


def run_walk_forward(
    df: pd.DataFrame,
    params: ECVTParams,
    train_months: int = 3,
    test_months: int = 1,
    step_months: int = 1,
    label: str = "WF"
) -> List[WalkForwardWindow]:
    """
    Run walk-forward validation.
    
    - Train on [t, t + train_months)
    - Test on [t + train_months, t + train_months + test_months)
    - Step forward by step_months
    - NO parameter optimization — we use fixed params throughout
    """
    results = []
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    start_date = df['timestamp'].min()
    end_date = df['timestamp'].max()
    
    window_id = 0
    train_start = start_date
    
    while True:
        train_end = train_start + pd.DateOffset(months=train_months)
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=test_months)
        
        if test_end > end_date:
            break
        
        # Split data
        train_mask = (df['timestamp'] >= train_start) & (df['timestamp'] < train_end)
        test_mask = (df['timestamp'] >= test_start) & (df['timestamp'] < test_end)
        
        df_train = df[train_mask].copy().reset_index(drop=True)
        df_test = df[test_mask].copy().reset_index(drop=True)
        
        if len(df_train) < params.entropy_lookback + params.entropy_window + 50:
            train_start += pd.DateOffset(months=step_months)
            continue
        
        if len(df_test) < 50:
            train_start += pd.DateOffset(months=step_months)
            continue
        
        # Generate signals on FULL data up to test end 
        # (entropy needs lookback context from training period)
        full_mask = (df['timestamp'] >= train_start) & (df['timestamp'] < test_end)
        df_full = df[full_mask].copy().reset_index(drop=True)
        
        try:
            df_signals = generate_signals(df_full, params)
        except Exception as e:
            print(f"  Window {window_id}: Signal generation failed: {e}")
            train_start += pd.DateOffset(months=step_months)
            continue
        
        # Split signals back into train/test
        train_len = len(df_train)
        df_train_signals = df_signals.iloc[:train_len].copy().reset_index(drop=True)
        df_test_signals = df_signals.iloc[train_len:].copy().reset_index(drop=True)
        
        # Run backtest on train (for reference)
        train_result = run_backtest(df_train_signals, params)
        
        # Run backtest on test (THE REAL RESULT)
        test_result = run_backtest(df_test_signals, params)
        
        # Collect exit reasons
        exit_reasons = {}
        for t in test_result.trades:
            exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1
        
        wf = WalkForwardWindow(
            window_id=window_id,
            train_start=str(train_start.date()),
            train_end=str(train_end.date()),
            test_start=str(test_start.date()),
            test_end=str(test_end.date()),
            train_bars=len(df_train),
            test_bars=len(df_test),
            num_trades=test_result.num_trades,
            win_rate=test_result.win_rate,
            total_pnl_bps=test_result.total_pnl_bps,
            profit_factor=test_result.profit_factor,
            max_drawdown_bps=test_result.max_drawdown_bps,
            sharpe_ratio=test_result.sharpe_ratio,
            avg_win_bps=test_result.avg_win_bps,
            avg_loss_bps=test_result.avg_loss_bps,
            exit_reasons=exit_reasons,
            train_pnl_bps=train_result.total_pnl_bps,
            train_win_rate=train_result.win_rate,
            train_num_trades=train_result.num_trades,
        )
        results.append(wf)
        
        print(f"  {label} Window {window_id}: "
              f"Train [{wf.train_start} → {wf.train_end}] "
              f"Test [{wf.test_start} → {wf.test_end}] "
              f"Trades={wf.num_trades} WR={wf.win_rate:.1%} "
              f"PnL={wf.total_pnl_bps:+.1f}bps PF={wf.profit_factor:.2f}")
        
        window_id += 1
        train_start += pd.DateOffset(months=step_months)
    
    return results


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def format_results_markdown(
    hourly_full: BacktestResult,
    hourly_wf: List[WalkForwardWindow],
    fivemin_full: Optional[BacktestResult],
    fivemin_wf: Optional[List[WalkForwardWindow]],
    tick_1min_full: Optional[BacktestResult],
    tick_1min_wf: Optional[List[WalkForwardWindow]],
    hourly_params: ECVTParams,
    fivemin_params: Optional[ECVTParams],
    tick_params: Optional[ECVTParams],
    hourly_df_info: Dict,
    fivemin_df_info: Optional[Dict],
    tick_df_info: Optional[Dict],
    entropy_stats: Dict,
) -> str:
    """Generate comprehensive Markdown results report."""
    
    lines = []
    lines.append("# ECVT Strategy — Real EURUSD Data Evaluation")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Strategy:** Entropy Collapse Volatility Timing (Singha 2025, arXiv:2512.15720)")
    lines.append(f"**Instrument:** EURUSD")
    lines.append("")
    
    # ---- Executive Summary ----
    lines.append("## Executive Summary")
    lines.append("")
    
    # Compute aggregate OOS stats
    if hourly_wf:
        oos_trades = sum(w.num_trades for w in hourly_wf)
        oos_pnl = sum(w.total_pnl_bps for w in hourly_wf)
        oos_wins = sum(w.num_trades * w.win_rate for w in hourly_wf)
        oos_wr = oos_wins / oos_trades if oos_trades > 0 else 0
        
        lines.append(f"**Hourly OOS (walk-forward aggregate):** "
                     f"{oos_trades} trades | PnL: {oos_pnl:+.1f} bps | "
                     f"Win rate: {oos_wr:.1%}")
    
    if fivemin_wf:
        oos5_trades = sum(w.num_trades for w in fivemin_wf)
        oos5_pnl = sum(w.total_pnl_bps for w in fivemin_wf)
        oos5_wins = sum(w.num_trades * w.win_rate for w in fivemin_wf)
        oos5_wr = oos5_wins / oos5_trades if oos5_trades > 0 else 0
        
        lines.append(f"**5-min OOS (walk-forward aggregate):** "
                     f"{oos5_trades} trades | PnL: {oos5_pnl:+.1f} bps | "
                     f"Win rate: {oos5_wr:.1%}")
    
    if tick_1min_wf:
        oost_trades = sum(w.num_trades for w in tick_1min_wf)
        oost_pnl = sum(w.total_pnl_bps for w in tick_1min_wf)
        oost_wins = sum(w.num_trades * w.win_rate for w in tick_1min_wf)
        oost_wr = oost_wins / oost_trades if oost_trades > 0 else 0
        
        lines.append(f"**Tick→1min OOS (walk-forward aggregate):** "
                     f"{oost_trades} trades | PnL: {oost_pnl:+.1f} bps | "
                     f"Win rate: {oost_wr:.1%}")
    
    lines.append("")
    
    # ---- Data Description ----
    lines.append("## Data Sources")
    lines.append("")
    lines.append("| Dataset | Source | Bars | Period | Volume |")
    lines.append("|---------|--------|------|--------|--------|")
    lines.append(f"| Hourly | yfinance EURUSD=X | {hourly_df_info['bars']} | "
                 f"{hourly_df_info['start']} → {hourly_df_info['end']} | "
                 f"{hourly_df_info['vol_type']} |")
    if fivemin_df_info:
        lines.append(f"| 5-min | yfinance EURUSD=X | {fivemin_df_info['bars']} | "
                     f"{fivemin_df_info['start']} → {fivemin_df_info['end']} | "
                     f"{fivemin_df_info['vol_type']} |")
    if tick_df_info:
        lines.append(f"| Tick→1min | Dukascopy | {tick_df_info['bars']} | "
                     f"{tick_df_info['start']} → {tick_df_info['end']} | "
                     f"{tick_df_info['vol_type']} |")
    lines.append("")
    
    # ---- Entropy Analysis ----
    lines.append("## Entropy Signal Analysis")
    lines.append("")
    for key, stats in entropy_stats.items():
        lines.append(f"### {key}")
        lines.append(f"- Mean entropy: {stats['mean']:.4f}")
        lines.append(f"- Std entropy: {stats['std']:.4f}")
        lines.append(f"- Min: {stats['min']:.4f} | Max: {stats['max']:.4f}")
        lines.append(f"- 5th percentile: {stats['p5']:.4f}")
        lines.append(f"- 10th percentile: {stats['p10']:.4f}")
        lines.append(f"- Valid values: {stats['valid']} / {stats['total']}")
        lines.append(f"- Signals generated: {stats['signals']}")
        lines.append("")
    
    # ---- Parameters ----
    lines.append("## Parameters Used")
    lines.append("")
    lines.append("### Hourly OHLCV")
    lines.append("```")
    for k, v in vars(hourly_params).items():
        lines.append(f"  {k}: {v}")
    lines.append("```")
    
    if fivemin_params:
        lines.append("\n### 5-Minute OHLCV")
        lines.append("```")
        for k, v in vars(fivemin_params).items():
            lines.append(f"  {k}: {v}")
        lines.append("```")
    
    if tick_params:
        lines.append("\n### Tick→1min")
        lines.append("```")
        for k, v in vars(tick_params).items():
            lines.append(f"  {k}: {v}")
        lines.append("```")
    lines.append("")
    
    # ---- Full-period results ----
    lines.append("## Full-Period Backtest Results")
    lines.append("")
    
    def result_table(result: BacktestResult, label: str) -> List[str]:
        rows = []
        rows.append(f"### {label}")
        rows.append("")
        rows.append(f"| Metric | Value |")
        rows.append(f"|--------|-------|")
        rows.append(f"| Total trades | {result.num_trades} |")
        rows.append(f"| Total PnL | {result.total_pnl_bps:+.1f} bps |")
        rows.append(f"| Win rate | {result.win_rate:.1%} |")
        rows.append(f"| Avg win | {result.avg_win_bps:+.1f} bps |")
        rows.append(f"| Avg loss | {result.avg_loss_bps:+.1f} bps |")
        rows.append(f"| Profit factor | {result.profit_factor:.2f} |")
        rows.append(f"| Max drawdown | {result.max_drawdown_bps:.1f} bps |")
        rows.append(f"| Sharpe (per-trade) | {result.sharpe_ratio:.3f} |")
        
        if result.num_trades > 0:
            exit_reasons = {}
            for t in result.trades:
                exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1
            rows.append(f"| **Exit reasons** | |")
            for reason, count in sorted(exit_reasons.items()):
                rows.append(f"| — {reason} | {count} ({count/result.num_trades:.0%}) |")
        
        rows.append("")
        return rows
    
    lines.extend(result_table(hourly_full, "Hourly OHLCV (Full Period)"))
    
    if fivemin_full:
        lines.extend(result_table(fivemin_full, "5-Minute OHLCV (Full Period)"))
    
    if tick_1min_full:
        lines.extend(result_table(tick_1min_full, "Tick→1min (Full Period)"))
    
    # ---- Walk-forward results ----
    lines.append("## Walk-Forward Validation (Out-of-Sample)")
    lines.append("")
    lines.append("**Methodology:** Fixed parameters (no optimization). "
                 "Train window provides entropy context. "
                 "Test window is pure out-of-sample.")
    lines.append("")
    
    def wf_table(windows: List[WalkForwardWindow], label: str) -> List[str]:
        rows = []
        rows.append(f"### {label}")
        rows.append("")
        rows.append("| Window | Test Period | Trades | Win Rate | PnL (bps) | PF | Max DD | Sharpe | Train PnL |")
        rows.append("|--------|-------------|--------|----------|-----------|-----|--------|--------|-----------|")
        
        for w in windows:
            rows.append(
                f"| {w.window_id} | {w.test_start} → {w.test_end} | "
                f"{w.num_trades} | {w.win_rate:.0%} | "
                f"{w.total_pnl_bps:+.1f} | {w.profit_factor:.2f} | "
                f"{w.max_drawdown_bps:.1f} | {w.sharpe_ratio:.3f} | "
                f"{w.train_pnl_bps:+.1f} |"
            )
        
        # Aggregate
        if windows:
            total_trades = sum(w.num_trades for w in windows)
            total_pnl = sum(w.total_pnl_bps for w in windows)
            avg_wr = np.mean([w.win_rate for w in windows if w.num_trades > 0]) if any(w.num_trades > 0 for w in windows) else 0
            avg_pf = np.mean([w.profit_factor for w in windows if w.num_trades > 0 and w.profit_factor < 100]) if any(w.num_trades > 0 for w in windows) else 0
            avg_sharpe = np.mean([w.sharpe_ratio for w in windows if w.num_trades > 0]) if any(w.num_trades > 0 for w in windows) else 0
            max_dd = max(w.max_drawdown_bps for w in windows) if windows else 0
            
            rows.append(
                f"| **AGG** | **{len(windows)} windows** | "
                f"**{total_trades}** | **{avg_wr:.0%}** | "
                f"**{total_pnl:+.1f}** | **{avg_pf:.2f}** | "
                f"**{max_dd:.1f}** | **{avg_sharpe:.3f}** | |"
            )
            
            # Consistency
            profitable_windows = sum(1 for w in windows if w.total_pnl_bps > 0)
            rows.append("")
            rows.append(f"**Consistency:** {profitable_windows}/{len(windows)} windows profitable "
                       f"({profitable_windows/len(windows):.0%})")
        
        rows.append("")
        return rows
    
    lines.extend(wf_table(hourly_wf, "Hourly Walk-Forward"))
    
    if fivemin_wf:
        lines.extend(wf_table(fivemin_wf, "5-Minute Walk-Forward"))
    
    if tick_1min_wf:
        lines.extend(wf_table(tick_1min_wf, "Tick→1min Walk-Forward"))
    
    # ---- Comparison ----
    lines.append("## Timeframe Comparison")
    lines.append("")
    lines.append("| Metric | Hourly | 5-min | Tick→1min |")
    lines.append("|--------|--------|-------|-----------|")
    
    def safe(v, fmt="+.1f"):
        return f"{v:{fmt}}" if v is not None else "N/A"
    
    h_oos_pnl = sum(w.total_pnl_bps for w in hourly_wf) if hourly_wf else None
    f_oos_pnl = sum(w.total_pnl_bps for w in fivemin_wf) if fivemin_wf else None
    t_oos_pnl = sum(w.total_pnl_bps for w in tick_1min_wf) if tick_1min_wf else None
    
    h_oos_trades = sum(w.num_trades for w in hourly_wf) if hourly_wf else None
    f_oos_trades = sum(w.num_trades for w in fivemin_wf) if fivemin_wf else None
    t_oos_trades = sum(w.num_trades for w in tick_1min_wf) if tick_1min_wf else None
    
    lines.append(f"| OOS Total PnL (bps) | {safe(h_oos_pnl)} | {safe(f_oos_pnl)} | {safe(t_oos_pnl)} |")
    lines.append(f"| OOS Total Trades | {safe(h_oos_trades, 'd') if h_oos_trades else 'N/A'} | "
                 f"{safe(f_oos_trades, 'd') if f_oos_trades else 'N/A'} | "
                 f"{safe(t_oos_trades, 'd') if t_oos_trades else 'N/A'} |")
    fivemin_pnl_str = f"{fivemin_full.total_pnl_bps:+.1f}" if fivemin_full else "N/A"
    tick_pnl_str = f"{tick_1min_full.total_pnl_bps:+.1f}" if tick_1min_full else "N/A"
    lines.append(f"| Full-period PnL (bps) | {hourly_full.total_pnl_bps:+.1f} | "
                 f"{fivemin_pnl_str} | "
                 f"{tick_pnl_str} |")
    lines.append("")
    
    # ---- Honest Assessment ----
    lines.append("## Honest Assessment")
    lines.append("")
    
    # Determine if profitable
    all_pnls = [h_oos_pnl]
    if f_oos_pnl is not None:
        all_pnls.append(f_oos_pnl)
    if t_oos_pnl is not None:
        all_pnls.append(t_oos_pnl)
    
    profitable_count = sum(1 for p in all_pnls if p is not None and p > 0)
    
    if profitable_count == len(all_pnls) and all(p > 50 for p in all_pnls if p is not None):
        lines.append("🟢 **PROMISING:** The entropy signal shows consistent profitability across "
                     "timeframes in out-of-sample testing. Worth further investigation with "
                     "live paper trading.")
    elif profitable_count > 0:
        lines.append("🟡 **MIXED:** The entropy signal shows some profitability but is inconsistent "
                     "across timeframes or walk-forward windows. The signal may be real but too "
                     "weak to trade profitably after costs.")
    else:
        lines.append("🔴 **NOT PROFITABLE:** The entropy signal does not survive contact with "
                     "real market data. The theoretical foundation from Singha (2025) may be "
                     "correct regarding entropy predicting move magnitude, but the practical "
                     "implementation tested here does not generate tradeable alpha.")
    
    lines.append("")
    lines.append("### Key Observations")
    lines.append("")
    
    # Auto-generate observations
    if hourly_wf:
        consistent = sum(1 for w in hourly_wf if w.total_pnl_bps > 0) / len(hourly_wf)
        lines.append(f"1. **Hourly consistency:** {consistent:.0%} of walk-forward windows are profitable")
        
        if hourly_full.num_trades > 0:
            sl_count = sum(1 for t in hourly_full.trades if t.exit_reason == 'stop_loss')
            tp_count = sum(1 for t in hourly_full.trades if t.exit_reason == 'take_profit')
            to_count = sum(1 for t in hourly_full.trades if t.exit_reason == 'timeout')
            total = hourly_full.num_trades
            lines.append(f"2. **Exit profile:** SL={sl_count/total:.0%} TP={tp_count/total:.0%} "
                         f"Timeout={to_count/total:.0%}")
            lines.append(f"3. **Asymmetry:** Avg win={hourly_full.avg_win_bps:+.1f}bps vs "
                         f"Avg loss={hourly_full.avg_loss_bps:+.1f}bps "
                         f"(ratio: {abs(hourly_full.avg_win_bps/hourly_full.avg_loss_bps) if hourly_full.avg_loss_bps != 0 else 'inf':.2f}x)")
    
    lines.append("")
    lines.append("### Caveats")
    lines.append("")
    lines.append("- yfinance forex data has **zero volume** — we synthesized a volatility-based proxy")
    lines.append("- Dukascopy tick data has real volume but limited coverage period")
    lines.append("- Spread/slippage beyond the 2bps flat cost is not modeled")
    lines.append("- The original paper (Singha 2025) used equity tick data, not forex")
    lines.append("- Walk-forward uses fixed parameters — adaptive params might perform differently")
    lines.append("")
    
    lines.append("---")
    lines.append(f"*Report generated by ECVT real-data evaluator, {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    
    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  ECVT STRATEGY — REAL EURUSD DATA EVALUATION")
    print("  Singha (2025) arXiv:2512.15720")
    print("=" * 70)
    
    entropy_stats = {}
    
    # =========================================================================
    # PART 1: HOURLY DATA (2 YEARS)
    # =========================================================================
    print("\n" + "=" * 70)
    print("  PART 1: HOURLY OHLCV (2 YEARS)")
    print("=" * 70)
    
    df_hourly = load_hourly_data()
    df_hourly['timestamp'] = pd.to_datetime(df_hourly['timestamp'], utc=True)
    df_hourly = synthesize_volume_from_volatility(df_hourly)
    
    hourly_df_info = {
        'bars': len(df_hourly),
        'start': str(df_hourly['timestamp'].min().date()),
        'end': str(df_hourly['timestamp'].max().date()),
        'vol_type': 'Synthesized (volatility proxy)'
    }
    
    # Hourly params — adapted from default for H1 timeframe
    hourly_params = ECVTParams(
        entropy_window=48,           # 2 days of hourly bars
        vol_quintile_window=48,
        entropy_percentile=5.0,      # Strict: bottom 5%
        entropy_lookback=500,        # ~21 trading days
        volume_percentile=90.0,
        volume_lookback=120,         # 5 trading days
        min_trail_return=0.001,      # 10 pips minimum
        max_trail_return=0.005,      # 50 pips maximum
        trail_return_window=12,      # 12 hours
        stop_loss_pct=0.0015,        # 15 pips on ~1.10
        take_profit_pct=0.005,       # 50 pips on ~1.10
        timeout_bars=24,             # 1 day timeout
        cost_per_trade_bps=3.0       # Conservative for hourly
    )
    
    print("\n[1.1] Generating signals on full hourly dataset...")
    df_hourly_signals = generate_signals(df_hourly, hourly_params)
    
    ent = df_hourly_signals['entropy'].dropna()
    n_signals = (df_hourly_signals['signal'] != 0).sum()
    print(f"  Entropy: mean={ent.mean():.4f}, std={ent.std():.4f}, "
          f"min={ent.min():.4f}, max={ent.max():.4f}")
    print(f"  Signals: {n_signals} raw signals")
    
    entropy_stats['Hourly'] = {
        'mean': float(ent.mean()),
        'std': float(ent.std()),
        'min': float(ent.min()),
        'max': float(ent.max()),
        'p5': float(ent.quantile(0.05)),
        'p10': float(ent.quantile(0.10)),
        'valid': int(ent.notna().sum()),
        'total': len(df_hourly_signals),
        'signals': int(n_signals),
    }
    
    print("\n[1.2] Running full-period backtest (hourly)...")
    hourly_full = run_backtest(df_hourly_signals, hourly_params)
    print_results(hourly_full, "Hourly OHLCV — Full Period (2 years)")
    
    print("\n[1.3] Running walk-forward validation (hourly)...")
    print("  Train=3mo, Test=1mo, Step=1mo")
    hourly_wf = run_walk_forward(
        df_hourly, hourly_params,
        train_months=3, test_months=1, step_months=1,
        label="H1"
    )
    
    # =========================================================================
    # PART 2: 5-MINUTE DATA (~80 days)
    # =========================================================================
    print("\n" + "=" * 70)
    print("  PART 2: 5-MINUTE OHLCV (~80 DAYS)")
    print("=" * 70)
    
    fivemin_full = None
    fivemin_wf = None
    fivemin_params = None
    fivemin_df_info = None
    
    try:
        df_5min = load_5min_data()
        df_5min['timestamp'] = pd.to_datetime(df_5min['timestamp'], utc=True)
        df_5min = synthesize_volume_from_volatility(df_5min)
        
        fivemin_df_info = {
            'bars': len(df_5min),
            'start': str(df_5min['timestamp'].min().date()),
            'end': str(df_5min['timestamp'].max().date()),
            'vol_type': 'Synthesized (volatility proxy)'
        }
        
        # 5-min params
        fivemin_params = ECVTParams(
            entropy_window=120,          # 10 hours of 5-min bars
            vol_quintile_window=120,
            entropy_percentile=5.0,
            entropy_lookback=1000,       # ~3.5 trading days
            volume_percentile=90.0,
            volume_lookback=288,         # 1 day of 5-min bars
            min_trail_return=0.0005,     # 5 pips
            max_trail_return=0.003,      # 30 pips
            trail_return_window=36,      # 3 hours of 5-min bars
            stop_loss_pct=0.001,         # 10 pips
            take_profit_pct=0.003,       # 30 pips
            timeout_bars=72,             # 6 hours
            cost_per_trade_bps=2.0
        )
        
        print("\n[2.1] Generating signals on 5-min dataset...")
        df_5min_signals = generate_signals(df_5min, fivemin_params)
        
        ent5 = df_5min_signals['entropy'].dropna()
        n_signals_5 = (df_5min_signals['signal'] != 0).sum()
        print(f"  Entropy: mean={ent5.mean():.4f}, std={ent5.std():.4f}")
        print(f"  Signals: {n_signals_5}")
        
        entropy_stats['5-Minute'] = {
            'mean': float(ent5.mean()),
            'std': float(ent5.std()),
            'min': float(ent5.min()),
            'max': float(ent5.max()),
            'p5': float(ent5.quantile(0.05)),
            'p10': float(ent5.quantile(0.10)),
            'valid': int(ent5.notna().sum()),
            'total': len(df_5min_signals),
            'signals': int(n_signals_5),
        }
        
        print("\n[2.2] Running full-period backtest (5-min)...")
        fivemin_full = run_backtest(df_5min_signals, fivemin_params)
        print_results(fivemin_full, "5-min OHLCV — Full Period")
        
        print("\n[2.3] Running walk-forward validation (5-min)...")
        # With ~80 days of data, use shorter windows: 30d train, 10d test
        fivemin_wf = run_walk_forward(
            df_5min, fivemin_params,
            train_months=1, test_months=1, step_months=1,
            label="M5"
        )
        
    except FileNotFoundError:
        print("  5-min data not available, skipping.")
    except Exception as e:
        print(f"  5-min analysis failed: {e}")
        import traceback; traceback.print_exc()
    
    # =========================================================================
    # PART 3: TICK DATA (DUKASCOPY)
    # =========================================================================
    print("\n" + "=" * 70)
    print("  PART 3: TICK DATA → 1-MINUTE BARS (DUKASCOPY)")
    print("=" * 70)
    
    tick_1min_full = None
    tick_1min_wf = None
    tick_params = None
    tick_df_info = None
    
    try:
        # Download ~6 weeks of tick data (reasonable scope for HTTP download)
        # ~720 trading hours = manageable requests
        tick_start = datetime(2025, 9, 1)
        tick_end = datetime(2025, 10, 15)  # 6 weeks
        
        tick_cache = DATA_DIR / "eurusd_ticks_dukascopy.parquet"
        
        if tick_cache.exists():
            print(f"  Loading cached tick data from {tick_cache}")
            df_ticks = pd.read_parquet(tick_cache)
        else:
            df_ticks = download_tick_data_range("EURUSD", tick_start, tick_end, max_hours=800)
            if not df_ticks.empty:
                os.makedirs(DATA_DIR, exist_ok=True)
                df_ticks.to_parquet(tick_cache, index=False)
                print(f"  Cached tick data to {tick_cache}")
        
        if not df_ticks.empty:
            print(f"  Ticks loaded: {len(df_ticks)} ticks, "
                  f"{df_ticks['timestamp'].min()} to {df_ticks['timestamp'].max()}")
            
            # Aggregate to 1-minute bars
            print("\n[3.1] Aggregating ticks to 1-minute bars...")
            df_tick_1min = aggregate_ticks_to_bars(df_ticks, freq='1min')
            print(f"  1-min bars: {len(df_tick_1min)}")
            
            tick_df_info = {
                'bars': len(df_tick_1min),
                'start': str(df_tick_1min['timestamp'].min().date()) if len(df_tick_1min) > 0 else 'N/A',
                'end': str(df_tick_1min['timestamp'].max().date()) if len(df_tick_1min) > 0 else 'N/A',
                'vol_type': 'Real (Dukascopy tick volume)'
            }
            
            if len(df_tick_1min) > 2000:
                # 1-min params (similar to 5-min but tighter)
                tick_params = ECVTParams(
                    entropy_window=120,          # 2 hours
                    vol_quintile_window=120,
                    entropy_percentile=5.0,
                    entropy_lookback=1000,
                    volume_percentile=90.0,
                    volume_lookback=360,         # 6 hours
                    min_trail_return=0.0003,     # 3 pips
                    max_trail_return=0.002,      # 20 pips
                    trail_return_window=60,      # 1 hour
                    stop_loss_pct=0.0008,        # 8 pips
                    take_profit_pct=0.0025,      # 25 pips
                    timeout_bars=120,            # 2 hours
                    cost_per_trade_bps=2.0
                )
                
                print("\n[3.2] Generating signals on tick→1min data...")
                df_tick_1min_signals = generate_signals(df_tick_1min, tick_params)
                
                ent_tick = df_tick_1min_signals['entropy'].dropna()
                n_signals_tick = (df_tick_1min_signals['signal'] != 0).sum()
                print(f"  Entropy: mean={ent_tick.mean():.4f}, std={ent_tick.std():.4f}")
                print(f"  Signals: {n_signals_tick}")
                
                entropy_stats['Tick→1min'] = {
                    'mean': float(ent_tick.mean()),
                    'std': float(ent_tick.std()),
                    'min': float(ent_tick.min()),
                    'max': float(ent_tick.max()),
                    'p5': float(ent_tick.quantile(0.05)),
                    'p10': float(ent_tick.quantile(0.10)),
                    'valid': int(ent_tick.notna().sum()),
                    'total': len(df_tick_1min_signals),
                    'signals': int(n_signals_tick),
                }
                
                print("\n[3.3] Running full-period backtest (tick→1min)...")
                tick_1min_full = run_backtest(df_tick_1min_signals, tick_params)
                print_results(tick_1min_full, "Tick→1min — Full Period")
                
                # Walk-forward on tick data (limited data — use 2-week windows)
                total_days = (df_tick_1min['timestamp'].max() - df_tick_1min['timestamp'].min()).days
                if total_days > 35:
                    print("\n[3.4] Running walk-forward validation (tick→1min)...")
                    tick_1min_wf = run_walk_forward(
                        df_tick_1min, tick_params,
                        train_months=1, test_months=1, step_months=1,
                        label="T1"
                    )
                else:
                    print(f"\n[3.4] Only {total_days} days of tick data — skipping walk-forward")
            else:
                print(f"  Only {len(df_tick_1min)} 1-min bars — too few for meaningful analysis")
        else:
            print("  No tick data downloaded. Skipping tick analysis.")
    
    except Exception as e:
        print(f"  Tick analysis failed: {e}")
        import traceback; traceback.print_exc()
    
    # =========================================================================
    # GENERATE REPORT
    # =========================================================================
    print("\n" + "=" * 70)
    print("  GENERATING REPORT")
    print("=" * 70)
    
    report = format_results_markdown(
        hourly_full=hourly_full,
        hourly_wf=hourly_wf,
        fivemin_full=fivemin_full,
        fivemin_wf=fivemin_wf,
        tick_1min_full=tick_1min_full,
        tick_1min_wf=tick_1min_wf,
        hourly_params=hourly_params,
        fivemin_params=fivemin_params,
        tick_params=tick_params,
        hourly_df_info=hourly_df_info,
        fivemin_df_info=fivemin_df_info,
        tick_df_info=tick_df_info,
        entropy_stats=entropy_stats,
    )
    
    RESULTS_FILE.write_text(report)
    print(f"\nResults saved to: {RESULTS_FILE}")
    print(f"Report length: {len(report)} characters")
    
    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70)


if __name__ == '__main__':
    main()
