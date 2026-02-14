#!/usr/bin/env python3
"""
ECVT + Hurst Combo Strategy
============================
Combines ECVT entropy collapse (timing) with Hurst exponent (direction/regime).

Logic:
  - ECVT signals when entropy drops to bottom percentile → big move imminent
  - Hurst exponent determines regime and direction:
    * H > 0.6: trending → follow EMA-20 slope
    * H < 0.4: mean-reverting → fade 10-bar momentum
    * 0.4-0.6: random walk → skip (no edge)
  - BOTH must agree: ECVT fires AND Hurst gives direction

Benchmark: Standalone ECVT hourly = 44 trades, +198 bps, PF 1.44, 41% WR
"""

import sys
import os
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd

# Import ECVT functions
sys.path.insert(0, str(Path(__file__).parent))
from ecvt_fast import (
    ECVTParams,
    states_from_ohlcv,
    compute_markov_entropy_fast,
    discretize_returns,
    compute_volume_quintiles_fast,
)


# =============================================================================
# HURST EXPONENT (adapted from hurst_signal.py, lowercase columns)
# =============================================================================

def hurst_rs(series: np.ndarray) -> float:
    """Hurst exponent via Rescaled Range (R/S) analysis."""
    N = len(series)
    if N < 20:
        return np.nan

    max_k = N // 2
    min_k = 8

    ks = []
    k = min_k
    while k <= max_k:
        ks.append(k)
        k = int(k * 1.5)
        if k == ks[-1]:
            k += 1

    if len(ks) < 3:
        return np.nan

    rs_values = []
    ns_values = []

    for k in ks:
        n_chunks = N // k
        if n_chunks < 1:
            continue

        rs_list = []
        for chunk_i in range(n_chunks):
            chunk = series[chunk_i * k:(chunk_i + 1) * k]
            mean_chunk = np.mean(chunk)
            deviations = chunk - mean_chunk
            cumulative = np.cumsum(deviations)
            R = np.max(cumulative) - np.min(cumulative)
            S = np.std(chunk, ddof=1)
            if S > 0:
                rs_list.append(R / S)

        if rs_list:
            rs_values.append(np.mean(rs_list))
            ns_values.append(k)

    if len(ns_values) < 3:
        return np.nan

    log_n = np.log(np.array(ns_values))
    log_rs = np.log(np.array(rs_values))

    try:
        coeffs = np.polyfit(log_n, log_rs, 1)
        H = max(0.0, min(1.0, coeffs[0]))
        return H
    except:
        return np.nan


def rolling_hurst(series: np.ndarray, window: int = 50) -> np.ndarray:
    """Rolling Hurst exponent on returns."""
    n = len(series)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        segment = series[i - window + 1:i + 1]
        if np.any(np.isnan(segment)):
            continue
        result[i] = hurst_rs(segment)
    return result


# =============================================================================
# VOLUME SYNTHESIS
# =============================================================================

def synthesize_volume(df: pd.DataFrame) -> pd.DataFrame:
    """Synthesize volume from price activity when real volume is zero."""
    df = df.copy()
    if df['volume'].sum() == 0 or df['volume'].isna().all():
        bar_range = (df['high'] - df['low']) / df['close']
        abs_return = df['close'].pct_change().abs().fillna(0)
        activity = (bar_range + abs_return) / 2
        activity = activity.fillna(activity.median())
        if activity.mean() > 0:
            df['volume'] = (activity / activity.mean() * 1000).clip(lower=1)
        else:
            df['volume'] = 1000
    return df


# =============================================================================
# COMBO PARAMETERS
# =============================================================================

@dataclass
class ComboParams:
    # ECVT params
    entropy_window: int = 48
    vol_quintile_window: int = 48
    entropy_percentile: float = 5.0
    entropy_lookback: int = 500
    volume_percentile: float = 90.0
    volume_lookback: int = 120
    min_trail_return: float = 0.001
    max_trail_return: float = 0.005
    trail_return_window: int = 12

    # Hurst params
    hurst_window: int = 50
    trending_threshold: float = 0.6
    mean_revert_threshold: float = 0.4
    ema_span: int = 20
    momentum_window: int = 10

    # ATR-based SL/TP
    atr_period: int = 14
    sl_atr_mult: float = 1.5
    tp_atr_mult: float = 3.0
    timeout_bars: int = 24

    # Cost
    cost_per_trade_bps: float = 3.0


# =============================================================================
# TRADE STRUCTURES
# =============================================================================

@dataclass
class Trade:
    entry_bar: int
    entry_time: object
    entry_price: float
    direction: int  # +1 long, -1 short
    stop_loss: float = 0.0
    take_profit: float = 0.0
    exit_bar: int = -1
    exit_time: object = None
    exit_price: float = 0.0
    exit_reason: str = ''
    pnl_bps: float = 0.0
    hurst_val: float = 0.0
    regime: str = ''


@dataclass
class BacktestResult:
    trades: List[Trade] = field(default_factory=list)
    total_pnl_bps: float = 0.0
    num_trades: int = 0
    win_rate: float = 0.0
    avg_win_bps: float = 0.0
    avg_loss_bps: float = 0.0
    max_drawdown_bps: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0


# =============================================================================
# SIGNAL GENERATION
# =============================================================================

def generate_combo_signals(df: pd.DataFrame, params: ComboParams) -> pd.DataFrame:
    """
    Generate combo signals:
    1. Compute ECVT entropy → flag bars where entropy < percentile threshold
    2. Compute rolling Hurst → determine regime
    3. ECVT fires + Hurst agrees → signal with Hurst-determined direction
    """
    df = df.copy()

    # --- ECVT entropy ---
    states = states_from_ohlcv(df, vol_window=params.vol_quintile_window)
    df['entropy'] = compute_markov_entropy_fast(
        states, window=params.entropy_window, n_states=15
    )
    df['entropy_threshold'] = df['entropy'].rolling(
        window=params.entropy_lookback,
        min_periods=params.entropy_lookback // 2,
    ).quantile(params.entropy_percentile / 100.0)

    df['volume_threshold'] = df['volume'].rolling(
        window=params.volume_lookback,
        min_periods=params.volume_lookback // 2,
    ).quantile(params.volume_percentile / 100.0)

    df['trail_return'] = df['close'].pct_change(params.trail_return_window)

    entropy_low = df['entropy'] < df['entropy_threshold']
    volume_high = df['volume'] > df['volume_threshold']
    trail_abs = df['trail_return'].abs()
    return_in_range = (trail_abs >= params.min_trail_return) & (trail_abs <= params.max_trail_return)

    ecvt_fires = entropy_low & volume_high & return_in_range
    
    # Store ECVT signal for analysis
    df['ecvt_signal'] = 0
    df.loc[ecvt_fires, 'ecvt_signal'] = 1

    # --- Hurst exponent ---
    log_ret = np.log(df['close'] / df['close'].shift(1)).fillna(0).values
    df['hurst'] = rolling_hurst(log_ret, window=params.hurst_window)

    # EMA slope for trending direction
    df['ema'] = df['close'].ewm(span=params.ema_span, adjust=False).mean()
    df['ema_slope'] = df['ema'] - df['ema'].shift(1)

    # Momentum for mean-reversion fade
    df['momentum'] = df['close'] - df['close'].shift(params.momentum_window)

    # ATR for SL/TP
    tr = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            (df['high'] - df['close'].shift(1)).abs(),
            (df['low'] - df['close'].shift(1)).abs(),
        )
    )
    df['atr'] = tr.rolling(params.atr_period).mean()

    # --- Combine ---
    df['signal'] = 0
    df['regime'] = ''

    for i in range(len(df)):
        if not ecvt_fires.iloc[i]:
            continue

        h = df['hurst'].iloc[i]
        if pd.isna(h) or pd.isna(df['atr'].iloc[i]):
            continue

        direction = 0
        regime = ''

        if h > params.trending_threshold:
            # Trending → follow EMA slope
            slope = df['ema_slope'].iloc[i]
            if pd.isna(slope) or slope == 0:
                continue
            direction = 1 if slope > 0 else -1
            regime = 'trending'
        elif h < params.mean_revert_threshold:
            # Mean-reverting → fade momentum
            mom = df['momentum'].iloc[i]
            if pd.isna(mom) or mom == 0:
                continue
            direction = -1 if mom > 0 else 1  # fade
            regime = 'mean_revert'
        else:
            # Random walk → skip
            continue

        df.iloc[i, df.columns.get_loc('signal')] = direction
        df.iloc[i, df.columns.get_loc('regime')] = regime

    return df


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

def run_backtest(df: pd.DataFrame, params: ComboParams) -> BacktestResult:
    """Run backtest with ATR-based SL/TP. Entry at NEXT bar open."""
    trades = []
    in_position = False
    current_trade = None

    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    signals = df['signal'].values
    atrs = df['atr'].values
    hursts = df['hurst'].values
    regimes = df['regime'].values
    timestamps = df['timestamp'].values if 'timestamp' in df.columns else np.arange(len(df))

    for i in range(1, len(df)):
        # --- Check exit for current position ---
        if in_position and current_trade is not None:
            direction = current_trade.direction
            bars_held = i - current_trade.entry_bar

            exited = False

            if direction == 1:  # long
                if lows[i] <= current_trade.stop_loss:
                    current_trade.exit_price = current_trade.stop_loss
                    current_trade.exit_reason = 'stop_loss'
                    exited = True
                elif highs[i] >= current_trade.take_profit:
                    current_trade.exit_price = current_trade.take_profit
                    current_trade.exit_reason = 'take_profit'
                    exited = True
            else:  # short
                if highs[i] >= current_trade.stop_loss:
                    current_trade.exit_price = current_trade.stop_loss
                    current_trade.exit_reason = 'stop_loss'
                    exited = True
                elif lows[i] <= current_trade.take_profit:
                    current_trade.exit_price = current_trade.take_profit
                    current_trade.exit_reason = 'take_profit'
                    exited = True

            if not exited and bars_held >= params.timeout_bars:
                current_trade.exit_price = closes[i]
                current_trade.exit_reason = 'timeout'
                exited = True

            if exited:
                current_trade.exit_bar = i
                current_trade.exit_time = timestamps[i]
                raw_pnl = direction * (current_trade.exit_price - current_trade.entry_price) / current_trade.entry_price
                current_trade.pnl_bps = raw_pnl * 10000 - params.cost_per_trade_bps
                trades.append(current_trade)
                in_position = False
                current_trade = None

        # --- Check for new entry (signal at i-1, enter at i open) ---
        if not in_position and signals[i - 1] != 0:
            direction = int(signals[i - 1])
            entry_price = opens[i]
            atr = atrs[i - 1]

            if pd.isna(atr) or atr <= 0:
                continue

            sl_dist = params.sl_atr_mult * atr
            tp_dist = params.tp_atr_mult * atr

            if direction == 1:
                sl = entry_price - sl_dist
                tp = entry_price + tp_dist
            else:
                sl = entry_price + sl_dist
                tp = entry_price - tp_dist

            current_trade = Trade(
                entry_bar=i,
                entry_time=timestamps[i],
                entry_price=entry_price,
                direction=direction,
                stop_loss=sl,
                take_profit=tp,
                hurst_val=float(hursts[i - 1]) if not pd.isna(hursts[i - 1]) else 0.0,
                regime=str(regimes[i - 1]),
            )
            in_position = True

    # Close open position at end
    if in_position and current_trade is not None:
        i = len(df) - 1
        current_trade.exit_bar = i
        current_trade.exit_time = timestamps[i]
        current_trade.exit_price = closes[i]
        current_trade.exit_reason = 'end_of_data'
        direction = current_trade.direction
        raw_pnl = direction * (current_trade.exit_price - current_trade.entry_price) / current_trade.entry_price
        current_trade.pnl_bps = raw_pnl * 10000 - params.cost_per_trade_bps
        trades.append(current_trade)

    # --- Compute stats ---
    result = BacktestResult()
    result.trades = trades
    result.num_trades = len(trades)

    if len(trades) == 0:
        return result

    pnls = np.array([t.pnl_bps for t in trades])
    result.total_pnl_bps = pnls.sum()
    result.win_rate = (pnls > 0).mean()

    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    result.avg_win_bps = float(wins.mean()) if len(wins) > 0 else 0.0
    result.avg_loss_bps = float(losses.mean()) if len(losses) > 0 else 0.0

    cumulative = np.cumsum(pnls)
    peak = np.maximum.accumulate(cumulative)
    drawdown = peak - cumulative
    result.max_drawdown_bps = float(drawdown.max()) if len(drawdown) > 0 else 0.0

    if pnls.std() > 0:
        result.sharpe_ratio = float(pnls.mean() / pnls.std())

    gross_profit = float(wins.sum()) if len(wins) > 0 else 0.0
    gross_loss = float(abs(losses.sum())) if len(losses) > 0 else 1.0
    result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    return result


# =============================================================================
# WALK-FORWARD VALIDATION
# =============================================================================

@dataclass
class WFWindow:
    window_id: int
    test_start: str
    test_end: str
    train_bars: int
    test_bars: int
    num_trades: int = 0
    win_rate: float = 0.0
    total_pnl_bps: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_bps: float = 0.0
    sharpe_ratio: float = 0.0
    regime_breakdown: Dict[str, int] = field(default_factory=dict)
    exit_breakdown: Dict[str, int] = field(default_factory=dict)


def run_walk_forward(
    df: pd.DataFrame,
    params: ComboParams,
    train_months: int = 3,
    test_months: int = 1,
    step_months: int = 1,
) -> List[WFWindow]:
    """Walk-forward with fixed params. No in-sample optimization."""
    results = []
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    start_date = df['timestamp'].min()
    end_date = df['timestamp'].max()

    window_id = 0
    train_start = start_date

    warmup = max(params.entropy_lookback, params.entropy_window, params.hurst_window) + 100

    while True:
        train_end = train_start + pd.DateOffset(months=train_months)
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=test_months)

        if test_end > end_date:
            break

        # Need full context from train_start to test_end for entropy lookback
        full_mask = (df['timestamp'] >= train_start) & (df['timestamp'] < test_end)
        df_full = df[full_mask].copy().reset_index(drop=True)

        if len(df_full) < warmup + 50:
            train_start += pd.DateOffset(months=step_months)
            continue

        train_mask_local = df_full['timestamp'] < test_start
        train_len = train_mask_local.sum()
        test_len = len(df_full) - train_len

        if test_len < 50:
            train_start += pd.DateOffset(months=step_months)
            continue

        try:
            df_signals = generate_combo_signals(df_full, params)
        except Exception as e:
            print(f"  WF {window_id}: signal gen failed: {e}")
            train_start += pd.DateOffset(months=step_months)
            continue

        # Only backtest on the TEST portion
        df_test = df_signals.iloc[train_len:].copy().reset_index(drop=True)
        test_result = run_backtest(df_test, params)

        # Regime breakdown
        regime_counts = {}
        exit_counts = {}
        for t in test_result.trades:
            regime_counts[t.regime] = regime_counts.get(t.regime, 0) + 1
            exit_counts[t.exit_reason] = exit_counts.get(t.exit_reason, 0) + 1

        wf = WFWindow(
            window_id=window_id,
            test_start=str(test_start.date()),
            test_end=str(test_end.date()),
            train_bars=train_len,
            test_bars=test_len,
            num_trades=test_result.num_trades,
            win_rate=test_result.win_rate,
            total_pnl_bps=test_result.total_pnl_bps,
            profit_factor=test_result.profit_factor,
            max_drawdown_bps=test_result.max_drawdown_bps,
            sharpe_ratio=test_result.sharpe_ratio,
            regime_breakdown=regime_counts,
            exit_breakdown=exit_counts,
        )
        results.append(wf)

        print(f"  WF {window_id}: [{wf.test_start} → {wf.test_end}] "
              f"Trades={wf.num_trades} WR={wf.win_rate:.0%} "
              f"PnL={wf.total_pnl_bps:+.1f}bps PF={wf.profit_factor:.2f} "
              f"Regimes={regime_counts}")

        window_id += 1
        train_start += pd.DateOffset(months=step_months)

    return results


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_report(
    full_result: BacktestResult,
    wf_results: List[WFWindow],
    relaxed_full: BacktestResult,
    relaxed_wf: List[WFWindow],
    params: ComboParams,
    relaxed_params: ComboParams,
    df_info: Dict,
) -> str:
    lines = []
    lines.append("# ECVT + Hurst Combo Strategy — Results")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Instrument:** EURUSD Hourly")
    lines.append(f"**Data:** {df_info['bars']} bars, {df_info['start']} → {df_info['end']}")
    lines.append("")

    # --- Executive Summary ---
    lines.append("## Executive Summary")
    lines.append("")

    # Strict variant
    if wf_results:
        oos_trades = sum(w.num_trades for w in wf_results)
        oos_pnl = sum(w.total_pnl_bps for w in wf_results)
        prof_windows = sum(1 for w in wf_results if w.total_pnl_bps > 0)
        lines.append(f"**Strict (entropy p5):** {oos_trades} OOS trades | "
                     f"{oos_pnl:+.1f} bps | {prof_windows}/{len(wf_results)} windows profitable")

    # Relaxed variant
    if relaxed_wf:
        r_oos_trades = sum(w.num_trades for w in relaxed_wf)
        r_oos_pnl = sum(w.total_pnl_bps for w in relaxed_wf)
        r_prof_windows = sum(1 for w in relaxed_wf if w.total_pnl_bps > 0)
        lines.append(f"**Relaxed (entropy p10):** {r_oos_trades} OOS trades | "
                     f"{r_oos_pnl:+.1f} bps | {r_prof_windows}/{len(relaxed_wf)} windows profitable")

    lines.append("")
    lines.append("### Benchmark: Standalone ECVT Hourly")
    lines.append("- 44 trades, +198 bps, PF 1.44, 41% WR")
    lines.append("- Walk-forward: +220 bps aggregate OOS, 10/20 windows profitable (50%)")
    lines.append("")

    # --- Parameters ---
    lines.append("## Parameters")
    lines.append("")
    lines.append("### Strict Variant (entropy p5)")
    lines.append("```")
    for k, v in vars(params).items():
        lines.append(f"  {k}: {v}")
    lines.append("```")
    lines.append("")
    lines.append("### Relaxed Variant (entropy p10)")
    lines.append("```")
    for k, v in vars(relaxed_params).items():
        if getattr(relaxed_params, k) != getattr(params, k):
            lines.append(f"  {k}: {v}  # changed from {getattr(params, k)}")
        else:
            lines.append(f"  {k}: {v}")
    lines.append("```")
    lines.append("")

    # --- Full-period results ---
    def result_section(result: BacktestResult, label: str) -> List[str]:
        rows = [f"### {label}", ""]
        rows.append("| Metric | Value |")
        rows.append("|--------|-------|")
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
            regime_counts = {}
            for t in result.trades:
                exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1
                regime_counts[t.regime] = regime_counts.get(t.regime, 0) + 1

            rows.append(f"| **Exit reasons** | |")
            for reason, count in sorted(exit_reasons.items()):
                rows.append(f"| — {reason} | {count} ({count/result.num_trades:.0%}) |")
            rows.append(f"| **Regimes** | |")
            for regime, count in sorted(regime_counts.items()):
                rows.append(f"| — {regime} | {count} ({count/result.num_trades:.0%}) |")

            # Per-regime PnL
            for regime in sorted(set(t.regime for t in result.trades)):
                r_trades = [t for t in result.trades if t.regime == regime]
                r_pnl = sum(t.pnl_bps for t in r_trades)
                r_wr = sum(1 for t in r_trades if t.pnl_bps > 0) / len(r_trades) if r_trades else 0
                rows.append(f"| **{regime} PnL** | {r_pnl:+.1f} bps ({len(r_trades)} trades, {r_wr:.0%} WR) |")

        rows.append("")
        return rows

    lines.append("## Full-Period Backtest Results")
    lines.append("")
    lines.extend(result_section(full_result, "Strict (entropy p5)"))
    lines.extend(result_section(relaxed_full, "Relaxed (entropy p10)"))

    # --- Walk-forward ---
    def wf_section(windows: List[WFWindow], label: str) -> List[str]:
        rows = [f"### {label}", ""]
        rows.append("| Win | Test Period | Trades | WR | PnL (bps) | PF | DD | Regimes |")
        rows.append("|-----|-------------|--------|-----|-----------|-----|-----|---------|")

        for w in windows:
            regimes_str = ", ".join(f"{k}:{v}" for k, v in sorted(w.regime_breakdown.items()))
            rows.append(
                f"| {w.window_id} | {w.test_start}→{w.test_end} | "
                f"{w.num_trades} | {w.win_rate:.0%} | "
                f"{w.total_pnl_bps:+.1f} | {w.profit_factor:.2f} | "
                f"{w.max_drawdown_bps:.1f} | {regimes_str} |"
            )

        if windows:
            total_trades = sum(w.num_trades for w in windows)
            total_pnl = sum(w.total_pnl_bps for w in windows)
            prof = sum(1 for w in windows if w.total_pnl_bps > 0)
            avg_wr = np.mean([w.win_rate for w in windows if w.num_trades > 0]) if any(w.num_trades > 0 for w in windows) else 0
            rows.append(f"| **AGG** | **{len(windows)} win** | **{total_trades}** | "
                       f"**{avg_wr:.0%}** | **{total_pnl:+.1f}** | — | — | — |")
            rows.append("")
            rows.append(f"**Consistency:** {prof}/{len(windows)} windows profitable ({prof/len(windows):.0%})")

        rows.append("")
        return rows

    lines.append("## Walk-Forward Validation (Out-of-Sample)")
    lines.append("")
    lines.append("**Method:** Train=3mo context, Test=1mo OOS, Step=1mo. Fixed params (no optimization).")
    lines.append("")
    lines.extend(wf_section(wf_results, "Strict Walk-Forward (entropy p5)"))
    lines.extend(wf_section(relaxed_wf, "Relaxed Walk-Forward (entropy p10)"))

    # --- Comparison table ---
    lines.append("## Comparison: Standalone ECVT vs ECVT+Hurst Combo")
    lines.append("")
    lines.append("| Metric | ECVT Standalone | Combo Strict (p5) | Combo Relaxed (p10) |")
    lines.append("|--------|----------------|-------------------|---------------------|")
    lines.append(f"| Full-period trades | 44 | {full_result.num_trades} | {relaxed_full.num_trades} |")
    lines.append(f"| Full-period PnL | +198 bps | {full_result.total_pnl_bps:+.1f} bps | {relaxed_full.total_pnl_bps:+.1f} bps |")
    lines.append(f"| Full-period WR | 41% | {full_result.win_rate:.0%} | {relaxed_full.win_rate:.0%} |")
    lines.append(f"| Full-period PF | 1.44 | {full_result.profit_factor:.2f} | {relaxed_full.profit_factor:.2f} |")

    if wf_results:
        combo_oos = sum(w.total_pnl_bps for w in wf_results)
        combo_oos_trades = sum(w.num_trades for w in wf_results)
        combo_prof = sum(1 for w in wf_results if w.total_pnl_bps > 0)
    else:
        combo_oos = combo_oos_trades = combo_prof = 0

    if relaxed_wf:
        relax_oos = sum(w.total_pnl_bps for w in relaxed_wf)
        relax_oos_trades = sum(w.num_trades for w in relaxed_wf)
        relax_prof = sum(1 for w in relaxed_wf if w.total_pnl_bps > 0)
    else:
        relax_oos = relax_oos_trades = relax_prof = 0

    lines.append(f"| OOS trades | 35 | {combo_oos_trades} | {relax_oos_trades} |")
    lines.append(f"| OOS PnL | +220 bps | {combo_oos:+.1f} bps | {relax_oos:+.1f} bps |")
    lines.append(f"| OOS consistency | 10/20 (50%) | {combo_prof}/{len(wf_results)} ({combo_prof/max(len(wf_results),1):.0%}) | {relax_prof}/{len(relaxed_wf)} ({relax_prof/max(len(relaxed_wf),1):.0%}) |")
    lines.append(f"| Max DD | ? | {full_result.max_drawdown_bps:.1f} bps | {relaxed_full.max_drawdown_bps:.1f} bps |")
    lines.append("")

    # --- Honest Assessment ---
    lines.append("## Honest Assessment")
    lines.append("")

    # Auto-assess
    ecvt_oos_pnl = 220
    if combo_oos > ecvt_oos_pnl * 1.1:
        lines.append("🟢 **Hurst filter IMPROVES the strategy.** The combo generates more PnL OOS than standalone ECVT.")
    elif combo_oos > 0 and combo_oos > ecvt_oos_pnl * 0.8:
        lines.append("🟡 **Hurst filter is NEUTRAL.** Similar OOS performance to standalone ECVT. The filter doesn't clearly help.")
    elif combo_oos > 0:
        lines.append("🟡 **Hurst filter REDUCES performance.** The combo is positive OOS but weaker than standalone ECVT. The additional filter is too restrictive.")
    else:
        lines.append("🔴 **Hurst filter HURTS the strategy.** The combo loses money OOS. Hurst adds noise, not signal.")

    lines.append("")

    # Relaxed assessment
    if relax_oos > combo_oos and relax_oos > ecvt_oos_pnl:
        lines.append("💡 **Relaxed variant (p10) outperforms both**, suggesting Hurst filter allows more aggressive entropy threshold while maintaining quality.")
    elif relax_oos > combo_oos:
        lines.append("💡 **Relaxed variant (p10) is better than strict**, suggesting the Hurst filter compensates for weaker entropy signals.")
    else:
        lines.append("💡 **Strict variant (p5) is better**, suggesting tighter entropy threshold matters more than Hurst filtering.")

    lines.append("")
    lines.append("### Caveats")
    lines.append("- Volume is synthesized (yfinance forex = zero volume)")
    lines.append("- Hurst R/S is noisy on short windows (50 bars = 50 hours ≈ 2 days)")
    lines.append("- ATR-based SL/TP differs from standalone ECVT's percentage-based SL/TP")
    lines.append("- Both strategies use same 2yr EURUSD hourly data")
    lines.append("")
    lines.append("---")
    lines.append(f"*Generated by ecvt_hurst_combo.py, {datetime.now().strftime('%Y-%m-%d %H:%M')}*")

    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main():
    DATA_DIR = Path(__file__).parent.parent / "curupira-backtests" / "data"

    print("=" * 70)
    print("  ECVT + HURST COMBO STRATEGY — EURUSD HOURLY")
    print("=" * 70)

    # Load data
    csv_path = DATA_DIR / "eurusd_hourly.csv"
    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = synthesize_volume(df)

    df_info = {
        'bars': len(df),
        'start': str(df['timestamp'].min().date()),
        'end': str(df['timestamp'].max().date()),
    }
    print(f"Loaded: {df_info['bars']} bars, {df_info['start']} → {df_info['end']}")

    # ---- STRICT VARIANT (entropy p5) ----
    print("\n" + "=" * 70)
    print("  STRICT VARIANT (entropy percentile = 5)")
    print("=" * 70)

    params = ComboParams()

    print("\n[1] Generating combo signals (strict)...")
    df_strict = generate_combo_signals(df.copy(), params)
    n_signals = (df_strict['signal'] != 0).sum()
    print(f"  Signals: {n_signals}")

    print("\n[2] Full-period backtest (strict)...")
    full_strict = run_backtest(df_strict, params)
    print(f"  Trades: {full_strict.num_trades} | PnL: {full_strict.total_pnl_bps:+.1f} bps | "
          f"WR: {full_strict.win_rate:.1%} | PF: {full_strict.profit_factor:.2f} | "
          f"DD: {full_strict.max_drawdown_bps:.1f} bps")

    if full_strict.num_trades > 0:
        for regime in sorted(set(t.regime for t in full_strict.trades)):
            r_trades = [t for t in full_strict.trades if t.regime == regime]
            r_pnl = sum(t.pnl_bps for t in r_trades)
            print(f"    {regime}: {len(r_trades)} trades, {r_pnl:+.1f} bps")

    print("\n[3] Walk-forward (strict)...")
    wf_strict = run_walk_forward(df.copy(), params)

    # ---- RELAXED VARIANT (entropy p10) ----
    print("\n" + "=" * 70)
    print("  RELAXED VARIANT (entropy percentile = 10)")
    print("=" * 70)

    relaxed_params = ComboParams(entropy_percentile=10.0)

    print("\n[4] Generating combo signals (relaxed)...")
    df_relaxed = generate_combo_signals(df.copy(), relaxed_params)
    n_signals_r = (df_relaxed['signal'] != 0).sum()
    print(f"  Signals: {n_signals_r}")

    print("\n[5] Full-period backtest (relaxed)...")
    full_relaxed = run_backtest(df_relaxed, relaxed_params)
    print(f"  Trades: {full_relaxed.num_trades} | PnL: {full_relaxed.total_pnl_bps:+.1f} bps | "
          f"WR: {full_relaxed.win_rate:.1%} | PF: {full_relaxed.profit_factor:.2f} | "
          f"DD: {full_relaxed.max_drawdown_bps:.1f} bps")

    if full_relaxed.num_trades > 0:
        for regime in sorted(set(t.regime for t in full_relaxed.trades)):
            r_trades = [t for t in full_relaxed.trades if t.regime == regime]
            r_pnl = sum(t.pnl_bps for t in r_trades)
            print(f"    {regime}: {len(r_trades)} trades, {r_pnl:+.1f} bps")

    print("\n[6] Walk-forward (relaxed)...")
    wf_relaxed = run_walk_forward(df.copy(), relaxed_params)

    # ---- REPORT ----
    print("\n" + "=" * 70)
    print("  GENERATING REPORT")
    print("=" * 70)

    report = generate_report(
        full_result=full_strict,
        wf_results=wf_strict,
        relaxed_full=full_relaxed,
        relaxed_wf=wf_relaxed,
        params=params,
        relaxed_params=relaxed_params,
        df_info=df_info,
    )

    results_path = Path(__file__).parent / "ecvt_hurst_combo_results.md"
    results_path.write_text(report)
    print(f"\nReport saved to: {results_path}")
    print(f"Report: {len(report)} chars")

    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70)


if __name__ == '__main__':
    main()
