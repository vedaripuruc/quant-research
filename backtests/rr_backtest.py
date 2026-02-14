#!/usr/bin/env python3
"""
R:R Sweep Backtester for active signal strategies
===================================================
Tests each signal type across multiple RR ratios (1:1, 1:1.25, 1:1.5, 1:2, 1:2.5, 1:3)
using historical data. Walk-forward: 6M train / 2M OOS / 2M step.

Answers: what RR maximizes edge for prop firm (high WR, low DD)?
"""

import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
import json, sys
from pathlib import Path

# ── Signal generation logic (imported inline to avoid side effects) ──

def compute_hurst(series, max_lag=20):
    """Simplified R/S Hurst exponent (vectorized)."""
    n = len(series)
    if n < max_lag * 2:
        return 0.5
    lags = np.arange(2, max_lag + 1)
    log_rs = np.zeros(len(lags))
    valid = 0
    for li, lag in enumerate(lags):
        num_chunks = n // lag
        if num_chunks < 1:
            continue
        arr = series[:num_chunks * lag].reshape(num_chunks, lag)
        means = arr.mean(axis=1, keepdims=True)
        devs = np.cumsum(arr - means, axis=1)
        R = devs.max(axis=1) - devs.min(axis=1)
        S = arr.std(axis=1, ddof=1)
        mask = S > 0
        if mask.any():
            log_rs[li] = np.log(np.mean(R[mask] / S[mask]))
            valid += 1
        else:
            log_rs[li] = np.nan
    if valid < 3:
        return 0.5
    mask = ~np.isnan(log_rs)
    H = np.polyfit(np.log(lags[mask]), log_rs[mask], 1)[0]
    return float(np.clip(H, 0, 1))


def compute_atr(highs, lows, closes, period=14):
    """ATR calculation."""
    trs = []
    for i in range(1, len(closes)):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        trs.append(tr)
    if len(trs) < period:
        return np.mean(trs) if trs else 0
    return np.mean(trs[-period:])


def compute_ema(series, span):
    return pd.Series(series).ewm(span=span).mean().values


@dataclass
class Trade:
    entry_time: datetime
    direction: str
    entry: float
    sl: float
    tp: float
    signal_type: str
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    outcome: Optional[str] = None
    pnl_r: float = 0.0


@dataclass
class BacktestResult:
    signal_type: str
    rr_ratio: float
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_r: float = 0.0
    avg_r: float = 0.0
    max_dd_r: float = 0.0  # max consecutive losing R
    expectancy: float = 0.0
    oos_windows: int = 0
    oos_profitable: int = 0
    trades: list = field(default_factory=list)


# ════════════════════════════════════════════════════════
# SIGNAL GENERATORS (per bar)
# ════════════════════════════════════════════════════════

def generate_hurst_signals(df, rr_ratio, sl_mult=1.5):
    """Hurst regime signals on hourly crypto data. Sample every 4 bars for speed."""
    signals = []
    lookback = 100
    for i in range(lookback, len(df), 4):  # every 4 bars
        closes = df['Close'].iloc[i-lookback:i].values
        returns = np.diff(np.log(closes))
        H = compute_hurst(returns)
        
        highs = df['High'].iloc[i-14:i].values
        lows = df['Low'].iloc[i-14:i].values
        close_arr = df['Close'].iloc[i-14:i].values
        atr = compute_atr(highs, lows, close_arr)
        if atr <= 0:
            continue
        
        entry = df['Open'].iloc[i]  # next-bar entry
        sl_dist = sl_mult * atr
        tp_dist = sl_dist * rr_ratio
        
        direction = None
        if H > 0.6:
            # Trending — follow EMA slope
            ema20 = compute_ema(df['Close'].iloc[i-30:i].values, 20)
            if len(ema20) >= 2:
                slope = ema20[-1] - ema20[-2]
                if slope > 0:
                    direction = "LONG"
                elif slope < 0:
                    direction = "SHORT"
        elif H < 0.4:
            # Mean-reverting — fade momentum
            mom = (closes[-1] - closes[-10]) / closes[-10] if closes[-10] != 0 else 0
            if mom > 0:
                direction = "SHORT"
            elif mom < 0:
                direction = "LONG"
        
        if direction is None:
            continue
        
        if direction == "LONG":
            sl = entry - sl_dist
            tp = entry + tp_dist
        else:
            sl = entry + sl_dist
            tp = entry - tp_dist
        
        signals.append({
            'bar_idx': i, 'time': df.index[i], 'direction': direction,
            'entry': entry, 'sl': sl, 'tp': tp, 'signal_type': 'hurst',
        })
    return signals


def generate_jump_signals(df, rr_ratio, fade_sl_mult=1.0, trend_sl_mult=2.0):
    """Jump detection signals on daily Gold data."""
    signals = []
    for i in range(22, len(df)):
        closes = df['Close'].iloc[:i+1].values
        opens = df['Open'].iloc[:i+1].values
        highs = df['High'].iloc[i-14:i].values
        lows = df['Low'].iloc[i-14:i].values
        close_arr = df['Close'].iloc[i-14:i].values
        
        atr = compute_atr(highs, lows, close_arr)
        if atr <= 0:
            continue
        
        # BPV and RV
        log_ret = np.diff(np.log(closes[max(0,i-21):i+1]))
        if len(log_ret) < 2:
            continue
        rv = np.sum(log_ret**2)
        bpv = (np.pi/2) * np.sum(np.abs(log_ret[1:]) * np.abs(log_ret[:-1]))
        
        jump_ratio = max(0, (rv - bpv) / rv) if rv > 0 else 0
        yesterday_ret = log_ret[-1] if len(log_ret) > 0 else 0
        
        entry = opens[i]  # next-bar open
        
        if jump_ratio > 0.3:
            # Fade
            if yesterday_ret > 0:
                direction = "SHORT"
            else:
                direction = "LONG"
            sl_dist = fade_sl_mult * atr
            tp_dist = sl_dist * rr_ratio
            sig_type = "jump_fade"
        elif jump_ratio < 0.1:
            # Trend
            mom_20 = (closes[i] - closes[i-20]) / closes[i-20] if i >= 20 else 0
            if abs(mom_20) < 0.02:
                continue
            direction = "LONG" if mom_20 > 0 else "SHORT"
            sl_dist = trend_sl_mult * atr
            tp_dist = sl_dist * rr_ratio
            sig_type = "jump_trend"
        else:
            continue
        
        if direction == "LONG":
            sl = entry - sl_dist
            tp = entry + tp_dist
        else:
            sl = entry + sl_dist
            tp = entry - tp_dist
        
        signals.append({
            'bar_idx': i, 'time': df.index[i], 'direction': direction,
            'entry': entry, 'sl': sl, 'tp': tp, 'signal_type': sig_type,
        })
    return signals


# ════════════════════════════════════════════════════════
# TRADE RESOLUTION
# ════════════════════════════════════════════════════════

def resolve_trades(signals, df, cooldown_bars=4):
    """Resolve signal list against price data. One position at a time, cooldown between trades."""
    trades = []
    next_allowed = 0
    
    for sig in signals:
        idx = sig['bar_idx']
        if idx < next_allowed or idx + 1 >= len(df):
            continue
        
        entry = sig['entry']
        sl = sig['sl']
        tp = sig['tp']
        direction = sig['direction']
        sl_dist = abs(entry - sl)
        
        # Scan forward for SL/TP hit
        for j in range(idx + 1, min(idx + 200, len(df))):  # max 200 bars
            h = df['High'].iloc[j]
            l = df['Low'].iloc[j]
            
            if direction == "LONG":
                sl_hit = l <= sl
                tp_hit = h >= tp
            else:
                sl_hit = h >= sl
                tp_hit = l <= tp
            
            if sl_hit and tp_hit:
                # Both hit same bar — use open proximity
                o = df['Open'].iloc[j]
                if direction == "LONG":
                    sl_first = abs(o - sl) < abs(o - tp)
                else:
                    sl_first = abs(o - sl) < abs(o - tp)
                outcome = "LOSS" if sl_first else "WIN"
            elif sl_hit:
                outcome = "LOSS"
            elif tp_hit:
                outcome = "WIN"
            else:
                continue
            
            pnl_r = (abs(tp - entry) / sl_dist) if outcome == "WIN" else -1.0
            trades.append(Trade(
                entry_time=sig['time'], direction=direction,
                entry=entry, sl=sl, tp=tp, signal_type=sig['signal_type'],
                exit_time=df.index[j], exit_price=tp if outcome == "WIN" else sl,
                outcome=outcome, pnl_r=pnl_r,
            ))
            next_allowed = j + cooldown_bars
            break
    
    return trades


# ════════════════════════════════════════════════════════
# WALK-FORWARD ENGINE
# ════════════════════════════════════════════════════════

def walk_forward_backtest(df, signal_fn, rr_ratio, train_months=6, oos_months=2, step_months=2):
    """Walk-forward with fixed params (no optimization — just OOS collection)."""
    all_oos_trades = []
    window_results = []
    
    dates = df.index
    start = dates[0]
    end = dates[-1]
    
    cursor = start
    while True:
        train_end = cursor + pd.DateOffset(months=train_months)
        oos_start = train_end
        oos_end = oos_start + pd.DateOffset(months=oos_months)
        
        if oos_end > end:
            break
        
        oos_df = df[oos_start:oos_end]
        if len(oos_df) < 20:
            cursor += pd.DateOffset(months=step_months)
            continue
        
        signals = signal_fn(oos_df, rr_ratio)
        trades = resolve_trades(signals, oos_df)
        
        w = sum(1 for t in trades if t.outcome == "WIN")
        l = sum(1 for t in trades if t.outcome == "LOSS")
        total_r = sum(t.pnl_r for t in trades)
        
        window_results.append({
            'start': str(oos_start.date()) if hasattr(oos_start, 'date') else str(oos_start),
            'end': str(oos_end.date()) if hasattr(oos_end, 'date') else str(oos_end),
            'trades': len(trades), 'wins': w, 'losses': l, 'total_r': round(total_r, 2),
        })
        all_oos_trades.extend(trades)
        cursor += pd.DateOffset(months=step_months)
    
    return all_oos_trades, window_results


def calculate_result(trades, signal_type, rr_ratio, windows):
    res = BacktestResult(signal_type=signal_type, rr_ratio=rr_ratio)
    res.total_trades = len(trades)
    res.wins = sum(1 for t in trades if t.outcome == "WIN")
    res.losses = sum(1 for t in trades if t.outcome == "LOSS")
    res.win_rate = round(100 * res.wins / res.total_trades, 1) if res.total_trades else 0
    
    gross_win = sum(t.pnl_r for t in trades if t.outcome == "WIN")
    gross_loss = abs(sum(t.pnl_r for t in trades if t.outcome == "LOSS"))
    res.profit_factor = round(gross_win / gross_loss, 2) if gross_loss > 0 else float('inf')
    
    res.total_r = round(sum(t.pnl_r for t in trades), 2)
    res.avg_r = round(res.total_r / res.total_trades, 3) if res.total_trades else 0
    
    # Expectancy
    if res.total_trades:
        wr = res.wins / res.total_trades
        res.expectancy = round(wr * rr_ratio - (1 - wr), 3)
    
    # Max drawdown in R
    cum = 0
    peak = 0
    max_dd = 0
    for t in trades:
        cum += t.pnl_r
        peak = max(peak, cum)
        dd = peak - cum
        max_dd = max(max_dd, dd)
    res.max_dd_r = round(max_dd, 2)
    
    res.oos_windows = len(windows)
    res.oos_profitable = sum(1 for w in windows if w['total_r'] > 0)
    
    return res


# ════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════

RR_RATIOS = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0]

ASSETS = {
    "XRP": {"symbol": "XRP-USD", "signal_fn": "hurst", "period": "2y", "interval": "1h"},
    "LINK": {"symbol": "LINK-USD", "signal_fn": "hurst", "period": "2y", "interval": "1h"},
    "ADA": {"symbol": "ADA-USD", "signal_fn": "hurst", "period": "2y", "interval": "1h"},
    "Gold": {"symbol": "GC=F", "signal_fn": "jump", "period": "2y", "interval": "1d"},
}


def main():
    import sys
    # Force unbuffered output
    sys.stdout.reconfigure(line_buffering=True)
    print("=" * 80)
    print("R:R SWEEP BACKTEST — Signal Strategy Optimization")
    print("=" * 80)
    
    all_results = []
    
    for asset_name, cfg in ASSETS.items():
        print(f"\n{'─' * 60}")
        print(f"  {asset_name} ({cfg['symbol']}) — {cfg['signal_fn']} signals")
        print(f"{'─' * 60}")
        
        ticker = yf.Ticker(cfg['symbol'])
        df = ticker.history(period=cfg['period'], interval=cfg['interval'])
        if df is None or df.empty:
            print(f"  ERROR: no data for {cfg['symbol']}")
            continue
        print(f"  Data: {len(df)} bars, {df.index[0].date()} → {df.index[-1].date()}")
        
        signal_fn = generate_hurst_signals if cfg['signal_fn'] == 'hurst' else generate_jump_signals
        
        for rr in RR_RATIOS:
            trades, windows = walk_forward_backtest(df, signal_fn, rr)
            res = calculate_result(trades, f"{asset_name}_{cfg['signal_fn']}", rr, windows)
            all_results.append(res)
            
            wr_bar = "█" * int(res.win_rate / 5) + "░" * (20 - int(res.win_rate / 5))
            pf_str = f"{res.profit_factor:.2f}" if res.profit_factor != float('inf') else "∞"
            dd_warn = " ⚠️DD" if res.max_dd_r > 5 else ""
            
            print(f"  RR 1:{rr:<4} | {res.total_trades:3d} trades | WR {res.win_rate:5.1f}% {wr_bar} | PF {pf_str:>5} | R: {res.total_r:+7.2f} | DD: {res.max_dd_r:.1f}R | E: {res.expectancy:+.3f} | OOS {res.oos_profitable}/{res.oos_windows}{dd_warn}")
    
    # ── Summary ──
    print(f"\n{'=' * 80}")
    print("SUMMARY — Best RR per asset (by expectancy)")
    print(f"{'=' * 80}")
    
    by_asset = {}
    for r in all_results:
        asset = r.signal_type.split("_")[0]
        if asset not in by_asset or r.expectancy > by_asset[asset].expectancy:
            by_asset[asset] = r
    
    for asset, r in sorted(by_asset.items()):
        pf_str = f"{r.profit_factor:.2f}" if r.profit_factor != float('inf') else "∞"
        print(f"  {asset:6s} → 1:{r.rr_ratio} | WR {r.win_rate}% | PF {pf_str} | {r.total_trades} trades | {r.total_r:+.1f}R | MaxDD {r.max_dd_r:.1f}R | E={r.expectancy:+.3f}")
    
    # ── Prop firm analysis ──
    print(f"\n{'─' * 60}")
    print("PROP FIRM LENS (need high WR, low DD, many trades)")
    print(f"{'─' * 60}")
    
    # Filter: WR > 40%, trades > 15, expectancy > 0
    viable = [r for r in all_results if r.win_rate > 40 and r.total_trades > 15 and r.expectancy > 0]
    viable.sort(key=lambda r: (-r.win_rate, r.max_dd_r))
    
    if viable:
        for r in viable[:10]:
            pf_str = f"{r.profit_factor:.2f}" if r.profit_factor != float('inf') else "∞"
            print(f"  {r.signal_type:15s} 1:{r.rr_ratio:<4} | WR {r.win_rate:5.1f}% | PF {pf_str:>5} | {r.total_trades:3d} trades | MaxDD {r.max_dd_r:.1f}R | E={r.expectancy:+.3f}")
    else:
        print("  No viable configs found (WR>40%, trades>15, E>0)")
    
    # Save results
    out_dir = Path("data/signals")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for r in all_results:
        out.append({
            "signal_type": r.signal_type, "rr_ratio": r.rr_ratio,
            "total_trades": r.total_trades, "wins": r.wins, "losses": r.losses,
            "win_rate": r.win_rate, "profit_factor": r.profit_factor if r.profit_factor != float('inf') else 999,
            "total_r": r.total_r, "avg_r": r.avg_r, "max_dd_r": r.max_dd_r,
            "expectancy": r.expectancy, "oos_windows": r.oos_windows, "oos_profitable": r.oos_profitable,
        })
    (out_dir / "rr_sweep_results.json").write_text(json.dumps(out, indent=2))
    print(f"\n  Results saved: {out_dir / 'rr_sweep_results.json'}")


if __name__ == "__main__":
    main()
