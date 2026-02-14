#!/usr/bin/env python3
"""
Jump Signal Tick Backtester — EURUSD tick-verified
====================================================
Builds OHLC bars from tick data at configurable timeframe,
generates jump (fade + trend) signals, resolves SL/TP at tick level.
Sweeps RR ratios. Walk-forward validation.

Usage:
  python jump_tick_backtest.py --bar-minutes 60    # 1H bars
  python jump_tick_backtest.py --bar-minutes 240   # 4H bars
  python jump_tick_backtest.py --bar-minutes 1440  # Daily bars (reference)
"""

import numpy as np
import pandas as pd
import gzip, glob, argparse, json, sys, os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

TICK_DIR = Path(__file__).parent / "tickdata" / "EURUSD"

# ════════════════════════════════════════════════════════
# TICK DATA → OHLC BARS
# ════════════════════════════════════════════════════════

def load_all_ticks(max_files=None):
    """Load all EURUSD tick files into a single DataFrame."""
    files = sorted(glob.glob(str(TICK_DIR / "EURUSD_BID_*.log.gz")))
    if max_files:
        files = files[:max_files]
    
    print(f"  Loading {len(files)} tick files...", flush=True)
    chunks = []
    for i, f in enumerate(files):
        if i % 2000 == 0 and i > 0:
            print(f"    {i}/{len(files)} files loaded...", flush=True)
        try:
            with gzip.open(f, 'rt') as fh:
                for line in fh:
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        chunks.append((int(parts[0]), float(parts[1])))
        except Exception:
            continue
    
    if not chunks:
        return pd.DataFrame()
    
    df = pd.DataFrame(chunks, columns=['timestamp_ms', 'price'])
    df['datetime'] = pd.to_datetime(df['timestamp_ms'], unit='ms', utc=True).dt.tz_localize(None)
    df = df.sort_values('datetime').reset_index(drop=True)
    print(f"  Loaded {len(df):,} ticks, {df['datetime'].iloc[0]} → {df['datetime'].iloc[-1]}", flush=True)
    return df


def ticks_to_bars(ticks_df, bar_minutes):
    """Resample ticks into OHLC bars."""
    ticks_df = ticks_df.set_index('datetime')
    freq = f'{bar_minutes}min' if bar_minutes < 1440 else '1D'
    bars = ticks_df['price'].resample(freq).ohlc()
    bars.columns = ['Open', 'High', 'Low', 'Close']
    bars = bars.dropna()
    print(f"  Built {len(bars)} bars ({bar_minutes}min)", flush=True)
    return bars


# ════════════════════════════════════════════════════════
# SIGNAL GENERATION
# ════════════════════════════════════════════════════════

def compute_atr(bars, period=14):
    """Vectorized ATR."""
    h = bars['High'].values
    l = bars['Low'].values
    c = bars['Close'].values
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    atr = pd.Series(np.concatenate([[np.nan], tr])).rolling(period).mean().values
    return atr


def generate_jump_signals(bars, rr_ratio, lookback=22, fade_sl_mult=1.0, trend_sl_mult=2.0):
    """Generate jump fade + trend signals from OHLC bars."""
    closes = bars['Close'].values
    opens = bars['Open'].values
    atr = compute_atr(bars)
    
    signals = []
    for i in range(lookback, len(bars)):
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue
        
        # Log returns for BPV/RV
        log_ret = np.diff(np.log(closes[max(0, i-lookback):i+1]))
        if len(log_ret) < 3:
            continue
        
        rv = np.sum(log_ret**2)
        bpv = (np.pi/2) * np.sum(np.abs(log_ret[1:]) * np.abs(log_ret[:-1]))
        jump_ratio = max(0, (rv - bpv) / rv) if rv > 0 else 0
        
        yesterday_ret = log_ret[-1]
        entry = opens[i]  # next-bar open
        
        if jump_ratio > 0.3:
            # Fade the jump
            direction = "SHORT" if yesterday_ret > 0 else "LONG"
            sl_dist = fade_sl_mult * atr[i]
            tp_dist = sl_dist * rr_ratio
            sig_type = "jump_fade"
        elif jump_ratio < 0.1:
            # Trend
            if i >= 20:
                mom = (closes[i] - closes[i-20]) / closes[i-20]
            else:
                continue
            if abs(mom) < 0.005:  # lower threshold for forex (vs 2% for gold)
                continue
            direction = "LONG" if mom > 0 else "SHORT"
            sl_dist = trend_sl_mult * atr[i]
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
            'bar_idx': i,
            'time': bars.index[i],
            'direction': direction,
            'entry': entry,
            'sl': sl,
            'tp': tp,
            'signal_type': sig_type,
            'jump_ratio': round(jump_ratio, 4),
            'atr': atr[i],
        })
    
    return signals


# ════════════════════════════════════════════════════════
# TICK-LEVEL TRADE RESOLUTION
# ════════════════════════════════════════════════════════

@dataclass
class Trade:
    entry_time: object
    direction: str
    entry: float
    sl: float
    tp: float
    signal_type: str
    exit_time: object = None
    exit_price: float = 0.0
    outcome: str = ""
    pnl_pips: float = 0.0
    pnl_r: float = 0.0
    bars_held: int = 0


def resolve_trades_tick(signals, bars, ticks_df, bar_minutes, cooldown_bars=4, max_bars=200):
    """Resolve trades using actual tick data for SL/TP verification.
    Uses numpy searchsorted for fast tick windowing instead of scanning 42M ticks per trade."""
    trades = []
    next_allowed_idx = 0
    
    ticks_sorted = ticks_df.sort_values('datetime')
    tick_times = ticks_sorted['datetime'].values.astype('datetime64[ns]')
    tick_prices = ticks_sorted['price'].values
    
    for sig in signals:
        idx = sig['bar_idx']
        if idx < next_allowed_idx or idx + 1 >= len(bars):
            continue
        
        entry = sig['entry']
        sl = sig['sl']
        tp = sig['tp']
        direction = sig['direction']
        sl_dist = abs(entry - sl)
        
        entry_time = bars.index[idx]
        max_exit_idx = min(idx + max_bars, len(bars) - 1)
        max_exit_time = bars.index[max_exit_idx]
        
        # Fast binary search for tick window
        i_start = np.searchsorted(tick_times, np.datetime64(entry_time, 'ns'))
        i_end = np.searchsorted(tick_times, np.datetime64(max_exit_time, 'ns'), side='right')
        
        if i_start >= i_end:
            continue
        
        window_prices = tick_prices[i_start:i_end]
        window_times = tick_times[i_start:i_end]
        
        # Vectorized SL/TP check
        if direction == "LONG":
            sl_hits = np.where(window_prices <= sl)[0]
            tp_hits = np.where(window_prices >= tp)[0]
        else:
            sl_hits = np.where(window_prices >= sl)[0]
            tp_hits = np.where(window_prices <= tp)[0]
        
        first_sl = sl_hits[0] if len(sl_hits) > 0 else len(window_prices) + 1
        first_tp = tp_hits[0] if len(tp_hits) > 0 else len(window_prices) + 1
        
        if first_sl <= first_tp and first_sl < len(window_prices):
            outcome = "LOSS"
            exit_idx = first_sl
            exit_price = sl
        elif first_tp < first_sl and first_tp < len(window_prices):
            outcome = "WIN"
            exit_idx = first_tp
            exit_price = tp
        else:
            outcome = "TIMEOUT"
            exit_idx = len(window_prices) - 1
            exit_price = window_prices[-1]
        
        exit_time = window_times[exit_idx]
        
        # P&L
        if direction == "LONG":
            pnl_pips = (exit_price - entry) * 10000
        else:
            pnl_pips = (entry - exit_price) * 10000
        
        pnl_r = pnl_pips / (sl_dist * 10000) if sl_dist > 0 else 0
        
        exit_dt = pd.Timestamp(exit_time)
        entry_dt = pd.Timestamp(entry_time)
        delta_s = (exit_dt - entry_dt).total_seconds()
        bars_held = max(1, int(delta_s / (bar_minutes * 60)))
        
        trades.append(Trade(
            entry_time=entry_time, direction=direction,
            entry=entry, sl=sl, tp=tp, signal_type=sig['signal_type'],
            exit_time=exit_time, exit_price=exit_price,
            outcome=outcome, pnl_pips=round(pnl_pips, 1),
            pnl_r=round(pnl_r, 2), bars_held=bars_held,
        ))
        
        # Cooldown: find next bar after exit
        if exit_time is not None:
            for ci in range(idx + 1, len(bars)):
                if bars.index[ci] >= exit_dt:
                    next_allowed_idx = ci + cooldown_bars
                    break
            else:
                next_allowed_idx = len(bars)
        else:
            next_allowed_idx = idx + cooldown_bars
    
    return trades


# ════════════════════════════════════════════════════════
# WALK-FORWARD
# ════════════════════════════════════════════════════════

def walk_forward(bars, ticks_df, bar_minutes, rr_ratio, 
                 train_months=6, oos_months=2, step_months=2):
    """Walk-forward with tick-verified resolution."""
    all_trades = []
    windows = []
    
    start = bars.index[0]
    end = bars.index[-1]
    cursor = start
    
    while True:
        train_end = cursor + pd.DateOffset(months=train_months)
        oos_start = train_end
        oos_end = oos_start + pd.DateOffset(months=oos_months)
        
        if oos_end > end:
            break
        
        oos_bars = bars[oos_start:oos_end]
        if len(oos_bars) < 20:
            cursor += pd.DateOffset(months=step_months)
            continue
        
        # Get ticks for OOS window (with buffer for trade resolution)
        tick_start = oos_start
        tick_end = oos_end + pd.DateOffset(months=1)
        oos_ticks = ticks_df[(ticks_df['datetime'] >= tick_start) & (ticks_df['datetime'] <= tick_end)]
        
        signals = generate_jump_signals(oos_bars, rr_ratio)
        trades = resolve_trades_tick(signals, oos_bars, oos_ticks, bar_minutes)
        
        w = sum(1 for t in trades if t.outcome == "WIN")
        l = sum(1 for t in trades if t.outcome == "LOSS")
        to = sum(1 for t in trades if t.outcome == "TIMEOUT")
        total_pips = sum(t.pnl_pips for t in trades)
        total_r = sum(t.pnl_r for t in trades)
        
        windows.append({
            'start': str(oos_start.date()) if hasattr(oos_start, 'date') else str(oos_start)[:10],
            'end': str(oos_end.date()) if hasattr(oos_end, 'date') else str(oos_end)[:10],
            'trades': len(trades), 'wins': w, 'losses': l, 'timeouts': to,
            'pips': round(total_pips, 1), 'total_r': round(total_r, 2),
        })
        all_trades.extend(trades)
        cursor += pd.DateOffset(months=step_months)
    
    return all_trades, windows


def print_results(trades, windows, rr_ratio, bar_minutes):
    """Print formatted results."""
    total = len(trades)
    wins = sum(1 for t in trades if t.outcome == "WIN")
    losses = sum(1 for t in trades if t.outcome == "LOSS")
    timeouts = sum(1 for t in trades if t.outcome == "TIMEOUT")
    
    wr = round(100 * wins / total, 1) if total else 0
    total_pips = sum(t.pnl_pips for t in trades)
    total_r = sum(t.pnl_r for t in trades)
    
    gross_win = sum(t.pnl_r for t in trades if t.outcome == "WIN")
    gross_loss = abs(sum(t.pnl_r for t in trades if t.outcome == "LOSS"))
    pf = round(gross_win / gross_loss, 2) if gross_loss > 0 else float('inf')
    
    # Max drawdown in R
    cum = 0; peak = 0; max_dd = 0
    for t in trades:
        cum += t.pnl_r
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)
    
    # Expectancy
    if total:
        expectancy = round((wins/total) * rr_ratio - (losses/total), 3)
    else:
        expectancy = 0
    
    oos_profitable = sum(1 for w in windows if w['total_r'] > 0)
    
    # By signal type
    fade_trades = [t for t in trades if t.signal_type == 'jump_fade']
    trend_trades = [t for t in trades if t.signal_type == 'jump_trend']
    
    pf_str = f"{pf:.2f}" if pf != float('inf') else "∞"
    dd_warn = " ⚠️DD" if max_dd > 10 else ""
    
    print(f"  RR 1:{rr_ratio:<4} | {total:4d} trades | WR {wr:5.1f}% | PF {pf_str:>5} | {total_pips:+8.1f} pips | {total_r:+7.1f}R | DD {max_dd:.1f}R | E {expectancy:+.3f} | OOS {oos_profitable}/{len(windows)}{dd_warn}")
    
    if fade_trades:
        fw = sum(1 for t in fade_trades if t.outcome == "WIN")
        fl = sum(1 for t in fade_trades if t.outcome == "LOSS")
        fwr = round(100 * fw / len(fade_trades), 1) if fade_trades else 0
        fp = sum(t.pnl_pips for t in fade_trades)
        print(f"         fade: {len(fade_trades):4d} trades | WR {fwr:5.1f}% | {fp:+8.1f} pips")
    
    if trend_trades:
        tw = sum(1 for t in trend_trades if t.outcome == "WIN")
        tl = sum(1 for t in trend_trades if t.outcome == "LOSS")
        twr = round(100 * tw / len(trend_trades), 1) if trend_trades else 0
        tp_pips = sum(t.pnl_pips for t in trend_trades)
        print(f"        trend: {len(trend_trades):4d} trades | WR {twr:5.1f}% | {tp_pips:+8.1f} pips")
    
    return {
        'bar_minutes': bar_minutes, 'rr_ratio': rr_ratio,
        'total_trades': total, 'wins': wins, 'losses': losses, 'timeouts': timeouts,
        'win_rate': wr, 'profit_factor': pf if pf != float('inf') else 999,
        'total_pips': round(total_pips, 1), 'total_r': round(total_r, 1),
        'max_dd_r': round(max_dd, 1), 'expectancy': expectancy,
        'oos_windows': len(windows), 'oos_profitable': oos_profitable,
        'windows': windows,
    }


# ════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════

RR_RATIOS = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0]

def main():
    sys.stdout.reconfigure(line_buffering=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--bar-minutes', type=int, nargs='+', default=[60, 240],
                        help='Bar sizes in minutes (default: 60 240)')
    parser.add_argument('--rr', type=float, nargs='+', default=None,
                        help='RR ratios to test (default: all)')
    args = parser.parse_args()
    
    rr_ratios = args.rr or RR_RATIOS
    
    print("=" * 90)
    print("JUMP SIGNAL TICK BACKTEST — EURUSD — Tick-Verified SL/TP")
    print("=" * 90)
    
    # Load ticks once
    ticks_df = load_all_ticks()
    if ticks_df.empty:
        print("ERROR: No tick data found")
        return
    
    all_results = []
    
    for bar_min in args.bar_minutes:
        print(f"\n{'━' * 70}")
        print(f"  EURUSD — {bar_min}min bars — Jump Detection")
        print(f"{'━' * 70}")
        
        bars = ticks_to_bars(ticks_df.copy(), bar_min)
        
        for rr in rr_ratios:
            trades, windows = walk_forward(bars, ticks_df, bar_min, rr)
            result = print_results(trades, windows, rr, bar_min)
            all_results.append(result)
    
    # ── Summary ──
    print(f"\n{'=' * 90}")
    print("SUMMARY — Best RR per timeframe (by expectancy, min 20 trades)")
    print(f"{'=' * 90}")
    
    for bar_min in args.bar_minutes:
        tf_results = [r for r in all_results if r['bar_minutes'] == bar_min and r['total_trades'] >= 20]
        if tf_results:
            best = max(tf_results, key=lambda r: r['expectancy'])
            pf_str = f"{best['profit_factor']:.2f}" if best['profit_factor'] < 999 else "∞"
            print(f"  {bar_min:4d}min → 1:{best['rr_ratio']} | WR {best['win_rate']}% | PF {pf_str} | {best['total_trades']} trades | {best['total_pips']:+.0f} pips | DD {best['max_dd_r']:.1f}R | E={best['expectancy']:+.3f}")
    
    # ── Prop firm lens ──
    print(f"\n{'─' * 70}")
    print("PROP FIRM LENS (WR > 40%, trades > 20, E > 0)")
    print(f"{'─' * 70}")
    
    viable = [r for r in all_results if r['win_rate'] > 40 and r['total_trades'] > 20 and r['expectancy'] > 0]
    viable.sort(key=lambda r: (-r['win_rate'], r['max_dd_r']))
    
    if viable:
        for r in viable[:8]:
            pf_str = f"{r['profit_factor']:.2f}" if r['profit_factor'] < 999 else "∞"
            print(f"  {r['bar_minutes']:4d}min 1:{r['rr_ratio']:<4} | WR {r['win_rate']:5.1f}% | PF {pf_str:>5} | {r['total_trades']:4d} trades | {r['total_pips']:+8.0f} pips | DD {r['max_dd_r']:.1f}R | E={r['expectancy']:+.3f}")
    else:
        print("  No viable configs found")
    
    # Save
    out = Path("data/signals/jump_tick_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\n  Results saved: {out}")


if __name__ == "__main__":
    main()
