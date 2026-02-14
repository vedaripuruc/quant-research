#!/usr/bin/env python3
"""
Jump ratio distribution profiler — what does jump_ratio actually look like
on different assets/timeframes? Calibrate thresholds from data, not gut.
"""
import numpy as np
import pandas as pd
import gzip, glob, sys, json
from pathlib import Path

TICK_DIR = Path(__file__).parent / "tickdata" / "EURUSD"
sys.stdout.reconfigure(line_buffering=True)


def load_ticks():
    files = sorted(glob.glob(str(TICK_DIR / "EURUSD_BID_*.log.gz")))
    print(f"Loading {len(files)} tick files...", flush=True)
    chunks = []
    for i, f in enumerate(files):
        if i % 3000 == 0 and i > 0:
            print(f"  {i}/{len(files)}...", flush=True)
        try:
            with gzip.open(f, 'rt') as fh:
                for line in fh:
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        chunks.append((int(parts[0]), float(parts[1])))
        except:
            continue
    df = pd.DataFrame(chunks, columns=['timestamp_ms', 'price'])
    df['datetime'] = pd.to_datetime(df['timestamp_ms'], unit='ms').sort_values()
    df = df.sort_values('datetime').reset_index(drop=True)
    print(f"Loaded {len(df):,} ticks", flush=True)
    return df


def make_bars(ticks, bar_min):
    t = ticks.set_index('datetime')
    freq = f'{bar_min}min' if bar_min < 1440 else '1D'
    bars = t['price'].resample(freq).ohlc().dropna()
    bars.columns = ['Open', 'High', 'Low', 'Close']
    return bars


def compute_jump_ratios(bars, lookback=22):
    """Compute jump_ratio for every bar."""
    closes = bars['Close'].values
    ratios = []
    for i in range(lookback, len(bars)):
        log_ret = np.diff(np.log(closes[max(0, i-lookback):i+1]))
        if len(log_ret) < 3:
            ratios.append(np.nan)
            continue
        rv = np.sum(log_ret**2)
        bpv = (np.pi/2) * np.sum(np.abs(log_ret[1:]) * np.abs(log_ret[:-1]))
        jr = max(0, (rv - bpv) / rv) if rv > 0 else 0
        ratios.append(jr)
    return np.array(ratios)


def profile_timeframe(bars, bar_min, ticks_df):
    """Profile jump_ratio distribution and test threshold impact."""
    jrs = compute_jump_ratios(bars)
    jrs = jrs[~np.isnan(jrs)]
    
    print(f"\n{'='*60}")
    print(f"  EURUSD {bar_min}min — Jump Ratio Distribution ({len(jrs)} bars)")
    print(f"{'='*60}")
    
    # Distribution stats
    pcts = [5, 10, 25, 50, 75, 90, 95, 99]
    vals = np.percentile(jrs, pcts)
    print(f"  Mean: {np.mean(jrs):.4f}  Std: {np.std(jrs):.4f}  Min: {np.min(jrs):.4f}  Max: {np.max(jrs):.4f}")
    print(f"  Percentiles:")
    for p, v in zip(pcts, vals):
        bar = "█" * int(v * 100)
        print(f"    P{p:2d}: {v:.4f}  {bar}")
    
    # How often does each threshold fire?
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    print(f"\n  Fade threshold (jr > X) — how often it fires:")
    for th in thresholds:
        count = np.sum(jrs > th)
        pct = 100 * count / len(jrs)
        bar = "█" * int(pct)
        print(f"    > {th:.2f}: {count:5d} ({pct:5.1f}%)  {bar}")
    
    print(f"\n  Trend threshold (jr < X) — how often it fires:")
    for th in [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]:
        count = np.sum(jrs < th)
        pct = 100 * count / len(jrs)
        bar = "█" * int(pct)
        print(f"    < {th:.2f}: {count:5d} ({pct:5.1f}%)  {bar}")
    
    # What's the optimal fade threshold? Test WR at each
    print(f"\n  Fade signal WR by threshold (next-bar direction accuracy):")
    closes = bars['Close'].values
    lookback = 22
    for th in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
        correct = 0
        total = 0
        for i in range(lookback, len(bars) - 1):
            log_ret = np.diff(np.log(closes[max(0, i-lookback):i+1]))
            if len(log_ret) < 3:
                continue
            rv = np.sum(log_ret**2)
            bpv = (np.pi/2) * np.sum(np.abs(log_ret[1:]) * np.abs(log_ret[:-1]))
            jr = max(0, (rv - bpv) / rv) if rv > 0 else 0
            
            if jr > th:
                yesterday_ret = log_ret[-1]
                # Fade = expect reversal
                if yesterday_ret > 0:
                    predicted = "DOWN"
                else:
                    predicted = "UP"
                
                actual_ret = closes[i+1] - closes[i]
                actual = "UP" if actual_ret > 0 else "DOWN"
                
                if predicted == actual:
                    correct += 1
                total += 1
        
        if total > 0:
            wr = 100 * correct / total
            print(f"    > {th:.2f}: {total:4d} signals, WR {wr:5.1f}% {'✅' if wr > 52 else '❌'}")
        else:
            print(f"    > {th:.2f}:    0 signals")
    
    # Trend signal accuracy
    print(f"\n  Trend signal WR by threshold (momentum continuation):")
    for th in [0.01, 0.02, 0.05, 0.10]:
        correct = 0
        total = 0
        for i in range(lookback, len(bars) - 1):
            if i < 20:
                continue
            log_ret = np.diff(np.log(closes[max(0, i-lookback):i+1]))
            if len(log_ret) < 3:
                continue
            rv = np.sum(log_ret**2)
            bpv = (np.pi/2) * np.sum(np.abs(log_ret[1:]) * np.abs(log_ret[:-1]))
            jr = max(0, (rv - bpv) / rv) if rv > 0 else 0
            
            if jr < th:
                mom = (closes[i] - closes[i-20]) / closes[i-20]
                if abs(mom) < 0.005:
                    continue
                predicted = "UP" if mom > 0 else "DOWN"
                actual_ret = closes[i+1] - closes[i]
                actual = "UP" if actual_ret > 0 else "DOWN"
                if predicted == actual:
                    correct += 1
                total += 1
        
        if total > 0:
            wr = 100 * correct / total
            print(f"    < {th:.2f}: {total:4d} signals, WR {wr:5.1f}% {'✅' if wr > 52 else '❌'}")
        else:
            print(f"    < {th:.2f}:    0 signals")
    
    return jrs


def main():
    ticks = load_ticks()
    
    for bar_min in [60, 240, 1440]:
        bars = make_bars(ticks.copy(), bar_min)
        print(f"\n  {bar_min}min: {len(bars)} bars")
        profile_timeframe(bars, bar_min, ticks)
    
    # Compare to Gold daily (yfinance)
    print(f"\n{'='*60}")
    print(f"  GOLD DAILY — Reference Distribution")
    print(f"{'='*60}")
    try:
        import yfinance as yf
        gold = yf.Ticker("GC=F").history(period="2y", interval="1d")
        if gold is not None and len(gold) > 30:
            jrs_gold = compute_jump_ratios(gold)
            jrs_gold = jrs_gold[~np.isnan(jrs_gold)]
            pcts = [5, 10, 25, 50, 75, 90, 95, 99]
            vals = np.percentile(jrs_gold, pcts)
            print(f"  {len(jrs_gold)} bars")
            print(f"  Mean: {np.mean(jrs_gold):.4f}  Std: {np.std(jrs_gold):.4f}")
            for p, v in zip(pcts, vals):
                print(f"    P{p:2d}: {v:.4f}")
            for th in [0.10, 0.20, 0.30, 0.40, 0.50]:
                count = np.sum(jrs_gold > th)
                pct = 100 * count / len(jrs_gold)
                print(f"    > {th:.2f}: {count:4d} ({pct:5.1f}%)")
    except:
        print("  (yfinance unavailable)")


if __name__ == "__main__":
    main()
