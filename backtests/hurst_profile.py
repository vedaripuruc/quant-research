#!/usr/bin/env python3
"""
Hurst exponent distribution profiler — are the H>0.6/H<0.4 thresholds
actually calibrated for crypto hourly? Same analysis as jump_ratio_profile.
"""
import numpy as np
import pandas as pd
import yfinance as yf
import sys

sys.stdout.reconfigure(line_buffering=True)

ASSETS = {
    "XRP": "XRP-USD",
    "LINK": "LINK-USD",
    "ADA": "ADA-USD",
}

def compute_hurst(series, max_lag=20):
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


def compute_ema_slope(closes, span=20):
    ema = pd.Series(closes).ewm(span=span).mean().values
    if len(ema) < 2:
        return 0
    return ema[-1] - ema[-2]


def profile_asset(name, symbol):
    print(f"\n{'='*65}")
    print(f"  {name} ({symbol}) — 1H — Hurst Distribution")
    print(f"{'='*65}")

    df = yf.Ticker(symbol).history(period="2y", interval="1h")
    if df is None or len(df) < 200:
        print("  Insufficient data"); return

    print(f"  {len(df)} bars, {df.index[0].date()} → {df.index[-1].date()}")

    closes = df['Close'].values
    lookback = 100

    # Compute H for every 4th bar (speed)
    hurst_vals = []
    bar_indices = []
    for i in range(lookback, len(df), 4):
        returns = np.diff(np.log(closes[i-lookback:i]))
        H = compute_hurst(returns)
        hurst_vals.append(H)
        bar_indices.append(i)

    H_arr = np.array(hurst_vals)
    print(f"  Computed {len(H_arr)} Hurst values (every 4 bars)")

    # Distribution
    pcts = [5, 10, 25, 50, 75, 90, 95, 99]
    vals = np.percentile(H_arr, pcts)
    print(f"  Mean: {np.mean(H_arr):.4f}  Std: {np.std(H_arr):.4f}  Min: {np.min(H_arr):.4f}  Max: {np.max(H_arr):.4f}")
    for p, v in zip(pcts, vals):
        pos = int(v * 50)
        bar = "░" * pos + "█"
        print(f"    P{p:2d}: {v:.4f}  {bar}")

    # Zone frequency
    zones = [
        ("H < 0.30 (strong MR)", lambda h: h < 0.30),
        ("H < 0.35", lambda h: h < 0.35),
        ("H < 0.40 (current MR)", lambda h: h < 0.40),
        ("0.40-0.60 (random)", lambda h: 0.40 <= h <= 0.60),
        ("H > 0.60 (current trend)", lambda h: h > 0.60),
        ("H > 0.65", lambda h: h > 0.65),
        ("H > 0.70", lambda h: h > 0.70),
        ("H > 0.75", lambda h: h > 0.75),
        ("H > 0.80", lambda h: h > 0.80),
    ]
    print(f"\n  Zone frequency:")
    for label, fn in zones:
        count = np.sum([fn(h) for h in H_arr])
        pct = 100 * count / len(H_arr)
        bar = "█" * int(pct / 2)
        print(f"    {label:30s}: {count:5d} ({pct:5.1f}%) {bar}")

    # Direction prediction WR by threshold
    print(f"\n  Trend signal WR (H > threshold, follow EMA slope → next bar):")
    for th in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        correct = 0; total = 0
        for idx, i in enumerate(bar_indices):
            if i + 1 >= len(df):
                continue
            H = hurst_vals[idx]
            if H <= th:
                continue
            # Follow EMA slope
            ema_slope = compute_ema_slope(closes[i-30:i], 20)
            if abs(ema_slope) < 1e-8:
                continue
            predicted = "UP" if ema_slope > 0 else "DOWN"
            actual = "UP" if closes[i+1] > closes[i] else "DOWN"
            if predicted == actual:
                correct += 1
            total += 1
        if total > 0:
            wr = 100 * correct / total
            print(f"    H > {th:.2f}: {total:4d} signals, WR {wr:5.1f}% {'✅' if wr > 52 else '❌'}")
        else:
            print(f"    H > {th:.2f}:    0 signals")

    # Fade signal WR
    print(f"\n  Fade signal WR (H < threshold, fade 10-bar momentum → next bar):")
    for th in [0.30, 0.35, 0.40, 0.45]:
        correct = 0; total = 0
        for idx, i in enumerate(bar_indices):
            if i + 1 >= len(df) or i < 10:
                continue
            H = hurst_vals[idx]
            if H >= th:
                continue
            mom = (closes[i] - closes[i-10]) / closes[i-10]
            if abs(mom) < 0.001:
                continue
            predicted = "DOWN" if mom > 0 else "UP"  # fade
            actual = "UP" if closes[i+1] > closes[i] else "DOWN"
            if predicted == actual:
                correct += 1
            total += 1
        if total > 0:
            wr = 100 * correct / total
            print(f"    H < {th:.2f}: {total:4d} signals, WR {wr:5.1f}% {'✅' if wr > 52 else '❌'}")
        else:
            print(f"    H < {th:.2f}:    0 signals")

    # Multi-bar prediction (does it predict 4-bar direction better?)
    print(f"\n  Trend signal WR — 4-bar lookahead (H > threshold):")
    for th in [0.55, 0.60, 0.65, 0.70, 0.75]:
        correct = 0; total = 0
        for idx, i in enumerate(bar_indices):
            if i + 4 >= len(df):
                continue
            H = hurst_vals[idx]
            if H <= th:
                continue
            ema_slope = compute_ema_slope(closes[i-30:i], 20)
            if abs(ema_slope) < 1e-8:
                continue
            predicted = "UP" if ema_slope > 0 else "DOWN"
            actual = "UP" if closes[i+4] > closes[i] else "DOWN"
            if predicted == actual:
                correct += 1
            total += 1
        if total > 0:
            wr = 100 * correct / total
            print(f"    H > {th:.2f}: {total:4d} signals, WR {wr:5.1f}% (4-bar) {'✅' if wr > 52 else '❌'}")

    return H_arr


def main():
    print("=" * 65)
    print("HURST THRESHOLD PROFILER — Crypto 1H")
    print("=" * 65)

    all_h = {}
    for name, sym in ASSETS.items():
        H_arr = profile_asset(name, sym)
        if H_arr is not None:
            all_h[name] = H_arr

    # Cross-asset summary
    print(f"\n{'='*65}")
    print("CROSS-ASSET SUMMARY")
    print(f"{'='*65}")
    for name, H_arr in all_h.items():
        trending = np.sum(H_arr > 0.6) / len(H_arr) * 100
        mr = np.sum(H_arr < 0.4) / len(H_arr) * 100
        rw = np.sum((H_arr >= 0.4) & (H_arr <= 0.6)) / len(H_arr) * 100
        print(f"  {name:6s}: trend(>0.6)={trending:4.1f}%  random(0.4-0.6)={rw:4.1f}%  MR(<0.4)={mr:4.1f}%  mean={np.mean(H_arr):.3f}")


if __name__ == "__main__":
    main()
