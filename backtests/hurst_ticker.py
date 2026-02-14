"""
HurstTicker — Tick-Level Hurst Regime Detector
================================================
Computes Hurst exponent directly on raw tick-to-tick log returns
for EURUSD, then evaluates regime detection accuracy and simulated P&L.

4 test weeks:
  - Trending:       Nov 4-8, 2024  (US election week, strong USD move)
  - Choppy/ranging: Aug 12-16, 2024 (summer doldrums)
  - High-volatility: Jan 6-10, 2025 (new-year return, central bank chatter)
  - Quiet/holiday:  Dec 23-27, 2024 (Christmas week)
"""

import gzip
import glob
import json
import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

# ──────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────

TICK_DIR = Path(__file__).parent / "tickdata" / "EURUSD"

# 4 diverse weeks (Mon-Fri dates)
WEEKS = {
    "trending_nov2024": [
        "2024-11-04", "2024-11-05", "2024-11-06", "2024-11-07", "2024-11-08"
    ],
    "choppy_aug2024": [
        "2024-08-12", "2024-08-13", "2024-08-14", "2024-08-15", "2024-08-16"
    ],
    "highvol_jan2025": [
        "2025-01-06", "2025-01-07", "2025-01-08", "2025-01-09", "2025-01-10"
    ],
    "quiet_dec2024": [
        "2024-12-23", "2024-12-24", "2024-12-25", "2024-12-26", "2024-12-27"
    ],
}

# Regime thresholds
H_TREND = 0.60
H_MEANREV = 0.40


# ──────────────────────────────────────────────────
# TICK DATA LOADING
# ──────────────────────────────────────────────────

def load_day_ticks(date_str: str) -> np.ndarray:
    """Load all hourly tick files for a given date.
    Returns array of (timestamp_ms, price) tuples, sorted by time.
    """
    pattern = str(TICK_DIR / f"EURUSD_BID_{date_str}_*.log.gz")
    files = sorted(glob.glob(pattern))
    if not files:
        return np.empty((0, 2))
    
    all_ticks = []
    for fpath in files:
        with gzip.open(fpath, "rt") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) >= 2:
                    try:
                        ts = float(parts[0])
                        price = float(parts[1])
                        all_ticks.append((ts, price))
                    except ValueError:
                        continue
    
    if not all_ticks:
        return np.empty((0, 2))
    
    arr = np.array(all_ticks)
    # Sort by timestamp
    arr = arr[arr[:, 0].argsort()]
    return arr


# ──────────────────────────────────────────────────
# HURST EXPONENT (R/S analysis)
# ──────────────────────────────────────────────────

def hurst_rs(series: np.ndarray) -> float:
    """
    Compute Hurst exponent using Rescaled Range (R/S) analysis.
    Optimised for large tick-level datasets (10K-200K points).
    """
    N = len(series)
    if N < 100:
        return np.nan

    max_k = N // 2
    min_k = 16

    # Log-spaced chunk sizes — more sizes for better regression
    ks = []
    k = min_k
    while k <= max_k:
        ks.append(int(k))
        k *= 1.4
        if len(ks) > 0 and int(k) == ks[-1]:
            k += 1

    if len(ks) < 4:
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
            if S > 1e-15:
                rs_list.append(R / S)

        if rs_list:
            rs_values.append(np.mean(rs_list))
            ns_values.append(k)

    if len(ns_values) < 4:
        return np.nan

    log_n = np.log(np.array(ns_values, dtype=float))
    log_rs = np.log(np.array(rs_values, dtype=float))

    try:
        coeffs = np.polyfit(log_n, log_rs, 1)
        H = float(coeffs[0])
        return max(0.0, min(1.0, H))
    except Exception:
        return np.nan


# ──────────────────────────────────────────────────
# DAILY STATS
# ──────────────────────────────────────────────────

def compute_day_stats(ticks: np.ndarray) -> dict:
    """
    From an array of (timestamp_ms, price), compute:
      - open, close, high, low, range
      - direction (close - open)
      - first-hour close price (for P&L sim)
      - tick-to-tick log returns
      - Hurst exponent on those returns
    """
    if len(ticks) < 200:
        return None

    prices = ticks[:, 1]
    timestamps = ticks[:, 0]

    open_price = prices[0]
    close_price = prices[-1]
    high_price = np.max(prices)
    low_price = np.min(prices)
    daily_range = high_price - low_price
    direction = close_price - open_price  # positive = bullish

    # First-hour boundary: ticks within first 60 min from first tick
    first_ts = timestamps[0]
    one_hour_ms = 3600 * 1000
    first_hour_mask = timestamps <= (first_ts + one_hour_ms)
    first_hour_prices = prices[first_hour_mask]
    if len(first_hour_prices) > 0:
        first_hour_close = first_hour_prices[-1]
        first_hour_dir = first_hour_close - open_price
    else:
        first_hour_close = open_price
        first_hour_dir = 0.0

    # Tick-to-tick log returns
    log_returns = np.diff(np.log(prices))
    # Remove any NaN/Inf
    log_returns = log_returns[np.isfinite(log_returns)]

    n_ticks = len(prices)
    n_returns = len(log_returns)

    # Compute Hurst on raw tick returns
    H = hurst_rs(log_returns)

    # Also compute Hurst on subsampled returns (every Nth tick → pseudo-candles)
    # This aggregates microstructure noise
    hurst_sub = {}
    for subsample in [50, 100, 500, 1000]:
        if len(prices) > subsample * 20:  # need at least 20 data points
            sub_prices = prices[::subsample]
            sub_returns = np.diff(np.log(sub_prices))
            sub_returns = sub_returns[np.isfinite(sub_returns)]
            if len(sub_returns) >= 100:
                hurst_sub[f"H_sub{subsample}"] = float(hurst_rs(sub_returns))
            else:
                hurst_sub[f"H_sub{subsample}"] = None
        else:
            hurst_sub[f"H_sub{subsample}"] = None

    # Also try Hurst on absolute returns (volatility clustering)
    abs_returns = np.abs(log_returns)
    H_abs = hurst_rs(abs_returns) if len(abs_returns) >= 100 else None

    # Regime classification — use subsampled Hurst if available (sub500 is ~minute-level)
    # Prefer sub100 as it aggregates noise but keeps enough data points
    H_regime = None
    for key in ["H_sub100", "H_sub50"]:
        if hurst_sub.get(key) is not None:
            H_regime = hurst_sub[key]
            break
    if H_regime is None:
        H_regime = H  # fallback to raw tick Hurst

    if H_regime is not None and not np.isnan(H_regime):
        if H_regime > H_TREND:
            regime = "trending"
        elif H_regime < H_MEANREV:
            regime = "mean-reverting"
        else:
            regime = "random"
    else:
        regime = "unknown"

    # Actual market behaviour classification
    if daily_range > 0:
        move_ratio = abs(direction) / daily_range
    else:
        move_ratio = 0.0

    if move_ratio > 0.3:
        actual_regime = "trending"
    elif move_ratio < 0.2:
        actual_regime = "mean-reverting"
    else:
        actual_regime = "ambiguous"

    return {
        "n_ticks": int(n_ticks),
        "n_returns": int(n_returns),
        "open": float(open_price),
        "close": float(close_price),
        "high": float(high_price),
        "low": float(low_price),
        "range_pips": float(daily_range * 10000),  # in pips
        "direction_pips": float(direction * 10000),
        "move_ratio": float(move_ratio),
        "first_hour_dir": float(first_hour_dir),
        "hurst_tick": float(H) if (H is not None and not np.isnan(H)) else None,
        "hurst_abs": float(H_abs) if (H_abs is not None and not np.isnan(H_abs)) else None,
        "hurst": float(H_regime) if (H_regime is not None and not np.isnan(H_regime)) else None,
        **{k: v for k, v in hurst_sub.items()},
        "regime_call": regime,
        "actual_regime": actual_regime,
    }


# ──────────────────────────────────────────────────
# EVALUATE REGIME ACCURACY
# ──────────────────────────────────────────────────

def evaluate_regime(day: dict) -> str | None:
    """Check if regime call matches actual market behaviour."""
    if day["hurst"] is None:
        return None
    regime = day["regime_call"]
    actual = day["actual_regime"]

    if regime == "random":
        return None  # skip

    if regime == actual:
        return "correct"
    elif actual == "ambiguous":
        return "ambiguous"
    else:
        return "wrong"


# ──────────────────────────────────────────────────
# SIMULATED P&L
# ──────────────────────────────────────────────────

def simulate_pnl(days: list[dict]) -> list[dict]:
    """
    Use previous day's Hurst to predict today.
    - Trending: follow first-hour direction → win if close is in same direction
    - Mean-reverting: fade first-hour direction → win if close reverses
    
    Returns list of trade results.
    """
    trades = []
    for i in range(1, len(days)):
        prev = days[i - 1]
        today = days[i]

        if prev["hurst"] is None or today["hurst"] is None:
            continue

        prev_regime = prev["regime_call"]
        if prev_regime == "random":
            continue

        first_hour_dir = today["first_hour_dir"]
        day_dir = today["close"] - today["open"]

        if abs(first_hour_dir) < 1e-6:
            continue  # no signal if first hour was flat

        if prev_regime == "trending":
            # Follow first-hour direction
            # Win if market closes in same direction as first hour
            won = (first_hour_dir > 0 and day_dir > 0) or (first_hour_dir < 0 and day_dir < 0)
            pnl_pips = abs(day_dir) * 10000 if won else -abs(day_dir) * 10000
        else:  # mean-reverting
            # Fade first-hour direction
            # Win if market reverses (closes opposite to first hour)
            won = (first_hour_dir > 0 and day_dir < 0) or (first_hour_dir < 0 and day_dir > 0)
            pnl_pips = abs(day_dir) * 10000 if won else -abs(day_dir) * 10000

        trades.append({
            "date": today["date"],
            "signal_from": prev["date"],
            "prev_hurst": prev["hurst"],
            "prev_regime": prev_regime,
            "first_hour_dir_pips": round(first_hour_dir * 10000, 1),
            "day_dir_pips": round(day_dir * 10000, 1),
            "won": won,
            "pnl_pips": round(pnl_pips, 1),
        })

    return trades


# ──────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("HurstTicker — Tick-Level Hurst Regime Detector")
    print("=" * 60)

    all_days = []

    for week_name, dates in WEEKS.items():
        print(f"\n--- {week_name} ---")
        for date_str in dates:
            # Skip weekends (shouldn't be in list, but just in case)
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            if dt.weekday() >= 5:
                print(f"  {date_str}: skipping (weekend)")
                continue

            print(f"  {date_str}: loading ticks...", end=" ", flush=True)
            ticks = load_day_ticks(date_str)
            if len(ticks) == 0:
                print("NO DATA")
                continue

            print(f"{len(ticks)} ticks, computing Hurst...", end=" ", flush=True)
            stats = compute_day_stats(ticks)
            if stats is None:
                print("insufficient data")
                continue

            stats["date"] = date_str
            stats["week"] = week_name

            # Evaluate regime
            result = evaluate_regime(stats)
            stats["regime_match"] = result

            all_days.append(stats)

            ht = f"Htick={stats['hurst_tick']:.3f}" if stats["hurst_tick"] else "Htick=N/A"
            h100 = f"H100={stats.get('H_sub100','N/A')}" if stats.get('H_sub100') else "H100=N/A"
            if isinstance(stats.get('H_sub100'), float):
                h100 = f"H100={stats['H_sub100']:.3f}"
            habs = f"Habs={stats['hurst_abs']:.3f}" if stats.get("hurst_abs") else ""
            print(f"{ht} {h100} {habs} [{stats['regime_call']}] actual={stats['actual_regime']} "
                  f"move={stats['direction_pips']:.1f}pip range={stats['range_pips']:.1f}pip "
                  f"ratio={stats['move_ratio']:.2f} → {result or 'skipped'}")

    # ── Accuracy stats ──
    print("\n" + "=" * 60)
    print("ACCURACY SUMMARY")
    print("=" * 60)

    calls = [d for d in all_days if d["regime_match"] is not None]
    correct = [d for d in calls if d["regime_match"] == "correct"]
    wrong = [d for d in calls if d["regime_match"] == "wrong"]
    ambiguous = [d for d in calls if d["regime_match"] == "ambiguous"]

    trend_calls = [d for d in all_days if d["regime_call"] == "trending"]
    trend_correct = [d for d in trend_calls if d["regime_match"] == "correct"]
    mr_calls = [d for d in all_days if d["regime_call"] == "mean-reverting"]
    mr_correct = [d for d in mr_calls if d["regime_match"] == "correct"]
    random_calls = [d for d in all_days if d["regime_call"] == "random"]

    total_days = len(all_days)
    print(f"Total days analysed: {total_days}")
    print(f"Regime calls made: {len(calls)} (skipped {len(random_calls)} random + {total_days - len(calls) - len(random_calls)} unknown)")
    print(f"  Correct: {len(correct)}")
    print(f"  Wrong:   {len(wrong)}")
    print(f"  Ambiguous: {len(ambiguous)}")
    if calls:
        acc = len(correct) / (len(correct) + len(wrong)) if (len(correct) + len(wrong)) > 0 else 0
        print(f"  Accuracy (excl ambiguous): {acc:.1%}")
    print(f"\nTrending calls: {len(trend_calls)}")
    if trend_calls:
        tc = len(trend_correct)
        tw = len([d for d in trend_calls if d['regime_match'] == 'wrong'])
        if tc + tw > 0:
            print(f"  Correct: {tc}/{tc+tw} = {tc/(tc+tw):.1%}")
    print(f"Mean-reverting calls: {len(mr_calls)}")
    if mr_calls:
        mc = len(mr_correct)
        mw = len([d for d in mr_calls if d['regime_match'] == 'wrong'])
        if mc + mw > 0:
            print(f"  Correct: {mc}/{mc+mw} = {mc/(mc+mw):.1%}")

    # Hurst distribution
    hurst_vals = [d["hurst"] for d in all_days if d["hurst"] is not None]
    if hurst_vals:
        print(f"\nHurst distribution: min={min(hurst_vals):.3f} max={max(hurst_vals):.3f} "
              f"mean={np.mean(hurst_vals):.3f} std={np.std(hurst_vals):.3f}")

    # ── P&L Simulation ──
    print("\n" + "=" * 60)
    print("SIMULATED P&L (previous day's H → today's trade)")
    print("=" * 60)

    # Do P&L per week (contiguous days)
    all_trades = []
    for week_name, dates in WEEKS.items():
        week_days = [d for d in all_days if d["week"] == week_name]
        week_days.sort(key=lambda x: x["date"])
        trades = simulate_pnl(week_days)
        all_trades.extend(trades)

    if all_trades:
        total_pnl = sum(t["pnl_pips"] for t in all_trades)
        wins = [t for t in all_trades if t["won"]]
        losses = [t for t in all_trades if not t["won"]]
        print(f"Total trades: {len(all_trades)}")
        print(f"Wins: {len(wins)}, Losses: {len(losses)}")
        if all_trades:
            print(f"Win rate: {len(wins)/len(all_trades):.1%}")
        print(f"Total P&L: {total_pnl:.1f} pips")
        if wins:
            print(f"Avg win: {np.mean([t['pnl_pips'] for t in wins]):.1f} pips")
        if losses:
            print(f"Avg loss: {np.mean([t['pnl_pips'] for t in losses]):.1f} pips")

        for t in all_trades:
            emoji = "✅" if t["won"] else "❌"
            print(f"  {t['date']} [{t['prev_regime']}←{t['signal_from']}] "
                  f"1h={t['first_hour_dir_pips']:+.1f}pip close={t['day_dir_pips']:+.1f}pip "
                  f"→ {t['pnl_pips']:+.1f}pip {emoji}")
    else:
        print("No trades generated.")

    # ── Save results ──
    output_dir = Path(__file__).parent

    # JSON
    results = {
        "days": all_days,
        "trades": all_trades,
        "summary": {
            "total_days": total_days,
            "regime_calls": len(calls),
            "correct": len(correct),
            "wrong": len(wrong),
            "ambiguous": len(ambiguous),
            "accuracy_pct": round(len(correct) / (len(correct) + len(wrong)) * 100, 1) if (len(correct) + len(wrong)) > 0 else None,
            "trend_calls": len(trend_calls),
            "mr_calls": len(mr_calls),
            "random_calls": len(random_calls),
            "hurst_mean": round(np.mean(hurst_vals), 4) if hurst_vals else None,
            "hurst_std": round(np.std(hurst_vals), 4) if hurst_vals else None,
            "total_trades": len(all_trades),
            "wins": len(wins) if all_trades else 0,
            "losses": len(losses) if all_trades else 0,
            "total_pnl_pips": round(total_pnl, 1) if all_trades else 0,
        }
    }
    json_path = output_dir / "results_hurst_ticker.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved raw data to {json_path}")

    # ── Markdown Report ──
    md_lines = []
    md_lines.append("# HurstTicker — Tick-Level Hurst Regime Detector Report")
    md_lines.append("")
    md_lines.append("## Concept")
    md_lines.append("Instead of computing Hurst on OHLC candle bars (50-200 data points), we compute it")
    md_lines.append("directly on raw tick-to-tick log returns (~10K-80K+ data points per day). This should")
    md_lines.append("give a much more statistically reliable Hurst estimate.")
    md_lines.append("")

    md_lines.append("## Test Weeks")
    md_lines.append("| Week | Period | Rationale |")
    md_lines.append("|------|--------|-----------|")
    md_lines.append("| Trending | Nov 4-8, 2024 | US election week, strong USD |")
    md_lines.append("| Choppy | Aug 12-16, 2024 | Summer doldrums |")
    md_lines.append("| High-vol | Jan 6-10, 2025 | New year return |")
    md_lines.append("| Quiet | Dec 23-27, 2024 | Christmas week |")
    md_lines.append("")

    # Day-by-day table
    md_lines.append("## Day-by-Day Results")
    md_lines.append("")
    md_lines.append("| Date | Week | Ticks | Hurst | Regime Call | Move (pip) | Range (pip) | Ratio | Actual | Match? |")
    md_lines.append("|------|------|-------|-------|------------|-----------|-----------|-------|--------|--------|")
    for d in all_days:
        h = f"{d['hurst']:.3f}" if d['hurst'] else "N/A"
        match = d['regime_match'] or "skip"
        wk = d['week'].split('_')[0]
        md_lines.append(
            f"| {d['date']} | {wk} | {d['n_ticks']:,} | {h} | {d['regime_call']} | "
            f"{d['direction_pips']:+.1f} | {d['range_pips']:.1f} | {d['move_ratio']:.2f} | "
            f"{d['actual_regime']} | {match} |"
        )
    md_lines.append("")

    # Accuracy
    md_lines.append("## Accuracy Statistics")
    md_lines.append("")
    md_lines.append(f"- **Total days analysed:** {total_days}")
    md_lines.append(f"- **Regime calls made:** {len(calls)} (of which {len(random_calls)} were 'random'/skipped)")
    md_lines.append(f"- **Correct:** {len(correct)}")
    md_lines.append(f"- **Wrong:** {len(wrong)}")
    md_lines.append(f"- **Ambiguous:** {len(ambiguous)}")
    if (len(correct) + len(wrong)) > 0:
        acc = len(correct) / (len(correct) + len(wrong))
        md_lines.append(f"- **Accuracy (excl. ambiguous):** {acc:.1%}")
    md_lines.append("")
    if trend_calls:
        tc = len(trend_correct)
        tw = len([d for d in trend_calls if d['regime_match'] == 'wrong'])
        md_lines.append(f"- **Trending calls:** {len(trend_calls)} → {tc} correct, {tw} wrong" +
                        (f" ({tc/(tc+tw):.0%})" if tc+tw > 0 else ""))
    if mr_calls:
        mc = len(mr_correct)
        mw = len([d for d in mr_calls if d['regime_match'] == 'wrong'])
        md_lines.append(f"- **Mean-reverting calls:** {len(mr_calls)} → {mc} correct, {mw} wrong" +
                        (f" ({mc/(mc+mw):.0%})" if mc+mw > 0 else ""))
    md_lines.append("")

    # Hurst distribution
    if hurst_vals:
        md_lines.append("## Hurst Distribution")
        md_lines.append(f"- Min: {min(hurst_vals):.3f}")
        md_lines.append(f"- Max: {max(hurst_vals):.3f}")
        md_lines.append(f"- Mean: {np.mean(hurst_vals):.3f}")
        md_lines.append(f"- Std: {np.std(hurst_vals):.3f}")
        md_lines.append("")

    # P&L
    md_lines.append("## Simulated P&L")
    md_lines.append("**Rule:** Use previous day's Hurst regime to trade today.")
    md_lines.append("- Trending signal → follow first-hour direction")
    md_lines.append("- Mean-reverting signal → fade first-hour direction")
    md_lines.append("")
    if all_trades:
        md_lines.append(f"| Date | Signal From | Prev Regime | Prev H | 1h Dir (pip) | Close Dir (pip) | P&L (pip) | Won? |")
        md_lines.append(f"|------|------------|-------------|--------|-------------|----------------|----------|------|")
        for t in all_trades:
            won_emoji = "✅" if t["won"] else "❌"
            md_lines.append(
                f"| {t['date']} | {t['signal_from']} | {t['prev_regime']} | "
                f"{t['prev_hurst']:.3f} | {t['first_hour_dir_pips']:+.1f} | "
                f"{t['day_dir_pips']:+.1f} | {t['pnl_pips']:+.1f} | {won_emoji} |"
            )
        md_lines.append("")
        md_lines.append(f"**Total trades:** {len(all_trades)}")
        md_lines.append(f"**Win rate:** {len(wins)/len(all_trades):.1%}")
        md_lines.append(f"**Total P&L:** {total_pnl:.1f} pips")
    else:
        md_lines.append("No trades generated.")
    md_lines.append("")

    # Key insights
    md_lines.append("## Key Insights")
    md_lines.append("")
    md_lines.append("### Does tick-level Hurst give better regime detection than candle-level?")
    md_lines.append("")
    md_lines.append("*To be filled after analysis — see raw numbers above.*")
    md_lines.append("")

    report_path = output_dir / "report_hurst_ticker.md"
    with open(report_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"Saved report to {report_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
