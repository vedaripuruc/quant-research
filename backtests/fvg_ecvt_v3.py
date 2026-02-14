#!/usr/bin/env python3
"""
FVG v2 + ECVT regime filter walk-forward backtest (v3).

Strategies:
1) FVG v2 baseline
2) FVG v2 with entropy regime filter (skip entropy collapse)
3) FVG v2 inverse filter (take only entropy collapse)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from ecvt_fast import compute_markov_entropy_fast, states_from_ohlcv
from fvg_barrier_v2_backtest import (
    Trade,
    build_walk_forward_windows,
    compute_metrics,
    load_hourly_data,
    run_backtest,
)

N_WINDOWS = 10
ENTROPY_WINDOW = 120
ENTROPY_LOOKBACK = 500
ENTROPY_PERCENTILE = 0.10


def round_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, int):
            out[key] = value
        else:
            out[key] = round(float(value), 4)
    return out


def compute_entropy_collapse_flags(hourly: pd.DataFrame) -> np.ndarray:
    states = states_from_ohlcv(hourly)
    entropy = compute_markov_entropy_fast(states, window=ENTROPY_WINDOW, n_states=15)

    threshold = (
        pd.Series(entropy)
        .rolling(window=ENTROPY_LOOKBACK, min_periods=ENTROPY_LOOKBACK)
        .quantile(ENTROPY_PERCENTILE)
        .to_numpy()
    )

    return np.isfinite(entropy) & np.isfinite(threshold) & (entropy < threshold)


def split_trades_by_regime(
    trades: Sequence[Trade],
    entropy_collapse_flags: np.ndarray,
) -> Tuple[List[Trade], List[Trade]]:
    filtered: List[Trade] = []
    inverse: List[Trade] = []

    n = len(entropy_collapse_flags)
    for trade in trades:
        signal_idx = trade.entry_idx - 1
        if signal_idx < 0 or signal_idx >= n:
            filtered.append(trade)
            continue

        if bool(entropy_collapse_flags[signal_idx]):
            inverse.append(trade)
        else:
            filtered.append(trade)

    return filtered, inverse


def run_walk_forward(hourly_full: pd.DataFrame, n_windows: int = N_WINDOWS) -> Dict:
    windows = build_walk_forward_windows(total_bars=len(hourly_full), n_windows=n_windows)

    baseline_windows: List[Dict] = []
    filtered_windows: List[Dict] = []
    inverse_windows: List[Dict] = []

    baseline_all: List[Trade] = []
    filtered_all: List[Trade] = []
    inverse_all: List[Trade] = []

    for w in windows:
        start = w["start"]
        train_end = w["train_end"]
        test_end = w["test_end"]
        train_size = w["train_size"]

        segment = hourly_full.iloc[start:test_end].reset_index(drop=True)
        _, all_trades = run_backtest(segment)
        oos_trades = [t for t in all_trades if t.entry_idx >= train_size]

        collapse_flags = compute_entropy_collapse_flags(segment)
        filtered_trades, inverse_trades = split_trades_by_regime(oos_trades, collapse_flags)

        baseline_all.extend(oos_trades)
        filtered_all.extend(filtered_trades)
        inverse_all.extend(inverse_trades)

        test_period = {
            "start": segment.iloc[train_size]["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
            "end": segment.iloc[-1]["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
        }

        baseline_windows.append(
            {
                "window": w["window"],
                "test_period": test_period,
                **round_metrics(compute_metrics(oos_trades)),
            }
        )
        filtered_windows.append(
            {
                "window": w["window"],
                "test_period": test_period,
                **round_metrics(compute_metrics(filtered_trades)),
            }
        )
        inverse_windows.append(
            {
                "window": w["window"],
                "test_period": test_period,
                **round_metrics(compute_metrics(inverse_trades)),
            }
        )

    baseline_agg = round_metrics(compute_metrics(baseline_all))
    filtered_agg = round_metrics(compute_metrics(filtered_all))
    inverse_agg = round_metrics(compute_metrics(inverse_all))

    baseline_pnl = float(baseline_agg["pnl_pips"])
    filtered_pnl = float(filtered_agg["pnl_pips"])
    inverse_pnl = float(inverse_agg["pnl_pips"])

    return {
        "fvg_v2_baseline": {
            "windows": baseline_windows,
            "aggregate": baseline_agg,
        },
        "fvg_v3_regime_filter": {
            "windows": filtered_windows,
            "aggregate": filtered_agg,
        },
        "fvg_v3_inverse": {
            "windows": inverse_windows,
            "aggregate": inverse_agg,
        },
        "comparison": {
            "baseline_pnl": round(baseline_pnl, 4),
            "filtered_pnl": round(filtered_pnl, 4),
            "inverse_pnl": round(inverse_pnl, 4),
            "improvement_pips": round(filtered_pnl - baseline_pnl, 4),
        },
    }


def fmt_pf(value: float) -> str:
    return f"{value:.2f}"


def print_comparison_table(results: Dict) -> None:
    base = results["fvg_v2_baseline"]["windows"]
    filt = results["fvg_v3_regime_filter"]["windows"]
    inv = results["fvg_v3_inverse"]["windows"]

    print("\nFVG v2 vs FVG v3 Regime Filter vs Inverse (OOS)")
    print("-" * 174)
    print(
        f"{'Win':>3}  {'Test Period':<35} "
        f"{'BASE T':>7} {'BASE WR%':>8} {'BASE PnL':>10} {'BASE PF':>8} "
        f"{'FILT T':>7} {'FILT WR%':>8} {'FILT PnL':>10} {'FILT PF':>8} "
        f"{'INV T':>7} {'INV WR%':>8} {'INV PnL':>10} {'INV PF':>8}"
    )
    print("-" * 174)

    n = min(len(base), len(filt), len(inv))
    for i in range(n):
        bw = base[i]
        fw = filt[i]
        iw = inv[i]
        period = f"{bw['test_period']['start']} -> {bw['test_period']['end']}"
        print(
            f"{bw['window']:>3}  {period:<35} "
            f"{bw['trades']:>7} {bw['win_rate']:>8.2f} {bw['pnl_pips']:>10.2f} {fmt_pf(float(bw['profit_factor'])):>8} "
            f"{fw['trades']:>7} {fw['win_rate']:>8.2f} {fw['pnl_pips']:>10.2f} {fmt_pf(float(fw['profit_factor'])):>8} "
            f"{iw['trades']:>7} {iw['win_rate']:>8.2f} {iw['pnl_pips']:>10.2f} {fmt_pf(float(iw['profit_factor'])):>8}"
        )

    print("-" * 174)
    ba = results["fvg_v2_baseline"]["aggregate"]
    fa = results["fvg_v3_regime_filter"]["aggregate"]
    ia = results["fvg_v3_inverse"]["aggregate"]
    print(
        f"ALL  {'Aggregate OOS':<35} "
        f"{ba['trades']:>7} {ba['win_rate']:>8.2f} {ba['pnl_pips']:>10.2f} {fmt_pf(float(ba['profit_factor'])):>8} "
        f"{fa['trades']:>7} {fa['win_rate']:>8.2f} {fa['pnl_pips']:>10.2f} {fmt_pf(float(fa['profit_factor'])):>8} "
        f"{ia['trades']:>7} {ia['win_rate']:>8.2f} {ia['pnl_pips']:>10.2f} {fmt_pf(float(ia['profit_factor'])):>8}"
    )
    print("-" * 174)


def main() -> None:
    parser = argparse.ArgumentParser(description="FVG v2 + ECVT regime filter backtest (v3).")
    parser.add_argument("--input", default="tickdata/EURUSD_1H_2Y.csv", help="Input 1H CSV")
    parser.add_argument("--output", default="fvg_ecvt_v3_results.json", help="Output JSON file")
    parser.add_argument("--windows", type=int, default=N_WINDOWS, help="Number of rolling windows")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    hourly = load_hourly_data(input_path)
    results = run_walk_forward(hourly_full=hourly, n_windows=args.windows)

    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print_comparison_table(results)
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
