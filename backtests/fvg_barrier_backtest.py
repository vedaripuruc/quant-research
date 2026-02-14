#!/usr/bin/env python3
"""
FVG Barrier backtest.

Builds daily FVG barrier zones from 1H data, classifies 1H reactions at active zones,
and evaluates fade/break entries with walk-forward validation.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

PIP_SIZE = 0.0001
SPREAD_PIPS = 1.0
RR_RATIO = 2.0


@dataclass
class FVGZone:
    zone_id: int
    direction: str  # bullish | bearish
    lower: float
    upper: float
    creation_date: str
    active_from: pd.Timestamp
    active_until: Optional[pd.Timestamp]


@dataclass
class Trade:
    zone_id: int
    zone_direction: str
    signal_type: str  # fade | break
    side: str  # long | short
    signal_time: str
    entry_time: str
    exit_time: str
    entry_price: float
    sl: float
    tp: float
    exit_price: float
    outcome: str  # win | loss | time_exit
    pnl_pips: float
    entry_idx: int


def load_hourly_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    required = {"timestamp", "open", "high", "low", "close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df


def resample_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    daily = (
        hourly.set_index("timestamp")
        .resample("1D")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "ticks": "sum",
            }
        )
        .dropna(subset=["open", "high", "low", "close"])
    )
    return daily


def detect_fvg_zones(daily: pd.DataFrame) -> List[FVGZone]:
    zones: List[FVGZone] = []
    zone_id = 1

    for i in range(2, len(daily)):
        left = daily.iloc[i - 2]
        right = daily.iloc[i]
        creation_day = daily.index[i]

        if left["high"] < right["low"]:
            lower = float(left["high"])
            upper = float(right["low"])
            active_until = find_fill_time(daily, i, "bullish", lower, upper)
            zones.append(
                FVGZone(
                    zone_id=zone_id,
                    direction="bullish",
                    lower=lower,
                    upper=upper,
                    creation_date=creation_day.strftime("%Y-%m-%d"),
                    active_from=creation_day + pd.Timedelta(days=1),
                    active_until=active_until,
                )
            )
            zone_id += 1

        if left["low"] > right["high"]:
            lower = float(right["high"])
            upper = float(left["low"])
            active_until = find_fill_time(daily, i, "bearish", lower, upper)
            zones.append(
                FVGZone(
                    zone_id=zone_id,
                    direction="bearish",
                    lower=lower,
                    upper=upper,
                    creation_date=creation_day.strftime("%Y-%m-%d"),
                    active_from=creation_day + pd.Timedelta(days=1),
                    active_until=active_until,
                )
            )
            zone_id += 1

    return zones


def find_fill_time(
    daily: pd.DataFrame,
    creation_idx: int,
    direction: str,
    lower: float,
    upper: float,
) -> Optional[pd.Timestamp]:
    """
    FVG deactivates after the first daily close that crosses through the entire zone.
    - Bullish zone fully filled if close <= lower edge
    - Bearish zone fully filled if close >= upper edge
    """
    for j in range(creation_idx + 1, len(daily)):
        close = float(daily.iloc[j]["close"])
        day = daily.index[j]
        if direction == "bullish" and close <= lower:
            return day + pd.Timedelta(days=1)
        if direction == "bearish" and close >= upper:
            return day + pd.Timedelta(days=1)
    return None


def classify_reaction(
    bar_open: float,
    bar_high: float,
    bar_low: float,
    bar_close: float,
    lower: float,
    upper: float,
) -> str:
    """
    Returns one of:
    - bounce_from_above
    - bounce_from_below
    - clean_break_up
    - clean_break_down
    - stall
    - no_touch
    """
    touches_zone = bar_high >= lower and bar_low <= upper
    if not touches_zone:
        return "no_touch"

    if bar_open > upper:
        if bar_close > upper:
            return "bounce_from_above"
        if bar_close < lower:
            return "clean_break_down"
        return "stall"

    if bar_open < lower:
        if bar_close < lower:
            return "bounce_from_below"
        if bar_close > upper:
            return "clean_break_up"
        return "stall"

    return "stall"


def trade_from_signal(
    reaction: str,
    zone: FVGZone,
    entry_price: float,
) -> Optional[Tuple[str, str, float, float]]:
    """
    Returns (signal_type, side, sl, tp) or None if no tradable signal.
    """
    side: Optional[str] = None
    signal_type: Optional[str] = None
    sl: Optional[float] = None

    if reaction == "bounce_from_above" and zone.direction == "bullish":
        signal_type = "fade"
        side = "long"
        sl = zone.lower
    elif reaction == "bounce_from_below" and zone.direction == "bearish":
        signal_type = "fade"
        side = "short"
        sl = zone.upper
    elif reaction == "clean_break_up":
        signal_type = "break"
        side = "long"
        sl = (zone.lower + zone.upper) / 2.0
    elif reaction == "clean_break_down":
        signal_type = "break"
        side = "short"
        sl = (zone.lower + zone.upper) / 2.0
    else:
        return None

    if side == "long":
        risk = entry_price - sl
        if risk <= 0:
            return None
        tp = entry_price + RR_RATIO * risk
    else:
        risk = sl - entry_price
        if risk <= 0:
            return None
        tp = entry_price - RR_RATIO * risk

    return signal_type, side, sl, tp


def simulate_exit(
    hourly: pd.DataFrame,
    entry_idx: int,
    side: str,
    sl: float,
    tp: float,
) -> Tuple[int, float, str]:
    for idx in range(entry_idx, len(hourly)):
        bar = hourly.iloc[idx]
        high = float(bar["high"])
        low = float(bar["low"])

        if side == "long":
            sl_hit = low <= sl
            tp_hit = high >= tp
        else:
            sl_hit = high >= sl
            tp_hit = low <= tp

        if sl_hit and tp_hit:
            return idx, sl, "loss"
        if sl_hit:
            return idx, sl, "loss"
        if tp_hit:
            return idx, tp, "win"

    last_idx = len(hourly) - 1
    last_close = float(hourly.iloc[last_idx]["close"])
    return last_idx, last_close, "time_exit"


def run_backtest(hourly: pd.DataFrame) -> Tuple[List[FVGZone], List[Trade]]:
    daily = resample_daily(hourly)
    zones = detect_fvg_zones(daily)
    trades: List[Trade] = []

    for i in range(len(hourly) - 1):
        bar = hourly.iloc[i]
        ts = pd.Timestamp(bar["timestamp"])
        bar_open = float(bar["open"])
        bar_high = float(bar["high"])
        bar_low = float(bar["low"])
        bar_close = float(bar["close"])

        active_zones = [
            z
            for z in zones
            if z.active_from <= ts and (z.active_until is None or ts < z.active_until)
        ]

        if not active_zones:
            continue

        for zone in active_zones:
            reaction = classify_reaction(
                bar_open=bar_open,
                bar_high=bar_high,
                bar_low=bar_low,
                bar_close=bar_close,
                lower=zone.lower,
                upper=zone.upper,
            )
            if reaction in {"no_touch", "stall"}:
                continue

            entry_idx = i + 1
            entry_bar = hourly.iloc[entry_idx]
            entry_price = float(entry_bar["open"])
            entry_time = pd.Timestamp(entry_bar["timestamp"])

            trade_setup = trade_from_signal(reaction, zone, entry_price)
            if trade_setup is None:
                continue
            signal_type, side, sl, tp = trade_setup

            exit_idx, exit_price, outcome = simulate_exit(
                hourly=hourly,
                entry_idx=entry_idx,
                side=side,
                sl=sl,
                tp=tp,
            )
            exit_time = pd.Timestamp(hourly.iloc[exit_idx]["timestamp"])

            if side == "long":
                pnl_pips = (exit_price - entry_price) / PIP_SIZE
            else:
                pnl_pips = (entry_price - exit_price) / PIP_SIZE
            pnl_pips -= SPREAD_PIPS

            trades.append(
                Trade(
                    zone_id=zone.zone_id,
                    zone_direction=zone.direction,
                    signal_type=signal_type,
                    side=side,
                    signal_time=ts.isoformat(),
                    entry_time=entry_time.isoformat(),
                    exit_time=exit_time.isoformat(),
                    entry_price=entry_price,
                    sl=sl,
                    tp=tp,
                    exit_price=exit_price,
                    outcome=outcome,
                    pnl_pips=pnl_pips,
                    entry_idx=entry_idx,
                )
            )

    return zones, trades


def compute_metrics(trades: List[Trade]) -> Dict[str, float]:
    if not trades:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "pnl_pips": 0.0,
            "profit_factor": 0.0,
            "gross_profit_pips": 0.0,
            "gross_loss_pips": 0.0,
        }

    pnl_values = [t.pnl_pips for t in trades]
    wins = sum(1 for pnl in pnl_values if pnl > 0)
    losses = len(trades) - wins
    gross_profit = sum(pnl for pnl in pnl_values if pnl > 0)
    gross_loss = -sum(pnl for pnl in pnl_values if pnl < 0)
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

    return {
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": (wins / len(trades)) * 100.0,
        "pnl_pips": sum(pnl_values),
        "profit_factor": profit_factor,
        "gross_profit_pips": gross_profit,
        "gross_loss_pips": gross_loss,
    }


def build_walk_forward_windows(
    total_bars: int,
    n_windows: int = 10,
    train_ratio: float = 0.6,
    test_ratio: float = 0.4,
) -> List[Dict[str, int]]:
    if total_bars < n_windows:
        raise ValueError("Not enough bars for requested number of windows")

    ratio = train_ratio / test_ratio
    test_size = int(round(total_bars / (n_windows + ratio)))
    if test_size <= 0:
        raise ValueError("Computed test size is zero")

    train_size = total_bars - (n_windows * test_size)
    if train_size <= 0:
        raise ValueError("Computed train size is zero")

    windows: List[Dict[str, int]] = []
    for i in range(n_windows):
        start = i * test_size
        train_end = start + train_size
        test_end = train_end + test_size
        if test_end > total_bars:
            break
        windows.append(
            {
                "window": i + 1,
                "start": start,
                "train_end": train_end,
                "test_end": test_end,
                "train_size": train_size,
                "test_size": test_size,
            }
        )
    return windows


def fmt_pf(value: float) -> str:
    if value == 0.0:
        return "0.00"
    return f"{value:.2f}"


def run_walk_forward(hourly_full: pd.DataFrame, n_windows: int = 10) -> Dict:
    windows = build_walk_forward_windows(total_bars=len(hourly_full), n_windows=n_windows)

    window_results: List[Dict] = []
    all_oos_trades: List[Trade] = []
    all_is_trades: List[Trade] = []

    for w in windows:
        start = w["start"]
        train_end = w["train_end"]
        test_end = w["test_end"]
        train_size = w["train_size"]

        segment = hourly_full.iloc[start:test_end].reset_index(drop=True)
        zones, trades = run_backtest(segment)

        is_trades = [t for t in trades if t.entry_idx < train_size]
        oos_trades = [t for t in trades if t.entry_idx >= train_size]

        is_metrics = compute_metrics(is_trades)
        oos_metrics = compute_metrics(oos_trades)

        all_is_trades.extend(is_trades)
        all_oos_trades.extend(oos_trades)

        window_results.append(
            {
                "window": w["window"],
                "train_period": {
                    "start": segment.iloc[0]["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                    "end": segment.iloc[train_size - 1]["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                },
                "test_period": {
                    "start": segment.iloc[train_size]["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                    "end": segment.iloc[-1]["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                },
                "zones_detected": len(zones),
                "in_sample": {k: round(v, 4) if isinstance(v, float) else v for k, v in is_metrics.items()},
                "out_of_sample": {k: round(v, 4) if isinstance(v, float) else v for k, v in oos_metrics.items()},
            }
        )

    agg_is = compute_metrics(all_is_trades)
    agg_oos = compute_metrics(all_oos_trades)

    results = {
        "config": {
            "n_windows": len(window_results),
            "train_ratio": 0.6,
            "test_ratio": 0.4,
            "rr_ratio": RR_RATIO,
            "spread_pips": SPREAD_PIPS,
            "entry_rule": "next_bar_open",
            "same_bar_sl_tp_priority": "sl",
        },
        "data": {
            "bars_1h": len(hourly_full),
            "start": hourly_full.iloc[0]["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
            "end": hourly_full.iloc[-1]["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
        },
        "windows": window_results,
        "aggregate": {
            "in_sample": {k: round(v, 4) if isinstance(v, float) else v for k, v in agg_is.items()},
            "out_of_sample": {k: round(v, 4) if isinstance(v, float) else v for k, v in agg_oos.items()},
        },
    }
    return results


def print_summary_table(results: Dict) -> None:
    print("\nFVG Barrier Walk-Forward (OOS)")
    print("-" * 88)
    header = f"{'Win':>3}  {'Test Period':<35} {'Trades':>6} {'WR%':>7} {'PnL(pips)':>11} {'PF':>7}"
    print(header)
    print("-" * 88)

    for window in results["windows"]:
        oos = window["out_of_sample"]
        period = f"{window['test_period']['start']} -> {window['test_period']['end']}"
        row = (
            f"{window['window']:>3}  "
            f"{period:<35} "
            f"{oos['trades']:>6} "
            f"{oos['win_rate']:>7.2f} "
            f"{oos['pnl_pips']:>11.2f} "
            f"{fmt_pf(oos['profit_factor']):>7}"
        )
        print(row)

    print("-" * 88)
    agg = results["aggregate"]["out_of_sample"]
    print(
        f"ALL  {'Aggregate OOS':<35} "
        f"{agg['trades']:>6} "
        f"{agg['win_rate']:>7.2f} "
        f"{agg['pnl_pips']:>11.2f} "
        f"{fmt_pf(agg['profit_factor']):>7}"
    )
    print("-" * 88)


def main() -> None:
    parser = argparse.ArgumentParser(description="FVG Barrier walk-forward backtest.")
    parser.add_argument("--input", default="tickdata/EURUSD_1H_2Y.csv", help="Input 1H CSV")
    parser.add_argument("--output", default="fvg_barrier_results.json", help="Output JSON file")
    parser.add_argument("--windows", type=int, default=10, help="Number of rolling windows")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    hourly = load_hourly_data(input_path)
    results = run_walk_forward(hourly, n_windows=args.windows)

    print_summary_table(results)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
