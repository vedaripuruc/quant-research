#!/usr/bin/env python3
"""
Combined FVG + ECVT walk-forward backtest.

Rules implemented from user spec:
- Daily FVG zones detected from 1H-resampled daily candles.
- FVG zones partially fill (shrink) on 1H bar penetration and die below 1 pip width.
- ECVT entropy collapse + volume spike trigger on 1H.
- Combined strategy: entropy+volume+FVG proximity (20 pips) with zone-direction entries.
- ECVT-only strategy: entropy+volume trigger with ECVT direction.
- Entry at next bar open; SL/TP on high/low; same-bar SL+TP => SL.
- 1 pip spread deducted per trade.
- 10 rolling walk-forward windows (60/40 train/test), OOS metrics reported.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ecvt_fast import compute_markov_entropy_fast, states_from_ohlcv

PIP_SIZE = 0.0001
SPREAD_PIPS = 1.0

N_WINDOWS = 10
TRAIN_RATIO = 0.6
TEST_RATIO = 0.4

ENTROPY_WINDOW = 120
ENTROPY_LOOKBACK = 500
ENTROPY_PERCENTILE = 0.05

VOLUME_LOOKBACK = 120
VOLUME_PERCENTILE = 0.90

TRAIL_DIR_WINDOW = 30

FVG_PROXIMITY_PIPS = 20.0
WIDE_ZONE_SL_PIPS = 50.0
FALLBACK_SL_PIPS = 25.0
RR_RATIO = 2.0


@dataclass
class FVGZone:
    zone_id: int
    direction: str  # bullish | bearish
    lower: float
    upper: float
    active_from: pd.Timestamp


def load_hourly_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df.columns = [str(c).strip().lower() for c in df.columns]
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def build_walk_forward_windows(
    total_bars: int,
    n_windows: int = N_WINDOWS,
    train_ratio: float = TRAIN_RATIO,
    test_ratio: float = TEST_RATIO,
) -> List[Dict[str, int]]:
    if total_bars <= 0:
        raise ValueError("No bars available")

    ratio = train_ratio / test_ratio
    test_size = int(round(total_bars / (n_windows + ratio)))
    if test_size <= 0:
        raise ValueError("Computed test size is zero")

    train_size = total_bars - (n_windows * test_size)
    if train_size <= 0:
        raise ValueError("Computed train size is zero")

    windows: List[Dict[str, int]] = []
    for idx in range(n_windows):
        start = idx * test_size
        train_end = start + train_size
        test_end = train_end + test_size
        if test_end > total_bars:
            break
        windows.append(
            {
                "window": idx + 1,
                "start": start,
                "train_end": train_end,
                "test_end": test_end,
                "train_size": train_size,
                "test_size": test_size,
            }
        )

    if len(windows) == 0:
        raise ValueError("No valid walk-forward windows produced")

    return windows


def detect_daily_fvg_zones(hourly: pd.DataFrame) -> List[FVGZone]:
    daily = (
        hourly.set_index("timestamp")
        .resample("1D")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna(subset=["open", "high", "low", "close"])
    )

    zones: List[FVGZone] = []
    zone_id = 1

    for i in range(2, len(daily)):
        left = daily.iloc[i - 2]
        right = daily.iloc[i]
        creation_day = daily.index[i]

        # Bullish FVG: left high < right low => gap [left.high, right.low]
        if float(left["high"]) < float(right["low"]):
            zones.append(
                FVGZone(
                    zone_id=zone_id,
                    direction="bullish",
                    lower=float(left["high"]),
                    upper=float(right["low"]),
                    active_from=creation_day + pd.Timedelta(days=1),
                )
            )
            zone_id += 1

        # Bearish FVG: left low > right high => gap [right.high, left.low]
        if float(left["low"]) > float(right["high"]):
            zones.append(
                FVGZone(
                    zone_id=zone_id,
                    direction="bearish",
                    lower=float(right["high"]),
                    upper=float(left["low"]),
                    active_from=creation_day + pd.Timedelta(days=1),
                )
            )
            zone_id += 1

    zones.sort(key=lambda z: z.active_from)
    return zones


def update_zone_partial_fill(zone: FVGZone, bar_high: float, bar_low: float) -> bool:
    """
    Apply partial fill shrink logic.
    Returns True if zone remains active, False if it dies (< 1 pip width).
    """
    if zone.direction == "bullish":
        # Bullish FVG [lower, upper]: upper shrinks down when price dips in.
        if bar_low < zone.upper:
            zone.upper = min(zone.upper, max(zone.lower, bar_low))
    else:
        # Bearish FVG [lower, upper]: lower shrinks up when price pushes in.
        if bar_high > zone.lower:
            zone.lower = max(zone.lower, min(zone.upper, bar_high))

    return (zone.upper - zone.lower) >= PIP_SIZE


def prepare_ecvt_features(hourly: pd.DataFrame) -> Dict[str, np.ndarray]:
    states = states_from_ohlcv(hourly)
    entropy = compute_markov_entropy_fast(states, window=ENTROPY_WINDOW, n_states=15)

    entropy_threshold = (
        pd.Series(entropy)
        .rolling(window=ENTROPY_LOOKBACK, min_periods=ENTROPY_LOOKBACK)
        .quantile(ENTROPY_PERCENTILE)
        .to_numpy()
    )

    volume_threshold = (
        hourly["volume"]
        .rolling(window=VOLUME_LOOKBACK, min_periods=VOLUME_LOOKBACK)
        .quantile(VOLUME_PERCENTILE)
        .to_numpy()
    )

    trail_return = hourly["close"].pct_change(TRAIL_DIR_WINDOW).to_numpy()
    direction = np.sign(trail_return)
    direction[~np.isfinite(direction)] = 0
    direction = direction.astype(np.int8)

    entropy_collapse = np.isfinite(entropy_threshold) & (entropy < entropy_threshold)
    volume_spike = np.isfinite(volume_threshold) & (hourly["volume"].to_numpy() > volume_threshold)

    return {
        "entropy": entropy,
        "entropy_threshold": entropy_threshold,
        "volume_threshold": volume_threshold,
        "entropy_collapse": entropy_collapse,
        "volume_spike": volume_spike,
        "direction": direction,
    }


def find_combined_signal(
    close_price: float,
    active_zones: List[FVGZone],
    proximity_pips: float = FVG_PROXIMITY_PIPS,
) -> Optional[Tuple[int, FVGZone, float]]:
    best_bull: Optional[Tuple[float, FVGZone]] = None
    best_bear: Optional[Tuple[float, FVGZone]] = None

    for zone in active_zones:
        if zone.direction == "bullish":
            barrier = zone.upper
            dist = abs(close_price - barrier) / PIP_SIZE
            if best_bull is None or dist < best_bull[0]:
                best_bull = (dist, zone)
        else:
            barrier = zone.lower
            dist = abs(close_price - barrier) / PIP_SIZE
            if best_bear is None or dist < best_bear[0]:
                best_bear = (dist, zone)

    bull_ok = best_bull is not None and best_bull[0] <= proximity_pips
    bear_ok = best_bear is not None and best_bear[0] <= proximity_pips

    if not bull_ok and not bear_ok:
        return None

    if bull_ok and bear_ok:
        bull_dist = best_bull[0]  # type: ignore[index]
        bear_dist = best_bear[0]  # type: ignore[index]
        if math.isclose(bull_dist, bear_dist, rel_tol=0.0, abs_tol=1e-9):
            return None
        if bull_dist < bear_dist:
            return 1, best_bull[1], bull_dist  # type: ignore[index]
        return -1, best_bear[1], bear_dist  # type: ignore[index]

    if bull_ok:
        return 1, best_bull[1], best_bull[0]  # type: ignore[index]

    return -1, best_bear[1], best_bear[0]  # type: ignore[index]


def build_zone_or_fallback_levels(entry_price: float, side: int, zone: FVGZone) -> Tuple[float, float, bool]:
    if side == 1:
        zone_sl = zone.lower
        zone_risk_pips = (entry_price - zone_sl) / PIP_SIZE
        if zone_risk_pips <= 0 or zone_risk_pips > WIDE_ZONE_SL_PIPS:
            sl = entry_price - FALLBACK_SL_PIPS * PIP_SIZE
            tp = entry_price + RR_RATIO * FALLBACK_SL_PIPS * PIP_SIZE
            return sl, tp, True
        sl = zone_sl
        tp = entry_price + RR_RATIO * (entry_price - sl)
        return sl, tp, False

    zone_sl = zone.upper
    zone_risk_pips = (zone_sl - entry_price) / PIP_SIZE
    if zone_risk_pips <= 0 or zone_risk_pips > WIDE_ZONE_SL_PIPS:
        sl = entry_price + FALLBACK_SL_PIPS * PIP_SIZE
        tp = entry_price - RR_RATIO * FALLBACK_SL_PIPS * PIP_SIZE
        return sl, tp, True
    sl = zone_sl
    tp = entry_price - RR_RATIO * (sl - entry_price)
    return sl, tp, False


def build_fixed_levels(entry_price: float, side: int) -> Tuple[float, float]:
    if side == 1:
        return (
            entry_price - FALLBACK_SL_PIPS * PIP_SIZE,
            entry_price + RR_RATIO * FALLBACK_SL_PIPS * PIP_SIZE,
        )
    return (
        entry_price + FALLBACK_SL_PIPS * PIP_SIZE,
        entry_price - RR_RATIO * FALLBACK_SL_PIPS * PIP_SIZE,
    )


def simulate_strategy(
    hourly: pd.DataFrame,
    test_start_idx: int,
    strategy: str,
) -> List[Dict]:
    if strategy not in {"ecvt_only", "fvg_ecvt_combined"}:
        raise ValueError(f"Unknown strategy: {strategy}")

    features = prepare_ecvt_features(hourly)
    entropy_collapse = features["entropy_collapse"]
    volume_spike = features["volume_spike"]
    ecvt_direction = features["direction"]

    opens = hourly["open"].to_numpy()
    highs = hourly["high"].to_numpy()
    lows = hourly["low"].to_numpy()
    closes = hourly["close"].to_numpy()
    timestamps = hourly["timestamp"].to_numpy()

    all_zones = detect_daily_fvg_zones(hourly)
    inactive_zone_idx = 0
    active_zones: List[FVGZone] = []

    trades: List[Dict] = []
    position: Optional[Dict] = None

    for i in range(len(hourly)):
        ts = pd.Timestamp(timestamps[i])
        bar_high = float(highs[i])
        bar_low = float(lows[i])

        # Activate any newly valid zones.
        while inactive_zone_idx < len(all_zones) and all_zones[inactive_zone_idx].active_from <= ts:
            z = all_zones[inactive_zone_idx]
            active_zones.append(
                FVGZone(
                    zone_id=z.zone_id,
                    direction=z.direction,
                    lower=z.lower,
                    upper=z.upper,
                    active_from=z.active_from,
                )
            )
            inactive_zone_idx += 1

        # Zones shrink as this bar trades through them.
        if active_zones:
            kept: List[FVGZone] = []
            for zone in active_zones:
                if update_zone_partial_fill(zone, bar_high=bar_high, bar_low=bar_low):
                    kept.append(zone)
            active_zones = kept

        # Manage exits first on current bar.
        if position is not None:
            side = int(position["side"])
            sl = float(position["sl"])
            tp = float(position["tp"])

            if side == 1:
                sl_hit = bar_low <= sl
                tp_hit = bar_high >= tp
            else:
                sl_hit = bar_high >= sl
                tp_hit = bar_low <= tp

            exit_price: Optional[float] = None
            exit_reason: Optional[str] = None

            if sl_hit and tp_hit:
                exit_price = sl
                exit_reason = "sl_tp_same_bar_sl"
            elif sl_hit:
                exit_price = sl
                exit_reason = "stop_loss"
            elif tp_hit:
                exit_price = tp
                exit_reason = "take_profit"
            elif i == len(hourly) - 1:
                exit_price = float(closes[i])
                exit_reason = "end_of_data"

            if exit_price is not None and exit_reason is not None:
                entry_price = float(position["entry_price"])
                gross_pips = ((exit_price - entry_price) / PIP_SIZE) * side
                pnl_pips = gross_pips - SPREAD_PIPS

                trades.append(
                    {
                        "strategy": strategy,
                        "side": side,
                        "signal_idx": int(position["signal_idx"]),
                        "entry_idx": int(position["entry_idx"]),
                        "exit_idx": i,
                        "entry_time": str(position["entry_time"]),
                        "exit_time": str(ts),
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "sl": sl,
                        "tp": tp,
                        "pnl_pips": pnl_pips,
                        "fallback_sl": bool(position.get("fallback_sl", False)),
                        "zone_id": position.get("zone_id"),
                        "exit_reason": exit_reason,
                    }
                )
                position = None

        # Entry decisions are made on bar i for entry at i+1 open.
        if position is not None:
            continue
        if i >= len(hourly) - 1:
            continue
        if i < test_start_idx:
            continue

        if not (bool(entropy_collapse[i]) and bool(volume_spike[i])):
            continue

        if strategy == "ecvt_only":
            side = int(ecvt_direction[i])
            if side == 0:
                continue

            entry_idx = i + 1
            entry_price = float(opens[entry_idx])
            sl, tp = build_fixed_levels(entry_price=entry_price, side=side)

            position = {
                "side": side,
                "signal_idx": i,
                "entry_idx": entry_idx,
                "entry_time": pd.Timestamp(timestamps[entry_idx]),
                "entry_price": entry_price,
                "sl": sl,
                "tp": tp,
                "fallback_sl": True,
                "zone_id": None,
            }
            continue

        # Combined strategy.
        signal = find_combined_signal(close_price=float(closes[i]), active_zones=active_zones)
        if signal is None:
            continue

        side, zone, _ = signal
        entry_idx = i + 1
        entry_price = float(opens[entry_idx])
        sl, tp, fallback_sl = build_zone_or_fallback_levels(entry_price=entry_price, side=side, zone=zone)

        position = {
            "side": side,
            "signal_idx": i,
            "entry_idx": entry_idx,
            "entry_time": pd.Timestamp(timestamps[entry_idx]),
            "entry_price": entry_price,
            "sl": sl,
            "tp": tp,
            "fallback_sl": fallback_sl,
            "zone_id": zone.zone_id,
        }

    return trades


def compute_metrics(trades: List[Dict]) -> Dict[str, Optional[float]]:
    if not trades:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "pnl_pips": 0.0,
            "profit_factor": 0.0,
            "gross_profit_pips": 0.0,
            "gross_loss_pips": 0.0,
        }

    pnl = np.array([float(t["pnl_pips"]) for t in trades], dtype=float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]

    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(-losses.sum()) if len(losses) else 0.0

    if gross_loss > 0:
        pf: Optional[float] = gross_profit / gross_loss
    elif gross_profit > 0:
        pf = None
    else:
        pf = 0.0

    return {
        "trades": int(len(trades)),
        "win_rate": float((pnl > 0).mean() * 100.0),
        "pnl_pips": float(pnl.sum()),
        "profit_factor": pf,
        "gross_profit_pips": gross_profit,
        "gross_loss_pips": gross_loss,
    }


def round_metrics(metrics: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    rounded: Dict[str, Optional[float]] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, np.integer)):
            rounded[key] = int(value)
        elif isinstance(value, (float, np.floating)):
            rounded[key] = round(float(value), 4)
        else:
            rounded[key] = value
    return rounded


def run_walk_forward(hourly: pd.DataFrame, strategy: str) -> Dict:
    windows = build_walk_forward_windows(len(hourly), n_windows=N_WINDOWS)
    window_results: List[Dict] = []
    aggregate_trades: List[Dict] = []

    for w in windows:
        start = w["start"]
        train_end = w["train_end"]
        test_end = w["test_end"]

        segment = hourly.iloc[start:test_end].reset_index(drop=True)
        local_test_start = train_end - start

        all_trades = simulate_strategy(
            hourly=segment,
            test_start_idx=local_test_start,
            strategy=strategy,
        )

        oos_trades = [t for t in all_trades if int(t["entry_idx"]) >= local_test_start]
        aggregate_trades.extend(oos_trades)

        metrics = round_metrics(compute_metrics(oos_trades))
        window_results.append(
            {
                "window": int(w["window"]),
                "test_period": {
                    "start": pd.Timestamp(segment.iloc[local_test_start]["timestamp"]).strftime("%Y-%m-%d %H:%M:%S"),
                    "end": pd.Timestamp(segment.iloc[-1]["timestamp"]).strftime("%Y-%m-%d %H:%M:%S"),
                },
                **metrics,
            }
        )

    aggregate = round_metrics(compute_metrics(aggregate_trades))
    return {"windows": window_results, "aggregate": aggregate}


def pf_label(value: Optional[float]) -> str:
    if value is None:
        return "inf"
    return f"{value:.2f}"


def print_comparison_table(ecvt_only: Dict, combined: Dict) -> None:
    print("\nFVG + ECVT Walk-Forward Comparison (OOS)")
    print("-" * 132)
    print(
        f"{'Win':>3}  {'Test Period':<35} "
        f"{'ECVT T':>6} {'ECVT WR%':>9} {'ECVT PnL':>10} {'ECVT PF':>8} "
        f"{'COMB T':>6} {'COMB WR%':>9} {'COMB PnL':>10} {'COMB PF':>8}"
    )
    print("-" * 132)

    ecvt_windows = ecvt_only["windows"]
    comb_windows = combined["windows"]
    n = min(len(ecvt_windows), len(comb_windows))

    for i in range(n):
        ew = ecvt_windows[i]
        cw = comb_windows[i]
        period = f"{ew['test_period']['start']} -> {ew['test_period']['end']}"
        print(
            f"{ew['window']:>3}  {period:<35} "
            f"{ew['trades']:>6} {ew['win_rate']:>9.2f} {ew['pnl_pips']:>10.2f} {pf_label(ew['profit_factor']):>8} "
            f"{cw['trades']:>6} {cw['win_rate']:>9.2f} {cw['pnl_pips']:>10.2f} {pf_label(cw['profit_factor']):>8}"
        )

    print("-" * 132)

    ea = ecvt_only["aggregate"]
    ca = combined["aggregate"]
    print(
        f"ALL  {'Aggregate OOS':<35} "
        f"{ea['trades']:>6} {ea['win_rate']:>9.2f} {ea['pnl_pips']:>10.2f} {pf_label(ea['profit_factor']):>8} "
        f"{ca['trades']:>6} {ca['win_rate']:>9.2f} {ca['pnl_pips']:>10.2f} {pf_label(ca['profit_factor']):>8}"
    )
    print("-" * 132)


def main() -> None:
    input_path = Path("tickdata/EURUSD_1H_2Y.csv")
    output_path = Path("fvg_ecvt_results.json")

    hourly = load_hourly_data(input_path)

    ecvt_only = run_walk_forward(hourly=hourly, strategy="ecvt_only")
    combined = run_walk_forward(hourly=hourly, strategy="fvg_ecvt_combined")

    ecvt_pnl = float(ecvt_only["aggregate"]["pnl_pips"])
    combined_pnl = float(combined["aggregate"]["pnl_pips"])

    results = {
        "ecvt_only": ecvt_only,
        "fvg_ecvt_combined": combined,
        "comparison": {
            "ecvt_pnl": round(ecvt_pnl, 4),
            "combined_pnl": round(combined_pnl, 4),
            "improvement_pips": round(combined_pnl - ecvt_pnl, 4),
        },
    }

    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print_comparison_table(ecvt_only=ecvt_only, combined=combined)
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
