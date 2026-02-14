#!/usr/bin/env python3
"""
FVG Lava backtest (fractal multi-timeframe).

Core idea:
- Detect FVG zones on Daily and 4H.
- Use 1H bars as execution timeframe.
- Trade "wick burns" where price wicks into higher-TF FVGs but body closes back outside.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

PIP_SIZE = 0.0001
SPREAD = 0.0001  # 1 pip spread, applied on entry
RR_RATIO = 2.0
MAX_HOLD_BARS = 48


@dataclass
class FVGZone:
    zone_id: str
    timeframe: str  # 1D | 4H
    direction: str  # bullish | bearish
    lower: float
    upper: float
    created_at: str
    active_from: pd.Timestamp
    invalidated_at: Optional[pd.Timestamp] = None


@dataclass
class SignalCandidate:
    side: str  # long | short
    tier: str  # Tier 1 | Tier 2 | Tier 3
    tier_rank: int
    sl: float
    penetration: float
    daily_zone_ids: List[str]
    h4_zone_ids: List[str]


@dataclass
class Trade:
    trade_id: int
    side: str
    tier: str
    signal_time: str
    entry_time: str
    exit_time: str
    entry_price: float
    sl: float
    tp: float
    exit_price: float
    outcome: str  # win | loss | time_exit
    pnl_pips: float
    hold_bars: int
    daily_zone_ids: List[str]
    h4_zone_ids: List[str]
    entry_idx: int


def load_hourly_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    required = {"timestamp", "open", "high", "low", "close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    return df


def resample_ohlc(hourly: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    if "volume" in hourly.columns:
        agg["volume"] = "sum"
    if "ticks" in hourly.columns:
        agg["ticks"] = "sum"

    resampled = (
        hourly.set_index("timestamp")
        .resample(rule)
        .agg(agg)
        .dropna(subset=["open", "high", "low", "close"])
    )
    return resampled


def detect_fvg_zones(df: pd.DataFrame, timeframe: str, bar_delta: pd.Timedelta) -> List[FVGZone]:
    zones: List[FVGZone] = []
    zone_number = 1

    for i in range(2, len(df)):
        candle1 = df.iloc[i - 2]
        candle3 = df.iloc[i]
        created_at = df.index[i]
        active_from = created_at + bar_delta

        # Bullish FVG: candle1.high < candle3.low
        if candle1["high"] < candle3["low"]:
            zones.append(
                FVGZone(
                    zone_id=f"{timeframe}-{zone_number}",
                    timeframe=timeframe,
                    direction="bullish",
                    lower=float(candle1["high"]),
                    upper=float(candle3["low"]),
                    created_at=created_at.isoformat(),
                    active_from=active_from,
                )
            )
            zone_number += 1

        # Bearish FVG: candle1.low > candle3.high
        if candle1["low"] > candle3["high"]:
            zones.append(
                FVGZone(
                    zone_id=f"{timeframe}-{zone_number}",
                    timeframe=timeframe,
                    direction="bearish",
                    lower=float(candle3["high"]),
                    upper=float(candle1["low"]),
                    created_at=created_at.isoformat(),
                    active_from=active_from,
                )
            )
            zone_number += 1

    return zones


def is_zone_active(zone: FVGZone, ts: pd.Timestamp) -> bool:
    return zone.active_from <= ts and zone.invalidated_at is None


def zones_overlap(zone_a: FVGZone, zone_b: FVGZone) -> bool:
    return max(zone_a.lower, zone_b.lower) <= min(zone_a.upper, zone_b.upper)


def is_long_lava_burn(bar_open: float, bar_close: float, bar_low: float, zone_upper: float) -> bool:
    body_above_zone = min(bar_open, bar_close) > zone_upper
    wick_into_zone = bar_low <= zone_upper
    return body_above_zone and wick_into_zone


def is_short_lava_burn(bar_open: float, bar_close: float, bar_high: float, zone_lower: float) -> bool:
    body_below_zone = max(bar_open, bar_close) < zone_lower
    wick_into_zone = bar_high >= zone_lower
    return body_below_zone and wick_into_zone


def build_long_candidate(
    daily_zones: Sequence[FVGZone],
    h4_zones: Sequence[FVGZone],
    bar_low: float,
) -> Optional[SignalCandidate]:
    if daily_zones and h4_zones:
        overlap_pairs = [(d, h) for d in daily_zones for h in h4_zones if zones_overlap(d, h)]
        if overlap_pairs:
            # Pick pair with highest near edge (closest support test from above).
            daily_zone, h4_zone = max(overlap_pairs, key=lambda pair: max(pair[0].upper, pair[1].upper))
            near_edge = max(daily_zone.upper, h4_zone.upper)
            return SignalCandidate(
                side="long",
                tier="Tier 3",
                tier_rank=3,
                sl=min(daily_zone.lower, h4_zone.lower),
                penetration=max(0.0, near_edge - bar_low),
                daily_zone_ids=[daily_zone.zone_id],
                h4_zone_ids=[h4_zone.zone_id],
            )

    if daily_zones:
        zone = max(daily_zones, key=lambda z: z.upper)
        return SignalCandidate(
            side="long",
            tier="Tier 1",
            tier_rank=2,
            sl=zone.lower,
            penetration=max(0.0, zone.upper - bar_low),
            daily_zone_ids=[zone.zone_id],
            h4_zone_ids=[],
        )

    if h4_zones:
        zone = max(h4_zones, key=lambda z: z.upper)
        return SignalCandidate(
            side="long",
            tier="Tier 2",
            tier_rank=1,
            sl=zone.lower,
            penetration=max(0.0, zone.upper - bar_low),
            daily_zone_ids=[],
            h4_zone_ids=[zone.zone_id],
        )

    return None


def build_short_candidate(
    daily_zones: Sequence[FVGZone],
    h4_zones: Sequence[FVGZone],
    bar_high: float,
) -> Optional[SignalCandidate]:
    if daily_zones and h4_zones:
        overlap_pairs = [(d, h) for d in daily_zones for h in h4_zones if zones_overlap(d, h)]
        if overlap_pairs:
            # Pick pair with lowest near edge (closest resistance test from below).
            daily_zone, h4_zone = min(overlap_pairs, key=lambda pair: min(pair[0].lower, pair[1].lower))
            near_edge = min(daily_zone.lower, h4_zone.lower)
            return SignalCandidate(
                side="short",
                tier="Tier 3",
                tier_rank=3,
                sl=max(daily_zone.upper, h4_zone.upper),
                penetration=max(0.0, bar_high - near_edge),
                daily_zone_ids=[daily_zone.zone_id],
                h4_zone_ids=[h4_zone.zone_id],
            )

    if daily_zones:
        zone = min(daily_zones, key=lambda z: z.lower)
        return SignalCandidate(
            side="short",
            tier="Tier 1",
            tier_rank=2,
            sl=zone.upper,
            penetration=max(0.0, bar_high - zone.lower),
            daily_zone_ids=[zone.zone_id],
            h4_zone_ids=[],
        )

    if h4_zones:
        zone = min(h4_zones, key=lambda z: z.lower)
        return SignalCandidate(
            side="short",
            tier="Tier 2",
            tier_rank=1,
            sl=zone.upper,
            penetration=max(0.0, bar_high - zone.lower),
            daily_zone_ids=[],
            h4_zone_ids=[zone.zone_id],
        )

    return None


def pick_candidate(candidates: Sequence[SignalCandidate]) -> Optional[SignalCandidate]:
    if not candidates:
        return None
    return max(candidates, key=lambda c: (c.tier_rank, c.penetration))


def simulate_exit(
    hourly: pd.DataFrame,
    entry_idx: int,
    side: str,
    sl: float,
    tp: float,
    max_hold_bars: int = MAX_HOLD_BARS,
) -> Tuple[int, float, str]:
    end_idx = min(len(hourly) - 1, entry_idx + max_hold_bars - 1)

    for idx in range(entry_idx, end_idx + 1):
        bar = hourly.iloc[idx]
        high = float(bar["high"])
        low = float(bar["low"])

        if side == "long":
            sl_hit = low <= sl
            tp_hit = high >= tp
        else:
            sl_hit = high >= sl
            tp_hit = low <= tp

        # Worst-case assumption if both hit in same bar: stop first.
        if sl_hit and tp_hit:
            return idx, sl, "loss"
        if sl_hit:
            return idx, sl, "loss"
        if tp_hit:
            return idx, tp, "win"

    exit_price = float(hourly.iloc[end_idx]["close"])
    return end_idx, exit_price, "time_exit"


def run_backtest(hourly: pd.DataFrame) -> Tuple[List[FVGZone], List[Trade]]:
    daily = resample_ohlc(hourly, "1D")
    h4 = resample_ohlc(hourly, "4h")

    zones_daily = detect_fvg_zones(daily, timeframe="1D", bar_delta=pd.Timedelta(days=1))
    zones_h4 = detect_fvg_zones(h4, timeframe="4H", bar_delta=pd.Timedelta(hours=4))
    zones = zones_daily + zones_h4

    trades: List[Trade] = []
    trade_id = 1

    for i in range(len(hourly) - 1):
        bar = hourly.iloc[i]
        ts = pd.Timestamp(bar["timestamp"])
        bar_open = float(bar["open"])
        bar_high = float(bar["high"])
        bar_low = float(bar["low"])
        bar_close = float(bar["close"])

        active_zones = [zone for zone in zones if is_zone_active(zone, ts)]

        if active_zones:
            long_daily: List[FVGZone] = []
            long_h4: List[FVGZone] = []
            short_daily: List[FVGZone] = []
            short_h4: List[FVGZone] = []

            for zone in active_zones:
                if zone.direction == "bullish":
                    if is_long_lava_burn(bar_open, bar_close, bar_low, zone.upper):
                        if zone.timeframe == "1D":
                            long_daily.append(zone)
                        else:
                            long_h4.append(zone)
                else:
                    if is_short_lava_burn(bar_open, bar_close, bar_high, zone.lower):
                        if zone.timeframe == "1D":
                            short_daily.append(zone)
                        else:
                            short_h4.append(zone)

            candidates: List[SignalCandidate] = []
            long_candidate = build_long_candidate(long_daily, long_h4, bar_low)
            short_candidate = build_short_candidate(short_daily, short_h4, bar_high)
            if long_candidate:
                candidates.append(long_candidate)
            if short_candidate:
                candidates.append(short_candidate)

            chosen = pick_candidate(candidates)
            if chosen:
                entry_idx = i + 1
                entry_bar = hourly.iloc[entry_idx]
                entry_time = pd.Timestamp(entry_bar["timestamp"])
                raw_entry_open = float(entry_bar["open"])

                if chosen.side == "long":
                    entry_price = raw_entry_open + SPREAD
                    sl = chosen.sl
                    risk = entry_price - sl
                    if risk <= 0:
                        chosen = None
                    else:
                        tp = entry_price + RR_RATIO * risk
                else:
                    entry_price = raw_entry_open - SPREAD
                    sl = chosen.sl
                    risk = sl - entry_price
                    if risk <= 0:
                        chosen = None
                    else:
                        tp = entry_price - RR_RATIO * risk

                if chosen:
                    exit_idx, exit_price, outcome = simulate_exit(
                        hourly=hourly,
                        entry_idx=entry_idx,
                        side=chosen.side,
                        sl=sl,
                        tp=tp,
                        max_hold_bars=MAX_HOLD_BARS,
                    )
                    exit_time = pd.Timestamp(hourly.iloc[exit_idx]["timestamp"])

                    if chosen.side == "long":
                        pnl_pips = (exit_price - entry_price) / PIP_SIZE
                    else:
                        pnl_pips = (entry_price - exit_price) / PIP_SIZE

                    trades.append(
                        Trade(
                            trade_id=trade_id,
                            side=chosen.side,
                            tier=chosen.tier,
                            signal_time=ts.isoformat(),
                            entry_time=entry_time.isoformat(),
                            exit_time=exit_time.isoformat(),
                            entry_price=entry_price,
                            sl=sl,
                            tp=tp,
                            exit_price=exit_price,
                            outcome=outcome,
                            pnl_pips=pnl_pips,
                            hold_bars=exit_idx - entry_idx + 1,
                            daily_zone_ids=chosen.daily_zone_ids,
                            h4_zone_ids=chosen.h4_zone_ids,
                            entry_idx=entry_idx,
                        )
                    )
                    trade_id += 1

        # Invalidate zones if price body closes through far edge.
        for zone in active_zones:
            if zone.direction == "bullish" and bar_close <= zone.lower:
                zone.invalidated_at = ts
            elif zone.direction == "bearish" and bar_close >= zone.upper:
                zone.invalidated_at = ts

    return zones, trades


def compute_metrics(trades: Sequence[Trade]) -> Dict[str, float]:
    if not trades:
        return {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "breakeven": 0,
            "win_rate": 0.0,
            "total_pnl_pips": 0.0,
            "profit_factor": 0.0,
            "avg_win_loss_ratio": 0.0,
            "avg_win_pips": 0.0,
            "avg_loss_pips": 0.0,
            "gross_profit_pips": 0.0,
            "gross_loss_pips": 0.0,
        }

    pnl = [t.pnl_pips for t in trades]
    wins = [x for x in pnl if x > 0]
    losses = [x for x in pnl if x < 0]
    breakeven = len([x for x in pnl if x == 0])

    gross_profit = sum(wins)
    gross_loss = -sum(losses)
    avg_win = gross_profit / len(wins) if wins else 0.0
    avg_loss = gross_loss / len(losses) if losses else 0.0

    return {
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "breakeven": breakeven,
        "win_rate": (len(wins) / len(trades)) * 100.0,
        "total_pnl_pips": sum(pnl),
        "profit_factor": (gross_profit / gross_loss) if gross_loss > 0 else 0.0,
        "avg_win_loss_ratio": (avg_win / avg_loss) if avg_loss > 0 else 0.0,
        "avg_win_pips": avg_win,
        "avg_loss_pips": avg_loss,
        "gross_profit_pips": gross_profit,
        "gross_loss_pips": gross_loss,
    }


def print_metrics(title: str, metrics: Dict[str, float]) -> None:
    print(f"\n{title}")
    print(f"  Total Trades:       {int(metrics['total_trades'])}")
    print(f"  Win Rate:           {metrics['win_rate']:.2f}%")
    print(f"  Total PnL (pips):   {metrics['total_pnl_pips']:.2f}")
    print(f"  Profit Factor:      {metrics['profit_factor']:.2f}")
    print(f"  Avg Win/Loss Ratio: {metrics['avg_win_loss_ratio']:.2f}")


def build_walk_forward_windows(
    hourly: pd.DataFrame,
    train_months: int = 6,
    test_months: int = 2,
    step_months: int = 2,
) -> List[Dict[str, pd.Timestamp]]:
    if hourly.empty:
        return []

    first_ts = pd.Timestamp(hourly.iloc[0]["timestamp"])
    last_ts = pd.Timestamp(hourly.iloc[-1]["timestamp"])

    windows: List[Dict[str, pd.Timestamp]] = []
    cursor = first_ts
    window_num = 1

    while True:
        train_end = cursor + pd.DateOffset(months=train_months)
        test_end = train_end + pd.DateOffset(months=test_months)
        if test_end > last_ts:
            break

        windows.append(
            {
                "window": window_num,
                "start": cursor,
                "train_end": train_end,
                "test_end": test_end,
            }
        )
        window_num += 1
        cursor = cursor + pd.DateOffset(months=step_months)
        if cursor >= last_ts:
            break

    return windows


def run_walk_forward(
    hourly_full: pd.DataFrame,
    train_months: int = 6,
    test_months: int = 2,
    step_months: int = 2,
) -> Dict:
    windows = build_walk_forward_windows(hourly_full, train_months, test_months, step_months)

    all_oos_trades: List[Trade] = []
    window_results: List[Dict] = []

    for window in windows:
        start = window["start"]
        train_end = window["train_end"]
        test_end = window["test_end"]

        segment = hourly_full[
            (hourly_full["timestamp"] >= start) & (hourly_full["timestamp"] < test_end)
        ].reset_index(drop=True)

        if len(segment) < 3:
            continue

        _, trades = run_backtest(segment)
        train_cutoff = pd.Timestamp(train_end)
        oos_trades = [t for t in trades if pd.Timestamp(t.entry_time) >= train_cutoff]
        oos_metrics = compute_metrics(oos_trades)
        all_oos_trades.extend(oos_trades)

        window_results.append(
            {
                "window": window["window"],
                "train_period": {
                    "start": start.strftime("%Y-%m-%d %H:%M:%S"),
                    "end": train_end.strftime("%Y-%m-%d %H:%M:%S"),
                },
                "test_period": {
                    "start": train_end.strftime("%Y-%m-%d %H:%M:%S"),
                    "end": test_end.strftime("%Y-%m-%d %H:%M:%S"),
                },
                "out_of_sample": {
                    k: round(v, 4) if isinstance(v, float) else v for k, v in oos_metrics.items()
                },
            }
        )

    aggregate_oos = compute_metrics(all_oos_trades)
    return {
        "config": {
            "train_months": train_months,
            "test_months": test_months,
            "step_months": step_months,
            "entry_rule": "next_bar_open",
            "spread": SPREAD,
            "rr_ratio": RR_RATIO,
            "max_hold_bars": MAX_HOLD_BARS,
            "same_bar_sl_tp_priority": "sl",
        },
        "windows": window_results,
        "aggregate_oos": {k: round(v, 4) if isinstance(v, float) else v for k, v in aggregate_oos.items()},
    }


def print_walk_forward_summary(results: Dict) -> None:
    print("\nWalk-Forward OOS (6m train / 2m test)")
    print("-" * 88)
    print(f"{'Win':>3}  {'Test Period':<35} {'Trades':>6} {'WR%':>7} {'PnL(pips)':>11} {'PF':>7}")
    print("-" * 88)

    for window in results["windows"]:
        oos = window["out_of_sample"]
        period = f"{window['test_period']['start']} -> {window['test_period']['end']}"
        print(
            f"{window['window']:>3}  "
            f"{period:<35} "
            f"{int(oos['total_trades']):>6} "
            f"{oos['win_rate']:>7.2f} "
            f"{oos['total_pnl_pips']:>11.2f} "
            f"{oos['profit_factor']:>7.2f}"
        )

    print("-" * 88)
    agg = results["aggregate_oos"]
    print(
        f"ALL  {'Aggregate OOS':<35} "
        f"{int(agg['total_trades']):>6} "
        f"{agg['win_rate']:>7.2f} "
        f"{agg['total_pnl_pips']:>11.2f} "
        f"{agg['profit_factor']:>7.2f}"
    )
    print("-" * 88)


def main() -> None:
    parser = argparse.ArgumentParser(description="FVG Lava backtest (fractal multi-timeframe).")
    parser.add_argument("--input", default="tickdata/EURUSD_1H_2Y.csv", help="Input 1H OHLC CSV path")
    parser.add_argument("--trades-output", default="fvg_lava_trades.json", help="Path to save trades JSON")
    parser.add_argument("--train-months", type=int, default=6, help="Walk-forward train months")
    parser.add_argument("--test-months", type=int, default=2, help="Walk-forward OOS months")
    parser.add_argument("--step-months", type=int, default=2, help="Walk-forward step months")
    args = parser.parse_args()

    input_path = Path(args.input)
    trades_output = Path(args.trades_output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    hourly = load_hourly_data(input_path)
    print(f"Loaded {len(hourly)} 1H bars from {hourly.iloc[0]['timestamp']} to {hourly.iloc[-1]['timestamp']}")

    zones, trades = run_backtest(hourly)

    print(f"\nDetected zones: {len(zones)} (Daily: {len([z for z in zones if z.timeframe == '1D'])}, "
          f"4H: {len([z for z in zones if z.timeframe == '4H'])})")

    overall_metrics = compute_metrics(trades)
    print_metrics("Overall Metrics", overall_metrics)

    tier_order = ["Tier 1", "Tier 2", "Tier 3"]
    for tier in tier_order:
        tier_trades = [t for t in trades if t.tier == tier]
        print_metrics(f"{tier} Metrics", compute_metrics(tier_trades))

    for side in ["long", "short"]:
        side_trades = [t for t in trades if t.side == side]
        print_metrics(f"{side.capitalize()} Metrics", compute_metrics(side_trades))

    with trades_output.open("w", encoding="utf-8") as f:
        json.dump([asdict(t) for t in trades], f, indent=2)
    print(f"\nSaved {len(trades)} trades to {trades_output}")

    wf_results = run_walk_forward(
        hourly_full=hourly,
        train_months=args.train_months,
        test_months=args.test_months,
        step_months=args.step_months,
    )
    print_walk_forward_summary(wf_results)


if __name__ == "__main__":
    main()
