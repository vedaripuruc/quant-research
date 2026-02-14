#!/usr/bin/env python3
"""
FVG Magnet - OFF-HOURS Formation Only.

Hypothesis: FVGs formed during Asian/London session (low liquidity)
are BETTER magnets because NY session fills them.

OFF-HOURS (UTC): 00:00-13:00 (Asian + London morning)
ON-HOURS (CME): 13:00-21:00 (NY session)
"""

from __future__ import annotations

import argparse
import gzip
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

PIP_SIZE = 0.0001
FILENAME_RE = re.compile(r"^(?P<symbol>[A-Z0-9]+)_BID_(?P<date>\d{4}-\d{2}-\d{2})_(?P<hour>\d{2})\.log\.gz$")

# Session definitions (UTC)
# OFF-HOURS: Asian + London morning (00:00-13:00 UTC)
# ON-HOURS: NY/CME session (13:00-21:00 UTC)
OFF_HOURS_START = 0
OFF_HOURS_END = 13


@dataclass
class FVG:
    fvg_type: str
    top: float
    bottom: float
    formed_at: datetime
    middle_high: float
    middle_low: float
    size_pips: float


@dataclass
class Trade:
    fvg_type: str
    formed_at: datetime
    entry_time: datetime
    entry_price: float
    sl: float
    tp: float
    exit_time: datetime
    exit_price: float
    outcome: str
    pnl_pips: float


def is_off_hours(dt: datetime) -> bool:
    """Check if datetime falls within off-hours (Asian/London)."""
    return OFF_HOURS_START <= dt.hour < OFF_HOURS_END


def parse_hour_from_filename(path: Path) -> Optional[datetime]:
    match = FILENAME_RE.match(path.name)
    if not match:
        return None
    date_str = match.group("date")
    hour_str = match.group("hour")
    return datetime.strptime(f"{date_str} {hour_str}", "%Y-%m-%d %H")


def build_tick_index(symbol_dir: Path) -> Dict[datetime, Path]:
    index: Dict[datetime, Path] = {}
    for path in symbol_dir.glob("*.log.gz"):
        ts = parse_hour_from_filename(path)
        if ts is not None:
            index[ts] = path
    return index


def iter_ticks(
    tick_index: Dict[datetime, Path],
    start_time: datetime,
    end_time: datetime,
) -> Iterable[Tuple[datetime, float]]:
    current = datetime(start_time.year, start_time.month, start_time.day, start_time.hour)
    end_hour = datetime(end_time.year, end_time.month, end_time.day, end_time.hour)
    while current <= end_hour:
        path = tick_index.get(current)
        if path and path.exists() and path.stat().st_size > 0:
            try:
                with gzip.open(path, "rt") as f:
                    while True:
                        try:
                            line = f.readline()
                        except OSError:
                            break
                        if not line:
                            break
                        parts = line.strip().split(",")
                        if len(parts) < 2:
                            continue
                        try:
                            ts_ms = int(parts[0])
                            price = float(parts[1])
                        except ValueError:
                            continue
                        ts = datetime.fromtimestamp(ts_ms / 1000.0, tz=None)
                        if ts < start_time:
                            continue
                        if ts > end_time:
                            return
                        yield ts, price
            except OSError:
                pass
        current += timedelta(hours=1)


def find_fvgs(df: pd.DataFrame, off_hours_only: bool = True) -> List[FVG]:
    """Find FVGs formed during off-hours only."""
    fvgs: List[FVG] = []
    for i in range(2, len(df)):
        left = df.iloc[i - 2]
        middle = df.iloc[i - 1]
        right = df.iloc[i]
        
        formed_at = pd.to_datetime(right["timestamp"]).to_pydatetime() + timedelta(hours=1)
        
        # Filter: only FVGs formed during off-hours
        if off_hours_only and not is_off_hours(formed_at):
            continue
        
        # Bullish FVG
        if left["high"] < right["low"]:
            bottom = float(left["high"])
            top = float(right["low"])
            size_pips = (top - bottom) / PIP_SIZE
            fvgs.append(
                FVG(
                    fvg_type="bullish",
                    top=top,
                    bottom=bottom,
                    formed_at=formed_at,
                    middle_high=float(middle["high"]),
                    middle_low=float(middle["low"]),
                    size_pips=size_pips,
                )
            )
        # Bearish FVG
        if left["low"] > right["high"]:
            bottom = float(right["high"])
            top = float(left["low"])
            size_pips = (top - bottom) / PIP_SIZE
            fvgs.append(
                FVG(
                    fvg_type="bearish",
                    top=top,
                    bottom=bottom,
                    formed_at=formed_at,
                    middle_high=float(middle["high"]),
                    middle_low=float(middle["low"]),
                    size_pips=size_pips,
                )
            )
    return fvgs


def backtest_fvg_magnet(
    df: pd.DataFrame,
    tick_index: Dict[datetime, Path],
    min_gap_pips: float,
    max_gap_pips: float,
    sl_buffer_pips: float = 2.0,
    rr_ratio: float = 2.0,
    max_hours: int = 48,
    off_hours_only: bool = True,
) -> List[Trade]:
    trades: List[Trade] = []
    fvgs = find_fvgs(df, off_hours_only=off_hours_only)

    for fvg in fvgs:
        if fvg.size_pips < min_gap_pips or fvg.size_pips > max_gap_pips:
            continue

        start_time = fvg.formed_at
        end_time = fvg.formed_at + timedelta(hours=max_hours)

        entry_time = None
        entry_price = None
        last_price = None

        for ts, price in iter_ticks(tick_index, start_time, end_time):
            if fvg.fvg_type == "bullish":
                if last_price is not None and last_price > fvg.top and price <= fvg.top:
                    entry_time = ts
                    entry_price = price
                    break
            else:
                if last_price is not None and last_price < fvg.bottom and price >= fvg.bottom:
                    entry_time = ts
                    entry_price = price
                    break
            last_price = price

        if entry_time is None or entry_price is None:
            trades.append(
                Trade(
                    fvg_type=fvg.fvg_type,
                    formed_at=fvg.formed_at,
                    entry_time=end_time,
                    entry_price=0.0,
                    sl=0.0,
                    tp=0.0,
                    exit_time=end_time,
                    exit_price=0.0,
                    outcome="no_touch",
                    pnl_pips=0.0,
                )
            )
            continue

        if fvg.fvg_type == "bullish":
            sl = fvg.bottom - sl_buffer_pips * PIP_SIZE
            risk = entry_price - sl
            tp = entry_price + rr_ratio * risk
        else:
            sl = fvg.top + sl_buffer_pips * PIP_SIZE
            risk = sl - entry_price
            tp = entry_price - rr_ratio * risk

        exit_time = entry_time
        exit_price = entry_price
        outcome = "timeout"

        for ts, price in iter_ticks(tick_index, entry_time, end_time):
            if fvg.fvg_type == "bullish":
                if price <= sl:
                    exit_time = ts
                    exit_price = price
                    outcome = "loss"
                    break
                if price >= tp:
                    exit_time = ts
                    exit_price = price
                    outcome = "win"
                    break
            else:
                if price >= sl:
                    exit_time = ts
                    exit_price = price
                    outcome = "loss"
                    break
                if price <= tp:
                    exit_time = ts
                    exit_price = price
                    outcome = "win"
                    break
            exit_time = ts
            exit_price = price

        if fvg.fvg_type == "bullish":
            pnl_pips = (exit_price - entry_price) / PIP_SIZE
        else:
            pnl_pips = (entry_price - exit_price) / PIP_SIZE

        trades.append(
            Trade(
                fvg_type=fvg.fvg_type,
                formed_at=fvg.formed_at,
                entry_time=entry_time,
                entry_price=entry_price,
                sl=sl,
                tp=tp,
                exit_time=exit_time,
                exit_price=exit_price,
                outcome=outcome,
                pnl_pips=pnl_pips,
            )
        )

    return trades


def summarize(trades: List[Trade]) -> Dict[str, float]:
    real_trades = [t for t in trades if t.outcome != "no_touch"]
    wins = [t for t in real_trades if t.outcome == "win"]
    losses = [t for t in real_trades if t.outcome == "loss"]
    timeouts = [t for t in real_trades if t.outcome == "timeout"]
    total = len(real_trades)
    win_rate = (len(wins) / total * 100.0) if total else 0.0
    total_pnl = sum(t.pnl_pips for t in real_trades)
    avg_pnl = total_pnl / total if total else 0.0
    return {
        "trades": total,
        "wins": len(wins),
        "losses": len(losses),
        "timeouts": len(timeouts),
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "avg_pnl": avg_pnl,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="FVG Magnet - Off-Hours Formation Only")
    parser.add_argument("--ohlc", default="tickdata/EURUSD_1H.csv")
    parser.add_argument("--tick-dir", default="tickdata/EURUSD")
    parser.add_argument("--min-gap", type=float, default=5.0)
    parser.add_argument("--max-gap", type=float, default=100.0)
    parser.add_argument("--all-hours", action="store_true", help="Use all hours (no filter)")
    args = parser.parse_args()

    df = pd.read_csv(args.ohlc, parse_dates=["timestamp"])
    tick_index = build_tick_index(Path(args.tick_dir))

    off_hours_only = not args.all_hours
    label = "OFF-HOURS" if off_hours_only else "ALL"
    
    trades = backtest_fvg_magnet(
        df=df,
        tick_index=tick_index,
        min_gap_pips=args.min_gap,
        max_gap_pips=args.max_gap,
        off_hours_only=off_hours_only,
    )

    stats = summarize(trades)
    print(
        f"[{label}] Trades={stats['trades']} Wins={stats['wins']} Losses={stats['losses']} "
        f"Timeouts={stats['timeouts']} WinRate={stats['win_rate']:.1f}% "
        f"TotalPnL(pips)={stats['total_pnl']:.1f}"
    )


if __name__ == "__main__":
    main()
