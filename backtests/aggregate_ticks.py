#!/usr/bin/env python3
"""
Aggregate Darwinex tick files into 1H OHLC.
Each hourly file contains ticks for that hour.
"""

from __future__ import annotations

import argparse
import gzip
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

FILENAME_RE = re.compile(r"^(?P<symbol>[A-Z0-9]+)_BID_(?P<date>\d{4}-\d{2}-\d{2})_(?P<hour>\d{2})\.log\.gz$")


@dataclass
class HourBar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    ticks: int


def parse_hour_from_filename(path: Path) -> Optional[datetime]:
    match = FILENAME_RE.match(path.name)
    if not match:
        return None
    date_str = match.group("date")
    hour_str = match.group("hour")
    return datetime.strptime(f"{date_str} {hour_str}", "%Y-%m-%d %H")


def iter_tick_files(symbol_dir: Path) -> Iterable[Path]:
    files = []
    for path in symbol_dir.glob("*.log.gz"):
        ts = parse_hour_from_filename(path)
        if ts is not None:
            files.append((ts, path))
    files.sort(key=lambda item: item[0])
    for _, path in files:
        yield path


def aggregate_file(path: Path) -> Optional[HourBar]:
    ts = parse_hour_from_filename(path)
    if ts is None:
        return None
    try:
        with gzip.open(path, "rt") as f:
            first = None
            last = None
            high = None
            low = None
            volume = 0.0
            ticks = 0
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
                    price = float(parts[1])
                    vol = float(parts[2]) if len(parts) > 2 else 1.0
                except ValueError:
                    continue
                if first is None:
                    first = price
                    high = price
                    low = price
                else:
                    if price > high:
                        high = price
                    if price < low:
                        low = price
                last = price
                volume += vol
                ticks += 1
            if first is None or last is None:
                return None
            return HourBar(
                timestamp=ts,
                open=first,
                high=high,
                low=low,
                close=last,
                volume=volume,
                ticks=ticks,
            )
    except (OSError, EOFError):
        return None


def aggregate_symbol(symbol_dir: Path) -> pd.DataFrame:
    rows: List[dict] = []
    for path in iter_tick_files(symbol_dir):
        bar = aggregate_file(path)
        if bar is None:
            continue
        rows.append(
            {
                "timestamp": bar.timestamp,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
                "ticks": bar.ticks,
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Darwinex tick files into 1H OHLC.")
    parser.add_argument("--symbol", default="EURUSD", help="Symbol directory name.")
    parser.add_argument("--data-dir", default="tickdata", help="Base tickdata directory.")
    parser.add_argument("--output", default=None, help="Output CSV path.")
    args = parser.parse_args()

    symbol_dir = Path(args.data_dir) / args.symbol
    if not symbol_dir.exists():
        raise SystemExit(f"Symbol directory not found: {symbol_dir}")

    df = aggregate_symbol(symbol_dir)
    if df.empty:
        raise SystemExit("No data aggregated. Check tick files.")

    output = Path(args.output) if args.output else Path(args.data_dir) / f"{args.symbol}_1H.csv"
    df.to_csv(output, index=False)
    print(f"Saved {len(df)} rows to {output}")


if __name__ == "__main__":
    main()
