#!/usr/bin/env python3
"""Basic candlestick replay animation with entry/exit markers.

Usage:
  python chart_replay.py --ohlc data/eurusd_hourly.csv --out charts/replay_demo.mp4

If no trade is provided, the script picks a mock trade from the OHLC data
(entry at the middle bar, exit 30 bars later) to prove the pipeline.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Rectangle


@dataclass
class Trade:
    entry_idx: int
    exit_idx: int


def load_ohlc(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError("OHLC file must include a 'timestamp' column")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            raise ValueError(f"OHLC file missing '{col}' column")
    return df


def resolve_trade(df: pd.DataFrame, entry_idx: Optional[int], exit_idx: Optional[int]) -> Trade:
    if entry_idx is None:
        entry_idx = len(df) // 2
    if exit_idx is None:
        exit_idx = min(entry_idx + 30, len(df) - 1)
    if entry_idx < 0 or exit_idx < 0 or entry_idx >= len(df) or exit_idx >= len(df):
        raise ValueError("Entry/exit indices out of range")
    if exit_idx <= entry_idx:
        raise ValueError("Exit index must be after entry index")
    return Trade(entry_idx=entry_idx, exit_idx=exit_idx)


def draw_candles(ax, df: pd.DataFrame, width: float = 0.6):
    for i, row in df.iterrows():
        color = "#26a69a" if row["close"] >= row["open"] else "#ef5350"
        ax.vlines(i, row["low"], row["high"], color=color, linewidth=1)
        lower = min(row["open"], row["close"])
        height = abs(row["close"] - row["open"])
        if height == 0:
            height = 1e-6
        rect = Rectangle((i - width / 2, lower), width, height, facecolor=color, edgecolor=color)
        ax.add_patch(rect)


def render_animation(
    df: pd.DataFrame,
    trade: Trade,
    out_path: Path,
    fps: int,
    before: int,
    after: int,
):
    start = max(trade.entry_idx - before, 0)
    end = min(trade.exit_idx + after, len(df) - 1)
    window = df.iloc[start : end + 1].reset_index(drop=True)

    entry_rel = trade.entry_idx - start
    exit_rel = trade.exit_idx - start

    fig, ax = plt.subplots(figsize=(10, 6))
    writer = FFMpegWriter(fps=fps, metadata={"title": "Trade Replay"})

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with writer.saving(fig, str(out_path), dpi=120):
        for i in range(len(window)):
            ax.clear()
            draw_candles(ax, window.iloc[: i + 1])

            ax.set_title("EURUSD Hourly Trade Replay")
            ax.set_xlim(-1, len(window))
            prices = window.iloc[: i + 1][["low", "high"]]
            ax.set_ylim(prices["low"].min() * 0.999, prices["high"].max() * 1.001)
            ax.set_xticks([])
            ax.grid(True, alpha=0.2)

            if i >= entry_rel:
                entry_price = window.loc[entry_rel, "open"]
                ax.scatter(entry_rel, entry_price, marker="^", s=80, color="#00c853", zorder=5, label="Entry")
            if i >= exit_rel:
                exit_price = window.loc[exit_rel, "close"]
                ax.scatter(exit_rel, exit_price, marker="x", s=80, color="#ff9800", zorder=5, label="Exit")

            writer.grab_frame()

    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Render a candlestick replay animation with entry/exit markers.")
    parser.add_argument("--ohlc", type=Path, default=Path("data/eurusd_hourly.csv"), help="Path to OHLC CSV")
    parser.add_argument("--entry-idx", type=int, default=None, help="Entry bar index in the OHLC file")
    parser.add_argument("--exit-idx", type=int, default=None, help="Exit bar index in the OHLC file")
    parser.add_argument("--before", type=int, default=50, help="Bars to show before entry")
    parser.add_argument("--after", type=int, default=50, help="Bars to show after exit")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second")
    parser.add_argument("--out", type=Path, default=Path("charts/replay_demo.mp4"), help="Output MP4 path")
    return parser.parse_args()


def main():
    args = parse_args()
    df = load_ohlc(args.ohlc)

    trade = resolve_trade(df, args.entry_idx, args.exit_idx)
    render_animation(df, trade, args.out, args.fps, args.before, args.after)

    print(f"Saved replay to {args.out}")


if __name__ == "__main__":
    main()
