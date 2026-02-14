#!/usr/bin/env python3
"""
Walk-Forward Validation for FVG Magnet Strategy.

Splits data into N windows and tests if the edge persists across all periods.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Import from our magnet backtest
from fvg_magnet_backtest import (
    build_tick_index,
    backtest_fvg_magnet,
    summarize,
)


def split_into_windows(df: pd.DataFrame, n_windows: int) -> List[pd.DataFrame]:
    """Split dataframe into N roughly equal windows."""
    total_rows = len(df)
    window_size = total_rows // n_windows
    
    windows = []
    for i in range(n_windows):
        start_idx = i * window_size
        end_idx = start_idx + window_size if i < n_windows - 1 else total_rows
        window_df = df.iloc[start_idx:end_idx].reset_index(drop=True)
        windows.append(window_df)
    
    return windows


def run_walk_forward(
    df: pd.DataFrame,
    tick_index: Dict[datetime, Path],
    n_windows: int = 6,
    min_gap_pips: float = 10.0,
    max_gap_pips: float = 100.0,
) -> Dict:
    """Run walk-forward validation."""
    
    windows = split_into_windows(df, n_windows)
    
    results = {
        "n_windows": n_windows,
        "min_gap_pips": min_gap_pips,
        "max_gap_pips": max_gap_pips,
        "windows": [],
        "summary": {}
    }
    
    total_trades = 0
    total_wins = 0
    total_losses = 0
    total_pnl = 0.0
    profitable_windows = 0
    
    for i, window_df in enumerate(windows):
        start_date = pd.to_datetime(window_df.iloc[0]["timestamp"]).strftime("%Y-%m-%d")
        end_date = pd.to_datetime(window_df.iloc[-1]["timestamp"]).strftime("%Y-%m-%d")
        n_bars = len(window_df)
        
        trades = backtest_fvg_magnet(
            df=window_df,
            tick_index=tick_index,
            min_gap_pips=min_gap_pips,
            max_gap_pips=max_gap_pips,
        )
        
        stats = summarize(trades)
        
        window_result = {
            "window": i + 1,
            "start": start_date,
            "end": end_date,
            "bars": n_bars,
            "trades": stats["trades"],
            "wins": stats["wins"],
            "losses": stats["losses"],
            "timeouts": stats["timeouts"],
            "win_rate": round(stats["win_rate"], 1),
            "pnl_pips": round(stats["total_pnl"], 1),
            "profitable": stats["total_pnl"] > 0
        }
        
        results["windows"].append(window_result)
        
        total_trades += stats["trades"]
        total_wins += stats["wins"]
        total_losses += stats["losses"]
        total_pnl += stats["total_pnl"]
        if stats["total_pnl"] > 0:
            profitable_windows += 1
        
        print(f"Window {i+1}: {start_date} to {end_date} | "
              f"Trades={stats['trades']} WR={stats['win_rate']:.1f}% "
              f"PnL={stats['total_pnl']:.1f} {'✅' if stats['total_pnl'] > 0 else '❌'}")
    
    # Overall summary
    overall_wr = (total_wins / total_trades * 100) if total_trades > 0 else 0
    results["summary"] = {
        "total_trades": total_trades,
        "total_wins": total_wins,
        "total_losses": total_losses,
        "overall_win_rate": round(overall_wr, 1),
        "total_pnl": round(total_pnl, 1),
        "profitable_windows": profitable_windows,
        "total_windows": n_windows,
        "consistency": f"{profitable_windows}/{n_windows}",
        "avg_pnl_per_window": round(total_pnl / n_windows, 1),
    }
    
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward validation for FVG Magnet.")
    parser.add_argument("--ohlc", default="tickdata/EURUSD_1H.csv", help="Path to 1H OHLC CSV.")
    parser.add_argument("--tick-dir", default="tickdata/EURUSD", help="Tick data directory.")
    parser.add_argument("--windows", type=int, default=6, help="Number of validation windows.")
    parser.add_argument("--min-gap", type=float, default=10.0, help="Minimum FVG size in pips.")
    parser.add_argument("--max-gap", type=float, default=100.0, help="Maximum FVG size in pips.")
    parser.add_argument("--output", default="fvg_walk_forward_results.json", help="Output JSON file.")
    args = parser.parse_args()

    print(f"Loading OHLC from {args.ohlc}...")
    df = pd.read_csv(args.ohlc, parse_dates=["timestamp"])
    print(f"Loaded {len(df)} bars")
    print(f"Date range: {df.iloc[0]['timestamp']} to {df.iloc[-1]['timestamp']}")
    
    print(f"\nBuilding tick index from {args.tick_dir}...")
    tick_index = build_tick_index(Path(args.tick_dir))
    print(f"Indexed {len(tick_index)} hourly tick files")
    
    print(f"\n{'='*60}")
    print(f"WALK-FORWARD VALIDATION: FVG Magnet")
    print(f"Windows: {args.windows} | Min Gap: {args.min_gap} pips | Max Gap: {args.max_gap} pips")
    print(f"{'='*60}\n")
    
    results = run_walk_forward(
        df=df,
        tick_index=tick_index,
        n_windows=args.windows,
        min_gap_pips=args.min_gap,
        max_gap_pips=args.max_gap,
    )
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total Trades: {results['summary']['total_trades']}")
    print(f"Overall Win Rate: {results['summary']['overall_win_rate']}%")
    print(f"Total PnL: {results['summary']['total_pnl']} pips")
    print(f"Profitable Windows: {results['summary']['consistency']}")
    print(f"Avg PnL/Window: {results['summary']['avg_pnl_per_window']} pips")
    
    # Verdict
    consistency = results['summary']['profitable_windows'] / results['summary']['total_windows']
    if consistency >= 0.67 and results['summary']['total_pnl'] > 0:
        verdict = "✅ VALIDATED - Edge appears robust"
    elif consistency >= 0.5 and results['summary']['total_pnl'] > 0:
        verdict = "⚠️ MARGINAL - Edge exists but inconsistent"
    else:
        verdict = "❌ FAILED - No reliable edge found"
    
    print(f"\nVERDICT: {verdict}")
    
    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
