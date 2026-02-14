#!/usr/bin/env python3
"""Quick test of ECVT+Hurst combo strategy on full dataset."""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Import combo components
sys.path.insert(0, str(Path(__file__).parent))
from ecvt_hurst_combo import ComboParams, generate_combo_signals, run_backtest

def synthesize_volume_from_volatility(df, window=20):
    """Synthesize volume proxy from price volatility when real volume is zero."""
    df = df.copy()
    
    if df['volume'].sum() == 0 or df['volume'].isna().all():
        print("  [INFO] Volume is zero — synthesizing from price activity")
        bar_range = (df['high'] - df['low']) / df['close']
        abs_return = df['close'].pct_change().abs().fillna(0)
        
        activity = (bar_range + abs_return) / 2
        activity = activity.fillna(activity.median())
        
        if activity.mean() > 0:
            df['volume'] = (activity / activity.mean() * 1000).clip(lower=1)
        else:
            df['volume'] = 1000
    
    return df

def main():
    print("=" * 60)
    print("  QUICK ECVT+HURST COMBO TEST")
    print("=" * 60)
    
    # Load data
    data_path = Path(__file__).parent.parent / "curupira-backtests" / "data" / "eurusd_hourly.csv"
    df = pd.read_csv(data_path, parse_dates=['timestamp'])
    print(f"Loaded {len(df)} rows: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = synthesize_volume_from_volatility(df)
    
    # Test strict version
    print("\n" + "=" * 60)
    print("  STRICT COMBO (entropy=5%)")
    print("=" * 60)
    
    strict_params = ComboParams(entropy_percentile=5.0)
    print("Generating signals (this may take ~2 minutes)...")
    df_strict = generate_combo_signals(df, strict_params)
    
    # Analyze signals
    ecvt_signals = (df_strict['ecvt_signal'] == 1).sum()
    combo_signals = (df_strict['signal'] != 0).sum()
    trending_count = (df_strict['regime'] == 'trending').sum()
    mean_rev_count = (df_strict['regime'] == 'mean_reverting').sum()
    
    print(f"ECVT raw signals:      {ecvt_signals}")
    print(f"Combo final signals:   {combo_signals}")
    print(f"Trending regime bars:  {trending_count}")
    print(f"Mean-rev regime bars:  {mean_rev_count}")
    print(f"Filter efficiency:     {combo_signals/ecvt_signals:.1%} signals passed Hurst filter")
    
    # Signal direction breakdown
    if combo_signals > 0:
        long_signals = (df_strict['signal'] == 1).sum()
        short_signals = (df_strict['signal'] == -1).sum()
        print(f"Direction: {long_signals} long, {short_signals} short")
    
    print("\nRunning backtest...")
    strict_result = run_backtest(df_strict, strict_params)
    
    print(f"\n{'='*60}")
    print(f"  STRICT COMBO RESULTS")
    print(f"{'='*60}")
    print(f"Total trades:        {strict_result.num_trades}")
    print(f"Total PnL:          {strict_result.total_pnl_bps:+.1f} bps")
    print(f"Win rate:           {strict_result.win_rate:.1%}")
    print(f"Avg win:            {strict_result.avg_win_bps:+.1f} bps")
    print(f"Avg loss:           {strict_result.avg_loss_bps:+.1f} bps")
    print(f"Profit factor:      {strict_result.profit_factor:.2f}")
    print(f"Max drawdown:       {strict_result.max_drawdown_bps:.1f} bps")
    print(f"Sharpe (per-trade): {strict_result.sharpe_ratio:.3f}")
    
    # Test relaxed version
    print("\n" + "=" * 60)
    print("  RELAXED COMBO (entropy=10%)")
    print("=" * 60)
    
    relaxed_params = ComboParams(entropy_percentile=10.0)
    print("Generating relaxed signals...")
    df_relaxed = generate_combo_signals(df, relaxed_params)
    
    ecvt_r = (df_relaxed['ecvt_signal'] == 1).sum()
    combo_r = (df_relaxed['signal'] != 0).sum()
    
    print(f"ECVT raw signals:      {ecvt_r}")
    print(f"Combo final signals:   {combo_r}")
    print(f"Filter efficiency:     {combo_r/ecvt_r:.1%} signals passed Hurst filter")
    
    print("\nRunning relaxed backtest...")
    relaxed_result = run_backtest(df_relaxed, relaxed_params)
    
    print(f"\n{'='*60}")
    print(f"  RELAXED COMBO RESULTS")
    print(f"{'='*60}")
    print(f"Total trades:        {relaxed_result.num_trades}")
    print(f"Total PnL:          {relaxed_result.total_pnl_bps:+.1f} bps")
    print(f"Win rate:           {relaxed_result.win_rate:.1%}")
    print(f"Avg win:            {relaxed_result.avg_win_bps:+.1f} bps")
    print(f"Avg loss:           {relaxed_result.avg_loss_bps:+.1f} bps")
    print(f"Profit factor:      {relaxed_result.profit_factor:.2f}")
    print(f"Max drawdown:       {relaxed_result.max_drawdown_bps:.1f} bps")
    print(f"Sharpe (per-trade): {relaxed_result.sharpe_ratio:.3f}")
    
    # Comparison
    print("\n" + "=" * 60)
    print("  COMPARISON TO ECVT STANDALONE")
    print("=" * 60)
    print(f"{'Strategy':<20} {'Trades':<8} {'PnL (bps)':<12} {'Win Rate':<10} {'PF':<6}")
    print(f"{'-'*56}")
    print(f"{'ECVT Standalone':<20} {'44':<8} {'+198':<12} {'41%':<10} {'1.44':<6}")
    strict_pnl_str = f"{strict_result.total_pnl_bps:+.1f}"
    strict_wr_str = f"{strict_result.win_rate:.0%}"
    strict_pf_str = f"{strict_result.profit_factor:.2f}"
    print(f"{'Combo Strict':<20} {strict_result.num_trades:<8} {strict_pnl_str:<12} {strict_wr_str:<10} {strict_pf_str:<6}")
    
    relaxed_pnl_str = f"{relaxed_result.total_pnl_bps:+.1f}"
    relaxed_wr_str = f"{relaxed_result.win_rate:.0%}"
    relaxed_pf_str = f"{relaxed_result.profit_factor:.2f}"
    print(f"{'Combo Relaxed':<20} {relaxed_result.num_trades:<8} {relaxed_pnl_str:<12} {relaxed_wr_str:<10} {relaxed_pf_str:<6}")
    
    # Assessment
    print("\n" + "=" * 60)
    print("  ASSESSMENT")
    print("=" * 60)
    
    baseline_pnl = 198  # ECVT standalone
    strict_improvement = strict_result.total_pnl_bps - baseline_pnl
    relaxed_improvement = relaxed_result.total_pnl_bps - baseline_pnl
    
    print(f"Strict improvement:  {strict_improvement:+.1f} bps")
    print(f"Relaxed improvement: {relaxed_improvement:+.1f} bps")
    
    if max(strict_improvement, relaxed_improvement) > 50:
        verdict = "🟢 POSITIVE: Hurst filter improves ECVT performance"
    elif max(strict_improvement, relaxed_improvement) > -50:
        verdict = "🟡 NEUTRAL: Hurst filter has mixed impact"
    else:
        verdict = "🔴 NEGATIVE: Hurst filter degrades ECVT performance"
    
    print(f"\n{verdict}")
    print("\nNote: This is full-period performance. Walk-forward validation needed for true OOS assessment.")
    print("=" * 60)

if __name__ == '__main__':
    main()