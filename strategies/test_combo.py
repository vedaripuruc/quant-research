#!/usr/bin/env python3
"""Test script for ECVT+Hurst combo strategy."""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Import combo components
sys.path.insert(0, str(Path(__file__).parent))
from ecvt_hurst_combo import ComboParams, generate_combo_signals

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
    print("Testing ECVT+Hurst combo signal generation...")
    
    # Load data
    data_path = Path(__file__).parent.parent / "curupira-backtests" / "data" / "eurusd_hourly.csv"
    df = pd.read_csv(data_path, parse_dates=['timestamp'])
    print(f"Loaded {len(df)} rows")
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = synthesize_volume_from_volatility(df)
    
    # Test with smaller subset first
    df_small = df.head(2000).copy()  # ~3 months of hourly data
    print(f"Testing with {len(df_small)} rows")
    
    params = ComboParams()
    print("Parameters:", vars(params))
    
    print("Generating signals...")
    df_signals = generate_combo_signals(df_small, params)
    
    # Analyze results
    ecvt_signals = (df_signals['ecvt_signal'] == 1).sum()
    combo_signals = (df_signals['signal'] != 0).sum()
    
    print(f"ECVT raw signals: {ecvt_signals}")
    print(f"Combo final signals: {combo_signals}")
    
    # Check regime distribution
    regimes = df_signals['regime'].value_counts()
    print("Hurst regimes:", regimes.to_dict())
    
    # Check signal distribution
    if combo_signals > 0:
        signal_dist = df_signals[df_signals['signal'] != 0]['signal'].value_counts()
        print("Signal direction distribution:", signal_dist.to_dict())
    
    print("Test completed successfully!")

if __name__ == '__main__':
    main()