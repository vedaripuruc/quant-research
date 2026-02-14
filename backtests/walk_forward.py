#!/usr/bin/env python3
"""
Walk-Forward Validation
-----------------------
Split data into training/testing windows to avoid overfitting.
More realistic than single-period backtest.
"""
import warnings
warnings.filterwarnings('ignore')
import sys
sys.stdout.reconfigure(line_buffering=True)

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from engine import BacktestEngine, BacktestConfig, calculate_metrics, fetch_data
from strategies_forex import FOREX_STRATEGIES

def walk_forward_test(symbol: str, strategy_fn, 
                      total_days: int = 180,
                      train_days: int = 60,
                      test_days: int = 30,
                      interval: str = '1h') -> dict:
    """
    Walk-forward validation:
    1. Train on first N days
    2. Test on next M days
    3. Slide window forward
    4. Repeat
    """
    
    df = fetch_data(symbol, interval, total_days)
    if len(df) < 100:
        return {'error': 'Insufficient data'}
    
    config = BacktestConfig(
        slippage_pct=0.0001,
        commission_pct=0.0003,
        position_size=1.0,
        compound=True
    )
    
    # Calculate window sizes in bars
    bars_per_day = 24 if interval == '1h' else 6 if interval == '4h' else 1
    train_bars = train_days * bars_per_day
    test_bars = test_days * bars_per_day
    step_bars = test_bars  # Non-overlapping test windows
    
    results = []
    start_idx = 0
    
    while start_idx + train_bars + test_bars <= len(df):
        # Training window (not used for parameter optimization in this simple version)
        train_end = start_idx + train_bars
        
        # Test window
        test_start = train_end
        test_end = test_start + test_bars
        
        test_df = df.iloc[test_start:test_end].reset_index(drop=True)
        test_df['Date'] = df.iloc[test_start:test_end]['Date'].values
        
        if len(test_df) < 20:
            break
        
        engine = BacktestEngine(config)
        trades = engine.run(test_df, strategy_fn)
        metrics = calculate_metrics(trades, test_df, config)
        
        results.append({
            'window': len(results) + 1,
            'test_start': df.iloc[test_start]['Date'],
            'test_end': df.iloc[min(test_end-1, len(df)-1)]['Date'],
            **metrics
        })
        
        start_idx += step_bars
    
    if not results:
        return {'error': 'No valid test windows'}
    
    # Aggregate results
    avg_return = np.mean([r['strategy_return'] for r in results])
    avg_wr = np.mean([r['win_rate'] for r in results if r['total_trades'] > 0])
    total_trades = sum(r['total_trades'] for r in results)
    positive_windows = sum(1 for r in results if r['strategy_return'] > 0)
    
    return {
        'symbol': symbol,
        'windows': len(results),
        'total_trades': total_trades,
        'avg_return': round(avg_return, 2),
        'avg_win_rate': round(avg_wr, 1),
        'positive_windows': positive_windows,
        'consistency': round(positive_windows / len(results) * 100, 1),
        'details': results
    }


print('WALK-FORWARD VALIDATION')
print('Train: 60 days | Test: 30 days | Rolling windows')
print('='*70)

PAIRS = ['EURUSD=X', 'USDJPY=X', 'EURJPY=X']
STRATS = ['breakout_fx', 'session_breakout_fx']

all_results = []

for pair in PAIRS:
    print(f'\n{pair}:')
    
    for strat_name in STRATS:
        strat_fn = FOREX_STRATEGIES[strat_name]
        result = walk_forward_test(pair, strat_fn, total_days=180)
        
        if 'error' not in result:
            all_results.append({'pair': pair, 'strat': strat_name, **result})
            print(f'  {strat_name:20} | {result["windows"]} windows | '
                  f'{result["total_trades"]}t | '
                  f'Avg:{result["avg_return"]:+.2f}% | '
                  f'Consistency:{result["consistency"]:.0f}%')
        else:
            print(f'  {strat_name:20} | ERROR: {result["error"]}')

# Summary
print('\n' + '='*70)
print('WALK-FORWARD SUMMARY')
print('='*70)

if all_results:
    best = max(all_results, key=lambda x: x['consistency'])
    print(f'\nMost Consistent: {best["pair"]} + {best["strat"]}')
    print(f'  Consistency: {best["consistency"]:.0f}% positive windows')
    print(f'  Avg Return: {best["avg_return"]:+.2f}% per window')
    
    print('\nWindow-by-window breakdown:')
    for w in best['details']:
        ret = w['strategy_return']
        wr = w['win_rate']
        trades = w['total_trades']
        status = '✓' if ret > 0 else '✗'
        print(f'  Window {w["window"]}: {ret:+6.2f}% | {trades}t | WR:{wr:.0f}% {status}')

print('\nDone!')
