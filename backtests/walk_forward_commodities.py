#!/usr/bin/env python3
"""Walk-forward for commodities (Gold, Oil)"""
import warnings
warnings.filterwarnings('ignore')
import sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
from engine import BacktestEngine, BacktestConfig, calculate_metrics, fetch_data
from strategies import breakout_signal, williams_r_signal

def walk_forward(symbol, strategy_fn, name, total_days=180, train_days=60, test_days=30):
    df = fetch_data(symbol, '1h', total_days)
    if len(df) < 100:
        return None
    
    config = BacktestConfig(slippage_pct=0.0005, commission_pct=0.001, position_size=1.0, compound=True)
    
    bars_per_day = 24
    train_bars = train_days * bars_per_day
    test_bars = test_days * bars_per_day
    
    results = []
    start_idx = 0
    
    while start_idx + train_bars + test_bars <= len(df):
        test_start = start_idx + train_bars
        test_end = test_start + test_bars
        
        test_df = df.iloc[test_start:test_end].reset_index(drop=True)
        test_df['Date'] = df.iloc[test_start:test_end]['Date'].values
        
        if len(test_df) < 20:
            break
        
        engine = BacktestEngine(config)
        trades = engine.run(test_df, strategy_fn)
        m = calculate_metrics(trades, test_df, config)
        
        results.append(m)
        start_idx += test_bars
    
    if not results:
        return None
    
    return {
        'symbol': name,
        'windows': len(results),
        'total_trades': sum(r['total_trades'] for r in results),
        'avg_return': np.mean([r['strategy_return'] for r in results]),
        'positive_windows': sum(1 for r in results if r['strategy_return'] > 0),
        'details': results
    }

print('COMMODITIES WALK-FORWARD')
print('='*70)

commodities = [('GC=F', 'GOLD'), ('CL=F', 'OIL')]

for symbol, name in commodities:
    print(f'\n{name}:')
    
    # Test breakout
    result = walk_forward(symbol, breakout_signal, name)
    if result:
        cons = result['positive_windows'] / result['windows'] * 100
        print(f'  breakout    | {result["windows"]} windows | {result["total_trades"]}t | Avg:{result["avg_return"]:+.2f}% | Consistency:{cons:.0f}%')
        for i, w in enumerate(result['details'], 1):
            status = '✓' if w['strategy_return'] > 0 else '✗'
            print(f'    Window {i}: {w["strategy_return"]:+6.2f}% | {w["total_trades"]}t | WR:{w["win_rate"]:.0f}% {status}')
    
    # Test williams_r
    result = walk_forward(symbol, williams_r_signal, name)
    if result:
        cons = result['positive_windows'] / result['windows'] * 100
        print(f'  williams_r  | {result["windows"]} windows | {result["total_trades"]}t | Avg:{result["avg_return"]:+.2f}% | Consistency:{cons:.0f}%')

print('\nDone!')
