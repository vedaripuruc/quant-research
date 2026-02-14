#!/usr/bin/env python3
"""Test indices - US30, NAS100, SPX, DAX"""
import warnings
warnings.filterwarnings('ignore')
import sys
sys.stdout.reconfigure(line_buffering=True)

from engine import BacktestEngine, BacktestConfig, calculate_metrics, fetch_data
from strategies import STRATEGIES

INDICES = [
    ('^DJI', 'US30'),
    ('^IXIC', 'NAS100'),
    ('^GSPC', 'SPX'),
    ('^GDAXI', 'DAX'),
]

# Also test gold and oil
COMMODITIES = [
    ('GC=F', 'GOLD'),
    ('CL=F', 'OIL'),
]

config = BacktestConfig(slippage_pct=0.0005, commission_pct=0.001, position_size=1.0, compound=True)

print('INDICES & COMMODITIES TEST (1h, 59 days)')
print('='*80)

all_results = []

for symbol, name in INDICES + COMMODITIES:
    print(f'\n{name} ({symbol}):')
    try:
        df = fetch_data(symbol, '1h', 59)
        if len(df) < 30:
            print(f'  Skip - only {len(df)} bars')
            continue
        
        for strat_name, strat_fn in STRATEGIES.items():
            engine = BacktestEngine(config)
            trades = engine.run(df, strat_fn)
            m = calculate_metrics(trades, df, config)
            
            all_results.append({'symbol': name, 'strat': strat_name, **m})
            
            if m['total_trades'] > 0:
                beat = '✓' if m['strategy_return'] > m['buy_hold_return'] else '✗'
                print(f'  {strat_name:20} | {m["total_trades"]:2}t | WR:{m["win_rate"]:5.1f}% | Ret:{m["strategy_return"]:7.2f}% {beat}')
    except Exception as e:
        print(f'  ERROR: {e}')

# Summary by instrument type
print('\n' + '='*80)
print('SUMMARY BY INSTRUMENT')
print('='*80)

import numpy as np

for group_name, symbols in [('INDICES', [x[1] for x in INDICES]), ('COMMODITIES', [x[1] for x in COMMODITIES])]:
    group_res = [r for r in all_results if r['symbol'] in symbols and r['total_trades'] > 0]
    if group_res:
        avg_ret = np.mean([r['strategy_return'] for r in group_res])
        avg_wr = np.mean([r['win_rate'] for r in group_res])
        beat = sum(1 for r in group_res if r['strategy_return'] > r['buy_hold_return'])
        print(f'\n{group_name} ({len(group_res)} tests):')
        print(f'  Avg Return: {avg_ret:.2f}%')
        print(f'  Avg WR: {avg_wr:.1f}%')
        print(f'  Beat B&H: {beat}/{len(group_res)}')

# Best performers
print('\n' + '='*80)
print('TOP 5 BY RETURN')
print('='*80)

top = sorted([r for r in all_results if r['total_trades'] >= 3], 
             key=lambda x: x['strategy_return'], reverse=True)[:5]

for r in top:
    print(f'  {r["symbol"]:8} + {r["strat"]:20} | Ret:{r["strategy_return"]:7.2f}% | WR:{r["win_rate"]:.1f}%')

print('\nDone!')
