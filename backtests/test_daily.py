#!/usr/bin/env python3
"""Test daily timeframe - longer history, bigger moves"""
import warnings
warnings.filterwarnings('ignore')
import sys
sys.stdout.reconfigure(line_buffering=True)

from engine import BacktestEngine, BacktestConfig, calculate_metrics, fetch_data
from strategies_forex import FOREX_STRATEGIES

FOREX = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'EURJPY=X', 'AUDUSD=X']
STRATS = ['breakout_fx', 'session_breakout_fx', 'swing_fx']

config = BacktestConfig(slippage_pct=0.0001, commission_pct=0.0003, position_size=1.0, compound=True)

print('DAILY TIMEFRAME TEST (365 days)')
print('='*70)

results = []

for pair in FOREX:
    print(f'\n{pair}:')
    try:
        df = fetch_data(pair, '1d', 365)
        if len(df) < 50:
            print(f'  Skip - only {len(df)} bars')
            continue
        
        for strat_name in STRATS:
            strat_fn = FOREX_STRATEGIES[strat_name]
            engine = BacktestEngine(config)
            trades = engine.run(df, strat_fn)
            m = calculate_metrics(trades, df, config)
            
            results.append({'pair': pair, 'strat': strat_name, **m})
            
            if m['total_trades'] > 0:
                beat = '✓' if m['strategy_return'] > m['buy_hold_return'] else '✗'
                print(f'  {strat_name:20} | {m["total_trades"]:2}t | WR:{m["win_rate"]:5.1f}% | Ret:{m["strategy_return"]:7.2f}% | MDD:{m["max_drawdown"]:6.2f}% {beat}')
    except Exception as e:
        print(f'  ERROR: {e}')

# Summary
print('\n' + '='*70)
print('DAILY SUMMARY')
print('='*70)

import numpy as np
for strat in STRATS:
    strat_res = [r for r in results if r['strat'] == strat and r['total_trades'] > 0]
    if strat_res:
        avg_ret = np.mean([r['strategy_return'] for r in strat_res])
        avg_wr = np.mean([r['win_rate'] for r in strat_res])
        avg_mdd = np.mean([r['max_drawdown'] for r in strat_res])
        print(f'{strat:20} | WR:{avg_wr:5.1f}% | Ret:{avg_ret:7.2f}% | MDD:{avg_mdd:6.2f}%')

print('\nDone!')
