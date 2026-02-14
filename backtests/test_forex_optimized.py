#!/usr/bin/env python3
"""
Test forex-optimized strategies
"""
import warnings
warnings.filterwarnings('ignore')
import sys
sys.stdout.reconfigure(line_buffering=True)

from engine import BacktestEngine, BacktestConfig, calculate_metrics, fetch_data
from strategies_forex import FOREX_STRATEGIES

FOREX = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'EURJPY=X']

# Forex-specific config: lower spread/commission
forex_config = BacktestConfig(
    slippage_pct=0.0001,   # 1 pip slippage 
    commission_pct=0.0003,  # 3 pips spread equiv
    position_size=1.0,
    compound=True
)

print("FOREX-OPTIMIZED STRATEGIES TEST")
print("Config: slippage=0.01%, commission=0.03% (forex-appropriate)")
print("=" * 70)

total_results = []

for pair in FOREX:
    print(f"\n{pair}:")
    try:
        df = fetch_data(pair, '1h', 59)  # Same 59 days as stock test
        if len(df) < 30:
            print(f"  Skip - only {len(df)} bars")
            continue
        
        for strat_name, strat_fn in FOREX_STRATEGIES.items():
            engine = BacktestEngine(forex_config)
            trades = engine.run(df, strat_fn)
            m = calculate_metrics(trades, df, forex_config)
            
            total_results.append({
                'symbol': pair,
                'strategy': strat_name,
                **m
            })
            
            if m['total_trades'] > 0:
                beat = '✓' if m['strategy_return'] > m['buy_hold_return'] else '✗'
                print(f"  {strat_name:20} | {m['total_trades']:2}t | WR:{m['win_rate']:5.1f}% | Ret:{m['strategy_return']:7.2f}% {beat}")
                
    except Exception as e:
        print(f"  ERROR: {e}")

# Summary
print("\n" + "=" * 70)
print("SUMMARY - FOREX OPTIMIZED")
print("=" * 70)

import numpy as np

for strat in FOREX_STRATEGIES.keys():
    strat_results = [r for r in total_results if r['strategy'] == strat and r['total_trades'] > 0]
    if strat_results:
        avg_wr = np.mean([r['win_rate'] for r in strat_results])
        avg_ret = np.mean([r['strategy_return'] for r in strat_results])
        beat_count = sum(1 for r in strat_results if r['strategy_return'] > r['buy_hold_return'])
        print(f"{strat:20} | WR:{avg_wr:5.1f}% | Ret:{avg_ret:7.2f}% | Beat:{beat_count}/{len(strat_results)}")

print("\nDone!")
