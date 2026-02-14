#!/usr/bin/env python3
"""Quick forex test - runs fast"""
import warnings
warnings.filterwarnings('ignore')
import sys
sys.stdout.reconfigure(line_buffering=True)

from engine import BacktestEngine, BacktestConfig, calculate_metrics, fetch_data
from strategies import STRATEGIES

FOREX = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'EURJPY=X']

config = BacktestConfig(slippage_pct=0.0005, commission_pct=0.001)

print("FOREX QUICK TEST")
print("="*70)

for pair in FOREX:
    print(f"\n{pair}:")
    try:
        df = fetch_data(pair, '1h', 30)
        if len(df) < 30:
            print(f"  Skip - only {len(df)} bars")
            continue
            
        for strat_name, strat_fn in STRATEGIES.items():
            engine = BacktestEngine(config)
            trades = engine.run(df, strat_fn)
            m = calculate_metrics(trades, df, config)
            
            if m['total_trades'] > 0:
                beat = '✓' if m['strategy_return'] > m['buy_hold_return'] else '✗'
                print(f"  {strat_name:20} | {m['total_trades']:2}t | WR:{m['win_rate']:5.1f}% | Ret:{m['strategy_return']:7.2f}% {beat}")
    except Exception as e:
        print(f"  ERROR: {e}")

print("\nDone!")
