#!/usr/bin/env python3
"""
Fade Retail v2 - Failed Breakouts, Traps, Liquidity Grabs
"""

import warnings
warnings.filterwarnings('ignore')
import sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
from datetime import datetime

from engine import BacktestEngine, BacktestConfig, calculate_metrics, fetch_data
from strategies import breakout_signal
from strategies_fade import FADE_STRATEGIES

SYMBOLS = [
    'EURUSD=X', 'USDJPY=X', 'EURJPY=X',
    'GC=F', 'CL=F',
    'SPY', 'QQQ', 'AAPL', 'TSLA'
]

forex_config = BacktestConfig(slippage_pct=0.0001, commission_pct=0.0003, position_size=1.0, compound=True)
stock_config = BacktestConfig(slippage_pct=0.0005, commission_pct=0.001, position_size=1.0, compound=True)

def run_test(symbol, days=59):
    df = fetch_data(symbol, '1h', days)
    if df.empty or len(df) < 100:
        return None
    
    config = forex_config if '=X' in symbol or '=F' in symbol else stock_config
    results = {}
    
    # Original breakout for comparison
    engine = BacktestEngine(config)
    trades = engine.run(df, breakout_signal)
    m = calculate_metrics(trades, df, config)
    results['breakout'] = m
    
    # Fade strategies
    for name, strat_fn in FADE_STRATEGIES.items():
        engine = BacktestEngine(config)
        trades = engine.run(df, strat_fn)
        m = calculate_metrics(trades, df, config)
        results[name] = m
    
    return results


print("=" * 90)
print("FADE RETAIL v2 - Failed Breakouts, Traps, Liquidity Grabs")
print("=" * 90)
print("\nNew strategies:")
print("  - failed_breakout_fade: Trade AGAINST failed breakouts (trapped retail)")
print("  - overextension_fade: Fade moves > 2x ATR (FOMO/panic)")
print("  - trap_and_reverse: Bull/bear traps (spike + reversal)")
print("  - liquidity_grab_fade: Fade wicks that hunt stops")
print()

all_results = []

for symbol in SYMBOLS:
    print(f"\n{symbol}:")
    result = run_test(symbol)
    
    if result is None:
        print("  Skip - insufficient data")
        continue
    
    all_results.append({'symbol': symbol, **result})
    
    # Header
    print(f"  {'Strategy':<25} | {'Trades':>6} | {'WR':>6} | {'Return':>8} | {'PF':>6}")
    print(f"  {'-'*65}")
    
    # Sort by return
    sorted_strats = sorted(result.items(), key=lambda x: x[1]['strategy_return'], reverse=True)
    
    for name, m in sorted_strats:
        if m['total_trades'] > 0:
            best = '🔥' if m['strategy_return'] > result['breakout']['strategy_return'] else ''
            print(f"  {name:<25} | {m['total_trades']:>6} | {m['win_rate']:>5.1f}% | {m['strategy_return']:>+7.2f}% | {m['profit_factor']:>5.2f} {best}")
        else:
            print(f"  {name:<25} | {m['total_trades']:>6} | {'N/A':>6} | {'N/A':>8} | {'N/A':>6}")

# Aggregate
print("\n" + "=" * 90)
print("AGGREGATE RESULTS")
print("=" * 90)

strategies = ['breakout'] + list(FADE_STRATEGIES.keys())

for strat in strategies:
    data = [r[strat] for r in all_results if strat in r and r[strat]['total_trades'] > 0]
    if not data:
        continue
    
    avg_ret = np.mean([d['strategy_return'] for d in data])
    avg_wr = np.mean([d['win_rate'] for d in data])
    total_trades = sum(d['total_trades'] for d in data)
    
    print(f"\n{strat.upper()}:")
    print(f"  Trades: {total_trades} | WR: {avg_wr:.1f}% | Avg Return: {avg_ret:+.2f}%")

# Winners
print("\n" + "=" * 90)
print("HEAD-TO-HEAD vs BREAKOUT")
print("=" * 90)

for strat in FADE_STRATEGIES.keys():
    wins = 0
    losses = 0
    for r in all_results:
        if strat in r and r[strat]['total_trades'] > 0:
            if r[strat]['strategy_return'] > r['breakout']['strategy_return']:
                wins += 1
            else:
                losses += 1
    if wins + losses > 0:
        print(f"  {strat}: {wins} wins, {losses} losses vs breakout")

print(f"\nCompleted: {datetime.now()}")
