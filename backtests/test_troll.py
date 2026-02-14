#!/usr/bin/env python3
"""
Troll Strategy Backtest
-----------------------
Compare original breakout vs troll (fade retail) versions.

Hypothesis: Waiting for pullback after breakout = better entry = more alpha
"""

import warnings
warnings.filterwarnings('ignore')
import sys
sys.stdout.reconfigure(line_buffering=True)

import pandas as pd
import numpy as np
from datetime import datetime

from engine import BacktestEngine, BacktestConfig, calculate_metrics, fetch_data
from strategies import breakout_signal  # Original
from strategies_troll import TROLL_STRATEGIES, troll_breakout_signal

# Test symbols
FOREX = ['EURUSD=X', 'USDJPY=X', 'EURJPY=X', 'GBPUSD=X', 'AUDUSD=X']
STOCKS = ['SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA']
COMMODITIES = ['GC=F', 'CL=F']

ALL_SYMBOLS = FOREX + STOCKS + COMMODITIES

# Configs
stock_config = BacktestConfig(
    slippage_pct=0.0005,
    commission_pct=0.001,
    position_size=1.0,
    compound=True
)

forex_config = BacktestConfig(
    slippage_pct=0.0001,
    commission_pct=0.0003,
    position_size=1.0,
    compound=True
)

def run_comparison(symbol: str, days: int = 59):
    """Compare original breakout vs troll versions"""
    
    df = fetch_data(symbol, '1h', days)
    if df.empty or len(df) < 150:
        return None
    
    config = forex_config if '=X' in symbol or '=F' in symbol else stock_config
    
    results = {}
    
    # Original breakout
    engine = BacktestEngine(config)
    trades = engine.run(df, breakout_signal)
    m = calculate_metrics(trades, df, config)
    results['original'] = {
        'trades': m['total_trades'],
        'win_rate': m['win_rate'],
        'return': m['strategy_return'],
        'mdd': m['max_drawdown'],
        'pf': m['profit_factor'],
    }
    
    # Troll versions
    for name, strat_fn in TROLL_STRATEGIES.items():
        engine = BacktestEngine(config)
        trades = engine.run(df, strat_fn)
        m = calculate_metrics(trades, df, config)
        results[name] = {
            'trades': m['total_trades'],
            'win_rate': m['win_rate'],
            'return': m['strategy_return'],
            'mdd': m['max_drawdown'],
            'pf': m['profit_factor'],
        }
    
    return results


def main():
    print("=" * 90)
    print("TROLL STRATEGY BACKTEST - Fade the Retail")
    print("=" * 90)
    print("\nConcept: Instead of chasing breakouts, wait for pullback and enter at better price")
    print("- LONG signal → Wait for dip → Buy cheap from weak hands")
    print("- SHORT signal → Wait for pump → Sell to FOMO buyers")
    print("\n" + "=" * 90)
    
    all_results = []
    
    for symbol in ALL_SYMBOLS:
        print(f"\n{symbol}:")
        result = run_comparison(symbol)
        
        if result is None:
            print("  Skip - insufficient data")
            continue
        
        all_results.append({'symbol': symbol, **result})
        
        # Print comparison
        orig = result['original']
        print(f"  {'Strategy':<20} | {'Trades':>6} | {'WR':>6} | {'Return':>8} | {'MDD':>8} | {'PF':>6}")
        print(f"  {'-'*70}")
        
        for name, r in result.items():
            better = '🔥' if r['return'] > orig['return'] else ''
            print(f"  {name:<20} | {r['trades']:>6} | {r['win_rate']:>5.1f}% | {r['return']:>+7.2f}% | {r['mdd']:>7.2f}% | {r['pf']:>5.2f} {better}")
    
    # Aggregate analysis
    print("\n" + "=" * 90)
    print("AGGREGATE ANALYSIS")
    print("=" * 90)
    
    strategies = ['original'] + list(TROLL_STRATEGIES.keys())
    
    for strat in strategies:
        strat_data = [r[strat] for r in all_results if strat in r]
        if not strat_data:
            continue
        
        avg_return = np.mean([s['return'] for s in strat_data])
        avg_wr = np.mean([s['win_rate'] for s in strat_data if s['trades'] > 0])
        avg_mdd = np.mean([s['mdd'] for s in strat_data])
        total_trades = sum(s['trades'] for s in strat_data)
        
        print(f"\n{strat.upper()}:")
        print(f"  Total Trades: {total_trades}")
        print(f"  Avg Win Rate: {avg_wr:.1f}%")
        print(f"  Avg Return:   {avg_return:+.2f}%")
        print(f"  Avg Max DD:   {avg_mdd:.2f}%")
    
    # Head-to-head comparison
    print("\n" + "=" * 90)
    print("HEAD-TO-HEAD: Original vs Troll")
    print("=" * 90)
    
    troll_wins = 0
    orig_wins = 0
    ties = 0
    
    for r in all_results:
        orig_ret = r['original']['return']
        troll_ret = r['troll_breakout']['return']
        
        if troll_ret > orig_ret + 0.5:  # Troll wins by >0.5%
            troll_wins += 1
            print(f"  🎭 TROLL wins on {r['symbol']}: {troll_ret:+.2f}% vs {orig_ret:+.2f}%")
        elif orig_ret > troll_ret + 0.5:  # Original wins
            orig_wins += 1
            print(f"  📈 ORIGINAL wins on {r['symbol']}: {orig_ret:+.2f}% vs {troll_ret:+.2f}%")
        else:
            ties += 1
    
    print(f"\n  Troll Wins: {troll_wins}")
    print(f"  Original Wins: {orig_wins}")
    print(f"  Ties: {ties}")
    
    # Best troll variant
    print("\n" + "=" * 90)
    print("BEST TROLL VARIANT")
    print("=" * 90)
    
    variant_totals = {}
    for strat in TROLL_STRATEGIES.keys():
        returns = [r[strat]['return'] for r in all_results if strat in r and r[strat]['trades'] > 0]
        if returns:
            variant_totals[strat] = np.mean(returns)
    
    if variant_totals:
        best = max(variant_totals, key=variant_totals.get)
        print(f"\n  Best variant: {best.upper()}")
        print(f"  Avg return: {variant_totals[best]:+.2f}%")
        
        for v, ret in sorted(variant_totals.items(), key=lambda x: -x[1]):
            print(f"    {v}: {ret:+.2f}%")
    
    print(f"\n{'='*90}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 90)


if __name__ == '__main__':
    main()
