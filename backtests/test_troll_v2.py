#!/usr/bin/env python3
"""
Troll Strategy Backtest v2 - With Fixed Engine
-----------------------------------------------
Tests the corrected troll strategies using the fixed engine
that eliminates look-ahead bias.

Run: python test_troll_v2.py
"""

import sys
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
import pandas as pd

from engine import BacktestEngine, BacktestConfig, calculate_metrics, fetch_data
from strategies_troll import make_troll_strategy, TROLL_STRATEGIES

# Test configuration
SYMBOLS = {
    'forex': ['EURUSD=X', 'USDJPY=X', 'GBPUSD=X'],
    'stocks': ['SPY', 'QQQ', 'AAPL'],
    'commodities': ['GC=F', 'CL=F'],
}

VARIANTS = ['troll_breakout', 'troll_aggressive', 'troll_patient', 'troll_ultimate']

INTERVAL = '1h'
DAYS = 59  # Max for 1h on yfinance


def run_single_test(symbol: str, variant: str, df: pd.DataFrame) -> dict:
    """Run a single backtest and return results."""
    config = BacktestConfig(
        slippage_pct=0.0005,
        commission_pct=0.001 if '=X' not in symbol else 0.0003,  # Lower for forex
    )
    
    engine = BacktestEngine(config)
    
    # Create strategy with proper context
    strategy = make_troll_strategy(
        variant=variant.replace('troll_', ''),
        symbol=symbol,
        interval=INTERVAL
    )
    
    trades_df = engine.run(df, strategy)
    metrics = calculate_metrics(trades_df, df, config)
    
    return {
        'symbol': symbol,
        'variant': variant,
        'trades': metrics['total_trades'],
        'win_rate': metrics['win_rate'],
        'return': metrics['strategy_return'],
        'bnh_return': metrics['buy_hold_return'],
        'max_dd': metrics['max_drawdown'],
        'sharpe': metrics['sharpe'],
        'pf': metrics['profit_factor'],
    }


def main():
    print("=" * 80)
    print("TROLL STRATEGY BACKTEST v2 - FIXED ENGINE (No Look-Ahead Bias)")
    print("=" * 80)
    print(f"Started: {datetime.now()}")
    print(f"Interval: {INTERVAL}, Days: {DAYS}")
    print()
    
    all_results = []
    
    for asset_class, symbols in SYMBOLS.items():
        print(f"\n{'='*40}")
        print(f"Testing {asset_class.upper()}")
        print(f"{'='*40}")
        
        for symbol in symbols:
            print(f"\nFetching {symbol}...")
            try:
                df = fetch_data(symbol, interval=INTERVAL, days=DAYS)
                if len(df) < 100:
                    print(f"  ⚠️ Insufficient data ({len(df)} bars), skipping")
                    continue
                print(f"  ✓ Got {len(df)} bars")
            except Exception as e:
                print(f"  ❌ Error fetching data: {e}")
                continue
            
            for variant in VARIANTS:
                try:
                    result = run_single_test(symbol, variant, df)
                    all_results.append(result)
                    
                    # Print inline result
                    emoji = "✅" if result['return'] > 0 else "❌"
                    print(f"  {variant:20} | {result['trades']:3} trades | "
                          f"WR {result['win_rate']:5.1f}% | "
                          f"Ret {result['return']:+6.2f}% | "
                          f"B&H {result['bnh_return']:+6.2f}% {emoji}")
                except Exception as e:
                    print(f"  {variant:20} | ❌ Error: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY BY VARIANT")
    print("=" * 80)
    
    if all_results:
        df_results = pd.DataFrame(all_results)
        
        for variant in VARIANTS:
            var_data = df_results[df_results['variant'] == variant]
            if len(var_data) > 0:
                avg_ret = var_data['return'].mean()
                avg_wr = var_data['win_rate'].mean()
                total_trades = var_data['trades'].sum()
                pos_count = (var_data['return'] > 0).sum()
                
                print(f"{variant:20} | Avg Return: {avg_ret:+6.2f}% | "
                      f"Avg WR: {avg_wr:5.1f}% | "
                      f"Total Trades: {total_trades:4} | "
                      f"Profitable: {pos_count}/{len(var_data)}")
    
    # Best variant
    print("\n" + "=" * 80)
    print("BEST VARIANT BY AVERAGE RETURN")
    print("=" * 80)
    
    if all_results:
        df_results = pd.DataFrame(all_results)
        variant_avg = df_results.groupby('variant')['return'].mean().sort_values(ascending=False)
        
        for i, (variant, avg_ret) in enumerate(variant_avg.items()):
            marker = "👑" if i == 0 else "  "
            print(f"{marker} {variant:20}: {avg_ret:+6.2f}%")
    
    print("\n" + "=" * 80)
    print(f"Completed: {datetime.now()}")
    print("=" * 80)
    
    return all_results


if __name__ == '__main__':
    results = main()
