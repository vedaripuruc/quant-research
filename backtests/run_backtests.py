#!/usr/bin/env python3
"""
Curupira Backtest Runner
------------------------
Run all strategies across multiple symbols and timeframes.
"""

import sys
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from engine import BacktestEngine, BacktestConfig, calculate_metrics, fetch_data
from strategies import STRATEGIES


def run_single_backtest(symbol: str, interval: str, strategy_name: str, 
                        days: int = 59, config: BacktestConfig = None) -> dict:
    """Run a single backtest and return results."""
    try:
        df = fetch_data(symbol, interval, days)
        if df.empty or len(df) < 30:
            return {'error': f'Insufficient data ({len(df)} bars)'}
        
        config = config or BacktestConfig()
        engine = BacktestEngine(config)
        
        strategy_fn = STRATEGIES[strategy_name]
        trades_df = engine.run(df, strategy_fn)
        
        metrics = calculate_metrics(trades_df, df, config)
        
        return {
            'symbol': symbol,
            'interval': interval,
            'strategy': strategy_name,
            'bars': len(df),
            **metrics
        }
    
    except Exception as e:
        return {
            'symbol': symbol,
            'interval': interval,
            'strategy': strategy_name,
            'error': str(e)
        }


def main():
    symbols = ['SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA', 'MSFT', 'META', 'AMZN']
    intervals = ['1h']  # Focus on hourly like original
    strategies = list(STRATEGIES.keys())
    
    config = BacktestConfig(
        slippage_pct=0.0005,  # 0.05%
        commission_pct=0.001,  # 0.1% round trip
        position_size=1.0,
        compound=True
    )
    
    all_results = []
    
    print("=" * 90)
    print("CURUPIRA BACKTESTS - Clean Engine")
    print(f"Config: slippage={config.slippage_pct*100:.2f}%, commission={config.commission_pct*100:.1f}%, compound={config.compound}")
    print("=" * 90)
    
    for interval in intervals:
        print(f"\n{'='*90}")
        print(f"INTERVAL: {interval}")
        print("=" * 90)
        
        for strategy_name in strategies:
            print(f"\n--- {strategy_name.upper()} ---")
            
            for symbol in symbols:
                result = run_single_backtest(symbol, interval, strategy_name, config=config)
                
                if 'error' not in result:
                    all_results.append(result)
                    beat_bnh = '✓' if result['strategy_return'] > result['buy_hold_return'] else '✗'
                    print(f"  {symbol:6} | {result['total_trades']:3}t | "
                          f"WR:{result['win_rate']:5.1f}% | "
                          f"PF:{result['profit_factor']:5.2f} | "
                          f"Ret:{result['strategy_return']:7.2f}% | "
                          f"B&H:{result['buy_hold_return']:7.2f}% {beat_bnh} | "
                          f"MDD:{result['max_drawdown']:6.2f}%")
                else:
                    print(f"  {symbol:6} | ERROR: {result['error'][:40]}")
    
    # Aggregate by strategy
    print("\n" + "=" * 90)
    print("STRATEGY SUMMARY")
    print("=" * 90)
    
    for strategy_name in strategies:
        strat_results = [r for r in all_results if r.get('strategy') == strategy_name and 'error' not in r]
        if not strat_results:
            continue
        
        avg_wr = np.mean([r['win_rate'] for r in strat_results])
        avg_pf = np.mean([r['profit_factor'] for r in strat_results if r['profit_factor'] < 100])
        avg_ret = np.mean([r['strategy_return'] for r in strat_results])
        avg_mdd = np.mean([r['max_drawdown'] for r in strat_results])
        beat_count = sum(1 for r in strat_results if r['strategy_return'] > r['buy_hold_return'])
        total_trades = sum(r['total_trades'] for r in strat_results)
        
        print(f"\n{strategy_name.upper()} ({len(strat_results)} tests):")
        print(f"  Total Trades:   {total_trades}")
        print(f"  Avg Win Rate:   {avg_wr:.1f}%")
        print(f"  Avg PF:         {avg_pf:.2f}")
        print(f"  Avg Return:     {avg_ret:.2f}%")
        print(f"  Avg Max DD:     {avg_mdd:.2f}%")
        print(f"  Beat B&H:       {beat_count}/{len(strat_results)}")
    
    # Top performers
    print("\n" + "=" * 90)
    print("TOP 10 BY RISK-ADJUSTED RETURN (Ret/MDD)")
    print("=" * 90)
    
    for r in all_results:
        if r.get('max_drawdown', 0) < 0:
            r['risk_adj'] = r['strategy_return'] / abs(r['max_drawdown'])
        else:
            r['risk_adj'] = r['strategy_return']
    
    top = sorted([r for r in all_results if 'error' not in r and r['total_trades'] > 0], 
                 key=lambda x: x.get('risk_adj', -999), reverse=True)[:10]
    
    for i, r in enumerate(top, 1):
        print(f"{i:2}. {r['symbol']:6} {r['strategy']:20} | "
              f"Ret:{r['strategy_return']:7.2f}% | MDD:{r['max_drawdown']:6.2f}% | "
              f"Ret/MDD:{r['risk_adj']:.2f}")
    
    print(f"\nTotal backtests: {len(all_results)}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return all_results


if __name__ == "__main__":
    main()
