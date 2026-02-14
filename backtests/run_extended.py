#!/usr/bin/env python3
"""
Extended Backtests - Forex, Indices, Multiple Timeframes
---------------------------------------------------------
FTMO-focused testing with proper risk metrics.
"""

import warnings
warnings.filterwarnings('ignore')

import os
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from engine import BacktestEngine, BacktestConfig, calculate_metrics, fetch_data
from strategies import STRATEGIES


# FTMO-style instruments
FOREX_PAIRS = [
    'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'USDCHF=X',  # Majors
    'AUDUSD=X', 'NZDUSD=X', 'USDCAD=X',              # Commodity
    'EURGBP=X', 'EURJPY=X', 'GBPJPY=X',              # Crosses
]

INDICES = [
    '^DJI',    # Dow Jones (US30)
    '^IXIC',   # Nasdaq (US100) 
    '^GSPC',   # S&P 500 (US500)
    '^GDAXI',  # DAX (GER40)
]

COMMODITIES = [
    'GC=F',    # Gold
    'SI=F',    # Silver
    'CL=F',    # Crude Oil
]

# Original stocks/ETFs for comparison
STOCKS = ['SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA', 'MSFT', 'META', 'AMZN']

TIMEFRAMES = ['1h', '1d']  # yfinance supports these well


def calculate_ftmo_metrics(trades_df: pd.DataFrame, initial_balance: float = 100000) -> dict:
    """
    Calculate FTMO-specific risk metrics.
    """
    if trades_df.empty:
        return {
            'max_daily_dd': 0,
            'max_total_dd': 0,
            'days_to_10pct': None,
            'would_pass': False,
            'breach_reason': 'No trades'
        }
    
    completed = trades_df[trades_df['exit_price'].notna()].copy()
    if completed.empty:
        return {
            'max_daily_dd': 0,
            'max_total_dd': 0,
            'days_to_10pct': None,
            'would_pass': False,
            'breach_reason': 'No completed trades'
        }
    
    # Calculate running P&L
    completed['pnl_dollars'] = completed['pnl_pct'] * initial_balance
    completed['cumulative_pnl'] = completed['pnl_dollars'].cumsum()
    completed['equity'] = initial_balance + completed['cumulative_pnl']
    
    # Track daily P&L (group by date)
    completed['trade_date'] = pd.to_datetime(completed['exit_time']).dt.date
    daily_pnl = completed.groupby('trade_date')['pnl_dollars'].sum()
    
    # Max daily drawdown
    max_daily_loss = daily_pnl.min() if len(daily_pnl) > 0 else 0
    max_daily_dd_pct = (max_daily_loss / initial_balance) * 100
    
    # Max total drawdown (peak to trough)
    peak = completed['equity'].expanding().max()
    drawdown = completed['equity'] - peak
    max_total_dd = drawdown.min()
    max_total_dd_pct = (max_total_dd / initial_balance) * 100
    
    # Days to reach 10% profit
    target = initial_balance * 1.10
    reached_target = completed[completed['equity'] >= target]
    if len(reached_target) > 0:
        first_target_date = pd.to_datetime(reached_target.iloc[0]['exit_time']).date()
        first_trade_date = pd.to_datetime(completed.iloc[0]['entry_time']).date()
        days_to_target = (first_target_date - first_trade_date).days
    else:
        days_to_target = None
    
    # Check if would pass FTMO
    breach_reason = None
    would_pass = True
    
    if max_daily_dd_pct < -5:
        would_pass = False
        breach_reason = f'Daily loss exceeded 5% ({max_daily_dd_pct:.2f}%)'
    elif max_total_dd_pct < -10:
        would_pass = False
        breach_reason = f'Total loss exceeded 10% ({max_total_dd_pct:.2f}%)'
    elif days_to_target is None:
        final_return = ((completed['equity'].iloc[-1] / initial_balance) - 1) * 100
        if final_return < 10:
            would_pass = False
            breach_reason = f'Did not reach 10% target ({final_return:.2f}%)'
    
    return {
        'max_daily_dd': round(max_daily_dd_pct, 2),
        'max_total_dd': round(max_total_dd_pct, 2),
        'days_to_10pct': days_to_target,
        'would_pass': would_pass,
        'breach_reason': breach_reason,
        'trading_days': len(daily_pnl),
        'final_return': round(((completed['equity'].iloc[-1] / initial_balance) - 1) * 100, 2)
    }


def run_extended_backtests(output_file: str = 'extended_results.json'):
    """Run backtests across all instrument types."""
    
    config = BacktestConfig(
        slippage_pct=0.0005,
        commission_pct=0.001,
        position_size=1.0,
        compound=True
    )
    
    all_results = []
    
    instrument_groups = [
        ('Forex', FOREX_PAIRS),
        ('Indices', INDICES),
        ('Commodities', COMMODITIES),
        ('Stocks', STOCKS),
    ]
    
    print("=" * 100)
    print("EXTENDED BACKTESTS - FTMO Focus")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)
    
    for group_name, symbols in instrument_groups:
        print(f"\n{'='*100}")
        print(f"GROUP: {group_name}")
        print("=" * 100)
        
        for interval in TIMEFRAMES:
            days = 59 if interval == '1h' else 365
            
            print(f"\n--- Timeframe: {interval} ({days} days) ---")
            
            for strategy_name in STRATEGIES.keys():
                for symbol in symbols:
                    try:
                        df = fetch_data(symbol, interval, days)
                        if df.empty or len(df) < 30:
                            continue
                        
                        engine = BacktestEngine(config)
                        trades_df = engine.run(df, STRATEGIES[strategy_name])
                        metrics = calculate_metrics(trades_df, df, config)
                        ftmo = calculate_ftmo_metrics(trades_df)
                        
                        result = {
                            'group': group_name,
                            'symbol': symbol,
                            'interval': interval,
                            'strategy': strategy_name,
                            'bars': len(df),
                            **metrics,
                            'ftmo': ftmo
                        }
                        all_results.append(result)
                        
                        # Print summary
                        beat = '✓' if metrics['strategy_return'] > metrics['buy_hold_return'] else '✗'
                        ftmo_ok = '✓' if ftmo['would_pass'] else '✗'
                        
                        print(f"  {symbol:12} {strategy_name:20} | "
                              f"{metrics['total_trades']:3}t | "
                              f"WR:{metrics['win_rate']:5.1f}% | "
                              f"Ret:{metrics['strategy_return']:7.2f}% {beat} | "
                              f"FTMO:{ftmo_ok}")
                        
                    except Exception as e:
                        print(f"  {symbol:12} {strategy_name:20} | ERROR: {str(e)[:40]}")
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Generate summary
    print("\n" + "=" * 100)
    print("SUMMARY BY INSTRUMENT GROUP")
    print("=" * 100)
    
    for group_name, _ in instrument_groups:
        group_results = [r for r in all_results if r.get('group') == group_name]
        if not group_results:
            continue
        
        total_trades = sum(r['total_trades'] for r in group_results)
        avg_wr = np.mean([r['win_rate'] for r in group_results if r['total_trades'] > 0])
        avg_ret = np.mean([r['strategy_return'] for r in group_results])
        ftmo_pass = sum(1 for r in group_results if r['ftmo']['would_pass'])
        
        print(f"\n{group_name} ({len(group_results)} tests):")
        print(f"  Total Trades: {total_trades}")
        print(f"  Avg Win Rate: {avg_wr:.1f}%")
        print(f"  Avg Return:   {avg_ret:.2f}%")
        print(f"  FTMO Pass:    {ftmo_pass}/{len(group_results)}")
    
    # Top FTMO candidates
    print("\n" + "=" * 100)
    print("TOP FTMO CANDIDATES (Would Pass + Best Return)")
    print("=" * 100)
    
    ftmo_candidates = [r for r in all_results 
                       if r['ftmo']['would_pass'] and r['total_trades'] >= 5]
    ftmo_candidates = sorted(ftmo_candidates, 
                            key=lambda x: x['strategy_return'], reverse=True)[:15]
    
    for i, r in enumerate(ftmo_candidates, 1):
        print(f"{i:2}. {r['symbol']:12} {r['strategy']:20} {r['interval']:3} | "
              f"Ret:{r['strategy_return']:7.2f}% | "
              f"DD:{r['ftmo']['max_total_dd']:6.2f}% | "
              f"Days:{r['ftmo'].get('days_to_10pct', 'N/A')}")
    
    print(f"\nTotal tests: {len(all_results)}")
    print(f"Results saved to: {output_file}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return all_results


if __name__ == "__main__":
    run_extended_backtests()
