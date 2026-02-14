#!/usr/bin/env python3
"""
FTMO Challenge Simulation
-------------------------
Simulate a $100k FTMO challenge with the best forex strategies.
Track daily drawdown (5% limit) and total drawdown (10% limit).
"""
import warnings
warnings.filterwarnings('ignore')
import sys
sys.stdout.reconfigure(line_buffering=True)

import pandas as pd
import numpy as np
from datetime import datetime

from engine import BacktestEngine, BacktestConfig, fetch_data
from strategies_forex import FOREX_STRATEGIES

# FTMO Parameters
INITIAL_BALANCE = 100000
PROFIT_TARGET = 10000  # 10%
MAX_DAILY_LOSS = 5000  # 5%
MAX_TOTAL_LOSS = 10000  # 10%
MIN_TRADING_DAYS = 4

# Best forex strategies from testing
BEST_STRATEGIES = ['breakout_fx', 'session_breakout_fx']
FOREX_PAIRS = ['EURUSD=X', 'USDJPY=X', 'EURJPY=X']  # Best performers

forex_config = BacktestConfig(
    slippage_pct=0.0001,
    commission_pct=0.0003,
    position_size=1.0,  # 100% position size (full capital, like earlier tests)
    compound=True
)

def simulate_ftmo_challenge(pair: str, strategy_name: str, strategy_fn, days: int = 59):
    """Simulate an FTMO challenge and return detailed results"""
    
    df = fetch_data(pair, '1h', days)
    if len(df) < 30:
        return None
    
    engine = BacktestEngine(forex_config)
    trades_df = engine.run(df, strategy_fn)
    
    if trades_df.empty:
        return {
            'pair': pair,
            'strategy': strategy_name,
            'passed': False,
            'reason': 'No trades',
            'final_balance': INITIAL_BALANCE,
            'max_daily_dd': 0,
            'max_total_dd': 0,
            'trading_days': 0,
            'total_trades': 0,
        }
    
    completed = trades_df[trades_df['exit_price'].notna()].copy()
    if completed.empty:
        return {
            'pair': pair,
            'strategy': strategy_name,
            'passed': False,
            'reason': 'No completed trades',
            'final_balance': INITIAL_BALANCE,
            'max_daily_dd': 0,
            'max_total_dd': 0,
            'trading_days': 0,
            'total_trades': 0,
        }
    
    # Calculate P&L in dollars
    completed['pnl_dollars'] = completed['pnl_pct'] * INITIAL_BALANCE * forex_config.position_size
    completed['cumulative_pnl'] = completed['pnl_dollars'].cumsum()
    completed['equity'] = INITIAL_BALANCE + completed['cumulative_pnl']
    
    # Daily P&L tracking
    completed['trade_date'] = pd.to_datetime(completed['exit_time']).dt.date
    daily_pnl = completed.groupby('trade_date')['pnl_dollars'].sum()
    
    # Check daily loss limit
    max_daily_loss_hit = daily_pnl.min()
    max_daily_dd_pct = (max_daily_loss_hit / INITIAL_BALANCE) * 100
    daily_breach = max_daily_loss_hit < -MAX_DAILY_LOSS
    
    # Check total loss limit
    peak = completed['equity'].expanding().max()
    drawdown = completed['equity'] - peak
    max_total_dd = drawdown.min()
    max_total_dd_pct = (max_total_dd / INITIAL_BALANCE) * 100
    total_breach = max_total_dd < -MAX_TOTAL_LOSS
    
    # Check profit target
    final_balance = completed['equity'].iloc[-1]
    hit_target = final_balance >= INITIAL_BALANCE + PROFIT_TARGET
    
    # Check min trading days
    trading_days = len(daily_pnl)
    enough_days = trading_days >= MIN_TRADING_DAYS
    
    # Determine pass/fail
    passed = hit_target and not daily_breach and not total_breach and enough_days
    
    reason = None
    if daily_breach:
        reason = f'Daily loss exceeded 5% ({max_daily_dd_pct:.2f}%)'
    elif total_breach:
        reason = f'Total loss exceeded 10% ({max_total_dd_pct:.2f}%)'
    elif not enough_days:
        reason = f'Not enough trading days ({trading_days} < {MIN_TRADING_DAYS})'
    elif not hit_target:
        final_return = ((final_balance / INITIAL_BALANCE) - 1) * 100
        reason = f'Did not hit 10% target ({final_return:.2f}%)'
    else:
        reason = 'PASSED!'
    
    return {
        'pair': pair,
        'strategy': strategy_name,
        'passed': passed,
        'reason': reason,
        'final_balance': round(final_balance, 2),
        'final_return': round(((final_balance / INITIAL_BALANCE) - 1) * 100, 2),
        'max_daily_dd': round(max_daily_dd_pct, 2),
        'max_total_dd': round(max_total_dd_pct, 2),
        'trading_days': trading_days,
        'total_trades': len(completed),
        'win_rate': round(len(completed[completed['result'] == 'win']) / len(completed) * 100, 1),
    }


print("=" * 80)
print("FTMO CHALLENGE SIMULATION")
print(f"Account: ${INITIAL_BALANCE:,} | Target: +${PROFIT_TARGET:,} (10%)")
print(f"Max Daily Loss: ${MAX_DAILY_LOSS:,} (5%) | Max Total Loss: ${MAX_TOTAL_LOSS:,} (10%)")
print(f"Position Size: {forex_config.position_size*100:.0f}% per trade")
print("=" * 80)

all_results = []

for strategy_name in BEST_STRATEGIES:
    strategy_fn = FOREX_STRATEGIES[strategy_name]
    
    print(f"\n--- {strategy_name.upper()} ---")
    
    for pair in FOREX_PAIRS:
        result = simulate_ftmo_challenge(pair, strategy_name, strategy_fn)
        if result:
            all_results.append(result)
            
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            print(f"  {pair:10} | {status} | Ret:{result['final_return']:+6.2f}% | "
                  f"Daily DD:{result['max_daily_dd']:5.2f}% | "
                  f"Total DD:{result['max_total_dd']:5.2f}% | "
                  f"{result['total_trades']}t/{result['trading_days']}d")
            if not result['passed']:
                print(f"            └─ {result['reason']}")

# Summary
print("\n" + "=" * 80)
print("FTMO SIMULATION SUMMARY")
print("=" * 80)

passed = [r for r in all_results if r['passed']]
failed = [r for r in all_results if not r['passed']]

print(f"\nPassed: {len(passed)}/{len(all_results)}")

if passed:
    print("\n✅ WOULD PASS FTMO:")
    for r in passed:
        print(f"   {r['pair']} + {r['strategy']}: +{r['final_return']:.2f}%")

if failed:
    print("\n❌ WOULD FAIL FTMO:")
    for r in failed:
        print(f"   {r['pair']} + {r['strategy']}: {r['reason']}")

# Recommendations
print("\n" + "=" * 80)
print("RECOMMENDATIONS FOR FTMO")
print("=" * 80)
print("""
1. POSITION SIZING is key - 2% per trade is conservative
   At 2% risk with 60%+ win rate, you need ~50 winning trades to hit 10%
   Consider 3-5% per trade for faster target (but higher risk)

2. BEST PAIRS for these strategies:
   - USD/JPY (higher ATR, cleaner trends)
   - EUR/JPY (good volatility)
   - EUR/USD (tightest spreads)

3. TIMEFRAME: 1h is good, 4h might be better for FTMO
   (fewer false signals, larger moves per trade)

4. AVOID:
   - Williams %R on forex (still negative edge)
   - MA Crossover (too many whipsaws)

5. FTMO-SPECIFIC:
   - No trading 2 min before/after news
   - Weekend gaps can trigger stops
   - Minimum 4 trading days required
""")

print("\nDone!")
