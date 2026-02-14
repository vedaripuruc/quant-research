"""
FVG Multi-Timeframe Research - Forex Focus
==========================================
Testing fractal price delivery across timeframes with session filtering.

Hypothesis: Lower TFs show cleaner gap fills, Asian session is more orderly.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from typing import Optional, Dict, List
import json
import yfinance as yf

from engine import BacktestEngine, BacktestConfig, calculate_metrics
from fvg_optimized import FVGTracker, strategy_magnet, strategy_wall, STRATEGIES


def fetch_multitf_data(symbol: str, interval: str, days: int) -> pd.DataFrame:
    """Fetch data at specified interval."""
    end = datetime.now()
    start = end - timedelta(days=days)
    
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start.strftime('%Y-%m-%d'), 
                        end=end.strftime('%Y-%m-%d'),
                        interval=interval)
    df.reset_index(inplace=True)
    
    if 'Datetime' in df.columns:
        df.rename(columns={'Datetime': 'Date'}, inplace=True)
    if 'index' in df.columns:
        df.rename(columns={'index': 'Date'}, inplace=True)
    
    return df


def is_asian_session(dt) -> bool:
    """
    Check if datetime is in Asian session (Tokyo).
    Asian session: 00:00 - 09:00 UTC (Tokyo 09:00 - 18:00)
    """
    if hasattr(dt, 'hour'):
        hour = dt.hour
    else:
        hour = pd.Timestamp(dt).hour
    
    # Asian session roughly 00:00-09:00 UTC
    return 0 <= hour < 9


def is_london_session(dt) -> bool:
    """London session: 08:00 - 17:00 UTC"""
    if hasattr(dt, 'hour'):
        hour = dt.hour
    else:
        hour = pd.Timestamp(dt).hour
    return 8 <= hour < 17


def is_ny_session(dt) -> bool:
    """NY session: 13:00 - 22:00 UTC"""
    if hasattr(dt, 'hour'):
        hour = dt.hour
    else:
        hour = pd.Timestamp(dt).hour
    return 13 <= hour < 22


SESSION_FILTERS = {
    'all': lambda dt: True,
    'asian': is_asian_session,
    'london': is_london_session,
    'ny': is_ny_session,
}


def run_backtest_with_session(symbol: str, strategy_name: str,
                               interval: str = '1h',
                               session: str = 'all',
                               sl_mult: float = 1.0, tp_mult: float = 2.0,
                               days: int = 30) -> Dict:
    """Run backtest with session filter."""
    
    df = fetch_multitf_data(symbol, interval, days)
    
    if len(df) < 50:
        return {'error': f'Insufficient data: {len(df)} bars'}
    
    # Adjust gap parameters based on timeframe
    if interval in ['5m', '15m']:
        min_gap_pct = 0.005  # Smaller gaps on lower TF
        max_age = 24  # 2 hours on 5m
    elif interval == '30m':
        min_gap_pct = 0.01
        max_age = 24  # 12 hours
    else:
        min_gap_pct = 0.01
        max_age = 48
    
    tracker = FVGTracker(min_gap_pct=min_gap_pct, max_age=max_age)
    strategy_fn = STRATEGIES[strategy_name]
    session_filter = SESSION_FILTERS[session]
    
    def signal_fn(df_inner, i):
        tracker.update(df_inner, i)
        
        # Check session filter
        bar_time = df_inner.iloc[i]['Date'] if 'Date' in df_inner.columns else df_inner.index[i]
        if not session_filter(bar_time):
            return None
        
        return strategy_fn(df_inner, i, tracker, sl_mult, tp_mult)
    
    config = BacktestConfig(
        slippage_pct=0.0005,
        commission_pct=0.0003,
        compound=True,
    )
    
    engine = BacktestEngine(config)
    trades_df = engine.run(df, signal_fn)
    metrics = calculate_metrics(trades_df, df, config)
    
    return {
        'symbol': symbol,
        'interval': interval,
        'session': session,
        'strategy': strategy_name,
        'params': {'sl_mult': sl_mult, 'tp_mult': tp_mult},
        'metrics': metrics,
        'bars': len(df),
    }


def run_full_research():
    """Run multi-TF, multi-session research on forex."""
    
    SYMBOLS = ['EURUSD=X', 'USDJPY=X', 'GBPUSD=X']
    INTERVALS = ['5m', '15m', '30m', '1h', '4h']
    SESSIONS = ['all', 'asian', 'london', 'ny']
    STRATEGIES_TO_TEST = ['magnet', 'wall']
    
    # Adjust days based on interval (yfinance limits)
    DAYS_BY_INTERVAL = {
        '5m': 7,    # Max ~7 days for 5m
        '15m': 30,
        '30m': 30,
        '1h': 59,
        '4h': 59,
    }
    
    results = {
        'meta': {
            'run_date': datetime.now().isoformat(),
            'symbols': SYMBOLS,
            'intervals': INTERVALS,
            'sessions': SESSIONS,
        },
        'all': [],
    }
    
    total = len(SYMBOLS) * len(INTERVALS) * len(SESSIONS) * len(STRATEGIES_TO_TEST)
    print(f"Running {total} backtests...")
    
    for symbol in SYMBOLS:
        print(f"\n{'='*60}")
        print(f"{symbol}")
        print(f"{'='*60}")
        
        for interval in INTERVALS:
            days = DAYS_BY_INTERVAL[interval]
            print(f"\n  {interval} ({days}d):")
            
            for session in SESSIONS:
                for strategy in STRATEGIES_TO_TEST:
                    result = run_backtest_with_session(
                        symbol, strategy,
                        interval=interval,
                        session=session,
                        days=days,
                    )
                    
                    if 'error' not in result:
                        results['all'].append(result)
                        m = result['metrics']
                        if m['total_trades'] > 0:
                            status = '✓' if m['strategy_return'] > 0 else ' '
                            print(f"    {status} {session:6} {strategy:8} "
                                  f"Ret={m['strategy_return']:+6.2f}% "
                                  f"WR={m['win_rate']:5.1f}% "
                                  f"PF={m['profit_factor']:4.2f} "
                                  f"T={m['total_trades']}")
    
    # Summarize by timeframe
    print("\n" + "="*60)
    print("SUMMARY BY TIMEFRAME")
    print("="*60)
    
    for interval in INTERVALS:
        tf_results = [r for r in results['all'] if r['interval'] == interval]
        if tf_results:
            profitable = len([r for r in tf_results if r['metrics']['strategy_return'] > 0])
            avg_ret = np.mean([r['metrics']['strategy_return'] for r in tf_results])
            avg_pf = np.mean([r['metrics']['profit_factor'] for r in tf_results])
            print(f"{interval:4}: Avg Ret: {avg_ret:+6.2f}% | PF: {avg_pf:.2f} | "
                  f"Profitable: {profitable}/{len(tf_results)}")
    
    # Summarize by session
    print("\n" + "="*60)
    print("SUMMARY BY SESSION")
    print("="*60)
    
    for session in SESSIONS:
        sess_results = [r for r in results['all'] if r['session'] == session]
        if sess_results:
            profitable = len([r for r in sess_results if r['metrics']['strategy_return'] > 0])
            avg_ret = np.mean([r['metrics']['strategy_return'] for r in sess_results])
            avg_pf = np.mean([r['metrics']['profit_factor'] for r in sess_results])
            print(f"{session:7}: Avg Ret: {avg_ret:+6.2f}% | PF: {avg_pf:.2f} | "
                  f"Profitable: {profitable}/{len(sess_results)}")
    
    # Find best combo
    print("\n" + "="*60)
    print("TOP 10 CONFIGURATIONS")
    print("="*60)
    
    sorted_results = sorted(results['all'], 
                           key=lambda x: x['metrics']['strategy_return'],
                           reverse=True)
    
    for r in sorted_results[:10]:
        m = r['metrics']
        print(f"{r['symbol']:10} {r['interval']:4} {r['session']:7} {r['strategy']:8} "
              f"Ret={m['strategy_return']:+6.2f}% WR={m['win_rate']:5.1f}% "
              f"PF={m['profit_factor']:.2f} T={m['total_trades']}")
    
    # Save
    with open('fvg_multitf_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("\nSaved: fvg_multitf_results.json")
    
    return results


if __name__ == '__main__':
    print("="*60)
    print("FVG MULTI-TIMEFRAME RESEARCH - FOREX")
    print("Timeframes: 5m, 15m, 30m, 1h, 4h")
    print("Sessions: All, Asian, London, NY")
    print("="*60)
    
    results = run_full_research()
