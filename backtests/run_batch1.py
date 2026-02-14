"""
Batch 1 Runner: Entropy + Hurst signals
----------------------------------------
Backtests both signals on 6 assets with:
- Full-period backtest
- Walk-forward validation (6 rolling windows: 8m train, 4m test)
- Results saved to JSON
"""

import json
import sys
import warnings
import traceback
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

from engine import BacktestEngine, BacktestConfig, calculate_metrics
from entropy_signal import calculate_indicators as entropy_indicators, entropy_signal
from hurst_signal import calculate_indicators as hurst_indicators, hurst_signal

warnings.filterwarnings('ignore')

SYMBOLS = ['EURUSD=X', 'LINK-USD', 'ADA-USD', 'XRP-USD', 'GC=F', 'BTC-USD']
SYMBOL_NAMES = {
    'EURUSD=X': 'EUR/USD',
    'LINK-USD': 'LINK',
    'ADA-USD': 'ADA',
    'XRP-USD': 'XRP',
    'GC=F': 'Gold',
    'BTC-USD': 'BTC',
}


def fetch_daily_data(symbol: str, years: int = 2) -> pd.DataFrame:
    """Fetch 2 years of daily data."""
    end = datetime.now()
    start = end - timedelta(days=years * 365 + 30)  # Extra buffer
    
    print(f"  Fetching {symbol}...")
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start.strftime('%Y-%m-%d'),
                        end=end.strftime('%Y-%m-%d'),
                        interval='1d')
    df.reset_index(inplace=True)
    if 'Datetime' in df.columns:
        df.rename(columns={'Datetime': 'Date'}, inplace=True)
    
    # Ensure Date is datetime
    df['Date'] = pd.to_datetime(df['Date'])
    # Remove timezone if present
    if df['Date'].dt.tz is not None:
        df['Date'] = df['Date'].dt.tz_localize(None)
    
    print(f"    Got {len(df)} bars from {df['Date'].iloc[0].date()} to {df['Date'].iloc[-1].date()}")
    return df


def run_backtest(df: pd.DataFrame, signal_fn, indicator_fn, symbol: str) -> dict:
    """Run a full-period backtest for one asset."""
    # Calculate indicators
    df_ind = indicator_fn(df)
    
    config = BacktestConfig(
        slippage_pct=0.0005,
        commission_pct=0.001,
        position_size=1.0,
        compound=True,
    )
    engine = BacktestEngine(config)
    trades_df = engine.run(df_ind, signal_fn)
    metrics = calculate_metrics(trades_df, df_ind, config)
    
    return metrics


def walk_forward_test(df: pd.DataFrame, signal_fn, indicator_fn,
                      n_windows: int = 6, train_months: int = 8, test_months: int = 4) -> list:
    """
    Walk-forward validation with rolling windows.
    Each window: train on train_months, test on test_months.
    Windows overlap by stepping forward test_months at a time.
    """
    total_days = (df['Date'].iloc[-1] - df['Date'].iloc[0]).days
    train_days = train_months * 30
    test_days = test_months * 30
    window_days = train_days + test_days
    step_days = test_days  # Step forward by test period
    
    results = []
    
    for w in range(n_windows):
        start_offset = w * step_days
        window_start = df['Date'].iloc[0] + timedelta(days=start_offset)
        train_end = window_start + timedelta(days=train_days)
        test_end = train_end + timedelta(days=test_days)
        
        # Get train and test slices
        train_df = df[(df['Date'] >= window_start) & (df['Date'] < train_end)].copy()
        test_df = df[(df['Date'] >= train_end) & (df['Date'] < test_end)].copy()
        
        if len(train_df) < 50 or len(test_df) < 20:
            continue
        
        # Calculate indicators on FULL data up to test end (to avoid NaN at start of test)
        full_slice = df[(df['Date'] >= window_start) & (df['Date'] < test_end)].copy()
        full_slice = full_slice.reset_index(drop=True)
        full_ind = indicator_fn(full_slice)
        
        # Find where test period starts in full_ind
        test_start_idx = len(full_ind[full_ind['Date'] < train_end])
        
        # Run backtest only on test period, but with indicators from full
        config = BacktestConfig(slippage_pct=0.0005, commission_pct=0.001, compound=True)
        engine = BacktestEngine(config)
        
        # Create a signal function that only fires during test period
        def test_signal(df_inner, i, _start=test_start_idx, _fn=signal_fn):
            if i < _start:
                return None
            return _fn(df_inner, i)
        
        trades_df = engine.run(full_ind, test_signal)
        
        # Filter trades to only those in test period
        if not trades_df.empty:
            trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
            trades_df = trades_df[trades_df['entry_time'] >= train_end]
        
        metrics = calculate_metrics(trades_df, test_df, config) if not trades_df.empty else {
            'total_trades': 0, 'win_rate': 0, 'profit_factor': 0,
            'strategy_return': 0, 'max_drawdown': 0,
        }
        
        results.append({
            'window': w + 1,
            'train_start': window_start.strftime('%Y-%m-%d'),
            'train_end': train_end.strftime('%Y-%m-%d'),
            'test_start': train_end.strftime('%Y-%m-%d'),
            'test_end': test_end.strftime('%Y-%m-%d'),
            'test_bars': len(test_df),
            **metrics,
        })
    
    return results


def main():
    print("=" * 70)
    print("BATCH 1: Entropy + Hurst Signal Backtests")
    print("=" * 70)
    
    # Fetch all data first
    print("\n📥 Fetching data...")
    data = {}
    for sym in SYMBOLS:
        try:
            data[sym] = fetch_daily_data(sym)
        except Exception as e:
            print(f"  ❌ Failed to fetch {sym}: {e}")
    
    # ============== SIGNAL 1: ENTROPY ==============
    print("\n" + "=" * 70)
    print("🔬 SIGNAL 1: Entropy Regime Detector")
    print("=" * 70)
    
    entropy_results = {'full_period': {}, 'walk_forward': {}}
    
    for sym in SYMBOLS:
        if sym not in data:
            continue
        name = SYMBOL_NAMES[sym]
        print(f"\n  📊 {name} ({sym})")
        
        df = data[sym].copy()
        
        # Full period backtest
        try:
            metrics = run_backtest(df, entropy_signal, entropy_indicators, sym)
            entropy_results['full_period'][sym] = metrics
            print(f"    Full: {metrics['total_trades']} trades, "
                  f"WR={metrics['win_rate']}%, PF={metrics['profit_factor']}, "
                  f"Return={metrics['strategy_return']}%, MDD={metrics['max_drawdown']}%")
        except Exception as e:
            print(f"    ❌ Full backtest error: {e}")
            traceback.print_exc()
            entropy_results['full_period'][sym] = {'error': str(e)}
        
        # Walk-forward
        try:
            wf = walk_forward_test(df, entropy_signal, entropy_indicators)
            entropy_results['walk_forward'][sym] = wf
            profitable_windows = sum(1 for w in wf if w.get('strategy_return', 0) > 0)
            print(f"    WF: {profitable_windows}/{len(wf)} profitable OOS windows")
        except Exception as e:
            print(f"    ❌ Walk-forward error: {e}")
            traceback.print_exc()
            entropy_results['walk_forward'][sym] = {'error': str(e)}
    
    # Save entropy results
    with open('results_entropy.json', 'w') as f:
        json.dump(entropy_results, f, indent=2, default=str)
    print("\n✅ Saved results_entropy.json")
    
    # ============== SIGNAL 2: HURST ==============
    print("\n" + "=" * 70)
    print("🔬 SIGNAL 2: Hurst Exponent Regime Switch")
    print("=" * 70)
    
    hurst_results = {'full_period': {}, 'walk_forward': {}}
    
    for sym in SYMBOLS:
        if sym not in data:
            continue
        name = SYMBOL_NAMES[sym]
        print(f"\n  📊 {name} ({sym})")
        
        df = data[sym].copy()
        
        # Full period backtest
        try:
            metrics = run_backtest(df, hurst_signal, hurst_indicators, sym)
            hurst_results['full_period'][sym] = metrics
            print(f"    Full: {metrics['total_trades']} trades, "
                  f"WR={metrics['win_rate']}%, PF={metrics['profit_factor']}, "
                  f"Return={metrics['strategy_return']}%, MDD={metrics['max_drawdown']}%")
        except Exception as e:
            print(f"    ❌ Full backtest error: {e}")
            traceback.print_exc()
            hurst_results['full_period'][sym] = {'error': str(e)}
        
        # Walk-forward
        try:
            wf = walk_forward_test(df, hurst_signal, hurst_indicators)
            hurst_results['walk_forward'][sym] = wf
            profitable_windows = sum(1 for w in wf if w.get('strategy_return', 0) > 0)
            print(f"    WF: {profitable_windows}/{len(wf)} profitable OOS windows")
        except Exception as e:
            print(f"    ❌ Walk-forward error: {e}")
            traceback.print_exc()
            hurst_results['walk_forward'][sym] = {'error': str(e)}
    
    # Save hurst results
    with open('results_hurst.json', 'w') as f:
        json.dump(hurst_results, f, indent=2, default=str)
    print("\n✅ Saved results_hurst.json")
    
    # ============== SUMMARY ==============
    print("\n" + "=" * 70)
    print("📋 SUMMARY")
    print("=" * 70)
    
    for signal_name, results in [("Entropy", entropy_results), ("Hurst", hurst_results)]:
        print(f"\n--- {signal_name} Signal ---")
        print(f"{'Asset':<10} {'Trades':>7} {'WinRate':>8} {'PF':>6} {'Return':>8} {'MaxDD':>8}")
        print("-" * 50)
        for sym in SYMBOLS:
            if sym in results['full_period'] and 'error' not in results['full_period'][sym]:
                m = results['full_period'][sym]
                name = SYMBOL_NAMES[sym]
                print(f"{name:<10} {m['total_trades']:>7} {m['win_rate']:>7.1f}% {m['profit_factor']:>6.2f} "
                      f"{m['strategy_return']:>7.2f}% {m['max_drawdown']:>7.2f}%")
        
        # Walk-forward summary
        print(f"\nWalk-Forward (profitable OOS windows):")
        for sym in SYMBOLS:
            if sym in results['walk_forward'] and isinstance(results['walk_forward'][sym], list):
                wf = results['walk_forward'][sym]
                profitable = sum(1 for w in wf if w.get('strategy_return', 0) > 0)
                total = len(wf)
                name = SYMBOL_NAMES[sym]
                edge = "✅ EDGE" if profitable > total / 2 else "❌ NO EDGE"
                print(f"  {name:<10}: {profitable}/{total} {edge}")
    
    print("\n🏁 Done!")


if __name__ == '__main__':
    main()
