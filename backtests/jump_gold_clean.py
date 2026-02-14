"""
Jump Detection on Gold (GC=F) — CLEAN Walk-Forward
----------------------------------------------------
Uses the existing jump_signal.py logic (which is correct).
Re-runs with Batch 1's clean walk-forward method:
- Suppress signals during training period
- Reset jump_state between windows
- Slippage 0.05%, commission 0.1%
"""

import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import yfinance as yf

from engine import BacktestEngine, BacktestConfig, calculate_metrics
from jump_signal import (
    jump_signal, reset_jump_state,
    compute_jump_measures, compute_atr
)

warnings.filterwarnings('ignore')


# ─── Data ────────────────────────────────────────────────────────────────

def fetch_gold_daily(years: int = 2) -> pd.DataFrame:
    """Fetch 2 years of daily Gold (GC=F) data."""
    end = datetime.now()
    start = end - timedelta(days=years * 365 + 30)

    print(f"  Fetching GC=F...")
    ticker = yf.Ticker('GC=F')
    df = ticker.history(start=start.strftime('%Y-%m-%d'),
                        end=end.strftime('%Y-%m-%d'),
                        interval='1d')
    df.reset_index(inplace=True)
    if 'Datetime' in df.columns:
        df.rename(columns={'Datetime': 'Date'}, inplace=True)

    df['Date'] = pd.to_datetime(df['Date'])
    if df['Date'].dt.tz is not None:
        df['Date'] = df['Date'].dt.tz_localize(None)

    print(f"  Got {len(df)} bars from {df['Date'].iloc[0].date()} to {df['Date'].iloc[-1].date()}")
    return df


# ─── Walk-forward ────────────────────────────────────────────────────────

def walk_forward_clean(df: pd.DataFrame, n_windows: int = 6,
                       train_months: int = 8, test_months: int = 4) -> list:
    """
    CLEAN walk-forward: suppress training signals, reset state between windows.
    """
    train_days = train_months * 30
    test_days = test_months * 30
    step_days = test_days

    config = BacktestConfig(slippage_pct=0.0005, commission_pct=0.001, compound=True)
    results = []

    for w in range(n_windows):
        start_offset = w * step_days
        window_start = df['Date'].iloc[0] + timedelta(days=start_offset)
        train_end = window_start + timedelta(days=train_days)
        test_end = train_end + timedelta(days=test_days)

        # Get full window slice
        full_slice = df[(df['Date'] >= window_start) & (df['Date'] < test_end)].copy()
        full_slice = full_slice.reset_index(drop=True)

        if len(full_slice) < 50:
            continue

        # Find where test period starts
        test_start_idx = len(full_slice[full_slice['Date'] < train_end])

        if test_start_idx >= len(full_slice) - 5:
            continue

        test_bar_count = len(full_slice) - test_start_idx

        # CRITICAL: Reset jump state for each window
        reset_jump_state()

        # Create signal wrapper that suppresses training signals
        def test_signal(df_inner, i, _start=test_start_idx, _fn=jump_signal):
            if i < _start:
                return None
            return _fn(df_inner, i)

        engine = BacktestEngine(config)
        trades_df = engine.run(full_slice, test_signal)

        # Filter trades to only those in test period
        if not trades_df.empty:
            trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
            trades_df = trades_df[trades_df['entry_time'] >= train_end]

        # Metrics on test slice
        test_slice = full_slice[full_slice['Date'] >= train_end].copy()
        if len(test_slice) < 5:
            continue

        metrics = calculate_metrics(trades_df, test_slice, config) if not trades_df.empty and len(trades_df) > 0 else {
            'total_trades': 0, 'win_rate': 0, 'profit_factor': 0,
            'strategy_return': 0, 'buy_hold_return': 0, 'max_drawdown': 0,
            'avg_win': 0, 'avg_loss': 0, 'expectancy': 0, 'sharpe': 0,
        }

        results.append({
            'window': w + 1,
            'train_start': window_start.strftime('%Y-%m-%d'),
            'train_end': train_end.strftime('%Y-%m-%d'),
            'test_start': train_end.strftime('%Y-%m-%d'),
            'test_end': test_end.strftime('%Y-%m-%d'),
            'test_bars': test_bar_count,
            **metrics,
        })

    return results


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("JUMP DETECTION on GOLD — CLEAN Walk-Forward")
    print("=" * 70)

    # Fetch data
    print("\n📥 Fetching Gold (GC=F) data...")
    df = fetch_gold_daily(years=2)

    # ── Full-period backtest ──
    print("\n📊 Full-period backtest...")
    config = BacktestConfig(slippage_pct=0.0005, commission_pct=0.001, compound=True)

    reset_jump_state()
    engine = BacktestEngine(config)
    trades_df = engine.run(df, jump_signal)
    full_metrics = calculate_metrics(trades_df, df, config)

    # Count trade types
    if not trades_df.empty and '_trade_type' in trades_df.columns:
        n_fade = (trades_df['_trade_type'] == 'fade').sum() if '_trade_type' in trades_df.columns else 'N/A'
        n_trend = (trades_df['_trade_type'] == 'trend').sum() if '_trade_type' in trades_df.columns else 'N/A'
    else:
        n_fade = 'N/A'
        n_trend = 'N/A'

    print(f"  Trades: {full_metrics['total_trades']} (fade={n_fade}, trend={n_trend})")
    print(f"  Win Rate: {full_metrics['win_rate']}%")
    print(f"  Profit Factor: {full_metrics['profit_factor']}")
    print(f"  Strategy Return: {full_metrics['strategy_return']}%")
    print(f"  Buy&Hold Return: {full_metrics['buy_hold_return']}%")
    print(f"  Max Drawdown: {full_metrics['max_drawdown']}%")
    print(f"  Expectancy: {full_metrics['expectancy']}%")
    print(f"  Sharpe: {full_metrics['sharpe']}")

    # ── Walk-Forward ──
    print("\n🔄 Clean Walk-Forward (6 windows, 8m train / 4m test)...")
    wf_results = walk_forward_clean(df)

    profitable_windows = 0
    for w in wf_results:
        status = "✓" if w.get('strategy_return', 0) > 0 else "✗"
        print(f"  Window {w['window']}: {w.get('strategy_return', 0):+.2f}% | "
              f"WR={w.get('win_rate', 0):.0f}% | PF={w.get('profit_factor', 0):.2f} | "
              f"Trades={w.get('total_trades', 0)} | Bars={w.get('test_bars', 0)} {status}")
        if w.get('strategy_return', 0) > 0:
            profitable_windows += 1

    print(f"\n  OOS Profitable: {profitable_windows}/{len(wf_results)}")
    edge = "✅ EDGE" if profitable_windows > len(wf_results) / 2 else "❌ NO EDGE"
    print(f"  Verdict: {edge}")

    # Save results
    all_results = {
        'asset': 'Gold (GC=F)',
        'timeframe': '1D',
        'bars': len(df),
        'date_range': f"{df['Date'].iloc[0].date()} to {df['Date'].iloc[-1].date()}",
        'full_backtest': full_metrics,
        'walk_forward': wf_results,
        'profitable_windows': profitable_windows,
        'total_windows': len(wf_results),
    }

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, pd.Timestamp):
                return str(obj)
            return super().default(obj)

    with open('results_jump_gold_clean.json', 'w') as f:
        json.dump(all_results, f, indent=2, cls=NpEncoder)

    print(f"\n✅ Saved results_jump_gold_clean.json")
    return all_results


if __name__ == '__main__':
    main()
