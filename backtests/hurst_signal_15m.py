"""
Hurst Exponent Regime Switch Signal — 15-Minute EURUSD
-------------------------------------------------------
Adapted from hurst_signal.py for intraday bars:
- Rolling Hurst on 200 bars (~3 trading days of 15m data)
- 80-bar EMA for trend direction (replaces 20-bar daily)
- 40-bar momentum for mean-reversion fade (replaces 10-bar daily)
- ATR on 56 bars (≈14 daily bars worth of 15m bars)

Walk-forward: 6 windows, suppress signals during training period.
"""

import json
import warnings
import numpy as np
import pandas as pd
from datetime import timedelta

from engine import BacktestEngine, BacktestConfig, calculate_metrics
from hurst_signal import hurst_rs, rolling_hurst

warnings.filterwarnings('ignore')


# ─── Indicators ──────────────────────────────────────────────────────────

def calculate_indicators_15m(df: pd.DataFrame) -> pd.DataFrame:
    """Pre-calculate all indicators for 15m Hurst signal."""
    df = df.copy()

    # Log returns for Hurst calculation
    df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))

    # Rolling Hurst exponent on log returns — 200-bar window
    log_rets = df['log_ret'].values
    df['hurst'] = rolling_hurst(log_rets, window=200)

    # 80-bar EMA and its slope (replaces 20-bar daily)
    df['ema_80'] = df['Close'].ewm(span=80, adjust=False).mean()
    df['ema_slope'] = df['ema_80'] - df['ema_80'].shift(1)

    # 40-bar momentum for mean-reversion fade (replaces 10-bar daily)
    df['mom_40'] = df['Close'] - df['Close'].shift(40)

    # ATR on 56 bars (≈14 daily bars of 15m data: 14 * 4 bars/hour * ~1h? No, 14 daily = 14*96 too many.
    # Actually task says 56 bars ≈ 14 daily bars worth of 15m bars. Just use 56.)
    df['tr'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(
            abs(df['High'] - df['Close'].shift(1)),
            abs(df['Low'] - df['Close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(56).mean()

    return df


# ─── Signal function ────────────────────────────────────────────────────

def hurst_signal_15m(df: pd.DataFrame, i: int) -> dict:
    """
    Hurst exponent regime switch signal — 15m version.

    - H > 0.6: Trending → follow 80-bar EMA slope
    - H < 0.4: Mean-reverting → fade 40-bar momentum
    - 0.4-0.6: Random walk → no trade
    """
    row = df.iloc[i]

    if pd.isna(row.get('hurst')) or pd.isna(row.get('atr')) or \
       pd.isna(row.get('ema_slope')) or pd.isna(row.get('mom_40')):
        return None

    H = row['hurst']
    atr = row['atr']
    ema_slope = row['ema_slope']
    mom = row['mom_40']
    close = row['Close']

    if atr <= 0 or not np.isfinite(H):
        return None

    direction = None

    if H > 0.6:
        # Trending regime → follow 80-EMA slope
        if ema_slope > 0:
            direction = 'long'
        elif ema_slope < 0:
            direction = 'short'
    elif H < 0.4:
        # Mean-reverting regime → fade momentum
        if mom > 0:
            direction = 'short'  # Fade up move
        elif mom < 0:
            direction = 'long'  # Fade down move
    # else: 0.4-0.6 is random walk, no trade

    if direction is None:
        return None

    # ATR-based SL/TP with 1:2 R:R
    sl_dist = 1.5 * atr
    tp_dist = 3.0 * atr

    if direction == 'long':
        stop_loss = close - sl_dist
        take_profit = close + tp_dist
    else:
        stop_loss = close + sl_dist
        take_profit = close - tp_dist

    return {
        'direction': direction,
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        '_signal_open': close,
    }


# ─── Data loading ───────────────────────────────────────────────────────

def load_eurusd_15m() -> pd.DataFrame:
    """Load and prepare the EURUSD 15M CSV."""
    df = pd.read_csv('tickdata/EURUSD_15M_2Y.csv')
    df.rename(columns={
        'timestamp': 'Date',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
    }, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    return df


# ─── Walk-forward ────────────────────────────────────────────────────────

def walk_forward_15m(df: pd.DataFrame, signal_fn, indicator_fn,
                     n_windows: int = 6, train_months: int = 8, test_months: int = 4) -> list:
    """
    Walk-forward with CLEAN suppression of training signals.
    Indicators computed on full window (train+test), but signals only fire during test.
    """
    total_days = (df['Date'].iloc[-1] - df['Date'].iloc[0]).days
    train_days = train_months * 30
    test_days = test_months * 30
    step_days = test_days  # Step forward by test period

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

        if len(full_slice) < 250:
            continue

        # Calculate indicators on full window
        full_ind = indicator_fn(full_slice)

        # Find where test period starts
        test_start_idx = len(full_ind[full_ind['Date'] < train_end])

        if test_start_idx >= len(full_ind) - 10:
            continue

        test_bar_count = len(full_ind) - test_start_idx

        # Create signal wrapper that suppresses training signals
        def test_signal(df_inner, i, _start=test_start_idx, _fn=signal_fn):
            if i < _start:
                return None
            return _fn(df_inner, i)

        engine = BacktestEngine(config)
        trades_df = engine.run(full_ind, test_signal)

        # Filter trades to only those in test period
        if not trades_df.empty:
            trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
            trades_df = trades_df[trades_df['entry_time'] >= train_end]

        # Metrics on test slice
        test_slice = full_ind[full_ind['Date'] >= train_end].copy()
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
    print("HURST REGIME SWITCH on EURUSD 15-Minute Data")
    print("=" * 70)

    # Load data
    print("\n📥 Loading EURUSD 15M data...")
    df = load_eurusd_15m()
    print(f"  {len(df)} bars from {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")

    # Calculate indicators
    print("\n🔧 Calculating indicators (Hurst window=200, this may take a minute)...")
    df_ind = calculate_indicators_15m(df)

    # Count valid Hurst values
    valid_hurst = df_ind['hurst'].notna().sum()
    print(f"  Valid Hurst values: {valid_hurst}/{len(df_ind)}")
    if valid_hurst > 0:
        h_mean = df_ind['hurst'].mean()
        h_std = df_ind['hurst'].std()
        h_trending = (df_ind['hurst'] > 0.6).sum()
        h_meanrev = (df_ind['hurst'] < 0.4).sum()
        h_random = ((df_ind['hurst'] >= 0.4) & (df_ind['hurst'] <= 0.6)).sum()
        print(f"  Hurst mean={h_mean:.3f}, std={h_std:.3f}")
        print(f"  Trending (H>0.6): {h_trending} bars ({h_trending/valid_hurst*100:.1f}%)")
        print(f"  Mean-rev (H<0.4): {h_meanrev} bars ({h_meanrev/valid_hurst*100:.1f}%)")
        print(f"  Random (0.4-0.6): {h_random} bars ({h_random/valid_hurst*100:.1f}%)")

    # ── Full-period backtest ──
    print("\n📊 Full-period backtest...")
    config = BacktestConfig(slippage_pct=0.0005, commission_pct=0.001, compound=True)
    engine = BacktestEngine(config)
    trades_df = engine.run(df_ind, hurst_signal_15m)
    full_metrics = calculate_metrics(trades_df, df_ind, config)

    print(f"  Trades: {full_metrics['total_trades']}")
    print(f"  Win Rate: {full_metrics['win_rate']}%")
    print(f"  Profit Factor: {full_metrics['profit_factor']}")
    print(f"  Strategy Return: {full_metrics['strategy_return']}%")
    print(f"  Buy&Hold Return: {full_metrics['buy_hold_return']}%")
    print(f"  Max Drawdown: {full_metrics['max_drawdown']}%")
    print(f"  Expectancy: {full_metrics['expectancy']}%")
    print(f"  Sharpe: {full_metrics['sharpe']}")

    # ── Walk-Forward ──
    print("\n🔄 Walk-Forward Validation (6 windows, 8m train / 4m test)...")
    wf_results = walk_forward_15m(df, hurst_signal_15m, calculate_indicators_15m)

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
        'asset': 'EURUSD',
        'timeframe': '15M',
        'bars': len(df),
        'date_range': f"{df['Date'].iloc[0]} to {df['Date'].iloc[-1]}",
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

    with open('results_hurst_15m.json', 'w') as f:
        json.dump(all_results, f, indent=2, cls=NpEncoder)

    print(f"\n✅ Saved results_hurst_15m.json")
    return all_results


if __name__ == '__main__':
    main()
