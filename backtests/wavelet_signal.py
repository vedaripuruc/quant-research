"""
Signal 3: Wavelet Multi-Scale Momentum
=======================================
Uses Discrete Wavelet Transform (DWT) with 'db4' wavelet to decompose
price into multiple scales, then aligns slow trend + medium cycle momentum.

LONG:  slow trend UP   AND medium cycle UP
SHORT: slow trend DOWN AND medium cycle DOWN
FLAT:  conflicting scales

Rolling window of 60 bars. SL = 1.5*ATR, TP = 3*ATR (1:2 R:R).
"""

import numpy as np
import pandas as pd
import pywt
import json
import sys
from datetime import datetime, timedelta
import yfinance as yf

# Import engine
from engine import BacktestEngine, BacktestConfig, calculate_metrics


# ─── Wavelet helpers ────────────────────────────────────────────────────

def wavelet_decompose(prices: np.ndarray, wavelet='db4', level=3):
    """
    Decompose price series using DWT.
    Returns reconstructed approximation (slow trend) and detail coefficients.
    """
    # Ensure minimum length for decomposition
    min_len = pywt.dwt_coeff_len(len(prices), pywt.Wavelet(wavelet).dec_len, mode='symmetric')
    if len(prices) < 2 ** level:
        return None, None, None, None

    coeffs = pywt.wavedec(prices, wavelet, level=level, mode='symmetric')
    # coeffs = [cA3, cD3, cD2, cD1]

    # Reconstruct each component
    # cA3 (slow trend): zero out all detail coeffs
    slow_coeffs = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
    slow_trend = pywt.waverec(slow_coeffs, wavelet, mode='symmetric')[:len(prices)]

    # cD3 (medium cycle): zero out approx and other details
    med_coeffs = [np.zeros_like(coeffs[0])] + [coeffs[1]] + [np.zeros_like(c) for c in coeffs[2:]]
    medium_cycle = pywt.waverec(med_coeffs, wavelet, mode='symmetric')[:len(prices)]

    # cD2 (fast cycle)
    fast_coeffs = [np.zeros_like(coeffs[0]), np.zeros_like(coeffs[1]), coeffs[2]] + \
                  [np.zeros_like(c) for c in coeffs[3:]]
    fast_cycle = pywt.waverec(fast_coeffs, wavelet, mode='symmetric')[:len(prices)]

    # cD1 (noise)
    noise_coeffs = [np.zeros_like(c) for c in coeffs[:-1]] + [coeffs[-1]]
    noise = pywt.waverec(noise_coeffs, wavelet, mode='symmetric')[:len(prices)]

    return slow_trend, medium_cycle, fast_cycle, noise


def compute_momentum(series: np.ndarray, lookback: int = 5) -> float:
    """Compute slope/momentum of last `lookback` points using simple regression."""
    if len(series) < lookback:
        return 0.0
    y = series[-lookback:]
    x = np.arange(lookback)
    # Simple linear regression slope
    slope = np.polyfit(x, y, 1)[0]
    return slope


# ─── ATR calculation ────────────────────────────────────────────────────

def compute_atr(df: pd.DataFrame, i: int, period: int = 14) -> float:
    """Compute ATR at bar i."""
    if i < period:
        return None
    highs = df['High'].iloc[i - period + 1:i + 1].values
    lows = df['Low'].iloc[i - period + 1:i + 1].values
    closes = df['Close'].iloc[i - period:i].values  # previous closes

    tr = np.maximum(highs - lows,
                    np.maximum(np.abs(highs - closes),
                               np.abs(lows - closes)))
    return np.mean(tr)


# ─── Signal function ────────────────────────────────────────────────────

def wavelet_signal(df: pd.DataFrame, i: int, window: int = 60, 
                   momentum_lookback: int = 8) -> dict:
    """
    Wavelet multi-scale momentum signal.
    
    Uses rolling 60-bar window for DWT decomposition.
    Signal fires when slow trend and medium cycle agree on direction.
    """
    if i < window:
        return None

    # Get ATR for SL/TP
    atr = compute_atr(df, i, period=14)
    if atr is None or atr <= 0:
        return None

    # Extract rolling window of closes
    closes = df['Close'].iloc[i - window + 1:i + 1].values.astype(float)

    # Decompose
    slow, medium, fast, noise = wavelet_decompose(closes)
    if slow is None:
        return None

    # Calculate momentum of slow trend and medium cycle
    slow_mom = compute_momentum(slow, lookback=momentum_lookback)
    med_mom = compute_momentum(medium, lookback=momentum_lookback)

    # Normalize momentum relative to price level for comparability
    price_level = closes[-1]
    slow_mom_norm = slow_mom / price_level if price_level > 0 else 0
    med_mom_norm = med_mom / price_level if price_level > 0 else 0

    # Thresholds: require meaningful momentum (not just noise)
    mom_threshold = 0.0001  # 0.01% per bar minimum

    direction = None
    if slow_mom_norm > mom_threshold and med_mom_norm > mom_threshold:
        direction = 'long'
    elif slow_mom_norm < -mom_threshold and med_mom_norm < -mom_threshold:
        direction = 'short'
    else:
        return None  # Conflicting scales

    close = df['Close'].iloc[i]

    if direction == 'long':
        sl = close - 1.5 * atr
        tp = close + 3.0 * atr
    else:
        sl = close + 1.5 * atr
        tp = close - 3.0 * atr

    return {
        'direction': direction,
        'stop_loss': sl,
        'take_profit': tp,
        '_signal_open': close,
    }


# ─── Fetch data ─────────────────────────────────────────────────────────

def fetch_daily_data(symbol: str, years: int = 2) -> pd.DataFrame:
    """Fetch daily OHLCV data for `years` years."""
    end = datetime.now()
    start = end - timedelta(days=years * 365)
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start.strftime('%Y-%m-%d'),
                        end=end.strftime('%Y-%m-%d'),
                        interval='1d')
    df.reset_index(inplace=True)
    if 'Datetime' in df.columns:
        df.rename(columns={'Datetime': 'Date'}, inplace=True)
    return df


# ─── Walk-forward ────────────────────────────────────────────────────────

def walk_forward(df: pd.DataFrame, signal_fn, n_windows=6,
                 train_months=8, test_months=4):
    """
    Rolling walk-forward validation.
    Returns list of per-window OOS metrics.
    """
    total_bars = len(df)
    window_bars = int(total_bars / (n_windows + (train_months / (train_months + test_months))))
    
    # Calculate bar counts
    total_period = train_months + test_months
    bars_per_month = total_bars / 24  # ~24 months of data
    train_bars = int(train_months * bars_per_month)
    test_bars = int(test_months * bars_per_month)
    step = test_bars  # Step forward by test period each window

    results = []
    for w in range(n_windows):
        start = w * step
        train_end = start + train_bars
        test_end = train_end + test_bars

        if test_end > total_bars:
            break

        # OOS test period only (we don't optimize, so train = just establishing context)
        test_df = df.iloc[start:test_end].reset_index(drop=True)

        engine = BacktestEngine(BacktestConfig())
        trades_df = engine.run(test_df, signal_fn)
        metrics = calculate_metrics(trades_df, test_df)

        results.append({
            'window': w + 1,
            'test_start': str(df.iloc[train_end]['Date'] if train_end < len(df) else 'N/A'),
            'test_end': str(df.iloc[min(test_end, len(df) - 1)]['Date']),
            'trades': metrics['total_trades'],
            'return': metrics['strategy_return'],
            'win_rate': metrics['win_rate'],
            'profit_factor': metrics['profit_factor'],
            'max_dd': metrics['max_drawdown'],
        })

    return results


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    symbols = ['EURUSD=X', 'LINK-USD', 'ADA-USD', 'XRP-USD', 'GC=F', 'BTC-USD']
    all_results = {}

    for sym in symbols:
        print(f"\n{'='*60}")
        print(f"  WAVELET SIGNAL: {sym}")
        print(f"{'='*60}")

        df = fetch_daily_data(sym, years=2)
        print(f"  Data: {len(df)} bars, {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")

        if len(df) < 100:
            print(f"  SKIP: Not enough data ({len(df)} bars)")
            all_results[sym] = {'error': 'insufficient data'}
            continue

        # Full backtest
        engine = BacktestEngine(BacktestConfig())
        trades_df = engine.run(df, wavelet_signal)
        metrics = calculate_metrics(trades_df, df)

        print(f"  Trades: {metrics['total_trades']}")
        print(f"  Win Rate: {metrics['win_rate']}%")
        print(f"  Profit Factor: {metrics['profit_factor']}")
        print(f"  Strategy Return: {metrics['strategy_return']}%")
        print(f"  Buy&Hold Return: {metrics['buy_hold_return']}%")
        print(f"  Max Drawdown: {metrics['max_drawdown']}%")

        # Walk-forward
        print(f"\n  Walk-Forward (6 windows, 8m train / 4m test):")
        wf_results = walk_forward(df, wavelet_signal)
        profitable_windows = 0
        for w in wf_results:
            status = "✓" if w['return'] > 0 else "✗"
            print(f"    Window {w['window']}: {w['return']:+.2f}% | "
                  f"WR={w['win_rate']:.0f}% | PF={w['profit_factor']:.2f} | "
                  f"Trades={w['trades']} {status}")
            if w['return'] > 0:
                profitable_windows += 1

        print(f"  OOS Profitable: {profitable_windows}/{len(wf_results)}")

        all_results[sym] = {
            'full_backtest': metrics,
            'walk_forward': wf_results,
            'profitable_windows': profitable_windows,
            'total_windows': len(wf_results),
        }

    # Save results
    # Convert any non-serializable types
    def clean(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, pd.Timestamp):
            return str(obj)
        return obj

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, pd.Timestamp):
                return str(obj)
            return super().default(obj)

    with open('results_wavelet.json', 'w') as f:
        json.dump(all_results, f, indent=2, cls=NpEncoder)

    print(f"\n\nResults saved to results_wavelet.json")
    return all_results


if __name__ == '__main__':
    main()
