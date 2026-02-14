"""
Hurst Regime Switch on Crypto H1 (Hourly)
==========================================
Adapts the daily Hurst exponent signal for hourly bars.

Key differences from daily version:
- Hurst window: 168 bars (1 week of hourly data) vs 50 daily bars
- EMA: 72 bars (~3 days) vs 20 daily bars
- Momentum: 48 bars (~2 days) vs 10 daily bars
- ATR: 56 bars vs 14 daily bars

Assets: LINK-USD, ADA-USD, XRP-USD
Walk-forward: 6 windows, 8 months train + 4 months test
Clean method: suppress signals during training period
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
import yfinance as yf

from engine import BacktestEngine, BacktestConfig, calculate_metrics


# ─── Hurst R/S calculation (same as daily version) ─────────────────────

def hurst_rs(series: np.ndarray) -> float:
    """
    Compute Hurst exponent using Rescaled Range (R/S) analysis.
    Returns H in [0, 1]. H=0.5 is random walk.
    """
    N = len(series)
    if N < 20:
        return np.nan
    
    max_k = N // 2
    min_k = 8
    
    ks = []
    k = min_k
    while k <= max_k:
        ks.append(k)
        k = int(k * 1.5)
        if k == ks[-1]:
            k += 1
    
    if len(ks) < 3:
        return np.nan
    
    rs_values = []
    ns_values = []
    
    for k in ks:
        n_chunks = N // k
        if n_chunks < 1:
            continue
        
        rs_list = []
        for chunk_i in range(n_chunks):
            chunk = series[chunk_i * k:(chunk_i + 1) * k]
            mean_chunk = np.mean(chunk)
            deviations = chunk - mean_chunk
            cumulative = np.cumsum(deviations)
            R = np.max(cumulative) - np.min(cumulative)
            S = np.std(chunk, ddof=1)
            if S > 0:
                rs_list.append(R / S)
        
        if rs_list:
            rs_values.append(np.mean(rs_list))
            ns_values.append(k)
    
    if len(ns_values) < 3:
        return np.nan
    
    log_n = np.log(np.array(ns_values))
    log_rs = np.log(np.array(rs_values))
    
    try:
        coeffs = np.polyfit(log_n, log_rs, 1)
        H = coeffs[0]
        H = max(0.0, min(1.0, H))
        return H
    except:
        return np.nan


def rolling_hurst(series: np.ndarray, window: int = 168, step: int = 4) -> np.ndarray:
    """
    Calculate rolling Hurst exponent on returns.
    Uses step parameter to compute every Nth bar and forward-fill.
    With 168-bar window, computing every 4 bars is fine (< 2.5% window drift).
    """
    n = len(series)
    result = np.full(n, np.nan)
    
    for i in range(window - 1, n, step):
        segment = series[i - window + 1:i + 1]
        if np.any(np.isnan(segment)):
            continue
        h = hurst_rs(segment)
        # Fill this and next step-1 bars
        for j in range(i, min(i + step, n)):
            result[j] = h
    
    return result


# ─── Data preparation ───────────────────────────────────────────────────

def prepare_h1_data(symbol: str) -> pd.DataFrame:
    """Download 2y of hourly data."""
    ticker = yf.Ticker(symbol)
    df = ticker.history(period='2y', interval='1h')
    df.reset_index(inplace=True)
    
    if 'Datetime' in df.columns:
        df.rename(columns={'Datetime': 'Date'}, inplace=True)
    if 'index' in df.columns:
        df.rename(columns={'index': 'Date'}, inplace=True)
    
    # Ensure timezone-naive
    if hasattr(df['Date'].dtype, 'tz') and df['Date'].dt.tz is not None:
        df['Date'] = df['Date'].dt.tz_localize(None)
    
    print(f"  Downloaded {len(df)} H1 bars for {symbol}")
    print(f"  Date range: {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")
    
    return df


def calculate_h1_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Pre-calculate all indicators for H1 Hurst signal."""
    df = df.copy()
    
    # Log returns for Hurst calculation
    df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Rolling Hurst exponent (168 bars = 1 week of hourly data)
    print("    Computing rolling Hurst (168-bar window)... this may take a minute")
    log_rets = df['log_ret'].values
    df['hurst'] = rolling_hurst(log_rets, window=168)
    
    # EMA 72 bars (~3 days) and its slope
    df['ema_72'] = df['Close'].ewm(span=72, adjust=False).mean()
    df['ema_slope'] = df['ema_72'] - df['ema_72'].shift(1)
    
    # Momentum 48 bars (~2 days) for mean-reversion fade
    df['mom_48'] = df['Close'] - df['Close'].shift(48)
    
    # ATR 56 bars
    df['tr'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(
            abs(df['High'] - df['Close'].shift(1)),
            abs(df['Low'] - df['Close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(56).mean()
    
    # Count valid Hurst values
    valid_hurst = df['hurst'].notna().sum()
    trending = (df['hurst'] > 0.6).sum()
    mean_rev = (df['hurst'] < 0.4).sum()
    print(f"    Valid Hurst values: {valid_hurst}/{len(df)}")
    print(f"    Trending (H>0.6): {trending} bars ({trending/max(valid_hurst,1)*100:.1f}%)")
    print(f"    Mean-reverting (H<0.4): {mean_rev} bars ({mean_rev/max(valid_hurst,1)*100:.1f}%)")
    
    return df


# ─── Signal function ────────────────────────────────────────────────────

def hurst_h1_signal(df: pd.DataFrame, i: int) -> dict:
    """
    Hurst exponent regime switch signal on H1 bars.
    
    - H > 0.6: Trending → follow 72-EMA slope
    - H < 0.4: Mean-reverting → fade 48-bar momentum
    - 0.4-0.6: No trade
    
    SL = 1.5×ATR, TP = 3×ATR (1:2 R:R)
    """
    row = df.iloc[i]
    
    # Need all indicators valid
    if pd.isna(row.get('hurst')) or pd.isna(row.get('atr')) or \
       pd.isna(row.get('ema_slope')) or pd.isna(row.get('mom_48')):
        return None
    
    H = row['hurst']
    atr = row['atr']
    ema_slope = row['ema_slope']
    mom = row['mom_48']
    close = row['Close']
    
    if atr <= 0 or not np.isfinite(H):
        return None
    
    direction = None
    
    if H > 0.6:
        # Trending regime → follow 72-EMA slope
        if ema_slope > 0:
            direction = 'long'
        elif ema_slope < 0:
            direction = 'short'
    elif H < 0.4:
        # Mean-reverting regime → fade momentum
        if mom > 0:
            direction = 'short'  # Fade up move
        elif mom < 0:
            direction = 'long'   # Fade down move
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


# ─── Walk-forward with clean suppression ─────────────────────────────────

def test_signal(df, i, start_idx, signal_fn):
    """Suppress signals during training period."""
    if i < start_idx:
        return None
    return signal_fn(df, i)


def walk_forward_clean(df: pd.DataFrame, signal_fn, n_windows=6,
                       train_months=8, test_months=4):
    """Walk-forward with clean method: suppress training-period signals."""
    total_bars = len(df)
    
    # Estimate bars per month from data
    date_range_secs = (df['Date'].iloc[-1] - df['Date'].iloc[0]).total_seconds()
    date_range_months = date_range_secs / (30 * 24 * 3600)
    bars_per_month = total_bars / date_range_months if date_range_months > 0 else total_bars / 24
    
    train_bars = int(train_months * bars_per_month)
    test_bars = int(test_months * bars_per_month)
    step = test_bars
    
    print(f"  WF: {total_bars} total bars, ~{bars_per_month:.0f} bars/month")
    print(f"  WF: train={train_bars} bars, test={test_bars} bars, step={step}")
    
    results = []
    
    for w in range(n_windows):
        start = w * step
        train_end = start + train_bars
        test_end = train_end + test_bars
        
        if test_end > total_bars:
            break
        
        window_df = df.iloc[start:test_end].reset_index(drop=True)
        test_start_idx = train_bars
        
        def make_signal_fn(start_idx):
            def wrapped(df_inner, i):
                return test_signal(df_inner, i, start_idx, signal_fn)
            return wrapped
        
        engine = BacktestEngine(BacktestConfig())
        wrapped_fn = make_signal_fn(test_start_idx)
        trades_df = engine.run(window_df, wrapped_fn)
        
        test_price_df = window_df.iloc[test_start_idx:].reset_index(drop=True)
        metrics = calculate_metrics(trades_df, test_price_df)
        
        test_start_date = str(df.iloc[min(train_end, len(df)-1)]['Date'])[:10]
        test_end_date = str(df.iloc[min(test_end-1, len(df)-1)]['Date'])[:10]
        
        results.append({
            'window': w + 1,
            'test_start': test_start_date,
            'test_end': test_end_date,
            'trades': metrics['total_trades'],
            'return': metrics['strategy_return'],
            'win_rate': metrics['win_rate'],
            'profit_factor': metrics['profit_factor'],
            'max_dd': metrics['max_drawdown'],
            'sharpe': metrics['sharpe'],
            'avg_win': metrics['avg_win'],
            'avg_loss': metrics['avg_loss'],
        })
    
    return results


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    symbols = ['LINK-USD', 'ADA-USD', 'XRP-USD']
    all_results = {}
    
    for sym in symbols:
        print(f"\n{'='*70}")
        print(f"  HURST REGIME SWITCH H1: {sym}")
        print(f"{'='*70}")
        
        # 1. Download data
        print(f"\n[1] Downloading {sym} hourly data...")
        df = prepare_h1_data(sym)
        
        if len(df) < 500:
            print(f"  SKIP: Not enough data ({len(df)} bars)")
            all_results[sym] = {'error': 'insufficient data'}
            continue
        
        # 2. Calculate indicators
        print(f"\n[2] Calculating H1 indicators...")
        df = calculate_h1_indicators(df)
        
        # 3. Full backtest
        print(f"\n[3] Full-period backtest...")
        engine = BacktestEngine(BacktestConfig())
        trades_df = engine.run(df, hurst_h1_signal)
        full_metrics = calculate_metrics(trades_df, df)
        
        print(f"  Trades: {full_metrics['total_trades']}")
        print(f"  Win Rate: {full_metrics['win_rate']}%")
        print(f"  Profit Factor: {full_metrics['profit_factor']}")
        print(f"  Strategy Return: {full_metrics['strategy_return']}%")
        print(f"  Buy&Hold Return: {full_metrics['buy_hold_return']}%")
        print(f"  Max Drawdown: {full_metrics['max_drawdown']}%")
        print(f"  Sharpe: {full_metrics['sharpe']}")
        
        # 4. Walk-forward
        print(f"\n[4] Walk-Forward (6 windows, 8m train / 4m test)...")
        wf_results = walk_forward_clean(df, hurst_h1_signal, n_windows=6,
                                         train_months=8, test_months=4)
        
        profitable_windows = 0
        for w in wf_results:
            status = "✓" if w['return'] > 0 else "✗"
            print(f"    Window {w['window']}: {w['return']:+.2f}% | "
                  f"WR={w['win_rate']:.0f}% | PF={w['profit_factor']:.2f} | "
                  f"Trades={w['trades']} | DD={w['max_dd']:.1f}% {status}")
            if w['return'] > 0:
                profitable_windows += 1
        
        print(f"\n  OOS Profitable: {profitable_windows}/{len(wf_results)}")
        
        all_results[sym] = {
            'timeframe': 'H1',
            'full_backtest': full_metrics,
            'walk_forward': wf_results,
            'profitable_windows': profitable_windows,
            'total_windows': len(wf_results),
            'data_bars': len(df),
        }
    
    # Save results
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, pd.Timestamp):
                return str(obj)
            return super().default(obj)
    
    with open('results_hurst_crypto_h1.json', 'w') as f:
        json.dump(all_results, f, indent=2, cls=NpEncoder)
    
    print(f"\n\nResults saved to results_hurst_crypto_h1.json")
    return all_results


if __name__ == '__main__':
    main()
