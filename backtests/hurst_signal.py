"""
Hurst Exponent Regime Switch Signal
------------------------------------
Uses rolling Hurst exponent (R/S analysis) to detect market regime:
- H > 0.6: Trending → follow 20-EMA slope
- H < 0.4: Mean-reverting → fade 10-bar momentum
- 0.4-0.6: Random walk → no trade

Signal fires at bar i, entry at bar i+1 Open.
"""

import numpy as np
import pandas as pd


def hurst_rs(series: np.ndarray) -> float:
    """
    Compute Hurst exponent using Rescaled Range (R/S) analysis.
    Returns H in [0, 1]. H=0.5 is random walk.
    """
    N = len(series)
    if N < 20:
        return np.nan
    
    # We need several sub-periods to do proper R/S regression
    # Use divisors of N (or close to them)
    max_k = N // 2
    min_k = 8  # Minimum subseries length
    
    # Use logarithmically spaced chunk sizes
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
    
    # Fit log(R/S) = H * log(n) + c
    log_n = np.log(np.array(ns_values))
    log_rs = np.log(np.array(rs_values))
    
    # Linear regression
    try:
        coeffs = np.polyfit(log_n, log_rs, 1)
        H = coeffs[0]
        # Clamp to [0, 1]
        H = max(0.0, min(1.0, H))
        return H
    except:
        return np.nan


def rolling_hurst(series: np.ndarray, window: int = 50) -> np.ndarray:
    """Calculate rolling Hurst exponent on returns."""
    n = len(series)
    result = np.full(n, np.nan)
    
    for i in range(window - 1, n):
        segment = series[i - window + 1:i + 1]
        if np.any(np.isnan(segment)):
            continue
        result[i] = hurst_rs(segment)
    
    return result


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Pre-calculate all indicators needed for the signal."""
    df = df.copy()
    
    # Log returns for Hurst calculation
    df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Rolling Hurst exponent on log returns
    log_rets = df['log_ret'].values
    df['hurst'] = rolling_hurst(log_rets, window=50)
    
    # 20-bar EMA and its slope
    df['ema_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['ema_slope'] = df['ema_20'] - df['ema_20'].shift(1)
    
    # 10-bar momentum (for mean-reversion fade)
    df['mom_10'] = df['Close'] - df['Close'].shift(10)
    
    # ATR for SL/TP
    df['tr'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(
            abs(df['High'] - df['Close'].shift(1)),
            abs(df['Low'] - df['Close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(14).mean()
    
    return df


def hurst_signal(df: pd.DataFrame, i: int) -> dict:
    """
    Hurst exponent regime switch signal function.
    
    - H > 0.6: Trending → enter in direction of EMA slope
    - H < 0.4: Mean-reverting → fade 10-bar momentum
    - 0.4-0.6: Random walk → no trade
    """
    row = df.iloc[i]
    
    # Need all indicators valid
    if pd.isna(row.get('hurst')) or pd.isna(row.get('atr')) or pd.isna(row.get('ema_slope')) or pd.isna(row.get('mom_10')):
        return None
    
    H = row['hurst']
    atr = row['atr']
    ema_slope = row['ema_slope']
    mom = row['mom_10']
    close = row['Close']
    
    if atr <= 0 or not np.isfinite(H):
        return None
    
    direction = None
    
    if H > 0.6:
        # Trending regime → follow 20-EMA slope
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
