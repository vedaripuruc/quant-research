"""
Entropy Regime Detector Signal
------------------------------
Uses Approximate Entropy (ApEn) to detect regime changes:
- Low entropy (predictable) → trend-follow
- High entropy (chaotic) → mean-revert

Signal fires at bar i, entry at bar i+1 Open.
"""

import numpy as np
import pandas as pd


def approx_entropy(data: np.ndarray, m: int = 2, r_mult: float = 0.2) -> float:
    """
    Calculate Approximate Entropy (ApEn) of a time series.
    More robust than SampEn for short series.
    m: embedding dimension
    r_mult: tolerance as fraction of std
    """
    N = len(data)
    if N < m + 2:
        return np.nan
    
    r = r_mult * np.std(data)
    if r == 0:
        return np.nan
    
    def phi(dim):
        templates = np.array([data[i:i + dim] for i in range(N - dim + 1)])
        n = len(templates)
        counts = np.zeros(n)
        for i in range(n):
            # Count how many templates are within tolerance (including self)
            dists = np.max(np.abs(templates - templates[i]), axis=1)
            counts[i] = np.sum(dists <= r)
        # Average of log(count / n)
        return np.mean(np.log(counts / n))
    
    try:
        return phi(m) - phi(m + 1)
    except:
        return np.nan


def rolling_approx_entropy(log_returns: np.ndarray, window: int = 20, 
                           m: int = 2, r_mult: float = 0.2) -> np.ndarray:
    """Calculate rolling ApEn over log returns."""
    n = len(log_returns)
    result = np.full(n, np.nan)
    
    for i in range(window - 1, n):
        segment = log_returns[i - window + 1:i + 1]
        if np.any(np.isnan(segment)):
            continue
        ae = approx_entropy(segment, m=m, r_mult=r_mult)
        if np.isfinite(ae):
            result[i] = ae
    
    return result


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Pre-calculate all indicators needed for the signal."""
    df = df.copy()
    
    # Log returns
    df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Rolling ApEn on log returns (more robust than SampEn for short windows)
    log_rets = df['log_ret'].values
    df['sampen'] = rolling_approx_entropy(log_rets, window=20, m=2, r_mult=0.2)
    
    # Use expanding z-score with min_periods=30 (more realistic for available data)
    df['sampen_mean'] = df['sampen'].expanding(min_periods=30).mean()
    df['sampen_std'] = df['sampen'].expanding(min_periods=30).std()
    df['sampen_zscore'] = (df['sampen'] - df['sampen_mean']) / df['sampen_std']
    
    # 10-bar momentum (for trend direction)
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


def entropy_signal(df: pd.DataFrame, i: int) -> dict:
    """
    Entropy regime detector signal function.
    
    - SampEn z-score < -1.5: Market becoming predictable → trend-follow
    - SampEn z-score > +1.5: Market becoming chaotic → mean-revert (fade)
    """
    row = df.iloc[i]
    
    # Need all indicators valid
    if pd.isna(row.get('sampen_zscore')) or pd.isna(row.get('atr')) or pd.isna(row.get('mom_10')):
        return None
    
    zscore = row['sampen_zscore']
    atr = row['atr']
    mom = row['mom_10']
    close = row['Close']
    
    if atr <= 0 or not np.isfinite(zscore):
        return None
    
    direction = None
    
    if zscore < -1.5:
        # Low entropy → predictable/trending → go WITH trend
        if mom > 0:
            direction = 'long'
        elif mom < 0:
            direction = 'short'
    elif zscore > 1.5:
        # High entropy → chaotic → FADE the move (mean reversion)
        if mom > 0:
            direction = 'short'  # Fade the up move
        elif mom < 0:
            direction = 'long'   # Fade the down move
    
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
