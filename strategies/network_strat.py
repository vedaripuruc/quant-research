"""
VG-TDA Regime-Aware Trading Strategy
=====================================
Combines Visibility Graph degree analysis with Topological Data Analysis
(persistent homology) for regime-aware intraday trading.

Based on:
- Serafino et al. (2017) - Visibility graph validation for financial instability
- Gidea & Katz (2017) - TDA persistence landscapes for crash detection
- Huang et al. (2024) - TDA features for stock index prediction

Works with 1-minute or 1-hour OHLCV data.
Entry at NEXT bar open after signal (no look-ahead bias).

Dependencies:
    pip install numpy pandas
    pip install ts2vg        # Fast visibility graph (C backend)
    pip install giotto-tda   # TDA / persistent homology
    # OR: pip install ripser persim  (alternative TDA stack)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PART 1: VISIBILITY GRAPH FEATURES
# ============================================================================

def natural_visibility_graph_degrees(series: np.ndarray) -> np.ndarray:
    """
    Build Natural Visibility Graph and return node degrees.
    
    Two points (t_a, y_a) and (t_b, y_b) are connected if for all t_c 
    between them: y_c < y_a + (y_b - y_a) * (t_c - t_a) / (t_b - t_a)
    
    This is the O(n^2) reference implementation. For production, use ts2vg.
    """
    n = len(series)
    degrees = np.zeros(n, dtype=int)
    
    for i in range(n):
        for j in range(i + 2, n):
            # Check if all intermediate points are below the line from i to j
            visible = True
            for k in range(i + 1, j):
                # Line from (i, series[i]) to (j, series[j]) evaluated at k
                line_val = series[i] + (series[j] - series[i]) * (k - i) / (j - i)
                if series[k] >= line_val:
                    visible = False
                    break
            if visible:
                degrees[i] += 1
                degrees[j] += 1
        # Adjacent nodes are always connected
        if i < n - 1:
            degrees[i] += 1
            if i == 0:
                pass  # Will be counted when j = i+1
            # Actually, adjacent already counted in j = i+1 loop
    
    # Simpler correct approach: just count adjacencies
    # Reset and use proper algorithm
    degrees = np.zeros(n, dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            visible = True
            for k in range(i + 1, j):
                line_val = series[i] + (series[j] - series[i]) * (k - i) / (j - i)
                if series[k] >= line_val:
                    visible = False
                    break
            if visible:
                degrees[i] += 1
                degrees[j] += 1
    
    return degrees


def vg_degree_fast(series: np.ndarray) -> np.ndarray:
    """
    Fast visibility graph using ts2vg library.
    Falls back to pure Python if ts2vg not installed.
    """
    try:
        from ts2vg import NaturalVG
        vg = NaturalVG()
        vg.build(np.array(series, dtype=np.float64, copy=True))
        degrees = vg.degrees
        return np.array(degrees)
    except ImportError:
        return natural_visibility_graph_degrees(series)


def compute_vg_features(close: np.ndarray, window: int = 50) -> dict:
    """
    Compute Visibility Graph features on a rolling window.
    
    Returns dict of arrays:
        - vg_last_degree_ratio: degree of last node / (window-1), normalized [0, 1]
        - vg_mean_degree: mean degree in window  
        - vg_degree_std: std of degrees in window
        - vg_max_degree_ratio: max degree / (window-1)
    """
    n = len(close)
    last_degree_ratio = np.full(n, np.nan)
    mean_degree = np.full(n, np.nan)
    degree_std = np.full(n, np.nan)
    max_degree_ratio = np.full(n, np.nan)
    
    for i in range(window - 1, n):
        chunk = close[i - window + 1: i + 1]
        degrees = vg_degree_fast(chunk)
        max_possible = window - 1  # Maximum degree in a VG of size window
        
        last_degree_ratio[i] = degrees[-1] / max_possible
        mean_degree[i] = np.mean(degrees) / max_possible
        degree_std[i] = np.std(degrees) / max_possible
        max_degree_ratio[i] = np.max(degrees) / max_possible
    
    return {
        'vg_last_degree_ratio': last_degree_ratio,
        'vg_mean_degree': mean_degree,
        'vg_degree_std': degree_std,
        'vg_max_degree_ratio': max_degree_ratio,
    }


# ============================================================================
# PART 2: TDA PERSISTENCE FEATURES (Takens Embedding + Persistent Homology)
# ============================================================================

def takens_embedding(series: np.ndarray, dim: int = 4, delay: int = 1) -> np.ndarray:
    """
    Create Takens delay embedding of a 1D time series.
    
    Returns point cloud in R^dim.
    Shape: (n - (dim-1)*delay, dim)
    """
    n = len(series)
    n_points = n - (dim - 1) * delay
    if n_points <= 0:
        return np.array([]).reshape(0, dim)
    
    embedding = np.zeros((n_points, dim))
    for d in range(dim):
        embedding[:, d] = series[d * delay: d * delay + n_points]
    
    return embedding


def persistence_l1_norm_giotto(point_cloud: np.ndarray) -> float:
    """
    Compute L1 norm of H1 persistence diagram using giotto-tda.
    L1 norm = sum of (death - birth) for all H1 features.
    """
    try:
        from gtda.homology import VietorisRipsPersistence
        
        # giotto-tda expects 3D input: (n_samples, n_points, n_features)
        pc = point_cloud.reshape(1, *point_cloud.shape)
        
        vr = VietorisRipsPersistence(
            homology_dimensions=[1],
            max_edge_length=np.inf,
            n_jobs=1
        )
        diagrams = vr.fit_transform(pc)
        
        # diagrams shape: (1, n_features, 3) where columns are (birth, death, dim)
        diag = diagrams[0]
        h1_mask = diag[:, 2] == 1
        h1_features = diag[h1_mask]
        
        if len(h1_features) == 0:
            return 0.0
        
        lifetimes = h1_features[:, 1] - h1_features[:, 0]
        # Filter out infinite lifetimes
        lifetimes = lifetimes[np.isfinite(lifetimes)]
        
        return np.sum(lifetimes)
    
    except ImportError:
        return persistence_l1_norm_ripser(point_cloud)


def persistence_l1_norm_ripser(point_cloud: np.ndarray) -> float:
    """
    Compute L1 norm of H1 persistence diagram using ripser.
    """
    try:
        from ripser import ripser
        
        result = ripser(point_cloud, maxdim=1)
        h1 = result['dgms'][1]  # H1 diagram
        
        if len(h1) == 0:
            return 0.0
        
        lifetimes = h1[:, 1] - h1[:, 0]
        lifetimes = lifetimes[np.isfinite(lifetimes)]
        
        return np.sum(lifetimes)
    
    except ImportError:
        return persistence_l1_norm_pure(point_cloud)


def persistence_l1_norm_pure(point_cloud: np.ndarray) -> float:
    """
    Lightweight proxy for persistent homology L1 norm using
    distance matrix statistics (when neither giotto-tda nor ripser available).
    
    Approximation: Use the spread of pairwise distances as a proxy for
    topological complexity. Not real persistent homology but captures
    similar information about point cloud structure.
    """
    from scipy.spatial.distance import pdist
    
    if len(point_cloud) < 4:
        return 0.0
    
    dists = pdist(point_cloud)
    
    # Proxy: coefficient of variation of pairwise distances
    # High CV = complex structure with multiple scales = "more loops"
    if np.mean(dists) == 0:
        return 0.0
    
    # Use a combination of distance statistics as TDA proxy
    mean_d = np.mean(dists)
    std_d = np.std(dists)
    q75 = np.percentile(dists, 75)
    q25 = np.percentile(dists, 25)
    
    # Interquartile range relative to mean ~ topological complexity
    proxy = (q75 - q25) * std_d / (mean_d + 1e-10)
    
    return proxy


def compute_tda_features(log_returns: np.ndarray, 
                          window: int = 50,
                          embed_dim: int = 4,
                          embed_delay: int = 1) -> dict:
    """
    Compute TDA persistence features on rolling windows.
    
    Returns dict of arrays:
        - tda_l1_norm: L1 norm of H1 persistence landscape per window
        - tda_l1_zscore: rolling z-score of L1 norm
    """
    n = len(log_returns)
    l1_norms = np.full(n, np.nan)
    
    min_points_needed = window  # We need at least 'window' returns
    
    for i in range(min_points_needed - 1, n):
        chunk = log_returns[i - min_points_needed + 1: i + 1]
        
        # Remove NaN
        chunk = chunk[~np.isnan(chunk)]
        if len(chunk) < embed_dim * embed_delay + 5:
            continue
        
        # Takens embedding
        pc = takens_embedding(chunk, dim=embed_dim, delay=embed_delay)
        
        if len(pc) < 5:
            continue
        
        # Compute persistence L1 norm
        l1_norms[i] = persistence_l1_norm_giotto(pc)
    
    # Rolling z-score of L1 norms (using 100-bar lookback)
    zscore_window = 100
    l1_zscore = np.full(n, np.nan)
    for i in range(zscore_window - 1, n):
        chunk = l1_norms[i - zscore_window + 1: i + 1]
        valid = chunk[~np.isnan(chunk)]
        if len(valid) >= 10:
            mean_val = np.mean(valid)
            std_val = np.std(valid)
            if std_val > 1e-10 and not np.isnan(l1_norms[i]):
                l1_zscore[i] = (l1_norms[i] - mean_val) / std_val
    
    return {
        'tda_l1_norm': l1_norms,
        'tda_l1_zscore': l1_zscore,
    }


# ============================================================================
# PART 3: SIGNAL GENERATION
# ============================================================================

def generate_signals(df: pd.DataFrame,
                     vg_window: int = 50,
                     tda_window: int = 50,
                     sma_period: int = 20,
                     vg_high_threshold: float = 0.5,
                     vg_low_threshold: float = 0.2,
                     tda_unstable_z: float = 1.5,
                     tda_stable_z: float = -0.5,
                     ) -> pd.DataFrame:
    """
    Generate trading signals from OHLCV data.
    
    Parameters:
        df: DataFrame with columns [timestamp, open, high, low, close, volume]
        vg_window: window size for visibility graph
        tda_window: window size for TDA features
        sma_period: SMA period for trend context
        vg_high_threshold: VG degree ratio above which = local extreme
        vg_low_threshold: VG degree ratio below which = trapped in channel
        tda_unstable_z: TDA z-score above which = unstable regime (reduce/avoid)
        tda_stable_z: TDA z-score below which = stable/trending regime
    
    Returns:
        df with added signal columns
    """
    df = df.copy()
    close = df['close'].values
    
    # ---- Compute features ----
    
    # 1. SMA for trend context
    df['sma'] = df['close'].rolling(sma_period).mean()
    
    # 2. Log returns
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # 3. VG features
    print(f"Computing VG features (window={vg_window})...")
    vg_feats = compute_vg_features(close, window=vg_window)
    for k, v in vg_feats.items():
        df[k] = v
    
    # 4. TDA features
    print(f"Computing TDA features (window={tda_window})...")
    tda_feats = compute_tda_features(
        df['log_return'].values, 
        window=tda_window,
        embed_dim=4,
        embed_delay=1
    )
    for k, v in tda_feats.items():
        df[k] = v
    
    # 5. Short-term momentum (5-bar return)
    df['momentum_5'] = df['close'].pct_change(5)
    
    # ---- Generate signals ----
    
    # Regime classification
    df['regime'] = 'normal'
    df.loc[df['tda_l1_zscore'] > tda_unstable_z, 'regime'] = 'unstable'
    df.loc[df['tda_l1_zscore'] < tda_stable_z, 'regime'] = 'stable'
    
    # Position sizing factor based on regime
    df['regime_size'] = 1.0
    df.loc[df['regime'] == 'unstable', 'regime_size'] = 0.0  # No trading in unstable
    df.loc[df['regime'] == 'stable', 'regime_size'] = 1.0
    # 'normal' = 0.5 (reduced)
    df.loc[df['regime'] == 'normal', 'regime_size'] = 0.5
    
    # Raw signal: -1, 0, +1
    df['raw_signal'] = 0
    
    for i in range(len(df)):
        vg_ratio = df['vg_last_degree_ratio'].iloc[i]
        close_val = df['close'].iloc[i]
        sma_val = df['sma'].iloc[i]
        momentum = df['momentum_5'].iloc[i]
        
        if pd.isna(vg_ratio) or pd.isna(sma_val) or pd.isna(momentum):
            continue
        
        # High VG degree = local extreme → mean reversion
        if vg_ratio > vg_high_threshold:
            if close_val > sma_val:
                df.iloc[i, df.columns.get_loc('raw_signal')] = -1  # SHORT (revert from high)
            else:
                df.iloc[i, df.columns.get_loc('raw_signal')] = 1   # LONG (revert from low)
        
        # Low VG degree = trapped/trending → momentum following
        elif vg_ratio < vg_low_threshold:
            if momentum > 0:
                df.iloc[i, df.columns.get_loc('raw_signal')] = 1   # LONG (follow momentum)
            elif momentum < 0:
                df.iloc[i, df.columns.get_loc('raw_signal')] = -1  # SHORT (follow momentum)
    
    # Apply regime filter
    df['signal'] = df['raw_signal'] * df['regime_size']
    
    # CRITICAL: Entry at NEXT bar open (shift signal forward by 1)
    df['position'] = df['signal'].shift(1)
    
    return df


# ============================================================================
# PART 4: BACKTEST ENGINE
# ============================================================================

def backtest(df: pd.DataFrame,
             initial_capital: float = 100000.0,
             position_size_pct: float = 0.10,  # 10% of capital per trade
             commission_pct: float = 0.001,     # 0.1% round trip
             holding_period: int = 5,           # Bars to hold
             ) -> pd.DataFrame:
    """
    Simple backtest loop.
    
    - Enters at open of bar AFTER signal (already shifted in generate_signals)
    - Holds for 'holding_period' bars
    - Exits at open of bar after holding period
    - No overlapping positions (one position at a time)
    
    Returns df with equity curve and trade log.
    """
    df = df.copy()
    
    equity = initial_capital
    position = 0          # +1 long, -1 short, 0 flat
    entry_price = 0.0
    entry_bar = 0
    trade_size = 0.0
    
    equity_curve = np.full(len(df), np.nan)
    trades = []
    
    for i in range(1, len(df)):
        current_open = df['open'].iloc[i]
        signal = df['position'].iloc[i]
        
        if pd.isna(signal):
            signal = 0
        
        # Check if we need to exit (holding period expired)
        if position != 0 and (i - entry_bar) >= holding_period:
            exit_price = current_open
            pnl = position * (exit_price - entry_price) * trade_size
            commission = abs(trade_size * exit_price * commission_pct)
            equity += pnl - commission
            
            trades.append({
                'entry_bar': entry_bar,
                'exit_bar': i,
                'direction': 'LONG' if position > 0 else 'SHORT',
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl - commission,
                'return_pct': (pnl - commission) / (trade_size * entry_price) * 100,
            })
            
            position = 0
        
        # Check for new entry
        if position == 0 and abs(signal) > 0.01:
            direction = np.sign(signal)
            sizing_factor = abs(signal)  # regime-adjusted
            
            trade_notional = equity * position_size_pct * sizing_factor
            trade_size = trade_notional / current_open
            
            position = direction
            entry_price = current_open
            entry_bar = i
            
            # Entry commission
            equity -= abs(trade_size * current_open * commission_pct * 0.5)
        
        equity_curve[i] = equity
    
    # Close any remaining position
    if position != 0 and len(df) > 0:
        exit_price = df['close'].iloc[-1]
        pnl = position * (exit_price - entry_price) * trade_size
        commission = abs(trade_size * exit_price * commission_pct)
        equity += pnl - commission
        equity_curve[-1] = equity
    
    df['equity'] = equity_curve
    
    # Trade statistics
    trades_df = pd.DataFrame(trades)
    
    return df, trades_df


def print_stats(trades_df: pd.DataFrame, equity_curve: np.ndarray, 
                initial_capital: float = 100000.0):
    """Print backtest statistics."""
    
    if len(trades_df) == 0:
        print("No trades generated.")
        return
    
    total_trades = len(trades_df)
    winners = trades_df[trades_df['pnl'] > 0]
    losers = trades_df[trades_df['pnl'] <= 0]
    
    win_rate = len(winners) / total_trades * 100
    avg_win = winners['pnl'].mean() if len(winners) > 0 else 0
    avg_loss = losers['pnl'].mean() if len(losers) > 0 else 0
    
    total_pnl = trades_df['pnl'].sum()
    total_return = (total_pnl / initial_capital) * 100
    
    # Sharpe (annualized, assuming ~252 trading days, ~10 trades/day for minute data)
    if len(trades_df) > 1:
        returns = trades_df['return_pct'].values / 100
        sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
    else:
        sharpe = 0
    
    # Max drawdown
    eq = equity_curve[~np.isnan(equity_curve)]
    if len(eq) > 0:
        running_max = np.maximum.accumulate(eq)
        drawdowns = (eq - running_max) / running_max
        max_dd = np.min(drawdowns) * 100
    else:
        max_dd = 0
    
    # Profit factor
    gross_profit = winners['pnl'].sum() if len(winners) > 0 else 0
    gross_loss = abs(losers['pnl'].sum()) if len(losers) > 0 else 1e-10
    profit_factor = gross_profit / gross_loss
    
    print("=" * 60)
    print("BACKTEST RESULTS: VG-TDA Regime-Aware Strategy")
    print("=" * 60)
    print(f"Total Trades:      {total_trades}")
    print(f"Win Rate:          {win_rate:.1f}%")
    print(f"Avg Winner:        ${avg_win:,.2f}")
    print(f"Avg Loser:         ${avg_loss:,.2f}")
    print(f"Profit Factor:     {profit_factor:.2f}")
    print(f"Total P&L:         ${total_pnl:,.2f}")
    print(f"Total Return:      {total_return:.2f}%")
    print(f"Sharpe Ratio:      {sharpe:.2f}")
    print(f"Max Drawdown:      {max_dd:.2f}%")
    print("=" * 60)
    
    # Regime breakdown
    print("\nTrade counts by direction:")
    for direction in ['LONG', 'SHORT']:
        subset = trades_df[trades_df['direction'] == direction]
        if len(subset) > 0:
            wr = len(subset[subset['pnl'] > 0]) / len(subset) * 100
            print(f"  {direction}: {len(subset)} trades, WR={wr:.1f}%, "
                  f"Avg PnL=${subset['pnl'].mean():,.2f}")


# ============================================================================
# PART 5: SYNTHETIC DATA GENERATOR (for testing without real data)
# ============================================================================

def generate_synthetic_ohlcv(n_bars: int = 5000,
                              timeframe: str = '1min',
                              seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data with regime changes.
    Includes trending, mean-reverting, and volatile regimes.
    """
    np.random.seed(seed)
    
    # Create regime sequence
    regime_length = n_bars // 5
    regimes = []
    for _ in range(5):
        regime_type = np.random.choice(['trend_up', 'trend_down', 'mean_revert', 'volatile'])
        regimes.extend([regime_type] * regime_length)
    regimes = regimes[:n_bars]
    
    # Generate returns per regime
    returns = np.zeros(n_bars)
    for i in range(n_bars):
        r = regimes[i]
        if r == 'trend_up':
            returns[i] = np.random.normal(0.0003, 0.002)
        elif r == 'trend_down':
            returns[i] = np.random.normal(-0.0003, 0.002)
        elif r == 'mean_revert':
            returns[i] = np.random.normal(0, 0.001)
            if i > 0:
                returns[i] -= 0.3 * returns[i-1]  # Mean reversion
        elif r == 'volatile':
            returns[i] = np.random.normal(0, 0.005)
    
    # Build price series
    price = 100.0
    prices = [price]
    for r in returns[1:]:
        price *= (1 + r)
        prices.append(price)
    prices = np.array(prices)
    
    # Generate OHLCV from close prices
    close = prices
    noise = np.abs(np.random.normal(0, 0.001, n_bars))
    high = close * (1 + noise)
    low = close * (1 - noise)
    open_price = np.roll(close, 1) * (1 + np.random.normal(0, 0.0005, n_bars))
    open_price[0] = close[0]
    volume = np.random.lognormal(10, 1, n_bars).astype(int)
    
    # Timestamps
    if timeframe == '1min':
        timestamps = pd.date_range('2025-01-02 09:30', periods=n_bars, freq='1min')
    else:
        timestamps = pd.date_range('2025-01-02', periods=n_bars, freq='1h')
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
    })
    
    # Store true regime for analysis
    df['true_regime'] = regimes
    
    return df


# ============================================================================
# PART 6: TICK DATA VARIANT (for reference - requires tick data)
# ============================================================================

def hawkes_ofi_proxy_from_ohlcv(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Approximate Order Flow Imbalance from OHLCV (no tick data needed).
    
    Uses the "close location value" (CLV) approach:
    CLV = ((close - low) - (high - close)) / (high - low)
    
    CLV > 0 → buying pressure (close near high)
    CLV < 0 → selling pressure (close near low)
    
    Then apply exponential decay (Hawkes-like self-excitation proxy):
    OFI_proxy[t] = alpha * OFI_proxy[t-1] + CLV[t] * volume[t]
    
    This is NOT real Hawkes process (that requires tick data) but captures
    similar dynamics: volume-weighted buying/selling pressure with memory.
    """
    hlr = df['high'] - df['low']
    hlr = hlr.replace(0, np.nan)  # Avoid division by zero
    
    clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / hlr
    clv = clv.fillna(0)
    
    # Volume-weighted CLV
    vw_clv = clv * df['volume']
    
    # Hawkes-like exponential decay accumulation
    alpha = 0.7  # Decay factor (higher = more memory)
    ofi_proxy = np.zeros(len(df))
    ofi_proxy[0] = vw_clv.iloc[0]
    
    for i in range(1, len(df)):
        ofi_proxy[i] = alpha * ofi_proxy[i-1] + vw_clv.iloc[i]
    
    # Normalize to rolling z-score
    ofi_series = pd.Series(ofi_proxy, index=df.index)
    ofi_zscore = (ofi_series - ofi_series.rolling(window).mean()) / (
        ofi_series.rolling(window).std() + 1e-10)
    
    return ofi_zscore


# ============================================================================
# PART 7: ENHANCED STRATEGY (VG + TDA + OFI Proxy)
# ============================================================================

def generate_enhanced_signals(df: pd.DataFrame,
                               vg_window: int = 30,
                               tda_window: int = 30,
                               ofi_window: int = 20,
                               sma_period: int = 20,
                               vg_high_threshold: float = 0.45,
                               vg_low_threshold: float = 0.15,
                               tda_unstable_z: float = 1.5,
                               ofi_threshold: float = 1.0,
                               ) -> pd.DataFrame:
    """
    Enhanced version: VG + TDA + Hawkes OFI Proxy.
    
    Entry logic:
    1. TDA regime filter (avoid unstable)
    2. VG degree extreme → mean reversion signal, confirmed by OFI
    3. VG degree low → momentum signal, confirmed by OFI direction
    """
    df = df.copy()
    close = df['close'].values
    
    # Features
    df['sma'] = df['close'].rolling(sma_period).mean()
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # VG
    print(f"Computing VG features (window={vg_window})...")
    vg_feats = compute_vg_features(close, window=vg_window)
    for k, v in vg_feats.items():
        df[k] = v
    
    # TDA
    print(f"Computing TDA features (window={tda_window})...")
    tda_feats = compute_tda_features(
        df['log_return'].values,
        window=tda_window,
        embed_dim=4,
        embed_delay=1
    )
    for k, v in tda_feats.items():
        df[k] = v
    
    # OFI proxy
    print("Computing OFI proxy...")
    df['ofi_zscore'] = hawkes_ofi_proxy_from_ohlcv(df, window=ofi_window)
    
    # Momentum
    df['momentum_5'] = df['close'].pct_change(5)
    
    # Regime
    df['regime'] = 'normal'
    df.loc[df['tda_l1_zscore'] > tda_unstable_z, 'regime'] = 'unstable'
    df.loc[df['tda_l1_zscore'] < -0.5, 'regime'] = 'stable'
    
    df['regime_size'] = 0.5
    df.loc[df['regime'] == 'unstable', 'regime_size'] = 0.0
    df.loc[df['regime'] == 'stable', 'regime_size'] = 1.0
    
    # Signals
    df['raw_signal'] = 0
    
    for i in range(len(df)):
        vg_ratio = df['vg_last_degree_ratio'].iloc[i]
        close_val = df['close'].iloc[i]
        sma_val = df['sma'].iloc[i]
        momentum = df['momentum_5'].iloc[i]
        ofi = df['ofi_zscore'].iloc[i]
        
        if any(pd.isna(x) for x in [vg_ratio, sma_val, momentum, ofi]):
            continue
        
        # High VG + OFI confirmation → mean reversion
        if vg_ratio > vg_high_threshold:
            if close_val > sma_val and ofi > ofi_threshold:
                # Overbought extreme with buying exhaustion
                df.iloc[i, df.columns.get_loc('raw_signal')] = -1
            elif close_val < sma_val and ofi < -ofi_threshold:
                # Oversold extreme with selling exhaustion
                df.iloc[i, df.columns.get_loc('raw_signal')] = 1
        
        # Low VG + OFI alignment → momentum
        elif vg_ratio < vg_low_threshold:
            if momentum > 0 and ofi > 0.5:
                df.iloc[i, df.columns.get_loc('raw_signal')] = 1
            elif momentum < 0 and ofi < -0.5:
                df.iloc[i, df.columns.get_loc('raw_signal')] = -1
    
    df['signal'] = df['raw_signal'] * df['regime_size']
    df['position'] = df['signal'].shift(1)  # NEXT bar entry
    
    return df


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run the strategy on synthetic data."""
    
    print("=" * 60)
    print("VG-TDA Regime-Aware Trading Strategy")
    print("Network Science + Topological Data Analysis")
    print("=" * 60)
    
    # Generate synthetic data
    print("\n[1] Generating synthetic OHLCV data (5000 bars, 1-hour)...")
    df = generate_synthetic_ohlcv(n_bars=2000, timeframe='1h', seed=42)
    print(f"    Data range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
    print(f"    Price range: {df['close'].min():.2f} to {df['close'].max():.2f}")
    
    # Generate signals (basic version for speed)
    print("\n[2] Generating signals...")
    # Use smaller windows for faster computation
    df = generate_signals(
        df,
        vg_window=30,    # Smaller for speed
        tda_window=30,
        sma_period=20,
        vg_high_threshold=0.45,
        vg_low_threshold=0.15,
    )
    
    # Run backtest
    print("\n[3] Running backtest...")
    df, trades = backtest(
        df,
        initial_capital=100000.0,
        position_size_pct=0.10,
        commission_pct=0.001,
        holding_period=5,
    )
    
    # Print results
    print("\n[4] Results:")
    print_stats(trades, df['equity'].values)
    
    # Show regime distribution
    if 'regime' in df.columns:
        print("\nRegime distribution:")
        counts = df['regime'].value_counts()
        for regime, count in counts.items():
            pct = count / len(df) * 100
            print(f"  {regime}: {count} bars ({pct:.1f}%)")
    
    # Show sample signals
    print("\nSample signals (last 20 with non-zero):")
    signals = df[df['raw_signal'] != 0].tail(20)
    if len(signals) > 0:
        print(signals[['timestamp', 'close', 'vg_last_degree_ratio', 
                       'tda_l1_zscore', 'regime', 'raw_signal']].to_string())
    
    # Show trade log
    if len(trades) > 0:
        print(f"\nFirst 10 trades:")
        print(trades.head(10).to_string())
    
    return df, trades


if __name__ == '__main__':
    main()
