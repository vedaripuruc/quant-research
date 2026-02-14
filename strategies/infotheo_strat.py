"""
Complexity-Gated Regime Momentum (CGRM) Strategy
=================================================
Uses information-theoretic complexity measures (Lempel-Ziv, Shannon block entropy,
permutation entropy) to detect predictability regimes in intraday data, then trades
momentum during low-complexity (structured) regimes and goes flat during high-complexity
(random) regimes.

Based on research from:
- Shternshis & Marmi (2024) — entropy-based predictability at ultra-high frequency
- Ponta & Murialdo (2022) — Shannon entropy of HF financial time series
- Brandouy, Delahaye & Ma (2015) — algorithmic complexity of stock markets

Author: Curupira (AI quant research agent)
Date: 2026-02-06
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from collections import Counter
from itertools import permutations
from math import factorial, log2


# =============================================================================
# 1. INFORMATION-THEORETIC COMPLEXITY MEASURES
# =============================================================================

def lempel_ziv_complexity(s: str) -> int:
    """
    Compute Lempel-Ziv complexity (LZ76) of a symbolic string.
    Uses set-based dictionary for O(n log n) average performance.
    Counts the number of distinct substrings encountered in a left-to-right scan.
    
    Reference: Lempel & Ziv, "On the Complexity of Finite Sequences" (1976)
    """
    n = len(s)
    if n == 0:
        return 0
    
    dictionary = set()
    complexity = 0
    start = 0
    
    while start < n:
        end = start + 1
        while end <= n:
            substr = s[start:end]
            if substr not in dictionary:
                dictionary.add(substr)
                complexity += 1
                start = end
                break
            end += 1
        else:
            # Reached the end without finding a new substring
            complexity += 1
            break
    
    return complexity


def normalized_lz_complexity(s: str) -> float:
    """
    Normalized LZ complexity: C(s) / C_max
    where C_max is estimated from a random shuffled version of the same string.
    Falls back to n / log2(alphabet_size * n) bound.
    Returns value in [0, 1], where 1.0 ≈ fully random.
    """
    n = len(s)
    if n <= 1:
        return 0.0
    
    c = lempel_ziv_complexity(s)
    
    # Theoretical upper bound for iid random sequence with given alphabet
    alphabet_size = max(len(set(s)), 2)
    # Lempel-Ziv (1976) theorem: for random iid, C ~ n / log_alphabet(n)
    log_base = log2(n) / log2(alphabet_size) if alphabet_size > 1 else log2(n)
    normalizer = n / log_base if log_base > 0 else n
    
    return min(c / normalizer, 1.0)  # clamp to [0, 1]


def shannon_block_entropy(s: str, k: int = 3) -> float:
    """
    Shannon entropy of k-grams (blocks of length k) in symbolic string s.
    Normalized by log2(num_possible_blocks) to return value in [0, 1].
    
    Low value → some patterns dominate → predictable
    High value (near 1) → all patterns equally likely → random
    """
    n = len(s)
    if n < k:
        return 1.0  # not enough data, assume random
    
    # Count k-gram frequencies
    blocks = [s[i:i + k] for i in range(n - k + 1)]
    counts = Counter(blocks)
    total = sum(counts.values())
    
    if total == 0:
        return 1.0
    
    # Compute entropy
    probs = np.array([c / total for c in counts.values()])
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log2(probs))
    
    # Normalize by max possible entropy
    # Number of unique symbols in alphabet
    alphabet_size = len(set(s))
    if alphabet_size <= 1:
        return 0.0  # constant sequence = zero randomness
    max_entropy = k * log2(alphabet_size)
    
    if max_entropy <= 0:
        return 0.0
    
    return max(0.0, min(1.0, entropy / max_entropy))


def permutation_entropy(x: np.ndarray, m: int = 3, delay: int = 1) -> float:
    """
    Permutation entropy of a numerical time series.
    Based on Bandt & Pompe (2002).
    
    Parameters:
        x: numerical time series (e.g., raw returns, not discretized)
        m: embedding dimension (order of permutation patterns)
        delay: time delay between elements
    
    Returns:
        Normalized permutation entropy in [0, 1].
        Low → ordered patterns dominate → predictable
        High → all ordinal patterns equally likely → random
    """
    n = len(x)
    if n < m * delay:
        return 1.0
    
    # Extract ordinal patterns
    patterns = []
    for i in range(n - (m - 1) * delay):
        # Extract m values with given delay
        window = [x[i + j * delay] for j in range(m)]
        # Get the rank order (permutation pattern)
        pattern = tuple(np.argsort(window))
        patterns.append(pattern)
    
    if len(patterns) == 0:
        return 1.0
    
    # Count pattern frequencies
    counts = Counter(patterns)
    total = len(patterns)
    probs = np.array([c / total for c in counts.values()])
    probs = probs[probs > 0]
    
    # Shannon entropy of permutation distribution
    entropy = -np.sum(probs * np.log2(probs))
    
    # Normalize by max entropy = log2(m!)
    max_entropy = log2(factorial(m))
    
    if max_entropy == 0:
        return 0.0  # degenerate case
    
    return max(0.0, min(1.0, entropy / max_entropy))


# =============================================================================
# 2. RETURN DISCRETIZATION
# =============================================================================

def discretize_returns(returns: pd.Series, threshold: Optional[float] = None) -> str:
    """
    Convert continuous returns to symbolic sequence {0, 1, 2}.
    0 = negative return, 1 = ~zero return, 2 = positive return.
    
    Threshold: if None, uses median absolute return as threshold for "zero".
    This avoids counting microstructure noise as directional.
    
    Reference: Shternshis & Marmi (2024) use binary discretization.
    We use ternary to capture zero-return staleness (Ponta & Murialdo 2022).
    """
    if threshold is None:
        threshold = returns.abs().median() * 0.5
        if threshold == 0:
            threshold = 1e-10  # fallback
    
    symbols = []
    for r in returns:
        if r < -threshold:
            symbols.append('0')
        elif r > threshold:
            symbols.append('2')
        else:
            symbols.append('1')
    
    return ''.join(symbols)


def discretize_returns_binary(returns: pd.Series) -> str:
    """
    Binary discretization: 0 = non-positive, 1 = positive.
    Simpler variant per Shternshis & Marmi (2024).
    """
    return ''.join(['1' if r > 0 else '0' for r in returns])


# =============================================================================
# 3. COMPOSITE PREDICTABILITY SCORE
# =============================================================================

def compute_predictability_score(
    returns: pd.Series,
    raw_returns: np.ndarray,
    lz_weight: float = 0.4,
    shannon_weight: float = 0.3,
    perm_weight: float = 0.3,
    block_k: int = 3,
    perm_m: int = 3,
    perm_delay: int = 1
) -> float:
    """
    Composite predictability score combining three complexity metrics.
    
    Returns a score in [0, 1] where:
        HIGH score → LOW complexity → market is PREDICTABLE → trade
        LOW score → HIGH complexity → market is RANDOM → stay flat
    """
    # Discretize for LZ and Shannon
    sym_ternary = discretize_returns(returns)
    sym_binary = discretize_returns_binary(returns)
    
    # 1. Normalized LZ complexity (use binary for cleaner LZ)
    lz = normalized_lz_complexity(sym_binary)
    
    # 2. Shannon block entropy (use ternary for richer patterns)
    sh = shannon_block_entropy(sym_ternary, k=block_k)
    
    # 3. Permutation entropy (use raw returns, not discretized)
    pe = permutation_entropy(raw_returns, m=perm_m, delay=perm_delay)
    
    # Invert: low complexity → high predictability
    pred_lz = max(0, 1.0 - lz)
    pred_sh = max(0, 1.0 - sh)
    pred_pe = max(0, 1.0 - pe)
    
    # Weighted combination
    score = (lz_weight * pred_lz + 
             shannon_weight * pred_sh + 
             perm_weight * pred_pe)
    
    return score


# =============================================================================
# 4. SIGNAL GENERATION
# =============================================================================

def compute_direction_bias(returns: pd.Series, lookback: int = 20) -> float:
    """
    Compute the directional bias of recent returns.
    Returns value in [-1, 1]:
        +1 = all returns positive (strong uptrend)
        -1 = all returns negative (strong downtrend)
         0 = balanced
    
    Uses sign-count ratio, more robust than simple mean for discrete signals.
    """
    recent = returns.iloc[-lookback:]
    if len(recent) == 0:
        return 0.0
    
    positive = (recent > 0).sum()
    negative = (recent < 0).sum()
    total = positive + negative
    
    if total == 0:
        return 0.0
    
    return (positive - negative) / total


def generate_signals_ohlcv(
    df: pd.DataFrame,
    window: int = 120,
    pred_threshold: float = 0.03,
    pred_drop_trigger: float = -0.10,
    direction_threshold: float = 0.3,
    direction_lookback: int = 20,
    max_hold_bars: int = 30,
    block_k: int = 3,
    perm_m: int = 3
) -> pd.DataFrame:
    """
    Generate entry/exit signals from OHLCV data.
    
    Parameters:
        df: DataFrame with columns [timestamp, open, high, low, close, volume]
        window: rolling window size for complexity computation
        pred_threshold: minimum predictability score to consider entry
        pred_drop_trigger: if predictability score drops by this much (rate of change),
                          trigger a regime-change entry signal
        direction_threshold: minimum directional bias magnitude to enter
        direction_lookback: bars to look back for direction
        max_hold_bars: maximum bars to hold a position
        block_k: k-gram length for Shannon block entropy
        perm_m: permutation order for permutation entropy
    
    Returns:
        df with added columns: returns, pred_score, direction, signal, position
    """
    df = df.copy()
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Compute log returns
    df['returns'] = np.log(df['close'] / df['close'].shift(1))
    df['returns'] = df['returns'].fillna(0)
    
    # Rolling predictability score
    pred_scores = []
    for i in range(len(df)):
        if i < window:
            pred_scores.append(np.nan)
            continue
        
        window_returns = df['returns'].iloc[i - window:i]
        raw_returns = window_returns.values
        
        score = compute_predictability_score(
            window_returns, raw_returns,
            block_k=block_k, perm_m=perm_m
        )
        pred_scores.append(score)
    
    df['pred_score'] = pred_scores
    
    # Rate of change of predictability (detect regime transitions)
    df['pred_roc'] = df['pred_score'].pct_change(periods=10)
    
    # Directional bias
    directions = []
    for i in range(len(df)):
        if i < window:
            directions.append(0.0)
            continue
        
        recent = df['returns'].iloc[max(0, i - direction_lookback):i]
        directions.append(compute_direction_bias(recent, len(recent)))
    
    df['direction'] = directions
    
    # Adaptive threshold: use rolling percentile of pred_score
    # Trade when predictability is in the top quantile of its own recent history
    pred_series = df['pred_score'].copy()
    df['pred_threshold_adaptive'] = pred_series.rolling(
        window=window, min_periods=window // 2
    ).quantile(0.75)  # top 25% of recent scores
    
    # Generate raw signals
    # Signal: +1 = long, -1 = short, 0 = flat
    signals = np.zeros(len(df))
    
    for i in range(window, len(df)):
        ps = df['pred_score'].iloc[i]
        direction = df['direction'].iloc[i]
        adaptive_thresh = df['pred_threshold_adaptive'].iloc[i]
        
        if pd.isna(ps) or pd.isna(adaptive_thresh):
            continue
        
        # Use the higher of fixed or adaptive threshold
        effective_threshold = max(pred_threshold, adaptive_thresh)
        
        # Condition 1: High predictability + directional bias
        if ps > effective_threshold and abs(direction) > direction_threshold:
            signals[i] = np.sign(direction)
        
        # Condition 2: Sharp predictability spike (regime becoming ordered)
        # This catches the ONSET of a low-entropy regime
        pred_roc = df['pred_roc'].iloc[i]
        if not pd.isna(pred_roc) and pred_roc > 0.20 and abs(direction) > 0.2:
            signals[i] = np.sign(direction)
    
    df['raw_signal'] = signals
    
    # Convert raw signals to positions with holding period logic
    # CRITICAL: entry at NEXT bar's open (no look-ahead bias)
    positions = np.zeros(len(df))
    entry_bar = -999
    
    for i in range(1, len(df)):
        # Check if we have an active position
        if positions[i - 1] != 0:
            bars_held = i - entry_bar
            
            # Exit conditions:
            # 1. Max hold period reached
            # 2. Predictability drops below threshold (regime ending)
            # 3. Direction flips
            exit_signal = False
            
            if bars_held >= max_hold_bars:
                exit_signal = True
            elif not pd.isna(df['pred_score'].iloc[i]) and df['pred_score'].iloc[i] < pred_threshold * 0.5:
                exit_signal = True
            elif np.sign(df['direction'].iloc[i]) != np.sign(positions[i - 1]) and abs(df['direction'].iloc[i]) > direction_threshold:
                exit_signal = True
            
            if exit_signal:
                positions[i] = 0
            else:
                positions[i] = positions[i - 1]
        else:
            # No position: check for entry at this bar (signal from previous bar)
            if df['raw_signal'].iloc[i - 1] != 0 and positions[i - 1] == 0:
                positions[i] = df['raw_signal'].iloc[i - 1]  # enter at bar i open
                entry_bar = i
    
    df['position'] = positions
    
    return df


# =============================================================================
# 5. TICK DATA VARIANT
# =============================================================================

def generate_signals_tick(
    df: pd.DataFrame,
    window: int = 500,
    pred_threshold: float = 0.28,
    direction_threshold: float = 0.25,
    direction_lookback: int = 50,
    max_hold_ticks: int = 200,
    block_k: int = 4,
    perm_m: int = 4
) -> pd.DataFrame:
    """
    Generate entry/exit signals from tick data.
    
    Parameters:
        df: DataFrame with columns [timestamp, bid, ask, volume]
        window: rolling window size (in ticks)
        Other params same as OHLCV variant but tuned for tick frequency.
    
    Returns:
        df with added signal/position columns
    """
    df = df.copy()
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Compute mid-price and returns
    df['mid'] = (df['bid'] + df['ask']) / 2.0
    df['spread'] = df['ask'] - df['bid']
    df['returns'] = df['mid'].pct_change().fillna(0)
    
    # Filter: only compute on ticks where price actually moved
    # (Ponta & Murialdo 2022: zero returns create false efficiency signal)
    df['price_changed'] = df['returns'].abs() > 0
    
    # Rolling predictability score
    pred_scores = []
    for i in range(len(df)):
        if i < window:
            pred_scores.append(np.nan)
            continue
        
        # Use only non-zero returns within window (filter staleness)
        window_slice = df.iloc[i - window:i]
        active_returns = window_slice.loc[window_slice['price_changed'], 'returns']
        
        if len(active_returns) < 30:  # not enough active ticks
            pred_scores.append(np.nan)
            continue
        
        raw_returns = active_returns.values
        
        score = compute_predictability_score(
            active_returns, raw_returns,
            block_k=block_k, perm_m=perm_m
        )
        pred_scores.append(score)
    
    df['pred_score'] = pred_scores
    df['pred_roc'] = df['pred_score'].pct_change(periods=20)
    
    # Direction bias
    directions = []
    for i in range(len(df)):
        if i < window:
            directions.append(0.0)
            continue
        
        recent = df['returns'].iloc[max(0, i - direction_lookback):i]
        directions.append(compute_direction_bias(recent, len(recent)))
    
    df['direction'] = directions
    
    # Generate signals (same logic as OHLCV)
    signals = np.zeros(len(df))
    
    for i in range(window, len(df)):
        ps = df['pred_score'].iloc[i]
        direction = df['direction'].iloc[i]
        
        if pd.isna(ps):
            continue
        
        if ps > pred_threshold and abs(direction) > direction_threshold:
            signals[i] = np.sign(direction)
        
        pred_roc = df['pred_roc'].iloc[i]
        if not pd.isna(pred_roc) and pred_roc > 0.15 and abs(direction) > 0.15:
            signals[i] = np.sign(direction)
    
    df['raw_signal'] = signals
    
    # Position management
    positions = np.zeros(len(df))
    entry_bar = -999
    
    for i in range(1, len(df)):
        if positions[i - 1] != 0:
            bars_held = i - entry_bar
            exit_signal = False
            
            if bars_held >= max_hold_ticks:
                exit_signal = True
            elif not pd.isna(df['pred_score'].iloc[i]) and df['pred_score'].iloc[i] < pred_threshold * 0.5:
                exit_signal = True
            
            if exit_signal:
                positions[i] = 0
            else:
                positions[i] = positions[i - 1]
        else:
            if df['raw_signal'].iloc[i - 1] != 0 and positions[i - 1] == 0:
                positions[i] = df['raw_signal'].iloc[i - 1]
                entry_bar = i
    
    df['position'] = positions
    
    return df


# =============================================================================
# 6. BACKTEST ENGINE
# =============================================================================

def backtest(
    df: pd.DataFrame,
    initial_capital: float = 100_000,
    position_size_pct: float = 0.10,
    commission_per_trade: float = 1.0,
    slippage_bps: float = 1.0,
    price_col: str = 'open',
    data_type: str = 'ohlcv',
    stop_loss_pct: Optional[float] = None,
    take_profit_pct: Optional[float] = None
) -> dict:
    """
    Bar-by-bar backtest loop with SL/TP support.
    
    CRITICAL: Entries happen at the OPEN of the bar when position changes.
    This avoids look-ahead bias since position[i] was determined from data up to bar i-1.
    
    SL/TP are checked against bar HIGH/LOW (not close) each bar while in a trade.
    When both SL and TP could be hit on the same bar, SL is assumed to hit first
    (conservative assumption).
    
    Exit priority per bar:
        1. Stop loss  (checked against high/low)
        2. Take profit (checked against high/low)
        3. Signal exit  (position column changed to 0 or flipped)
        4. Max hold / regime exit (already encoded in position column)
    
    Parameters:
        df: DataFrame with 'position' column and price data
        initial_capital: starting equity
        position_size_pct: fraction of equity per trade
        commission_per_trade: fixed commission per entry/exit
        slippage_bps: slippage in basis points per trade
        price_col: column to use for execution price ('open' for OHLCV, 'mid' for tick)
        data_type: 'ohlcv' or 'tick'
        stop_loss_pct: stop loss as fraction (e.g. 0.002 = 0.2%). None = disabled.
        take_profit_pct: take profit as fraction (e.g. 0.004 = 0.4%). None = disabled.
    
    Returns:
        Dictionary with performance metrics
    """
    if data_type == 'tick':
        price_col = 'mid'
    
    df = df.copy()
    
    # Determine if we have OHLC data for intra-bar SL/TP checks
    has_ohlc = all(c in df.columns for c in ['high', 'low'])
    
    equity = initial_capital
    equity_curve = [initial_capital]
    trades = []
    realized_pnl_sum = 0.0  # running sum for faster MTM
    
    in_trade = False
    entry_price = 0.0
    entry_dir = 0
    entry_idx = 0
    shares = 0.0
    
    for i in range(1, len(df)):
        prev_pos = df['position'].iloc[i - 1]
        curr_pos = df['position'].iloc[i]
        exec_price = df[price_col].iloc[i]
        
        if pd.isna(exec_price) or exec_price <= 0:
            equity_curve.append(equity if not in_trade else
                                initial_capital + realized_pnl_sum)
            continue
        
        exit_reason = None
        forced_exit_price = None
        
        # --- CHECK SL/TP BEFORE processing signal exits ---
        if in_trade and has_ohlc and (stop_loss_pct is not None or take_profit_pct is not None):
            bar_high = df['high'].iloc[i]
            bar_low = df['low'].iloc[i]
            
            if entry_dir > 0:  # LONG position
                sl_price = entry_price * (1.0 - stop_loss_pct) if stop_loss_pct is not None else None
                tp_price = entry_price * (1.0 + take_profit_pct) if take_profit_pct is not None else None
                sl_hit = sl_price is not None and bar_low <= sl_price
                tp_hit = tp_price is not None and bar_high >= tp_price
            else:  # SHORT position
                sl_price = entry_price * (1.0 + stop_loss_pct) if stop_loss_pct is not None else None
                tp_price = entry_price * (1.0 - take_profit_pct) if take_profit_pct is not None else None
                sl_hit = sl_price is not None and bar_high >= sl_price
                tp_hit = tp_price is not None and bar_low <= tp_price
            
            # Conservative: if both could hit on same bar, SL wins
            if sl_hit:
                exit_reason = 'stop_loss'
                forced_exit_price = sl_price
            elif tp_hit:
                exit_reason = 'take_profit'
                forced_exit_price = tp_price
        
        # --- PROCESS EXITS ---
        if in_trade and (exit_reason is not None or curr_pos != prev_pos):
            if exit_reason is not None:
                # SL or TP hit — use the exact SL/TP level as exit price
                raw_exit = forced_exit_price
            else:
                # Signal-based exit — execute at bar open with slippage
                raw_exit = exec_price
                exit_reason = 'signal'
            
            # Apply slippage (directional: adverse to the trade)
            slippage = raw_exit * slippage_bps / 10000
            if entry_dir > 0:
                exit_px = raw_exit - slippage
            else:
                exit_px = raw_exit + slippage
            
            pnl = entry_dir * (exit_px - entry_price) * shares
            pnl -= commission_per_trade
            equity += pnl
            realized_pnl_sum += pnl
            
            trades.append({
                'entry_idx': entry_idx,
                'exit_idx': i,
                'entry_price': entry_price,
                'exit_price': exit_px,
                'direction': entry_dir,
                'pnl': pnl,
                'bars_held': i - entry_idx,
                'exit_reason': exit_reason,
            })
            
            in_trade = False
            
            # If SL/TP forced the exit, override the position column
            # so we don't re-enter on the same bar from the original signal
            if forced_exit_price is not None:
                curr_pos = 0  # flatten — don't open new position on SL/TP bar
        
        # --- PROCESS ENTRIES ---
        if not in_trade and curr_pos != 0 and curr_pos != prev_pos:
            slippage = exec_price * slippage_bps / 10000
            if curr_pos > 0:
                entry_price = exec_price + slippage
            else:
                entry_price = exec_price - slippage
            
            trade_capital = equity * position_size_pct
            shares = trade_capital / entry_price if entry_price > 0 else 0
            entry_dir = int(curr_pos)
            entry_idx = i
            equity -= commission_per_trade
            in_trade = True
        
        # --- EQUITY CURVE (mark-to-market) ---
        if in_trade:
            mtm = entry_dir * (exec_price - entry_price) * shares
            equity_curve.append(initial_capital + realized_pnl_sum + mtm)
        else:
            equity_curve.append(equity)
    
    # Close any remaining position at end of data
    if in_trade and len(df) > 0:
        final_price = df[price_col].iloc[-1]
        slippage = final_price * slippage_bps / 10000
        if entry_dir > 0:
            exit_px = final_price - slippage
        else:
            exit_px = final_price + slippage
        pnl = entry_dir * (exit_px - entry_price) * shares - commission_per_trade
        equity += pnl
        realized_pnl_sum += pnl
        trades.append({
            'entry_idx': entry_idx,
            'exit_idx': len(df) - 1,
            'entry_price': entry_price,
            'exit_price': exit_px,
            'direction': entry_dir,
            'pnl': pnl,
            'bars_held': len(df) - 1 - entry_idx,
            'exit_reason': 'end_of_data',
        })
    
    # --- COMPUTE METRICS ---
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    equity_series = pd.Series(equity_curve)
    
    total_pnl = sum(t['pnl'] for t in trades) if trades else 0
    num_trades = len(trades)
    
    if num_trades > 0:
        wins = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] <= 0]
        win_rate = len(wins) / num_trades
        avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
        avg_loss = np.mean([t['pnl'] for t in losses]) if losses else 0
        gross_wins = sum(t['pnl'] for t in wins)
        gross_losses = sum(t['pnl'] for t in losses)
        profit_factor = abs(gross_wins / gross_losses) if gross_losses != 0 else float('inf')
        avg_bars_held = np.mean([t['bars_held'] for t in trades])
        
        # Exit reason breakdown
        exit_reasons = {}
        for t in trades:
            r = t.get('exit_reason', 'unknown')
            exit_reasons[r] = exit_reasons.get(r, 0) + 1
    else:
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0
        avg_bars_held = 0
        exit_reasons = {}
    
    # Equity curve stats
    returns_series = equity_series.pct_change().dropna()
    if len(returns_series) > 0 and returns_series.std() > 0:
        sharpe = returns_series.mean() / returns_series.std() * np.sqrt(252 * 390)  # annualized for 1-min
    else:
        sharpe = 0
    
    # Max drawdown
    peak = equity_series.expanding().max()
    drawdown = (equity_series - peak) / peak
    max_dd = drawdown.min()
    
    return {
        'total_pnl': total_pnl,
        'num_trades': num_trades,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'avg_bars_held': avg_bars_held,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'final_equity': equity,
        'trades': trades_df,
        'equity_curve': equity_series,
        'exit_reasons': exit_reasons,
    }


# =============================================================================
# 7. SYNTHETIC DATA GENERATOR (for testing)
# =============================================================================

def generate_synthetic_ohlcv(
    n_bars: int = 2000,
    freq: str = '1min',
    regime_length: int = 200,
    trending_drift: float = 0.0003,
    random_vol: float = 0.001,
    trending_vol: float = 0.0005,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data with alternating regimes:
    - Trending regime (low complexity): directional with lower vol
    - Random regime (high complexity): noise with higher vol
    
    This is for testing the strategy's ability to detect regimes.
    """
    np.random.seed(seed)
    
    timestamps = pd.date_range(
        start='2026-01-05 09:30:00',
        periods=n_bars,
        freq=freq
    )
    
    closes = [100.0]
    volumes = []
    
    for i in range(1, n_bars):
        regime_idx = (i // regime_length) % 2  # alternating
        
        if regime_idx == 0:
            # Trending regime: persistent direction + low vol
            direction = 1 if (i // regime_length) % 4 < 2 else -1
            drift = direction * trending_drift
            vol = trending_vol
            # Add autocorrelation (persistence)
            noise = np.random.normal(drift, vol)
            if i > 1 and np.random.random() < 0.6:
                # 60% chance of same direction as previous
                noise = abs(noise) * np.sign(closes[-1] - closes[-2]) if len(closes) > 1 else noise
        else:
            # Random regime: no drift, higher vol
            noise = np.random.normal(0, random_vol)
        
        new_price = closes[-1] * (1 + noise)
        closes.append(max(new_price, 0.01))  # floor at penny
        volumes.append(np.random.randint(1000, 50000) * (2 if regime_idx == 0 else 1))
    
    volumes.append(np.random.randint(1000, 50000))
    
    closes = np.array(closes)
    highs = closes * (1 + np.abs(np.random.normal(0, 0.0003, n_bars)))
    lows = closes * (1 - np.abs(np.random.normal(0, 0.0003, n_bars)))
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    })


def generate_synthetic_ticks(
    n_ticks: int = 5000,
    seed: int = 42
) -> pd.DataFrame:
    """Generate synthetic tick data with alternating regimes."""
    np.random.seed(seed)
    
    timestamps = pd.date_range(
        start='2026-01-05 09:30:00',
        periods=n_ticks,
        freq='200ms'
    )
    
    mid_prices = [1.1000]  # e.g., EUR/USD
    regime_length = 500
    
    for i in range(1, n_ticks):
        regime_idx = (i // regime_length) % 2
        
        if regime_idx == 0:
            # Trending: persistent microstructure
            drift = 0.00001 * (1 if (i // regime_length) % 4 < 2 else -1)
            vol = 0.00002
            noise = np.random.normal(drift, vol)
            if i > 1 and np.random.random() < 0.65:
                noise = abs(noise) * np.sign(mid_prices[-1] - mid_prices[-2])
        else:
            noise = np.random.normal(0, 0.00005)
        
        mid_prices.append(mid_prices[-1] + noise)
    
    mid_prices = np.array(mid_prices)
    half_spread = 0.00005 + np.abs(np.random.normal(0, 0.00002, n_ticks))
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'bid': mid_prices - half_spread,
        'ask': mid_prices + half_spread,
        'volume': np.random.randint(1, 20, n_ticks)
    })


# =============================================================================
# 8. MAIN: RUN DEMO
# =============================================================================

def print_results(results: dict, label: str):
    """Pretty-print backtest results."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Total PnL:       ${results['total_pnl']:>12,.2f}")
    print(f"  Final Equity:    ${results['final_equity']:>12,.2f}")
    print(f"  Num Trades:       {results['num_trades']:>12d}")
    print(f"  Win Rate:         {results['win_rate']:>12.1%}")
    print(f"  Avg Win:         ${results['avg_win']:>12,.2f}")
    print(f"  Avg Loss:        ${results['avg_loss']:>12,.2f}")
    print(f"  Profit Factor:    {results['profit_factor']:>12.2f}")
    print(f"  Avg Bars Held:    {results['avg_bars_held']:>12.1f}")
    print(f"  Sharpe Ratio:     {results['sharpe_ratio']:>12.2f}")
    print(f"  Max Drawdown:     {results['max_drawdown']:>12.2%}")
    print(f"{'='*60}\n")


def run_sltp_sweep(
    signals_df: pd.DataFrame,
    sl_values: list,
    tp_values: list,
    initial_capital: float = 100_000,
    position_size_pct: float = 0.10,
    commission_per_trade: float = 1.0,
    slippage_bps: float = 1.0,
) -> pd.DataFrame:
    """
    Run a parameter sweep over SL/TP combinations.
    
    Parameters:
        signals_df: DataFrame with signals already generated (has 'position' column)
        sl_values: list of stop loss fractions (e.g. [0.001, 0.002])
        tp_values: list of take profit fractions (e.g. [0.002, 0.004])
    
    Returns:
        DataFrame with one row per SL/TP combination and performance metrics.
    """
    rows = []
    total = len(sl_values) * len(tp_values)
    done = 0
    
    for sl in sl_values:
        for tp in tp_values:
            done += 1
            results = backtest(
                signals_df,
                initial_capital=initial_capital,
                position_size_pct=position_size_pct,
                commission_per_trade=commission_per_trade,
                slippage_bps=slippage_bps,
                stop_loss_pct=sl,
                take_profit_pct=tp,
            )
            
            exit_reasons = results.get('exit_reasons', {})
            
            rows.append({
                'SL%': f"{sl * 100:.1f}%",
                'TP%': f"{tp * 100:.1f}%",
                'sl_raw': sl,
                'tp_raw': tp,
                'num_trades': results['num_trades'],
                'win_rate': results['win_rate'],
                'total_pnl': results['total_pnl'],
                'profit_factor': results['profit_factor'],
                'max_drawdown': results['max_drawdown'],
                'sharpe_ratio': results['sharpe_ratio'],
                'avg_bars_held': results['avg_bars_held'],
                'exits_sl': exit_reasons.get('stop_loss', 0),
                'exits_tp': exit_reasons.get('take_profit', 0),
                'exits_signal': exit_reasons.get('signal', 0),
                'exits_eod': exit_reasons.get('end_of_data', 0),
            })
            
            print(f"  [{done:>2}/{total}] SL={sl*100:.1f}% TP={tp*100:.1f}% → "
                  f"trades={results['num_trades']:>3} "
                  f"wr={results['win_rate']:.1%} "
                  f"pnl=${results['total_pnl']:>8.2f} "
                  f"pf={results['profit_factor']:.2f} "
                  f"dd={results['max_drawdown']:.2%}")
    
    return pd.DataFrame(rows)


def save_sweep_results_md(sweep_df: pd.DataFrame, filepath: str):
    """Save SL/TP sweep results as a markdown file."""
    import datetime
    
    lines = [
        "# InfoTheo CGRM — SL/TP Parameter Sweep Results",
        "",
        f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Data:** Synthetic OHLCV (1-min, 2000 bars, alternating regimes)",
        f"**Capital:** $100,000 | **Position:** 10% equity | **Commission:** $1/trade | **Slippage:** 1 bps",
        "",
        "## Results Table",
        "",
    ]
    
    # Build markdown table
    headers = ['SL%', 'TP%', 'Trades', 'Win Rate', 'Total PnL', 'Profit Factor', 'Max DD',
               'Sharpe', 'Avg Bars', 'SL Exits', 'TP Exits', 'Sig Exits']
    lines.append('| ' + ' | '.join(headers) + ' |')
    lines.append('| ' + ' | '.join(['---'] * len(headers)) + ' |')
    
    for _, row in sweep_df.iterrows():
        vals = [
            row['SL%'],
            row['TP%'],
            str(int(row['num_trades'])),
            f"{row['win_rate']:.1%}",
            f"${row['total_pnl']:.2f}",
            f"{row['profit_factor']:.2f}" if row['profit_factor'] != float('inf') else "∞",
            f"{row['max_drawdown']:.2%}",
            f"{row['sharpe_ratio']:.2f}",
            f"{row['avg_bars_held']:.1f}",
            str(int(row['exits_sl'])),
            str(int(row['exits_tp'])),
            str(int(row['exits_signal'])),
        ]
        lines.append('| ' + ' | '.join(vals) + ' |')
    
    # Best configurations
    lines.append("")
    lines.append("## Best Configurations")
    lines.append("")
    
    if len(sweep_df) > 0 and sweep_df['num_trades'].max() > 0:
        valid = sweep_df[sweep_df['num_trades'] > 0].copy()
        
        if len(valid) > 0:
            best_pnl = valid.loc[valid['total_pnl'].idxmax()]
            lines.append(f"**Highest PnL:** SL={best_pnl['SL%']} TP={best_pnl['TP%']} → "
                        f"${best_pnl['total_pnl']:.2f} ({best_pnl['num_trades']:.0f} trades, "
                        f"{best_pnl['win_rate']:.1%} WR)")
            
            best_pf = valid[valid['profit_factor'] != float('inf')]
            if len(best_pf) > 0:
                best_pf_row = best_pf.loc[best_pf['profit_factor'].idxmax()]
                lines.append(f"**Best Profit Factor:** SL={best_pf_row['SL%']} TP={best_pf_row['TP%']} → "
                            f"PF={best_pf_row['profit_factor']:.2f} ({best_pf_row['num_trades']:.0f} trades)")
            
            best_wr = valid.loc[valid['win_rate'].idxmax()]
            lines.append(f"**Highest Win Rate:** SL={best_wr['SL%']} TP={best_wr['TP%']} → "
                        f"{best_wr['win_rate']:.1%} ({best_wr['num_trades']:.0f} trades)")
            
            best_dd = valid.loc[valid['max_drawdown'].idxmax()]  # least negative
            lines.append(f"**Smallest Drawdown:** SL={best_dd['SL%']} TP={best_dd['TP%']} → "
                        f"{best_dd['max_drawdown']:.2%}")
    
    lines.append("")
    lines.append("## Exit Reason Analysis")
    lines.append("")
    if len(sweep_df) > 0:
        total_exits = (sweep_df['exits_sl'].sum() + sweep_df['exits_tp'].sum() + 
                      sweep_df['exits_signal'].sum() + sweep_df['exits_eod'].sum())
        if total_exits > 0:
            lines.append(f"- **Stop Loss exits:** {sweep_df['exits_sl'].sum():.0f} "
                        f"({sweep_df['exits_sl'].sum()/total_exits:.1%})")
            lines.append(f"- **Take Profit exits:** {sweep_df['exits_tp'].sum():.0f} "
                        f"({sweep_df['exits_tp'].sum()/total_exits:.1%})")
            lines.append(f"- **Signal exits:** {sweep_df['exits_signal'].sum():.0f} "
                        f"({sweep_df['exits_signal'].sum()/total_exits:.1%})")
            lines.append(f"- **End-of-data exits:** {sweep_df['exits_eod'].sum():.0f} "
                        f"({sweep_df['exits_eod'].sum()/total_exits:.1%})")
    
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Entry: next bar open (no look-ahead bias)")
    lines.append("- SL/TP checked against bar HIGH/LOW (not close)")
    lines.append("- When both SL and TP could hit same bar, SL assumed first (conservative)")
    lines.append("- Synthetic data has alternating trending/random regimes (200 bars each)")
    lines.append("- Results on synthetic data — real data may differ significantly")
    lines.append("")
    
    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"\n  Results saved to: {filepath}")


if __name__ == '__main__':
    print("=" * 60)
    print("  COMPLEXITY-GATED REGIME MOMENTUM (CGRM) STRATEGY")
    print("  Information Theory + Intraday Trading")
    print("=" * 60)
    
    # --- OHLCV Demo ---
    print("\n[1] Generating synthetic 1-min OHLCV data (2000 bars)...")
    ohlcv_data = generate_synthetic_ohlcv(n_bars=2000)
    print(f"    Date range: {ohlcv_data['timestamp'].iloc[0]} to {ohlcv_data['timestamp'].iloc[-1]}")
    print(f"    Price range: {ohlcv_data['close'].min():.2f} - {ohlcv_data['close'].max():.2f}")
    
    print("\n[2] Computing complexity metrics & generating signals...")
    signals_df = generate_signals_ohlcv(
        ohlcv_data,
        window=120,
        pred_threshold=0.25,
        direction_threshold=0.3,
        max_hold_bars=30
    )
    
    n_signals = (signals_df['raw_signal'] != 0).sum()
    n_long = (signals_df['position'] > 0).sum()
    n_short = (signals_df['position'] < 0).sum()
    n_flat = (signals_df['position'] == 0).sum()
    
    print(f"    Raw signals generated: {n_signals}")
    print(f"    Bars long: {n_long} | short: {n_short} | flat: {n_flat}")
    
    # Show predictability score distribution
    valid_pred = signals_df['pred_score'].dropna()
    if len(valid_pred) > 0:
        print(f"    Pred score: min={valid_pred.min():.3f} median={valid_pred.median():.3f} max={valid_pred.max():.3f}")
    
    # --- Baseline (no SL/TP) ---
    print("\n[3] Running baseline backtest (NO SL/TP)...")
    results_baseline = backtest(
        signals_df,
        initial_capital=100_000,
        position_size_pct=0.10,
        commission_per_trade=1.0,
        slippage_bps=1.0
    )
    print_results(results_baseline, "BASELINE (no SL/TP)")
    
    # --- SL/TP Parameter Sweep ---
    print("\n[4] Running SL/TP parameter sweep (16 combinations)...")
    sl_values = [0.001, 0.002, 0.003, 0.005]   # 0.1%, 0.2%, 0.3%, 0.5%
    tp_values = [0.002, 0.004, 0.006, 0.010]    # 0.2%, 0.4%, 0.6%, 1.0%
    
    sweep_results = run_sltp_sweep(
        signals_df,
        sl_values=sl_values,
        tp_values=tp_values,
        initial_capital=100_000,
        position_size_pct=0.10,
        commission_per_trade=1.0,
        slippage_bps=1.0,
    )
    
    # Print summary table
    print(f"\n{'='*90}")
    print("  SL/TP SWEEP RESULTS SUMMARY")
    print(f"{'='*90}")
    print(f"  {'SL%':>5} {'TP%':>5} {'Trades':>7} {'WinRate':>8} {'TotalPnL':>10} {'PF':>6} {'MaxDD':>8} {'Sharpe':>7}")
    print(f"  {'-'*5} {'-'*5} {'-'*7} {'-'*8} {'-'*10} {'-'*6} {'-'*8} {'-'*7}")
    for _, row in sweep_results.iterrows():
        pf_str = f"{row['profit_factor']:.2f}" if row['profit_factor'] != float('inf') else "  inf"
        print(f"  {row['SL%']:>5} {row['TP%']:>5} {row['num_trades']:>7.0f} "
              f"{row['win_rate']:>7.1%} {row['total_pnl']:>10.2f} {pf_str:>6} "
              f"{row['max_drawdown']:>7.2%} {row['sharpe_ratio']:>7.2f}")
    
    # Save to markdown
    results_path = '/home/thiago/.openclaw/workspace/projects/strat-research/infotheo_sltp_results.md'
    save_sweep_results_md(sweep_results, results_path)
    
    # --- Complexity Metric Showcase ---
    print("\n[5] Complexity metric examples:")
    print("    Test string 'AAAAAAAAAA' (pure pattern):")
    print(f"      LZ complexity (norm): {normalized_lz_complexity('0000000000'):.3f}")
    print(f"      Shannon block entropy: {shannon_block_entropy('0000000000', k=3):.3f}")
    print(f"      Permutation entropy:   {permutation_entropy(np.zeros(10), m=3):.3f}")
    
    random_str = ''.join([str(np.random.randint(0, 2)) for _ in range(100)])
    random_arr = np.random.randn(100)
    print(f"\n    Random binary string (100 chars):")
    print(f"      LZ complexity (norm): {normalized_lz_complexity(random_str):.3f}")
    print(f"      Shannon block entropy: {shannon_block_entropy(random_str, k=3):.3f}")
    print(f"      Permutation entropy:   {permutation_entropy(random_arr, m=3):.3f}")
    
    print("\n✅ Strategy with SL/TP sweep complete.")
