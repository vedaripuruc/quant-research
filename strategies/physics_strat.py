"""
Entropy Collapse Volatility Timing (ECVT) Strategy
====================================================

Based on: Singha (2025) "Hidden Order in Trades Predicts the Size of Price Moves"
          arXiv:2512.15720

Core Idea:
    Order-flow entropy (Shannon entropy of Markov transition matrix over
    discretized price-change × volume states) predicts the MAGNITUDE of
    future price moves but NOT the direction. When entropy collapses (low
    values), a large move is imminent.

    We use asymmetric risk management (tight stop, wide target) so that
    correct magnitude timing generates profit despite ~50% directional accuracy.

Two Implementations:
    1. OHLCV (1-min or 1-hour bars) — the practical version for most traders
    2. Tick data (bid/ask/volume) — closer to the original paper

NO LOOK-AHEAD BIAS: All signals use only data available at time t.
ENTRY: Always at the NEXT bar's open after signal fires.

Author: Curupira (research subagent), 2026-02-06
License: Research use only. Not financial advice.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass, field


# =============================================================================
# CORE ENTROPY ENGINE
# =============================================================================

def discretize_returns(returns: np.ndarray) -> np.ndarray:
    """Map returns to {0: down, 1: flat, 2: up}."""
    signs = np.sign(returns)
    # Map: -1 -> 0, 0 -> 1, +1 -> 2
    return (signs + 1).astype(int)


def compute_volume_quintiles(volumes: np.ndarray, window: int) -> np.ndarray:
    """
    Rolling quintile rank of volume.
    Returns values in {0, 1, 2, 3, 4} (quintile index).
    Uses only past data (no look-ahead).
    """
    n = len(volumes)
    quintiles = np.zeros(n, dtype=int)
    for i in range(n):
        start = max(0, i - window + 1)
        window_vols = volumes[start:i + 1]
        if len(window_vols) < 5:
            # Not enough data for meaningful quintile
            quintiles[i] = 2  # median quintile as default
        else:
            rank = np.searchsorted(np.sort(window_vols), volumes[i])
            quintiles[i] = min(4, int(5 * rank / len(window_vols)))
    return quintiles


def states_from_ohlcv(df: pd.DataFrame, vol_window: int = 120) -> np.ndarray:
    """
    Construct 15-state encoding from OHLCV data.
    State = return_sign (3) × volume_quintile (5) = 15 states.
    
    States are indexed 0..14:
        state = return_sign_code * 5 + volume_quintile
    """
    returns = df['close'].pct_change().fillna(0).values
    ret_signs = discretize_returns(returns)  # {0, 1, 2}
    vol_quints = compute_volume_quintiles(df['volume'].values, vol_window)  # {0..4}
    states = ret_signs * 5 + vol_quints  # {0..14}
    return states


def compute_markov_entropy(
    states: np.ndarray,
    window: int = 120,
    n_states: int = 15,
    min_transitions: int = 20
) -> np.ndarray:
    """
    Compute normalized Shannon entropy of Markov transition matrix
    over a rolling window.
    
    For each time t, we estimate the transition matrix from
    states[t-window+1:t+1] and compute:
    
        H_t = -1/log(K) * sum_i pi_i * sum_j p_ij * log(p_ij)
    
    where K = n_states, pi is the stationary distribution, and
    p_ij are transition probabilities.
    
    Returns normalized entropy in [0, 1]. High = random. Low = structured.
    """
    n = len(states)
    entropy = np.full(n, np.nan)
    log_k = np.log(n_states)
    
    for t in range(window, n):
        window_states = states[t - window + 1:t + 1]
        
        # Build transition count matrix
        trans_counts = np.zeros((n_states, n_states))
        for i in range(len(window_states) - 1):
            s_from = window_states[i]
            s_to = window_states[i + 1]
            trans_counts[s_from, s_to] += 1
        
        total_transitions = trans_counts.sum()
        if total_transitions < min_transitions:
            entropy[t] = np.nan
            continue
        
        # Normalize rows to get transition probabilities
        row_sums = trans_counts.sum(axis=1, keepdims=True)
        # For rows with no transitions, use uniform
        row_sums[row_sums == 0] = 1
        trans_probs = trans_counts / row_sums
        # Fill zero-count rows with uniform
        zero_rows = trans_counts.sum(axis=1) == 0
        trans_probs[zero_rows, :] = 1.0 / n_states
        
        # Compute stationary distribution via eigendecomposition
        try:
            eigenvalues, eigenvectors = np.linalg.eig(trans_probs.T)
            # Find eigenvector for eigenvalue closest to 1
            idx = np.argmin(np.abs(eigenvalues - 1.0))
            pi = np.abs(eigenvectors[:, idx].real)
            pi = pi / pi.sum()
        except np.linalg.LinAlgError:
            # Fallback: use empirical state frequencies
            pi = trans_counts.sum(axis=1)
            pi = pi / pi.sum() if pi.sum() > 0 else np.ones(n_states) / n_states
        
        # Compute normalized entropy
        h = 0.0
        for i in range(n_states):
            if pi[i] < 1e-12:
                continue
            for j in range(n_states):
                p = trans_probs[i, j]
                if p > 1e-12:
                    h -= pi[i] * p * np.log(p)
        
        entropy[t] = h / log_k  # Normalize to [0, 1]
    
    return entropy


# =============================================================================
# SIGNAL GENERATION
# =============================================================================

@dataclass
class ECVTParams:
    """Parameters for the Entropy Collapse Volatility Timing strategy."""
    # Entropy computation
    entropy_window: int = 120        # Rolling window for transition matrix (bars)
    vol_quintile_window: int = 120   # Window for volume quintile computation
    
    # Signal thresholds
    entropy_percentile: float = 5.0  # Signal when entropy < this percentile
    entropy_lookback: int = 500      # Lookback for percentile calculation
    volume_percentile: float = 90.0  # Volume must be above this percentile
    volume_lookback: int = 120       # Lookback for volume percentile
    min_trail_return: float = 0.001  # Min trailing return magnitude (0.1%)
    max_trail_return: float = 0.005  # Max trailing return magnitude (0.5%)
    trail_return_window: int = 30    # Bars to compute trailing return
    
    # Risk management
    stop_loss_pct: float = 0.0015   # 0.15% stop loss (tight)
    take_profit_pct: float = 0.005  # 0.50% take profit (wide)
    timeout_bars: int = 30          # Max bars to hold position
    
    # Transaction costs
    cost_per_trade_bps: float = 2.0  # Round-trip cost in basis points


def generate_signals(
    df: pd.DataFrame,
    params: ECVTParams = ECVTParams()
) -> pd.DataFrame:
    """
    Generate ECVT signals from OHLCV DataFrame.
    
    Input DataFrame must have columns: timestamp, open, high, low, close, volume
    
    Returns DataFrame with additional columns:
        - entropy: normalized Markov entropy
        - signal: {1: long, -1: short, 0: no signal}
        - entropy_threshold: rolling entropy threshold
    """
    df = df.copy()
    
    # Step 1: Compute states and entropy
    states = states_from_ohlcv(df, vol_window=params.vol_quintile_window)
    df['entropy'] = compute_markov_entropy(
        states, 
        window=params.entropy_window,
        n_states=15
    )
    
    # Step 2: Compute rolling entropy threshold (percentile)
    df['entropy_threshold'] = df['entropy'].rolling(
        window=params.entropy_lookback, 
        min_periods=params.entropy_lookback // 2
    ).quantile(params.entropy_percentile / 100.0)
    
    # Step 3: Compute rolling volume percentile threshold
    df['volume_threshold'] = df['volume'].rolling(
        window=params.volume_lookback,
        min_periods=params.volume_lookback // 2
    ).quantile(params.volume_percentile / 100.0)
    
    # Step 4: Compute trailing return (for direction and magnitude filter)
    df['trail_return'] = df['close'].pct_change(params.trail_return_window)
    
    # Step 5: Generate raw signals
    entropy_low = df['entropy'] < df['entropy_threshold']
    volume_high = df['volume'] > df['volume_threshold']
    trail_abs = df['trail_return'].abs()
    return_in_range = (trail_abs >= params.min_trail_return) & \
                      (trail_abs <= params.max_trail_return)
    
    # Combined signal condition
    signal_condition = entropy_low & volume_high & return_in_range
    
    # Direction from trailing return (momentum heuristic)
    direction = np.sign(df['trail_return'])
    
    df['signal'] = 0
    df.loc[signal_condition, 'signal'] = direction[signal_condition].astype(int)
    
    # Step 6: Prevent signals during existing positions (debounce)
    # We'll handle this in the backtest loop, but mark raw signals here
    df['raw_signal'] = df['signal'].copy()
    
    return df


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

@dataclass
class Trade:
    """Record of a single trade."""
    entry_bar: int
    entry_time: object
    entry_price: float
    direction: int  # 1 = long, -1 = short
    exit_bar: int = -1
    exit_time: object = None
    exit_price: float = 0.0
    exit_reason: str = ''
    pnl: float = 0.0
    pnl_bps: float = 0.0


@dataclass
class BacktestResult:
    """Results of a backtest run."""
    trades: List[Trade] = field(default_factory=list)
    total_pnl_bps: float = 0.0
    num_trades: int = 0
    win_rate: float = 0.0
    avg_win_bps: float = 0.0
    avg_loss_bps: float = 0.0
    max_drawdown_bps: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0


def run_backtest(
    df: pd.DataFrame,
    params: ECVTParams = ECVTParams()
) -> BacktestResult:
    """
    Run a simple event-driven backtest.
    
    Rules:
        - Entry at NEXT bar's open after signal fires
        - Stop loss and take profit checked against each bar's high/low
        - Timeout exit at bar's close
        - No overlapping positions
        - Transaction costs deducted from each trade
    
    Input df must already have 'signal' column (from generate_signals).
    """
    trades = []
    in_position = False
    current_trade = None
    
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        
        # Check if we need to exit current position
        if in_position and current_trade is not None:
            entry_price = current_trade.entry_price
            direction = current_trade.direction
            bars_held = i - current_trade.entry_bar
            
            stop_price = entry_price * (1 - direction * params.stop_loss_pct)
            target_price = entry_price * (1 + direction * params.take_profit_pct)
            
            exited = False
            
            # Check stop loss (using bar's adverse extreme)
            if direction == 1:  # Long
                if row['low'] <= stop_price:
                    current_trade.exit_price = stop_price
                    current_trade.exit_reason = 'stop_loss'
                    exited = True
                elif row['high'] >= target_price:
                    current_trade.exit_price = target_price
                    current_trade.exit_reason = 'take_profit'
                    exited = True
            else:  # Short
                if row['high'] >= stop_price:
                    current_trade.exit_price = stop_price
                    current_trade.exit_reason = 'stop_loss'
                    exited = True
                elif row['low'] <= target_price:
                    current_trade.exit_price = target_price
                    current_trade.exit_reason = 'take_profit'
                    exited = True
            
            # Check timeout
            if not exited and bars_held >= params.timeout_bars:
                current_trade.exit_price = row['close']
                current_trade.exit_reason = 'timeout'
                exited = True
            
            if exited:
                current_trade.exit_bar = i
                current_trade.exit_time = row.get('timestamp', i)
                raw_pnl = direction * (current_trade.exit_price - entry_price) / entry_price
                current_trade.pnl = raw_pnl
                current_trade.pnl_bps = raw_pnl * 10000 - params.cost_per_trade_bps
                trades.append(current_trade)
                in_position = False
                current_trade = None
        
        # Check for new entry signal (only if not in position)
        if not in_position and prev['signal'] != 0:
            # Enter at THIS bar's open (signal was at previous bar)
            current_trade = Trade(
                entry_bar=i,
                entry_time=row.get('timestamp', i),
                entry_price=row['open'],
                direction=int(prev['signal'])
            )
            in_position = True
    
    # Close any open position at last bar
    if in_position and current_trade is not None:
        last = df.iloc[-1]
        current_trade.exit_bar = len(df) - 1
        current_trade.exit_time = last.get('timestamp', len(df) - 1)
        current_trade.exit_price = last['close']
        current_trade.exit_reason = 'end_of_data'
        direction = current_trade.direction
        raw_pnl = direction * (current_trade.exit_price - current_trade.entry_price) / current_trade.entry_price
        current_trade.pnl = raw_pnl
        current_trade.pnl_bps = raw_pnl * 10000 - params.cost_per_trade_bps
        trades.append(current_trade)
    
    # Compute summary statistics
    result = BacktestResult()
    result.trades = trades
    result.num_trades = len(trades)
    
    if len(trades) == 0:
        return result
    
    pnls = np.array([t.pnl_bps for t in trades])
    result.total_pnl_bps = pnls.sum()
    result.win_rate = (pnls > 0).mean()
    
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    result.avg_win_bps = wins.mean() if len(wins) > 0 else 0.0
    result.avg_loss_bps = losses.mean() if len(losses) > 0 else 0.0
    
    # Max drawdown
    cumulative = np.cumsum(pnls)
    peak = np.maximum.accumulate(cumulative)
    drawdown = peak - cumulative
    result.max_drawdown_bps = drawdown.max() if len(drawdown) > 0 else 0.0
    
    # Sharpe-like ratio (per-trade)
    if pnls.std() > 0:
        result.sharpe_ratio = pnls.mean() / pnls.std()
    
    # Profit factor
    gross_profit = wins.sum() if len(wins) > 0 else 0.0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 1.0
    result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    
    return result


# =============================================================================
# TICK DATA VERSION
# =============================================================================

def states_from_ticks(
    df: pd.DataFrame,
    agg_seconds: int = 1,
    vol_window: int = 120
) -> np.ndarray:
    """
    Construct 15-state encoding from tick data.
    
    Input DataFrame columns: timestamp, bid, ask, volume
    
    Process:
        1. Compute mid-price = (bid + ask) / 2
        2. Aggregate to `agg_seconds` resolution
        3. Compute price-change signs and volume quintiles
        4. Encode into 15 states
    """
    df = df.copy()
    df['mid'] = (df['bid'] + df['ask']) / 2
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Aggregate to second-resolution bars
    agg = df.resample(f'{agg_seconds}s', on='timestamp').agg({
        'mid': ['first', 'last'],
        'volume': 'sum',
        'bid': 'last',
        'ask': 'last'
    })
    agg.columns = ['open', 'close', 'volume', 'bid', 'ask']
    agg = agg.dropna()
    
    # Compute returns and states
    returns = agg['close'].pct_change().fillna(0).values
    ret_signs = discretize_returns(returns)
    vol_quints = compute_volume_quintiles(agg['volume'].values, vol_window)
    states = ret_signs * 5 + vol_quints
    
    return states, agg


def generate_tick_signals(
    df: pd.DataFrame,
    params: ECVTParams = ECVTParams(),
    agg_seconds: int = 1
) -> pd.DataFrame:
    """
    Generate ECVT signals from tick data.
    
    Input DataFrame columns: timestamp, bid, ask, volume
    
    This is closer to Singha (2025) original methodology.
    """
    states, agg_df = states_from_ticks(df, agg_seconds=agg_seconds)
    
    # Add OHLCV-like columns for the rest of the pipeline
    agg_df['high'] = agg_df[['open', 'close']].max(axis=1)
    agg_df['low'] = agg_df[['open', 'close']].min(axis=1)
    agg_df = agg_df.reset_index()
    agg_df = agg_df.rename(columns={'timestamp': 'timestamp'})
    
    # Compute entropy directly from pre-computed states
    agg_df['entropy'] = compute_markov_entropy(
        states,
        window=params.entropy_window,
        n_states=15
    )
    
    # Rolling thresholds
    agg_df['entropy_threshold'] = agg_df['entropy'].rolling(
        window=params.entropy_lookback,
        min_periods=params.entropy_lookback // 2
    ).quantile(params.entropy_percentile / 100.0)
    
    agg_df['volume_threshold'] = agg_df['volume'].rolling(
        window=params.volume_lookback,
        min_periods=params.volume_lookback // 2
    ).quantile(params.volume_percentile / 100.0)
    
    agg_df['trail_return'] = agg_df['close'].pct_change(params.trail_return_window)
    
    # Signal logic
    entropy_low = agg_df['entropy'] < agg_df['entropy_threshold']
    volume_high = agg_df['volume'] > agg_df['volume_threshold']
    trail_abs = agg_df['trail_return'].abs()
    return_in_range = (trail_abs >= params.min_trail_return) & \
                      (trail_abs <= params.max_trail_return)
    
    signal_condition = entropy_low & volume_high & return_in_range
    direction = np.sign(agg_df['trail_return'])
    
    agg_df['signal'] = 0
    agg_df.loc[signal_condition, 'signal'] = direction[signal_condition].astype(int)
    agg_df['raw_signal'] = agg_df['signal'].copy()
    
    return agg_df


# =============================================================================
# DEMO / SYNTHETIC DATA TEST
# =============================================================================

def generate_synthetic_data(
    n_bars: int = 5000,
    freq: str = '1min',
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data with embedded regime structure.
    
    Creates two regimes:
        1. Normal: random walk with moderate volatility
        2. "Informed": periods of low entropy (structured transitions)
           followed by large directional moves
    
    This lets us validate that the entropy measure actually detects structure.
    """
    rng = np.random.RandomState(seed)
    
    timestamps = pd.date_range('2025-01-06 09:30:00', periods=n_bars, freq=freq)
    
    price = 100.0
    prices = []
    volumes = []
    
    # Create regime schedule: mostly random, with 5% of bars being "informed"
    regime = np.zeros(n_bars, dtype=int)  # 0 = normal, 1 = informed
    
    # Place informed periods (clusters of 30-60 bars followed by breakout)
    i = 200
    while i < n_bars - 100:
        if rng.random() < 0.03:  # 3% chance of starting informed period
            duration = rng.randint(20, 50)
            regime[i:i + duration] = 1
            # After informed period, large move
            direction = rng.choice([-1, 1])
            for j in range(i + duration, min(i + duration + 10, n_bars)):
                regime[j] = 2  # breakout
            i += duration + 50  # cool-off
        i += 1
    
    for i in range(n_bars):
        if regime[i] == 0:
            # Normal regime: random walk
            ret = rng.normal(0, 0.001)
            vol = abs(rng.normal(1000, 300))
        elif regime[i] == 1:
            # Informed regime: small returns but structured
            # Alternate directions with high probability (creating low entropy)
            if i > 0 and regime[i - 1] == 1:
                prev_ret = prices[-1] / prices[-2] - 1 if len(prices) > 1 else 0
                # 80% chance of continuing same direction (creates serial correlation)
                if rng.random() < 0.8:
                    ret = abs(rng.normal(0.0003, 0.0001)) * np.sign(prev_ret) if prev_ret != 0 else rng.normal(0, 0.0003)
                else:
                    ret = rng.normal(0, 0.0003)
            else:
                ret = rng.normal(0, 0.0003)
            vol = abs(rng.normal(2500, 500))  # Higher volume during informed trading
        else:  # regime == 2: breakout
            # Large directional move
            ret = rng.choice([-1, 1]) * abs(rng.normal(0.003, 0.001))
            vol = abs(rng.normal(4000, 1000))
        
        price *= (1 + ret)
        prices.append(price)
        volumes.append(max(100, vol))
    
    prices = np.array(prices)
    volumes = np.array(volumes)
    
    # Generate OHLCV
    noise_high = np.abs(rng.normal(0, 0.0005, n_bars))
    noise_low = np.abs(rng.normal(0, 0.0005, n_bars))
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': np.roll(prices, 1),  # open = previous close
        'high': prices * (1 + noise_high),
        'low': prices * (1 - noise_low),
        'close': prices,
        'volume': volumes
    })
    df.iloc[0, df.columns.get_loc('open')] = 100.0
    df['_regime'] = regime  # Hidden label for validation
    
    return df


def generate_synthetic_ticks(
    n_ticks: int = 100000,
    seed: int = 42
) -> pd.DataFrame:
    """Generate synthetic tick data for testing tick-data path."""
    rng = np.random.RandomState(seed)
    
    base_time = pd.Timestamp('2025-01-06 09:30:00')
    timestamps = [base_time + pd.Timedelta(milliseconds=i * rng.randint(50, 500))
                  for i in range(n_ticks)]
    
    mid = 100.0
    mids = []
    for _ in range(n_ticks):
        mid *= 1 + rng.normal(0, 0.0001)
        mids.append(mid)
    
    mids = np.array(mids)
    spread = 0.01  # 1 cent spread
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'bid': mids - spread / 2,
        'ask': mids + spread / 2,
        'volume': np.abs(rng.normal(100, 50, n_ticks)).astype(int) + 1
    })


def print_results(result: BacktestResult, label: str = "Backtest"):
    """Pretty-print backtest results."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  Total trades:       {result.num_trades}")
    print(f"  Total PnL:          {result.total_pnl_bps:+.1f} bps")
    print(f"  Win rate:           {result.win_rate:.1%}")
    print(f"  Avg win:            {result.avg_win_bps:+.1f} bps")
    print(f"  Avg loss:           {result.avg_loss_bps:+.1f} bps")
    print(f"  Max drawdown:       {result.max_drawdown_bps:.1f} bps")
    print(f"  Sharpe (per-trade): {result.sharpe_ratio:.3f}")
    print(f"  Profit factor:      {result.profit_factor:.2f}")
    
    if result.num_trades > 0:
        exit_reasons = {}
        for t in result.trades:
            exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1
        print(f"\n  Exit reasons:")
        for reason, count in sorted(exit_reasons.items()):
            print(f"    {reason}: {count} ({count/result.num_trades:.1%})")
    print(f"{'=' * 60}\n")


# =============================================================================
# MAIN: RUN DEMO
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  ENTROPY COLLAPSE VOLATILITY TIMING (ECVT) STRATEGY")
    print("  Based on: Singha (2025) arXiv:2512.15720")
    print("=" * 60)
    
    # --- OHLCV Demo ---
    print("\n[1] Generating synthetic 1-minute OHLCV data...")
    df = generate_synthetic_data(n_bars=5000, freq='1min')
    print(f"    {len(df)} bars, price range: {df['close'].min():.2f} - {df['close'].max():.2f}")
    
    # Use slightly relaxed params for synthetic data
    params = ECVTParams(
        entropy_window=60,           # 1 hour lookback
        vol_quintile_window=60,
        entropy_percentile=10.0,     # More relaxed for synthetic data
        entropy_lookback=300,
        volume_percentile=80.0,
        volume_lookback=60,
        min_trail_return=0.0005,     # 0.05%
        max_trail_return=0.01,       # 1.0%
        trail_return_window=15,
        stop_loss_pct=0.002,         # 0.2%
        take_profit_pct=0.006,       # 0.6%
        timeout_bars=30,
        cost_per_trade_bps=2.0
    )
    
    print("\n[2] Computing Markov entropy and generating signals...")
    df_signals = generate_signals(df, params)
    
    n_signals = (df_signals['signal'] != 0).sum()
    entropy_valid = df_signals['entropy'].notna().sum()
    print(f"    Valid entropy values: {entropy_valid}")
    print(f"    Raw signals generated: {n_signals}")
    
    if entropy_valid > 0:
        print(f"    Entropy range: [{df_signals['entropy'].min():.4f}, {df_signals['entropy'].max():.4f}]")
        print(f"    Mean entropy: {df_signals['entropy'].mean():.4f}")
    
    print("\n[3] Running backtest...")
    result = run_backtest(df_signals, params)
    print_results(result, "ECVT Strategy - 1-min OHLCV (Synthetic Data)")
    
    # Validate: check if entropy was lower during "informed" regime
    if '_regime' in df_signals.columns:
        informed_mask = df_signals['_regime'] == 1
        normal_mask = df_signals['_regime'] == 0
        if informed_mask.any() and normal_mask.any():
            ent_informed = df_signals.loc[informed_mask, 'entropy'].mean()
            ent_normal = df_signals.loc[normal_mask, 'entropy'].mean()
            print(f"  [VALIDATION] Mean entropy during 'informed' regime: {ent_informed:.4f}")
            print(f"  [VALIDATION] Mean entropy during 'normal' regime:   {ent_normal:.4f}")
            print(f"  [VALIDATION] Entropy drop: {(ent_normal - ent_informed) / ent_normal:.1%}")
            if ent_informed < ent_normal:
                print("  ✅ Entropy correctly detects structured (informed) periods!")
            else:
                print("  ❌ Entropy did NOT detect regime difference (may need tuning)")
    
    # --- Show example of using with hourly data ---
    print("\n" + "=" * 60)
    print("  HOURLY DATA VARIANT")
    print("=" * 60)
    
    print("\n[4] Generating synthetic 1-hour OHLCV data...")
    df_hourly = generate_synthetic_data(n_bars=2000, freq='1h', seed=123)
    
    hourly_params = ECVTParams(
        entropy_window=48,           # 2 days lookback
        vol_quintile_window=48,
        entropy_percentile=10.0,
        entropy_lookback=200,
        volume_percentile=80.0,
        volume_lookback=48,
        min_trail_return=0.001,
        max_trail_return=0.01,
        trail_return_window=12,      # 12 hours trailing
        stop_loss_pct=0.003,         # 0.3% stop
        take_profit_pct=0.01,        # 1.0% target
        timeout_bars=24,             # 24 hours timeout
        cost_per_trade_bps=3.0       # Higher cost for slower frequency
    )
    
    df_hourly_signals = generate_signals(df_hourly, hourly_params)
    n_hourly_signals = (df_hourly_signals['signal'] != 0).sum()
    print(f"    Signals generated: {n_hourly_signals}")
    
    hourly_result = run_backtest(df_hourly_signals, hourly_params)
    print_results(hourly_result, "ECVT Strategy - 1-hour OHLCV (Synthetic Data)")
    
    # --- Summary ---
    print("\n" + "=" * 60)
    print("  HOW TO USE WITH REAL DATA")
    print("=" * 60)
    print("""
    import pandas as pd
    from physics_strat import generate_signals, run_backtest, ECVTParams
    
    # Load your data (must have: timestamp, open, high, low, close, volume)
    df = pd.read_csv('your_1min_data.csv')
    
    # Configure parameters (tune for your instrument/timeframe)
    params = ECVTParams(
        entropy_window=120,         # 2 hours for 1-min data
        entropy_percentile=5.0,     # Strict: only extreme entropy drops
        stop_loss_pct=0.0015,       # 15 bps
        take_profit_pct=0.005,      # 50 bps
        cost_per_trade_bps=2.0,     # Adjust for your broker
    )
    
    # Generate signals
    df_with_signals = generate_signals(df, params)
    
    # Backtest
    result = run_backtest(df_with_signals, params)
    print(f"PnL: {result.total_pnl_bps:.0f} bps over {result.num_trades} trades")
    
    # For tick data:
    from physics_strat import generate_tick_signals
    tick_df = pd.read_csv('your_tick_data.csv')  # timestamp, bid, ask, volume
    tick_signals = generate_tick_signals(tick_df, params, agg_seconds=1)
    tick_result = run_backtest(tick_signals, params)
    """)
