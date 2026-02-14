"""
Fast ECVT implementation — optimized entropy computation using numpy.
Drop-in replacement for physics_strat functions.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class ECVTParams:
    entropy_window: int = 120
    vol_quintile_window: int = 120
    entropy_percentile: float = 5.0
    entropy_lookback: int = 500
    volume_percentile: float = 90.0
    volume_lookback: int = 120
    min_trail_return: float = 0.001
    max_trail_return: float = 0.005
    trail_return_window: int = 30
    stop_loss_pct: float = 0.0015
    take_profit_pct: float = 0.005
    timeout_bars: int = 30
    cost_per_trade_bps: float = 2.0


@dataclass
class Trade:
    entry_bar: int
    entry_time: object
    entry_price: float
    direction: int
    exit_bar: int = -1
    exit_time: object = None
    exit_price: float = 0.0
    exit_reason: str = ''
    pnl: float = 0.0
    pnl_bps: float = 0.0


@dataclass
class BacktestResult:
    trades: List[Trade] = field(default_factory=list)
    total_pnl_bps: float = 0.0
    num_trades: int = 0
    win_rate: float = 0.0
    avg_win_bps: float = 0.0
    avg_loss_bps: float = 0.0
    max_drawdown_bps: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0


def discretize_returns(returns: np.ndarray) -> np.ndarray:
    signs = np.sign(returns)
    return (signs + 1).astype(int)


def compute_volume_quintiles_fast(volumes: np.ndarray, window: int) -> np.ndarray:
    """Vectorized rolling quintile computation."""
    n = len(volumes)
    quintiles = np.full(n, 2, dtype=int)  # default median
    
    for i in range(min(window, n), n):
        start = max(0, i - window + 1)
        w = volumes[start:i + 1]
        rank = np.searchsorted(np.sort(w), volumes[i])
        quintiles[i] = min(4, int(5 * rank / len(w)))
    
    return quintiles


def states_from_ohlcv(df: pd.DataFrame, vol_window: int = 120) -> np.ndarray:
    returns = df['close'].pct_change().fillna(0).values
    ret_signs = discretize_returns(returns)
    vol_quints = compute_volume_quintiles_fast(df['volume'].values, vol_window)
    return ret_signs * 5 + vol_quints


def compute_markov_entropy_fast(
    states: np.ndarray,
    window: int = 120,
    n_states: int = 15,
    min_transitions: int = 20
) -> np.ndarray:
    """
    Optimized Markov entropy: uses numpy for transition counting
    and avoids eigendecomposition by using empirical state frequencies
    (much faster, nearly identical results for large windows).
    """
    n = len(states)
    entropy = np.full(n, np.nan)
    log_k = np.log(n_states)
    
    # Pre-compute transition pairs
    pairs_from = states[:-1]
    pairs_to = states[1:]
    
    for t in range(window, n):
        w_from = pairs_from[t - window:t]
        w_to = pairs_to[t - window:t]
        
        # Build transition matrix with numpy
        trans_counts = np.zeros((n_states, n_states))
        np.add.at(trans_counts, (w_from, w_to), 1)
        
        total = trans_counts.sum()
        if total < min_transitions:
            continue
        
        # Row sums for normalization
        row_sums = trans_counts.sum(axis=1)
        
        # Use empirical state frequencies instead of eigendecomposition
        pi = row_sums / total
        
        # Normalize rows to get transition probabilities
        mask = row_sums > 0
        trans_probs = np.zeros_like(trans_counts)
        trans_probs[mask] = trans_counts[mask] / row_sums[mask, np.newaxis]
        
        # Compute entropy: H = -sum_i pi_i * sum_j p_ij * log(p_ij)
        log_probs = np.zeros_like(trans_probs)
        pos = trans_probs > 1e-12
        log_probs[pos] = np.log(trans_probs[pos])
        
        # Weighted row entropies
        row_ent = -(trans_probs * log_probs).sum(axis=1)
        h = (pi * row_ent).sum()
        
        entropy[t] = h / log_k
    
    return entropy


def generate_signals(df: pd.DataFrame, params: ECVTParams = ECVTParams()) -> pd.DataFrame:
    df = df.copy()
    
    states = states_from_ohlcv(df, vol_window=params.vol_quintile_window)
    df['entropy'] = compute_markov_entropy_fast(
        states, window=params.entropy_window, n_states=15
    )
    
    df['entropy_threshold'] = df['entropy'].rolling(
        window=params.entropy_lookback,
        min_periods=params.entropy_lookback // 2
    ).quantile(params.entropy_percentile / 100.0)
    
    df['volume_threshold'] = df['volume'].rolling(
        window=params.volume_lookback,
        min_periods=params.volume_lookback // 2
    ).quantile(params.volume_percentile / 100.0)
    
    df['trail_return'] = df['close'].pct_change(params.trail_return_window)
    
    entropy_low = df['entropy'] < df['entropy_threshold']
    volume_high = df['volume'] > df['volume_threshold']
    trail_abs = df['trail_return'].abs()
    return_in_range = (trail_abs >= params.min_trail_return) & \
                      (trail_abs <= params.max_trail_return)
    
    signal_condition = entropy_low & volume_high & return_in_range
    direction = np.sign(df['trail_return'])
    
    df['signal'] = 0
    df.loc[signal_condition, 'signal'] = direction[signal_condition].astype(int)
    df['raw_signal'] = df['signal'].copy()
    
    return df


def run_backtest(df: pd.DataFrame, params: ECVTParams = ECVTParams()) -> BacktestResult:
    trades = []
    in_position = False
    current_trade = None
    
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    signals = df['signal'].values
    timestamps = df['timestamp'].values if 'timestamp' in df.columns else np.arange(len(df))
    
    for i in range(1, len(df)):
        if in_position and current_trade is not None:
            entry_price = current_trade.entry_price
            direction = current_trade.direction
            bars_held = i - current_trade.entry_bar
            
            stop_price = entry_price * (1 - direction * params.stop_loss_pct)
            target_price = entry_price * (1 + direction * params.take_profit_pct)
            
            exited = False
            
            if direction == 1:
                if lows[i] <= stop_price:
                    current_trade.exit_price = stop_price
                    current_trade.exit_reason = 'stop_loss'
                    exited = True
                elif highs[i] >= target_price:
                    current_trade.exit_price = target_price
                    current_trade.exit_reason = 'take_profit'
                    exited = True
            else:
                if highs[i] >= stop_price:
                    current_trade.exit_price = stop_price
                    current_trade.exit_reason = 'stop_loss'
                    exited = True
                elif lows[i] <= target_price:
                    current_trade.exit_price = target_price
                    current_trade.exit_reason = 'take_profit'
                    exited = True
            
            if not exited and bars_held >= params.timeout_bars:
                current_trade.exit_price = closes[i]
                current_trade.exit_reason = 'timeout'
                exited = True
            
            if exited:
                current_trade.exit_bar = i
                current_trade.exit_time = timestamps[i]
                raw_pnl = direction * (current_trade.exit_price - entry_price) / entry_price
                current_trade.pnl = raw_pnl
                current_trade.pnl_bps = raw_pnl * 10000 - params.cost_per_trade_bps
                trades.append(current_trade)
                in_position = False
                current_trade = None
        
        if not in_position and signals[i - 1] != 0:
            current_trade = Trade(
                entry_bar=i,
                entry_time=timestamps[i],
                entry_price=opens[i],
                direction=int(signals[i - 1])
            )
            in_position = True
    
    if in_position and current_trade is not None:
        i = len(df) - 1
        current_trade.exit_bar = i
        current_trade.exit_time = timestamps[i]
        current_trade.exit_price = closes[i]
        current_trade.exit_reason = 'end_of_data'
        direction = current_trade.direction
        raw_pnl = direction * (current_trade.exit_price - current_trade.entry_price) / current_trade.entry_price
        current_trade.pnl = raw_pnl
        current_trade.pnl_bps = raw_pnl * 10000 - params.cost_per_trade_bps
        trades.append(current_trade)
    
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
    
    cumulative = np.cumsum(pnls)
    peak = np.maximum.accumulate(cumulative)
    drawdown = peak - cumulative
    result.max_drawdown_bps = drawdown.max() if len(drawdown) > 0 else 0.0
    
    if pnls.std() > 0:
        result.sharpe_ratio = pnls.mean() / pnls.std()
    
    gross_profit = wins.sum() if len(wins) > 0 else 0.0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 1.0
    result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    
    return result


def print_results(result: BacktestResult, label: str = "Backtest"):
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
