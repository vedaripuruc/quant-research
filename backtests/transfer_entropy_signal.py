#!/usr/bin/env python3
"""
Signal 5: Cross-Asset Transfer Entropy Lead-Lag
================================================
Uses Transfer Entropy to measure causal information flow between BTC and altcoins.
When BTC "leads" a target asset (high TE), trade the target in BTC's direction.
When TE is low (target is independent), use mean-reversion (RSI) on the target.

For forex/gold: Use BTC as the leader asset instead.

Entry at NEXT bar's open (no look-ahead bias).
"""

import json
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Add parent to path for engine
sys.path.insert(0, '.')
from engine import BacktestEngine, BacktestConfig, calculate_metrics


# ──────────────────────────────────────────
# Transfer Entropy (manual implementation)
# ──────────────────────────────────────────
def discretize_returns(returns: np.ndarray, n_bins: int = 3) -> np.ndarray:
    """Discretize continuous returns into bins for TE calculation.
    Uses quantile-based binning: negative / neutral / positive.
    """
    # Use percentile-based bins for robustness
    q_low = np.nanpercentile(returns, 33.3)
    q_high = np.nanpercentile(returns, 66.6)
    
    out = np.zeros(len(returns), dtype=int)
    out[returns < q_low] = 0   # negative
    out[returns > q_high] = 2  # positive
    out[(returns >= q_low) & (returns <= q_high)] = 1  # neutral
    
    return out


def transfer_entropy(source: np.ndarray, target: np.ndarray, k: int = 1) -> float:
    """
    Calculate Transfer Entropy from source → target.
    TE(X→Y) = H(Y_future | Y_past) - H(Y_future | Y_past, X_past)
    
    Uses discrete (binned) time series.
    k = history length (lag order).
    
    Returns TE in bits. Higher = more causal influence from source to target.
    """
    n = len(source)
    if n < k + 2:
        return 0.0
    
    # Build joint distributions
    # Y_future = target[k:]
    # Y_past = target[k-1:-1] (for k=1)
    # X_past = source[k-1:-1]
    
    y_future = target[k:]
    y_past = target[:n-k]
    x_past = source[:n-k]
    
    # Trim to same length
    min_len = min(len(y_future), len(y_past), len(x_past))
    y_future = y_future[:min_len]
    y_past = y_past[:min_len]
    x_past = x_past[:min_len]
    
    if min_len < 10:
        return 0.0
    
    # Compute probabilities using counting
    n_bins = 3  # Our discretization uses 3 bins
    
    # P(y_future, y_past, x_past)
    joint_xyz = np.zeros((n_bins, n_bins, n_bins))
    # P(y_future, y_past)
    joint_yz = np.zeros((n_bins, n_bins))
    # P(y_past, x_past)
    joint_zx = np.zeros((n_bins, n_bins))
    # P(y_past)
    marg_z = np.zeros(n_bins)
    
    for i in range(min_len):
        yf_i = int(y_future[i])
        yp_i = int(y_past[i])
        xp_i = int(x_past[i])
        
        if 0 <= yf_i < n_bins and 0 <= yp_i < n_bins and 0 <= xp_i < n_bins:
            joint_xyz[yf_i, yp_i, xp_i] += 1
            joint_yz[yf_i, yp_i] += 1
            joint_zx[yp_i, xp_i] += 1
            marg_z[yp_i] += 1
    
    # Normalize
    total = joint_xyz.sum()
    if total == 0:
        return 0.0
    
    joint_xyz /= total
    joint_yz /= total
    joint_zx /= total
    marg_z /= total
    
    # TE = sum over all states of:
    # p(y_f, y_p, x_p) * log2( p(y_f, y_p, x_p) * p(y_p) / (p(y_p, x_p) * p(y_f, y_p)) )
    te = 0.0
    eps = 1e-12
    
    for yf_i in range(n_bins):
        for yp_i in range(n_bins):
            for xp_i in range(n_bins):
                p_xyz = joint_xyz[yf_i, yp_i, xp_i]
                p_yz = joint_yz[yf_i, yp_i]
                p_zx = joint_zx[yp_i, xp_i]
                p_z = marg_z[yp_i]
                
                if p_xyz > eps and p_yz > eps and p_zx > eps and p_z > eps:
                    te += p_xyz * np.log2((p_xyz * p_z) / (p_zx * p_yz))
    
    return max(0.0, te)


def rolling_transfer_entropy(source_returns: np.ndarray, target_returns: np.ndarray,
                              window: int = 30, k: int = 1) -> np.ndarray:
    """Compute rolling transfer entropy over a sliding window."""
    n = len(source_returns)
    te_values = np.full(n, np.nan)
    
    for i in range(window, n):
        src_window = source_returns[i-window:i]
        tgt_window = target_returns[i-window:i]
        
        # Skip if too many NaNs
        valid = ~(np.isnan(src_window) | np.isnan(tgt_window))
        if valid.sum() < window * 0.7:
            continue
        
        src_disc = discretize_returns(src_window[valid])
        tgt_disc = discretize_returns(tgt_window[valid])
        
        te_values[i] = transfer_entropy(src_disc, tgt_disc, k=k)
    
    return te_values


# ──────────────────────────────────────────
# Technical Indicators
# ──────────────────────────────────────────
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Standard RSI calculation."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    high = df['High']
    low = df['Low']
    close = df['Close'].shift(1)
    
    tr = pd.concat([
        high - low,
        (high - close).abs(),
        (low - close).abs()
    ], axis=1).max(axis=1)
    
    return tr.rolling(window=period).mean()


# ──────────────────────────────────────────
# Data Fetching
# ──────────────────────────────────────────
SYMBOLS = ['EURUSD=X', 'LINK-USD', 'ADA-USD', 'XRP-USD', 'GC=F', 'BTC-USD']
CRYPTO_TARGETS = ['LINK-USD', 'ADA-USD', 'XRP-USD']
LEADER_MAP = {
    'LINK-USD': 'BTC-USD',
    'ADA-USD': 'BTC-USD',
    'XRP-USD': 'BTC-USD',
    'EURUSD=X': 'BTC-USD',  # Use BTC as leader for forex too
    'GC=F': 'BTC-USD',       # Use BTC as leader for gold
}


def fetch_all_data(period: str = '2y', interval: str = '1d') -> dict:
    """Download all required data."""
    data = {}
    for symbol in SYMBOLS:
        print(f"  Fetching {symbol}...")
        try:
            df = yf.download(symbol, period=period, interval=interval, progress=False)
            df.reset_index(inplace=True)
            # Handle multi-level columns from yfinance
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            # Rename 'Datetime' to 'Date' if needed
            if 'Datetime' in df.columns:
                df.rename(columns={'Datetime': 'Date'}, inplace=True)
            data[symbol] = df
            print(f"    → {len(df)} bars")
        except Exception as e:
            print(f"    → ERROR: {e}")
    return data


# ──────────────────────────────────────────
# Signal Generator
# ──────────────────────────────────────────
def build_te_signal_fn(target_df: pd.DataFrame, leader_df: pd.DataFrame,
                        te_window: int = 30, te_high_pct: float = 75,
                        te_low_pct: float = 25, atr_mult_sl: float = 1.5,
                        rr_ratio: float = 2.0, btc_move_threshold: float = 0.01):
    """
    Build a signal function for the Transfer Entropy strategy.
    
    Returns a closure that the BacktestEngine can call.
    """
    # Pre-compute all indicators
    target_returns = target_df['Close'].pct_change().values
    leader_returns = leader_df['Close'].pct_change().values
    
    # Align data by date
    target_dates = pd.to_datetime(target_df['Date']).values
    leader_dates = pd.to_datetime(leader_df['Date']).values
    
    # Create aligned returns arrays using date matching
    date_to_leader_ret = dict(zip(leader_dates, leader_returns))
    
    aligned_leader_returns = np.array([
        date_to_leader_ret.get(d, np.nan) for d in target_dates
    ])
    
    # Compute rolling TE (BTC → target)
    te_forward = rolling_transfer_entropy(aligned_leader_returns, target_returns, window=te_window)
    
    # Compute rolling TE (target → BTC) - reverse direction
    te_reverse = rolling_transfer_entropy(target_returns, aligned_leader_returns, window=te_window)
    
    # Compute rolling percentiles of TE
    te_high_thresh = pd.Series(te_forward).rolling(window=120, min_periods=30).quantile(te_high_pct / 100).values
    te_low_thresh = pd.Series(te_forward).rolling(window=120, min_periods=30).quantile(te_low_pct / 100).values
    
    # RSI for mean-reversion
    rsi = compute_rsi(target_df['Close']).values
    
    # ATR for SL/TP
    atr = compute_atr(target_df).values
    
    def signal_fn(df, i):
        """Generate signal at bar i. Entry will be at bar i+1."""
        if i < te_window + 10 or i >= len(te_forward):
            return None
        
        current_te = te_forward[i]
        high_threshold = te_high_thresh[i]
        low_threshold = te_low_thresh[i]
        current_atr = atr[i]
        current_rsi = rsi[i]
        
        if np.isnan(current_te) or np.isnan(current_atr) or current_atr <= 0:
            return None
        if np.isnan(high_threshold) or np.isnan(low_threshold):
            return None
        
        current_close = df.iloc[i]['Close']
        leader_ret_today = aligned_leader_returns[i] if i < len(aligned_leader_returns) else np.nan
        
        if np.isnan(leader_ret_today):
            return None
        
        direction = None
        
        # MODE 1: High TE — BTC leads target
        if current_te > high_threshold:
            # Only act on strong BTC moves
            if abs(leader_ret_today) > btc_move_threshold:
                if leader_ret_today > 0:
                    direction = 'long'   # BTC went up, target should follow
                else:
                    direction = 'short'  # BTC went down, target should follow
        
        # MODE 2: Low TE — target is independent, use mean reversion
        elif current_te < low_threshold:
            if current_rsi < 30:
                direction = 'long'   # Oversold → buy
            elif current_rsi > 70:
                direction = 'short'  # Overbought → sell
        
        if direction is None:
            return None
        
        # SL/TP based on ATR
        sl_distance = current_atr * atr_mult_sl
        tp_distance = sl_distance * rr_ratio
        
        if direction == 'long':
            stop_loss = current_close - sl_distance
            take_profit = current_close + tp_distance
        else:
            stop_loss = current_close + sl_distance
            take_profit = current_close - tp_distance
        
        return {
            'direction': direction,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            '_signal_open': current_close,
        }
    
    return signal_fn


# ──────────────────────────────────────────
# Walk-Forward Testing
# ──────────────────────────────────────────
def walk_forward_test(target_df: pd.DataFrame, leader_df: pd.DataFrame,
                       n_splits: int = 5, train_pct: float = 0.6) -> dict:
    """
    Walk-forward test: train on train_pct, test on remaining, slide forward.
    """
    n = len(target_df)
    window_size = n // n_splits
    
    all_oos_trades = []
    oos_windows = []
    
    for fold in range(n_splits):
        start_idx = fold * (window_size // 2)  # 50% overlap
        end_idx = min(start_idx + window_size, n)
        
        if end_idx - start_idx < 60:
            continue
        
        train_end = start_idx + int((end_idx - start_idx) * train_pct)
        
        # Build signal on full data (TE needs history) but only count OOS trades
        signal_fn = build_te_signal_fn(target_df, leader_df)
        
        # Run backtest on OOS portion
        oos_df = target_df.iloc[train_end:end_idx].copy().reset_index(drop=True)
        if len(oos_df) < 20:
            continue
        
        engine = BacktestEngine(BacktestConfig(
            slippage_pct=0.001,
            commission_pct=0.001,
        ))
        
        # We need to run on the full dataset but only count OOS trades
        # So run on full data and filter
        trades_df = engine.run(target_df.copy().reset_index(drop=True), signal_fn)
        
        # Filter to OOS period
        if not trades_df.empty and 'signal_bar' in trades_df.columns:
            oos_trades = trades_df[
                (trades_df['signal_bar'] >= train_end) & 
                (trades_df['signal_bar'] < end_idx)
            ]
        else:
            oos_trades = pd.DataFrame()
        
        oos_pnl = oos_trades['pnl_pct'].sum() * 100 if not oos_trades.empty else 0
        oos_n = len(oos_trades)
        
        oos_windows.append({
            'fold': fold,
            'oos_start': train_end,
            'oos_end': end_idx,
            'n_trades': oos_n,
            'oos_return': round(oos_pnl, 2),
            'profitable': oos_pnl > 0,
        })
        
        if not oos_trades.empty:
            all_oos_trades.append(oos_trades)
    
    profitable_windows = sum(1 for w in oos_windows if w['profitable'])
    total_windows = len(oos_windows)
    
    combined_oos = pd.concat(all_oos_trades) if all_oos_trades else pd.DataFrame()
    
    return {
        'windows': oos_windows,
        'profitable_pct': round(profitable_windows / total_windows * 100, 1) if total_windows > 0 else 0,
        'total_windows': total_windows,
        'profitable_windows': profitable_windows,
        'combined_oos_trades': combined_oos,
    }


# ──────────────────────────────────────────
# Main Backtest Runner
# ──────────────────────────────────────────
def run_full_backtest():
    """Run transfer entropy signal on all assets."""
    print("=" * 70)
    print("SIGNAL 5: Cross-Asset Transfer Entropy Lead-Lag")
    print("=" * 70)
    
    # Fetch data
    print("\n📥 Fetching 2Y daily data...")
    data = fetch_all_data(period='2y', interval='1d')
    
    if 'BTC-USD' not in data:
        print("ERROR: Could not fetch BTC-USD data!")
        return {}
    
    results = {}
    
    for symbol in SYMBOLS:
        if symbol == 'BTC-USD':
            continue  # BTC is the leader, not a target
        
        if symbol not in data:
            print(f"\n⚠️ Skipping {symbol} — no data")
            continue
        
        leader_symbol = LEADER_MAP.get(symbol, 'BTC-USD')
        if leader_symbol not in data:
            print(f"\n⚠️ Skipping {symbol} — no leader data ({leader_symbol})")
            continue
        
        target_df = data[symbol].copy()
        leader_df = data[leader_symbol].copy()
        
        print(f"\n{'─' * 50}")
        print(f"📊 {symbol} (leader: {leader_symbol})")
        print(f"   Target bars: {len(target_df)}, Leader bars: {len(leader_df)}")
        
        # Full backtest
        signal_fn = build_te_signal_fn(target_df, leader_df)
        
        engine = BacktestEngine(BacktestConfig(
            slippage_pct=0.001,
            commission_pct=0.001,
        ))
        
        trades_df = engine.run(target_df.copy().reset_index(drop=True), signal_fn)
        metrics = calculate_metrics(trades_df, target_df)
        
        print(f"   Trades: {metrics['total_trades']}")
        print(f"   Win Rate: {metrics['win_rate']}%")
        print(f"   Profit Factor: {metrics['profit_factor']}")
        print(f"   Strategy Return: {metrics['strategy_return']}%")
        print(f"   Buy & Hold: {metrics['buy_hold_return']}%")
        print(f"   Max DD: {metrics['max_drawdown']}%")
        print(f"   Sharpe: {metrics['sharpe']}")
        
        # Walk-forward
        print(f"\n   🔄 Walk-Forward Test (5 folds)...")
        wf = walk_forward_test(target_df, leader_df, n_splits=5)
        
        print(f"   OOS Windows Profitable: {wf['profitable_windows']}/{wf['total_windows']} ({wf['profitable_pct']}%)")
        for w in wf['windows']:
            print(f"      Fold {w['fold']}: {w['n_trades']} trades, return={w['oos_return']}% {'✅' if w['profitable'] else '❌'}")
        
        results[symbol] = {
            'metrics': metrics,
            'walk_forward': {
                'profitable_pct': wf['profitable_pct'],
                'total_windows': wf['total_windows'],
                'profitable_windows': wf['profitable_windows'],
                'windows': wf['windows'],
            }
        }
    
    # Save results
    output_path = 'results_transfer_entropy.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n💾 Results saved to {output_path}")
    
    return results


if __name__ == '__main__':
    results = run_full_backtest()
