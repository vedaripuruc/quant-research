"""
Signal 4: Jump Detection (Bipower Variation)
=============================================
Uses Realized Variance vs Bipower Variation to detect price jumps.

Jump ratio = (RV - BPV) / RV — fraction of variance from jumps.

FADE:  jump ratio > 0.3 → wait 1 bar, fade the jump (mean reversion)
TREND: jump ratio < 0.1 AND smooth trending (20-bar momentum) → ride trend
FLAT:  otherwise

Fade trades: SL = 1*ATR, TP = 2*ATR (tight)
Trend trades: SL = 2*ATR, TP = 4*ATR (wide)
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
import yfinance as yf

from engine import BacktestEngine, BacktestConfig, calculate_metrics


# ─── Realized measures ──────────────────────────────────────────────────

def compute_realized_variance(returns: np.ndarray) -> float:
    """RV = sum of squared returns over the window."""
    return np.sum(returns ** 2)


def compute_bipower_variation(returns: np.ndarray) -> float:
    """
    BPV = (π/2) * sum(|r_i| * |r_{i-1}|)
    Robust to jumps — captures smooth/continuous variation.
    """
    abs_ret = np.abs(returns)
    bpv = (np.pi / 2) * np.sum(abs_ret[1:] * abs_ret[:-1])
    return bpv


def compute_jump_measures(df: pd.DataFrame, i: int, window: int = 20):
    """
    Compute RV, BPV, jump component, and jump ratio at bar i.
    Returns (rv, bpv, jump, jump_ratio) or None if insufficient data.
    """
    if i < window + 1:  # Need window+1 for returns
        return None

    # Daily log returns
    closes = df['Close'].iloc[i - window:i + 1].values.astype(float)
    returns = np.diff(np.log(closes))  # window returns

    if len(returns) < window:
        return None

    rv = compute_realized_variance(returns)
    bpv = compute_bipower_variation(returns)

    jump = max(rv - bpv, 0)
    jump_ratio = jump / rv if rv > 0 else 0

    return rv, bpv, jump, jump_ratio


# ─── ATR ─────────────────────────────────────────────────────────────────

def compute_atr(df: pd.DataFrame, i: int, period: int = 14) -> float:
    """Compute ATR at bar i."""
    if i < period:
        return None
    highs = df['High'].iloc[i - period + 1:i + 1].values
    lows = df['Low'].iloc[i - period + 1:i + 1].values
    closes = df['Close'].iloc[i - period:i].values

    tr = np.maximum(highs - lows,
                    np.maximum(np.abs(highs - closes),
                               np.abs(lows - closes)))
    return np.mean(tr)


# ─── Signal function ────────────────────────────────────────────────────

# State tracking for the "wait 1 bar" logic
_jump_state = {}

def reset_jump_state():
    """Reset state between runs."""
    global _jump_state
    _jump_state = {}

def jump_signal(df: pd.DataFrame, i: int, window: int = 20) -> dict:
    """
    Jump detection signal using Bipower Variation.
    
    - Jump ratio > 0.3: FADE (after 1 bar wait)
    - Jump ratio < 0.1 + trending: TREND
    - Otherwise: no trade
    """
    global _jump_state

    if i < window + 2:
        return None

    atr = compute_atr(df, i, period=14)
    if atr is None or atr <= 0:
        return None

    measures = compute_jump_measures(df, i, window)
    if measures is None:
        return None

    rv, bpv, jump, jump_ratio = measures
    close = df['Close'].iloc[i]

    # ── Check if we're in "wait 1 bar after jump" state ──
    if _jump_state.get('waiting'):
        if _jump_state['wait_bar'] == i - 1:
            # This is the bar AFTER the wait bar — NOW we fade
            jump_dir = _jump_state['jump_direction']
            _jump_state = {}  # Clear state

            # Fade the jump: go opposite
            if jump_dir > 0:
                direction = 'short'  # Jump was up, fade down
                sl = close + 1.0 * atr
                tp = close - 2.0 * atr
            else:
                direction = 'long'  # Jump was down, fade up
                sl = close - 1.0 * atr
                tp = close + 2.0 * atr

            return {
                'direction': direction,
                'stop_loss': sl,
                'take_profit': tp,
                '_signal_open': close,
                '_trade_type': 'fade',
            }
        elif i > _jump_state['wait_bar'] + 1:
            _jump_state = {}  # Expired, clear

    # ── Detect new jump ──
    if jump_ratio > 0.3:
        # Big jump detected — determine direction
        ret_today = np.log(df['Close'].iloc[i] / df['Close'].iloc[i - 1])
        _jump_state = {
            'waiting': True,
            'wait_bar': i,  # We wait this bar, trade next
            'jump_direction': 1 if ret_today > 0 else -1,
        }
        return None  # Wait 1 bar

    # ── Smooth trend regime ──
    if jump_ratio < 0.1:
        # Check 20-bar momentum for Hurst-like trending
        if i < 20:
            return None

        # Simple momentum: price change over 20 bars
        mom = (df['Close'].iloc[i] - df['Close'].iloc[i - 20]) / df['Close'].iloc[i - 20]

        # Require meaningful momentum (> 2% over 20 bars)
        if abs(mom) < 0.02:
            return None

        if mom > 0:
            direction = 'long'
            sl = close - 2.0 * atr
            tp = close + 4.0 * atr
        else:
            direction = 'short'
            sl = close + 2.0 * atr
            tp = close - 4.0 * atr

        return {
            'direction': direction,
            'stop_loss': sl,
            'take_profit': tp,
            '_signal_open': close,
            '_trade_type': 'trend',
        }

    return None  # Middle ground — no trade


# ─── Fetch data ──────────────────────────────────────────────────────────

def fetch_daily_data(symbol: str, years: int = 2) -> pd.DataFrame:
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
    total_bars = len(df)
    bars_per_month = total_bars / 24
    train_bars = int(train_months * bars_per_month)
    test_bars = int(test_months * bars_per_month)
    step = test_bars

    results = []
    for w in range(n_windows):
        start = w * step
        train_end = start + train_bars
        test_end = train_end + test_bars

        if test_end > total_bars:
            break

        test_df = df.iloc[start:test_end].reset_index(drop=True)

        reset_jump_state()  # Clean state for each window
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
        print(f"  JUMP DETECTION SIGNAL: {sym}")
        print(f"{'='*60}")

        df = fetch_daily_data(sym, years=2)
        print(f"  Data: {len(df)} bars, {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")

        if len(df) < 100:
            print(f"  SKIP: Not enough data ({len(df)} bars)")
            all_results[sym] = {'error': 'insufficient data'}
            continue

        # Full backtest
        reset_jump_state()
        engine = BacktestEngine(BacktestConfig())
        trades_df = engine.run(df, jump_signal)
        metrics = calculate_metrics(trades_df, df)

        # Count trade types
        n_fade = 0
        n_trend = 0
        if not trades_df.empty:
            # We can't easily get trade type from trades_df, count from signal runs
            pass

        print(f"  Trades: {metrics['total_trades']}")
        print(f"  Win Rate: {metrics['win_rate']}%")
        print(f"  Profit Factor: {metrics['profit_factor']}")
        print(f"  Strategy Return: {metrics['strategy_return']}%")
        print(f"  Buy&Hold Return: {metrics['buy_hold_return']}%")
        print(f"  Max Drawdown: {metrics['max_drawdown']}%")

        # Walk-forward
        print(f"\n  Walk-Forward (6 windows, 8m train / 4m test):")
        wf_results = walk_forward(df, jump_signal)
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

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, pd.Timestamp):
                return str(obj)
            return super().default(obj)

    with open('results_jump.json', 'w') as f:
        json.dump(all_results, f, indent=2, cls=NpEncoder)

    print(f"\n\nResults saved to results_jump.json")
    return all_results


if __name__ == '__main__':
    main()
