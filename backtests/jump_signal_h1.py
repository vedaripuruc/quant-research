"""
Jump Detection on Gold H1 (Hourly)
===================================
Adapts the daily jump detection signal for hourly bars.

Key differences from daily version:
- RV and BPV computed from intraday hourly returns grouped by trading day
- Signal fires at the LAST bar of each trading day
- Entry at next day's FIRST bar (pending_signal via engine)
- SL/TP based on daily ATR (computed from daily OHLC aggregated from H1)

Walk-forward: 6 windows, 8 months train + 4 months test
Clean method: suppress signals during training period
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
import yfinance as yf

from engine import BacktestEngine, BacktestConfig, calculate_metrics


# ─── Realized measures (same math, applied to hourly returns per day) ────

def compute_rv(returns: np.ndarray) -> float:
    """RV = sum of squared returns."""
    return np.sum(returns ** 2)


def compute_bpv(returns: np.ndarray) -> float:
    """BPV = (π/2) * sum(|r_i| * |r_{i-1}|)"""
    if len(returns) < 2:
        return 0.0
    abs_ret = np.abs(returns)
    return (np.pi / 2) * np.sum(abs_ret[1:] * abs_ret[:-1])


def prepare_h1_data(symbol: str = 'GC=F') -> pd.DataFrame:
    """
    Download 2y of hourly data for gold futures.
    yfinance limits H1 to ~730 days max with period='2y'.
    """
    ticker = yf.Ticker(symbol)
    # yfinance H1 data: max ~730 days
    df = ticker.history(period='2y', interval='1h')
    df.reset_index(inplace=True)
    
    if 'Datetime' in df.columns:
        df.rename(columns={'Datetime': 'Date'}, inplace=True)
    if 'index' in df.columns:
        df.rename(columns={'index': 'Date'}, inplace=True)
    
    # Ensure timezone-naive for consistency
    if hasattr(df['Date'].dtype, 'tz') and df['Date'].dt.tz is not None:
        df['Date'] = df['Date'].dt.tz_localize(None)
    
    # Add trading date column (date part of each H1 bar)
    df['trading_date'] = df['Date'].dt.date
    
    print(f"  Downloaded {len(df)} H1 bars for {symbol}")
    print(f"  Date range: {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")
    
    return df


def compute_daily_jump_measures(df: pd.DataFrame) -> dict:
    """
    For each trading day, compute RV, BPV, jump_ratio, and direction
    from the hourly returns within that day.
    
    Returns dict: {date: {'jump_ratio': float, 'direction': int, 'daily_close': float, 'daily_open': float}}
    """
    daily_measures = {}
    
    for date, group in df.groupby('trading_date'):
        if len(group) < 10:  # Need ≥10 bars for meaningful measures
            continue
        
        closes = group['Close'].values.astype(float)
        log_rets = np.diff(np.log(closes))
        
        if len(log_rets) < 2:
            continue
        
        rv = compute_rv(log_rets)
        bpv = compute_bpv(log_rets)
        
        jump = max(rv - bpv, 0)
        jump_ratio = jump / rv if rv > 0 else 0
        
        # Direction of the day's total move
        day_return = np.log(closes[-1] / closes[0])
        direction = 1 if day_return > 0 else -1
        
        daily_measures[date] = {
            'jump_ratio': jump_ratio,
            'direction': direction,
            'daily_close': closes[-1],
            'daily_open': closes[0],
            'daily_high': group['High'].max(),
            'daily_low': group['Low'].min(),
        }
    
    return daily_measures


def compute_daily_atr(daily_measures: dict, period: int = 14) -> dict:
    """
    Compute ATR from the daily OHLC data derived from H1 bars.
    Returns dict: {date: atr_value}
    """
    dates = sorted(daily_measures.keys())
    daily_atr = {}
    
    # Build arrays for TR calculation
    trs = []
    for i, date in enumerate(dates):
        m = daily_measures[date]
        high = m['daily_high']
        low = m['daily_low']
        
        if i > 0:
            prev_close = daily_measures[dates[i-1]]['daily_close']
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        else:
            tr = high - low
        
        trs.append(tr)
        
        if len(trs) >= period:
            daily_atr[date] = np.mean(trs[-period:])
    
    return daily_atr


def compute_trend_direction(daily_measures: dict, lookback: int = 20) -> dict:
    """
    Compute 20-day trend direction from daily closes.
    Returns dict: {date: trend_direction (+1 or -1 or 0)}
    """
    dates = sorted(daily_measures.keys())
    trend = {}
    
    for i, date in enumerate(dates):
        if i < lookback:
            trend[date] = 0
            continue
        
        current_close = daily_measures[date]['daily_close']
        past_close = daily_measures[dates[i - lookback]]['daily_close']
        mom = (current_close - past_close) / past_close
        
        if mom > 0.01:  # >1% trend
            trend[date] = 1
        elif mom < -0.01:
            trend[date] = -1
        else:
            trend[date] = 0
    
    return trend


def build_h1_signal_data(df: pd.DataFrame):
    """
    Pre-compute all daily-level measures and map them to H1 bars.
    Returns the df with added columns for signal generation.
    """
    df = df.copy()
    
    # Compute daily jump measures
    daily_measures = compute_daily_jump_measures(df)
    daily_atr = compute_daily_atr(daily_measures, period=14)
    daily_trend = compute_trend_direction(daily_measures, lookback=20)
    
    # Identify the LAST bar of each trading day
    df['is_last_bar_of_day'] = False
    for date, group in df.groupby('trading_date'):
        last_idx = group.index[-1]
        df.loc[last_idx, 'is_last_bar_of_day'] = True
    
    # Identify the FIRST bar of each trading day
    df['is_first_bar_of_day'] = False
    for date, group in df.groupby('trading_date'):
        first_idx = group.index[0]
        df.loc[first_idx, 'is_first_bar_of_day'] = True
    
    # Map daily measures to H1 bars
    df['jump_ratio'] = df['trading_date'].map(
        lambda d: daily_measures.get(d, {}).get('jump_ratio', np.nan))
    df['jump_direction'] = df['trading_date'].map(
        lambda d: daily_measures.get(d, {}).get('direction', 0))
    df['daily_atr'] = df['trading_date'].map(
        lambda d: daily_atr.get(d, np.nan))
    df['trend_20d'] = df['trading_date'].map(
        lambda d: daily_trend.get(d, 0))
    
    return df, daily_measures


def jump_h1_signal(df: pd.DataFrame, i: int) -> dict:
    """
    Jump detection signal on H1 bars.
    
    Signal fires ONLY at the LAST bar of each trading day.
    Engine will execute entry at next bar (= next day's first bar).
    
    - jump_ratio > 0.3: FADE the jump direction
    - jump_ratio < 0.1 + trend: RIDE the trend
    """
    row = df.iloc[i]
    
    # Only signal at end of day
    if not row.get('is_last_bar_of_day', False):
        return None
    
    jump_ratio = row.get('jump_ratio', np.nan)
    atr = row.get('daily_atr', np.nan)
    
    if pd.isna(jump_ratio) or pd.isna(atr) or atr <= 0:
        return None
    
    close = row['Close']
    jump_dir = row.get('jump_direction', 0)
    trend = row.get('trend_20d', 0)
    
    # ── FADE: High jump ratio ──
    if jump_ratio > 0.3:
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
    
    # ── TREND: Low jump ratio + trending ──
    if jump_ratio < 0.1 and trend != 0:
        if trend > 0:
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
    
    return None


# ─── Walk-forward with clean suppression ─────────────────────────────────

def test_signal(df, i, start_idx, signal_fn):
    """Suppress signals during training period."""
    if i < start_idx:
        return None
    return signal_fn(df, i)


def walk_forward_clean(df: pd.DataFrame, signal_fn, n_windows=6,
                       train_months=8, test_months=4):
    """
    Walk-forward with clean method: suppress training-period signals.
    """
    total_bars = len(df)
    window_months = train_months + test_months  # 12 months total per window
    
    # Estimate bars per month from data
    date_range = (df['Date'].iloc[-1] - df['Date'].iloc[0]).total_seconds() / (30 * 24 * 3600)
    bars_per_month = total_bars / date_range if date_range > 0 else total_bars / 24
    
    train_bars = int(train_months * bars_per_month)
    test_bars = int(test_months * bars_per_month)
    step = test_bars  # Step by test_bars for rolling windows
    
    print(f"  WF: {total_bars} total bars, ~{bars_per_month:.0f} bars/month")
    print(f"  WF: train={train_bars} bars, test={test_bars} bars, step={step}")
    
    results = []
    
    for w in range(n_windows):
        start = w * step
        train_end = start + train_bars
        test_end = train_end + test_bars
        
        if test_end > total_bars:
            break
        
        # Use full window (train+test), but suppress signals before train_end
        window_df = df.iloc[start:test_end].reset_index(drop=True)
        test_start_idx = train_bars  # Index within window_df where test begins
        
        # Create wrapped signal fn that suppresses training period
        def make_signal_fn(start_idx):
            def wrapped(df_inner, i):
                return test_signal(df_inner, i, start_idx, signal_fn)
            return wrapped
        
        engine = BacktestEngine(BacktestConfig())
        wrapped_fn = make_signal_fn(test_start_idx)
        trades_df = engine.run(window_df, wrapped_fn)
        
        # Calculate metrics on test period data only
        test_price_df = window_df.iloc[test_start_idx:].reset_index(drop=True)
        metrics = calculate_metrics(trades_df, test_price_df)
        
        # Get date range for reporting
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
    print("=" * 70)
    print("  JUMP DETECTION ON GOLD H1 (Hourly)")
    print("=" * 70)
    
    # 1. Download data
    print("\n[1] Downloading GC=F hourly data...")
    df = prepare_h1_data('GC=F')
    
    # 2. Build signal data (daily measures mapped to H1)
    print("\n[2] Computing daily jump measures from H1 bars...")
    df, daily_measures = build_h1_signal_data(df)
    
    n_days = len(daily_measures)
    jump_days = sum(1 for d in daily_measures.values() if d['jump_ratio'] > 0.3)
    smooth_days = sum(1 for d in daily_measures.values() if d['jump_ratio'] < 0.1)
    print(f"  Total trading days: {n_days}")
    print(f"  Jump days (ratio > 0.3): {jump_days} ({jump_days/n_days*100:.1f}%)")
    print(f"  Smooth days (ratio < 0.1): {smooth_days} ({smooth_days/n_days*100:.1f}%)")
    
    # 3. Full backtest (no WF suppression)
    print("\n[3] Full-period backtest...")
    engine = BacktestEngine(BacktestConfig())
    trades_df = engine.run(df, jump_h1_signal)
    full_metrics = calculate_metrics(trades_df, df)
    
    print(f"  Trades: {full_metrics['total_trades']}")
    print(f"  Win Rate: {full_metrics['win_rate']}%")
    print(f"  Profit Factor: {full_metrics['profit_factor']}")
    print(f"  Strategy Return: {full_metrics['strategy_return']}%")
    print(f"  Buy&Hold Return: {full_metrics['buy_hold_return']}%")
    print(f"  Max Drawdown: {full_metrics['max_drawdown']}%")
    print(f"  Sharpe: {full_metrics['sharpe']}")
    
    # 4. Walk-forward
    print("\n[4] Walk-Forward (6 windows, 8m train / 4m test)...")
    wf_results = walk_forward_clean(df, jump_h1_signal, n_windows=6,
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
    
    # 5. Save results
    results = {
        'GC=F': {
            'timeframe': 'H1',
            'full_backtest': full_metrics,
            'walk_forward': wf_results,
            'profitable_windows': profitable_windows,
            'total_windows': len(wf_results),
            'data_bars': len(df),
            'trading_days': n_days,
            'jump_days': jump_days,
            'smooth_days': smooth_days,
        }
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
    
    with open('results_jump_gold_h1.json', 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    
    print(f"\n  Results saved to results_jump_gold_h1.json")
    return results


if __name__ == '__main__':
    main()
