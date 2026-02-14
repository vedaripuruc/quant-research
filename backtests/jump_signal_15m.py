"""
Bipower Variation Jump Detection on EURUSD 15-minute bars
=========================================================
Proper implementation using Barndorff-Nielsen & Shephard (2004):
- Compute RV and BPV from INTRADAY 15m returns within each trading day
- Signal at end of day → entry at first bar of NEXT day (no look-ahead)
- Walk-forward validation with 6 windows (8m train / 4m test)
- Exit via SL/TP checked on 15m bars (intrabar High/Low simulation)

Uses raw tick-aggregated 15m data from tickdata/EURUSD_15M_2Y.csv
"""

import numpy as np
import pandas as pd
import json
import sys
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, List, Literal


# ─── Configuration ──────────────────────────────────────────────────────

SLIPPAGE_PCT = 0.0005    # 0.05%
COMMISSION_PCT = 0.001   # 0.1% round trip
JUMP_THRESHOLD = 0.3     # Jump ratio > this → fade
SMOOTH_THRESHOLD = 0.1   # Jump ratio < this → trend
MOMENTUM_WINDOW = 20     # Days for momentum calculation
MOMENTUM_MIN = 0.005     # Min 0.5% momentum over 20 days (forex is low vol)
ATR_PERIOD = 14          # Days for ATR calculation
MIN_BARS_PER_DAY = 20    # Skip days with fewer bars (holidays/half-days)


# ─── Data Loading ───────────────────────────────────────────────────────

def load_15m_data(path: str = "tickdata/EURUSD_15M_2Y.csv") -> pd.DataFrame:
    """Load 15-minute OHLC data."""
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df.rename(columns={'timestamp': 'Date'}, inplace=True)
    df['date'] = df['Date'].dt.date
    print(f"Loaded {len(df)} 15m bars from {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")
    return df


def build_daily_ohlc(df_15m: pd.DataFrame) -> pd.DataFrame:
    """Build daily OHLC from 15m bars (for ATR computation)."""
    daily = df_15m.groupby('date').agg(
        Open=('open', 'first'),
        High=('high', 'max'),
        Low=('low', 'min'),
        Close=('close', 'last'),
        Volume=('volume', 'sum'),
        Bars=('ticks', 'count'),
    ).reset_index()
    daily.rename(columns={'date': 'Date'}, inplace=True)
    daily['Date'] = pd.to_datetime(daily['Date'])
    return daily


# ─── Jump Detection (per-day from intraday returns) ────────────────────

def compute_daily_jump_stats(df_15m: pd.DataFrame) -> pd.DataFrame:
    """
    For each trading day, compute RV, BPV, jump ratio from 15m log-returns.
    Also compute the day's direction (close vs open) for fade signals.
    
    Returns DataFrame indexed by date with columns:
    rv, bpv, jump, jump_ratio, day_return, n_bars
    """
    results = []
    
    for date, group in df_15m.groupby('date'):
        group = group.sort_values('Date')
        closes = group['close'].values.astype(float)
        
        if len(closes) < MIN_BARS_PER_DAY:
            # Half-day or holiday — skip (not enough bars for reliable RV/BPV)
            results.append({
                'date': date,
                'rv': np.nan,
                'bpv': np.nan,
                'jump': np.nan,
                'jump_ratio': np.nan,
                'day_return': 0.0,
                'day_open': closes[0],
                'day_close': closes[-1],
                'n_bars': len(closes),
                'skip': True,
            })
            continue
        
        # Log-returns from consecutive 15m closes
        log_returns = np.diff(np.log(closes))
        
        if len(log_returns) < 2:
            results.append({
                'date': date,
                'rv': np.nan, 'bpv': np.nan, 'jump': np.nan,
                'jump_ratio': np.nan, 'day_return': 0.0,
                'day_open': closes[0], 'day_close': closes[-1],
                'n_bars': len(closes), 'skip': True,
            })
            continue
        
        # Realized Variance: sum of squared 15m returns
        rv = np.sum(log_returns ** 2)
        
        # Bipower Variation: (π/2) × Σ|r_i| × |r_{i-1}|
        abs_ret = np.abs(log_returns)
        bpv = (np.pi / 2) * np.sum(abs_ret[1:] * abs_ret[:-1])
        
        # Jump component
        jump = max(rv - bpv, 0)
        jump_ratio = jump / rv if rv > 1e-20 else 0
        
        # Day return: log(close/open)
        day_return = np.log(closes[-1] / closes[0])
        
        results.append({
            'date': date,
            'rv': rv,
            'bpv': bpv,
            'jump': jump,
            'jump_ratio': jump_ratio,
            'day_return': day_return,
            'day_open': closes[0],
            'day_close': closes[-1],
            'n_bars': len(closes),
            'skip': False,
        })
    
    return pd.DataFrame(results)


def compute_daily_atr(daily_df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    """Compute ATR from daily OHLC."""
    high = daily_df['High'].values
    low = daily_df['Low'].values
    close = daily_df['Close'].values
    
    tr = np.zeros(len(daily_df))
    tr[0] = high[0] - low[0]
    for i in range(1, len(daily_df)):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    
    atr = pd.Series(tr).rolling(period).mean()
    return atr.values


# ─── Signal Generation ──────────────────────────────────────────────────

def generate_signals(jump_stats: pd.DataFrame, daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate trading signals based on daily jump stats.
    Signal at end of day D → trade on day D+1.
    
    Returns DataFrame with columns: signal_date, trade_date, direction, trade_type, atr
    """
    # Compute ATR on daily data
    atr_values = compute_daily_atr(daily_df, ATR_PERIOD)
    daily_df = daily_df.copy()
    daily_df['atr'] = atr_values
    
    # Merge jump stats with daily data
    jump_stats = jump_stats.copy()
    jump_stats['Date'] = pd.to_datetime(jump_stats['date'])
    
    # Create date lookup for ATR
    atr_lookup = dict(zip(daily_df['Date'].dt.date, daily_df['atr']))
    
    # 20-day momentum: rolling close changes
    daily_closes = dict(zip(daily_df['Date'].dt.date, daily_df['Close']))
    daily_dates = sorted(daily_closes.keys())
    
    signals = []
    
    for idx, row in jump_stats.iterrows():
        if row['skip'] or pd.isna(row['jump_ratio']):
            continue
        
        signal_date = row['date']
        atr = atr_lookup.get(signal_date)
        
        if atr is None or pd.isna(atr) or atr <= 0:
            continue
        
        # Find next trading day
        date_idx = daily_dates.index(signal_date) if signal_date in daily_dates else -1
        if date_idx < 0 or date_idx + 1 >= len(daily_dates):
            continue
        trade_date = daily_dates[date_idx + 1]
        
        jump_ratio = row['jump_ratio']
        
        # ── FADE: Jump detected ──
        if jump_ratio > JUMP_THRESHOLD:
            day_ret = row['day_return']
            if abs(day_ret) < 1e-8:
                continue  # No clear direction to fade
            
            # Fade the jump direction
            if day_ret > 0:
                direction = 'short'  # Jump was up, fade down
            else:
                direction = 'long'   # Jump was down, fade up
            
            signals.append({
                'signal_date': signal_date,
                'trade_date': trade_date,
                'direction': direction,
                'trade_type': 'fade',
                'atr': atr,
                'sl_mult': 1.0,
                'tp_mult': 2.0,
                'jump_ratio': jump_ratio,
            })
        
        # ── TREND: Smooth regime with clear momentum ──
        elif jump_ratio < SMOOTH_THRESHOLD:
            # 20-day momentum
            if date_idx < MOMENTUM_WINDOW:
                continue
            mom_date = daily_dates[date_idx - MOMENTUM_WINDOW]
            mom_close = daily_closes.get(mom_date)
            current_close = daily_closes.get(signal_date)
            
            if mom_close is None or current_close is None or mom_close <= 0:
                continue
            
            momentum = (current_close - mom_close) / mom_close
            
            if abs(momentum) < MOMENTUM_MIN:
                continue  # Not enough momentum
            
            if momentum > 0:
                direction = 'long'
            else:
                direction = 'short'
            
            signals.append({
                'signal_date': signal_date,
                'trade_date': trade_date,
                'direction': direction,
                'trade_type': 'trend',
                'atr': atr,
                'sl_mult': 2.0,
                'tp_mult': 4.0,
                'jump_ratio': jump_ratio,
            })
    
    return pd.DataFrame(signals)


# ─── Trade Simulation on 15m Bars ───────────────────────────────────────

@dataclass
class Trade:
    entry_time: object
    entry_price: float
    direction: str
    stop_loss: float
    take_profit: float
    trade_type: str
    signal_date: object
    exit_time: object = None
    exit_price: float = None
    result: str = None
    exit_reason: str = None
    pnl_pct: float = None

    def close(self, exit_time, exit_price, reason):
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.exit_reason = reason
        
        if self.direction == 'long':
            self.pnl_pct = (exit_price - self.entry_price) / self.entry_price
        else:
            self.pnl_pct = (self.entry_price - exit_price) / self.entry_price
        
        # Apply commission
        self.pnl_pct -= COMMISSION_PCT
        
        if self.pnl_pct > 0.0001:
            self.result = 'win'
        elif self.pnl_pct < -0.0001:
            self.result = 'loss'
        else:
            self.result = 'breakeven'


def simulate_trades(df_15m: pd.DataFrame, signals_df: pd.DataFrame) -> List[Trade]:
    """
    Simulate trades on 15m bars.
    
    For each signal:
    - Entry at the OPEN of the first 15m bar of trade_date
    - SL/TP set based on ATR
    - Check High/Low of each 15m bar for SL/TP hit
    - Only one trade at a time
    """
    trades = []
    current_trade: Optional[Trade] = None
    
    # Build a lookup: trade_date → signal info
    signal_lookup = {}
    for _, sig in signals_df.iterrows():
        td = sig['trade_date']
        if td not in signal_lookup:  # First signal per day wins
            signal_lookup[td] = sig
    
    # Track which dates have a pending signal
    pending_signal = None
    entered_today = False
    current_day = None
    
    for i, bar in df_15m.iterrows():
        bar_date = bar['Date'].date() if hasattr(bar['Date'], 'date') else bar['date']
        bar_time = bar['Date']
        
        # New day?
        if bar_date != current_day:
            current_day = bar_date
            entered_today = False
            
            # Check if we have a signal for today
            if bar_date in signal_lookup and current_trade is None:
                pending_signal = signal_lookup[bar_date]
        
        # Enter on first bar of the day if we have a pending signal
        if pending_signal is not None and current_trade is None and not entered_today:
            sig = pending_signal
            pending_signal = None
            entered_today = True
            
            entry_price = bar['open']
            
            # Apply slippage
            if sig['direction'] == 'long':
                entry_price *= (1 + SLIPPAGE_PCT)
            else:
                entry_price *= (1 - SLIPPAGE_PCT)
            
            # Set SL/TP based on ATR
            atr = sig['atr']
            if sig['direction'] == 'long':
                sl = entry_price - sig['sl_mult'] * atr
                tp = entry_price + sig['tp_mult'] * atr
            else:
                sl = entry_price + sig['sl_mult'] * atr
                tp = entry_price - sig['tp_mult'] * atr
            
            current_trade = Trade(
                entry_time=bar_time,
                entry_price=entry_price,
                direction=sig['direction'],
                stop_loss=sl,
                take_profit=tp,
                trade_type=sig['trade_type'],
                signal_date=sig['signal_date'],
            )
        
        # Check exits on current bar
        if current_trade is not None:
            high = bar['high']
            low = bar['low']
            
            if current_trade.direction == 'long':
                # SL first (worst case)
                if low <= current_trade.stop_loss:
                    exit_price = current_trade.stop_loss * (1 - SLIPPAGE_PCT)
                    current_trade.close(bar_time, exit_price, 'sl')
                    trades.append(current_trade)
                    current_trade = None
                elif high >= current_trade.take_profit:
                    exit_price = current_trade.take_profit * (1 - SLIPPAGE_PCT)
                    current_trade.close(bar_time, exit_price, 'tp')
                    trades.append(current_trade)
                    current_trade = None
            else:  # short
                if high >= current_trade.stop_loss:
                    exit_price = current_trade.stop_loss * (1 + SLIPPAGE_PCT)
                    current_trade.close(bar_time, exit_price, 'sl')
                    trades.append(current_trade)
                    current_trade = None
                elif low <= current_trade.take_profit:
                    exit_price = current_trade.take_profit * (1 + SLIPPAGE_PCT)
                    current_trade.close(bar_time, exit_price, 'tp')
                    trades.append(current_trade)
                    current_trade = None
    
    # Close any remaining position at end
    if current_trade is not None:
        last = df_15m.iloc[-1]
        current_trade.close(last['Date'], last['close'], 'eod')
        trades.append(current_trade)
    
    return trades


# ─── Metrics ────────────────────────────────────────────────────────────

def compute_metrics(trades: List[Trade], df_15m: pd.DataFrame) -> dict:
    """Compute comprehensive backtest metrics."""
    if not trades:
        first_close = df_15m['close'].iloc[0]
        last_close = df_15m['close'].iloc[-1]
        bnh = (last_close / first_close - 1) * 100
        return {
            'total_trades': 0, 'wins': 0, 'losses': 0,
            'win_rate': 0, 'profit_factor': 0,
            'strategy_return': 0, 'buy_hold_return': round(bnh, 2),
            'max_drawdown': 0, 'avg_win': 0, 'avg_loss': 0,
            'expectancy': 0, 'sharpe': 0,
            'fade_trades': 0, 'trend_trades': 0,
            'fade_wr': 0, 'trend_wr': 0,
        }
    
    pnls = [t.pnl_pct for t in trades]
    wins = [t for t in trades if t.result == 'win']
    losses = [t for t in trades if t.result == 'loss']
    fade_trades = [t for t in trades if t.trade_type == 'fade']
    trend_trades = [t for t in trades if t.trade_type == 'trend']
    
    total = len(trades)
    win_rate = len(wins) / total * 100 if total > 0 else 0
    
    total_profit = sum(t.pnl_pct for t in wins) if wins else 0
    total_loss = abs(sum(t.pnl_pct for t in losses)) if losses else 0
    profit_factor = total_profit / total_loss if total_loss > 0 else (999 if total_profit > 0 else 0)
    
    avg_win = np.mean([t.pnl_pct for t in wins]) * 100 if wins else 0
    avg_loss = np.mean([t.pnl_pct for t in losses]) * 100 if losses else 0
    
    # Compounded return
    equity = np.cumprod([1 + p for p in pnls])
    strategy_return = (equity[-1] - 1) * 100
    
    # Max drawdown
    peak = np.maximum.accumulate(equity)
    dd = (equity / peak - 1) * 100
    max_drawdown = np.min(dd)
    
    # Buy & hold
    first_close = df_15m['close'].iloc[0]
    last_close = df_15m['close'].iloc[-1]
    bnh = (last_close / first_close - 1) * 100
    
    # Expectancy
    expectancy = np.mean(pnls) * 100
    
    # Sharpe (annualize based on avg trade duration, approximate as daily)
    if len(pnls) > 1 and np.std(pnls) > 0:
        # Estimate trades per year
        first_entry = trades[0].entry_time
        last_entry = trades[-1].entry_time
        if hasattr(first_entry, 'timestamp'):
            days_span = (last_entry - first_entry).days
        else:
            days_span = 365  # fallback
        if days_span > 0:
            trades_per_year = len(trades) / days_span * 252
        else:
            trades_per_year = 252
        sharpe = np.sqrt(trades_per_year) * np.mean(pnls) / np.std(pnls)
    else:
        sharpe = 0
    
    # Per-type stats
    fade_wins = [t for t in fade_trades if t.result == 'win']
    trend_wins = [t for t in trend_trades if t.result == 'win']
    fade_wr = len(fade_wins) / len(fade_trades) * 100 if fade_trades else 0
    trend_wr = len(trend_wins) / len(trend_trades) * 100 if trend_trades else 0
    
    return {
        'total_trades': total,
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': round(win_rate, 1),
        'profit_factor': round(profit_factor, 2),
        'strategy_return': round(strategy_return, 2),
        'buy_hold_return': round(bnh, 2),
        'max_drawdown': round(max_drawdown, 2),
        'avg_win': round(avg_win, 3),
        'avg_loss': round(avg_loss, 3),
        'expectancy': round(expectancy, 3),
        'sharpe': round(sharpe, 2),
        'fade_trades': len(fade_trades),
        'trend_trades': len(trend_trades),
        'fade_wr': round(fade_wr, 1),
        'trend_wr': round(trend_wr, 1),
    }


# ─── Walk-Forward ───────────────────────────────────────────────────────

def walk_forward(df_15m: pd.DataFrame, n_windows: int = 6,
                 train_months: int = 8, test_months: int = 4) -> list:
    """
    Walk-forward validation with rolling windows.
    
    Each window:
    - Train: 8 months (signals are computed but trades suppressed)
    - Test: 4 months (trades executed)
    
    The key: we compute jump stats on ALL data up to the test period,
    but only ENTER trades during the test period.
    """
    total_days = (df_15m['Date'].iloc[-1] - df_15m['Date'].iloc[0]).days
    
    # Get unique dates
    all_dates = sorted(df_15m['date'].unique())
    start_date = pd.Timestamp(all_dates[0])
    end_date = pd.Timestamp(all_dates[-1])
    
    results = []
    
    for w in range(n_windows):
        window_start = start_date + pd.DateOffset(months=w * test_months)
        train_end = window_start + pd.DateOffset(months=train_months)
        test_end = train_end + pd.DateOffset(months=test_months)
        
        if test_end > end_date + pd.DateOffset(days=30):
            break
        
        # Get 15m data for the full window (train + test)
        mask = (df_15m['Date'] >= window_start) & (df_15m['Date'] < test_end)
        window_df = df_15m[mask].copy().reset_index(drop=True)
        
        if len(window_df) < 1000:
            continue
        
        # Build daily OHLC for the window
        daily_df = build_daily_ohlc(window_df)
        
        # Compute jump stats for all days in window
        jump_stats = compute_daily_jump_stats(window_df)
        
        # Generate ALL signals
        all_signals = generate_signals(jump_stats, daily_df)
        
        if all_signals.empty:
            results.append({
                'window': w + 1,
                'train_start': window_start.strftime('%Y-%m-%d'),
                'train_end': train_end.strftime('%Y-%m-%d'),
                'test_start': train_end.strftime('%Y-%m-%d'),
                'test_end': test_end.strftime('%Y-%m-%d'),
                'total_trades': 0, 'win_rate': 0, 'profit_factor': 0,
                'strategy_return': 0, 'max_drawdown': 0,
                'fade_trades': 0, 'trend_trades': 0,
            })
            continue
        
        # SUPPRESS signals during training period — only keep test period signals
        train_end_date = train_end.date()
        test_signals = all_signals[all_signals['trade_date'] >= train_end_date].copy()
        
        if test_signals.empty:
            results.append({
                'window': w + 1,
                'train_start': window_start.strftime('%Y-%m-%d'),
                'train_end': train_end.strftime('%Y-%m-%d'),
                'test_start': train_end.strftime('%Y-%m-%d'),
                'test_end': test_end.strftime('%Y-%m-%d'),
                'total_trades': 0, 'win_rate': 0, 'profit_factor': 0,
                'strategy_return': 0, 'max_drawdown': 0,
                'fade_trades': 0, 'trend_trades': 0,
            })
            continue
        
        # Get test period 15m data for simulation
        test_mask = (window_df['Date'] >= train_end)
        test_15m = window_df[test_mask].copy().reset_index(drop=True)
        
        # Simulate trades only on test period
        trades = simulate_trades(test_15m, test_signals)
        
        # Also get just the test daily data for B&H
        test_daily_mask = (daily_df['Date'] >= train_end) & (daily_df['Date'] < test_end)
        test_daily = daily_df[test_daily_mask].copy()
        
        metrics = compute_metrics(trades, test_15m)
        
        results.append({
            'window': w + 1,
            'train_start': window_start.strftime('%Y-%m-%d'),
            'train_end': train_end.strftime('%Y-%m-%d'),
            'test_start': train_end.strftime('%Y-%m-%d'),
            'test_end': test_end.strftime('%Y-%m-%d'),
            **metrics,
        })
    
    return results


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  BIPOWER VARIATION JUMP DETECTION — EURUSD 15M")
    print("  Barndorff-Nielsen & Shephard (2004) proper implementation")
    print("=" * 70)
    
    # Load data
    df_15m = load_15m_data()
    daily_df = build_daily_ohlc(df_15m)
    
    print(f"Daily bars: {len(daily_df)}")
    print(f"Date range: {daily_df['Date'].iloc[0].date()} to {daily_df['Date'].iloc[-1].date()}")
    
    # ── Compute jump stats ──
    print("\n📊 Computing daily jump statistics from intraday returns...")
    jump_stats = compute_daily_jump_stats(df_15m)
    valid_stats = jump_stats[~jump_stats['skip']]
    
    print(f"  Total days: {len(jump_stats)}")
    print(f"  Valid days (≥{MIN_BARS_PER_DAY} bars): {len(valid_stats)}")
    print(f"  Skipped days: {jump_stats['skip'].sum()}")
    
    # Stats on jump ratios
    ratios = valid_stats['jump_ratio']
    print(f"\n  Jump ratio distribution:")
    print(f"    Mean: {ratios.mean():.4f}")
    print(f"    Median: {ratios.median():.4f}")
    print(f"    Std: {ratios.std():.4f}")
    print(f"    >0.3 (jump days): {(ratios > JUMP_THRESHOLD).sum()} ({(ratios > JUMP_THRESHOLD).mean()*100:.1f}%)")
    print(f"    <0.1 (smooth days): {(ratios < SMOOTH_THRESHOLD).sum()} ({(ratios < SMOOTH_THRESHOLD).mean()*100:.1f}%)")
    
    # ── Generate signals ──
    print("\n🎯 Generating trading signals...")
    signals = generate_signals(jump_stats, daily_df)
    
    if signals.empty:
        print("  ❌ No signals generated! Check thresholds.")
        return
    
    fade_sigs = signals[signals['trade_type'] == 'fade']
    trend_sigs = signals[signals['trade_type'] == 'trend']
    print(f"  Total signals: {len(signals)}")
    print(f"    Fade signals: {len(fade_sigs)}")
    print(f"    Trend signals: {len(trend_sigs)}")
    
    # ── Full-period backtest ──
    print("\n💰 Running full-period backtest...")
    trades = simulate_trades(df_15m, signals)
    metrics = compute_metrics(trades, df_15m)
    
    print(f"\n  FULL PERIOD RESULTS:")
    print(f"  {'─'*50}")
    print(f"  Total trades:     {metrics['total_trades']}")
    print(f"    Fade trades:    {metrics['fade_trades']} (WR: {metrics['fade_wr']}%)")
    print(f"    Trend trades:   {metrics['trend_trades']} (WR: {metrics['trend_wr']}%)")
    print(f"  Win rate:         {metrics['win_rate']}%")
    print(f"  Profit factor:    {metrics['profit_factor']}")
    print(f"  Strategy return:  {metrics['strategy_return']}%")
    print(f"  Buy & hold:       {metrics['buy_hold_return']}%")
    print(f"  Max drawdown:     {metrics['max_drawdown']}%")
    print(f"  Avg win:          {metrics['avg_win']}%")
    print(f"  Avg loss:         {metrics['avg_loss']}%")
    print(f"  Expectancy:       {metrics['expectancy']}%")
    print(f"  Sharpe:           {metrics['sharpe']}")
    
    # ── Walk-forward ──
    print(f"\n📈 Walk-Forward Validation (6 windows, 8m train / 4m test):")
    print(f"  {'─'*70}")
    
    wf_results = walk_forward(df_15m)
    
    profitable_windows = 0
    total_wf_return = 0
    
    for w in wf_results:
        ret = w.get('strategy_return', 0)
        total_wf_return += ret
        status = "✓" if ret > 0 else "✗"
        if ret > 0:
            profitable_windows += 1
        
        print(f"  Window {w['window']}: {w['test_start']} → {w['test_end']} | "
              f"{ret:+.2f}% | WR={w.get('win_rate', 0):.0f}% | "
              f"PF={w.get('profit_factor', 0):.2f} | "
              f"Trades={w.get('total_trades', 0)} "
              f"(F:{w.get('fade_trades', 0)} T:{w.get('trend_trades', 0)}) {status}")
    
    print(f"\n  OOS Profitable:    {profitable_windows}/{len(wf_results)}")
    print(f"  Cumulative OOS:    {total_wf_return:+.2f}%")
    
    edge = profitable_windows > len(wf_results) / 2
    print(f"  Edge survived?     {'✅ YES' if edge else '❌ NO'}")
    
    # ── Save results ──
    all_results = {
        'full_period': metrics,
        'walk_forward': wf_results,
        'config': {
            'slippage_pct': SLIPPAGE_PCT,
            'commission_pct': COMMISSION_PCT,
            'jump_threshold': JUMP_THRESHOLD,
            'smooth_threshold': SMOOTH_THRESHOLD,
            'momentum_window': MOMENTUM_WINDOW,
            'momentum_min': MOMENTUM_MIN,
            'atr_period': ATR_PERIOD,
            'min_bars_per_day': MIN_BARS_PER_DAY,
        },
        'jump_stats_summary': {
            'total_days': len(jump_stats),
            'valid_days': int(len(valid_stats)),
            'jump_days': int((ratios > JUMP_THRESHOLD).sum()),
            'smooth_days': int((ratios < SMOOTH_THRESHOLD).sum()),
            'mean_jump_ratio': round(float(ratios.mean()), 4),
            'median_jump_ratio': round(float(ratios.median()), 4),
        },
        'signal_counts': {
            'total': len(signals),
            'fade': len(fade_sigs),
            'trend': len(trend_sigs),
        },
    }
    
    with open('results_jump_15m.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n✅ Saved results_jump_15m.json")
    
    # ── Write report ──
    write_report(all_results, wf_results, metrics, jump_stats, signals)
    print(f"✅ Saved report_jump_15m.md")


def write_report(all_results, wf_results, metrics, jump_stats, signals):
    """Write markdown report."""
    valid = jump_stats[~jump_stats['skip']]
    ratios = valid['jump_ratio']
    
    report = f"""# Bipower Variation Jump Detection — EURUSD 15M
## Barndorff-Nielsen & Shephard (2004) Implementation

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  
**Data:** EURUSD tick data aggregated to 15-minute bars  
**Period:** 2024-01-02 to 2026-02-04 ({len(jump_stats)} trading days, 28,443 bars)  
**Methodology:** Realized Variance vs Bipower Variation from intraday 15m returns  

---

## Methodology

### Theory
Unlike the daily version (which computed RV/BPV from rolling windows of daily returns), this implementation uses the **theoretically correct** approach:

1. For each trading day, collect all 15-minute log-returns (~60-96 per day)
2. **Realized Variance (RV):** Σ r²ᵢ — total variance including jumps
3. **Bipower Variation (BPV):** (π/2) × Σ|rᵢ||rᵢ₋₁| — robust to jumps, captures continuous variation
4. **Jump component:** J = max(RV - BPV, 0)
5. **Jump ratio:** J / RV — fraction of daily variance attributable to jumps

### Signal Logic
- **FADE (jump_ratio > {JUMP_THRESHOLD}):** Jump day detected → next day, trade AGAINST jump direction (mean reversion)
  - SL = 1×ATR, TP = 2×ATR (tight stops, expecting reversion)
- **TREND (jump_ratio < {SMOOTH_THRESHOLD} + 20-day momentum > {MOMENTUM_MIN*100}%):** Smooth trending regime → ride the trend
  - SL = 2×ATR, TP = 4×ATR (wide stops, letting trends develop)
- Entry: First 15m bar of next trading day (no look-ahead bias)
- Exit: SL/TP checked on 15m High/Low (intrabar simulation)
- Costs: {SLIPPAGE_PCT*100}% slippage + {COMMISSION_PCT*100}% commission round trip

---

## Jump Statistics

| Metric | Value |
|--------|-------|
| Total trading days | {len(jump_stats)} |
| Valid days (≥{MIN_BARS_PER_DAY} bars) | {len(valid)} |
| Mean jump ratio | {ratios.mean():.4f} |
| Median jump ratio | {ratios.median():.4f} |
| Jump days (ratio > {JUMP_THRESHOLD}) | {(ratios > JUMP_THRESHOLD).sum()} ({(ratios > JUMP_THRESHOLD).mean()*100:.1f}%) |
| Smooth days (ratio < {SMOOTH_THRESHOLD}) | {(ratios < SMOOTH_THRESHOLD).sum()} ({(ratios < SMOOTH_THRESHOLD).mean()*100:.1f}%) |

Signals generated: **{len(signals)}** ({len(signals[signals['trade_type']=='fade'])} fade, {len(signals[signals['trade_type']=='trend'])} trend)

---

## Full Period Results

| Metric | Value |
|--------|-------|
| Total Trades | {metrics['total_trades']} |
| Win Rate | {metrics['win_rate']}% |
| Profit Factor | {metrics['profit_factor']} |
| Strategy Return | {metrics['strategy_return']}% |
| Buy & Hold Return | {metrics['buy_hold_return']}% |
| Max Drawdown | {metrics['max_drawdown']}% |
| Avg Win | {metrics['avg_win']}% |
| Avg Loss | {metrics['avg_loss']}% |
| Expectancy | {metrics['expectancy']}% |
| Sharpe Ratio | {metrics['sharpe']} |

### By Trade Type
| Type | Trades | Win Rate |
|------|--------|----------|
| Fade | {metrics['fade_trades']} | {metrics['fade_wr']}% |
| Trend | {metrics['trend_trades']} | {metrics['trend_wr']}% |

---

## Walk-Forward Validation

6 rolling windows: 8 months training (signals suppressed) + 4 months out-of-sample testing.

| Window | Test Period | Return | WR | PF | Trades | Fade | Trend |
|--------|-------------|--------|-----|-----|--------|------|-------|
"""
    
    profitable_windows = 0
    total_oos_return = 0
    for w in wf_results:
        ret = w.get('strategy_return', 0)
        total_oos_return += ret
        if ret > 0:
            profitable_windows += 1
        status = "✓" if ret > 0 else "✗"
        report += (f"| {w['window']} | {w['test_start']} → {w['test_end']} | "
                   f"{ret:+.2f}% | {w.get('win_rate', 0):.0f}% | "
                   f"{w.get('profit_factor', 0):.2f} | {w.get('total_trades', 0)} | "
                   f"{w.get('fade_trades', 0)} | {w.get('trend_trades', 0)} | {status} |\n")
    
    report += f"""
**OOS Profitable Windows:** {profitable_windows}/{len(wf_results)}  
**Cumulative OOS Return:** {total_oos_return:+.2f}%  

---

## Comparison

| Version | Asset | PF | Return | Max DD | Method |
|---------|-------|----|--------|--------|--------|
| Daily (original) | Gold (GC=F) | 1.95 | +15.2% | -4.5% | Rolling window of daily returns |
| **15M (this)** | **EURUSD** | **{metrics['profit_factor']}** | **{metrics['strategy_return']}%** | **{metrics['max_drawdown']}%** | **Intraday BPV from 15m returns** |

The daily version used a rolling window of daily returns (20 bars) to approximate RV/BPV — a convenient but theoretically incorrect shortcut. This version computes RV and BPV from **actual intraday returns** within each trading day, which is the proper Barndorff-Nielsen & Shephard approach.

---

## Honest Assessment

"""
    
    if metrics['profit_factor'] > 1.0 and profitable_windows > len(wf_results) / 2:
        report += """The signal shows a positive edge that **survives walk-forward validation**. 
The intraday BPV calculation provides statistically meaningful jump detection, 
and the mean-reversion after jumps appears to be a genuine microstructure effect in EURUSD.

However, caveats:
- EURUSD is the most liquid pair — spreads and slippage may be lower than modeled
- The edge is modest — transaction costs eat significantly into profits
- Walk-forward windows may overlap in training data
"""
    elif metrics['profit_factor'] > 1.0:
        report += """The full-period backtest shows a positive profit factor, but **walk-forward validation is mixed**. 
This suggests potential overfitting to the full dataset or parameter sensitivity.

The edge may be real but fragile — it doesn't consistently survive out-of-sample testing.
"""
    else:
        report += """**The strategy does NOT show a tradeable edge on EURUSD with 15-minute bars.**

Despite the theoretically sound methodology (proper intraday BPV), the signal fails 
to generate consistent profits after accounting for slippage and commissions.

Possible reasons:
- EURUSD is extremely efficient — jumps are priced in quickly
- Mean reversion after jumps may not be strong enough in forex to overcome costs
- The daily signal on Gold (PF 1.95) may have worked due to commodity-specific dynamics
- 15-minute frequency may be too granular for a daily-signal strategy (entry timing noise)

**This is an honest result.** Not every theoretically sound signal is profitable.
"""
    
    with open('report_jump_15m.md', 'w') as f:
        f.write(report)


if __name__ == '__main__':
    main()
