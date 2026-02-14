"""
Forex-Optimized Strategies
--------------------------
Adjusted parameters for lower volatility forex markets.
Key changes:
- Tighter SL/TP ratios (0.75x / 2x ATR instead of 1.5x / 3x)
- Lower commission assumption (0.03% for forex vs 0.1% for stocks)
- Or: Use fixed pip-based exits
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict


def williams_r_forex(df: pd.DataFrame, i: int,
                     period: int = 14,
                     oversold: float = -80,
                     overbought: float = -20,
                     atr_sl_mult: float = 0.75,  # Tighter for forex
                     atr_tp_mult: float = 2.0) -> Optional[Dict]:
    """Williams %R for forex - tighter stops"""
    if i < period + 1:
        return None
    
    window = df.iloc[:i+1]
    
    highest_high = window['High'].rolling(window=period).max()
    lowest_low = window['Low'].rolling(window=period).min()
    wr = -100 * ((highest_high - window['Close']) / (highest_high - lowest_low))
    
    current_wr = wr.iloc[-1]
    prev_wr = wr.iloc[-2]
    
    tr = pd.concat([
        window['High'] - window['Low'],
        abs(window['High'] - window['Close'].shift(1)),
        abs(window['Low'] - window['Close'].shift(1))
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean().iloc[-1]
    
    entry_price = df.iloc[i]['Open']
    
    if prev_wr < oversold and current_wr >= oversold:
        return {
            'direction': 'long',
            'stop_loss': entry_price - atr_sl_mult * atr,
            'take_profit': entry_price + atr_tp_mult * atr,
        }
    
    if prev_wr > overbought and current_wr <= overbought:
        return {
            'direction': 'short',
            'stop_loss': entry_price + atr_sl_mult * atr,
            'take_profit': entry_price - atr_tp_mult * atr,
        }
    
    return None


def breakout_forex(df: pd.DataFrame, i: int,
                   period: int = 20,
                   atr_period: int = 14,
                   atr_sl_mult: float = 1.0,  # Tighter
                   atr_tp_mult: float = 2.5) -> Optional[Dict]:
    """Breakout for forex - tighter parameters"""
    if i < period + 1:
        return None
    
    window = df.iloc[i-period:i]
    highest_high = window['High'].max()
    lowest_low = window['Low'].min()
    
    current_close = df.iloc[i]['Close']
    prev_close = df.iloc[i-1]['Close']
    open_price = df.iloc[i]['Open']
    
    tr = pd.concat([
        df['High'] - df['Low'],
        abs(df['High'] - df['Close'].shift(1)),
        abs(df['Low'] - df['Close'].shift(1))
    ], axis=1).max(axis=1)
    atr = tr.iloc[i-atr_period:i].mean()
    
    if prev_close <= highest_high < current_close:
        return {
            'direction': 'long',
            'stop_loss': open_price - atr_sl_mult * atr,
            'take_profit': open_price + atr_tp_mult * atr,
        }
    
    if prev_close >= lowest_low > current_close:
        return {
            'direction': 'short',
            'stop_loss': open_price + atr_sl_mult * atr,
            'take_profit': open_price - atr_tp_mult * atr,
        }
    
    return None


def ma_crossover_forex(df: pd.DataFrame, i: int,
                       fast_period: int = 8,  # Faster for forex
                       slow_period: int = 21,
                       atr_period: int = 14,
                       atr_sl_mult: float = 1.0,
                       atr_tp_mult: float = 2.5) -> Optional[Dict]:
    """MA Crossover for forex - faster periods"""
    if i < slow_period + 1:
        return None
    
    window = df.iloc[:i+1]
    
    fast_ma = window['Close'].rolling(fast_period).mean()
    slow_ma = window['Close'].rolling(slow_period).mean()
    
    current_fast = fast_ma.iloc[-1]
    current_slow = slow_ma.iloc[-1]
    prev_fast = fast_ma.iloc[-2]
    prev_slow = slow_ma.iloc[-2]
    
    open_price = df.iloc[i]['Open']
    
    tr = pd.concat([
        window['High'] - window['Low'],
        abs(window['High'] - window['Close'].shift(1)),
        abs(window['Low'] - window['Close'].shift(1))
    ], axis=1).max(axis=1)
    atr = tr.iloc[-atr_period:].mean()
    
    if prev_fast <= prev_slow and current_fast > current_slow:
        return {
            'direction': 'long',
            'stop_loss': open_price - atr_sl_mult * atr,
            'take_profit': open_price + atr_tp_mult * atr,
        }
    
    if prev_fast >= prev_slow and current_fast < current_slow:
        return {
            'direction': 'short',
            'stop_loss': open_price + atr_sl_mult * atr,
            'take_profit': open_price - atr_tp_mult * atr,
        }
    
    return None


# Higher timeframe strategy for forex - catches bigger moves
def swing_forex(df: pd.DataFrame, i: int,
                ema_period: int = 50,
                atr_period: int = 14,
                atr_sl_mult: float = 1.5,
                atr_tp_mult: float = 3.0) -> Optional[Dict]:
    """
    Swing trading strategy for forex.
    Entry: Price crosses above/below 50 EMA with momentum confirmation.
    Works better on 4h/daily.
    """
    if i < ema_period + 5:
        return None
    
    window = df.iloc[:i+1]
    
    ema = window['Close'].ewm(span=ema_period, adjust=False).mean()
    
    current_price = df.iloc[i]['Close']
    prev_price = df.iloc[i-1]['Close']
    current_ema = ema.iloc[-1]
    prev_ema = ema.iloc[-2]
    
    # Momentum: 3-bar ROC
    roc = (current_price / df.iloc[i-3]['Close'] - 1) * 100
    
    open_price = df.iloc[i]['Open']
    
    tr = pd.concat([
        window['High'] - window['Low'],
        abs(window['High'] - window['Close'].shift(1)),
        abs(window['Low'] - window['Close'].shift(1))
    ], axis=1).max(axis=1)
    atr = tr.iloc[-atr_period:].mean()
    
    # Cross above EMA with positive momentum
    if prev_price < prev_ema and current_price > current_ema and roc > 0.5:
        return {
            'direction': 'long',
            'stop_loss': open_price - atr_sl_mult * atr,
            'take_profit': open_price + atr_tp_mult * atr,
        }
    
    # Cross below EMA with negative momentum
    if prev_price > prev_ema and current_price < current_ema and roc < -0.5:
        return {
            'direction': 'short',
            'stop_loss': open_price + atr_sl_mult * atr,
            'take_profit': open_price - atr_tp_mult * atr,
        }
    
    return None


# Session-based strategy - trade London/NY overlap
def session_breakout_forex(df: pd.DataFrame, i: int,
                           session_bars: int = 4,  # Look at last N bars
                           atr_sl_mult: float = 0.75,
                           atr_tp_mult: float = 2.0) -> Optional[Dict]:
    """
    Session breakout for forex.
    Trade the break of the Asian session range during London/NY.
    Best on 1h timeframe.
    """
    if i < session_bars + 14:
        return None
    
    window = df.iloc[:i+1]
    session_window = df.iloc[i-session_bars:i]
    
    session_high = session_window['High'].max()
    session_low = session_window['Low'].min()
    
    current_close = df.iloc[i]['Close']
    prev_close = df.iloc[i-1]['Close']
    open_price = df.iloc[i]['Open']
    
    tr = pd.concat([
        window['High'] - window['Low'],
        abs(window['High'] - window['Close'].shift(1)),
        abs(window['Low'] - window['Close'].shift(1))
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]
    
    # Break above session high
    if prev_close <= session_high < current_close:
        return {
            'direction': 'long',
            'stop_loss': session_low,  # Session low as stop
            'take_profit': open_price + atr_tp_mult * atr,
        }
    
    # Break below session low
    if prev_close >= session_low > current_close:
        return {
            'direction': 'short',
            'stop_loss': session_high,  # Session high as stop
            'take_profit': open_price - atr_tp_mult * atr,
        }
    
    return None


FOREX_STRATEGIES = {
    'williams_r_fx': williams_r_forex,
    'breakout_fx': breakout_forex,
    'ma_crossover_fx': ma_crossover_forex,
    'swing_fx': swing_forex,
    'session_breakout_fx': session_breakout_forex,
}
