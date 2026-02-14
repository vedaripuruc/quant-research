"""
Trading Strategies
------------------
Clean implementations without look-ahead bias.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict


# =============================================================================
# WILLIAMS %R STRATEGY
# =============================================================================

def williams_r_indicator(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Williams %R indicator."""
    highest_high = df['High'].rolling(window=period).max()
    lowest_low = df['Low'].rolling(window=period).min()
    williams_r = -100 * ((highest_high - df['Close']) / (highest_high - lowest_low))
    return williams_r


def williams_r_signal(df: pd.DataFrame, i: int, 
                      period: int = 14,
                      oversold: float = -80,
                      overbought: float = -20,
                      atr_sl_mult: float = 1.5,
                      atr_tp_mult: float = 3.0) -> Optional[Dict]:
    """
    Williams %R mean reversion strategy.
    
    Entry:
    - Long: Williams %R crosses above oversold (-80)
    - Short: Williams %R crosses below overbought (-20)
    
    Exit: ATR-based SL/TP
    """
    if i < period + 1:
        return None
    
    # Calculate Williams %R for current and previous bar
    # Using only data available at time i (no look-ahead)
    window = df.iloc[:i+1]
    
    highest_high = window['High'].rolling(window=period).max()
    lowest_low = window['Low'].rolling(window=period).min()
    wr = -100 * ((highest_high - window['Close']) / (highest_high - lowest_low))
    
    current_wr = wr.iloc[-1]
    prev_wr = wr.iloc[-2]
    
    # Calculate ATR for position sizing
    tr = pd.concat([
        window['High'] - window['Low'],
        abs(window['High'] - window['Close'].shift(1)),
        abs(window['Low'] - window['Close'].shift(1))
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean().iloc[-1]
    
    entry_price = df.iloc[i]['Open']
    
    # Long signal: cross above oversold
    if prev_wr < oversold and current_wr >= oversold:
        return {
            'direction': 'long',
            'stop_loss': entry_price - atr_sl_mult * atr,
            'take_profit': entry_price + atr_tp_mult * atr,
        }
    
    # Short signal: cross below overbought
    if prev_wr > overbought and current_wr <= overbought:
        return {
            'direction': 'short',
            'stop_loss': entry_price + atr_sl_mult * atr,
            'take_profit': entry_price - atr_tp_mult * atr,
        }
    
    return None


# =============================================================================
# FAIR VALUE GAP (FVG) STRATEGY
# =============================================================================

def find_fvgs(df: pd.DataFrame, i: int, lookback: int = 50) -> list:
    """
    Find unfilled Fair Value Gaps up to current bar.
    Uses only historical data (no look-ahead).
    """
    start_idx = max(2, i - lookback)
    fvgs = []
    
    for j in range(start_idx, i):
        if j < 2:
            continue
            
        prev2_high = df.iloc[j-2]['High']
        prev2_low = df.iloc[j-2]['Low']
        curr_high = df.iloc[j]['High']
        curr_low = df.iloc[j]['Low']
        
        # Bullish FVG: gap up (current low > 2-bars-ago high)
        if curr_low > prev2_high:
            gap = {
                'type': 'bullish',
                'top': curr_low,
                'bottom': prev2_high,
                'midpoint': (curr_low + prev2_high) / 2,
                'created_at': j,
            }
            # Check if filled by subsequent price action
            filled = False
            for k in range(j+1, i+1):
                if df.iloc[k]['Low'] <= prev2_high:
                    filled = True
                    break
            if not filled:
                fvgs.append(gap)
        
        # Bearish FVG: gap down (2-bars-ago low > current high)
        elif prev2_low > curr_high:
            gap = {
                'type': 'bearish',
                'top': prev2_low,
                'bottom': curr_high,
                'midpoint': (prev2_low + curr_high) / 2,
                'created_at': j,
            }
            # Check if filled
            filled = False
            for k in range(j+1, i+1):
                if df.iloc[k]['High'] >= prev2_low:
                    filled = True
                    break
            if not filled:
                fvgs.append(gap)
    
    return fvgs


def fvg_signal(df: pd.DataFrame, i: int,
               sl_mult: float = 1.5,
               tp_mult: float = 3.0) -> Optional[Dict]:
    """
    Fair Value Gap mean reversion strategy.
    
    Entry: Price retraces to FVG midpoint
    Exit: Gap-size based SL/TP
    """
    if i < 3:
        return None
    
    fvgs = find_fvgs(df, i)
    if not fvgs:
        return None
    
    current_price = df.iloc[i]['Close']
    prev_price = df.iloc[i-1]['Close']
    open_price = df.iloc[i]['Open']
    
    # Find nearest untouched FVG that price is entering
    for gap in reversed(fvgs):  # Most recent first
        midpoint = gap['midpoint']
        gap_size = gap['top'] - gap['bottom']
        
        if gap['type'] == 'bullish':
            # Price crossing up through midpoint
            if prev_price < midpoint <= current_price:
                return {
                    'direction': 'long',
                    'stop_loss': open_price - sl_mult * gap_size,
                    'take_profit': open_price + tp_mult * gap_size,
                }
        
        else:  # bearish
            # Price crossing down through midpoint
            if prev_price > midpoint >= current_price:
                return {
                    'direction': 'short',
                    'stop_loss': open_price + sl_mult * gap_size,
                    'take_profit': open_price - tp_mult * gap_size,
                }
    
    return None


# =============================================================================
# BREAKOUT STRATEGY (Donchian Channel)
# =============================================================================

def breakout_signal(df: pd.DataFrame, i: int,
                    period: int = 20,
                    atr_period: int = 14,
                    atr_sl_mult: float = 2.0,
                    atr_tp_mult: float = 4.0) -> Optional[Dict]:
    """
    Donchian Channel Breakout Strategy.
    
    Entry:
    - Long: Close breaks above highest high of last N bars
    - Short: Close breaks below lowest low of last N bars
    
    Exit: ATR-based trailing stop (simplified to fixed TP/SL here)
    """
    if i < period + 1:
        return None
    
    window = df.iloc[i-period:i]
    highest_high = window['High'].max()
    lowest_low = window['Low'].min()
    
    current_close = df.iloc[i]['Close']
    prev_close = df.iloc[i-1]['Close']
    open_price = df.iloc[i]['Open']
    
    # ATR for position sizing
    tr = pd.concat([
        df['High'] - df['Low'],
        abs(df['High'] - df['Close'].shift(1)),
        abs(df['Low'] - df['Close'].shift(1))
    ], axis=1).max(axis=1)
    atr = tr.iloc[i-atr_period:i].mean()
    
    # Breakout above
    if prev_close <= highest_high < current_close:
        return {
            'direction': 'long',
            'stop_loss': open_price - atr_sl_mult * atr,
            'take_profit': open_price + atr_tp_mult * atr,
        }
    
    # Breakout below
    if prev_close >= lowest_low > current_close:
        return {
            'direction': 'short',
            'stop_loss': open_price + atr_sl_mult * atr,
            'take_profit': open_price - atr_tp_mult * atr,
        }
    
    return None


# =============================================================================
# RSI DIVERGENCE STRATEGY
# =============================================================================

def rsi_indicator(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def rsi_divergence_signal(df: pd.DataFrame, i: int,
                          rsi_period: int = 14,
                          lookback: int = 10,
                          oversold: float = 30,
                          overbought: float = 70,
                          atr_sl_mult: float = 1.5,
                          atr_tp_mult: float = 3.0) -> Optional[Dict]:
    """
    RSI Divergence Strategy.
    
    Entry:
    - Bullish divergence: Price makes lower low, RSI makes higher low (near oversold)
    - Bearish divergence: Price makes higher high, RSI makes lower high (near overbought)
    """
    if i < rsi_period + lookback:
        return None
    
    window = df.iloc[:i+1]
    rsi = rsi_indicator(window['Close'], rsi_period)
    
    current_rsi = rsi.iloc[-1]
    price_window = window['Close'].iloc[-lookback:]
    rsi_window = rsi.iloc[-lookback:]
    
    open_price = df.iloc[i]['Open']
    
    # ATR for exits
    tr = pd.concat([
        window['High'] - window['Low'],
        abs(window['High'] - window['Close'].shift(1)),
        abs(window['Low'] - window['Close'].shift(1))
    ], axis=1).max(axis=1)
    atr = tr.iloc[-rsi_period:].mean()
    
    # Find local lows/highs
    price_low_idx = price_window.idxmin()
    price_high_idx = price_window.idxmax()
    
    # Bullish divergence
    if current_rsi < 40:  # Near oversold zone
        current_price = window['Close'].iloc[-1]
        prev_low_price = price_window.min()
        
        # Price at new low but RSI higher than at previous low
        if current_price <= prev_low_price * 1.01:  # Within 1% of low
            prev_rsi_at_low = rsi.loc[price_low_idx] if price_low_idx in rsi.index else rsi.iloc[-lookback]
            if current_rsi > prev_rsi_at_low + 5:  # RSI higher by at least 5
                return {
                    'direction': 'long',
                    'stop_loss': open_price - atr_sl_mult * atr,
                    'take_profit': open_price + atr_tp_mult * atr,
                }
    
    # Bearish divergence
    if current_rsi > 60:  # Near overbought zone
        current_price = window['Close'].iloc[-1]
        prev_high_price = price_window.max()
        
        if current_price >= prev_high_price * 0.99:  # Within 1% of high
            prev_rsi_at_high = rsi.loc[price_high_idx] if price_high_idx in rsi.index else rsi.iloc[-lookback]
            if current_rsi < prev_rsi_at_high - 5:  # RSI lower by at least 5
                return {
                    'direction': 'short',
                    'stop_loss': open_price + atr_sl_mult * atr,
                    'take_profit': open_price - atr_tp_mult * atr,
                }
    
    return None


# =============================================================================
# MOVING AVERAGE CROSSOVER
# =============================================================================

def ma_crossover_signal(df: pd.DataFrame, i: int,
                        fast_period: int = 10,
                        slow_period: int = 30,
                        atr_period: int = 14,
                        atr_sl_mult: float = 2.0,
                        atr_tp_mult: float = 4.0) -> Optional[Dict]:
    """
    Simple Moving Average Crossover.
    
    Entry:
    - Long: Fast MA crosses above Slow MA
    - Short: Fast MA crosses below Slow MA
    """
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
    
    # ATR for exits
    tr = pd.concat([
        window['High'] - window['Low'],
        abs(window['High'] - window['Close'].shift(1)),
        abs(window['Low'] - window['Close'].shift(1))
    ], axis=1).max(axis=1)
    atr = tr.iloc[-atr_period:].mean()
    
    # Golden cross
    if prev_fast <= prev_slow and current_fast > current_slow:
        return {
            'direction': 'long',
            'stop_loss': open_price - atr_sl_mult * atr,
            'take_profit': open_price + atr_tp_mult * atr,
        }
    
    # Death cross
    if prev_fast >= prev_slow and current_fast < current_slow:
        return {
            'direction': 'short',
            'stop_loss': open_price + atr_sl_mult * atr,
            'take_profit': open_price - atr_tp_mult * atr,
        }
    
    return None


# =============================================================================
# VOLATILITY SQUEEZE (Bollinger + Keltner)
# =============================================================================

def volatility_squeeze_signal(df: pd.DataFrame, i: int,
                              bb_period: int = 20,
                              bb_std: float = 2.0,
                              kc_period: int = 20,
                              kc_mult: float = 1.5,
                              atr_sl_mult: float = 2.0,
                              atr_tp_mult: float = 4.0) -> Optional[Dict]:
    """
    Volatility Squeeze Strategy (TTM Squeeze concept).
    
    Entry: When Bollinger Bands come inside Keltner Channels (squeeze),
           trade the breakout direction.
    """
    if i < max(bb_period, kc_period) + 1:
        return None
    
    window = df.iloc[:i+1]
    close = window['Close']
    high = window['High']
    low = window['Low']
    
    # Bollinger Bands
    bb_ma = close.rolling(bb_period).mean()
    bb_std_val = close.rolling(bb_period).std()
    bb_upper = bb_ma + bb_std * bb_std_val
    bb_lower = bb_ma - bb_std * bb_std_val
    
    # Keltner Channels
    tr = pd.concat([
        high - low,
        abs(high - close.shift(1)),
        abs(low - close.shift(1))
    ], axis=1).max(axis=1)
    atr = tr.rolling(kc_period).mean()
    kc_ma = close.rolling(kc_period).mean()
    kc_upper = kc_ma + kc_mult * atr
    kc_lower = kc_ma - kc_mult * atr
    
    # Squeeze detection
    in_squeeze = (bb_lower.iloc[-1] > kc_lower.iloc[-1]) and (bb_upper.iloc[-1] < kc_upper.iloc[-1])
    was_in_squeeze = (bb_lower.iloc[-2] > kc_lower.iloc[-2]) and (bb_upper.iloc[-2] < kc_upper.iloc[-2])
    
    open_price = df.iloc[i]['Open']
    current_atr = atr.iloc[-1]
    
    # Breakout from squeeze
    if was_in_squeeze and not in_squeeze:
        current_price = close.iloc[-1]
        prev_price = close.iloc[-2]
        
        # Upward breakout
        if current_price > bb_upper.iloc[-1]:
            return {
                'direction': 'long',
                'stop_loss': open_price - atr_sl_mult * current_atr,
                'take_profit': open_price + atr_tp_mult * current_atr,
            }
        
        # Downward breakout
        if current_price < bb_lower.iloc[-1]:
            return {
                'direction': 'short',
                'stop_loss': open_price + atr_sl_mult * current_atr,
                'take_profit': open_price - atr_tp_mult * current_atr,
            }
    
    return None


# Strategy registry
STRATEGIES = {
    'williams_r': williams_r_signal,
    'fvg': fvg_signal,
    'breakout': breakout_signal,
    'rsi_divergence': rsi_divergence_signal,
    'ma_crossover': ma_crossover_signal,
    'volatility_squeeze': volatility_squeeze_signal,
}
