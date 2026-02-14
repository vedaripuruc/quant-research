"""
Fade Retail Strategy v2 - Failed Breakout Edition
--------------------------------------------------
The REAL way to fade retail:

When retail buys a breakout that FAILS, we sell their bags.
When retail sells a breakdown that FAILS, we buy their panic.

Failed breakout = price breaks out, then comes BACK inside the range.
That's when retail is trapped. That's when we strike.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = pd.concat([
        df['High'] - df['Low'],
        abs(df['High'] - df['Close'].shift(1)),
        abs(df['Low'] - df['Close'].shift(1))
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def failed_breakout_fade(df: pd.DataFrame, i: int,
                         period: int = 20,
                         confirm_bars: int = 3,  # Bars to confirm failure
                         atr_sl_mult: float = 1.5,
                         atr_tp_mult: float = 3.0) -> Optional[Dict]:
    """
    Failed Breakout Fade Strategy
    
    Logic:
    1. Detect when price broke above/below channel (breakout)
    2. Wait for price to come BACK inside the channel (failure)
    3. Enter OPPOSITE direction - fade the trapped retail
    
    - If there was a LONG breakout that failed → SHORT (trapped longs will puke)
    - If there was a SHORT breakout that failed → LONG (trapped shorts will cover)
    """
    if i < period + confirm_bars + 5:
        return None
    
    # Get channel bounds
    lookback = df.iloc[i-period-confirm_bars:i-confirm_bars]
    channel_high = lookback['High'].max()
    channel_low = lookback['Low'].min()
    
    # Check for breakout in recent bars
    recent = df.iloc[i-confirm_bars:i]
    current = df.iloc[i]
    
    atr = calculate_atr(df.iloc[:i+1]).iloc[-1]
    open_price = current['Open']
    
    # Check for failed LONG breakout (broke above, now back inside)
    broke_above = (recent['High'] > channel_high).any()
    back_inside_from_above = current['Close'] < channel_high
    
    if broke_above and back_inside_from_above:
        # Failed long breakout - retail longs are trapped above
        # SHORT their pain
        return {
            'direction': 'short',
            'stop_loss': channel_high + atr_sl_mult * atr,  # Above the failed breakout
            'take_profit': open_price - atr_tp_mult * atr,
            'reason': f'Failed LONG breakout at {channel_high:.5f}, fading trapped longs',
            'channel_high': float(channel_high),
            'channel_low': float(channel_low),
        }
    
    # Check for failed SHORT breakout (broke below, now back inside)
    broke_below = (recent['Low'] < channel_low).any()
    back_inside_from_below = current['Close'] > channel_low
    
    if broke_below and back_inside_from_below:
        # Failed short breakout - retail shorts are trapped below
        # LONG their cover
        return {
            'direction': 'long',
            'stop_loss': channel_low - atr_sl_mult * atr,  # Below the failed breakdown
            'take_profit': open_price + atr_tp_mult * atr,
            'reason': f'Failed SHORT breakout at {channel_low:.5f}, fading trapped shorts',
            'channel_high': float(channel_high),
            'channel_low': float(channel_low),
        }
    
    return None


def overextension_fade(df: pd.DataFrame, i: int,
                       lookback: int = 5,
                       extension_threshold: float = 2.0,  # 2x ATR move
                       atr_sl_mult: float = 1.0,
                       atr_tp_mult: float = 2.0) -> Optional[Dict]:
    """
    Overextension Fade Strategy
    
    When price moves too far too fast (overextended), fade it.
    Retail FOMO'd in, now they'll get rekt.
    
    Logic:
    - Calculate move over last N bars
    - If move > X * ATR, it's overextended
    - Fade the overextension
    """
    if i < lookback + 20:
        return None
    
    window = df.iloc[i-lookback:i+1]
    current = df.iloc[i]
    
    atr = calculate_atr(df.iloc[:i+1]).iloc[-1]
    
    # Calculate move
    start_price = window['Close'].iloc[0]
    end_price = window['Close'].iloc[-1]
    move = end_price - start_price
    move_in_atr = abs(move) / atr
    
    open_price = current['Open']
    
    if move_in_atr >= extension_threshold:
        if move > 0:  # Overextended UP - fade with short
            return {
                'direction': 'short',
                'stop_loss': open_price + atr_sl_mult * atr,
                'take_profit': open_price - atr_tp_mult * atr,
                'reason': f'Overextended UP by {move_in_atr:.1f}x ATR, fading FOMO longs',
                'move_atr': float(move_in_atr),
            }
        else:  # Overextended DOWN - fade with long
            return {
                'direction': 'long',
                'stop_loss': open_price - atr_sl_mult * atr,
                'take_profit': open_price + atr_tp_mult * atr,
                'reason': f'Overextended DOWN by {move_in_atr:.1f}x ATR, fading panic sellers',
                'move_atr': float(move_in_atr),
            }
    
    return None


def trap_and_reverse(df: pd.DataFrame, i: int,
                     period: int = 20,
                     atr_sl_mult: float = 1.5,
                     atr_tp_mult: float = 3.0) -> Optional[Dict]:
    """
    Trap & Reverse Strategy
    
    The ultimate retail trap:
    1. Price makes a new high/low (retail thinks breakout)
    2. Immediately reverses and closes in opposite direction
    3. Retail is trapped, we enter their pain
    
    This is a single-bar trap: spike + reversal in same bar or next bar.
    """
    if i < period + 2:
        return None
    
    lookback = df.iloc[i-period:i-1]  # Exclude last 2 bars
    channel_high = lookback['High'].max()
    channel_low = lookback['Low'].min()
    
    prev = df.iloc[i-1]
    current = df.iloc[i]
    
    atr = calculate_atr(df.iloc[:i+1]).iloc[-1]
    open_price = current['Open']
    
    # Bull trap: Previous bar made new high, current bar closes below previous low
    if prev['High'] > channel_high and current['Close'] < prev['Low']:
        return {
            'direction': 'short',
            'stop_loss': prev['High'] + atr_sl_mult * atr,
            'take_profit': open_price - atr_tp_mult * atr,
            'reason': f'BULL TRAP: Spiked to {prev["High"]:.5f}, now reversing hard',
            'trap_price': float(prev['High']),
        }
    
    # Bear trap: Previous bar made new low, current bar closes above previous high
    if prev['Low'] < channel_low and current['Close'] > prev['High']:
        return {
            'direction': 'long',
            'stop_loss': prev['Low'] - atr_sl_mult * atr,
            'take_profit': open_price + atr_tp_mult * atr,
            'reason': f'BEAR TRAP: Dumped to {prev["Low"]:.5f}, now reversing hard',
            'trap_price': float(prev['Low']),
        }
    
    return None


def liquidity_grab_fade(df: pd.DataFrame, i: int,
                        period: int = 20,
                        wick_ratio: float = 0.6,  # Wick must be 60%+ of bar
                        atr_sl_mult: float = 1.5,
                        atr_tp_mult: float = 3.0) -> Optional[Dict]:
    """
    Liquidity Grab Fade Strategy
    
    When price wicks through a level but closes back inside,
    it's a liquidity grab - retail stops got hunted.
    
    Now we fade in the direction of the close.
    """
    if i < period + 2:
        return None
    
    lookback = df.iloc[i-period:i]
    channel_high = lookback['High'].max()
    channel_low = lookback['Low'].min()
    
    current = df.iloc[i]
    bar_range = current['High'] - current['Low']
    
    if bar_range == 0:
        return None
    
    atr = calculate_atr(df.iloc[:i+1]).iloc[-1]
    open_price = current['Open']
    
    # Upside liquidity grab: wick above channel but close inside
    if current['High'] > channel_high and current['Close'] < channel_high:
        upper_wick = current['High'] - max(current['Open'], current['Close'])
        if upper_wick / bar_range >= wick_ratio:
            return {
                'direction': 'short',
                'stop_loss': current['High'] + atr_sl_mult * atr,
                'take_profit': open_price - atr_tp_mult * atr,
                'reason': f'Liquidity grab above {channel_high:.5f}, shorts hunted, now fading',
            }
    
    # Downside liquidity grab: wick below channel but close inside
    if current['Low'] < channel_low and current['Close'] > channel_low:
        lower_wick = min(current['Open'], current['Close']) - current['Low']
        if lower_wick / bar_range >= wick_ratio:
            return {
                'direction': 'long',
                'stop_loss': current['Low'] - atr_sl_mult * atr,
                'take_profit': open_price + atr_tp_mult * atr,
                'reason': f'Liquidity grab below {channel_low:.5f}, longs hunted, now fading',
            }
    
    return None


# Strategy registry
FADE_STRATEGIES = {
    'failed_breakout_fade': failed_breakout_fade,
    'overextension_fade': overextension_fade,
    'trap_and_reverse': trap_and_reverse,
    'liquidity_grab_fade': liquidity_grab_fade,
}
