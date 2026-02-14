"""
Troll Strategy v2 - Fade the Retail (FIXED)
--------------------------------------------
FIXED:
- Dynamic lookback based on asset class (forex=24h, stocks=7h)
- Proper volatility calculation per market
- Longer signal validity window

Logic:
- LONG signal → Wait for dip (weak hands selling) → Buy cheap
- SHORT signal → Wait for pump (weak hands buying) → Sell high

"Little bit" = % of N-day average volatility
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict


def get_asset_class(symbol: str) -> str:
    """Detect asset class from symbol."""
    symbol = str(symbol).upper() if symbol else ''
    if '=X' in symbol:
        return 'forex'
    elif '=F' in symbol:
        return 'futures'
    elif symbol.startswith('^'):
        return 'index'
    else:
        return 'stock'


def get_bars_per_day(symbol: str = '', interval: str = '1h') -> int:
    """Get expected number of bars per trading day."""
    asset_class = get_asset_class(symbol)
    
    hours_per_day = {
        'forex': 24,
        'futures': 23,
        'index': 7,
        'stock': 7,
    }
    
    hours = hours_per_day.get(asset_class, 7)
    
    if interval == '1h':
        return hours
    elif interval == '4h':
        return max(1, hours // 4)
    elif interval == '15m':
        return hours * 4
    elif interval == '30m':
        return hours * 2
    elif interval == '1d':
        return 1
    else:
        return hours


def calculate_volatility(df: pd.DataFrame, i: int, 
                         days: int = 5, 
                         symbol: str = '',
                         interval: str = '1h') -> float:
    """
    Calculate average daily volatility over last N trading days.
    Returns as percentage of price.
    
    FIXED: Uses correct bars per day based on asset class.
    """
    bars_per_day = get_bars_per_day(symbol, interval)
    lookback_bars = min(days * bars_per_day, i)
    
    if lookback_bars < bars_per_day:
        return 0.01  # Default 1% if insufficient data
    
    window = df.iloc[i-lookback_bars:i]
    
    # Calculate True Range
    tr = pd.concat([
        window['High'] - window['Low'],
        abs(window['High'] - window['Close'].shift(1)),
        abs(window['Low'] - window['Close'].shift(1))
    ], axis=1).max(axis=1)
    
    # Average TR as % of price
    avg_tr = tr.mean()
    avg_price = window['Close'].mean()
    avg_volatility = (avg_tr / avg_price) * 100
    
    return max(0.1, avg_volatility)


def get_breakout_signal(df: pd.DataFrame, i: int, period: int = 20) -> Optional[str]:
    """
    Check if we have a breakout signal at bar i.
    Returns 'LONG', 'SHORT', or None.
    
    Uses CROSSOVER logic (prev must be inside, current must be outside).
    """
    if i < period + 1:
        return None
    
    lookback = df.iloc[i-period:i]  # Bars BEFORE current
    current = df.iloc[i]
    prev = df.iloc[i-1]
    
    highest = lookback['High'].max()
    lowest = lookback['Low'].min()
    
    # Breakout above (crossover)
    if prev['Close'] <= highest and current['Close'] > highest:
        return 'LONG'
    
    # Breakout below (crossover)
    if prev['Close'] >= lowest and current['Close'] < lowest:
        return 'SHORT'
    
    return None


def troll_breakout_signal(df: pd.DataFrame, i: int,
                          period: int = 20,
                          fade_pct: float = 0.5,
                          atr_sl_mult: float = 1.5,
                          atr_tp_mult: float = 3.0,
                          symbol: str = '',
                          interval: str = '1h',
                          lookback_days: float = 2.0) -> Optional[Dict]:
    """
    Troll Breakout Strategy - Fade the Retail
    
    Instead of entering on breakout, we:
    1. Detect breakout signal
    2. Wait for price to retrace against the signal
    3. Enter at the better price
    
    FIXED:
    - lookback_days is now configurable and asset-aware
    - Uses correct bars per day calculation
    
    Args:
        lookback_days: How many TRADING DAYS to keep signal valid (default 2)
    """
    bars_per_day = get_bars_per_day(symbol, interval)
    min_history = max(period + 1, int(5 * bars_per_day))  # Need 5 days for volatility
    
    if i < min_history:
        return None
    
    # FIXED: Dynamic lookback based on asset class and timeframe
    lookback_for_signal = int(lookback_days * bars_per_day)
    
    # Search for pending breakout signals
    pending_signal = None
    signal_bar = None
    signal_price = None
    
    search_start = max(period + 1, i - lookback_for_signal)
    for j in range(search_start, i):
        sig = get_breakout_signal(df, j, period)
        if sig:
            pending_signal = sig
            signal_bar = j
            signal_price = df.iloc[j]['Close']
    
    if not pending_signal:
        return None
    
    # Calculate required retracement
    volatility = calculate_volatility(df, i, days=5, symbol=symbol, interval=interval)
    required_fade = volatility * fade_pct / 100  # Convert to decimal
    
    current = df.iloc[i]
    current_price = current['Close']
    open_price = current['Open']
    
    # Calculate ATR for SL/TP
    atr_period = 14
    atr_lookback = min(atr_period * 2, i)
    tr = pd.concat([
        df['High'].iloc[i-atr_lookback:i+1] - df['Low'].iloc[i-atr_lookback:i+1],
        abs(df['High'].iloc[i-atr_lookback:i+1] - df['Close'].iloc[i-atr_lookback:i+1].shift(1)),
        abs(df['Low'].iloc[i-atr_lookback:i+1] - df['Close'].iloc[i-atr_lookback:i+1].shift(1))
    ], axis=1).max(axis=1)
    atr = tr.rolling(atr_period).mean().iloc[-1]
    
    if pd.isna(atr) or atr <= 0:
        return None
    
    if pending_signal == 'LONG':
        # For LONG signal, wait for price to DIP
        dip_pct = (signal_price - current_price) / signal_price
        
        if dip_pct >= required_fade:
            return {
                'direction': 'long',
                'stop_loss': open_price - atr_sl_mult * atr,
                'take_profit': open_price + atr_tp_mult * atr,
                '_signal_open': open_price,  # For engine SL/TP adjustment
                'original_signal': pending_signal,
                'signal_price': signal_price,
                'fade_entry': current_price,
                'improvement': dip_pct * 100,
                'volatility_5d': volatility,
                'bars_since_signal': i - signal_bar,
            }
    
    elif pending_signal == 'SHORT':
        # For SHORT signal, wait for price to PUMP
        pump_pct = (current_price - signal_price) / signal_price
        
        if pump_pct >= required_fade:
            return {
                'direction': 'short',
                'stop_loss': open_price + atr_sl_mult * atr,
                'take_profit': open_price - atr_tp_mult * atr,
                '_signal_open': open_price,
                'original_signal': pending_signal,
                'signal_price': signal_price,
                'fade_entry': current_price,
                'improvement': pump_pct * 100,
                'volatility_5d': volatility,
                'bars_since_signal': i - signal_bar,
            }
    
    return None


def aggressive_troll_signal(df: pd.DataFrame, i: int,
                            period: int = 20,
                            symbol: str = '',
                            interval: str = '1h') -> Optional[Dict]:
    """
    Aggressive Troll - Less waiting, tighter stops, faster entries.
    - 30% fade requirement
    - 1 day lookback
    - Tighter SL/TP
    """
    return troll_breakout_signal(
        df, i, period,
        fade_pct=0.3,
        atr_sl_mult=1.0,
        atr_tp_mult=2.5,
        symbol=symbol,
        interval=interval,
        lookback_days=1.0
    )


def patient_troll_signal(df: pd.DataFrame, i: int,
                         period: int = 20,
                         symbol: str = '',
                         interval: str = '1h') -> Optional[Dict]:
    """
    Patient Troll - Wait for bigger pullback, wider stops, bigger targets.
    
    FIXED: 
    - Reduced fade from 75% to 50% (75% was too demanding)
    - Increased lookback to 3 days (was 10 bars regardless of TF)
    """
    return troll_breakout_signal(
        df, i, period,
        fade_pct=0.5,  # Was 0.75 - too demanding
        atr_sl_mult=2.0,
        atr_tp_mult=4.0,
        symbol=symbol,
        interval=interval,
        lookback_days=3.0  # 3 days to wait for pullback
    )


def ultimate_troll_signal(df: pd.DataFrame, i: int,
                          period: int = 20,
                          symbol: str = '',
                          interval: str = '1h') -> Optional[Dict]:
    """
    Ultimate Troll - Dynamic fade based on volatility regime.
    
    High volatility = wait for bigger pullback
    Low volatility = accept smaller pullback
    """
    bars_per_day = get_bars_per_day(symbol, interval)
    min_history = max(period + 1, int(5 * bars_per_day))
    
    if i < min_history:
        return None
    
    volatility = calculate_volatility(df, i, days=5, symbol=symbol, interval=interval)
    
    # Dynamic fade percentage based on volatility regime
    if volatility > 2.0:  # High vol (>2% daily range)
        fade_pct = 0.6
        lookback_days = 2.5
    elif volatility > 1.0:  # Medium vol
        fade_pct = 0.5
        lookback_days = 2.0
    else:  # Low vol (<1%)
        fade_pct = 0.4
        lookback_days = 1.5
    
    return troll_breakout_signal(
        df, i, period,
        fade_pct=fade_pct,
        atr_sl_mult=1.5,
        atr_tp_mult=3.0,
        symbol=symbol,
        interval=interval,
        lookback_days=lookback_days
    )


# Strategy registry with symbol/interval support
def make_troll_strategy(variant: str = 'default', symbol: str = '', interval: str = '1h'):
    """Factory function to create troll strategies with proper context."""
    
    def strategy(df, i):
        if variant == 'aggressive':
            return aggressive_troll_signal(df, i, symbol=symbol, interval=interval)
        elif variant == 'patient':
            return patient_troll_signal(df, i, symbol=symbol, interval=interval)
        elif variant == 'ultimate':
            return ultimate_troll_signal(df, i, symbol=symbol, interval=interval)
        else:
            return troll_breakout_signal(df, i, symbol=symbol, interval=interval)
    
    return strategy


# Legacy registry (for backward compatibility)
TROLL_STRATEGIES = {
    'troll_breakout': lambda df, i: troll_breakout_signal(df, i),
    'troll_aggressive': lambda df, i: aggressive_troll_signal(df, i),
    'troll_patient': lambda df, i: patient_troll_signal(df, i),
    'troll_ultimate': lambda df, i: ultimate_troll_signal(df, i),
}
