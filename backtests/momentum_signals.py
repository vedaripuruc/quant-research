#!/usr/bin/env python3
"""
Momentum Signal Tracker - LINK/ADA/XRP
======================================
Simple momentum-based entry signals with proper position sizing.

Signals:
- RSI oversold/overbought reversals
- EMA crossovers
- Breakout from consolidation

Position Sizing:
- ATR-based stop loss distance
- Risk per trade: 1% of account (configurable)
- Formula: position_size = (account * risk_pct) / (entry - stop_loss)
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import os

# Configuration
ASSETS = {
    "LINKUSDT": {"name": "Chainlink", "pip": 0.001},
    "ADAUSDT": {"name": "Cardano", "pip": 0.0001},
    "XRPUSDT": {"name": "XRP", "pip": 0.0001},
}

# Position Sizing Config
DEFAULT_ACCOUNT_SIZE = 100_000  # $100,000
DEFAULT_RISK_PCT = 0.01  # 1% risk per trade
MAX_RISK_PCT = 0.02  # Never risk more than 2%

SIGNAL_FILE = os.path.join(os.path.dirname(__file__), "active_signals.json")


def fetch_binance_klines(symbol: str, interval: str = "1h", limit: int = 100, 
                         start_time: int = None, end_time: int = None) -> pd.DataFrame:
    """Fetch klines from Binance. Supports historical fetching with start/end times."""
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    
    if start_time:
        params["startTime"] = start_time
    if end_time:
        params["endTime"] = end_time
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        return df[['open', 'high', 'low', 'close', 'volume']]
    
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return pd.DataFrame()


def fetch_live_price(symbol: str) -> Optional[float]:
    """Fetch current live price from Binance ticker."""
    url = "https://api.binance.com/api/v3/ticker/price"
    params = {"symbol": symbol}
    
    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        return float(data['price'])
    except Exception as e:
        print(f"Error fetching live price for {symbol}: {e}")
        return None


def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate RSI."""
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_ema(df: pd.DataFrame, period: int) -> pd.Series:
    """Calculate EMA."""
    return df['close'].ewm(span=period, adjust=False).mean()


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate ATR."""
    high = df['high']
    low = df['low']
    close = df['close'].shift(1)
    
    tr = pd.DataFrame({
        'hl': high - low,
        'hc': abs(high - close),
        'lc': abs(low - close)
    }).max(axis=1)
    
    return tr.rolling(window=period).mean()


def calculate_position_size(entry: float, stop_loss: float, 
                            account_size: float = DEFAULT_ACCOUNT_SIZE,
                            risk_pct: float = DEFAULT_RISK_PCT,
                            atr: float = None) -> Dict:
    """
    Calculate position size based on risk management.
    
    Formula: position_size_usd = (account * risk_pct) / abs(entry - stop_loss) * entry
    Position in units = risk_amount / abs(entry - stop_loss)
    
    Returns dict with position sizing info.
    """
    # Dollar risk
    risk_amount = account_size * risk_pct
    
    # Stop loss distance
    sl_distance = abs(entry - stop_loss)
    
    # Prevent division by zero
    if sl_distance == 0:
        return {
            "position_size_units": 0,
            "position_size_usd": 0,
            "risk_amount_usd": risk_amount,
            "sl_distance": 0,
            "sl_distance_pct": 0,
            "error": "SL distance is zero"
        }
    
    # Position size in units (how many coins/tokens to buy)
    position_size_units = risk_amount / sl_distance
    
    # Position size in USD
    position_size_usd = position_size_units * entry
    
    # SL distance as percentage
    sl_distance_pct = (sl_distance / entry) * 100
    
    return {
        "position_size_units": round(position_size_units, 4),
        "position_size_usd": round(position_size_usd, 2),
        "risk_amount_usd": round(risk_amount, 2),
        "sl_distance": round(sl_distance, 6),
        "sl_distance_pct": round(sl_distance_pct, 2),
        "account_size": account_size,
        "risk_pct": risk_pct * 100,  # As percentage for display
    }


def detect_signals(symbol: str, config: Dict, 
                   df: pd.DataFrame = None,
                   use_live_price: bool = True,
                   account_size: float = DEFAULT_ACCOUNT_SIZE,
                   risk_pct: float = DEFAULT_RISK_PCT) -> Optional[Dict]:
    """
    Detect momentum signals for a single asset.
    
    Signal types:
    1. RSI Reversal: RSI crosses above 30 (long) or below 70 (short)
    2. EMA Cross: Fast EMA crosses slow EMA
    3. Breakout: Price breaks above/below consolidation range
    
    Args:
        symbol: Trading pair symbol
        config: Asset configuration
        df: Optional pre-fetched dataframe (for backtesting)
        use_live_price: Whether to fetch live price (False for backtesting)
        account_size: Account size for position sizing
        risk_pct: Risk percentage per trade
    """
    if df is None:
        df = fetch_binance_klines(symbol, interval="1h", limit=100)
    
    if df.empty or len(df) < 50:
        return None
    
    # Calculate indicators
    df = df.copy()
    df['rsi'] = calculate_rsi(df)
    df['ema_fast'] = calculate_ema(df, 9)
    df['ema_slow'] = calculate_ema(df, 21)
    df['atr'] = calculate_atr(df)
    df['vol_avg_20'] = df['volume'].rolling(window=20).mean()
    
    # Current values
    current = df.iloc[-1]
    prev = df.iloc[-2]
    
    price = current['close']
    rsi = current['rsi']
    atr = current['atr']
    ema_fast = current['ema_fast']
    ema_slow = current['ema_slow']
    
    # Use PREVIOUS candle's volume (current candle is incomplete/partial)
    volume = prev['volume']
    vol_avg_20 = prev['vol_avg_20'] if 'vol_avg_20' in prev else current['vol_avg_20']
    
    # Volume filter: require 1.5x average volume for signal confirmation
    # Valid for CRYPTO (exchange volume is real). DO NOT use for forex.
    if pd.notna(vol_avg_20) and volume <= (1.5 * vol_avg_20):
        return None  # Low volume = weak signal, skip
    
    signals = []
    
    # 1. RSI Reversal Signal
    if prev['rsi'] < 30 and rsi >= 30:
        signals.append({
            "type": "RSI_REVERSAL",
            "direction": "LONG",
            "reason": f"RSI crossed above 30 ({rsi:.1f})",
            "strength": "STRONG" if rsi < 35 else "MEDIUM"
        })
    elif prev['rsi'] > 70 and rsi <= 70:
        signals.append({
            "type": "RSI_REVERSAL",
            "direction": "SHORT",
            "reason": f"RSI crossed below 70 ({rsi:.1f})",
            "strength": "STRONG" if rsi > 65 else "MEDIUM"
        })
    
    # 2. EMA Crossover
    prev_diff = prev['ema_fast'] - prev['ema_slow']
    curr_diff = ema_fast - ema_slow
    
    if prev_diff < 0 and curr_diff > 0:
        signals.append({
            "type": "EMA_CROSS",
            "direction": "LONG",
            "reason": f"EMA9 crossed above EMA21",
            "strength": "MEDIUM"
        })
    elif prev_diff > 0 and curr_diff < 0:
        signals.append({
            "type": "EMA_CROSS",
            "direction": "SHORT",
            "reason": f"EMA9 crossed below EMA21",
            "strength": "MEDIUM"
        })
    
    # 3. Breakout from recent range
    lookback = df.iloc[-24:-1]  # Last 24 hours excluding current
    range_high = lookback['high'].max()
    range_low = lookback['low'].min()
    range_size = range_high - range_low
    
    # Tight range = consolidation (less than 2x ATR)
    if range_size < 2 * atr:
        if current['close'] > range_high:
            signals.append({
                "type": "BREAKOUT",
                "direction": "LONG",
                "reason": f"Broke above {range_high:.4f} consolidation",
                "strength": "STRONG"
            })
        elif current['close'] < range_low:
            signals.append({
                "type": "BREAKOUT",
                "direction": "SHORT",
                "reason": f"Broke below {range_low:.4f} consolidation",
                "strength": "STRONG"
            })
    
    if not signals:
        return None
    
    # Calculate entry/SL/TP for the strongest signal
    best_signal = max(signals, key=lambda x: 1 if x['strength'] == 'STRONG' else 0)
    direction = best_signal['direction']
    
    # Entry price: use live or candle close
    if use_live_price:
        live_price = fetch_live_price(symbol)
        if live_price is None:
            live_price = price
        entry = live_price
    else:
        entry = price
        live_price = price
    
    # SL: 1.5x ATR
    sl_distance = atr * 1.5
    
    # TP: 2x SL distance (1:2 RR)
    tp_distance = sl_distance * 2
    
    if direction == "LONG":
        stop_loss = entry - sl_distance
        take_profit = entry + tp_distance
    else:
        stop_loss = entry + sl_distance
        take_profit = entry - tp_distance
    
    # VALIDATION: Check if price is already beyond SL (signal invalid)
    if direction == "LONG" and live_price <= stop_loss:
        print(f"⚠ {symbol}: Signal invalidated - live price ${live_price:.4f} already at/below SL ${stop_loss:.4f}")
        return None
    elif direction == "SHORT" and live_price >= stop_loss:
        print(f"⚠ {symbol}: Signal invalidated - live price ${live_price:.4f} already at/above SL ${stop_loss:.4f}")
        return None
    
    # Calculate position sizing
    position = calculate_position_size(
        entry=entry,
        stop_loss=stop_loss,
        account_size=account_size,
        risk_pct=risk_pct,
        atr=atr
    )
    
    return {
        "symbol": symbol,
        "name": config['name'],
        "timestamp": datetime.now().isoformat(),
        "price": live_price,
        "candle_close": price,
        "direction": direction,
        "signals": signals,
        "entry": entry,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "atr": atr,
        "rsi": rsi,
        "risk_reward": "1:2",
        # Volume confirmation
        "volume": volume,
        "vol_avg_20": vol_avg_20,
        "vol_ratio": round(volume / vol_avg_20, 2) if vol_avg_20 else None,
        # Position sizing fields
        "position_size_usd": position["position_size_usd"],
        "position_size_units": position["position_size_units"],
        "risk_amount_usd": position["risk_amount_usd"],
        "sl_distance_pct": position["sl_distance_pct"],
    }


def check_all_signals(account_size: float = DEFAULT_ACCOUNT_SIZE,
                      risk_pct: float = DEFAULT_RISK_PCT) -> Dict:
    """Check signals for all configured assets."""
    results = {
        "timestamp": datetime.now().isoformat(),
        "status": "ok",
        "signals": [],
        "no_signal": [],
        "config": {
            "account_size": account_size,
            "risk_pct": risk_pct * 100,
        }
    }

    # ══════════════════════════════════════════════════════════════
    # MOMENTUM SIGNALS DISABLED (2026-02-13)
    # Hurst H > 0.6 fires 100% on crypto hourly (mean=0.758).
    # Zero directional power (48-50% WR). See hurst_profile.py.
    # Re-enable when recalibrated or replaced with LLM agent.
    # ══════════════════════════════════════════════════════════════
    SIGNALS_ENABLED = False

    if not SIGNALS_ENABLED:
        print("  ⚠️ Momentum signals DISABLED — Hurst phantom signals killed 2026-02-13")
        for config in ASSETS.values():
            results["no_signal"].append(config['name'])
        return results

    for symbol, config in ASSETS.items():
        signal = detect_signals(symbol, config, 
                               account_size=account_size, 
                               risk_pct=risk_pct)
        
        if signal:
            results["signals"].append(signal)
        else:
            results["no_signal"].append(config['name'])
    
    if results["signals"]:
        results["status"] = "signals"
    
    return results


def format_signal_alert(signal: Dict) -> str:
    """Format a signal for human-readable alert."""
    direction_emoji = "🟢" if signal['direction'] == "LONG" else "🔴"
    
    alert = f"""
{direction_emoji} **{signal['direction']} SIGNAL - {signal['name']}**

📊 Price: ${signal['price']:.4f}
📈 RSI: {signal['rsi']:.1f}

**Trade Setup:**
• Entry: ${signal['entry']:.4f}
• Stop Loss: ${signal['stop_loss']:.4f} ({signal['sl_distance_pct']:.2f}% away)
• Take Profit: ${signal['take_profit']:.4f}
• R:R: {signal['risk_reward']}

**Position Sizing:**
• Size: {signal['position_size_units']:.2f} units (${signal['position_size_usd']:,.2f})
• Risk: ${signal['risk_amount_usd']:,.2f}

**Signals Detected:**
"""
    
    for s in signal['signals']:
        strength_emoji = "🔥" if s['strength'] == "STRONG" else "⚡"
        alert += f"• {strength_emoji} {s['type']}: {s['reason']}\n"
    
    alert += f"\n⏰ {signal['timestamp']}"
    
    return alert.strip()


def main():
    """Main entry point - check signals and save to file."""
    print("="*50)
    print("MOMENTUM SIGNAL CHECK - LINK/ADA/XRP")
    print("="*50)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Account: ${DEFAULT_ACCOUNT_SIZE:,} | Risk: {DEFAULT_RISK_PCT*100:.1f}%\n")
    
    results = check_all_signals()
    
    # Save to file
    with open(SIGNAL_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    
    if results["signals"]:
        print(f"🎯 SIGNALS FOUND: {len(results['signals'])}\n")
        
        for signal in results["signals"]:
            print(format_signal_alert(signal))
            print("-"*40)
    else:
        print("No signals detected")
        print(f"Assets checked: {', '.join(results['no_signal'])}")
    
    print(f"\n✅ Results saved to {SIGNAL_FILE}")
    
    return results


if __name__ == "__main__":
    main()
