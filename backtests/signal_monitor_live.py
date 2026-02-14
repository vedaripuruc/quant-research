#!/usr/bin/env python3
"""
Live Signal Monitor (with real-time data support)
--------------------------------------------------
Uses Finnhub/TwelveData when available, falls back to yfinance.

FIXED to match backtest logic:
- Uses only CLOSED candles (drops last incomplete candle)
- Uses CROSSOVER conditions for entry signals
- Uses Open price as entry reference, not Close

Run: python signal_monitor_live.py --once
     python signal_monitor_live.py --daemon
"""

import os
import sys
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

# Load .env if exists
env_file = Path(__file__).parent / '.env'
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            if '=' in line and not line.startswith('#'):
                key, val = line.strip().split('=', 1)
                os.environ[key] = val

from live_data import get_live_quote, get_live_candles, check_api_keys

# ============================================================================
# CONFIG
# ============================================================================

CONFIG = {
    'instruments': [
        'EURUSD=X', 'USDJPY=X', 'EURJPY=X', 'GBPUSD=X',
        'GC=F', 'CL=F',  # Gold, Oil
    ],
    'strategies': ['session_breakout', 'breakout'],
    'scan_interval': 900,  # 15 minutes
    'candle_count': 100,
    'state_file': Path(__file__).parent / 'signal_state.json',
    'log_file': Path(__file__).parent / 'signal_monitor.log',
}

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(CONFIG['log_file']),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ============================================================================
# STRATEGIES (matching backtest logic in strategies_forex.py)
# ============================================================================

def candles_to_df(candles):
    """Convert candle list to DataFrame and drop last incomplete candle"""
    df = pd.DataFrame(candles)
    df['Date'] = pd.to_datetime(df['time'])
    df = df.rename(columns={
        'open': 'Open', 'high': 'High', 
        'low': 'Low', 'close': 'Close', 'volume': 'Volume'
    })
    
    # Drop the last row - it's likely an incomplete/forming candle
    if len(df) > 1:
        df = df.iloc[:-1]
    
    return df


def calculate_atr(df, period=14):
    tr = pd.concat([
        df['High'] - df['Low'],
        abs(df['High'] - df['Close'].shift(1)),
        abs(df['Low'] - df['Close'].shift(1))
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def check_session_breakout(df, symbol, session_bars=4, atr_sl_mult=0.75, atr_tp_mult=2.0):
    """
    Session Breakout Signal (matches session_breakout_forex in backtest)
    Entry: Price CROSSES above/below last N bars range
    Uses Open price for SL/TP calculations
    """
    if len(df) < session_bars + 14:
        return None
    
    # Session is the N bars before the current bar
    session_window = df.iloc[-(session_bars + 1):-1]
    current = df.iloc[-1]
    prev = df.iloc[-2]
    
    session_high = session_window['High'].max()
    session_low = session_window['Low'].min()
    
    atr = calculate_atr(df, 14).iloc[-1]
    
    # Entry based on Open price (matching backtest)
    open_price = current['Open']
    current_close = current['Close']
    prev_close = prev['Close']
    
    # CROSSOVER: was below/at session high, now above
    if prev_close <= session_high < current_close:
        return {
            'type': 'LONG',
            'strategy': 'session_breakout',
            'symbol': symbol,
            'entry': float(open_price),
            'stop_loss': float(session_low),  # Session low as stop
            'take_profit': float(open_price + atr_tp_mult * atr),
            'session_high': float(session_high),
            'session_low': float(session_low),
            'atr': float(atr),
            'reason': f'Crossover: prev {prev_close:.5f} <= high {session_high:.5f} < curr {current_close:.5f}',
        }
    
    # CROSSOVER: was above/at session low, now below
    elif prev_close >= session_low > current_close:
        return {
            'type': 'SHORT',
            'strategy': 'session_breakout',
            'symbol': symbol,
            'entry': float(open_price),
            'stop_loss': float(session_high),  # Session high as stop
            'take_profit': float(open_price - atr_tp_mult * atr),
            'session_high': float(session_high),
            'session_low': float(session_low),
            'atr': float(atr),
            'reason': f'Crossover: prev {prev_close:.5f} >= low {session_low:.5f} > curr {current_close:.5f}',
        }
    
    return None


def check_breakout(df, symbol, period=20, atr_period=14, atr_sl_mult=1.0, atr_tp_mult=2.5):
    """
    Donchian Breakout Signal (matches breakout_forex in backtest)
    Entry: Price CROSSES above highest high / below lowest low of N bars
    Uses Open price for SL/TP calculations
    """
    if len(df) < period + 2:
        return None
    
    # Lookback window is the N bars before the current bar (exclusive of current)
    lookback = df.iloc[-(period + 1):-1]
    current = df.iloc[-1]
    prev = df.iloc[-2]
    
    highest = lookback['High'].max()
    lowest = lookback['Low'].min()
    
    atr = calculate_atr(df, atr_period).iloc[-1]
    
    # Entry based on Open price (matching backtest)
    open_price = current['Open']
    current_close = current['Close']
    prev_close = prev['Close']
    
    # CROSSOVER: was below/at highest, now above
    if prev_close <= highest < current_close:
        return {
            'type': 'LONG',
            'strategy': 'breakout',
            'symbol': symbol,
            'entry': float(open_price),
            'stop_loss': float(open_price - atr_sl_mult * atr),
            'take_profit': float(open_price + atr_tp_mult * atr),
            'channel_high': float(highest),
            'channel_low': float(lowest),
            'atr': float(atr),
            'reason': f'Crossover: prev {prev_close:.5f} <= high {highest:.5f} < curr {current_close:.5f}',
        }
    
    # CROSSOVER: was above/at lowest, now below
    elif prev_close >= lowest > current_close:
        return {
            'type': 'SHORT',
            'strategy': 'breakout',
            'symbol': symbol,
            'entry': float(open_price),
            'stop_loss': float(open_price + atr_sl_mult * atr),
            'take_profit': float(open_price - atr_tp_mult * atr),
            'channel_high': float(highest),
            'channel_low': float(lowest),
            'atr': float(atr),
            'reason': f'Crossover: prev {prev_close:.5f} >= low {lowest:.5f} > curr {current_close:.5f}',
        }
    
    return None


STRATEGY_FUNCS = {
    'session_breakout': check_session_breakout,
    'breakout': check_breakout,
}

# ============================================================================
# SCANNER
# ============================================================================

def scan_all():
    """Scan all instruments"""
    signals = []
    scanned = []
    errors = []
    data_sources = {}
    
    # Check API status
    api_status = check_api_keys()
    
    for symbol in CONFIG['instruments']:
        try:
            # Get live candles
            candles = get_live_candles(symbol, '1h', CONFIG['candle_count'])
            
            if not candles or len(candles) < 25:
                errors.append(f"{symbol}: No data or insufficient bars")
                continue
            
            df = candles_to_df(candles)  # This now drops the last candle
            
            if len(df) < 25:
                errors.append(f"{symbol}: Insufficient closed bars")
                continue
            
            scanned.append(symbol)
            
            # Track data source
            quote = get_live_quote(symbol)
            if quote:
                data_sources[symbol] = quote.get('source', 'unknown')
            
            # Run strategies
            for strat_name in CONFIG['strategies']:
                strat_func = STRATEGY_FUNCS.get(strat_name)
                if not strat_func:
                    continue
                
                signal = strat_func(df, symbol)
                if signal:
                    signal['scanned_at'] = datetime.now().isoformat()
                    signal['data_source'] = data_sources.get(symbol, 'unknown')
                    signals.append(signal)
                    log.info(f"🎯 SIGNAL: {signal['type']} {symbol} via {strat_name} ({signal['data_source']})")
        
        except Exception as e:
            errors.append(f"{symbol}: {str(e)}")
            log.error(f"Error scanning {symbol}: {e}")
    
    return {
        'signals': signals,
        'scanned': scanned,
        'errors': errors,
        'data_sources': data_sources,
        'api_status': api_status,
        'scan_time': datetime.now().isoformat(),
        'next_scan': (datetime.now() + timedelta(seconds=CONFIG['scan_interval'])).isoformat(),
    }


def save_state(state):
    with open(CONFIG['state_file'], 'w') as f:
        json.dump(state, f, indent=2, default=str)


def run_once():
    """Single scan"""
    log.info("Running single scan...")
    
    # Check API status
    api_status = check_api_keys()
    log.info(f"Data sources: Finnhub={'✓' if api_status['finnhub'] else '✗'}, "
             f"TwelveData={'✓' if api_status['twelvedata'] else '✗'}, "
             f"yfinance=✓")
    
    state = scan_all()
    save_state(state)
    
    print(f"\n{'='*60}")
    print(f"SCAN COMPLETE: {state['scan_time']}")
    print(f"{'='*60}")
    print(f"Scanned: {len(state['scanned'])} instruments")
    print(f"Signals: {len(state['signals'])}")
    print(f"Data sources used: {set(state['data_sources'].values()) or {'yfinance'}}")
    
    if state['signals']:
        print(f"\n🎯 ACTIVE SIGNALS:")
        for s in state['signals']:
            print(f"  {s['type']:5} {s['symbol']:10} @ {s['entry']:.5f}")
            print(f"        SL: {s['stop_loss']:.5f} | TP: {s['take_profit']:.5f}")
            print(f"        Reason: {s['reason']}")
            print(f"        Source: {s.get('data_source', 'unknown')}")
    else:
        print("\n😴 No signals at this time")
    
    return state


def run_daemon():
    """Continuous scanning"""
    log.info("Starting live signal monitor daemon...")
    log.info(f"Scan interval: {CONFIG['scan_interval']}s")
    
    while True:
        try:
            state = scan_all()
            save_state(state)
            
            if state['signals']:
                log.info(f"🎯 {len(state['signals'])} active signals")
            else:
                log.info("😴 No signals")
            
            time.sleep(CONFIG['scan_interval'])
            
        except KeyboardInterrupt:
            log.info("Shutting down...")
            break
        except Exception as e:
            log.error(f"Scan error: {e}")
            time.sleep(60)


if __name__ == '__main__':
    if '--once' in sys.argv:
        run_once()
    else:
        run_daemon()
