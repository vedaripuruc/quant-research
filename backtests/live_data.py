#!/usr/bin/env python3
"""
Live Forex Data Sources
-----------------------
Multiple free sources for real-time(ish) forex data.

Priority:
1. Finnhub WebSocket (real-time, free)
2. Twelve Data REST (near real-time, free tier)
3. Yahoo Finance (15-20min delayed, fallback)
"""

import os
import json
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List
import requests

# ============================================================================
# FINNHUB - Real-time WebSocket (FREE)
# ============================================================================
# Get free API key at: https://finnhub.io/register
# Free tier: 60 API calls/min, real-time websocket for forex

FINNHUB_API_KEY = os.environ.get('FINNHUB_API_KEY', '')

def get_finnhub_quote(symbol: str) -> Optional[Dict]:
    """
    Get current quote from Finnhub REST API.
    Symbol format: OANDA:EUR_USD
    """
    if not FINNHUB_API_KEY:
        return None
    
    # Convert symbol format
    # EURUSD=X -> OANDA:EUR_USD
    clean = symbol.replace('=X', '').replace('/', '')
    if len(clean) == 6:
        finnhub_symbol = f"OANDA:{clean[:3]}_{clean[3:]}"
    else:
        finnhub_symbol = symbol
    
    try:
        url = f"https://finnhub.io/api/v1/quote"
        params = {
            'symbol': finnhub_symbol,
            'token': FINNHUB_API_KEY
        }
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        
        if 'c' in data and data['c'] > 0:
            return {
                'symbol': symbol,
                'price': data['c'],  # Current price
                'high': data['h'],   # Day high
                'low': data['l'],    # Day low
                'open': data['o'],   # Day open
                'prev_close': data['pc'],
                'timestamp': datetime.now().isoformat(),
                'source': 'finnhub'
            }
    except Exception as e:
        print(f"Finnhub error for {symbol}: {e}")
    
    return None


def get_finnhub_candles(symbol: str, resolution: str = '60', count: int = 100) -> Optional[List]:
    """
    Get historical candles from Finnhub.
    Resolution: 1, 5, 15, 30, 60, D, W, M
    """
    if not FINNHUB_API_KEY:
        return None
    
    clean = symbol.replace('=X', '').replace('/', '')
    if len(clean) == 6:
        finnhub_symbol = f"OANDA:{clean[:3]}_{clean[3:]}"
    else:
        finnhub_symbol = symbol
    
    try:
        now = int(time.time())
        # Calculate start time based on count and resolution
        if resolution == '60':
            start = now - (count * 3600)
        elif resolution == 'D':
            start = now - (count * 86400)
        else:
            start = now - (count * 3600)  # Default to hourly
        
        url = f"https://finnhub.io/api/v1/forex/candle"
        params = {
            'symbol': finnhub_symbol,
            'resolution': resolution,
            'from': start,
            'to': now,
            'token': FINNHUB_API_KEY
        }
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        
        if data.get('s') == 'ok' and 'c' in data:
            candles = []
            for i in range(len(data['c'])):
                candles.append({
                    'time': datetime.fromtimestamp(data['t'][i]).isoformat(),
                    'open': data['o'][i],
                    'high': data['h'][i],
                    'low': data['l'][i],
                    'close': data['c'][i],
                    'volume': data.get('v', [0] * len(data['c']))[i]
                })
            return candles
    except Exception as e:
        print(f"Finnhub candles error for {symbol}: {e}")
    
    return None


# ============================================================================
# TWELVE DATA - Near real-time REST (FREE tier: 800/day)
# ============================================================================
# Get free API key at: https://twelvedata.com/account/api-keys

TWELVEDATA_API_KEY = os.environ.get('TWELVEDATA_API_KEY', '')

def get_twelvedata_quote(symbol: str) -> Optional[Dict]:
    """Get current quote from Twelve Data."""
    if not TWELVEDATA_API_KEY:
        return None
    
    # Convert symbol format: EURUSD=X -> EUR/USD
    clean = symbol.replace('=X', '')
    if len(clean) == 6 and '/' not in clean:
        td_symbol = f"{clean[:3]}/{clean[3:]}"
    else:
        td_symbol = clean
    
    try:
        url = "https://api.twelvedata.com/price"
        params = {
            'symbol': td_symbol,
            'apikey': TWELVEDATA_API_KEY
        }
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        
        if 'price' in data:
            return {
                'symbol': symbol,
                'price': float(data['price']),
                'timestamp': datetime.now().isoformat(),
                'source': 'twelvedata'
            }
    except Exception as e:
        print(f"TwelveData error for {symbol}: {e}")
    
    return None


def get_twelvedata_candles(symbol: str, interval: str = '1h', count: int = 100) -> Optional[List]:
    """Get historical candles from Twelve Data."""
    if not TWELVEDATA_API_KEY:
        return None
    
    clean = symbol.replace('=X', '')
    if len(clean) == 6 and '/' not in clean:
        td_symbol = f"{clean[:3]}/{clean[3:]}"
    else:
        td_symbol = clean
    
    try:
        url = "https://api.twelvedata.com/time_series"
        params = {
            'symbol': td_symbol,
            'interval': interval,
            'outputsize': count,
            'apikey': TWELVEDATA_API_KEY
        }
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        
        if 'values' in data:
            candles = []
            for v in reversed(data['values']):  # Oldest first
                candles.append({
                    'time': v['datetime'],
                    'open': float(v['open']),
                    'high': float(v['high']),
                    'low': float(v['low']),
                    'close': float(v['close']),
                    'volume': 0
                })
            return candles
    except Exception as e:
        print(f"TwelveData candles error for {symbol}: {e}")
    
    return None


# ============================================================================
# YAHOO FINANCE - Fallback (15-20min delayed)
# ============================================================================

def get_yfinance_quote(symbol: str) -> Optional[Dict]:
    """Fallback to yfinance for delayed quotes."""
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        if 'regularMarketPrice' in info:
            return {
                'symbol': symbol,
                'price': info['regularMarketPrice'],
                'high': info.get('dayHigh', 0),
                'low': info.get('dayLow', 0),
                'open': info.get('regularMarketOpen', 0),
                'prev_close': info.get('regularMarketPreviousClose', 0),
                'timestamp': datetime.now().isoformat(),
                'source': 'yfinance',
                'delayed': True
            }
    except Exception as e:
        print(f"yfinance error for {symbol}: {e}")
    
    return None


def get_yfinance_candles(symbol: str, interval: str = '1h', days: int = 7) -> Optional[List]:
    """Get candles from yfinance."""
    try:
        import yfinance as yf
        from datetime import timedelta
        
        end = datetime.now()
        start = end - timedelta(days=days)
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start.strftime('%Y-%m-%d'),
                           end=end.strftime('%Y-%m-%d'),
                           interval=interval)
        
        if df.empty:
            return None
        
        candles = []
        for idx, row in df.iterrows():
            candles.append({
                'time': str(idx),
                'open': row['Open'],
                'high': row['High'],
                'low': row['Low'],
                'close': row['Close'],
                'volume': row['Volume']
            })
        return candles
    except Exception as e:
        print(f"yfinance candles error for {symbol}: {e}")
    
    return None


# ============================================================================
# UNIFIED INTERFACE
# ============================================================================

def get_live_quote(symbol: str) -> Optional[Dict]:
    """
    Get live quote, trying sources in order of freshness.
    """
    # Try Finnhub first (real-time)
    quote = get_finnhub_quote(symbol)
    if quote:
        return quote
    
    # Try Twelve Data (near real-time)
    quote = get_twelvedata_quote(symbol)
    if quote:
        return quote
    
    # Fallback to yfinance (delayed)
    quote = get_yfinance_quote(symbol)
    if quote:
        return quote
    
    return None


def get_live_candles(symbol: str, interval: str = '1h', count: int = 100) -> Optional[List]:
    """
    Get historical candles, trying sources in order.
    """
    # Try Finnhub first
    if interval in ['60', '1h']:
        candles = get_finnhub_candles(symbol, '60', count)
        if candles:
            return candles
    
    # Try Twelve Data
    candles = get_twelvedata_candles(symbol, interval, count)
    if candles:
        return candles
    
    # Fallback to yfinance
    days = max(7, count // 24 + 2)
    candles = get_yfinance_candles(symbol, interval, days)
    if candles:
        return candles[-count:]  # Limit to requested count
    
    return None


def check_api_keys():
    """Check which API keys are configured."""
    status = {
        'finnhub': bool(FINNHUB_API_KEY),
        'twelvedata': bool(TWELVEDATA_API_KEY),
        'yfinance': True  # Always available
    }
    return status


# ============================================================================
# TEST
# ============================================================================

if __name__ == '__main__':
    print("Live Data Source Check")
    print("=" * 50)
    
    status = check_api_keys()
    print(f"Finnhub API Key: {'✓ Set' if status['finnhub'] else '✗ Not set (FINNHUB_API_KEY)'}")
    print(f"TwelveData API Key: {'✓ Set' if status['twelvedata'] else '✗ Not set (TWELVEDATA_API_KEY)'}")
    print(f"yfinance: ✓ Available (fallback, delayed)")
    
    print("\n" + "=" * 50)
    print("Testing EUR/USD quote...")
    
    quote = get_live_quote('EURUSD=X')
    if quote:
        print(f"  Source: {quote['source']}")
        print(f"  Price: {quote['price']}")
        print(f"  Time: {quote['timestamp']}")
    else:
        print("  Failed to get quote")
    
    print("\nTo enable real-time data, set environment variables:")
    print("  export FINNHUB_API_KEY='your_key'  # Get free at finnhub.io")
    print("  export TWELVEDATA_API_KEY='your_key'  # Get free at twelvedata.com")
