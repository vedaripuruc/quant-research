"""
FVG Professional Backtesting Framework
======================================
- Multiple data sources (Polygon, Alpha Vantage, Twelve Data)
- Monte Carlo validation (random entry baseline)
- Statistical significance testing
- Walk-forward validation
- Proper sample size requirements

Focus: FOREX for data granularity
"""

import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import json
import os
import requests
from dataclasses import dataclass
import time

# =============================================================================
# DATA SOURCES
# =============================================================================

class DataFetcher:
    """Multi-source forex data fetcher."""
    
    def __init__(self):
        self.polygon_key = os.getenv('POLYGON_API_KEY')
        self.alpha_key = os.getenv('ALPHA_VANTAGE_KEY')
        self.twelve_key = os.getenv('TWELVE_DATA_KEY')
        self.cache_dir = 'data_cache'
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def fetch_polygon(self, symbol: str, interval: str, days: int) -> pd.DataFrame:
        """Fetch from Polygon.io (2 years free history)."""
        if not self.polygon_key:
            raise ValueError("POLYGON_API_KEY not set")
        
        # Convert interval
        multiplier, timespan = {
            '1m': (1, 'minute'), '5m': (5, 'minute'), '15m': (15, 'minute'),
            '30m': (30, 'minute'), '1h': (1, 'hour'), '4h': (4, 'hour'),
            '1d': (1, 'day'),
        }.get(interval, (1, 'hour'))
        
        # Polygon uses C:EURUSD format for forex
        ticker = f"C:{symbol.replace('=X', '').replace('/', '')}"
        
        end = datetime.now()
        start = end - timedelta(days=days)
        
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
        params = {'apiKey': self.polygon_key, 'limit': 50000, 'sort': 'asc'}
        
        resp = requests.get(url, params=params)
        data = resp.json()
        
        if 'results' not in data:
            raise ValueError(f"Polygon error: {data.get('error', 'Unknown')}")
        
        df = pd.DataFrame(data['results'])
        df['Date'] = pd.to_datetime(df['t'], unit='ms')
        df = df.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'})
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        return df
    
    def fetch_twelve_data(self, symbol: str, interval: str, days: int) -> pd.DataFrame:
        """Fetch from Twelve Data (800 free calls/day)."""
        if not self.twelve_key:
            raise ValueError("TWELVE_DATA_KEY not set")
        
        # Twelve Data uses EUR/USD format
        forex_symbol = symbol.replace('=X', '').replace('USD', '/USD').replace('EUR/', 'EUR/')
        if '/' not in forex_symbol:
            forex_symbol = forex_symbol[:3] + '/' + forex_symbol[3:]
        
        url = "https://api.twelvedata.com/time_series"
        params = {
            'symbol': forex_symbol,
            'interval': interval,
            'outputsize': min(5000, days * 24 if 'h' in interval else days * 24 * 60),
            'apikey': self.twelve_key,
        }
        
        resp = requests.get(url, params=params)
        data = resp.json()
        
        if 'values' not in data:
            raise ValueError(f"Twelve Data error: {data.get('message', 'Unknown')}")
        
        df = pd.DataFrame(data['values'])
        df['Date'] = pd.to_datetime(df['datetime'])
        df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'})
        df[['Open', 'High', 'Low', 'Close']] = df[['Open', 'High', 'Low', 'Close']].astype(float)
        df = df[['Date', 'Open', 'High', 'Low', 'Close']]
        df = df.sort_values('Date').reset_index(drop=True)
        
        return df
    
    def fetch_alpha_vantage(self, symbol: str, interval: str) -> pd.DataFrame:
        """Fetch from Alpha Vantage (full history but limited calls)."""
        if not self.alpha_key:
            raise ValueError("ALPHA_VANTAGE_KEY not set")
        
        forex_symbol = symbol.replace('=X', '')
        from_symbol = forex_symbol[:3]
        to_symbol = forex_symbol[3:]
        
        # Map interval
        av_interval = {'1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min', '1h': '60min'}.get(interval)
        
        if av_interval:
            function = 'FX_INTRADAY'
            params = {
                'function': function,
                'from_symbol': from_symbol,
                'to_symbol': to_symbol,
                'interval': av_interval,
                'outputsize': 'full',
                'apikey': self.alpha_key,
            }
        else:
            function = 'FX_DAILY'
            params = {
                'function': function,
                'from_symbol': from_symbol,
                'to_symbol': to_symbol,
                'outputsize': 'full',
                'apikey': self.alpha_key,
            }
        
        url = 'https://www.alphavantage.co/query'
        resp = requests.get(url, params=params)
        data = resp.json()
        
        # Find the time series key
        ts_key = [k for k in data.keys() if 'Time Series' in k]
        if not ts_key:
            raise ValueError(f"Alpha Vantage error: {data.get('Note', data.get('Error Message', 'Unknown'))}")
        
        ts = data[ts_key[0]]
        df = pd.DataFrame.from_dict(ts, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.reset_index().rename(columns={'index': 'Date'})
        df = df.rename(columns={
            '1. open': 'Open', '2. high': 'High', 
            '3. low': 'Low', '4. close': 'Close'
        })
        df[['Open', 'High', 'Low', 'Close']] = df[['Open', 'High', 'Low', 'Close']].astype(float)
        df = df[['Date', 'Open', 'High', 'Low', 'Close']]
        df = df.sort_values('Date').reset_index(drop=True)
        
        return df
    
    def fetch(self, symbol: str, interval: str, days: int, source: str = 'auto') -> pd.DataFrame:
        """Fetch data with automatic source selection and caching."""
        
        cache_file = f"{self.cache_dir}/{symbol.replace('/', '')}_{interval}_{days}d.parquet"
        
        # Check cache (valid for 1 hour)
        if os.path.exists(cache_file):
            mtime = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if datetime.now() - mtime < timedelta(hours=1):
                return pd.read_parquet(cache_file)
        
        # Try sources in order
        sources = ['polygon', 'twelve', 'alpha'] if source == 'auto' else [source]
        
        for src in sources:
            try:
                if src == 'polygon' and self.polygon_key:
                    df = self.fetch_polygon(symbol, interval, days)
                elif src == 'twelve' and self.twelve_key:
                    df = self.fetch_twelve_data(symbol, interval, days)
                elif src == 'alpha' and self.alpha_key:
                    df = self.fetch_alpha_vantage(symbol, interval)
                else:
                    continue
                
                # Cache it
                df.to_parquet(cache_file)
                print(f"Fetched {symbol} {interval} from {src}: {len(df)} bars")
                return df
                
            except Exception as e:
                print(f"  {src} failed: {e}")
                continue
        
        # Fallback to yfinance
        print(f"  Falling back to yfinance...")
        import yfinance as yf
        end = datetime.now()
        start = end - timedelta(days=days)
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end, interval=interval)
        df.reset_index(inplace=True)
        if 'Datetime' in df.columns:
            df.rename(columns={'Datetime': 'Date'}, inplace=True)
        return df


# =============================================================================
# FVG DETECTION (Optimized)
# =============================================================================

class FVGTracker:
    def __init__(self, min_gap_pct: float = 0.005, max_age: int = 48):
        self.gaps = []
        self.min_gap_pct = min_gap_pct
        self.max_age = max_age
    
    def update(self, df: pd.DataFrame, i: int):
        if i < 2:
            return
        
        current = df.iloc[i]
        bar_2ago = df.iloc[i-2]
        high, low = current['High'], current['Low']
        high_2ago, low_2ago = bar_2ago['High'], bar_2ago['Low']
        
        # Detect new gaps
        if low > high_2ago:
            gap_pct = (low - high_2ago) / high_2ago * 100
            if gap_pct >= self.min_gap_pct:
                self.gaps.append({
                    'type': 'bullish', 'top': low, 'bottom': high_2ago,
                    'mid': (low + high_2ago) / 2, 'size': low - high_2ago,
                    'bar': i, 'filled': False, 'used': False
                })
        elif low_2ago > high:
            gap_pct = (low_2ago - high) / high * 100
            if gap_pct >= self.min_gap_pct:
                self.gaps.append({
                    'type': 'bearish', 'top': low_2ago, 'bottom': high,
                    'mid': (low_2ago + high) / 2, 'size': low_2ago - high,
                    'bar': i, 'filled': False, 'used': False
                })
        
        # Update fills
        for gap in self.gaps:
            if not gap['filled']:
                if gap['type'] == 'bullish' and low <= gap['bottom']:
                    gap['filled'] = True
                elif gap['type'] == 'bearish' and high >= gap['top']:
                    gap['filled'] = True
        
        # Prune old
        self.gaps = [g for g in self.gaps if i - g['bar'] <= self.max_age]


def fvg_magnet_signal(df, i, tracker, sl_mult=1.0, tp_mult=2.0):
    if i < 3:
        return None
    curr, prev = df.iloc[i], df.iloc[i-1]
    for gap in [g for g in tracker.gaps if not g['filled'] and not g['used']]:
        mid = gap['mid']
        if gap['type'] == 'bullish' and prev['Close'] < mid <= curr['Close']:
            gap['used'] = True
            return {'dir': 'long', 'sl': curr['Open'] - sl_mult * gap['size'], 
                    'tp': curr['Open'] + tp_mult * gap['size']}
        elif gap['type'] == 'bearish' and prev['Close'] > mid >= curr['Close']:
            gap['used'] = True
            return {'dir': 'short', 'sl': curr['Open'] + sl_mult * gap['size'],
                    'tp': curr['Open'] - tp_mult * gap['size']}
    return None


def fvg_wall_signal(df, i, tracker, sl_mult=1.0, tp_mult=2.0):
    if i < 3:
        return None
    curr, prev = df.iloc[i], df.iloc[i-1]
    for gap in [g for g in tracker.gaps if not g['filled'] and not g['used']]:
        if gap['type'] == 'bullish':
            if prev['Low'] > gap['top'] >= curr['Low'] and curr['Close'] > gap['top']:
                gap['used'] = True
                return {'dir': 'long', 'sl': gap['bottom'] - sl_mult * gap['size'],
                        'tp': curr['Open'] + tp_mult * gap['size']}
        else:
            if prev['High'] < gap['bottom'] <= curr['High'] and curr['Close'] < gap['bottom']:
                gap['used'] = True
                return {'dir': 'short', 'sl': gap['top'] + sl_mult * gap['size'],
                        'tp': curr['Open'] - tp_mult * gap['size']}
    return None


STRATEGIES = {'magnet': fvg_magnet_signal, 'wall': fvg_wall_signal}


# =============================================================================
# BACKTESTING ENGINE
# =============================================================================

def run_backtest(df: pd.DataFrame, strategy_fn, sl_mult=1.0, tp_mult=2.0,
                 slippage=0.0002, commission=0.0001) -> Dict:
    """Simple vectorized backtest."""
    
    tracker = FVGTracker()
    trades = []
    position = None
    
    for i in range(len(df)):
        tracker.update(df, i)
        bar = df.iloc[i]
        
        # Check exit
        if position:
            if position['dir'] == 'long':
                if bar['Low'] <= position['sl']:
                    pnl = (position['sl'] - position['entry']) / position['entry'] - commission
                    trades.append({'pnl': pnl, 'result': 'loss'})
                    position = None
                elif bar['High'] >= position['tp']:
                    pnl = (position['tp'] - position['entry']) / position['entry'] - commission
                    trades.append({'pnl': pnl, 'result': 'win'})
                    position = None
            else:
                if bar['High'] >= position['sl']:
                    pnl = (position['entry'] - position['sl']) / position['entry'] - commission
                    trades.append({'pnl': pnl, 'result': 'loss'})
                    position = None
                elif bar['Low'] <= position['tp']:
                    pnl = (position['entry'] - position['tp']) / position['entry'] - commission
                    trades.append({'pnl': pnl, 'result': 'win'})
                    position = None
        
        # Check entry
        if not position and i > 2:
            signal = strategy_fn(df, i, tracker, sl_mult, tp_mult)
            if signal:
                entry = bar['Open'] * (1 + slippage if signal['dir'] == 'long' else 1 - slippage)
                position = {'dir': signal['dir'], 'entry': entry, 'sl': signal['sl'], 'tp': signal['tp']}
    
    # Close open position
    if position:
        final = df.iloc[-1]['Close']
        if position['dir'] == 'long':
            pnl = (final - position['entry']) / position['entry'] - commission
        else:
            pnl = (position['entry'] - final) / position['entry'] - commission
        trades.append({'pnl': pnl, 'result': 'open'})
    
    return trades


def calculate_metrics(trades: List[Dict]) -> Dict:
    """Calculate trading metrics."""
    if not trades:
        return {'trades': 0, 'return': 0, 'win_rate': 0, 'pf': 0, 'sharpe': 0}
    
    pnls = [t['pnl'] for t in trades]
    wins = [t for t in trades if t['result'] == 'win']
    losses = [t for t in trades if t['result'] == 'loss']
    
    total_return = (np.prod([1 + p for p in pnls]) - 1) * 100
    win_rate = len(wins) / len(trades) * 100 if trades else 0
    
    total_win = sum(t['pnl'] for t in wins)
    total_loss = abs(sum(t['pnl'] for t in losses))
    pf = total_win / total_loss if total_loss > 0 else 999
    
    sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(252) if len(pnls) > 1 and np.std(pnls) > 0 else 0
    
    return {
        'trades': len(trades),
        'return': round(total_return, 2),
        'win_rate': round(win_rate, 1),
        'pf': round(pf, 2),
        'sharpe': round(sharpe, 2),
    }


# =============================================================================
# MONTE CARLO VALIDATION
# =============================================================================

def random_entry_baseline(df: pd.DataFrame, n_trades: int, sl_pct=0.01, tp_pct=0.02,
                          n_simulations=1000) -> Dict:
    """
    Generate random entry baseline for comparison.
    Returns distribution of returns from random trading.
    """
    random_returns = []
    
    for _ in range(n_simulations):
        trades = []
        # Random entry points
        entry_bars = np.random.choice(range(10, len(df)-10), size=min(n_trades, len(df)//10), replace=False)
        
        for bar_idx in entry_bars:
            bar = df.iloc[bar_idx]
            direction = np.random.choice(['long', 'short'])
            entry = bar['Open']
            
            if direction == 'long':
                sl = entry * (1 - sl_pct)
                tp = entry * (1 + tp_pct)
            else:
                sl = entry * (1 + sl_pct)
                tp = entry * (1 - tp_pct)
            
            # Simulate trade
            for j in range(bar_idx + 1, min(bar_idx + 50, len(df))):
                future = df.iloc[j]
                if direction == 'long':
                    if future['Low'] <= sl:
                        trades.append(-sl_pct)
                        break
                    elif future['High'] >= tp:
                        trades.append(tp_pct)
                        break
                else:
                    if future['High'] >= sl:
                        trades.append(-sl_pct)
                        break
                    elif future['Low'] <= tp:
                        trades.append(tp_pct)
                        break
        
        if trades:
            total_return = (np.prod([1 + p for p in trades]) - 1) * 100
            random_returns.append(total_return)
    
    return {
        'mean': np.mean(random_returns),
        'std': np.std(random_returns),
        'p5': np.percentile(random_returns, 5),
        'p95': np.percentile(random_returns, 95),
        'distribution': random_returns,
    }


def calculate_significance(strategy_return: float, baseline: Dict) -> Dict:
    """Calculate statistical significance vs random baseline."""
    
    z_score = (strategy_return - baseline['mean']) / baseline['std'] if baseline['std'] > 0 else 0
    p_value = 1 - stats.norm.cdf(z_score)  # One-tailed test
    
    return {
        'z_score': round(z_score, 2),
        'p_value': round(p_value, 4),
        'significant_95': p_value < 0.05,
        'significant_99': p_value < 0.01,
        'vs_random': f"{strategy_return:+.2f}% vs {baseline['mean']:+.2f}% random",
    }


# =============================================================================
# WALK-FORWARD VALIDATION
# =============================================================================

def walk_forward_test(df: pd.DataFrame, strategy_fn, 
                      train_pct=0.7, n_windows=5) -> Dict:
    """
    Walk-forward validation with rolling windows.
    """
    total_bars = len(df)
    window_size = total_bars // n_windows
    train_size = int(window_size * train_pct)
    
    results = []
    
    for i in range(n_windows):
        start = i * window_size
        train_end = start + train_size
        test_end = start + window_size
        
        if test_end > total_bars:
            break
        
        # Test on out-of-sample
        test_df = df.iloc[train_end:test_end].reset_index(drop=True)
        trades = run_backtest(test_df, strategy_fn)
        metrics = calculate_metrics(trades)
        
        results.append({
            'window': i + 1,
            'test_bars': len(test_df),
            **metrics
        })
    
    # Aggregate
    if results:
        avg_return = np.mean([r['return'] for r in results])
        consistency = len([r for r in results if r['return'] > 0]) / len(results)
    else:
        avg_return, consistency = 0, 0
    
    return {
        'windows': results,
        'avg_return': round(avg_return, 2),
        'consistency': round(consistency * 100, 1),
    }


# =============================================================================
# MAIN RESEARCH
# =============================================================================

def run_professional_research():
    """Run full professional FVG research."""
    
    print("="*70)
    print("FVG PROFESSIONAL RESEARCH - FOREX FOCUS")
    print("="*70)
    
    fetcher = DataFetcher()
    
    SYMBOLS = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'EURJPY=X', 'AUDUSD=X', 'USDCAD=X']
    INTERVALS = ['15m', '1h', '4h']
    DAYS = {'15m': 30, '1h': 90, '4h': 180}
    
    results = {
        'meta': {'timestamp': datetime.now().isoformat()},
        'by_config': [],
        'summary': {},
    }
    
    for symbol in SYMBOLS:
        print(f"\n{'='*70}")
        print(f"{symbol}")
        print(f"{'='*70}")
        
        for interval in INTERVALS:
            days = DAYS[interval]
            print(f"\n  {interval} ({days}d):")
            
            try:
                df = fetcher.fetch(symbol, interval, days)
                if len(df) < 100:
                    print(f"    Insufficient data: {len(df)} bars")
                    continue
            except Exception as e:
                print(f"    Data error: {e}")
                continue
            
            for strategy_name, strategy_fn in STRATEGIES.items():
                # Run backtest
                trades = run_backtest(df, strategy_fn)
                metrics = calculate_metrics(trades)
                
                if metrics['trades'] < 5:
                    print(f"    {strategy_name:8} Too few trades ({metrics['trades']})")
                    continue
                
                # Monte Carlo baseline
                baseline = random_entry_baseline(df, metrics['trades'], n_simulations=500)
                significance = calculate_significance(metrics['return'], baseline)
                
                # Walk-forward
                wf = walk_forward_test(df, strategy_fn, n_windows=3)
                
                result = {
                    'symbol': symbol,
                    'interval': interval,
                    'strategy': strategy_name,
                    'metrics': metrics,
                    'significance': significance,
                    'walk_forward': wf,
                }
                results['by_config'].append(result)
                
                sig_marker = '***' if significance['significant_99'] else ('**' if significance['significant_95'] else '')
                print(f"    {strategy_name:8} Ret={metrics['return']:+6.2f}% "
                      f"WR={metrics['win_rate']:5.1f}% T={metrics['trades']:3} "
                      f"p={significance['p_value']:.3f}{sig_marker} "
                      f"WF={wf['consistency']:.0f}%")
    
    # Summary
    print("\n" + "="*70)
    print("STATISTICALLY SIGNIFICANT RESULTS (p < 0.05)")
    print("="*70)
    
    significant = [r for r in results['by_config'] if r['significance']['significant_95']]
    significant = sorted(significant, key=lambda x: x['metrics']['return'], reverse=True)
    
    if significant:
        for r in significant[:10]:
            print(f"{r['symbol']:10} {r['interval']:4} {r['strategy']:8} "
                  f"Ret={r['metrics']['return']:+6.2f}% "
                  f"p={r['significance']['p_value']:.4f} "
                  f"WF={r['walk_forward']['consistency']:.0f}%")
    else:
        print("No statistically significant results found.")
    
    # Save
    with open('fvg_professional_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("\nSaved: fvg_professional_results.json")
    
    return results


if __name__ == '__main__':
    run_professional_research()
