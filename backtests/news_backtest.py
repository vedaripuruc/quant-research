"""
News Event Backtest - Prop Firm Challenge Strategy
===================================================
1. Get historical high-impact news events from ForexFactory
2. Fetch granular price data around those events
3. Test pre-restriction entry (before 2-min window)
4. Measure SL/TP outcomes

Strategy:
- Enter position 3-5 minutes before news
- 5% risk, 1:2 RR
- Let it ride through the event
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import json
import os
import sqlite3
from typing import List, Dict, Optional
import yfinance as yf
import time

# =============================================================================
# DATA STORAGE (Local SQLite cache)
# =============================================================================

class PriceCache:
    """Local SQLite cache for price data to avoid API abuse."""
    
    def __init__(self, db_path: str = 'price_cache.db'):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS prices (
                symbol TEXT,
                interval TEXT,
                timestamp INTEGER,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                PRIMARY KEY (symbol, interval, timestamp)
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS news_events (
                id INTEGER PRIMARY KEY,
                timestamp INTEGER,
                currency TEXT,
                event TEXT,
                impact TEXT,
                actual TEXT,
                forecast TEXT,
                previous TEXT
            )
        ''')
        conn.commit()
        conn.close()
    
    def store_prices(self, symbol: str, interval: str, df: pd.DataFrame):
        """Store price data in cache."""
        conn = sqlite3.connect(self.db_path)
        for _, row in df.iterrows():
            ts = int(row['Date'].timestamp()) if hasattr(row['Date'], 'timestamp') else int(pd.Timestamp(row['Date']).timestamp())
            try:
                conn.execute('''
                    INSERT OR REPLACE INTO prices VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (symbol, interval, ts, row['Open'], row['High'], row['Low'], row['Close'], row.get('Volume', 0)))
            except:
                pass
        conn.commit()
        conn.close()
    
    def get_prices(self, symbol: str, interval: str, start_ts: int, end_ts: int) -> pd.DataFrame:
        """Get cached prices."""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query('''
            SELECT timestamp, open, high, low, close, volume 
            FROM prices 
            WHERE symbol = ? AND interval = ? AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp
        ''', conn, params=(symbol, interval, start_ts, end_ts))
        conn.close()
        
        if not df.empty:
            df['Date'] = pd.to_datetime(df['timestamp'], unit='s')
            df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
        
        return df
    
    def store_news_events(self, events: List[Dict]):
        """Store news events."""
        conn = sqlite3.connect(self.db_path)
        for e in events:
            ts = int(e['timestamp'].timestamp()) if hasattr(e['timestamp'], 'timestamp') else e['timestamp']
            try:
                conn.execute('''
                    INSERT OR IGNORE INTO news_events (timestamp, currency, event, impact, actual, forecast, previous)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (ts, e.get('currency'), e.get('event'), e.get('impact'), 
                      e.get('actual'), e.get('forecast'), e.get('previous')))
            except:
                pass
        conn.commit()
        conn.close()


# =============================================================================
# FOREX FACTORY SCRAPER
# =============================================================================

def scrape_forexfactory_week(year: int, week: int) -> List[Dict]:
    """
    Scrape ForexFactory calendar for a specific week.
    Note: ForexFactory may block scrapers - use respectfully.
    """
    # ForexFactory uses week numbers
    url = f"https://www.forexfactory.com/calendar?week={year}{week:02d}"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            print(f"  ForexFactory returned {resp.status_code}")
            return []
        
        soup = BeautifulSoup(resp.text, 'html.parser')
        events = []
        
        # Parse calendar table (structure may change)
        rows = soup.select('tr.calendar__row')
        
        current_date = None
        for row in rows:
            # Get date if present
            date_cell = row.select_one('.calendar__date')
            if date_cell and date_cell.text.strip():
                current_date = date_cell.text.strip()
            
            # Get event details
            currency = row.select_one('.calendar__currency')
            event = row.select_one('.calendar__event')
            impact = row.select_one('.calendar__impact')
            time_cell = row.select_one('.calendar__time')
            
            if event and currency:
                impact_level = 'low'
                if impact:
                    impact_classes = impact.get('class', [])
                    if 'high' in str(impact_classes).lower() or 'icon--ff-impact-red' in str(impact):
                        impact_level = 'high'
                    elif 'medium' in str(impact_classes).lower() or 'icon--ff-impact-ora' in str(impact):
                        impact_level = 'medium'
                
                events.append({
                    'date': current_date,
                    'time': time_cell.text.strip() if time_cell else '',
                    'currency': currency.text.strip(),
                    'event': event.text.strip(),
                    'impact': impact_level,
                })
        
        return events
    
    except Exception as e:
        print(f"  Error scraping ForexFactory: {e}")
        return []


def get_known_high_impact_events() -> List[Dict]:
    """
    Return known high-impact event dates for backtesting.
    More reliable than scraping.
    """
    # Key US events - these are the ones that matter most for EUR/USD, GBP/USD
    events = []
    
    # NFP - First Friday of each month (approximate)
    nfp_dates = [
        '2025-12-06', '2025-11-07', '2025-10-03', '2025-09-05', '2025-08-01',
        '2025-07-04', '2025-06-06', '2025-05-02', '2025-04-04', '2025-03-07',
        '2025-02-07', '2025-01-10',
        '2026-01-03', '2026-02-07',  # Recent
    ]
    
    for d in nfp_dates:
        events.append({
            'timestamp': pd.Timestamp(d + ' 13:30:00', tz='UTC'),  # 8:30 AM ET
            'currency': 'USD',
            'event': 'Non-Farm Payrolls',
            'impact': 'high',
        })
    
    # FOMC - 8 meetings per year
    fomc_dates = [
        '2025-12-18', '2025-11-07', '2025-09-18', '2025-07-30',
        '2025-06-18', '2025-05-07', '2025-03-19', '2025-01-29',
        '2026-01-29',  # Next one
    ]
    
    for d in fomc_dates:
        events.append({
            'timestamp': pd.Timestamp(d + ' 19:00:00', tz='UTC'),  # 2:00 PM ET
            'currency': 'USD',
            'event': 'FOMC Rate Decision',
            'impact': 'high',
        })
    
    # CPI - Around 10th-15th of each month
    cpi_dates = [
        '2025-12-11', '2025-11-13', '2025-10-10', '2025-09-11', '2025-08-14',
        '2025-07-11', '2025-06-11', '2025-05-13', '2025-04-10', '2025-03-12',
        '2025-02-12', '2025-01-15',
        '2026-01-15', '2026-02-12',
    ]
    
    for d in cpi_dates:
        events.append({
            'timestamp': pd.Timestamp(d + ' 13:30:00', tz='UTC'),  # 8:30 AM ET
            'currency': 'USD',
            'event': 'CPI',
            'impact': 'high',
        })
    
    return events


# =============================================================================
# PRICE DATA FETCHING
# =============================================================================

def fetch_price_data_around_event(
    symbol: str,
    event_time: pd.Timestamp,
    minutes_before: int = 30,
    minutes_after: int = 60,
    interval: str = '1m',
    cache: Optional[PriceCache] = None
) -> pd.DataFrame:
    """
    Fetch minute-level data around a news event.
    Uses cache if available.
    """
    start = event_time - timedelta(minutes=minutes_before)
    end = event_time + timedelta(minutes=minutes_after)
    
    # Check cache first
    if cache:
        cached = cache.get_prices(symbol, interval, int(start.timestamp()), int(end.timestamp()))
        if len(cached) > 10:  # Got enough cached data
            return cached
    
    # Fetch from yfinance (limited to 7 days for 1m data)
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end, interval=interval)
        df.reset_index(inplace=True)
        
        if 'Datetime' in df.columns:
            df.rename(columns={'Datetime': 'Date'}, inplace=True)
        
        # Cache it
        if cache and not df.empty:
            cache.store_prices(symbol, interval, df)
        
        return df
    
    except Exception as e:
        print(f"  Error fetching {symbol}: {e}")
        return pd.DataFrame()


# =============================================================================
# BACKTEST LOGIC
# =============================================================================

def simulate_news_trade(
    df: pd.DataFrame,
    event_time: pd.Timestamp,
    entry_minutes_before: int = 3,
    risk_pct: float = 0.05,
    rr_ratio: float = 2.0,
    direction: str = 'long'  # or 'short' or 'random'
) -> Dict:
    """
    Simulate a single news trade.
    
    Entry: X minutes before news
    SL/TP: Based on risk_pct and RR
    """
    if df.empty:
        return {'result': 'no_data'}
    
    # Find entry bar (X minutes before event)
    entry_time = event_time - timedelta(minutes=entry_minutes_before)
    
    # Find closest bar to entry time
    df['time_diff'] = abs(df['Date'] - entry_time)
    entry_idx = df['time_diff'].idxmin()
    
    if pd.isna(entry_idx):
        return {'result': 'no_entry_bar'}
    
    entry_bar = df.loc[entry_idx]
    entry_price = entry_bar['Close']
    
    # Determine direction
    if direction == 'random':
        direction = np.random.choice(['long', 'short'])
    
    # Calculate SL/TP
    if direction == 'long':
        sl = entry_price * (1 - risk_pct)
        tp = entry_price * (1 + risk_pct * rr_ratio)
    else:
        sl = entry_price * (1 + risk_pct)
        tp = entry_price * (1 - risk_pct * rr_ratio)
    
    # Simulate trade outcome
    result = 'open'
    exit_price = None
    exit_time = None
    
    for idx in range(entry_idx + 1, len(df)):
        bar = df.iloc[idx]
        
        if direction == 'long':
            if bar['Low'] <= sl:
                result = 'loss'
                exit_price = sl
                exit_time = bar['Date']
                break
            elif bar['High'] >= tp:
                result = 'win'
                exit_price = tp
                exit_time = bar['Date']
                break
        else:
            if bar['High'] >= sl:
                result = 'loss'
                exit_price = sl
                exit_time = bar['Date']
                break
            elif bar['Low'] <= tp:
                result = 'win'
                exit_price = tp
                exit_time = bar['Date']
                break
    
    # If still open at end of data, use last close
    if result == 'open':
        exit_price = df.iloc[-1]['Close']
        exit_time = df.iloc[-1]['Date']
        if direction == 'long':
            result = 'win' if exit_price > entry_price else 'loss'
        else:
            result = 'win' if exit_price < entry_price else 'loss'
    
    return {
        'result': result,
        'direction': direction,
        'entry_price': entry_price,
        'entry_time': entry_bar['Date'],
        'exit_price': exit_price,
        'exit_time': exit_time,
        'sl': sl,
        'tp': tp,
        'pnl_pct': (exit_price - entry_price) / entry_price * 100 if direction == 'long' else (entry_price - exit_price) / entry_price * 100,
    }


def run_news_backtest(
    symbols: List[str] = ['EURUSD=X', 'GBPUSD=X'],
    entry_minutes_before: int = 3,
    risk_pct: float = 0.05,
    rr_ratio: float = 2.0,
    direction: str = 'random',
    max_events: int = 50
) -> Dict:
    """
    Run full backtest across multiple events and symbols.
    """
    events = get_known_high_impact_events()
    
    # Filter to recent events (within yfinance 1m data range - 7 days)
    cutoff = datetime.now() - timedelta(days=7)
    recent_events = [e for e in events if e['timestamp'].replace(tzinfo=None) > cutoff]
    
    if not recent_events:
        print("No recent events within 7-day window. Using older events (may lack 1m data).")
        # Use events anyway, will rely on cache or hourly data
        recent_events = events[:max_events]
    
    cache = PriceCache()
    results = []
    
    print(f"\nBacktesting {len(recent_events)} events across {len(symbols)} symbols...")
    print(f"Strategy: Enter {entry_minutes_before}min before, {risk_pct*100:.1f}% risk, 1:{rr_ratio} RR, {direction} direction")
    print("-" * 70)
    
    for event in recent_events[:max_events]:
        event_time = event['timestamp']
        event_name = event['event']
        
        for symbol in symbols:
            # Fetch data
            df = fetch_price_data_around_event(symbol, event_time, cache=cache)
            
            if df.empty:
                continue
            
            # Simulate trade
            trade = simulate_news_trade(
                df, event_time, entry_minutes_before, risk_pct, rr_ratio, direction
            )
            
            if trade['result'] in ['win', 'loss']:
                trade['symbol'] = symbol
                trade['event'] = event_name
                trade['event_time'] = event_time
                results.append(trade)
                
                status = '✓' if trade['result'] == 'win' else '✗'
                print(f"  {status} {event_time.strftime('%Y-%m-%d')} {event_name[:15]:15} {symbol:10} {trade['direction']:5} {trade['pnl_pct']:+.2f}%")
    
    # Calculate stats
    if not results:
        return {'error': 'No trades executed'}
    
    wins = [r for r in results if r['result'] == 'win']
    losses = [r for r in results if r['result'] == 'loss']
    
    win_rate = len(wins) / len(results) * 100
    avg_win = np.mean([r['pnl_pct'] for r in wins]) if wins else 0
    avg_loss = np.mean([r['pnl_pct'] for r in losses]) if losses else 0
    total_pnl = sum(r['pnl_pct'] for r in results)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Total trades: {len(results)}")
    print(f"Wins: {len(wins)} | Losses: {len(losses)}")
    print(f"Win rate: {win_rate:.1f}%")
    print(f"Avg win: {avg_win:.2f}% | Avg loss: {avg_loss:.2f}%")
    print(f"Total P&L: {total_pnl:.2f}%")
    
    # Challenge simulation
    print("\n" + "=" * 70)
    print("CHALLENGE SIMULATION")
    print("=" * 70)
    
    # Simulate with 5% risk per trade
    equity = 0
    max_dd = 0
    peak = 0
    
    for r in results:
        if r['result'] == 'win':
            equity += risk_pct * rr_ratio * 100
        else:
            equity -= risk_pct * 100
        
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd
        
        # Check barriers
        if equity >= 10:
            print(f"  ✓ PASSED at {equity:.1f}% after {results.index(r)+1} trades")
            break
        elif equity <= -10:
            print(f"  ✗ BLOWN at {equity:.1f}% after {results.index(r)+1} trades")
            break
    else:
        print(f"  ? Ended at {equity:.1f}% (neither passed nor blown)")
    
    print(f"  Max drawdown: {max_dd:.1f}%")
    
    return {
        'trades': results,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'n_trades': len(results),
    }


if __name__ == '__main__':
    print("=" * 70)
    print("NEWS EVENT BACKTEST - PROP FIRM STRATEGY")
    print("=" * 70)
    
    # Run backtest
    results = run_news_backtest(
        symbols=['EURUSD=X', 'GBPUSD=X', 'USDJPY=X'],
        entry_minutes_before=3,
        risk_pct=0.05,
        rr_ratio=2.0,
        direction='random',
        max_events=20
    )
    
    # Save results
    with open('news_backtest_results.json', 'w') as f:
        json.dump({
            'win_rate': results.get('win_rate'),
            'total_pnl': results.get('total_pnl'),
            'n_trades': results.get('n_trades'),
        }, f, indent=2, default=str)
    
    print("\nSaved: news_backtest_results.json")
