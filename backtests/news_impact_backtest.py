"""
High-Impact News Trading Backtest
=================================
Strategy: Trade only during high-volatility economic events
Goal: Pass FTMO challenge using variance, not alpha

FTMO Rules:
- Profit Target: 10%
- Daily Drawdown: 5%
- Max Drawdown: 10%
- Min Trading Days: 4

Our Approach:
- Trade only NFP, CPI, FOMC, etc.
- 5% risk per trade (full daily DD allowance)
- 1:2 RR target (10% reward per win)
- Don't need edge - structure is +EV
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import json
import requests

# =============================================================================
# ECONOMIC CALENDAR
# =============================================================================

# High-impact events that move markets
HIGH_IMPACT_EVENTS = {
    "NFP": {
        "name": "Non-Farm Payrolls",
        "frequency": "monthly",
        "day": "first_friday",
        "time": "08:30",
        "timezone": "America/New_York",
        "affected": ["EURUSD", "USDJPY", "GBPUSD", "XAUUSD", "US30"],
        "expected_move_pips": {"EURUSD": 50, "USDJPY": 70, "GBPUSD": 60}
    },
    "CPI": {
        "name": "Consumer Price Index",
        "frequency": "monthly", 
        "day": "~13th",
        "time": "08:30",
        "timezone": "America/New_York",
        "affected": ["EURUSD", "USDJPY", "XAUUSD", "US30"],
        "expected_move_pips": {"EURUSD": 40, "USDJPY": 50}
    },
    "FOMC": {
        "name": "Federal Reserve Decision",
        "frequency": "8x_year",
        "time": "14:00",
        "timezone": "America/New_York",
        "affected": ["EURUSD", "USDJPY", "GBPUSD", "XAUUSD", "US30", "NAS100"],
        "expected_move_pips": {"EURUSD": 80, "USDJPY": 100}
    },
    "ECB": {
        "name": "ECB Interest Rate Decision",
        "frequency": "6x_year",
        "time": "13:45",
        "timezone": "Europe/Frankfurt",
        "affected": ["EURUSD", "EURGBP", "EURJPY"],
        "expected_move_pips": {"EURUSD": 60}
    },
    "BOE": {
        "name": "Bank of England Decision",
        "frequency": "8x_year",
        "time": "12:00",
        "timezone": "Europe/London",
        "affected": ["GBPUSD", "EURGBP", "GBPJPY"],
        "expected_move_pips": {"GBPUSD": 70}
    },
    "GDP": {
        "name": "GDP Release",
        "frequency": "quarterly",
        "time": "08:30",
        "timezone": "America/New_York",
        "affected": ["EURUSD", "USDJPY", "US30"],
        "expected_move_pips": {"EURUSD": 30, "USDJPY": 40}
    }
}

# Historical high-impact dates for 2024-2025 (for backtesting)
# Format: (date, event_type, actual_move_pips_eurusd)
HISTORICAL_EVENTS_2024_2025 = [
    # 2024
    ("2024-01-05", "NFP", 45),
    ("2024-01-11", "CPI", 38),
    ("2024-01-31", "FOMC", 65),
    ("2024-02-02", "NFP", 52),
    ("2024-02-13", "CPI", 55),
    ("2024-03-08", "NFP", 48),
    ("2024-03-12", "CPI", 42),
    ("2024-03-20", "FOMC", 72),
    ("2024-04-05", "NFP", 40),
    ("2024-04-10", "CPI", 35),
    ("2024-05-01", "FOMC", 58),
    ("2024-05-03", "NFP", 62),
    ("2024-05-15", "CPI", 48),
    ("2024-06-07", "NFP", 55),
    ("2024-06-12", "FOMC", 85),
    ("2024-06-12", "CPI", 52),
    ("2024-07-05", "NFP", 38),
    ("2024-07-11", "CPI", 45),
    ("2024-07-31", "FOMC", 68),
    ("2024-08-02", "NFP", 72),
    ("2024-08-14", "CPI", 40),
    ("2024-09-06", "NFP", 58),
    ("2024-09-11", "CPI", 35),
    ("2024-09-18", "FOMC", 95),  # Rate cut
    ("2024-10-04", "NFP", 65),
    ("2024-10-10", "CPI", 42),
    ("2024-11-01", "NFP", 48),
    ("2024-11-07", "FOMC", 55),
    ("2024-11-13", "CPI", 38),
    ("2024-12-06", "NFP", 52),
    ("2024-12-11", "CPI", 45),
    ("2024-12-18", "FOMC", 78),
    # 2025
    ("2025-01-10", "NFP", 58),
    ("2025-01-15", "CPI", 50),
    ("2025-01-29", "FOMC", 62),
]


@dataclass
class Trade:
    entry_time: datetime
    direction: str  # "LONG" or "SHORT"
    entry_price: float
    stop_loss: float
    take_profit: float
    event_type: str
    symbol: str
    risk_pct: float = 5.0  # % of account risked
    
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl_pct: Optional[float] = None
    result: Optional[str] = None  # "WIN", "LOSS", "TIMEOUT"


@dataclass 
class ChallengeState:
    starting_balance: float = 100000
    current_balance: float = 100000
    high_water_mark: float = 100000
    daily_start_balance: float = 100000
    current_day: Optional[str] = None
    trades: List[Trade] = None
    trading_days: set = None
    
    # FTMO limits
    profit_target_pct: float = 10.0
    daily_dd_limit_pct: float = 5.0
    max_dd_limit_pct: float = 10.0
    min_trading_days: int = 4
    
    # Status
    passed: bool = False
    failed: bool = False
    fail_reason: Optional[str] = None
    
    def __post_init__(self):
        if self.trades is None:
            self.trades = []
        if self.trading_days is None:
            self.trading_days = set()
    
    @property
    def current_profit_pct(self) -> float:
        return ((self.current_balance - self.starting_balance) / self.starting_balance) * 100
    
    @property
    def daily_dd_pct(self) -> float:
        return ((self.daily_start_balance - self.current_balance) / self.starting_balance) * 100
    
    @property
    def max_dd_pct(self) -> float:
        return ((self.high_water_mark - self.current_balance) / self.starting_balance) * 100
    
    def new_day(self, date: str):
        if self.current_day != date:
            self.current_day = date
            self.daily_start_balance = self.current_balance
    
    def check_limits(self) -> bool:
        """Check if challenge limits are breached. Returns True if still valid."""
        # Check daily drawdown
        if self.daily_dd_pct >= self.daily_dd_limit_pct:
            self.failed = True
            self.fail_reason = f"Daily DD exceeded: {self.daily_dd_pct:.2f}%"
            return False
        
        # Check max drawdown
        if self.max_dd_pct >= self.max_dd_limit_pct:
            self.failed = True
            self.fail_reason = f"Max DD exceeded: {self.max_dd_pct:.2f}%"
            return False
        
        return True
    
    def check_passed(self) -> bool:
        """Check if challenge is passed."""
        if self.current_profit_pct >= self.profit_target_pct:
            if len(self.trading_days) >= self.min_trading_days:
                self.passed = True
                return True
        return False
    
    def apply_trade(self, trade: Trade):
        """Apply trade result to challenge state."""
        self.trades.append(trade)
        self.trading_days.add(trade.entry_time.strftime("%Y-%m-%d"))
        
        if trade.pnl_pct:
            pnl_amount = self.current_balance * (trade.pnl_pct / 100)
            self.current_balance += pnl_amount
            
            # Update high water mark
            if self.current_balance > self.high_water_mark:
                self.high_water_mark = self.current_balance


# =============================================================================
# DATA FETCHING
# =============================================================================

def fetch_price_data(symbol: str, start_date: str, end_date: str, interval: str = "1h") -> pd.DataFrame:
    """Fetch OHLCV data from yfinance."""
    # Convert forex symbols for yfinance
    yf_symbol = symbol
    if symbol in ["EURUSD", "USDJPY", "GBPUSD", "EURJPY", "GBPJPY", "AUDUSD"]:
        yf_symbol = f"{symbol}=X"
    elif symbol == "XAUUSD":
        yf_symbol = "GC=F"  # Gold futures
    elif symbol == "US30":
        yf_symbol = "YM=F"  # Dow futures
    elif symbol == "NAS100":
        yf_symbol = "NQ=F"  # Nasdaq futures
    
    ticker = yf.Ticker(yf_symbol)
    df = ticker.history(start=start_date, end=end_date, interval=interval)
    
    if df.empty:
        print(f"Warning: No data for {symbol}")
        return pd.DataFrame()
    
    df = df.rename(columns={
        'Open': 'open', 'High': 'high', 'Low': 'low', 
        'Close': 'close', 'Volume': 'volume'
    })
    
    return df[['open', 'high', 'low', 'close', 'volume']]


def get_event_candles(df: pd.DataFrame, event_date: str, hours_before: int = 1, hours_after: int = 4) -> pd.DataFrame:
    """Get candles around an event."""
    event_dt = pd.to_datetime(event_date)
    
    # Make timezone-aware if needed
    if df.index.tz is not None:
        event_dt = event_dt.tz_localize(df.index.tz)
    
    # Find candles in the window
    start_time = event_dt - timedelta(hours=hours_before)
    end_time = event_dt + timedelta(hours=hours_after)
    
    mask = (df.index >= start_time) & (df.index <= end_time)
    return df[mask]


# =============================================================================
# TRADING LOGIC
# =============================================================================

def calculate_event_volatility(df: pd.DataFrame, event_time: datetime) -> Dict:
    """Calculate volatility metrics around an event."""
    # Make event_time timezone-aware if df index is timezone-aware
    if df.index.tz is not None:
        event_time = pd.Timestamp(event_time).tz_localize(df.index.tz)
    
    # Pre-event (1 hour before)
    pre_event = df[df.index < event_time].tail(4)  # 4 x 15min or 1 x 1h
    
    # Post-event (4 hours after)
    post_event = df[df.index >= event_time].head(16)  # 16 x 15min or 4 x 1h
    
    if pre_event.empty or post_event.empty:
        return None
    
    pre_range = pre_event['high'].max() - pre_event['low'].min()
    post_range = post_event['high'].max() - post_event['low'].min()
    
    # Direction of move
    event_open = post_event.iloc[0]['open']
    event_close = post_event.iloc[-1]['close']
    move_direction = "UP" if event_close > event_open else "DOWN"
    move_pips = abs(event_close - event_open) * 10000 if "JPY" not in str(df.index[0]) else abs(event_close - event_open) * 100
    
    return {
        "pre_range": pre_range,
        "post_range": post_range,
        "volatility_expansion": post_range / pre_range if pre_range > 0 else 0,
        "move_direction": move_direction,
        "move_pips": move_pips,
        "event_open": event_open,
        "event_close": event_close
    }


def simulate_straddle_trade(
    df: pd.DataFrame, 
    event_time: datetime,
    risk_pct: float = 5.0,
    rr_ratio: float = 2.0,
    atr_multiple_sl: float = 1.5
) -> Tuple[Optional[Trade], Optional[Trade]]:
    """
    Simulate a straddle trade around news event.
    Places both LONG and SHORT pending orders at recent highs/lows.
    One gets triggered, the other gets cancelled.
    """
    # Make timezone-aware if needed
    if df.index.tz is not None and event_time.tzinfo is None:
        event_time = pd.Timestamp(event_time).tz_localize(df.index.tz)
    
    # Get pre-event candles for entry levels
    pre_event = df[df.index < event_time].tail(4)
    
    if len(pre_event) < 2:
        return None, None
    
    # Entry levels: recent high and low
    recent_high = pre_event['high'].max()
    recent_low = pre_event['low'].min()
    range_size = recent_high - recent_low
    
    if range_size == 0:
        return None, None
    
    # ATR for stop sizing
    atr = range_size  # Simplified: use recent range as ATR proxy
    
    # Long trade setup
    long_entry = recent_high + (atr * 0.1)  # Slight buffer above high
    long_sl = long_entry - (atr * atr_multiple_sl)
    long_tp = long_entry + (atr * atr_multiple_sl * rr_ratio)
    
    # Short trade setup  
    short_entry = recent_low - (atr * 0.1)  # Slight buffer below low
    short_sl = short_entry + (atr * atr_multiple_sl)
    short_tp = short_entry - (atr * atr_multiple_sl * rr_ratio)
    
    long_trade = Trade(
        entry_time=event_time,
        direction="LONG",
        entry_price=long_entry,
        stop_loss=long_sl,
        take_profit=long_tp,
        event_type="NEWS",
        symbol="EURUSD",
        risk_pct=risk_pct
    )
    
    short_trade = Trade(
        entry_time=event_time,
        direction="SHORT",
        entry_price=short_entry,
        stop_loss=short_sl,
        take_profit=short_tp,
        event_type="NEWS",
        symbol="EURUSD",
        risk_pct=risk_pct
    )
    
    return long_trade, short_trade


def execute_trade(trade: Trade, df: pd.DataFrame, max_hours: int = 4) -> Trade:
    """Execute a trade against price data and determine outcome."""
    entry_time = trade.entry_time
    
    # Make timezone-aware if needed
    if df.index.tz is not None and (not hasattr(entry_time, 'tzinfo') or entry_time.tzinfo is None):
        entry_time = pd.Timestamp(entry_time).tz_localize(df.index.tz)
    
    # Find candles after entry time
    post_entry = df[df.index >= entry_time].head(max_hours * 4)  # Assuming 15min candles
    
    if post_entry.empty:
        trade.result = "NO_DATA"
        return trade
    
    # Check if entry was triggered
    entry_triggered = False
    for idx, candle in post_entry.iterrows():
        if trade.direction == "LONG" and candle['high'] >= trade.entry_price:
            entry_triggered = True
            trade.entry_time = idx
            break
        elif trade.direction == "SHORT" and candle['low'] <= trade.entry_price:
            entry_triggered = True
            trade.entry_time = idx
            break
    
    if not entry_triggered:
        trade.result = "NOT_TRIGGERED"
        return trade
    
    # Now check for SL or TP hit
    post_trigger = df[df.index >= trade.entry_time]
    
    for idx, candle in post_trigger.iterrows():
        if trade.direction == "LONG":
            # Check SL first (worst case)
            if candle['low'] <= trade.stop_loss:
                trade.exit_time = idx
                trade.exit_price = trade.stop_loss
                trade.result = "LOSS"
                trade.pnl_pct = -trade.risk_pct
                return trade
            # Check TP
            if candle['high'] >= trade.take_profit:
                trade.exit_time = idx
                trade.exit_price = trade.take_profit
                trade.result = "WIN"
                trade.pnl_pct = trade.risk_pct * 2  # 1:2 RR
                return trade
        else:  # SHORT
            # Check SL first
            if candle['high'] >= trade.stop_loss:
                trade.exit_time = idx
                trade.exit_price = trade.stop_loss
                trade.result = "LOSS"
                trade.pnl_pct = -trade.risk_pct
                return trade
            # Check TP
            if candle['low'] <= trade.take_profit:
                trade.exit_time = idx
                trade.exit_price = trade.take_profit
                trade.result = "WIN"
                trade.pnl_pct = trade.risk_pct * 2
                return trade
    
    # Timeout - close at last price
    trade.exit_time = post_trigger.index[-1]
    trade.exit_price = post_trigger.iloc[-1]['close']
    
    if trade.direction == "LONG":
        pnl_raw = (trade.exit_price - trade.entry_price) / (trade.entry_price - trade.stop_loss)
    else:
        pnl_raw = (trade.entry_price - trade.exit_price) / (trade.stop_loss - trade.entry_price)
    
    trade.pnl_pct = pnl_raw * trade.risk_pct
    trade.result = "TIMEOUT"
    
    return trade


# =============================================================================
# SIMULATION
# =============================================================================

def simulate_challenge(
    events: List[Tuple[str, str, int]],
    symbol: str = "EURUSD",
    risk_pct: float = 5.0,
    rr_ratio: float = 2.0,
    direction_strategy: str = "straddle"  # "straddle", "random", "trend"
) -> ChallengeState:
    """
    Simulate an FTMO challenge trading only high-impact news events.
    
    direction_strategy:
    - "straddle": Place both long and short pending orders, one triggers
    - "random": Random direction (50/50)
    - "trend": Follow the initial move direction
    """
    # Fetch data
    start_date = events[0][0]
    end_date = events[-1][0]
    
    # Extend date range for data availability
    start_dt = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=7)
    end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=7)
    
    df = fetch_price_data(symbol, start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"), "1h")
    
    if df.empty:
        print("No data available")
        return None
    
    challenge = ChallengeState()
    
    for event_date, event_type, expected_move in events:
        if challenge.failed or challenge.passed:
            break
        
        event_dt = pd.to_datetime(event_date)
        challenge.new_day(event_date)
        
        # Get event volatility
        vol_data = calculate_event_volatility(df, event_dt)
        
        if vol_data is None:
            continue
        
        # Create trade based on strategy
        if direction_strategy == "straddle":
            long_trade, short_trade = simulate_straddle_trade(
                df, event_dt, risk_pct=risk_pct, rr_ratio=rr_ratio
            )
            
            if long_trade and short_trade:
                # Execute both, but only one should trigger
                long_result = execute_trade(long_trade, df)
                short_result = execute_trade(short_trade, df)
                
                # Take the one that triggered (or the winning one if both triggered)
                if long_result.result in ["WIN", "LOSS", "TIMEOUT"]:
                    trade = long_result
                elif short_result.result in ["WIN", "LOSS", "TIMEOUT"]:
                    trade = short_result
                else:
                    continue
                
                trade.event_type = event_type
                challenge.apply_trade(trade)
        
        elif direction_strategy == "random":
            direction = np.random.choice(["LONG", "SHORT"])
            # Simplified: use vol_data to create trade
            entry = vol_data["event_open"]
            atr = vol_data["post_range"] / 2
            
            if direction == "LONG":
                sl = entry - atr * 1.5
                tp = entry + atr * 1.5 * rr_ratio
            else:
                sl = entry + atr * 1.5
                tp = entry - atr * 1.5 * rr_ratio
            
            trade = Trade(
                entry_time=event_dt,
                direction=direction,
                entry_price=entry,
                stop_loss=sl,
                take_profit=tp,
                event_type=event_type,
                symbol=symbol,
                risk_pct=risk_pct
            )
            
            trade = execute_trade(trade, df)
            if trade.result != "NOT_TRIGGERED":
                challenge.apply_trade(trade)
        
        # Check limits after each trade
        if not challenge.check_limits():
            break
        
        # Check if passed
        if challenge.check_passed():
            break
    
    return challenge


def run_monte_carlo_simulation(
    n_simulations: int = 1000,
    events_per_challenge: int = 15,
    risk_pct: float = 5.0,
    rr_ratio: float = 2.0,
    win_rate: float = 0.50  # Assume 50% - no edge
) -> Dict:
    """
    Monte Carlo simulation of FTMO challenge pass rate.
    Uses simplified model (no real price data) for speed.
    """
    results = {
        "passed": 0,
        "failed_daily_dd": 0,
        "failed_max_dd": 0,
        "failed_no_profit": 0,
        "final_balances": [],
        "trades_to_pass": [],
        "days_to_pass": []
    }
    
    for _ in range(n_simulations):
        balance = 100000
        high_water = 100000
        daily_start = 100000
        trades_taken = 0
        trading_days = set()
        
        failed = False
        passed = False
        
        for day in range(30):  # Max 30 days
            daily_start = balance
            
            # Simulate 1-2 events per week (random days)
            if np.random.random() < 0.3:  # ~30% chance of event per day
                trading_days.add(day)
                trades_taken += 1
                
                # Trade outcome
                if np.random.random() < win_rate:
                    pnl = risk_pct * rr_ratio  # Win
                else:
                    pnl = -risk_pct  # Loss
                
                balance = balance * (1 + pnl / 100)
                
                if balance > high_water:
                    high_water = balance
                
                # Check daily DD
                daily_dd = (daily_start - balance) / 100000 * 100
                if daily_dd >= 5.0:
                    failed = True
                    results["failed_daily_dd"] += 1
                    break
                
                # Check max DD
                max_dd = (high_water - balance) / 100000 * 100
                if max_dd >= 10.0:
                    failed = True
                    results["failed_max_dd"] += 1
                    break
                
                # Check if passed
                profit = (balance - 100000) / 100000 * 100
                if profit >= 10.0 and len(trading_days) >= 4:
                    passed = True
                    results["passed"] += 1
                    results["trades_to_pass"].append(trades_taken)
                    results["days_to_pass"].append(day + 1)
                    break
        
        if not failed and not passed:
            results["failed_no_profit"] += 1
        
        results["final_balances"].append(balance)
    
    # Calculate statistics
    results["pass_rate"] = results["passed"] / n_simulations * 100
    results["avg_final_balance"] = np.mean(results["final_balances"])
    results["median_final_balance"] = np.median(results["final_balances"])
    
    if results["trades_to_pass"]:
        results["avg_trades_to_pass"] = np.mean(results["trades_to_pass"])
        results["avg_days_to_pass"] = np.mean(results["days_to_pass"])
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("HIGH-IMPACT NEWS TRADING BACKTEST")
    print("=" * 60)
    
    # Run Monte Carlo with different risk sizes
    print("\n📊 Monte Carlo Simulation (1000 runs)")
    print("-" * 40)
    
    # Test different risk sizes
    risk_scenarios = [
        ("2% risk, 50% WR", 2.0, 0.50),
        ("2% risk, 55% WR", 2.0, 0.55),
        ("3% risk, 50% WR", 3.0, 0.50),
        ("3% risk, 55% WR", 3.0, 0.55),
        ("2.5% risk, 52% WR", 2.5, 0.52),
    ]
    
    for name, risk, wr in risk_scenarios:
        results = run_monte_carlo_simulation(
            n_simulations=1000,
            win_rate=wr,
            risk_pct=risk,
            rr_ratio=2.0
        )
        
        print(f"\n{name}:")
        print(f"  Pass Rate: {results['pass_rate']:.1f}%")
        print(f"  Avg Balance: ${results['avg_final_balance']:,.0f}")
        print(f"  Failed DD (daily): {results['failed_daily_dd']}")
        print(f"  Failed DD (max): {results['failed_max_dd']}")
        print(f"  Failed (no profit): {results['failed_no_profit']}")
        if results.get('avg_trades_to_pass'):
            print(f"  Avg Trades to Pass: {results['avg_trades_to_pass']:.1f}")
            print(f"  Avg Days to Pass: {results['avg_days_to_pass']:.1f}")
    
    # Now test with real data (using recent dates that yfinance can handle)
    print("\n" + "=" * 60)
    print("REAL DATA BACKTEST (EURUSD, Recent Events)")
    print("=" * 60)
    
    # Use only 2025 events (within yfinance 730 day limit for hourly data)
    test_events = [e for e in HISTORICAL_EVENTS_2024_2025 if e[0] >= "2024-06-01"]
    
    challenge = simulate_challenge(
        events=test_events,
        symbol="EURUSD",
        risk_pct=2.5,  # More conservative
        rr_ratio=2.0,
        direction_strategy="straddle"
    )
    
    if challenge:
        print(f"\nChallenge Result: {'PASSED ✅' if challenge.passed else 'FAILED ❌' if challenge.failed else 'INCOMPLETE'}")
        print(f"Final Balance: ${challenge.current_balance:,.2f}")
        print(f"Profit: {challenge.current_profit_pct:.2f}%")
        print(f"Max DD: {challenge.max_dd_pct:.2f}%")
        print(f"Trading Days: {len(challenge.trading_days)}")
        print(f"Total Trades: {len(challenge.trades)}")
        
        if challenge.fail_reason:
            print(f"Fail Reason: {challenge.fail_reason}")
        
        # Trade breakdown
        wins = sum(1 for t in challenge.trades if t.result == "WIN")
        losses = sum(1 for t in challenge.trades if t.result == "LOSS")
        print(f"\nTrade Results: {wins}W / {losses}L ({wins/(wins+losses)*100:.1f}% WR)" if wins+losses > 0 else "")
    
    # Save results
    output = {
        "monte_carlo": {name: run_monte_carlo_simulation(n_simulations=1000, win_rate=wr, risk_pct=risk) 
                       for name, risk, wr in risk_scenarios},
        "timestamp": datetime.now().isoformat()
    }
    
    with open("news_impact_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print("\n✅ Results saved to news_impact_results.json")


if __name__ == "__main__":
    main()
