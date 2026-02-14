"""
Fade The Spike Strategy - News Trading
=======================================
Wait for initial spike after news, then fade the direction.

HYPOTHESIS: News spikes overshoot, then revert within 2-4 hours.

LOOK-AHEAD BIAS AUDIT:
✅ Entry decision: Uses only data available at decision time
✅ ATR calculation: Uses PRE-event data only
✅ Spike detection: Waits for spike to form (2h after event)
✅ Entry execution: Next bar after signal
✅ SL/TP resolution: Worst-case (SL) first when ambiguous
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import json

# =============================================================================
# HISTORICAL EVENTS (2024-2025)
# =============================================================================

HISTORICAL_EVENTS = [
    # 2024 (only dates within yfinance 730-day limit)
    ("2024-06-07", "NFP"),
    ("2024-06-12", "CPI"),
    ("2024-06-12", "FOMC"),
    ("2024-07-05", "NFP"),
    ("2024-07-11", "CPI"),
    ("2024-07-31", "FOMC"),
    ("2024-08-02", "NFP"),
    ("2024-08-14", "CPI"),
    ("2024-09-06", "NFP"),
    ("2024-09-11", "CPI"),
    ("2024-09-18", "FOMC"),
    ("2024-10-04", "NFP"),
    ("2024-10-10", "CPI"),
    ("2024-11-01", "NFP"),
    ("2024-11-07", "FOMC"),
    ("2024-11-13", "CPI"),
    ("2024-12-06", "NFP"),
    ("2024-12-11", "CPI"),
    ("2024-12-18", "FOMC"),
    # 2025
    ("2025-01-10", "NFP"),
    ("2025-01-15", "CPI"),
    ("2025-01-29", "FOMC"),
]


@dataclass
class Trade:
    signal_time: datetime      # When signal was generated
    entry_time: datetime       # When entry actually executed (next bar)
    direction: str             # "LONG" or "SHORT"
    entry_price: float
    stop_loss: float
    take_profit: float
    event_type: str
    symbol: str
    risk_pct: float = 2.5
    spike_direction: str = ""  # Direction of initial spike we're fading
    
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl_pct: Optional[float] = None
    result: Optional[str] = None


@dataclass
class BacktestResult:
    symbol: str
    trades: List[Trade] = field(default_factory=list)
    
    @property
    def total_trades(self) -> int:
        return len([t for t in self.trades if t.result in ["WIN", "LOSS", "TIMEOUT"]])
    
    @property
    def wins(self) -> int:
        return len([t for t in self.trades if t.result == "WIN"])
    
    @property
    def losses(self) -> int:
        return len([t for t in self.trades if t.result == "LOSS"])
    
    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0
        return self.wins / self.total_trades * 100
    
    @property
    def total_pnl(self) -> float:
        return sum(t.pnl_pct or 0 for t in self.trades)
    
    @property
    def profit_factor(self) -> float:
        gross_profit = sum(t.pnl_pct for t in self.trades if t.pnl_pct and t.pnl_pct > 0)
        gross_loss = abs(sum(t.pnl_pct for t in self.trades if t.pnl_pct and t.pnl_pct < 0))
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0
        return gross_profit / gross_loss


def fetch_data(symbol: str, start: str, end: str, interval: str = "1h") -> pd.DataFrame:
    """Fetch OHLCV data."""
    yf_symbol = f"{symbol}=X" if symbol in ["EURUSD", "USDJPY", "GBPUSD"] else symbol
    
    ticker = yf.Ticker(yf_symbol)
    df = ticker.history(start=start, end=end, interval=interval)
    
    if df.empty:
        return pd.DataFrame()
    
    df = df.rename(columns={
        'Open': 'open', 'High': 'high', 'Low': 'low',
        'Close': 'close', 'Volume': 'volume'
    })
    return df[['open', 'high', 'low', 'close', 'volume']]


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate ATR using only past data (no look-ahead)."""
    high = df['high']
    low = df['low']
    close = df['close'].shift(1)  # Previous close
    
    tr = pd.DataFrame({
        'hl': high - low,
        'hc': abs(high - close),
        'lc': abs(low - close)
    }).max(axis=1)
    
    return tr.rolling(window=period).mean()


def detect_spike(
    df: pd.DataFrame,
    event_time: datetime,
    wait_hours: int = 2,
    min_move_atr: float = 1.5
) -> Optional[Dict]:
    """
    Detect if a significant spike occurred after news event.
    Returns spike info ONLY using data up to (event_time + wait_hours).
    
    NO LOOK-AHEAD: We wait for the spike to form before making decisions.
    """
    # Make timezone aware if needed
    if df.index.tz is not None and event_time.tzinfo is None:
        event_time = pd.Timestamp(event_time).tz_localize(df.index.tz)
    
    # Get PRE-event data for ATR (no look-ahead)
    pre_event = df[df.index < event_time].tail(50)
    if len(pre_event) < 20:
        return None
    
    atr = calculate_atr(pre_event, 14).iloc[-1]
    if pd.isna(atr) or atr == 0:
        return None
    
    # Get SPIKE WINDOW (event to event + wait_hours)
    spike_end_time = event_time + timedelta(hours=wait_hours)
    spike_window = df[(df.index >= event_time) & (df.index <= spike_end_time)]
    
    if len(spike_window) < 2:
        return None
    
    # Measure the move during spike window
    event_open = spike_window.iloc[0]['open']
    spike_high = spike_window['high'].max()
    spike_low = spike_window['low'].min()
    spike_close = spike_window.iloc[-1]['close']
    
    up_move = spike_high - event_open
    down_move = event_open - spike_low
    
    # Determine spike direction (which way did it go more?)
    if up_move > down_move and up_move > atr * min_move_atr:
        return {
            "direction": "UP",
            "magnitude": up_move,
            "atr": atr,
            "atr_multiple": up_move / atr,
            "spike_high": spike_high,
            "spike_low": spike_low,
            "event_open": event_open,
            "spike_end_time": spike_end_time,
            "last_close": spike_close
        }
    elif down_move > up_move and down_move > atr * min_move_atr:
        return {
            "direction": "DOWN",
            "magnitude": down_move,
            "atr": atr,
            "atr_multiple": down_move / atr,
            "spike_high": spike_high,
            "spike_low": spike_low,
            "event_open": event_open,
            "spike_end_time": spike_end_time,
            "last_close": spike_close
        }
    
    return None  # No significant spike


def create_fade_trade(
    df: pd.DataFrame,
    spike: Dict,
    risk_pct: float = 2.5,
    rr_ratio: float = 2.0,
    sl_atr_mult: float = 1.0,
    event_type: str = "",
    symbol: str = ""
) -> Optional[Trade]:
    """
    Create a fade trade AFTER spike is detected.
    
    ENTRY: Next bar after spike_end_time (NO LOOK-AHEAD)
    SL: Beyond spike extreme
    TP: Back toward event_open (fade the move)
    """
    spike_end = spike["spike_end_time"]
    
    # Find the NEXT bar after spike window for entry
    future_bars = df[df.index > spike_end]
    if len(future_bars) < 1:
        return None
    
    entry_bar = future_bars.iloc[0]
    entry_time = future_bars.index[0]
    entry_price = entry_bar['open']  # Enter at open of next bar
    
    atr = spike["atr"]
    
    if spike["direction"] == "UP":
        # Spike went UP → Fade by going SHORT
        direction = "SHORT"
        stop_loss = spike["spike_high"] + (atr * sl_atr_mult)  # SL above spike high
        risk_distance = stop_loss - entry_price
        take_profit = entry_price - (risk_distance * rr_ratio)  # TP below
    else:
        # Spike went DOWN → Fade by going LONG
        direction = "LONG"
        stop_loss = spike["spike_low"] - (atr * sl_atr_mult)  # SL below spike low
        risk_distance = entry_price - stop_loss
        take_profit = entry_price + (risk_distance * rr_ratio)  # TP above
    
    return Trade(
        signal_time=spike_end,
        entry_time=entry_time,
        direction=direction,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        event_type=event_type,
        symbol=symbol,
        risk_pct=risk_pct,
        spike_direction=spike["direction"]
    )


def execute_trade(trade: Trade, df: pd.DataFrame, max_hours: int = 8) -> Trade:
    """
    Execute trade and determine outcome.
    
    LOOK-AHEAD SAFE:
    - Entry is at bar open (already set)
    - We walk forward bar by bar
    - When SL and TP could both be hit in same bar, assume SL (worst case)
    """
    entry_time = trade.entry_time
    
    # Get bars after entry
    post_entry = df[df.index >= entry_time].head(max_hours)
    
    if len(post_entry) == 0:
        trade.result = "NO_DATA"
        return trade
    
    # Walk through each bar
    for idx, bar in post_entry.iterrows():
        if trade.direction == "LONG":
            # Check if BOTH SL and TP could be hit (worst case: SL first)
            sl_hit = bar['low'] <= trade.stop_loss
            tp_hit = bar['high'] >= trade.take_profit
            
            if sl_hit and tp_hit:
                # Ambiguous - assume worst case (SL hit first)
                trade.exit_time = idx
                trade.exit_price = trade.stop_loss
                trade.result = "LOSS"
                trade.pnl_pct = -trade.risk_pct
                return trade
            elif sl_hit:
                trade.exit_time = idx
                trade.exit_price = trade.stop_loss
                trade.result = "LOSS"
                trade.pnl_pct = -trade.risk_pct
                return trade
            elif tp_hit:
                trade.exit_time = idx
                trade.exit_price = trade.take_profit
                trade.result = "WIN"
                trade.pnl_pct = trade.risk_pct * 2  # 1:2 RR
                return trade
                
        else:  # SHORT
            sl_hit = bar['high'] >= trade.stop_loss
            tp_hit = bar['low'] <= trade.take_profit
            
            if sl_hit and tp_hit:
                trade.exit_time = idx
                trade.exit_price = trade.stop_loss
                trade.result = "LOSS"
                trade.pnl_pct = -trade.risk_pct
                return trade
            elif sl_hit:
                trade.exit_time = idx
                trade.exit_price = trade.stop_loss
                trade.result = "LOSS"
                trade.pnl_pct = -trade.risk_pct
                return trade
            elif tp_hit:
                trade.exit_time = idx
                trade.exit_price = trade.take_profit
                trade.result = "WIN"
                trade.pnl_pct = trade.risk_pct * 2
                return trade
    
    # Timeout - close at last price
    last_bar = post_entry.iloc[-1]
    trade.exit_time = post_entry.index[-1]
    trade.exit_price = last_bar['close']
    
    if trade.direction == "LONG":
        raw_pnl = (trade.exit_price - trade.entry_price) / (trade.entry_price - trade.stop_loss)
    else:
        raw_pnl = (trade.entry_price - trade.exit_price) / (trade.stop_loss - trade.entry_price)
    
    trade.pnl_pct = raw_pnl * trade.risk_pct
    trade.result = "TIMEOUT"
    
    return trade


def run_backtest(
    symbol: str,
    events: List[Tuple[str, str]],
    wait_hours: int = 2,
    min_move_atr: float = 1.5,
    risk_pct: float = 2.5,
    rr_ratio: float = 2.0
) -> BacktestResult:
    """Run fade-the-spike backtest on historical events."""
    
    # Fetch all data at once
    start_date = (datetime.strptime(events[0][0], "%Y-%m-%d") - timedelta(days=30)).strftime("%Y-%m-%d")
    end_date = (datetime.strptime(events[-1][0], "%Y-%m-%d") + timedelta(days=7)).strftime("%Y-%m-%d")
    
    print(f"Fetching {symbol} data from {start_date} to {end_date}...")
    df = fetch_data(symbol, start_date, end_date, "1h")
    
    if df.empty:
        print(f"No data for {symbol}")
        return BacktestResult(symbol=symbol)
    
    print(f"Got {len(df)} bars")
    
    result = BacktestResult(symbol=symbol)
    
    for event_date, event_type in events:
        event_time = pd.to_datetime(event_date)
        
        # Detect spike
        spike = detect_spike(df, event_time, wait_hours=wait_hours, min_move_atr=min_move_atr)
        
        if spike is None:
            continue
        
        # Create fade trade
        trade = create_fade_trade(
            df, spike,
            risk_pct=risk_pct,
            rr_ratio=rr_ratio,
            event_type=event_type,
            symbol=symbol
        )
        
        if trade is None:
            continue
        
        # Execute trade
        trade = execute_trade(trade, df)
        result.trades.append(trade)
    
    return result


def print_results(result: BacktestResult):
    """Print backtest results."""
    print(f"\n{'='*60}")
    print(f"FADE THE SPIKE - {result.symbol}")
    print(f"{'='*60}")
    
    if result.total_trades == 0:
        print("No trades executed")
        return
    
    print(f"Total Trades: {result.total_trades}")
    print(f"Wins: {result.wins}")
    print(f"Losses: {result.losses}")
    print(f"Win Rate: {result.win_rate:.1f}%")
    print(f"Total PnL: {result.total_pnl:+.2f}%")
    print(f"Profit Factor: {result.profit_factor:.2f}")
    
    print(f"\n--- Trade Details ---")
    for i, trade in enumerate(result.trades):
        if trade.result in ["WIN", "LOSS", "TIMEOUT"]:
            print(f"{i+1}. {trade.event_type} | {trade.direction} (fade {trade.spike_direction}) | "
                  f"{trade.result} | PnL: {trade.pnl_pct:+.2f}%")


def run_parameter_sweep():
    """Test different parameter combinations."""
    symbols = ["EURUSD", "GBPUSD", "USDJPY"]
    wait_hours_options = [1, 2, 3, 4]
    min_move_options = [1.0, 1.5, 2.0]
    
    all_results = []
    
    for symbol in symbols:
        for wait_h in wait_hours_options:
            for min_move in min_move_options:
                result = run_backtest(
                    symbol=symbol,
                    events=HISTORICAL_EVENTS,
                    wait_hours=wait_h,
                    min_move_atr=min_move
                )
                
                if result.total_trades >= 3:
                    all_results.append({
                        "symbol": symbol,
                        "wait_hours": wait_h,
                        "min_move_atr": min_move,
                        "trades": result.total_trades,
                        "win_rate": result.win_rate,
                        "pnl": result.total_pnl,
                        "pf": result.profit_factor
                    })
    
    # Sort by PnL
    all_results.sort(key=lambda x: x["pnl"], reverse=True)
    
    print("\n" + "="*80)
    print("PARAMETER SWEEP RESULTS (sorted by PnL)")
    print("="*80)
    print(f"{'Symbol':<10} {'Wait':<6} {'MinATR':<8} {'Trades':<8} {'WR%':<8} {'PnL%':<10} {'PF':<8}")
    print("-"*80)
    
    for r in all_results[:20]:  # Top 20
        print(f"{r['symbol']:<10} {r['wait_hours']:<6} {r['min_move_atr']:<8.1f} "
              f"{r['trades']:<8} {r['win_rate']:<8.1f} {r['pnl']:<10.2f} {r['pf']:<8.2f}")
    
    # Save results
    with open("fade_spike_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ Saved {len(all_results)} results to fade_spike_results.json")


def main():
    print("="*60)
    print("FADE THE SPIKE STRATEGY")
    print("Wait for news spike, then fade the direction")
    print("="*60)
    
    # Quick test on EURUSD
    result = run_backtest(
        symbol="EURUSD",
        events=HISTORICAL_EVENTS,
        wait_hours=2,
        min_move_atr=1.5
    )
    
    print_results(result)
    
    # Ask user if they want full sweep
    print("\n" + "="*60)
    print("Running parameter sweep on 3 pairs...")
    run_parameter_sweep()


if __name__ == "__main__":
    main()
