"""
Curupira Backtesting Engine v2
------------------------------
FIXED: No look-ahead bias
- Signal generated at bar i (using Close[i]) 
- Entry executed at bar i+1 Open (realistic)

Features:
- Proper intrabar TP/SL checking (High/Low, not Close)
- Compounded returns
- Slippage modeling
- Position sizing
- Walk-forward ready
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Literal, Callable
from datetime import datetime
import yfinance as yf


@dataclass
class Trade:
    entry_time: datetime
    entry_price: float
    direction: Literal['long', 'short']
    stop_loss: float
    take_profit: float
    size: float = 1.0  # Position size as fraction of capital
    signal_bar: Optional[int] = None  # Bar index where signal was generated
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    result: Optional[Literal['win', 'loss', 'breakeven']] = None
    exit_reason: Optional[str] = None  # 'tp', 'sl', 'signal', 'eod'
    pnl_pct: Optional[float] = None

    def close(self, exit_time: datetime, exit_price: float, reason: str):
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.exit_reason = reason
        
        if self.direction == 'long':
            self.pnl_pct = (exit_price - self.entry_price) / self.entry_price
        else:
            self.pnl_pct = (self.entry_price - exit_price) / self.entry_price
        
        if self.pnl_pct > 0.0001:
            self.result = 'win'
        elif self.pnl_pct < -0.0001:
            self.result = 'loss'
        else:
            self.result = 'breakeven'


@dataclass
class BacktestConfig:
    slippage_pct: float = 0.0005  # 0.05% slippage
    commission_pct: float = 0.001  # 0.1% commission per trade
    position_size: float = 1.0  # Fraction of capital per trade
    compound: bool = True  # Compound returns


class BacktestEngine:
    """
    Core backtesting engine with proper trade simulation.
    
    CRITICAL FIX: Uses pending_signal pattern to avoid look-ahead bias.
    - Signal fires at bar i (can use bar i's Close)
    - Entry executes at bar i+1's Open (realistic)
    """
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.trades: List[Trade] = []
        self.current_position: Optional[Trade] = None
    
    def run(self, df: pd.DataFrame, signal_fn: Callable) -> pd.DataFrame:
        """
        Run backtest with a signal function.
        
        signal_fn(df, i) -> dict or None
            Returns None for no signal, or:
            {
                'direction': 'long' or 'short',
                'stop_loss': price,
                'take_profit': price,
            }
            
        TIMING:
        - signal_fn evaluates at bar i (can use bar i's Close - it's "closed")
        - If signal fires, entry happens at bar i+1's Open
        """
        self.trades = []
        self.current_position = None
        pending_signal = None  # Signal waiting to be executed next bar
        pending_signal_bar = None  # Which bar generated the pending signal
        
        for i in range(1, len(df)):
            row = df.iloc[i]
            current_time = row['Date'] if 'Date' in df.columns else df.index[i]
            
            # 1. EXECUTE pending signal from PREVIOUS bar at THIS bar's Open
            if pending_signal and not self.current_position:
                self._enter_trade(row, current_time, pending_signal, pending_signal_bar)
                pending_signal = None
                pending_signal_bar = None
            
            # 2. Check exits (if we have a position)
            if self.current_position:
                self._check_exit(row, current_time)
            
            # 3. Generate new signal (will execute NEXT bar)
            #    Only if we're flat AND no pending signal
            if not self.current_position and not pending_signal:
                signal = signal_fn(df, i)
                if signal:
                    pending_signal = signal
                    pending_signal_bar = i
        
        # Close any remaining position at end
        if self.current_position:
            last = df.iloc[-1]
            last_time = last['Date'] if 'Date' in df.columns else df.index[-1]
            self.current_position.close(last_time, last['Close'], 'eod')
            self.trades.append(self.current_position)
            self.current_position = None
        
        return pd.DataFrame([t.__dict__ for t in self.trades])
    
    def _enter_trade(self, row: pd.Series, time: datetime, signal: dict, signal_bar: int):
        """Enter a new trade with slippage at current bar's Open."""
        direction = signal['direction']
        base_price = row['Open']  # Enter at THIS bar's Open (signal was from previous bar)
        
        # Apply slippage (worse price)
        if direction == 'long':
            entry_price = base_price * (1 + self.config.slippage_pct)
        else:
            entry_price = base_price * (1 - self.config.slippage_pct)
        
        # Adjust SL/TP relative to actual entry price
        # Signal's SL/TP were calculated from signal bar's Open
        # We need to shift them by the difference
        signal_open = signal.get('_signal_open')  # If provided by strategy
        
        if signal_open and signal_open != base_price:
            # Shift SL/TP by the same amount as entry price shifted
            price_shift = base_price - signal_open
            adjusted_sl = signal['stop_loss'] + price_shift
            adjusted_tp = signal['take_profit'] + price_shift
        else:
            # Use signal's SL/TP as-is (slightly less accurate but still valid)
            adjusted_sl = signal['stop_loss']
            adjusted_tp = signal['take_profit']
        
        self.current_position = Trade(
            entry_time=time,
            entry_price=entry_price,
            direction=direction,
            stop_loss=adjusted_sl,
            take_profit=adjusted_tp,
            size=self.config.position_size,
            signal_bar=signal_bar
        )
    
    def _check_exit(self, row: pd.Series, time: datetime):
        """
        Check if TP/SL hit using High/Low (proper intrabar simulation).
        Order matters: assume worst case (SL checked before TP if both could hit).
        """
        pos = self.current_position
        high, low, close = row['High'], row['Low'], row['Close']
        
        if pos.direction == 'long':
            # Check SL first (worst case)
            if low <= pos.stop_loss:
                exit_price = pos.stop_loss * (1 - self.config.slippage_pct)
                pos.close(time, exit_price, 'sl')
                self.trades.append(pos)
                self.current_position = None
            # Then check TP
            elif high >= pos.take_profit:
                exit_price = pos.take_profit * (1 - self.config.slippage_pct)
                pos.close(time, exit_price, 'tp')
                self.trades.append(pos)
                self.current_position = None
        
        else:  # short
            # Check SL first (worst case)
            if high >= pos.stop_loss:
                exit_price = pos.stop_loss * (1 + self.config.slippage_pct)
                pos.close(time, exit_price, 'sl')
                self.trades.append(pos)
                self.current_position = None
            # Then check TP
            elif low <= pos.take_profit:
                exit_price = pos.take_profit * (1 + self.config.slippage_pct)
                pos.close(time, exit_price, 'tp')
                self.trades.append(pos)
                self.current_position = None


def calculate_metrics(trades_df: pd.DataFrame, price_df: pd.DataFrame, 
                      config: BacktestConfig = None) -> dict:
    """Calculate comprehensive backtest metrics."""
    config = config or BacktestConfig()
    
    if trades_df.empty:
        bnh_ret = (price_df['Close'].iloc[-1] / price_df['Close'].iloc[0] - 1) * 100
        return {
            'total_trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'strategy_return': 0,
            'buy_hold_return': round(bnh_ret, 2),
            'max_drawdown': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'expectancy': 0,
            'sharpe': 0,
        }
    
    # Filter completed trades
    completed = trades_df[trades_df['exit_price'].notna()].copy()
    
    if completed.empty:
        bnh_ret = (price_df['Close'].iloc[-1] / price_df['Close'].iloc[0] - 1) * 100
        return {
            'total_trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'strategy_return': 0,
            'buy_hold_return': round(bnh_ret, 2),
            'max_drawdown': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'expectancy': 0,
            'sharpe': 0,
        }
    
    # Apply commission
    completed['pnl_pct'] = completed['pnl_pct'] - (2 * config.commission_pct)
    
    # Calculate compounded returns
    if config.compound:
        equity_curve = (1 + completed['pnl_pct'] * completed['size']).cumprod()
        strategy_return = (equity_curve.iloc[-1] - 1) * 100 if len(equity_curve) > 0 else 0
    else:
        strategy_return = completed['pnl_pct'].sum() * 100
    
    # Win/Loss stats
    wins = completed[completed['result'] == 'win']
    losses = completed[completed['result'] == 'loss']
    
    total_trades = len(completed)
    win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
    
    avg_win = wins['pnl_pct'].mean() * 100 if len(wins) > 0 else 0
    avg_loss = losses['pnl_pct'].mean() * 100 if len(losses) > 0 else 0
    
    # Profit factor
    total_profit = wins['pnl_pct'].sum() if len(wins) > 0 else 0
    total_loss = abs(losses['pnl_pct'].sum()) if len(losses) > 0 else 0
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
    
    # Max drawdown
    if config.compound:
        equity = (1 + completed['pnl_pct'] * completed['size']).cumprod()
    else:
        equity = 1 + (completed['pnl_pct'] * completed['size']).cumsum()
    
    peak = equity.expanding().max()
    drawdown = (equity / peak - 1) * 100
    max_drawdown = drawdown.min()
    
    # Buy and hold
    bnh_return = (price_df['Close'].iloc[-1] / price_df['Close'].iloc[0] - 1) * 100
    
    # Expectancy
    expectancy = completed['pnl_pct'].mean() * 100 if total_trades > 0 else 0
    
    # Sharpe (annualized, assuming daily returns)
    if len(completed) > 1 and completed['pnl_pct'].std() > 0:
        sharpe = np.sqrt(252) * completed['pnl_pct'].mean() / completed['pnl_pct'].std()
    else:
        sharpe = 0
    
    return {
        'total_trades': total_trades,
        'win_rate': round(win_rate, 1),
        'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else 999,
        'strategy_return': round(strategy_return, 2),
        'buy_hold_return': round(bnh_return, 2),
        'max_drawdown': round(max_drawdown, 2),
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2),
        'expectancy': round(expectancy, 2),
        'sharpe': round(sharpe, 2),
    }


def fetch_data(symbol: str, interval: str = '1h', days: int = 59) -> pd.DataFrame:
    """Fetch OHLCV data from yfinance."""
    from datetime import timedelta
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date.strftime('%Y-%m-%d'), 
                        end=end_date.strftime('%Y-%m-%d'),
                        interval=interval)
    df.reset_index(inplace=True)
    
    if 'Datetime' in df.columns:
        df.rename(columns={'Datetime': 'Date'}, inplace=True)
    if 'index' in df.columns:
        df.rename(columns={'index': 'Date'}, inplace=True)
    
    return df


# Utility: Detect asset class from symbol
def get_asset_class(symbol: str) -> str:
    """Detect asset class from symbol for proper bar counting."""
    symbol = symbol.upper()
    if '=X' in symbol:
        return 'forex'  # 24h market
    elif '=F' in symbol:
        return 'futures'  # Nearly 24h
    elif symbol.startswith('^'):
        return 'index'  # ~7h/day
    else:
        return 'stock'  # ~7h/day (6.5h regular + some extended)


def get_bars_per_day(symbol: str, interval: str = '1h') -> int:
    """Get expected number of bars per trading day."""
    asset_class = get_asset_class(symbol)
    
    # Hours per day by asset class
    hours_per_day = {
        'forex': 24,
        'futures': 23,  # Brief maintenance break
        'index': 7,
        'stock': 7,  # 6.5h regular, round to 7
    }
    
    hours = hours_per_day.get(asset_class, 7)
    
    # Adjust for interval
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
        return hours  # Default to 1h assumption
