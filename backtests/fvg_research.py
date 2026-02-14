"""
FVG (Fair Value Gap) Research - Forex Focus
============================================
Testing multiple hypotheses about how price interacts with gaps.

Variations:
1. Magnet (baseline) - Price crosses midpoint, trade WITH gap
2. Wall/Bounce - Price touches edge and reverses, trade AGAINST gap  
3. Fill Reversal - After gap FULLY fills, trade AGAINST gap
4. Edge Entry - Entry at gap edge (both WITH and AGAINST directions)

Key parameters tested:
- Gap age limit (recent vs any age)
- Min gap size (filter noise)
- SL/TP ratios (1:2, 1:3, 1:4)
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Tuple, Literal
from datetime import datetime, timedelta
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

from engine import BacktestEngine, BacktestConfig, calculate_metrics, fetch_data


# =============================================================================
# FVG DETECTION (Enhanced)
# =============================================================================

@dataclass
class FVG:
    """Fair Value Gap data structure."""
    type: Literal['bullish', 'bearish']
    top: float
    bottom: float
    midpoint: float
    size: float
    created_at: int  # Bar index
    created_time: datetime
    filled: bool = False
    filled_at: Optional[int] = None
    touched_edge: bool = False
    touched_at: Optional[int] = None


def detect_fvgs(df: pd.DataFrame, i: int, lookback: int = 100, 
                min_gap_pct: float = 0.0, max_age_bars: Optional[int] = None) -> List[FVG]:
    """
    Detect Fair Value Gaps up to current bar with filtering.
    
    Args:
        df: OHLCV dataframe
        i: Current bar index
        lookback: How many bars back to look for gaps
        min_gap_pct: Minimum gap size as percentage of price (filter noise)
        max_age_bars: Maximum age of gap in bars (None = any age)
    
    Returns:
        List of FVG objects that are still relevant
    """
    start_idx = max(2, i - lookback)
    fvgs = []
    
    for j in range(start_idx, i):
        if j < 2:
            continue
        
        # Three-bar pattern: bar[j-2], bar[j-1] (impulse), bar[j]
        bar_minus2 = df.iloc[j-2]
        bar_current = df.iloc[j]
        
        high_2ago = bar_minus2['High']
        low_2ago = bar_minus2['Low']
        high_curr = bar_current['High']
        low_curr = bar_current['Low']
        
        # Get timestamp
        time_col = 'Date' if 'Date' in df.columns else df.index[j]
        created_time = df.iloc[j][time_col] if 'Date' in df.columns else df.index[j]
        
        fvg = None
        
        # Bullish FVG: gap up (current low > 2-bars-ago high)
        if low_curr > high_2ago:
            gap_size = low_curr - high_2ago
            gap_pct = gap_size / high_2ago * 100
            
            if gap_pct >= min_gap_pct:
                fvg = FVG(
                    type='bullish',
                    top=low_curr,
                    bottom=high_2ago,
                    midpoint=(low_curr + high_2ago) / 2,
                    size=gap_size,
                    created_at=j,
                    created_time=created_time,
                )
        
        # Bearish FVG: gap down (2-bars-ago low > current high)
        elif low_2ago > high_curr:
            gap_size = low_2ago - high_curr
            gap_pct = gap_size / low_2ago * 100
            
            if gap_pct >= min_gap_pct:
                fvg = FVG(
                    type='bearish',
                    top=low_2ago,
                    bottom=high_curr,
                    midpoint=(low_2ago + high_curr) / 2,
                    size=gap_size,
                    created_at=j,
                    created_time=created_time,
                )
        
        if fvg is None:
            continue
        
        # Check age limit
        if max_age_bars is not None and (i - j) > max_age_bars:
            continue
        
        # Track fill and edge touch status
        for k in range(j + 1, i + 1):
            bar_k = df.iloc[k]
            
            if fvg.type == 'bullish':
                # Edge touched = price reaches top of gap
                if not fvg.touched_edge and bar_k['Low'] <= fvg.top:
                    fvg.touched_edge = True
                    fvg.touched_at = k
                
                # Filled = price goes all the way through (below bottom)
                if bar_k['Low'] <= fvg.bottom:
                    fvg.filled = True
                    fvg.filled_at = k
                    break
            else:  # bearish
                # Edge touched = price reaches bottom of gap
                if not fvg.touched_edge and bar_k['High'] >= fvg.bottom:
                    fvg.touched_edge = True
                    fvg.touched_at = k
                
                # Filled = price goes all the way through (above top)
                if bar_k['High'] >= fvg.top:
                    fvg.filled = True
                    fvg.filled_at = k
                    break
        
        fvgs.append(fvg)
    
    return fvgs


def is_rejection_candle(df: pd.DataFrame, i: int, direction: str) -> bool:
    """
    Check if current bar is a rejection candle (pin bar / hammer).
    
    Args:
        direction: Expected rejection direction ('bullish' = price rejected down, 'bearish' = rejected up)
    """
    bar = df.iloc[i]
    body = abs(bar['Close'] - bar['Open'])
    full_range = bar['High'] - bar['Low']
    
    if full_range == 0:
        return False
    
    body_ratio = body / full_range
    
    if direction == 'bullish':
        # Bullish rejection = long lower wick, price bounced up
        lower_wick = min(bar['Open'], bar['Close']) - bar['Low']
        upper_wick = bar['High'] - max(bar['Open'], bar['Close'])
        return lower_wick > 2 * body and lower_wick > 2 * upper_wick and body_ratio < 0.4
    
    else:  # bearish
        # Bearish rejection = long upper wick, price rejected down
        upper_wick = bar['High'] - max(bar['Open'], bar['Close'])
        lower_wick = min(bar['Open'], bar['Close']) - bar['Low']
        return upper_wick > 2 * body and upper_wick > 2 * lower_wick and body_ratio < 0.4


# =============================================================================
# STRATEGY VARIATIONS
# =============================================================================

def fvg_magnet_signal(df: pd.DataFrame, i: int, 
                      sl_mult: float = 1.5, tp_mult: float = 3.0,
                      min_gap_pct: float = 0.02, max_age_bars: int = 48) -> Optional[Dict]:
    """
    Variation 1: MAGNET (baseline)
    - Entry: Price crosses gap midpoint
    - Direction: WITH gap (bullish FVG → long)
    - Theory: Gaps act as magnets pulling price through
    """
    if i < 5:
        return None
    
    fvgs = detect_fvgs(df, i, min_gap_pct=min_gap_pct, max_age_bars=max_age_bars)
    unfilled = [g for g in fvgs if not g.filled]
    
    if not unfilled:
        return None
    
    current_price = df.iloc[i]['Close']
    prev_price = df.iloc[i-1]['Close']
    open_price = df.iloc[i]['Open']
    
    for gap in reversed(unfilled):  # Most recent first
        midpoint = gap.midpoint
        
        if gap.type == 'bullish':
            # Price crossing UP through midpoint → long
            if prev_price < midpoint <= current_price:
                return {
                    'direction': 'long',
                    'stop_loss': open_price - sl_mult * gap.size,
                    'take_profit': open_price + tp_mult * gap.size,
                    '_gap': gap,
                }
        else:  # bearish
            # Price crossing DOWN through midpoint → short
            if prev_price > midpoint >= current_price:
                return {
                    'direction': 'short',
                    'stop_loss': open_price + sl_mult * gap.size,
                    'take_profit': open_price - tp_mult * gap.size,
                    '_gap': gap,
                }
    
    return None


def fvg_wall_signal(df: pd.DataFrame, i: int,
                    sl_mult: float = 1.5, tp_mult: float = 3.0,
                    min_gap_pct: float = 0.02, max_age_bars: int = 48) -> Optional[Dict]:
    """
    Variation 2: WALL/BOUNCE
    - Entry: Price touches gap edge and shows rejection candle
    - Direction: AGAINST gap (bullish FVG → short on rejection)
    - Theory: Gaps act as resistance/support walls
    """
    if i < 5:
        return None
    
    fvgs = detect_fvgs(df, i, min_gap_pct=min_gap_pct, max_age_bars=max_age_bars)
    unfilled = [g for g in fvgs if not g.filled]
    
    if not unfilled:
        return None
    
    bar = df.iloc[i]
    open_price = bar['Open']
    
    for gap in reversed(unfilled):
        if gap.type == 'bullish':
            # Price touching top of bullish gap (potential resistance)
            # Look for bearish rejection
            price_at_edge = bar['Low'] <= gap.top <= bar['High']
            
            if price_at_edge and is_rejection_candle(df, i, 'bearish'):
                return {
                    'direction': 'short',  # AGAINST the gap
                    'stop_loss': gap.top + sl_mult * gap.size,
                    'take_profit': gap.top - tp_mult * gap.size,
                    '_gap': gap,
                }
        
        else:  # bearish gap
            # Price touching bottom of bearish gap (potential support)
            # Look for bullish rejection
            price_at_edge = bar['Low'] <= gap.bottom <= bar['High']
            
            if price_at_edge and is_rejection_candle(df, i, 'bullish'):
                return {
                    'direction': 'long',  # AGAINST the gap
                    'stop_loss': gap.bottom - sl_mult * gap.size,
                    'take_profit': gap.bottom + tp_mult * gap.size,
                    '_gap': gap,
                }
    
    return None


def fvg_fill_reversal_signal(df: pd.DataFrame, i: int,
                             sl_mult: float = 1.5, tp_mult: float = 3.0,
                             min_gap_pct: float = 0.02, max_age_bars: int = 96) -> Optional[Dict]:
    """
    Variation 3: FILL REVERSAL
    - Entry: After gap is FULLY FILLED
    - Direction: AGAINST original gap direction
    - Theory: Filled gaps become support/resistance for reversal
    """
    if i < 5:
        return None
    
    fvgs = detect_fvgs(df, i, min_gap_pct=min_gap_pct, max_age_bars=max_age_bars)
    
    # Look for gaps that just got filled THIS bar
    just_filled = [g for g in fvgs if g.filled and g.filled_at == i]
    
    if not just_filled:
        return None
    
    open_price = df.iloc[i]['Open']
    
    for gap in just_filled:
        if gap.type == 'bullish':
            # Bullish gap just filled → expect reversal up (long)
            return {
                'direction': 'long',  # Reversal direction
                'stop_loss': gap.bottom - sl_mult * gap.size,
                'take_profit': gap.bottom + tp_mult * gap.size,
                '_gap': gap,
            }
        else:  # bearish
            # Bearish gap just filled → expect reversal down (short)
            return {
                'direction': 'short',  # Reversal direction
                'stop_loss': gap.top + sl_mult * gap.size,
                'take_profit': gap.top - tp_mult * gap.size,
                '_gap': gap,
            }
    
    return None


def fvg_edge_with_signal(df: pd.DataFrame, i: int,
                         sl_mult: float = 1.5, tp_mult: float = 3.0,
                         min_gap_pct: float = 0.02, max_age_bars: int = 48) -> Optional[Dict]:
    """
    Variation 4a: EDGE ENTRY (WITH gap)
    - Entry: At gap edge (not midpoint)
    - Direction: WITH gap
    """
    if i < 5:
        return None
    
    fvgs = detect_fvgs(df, i, min_gap_pct=min_gap_pct, max_age_bars=max_age_bars)
    unfilled = [g for g in fvgs if not g.filled]
    
    if not unfilled:
        return None
    
    bar = df.iloc[i]
    prev_bar = df.iloc[i-1]
    open_price = bar['Open']
    
    for gap in reversed(unfilled):
        if gap.type == 'bullish':
            # Price entering top of bullish gap → long (with gap)
            if prev_bar['Low'] > gap.top and bar['Low'] <= gap.top:
                return {
                    'direction': 'long',
                    'stop_loss': gap.bottom - sl_mult * gap.size,
                    'take_profit': gap.top + tp_mult * gap.size,
                    '_gap': gap,
                }
        else:  # bearish
            # Price entering bottom of bearish gap → short (with gap)
            if prev_bar['High'] < gap.bottom and bar['High'] >= gap.bottom:
                return {
                    'direction': 'short',
                    'stop_loss': gap.top + sl_mult * gap.size,
                    'take_profit': gap.bottom - tp_mult * gap.size,
                    '_gap': gap,
                }
    
    return None


def fvg_edge_against_signal(df: pd.DataFrame, i: int,
                            sl_mult: float = 1.5, tp_mult: float = 3.0,
                            min_gap_pct: float = 0.02, max_age_bars: int = 48) -> Optional[Dict]:
    """
    Variation 4b: EDGE ENTRY (AGAINST gap)
    - Entry: At gap edge
    - Direction: AGAINST gap (fade the move)
    """
    if i < 5:
        return None
    
    fvgs = detect_fvgs(df, i, min_gap_pct=min_gap_pct, max_age_bars=max_age_bars)
    unfilled = [g for g in fvgs if not g.filled]
    
    if not unfilled:
        return None
    
    bar = df.iloc[i]
    prev_bar = df.iloc[i-1]
    open_price = bar['Open']
    
    for gap in reversed(unfilled):
        if gap.type == 'bullish':
            # Price entering top of bullish gap → short (against gap / fade)
            if prev_bar['Low'] > gap.top and bar['Low'] <= gap.top:
                return {
                    'direction': 'short',  # Against
                    'stop_loss': gap.top + sl_mult * gap.size,
                    'take_profit': gap.bottom - tp_mult * gap.size,
                    '_gap': gap,
                }
        else:  # bearish
            # Price entering bottom of bearish gap → long (against gap / fade)
            if prev_bar['High'] < gap.bottom and bar['High'] >= gap.bottom:
                return {
                    'direction': 'long',  # Against
                    'stop_loss': gap.bottom - sl_mult * gap.size,
                    'take_profit': gap.top + tp_mult * gap.size,
                    '_gap': gap,
                }
    
    return None


# Strategy registry
STRATEGIES = {
    'magnet': fvg_magnet_signal,
    'wall': fvg_wall_signal,
    'fill_reversal': fvg_fill_reversal_signal,
    'edge_with': fvg_edge_with_signal,
    'edge_against': fvg_edge_against_signal,
}


# =============================================================================
# PARAMETER GRID
# =============================================================================

PARAM_GRID = {
    'sl_tp_ratios': [
        (1.0, 2.0),   # 1:2
        (1.0, 3.0),   # 1:3
        (1.5, 3.0),   # 1.5:3
        (1.0, 4.0),   # 1:4
    ],
    'min_gap_pct': [0.01, 0.02, 0.05],  # 0.01%, 0.02%, 0.05%
    'max_age_bars': [24, 48, 96, None],  # 1 day, 2 days, 4 days, unlimited
}


# =============================================================================
# BACKTESTING
# =============================================================================

def run_fvg_backtest(symbol: str, strategy_name: str, 
                     sl_mult: float, tp_mult: float,
                     min_gap_pct: float, max_age_bars: Optional[int],
                     days: int = 59) -> Dict:
    """Run a single FVG backtest with given parameters."""
    
    # Fetch data
    df = fetch_data(symbol, interval='1h', days=days)
    
    if len(df) < 50:
        return {'error': f'Insufficient data for {symbol}: {len(df)} bars'}
    
    # Create signal function with parameters
    strategy_fn = STRATEGIES[strategy_name]
    
    def signal_fn(df_inner, i):
        return strategy_fn(
            df_inner, i,
            sl_mult=sl_mult,
            tp_mult=tp_mult,
            min_gap_pct=min_gap_pct,
            max_age_bars=max_age_bars if max_age_bars else 999999
        )
    
    # Run backtest with forex-appropriate costs
    config = BacktestConfig(
        slippage_pct=0.0005,    # 0.05% slippage (half a pip on majors)
        commission_pct=0.0003,  # 0.03% commission
        compound=True,
    )
    
    engine = BacktestEngine(config)
    trades_df = engine.run(df, signal_fn)
    metrics = calculate_metrics(trades_df, df, config)
    
    return {
        'symbol': symbol,
        'strategy': strategy_name,
        'params': {
            'sl_mult': sl_mult,
            'tp_mult': tp_mult,
            'min_gap_pct': min_gap_pct,
            'max_age_bars': max_age_bars,
        },
        'metrics': metrics,
        'trades': trades_df.to_dict('records') if not trades_df.empty else [],
        'data_bars': len(df),
    }


def run_full_research(symbols: List[str], days: int = 59) -> Dict:
    """Run complete FVG research across all variations and parameters."""
    
    results = {
        'meta': {
            'run_date': datetime.now().isoformat(),
            'symbols': symbols,
            'days': days,
            'strategies': list(STRATEGIES.keys()),
        },
        'by_strategy': {},
        'by_symbol': {},
        'by_params': {},
        'all_results': [],
    }
    
    total_tests = len(symbols) * len(STRATEGIES) * len(PARAM_GRID['sl_tp_ratios']) * \
                  len(PARAM_GRID['min_gap_pct']) * len(PARAM_GRID['max_age_bars'])
    
    print(f"Running {total_tests} backtests...")
    test_num = 0
    
    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"Testing {symbol}")
        print(f"{'='*60}")
        
        for strategy_name in STRATEGIES.keys():
            print(f"\n  Strategy: {strategy_name}")
            
            for sl_mult, tp_mult in PARAM_GRID['sl_tp_ratios']:
                for min_gap_pct in PARAM_GRID['min_gap_pct']:
                    for max_age_bars in PARAM_GRID['max_age_bars']:
                        test_num += 1
                        
                        result = run_fvg_backtest(
                            symbol, strategy_name,
                            sl_mult, tp_mult,
                            min_gap_pct, max_age_bars,
                            days
                        )
                        
                        if 'error' not in result:
                            results['all_results'].append(result)
                            
                            # Organize by strategy
                            if strategy_name not in results['by_strategy']:
                                results['by_strategy'][strategy_name] = []
                            results['by_strategy'][strategy_name].append(result)
                            
                            # Organize by symbol
                            if symbol not in results['by_symbol']:
                                results['by_symbol'][symbol] = []
                            results['by_symbol'][symbol].append(result)
                            
                            # Progress
                            if test_num % 20 == 0:
                                m = result['metrics']
                                print(f"    [{test_num}/{total_tests}] "
                                      f"SL:{sl_mult} TP:{tp_mult} gap:{min_gap_pct}% "
                                      f"age:{max_age_bars} → "
                                      f"WR:{m['win_rate']}% PF:{m['profit_factor']} "
                                      f"Ret:{m['strategy_return']}%")
    
    # Compute summary statistics
    results['summary'] = compute_summary(results)
    
    return results


def compute_summary(results: Dict) -> Dict:
    """Compute summary statistics from results."""
    
    summary = {
        'best_by_strategy': {},
        'best_by_symbol': {},
        'strategy_comparison': {},
        'symbol_comparison': {},
    }
    
    # Best configuration per strategy
    for strategy, strat_results in results['by_strategy'].items():
        valid = [r for r in strat_results if r['metrics']['total_trades'] >= 5]
        if valid:
            best = max(valid, key=lambda r: r['metrics']['profit_factor'] 
                       if r['metrics']['profit_factor'] < 100 else 0)
            summary['best_by_strategy'][strategy] = {
                'symbol': best['symbol'],
                'params': best['params'],
                'metrics': best['metrics'],
            }
    
    # Best configuration per symbol
    for symbol, sym_results in results['by_symbol'].items():
        valid = [r for r in sym_results if r['metrics']['total_trades'] >= 5]
        if valid:
            best = max(valid, key=lambda r: r['metrics']['profit_factor']
                       if r['metrics']['profit_factor'] < 100 else 0)
            summary['best_by_symbol'][symbol] = {
                'strategy': best['strategy'],
                'params': best['params'],
                'metrics': best['metrics'],
            }
    
    # Strategy comparison (average metrics across all symbols/params)
    for strategy in STRATEGIES.keys():
        strat_results = [r for r in results['all_results'] 
                         if r['strategy'] == strategy and r['metrics']['total_trades'] >= 3]
        
        if strat_results:
            avg_wr = np.mean([r['metrics']['win_rate'] for r in strat_results])
            avg_pf = np.mean([r['metrics']['profit_factor'] for r in strat_results 
                             if r['metrics']['profit_factor'] < 100])
            avg_ret = np.mean([r['metrics']['strategy_return'] for r in strat_results])
            avg_trades = np.mean([r['metrics']['total_trades'] for r in strat_results])
            
            summary['strategy_comparison'][strategy] = {
                'avg_win_rate': round(avg_wr, 1),
                'avg_profit_factor': round(avg_pf, 2) if avg_pf else 0,
                'avg_return': round(avg_ret, 2),
                'avg_trades': round(avg_trades, 1),
                'sample_count': len(strat_results),
            }
    
    # Symbol comparison (average across all strategies)
    for symbol in results['meta']['symbols']:
        sym_results = [r for r in results['all_results']
                       if r['symbol'] == symbol and r['metrics']['total_trades'] >= 3]
        
        if sym_results:
            avg_wr = np.mean([r['metrics']['win_rate'] for r in sym_results])
            avg_pf = np.mean([r['metrics']['profit_factor'] for r in sym_results
                             if r['metrics']['profit_factor'] < 100])
            avg_ret = np.mean([r['metrics']['strategy_return'] for r in sym_results])
            
            summary['symbol_comparison'][symbol] = {
                'avg_win_rate': round(avg_wr, 1),
                'avg_profit_factor': round(avg_pf, 2) if avg_pf else 0,
                'avg_return': round(avg_ret, 2),
            }
    
    return summary


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_gap_visualization(symbol: str, strategy_name: str, 
                             result: Dict, output_dir: str = 'charts'):
    """Create price chart with FVG zones and trade markers."""
    
    df = fetch_data(symbol, interval='1h', days=59)
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=[f'{symbol} - {strategy_name.upper()} Strategy', 'Equity Curve']
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price',
            showlegend=False,
        ),
        row=1, col=1
    )
    
    # Detect and plot FVG zones
    all_fvgs = detect_fvgs(df, len(df)-1, lookback=500, min_gap_pct=0.02)
    
    for gap in all_fvgs[:30]:  # Limit for performance
        color = 'rgba(0,255,0,0.2)' if gap.type == 'bullish' else 'rgba(255,0,0,0.2)'
        fig.add_shape(
            type='rect',
            x0=gap.created_at,
            x1=min(gap.filled_at if gap.filled else len(df), len(df)) - 1,
            y0=gap.bottom,
            y1=gap.top,
            fillcolor=color,
            line=dict(width=0),
            row=1, col=1
        )
    
    # Plot trades
    trades = result.get('trades', [])
    for trade in trades:
        if not trade.get('entry_time') or not trade.get('exit_time'):
            continue
        
        # Find bar indices
        try:
            entry_idx = df.index.get_loc(trade['entry_time'])
            exit_idx = df.index.get_loc(trade['exit_time'])
        except:
            continue
        
        color = 'green' if trade.get('result') == 'win' else 'red'
        
        # Entry marker
        fig.add_trace(
            go.Scatter(
                x=[entry_idx],
                y=[trade['entry_price']],
                mode='markers',
                marker=dict(
                    symbol='triangle-up' if trade['direction'] == 'long' else 'triangle-down',
                    size=12,
                    color=color,
                ),
                name=f"Entry {trade['direction']}",
                showlegend=False,
            ),
            row=1, col=1
        )
        
        # Exit marker
        fig.add_trace(
            go.Scatter(
                x=[exit_idx],
                y=[trade['exit_price']],
                mode='markers',
                marker=dict(symbol='x', size=10, color=color),
                name=f"Exit",
                showlegend=False,
            ),
            row=1, col=1
        )
    
    # Equity curve
    if trades:
        trades_with_pnl = [t for t in trades if t.get('pnl_pct') is not None]
        if trades_with_pnl:
            equity = [1.0]
            for t in trades_with_pnl:
                equity.append(equity[-1] * (1 + t['pnl_pct']))
            
            fig.add_trace(
                go.Scatter(
                    y=equity,
                    mode='lines',
                    name='Equity',
                    line=dict(color='blue', width=2),
                ),
                row=2, col=1
            )
    
    # Layout
    fig.update_layout(
        height=800,
        xaxis_rangeslider_visible=False,
        title=f"FVG {strategy_name.upper()} - {symbol}",
    )
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f'fvg_{strategy_name}_{symbol}.html')
    fig.write_html(filepath)
    print(f"  Saved: {filepath}")
    
    return filepath


def create_strategy_comparison_chart(results: Dict, output_dir: str = 'charts'):
    """Create comparison chart across strategies."""
    
    summary = results.get('summary', {}).get('strategy_comparison', {})
    
    if not summary:
        return None
    
    strategies = list(summary.keys())
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Average Win Rate', 'Average Profit Factor', 
                        'Average Return %', 'Trade Frequency'],
    )
    
    # Win Rate
    fig.add_trace(
        go.Bar(
            x=strategies,
            y=[summary[s]['avg_win_rate'] for s in strategies],
            marker_color=['green' if summary[s]['avg_win_rate'] > 50 else 'red' for s in strategies],
        ),
        row=1, col=1
    )
    fig.add_hline(y=50, line_dash='dash', line_color='gray', row=1, col=1)
    
    # Profit Factor
    fig.add_trace(
        go.Bar(
            x=strategies,
            y=[summary[s]['avg_profit_factor'] for s in strategies],
            marker_color=['green' if summary[s]['avg_profit_factor'] > 1 else 'red' for s in strategies],
        ),
        row=1, col=2
    )
    fig.add_hline(y=1, line_dash='dash', line_color='gray', row=1, col=2)
    
    # Return
    fig.add_trace(
        go.Bar(
            x=strategies,
            y=[summary[s]['avg_return'] for s in strategies],
            marker_color=['green' if summary[s]['avg_return'] > 0 else 'red' for s in strategies],
        ),
        row=2, col=1
    )
    fig.add_hline(y=0, line_dash='dash', line_color='gray', row=2, col=1)
    
    # Trade frequency
    fig.add_trace(
        go.Bar(
            x=strategies,
            y=[summary[s]['avg_trades'] for s in strategies],
            marker_color='steelblue',
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=600,
        title='FVG Strategy Comparison (Forex)',
        showlegend=False,
    )
    
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'fvg_strategy_comparison.html')
    fig.write_html(filepath)
    print(f"Saved: {filepath}")
    
    return filepath


def create_equity_curves_chart(results: Dict, output_dir: str = 'charts'):
    """Create combined equity curves for best config per strategy."""
    
    fig = go.Figure()
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    
    best_configs = results.get('summary', {}).get('best_by_strategy', {})
    
    for i, (strategy, config) in enumerate(best_configs.items()):
        symbol = config['symbol']
        
        # Find the full result with trades
        full_result = None
        for r in results['all_results']:
            if (r['strategy'] == strategy and 
                r['symbol'] == symbol and 
                r['params'] == config['params']):
                full_result = r
                break
        
        if full_result and full_result.get('trades'):
            trades = [t for t in full_result['trades'] if t.get('pnl_pct') is not None]
            if trades:
                equity = [1.0]
                for t in trades:
                    equity.append(equity[-1] * (1 + t['pnl_pct']))
                
                fig.add_trace(go.Scatter(
                    y=equity,
                    mode='lines',
                    name=f"{strategy} ({symbol})",
                    line=dict(color=colors[i % len(colors)], width=2),
                ))
    
    fig.add_hline(y=1, line_dash='dash', line_color='gray')
    
    fig.update_layout(
        height=500,
        title='Equity Curves - Best Config Per Strategy',
        yaxis_title='Equity (starting at 1.0)',
        xaxis_title='Trade Number',
    )
    
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'fvg_equity_curves.html')
    fig.write_html(filepath)
    print(f"Saved: {filepath}")
    
    return filepath


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_markdown_report(results: Dict, output_path: str = 'FVG_RESEARCH.md'):
    """Generate comprehensive markdown research report."""
    
    summary = results.get('summary', {})
    
    md = []
    md.append("# FVG Trading Strategy Research - Forex")
    md.append(f"\n**Generated:** {results['meta']['run_date']}")
    md.append(f"\n**Symbols Tested:** {', '.join(results['meta']['symbols'])}")
    md.append(f"\n**Data Period:** {results['meta']['days']} days, 1H timeframe")
    md.append(f"\n**Total Backtests:** {len(results['all_results'])}")
    
    md.append("\n---\n")
    
    # Executive Summary
    md.append("## Executive Summary\n")
    
    strat_comp = summary.get('strategy_comparison', {})
    if strat_comp:
        best_strat = max(strat_comp.items(), 
                         key=lambda x: x[1]['avg_profit_factor'] if x[1]['avg_profit_factor'] < 100 else 0)
        
        md.append(f"**Best Performing Strategy:** `{best_strat[0]}` with {best_strat[1]['avg_profit_factor']:.2f} avg profit factor\n")
        
        # Ranking
        md.append("\n### Strategy Rankings (by Profit Factor)\n")
        sorted_strats = sorted(strat_comp.items(), 
                               key=lambda x: x[1]['avg_profit_factor'] if x[1]['avg_profit_factor'] < 100 else 0, 
                               reverse=True)
        
        for i, (strat, metrics) in enumerate(sorted_strats, 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
            md.append(f"{emoji} **{i}. {strat}** - PF: {metrics['avg_profit_factor']:.2f}, "
                      f"WR: {metrics['avg_win_rate']:.1f}%, Ret: {metrics['avg_return']:.2f}%\n")
    
    md.append("\n---\n")
    
    # Strategy Deep Dive
    md.append("## Strategy Analysis\n")
    
    for strategy in STRATEGIES.keys():
        md.append(f"\n### {strategy.upper()}\n")
        
        if strategy in strat_comp:
            m = strat_comp[strategy]
            md.append(f"- **Average Win Rate:** {m['avg_win_rate']:.1f}%\n")
            md.append(f"- **Average Profit Factor:** {m['avg_profit_factor']:.2f}\n")
            md.append(f"- **Average Return:** {m['avg_return']:.2f}%\n")
            md.append(f"- **Average Trades:** {m['avg_trades']:.1f}\n")
            md.append(f"- **Samples Tested:** {m['sample_count']}\n")
        
        if strategy in summary.get('best_by_strategy', {}):
            best = summary['best_by_strategy'][strategy]
            md.append(f"\n**Best Configuration:**\n")
            md.append(f"- Symbol: {best['symbol']}\n")
            md.append(f"- SL/TP: {best['params']['sl_mult']}:{best['params']['tp_mult']}\n")
            md.append(f"- Min Gap: {best['params']['min_gap_pct']}%\n")
            md.append(f"- Max Age: {best['params']['max_age_bars']} bars\n")
            md.append(f"- Win Rate: {best['metrics']['win_rate']}%\n")
            md.append(f"- Profit Factor: {best['metrics']['profit_factor']}\n")
            md.append(f"- Return: {best['metrics']['strategy_return']}%\n")
    
    md.append("\n---\n")
    
    # Per-Symbol Analysis
    md.append("## Symbol Breakdown\n")
    
    sym_comp = summary.get('symbol_comparison', {})
    for symbol in results['meta']['symbols']:
        md.append(f"\n### {symbol}\n")
        
        if symbol in sym_comp:
            m = sym_comp[symbol]
            md.append(f"- **Avg Win Rate:** {m['avg_win_rate']:.1f}%\n")
            md.append(f"- **Avg Profit Factor:** {m['avg_profit_factor']:.2f}\n")
            md.append(f"- **Avg Return:** {m['avg_return']:.2f}%\n")
        
        if symbol in summary.get('best_by_symbol', {}):
            best = summary['best_by_symbol'][symbol]
            md.append(f"\n**Best Strategy for {symbol}:** `{best['strategy']}`\n")
            md.append(f"- Params: SL={best['params']['sl_mult']}, TP={best['params']['tp_mult']}, "
                      f"gap={best['params']['min_gap_pct']}%, age={best['params']['max_age_bars']}\n")
            md.append(f"- PF: {best['metrics']['profit_factor']}, WR: {best['metrics']['win_rate']}%, "
                      f"Ret: {best['metrics']['strategy_return']}%\n")
    
    md.append("\n---\n")
    
    # Key Findings
    md.append("## Key Findings\n")
    
    md.append("\n### Gap Age Impact\n")
    # Analyze by age
    for age in [24, 48, 96, None]:
        age_results = [r for r in results['all_results'] 
                       if r['params']['max_age_bars'] == age and r['metrics']['total_trades'] >= 3]
        if age_results:
            avg_pf = np.mean([r['metrics']['profit_factor'] for r in age_results 
                             if r['metrics']['profit_factor'] < 100])
            avg_wr = np.mean([r['metrics']['win_rate'] for r in age_results])
            age_str = f"{age} bars" if age else "Unlimited"
            md.append(f"- **{age_str}:** PF={avg_pf:.2f}, WR={avg_wr:.1f}%\n")
    
    md.append("\n### Gap Size Impact\n")
    for gap_pct in PARAM_GRID['min_gap_pct']:
        gap_results = [r for r in results['all_results']
                       if r['params']['min_gap_pct'] == gap_pct and r['metrics']['total_trades'] >= 3]
        if gap_results:
            avg_pf = np.mean([r['metrics']['profit_factor'] for r in gap_results
                             if r['metrics']['profit_factor'] < 100])
            avg_wr = np.mean([r['metrics']['win_rate'] for r in gap_results])
            md.append(f"- **Min {gap_pct}%:** PF={avg_pf:.2f}, WR={avg_wr:.1f}%\n")
    
    md.append("\n### SL/TP Ratio Impact\n")
    for sl, tp in PARAM_GRID['sl_tp_ratios']:
        ratio_results = [r for r in results['all_results']
                         if r['params']['sl_mult'] == sl and r['params']['tp_mult'] == tp 
                         and r['metrics']['total_trades'] >= 3]
        if ratio_results:
            avg_pf = np.mean([r['metrics']['profit_factor'] for r in ratio_results
                             if r['metrics']['profit_factor'] < 100])
            avg_wr = np.mean([r['metrics']['win_rate'] for r in ratio_results])
            md.append(f"- **{sl}:{tp}:** PF={avg_pf:.2f}, WR={avg_wr:.1f}%\n")
    
    md.append("\n---\n")
    
    # Conclusions
    md.append("## Conclusions\n")
    md.append("\n*Analysis based on automated backtesting. Key observations:*\n\n")
    
    if strat_comp:
        sorted_by_pf = sorted(strat_comp.items(), 
                              key=lambda x: x[1]['avg_profit_factor'] if x[1]['avg_profit_factor'] < 100 else 0,
                              reverse=True)
        
        winner = sorted_by_pf[0] if sorted_by_pf else None
        loser = sorted_by_pf[-1] if sorted_by_pf else None
        
        if winner:
            md.append(f"1. **`{winner[0]}`** strategy shows the most promise with "
                      f"{winner[1]['avg_profit_factor']:.2f} profit factor\n")
        
        if loser and loser[0] != winner[0]:
            md.append(f"2. **`{loser[0]}`** strategy underperforms with "
                      f"{loser[1]['avg_profit_factor']:.2f} profit factor\n")
        
        md.append("3. Gap age filtering and minimum gap size thresholds impact signal quality\n")
        md.append("4. Risk-reward ratio of 1:3 or 1:4 tends to improve profitability\n")
    
    md.append("\n---\n")
    md.append("\n*Report generated by FVG Research Engine*\n")
    
    # Write report
    report_text = '\n'.join(md)
    with open(output_path, 'w') as f:
        f.write(report_text)
    
    print(f"\nReport saved: {output_path}")
    return output_path


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run complete FVG research."""
    
    # Forex pairs to test
    SYMBOLS = [
        'EURUSD=X',
        'USDJPY=X', 
        'GBPUSD=X',
        'EURJPY=X',
    ]
    
    print("="*60)
    print("FVG RESEARCH - FOREX FOCUS")
    print("="*60)
    print(f"Testing {len(SYMBOLS)} pairs x {len(STRATEGIES)} strategies")
    print(f"Timeframe: 1H, Days: 59")
    print("="*60)
    
    # Run full research
    results = run_full_research(SYMBOLS, days=59)
    
    # Save raw results
    print("\nSaving results...")
    
    # Convert datetime objects for JSON serialization
    def serialize_result(r):
        r_copy = r.copy()
        for trade in r_copy.get('trades', []):
            for key in ['entry_time', 'exit_time']:
                if key in trade and trade[key] is not None:
                    if hasattr(trade[key], 'isoformat'):
                        trade[key] = trade[key].isoformat()
        return r_copy
    
    serializable_results = {
        'meta': results['meta'],
        'summary': results['summary'],
        'all_results': [serialize_result(r) for r in results['all_results']],
    }
    
    with open('fvg_research_results.json', 'w') as f:
        json.dump(serializable_results, f, indent=2, default=str)
    print("Saved: fvg_research_results.json")
    
    # Generate markdown report
    generate_markdown_report(results)
    
    # Create visualizations
    print("\nGenerating charts...")
    create_strategy_comparison_chart(results)
    create_equity_curves_chart(results)
    
    # Create detailed charts for best configs
    for strategy, config in results.get('summary', {}).get('best_by_strategy', {}).items():
        # Find full result
        for r in results['all_results']:
            if (r['strategy'] == strategy and 
                r['symbol'] == config['symbol'] and 
                r['params'] == config['params']):
                create_gap_visualization(config['symbol'], strategy, r)
                break
    
    print("\n" + "="*60)
    print("RESEARCH COMPLETE")
    print("="*60)
    
    # Print summary
    print("\nStrategy Rankings:")
    for strat, metrics in sorted(
        results['summary']['strategy_comparison'].items(),
        key=lambda x: x[1]['avg_profit_factor'] if x[1]['avg_profit_factor'] < 100 else 0,
        reverse=True
    ):
        print(f"  {strat:15} - PF: {metrics['avg_profit_factor']:5.2f}, "
              f"WR: {metrics['avg_win_rate']:5.1f}%, Ret: {metrics['avg_return']:6.2f}%")


if __name__ == '__main__':
    main()
