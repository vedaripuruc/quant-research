"""
FVG Visualization with Clear Entry/Exit/TP/SL
==============================================
Creates interactive charts showing:
- Candlesticks
- FVG zones (bullish=green, bearish=red)
- Trade entries (triangles)
- TP/SL levels (horizontal lines)
- Exit points
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import json

from engine import fetch_data
from fvg_optimized import FVGTracker, strategy_magnet, strategy_wall, STRATEGIES


def visualize_fvg_strategy(symbol: str, strategy_name: str, 
                           sl_mult: float = 1.5, tp_mult: float = 3.0,
                           days: int = 59, show_last_n: int = 200):
    """
    Create detailed visualization of FVG strategy with trades.
    """
    print(f"Fetching {symbol}...")
    df = fetch_data(symbol, interval='1h', days=days)
    
    if len(df) < 50:
        print(f"Insufficient data: {len(df)} bars")
        return None
    
    # Collect all gaps and trades
    tracker = FVGTracker(min_gap_pct=0.01, max_age=48)
    strategy_fn = STRATEGIES[strategy_name]
    
    all_gaps = []
    trades = []
    
    for i in range(len(df)):
        # Update tracker
        tracker.update(df, i)
        
        # Record all new gaps
        for gap in tracker.gaps:
            if gap.bar_idx == i:
                all_gaps.append({
                    'type': gap.type,
                    'top': gap.top,
                    'bottom': gap.bottom,
                    'midpoint': gap.midpoint,
                    'bar_idx': gap.bar_idx,
                    'time': df.iloc[i]['Date'] if 'Date' in df.columns else df.index[i],
                })
        
        # Check for signals
        if i > 2:
            signal = strategy_fn(df, i, tracker, sl_mult, tp_mult)
            if signal:
                entry_price = df.iloc[i]['Open']
                trades.append({
                    'bar_idx': i,
                    'time': df.iloc[i]['Date'] if 'Date' in df.columns else df.index[i],
                    'direction': signal['direction'],
                    'entry': entry_price,
                    'sl': signal['stop_loss'],
                    'tp': signal['take_profit'],
                    'gap_type': signal['gap'].type,
                })
    
    print(f"Found {len(all_gaps)} gaps, {len(trades)} trades")
    
    # Use last N bars for cleaner chart
    df_plot = df.tail(show_last_n).reset_index(drop=True)
    start_idx = len(df) - show_last_n
    
    # Filter gaps and trades to visible range
    visible_gaps = [g for g in all_gaps if g['bar_idx'] >= start_idx]
    visible_trades = [t for t in trades if t['bar_idx'] >= start_idx]
    
    # Adjust indices
    for g in visible_gaps:
        g['plot_idx'] = g['bar_idx'] - start_idx
    for t in visible_trades:
        t['plot_idx'] = t['bar_idx'] - start_idx
    
    # Create figure
    fig = make_subplots(rows=1, cols=1)
    
    # Add candlesticks
    fig.add_trace(go.Candlestick(
        x=list(range(len(df_plot))),
        open=df_plot['Open'],
        high=df_plot['High'],
        low=df_plot['Low'],
        close=df_plot['Close'],
        name='Price',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350',
    ))
    
    # Add FVG zones
    for gap in visible_gaps:
        color = 'rgba(0, 255, 0, 0.15)' if gap['type'] == 'bullish' else 'rgba(255, 0, 0, 0.15)'
        border_color = 'green' if gap['type'] == 'bullish' else 'red'
        
        # Gap zone (extends to the right)
        end_idx = min(gap['plot_idx'] + 48, len(df_plot) - 1)
        
        fig.add_shape(
            type='rect',
            x0=gap['plot_idx'], x1=end_idx,
            y0=gap['bottom'], y1=gap['top'],
            fillcolor=color,
            line=dict(color=border_color, width=1),
            layer='below',
        )
        
        # Midpoint line
        fig.add_shape(
            type='line',
            x0=gap['plot_idx'], x1=end_idx,
            y0=gap['midpoint'], y1=gap['midpoint'],
            line=dict(color=border_color, width=1, dash='dash'),
            layer='below',
        )
    
    # Add trades with entry, TP, SL
    for trade in visible_trades:
        idx = trade['plot_idx']
        entry = trade['entry']
        sl = trade['sl']
        tp = trade['tp']
        direction = trade['direction']
        
        # Entry marker
        marker_symbol = 'triangle-up' if direction == 'long' else 'triangle-down'
        marker_color = 'blue'
        
        fig.add_trace(go.Scatter(
            x=[idx],
            y=[entry],
            mode='markers',
            marker=dict(symbol=marker_symbol, size=15, color=marker_color, line=dict(width=2, color='white')),
            name=f'{direction.upper()} Entry',
            hovertemplate=f'{direction.upper()}<br>Entry: {entry:.4f}<br>SL: {sl:.4f}<br>TP: {tp:.4f}',
            showlegend=False,
        ))
        
        # SL level (red dashed line)
        fig.add_shape(
            type='line',
            x0=idx, x1=idx + 20,
            y0=sl, y1=sl,
            line=dict(color='red', width=2, dash='dot'),
        )
        
        # TP level (green dashed line)
        fig.add_shape(
            type='line',
            x0=idx, x1=idx + 20,
            y0=tp, y1=tp,
            line=dict(color='green', width=2, dash='dot'),
        )
        
        # Entry level reference (blue)
        fig.add_shape(
            type='line',
            x0=idx, x1=idx + 20,
            y0=entry, y1=entry,
            line=dict(color='blue', width=1, dash='solid'),
        )
        
        # Add annotations
        fig.add_annotation(
            x=idx + 21, y=tp,
            text=f'TP {tp:.4f}',
            showarrow=False,
            font=dict(color='green', size=10),
            xanchor='left',
        )
        fig.add_annotation(
            x=idx + 21, y=sl,
            text=f'SL {sl:.4f}',
            showarrow=False,
            font=dict(color='red', size=10),
            xanchor='left',
        )
    
    # Layout
    fig.update_layout(
        title=f'{symbol} - {strategy_name.upper()} Strategy (SL:{sl_mult} TP:{tp_mult})',
        xaxis_title='Bar',
        yaxis_title='Price',
        template='plotly_dark',
        height=700,
        xaxis_rangeslider_visible=False,
        showlegend=False,
    )
    
    # Save
    filename = f'charts/fvg_{symbol.replace("=", "")}_{strategy_name}.html'
    fig.write_html(filename)
    print(f"Saved: {filename}")
    
    return fig


def create_comparison_chart(results_file: str = 'fvg_stocks_results.json'):
    """Create bar chart comparing all strategies and assets."""
    
    with open(results_file) as f:
        results = json.load(f)
    
    # Extract data
    data = []
    for r in results['all']:
        data.append({
            'symbol': r['symbol'],
            'strategy': r['strategy'],
            'params': f"{r['params']['sl_mult']}:{r['params']['tp_mult']}",
            'return': r['metrics']['strategy_return'],
            'win_rate': r['metrics']['win_rate'],
            'pf': r['metrics']['profit_factor'],
        })
    
    df = pd.DataFrame(data)
    
    # Create figure with subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=['Strategy Return by Asset', 'Profit Factor by Asset'],
        vertical_spacing=0.15,
    )
    
    colors = {'magnet': '#2196F3', 'wall': '#FF9800', 'fill_reversal': '#9C27B0'}
    
    for strategy in ['magnet', 'wall', 'fill_reversal']:
        strat_df = df[(df['strategy'] == strategy) & (df['params'] == '1.0:2.0')]
        
        fig.add_trace(go.Bar(
            x=strat_df['symbol'],
            y=strat_df['return'],
            name=strategy,
            marker_color=colors[strategy],
        ), row=1, col=1)
        
        fig.add_trace(go.Bar(
            x=strat_df['symbol'],
            y=strat_df['pf'],
            name=strategy,
            marker_color=colors[strategy],
            showlegend=False,
        ), row=2, col=1)
    
    fig.update_layout(
        title='FVG Strategy Comparison (1:2 Risk:Reward)',
        template='plotly_dark',
        height=800,
        barmode='group',
    )
    
    fig.add_hline(y=0, line_dash='dash', line_color='white', row=1, col=1)
    fig.add_hline(y=1, line_dash='dash', line_color='white', row=2, col=1)
    
    fig.write_html('charts/fvg_comparison.html')
    print("Saved: charts/fvg_comparison.html")
    
    return fig


if __name__ == '__main__':
    import os
    os.makedirs('charts', exist_ok=True)
    
    # Best performers
    print("\n" + "="*60)
    print("Creating visualizations for best performers...")
    print("="*60)
    
    # Gold Magnet (best)
    visualize_fvg_strategy('GC=F', 'magnet', sl_mult=1.0, tp_mult=2.0, show_last_n=150)
    
    # AAPL Wall
    visualize_fvg_strategy('AAPL', 'wall', sl_mult=1.0, tp_mult=2.0, show_last_n=150)
    
    # Comparison chart
    create_comparison_chart()
    
    print("\n✅ Visualizations complete!")
    print("View at: charts/fvg_*.html")
