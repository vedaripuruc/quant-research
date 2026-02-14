#!/usr/bin/env python3
"""
Detailed visualization with proper spread and clear SL/TP
"""
import warnings
warnings.filterwarnings('ignore')

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from engine import BacktestEngine, BacktestConfig, calculate_metrics, fetch_data
from strategies_forex import session_breakout_forex, breakout_forex

# Realistic forex spreads (in price terms, roughly)
# EUR/USD: ~1 pip = 0.0001
# USD/JPY: ~1.5 pip = 0.015
# EUR/JPY: ~2 pip = 0.02

SPREAD_PIPS = {
    'EURUSD=X': 0.00015,  # 1.5 pips
    'USDJPY=X': 0.02,     # 2 pips (in JPY terms)
    'EURJPY=X': 0.025,    # 2.5 pips
    'GBPUSD=X': 0.0002,   # 2 pips
}

def backtest_with_spread(symbol, strategy_fn, days=59):
    """Backtest with realistic spread modeling"""
    df = fetch_data(symbol, '1h', days)
    if len(df) < 30:
        return None, None, None
    
    spread = SPREAD_PIPS.get(symbol, 0.0002)
    
    # Spread = slippage on entry AND exit
    # Total cost per trade = spread * 2 (round trip)
    config = BacktestConfig(
        slippage_pct=0,  # We'll model spread directly
        commission_pct=0,  # Spread includes commission for forex
        position_size=1.0,
        compound=True
    )
    
    # Adjust prices to simulate spread
    df_adjusted = df.copy()
    # Buy at ask (higher), sell at bid (lower)
    # For simplicity, widen the high/low by half spread each way
    df_adjusted['High'] = df['High'] + spread/2
    df_adjusted['Low'] = df['Low'] - spread/2
    
    engine = BacktestEngine(config)
    trades_df = engine.run(df_adjusted, strategy_fn)
    
    # Manually deduct spread from each trade's P&L
    if not trades_df.empty:
        completed = trades_df[trades_df['exit_price'].notna()].copy()
        price = df['Close'].mean()
        spread_pct = (spread / price) * 100
        # Deduct spread from pnl_pct (entry + exit = 2x spread)
        completed['pnl_pct'] = completed['pnl_pct'] - (2 * spread / price)
        trades_df = completed
    
    metrics = calculate_metrics(trades_df, df, config)
    
    return df, trades_df, metrics


def create_detailed_chart(df, trades_df, symbol, strategy, metrics, spread):
    """Create a detailed, zoomable chart with clear SL/TP"""
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(
            f'{symbol} - {strategy}',
            'Trade P&L (%)',
            'Cumulative P&L (%)'
        )
    )
    
    # Main candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350',
            increasing_fillcolor='#26a69a',
            decreasing_fillcolor='#ef5350',
        ),
        row=1, col=1
    )
    
    if not trades_df.empty:
        completed = trades_df[trades_df['exit_price'].notna()].copy()
        
        # Draw each trade with clear SL/TP boxes
        for idx, trade in completed.iterrows():
            entry_time = trade['entry_time']
            exit_time = trade['exit_time']
            entry_price = trade['entry_price']
            exit_price = trade['exit_price']
            sl = trade['stop_loss']
            tp = trade['take_profit']
            direction = trade['direction']
            result = trade['result']
            
            # Color based on result
            if result == 'win':
                trade_color = 'rgba(0, 200, 83, 0.3)'
                line_color = '#00c853'
            else:
                trade_color = 'rgba(255, 82, 82, 0.3)'
                line_color = '#ff5252'
            
            # Draw trade zone (entry to exit)
            fig.add_shape(
                type="rect",
                x0=entry_time, x1=exit_time,
                y0=min(entry_price, exit_price, sl, tp),
                y1=max(entry_price, exit_price, sl, tp),
                fillcolor=trade_color,
                line=dict(color=line_color, width=1),
                row=1, col=1
            )
            
            # Entry line (thick)
            fig.add_shape(
                type="line",
                x0=entry_time, x1=exit_time,
                y0=entry_price, y1=entry_price,
                line=dict(color='white', width=2, dash='solid'),
                row=1, col=1
            )
            
            # Stop Loss line (red dashed)
            fig.add_shape(
                type="line",
                x0=entry_time, x1=exit_time,
                y0=sl, y1=sl,
                line=dict(color='#ff1744', width=2, dash='dash'),
                row=1, col=1
            )
            
            # Take Profit line (green dashed)
            fig.add_shape(
                type="line",
                x0=entry_time, x1=exit_time,
                y0=tp, y1=tp,
                line=dict(color='#00e676', width=2, dash='dash'),
                row=1, col=1
            )
            
            # Entry marker
            marker_symbol = 'triangle-up' if direction == 'long' else 'triangle-down'
            marker_color = '#00c853' if direction == 'long' else '#ff1744'
            
            fig.add_trace(
                go.Scatter(
                    x=[entry_time],
                    y=[entry_price],
                    mode='markers',
                    marker=dict(symbol=marker_symbol, size=15, color=marker_color,
                               line=dict(width=2, color='white')),
                    name=f'{direction.upper()} Entry',
                    showlegend=False,
                    hovertemplate=f'<b>{direction.upper()} ENTRY</b><br>'
                                  f'Price: {entry_price:.5f}<br>'
                                  f'SL: {sl:.5f}<br>'
                                  f'TP: {tp:.5f}<br>'
                                  f'<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Exit marker
            exit_symbol = 'circle' if result == 'win' else 'x'
            exit_color = '#2196f3' if result == 'win' else '#ff9800'
            
            fig.add_trace(
                go.Scatter(
                    x=[exit_time],
                    y=[exit_price],
                    mode='markers',
                    marker=dict(symbol=exit_symbol, size=12, color=exit_color,
                               line=dict(width=2, color='white')),
                    name=f'{result.upper()} Exit',
                    showlegend=False,
                    hovertemplate=f'<b>{result.upper()} EXIT</b><br>'
                                  f'Price: {exit_price:.5f}<br>'
                                  f'P&L: {trade["pnl_pct"]*100:.2f}%<br>'
                                  f'<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Trade P&L bar chart
        colors = ['#00c853' if r == 'win' else '#ff5252' for r in completed['result']]
        fig.add_trace(
            go.Bar(
                x=completed['exit_time'],
                y=completed['pnl_pct'] * 100,
                marker_color=colors,
                name='Trade P&L',
                hovertemplate='P&L: %{y:.2f}%<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Cumulative P&L line
        cum_pnl = (1 + completed['pnl_pct']).cumprod() - 1
        fig.add_trace(
            go.Scatter(
                x=completed['exit_time'],
                y=cum_pnl * 100,
                mode='lines+markers',
                line=dict(color='#2196f3', width=2),
                marker=dict(size=6),
                name='Cumulative P&L',
                hovertemplate='Cumulative: %{y:.2f}%<extra></extra>'
            ),
            row=3, col=1
        )
        
        # Zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)
    
    # Calculate stats
    if not trades_df.empty and len(trades_df) > 0:
        wins = len(trades_df[trades_df['result'] == 'win'])
        total = len(trades_df)
        final_return = (1 + trades_df['pnl_pct']).prod() - 1
    else:
        wins = 0
        total = 0
        final_return = 0
    
    # Layout
    spread_pips = spread * 10000 if 'JPY' not in symbol else spread * 100
    
    fig.update_layout(
        title=dict(
            text=f"<b>{symbol} - {strategy}</b><br>"
                 f"<sup>Spread: {spread_pips:.1f} pips | "
                 f"Trades: {total} | Wins: {wins} | "
                 f"WR: {wins/total*100:.1f}% | "
                 f"Return: {final_return*100:.2f}%</sup>",
            x=0.5,
            font=dict(size=16)
        ),
        height=900,
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        # Enable zooming
        dragmode='zoom',
    )
    
    # Y-axis labels
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="P&L %", row=2, col=1)
    fig.update_yaxes(title_text="Cum %", row=3, col=1)
    
    # Add annotation explaining colors
    fig.add_annotation(
        text="🟢 TP | 🔴 SL | ⬜ Entry | Green box = Win | Red box = Loss",
        xref="paper", yref="paper",
        x=0, y=-0.05,
        showarrow=False,
        font=dict(size=10, color="gray")
    )
    
    return fig


# Generate charts for FTMO candidates
print("Generating detailed charts with spread...")

pairs = [
    ('USDJPY=X', 'USD/JPY'),
    ('EURJPY=X', 'EUR/JPY'),
    ('EURUSD=X', 'EUR/USD'),
]

for symbol, name in pairs:
    spread = SPREAD_PIPS.get(symbol, 0.0002)
    
    print(f"\n{name}:")
    
    # Session Breakout
    df, trades, metrics = backtest_with_spread(symbol, session_breakout_forex)
    if df is not None and trades is not None:
        fig = create_detailed_chart(df, trades, name, 'Session Breakout', metrics, spread)
        filename = f"charts/{symbol.replace('=X', '')}_session_breakout_DETAILED.html"
        fig.write_html(filename, include_plotlyjs='cdn')
        
        wins = len(trades[trades['result'] == 'win']) if not trades.empty else 0
        total = len(trades) if not trades.empty else 0
        ret = ((1 + trades['pnl_pct']).prod() - 1) * 100 if not trades.empty else 0
        print(f"  Session Breakout: {total}t | {wins}W | {ret:.2f}% → {filename}")
    
    # Breakout FX
    df, trades, metrics = backtest_with_spread(symbol, breakout_forex)
    if df is not None and trades is not None:
        fig = create_detailed_chart(df, trades, name, 'Breakout FX', metrics, spread)
        filename = f"charts/{symbol.replace('=X', '')}_breakout_DETAILED.html"
        fig.write_html(filename, include_plotlyjs='cdn')
        
        wins = len(trades[trades['result'] == 'win']) if not trades.empty else 0
        total = len(trades) if not trades.empty else 0
        ret = ((1 + trades['pnl_pct']).prod() - 1) * 100 if not trades.empty else 0
        print(f"  Breakout FX: {total}t | {wins}W | {ret:.2f}% → {filename}")

print("\nDone! Check charts/*_DETAILED.html")
