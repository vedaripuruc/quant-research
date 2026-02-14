#!/usr/bin/env python3
"""Generate forex strategy charts"""
import warnings
warnings.filterwarnings('ignore')

import os
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from engine import BacktestEngine, BacktestConfig, calculate_metrics, fetch_data
from strategies_forex import FOREX_STRATEGIES

FOREX = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'EURJPY=X']
BEST_STRATS = ['breakout_fx', 'session_breakout_fx']

forex_config = BacktestConfig(
    slippage_pct=0.0001,
    commission_pct=0.0003,
    position_size=1.0,
    compound=True
)

def create_chart(df, trades_df, symbol, strategy, metrics):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, row_heights=[0.7, 0.3])
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df['Date'], open=df['Open'], high=df['High'], 
        low=df['Low'], close=df['Close'], name='Price'
    ), row=1, col=1)
    
    if not trades_df.empty:
        completed = trades_df[trades_df['exit_price'].notna()]
        
        # Entries
        longs = completed[completed['direction'] == 'long']
        shorts = completed[completed['direction'] == 'short']
        
        if len(longs) > 0:
            fig.add_trace(go.Scatter(
                x=longs['entry_time'], y=longs['entry_price'],
                mode='markers', marker=dict(symbol='triangle-up', size=12, color='green'),
                name='Long'
            ), row=1, col=1)
        
        if len(shorts) > 0:
            fig.add_trace(go.Scatter(
                x=shorts['entry_time'], y=shorts['entry_price'],
                mode='markers', marker=dict(symbol='triangle-down', size=12, color='red'),
                name='Short'
            ), row=1, col=1)
        
        # Exits
        wins = completed[completed['result'] == 'win']
        losses = completed[completed['result'] == 'loss']
        
        if len(wins) > 0:
            fig.add_trace(go.Scatter(
                x=wins['exit_time'], y=wins['exit_price'],
                mode='markers', marker=dict(symbol='circle', size=10, color='blue'),
                name='Win'
            ), row=1, col=1)
        
        if len(losses) > 0:
            fig.add_trace(go.Scatter(
                x=losses['exit_time'], y=losses['exit_price'],
                mode='markers', marker=dict(symbol='x', size=10, color='orange'),
                name='Loss'
            ), row=1, col=1)
        
        # Entry-exit lines
        for _, t in completed.iterrows():
            color = 'green' if t['result'] == 'win' else 'orange'
            fig.add_trace(go.Scatter(
                x=[t['entry_time'], t['exit_time']],
                y=[t['entry_price'], t['exit_price']],
                mode='lines', line=dict(color=color, width=1, dash='dot'),
                showlegend=False
            ), row=1, col=1)
    
    # Volume
    fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], name='Volume', 
                         marker_color='gray', opacity=0.5), row=2, col=1)
    
    beat = "✓" if metrics['strategy_return'] > metrics['buy_hold_return'] else "✗"
    fig.update_layout(
        title=f"<b>{symbol} - {strategy}</b><br>"
              f"<sup>{metrics['total_trades']}t | WR:{metrics['win_rate']:.1f}% | "
              f"Ret:{metrics['strategy_return']:.2f}% vs B&H:{metrics['buy_hold_return']:.2f}% {beat}</sup>",
        xaxis_rangeslider_visible=False,
        height=600,
        template='plotly_dark'
    )
    
    return fig

output_dir = 'charts'
os.makedirs(output_dir, exist_ok=True)

print("Generating forex charts...")

for pair in FOREX:
    df = fetch_data(pair, '1h', 59)
    if len(df) < 30:
        continue
    
    for strat_name in BEST_STRATS:
        print(f"  {pair} - {strat_name}...", end=" ")
        
        engine = BacktestEngine(forex_config)
        trades = engine.run(df, FOREX_STRATEGIES[strat_name])
        metrics = calculate_metrics(trades, df, forex_config)
        
        fig = create_chart(df, trades, pair, strat_name, metrics)
        
        clean_name = pair.replace('=X', '').replace('/', '')
        filename = f"{clean_name}_{strat_name}.html"
        fig.write_html(os.path.join(output_dir, filename), include_plotlyjs='cdn')
        
        print(f"OK ({metrics['total_trades']}t, {metrics['strategy_return']:.2f}%)")

print(f"\nCharts saved to {output_dir}/")
print("Done!")
