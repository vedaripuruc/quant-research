#!/usr/bin/env python3
"""
FVG Confluence Combiner
========================
Combines each of the 5 novel signals with Fair Value Gap (FVG) detection.
Novel signal fires → Check if active FVG in same direction → Take/Skip.

Uses daily OHLC data (yfinance) for both FVG detection and signal generation.
FVG detection adapted from fvg_magnet_backtest.py to work on daily bars.

Entry at NEXT bar's open (no look-ahead bias).
"""

import json
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Callable

sys.path.insert(0, '.')
from engine import BacktestEngine, BacktestConfig, calculate_metrics

# Import signal modules
from entropy_signal import calculate_indicators as entropy_indicators, entropy_signal
from hurst_signal import calculate_indicators as hurst_indicators, hurst_signal
from wavelet_signal import wavelet_signal
from jump_signal import jump_signal, reset_jump_state
from transfer_entropy_signal import (
    build_te_signal_fn, fetch_all_data, SYMBOLS, LEADER_MAP,
    compute_rsi, compute_atr
)


# ──────────────────────────────────────────
# FVG Detection on Daily OHLC
# ──────────────────────────────────────────
class DailyFVG:
    """A Fair Value Gap detected on daily bars."""
    def __init__(self, fvg_type: str, top: float, bottom: float, bar_idx: int, 
                 size: float, formed_close: float):
        self.fvg_type = fvg_type    # 'bullish' or 'bearish'
        self.top = top
        self.bottom = bottom
        self.bar_idx = bar_idx       # Bar index where FVG completed
        self.size = size             # Size in absolute price terms
        self.formed_close = formed_close
        self.filled = False          # Has price filled the gap?
        self.mid = (top + bottom) / 2


def detect_fvgs(df: pd.DataFrame, min_size_pct: float = 0.001) -> List[DailyFVG]:
    """
    Detect all FVGs in daily OHLC data.
    
    Bullish FVG: bar[i-2].High < bar[i].Low  (gap up — demand zone)
    Bearish FVG: bar[i-2].Low > bar[i].High  (gap down — supply zone)
    
    min_size_pct: minimum gap size as % of price (default 0.1%)
    """
    fvgs = []
    
    for i in range(2, len(df)):
        left = df.iloc[i - 2]
        right = df.iloc[i]
        price_level = df.iloc[i]['Close']
        
        # Bullish FVG (gap up)
        if left['High'] < right['Low']:
            bottom = left['High']
            top = right['Low']
            size = top - bottom
            size_pct = size / price_level if price_level > 0 else 0
            
            if size_pct >= min_size_pct:
                fvgs.append(DailyFVG(
                    fvg_type='bullish',
                    top=top,
                    bottom=bottom,
                    bar_idx=i,
                    size=size,
                    formed_close=price_level,
                ))
        
        # Bearish FVG (gap down)
        if left['Low'] > right['High']:
            top = left['Low']
            bottom = right['High']
            size = top - bottom
            size_pct = size / price_level if price_level > 0 else 0
            
            if size_pct >= min_size_pct:
                fvgs.append(DailyFVG(
                    fvg_type='bearish',
                    top=top,
                    bottom=bottom,
                    bar_idx=i,
                    size=size,
                    formed_close=price_level,
                ))
    
    return fvgs


def get_active_fvgs(all_fvgs: List[DailyFVG], df: pd.DataFrame, 
                     current_bar: int, max_age_bars: int = 20) -> List[DailyFVG]:
    """
    Get FVGs that are still active (not filled) at current_bar.
    
    An FVG is "filled" when price passes completely through it:
    - Bullish FVG filled when Low goes below bottom
    - Bearish FVG filled when High goes above top
    """
    active = []
    
    for fvg in all_fvgs:
        # Must be formed before current bar
        if fvg.bar_idx >= current_bar:
            continue
        
        # Check age
        if current_bar - fvg.bar_idx > max_age_bars:
            continue
        
        # Check if filled between formation and now
        filled = False
        for j in range(fvg.bar_idx + 1, current_bar + 1):
            if j >= len(df):
                break
            if fvg.fvg_type == 'bullish':
                if df.iloc[j]['Low'] < fvg.bottom:
                    filled = True
                    break
            else:
                if df.iloc[j]['High'] > fvg.top:
                    filled = True
                    break
        
        if not filled:
            active.append(fvg)
    
    return active


def fvg_confirms_direction(active_fvgs: List[DailyFVG], direction: str, 
                            current_price: float) -> bool:
    """
    Check if any active FVG confirms the signal direction.
    
    Bullish FVG (demand zone below) confirms LONG signals
    - Price should be near/above the FVG (FVG acts as magnet pulling up)
    Bearish FVG (supply zone above) confirms SHORT signals
    - Price should be near/below the FVG (FVG acts as magnet pulling down)
    """
    for fvg in active_fvgs:
        if direction == 'long' and fvg.fvg_type == 'bullish':
            # Bullish FVG below price → confirms long
            # Price should be within 3x FVG size above the FVG
            distance = current_price - fvg.top
            if -fvg.size <= distance <= fvg.size * 3:
                return True
        
        elif direction == 'short' and fvg.fvg_type == 'bearish':
            # Bearish FVG above price → confirms short
            distance = fvg.bottom - current_price
            if -fvg.size <= distance <= fvg.size * 3:
                return True
    
    return False


# ──────────────────────────────────────────
# Confluence Signal Wrappers
# ──────────────────────────────────────────
def make_confluence_signal(base_signal_fn: Callable, all_fvgs: List[DailyFVG],
                            max_fvg_age: int = 20) -> Callable:
    """
    Wrap a base signal function with FVG confluence filter.
    Only passes signals where an active FVG confirms the direction.
    """
    def confluence_fn(df, i):
        # Get base signal
        signal = base_signal_fn(df, i)
        if signal is None:
            return None
        
        # Get active FVGs at this bar
        current_price = df.iloc[i]['Close']
        active = get_active_fvgs(all_fvgs, df, i, max_age_bars=max_fvg_age)
        
        # Check if FVG confirms
        if fvg_confirms_direction(active, signal['direction'], current_price):
            return signal
        
        # No FVG confirmation → skip
        return None
    
    return confluence_fn


# ──────────────────────────────────────────
# Data Fetching
# ──────────────────────────────────────────
def fetch_data(symbol: str, period: str = '2y') -> pd.DataFrame:
    """Fetch daily data for a symbol."""
    df = yf.download(symbol, period=period, interval='1d', progress=False)
    df.reset_index(inplace=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    if 'Datetime' in df.columns:
        df.rename(columns={'Datetime': 'Date'}, inplace=True)
    return df


# ──────────────────────────────────────────
# Walk-Forward Test
# ──────────────────────────────────────────
def walk_forward(df: pd.DataFrame, signal_fn: Callable, n_windows: int = 4) -> dict:
    """Simple walk-forward: split data into n_windows, test each."""
    n = len(df)
    window_size = n // n_windows
    
    results = []
    for w in range(n_windows):
        start = w * window_size
        end = min(start + window_size, n)
        
        if end - start < 40:
            continue
        
        window_df = df.iloc[start:end].copy().reset_index(drop=True)
        
        engine = BacktestEngine(BacktestConfig(
            slippage_pct=0.001,
            commission_pct=0.001,
        ))
        trades_df = engine.run(window_df, signal_fn)
        metrics = calculate_metrics(trades_df, window_df)
        
        results.append({
            'window': w + 1,
            'trades': metrics['total_trades'],
            'return': metrics['strategy_return'],
            'win_rate': metrics['win_rate'],
            'profit_factor': metrics['profit_factor'],
        })
    
    profitable = sum(1 for r in results if r['return'] > 0)
    
    return {
        'windows': results,
        'profitable_pct': round(profitable / len(results) * 100, 1) if results else 0,
        'profitable_windows': profitable,
        'total_windows': len(results),
    }


# ──────────────────────────────────────────
# Main Runner
# ──────────────────────────────────────────
def run_confluence():
    print("=" * 70)
    print("FVG CONFLUENCE COMBINER")
    print("=" * 70)
    
    # Fetch all data
    print("\n📥 Fetching 2Y daily data...")
    symbols = ['EURUSD=X', 'LINK-USD', 'ADA-USD', 'XRP-USD', 'GC=F', 'BTC-USD']
    data = {}
    for sym in symbols:
        print(f"  {sym}...", end=" ")
        df = fetch_data(sym)
        data[sym] = df
        print(f"{len(df)} bars")
    
    all_results = {}
    
    # ─── Signal definitions ───
    signal_configs = {
        # SKIP entropy — it produced 0 trades on ALL assets (SampEn too restrictive)
        # 'entropy': {
        #     'name': 'Entropy Regime',
        #     'prepare': lambda df: entropy_indicators(df),
        #     'signal_fn': entropy_signal,
        #     'results_file': 'results_entropy.json',
        # },
        'hurst': {
            'name': 'Hurst Exponent',
            'prepare': lambda df: hurst_indicators(df),
            'signal_fn': hurst_signal,
            'results_file': 'results_hurst.json',
        },
        'wavelet': {
            'name': 'Wavelet Momentum',
            'prepare': lambda df: df,  # wavelet_signal handles internally
            'signal_fn': wavelet_signal,
            'results_file': 'results_wavelet.json',
        },
        'jump': {
            'name': 'Jump Detection',
            'prepare': lambda df: df,
            'signal_fn': jump_signal,
            'results_file': 'results_jump.json',
            'pre_run': reset_jump_state,
        },
        'transfer_entropy': {
            'name': 'Transfer Entropy',
            'results_file': 'results_transfer_entropy.json',
            # Special handling — needs leader data
        },
    }
    
    for sig_key, sig_config in signal_configs.items():
        print(f"\n{'━' * 70}")
        print(f"🔬 {sig_config['name']} + FVG Confluence")
        print(f"{'━' * 70}")
        
        sig_results = {}
        
        for symbol in symbols:
            if symbol == 'BTC-USD' and sig_key == 'transfer_entropy':
                continue  # BTC is the leader
            
            df = data[symbol].copy()
            if len(df) < 100:
                print(f"  {symbol}: SKIP (insufficient data)")
                continue
            
            # Detect all FVGs in the data
            all_fvgs = detect_fvgs(df, min_size_pct=0.001)
            
            # Prepare data and signal function
            if sig_key == 'transfer_entropy':
                leader_sym = LEADER_MAP.get(symbol, 'BTC-USD')
                if leader_sym not in data:
                    print(f"  {symbol}: SKIP (no leader data)")
                    continue
                leader_df = data[leader_sym].copy()
                base_signal_fn = build_te_signal_fn(df, leader_df)
                prepared_df = df.copy().reset_index(drop=True)
            else:
                prepare_fn = sig_config['prepare']
                prepared_df = prepare_fn(df).reset_index(drop=True)
                base_signal_fn = sig_config['signal_fn']
            
            # Pre-run callback (e.g., reset state)
            if 'pre_run' in sig_config:
                sig_config['pre_run']()
            
            # Run STANDALONE backtest
            engine = BacktestEngine(BacktestConfig(
                slippage_pct=0.001,
                commission_pct=0.001,
            ))
            standalone_trades = engine.run(prepared_df.copy(), base_signal_fn)
            standalone_metrics = calculate_metrics(standalone_trades, prepared_df)
            
            # Detect FVGs on prepared_df
            all_fvgs_prep = detect_fvgs(prepared_df, min_size_pct=0.001)
            
            # Build confluence signal
            if 'pre_run' in sig_config:
                sig_config['pre_run']()
            
            confluence_fn = make_confluence_signal(base_signal_fn, all_fvgs_prep, max_fvg_age=20)
            
            # Run CONFLUENCE backtest
            engine2 = BacktestEngine(BacktestConfig(
                slippage_pct=0.001,
                commission_pct=0.001,
            ))
            confluence_trades = engine2.run(prepared_df.copy(), confluence_fn)
            confluence_metrics = calculate_metrics(confluence_trades, prepared_df)
            
            # Walk-forward on confluence
            if 'pre_run' in sig_config:
                sig_config['pre_run']()
            
            wf = walk_forward(prepared_df, confluence_fn, n_windows=4)
            
            # FVG stats
            n_fvgs = len(all_fvgs_prep)
            n_bullish = sum(1 for f in all_fvgs_prep if f.fvg_type == 'bullish')
            n_bearish = n_fvgs - n_bullish
            
            print(f"\n  📊 {symbol}")
            print(f"     FVGs detected: {n_fvgs} ({n_bullish} bullish, {n_bearish} bearish)")
            print(f"     Standalone: {standalone_metrics['total_trades']} trades, "
                  f"WR={standalone_metrics['win_rate']}%, "
                  f"PF={standalone_metrics['profit_factor']}, "
                  f"Return={standalone_metrics['strategy_return']}%")
            print(f"     Confluence: {confluence_metrics['total_trades']} trades, "
                  f"WR={confluence_metrics['win_rate']}%, "
                  f"PF={confluence_metrics['profit_factor']}, "
                  f"Return={confluence_metrics['strategy_return']}%")
            print(f"     WF OOS Profitable: {wf['profitable_windows']}/{wf['total_windows']} "
                  f"({wf['profitable_pct']}%)")
            
            # Compare
            trade_reduction = 0
            if standalone_metrics['total_trades'] > 0:
                trade_reduction = round((1 - confluence_metrics['total_trades'] / 
                                        standalone_metrics['total_trades']) * 100, 1)
            
            pf_delta = round(confluence_metrics['profit_factor'] - standalone_metrics['profit_factor'], 2)
            wr_delta = round(confluence_metrics['win_rate'] - standalone_metrics['win_rate'], 1)
            
            improved = (confluence_metrics['profit_factor'] > standalone_metrics['profit_factor'] and
                       confluence_metrics['total_trades'] >= 3)
            
            print(f"     Trade reduction: {trade_reduction}%")
            print(f"     PF delta: {'+' if pf_delta >= 0 else ''}{pf_delta}")
            print(f"     WR delta: {'+' if wr_delta >= 0 else ''}{wr_delta}%")
            print(f"     {'✅ IMPROVED' if improved else '❌ NO IMPROVEMENT'}")
            
            sig_results[symbol] = {
                'standalone': standalone_metrics,
                'confluence': confluence_metrics,
                'walk_forward_confluence': {
                    'profitable_pct': wf['profitable_pct'],
                    'profitable_windows': wf['profitable_windows'],
                    'total_windows': wf['total_windows'],
                    'windows': wf['windows'],
                },
                'fvg_stats': {
                    'total_fvgs': n_fvgs,
                    'bullish': n_bullish,
                    'bearish': n_bearish,
                },
                'comparison': {
                    'trade_reduction_pct': trade_reduction,
                    'pf_delta': pf_delta,
                    'wr_delta': wr_delta,
                    'improved': improved,
                },
            }
        
        all_results[sig_key] = {
            'signal_name': sig_config['name'],
            'assets': sig_results,
        }
    
    # Save
    output_path = 'results_fvg_confluence.json'
    
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, pd.Timestamp):
                return str(obj)
            return super().default(obj)
    
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, cls=NpEncoder, default=str)
    print(f"\n\n💾 Confluence results saved to {output_path}")
    
    return all_results


if __name__ == '__main__':
    results = run_confluence()
