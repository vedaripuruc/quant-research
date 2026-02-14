"""
FVG Research - Optimized Version
================================
Key optimization: Incremental gap tracking instead of O(n²) rescan.

Strategies:
1. Magnet - Price crosses midpoint, trade WITH gap
2. Wall - Price touches edge and reverses, trade AGAINST gap
3. Fill Reversal - After gap fills, trade reversal
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List, Literal
from datetime import datetime, timedelta
import json

from engine import BacktestEngine, BacktestConfig, calculate_metrics, fetch_data


@dataclass
class Gap:
    """Fair Value Gap."""
    type: Literal['bullish', 'bearish']
    top: float  # Upper bound
    bottom: float  # Lower bound
    midpoint: float
    size: float
    bar_idx: int
    filled: bool = False
    used: bool = False


class FVGTracker:
    """
    Incrementally track FVGs - O(n) per bar instead of O(n²).
    """
    def __init__(self, min_gap_pct: float = 0.01, max_age: int = 48):
        self.gaps: List[Gap] = []
        self.min_gap_pct = min_gap_pct
        self.max_age = max_age
    
    def update(self, df: pd.DataFrame, i: int):
        """Update gap state for bar i. Call once per bar."""
        if i < 2:
            return
        
        current = df.iloc[i]
        high, low = current['High'], current['Low']
        
        # 1. Check for new gap (3-bar pattern)
        bar_2ago = df.iloc[i-2]
        high_2ago, low_2ago = bar_2ago['High'], bar_2ago['Low']
        
        # Bullish FVG: current low > 2-bars-ago high
        if low > high_2ago:
            gap_size = low - high_2ago
            gap_pct = gap_size / high_2ago * 100
            if gap_pct >= self.min_gap_pct:
                self.gaps.append(Gap(
                    type='bullish',
                    top=low,
                    bottom=high_2ago,
                    midpoint=(low + high_2ago) / 2,
                    size=gap_size,
                    bar_idx=i,
                ))
        
        # Bearish FVG: 2-bars-ago low > current high
        elif low_2ago > high:
            gap_size = low_2ago - high
            gap_pct = gap_size / high * 100
            if gap_pct >= self.min_gap_pct:
                self.gaps.append(Gap(
                    type='bearish',
                    top=low_2ago,
                    bottom=high,
                    midpoint=(low_2ago + high) / 2,
                    size=gap_size,
                    bar_idx=i,
                ))
        
        # 2. Update fill status for existing gaps
        for gap in self.gaps:
            if gap.filled:
                continue
            if gap.type == 'bullish' and low <= gap.bottom:
                gap.filled = True
            elif gap.type == 'bearish' and high >= gap.top:
                gap.filled = True
        
        # 3. Remove old gaps
        if self.max_age:
            self.gaps = [g for g in self.gaps if i - g.bar_idx <= self.max_age]
    
    def get_open_gaps(self) -> List[Gap]:
        """Get unfilled, unused gaps."""
        return [g for g in self.gaps if not g.filled and not g.used]
    
    def get_filled_gaps(self) -> List[Gap]:
        """Get filled but unused gaps."""
        return [g for g in self.gaps if g.filled and not g.used]


# =============================================================================
# STRATEGIES
# =============================================================================

def strategy_magnet(df: pd.DataFrame, i: int, tracker: FVGTracker,
                    sl_mult: float = 1.5, tp_mult: float = 3.0) -> Optional[Dict]:
    """
    MAGNET: Price is drawn to fill gaps.
    Entry: Price crosses through gap midpoint
    Direction: WITH the gap (bullish gap → long)
    """
    if i < 3:
        return None
    
    current = df.iloc[i]
    prev = df.iloc[i-1]
    
    for gap in tracker.get_open_gaps():
        mid = gap.midpoint
        
        if gap.type == 'bullish':
            # Price crossing UP through midpoint
            if prev['Close'] < mid <= current['Close']:
                gap.used = True
                return {
                    'direction': 'long',
                    'stop_loss': current['Open'] - sl_mult * gap.size,
                    'take_profit': current['Open'] + tp_mult * gap.size,
                    'gap': gap,
                }
        else:  # bearish
            # Price crossing DOWN through midpoint
            if prev['Close'] > mid >= current['Close']:
                gap.used = True
                return {
                    'direction': 'short',
                    'stop_loss': current['Open'] + sl_mult * gap.size,
                    'take_profit': current['Open'] - tp_mult * gap.size,
                    'gap': gap,
                }
    return None


def strategy_wall(df: pd.DataFrame, i: int, tracker: FVGTracker,
                  sl_mult: float = 1.5, tp_mult: float = 3.0) -> Optional[Dict]:
    """
    WALL: Gaps act as support/resistance.
    Entry: Price touches gap edge and shows rejection
    Direction: AGAINST the gap (bullish gap touched from above → short)
    """
    if i < 3:
        return None
    
    current = df.iloc[i]
    prev = df.iloc[i-1]
    
    for gap in tracker.get_open_gaps():
        if gap.type == 'bullish':
            # Price coming down, touching top of bullish gap, bouncing
            if prev['Low'] > gap.top >= current['Low'] and current['Close'] > gap.top:
                gap.used = True
                return {
                    'direction': 'long',  # Bounce up off support
                    'stop_loss': gap.bottom - sl_mult * gap.size,
                    'take_profit': current['Open'] + tp_mult * gap.size,
                    'gap': gap,
                }
        else:  # bearish
            # Price coming up, touching bottom of bearish gap, bouncing
            if prev['High'] < gap.bottom <= current['High'] and current['Close'] < gap.bottom:
                gap.used = True
                return {
                    'direction': 'short',  # Bounce down off resistance
                    'stop_loss': gap.top + sl_mult * gap.size,
                    'take_profit': current['Open'] - tp_mult * gap.size,
                    'gap': gap,
                }
    return None


def strategy_fill_reversal(df: pd.DataFrame, i: int, tracker: FVGTracker,
                           sl_mult: float = 1.5, tp_mult: float = 3.0) -> Optional[Dict]:
    """
    FILL REVERSAL: After gap fills, expect mean reversion.
    Entry: Gap just got filled this bar
    Direction: AGAINST original gap direction (filled bullish → short)
    """
    if i < 3:
        return None
    
    current = df.iloc[i]
    prev = df.iloc[i-1]
    
    for gap in tracker.gaps:
        if gap.used:
            continue
        
        # Check if gap JUST got filled this bar
        if gap.type == 'bullish':
            # Was open, now filled (price broke below gap bottom)
            if prev['Low'] > gap.bottom >= current['Low']:
                gap.used = True
                return {
                    'direction': 'short',  # Reversal after fill
                    'stop_loss': gap.top + sl_mult * gap.size,
                    'take_profit': current['Open'] - tp_mult * gap.size,
                    'gap': gap,
                }
        else:  # bearish
            # Was open, now filled (price broke above gap top)
            if prev['High'] < gap.top <= current['High']:
                gap.used = True
                return {
                    'direction': 'long',  # Reversal after fill
                    'stop_loss': gap.bottom - sl_mult * gap.size,
                    'take_profit': current['Open'] + tp_mult * gap.size,
                    'gap': gap,
                }
    return None


STRATEGIES = {
    'magnet': strategy_magnet,
    'wall': strategy_wall,
    'fill_reversal': strategy_fill_reversal,
}


# =============================================================================
# BACKTESTING
# =============================================================================

def run_backtest(symbol: str, strategy_name: str, 
                 sl_mult: float = 1.5, tp_mult: float = 3.0,
                 min_gap_pct: float = 0.01, max_age: int = 48,
                 days: int = 59) -> Dict:
    """Run single backtest."""
    
    df = fetch_data(symbol, interval='1h', days=days)
    if len(df) < 50:
        return {'error': f'Insufficient data: {len(df)} bars'}
    
    # Create tracker
    tracker = FVGTracker(min_gap_pct=min_gap_pct, max_age=max_age)
    strategy_fn = STRATEGIES[strategy_name]
    
    # Signal function that updates tracker each bar
    def signal_fn(df_inner, i):
        tracker.update(df_inner, i)
        return strategy_fn(df_inner, i, tracker, sl_mult, tp_mult)
    
    config = BacktestConfig(
        slippage_pct=0.0005,
        commission_pct=0.0003,
        compound=True,
    )
    
    engine = BacktestEngine(config)
    trades_df = engine.run(df, signal_fn)
    metrics = calculate_metrics(trades_df, df, config)
    
    return {
        'symbol': symbol,
        'strategy': strategy_name,
        'params': {'sl_mult': sl_mult, 'tp_mult': tp_mult, 
                   'min_gap_pct': min_gap_pct, 'max_age': max_age},
        'metrics': metrics,
        'trades': trades_df.to_dict('records') if not trades_df.empty else [],
        'bars': len(df),
    }


def run_research(symbols: List[str], days: int = 59) -> Dict:
    """Run FVG research with reduced parameter grid."""
    
    # Reduced grid for speed
    PARAMS = [
        {'sl_mult': 1.0, 'tp_mult': 2.0},  # 1:2
        {'sl_mult': 1.0, 'tp_mult': 3.0},  # 1:3
        {'sl_mult': 1.5, 'tp_mult': 3.0},  # 1.5:3
    ]
    
    results = {
        'meta': {
            'run_date': datetime.now().isoformat(),
            'symbols': symbols,
            'strategies': list(STRATEGIES.keys()),
        },
        'all': [],
        'by_strategy': {},
        'summary': {},
    }
    
    total = len(symbols) * len(STRATEGIES) * len(PARAMS)
    print(f"Running {total} backtests...")
    
    count = 0
    for symbol in symbols:
        print(f"\n{symbol}:")
        for strat_name in STRATEGIES:
            for params in PARAMS:
                count += 1
                result = run_backtest(
                    symbol, strat_name,
                    sl_mult=params['sl_mult'],
                    tp_mult=params['tp_mult'],
                    days=days,
                )
                
                if 'error' not in result:
                    results['all'].append(result)
                    m = result['metrics']
                    print(f"  {strat_name:15} {params['sl_mult']}:{params['tp_mult']} "
                          f"WR={m['win_rate']:5.1f}% Ret={m['strategy_return']:+6.2f}% "
                          f"PF={m['profit_factor']:4.2f} T={m['total_trades']}")
                else:
                    print(f"  {strat_name:15} ERROR: {result['error']}")
    
    # Summarize
    for strat in STRATEGIES:
        strat_results = [r for r in results['all'] if r['strategy'] == strat]
        if strat_results:
            avg_ret = np.mean([r['metrics']['strategy_return'] for r in strat_results])
            avg_wr = np.mean([r['metrics']['win_rate'] for r in strat_results])
            avg_pf = np.mean([r['metrics']['profit_factor'] for r in strat_results])
            profitable = len([r for r in strat_results if r['metrics']['strategy_return'] > 0])
            results['summary'][strat] = {
                'avg_return': round(avg_ret, 2),
                'avg_win_rate': round(avg_wr, 1),
                'avg_profit_factor': round(avg_pf, 2),
                'profitable_count': profitable,
                'total_tests': len(strat_results),
            }
    
    return results


if __name__ == '__main__':
    SYMBOLS = ['EURUSD=X', 'USDJPY=X', 'GBPUSD=X', 'EURJPY=X']
    
    print("="*60)
    print("FVG OPTIMIZED RESEARCH")
    print("="*60)
    
    results = run_research(SYMBOLS, days=59)
    
    # Save
    with open('fvg_optimized_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for strat, stats in results['summary'].items():
        print(f"{strat:15} Avg Ret: {stats['avg_return']:+6.2f}% "
              f"WR: {stats['avg_win_rate']:5.1f}% PF: {stats['avg_profit_factor']:.2f} "
              f"({stats['profitable_count']}/{stats['total_tests']} profitable)")
