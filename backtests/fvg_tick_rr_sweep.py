#!/usr/bin/env python3
"""FVG tick backtest with R:R sweep — tests multiple TP multiples on 60min bars."""

from fvg_tick_backtest import (
    DATA_DIR, DATA_START, DATA_END, RESULTS_DIR, MIN_GAP, PIP_MULTIPLIER,
    WF_PERIOD_START, WF_PERIOD_END, TRAIN_MONTHS,
    Bar, PendingOrder, ActiveTrade, ClosedTrade,
    dt_to_ms, ms_to_iso, add_months, iter_hours,
    calculate_metrics, build_walk_forward_windows, detect_fvg_and_create_order,
)
import gzip
import json
from datetime import datetime, timezone
from pathlib import Path

# SL stays at 1.5x gap. Sweep TP multiplier:
TP_MULTIPLES = [1.5, 2.0, 2.25, 2.5, 3.0, 4.0]
SL_MULTIPLE = 1.5
BAR_MINUTES = 60


def process_ticks_multi_rr():
    """Stream ticks once, track separate trade sets per R:R ratio."""
    
    # Per-RR state
    rr_states = {}
    for tp_mult in TP_MULTIPLES:
        rr_states[tp_mult] = {
            "pending": [],
            "active": [],
            "closed": [],
            "fills": 0,
        }
    
    bars = []
    next_order_id = 1
    files_processed = 0
    
    bar_interval_ms = BAR_MINUTES * 60 * 1000
    
    print(f"Streaming ticks, tracking {len(TP_MULTIPLES)} R:R ratios simultaneously...")
    
    for hour_start in iter_hours(DATA_START, DATA_END):
        file_name = f"EURUSD_BID_{hour_start:%Y-%m-%d_%H}.log.gz"
        file_path = DATA_DIR / file_name
        if not file_path.exists():
            continue
        
        hour_start_ms = dt_to_ms(hour_start)
        bar_open = bar_high = bar_low = bar_close = 0.0
        bar_vol = 0.0
        bar_ticks = 0
        
        with gzip.open(file_path, "rt", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) != 3:
                    continue
                try:
                    ts_ms = int(parts[0])
                    price = float(parts[1])
                    volume = float(parts[2])
                except ValueError:
                    continue
                
                # Update bar
                if bar_ticks == 0:
                    bar_open = bar_high = bar_low = bar_close = price
                else:
                    if price > bar_high: bar_high = price
                    if price < bar_low: bar_low = price
                    bar_close = price
                bar_vol += volume
                bar_ticks += 1
                
                # Process tick against ALL R:R variants
                for tp_mult in TP_MULTIPLES:
                    state = rr_states[tp_mult]
                    
                    # Check pending orders
                    still_pending = []
                    for order in state["pending"]:
                        filled = False
                        cancelled = False
                        
                        if order.fvg_type == "bullish":
                            if price <= order.midpoint:
                                filled = True
                            elif price < order.zone_start:
                                cancelled = True
                        else:
                            if price >= order.midpoint:
                                filled = True
                            elif price > order.zone_start:
                                cancelled = True
                        
                        if filled:
                            state["fills"] += 1
                            sl = order.midpoint - SL_MULTIPLE * order.gap_size if order.fvg_type == "bullish" else order.midpoint + SL_MULTIPLE * order.gap_size
                            tp = order.midpoint + tp_mult * order.gap_size if order.fvg_type == "bullish" else order.midpoint - tp_mult * order.gap_size
                            state["active"].append(ActiveTrade(
                                order_id=order.order_id,
                                fvg_time_ms=order.fvg_time_ms,
                                fvg_type=order.fvg_type,
                                midpoint=order.midpoint,
                                gap_size=order.gap_size,
                                entry_time_ms=ts_ms,
                                entry_price=order.midpoint,
                                sl=sl, tp=tp,
                            ))
                        elif not cancelled:
                            still_pending.append(order)
                    state["pending"] = still_pending
                    
                    # Check active trades
                    still_active = []
                    for trade in state["active"]:
                        exit_price = None
                        if trade.fvg_type == "bullish":
                            if price <= trade.sl: exit_price = trade.sl
                            elif price >= trade.tp: exit_price = trade.tp
                        else:
                            if price >= trade.sl: exit_price = trade.sl
                            elif price <= trade.tp: exit_price = trade.tp
                        
                        if exit_price is None:
                            still_active.append(trade)
                            continue
                        
                        pips = ((exit_price - trade.entry_price) if trade.fvg_type == "bullish" else (trade.entry_price - exit_price)) * PIP_MULTIPLIER
                        state["closed"].append(ClosedTrade(
                            fvg_time_ms=trade.fvg_time_ms, fvg_type=trade.fvg_type,
                            midpoint=trade.midpoint, gap_size=trade.gap_size,
                            entry_time_ms=trade.entry_time_ms, entry_price=trade.entry_price,
                            exit_time_ms=ts_ms, exit_price=exit_price,
                            sl=trade.sl, tp=trade.tp,
                            result="Win" if pips > 0 else "Loss",
                            pips=pips,
                            duration_seconds=max(0, (ts_ms - trade.entry_time_ms) / 1000),
                        ))
                    state["active"] = still_active
        
        files_processed += 1
        if files_processed % 500 == 0:
            print(f"  {files_processed} files... ({hour_start:%Y-%m-%d})")
        
        if bar_ticks == 0:
            continue
        
        bars.append(Bar(
            timestamp=hour_start, open=bar_open, high=bar_high,
            low=bar_low, close=bar_close, volume=bar_vol, tick_count=bar_ticks,
        ))
        
        # Detect FVG and create pending orders for ALL variants
        if len(bars) >= 3:
            left = bars[-3]
            right = bars[-1]
            
            order = None
            if right.low > left.high and (right.low - left.high) >= MIN_GAP:
                order = PendingOrder(
                    order_id=next_order_id, fvg_time_ms=dt_to_ms(right.timestamp),
                    fvg_type="bullish", midpoint=(left.high + right.low) / 2,
                    gap_size=right.low - left.high,
                    zone_start=left.high, zone_end=right.low,
                )
            elif left.low > right.high and (left.low - right.high) >= MIN_GAP:
                order = PendingOrder(
                    order_id=next_order_id, fvg_time_ms=dt_to_ms(right.timestamp),
                    fvg_type="bearish", midpoint=(right.high + left.low) / 2,
                    gap_size=left.low - right.high,
                    zone_start=left.low, zone_end=right.high,
                )
            
            if order:
                next_order_id += 1
                for tp_mult in TP_MULTIPLES:
                    # Each variant gets its own copy of the order
                    rr_states[tp_mult]["pending"].append(PendingOrder(
                        order_id=order.order_id, fvg_time_ms=order.fvg_time_ms,
                        fvg_type=order.fvg_type, midpoint=order.midpoint,
                        gap_size=order.gap_size, zone_start=order.zone_start,
                        zone_end=order.zone_end,
                    ))
    
    return rr_states, next_order_id - 1


def main():
    aggregate_start_ms = dt_to_ms(add_months(WF_PERIOD_START, TRAIN_MONTHS))
    aggregate_end_ms = dt_to_ms(WF_PERIOD_END)
    
    rr_states, total_fvgs = process_ticks_multi_rr()
    
    print(f"\nTotal FVG zones: {total_fvgs}")
    print(f"\n{'='*90}")
    print(f"{'TP mult':>8} {'RR':>5} {'Trades':>7} {'WR%':>7} {'PF':>7} {'Pips':>9} {'AvgW':>7} {'AvgL':>7} {'MaxDD':>8} {'Sharpe':>8}")
    print(f"{'='*90}")
    
    results = []
    for tp_mult in TP_MULTIPLES:
        rr = tp_mult / SL_MULTIPLE
        state = rr_states[tp_mult]
        oos_trades = [t for t in state["closed"] if aggregate_start_ms <= t.entry_time_ms < aggregate_end_ms]
        m = calculate_metrics(oos_trades)
        
        pf_str = f"{m['profit_factor']:.3f}" if m['profit_factor'] is not None else "inf"
        print(f"{tp_mult:>7.1f}x {rr:>4.1f}:1 {m['total_trades']:>7} {m['win_rate_pct']:>6.1f}% {pf_str:>7} {m['total_return_pips']:>+9.0f} {m['avg_win_pips']:>+6.1f} {m['avg_loss_pips']:>+6.1f} {m['max_drawdown_pips']:>8.0f} {m['sharpe_annualized']:>+7.2f}")
        
        results.append({
            "tp_multiple": tp_mult,
            "sl_multiple": SL_MULTIPLE,
            "risk_reward": rr,
            "metrics": m,
        })
    
    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "fvg_tick_rr_sweep.json", "w") as f:
        json.dump({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "bar_minutes": BAR_MINUTES,
            "sl_multiple": SL_MULTIPLE,
            "results": results,
        }, f, indent=2)
    
    print(f"\nSaved: results/fvg_tick_rr_sweep.json")


if __name__ == "__main__":
    main()
