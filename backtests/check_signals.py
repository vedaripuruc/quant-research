#!/usr/bin/env python3
"""
Check Signals - Heartbeat wrapper with position sizing and tracking
===================================================================
1. First runs signal_tracker to update outcomes of existing signals
2. Checks MOMENTUM signals (LINK/ADA/XRP hourly, Binance)
3. Checks DAILY signals (Gold Jump + Hurst Regime, yfinance)
4. Outputs format compatible with heartbeat alerts
"""

import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(__file__))

from momentum_signals import check_all_signals, format_signal_alert, DEFAULT_ACCOUNT_SIZE, DEFAULT_RISK_PCT
from daily_signals import check_daily_signals, format_daily_signal_alert
from signal_tracker import update_signals, calculate_stats
import json
from datetime import datetime


def main():
    """Main entry point for heartbeat monitoring."""
    print("=" * 55)
    print("SIGNAL CHECK + TRACKING")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)

    # Step 1: Update existing signal outcomes
    print("\n📊 TRACKING EXISTING SIGNALS...")
    print("-" * 40)
    history, stats = update_signals()

    # ─── Step 2: Momentum signals (hourly, Binance) ────────────────────
    print("\n🔍 CHECKING MOMENTUM SIGNALS (LINK/ADA/XRP)...")
    print("-" * 40)
    momentum_results = check_all_signals()

    # ─── Step 3: Daily signals (Gold Jump + Hurst Regime) ──────────────
    print("\n🔍 CHECKING DAILY SIGNALS (Gold Jump + Hurst)...")
    print("-" * 40)
    daily_results = check_daily_signals()

    # ─── Merge results ─────────────────────────────────────────────────
    all_signals = []
    all_no_signal = []

    if momentum_results["signals"]:
        all_signals.extend(momentum_results["signals"])
    all_no_signal.extend(momentum_results.get("no_signal", []))

    if daily_results["signals"]:
        all_signals.extend(daily_results["signals"])
    all_no_signal.extend(daily_results.get("no_signal", []))

    combined_status = "signals" if all_signals else "ok"

    # ─── Output ────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("SUMMARY")
    print("=" * 55)

    # Momentum section
    if momentum_results["signals"]:
        print(f"\n⚡ MOMENTUM SIGNALS: {len(momentum_results['signals'])}")
        for signal in momentum_results["signals"]:
            print(format_signal_alert(signal))
            print("---")
    else:
        checked = momentum_results.get("no_signal", [])
        print(f"\n⚡ Momentum: no signals (checked: {', '.join(checked)})")

    # Daily section
    if daily_results["signals"]:
        print(f"\n🌅 DAILY SIGNALS: {len(daily_results['signals'])}")
        for signal in daily_results["signals"]:
            print(format_daily_signal_alert(signal))
            print("---")
    else:
        checked = daily_results.get("no_signal", [])
        print(f"\n🌅 Daily: no signals (checked: {', '.join(checked)})")

    # Overall status
    print(f"\nstatus: {combined_status}")
    if all_signals:
        print(f"total_signals: {len(all_signals)}")
    if all_no_signal:
        print(f"no_signal: {', '.join(all_no_signal)}")

    # Performance summary
    if stats.get("total_signals", 0) > 0:
        print(f"\n📈 PERFORMANCE (All Time):")
        print(f"  Signals: {stats['total_signals']} ({stats.get('wins', 0)}W/{stats.get('losses', 0)}L)")
        print(f"  Win Rate: {stats.get('win_rate', 0)}%")
        print(f"  Total P&L: ${stats.get('total_pnl_usd', 0):+,.2f}")
        print(f"  Avg R: {stats.get('avg_r', 0)}")

    # Open positions reminder
    open_count = stats.get('open', 0)
    if open_count > 0:
        print(f"\n⚠️  {open_count} open position(s) - monitor TP/SL levels!")

    return {
        "status": combined_status,
        "momentum": momentum_results,
        "daily": daily_results,
        "signals": all_signals,
        "no_signal": all_no_signal,
    }


if __name__ == "__main__":
    main()
