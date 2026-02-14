#!/usr/bin/env python3
"""
Quick test script - verify all data feeds work.
No private key or API key needed for market data.
"""

import sys
import time
sys.path.insert(0, ".")

from onchain_feed import OnChainFeed
from technical_indicators import format_technical_context
from agent_memory import AgentMemory

# Use mainnet for testing market data (testnet may have less liquidity)
USE_MAINNET = True


def test_onchain():
    """Test on-chain data feed."""
    print("=" * 60)
    print("Testing On-Chain Feed...")
    print("=" * 60)

    feed = OnChainFeed(testnet=not USE_MAINNET)
    snapshots = feed.get_all_funding_and_oi()

    print(f"\nFound {len(snapshots)} assets")
    # Show top 6 by OI
    top = sorted(snapshots, key=lambda s: s.open_interest, reverse=True)[:6]
    for s in top:
        cls = feed.classify_funding(s.funding_rate)
        print(f"  {s.coin:>6}: Price=${s.mark_price:>10,.2f} | "
              f"Funding={s.funding_rate:>10.6f} ({cls}) | "
              f"OI=${s.open_interest:>12,.0f}")

    # Full context for BTC
    print("\n" + feed.format_onchain_context(["BTC", "ETH"]))


def test_technical():
    """Test technical indicators using Hyperliquid candle data."""
    print("\n" + "=" * 60)
    print("Testing Technical Indicators...")
    print("=" * 60)

    # Fetch candles from Hyperliquid directly (no auth needed for info)
    from hyperliquid.info import Info
    from hyperliquid.utils import constants
    import pandas as pd

    base_url = constants.MAINNET_API_URL if USE_MAINNET else constants.TESTNET_API_URL
    info = Info(base_url, skip_ws=True)

    now_ms = int(time.time() * 1000)
    start_ms = now_ms - 100 * 3600 * 1000  # 100 hours

    candles = info.candles_snapshot("BTC", "1h", start_ms, now_ms)
    if not candles:
        print("  No candle data available")
        return

    rows = []
    for c in candles:
        rows.append({
            "timestamp": pd.to_datetime(int(c["t"]), unit="ms"),
            "open": float(c["o"]),
            "high": float(c["h"]),
            "low": float(c["l"]),
            "close": float(c["c"]),
            "volume": float(c["v"]),
        })
    df = pd.DataFrame(rows).set_index("timestamp")
    print(f"\nFetched {len(df)} candles for BTC")
    print(f"Range: {df.index[0]} to {df.index[-1]}")

    ctx = format_technical_context(df, "BTC")
    print("\n" + ctx)


def test_memory():
    """Test agent memory."""
    print("\n" + "=" * 60)
    print("Testing Agent Memory...")
    print("=" * 60)

    mem = AgentMemory(db_path="/tmp/test_trade_journal.db")

    # Record a test trade
    trade_id = mem.record_trade_open(
        coin="BTC", direction="LONG",
        entry_price=96000, size=0.01,
        leverage=10, stop_loss=94000, take_profit=100000,
        entry_reason="Test trade: RSI oversold + extreme short funding",
        market_conditions="RSI=28, Funding=-0.001, MACD bullish cross",
    )
    print(f"\n  Opened test trade #{trade_id}")

    # Close it
    mem.record_trade_close(
        trade_id=trade_id, exit_price=99000,
        pnl_usd=300, pnl_pct=0.03125,
        outcome="WIN", exit_reason="Take profit hit",
        lessons="RSI oversold + extreme funding is reliable reversal signal",
    )
    print(f"  Closed test trade #{trade_id}")

    # Print context
    ctx = mem.format_memory_context(["BTC", "ETH"])
    print("\n" + ctx)

    # Stats
    stats = mem.get_performance_stats()
    print(f"\nStats: {stats}")

    # Cleanup
    import os
    os.remove("/tmp/test_trade_journal.db")
    print("  Cleaned up test DB")


if __name__ == "__main__":
    test_onchain()
    test_technical()
    test_memory()
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
