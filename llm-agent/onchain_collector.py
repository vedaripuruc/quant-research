#!/usr/bin/env python3
"""
On-Chain Data Collector — saves snapshots every 30 min for replay backtesting.
Stores funding rates, OI, liquidation zones, and price/indicators.
Builds the historical dataset we can't get retroactively.

Run via systemd timer. Each snapshot ~5KB JSON.
30 days @ 30min = ~1,440 snapshots = ~7MB. Cheap insurance.
"""

import json, sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from onchain_feed import OnChainFeed
from hyperliquid.info import Info
from hyperliquid.utils import constants
import pandas as pd

DATA_DIR = Path(__file__).parent / "data" / "snapshots"
COINS = ["BTC", "ETH", "SOL", "DOGE", "AVAX", "LINK"]


def collect_snapshot():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)

    feed = OnChainFeed(testnet=False)
    info = Info(constants.MAINNET_API_URL, skip_ws=True)  # read-only, no key needed

    # Get all funding/OI in one call
    all_funding = feed.get_all_funding_and_oi()
    funding_map = {s.coin: s for s in all_funding}

    snapshot = {
        "timestamp": now.isoformat(),
        "timestamp_unix": int(now.timestamp()),
        "coins": {},
    }

    for coin in COINS:
        entry = {}
        try:
            # On-chain
            fs = funding_map.get(coin)
            if fs:
                mark = fs.mark_price
                entry["funding_rate"] = fs.funding_rate
                entry["premium"] = fs.premium
                entry["open_interest"] = fs.open_interest
                entry["mark_price"] = mark

                # Liquidation zones
                liq = feed.estimate_liquidation_zones(coin, mark, fs.open_interest)
                entry["liquidation_zones"] = liq

            # Funding history (24h)
            fh = feed.get_funding_history(coin, lookback_hours=24)
            if fh:
                rates = [x["funding_rate"] for x in fh]
                entry["funding_24h_avg"] = sum(rates) / len(rates) if rates else 0
                entry["funding_24h_trend"] = "rising" if len(rates) > 1 and rates[-1] > rates[0] else "falling"

            # Price candles (last 50 1h bars)
            import time as _time
            now_ms = int(_time.time() * 1000)
            start_ms = now_ms - 50 * 3600 * 1000
            try:
                raw = info.candles_snapshot(coin, "1h", start_ms, now_ms)
                if raw and len(raw) > 0:
                    last = raw[-1]
                    entry["price"] = float(last.get("c", last.get("close", 0)))
                    entry["candle_count"] = len(raw)
                    # Save last 5 candles for context
                    entry["recent_candles"] = [
                        {"t": c.get("t"), "o": c.get("o"), "h": c.get("h"), "l": c.get("l"), "c": c.get("c")}
                        for c in raw[-5:]
                    ]
                else:
                    entry["price"] = fs.mark_price if fs else None
            except Exception as ce:
                entry["price"] = fs.mark_price if fs else None
                entry["candle_error"] = str(ce)

        except Exception as e:
            entry["error"] = str(e)

        snapshot["coins"][coin] = entry

    # Save as daily JSONL
    date_str = now.strftime("%Y-%m-%d")
    out_file = DATA_DIR / f"snapshots_{date_str}.jsonl"
    with out_file.open("a") as f:
        f.write(json.dumps(snapshot, default=str) + "\n")

    n_ok = sum(1 for c in snapshot["coins"].values() if "error" not in c)
    print(f"[{now.strftime('%H:%M:%S')}] Snapshot: {n_ok}/{len(COINS)} coins → {out_file.name}")
    return snapshot


if __name__ == "__main__":
    collect_snapshot()
