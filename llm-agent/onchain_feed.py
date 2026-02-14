#!/usr/bin/env python3
"""
On-Chain Data Feed
===================
Pulls funding rates, open interest, and liquidation-level
estimates from Hyperliquid API. This is our unique edge -
data that competition agents (Grok, DeepSeek) don't use.

All data is free via Hyperliquid's public API.
"""

import time
import requests
import json
from typing import Dict, List, Optional
from dataclasses import dataclass
from hyperliquid.info import Info
from hyperliquid.utils import constants


@dataclass
class FundingSnapshot:
    coin: str
    funding_rate: float  # current 1h funding rate
    premium: float
    open_interest: float
    mark_price: float


class OnChainFeed:
    """On-chain data feed from Hyperliquid."""

    # Extreme funding thresholds (based on competition research)
    EXTREME_FUNDING_LONG = 0.0005   # 0.05% per 8h = crowded long
    EXTREME_FUNDING_SHORT = -0.0005  # -0.05% per 8h = crowded short

    def __init__(self, testnet: bool = True):
        self.testnet = testnet
        base_url = (
            constants.TESTNET_API_URL if testnet else constants.MAINNET_API_URL
        )
        self.info = Info(base_url, skip_ws=True)
        self.api_url = base_url + "/info"

    def get_all_funding_and_oi(self) -> List[FundingSnapshot]:
        """Get current funding rates and open interest for all perps."""
        try:
            # meta_and_asset_ctxs returns metadata + asset contexts
            data = self._post_info({"type": "metaAndAssetCtxs"})
            if not data or len(data) < 2:
                return []

            meta = data[0]
            ctxs = data[1]
            universe = meta.get("universe", [])

            snapshots = []
            for i, ctx in enumerate(ctxs):
                if i >= len(universe):
                    break
                coin = universe[i]["name"]
                try:
                    snapshots.append(FundingSnapshot(
                        coin=coin,
                        funding_rate=float(ctx.get("funding") or 0),
                        premium=float(ctx.get("premium") or 0),
                        open_interest=float(ctx.get("openInterest") or 0),
                        mark_price=float(ctx.get("markPx") or 0),
                    ))
                except (ValueError, TypeError):
                    continue
            return snapshots
        except Exception as e:
            print(f"Error fetching funding/OI: {e}")
            return []

    def get_funding_history(self, coin: str, lookback_hours: int = 48) -> List[Dict]:
        """Get historical funding rates for a coin."""
        start_ms = int((time.time() - lookback_hours * 3600) * 1000)
        try:
            data = self._post_info({
                "type": "fundingHistory",
                "coin": coin,
                "startTime": start_ms,
            })
            return [
                {
                    "time": entry.get("time"),
                    "coin": entry.get("coin", coin),
                    "funding_rate": float(entry.get("fundingRate") or 0),
                    "premium": float(entry.get("premium") or 0),
                }
                for entry in (data or [])
            ]
        except Exception as e:
            print(f"Error fetching funding history for {coin}: {e}")
            return []

    def estimate_liquidation_zones(self, coin: str,
                                   mark_price: float,
                                   open_interest: float) -> Dict:
        """
        Estimate where leveraged positions will get liquidated.

        These act as price magnets (similar to FVG concept from curupira
        but based on actual position data).

        Assumptions:
        - 10x longs liquidated ~9-10% below entry
        - 20x longs liquidated ~4-5% below entry
        - 10x shorts liquidated ~9-10% above entry
        - 20x shorts liquidated ~4-5% above entry
        """
        zones = {
            "10x_long_liq": round(mark_price * 0.91, 2),
            "20x_long_liq": round(mark_price * 0.955, 2),
            "10x_short_liq": round(mark_price * 1.09, 2),
            "20x_short_liq": round(mark_price * 1.045, 2),
        }
        return zones

    def classify_funding(self, rate: float) -> str:
        """Classify funding rate as a trading signal."""
        if rate > self.EXTREME_FUNDING_LONG:
            return "EXTREME_LONG (crowded long, reversal risk)"
        elif rate < self.EXTREME_FUNDING_SHORT:
            return "EXTREME_SHORT (crowded short, squeeze risk)"
        elif rate > 0.0001:
            return "moderately long"
        elif rate < -0.0001:
            return "moderately short"
        else:
            return "neutral"

    def format_onchain_context(self, coins: List[str]) -> str:
        """Format on-chain data as text for LLM context injection."""
        all_snapshots = self.get_all_funding_and_oi()
        snapshot_map = {s.coin: s for s in all_snapshots}

        lines = ["=== On-Chain Data (Hyperliquid) ===", ""]

        for coin in coins:
            snap = snapshot_map.get(coin)
            if not snap:
                lines.append(f"{coin}: no data available")
                continue

            funding_class = self.classify_funding(snap.funding_rate)
            liq_zones = self.estimate_liquidation_zones(
                coin, snap.mark_price, snap.open_interest
            )

            # Get recent funding history for trend
            history = self.get_funding_history(coin, lookback_hours=24)
            avg_funding = 0
            if history:
                avg_funding = sum(h["funding_rate"] for h in history) / len(history)
                funding_trend = "rising" if history[-1]["funding_rate"] > avg_funding else "falling"
            else:
                funding_trend = "unknown"

            lines.extend([
                f"--- {coin} ---",
                f"Mark Price: ${snap.mark_price:,.2f}",
                f"Funding Rate: {snap.funding_rate:.6f} ({funding_class})",
                f"Funding Trend (24h): {funding_trend} (avg: {avg_funding:.6f})",
                f"Open Interest: {snap.open_interest:,.2f} contracts (${snap.open_interest * snap.mark_price:,.0f} notional)",
                f"Premium: {snap.premium:.6f}",
                f"Liquidation Zones:",
                f"  20x longs liquidated near: ${liq_zones['20x_long_liq']:,.2f}",
                f"  10x longs liquidated near: ${liq_zones['10x_long_liq']:,.2f}",
                f"  10x shorts liquidated near: ${liq_zones['10x_short_liq']:,.2f}",
                f"  20x shorts liquidated near: ${liq_zones['20x_short_liq']:,.2f}",
                "",
            ])

        return "\n".join(lines)

    def _post_info(self, payload: Dict) -> Optional[Dict]:
        """Direct POST to Hyperliquid info endpoint."""
        try:
            resp = requests.post(self.api_url, json=payload, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"Info API error: {e}")
            return None
