#!/usr/bin/env python3
"""
Hyperliquid Client
===================
Exchange integration for testnet/mainnet paper trading.
Wraps the hyperliquid-python-sdk for our trading agent.

Features:
- Market data (OHLCV candles, order book)
- Position management (open/close/modify)
- Order placement (market, limit, stop-loss, take-profit)
- Account state queries
"""

import time
import json
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants
from eth_account import Account


@dataclass
class Position:
    coin: str
    size: float  # positive = long, negative = short
    entry_price: float
    leverage: float
    liquidation_price: float
    unrealized_pnl: float
    return_on_equity: float
    margin_used: float


@dataclass
class OrderResult:
    success: bool
    order_id: Optional[str] = None
    error: Optional[str] = None
    raw: Optional[Dict] = None


class HyperliquidClient:
    """Unified client for Hyperliquid testnet/mainnet."""

    SUPPORTED_COINS = [
        "BTC", "ETH", "SOL", "XRP", "DOGE", "ADA",
        "LINK", "AVAX", "DOT", "MATIC", "UNI", "AAVE",
    ]

    def __init__(self, private_key: str, testnet: bool = True):
        self.testnet = testnet
        self.base_url = (
            constants.TESTNET_API_URL if testnet else constants.MAINNET_API_URL
        )
        self.info = Info(self.base_url, skip_ws=True)

        # Set up wallet and exchange for trading
        self.account = Account.from_key(private_key)
        self.address = self.account.address
        self.exchange = Exchange(self.account, self.base_url)

    # ── Market Data ──────────────────────────────────────────────

    def get_candles(self, coin: str, interval: str = "1h",
                    lookback_hours: int = 100) -> pd.DataFrame:
        """Fetch OHLCV candles and return as pandas DataFrame."""
        now_ms = int(time.time() * 1000)
        interval_ms = self._interval_to_ms(interval)
        start_ms = now_ms - lookback_hours * interval_ms

        try:
            candles = self.info.candles_snapshot(coin, interval, start_ms, now_ms)
        except Exception as e:
            print(f"Error fetching candles for {coin}: {e}")
            return pd.DataFrame()

        if not candles:
            return pd.DataFrame()

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
        return df

    def get_orderbook(self, coin: str) -> Dict:
        """Get L2 order book snapshot."""
        try:
            book = self.info.l2_snapshot(coin)
            return {
                "bids": [(float(p), float(s)) for p, s in book["levels"][0][:10]],
                "asks": [(float(p), float(s)) for p, s in book["levels"][1][:10]],
            }
        except Exception as e:
            print(f"Error fetching order book for {coin}: {e}")
            return {"bids": [], "asks": []}

    def get_mid_price(self, coin: str) -> Optional[float]:
        """Get current mid price from order book."""
        book = self.get_orderbook(coin)
        if book["bids"] and book["asks"]:
            return (book["bids"][0][0] + book["asks"][0][0]) / 2
        return None

    def get_all_mids(self) -> Dict[str, float]:
        """Get mid prices for all supported coins."""
        try:
            mids = self.info.all_mids()
            return {k: float(v) for k, v in mids.items()}
        except Exception as e:
            print(f"Error fetching mids: {e}")
            return {}

    # ── Account & Positions ──────────────────────────────────────

    def get_account_state(self) -> Dict:
        """Get full account state including margin and positions."""
        try:
            state = self.info.user_state(self.address)
            return state
        except Exception as e:
            print(f"Error fetching account state: {e}")
            return {}

    def get_account_value(self) -> float:
        """Get total account value in USD."""
        state = self.get_account_state()
        if state and "marginSummary" in state:
            return float(state["marginSummary"]["accountValue"])
        return 0.0

    def get_positions(self) -> List[Position]:
        """Get all open positions."""
        state = self.get_account_state()
        if not state or "assetPositions" not in state:
            return []

        positions = []
        for ap in state["assetPositions"]:
            pos = ap["position"]
            size = float(pos["szi"])
            if size == 0:
                continue
            positions.append(Position(
                coin=pos["coin"],
                size=size,
                entry_price=float(pos["entryPx"]),
                leverage=float(pos["leverage"]["value"])
                if isinstance(pos["leverage"], dict) else float(pos["leverage"]),
                liquidation_price=float(pos.get("liquidationPx", 0) or 0),
                unrealized_pnl=float(pos["unrealizedPnl"]),
                return_on_equity=float(pos.get("returnOnEquity", 0)),
                margin_used=float(pos.get("marginUsed", 0)),
            ))
        return positions

    def get_position(self, coin: str) -> Optional[Position]:
        """Get position for a specific coin."""
        for p in self.get_positions():
            if p.coin == coin:
                return p
        return None

    # ── Order Placement ──────────────────────────────────────────

    def market_open(self, coin: str, is_buy: bool, size: float,
                    slippage_pct: float = 0.01) -> OrderResult:
        """Open a position with a market order."""
        mid = self.get_mid_price(coin)
        if mid is None:
            return OrderResult(success=False, error=f"Cannot get price for {coin}")

        # Set slippage price
        if is_buy:
            slippage_px = mid * (1 + slippage_pct)
        else:
            slippage_px = mid * (1 - slippage_pct)

        try:
            result = self.exchange.market_open(
                coin, is_buy, size, slippage_px
            )
            return self._parse_order_result(result)
        except Exception as e:
            return OrderResult(success=False, error=str(e))

    def market_close(self, coin: str, slippage_pct: float = 0.01) -> OrderResult:
        """Close an entire position with market order."""
        pos = self.get_position(coin)
        if pos is None:
            return OrderResult(success=False, error=f"No open position for {coin}")

        mid = self.get_mid_price(coin)
        if mid is None:
            return OrderResult(success=False, error=f"Cannot get price for {coin}")

        try:
            result = self.exchange.market_close(coin)
            return self._parse_order_result(result)
        except Exception as e:
            return OrderResult(success=False, error=str(e))

    def limit_order(self, coin: str, is_buy: bool, size: float,
                    price: float) -> OrderResult:
        """Place a limit order (GTC)."""
        try:
            result = self.exchange.order(
                coin, is_buy, size, price,
                order_type={"limit": {"tif": "Gtc"}},
                reduce_only=False,
            )
            return self._parse_order_result(result)
        except Exception as e:
            return OrderResult(success=False, error=str(e))

    def set_leverage(self, coin: str, leverage: int, is_cross: bool = False) -> bool:
        """Set leverage for a coin."""
        try:
            self.exchange.update_leverage(
                leverage, coin, is_cross=is_cross
            )
            return True
        except Exception as e:
            print(f"Error setting leverage for {coin}: {e}")
            return False

    def cancel_all_orders(self, coin: Optional[str] = None) -> bool:
        """Cancel all open orders, optionally for a specific coin."""
        try:
            open_orders = self.info.open_orders(self.address)
            if coin:
                open_orders = [o for o in open_orders if o["coin"] == coin]
            if not open_orders:
                return True
            cancels = [
                {"coin": o["coin"], "oid": o["oid"]}
                for o in open_orders
            ]
            self.exchange.cancel(cancels)
            return True
        except Exception as e:
            print(f"Error canceling orders: {e}")
            return False

    # ── Helpers ───────────────────────────────────────────────────

    def _parse_order_result(self, result: Dict) -> OrderResult:
        """Parse SDK order result into OrderResult."""
        status = result.get("status", "")
        if status == "ok":
            response = result.get("response", {})
            data = response.get("data", {})
            statuses = data.get("statuses", [{}])
            if statuses and "filled" in statuses[0]:
                return OrderResult(
                    success=True,
                    order_id=str(statuses[0]["filled"].get("oid", "")),
                    raw=result,
                )
            elif statuses and "resting" in statuses[0]:
                return OrderResult(
                    success=True,
                    order_id=str(statuses[0]["resting"].get("oid", "")),
                    raw=result,
                )
        return OrderResult(success=False, error=json.dumps(result), raw=result)

    @staticmethod
    def _interval_to_ms(interval: str) -> int:
        """Convert interval string to milliseconds."""
        multipliers = {
            "1m": 60_000, "3m": 180_000, "5m": 300_000,
            "15m": 900_000, "30m": 1_800_000,
            "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000,
            "1d": 86_400_000,
        }
        return multipliers.get(interval, 3_600_000)

    def format_account_context(self) -> str:
        """Format account state as text for LLM context."""
        value = self.get_account_value()
        positions = self.get_positions()

        lines = [
            f"=== Account State ({'TESTNET' if self.testnet else 'MAINNET'}) ===",
            f"Account Value: ${value:,.2f}",
            f"Open Positions: {len(positions)}",
        ]

        if positions:
            lines.append("")
            for p in positions:
                direction = "LONG" if p.size > 0 else "SHORT"
                lines.append(
                    f"  {p.coin}: {direction} {abs(p.size):.4f} @ ${p.entry_price:,.2f} "
                    f"| Leverage: {p.leverage:.0f}x "
                    f"| PnL: ${p.unrealized_pnl:+,.2f} ({p.return_on_equity:+.1%}) "
                    f"| Liq: ${p.liquidation_price:,.2f}"
                )
        else:
            lines.append("  No open positions")

        return "\n".join(lines)
