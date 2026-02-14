#!/usr/bin/env python3
"""
Liquidation Watcher — WebSocket cascade detector + panic scorer
================================================================
Connects to Hyperliquid WebSocket, monitors trades/book/OI for
liquidation cascade patterns. When panic score > threshold,
triggers Curupira wake via OpenClaw cron.

Run as systemd service (persistent).
"""

import json, time, sys, os, signal, threading, logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List
import numpy as np

try:
    import websocket
except ImportError:
    print("pip install websocket-client")
    sys.exit(1)

try:
    import requests
except ImportError:
    print("pip install requests")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent))

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

WS_URL = "wss://api.hyperliquid.xyz/ws"
COINS = ["BTC", "ETH", "SOL"]  # start focused

# Panic thresholds
PANIC_WAKE_THRESHOLD = 60   # wake Curupira
PANIC_URGENT_THRESHOLD = 80 # high urgency

# Rolling windows
TRADE_WINDOW_SEC = 300      # 5 min trade accumulation
VOLUME_BASELINE_SEC = 3600  # 1h baseline for volume Z-score
PRICE_VELOCITY_SEC = 300    # 5 min price velocity
OI_CHECK_INTERVAL_SEC = 60  # check OI every 60s

# Scoring weights
W_VOLUME = 0.25
W_PRICE_VEL = 0.25
W_OI_DROP = 0.20
W_FUNDING = 0.15
W_BOOK_IMBAL = 0.15

# Paths
DATA_DIR = Path(__file__).parent / "data" / "events"
ALERT_FILE = Path("/tmp/cascade-alert.json")
LOG_FILE = Path(__file__).parent / "data" / "watcher.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_FILE), mode='a'),
    ]
)
log = logging.getLogger("watcher")


# ═══════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════

@dataclass
class TradeEvent:
    time: float         # unix timestamp
    coin: str
    side: str           # "B" or "A" (buy/sell)
    price: float
    size: float
    notional: float     # price * size


@dataclass
class CoinState:
    """Rolling state for one coin."""
    coin: str
    trades: deque = field(default_factory=lambda: deque(maxlen=10000))
    prices: deque = field(default_factory=lambda: deque(maxlen=5000))
    bid_depth: float = 0.0
    ask_depth: float = 0.0
    last_price: float = 0.0
    last_oi: float = 0.0
    prev_oi: float = 0.0
    last_oi_check: float = 0.0
    funding_rate: float = 0.0
    panic_score: float = 0.0
    last_cascade_time: float = 0.0

    def recent_trades(self, window_sec: float) -> List[TradeEvent]:
        cutoff = time.time() - window_sec
        return [t for t in self.trades if t.time >= cutoff]

    def volume_in_window(self, window_sec: float) -> float:
        return sum(t.notional for t in self.recent_trades(window_sec))

    def volume_by_side(self, window_sec: float) -> tuple:
        trades = self.recent_trades(window_sec)
        buy = sum(t.notional for t in trades if t.side == "B")
        sell = sum(t.notional for t in trades if t.side == "A")
        return buy, sell

    def price_velocity(self, window_sec: float) -> float:
        """% price change per minute over window."""
        cutoff = time.time() - window_sec
        recent = [(t.time, t.price) for t in self.trades if t.time >= cutoff]
        if len(recent) < 2:
            return 0.0
        first_price = recent[0][1]
        last_price = recent[-1][1]
        elapsed_min = (recent[-1][0] - recent[0][0]) / 60
        if elapsed_min < 0.1 or first_price == 0:
            return 0.0
        return abs(last_price - first_price) / first_price * 100 / elapsed_min


@dataclass
class CascadeEvent:
    """A detected cascade for logging and analysis."""
    timestamp: str
    coin: str
    panic_score: float
    direction: str       # "LONG_LIQUIDATION" or "SHORT_SQUEEZE"
    price_at_detection: float
    price_velocity: float  # %/min
    volume_spike: float    # multiplier vs baseline
    oi_change_pct: float
    funding_rate: float
    book_imbalance: float
    context: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# PANIC SCORER
# ═══════════════════════════════════════════════════════════════

class PanicScorer:
    """Scores cascade intensity 0-100."""

    @staticmethod
    def score_volume(state: CoinState) -> float:
        """Volume spike vs baseline. Returns 0-100."""
        recent_vol = state.volume_in_window(TRADE_WINDOW_SEC)
        baseline_vol = state.volume_in_window(VOLUME_BASELINE_SEC)
        if baseline_vol == 0:
            return 0
        # Normalize: baseline per 5 min
        baseline_5m = baseline_vol * (TRADE_WINDOW_SEC / VOLUME_BASELINE_SEC)
        if baseline_5m == 0:
            return 0
        ratio = recent_vol / baseline_5m
        # ratio of 1 = normal, 3+ = spike
        return min(100, max(0, (ratio - 1) * 33))

    @staticmethod
    def score_price_velocity(state: CoinState) -> float:
        """Price velocity. Returns 0-100."""
        vel = state.price_velocity(PRICE_VELOCITY_SEC)
        # BTC: 0.1%/min is notable, 0.5%/min is extreme
        # Normalize: 0.5%/min = 100
        return min(100, vel * 200)

    @staticmethod
    def score_oi_drop(state: CoinState) -> float:
        """OI drop rate. Returns 0-100."""
        if state.prev_oi == 0 or state.last_oi == 0:
            return 0
        change_pct = (state.last_oi - state.prev_oi) / state.prev_oi * 100
        # Only care about drops (negative change)
        if change_pct >= 0:
            return 0
        # -1% is notable, -5% is extreme
        return min(100, abs(change_pct) * 20)

    @staticmethod
    def score_funding(state: CoinState) -> float:
        """Funding rate extremity. Returns 0-100."""
        # 0.01% = normal, 0.05% = notable, 0.1%+ = extreme
        rate = abs(state.funding_rate)
        return min(100, rate * 100000 * 2)  # 0.0005 = 100

    @staticmethod
    def score_book_imbalance(state: CoinState) -> float:
        """Order book imbalance. Returns 0-100."""
        if state.bid_depth == 0 or state.ask_depth == 0:
            return 0
        ratio = max(state.bid_depth, state.ask_depth) / min(state.bid_depth, state.ask_depth)
        # ratio of 1 = balanced, 3+ = heavily imbalanced
        return min(100, max(0, (ratio - 1) * 50))

    @classmethod
    def compute(cls, state: CoinState) -> float:
        vol = cls.score_volume(state)
        vel = cls.score_price_velocity(state)
        oi = cls.score_oi_drop(state)
        fund = cls.score_funding(state)
        book = cls.score_book_imbalance(state)

        score = (
            vol * W_VOLUME +
            vel * W_PRICE_VEL +
            oi * W_OI_DROP +
            fund * W_FUNDING +
            book * W_BOOK_IMBAL
        )

        return round(score, 1)

    @classmethod
    def components(cls, state: CoinState) -> dict:
        return {
            "volume_spike": round(cls.score_volume(state), 1),
            "price_velocity": round(cls.score_price_velocity(state), 1),
            "oi_drop": round(cls.score_oi_drop(state), 1),
            "funding_extreme": round(cls.score_funding(state), 1),
            "book_imbalance": round(cls.score_book_imbalance(state), 1),
        }


# ═══════════════════════════════════════════════════════════════
# CASCADE DETECTOR
# ═══════════════════════════════════════════════════════════════

class CascadeDetector:
    """Detects liquidation cascades and triggers alerts."""

    def __init__(self):
        self.states: Dict[str, CoinState] = {c: CoinState(coin=c) for c in COINS}
        self.scorer = PanicScorer()
        self.last_alert_time = 0
        self.alert_cooldown = 300  # 5 min between alerts

    def on_trade(self, coin: str, trades: list):
        """Process incoming trade events."""
        state = self.states.get(coin)
        if not state:
            return

        for t in trades:
            try:
                price = float(t.get("px", 0))
                size = float(t.get("sz", 0))
                side = t.get("side", "B")
                ts = t.get("time", time.time() * 1000) / 1000

                event = TradeEvent(
                    time=ts, coin=coin, side=side,
                    price=price, size=size, notional=price * size,
                )
                state.trades.append(event)
                state.last_price = price
                state.prices.append((ts, price))
            except (ValueError, TypeError):
                continue

        self._check_panic(coin)

    def on_book(self, coin: str, book_data: dict):
        """Process order book update."""
        state = self.states.get(coin)
        if not state:
            return

        try:
            levels = book_data.get("levels", [[], []])
            bids = levels[0] if len(levels) > 0 else []
            asks = levels[1] if len(levels) > 1 else []

            # Sum top 10 levels of depth
            state.bid_depth = sum(float(l.get("sz", 0)) * float(l.get("px", 0)) for l in bids[:10])
            state.ask_depth = sum(float(l.get("sz", 0)) * float(l.get("px", 0)) for l in asks[:10])
        except (ValueError, TypeError):
            pass

    def update_oi(self, coin: str, new_oi: float):
        """Update open interest (called periodically from REST API)."""
        state = self.states.get(coin)
        if not state:
            return
        state.prev_oi = state.last_oi
        state.last_oi = new_oi
        state.last_oi_check = time.time()

    def update_funding(self, coin: str, rate: float):
        """Update funding rate."""
        state = self.states.get(coin)
        if state:
            state.funding_rate = rate

    def _check_panic(self, coin: str):
        """Check if panic threshold is breached."""
        state = self.states[coin]
        score = self.scorer.compute(state)
        state.panic_score = score

        if score >= PANIC_WAKE_THRESHOLD:
            now = time.time()
            if now - self.last_alert_time < self.alert_cooldown:
                return  # cooldown

            components = self.scorer.components(state)
            buy_vol, sell_vol = state.volume_by_side(TRADE_WINDOW_SEC)

            # Determine cascade direction
            if sell_vol > buy_vol * 1.5:
                direction = "LONG_LIQUIDATION"  # longs getting wrecked, price dumping
            elif buy_vol > sell_vol * 1.5:
                direction = "SHORT_SQUEEZE"     # shorts getting wrecked, price pumping
            else:
                direction = "MIXED"

            cascade = CascadeEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                coin=coin,
                panic_score=score,
                direction=direction,
                price_at_detection=state.last_price,
                price_velocity=round(state.price_velocity(PRICE_VELOCITY_SEC), 4),
                volume_spike=round(components["volume_spike"], 1),
                oi_change_pct=round((state.last_oi - state.prev_oi) / state.prev_oi * 100, 2) if state.prev_oi else 0,
                funding_rate=state.funding_rate,
                book_imbalance=round(state.bid_depth / state.ask_depth, 2) if state.ask_depth else 0,
                context=components,
            )

            self._trigger_alert(cascade)
            state.last_cascade_time = now
            self.last_alert_time = now

    def _trigger_alert(self, cascade: CascadeEvent):
        """Write alert file and trigger OpenClaw wake."""
        urgency = "URGENT" if cascade.panic_score >= PANIC_URGENT_THRESHOLD else "ALERT"
        log.warning(f"🚨 {urgency} [{cascade.coin}] score={cascade.panic_score} "
                    f"dir={cascade.direction} price=${cascade.price_at_detection:.2f} "
                    f"vel={cascade.price_velocity:.3f}%/min")

        # Save cascade event
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now().strftime("%Y-%m-%d")
        event_file = DATA_DIR / f"cascades_{date_str}.jsonl"
        with event_file.open("a") as f:
            f.write(json.dumps(asdict(cascade), default=str) + "\n")

        # Write alert for Curupira
        alert = {
            "type": "cascade_alert",
            "urgency": urgency,
            "cascade": asdict(cascade),
            "recommended_action": self._recommend_action(cascade),
            "written_at": datetime.now(timezone.utc).isoformat(),
        }
        ALERT_FILE.write_text(json.dumps(alert, indent=2, default=str))

        # Wake OpenClaw
        self._wake_curupira(cascade, urgency)

    def _recommend_action(self, cascade: CascadeEvent) -> str:
        if cascade.direction == "LONG_LIQUIDATION":
            return f"Potential LONG fade on {cascade.coin} — longs liquidating, price overshooting down"
        elif cascade.direction == "SHORT_SQUEEZE":
            return f"Potential SHORT fade on {cascade.coin} — shorts squeezed, price overshooting up"
        return f"Mixed cascade on {cascade.coin} — analyze further before acting"

    def _wake_curupira(self, cascade: CascadeEvent, urgency: str):
        """Trigger OpenClaw cron wake event."""
        try:
            # Use openclaw CLI to wake
            import subprocess
            text = (
                f"🚨 LIQUIDATION CASCADE DETECTED — {urgency}\n"
                f"Coin: {cascade.coin} | Score: {cascade.panic_score}/100\n"
                f"Direction: {cascade.direction}\n"
                f"Price: ${cascade.price_at_detection:.2f} | Velocity: {cascade.price_velocity:.3f}%/min\n"
                f"Read /tmp/cascade-alert.json for full context.\n"
                f"Analyze and decide: TRADE or HOLD."
            )
            subprocess.run(
                ["openclaw", "cron", "wake", "--text", text, "--mode", "now"],
                timeout=10, capture_output=True,
            )
            log.info("OpenClaw wake triggered")
        except Exception as e:
            log.error(f"Failed to wake OpenClaw: {e}")

    def get_dashboard_state(self) -> dict:
        """Get current state for command center dashboard."""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "coins": {
                coin: {
                    "panic_score": state.panic_score,
                    "last_price": state.last_price,
                    "components": self.scorer.components(state),
                    "volume_5m": round(state.volume_in_window(300), 2),
                    "volume_1h": round(state.volume_in_window(3600), 2),
                    "price_velocity": round(state.price_velocity(300), 4),
                    "bid_depth": round(state.bid_depth, 2),
                    "ask_depth": round(state.ask_depth, 2),
                    "oi": state.last_oi,
                    "funding": state.funding_rate,
                    "trade_count_5m": len(state.recent_trades(300)),
                }
                for coin, state in self.states.items()
            }
        }


# ═══════════════════════════════════════════════════════════════
# WEBSOCKET CLIENT
# ═══════════════════════════════════════════════════════════════

class LiquidationWatcher:
    """Main WebSocket client that feeds the cascade detector."""

    def __init__(self):
        self.detector = CascadeDetector()
        self.ws: Optional[websocket.WebSocketApp] = None
        self.running = True
        self.oi_thread: Optional[threading.Thread] = None

        # Import on-chain feed for OI/funding updates
        from onchain_feed import OnChainFeed
        self.feed = OnChainFeed(testnet=False)

    def start(self):
        """Start the watcher (blocking)."""
        log.info(f"Starting liquidation watcher — coins: {COINS}")
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        # Start OI polling thread
        self.oi_thread = threading.Thread(target=self._oi_poll_loop, daemon=True)
        self.oi_thread.start()

        # Start dashboard state writer
        self.dash_thread = threading.Thread(target=self._dashboard_loop, daemon=True)
        self.dash_thread.start()

        # Connect WebSocket with auto-reconnect
        while self.running:
            try:
                self._connect()
            except Exception as e:
                log.error(f"WebSocket error: {e}")
                if self.running:
                    log.info("Reconnecting in 5s...")
                    time.sleep(5)

    def stop(self):
        self.running = False
        if self.ws:
            self.ws.close()

    def _connect(self):
        self.ws = websocket.WebSocketApp(
            WS_URL,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self.ws.run_forever(ping_interval=30, ping_timeout=10)

    def _on_open(self, ws):
        log.info("WebSocket connected")
        # Subscribe to trades and order book for each coin
        for coin in COINS:
            ws.send(json.dumps({
                "method": "subscribe",
                "subscription": {"type": "trades", "coin": coin}
            }))
            ws.send(json.dumps({
                "method": "subscribe",
                "subscription": {"type": "l2Book", "coin": coin}
            }))
        log.info(f"Subscribed to {len(COINS)} coins (trades + l2Book)")

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
            channel = data.get("channel", "")

            if channel == "trades":
                coin_data = data.get("data", [])
                if coin_data and len(coin_data) > 0:
                    coin = coin_data[0].get("coin", "")
                    if coin in self.detector.states:
                        self.detector.on_trade(coin, coin_data)

            elif channel == "l2Book":
                book = data.get("data", {})
                coin = book.get("coin", "")
                if coin in self.detector.states:
                    self.detector.on_book(coin, book)

        except json.JSONDecodeError:
            pass
        except Exception as e:
            log.debug(f"Message processing error: {e}")

    def _on_error(self, ws, error):
        log.error(f"WebSocket error: {error}")

    def _on_close(self, ws, close_code, close_msg):
        log.info(f"WebSocket closed: {close_code} {close_msg}")

    def _oi_poll_loop(self):
        """Poll OI and funding every 60s via REST."""
        while self.running:
            try:
                snapshots = self.feed.get_all_funding_and_oi()
                for snap in snapshots:
                    if snap.coin in self.detector.states:
                        self.detector.update_oi(snap.coin, snap.open_interest)
                        self.detector.update_funding(snap.coin, snap.funding_rate)
            except Exception as e:
                log.debug(f"OI poll error: {e}")
            time.sleep(OI_CHECK_INTERVAL_SEC)

    def _dashboard_loop(self):
        """Write dashboard state every 10s."""
        state_file = Path(__file__).parent / "data" / "watcher_state.json"
        while self.running:
            try:
                state = self.detector.get_dashboard_state()
                state_file.write_text(json.dumps(state, indent=2, default=str))
            except Exception as e:
                log.debug(f"Dashboard write error: {e}")
            time.sleep(10)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    watcher = LiquidationWatcher()

    def sighandler(sig, frame):
        log.info("Shutting down...")
        watcher.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, sighandler)
    signal.signal(signal.SIGTERM, sighandler)

    watcher.start()


if __name__ == "__main__":
    main()
