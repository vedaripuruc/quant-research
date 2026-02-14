#!/usr/bin/env python3
"""
Daily Signal Monitor — Jump Detection (Gold) + Hurst Regime (Crypto)
=====================================================================
Checked once per day from heartbeat cron.

Signal A: Jump Detection on Gold (GC=F)
  - RV vs BPV jump ratio on daily bars
  - Fade jumps (ratio > 0.3), ride trends (ratio < 0.1 + momentum)

Signal B: Hurst Regime on LINK/ADA/XRP
  - Rolling Hurst exponent (R/S analysis, window=50)
  - H > 0.6: trend-follow EMA slope
  - H < 0.4: fade momentum
  - 0.4-0.6: no trade (random walk)
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional


# ═══════════════════════════════════════════════════════════════════════════
# SHARED UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def compute_atr_series(highs, lows, closes, period=14):
    """Compute ATR from arrays. Returns the last ATR value."""
    n = len(highs)
    if n < period + 1:
        return None
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(
            np.abs(highs[1:] - closes[:-1]),
            np.abs(lows[1:] - closes[:-1])
        )
    )
    if len(tr) < period:
        return None
    return np.mean(tr[-period:])


# ═══════════════════════════════════════════════════════════════════════════
# SIGNAL A: JUMP DETECTION ON GOLD (GC=F)
# ═══════════════════════════════════════════════════════════════════════════

def _compute_realized_variance(returns: np.ndarray) -> float:
    return np.sum(returns ** 2)


def _compute_bipower_variation(returns: np.ndarray) -> float:
    abs_ret = np.abs(returns)
    return (np.pi / 2) * np.sum(abs_ret[1:] * abs_ret[:-1])


def check_gold_jump(window: int = 20) -> Optional[Dict]:
    """
    Check Gold futures for jump signal.
    
    Fetches 60 days of GC=F daily data.
    Uses last `window` days of log returns to compute RV, BPV, jump_ratio.
    
    Returns a signal dict or None.
    """
    try:
        ticker = yf.Ticker("GC=F")
        df = ticker.history(period="60d", interval="1d")
    except Exception as e:
        print(f"  ⚠ Error fetching GC=F: {e}")
        return None

    if df is None or df.empty or len(df) < window + 5:
        print(f"  ⚠ GC=F: insufficient data ({len(df) if df is not None else 0} bars)")
        return None

    df = df.reset_index()
    closes = df['Close'].values.astype(float)
    highs = df['High'].values.astype(float)
    lows = df['Low'].values.astype(float)
    opens = df['Open'].values.astype(float)

    # Log returns for the last `window` days (using yesterday's completed close)
    # We use indices [-window-1:] to get window+1 closes → window returns
    if len(closes) < window + 2:
        print(f"  ⚠ GC=F: not enough closes for window={window}")
        return None

    # Yesterday = second-to-last bar (last completed day)
    # Today = last bar (potentially incomplete / today's open)
    yesterday_idx = -2
    today_idx = -1

    # Compute log returns over last window days ending at yesterday
    segment = closes[yesterday_idx - window:yesterday_idx + 1]  # window+1 values → window returns
    log_returns = np.diff(np.log(segment))

    if len(log_returns) < window:
        print(f"  ⚠ GC=F: not enough returns ({len(log_returns)})")
        return None

    rv = _compute_realized_variance(log_returns)
    bpv = _compute_bipower_variation(log_returns)
    jump = max(rv - bpv, 0)
    jump_ratio = jump / rv if rv > 0 else 0

    # ATR for SL/TP (14-day)
    atr = compute_atr_series(highs, lows, closes, period=14)
    if atr is None or atr <= 0:
        print(f"  ⚠ GC=F: could not compute ATR")
        return None

    # Yesterday's return direction
    yesterday_ret = np.log(closes[yesterday_idx] / closes[yesterday_idx - 1])
    yesterday_pct = (closes[yesterday_idx] / closes[yesterday_idx - 1] - 1) * 100

    # Entry price = today's Open
    entry = opens[today_idx]

    # ── FADE signal: jump_ratio > 0.3 ──
    if jump_ratio > 0.3:
        # Fade yesterday's direction
        if yesterday_ret > 0:
            direction = "SHORT"
            stop_loss = entry + 1.0 * atr
            take_profit = entry - 2.0 * atr
        else:
            direction = "LONG"
            stop_loss = entry - 1.0 * atr
            take_profit = entry + 2.0 * atr

        return {
            "asset": "Gold",
            "symbol": "GC=F",
            "direction": direction,
            "entry": round(float(entry), 2),
            "stop_loss": round(float(stop_loss), 2),
            "take_profit": round(float(take_profit), 2),
            "signal_type": "jump_fade",
            "details": f"Jump ratio: {jump_ratio:.3f}, fading yesterday's {yesterday_pct:+.1f}% move | RV={rv:.6f} BPV={bpv:.6f}",
            "r_r": "1:2",
            "atr": round(float(atr), 2),
            "jump_ratio": round(float(jump_ratio), 4),
        }

    # ── TREND signal: jump_ratio < 0.1 AND clear 20-day momentum ──
    if jump_ratio < 0.1:
        # 20-day momentum
        if len(closes) >= 22:
            mom = (closes[yesterday_idx] - closes[yesterday_idx - 20]) / closes[yesterday_idx - 20]
        else:
            mom = 0

        # Require meaningful momentum (> 2%)
        if abs(mom) >= 0.02:
            if mom > 0:
                direction = "LONG"
                stop_loss = entry - 2.0 * atr
                take_profit = entry + 4.0 * atr
            else:
                direction = "SHORT"
                stop_loss = entry + 2.0 * atr
                take_profit = entry - 4.0 * atr

            return {
                "asset": "Gold",
                "symbol": "GC=F",
                "direction": direction,
                "entry": round(float(entry), 2),
                "stop_loss": round(float(stop_loss), 2),
                "take_profit": round(float(take_profit), 2),
                "signal_type": "jump_trend",
                "details": f"Jump ratio: {jump_ratio:.3f} (smooth), 20d momentum: {mom*100:+.1f}%",
                "r_r": "1:2",
                "atr": round(float(atr), 2),
                "jump_ratio": round(float(jump_ratio), 4),
            }

    # No signal — middle ground
    return None


# ═══════════════════════════════════════════════════════════════════════════
# SIGNAL B: HURST REGIME ON LINK / ADA / XRP
# ═══════════════════════════════════════════════════════════════════════════

HURST_ASSETS = {
    "LINK-USD": "LINK",
    "ADA-USD": "ADA",
    "XRP-USD": "XRP",
}


def _hurst_rs(series: np.ndarray) -> float:
    """
    Compute Hurst exponent via Rescaled Range (R/S) analysis.
    Returns H ∈ [0, 1]. H=0.5 → random walk.
    """
    N = len(series)
    if N < 20:
        return np.nan

    max_k = N // 2
    min_k = 8

    ks = []
    k = min_k
    while k <= max_k:
        ks.append(k)
        k = int(k * 1.5)
        if k == ks[-1]:
            k += 1

    if len(ks) < 3:
        return np.nan

    rs_values = []
    ns_values = []

    for k in ks:
        n_chunks = N // k
        if n_chunks < 1:
            continue

        rs_list = []
        for chunk_i in range(n_chunks):
            chunk = series[chunk_i * k:(chunk_i + 1) * k]
            mean_chunk = np.mean(chunk)
            deviations = chunk - mean_chunk
            cumulative = np.cumsum(deviations)
            R = np.max(cumulative) - np.min(cumulative)
            S = np.std(chunk, ddof=1)
            if S > 0:
                rs_list.append(R / S)

        if rs_list:
            rs_values.append(np.mean(rs_list))
            ns_values.append(k)

    if len(ns_values) < 3:
        return np.nan

    log_n = np.log(np.array(ns_values))
    log_rs = np.log(np.array(rs_values))

    try:
        coeffs = np.polyfit(log_n, log_rs, 1)
        H = coeffs[0]
        return max(0.0, min(1.0, H))
    except:
        return np.nan


def _rolling_hurst(returns: np.ndarray, window: int = 50) -> np.ndarray:
    """Rolling Hurst exponent over log returns."""
    n = len(returns)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        segment = returns[i - window + 1:i + 1]
        if np.any(np.isnan(segment)):
            continue
        result[i] = _hurst_rs(segment)
    return result


def check_hurst_crypto() -> List[Optional[Dict]]:
    """
    Check Hurst regime signals for LINK, ADA, XRP.
    
    Returns list of signal dicts (one per asset, None if no signal).
    """
    signals = []

    for symbol, name in HURST_ASSETS.items():
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="100d", interval="1d")
        except Exception as e:
            print(f"  ⚠ Error fetching {symbol}: {e}")
            signals.append(None)
            continue

        if df is None or df.empty or len(df) < 60:
            print(f"  ⚠ {symbol}: insufficient data ({len(df) if df is not None else 0} bars)")
            signals.append(None)
            continue

        df = df.reset_index()
        closes = df['Close'].values.astype(float)
        highs = df['High'].values.astype(float)
        lows = df['Low'].values.astype(float)

        # Log returns
        log_rets = np.diff(np.log(closes))

        # Rolling Hurst (window=50)
        hurst_vals = _rolling_hurst(log_rets, window=50)

        # We need the last completed bar (second-to-last index in log_rets)
        # log_rets has len(closes)-1 elements
        last_idx = len(log_rets) - 1  # Last completed return

        if last_idx < 50:
            print(f"  ⚠ {symbol}: not enough data for Hurst (need 50 returns)")
            signals.append(None)
            continue

        H = hurst_vals[last_idx]
        if np.isnan(H):
            print(f"  ⚠ {symbol}: Hurst is NaN")
            signals.append(None)
            continue

        # 20-bar EMA and its slope
        close_series = pd.Series(closes)
        ema_20 = close_series.ewm(span=20, adjust=False).mean().values
        ema_slope = ema_20[-1] - ema_20[-2]  # Slope at last bar

        # 10-bar momentum (close[-1] - close[-11])
        if len(closes) >= 11:
            mom_10 = closes[-1] - closes[-11]
        else:
            mom_10 = 0

        # ATR (14-bar)
        atr = compute_atr_series(highs, lows, closes, period=14)
        if atr is None or atr <= 0:
            print(f"  ⚠ {symbol}: could not compute ATR")
            signals.append(None)
            continue

        # Entry = latest close (as proxy for next open — crypto trades 24/7)
        entry = float(closes[-1])

        direction = None
        signal_type = None
        details_prefix = f"Hurst: {H:.3f}"

        if H > 0.6:
            # Trending regime → follow EMA slope
            if ema_slope > 0:
                direction = "LONG"
                signal_type = "hurst_trend"
                details_prefix += f", EMA slope: +{ema_slope:.4f} (trending up)"
            elif ema_slope < 0:
                direction = "SHORT"
                signal_type = "hurst_trend"
                details_prefix += f", EMA slope: {ema_slope:.4f} (trending down)"
        elif H < 0.4:
            # Mean-reverting → fade momentum
            if mom_10 > 0:
                direction = "SHORT"
                signal_type = "hurst_fade"
                details_prefix += f", mom10: +{mom_10:.4f} (fading up)"
            elif mom_10 < 0:
                direction = "LONG"
                signal_type = "hurst_fade"
                details_prefix += f", mom10: {mom_10:.4f} (fading down)"
        # else: 0.4-0.6 → random walk, skip

        if direction is None:
            print(f"  ℹ {name}: H={H:.3f} — no signal (zone: {'random walk' if 0.4 <= H <= 0.6 else 'edge'})")
            signals.append(None)
            continue

        # SL/TP: 1.5×ATR / 3×ATR
        sl_dist = 1.5 * atr
        tp_dist = 3.0 * atr

        if direction == "LONG":
            stop_loss = entry - sl_dist
            take_profit = entry + tp_dist
        else:
            stop_loss = entry + sl_dist
            take_profit = entry - tp_dist

        sig = {
            "asset": name,
            "symbol": symbol,
            "direction": direction,
            "entry": round(entry, 6),
            "stop_loss": round(float(stop_loss), 6),
            "take_profit": round(float(take_profit), 6),
            "signal_type": signal_type,
            "details": details_prefix,
            "r_r": "1:2",
            "atr": round(float(atr), 6),
            "hurst": round(float(H), 4),
        }
        signals.append(sig)

    return signals


# ═══════════════════════════════════════════════════════════════════════════
# MAIN INTERFACE
# ═══════════════════════════════════════════════════════════════════════════

def check_daily_signals() -> Dict:
    """
    Check all daily signals: Gold Jump + Hurst Regime.
    
    Returns dict matching the format of momentum_signals.check_all_signals():
    {
        "status": "signals" | "ok",
        "signals": [...],
        "no_signal": [...]
    }
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "status": "ok",
        "signals": [],
        "no_signal": [],
    }

    # ── Signal A: Gold Jump Detection ──
    # ══════════════════════════════════════════════════════════════
    # SIGNALS DISABLED (2026-02-13)
    # Distribution audit proved both signals are phantom:
    #   - Jump fade: 50-51% WR at ALL thresholds on forex hourly
    #   - Hurst: H > 0.6 fires 100% of the time on crypto hourly
    # See: jump_ratio_profile.py, hurst_profile.py
    # Re-enable when properly recalibrated or replaced with ECVT/LLM agent
    # ══════════════════════════════════════════════════════════════
    SIGNALS_ENABLED = False

    if not SIGNALS_ENABLED:
        print("  ⚠️ Daily signals DISABLED — phantom signals killed 2026-02-13")
        print("  Rebuilding around ECVT + LLM trading agent")
        result["no_signal"].extend(["Gold", "LINK", "ADA", "XRP"])
        return result

    print("  📈 Gold (GC=F) — Jump Detection...")
    try:
        gold_signal = check_gold_jump()
        if gold_signal:
            result["signals"].append(gold_signal)
        else:
            result["no_signal"].append("Gold")
    except Exception as e:
        print(f"  ❌ Gold jump check failed: {e}")
        result["no_signal"].append("Gold")

    # ── Signal B: Hurst Regime (LINK/ADA/XRP) ──
    print("  📈 Crypto (LINK/ADA/XRP) — Hurst Regime...")
    try:
        hurst_signals = check_hurst_crypto()
        for i, (symbol, name) in enumerate(HURST_ASSETS.items()):
            sig = hurst_signals[i] if i < len(hurst_signals) else None
            if sig:
                result["signals"].append(sig)
            else:
                result["no_signal"].append(name)
    except Exception as e:
        print(f"  ❌ Hurst check failed: {e}")
        for name in HURST_ASSETS.values():
            if name not in [s["asset"] for s in result["signals"]]:
                result["no_signal"].append(name)

    if result["signals"]:
        result["status"] = "signals"

    return result


def format_daily_signal_alert(signal: Dict) -> str:
    """Format a daily signal for human-readable display."""
    direction_emoji = "🟢" if signal["direction"] == "LONG" else "🔴"
    type_label = {
        "jump_fade": "⚡ JUMP FADE",
        "jump_trend": "📈 JUMP TREND",
        "hurst_trend": "🌊 HURST TREND",
        "hurst_fade": "🔄 HURST FADE",
    }.get(signal["signal_type"], signal["signal_type"])

    # Determine decimal places based on asset
    if signal["symbol"] == "GC=F":
        fmt = ".2f"
    else:
        fmt = ".4f" if signal["entry"] > 1 else ".6f"

    alert = f"""{direction_emoji} **{signal['direction']} {signal['asset']}** — {type_label}

• Entry: ${signal['entry']:{fmt}}
• Stop Loss: ${signal['stop_loss']:{fmt}}
• Take Profit: ${signal['take_profit']:{fmt}}
• R:R: {signal['r_r']}
• ATR: ${signal['atr']:{fmt}}

📝 {signal['details']}"""

    return alert


# ── Standalone run ──
if __name__ == "__main__":
    print("=" * 55)
    print("DAILY SIGNAL MONITOR")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)

    results = check_daily_signals()

    print("\n" + "-" * 55)
    if results["status"] == "signals":
        print(f"🎯 SIGNALS: {len(results['signals'])}\n")
        for sig in results["signals"]:
            print(format_daily_signal_alert(sig))
            print("-" * 40)
    else:
        print("No daily signals.")

    if results["no_signal"]:
        print(f"\nNo signal: {', '.join(results['no_signal'])}")
