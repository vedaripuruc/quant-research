#!/usr/bin/env python3
"""
Technical Indicators Module
============================
MACD, RSI, ATR, EMA - pre-calculated and formatted as structured
context for the LLM trading agent.

Reuses patterns from curupira-backtests/momentum_signals.py
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
    """Calculate EMA."""
    return prices.ewm(span=period, adjust=False).mean()


def calculate_macd(prices: pd.Series,
                   fast: int = 12, slow: int = 26, signal: int = 9
                   ) -> Dict[str, pd.Series]:
    """Calculate MACD, signal line, and histogram."""
    ema_fast = calculate_ema(prices, fast)
    ema_slow = calculate_ema(prices, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return {
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram,
    }


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate ATR from OHLCV dataframe."""
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def calculate_bollinger(prices: pd.Series, period: int = 20, std_mult: float = 2.0):
    """Calculate Bollinger Bands."""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    return {
        "upper": sma + std_mult * std,
        "middle": sma,
        "lower": sma - std_mult * std,
    }


def calculate_volume_profile(df: pd.DataFrame, lookback: int = 50) -> Dict:
    """Simple volume profile: average volume and relative current volume."""
    if len(df) < lookback:
        return {"avg_volume": 0, "relative_volume": 0}
    recent = df.tail(lookback)
    avg_vol = recent["volume"].mean()
    current_vol = df["volume"].iloc[-1]
    return {
        "avg_volume": round(avg_vol, 2),
        "relative_volume": round(current_vol / avg_vol, 2) if avg_vol > 0 else 0,
    }


def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicators to a dataframe."""
    df = df.copy()
    df["rsi_14"] = calculate_rsi(df["close"], 14)
    df["ema_9"] = calculate_ema(df["close"], 9)
    df["ema_21"] = calculate_ema(df["close"], 21)
    df["atr_14"] = calculate_atr(df, 14)

    macd = calculate_macd(df["close"])
    df["macd"] = macd["macd"]
    df["macd_signal"] = macd["signal"]
    df["macd_histogram"] = macd["histogram"]

    bb = calculate_bollinger(df["close"])
    df["bb_upper"] = bb["upper"]
    df["bb_middle"] = bb["middle"]
    df["bb_lower"] = bb["lower"]
    return df


def format_technical_context(df: pd.DataFrame, coin: str, lookback: int = 5) -> str:
    """
    Format the last `lookback` bars into a concise text summary
    suitable for LLM context injection.
    """
    df = enrich_dataframe(df)
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    vol_profile = calculate_volume_profile(df)

    # MACD crossover detection
    macd_cross = ""
    if prev["macd"] < prev["macd_signal"] and latest["macd"] > latest["macd_signal"]:
        macd_cross = "BULLISH CROSSOVER (MACD crossed above signal)"
    elif prev["macd"] > prev["macd_signal"] and latest["macd"] < latest["macd_signal"]:
        macd_cross = "BEARISH CROSSOVER (MACD crossed below signal)"
    else:
        macd_cross = "no crossover"

    # RSI zones
    rsi_val = latest["rsi_14"]
    if rsi_val > 70:
        rsi_zone = "OVERBOUGHT"
    elif rsi_val < 30:
        rsi_zone = "OVERSOLD"
    elif rsi_val > 60:
        rsi_zone = "bullish"
    elif rsi_val < 40:
        rsi_zone = "bearish"
    else:
        rsi_zone = "neutral"

    # EMA trend
    ema_trend = "bullish" if latest["ema_9"] > latest["ema_21"] else "bearish"

    # Bollinger Band position
    price = latest["close"]
    bb_pos = (price - latest["bb_lower"]) / (latest["bb_upper"] - latest["bb_lower"])

    # Price change
    bars = df.tail(lookback)
    pct_change = (bars["close"].iloc[-1] / bars["close"].iloc[0] - 1) * 100

    lines = [
        f"=== {coin} Technical Analysis ===",
        f"Price: {price:.4f}",
        f"Change ({lookback} bars): {pct_change:+.2f}%",
        f"",
        f"RSI(14): {rsi_val:.1f} ({rsi_zone})",
        f"MACD: {latest['macd']:.6f} | Signal: {latest['macd_signal']:.6f} | Histogram: {latest['macd_histogram']:.6f}",
        f"MACD Status: {macd_cross}",
        f"EMA(9/21): {latest['ema_9']:.4f} / {latest['ema_21']:.4f} ({ema_trend})",
        f"ATR(14): {latest['atr_14']:.4f}",
        f"Bollinger: {latest['bb_lower']:.4f} - {latest['bb_middle']:.4f} - {latest['bb_upper']:.4f} (price at {bb_pos:.0%})",
        f"Volume: {vol_profile['relative_volume']:.1f}x average",
    ]

    # Last N candles summary
    lines.append(f"\nLast {lookback} candles:")
    for _, row in bars.iterrows():
        candle_type = "GREEN" if row["close"] > row["open"] else "RED"
        body_pct = abs(row["close"] - row["open"]) / row["open"] * 100
        lines.append(
            f"  {row.name}: O={row['open']:.4f} H={row['high']:.4f} "
            f"L={row['low']:.4f} C={row['close']:.4f} ({candle_type} {body_pct:.2f}%)"
        )

    return "\n".join(lines)
