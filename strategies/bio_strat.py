"""
Flocking Contagion Detector — Bio-Ecological Intraday Trading Strategy
======================================================================

Combines two biological models:
  1. Vicsek Flocking Order Parameter (Φ) — regime detection
  2. SIR Epidemic R₀ — momentum contagion signal

Papers grounding this approach:
  - Kazemian et al. (2020) "Market of Stocks during Crisis Looks Like a Flock of Birds" (Entropy)
  - Montero (2009) "Predator-Prey Model for Stock Market Fluctuations" (arXiv:0810.4844)
  - Miyahara et al. (2024) "Emergent Invariance & Scaling" (arXiv:2212.12703)
  - Gerig (2012) "HFT Synchronizes Prices" (arXiv:1211.1919)

Author: Curupira (autonomous agent)
Date: 2026-02-06
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Tuple, List


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class FlockingConfig:
    """Strategy hyperparameters."""
    # --- Vicsek Flocking Parameters ---
    zscore_window: int = 20           # Rolling window for z-score normalization
    phi_threshold_enter: float = 0.40 # Φ above this → herding regime
    phi_threshold_exit: float = 0.30  # Φ below this → flock dispersed

    # --- SIR Epidemic Parameters ---
    sir_lookback: int = 10            # Bars to estimate β (infection rate)
    sir_momentum_threshold: float = 1.0  # z-score threshold for "infected"
    sir_recovery_window: int = 5      # Bars of mean-reversion → "recovered"
    r0_threshold: float = 1.0        # R₀ above this → epidemic spreading

    # --- Risk Management ---
    stop_loss_pct: float = 0.02      # 2% stop loss
    take_profit_pct: float = 0.04    # 4% take profit (2:1 R/R)
    max_hold_bars: int = 60          # Maximum bars to hold (1 hour at 1-min)


# =============================================================================
# VICSEK FLOCKING ORDER PARAMETER
# =============================================================================

def compute_vicsek_phi(
    returns_matrix: pd.DataFrame,
    volume_changes_matrix: pd.DataFrame,
    zscore_window: int = 20,
) -> pd.Series:
    """
    Compute Vicsek order parameter Φ for a basket of stocks.

    Each stock is a "particle" in 2D phase space:
        X = z-scored return
        Y = z-scored volume change

    The velocity vector is the bar-to-bar change in (X, Y).
    Φ = |mean of normalized velocity vectors| ∈ [0, 1]
        Φ → 1: all stocks moving in same direction (flocking/herding)
        Φ → 0: random movement (normal market)

    Args:
        returns_matrix: DataFrame of log-returns, columns = stock symbols
        volume_changes_matrix: DataFrame of volume pct changes, columns = stocks
        zscore_window: Rolling window for z-score normalization

    Returns:
        Series of Φ values indexed by timestamp
    """
    # Z-score normalize returns and volume changes
    ret_mean = returns_matrix.rolling(zscore_window, min_periods=5).mean()
    ret_std = returns_matrix.rolling(zscore_window, min_periods=5).std()
    ret_std = ret_std.where(ret_std > 1e-10, 1.0)  # Avoid div by zero; use raw value
    x = (returns_matrix - ret_mean) / ret_std

    vol_mean = volume_changes_matrix.rolling(zscore_window, min_periods=5).mean()
    vol_std = volume_changes_matrix.rolling(zscore_window, min_periods=5).std()
    vol_std = vol_std.where(vol_std > 1e-10, 1.0)  # Avoid div by zero
    y = (volume_changes_matrix - vol_mean) / vol_std

    # Fill remaining NaN with 0 (early bars before enough history)
    x = x.fillna(0)
    y = y.fillna(0)

    # Velocity = bar-to-bar change in phase space position
    vx = x.diff().fillna(0)
    vy = y.diff().fillna(0)

    # Normalize each velocity vector to unit length
    speed = np.sqrt(vx**2 + vy**2)
    speed = speed.where(speed > 1e-10, 1.0)  # Avoid div by zero
    vx_norm = vx / speed
    vy_norm = vy / speed

    # Order parameter: magnitude of mean normalized velocity
    mean_vx = vx_norm.mean(axis=1)
    mean_vy = vy_norm.mean(axis=1)
    phi = np.sqrt(mean_vx**2 + mean_vy**2)

    # Also compute the flock direction (angle in radians)
    flock_angle = np.arctan2(mean_vy, mean_vx)

    return phi, flock_angle


def classify_flock_direction(flock_angle: pd.Series) -> pd.Series:
    """
    Classify the flock direction into bullish/bearish.

    If the X-component (return dimension) of the mean velocity is positive,
    the flock is moving toward higher returns → bullish.
    If negative → bearish.
    """
    # Angle between -π/2 and π/2 means positive X component → bullish
    direction = pd.Series(0, index=flock_angle.index)
    direction[np.cos(flock_angle) > 0] = 1   # Bullish
    direction[np.cos(flock_angle) < 0] = -1   # Bearish
    return direction


# =============================================================================
# SIR MOMENTUM CONTAGION MODEL
# =============================================================================

def compute_sir_r0(
    returns_matrix: pd.DataFrame,
    lookback: int = 10,
    momentum_threshold: float = 1.0,
    recovery_window: int = 5,
) -> pd.Series:
    """
    Compute the SIR basic reproduction number R₀ = β/γ for momentum contagion.

    Key insight: "infection" is measured via TIME-SERIES z-scores (each stock vs
    its own recent history), NOT cross-sectional. This way, when ALL stocks move
    together (crisis), they are ALL detected as "infected" — the epidemic is
    spreading through the population.

    - A stock is "Infected" if |return| exceeds threshold vs its own rolling σ
    - β (infection rate): rate at which new stocks join the trend
    - γ (recovery rate): rate at which infected stocks mean-revert
    - R₀ = β/γ; if R₀ > 1, the momentum epidemic is spreading

    Args:
        returns_matrix: DataFrame of log-returns, columns = stock symbols
        lookback: Bars to estimate rates
        momentum_threshold: z-score threshold to classify as "infected"
        recovery_window: Bars of sub-threshold returns → "recovered"

    Returns:
        Series of R₀ values
    """
    n_stocks = returns_matrix.shape[1]

    # Time-series z-score: each stock vs its own rolling mean/std
    ts_mean = returns_matrix.rolling(lookback * 2, min_periods=5).mean()
    ts_std = returns_matrix.rolling(lookback * 2, min_periods=5).std()
    ts_std = ts_std.where(ts_std > 1e-10, 1e-10)
    z_ts = (returns_matrix - ts_mean) / ts_std

    # Determine dominant direction: sign of the cross-sectional mean return
    cross_mean = returns_matrix.mean(axis=1)
    direction = np.sign(cross_mean).replace(0, 1)

    # "Infected" = stock's directional z-score exceeds threshold
    # (positive z in the trend direction means it caught the "virus")
    directional_z = z_ts.mul(direction, axis=0)
    infected = (directional_z > momentum_threshold).astype(float)

    # Count infected stocks
    n_infected = infected.sum(axis=1)
    frac_infected = n_infected / n_stocks

    # New infections: stocks that just became infected
    new_infections = infected.diff().clip(lower=0).fillna(0)
    n_new = new_infections.sum(axis=1)

    # Recoveries: stocks that were infected but returned below threshold
    recoveries = (-infected.diff()).clip(lower=0).fillna(0)
    n_recovered = recoveries.sum(axis=1)

    # Estimate β and γ over lookback window
    susceptible_frac = (1 - frac_infected).clip(lower=0.01)
    beta_raw = n_new / (n_stocks * susceptible_frac)
    beta = beta_raw.rolling(lookback, min_periods=3).mean()

    gamma_raw = n_recovered / (n_stocks * frac_infected.clip(lower=0.01))
    gamma = gamma_raw.rolling(lookback, min_periods=3).mean()

    # R₀ = β / γ (clipped to avoid infinity)
    r0 = beta / gamma.clip(lower=1e-6)
    r0 = r0.clip(upper=10.0).fillna(0)

    return r0, frac_infected


# =============================================================================
# COMBINED STRATEGY — OHLCV VERSION
# =============================================================================

def prepare_basket_data(
    ohlcv_dict: dict[str, pd.DataFrame],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare returns and volume changes matrices from dict of OHLCV DataFrames.

    Args:
        ohlcv_dict: {symbol: DataFrame with columns [timestamp, open, high, low, close, volume]}

    Returns:
        (returns_matrix, volume_changes_matrix) both indexed by timestamp
    """
    closes = {}
    volumes = {}
    for sym, df in ohlcv_dict.items():
        df = df.set_index('timestamp') if 'timestamp' in df.columns else df
        closes[sym] = df['close']
        volumes[sym] = df['volume']

    close_df = pd.DataFrame(closes)
    volume_df = pd.DataFrame(volumes)

    # Log returns
    returns = np.log(close_df / close_df.shift(1))

    # Volume percent change
    vol_change = volume_df.pct_change()

    return returns.dropna(), vol_change.dropna()


def generate_signals_ohlcv(
    ohlcv_dict: dict[str, pd.DataFrame],
    target_symbol: str,
    config: FlockingConfig = None,
) -> pd.DataFrame:
    """
    Generate entry/exit signals using the Flocking Contagion Detector.

    Works with 1-minute or 1-hour OHLCV bars for a basket of correlated stocks.
    Signals are generated for a single target symbol.

    Args:
        ohlcv_dict: {symbol: DataFrame with [timestamp, open, high, low, close, volume]}
        target_symbol: The symbol we actually trade
        config: Strategy configuration

    Returns:
        DataFrame with columns: [timestamp, phi, r0, flock_dir, signal, position]
        signal: 1 = long entry, -1 = short entry, 0 = no signal
    """
    if config is None:
        config = FlockingConfig()

    returns_mat, vol_changes_mat = prepare_basket_data(ohlcv_dict)

    # Align all data
    common_idx = returns_mat.index.intersection(vol_changes_mat.index)
    returns_mat = returns_mat.loc[common_idx]
    vol_changes_mat = vol_changes_mat.loc[common_idx]

    # --- Component 1: Vicsek Flocking ---
    phi, flock_angle = compute_vicsek_phi(
        returns_mat, vol_changes_mat, config.zscore_window
    )
    flock_dir = classify_flock_direction(flock_angle)

    # --- Component 2: SIR Contagion ---
    r0, frac_infected = compute_sir_r0(
        returns_mat,
        lookback=config.sir_lookback,
        momentum_threshold=config.sir_momentum_threshold,
        recovery_window=config.sir_recovery_window,
    )

    # --- Signal Generation ---
    # Herding regime: Φ above entry threshold
    in_herding = phi > config.phi_threshold_enter

    # Epidemic spreading: R₀ above threshold
    # Use smoothed R₀ (3-bar EMA) to avoid single-bar spikes
    r0_smooth = r0.ewm(span=3, min_periods=1).mean()
    epidemic_active = r0_smooth > config.r0_threshold

    # Combined condition: herding + epidemic + direction
    raw_signal = pd.Series(0, index=common_idx)
    bullish = in_herding & epidemic_active & (flock_dir == 1)
    bearish = in_herding & epidemic_active & (flock_dir == -1)
    raw_signal[bullish] = 1
    raw_signal[bearish] = -1

    # Also allow: if Φ very high (>0.6) even without R₀, the flock is undeniable
    strong_flock = phi > 0.6
    raw_signal[(strong_flock) & (flock_dir == 1) & (raw_signal == 0)] = 1
    raw_signal[(strong_flock) & (flock_dir == -1) & (raw_signal == 0)] = -1

    # Smooth flock direction with 3-bar majority vote to avoid flip-flop
    flock_dir_smooth = flock_dir.rolling(3, min_periods=1).median()

    # Re-apply direction to signal
    raw_signal_smooth = pd.Series(0, index=common_idx)
    condition = in_herding & (epidemic_active | strong_flock)
    raw_signal_smooth[condition & (flock_dir_smooth > 0)] = 1
    raw_signal_smooth[condition & (flock_dir_smooth < 0)] = -1

    # Entry: trigger on first bar of a new signal direction
    entry_signal = pd.Series(0, index=common_idx)
    prev_raw = raw_signal_smooth.shift(1).fillna(0)
    # New long entry
    entry_signal[(raw_signal_smooth == 1) & (prev_raw != 1)] = 1
    # New short entry
    entry_signal[(raw_signal_smooth == -1) & (prev_raw != -1)] = -1

    # Cooldown: suppress signals within 5 bars of a previous signal
    cooldown = 5
    last_signal_bar = -cooldown - 1
    for i in range(len(entry_signal)):
        if entry_signal.iloc[i] != 0:
            if i - last_signal_bar <= cooldown:
                entry_signal.iloc[i] = 0
            else:
                last_signal_bar = i

    # Exit signal: Φ drops below exit threshold AND R₀ drops below 1
    # (both must weaken for exit)
    phi_weak = phi < config.phi_threshold_exit
    r0_weak = r0_smooth < config.r0_threshold
    exit_condition = phi_weak & r0_weak

    result = pd.DataFrame({
        'phi': phi,
        'r0': r0,
        'frac_infected': frac_infected,
        'flock_dir': flock_dir,
        'signal': entry_signal,
        'exit_signal': exit_condition.astype(int),
    }, index=common_idx)

    return result


# =============================================================================
# TICK DATA VARIANT
# =============================================================================

def generate_signals_tick(
    tick_dict: dict[str, pd.DataFrame],
    target_symbol: str,
    resample_freq: str = '1min',
    config: FlockingConfig = None,
) -> pd.DataFrame:
    """
    Tick data variant: resample tick data to bars, then apply flocking strategy.

    Tick data provides richer information:
    - Mid-price from bid/ask (less noisy than trade price)
    - Bid-ask spread as additional "dimension" in phase space

    Args:
        tick_dict: {symbol: DataFrame with [timestamp, bid, ask, volume]}
        target_symbol: Symbol to trade
        resample_freq: Resampling frequency ('1min', '5min', etc.)
        config: Strategy configuration

    Returns:
        Signal DataFrame
    """
    if config is None:
        config = FlockingConfig()

    ohlcv_dict = {}
    for sym, ticks in tick_dict.items():
        ticks = ticks.set_index('timestamp') if 'timestamp' in ticks.columns else ticks
        mid = (ticks['bid'] + ticks['ask']) / 2

        # Resample to OHLCV bars
        bars = mid.resample(resample_freq).ohlc()
        bars.columns = ['open', 'high', 'low', 'close']
        bars['volume'] = ticks['volume'].resample(resample_freq).sum()
        bars = bars.dropna()
        ohlcv_dict[sym] = bars.reset_index().rename(columns={'index': 'timestamp'})

    return generate_signals_ohlcv(ohlcv_dict, target_symbol, config)


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

@dataclass
class Trade:
    """Represents a single trade."""
    entry_time: pd.Timestamp
    entry_price: float
    direction: int  # 1 = long, -1 = short
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    exit_reason: str = ''


def backtest_ohlcv(
    signals: pd.DataFrame,
    target_ohlcv: pd.DataFrame,
    config: FlockingConfig = None,
) -> Tuple[List[Trade], pd.DataFrame]:
    """
    Simple backtest loop. NO look-ahead bias.

    Rules:
    - Entry at NEXT bar's OPEN after signal fires
    - Exit when: exit_signal fires, stop-loss hit, take-profit hit, or max hold exceeded
    - One position at a time

    Args:
        signals: Output from generate_signals_ohlcv
        target_ohlcv: OHLCV DataFrame for the target symbol
        config: Strategy configuration

    Returns:
        (trades, equity_curve)
    """
    if config is None:
        config = FlockingConfig()

    target = target_ohlcv.set_index('timestamp') if 'timestamp' in target_ohlcv.columns else target_ohlcv
    common_idx = signals.index.intersection(target.index)
    signals = signals.loc[common_idx]
    target = target.loc[common_idx]

    trades: List[Trade] = []
    current_trade: Optional[Trade] = None
    pending_signal: int = 0
    equity = 10000.0
    equity_curve = []
    bars_held = 0

    for i in range(len(common_idx)):
        ts = common_idx[i]
        bar = target.loc[ts]

        # --- Execute pending entry at this bar's open ---
        if pending_signal != 0 and current_trade is None:
            current_trade = Trade(
                entry_time=ts,
                entry_price=bar['open'],
                direction=pending_signal,
            )
            bars_held = 0
            pending_signal = 0

        # --- Check exits for existing position ---
        if current_trade is not None:
            bars_held += 1
            direction = current_trade.direction
            entry_price = current_trade.entry_price

            # Stop loss check (use high/low of current bar)
            if direction == 1:
                # Long: stop if low drops below entry * (1 - SL)
                sl_price = entry_price * (1 - config.stop_loss_pct)
                tp_price = entry_price * (1 + config.take_profit_pct)
                if bar['low'] <= sl_price:
                    current_trade.exit_time = ts
                    current_trade.exit_price = sl_price
                    current_trade.exit_reason = 'stop_loss'
                elif bar['high'] >= tp_price:
                    current_trade.exit_time = ts
                    current_trade.exit_price = tp_price
                    current_trade.exit_reason = 'take_profit'
            else:
                # Short: stop if high rises above entry * (1 + SL)
                sl_price = entry_price * (1 + config.stop_loss_pct)
                tp_price = entry_price * (1 - config.take_profit_pct)
                if bar['high'] >= sl_price:
                    current_trade.exit_time = ts
                    current_trade.exit_price = sl_price
                    current_trade.exit_reason = 'stop_loss'
                elif bar['low'] <= tp_price:
                    current_trade.exit_time = ts
                    current_trade.exit_price = tp_price
                    current_trade.exit_reason = 'take_profit'

            # Exit signal from strategy
            if current_trade.exit_time is None and signals.loc[ts, 'exit_signal'] == 1:
                current_trade.exit_time = ts
                current_trade.exit_price = bar['close']
                current_trade.exit_reason = 'signal_exit'

            # Max hold bars
            if current_trade.exit_time is None and bars_held >= config.max_hold_bars:
                current_trade.exit_time = ts
                current_trade.exit_price = bar['close']
                current_trade.exit_reason = 'max_hold'

            # Close trade if exited
            if current_trade.exit_time is not None:
                pnl = (current_trade.exit_price - current_trade.entry_price) * direction
                current_trade.pnl = pnl / current_trade.entry_price  # Return %
                equity *= (1 + current_trade.pnl)
                trades.append(current_trade)
                current_trade = None

        # --- Check for new entry signal (will execute at NEXT bar open) ---
        sig = signals.loc[ts, 'signal']
        if sig != 0 and current_trade is None:
            pending_signal = int(sig)

        equity_curve.append({'timestamp': ts, 'equity': equity})

    # Close any open trade at last bar
    if current_trade is not None:
        ts = common_idx[-1]
        bar = target.loc[ts]
        current_trade.exit_time = ts
        current_trade.exit_price = bar['close']
        current_trade.exit_reason = 'end_of_data'
        pnl = (current_trade.exit_price - current_trade.entry_price) * current_trade.direction
        current_trade.pnl = pnl / current_trade.entry_price
        equity *= (1 + current_trade.pnl)
        trades.append(current_trade)

    equity_df = pd.DataFrame(equity_curve)
    return trades, equity_df


# =============================================================================
# ANALYTICS
# =============================================================================

def print_backtest_report(trades: List[Trade], equity_df: pd.DataFrame):
    """Print summary statistics for the backtest."""
    if not trades:
        print("No trades generated.")
        return

    pnls = [t.pnl for t in trades if t.pnl is not None]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    print("=" * 60)
    print("FLOCKING CONTAGION DETECTOR — BACKTEST REPORT")
    print("=" * 60)
    print(f"Total trades:     {len(pnls)}")
    print(f"Winning trades:   {len(wins)} ({100*len(wins)/max(len(pnls),1):.1f}%)")
    print(f"Losing trades:    {len(losses)} ({100*len(losses)/max(len(pnls),1):.1f}%)")
    print(f"")
    print(f"Total return:     {100*(equity_df['equity'].iloc[-1]/10000 - 1):.2f}%")
    print(f"Avg trade return: {100*np.mean(pnls):.3f}%")
    print(f"Avg win:          {100*np.mean(wins):.3f}%" if wins else "Avg win: N/A")
    print(f"Avg loss:         {100*np.mean(losses):.3f}%" if losses else "Avg loss: N/A")
    print(f"Profit factor:    {abs(sum(wins)/sum(losses)):.2f}" if losses and sum(losses) != 0 else "Profit factor: ∞")
    print(f"Max drawdown:     {100*max_drawdown(equity_df['equity']):.2f}%")
    print(f"")

    # Exit reasons
    reasons = {}
    for t in trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
    print("Exit reasons:")
    for reason, count in sorted(reasons.items()):
        print(f"  {reason}: {count}")
    print("=" * 60)


def max_drawdown(equity_series: pd.Series) -> float:
    """Compute maximum drawdown as a fraction."""
    peak = equity_series.expanding().max()
    dd = (equity_series - peak) / peak
    return abs(dd.min())


# =============================================================================
# SYNTHETIC DATA GENERATOR (for testing)
# =============================================================================

def generate_synthetic_basket(
    n_stocks: int = 30,
    n_bars: int = 1000,
    crisis_start: int = 400,
    crisis_end: int = 500,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """
    Generate synthetic OHLCV data with a crisis period where stocks flock.

    During normal period: stocks have independent random walks.
    During crisis: stocks correlate strongly (mimicking herding).

    This is for testing ONLY — real data is needed for actual validation.
    """
    np.random.seed(seed)
    timestamps = pd.date_range('2024-01-02 09:30', periods=n_bars, freq='1min')

    basket = {}
    for i in range(n_stocks):
        base_price = 100 + np.random.randn() * 20
        base_volume = 10000 + np.random.rand() * 5000

        returns = np.random.randn(n_bars) * 0.001  # Normal: small random returns
        # Volume has meaningful bar-to-bar variation (log-normal like real data)
        volume_mult = np.exp(np.random.randn(n_bars) * 0.15)  # ~15% vol variation

        # Crisis: inject correlated negative returns + volume spike
        # Phase 1 (first 30 bars): sharp onset — stocks suddenly align
        # Phase 2 (next 40 bars): sustained herding with waves
        # Phase 3 (last 30 bars): flock disperses gradually
        for t in range(crisis_start, min(crisis_end, n_bars)):
            progress = (t - crisis_start) / max(crisis_end - crisis_start, 1)
            if progress < 0.3:
                # Sharp onset: strongly correlated crash
                returns[t] = -0.004 + np.random.randn() * 0.0003
                volume_mult[t] = 4.0 + np.random.randn() * 0.2
            elif progress < 0.7:
                # Sustained: waves of selling with some bounces
                wave = np.sin(2 * np.pi * progress * 3) * 0.002
                returns[t] = -0.002 + wave + np.random.randn() * 0.0005
                volume_mult[t] = 3.0 + np.random.randn() * 0.3
            else:
                # Dispersal: some stocks recover, others don't
                if np.random.rand() < 0.3:
                    returns[t] = 0.001 + np.random.randn() * 0.001  # Recovery
                else:
                    returns[t] = -0.001 + np.random.randn() * 0.0008
                volume_mult[t] = 2.0 + np.random.randn() * 0.4

        # Build price series
        prices = base_price * np.exp(np.cumsum(returns))
        volumes = (base_volume * volume_mult).clip(min=100)

        # OHLCV from price
        noise = np.random.rand(n_bars) * 0.001
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices * (1 - noise / 2),
            'high': prices * (1 + abs(noise)),
            'low': prices * (1 - abs(noise)),
            'close': prices,
            'volume': volumes.astype(int),
        })
        basket[f'STOCK_{i:02d}'] = df

    return basket


# =============================================================================
# MAIN: Demo with synthetic data
# =============================================================================

if __name__ == '__main__':
    print("Generating synthetic basket data (30 stocks, 1000 1-min bars)...")
    print("Crisis period: bars 400-500 (stocks flock together)\n")

    basket = generate_synthetic_basket(
        n_stocks=30, n_bars=1000,
        crisis_start=400, crisis_end=500,
    )
    target = 'STOCK_00'

    config = FlockingConfig(
        zscore_window=20,
        phi_threshold_enter=0.35,  # Slightly lower for synthetic data
        phi_threshold_exit=0.25,
        sir_lookback=10,
        sir_momentum_threshold=0.8,
        r0_threshold=1.0,
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
        max_hold_bars=60,
    )

    print("Computing Vicsek flocking order parameter + SIR R₀...")
    signals = generate_signals_ohlcv(basket, target, config)

    print(f"\nSignal summary:")
    print(f"  Long entries:  {(signals['signal'] == 1).sum()}")
    print(f"  Short entries: {(signals['signal'] == -1).sum()}")
    print(f"  Bars in herding regime (Φ > {config.phi_threshold_enter}): "
          f"{(signals['phi'] > config.phi_threshold_enter).sum()}")
    print(f"  Bars with epidemic (R₀ > {config.r0_threshold}): "
          f"{(signals['r0'] > config.r0_threshold).sum()}")
    print(f"  Peak Φ: {signals['phi'].max():.3f}")
    print(f"  Peak R₀: {signals['r0'].max():.3f}")
    print(f"  Peak frac infected: {signals['frac_infected'].max():.3f}")

    # Show Φ around crisis
    print(f"\nΦ around crisis (bars 390-510):")
    crisis_window = signals.iloc[390:510]
    for i in range(0, len(crisis_window), 10):
        row = crisis_window.iloc[i]
        print(f"  Bar {390+i}: Φ={row['phi']:.3f}, R₀={row['r0']:.2f}, "
              f"dir={row['flock_dir']:.0f}, signal={row['signal']:.0f}")

    print("\nRunning backtest...")
    trades, equity_df = backtest_ohlcv(signals, basket[target], config)

    print_backtest_report(trades, equity_df)

    # Show individual trades
    if trades:
        print("\nTrade log:")
        for i, t in enumerate(trades[:20]):  # Show first 20
            print(f"  #{i+1}: {t.direction:+d} @ {t.entry_price:.2f} "
                  f"({t.entry_time}) → {t.exit_price:.2f} "
                  f"({t.exit_time}) | {100*t.pnl:+.3f}% | {t.exit_reason}")
