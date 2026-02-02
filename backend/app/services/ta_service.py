"""
Technical Analysis Service
==========================
Technical indicator calculations.

Ported from services/ta_service.py with identical calculations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# MOVING AVERAGES
# ============================================================================

def calculate_sma(prices: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average."""
    if len(prices) < period:
        return pd.Series(index=prices.index, dtype=float)
    return prices.rolling(window=period, min_periods=period).mean()


def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    if len(prices) < period:
        return pd.Series(index=prices.index, dtype=float)
    return prices.ewm(span=period, adjust=False, min_periods=period).mean()


# ============================================================================
# MOMENTUM INDICATORS
# ============================================================================

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index.
    
    RSI = 100 - (100 / (1 + RS))
    RS = Average Gain / Average Loss
    """
    if len(prices) < period + 1:
        return pd.Series(index=prices.index, dtype=float)
    
    delta = prices.diff()
    gains = delta.where(delta > 0, 0.0)
    losses = (-delta).where(delta < 0, 0.0)
    
    avg_gains = gains.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_losses = losses.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))
    
    rsi = rsi.replace([np.inf, -np.inf], 100)
    rsi = rsi.fillna(50)
    
    return rsi


def calculate_macd(
    prices: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Returns (MACD line, Signal line, Histogram)
    """
    min_data = slow_period + signal_period
    if len(prices) < min_data:
        empty = pd.Series(index=prices.index, dtype=float)
        return empty, empty.copy(), empty.copy()
    
    ema_fast = prices.ewm(span=fast_period, adjust=False, min_periods=fast_period).mean()
    ema_slow = prices.ewm(span=slow_period, adjust=False, min_periods=slow_period).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False, min_periods=signal_period).mean()
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


# ============================================================================
# VOLATILITY INDICATORS
# ============================================================================

def calculate_bollinger_bands(
    prices: pd.Series,
    period: int = 20,
    num_std: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands.
    
    Returns (Upper band, Middle band, Lower band)
    """
    if len(prices) < period:
        empty = pd.Series(index=prices.index, dtype=float)
        return empty, empty.copy(), empty.copy()
    
    middle_band = prices.rolling(window=period, min_periods=period).mean()
    rolling_std = prices.rolling(window=period, min_periods=period).std()
    
    upper_band = middle_band + (rolling_std * num_std)
    lower_band = middle_band - (rolling_std * num_std)
    
    return upper_band, middle_band, lower_band


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """
    Calculate Average True Range (ATR).
    """
    if len(close) < period + 1:
        return pd.Series(index=close.index, dtype=float)
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    return atr


# ============================================================================
# TREND INDICATORS
# ============================================================================

def calculate_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Average Directional Index (ADX) with +DI and -DI.
    
    Returns (ADX, +DI, -DI)
    """
    min_data = period * 2
    if len(close) < min_data:
        empty = pd.Series(index=close.index, dtype=float)
        return empty, empty.copy(), empty.copy()
    
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    plus_dm = pd.Series(0.0, index=close.index)
    minus_dm = pd.Series(0.0, index=close.index)
    
    plus_dm[(up_move > down_move) & (up_move > 0)] = up_move[(up_move > down_move) & (up_move > 0)]
    minus_dm[(down_move > up_move) & (down_move > 0)] = down_move[(down_move > up_move) & (down_move > 0)]
    
    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = dx.replace([np.inf, -np.inf], 0).fillna(0)
    
    adx = dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    return adx, plus_di, minus_di


def calculate_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Stochastic Oscillator (%K and %D).
    
    Returns (%K, %D)
    """
    if len(close) < k_period + d_period:
        empty = pd.Series(index=close.index, dtype=float)
        return empty, empty.copy()
    
    lowest_low = low.rolling(window=k_period, min_periods=k_period).min()
    highest_high = high.rolling(window=k_period, min_periods=k_period).max()
    
    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    stoch_k = stoch_k.replace([np.inf, -np.inf], 50).fillna(50)
    
    stoch_d = stoch_k.rolling(window=d_period, min_periods=d_period).mean()
    
    return stoch_k, stoch_d


# ============================================================================
# COMPREHENSIVE INDICATOR FUNCTION
# ============================================================================

def get_all_indicators(
    df: pd.DataFrame,
    sma_periods: List[int] = None,
    ema_periods: List[int] = None
) -> Dict[str, pd.Series]:
    """
    Calculate all technical indicators.
    """
    if sma_periods is None:
        sma_periods = [9, 21, 50]
    if ema_periods is None:
        ema_periods = [12, 26]
    
    # Handle column extraction
    close = df['Close']
    high = df['High'] if 'High' in df.columns else close
    low = df['Low'] if 'Low' in df.columns else close
    
    indicators = {}
    
    # SMAs
    for period in sma_periods:
        indicators[f'SMA_{period}'] = calculate_sma(close, period)
    
    # EMAs
    for period in ema_periods:
        indicators[f'EMA_{period}'] = calculate_ema(close, period)
    
    # RSI
    indicators['RSI'] = calculate_rsi(close)
    
    # MACD
    macd_line, signal_line, histogram = calculate_macd(close)
    indicators['MACD'] = macd_line
    indicators['MACD_Signal'] = signal_line
    indicators['MACD_Histogram'] = histogram
    
    # Bollinger Bands
    upper, middle, lower = calculate_bollinger_bands(close)
    indicators['BB_Upper'] = upper
    indicators['BB_Middle'] = middle
    indicators['BB_Lower'] = lower
    
    # ADX
    adx, plus_di, minus_di = calculate_adx(high, low, close)
    indicators['ADX'] = adx
    indicators['Plus_DI'] = plus_di
    indicators['Minus_DI'] = minus_di
    
    # Stochastic
    stoch_k, stoch_d = calculate_stochastic(high, low, close)
    indicators['Stoch_K'] = stoch_k
    indicators['Stoch_D'] = stoch_d
    
    # ATR
    indicators['ATR'] = calculate_atr(high, low, close)
    
    return indicators
