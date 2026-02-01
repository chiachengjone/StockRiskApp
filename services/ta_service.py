"""
Technical Analysis Service
==========================
Comprehensive Technical Indicators Library

Provides:
- SMA, EMA (Multiple periods)
- RSI, MACD
- Bollinger Bands
- ADX, Stochastic, ATR

Author: Stock Risk App | Feb 2026
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================

# Moving Average periods
SMA_PERIODS = [9, 21, 50]
EMA_PERIODS = [12, 26]

# RSI
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# MACD
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Bollinger Bands
BB_PERIOD = 20
BB_STD = 2

# ADX
ADX_PERIOD = 14

# Stochastic
STOCH_K_PERIOD = 14
STOCH_D_PERIOD = 3

# ATR
ATR_PERIOD = 14


# ============================================================================
# INDIVIDUAL INDICATOR FUNCTIONS
# ============================================================================

def calculate_sma(
    prices: pd.Series,
    period: int
) -> pd.Series:
    """
    Calculate Simple Moving Average.
    
    Args:
        prices: Price series (typically Close prices)
        period: Number of periods for the moving average
        
    Returns:
        Series with SMA values
    """
    if len(prices) < period:
        logger.warning(f"Insufficient data for SMA({period})")
        return pd.Series(index=prices.index, dtype=float)
    
    return prices.rolling(window=period, min_periods=period).mean()


def calculate_ema(
    prices: pd.Series,
    period: int
) -> pd.Series:
    """
    Calculate Exponential Moving Average.
    
    Args:
        prices: Price series (typically Close prices)
        period: Number of periods for the moving average
        
    Returns:
        Series with EMA values
    """
    if len(prices) < period:
        logger.warning(f"Insufficient data for EMA({period})")
        return pd.Series(index=prices.index, dtype=float)
    
    return prices.ewm(span=period, adjust=False, min_periods=period).mean()


def calculate_rsi(
    prices: pd.Series,
    period: int = RSI_PERIOD
) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    
    RSI measures momentum by comparing the magnitude of recent gains
    to recent losses.
    
    Args:
        prices: Price series (typically Close prices)
        period: Lookback period (default: 14)
        
    Returns:
        Series with RSI values (0-100)
    """
    if len(prices) < period + 1:
        logger.warning(f"Insufficient data for RSI({period})")
        return pd.Series(index=prices.index, dtype=float)
    
    # Calculate price changes
    delta = prices.diff()
    
    # Separate gains and losses
    gains = delta.where(delta > 0, 0.0)
    losses = (-delta).where(delta < 0, 0.0)
    
    # Calculate average gains and losses using Wilder's smoothing (EMA)
    avg_gains = gains.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_losses = losses.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    # Calculate RS and RSI
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))
    
    # Handle division by zero (when avg_losses is 0)
    rsi = rsi.replace([np.inf, -np.inf], 100)
    rsi = rsi.fillna(50)  # Neutral RSI when undefined
    
    return rsi


def calculate_macd(
    prices: pd.Series,
    fast_period: int = MACD_FAST,
    slow_period: int = MACD_SLOW,
    signal_period: int = MACD_SIGNAL
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        prices: Price series (typically Close prices)
        fast_period: Fast EMA period (default: 12)
        slow_period: Slow EMA period (default: 26)
        signal_period: Signal line period (default: 9)
        
    Returns:
        Tuple of (MACD line, Signal line, Histogram)
    """
    min_data = slow_period + signal_period
    if len(prices) < min_data:
        logger.warning(f"Insufficient data for MACD")
        empty = pd.Series(index=prices.index, dtype=float)
        return empty, empty.copy(), empty.copy()
    
    # Calculate fast and slow EMAs
    ema_fast = prices.ewm(span=fast_period, adjust=False, min_periods=fast_period).mean()
    ema_slow = prices.ewm(span=slow_period, adjust=False, min_periods=slow_period).mean()
    
    # MACD line
    macd_line = ema_fast - ema_slow
    
    # Signal line (EMA of MACD)
    signal_line = macd_line.ewm(span=signal_period, adjust=False, min_periods=signal_period).mean()
    
    # Histogram
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_bollinger_bands(
    prices: pd.Series,
    period: int = BB_PERIOD,
    num_std: float = BB_STD
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands.
    
    Args:
        prices: Price series (typically Close prices)
        period: Moving average period (default: 20)
        num_std: Number of standard deviations (default: 2)
        
    Returns:
        Tuple of (Upper band, Middle band/SMA, Lower band)
    """
    if len(prices) < period:
        logger.warning(f"Insufficient data for Bollinger Bands")
        empty = pd.Series(index=prices.index, dtype=float)
        return empty, empty.copy(), empty.copy()
    
    # Middle band (SMA)
    middle_band = prices.rolling(window=period, min_periods=period).mean()
    
    # Standard deviation
    rolling_std = prices.rolling(window=period, min_periods=period).std()
    
    # Upper and lower bands
    upper_band = middle_band + (rolling_std * num_std)
    lower_band = middle_band - (rolling_std * num_std)
    
    return upper_band, middle_band, lower_band


def calculate_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = ADX_PERIOD
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Average Directional Index (ADX) with +DI and -DI.
    
    ADX measures trend strength regardless of direction.
    +DI measures upward trend strength, -DI measures downward trend strength.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        period: Lookback period (default: 14)
        
    Returns:
        Tuple of (ADX, +DI, -DI)
    """
    min_data = period * 2
    if len(close) < min_data:
        logger.warning(f"Insufficient data for ADX({period})")
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
    
    # +DM and -DM
    plus_dm = pd.Series(0.0, index=close.index)
    minus_dm = pd.Series(0.0, index=close.index)
    
    plus_dm[(up_move > down_move) & (up_move > 0)] = up_move[(up_move > down_move) & (up_move > 0)]
    minus_dm[(down_move > up_move) & (down_move > 0)] = down_move[(down_move > up_move) & (down_move > 0)]
    
    # Smoothed averages using Wilder's method
    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr)
    
    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = dx.replace([np.inf, -np.inf], 0).fillna(0)
    
    adx = dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    return adx, plus_di, minus_di


def calculate_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = STOCH_K_PERIOD,
    d_period: int = STOCH_D_PERIOD
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Stochastic Oscillator (%K and %D).
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        k_period: %K lookback period (default: 14)
        d_period: %D smoothing period (default: 3)
        
    Returns:
        Tuple of (%K, %D)
    """
    if len(close) < k_period + d_period:
        logger.warning(f"Insufficient data for Stochastic")
        empty = pd.Series(index=close.index, dtype=float)
        return empty, empty.copy()
    
    # Calculate %K
    lowest_low = low.rolling(window=k_period, min_periods=k_period).min()
    highest_high = high.rolling(window=k_period, min_periods=k_period).max()
    
    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    stoch_k = stoch_k.replace([np.inf, -np.inf], 50).fillna(50)
    
    # Calculate %D (SMA of %K)
    stoch_d = stoch_k.rolling(window=d_period, min_periods=d_period).mean()
    
    return stoch_k, stoch_d


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = ATR_PERIOD
) -> pd.Series:
    """
    Calculate Average True Range (ATR).
    
    ATR measures market volatility by decomposing the entire range
    of an asset price for that period.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        period: Lookback period (default: 14)
        
    Returns:
        Series with ATR values
    """
    if len(close) < period + 1:
        logger.warning(f"Insufficient data for ATR({period})")
        return pd.Series(index=close.index, dtype=float)
    
    # True Range components
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    # True Range is the maximum
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # ATR using Wilder's smoothing method
    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    return atr


def get_all_indicators(
    df: pd.DataFrame,
    sma_periods: List[int] = None,
    ema_periods: List[int] = None
) -> Dict[str, Any]:
    """
    Calculate all technical indicators for a DataFrame.
    
    Args:
        df: DataFrame with OHLCV data (columns: Open, High, Low, Close, Volume)
        sma_periods: List of SMA periods (default: [9, 21, 50])
        ema_periods: List of EMA periods (default: [12, 26])
        
    Returns:
        Dictionary containing all indicator series
    """
    if sma_periods is None:
        sma_periods = SMA_PERIODS
    if ema_periods is None:
        ema_periods = EMA_PERIODS
    
    # Handle both single and multi-column DataFrames
    if isinstance(df.columns, pd.MultiIndex):
        # Multi-ticker DataFrame - extract first ticker
        close = df['Close'].iloc[:, 0] if isinstance(df['Close'], pd.DataFrame) else df['Close']
        high = df['High'].iloc[:, 0] if isinstance(df['High'], pd.DataFrame) else df['High']
        low = df['Low'].iloc[:, 0] if isinstance(df['Low'], pd.DataFrame) else df['Low']
        volume = df['Volume'].iloc[:, 0] if isinstance(df['Volume'], pd.DataFrame) else df['Volume']
    else:
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume'] if 'Volume' in df.columns else pd.Series(index=df.index, dtype=float)
    
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
    macd, signal, histogram = calculate_macd(close)
    indicators['MACD'] = macd
    indicators['MACD_Signal'] = signal
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
    
    # Volume
    indicators['Volume'] = volume
    
    # Close price for reference
    indicators['Close'] = close
    
    return indicators


# ============================================================================
# TA SERVICE CLASS
# ============================================================================

@dataclass
class IndicatorConfig:
    """Configuration for technical indicators."""
    sma_periods: List[int]
    ema_periods: List[int]
    rsi_period: int
    macd_fast: int
    macd_slow: int
    macd_signal: int
    bb_period: int
    bb_std: float
    adx_period: int
    stoch_k_period: int
    stoch_d_period: int
    atr_period: int


class TAService:
    """
    Technical Analysis Service.
    
    Provides comprehensive technical indicator calculations
    and analysis for stock data.
    """
    
    def __init__(self, config: Optional[IndicatorConfig] = None):
        """
        Initialize the TA Service.
        
        Args:
            config: Optional indicator configuration. Uses defaults if not provided.
        """
        if config is None:
            self.config = IndicatorConfig(
                sma_periods=SMA_PERIODS,
                ema_periods=EMA_PERIODS,
                rsi_period=RSI_PERIOD,
                macd_fast=MACD_FAST,
                macd_slow=MACD_SLOW,
                macd_signal=MACD_SIGNAL,
                bb_period=BB_PERIOD,
                bb_std=BB_STD,
                adx_period=ADX_PERIOD,
                stoch_k_period=STOCH_K_PERIOD,
                stoch_d_period=STOCH_D_PERIOD,
                atr_period=ATR_PERIOD
            )
        else:
            self.config = config
        
        self._indicators_cache: Dict[str, Any] = {}
        logger.info("TAService initialized")
    
    def calculate_all(
        self,
        df: pd.DataFrame,
        use_cache: bool = True
    ) -> Dict[str, pd.Series]:
        """
        Calculate all technical indicators for a DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            use_cache: Whether to use cached results
            
        Returns:
            Dictionary of indicator name -> Series
        """
        cache_key = f"{id(df)}_{len(df)}"
        
        if use_cache and cache_key in self._indicators_cache:
            logger.debug("Returning cached indicators")
            return self._indicators_cache[cache_key]
        
        try:
            indicators = get_all_indicators(
                df,
                sma_periods=self.config.sma_periods,
                ema_periods=self.config.ema_periods
            )
            
            if use_cache:
                self._indicators_cache[cache_key] = indicators
            
            logger.info(f"Calculated {len(indicators)} indicators")
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            raise
    
    def get_indicator(
        self,
        df: pd.DataFrame,
        indicator_name: str
    ) -> Optional[pd.Series]:
        """
        Get a specific indicator.
        
        Args:
            df: DataFrame with OHLCV data
            indicator_name: Name of the indicator (e.g., 'RSI', 'SMA_50')
            
        Returns:
            Indicator series or None if not found
        """
        indicators = self.calculate_all(df)
        return indicators.get(indicator_name)
    
    def get_trend_indicators(
        self,
        df: pd.DataFrame
    ) -> Dict[str, pd.Series]:
        """
        Get trend-following indicators (MAs, ADX).
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary of trend indicators
        """
        all_indicators = self.calculate_all(df)
        trend_keys = [k for k in all_indicators.keys() 
                     if k.startswith(('SMA', 'EMA', 'ADX', 'Plus_DI', 'Minus_DI'))]
        return {k: all_indicators[k] for k in trend_keys}
    
    def get_momentum_indicators(
        self,
        df: pd.DataFrame
    ) -> Dict[str, pd.Series]:
        """
        Get momentum indicators (RSI, MACD, Stochastic).
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary of momentum indicators
        """
        all_indicators = self.calculate_all(df)
        momentum_keys = ['RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram', 
                        'Stoch_K', 'Stoch_D']
        return {k: all_indicators[k] for k in momentum_keys if k in all_indicators}
    
    def get_volatility_indicators(
        self,
        df: pd.DataFrame
    ) -> Dict[str, pd.Series]:
        """
        Get volatility indicators (BB, ATR).
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary of volatility indicators
        """
        all_indicators = self.calculate_all(df)
        volatility_keys = ['BB_Upper', 'BB_Middle', 'BB_Lower', 'ATR']
        return {k: all_indicators[k] for k in volatility_keys if k in all_indicators}
    
    def get_indicator_summary(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Get a summary of current indicator values and signals.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary with current values and interpretations
        """
        indicators = self.calculate_all(df)
        
        # Get latest values
        latest = {}
        for key, series in indicators.items():
            if len(series.dropna()) > 0:
                latest[key] = float(series.dropna().iloc[-1])
            else:
                latest[key] = None
        
        # Build summary
        summary = {
            'current_price': latest.get('Close'),
            'indicators': latest,
            'interpretations': {}
        }
        
        # RSI interpretation
        rsi_val = latest.get('RSI')
        if rsi_val is not None:
            if rsi_val >= RSI_OVERBOUGHT:
                summary['interpretations']['RSI'] = 'Overbought'
            elif rsi_val <= RSI_OVERSOLD:
                summary['interpretations']['RSI'] = 'Oversold'
            else:
                summary['interpretations']['RSI'] = 'Neutral'
        
        # MACD interpretation
        macd_val = latest.get('MACD')
        signal_val = latest.get('MACD_Signal')
        if macd_val is not None and signal_val is not None:
            if macd_val > signal_val:
                summary['interpretations']['MACD'] = 'Bullish'
            else:
                summary['interpretations']['MACD'] = 'Bearish'
        
        # ADX interpretation
        adx_val = latest.get('ADX')
        if adx_val is not None:
            if adx_val >= 25:
                summary['interpretations']['ADX'] = 'Strong Trend'
            else:
                summary['interpretations']['ADX'] = 'Weak/No Trend'
        
        # Bollinger Bands interpretation
        close_val = latest.get('Close')
        bb_upper = latest.get('BB_Upper')
        bb_lower = latest.get('BB_Lower')
        if all(v is not None for v in [close_val, bb_upper, bb_lower]):
            if close_val >= bb_upper:
                summary['interpretations']['BB'] = 'Upper Band - Potential Resistance'
            elif close_val <= bb_lower:
                summary['interpretations']['BB'] = 'Lower Band - Potential Support'
            else:
                summary['interpretations']['BB'] = 'Within Bands'
        
        return summary
    
    def clear_cache(self) -> None:
        """Clear the indicators cache."""
        self._indicators_cache.clear()
        logger.info("TAService cache cleared")
