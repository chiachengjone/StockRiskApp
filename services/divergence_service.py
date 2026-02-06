"""
Divergence Detection Service
=============================
Detect price-indicator divergences

Provides:
- RSI divergence detection
- MACD divergence detection  
- Volume divergence detection
- OBV divergence detection
- Divergence strength scoring

Author: Stock Risk App | Feb 2026
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging

from .ta_service import calculate_rsi, calculate_macd
from .pattern_service import find_local_extrema

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class DivergenceType(Enum):
    """Types of divergence."""
    REGULAR_BULLISH = "Regular Bullish"  # Price lower low, indicator higher low
    REGULAR_BEARISH = "Regular Bearish"  # Price higher high, indicator lower high
    HIDDEN_BULLISH = "Hidden Bullish"    # Price higher low, indicator lower low
    HIDDEN_BEARISH = "Hidden Bearish"    # Price lower high, indicator higher high


class DivergenceIndicator(Enum):
    """Indicator used for divergence detection."""
    RSI = "RSI"
    MACD = "MACD"
    MACD_HISTOGRAM = "MACD Histogram"
    VOLUME = "Volume"
    OBV = "OBV"


class DivergenceStrength(Enum):
    """Divergence strength level."""
    STRONG = "Strong"
    MODERATE = "Moderate"
    WEAK = "Weak"


@dataclass
class Divergence:
    """Represents a detected divergence."""
    divergence_type: DivergenceType
    indicator: DivergenceIndicator
    strength: DivergenceStrength
    start_idx: int
    end_idx: int
    start_date: datetime
    end_date: datetime
    price_start: float
    price_end: float
    indicator_start: float
    indicator_end: float
    confidence: float  # 0-100
    description: str
    expected_move: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'divergence_type': self.divergence_type.value,
            'indicator': self.indicator.value,
            'strength': self.strength.value,
            'start_date': self.start_date.isoformat() if hasattr(self.start_date, 'isoformat') else str(self.start_date),
            'end_date': self.end_date.isoformat() if hasattr(self.end_date, 'isoformat') else str(self.end_date),
            'price_start': self.price_start,
            'price_end': self.price_end,
            'indicator_start': self.indicator_start,
            'indicator_end': self.indicator_end,
            'confidence': self.confidence,
            'description': self.description,
            'expected_move': self.expected_move,
            'metadata': self.metadata
        }


# ============================================================================
# DIVERGENCE DETECTION FUNCTIONS
# ============================================================================

def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculate On-Balance Volume (OBV).
    
    Args:
        close: Close price series
        volume: Volume series
        
    Returns:
        OBV series
    """
    obv = pd.Series(index=close.index, dtype=float)
    obv.iloc[0] = volume.iloc[0]
    
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    return obv


def detect_regular_bullish_divergence(
    price: pd.Series,
    indicator: pd.Series,
    price_troughs: pd.Series,
    indicator_window: int = 3
) -> List[Tuple[int, int, float, float, float, float]]:
    """
    Detect regular bullish divergence.
    Price makes lower low, indicator makes higher low.
    
    Returns:
        List of (start_idx, end_idx, price_start, price_end, ind_start, ind_end)
    """
    divergences = []
    
    trough_indices = list(price_troughs.index)
    
    for i in range(len(trough_indices) - 1):
        for j in range(i + 1, min(i + 6, len(trough_indices))):  # Look within 5 troughs
            idx1, idx2 = trough_indices[i], trough_indices[j]
            
            pos1 = price.index.get_loc(idx1)
            pos2 = price.index.get_loc(idx2)
            
            # Price lower low
            if price.iloc[pos2] >= price.iloc[pos1]:
                continue
            
            # Get indicator values (average around trough)
            ind_slice1 = indicator.iloc[max(0, pos1-indicator_window):pos1+indicator_window+1]
            ind_slice2 = indicator.iloc[max(0, pos2-indicator_window):pos2+indicator_window+1]
            
            if len(ind_slice1) == 0 or len(ind_slice2) == 0:
                continue
            
            ind_val1 = ind_slice1.min()
            ind_val2 = ind_slice2.min()
            
            # Indicator higher low (bullish divergence)
            if ind_val2 > ind_val1:
                divergences.append((
                    pos1, pos2,
                    float(price.iloc[pos1]), float(price.iloc[pos2]),
                    float(ind_val1), float(ind_val2)
                ))
    
    return divergences


def detect_regular_bearish_divergence(
    price: pd.Series,
    indicator: pd.Series,
    price_peaks: pd.Series,
    indicator_window: int = 3
) -> List[Tuple[int, int, float, float, float, float]]:
    """
    Detect regular bearish divergence.
    Price makes higher high, indicator makes lower high.
    """
    divergences = []
    
    peak_indices = list(price_peaks.index)
    
    for i in range(len(peak_indices) - 1):
        for j in range(i + 1, min(i + 6, len(peak_indices))):
            idx1, idx2 = peak_indices[i], peak_indices[j]
            
            pos1 = price.index.get_loc(idx1)
            pos2 = price.index.get_loc(idx2)
            
            # Price higher high
            if price.iloc[pos2] <= price.iloc[pos1]:
                continue
            
            # Get indicator values
            ind_slice1 = indicator.iloc[max(0, pos1-indicator_window):pos1+indicator_window+1]
            ind_slice2 = indicator.iloc[max(0, pos2-indicator_window):pos2+indicator_window+1]
            
            if len(ind_slice1) == 0 or len(ind_slice2) == 0:
                continue
            
            ind_val1 = ind_slice1.max()
            ind_val2 = ind_slice2.max()
            
            # Indicator lower high (bearish divergence)
            if ind_val2 < ind_val1:
                divergences.append((
                    pos1, pos2,
                    float(price.iloc[pos1]), float(price.iloc[pos2]),
                    float(ind_val1), float(ind_val2)
                ))
    
    return divergences


def detect_hidden_bullish_divergence(
    price: pd.Series,
    indicator: pd.Series,
    price_troughs: pd.Series,
    indicator_window: int = 3
) -> List[Tuple[int, int, float, float, float, float]]:
    """
    Detect hidden bullish divergence.
    Price makes higher low, indicator makes lower low.
    Signals trend continuation.
    """
    divergences = []
    
    trough_indices = list(price_troughs.index)
    
    for i in range(len(trough_indices) - 1):
        for j in range(i + 1, min(i + 6, len(trough_indices))):
            idx1, idx2 = trough_indices[i], trough_indices[j]
            
            pos1 = price.index.get_loc(idx1)
            pos2 = price.index.get_loc(idx2)
            
            # Price higher low
            if price.iloc[pos2] <= price.iloc[pos1]:
                continue
            
            ind_slice1 = indicator.iloc[max(0, pos1-indicator_window):pos1+indicator_window+1]
            ind_slice2 = indicator.iloc[max(0, pos2-indicator_window):pos2+indicator_window+1]
            
            if len(ind_slice1) == 0 or len(ind_slice2) == 0:
                continue
            
            ind_val1 = ind_slice1.min()
            ind_val2 = ind_slice2.min()
            
            # Indicator lower low (hidden bullish)
            if ind_val2 < ind_val1:
                divergences.append((
                    pos1, pos2,
                    float(price.iloc[pos1]), float(price.iloc[pos2]),
                    float(ind_val1), float(ind_val2)
                ))
    
    return divergences


def detect_hidden_bearish_divergence(
    price: pd.Series,
    indicator: pd.Series,
    price_peaks: pd.Series,
    indicator_window: int = 3
) -> List[Tuple[int, int, float, float, float, float]]:
    """
    Detect hidden bearish divergence.
    Price makes lower high, indicator makes higher high.
    """
    divergences = []
    
    peak_indices = list(price_peaks.index)
    
    for i in range(len(peak_indices) - 1):
        for j in range(i + 1, min(i + 6, len(peak_indices))):
            idx1, idx2 = peak_indices[i], peak_indices[j]
            
            pos1 = price.index.get_loc(idx1)
            pos2 = price.index.get_loc(idx2)
            
            # Price lower high
            if price.iloc[pos2] >= price.iloc[pos1]:
                continue
            
            ind_slice1 = indicator.iloc[max(0, pos1-indicator_window):pos1+indicator_window+1]
            ind_slice2 = indicator.iloc[max(0, pos2-indicator_window):pos2+indicator_window+1]
            
            if len(ind_slice1) == 0 or len(ind_slice2) == 0:
                continue
            
            ind_val1 = ind_slice1.max()
            ind_val2 = ind_slice2.max()
            
            # Indicator higher high (hidden bearish)
            if ind_val2 > ind_val1:
                divergences.append((
                    pos1, pos2,
                    float(price.iloc[pos1]), float(price.iloc[pos2]),
                    float(ind_val1), float(ind_val2)
                ))
    
    return divergences


def calculate_divergence_strength(
    price_change_pct: float,
    indicator_change_pct: float
) -> DivergenceStrength:
    """
    Calculate divergence strength based on magnitude.
    """
    total_divergence = abs(price_change_pct - indicator_change_pct)
    
    if total_divergence > 15:
        return DivergenceStrength.STRONG
    elif total_divergence > 7:
        return DivergenceStrength.MODERATE
    else:
        return DivergenceStrength.WEAK


# ============================================================================
# DIVERGENCE SERVICE CLASS
# ============================================================================

class DivergenceService:
    """
    Divergence Detection Service.
    
    Provides comprehensive divergence analysis for trading signals.
    """
    
    def __init__(self):
        """Initialize the Divergence Service."""
        self._cache: Dict[str, List[Divergence]] = {}
        logger.info("DivergenceService initialized")
    
    def detect_rsi_divergence(
        self,
        df: pd.DataFrame,
        rsi_period: int = 14,
        lookback: int = 100
    ) -> List[Divergence]:
        """
        Detect RSI divergences.
        
        Args:
            df: OHLCV DataFrame
            rsi_period: RSI calculation period
            lookback: Bars to look back
            
        Returns:
            List of detected divergences
        """
        divergences = []
        
        # Handle DataFrame structure
        if isinstance(df.columns, pd.MultiIndex):
            close = df['Close'].iloc[:, 0]
        else:
            close = df['Close']
        
        if len(close) < lookback:
            lookback = len(close)
        
        close = close.iloc[-lookback:]
        rsi = calculate_rsi(close, rsi_period)
        
        # Find extrema
        price_peaks, price_troughs = find_local_extrema(close, window=5)
        
        # Detect all divergence types
        # Regular Bullish
        for start, end, p1, p2, i1, i2 in detect_regular_bullish_divergence(close, rsi, price_troughs):
            price_change = (p2 - p1) / p1 * 100
            ind_change = (i2 - i1) / max(i1, 1) * 100
            strength = calculate_divergence_strength(price_change, ind_change)
            
            divergences.append(Divergence(
                divergence_type=DivergenceType.REGULAR_BULLISH,
                indicator=DivergenceIndicator.RSI,
                strength=strength,
                start_idx=start,
                end_idx=end,
                start_date=close.index[start],
                end_date=close.index[end],
                price_start=p1,
                price_end=p2,
                indicator_start=i1,
                indicator_end=i2,
                confidence=70 + (10 if strength == DivergenceStrength.STRONG else 0),
                description=f"RSI bullish divergence: Price lower low ({p1:.2f}→{p2:.2f}), RSI higher low ({i1:.1f}→{i2:.1f})",
                expected_move="Bullish reversal expected"
            ))
        
        # Regular Bearish
        for start, end, p1, p2, i1, i2 in detect_regular_bearish_divergence(close, rsi, price_peaks):
            price_change = (p2 - p1) / p1 * 100
            ind_change = (i2 - i1) / max(i1, 1) * 100
            strength = calculate_divergence_strength(price_change, ind_change)
            
            divergences.append(Divergence(
                divergence_type=DivergenceType.REGULAR_BEARISH,
                indicator=DivergenceIndicator.RSI,
                strength=strength,
                start_idx=start,
                end_idx=end,
                start_date=close.index[start],
                end_date=close.index[end],
                price_start=p1,
                price_end=p2,
                indicator_start=i1,
                indicator_end=i2,
                confidence=70 + (10 if strength == DivergenceStrength.STRONG else 0),
                description=f"RSI bearish divergence: Price higher high ({p1:.2f}→{p2:.2f}), RSI lower high ({i1:.1f}→{i2:.1f})",
                expected_move="Bearish reversal expected"
            ))
        
        # Hidden Bullish
        for start, end, p1, p2, i1, i2 in detect_hidden_bullish_divergence(close, rsi, price_troughs):
            price_change = (p2 - p1) / p1 * 100
            ind_change = (i2 - i1) / max(i1, 1) * 100
            strength = calculate_divergence_strength(price_change, ind_change)
            
            divergences.append(Divergence(
                divergence_type=DivergenceType.HIDDEN_BULLISH,
                indicator=DivergenceIndicator.RSI,
                strength=strength,
                start_idx=start,
                end_idx=end,
                start_date=close.index[start],
                end_date=close.index[end],
                price_start=p1,
                price_end=p2,
                indicator_start=i1,
                indicator_end=i2,
                confidence=65 + (10 if strength == DivergenceStrength.STRONG else 0),
                description=f"Hidden bullish divergence: Price higher low, RSI lower low",
                expected_move="Trend continuation (bullish)"
            ))
        
        # Hidden Bearish
        for start, end, p1, p2, i1, i2 in detect_hidden_bearish_divergence(close, rsi, price_peaks):
            price_change = (p2 - p1) / p1 * 100
            ind_change = (i2 - i1) / max(i1, 1) * 100
            strength = calculate_divergence_strength(price_change, ind_change)
            
            divergences.append(Divergence(
                divergence_type=DivergenceType.HIDDEN_BEARISH,
                indicator=DivergenceIndicator.RSI,
                strength=strength,
                start_idx=start,
                end_idx=end,
                start_date=close.index[start],
                end_date=close.index[end],
                price_start=p1,
                price_end=p2,
                indicator_start=i1,
                indicator_end=i2,
                confidence=65 + (10 if strength == DivergenceStrength.STRONG else 0),
                description=f"Hidden bearish divergence: Price lower high, RSI higher high",
                expected_move="Trend continuation (bearish)"
            ))
        
        return divergences
    
    def detect_macd_divergence(
        self,
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        use_histogram: bool = True,
        lookback: int = 100
    ) -> List[Divergence]:
        """
        Detect MACD divergences.
        
        Args:
            df: OHLCV DataFrame
            fast, slow, signal: MACD parameters
            use_histogram: Use histogram instead of MACD line
            lookback: Bars to look back
            
        Returns:
            List of detected divergences
        """
        divergences = []
        
        if isinstance(df.columns, pd.MultiIndex):
            close = df['Close'].iloc[:, 0]
        else:
            close = df['Close']
        
        if len(close) < lookback:
            lookback = len(close)
        
        close = close.iloc[-lookback:]
        macd_line, signal_line, histogram = calculate_macd(close, fast, slow, signal)
        
        indicator = histogram if use_histogram else macd_line
        indicator_type = DivergenceIndicator.MACD_HISTOGRAM if use_histogram else DivergenceIndicator.MACD
        
        price_peaks, price_troughs = find_local_extrema(close, window=5)
        
        # Regular Bullish
        for start, end, p1, p2, i1, i2 in detect_regular_bullish_divergence(close, indicator, price_troughs):
            strength = DivergenceStrength.STRONG if abs(i2 - i1) > 0.5 else DivergenceStrength.MODERATE
            
            divergences.append(Divergence(
                divergence_type=DivergenceType.REGULAR_BULLISH,
                indicator=indicator_type,
                strength=strength,
                start_idx=start,
                end_idx=end,
                start_date=close.index[start],
                end_date=close.index[end],
                price_start=p1,
                price_end=p2,
                indicator_start=i1,
                indicator_end=i2,
                confidence=72,
                description=f"MACD bullish divergence detected",
                expected_move="Bullish reversal expected"
            ))
        
        # Regular Bearish
        for start, end, p1, p2, i1, i2 in detect_regular_bearish_divergence(close, indicator, price_peaks):
            strength = DivergenceStrength.STRONG if abs(i2 - i1) > 0.5 else DivergenceStrength.MODERATE
            
            divergences.append(Divergence(
                divergence_type=DivergenceType.REGULAR_BEARISH,
                indicator=indicator_type,
                strength=strength,
                start_idx=start,
                end_idx=end,
                start_date=close.index[start],
                end_date=close.index[end],
                price_start=p1,
                price_end=p2,
                indicator_start=i1,
                indicator_end=i2,
                confidence=72,
                description=f"MACD bearish divergence detected",
                expected_move="Bearish reversal expected"
            ))
        
        return divergences
    
    def detect_volume_divergence(
        self,
        df: pd.DataFrame,
        lookback: int = 100
    ) -> List[Divergence]:
        """
        Detect volume divergences (price vs OBV).
        
        Args:
            df: OHLCV DataFrame
            lookback: Bars to look back
            
        Returns:
            List of detected divergences
        """
        divergences = []
        
        if isinstance(df.columns, pd.MultiIndex):
            close = df['Close'].iloc[:, 0]
            volume = df['Volume'].iloc[:, 0]
        else:
            close = df['Close']
            volume = df['Volume']
        
        if len(close) < lookback:
            lookback = len(close)
        
        close = close.iloc[-lookback:]
        volume = volume.iloc[-lookback:]
        
        obv = calculate_obv(close, volume)
        
        price_peaks, price_troughs = find_local_extrema(close, window=5)
        
        # Regular Bullish (price lower, OBV higher)
        for start, end, p1, p2, i1, i2 in detect_regular_bullish_divergence(close, obv, price_troughs):
            divergences.append(Divergence(
                divergence_type=DivergenceType.REGULAR_BULLISH,
                indicator=DivergenceIndicator.OBV,
                strength=DivergenceStrength.MODERATE,
                start_idx=start,
                end_idx=end,
                start_date=close.index[start],
                end_date=close.index[end],
                price_start=p1,
                price_end=p2,
                indicator_start=i1,
                indicator_end=i2,
                confidence=65,
                description="Volume accumulation detected with price weakness",
                expected_move="Bullish - smart money accumulating"
            ))
        
        # Regular Bearish (price higher, OBV lower)
        for start, end, p1, p2, i1, i2 in detect_regular_bearish_divergence(close, obv, price_peaks):
            divergences.append(Divergence(
                divergence_type=DivergenceType.REGULAR_BEARISH,
                indicator=DivergenceIndicator.OBV,
                strength=DivergenceStrength.MODERATE,
                start_idx=start,
                end_idx=end,
                start_date=close.index[start],
                end_date=close.index[end],
                price_start=p1,
                price_end=p2,
                indicator_start=i1,
                indicator_end=i2,
                confidence=65,
                description="Volume distribution detected with price strength",
                expected_move="Bearish - smart money distributing"
            ))
        
        return divergences
    
    def detect_all_divergences(
        self,
        df: pd.DataFrame,
        lookback: int = 100
    ) -> List[Divergence]:
        """
        Detect all types of divergences.
        
        Args:
            df: OHLCV DataFrame
            lookback: Bars to look back
            
        Returns:
            List of all detected divergences
        """
        all_divergences = []
        
        all_divergences.extend(self.detect_rsi_divergence(df, lookback=lookback))
        all_divergences.extend(self.detect_macd_divergence(df, lookback=lookback))
        all_divergences.extend(self.detect_volume_divergence(df, lookback=lookback))
        
        # Sort by end date (most recent first)
        all_divergences.sort(key=lambda d: d.end_idx, reverse=True)
        
        return all_divergences
    
    def get_recent_divergences(
        self,
        df: pd.DataFrame,
        days: int = 10
    ) -> List[Divergence]:
        """
        Get divergences detected in recent bars.
        
        Args:
            df: OHLCV DataFrame
            days: Number of recent days to check
            
        Returns:
            List of recent divergences
        """
        all_divergences = self.detect_all_divergences(df)
        
        if len(df) == 0:
            return []
        
        cutoff_idx = len(df) - days
        recent = [d for d in all_divergences if d.end_idx >= cutoff_idx]
        
        return recent
    
    def get_divergence_summary(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Get a summary of divergence analysis.
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            Dictionary with divergence summary
        """
        all_divergences = self.detect_all_divergences(df)
        recent = self.get_recent_divergences(df, days=10)
        
        bullish_types = [DivergenceType.REGULAR_BULLISH, DivergenceType.HIDDEN_BULLISH]
        bearish_types = [DivergenceType.REGULAR_BEARISH, DivergenceType.HIDDEN_BEARISH]
        
        bullish = [d for d in all_divergences if d.divergence_type in bullish_types]
        bearish = [d for d in all_divergences if d.divergence_type in bearish_types]
        
        recent_bullish = [d for d in recent if d.divergence_type in bullish_types]
        recent_bearish = [d for d in recent if d.divergence_type in bearish_types]
        
        # Calculate bias
        if recent:
            bullish_weight = sum(d.confidence for d in recent_bullish)
            bearish_weight = sum(d.confidence for d in recent_bearish)
            total_weight = bullish_weight + bearish_weight
            bias_score = ((bullish_weight - bearish_weight) / total_weight * 50 + 50) if total_weight > 0 else 50
        else:
            bias_score = 50
        
        return {
            'total_divergences': len(all_divergences),
            'recent_divergences': len(recent),
            'bullish_count': len(bullish),
            'bearish_count': len(bearish),
            'recent_bullish': len(recent_bullish),
            'recent_bearish': len(recent_bearish),
            'bias_score': bias_score,
            'recent_list': [d.to_dict() for d in recent[:10]],
            'strongest_recent': recent[0].to_dict() if recent else None,
            'indicators_with_divergence': list(set(d.indicator.value for d in recent))
        }
    
    def clear_cache(self) -> None:
        """Clear the divergence cache."""
        self._cache.clear()
        logger.info("DivergenceService cache cleared")
