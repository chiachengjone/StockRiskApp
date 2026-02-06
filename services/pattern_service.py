"""
Pattern Recognition Service
============================
Candlestick and Chart Pattern Detection

Provides:
- Candlestick patterns (Doji, Hammer, Engulfing, Morning/Evening Star, etc.)
- Chart patterns (Double Top/Bottom, Head & Shoulders, Triangles, etc.)
- Pattern scanning across watchlists
- Pattern strength scoring

Author: Stock Risk App | Feb 2026
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class PatternType(Enum):
    """Types of patterns."""
    # Candlestick patterns
    DOJI = "Doji"
    HAMMER = "Hammer"
    INVERTED_HAMMER = "Inverted Hammer"
    HANGING_MAN = "Hanging Man"
    SHOOTING_STAR = "Shooting Star"
    BULLISH_ENGULFING = "Bullish Engulfing"
    BEARISH_ENGULFING = "Bearish Engulfing"
    MORNING_STAR = "Morning Star"
    EVENING_STAR = "Evening Star"
    THREE_WHITE_SOLDIERS = "Three White Soldiers"
    THREE_BLACK_CROWS = "Three Black Crows"
    PIERCING_LINE = "Piercing Line"
    DARK_CLOUD_COVER = "Dark Cloud Cover"
    TWEEZER_TOP = "Tweezer Top"
    TWEEZER_BOTTOM = "Tweezer Bottom"
    SPINNING_TOP = "Spinning Top"
    MARUBOZU = "Marubozu"
    
    # Chart patterns
    DOUBLE_TOP = "Double Top"
    DOUBLE_BOTTOM = "Double Bottom"
    TRIPLE_TOP = "Triple Top"
    TRIPLE_BOTTOM = "Triple Bottom"
    HEAD_AND_SHOULDERS = "Head and Shoulders"
    INVERSE_HEAD_AND_SHOULDERS = "Inverse Head and Shoulders"
    ASCENDING_TRIANGLE = "Ascending Triangle"
    DESCENDING_TRIANGLE = "Descending Triangle"
    SYMMETRICAL_TRIANGLE = "Symmetrical Triangle"
    RISING_WEDGE = "Rising Wedge"
    FALLING_WEDGE = "Falling Wedge"
    BULL_FLAG = "Bull Flag"
    BEAR_FLAG = "Bear Flag"
    CUP_AND_HANDLE = "Cup and Handle"
    ROUNDING_BOTTOM = "Rounding Bottom"


class PatternDirection(Enum):
    """Pattern directional bias."""
    BULLISH = "Bullish"
    BEARISH = "Bearish"
    NEUTRAL = "Neutral"


class PatternReliability(Enum):
    """Pattern reliability level."""
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


@dataclass
class Pattern:
    """Represents a detected pattern."""
    pattern_type: PatternType
    direction: PatternDirection
    start_idx: int
    end_idx: int
    start_date: datetime
    end_date: datetime
    price_at_detection: float
    confidence: float  # 0-100
    reliability: PatternReliability
    description: str
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert pattern to dictionary."""
        return {
            'pattern_type': self.pattern_type.value,
            'direction': self.direction.value,
            'start_date': self.start_date.isoformat() if hasattr(self.start_date, 'isoformat') else str(self.start_date),
            'end_date': self.end_date.isoformat() if hasattr(self.end_date, 'isoformat') else str(self.end_date),
            'price': self.price_at_detection,
            'confidence': self.confidence,
            'reliability': self.reliability.value,
            'description': self.description,
            'target_price': self.target_price,
            'stop_loss': self.stop_loss,
            'metadata': self.metadata
        }


# ============================================================================
# CANDLESTICK PATTERN DETECTION
# ============================================================================

def _body_size(open_price: float, close: float) -> float:
    """Calculate candle body size."""
    return abs(close - open_price)


def _upper_shadow(high: float, open_price: float, close: float) -> float:
    """Calculate upper shadow size."""
    return high - max(open_price, close)


def _lower_shadow(low: float, open_price: float, close: float) -> float:
    """Calculate lower shadow size."""
    return min(open_price, close) - low


def _is_bullish(open_price: float, close: float) -> bool:
    """Check if candle is bullish."""
    return close > open_price


def _candle_range(high: float, low: float) -> float:
    """Calculate total candle range."""
    return high - low


def detect_doji(
    open_price: float, high: float, low: float, close: float,
    threshold: float = 0.1
) -> bool:
    """
    Detect Doji pattern - open and close are nearly equal.
    
    Args:
        open_price, high, low, close: OHLC values
        threshold: Body/range ratio threshold
        
    Returns:
        True if Doji detected
    """
    candle_range = _candle_range(high, low)
    if candle_range == 0:
        return False
    
    body = _body_size(open_price, close)
    return (body / candle_range) < threshold


def detect_hammer(
    open_price: float, high: float, low: float, close: float,
    body_ratio: float = 0.3,
    shadow_ratio: float = 2.0
) -> bool:
    """
    Detect Hammer pattern - small body at top with long lower shadow.
    
    Args:
        open_price, high, low, close: OHLC values
        body_ratio: Max body/range ratio
        shadow_ratio: Min lower shadow/body ratio
        
    Returns:
        True if Hammer detected
    """
    candle_range = _candle_range(high, low)
    if candle_range == 0:
        return False
    
    body = _body_size(open_price, close)
    lower_shadow = _lower_shadow(low, open_price, close)
    upper_shadow = _upper_shadow(high, open_price, close)
    
    # Small body at top, long lower shadow, minimal upper shadow
    return (
        (body / candle_range) < body_ratio and
        body > 0 and
        (lower_shadow / body) >= shadow_ratio and
        upper_shadow < body
    )


def detect_inverted_hammer(
    open_price: float, high: float, low: float, close: float,
    body_ratio: float = 0.3,
    shadow_ratio: float = 2.0
) -> bool:
    """
    Detect Inverted Hammer - small body at bottom with long upper shadow.
    """
    candle_range = _candle_range(high, low)
    if candle_range == 0:
        return False
    
    body = _body_size(open_price, close)
    lower_shadow = _lower_shadow(low, open_price, close)
    upper_shadow = _upper_shadow(high, open_price, close)
    
    return (
        (body / candle_range) < body_ratio and
        body > 0 and
        (upper_shadow / body) >= shadow_ratio and
        lower_shadow < body
    )


def detect_shooting_star(
    prev_close: float,
    open_price: float, high: float, low: float, close: float,
    body_ratio: float = 0.3,
    shadow_ratio: float = 2.0
) -> bool:
    """
    Detect Shooting Star - inverted hammer after uptrend.
    """
    # Must be after uptrend (gap up or opening higher)
    if open_price <= prev_close:
        return False
    
    return detect_inverted_hammer(open_price, high, low, close, body_ratio, shadow_ratio)


def detect_hanging_man(
    prev_close: float,
    open_price: float, high: float, low: float, close: float,
    body_ratio: float = 0.3,
    shadow_ratio: float = 2.0
) -> bool:
    """
    Detect Hanging Man - hammer after uptrend (bearish signal).
    """
    # Must be after uptrend
    if open_price <= prev_close:
        return False
    
    return detect_hammer(open_price, high, low, close, body_ratio, shadow_ratio)


def detect_bullish_engulfing(
    prev_open: float, prev_close: float,
    open_price: float, close: float
) -> bool:
    """
    Detect Bullish Engulfing - bullish candle engulfs previous bearish candle.
    """
    # Previous candle must be bearish
    if prev_close >= prev_open:
        return False
    
    # Current candle must be bullish and engulf previous
    return (
        close > open_price and  # Current is bullish
        open_price < prev_close and  # Opens below previous close
        close > prev_open  # Closes above previous open
    )


def detect_bearish_engulfing(
    prev_open: float, prev_close: float,
    open_price: float, close: float
) -> bool:
    """
    Detect Bearish Engulfing - bearish candle engulfs previous bullish candle.
    """
    # Previous candle must be bullish
    if prev_close <= prev_open:
        return False
    
    # Current candle must be bearish and engulf previous
    return (
        close < open_price and  # Current is bearish
        open_price > prev_close and  # Opens above previous close
        close < prev_open  # Closes below previous open
    )


def detect_morning_star(
    candles: List[Tuple[float, float, float, float]],  # List of (O, H, L, C)
    body_threshold: float = 0.3
) -> bool:
    """
    Detect Morning Star - 3-candle bullish reversal pattern.
    
    1. Large bearish candle
    2. Small body (star) that gaps down
    3. Large bullish candle that closes above candle 1's midpoint
    """
    if len(candles) < 3:
        return False
    
    o1, h1, l1, c1 = candles[0]  # First candle
    o2, h2, l2, c2 = candles[1]  # Star
    o3, h3, l3, c3 = candles[2]  # Third candle
    
    range1 = h1 - l1
    range2 = h2 - l2
    range3 = h3 - l3
    
    if range1 == 0 or range3 == 0:
        return False
    
    body1 = _body_size(o1, c1)
    body2 = _body_size(o2, c2)
    body3 = _body_size(o3, c3)
    
    midpoint1 = (o1 + c1) / 2
    
    return (
        c1 < o1 and  # First is bearish
        body1 / range1 > 0.5 and  # Large body
        max(o2, c2) < c1 and  # Star gaps down
        body2 / range2 < body_threshold if range2 > 0 else True and  # Small star
        c3 > o3 and  # Third is bullish
        body3 / range3 > 0.5 and  # Large body
        c3 > midpoint1  # Closes above first candle's midpoint
    )


def detect_evening_star(
    candles: List[Tuple[float, float, float, float]],
    body_threshold: float = 0.3
) -> bool:
    """
    Detect Evening Star - 3-candle bearish reversal pattern.
    """
    if len(candles) < 3:
        return False
    
    o1, h1, l1, c1 = candles[0]
    o2, h2, l2, c2 = candles[1]
    o3, h3, l3, c3 = candles[2]
    
    range1 = h1 - l1
    range2 = h2 - l2
    range3 = h3 - l3
    
    if range1 == 0 or range3 == 0:
        return False
    
    body1 = _body_size(o1, c1)
    body2 = _body_size(o2, c2)
    body3 = _body_size(o3, c3)
    
    midpoint1 = (o1 + c1) / 2
    
    return (
        c1 > o1 and  # First is bullish
        body1 / range1 > 0.5 and  # Large body
        min(o2, c2) > c1 and  # Star gaps up
        body2 / range2 < body_threshold if range2 > 0 else True and  # Small star
        c3 < o3 and  # Third is bearish
        body3 / range3 > 0.5 and  # Large body
        c3 < midpoint1  # Closes below first candle's midpoint
    )


def detect_three_white_soldiers(
    candles: List[Tuple[float, float, float, float]],
    min_body_ratio: float = 0.6
) -> bool:
    """
    Detect Three White Soldiers - 3 consecutive bullish candles.
    """
    if len(candles) < 3:
        return False
    
    for i in range(3):
        o, h, l, c = candles[i]
        body = _body_size(o, c)
        candle_range = _candle_range(h, l)
        
        if candle_range == 0:
            return False
        
        # Each candle must be bullish with large body
        if c <= o or (body / candle_range) < min_body_ratio:
            return False
        
        # Each candle opens within previous body and closes higher
        if i > 0:
            prev_o, _, _, prev_c = candles[i-1]
            if not (prev_o < o < prev_c and c > prev_c):
                return False
    
    return True


def detect_three_black_crows(
    candles: List[Tuple[float, float, float, float]],
    min_body_ratio: float = 0.6
) -> bool:
    """
    Detect Three Black Crows - 3 consecutive bearish candles.
    """
    if len(candles) < 3:
        return False
    
    for i in range(3):
        o, h, l, c = candles[i]
        body = _body_size(o, c)
        candle_range = _candle_range(h, l)
        
        if candle_range == 0:
            return False
        
        # Each candle must be bearish with large body
        if c >= o or (body / candle_range) < min_body_ratio:
            return False
        
        # Each candle opens within previous body and closes lower
        if i > 0:
            prev_o, _, _, prev_c = candles[i-1]
            if not (prev_c < o < prev_o and c < prev_c):
                return False
    
    return True


def detect_marubozu(
    open_price: float, high: float, low: float, close: float,
    shadow_threshold: float = 0.05
) -> Tuple[bool, bool]:
    """
    Detect Marubozu - candle with no/minimal shadows.
    
    Returns:
        Tuple of (is_marubozu, is_bullish)
    """
    candle_range = _candle_range(high, low)
    if candle_range == 0:
        return False, False
    
    upper = _upper_shadow(high, open_price, close)
    lower = _lower_shadow(low, open_price, close)
    
    is_marubozu = (upper / candle_range < shadow_threshold and 
                   lower / candle_range < shadow_threshold)
    is_bullish = close > open_price
    
    return is_marubozu, is_bullish


def detect_spinning_top(
    open_price: float, high: float, low: float, close: float,
    body_threshold: float = 0.3,
    shadow_min: float = 0.2
) -> bool:
    """
    Detect Spinning Top - small body with shadows on both sides.
    """
    candle_range = _candle_range(high, low)
    if candle_range == 0:
        return False
    
    body = _body_size(open_price, close)
    upper = _upper_shadow(high, open_price, close)
    lower = _lower_shadow(low, open_price, close)
    
    return (
        (body / candle_range) < body_threshold and
        (upper / candle_range) > shadow_min and
        (lower / candle_range) > shadow_min
    )


def detect_tweezer_top(
    candle1: Tuple[float, float, float, float],
    candle2: Tuple[float, float, float, float],
    tolerance: float = 0.001
) -> bool:
    """
    Detect Tweezer Top - two candles with matching highs at resistance.
    """
    o1, h1, l1, c1 = candle1
    o2, h2, l2, c2 = candle2
    
    avg_high = (h1 + h2) / 2
    if avg_high == 0:
        return False
    
    # First bullish, second bearish
    # Highs are approximately equal
    return (
        c1 > o1 and  # First bullish
        c2 < o2 and  # Second bearish
        abs(h1 - h2) / avg_high < tolerance
    )


def detect_tweezer_bottom(
    candle1: Tuple[float, float, float, float],
    candle2: Tuple[float, float, float, float],
    tolerance: float = 0.001
) -> bool:
    """
    Detect Tweezer Bottom - two candles with matching lows at support.
    """
    o1, h1, l1, c1 = candle1
    o2, h2, l2, c2 = candle2
    
    avg_low = (l1 + l2) / 2
    if avg_low == 0:
        return False
    
    # First bearish, second bullish
    # Lows are approximately equal
    return (
        c1 < o1 and  # First bearish
        c2 > o2 and  # Second bullish
        abs(l1 - l2) / avg_low < tolerance
    )


def detect_piercing_line(
    candle1: Tuple[float, float, float, float],
    candle2: Tuple[float, float, float, float]
) -> bool:
    """
    Detect Piercing Line - bullish reversal pattern.
    """
    o1, h1, l1, c1 = candle1
    o2, h2, l2, c2 = candle2
    
    midpoint1 = (o1 + c1) / 2
    
    return (
        c1 < o1 and  # First is bearish
        o2 < c1 and  # Opens below first close
        c2 > o2 and  # Second is bullish
        c2 > midpoint1 and  # Closes above midpoint
        c2 < o1  # But not above first open
    )


def detect_dark_cloud_cover(
    candle1: Tuple[float, float, float, float],
    candle2: Tuple[float, float, float, float]
) -> bool:
    """
    Detect Dark Cloud Cover - bearish reversal pattern.
    """
    o1, h1, l1, c1 = candle1
    o2, h2, l2, c2 = candle2
    
    midpoint1 = (o1 + c1) / 2
    
    return (
        c1 > o1 and  # First is bullish
        o2 > c1 and  # Opens above first close
        c2 < o2 and  # Second is bearish
        c2 < midpoint1 and  # Closes below midpoint
        c2 > o1  # But not below first open
    )


# ============================================================================
# CHART PATTERN DETECTION
# ============================================================================

def find_local_extrema(
    prices: pd.Series,
    window: int = 5
) -> Tuple[pd.Series, pd.Series]:
    """
    Find local maxima (peaks) and minima (troughs) in price series.
    
    Args:
        prices: Price series
        window: Lookback/lookahead window for extrema detection
        
    Returns:
        Tuple of (peaks Series, troughs Series)
    """
    peaks = pd.Series(index=prices.index, dtype=float)
    troughs = pd.Series(index=prices.index, dtype=float)
    
    for i in range(window, len(prices) - window):
        # Check if local max
        if prices.iloc[i] == prices.iloc[i-window:i+window+1].max():
            peaks.iloc[i] = prices.iloc[i]
        
        # Check if local min
        if prices.iloc[i] == prices.iloc[i-window:i+window+1].min():
            troughs.iloc[i] = prices.iloc[i]
    
    return peaks.dropna(), troughs.dropna()


def detect_double_top(
    high: pd.Series,
    close: pd.Series,
    tolerance: float = 0.02,
    min_distance: int = 10
) -> List[Pattern]:
    """
    Detect Double Top pattern.
    
    Args:
        high: High price series
        close: Close price series
        tolerance: Price tolerance for matching peaks
        min_distance: Minimum bars between peaks
        
    Returns:
        List of detected Double Top patterns
    """
    patterns = []
    peaks, _ = find_local_extrema(high, window=5)
    
    if len(peaks) < 2:
        return patterns
    
    peak_indices = list(peaks.index)
    peak_values = list(peaks.values)
    
    for i in range(len(peak_indices) - 1):
        for j in range(i + 1, len(peak_indices)):
            idx1, val1 = peak_indices[i], peak_values[i]
            idx2, val2 = peak_indices[j], peak_values[j]
            
            # Get numeric positions
            pos1 = high.index.get_loc(idx1)
            pos2 = high.index.get_loc(idx2)
            
            # Check distance
            if pos2 - pos1 < min_distance:
                continue
            
            # Check if peaks are similar height
            avg_peak = (val1 + val2) / 2
            if abs(val1 - val2) / avg_peak > tolerance:
                continue
            
            # Find neckline (lowest point between peaks)
            between = close.iloc[pos1:pos2+1]
            neckline = between.min()
            neckline_idx = between.idxmin()
            
            # Check if current price broke below neckline
            current_price = close.iloc[-1]
            if current_price < neckline:
                # Pattern confirmed!
                target = neckline - (avg_peak - neckline)  # Measured move
                
                patterns.append(Pattern(
                    pattern_type=PatternType.DOUBLE_TOP,
                    direction=PatternDirection.BEARISH,
                    start_idx=pos1,
                    end_idx=pos2,
                    start_date=idx1,
                    end_date=idx2,
                    price_at_detection=current_price,
                    confidence=75.0,
                    reliability=PatternReliability.HIGH,
                    description=f"Double Top at {avg_peak:.2f}, neckline {neckline:.2f}",
                    target_price=target,
                    stop_loss=avg_peak * 1.02,
                    metadata={
                        'peak1': val1,
                        'peak2': val2,
                        'neckline': neckline,
                        'neckline_date': str(neckline_idx)
                    }
                ))
    
    return patterns


def detect_double_bottom(
    low: pd.Series,
    close: pd.Series,
    tolerance: float = 0.02,
    min_distance: int = 10
) -> List[Pattern]:
    """
    Detect Double Bottom pattern.
    """
    patterns = []
    _, troughs = find_local_extrema(low, window=5)
    
    if len(troughs) < 2:
        return patterns
    
    trough_indices = list(troughs.index)
    trough_values = list(troughs.values)
    
    for i in range(len(trough_indices) - 1):
        for j in range(i + 1, len(trough_indices)):
            idx1, val1 = trough_indices[i], trough_values[i]
            idx2, val2 = trough_indices[j], trough_values[j]
            
            pos1 = low.index.get_loc(idx1)
            pos2 = low.index.get_loc(idx2)
            
            if pos2 - pos1 < min_distance:
                continue
            
            avg_trough = (val1 + val2) / 2
            if abs(val1 - val2) / avg_trough > tolerance:
                continue
            
            # Find neckline (highest point between troughs)
            between = close.iloc[pos1:pos2+1]
            neckline = between.max()
            neckline_idx = between.idxmax()
            
            current_price = close.iloc[-1]
            if current_price > neckline:
                target = neckline + (neckline - avg_trough)
                
                patterns.append(Pattern(
                    pattern_type=PatternType.DOUBLE_BOTTOM,
                    direction=PatternDirection.BULLISH,
                    start_idx=pos1,
                    end_idx=pos2,
                    start_date=idx1,
                    end_date=idx2,
                    price_at_detection=current_price,
                    confidence=75.0,
                    reliability=PatternReliability.HIGH,
                    description=f"Double Bottom at {avg_trough:.2f}, neckline {neckline:.2f}",
                    target_price=target,
                    stop_loss=avg_trough * 0.98,
                    metadata={
                        'trough1': val1,
                        'trough2': val2,
                        'neckline': neckline
                    }
                ))
    
    return patterns


def detect_head_and_shoulders(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    tolerance: float = 0.03
) -> List[Pattern]:
    """
    Detect Head and Shoulders pattern.
    """
    patterns = []
    peaks, _ = find_local_extrema(high, window=5)
    
    if len(peaks) < 3:
        return patterns
    
    peak_indices = list(peaks.index)
    peak_values = list(peaks.values)
    
    # Look for 3 peaks where middle is higher
    for i in range(len(peak_indices) - 2):
        left_shoulder_idx, left_shoulder = peak_indices[i], peak_values[i]
        head_idx, head = peak_indices[i+1], peak_values[i+1]
        right_shoulder_idx, right_shoulder = peak_indices[i+2], peak_values[i+2]
        
        # Head must be higher than shoulders
        if head <= left_shoulder or head <= right_shoulder:
            continue
        
        # Shoulders should be roughly equal
        avg_shoulder = (left_shoulder + right_shoulder) / 2
        if abs(left_shoulder - right_shoulder) / avg_shoulder > tolerance:
            continue
        
        # Get positions
        pos_ls = high.index.get_loc(left_shoulder_idx)
        pos_h = high.index.get_loc(head_idx)
        pos_rs = high.index.get_loc(right_shoulder_idx)
        
        # Find neckline troughs
        trough1 = low.iloc[pos_ls:pos_h+1].min()
        trough2 = low.iloc[pos_h:pos_rs+1].min()
        neckline = (trough1 + trough2) / 2
        
        current_price = close.iloc[-1]
        if current_price < neckline:
            target = neckline - (head - neckline)
            
            patterns.append(Pattern(
                pattern_type=PatternType.HEAD_AND_SHOULDERS,
                direction=PatternDirection.BEARISH,
                start_idx=pos_ls,
                end_idx=pos_rs,
                start_date=left_shoulder_idx,
                end_date=right_shoulder_idx,
                price_at_detection=current_price,
                confidence=80.0,
                reliability=PatternReliability.HIGH,
                description=f"H&S: Head {head:.2f}, Shoulders ~{avg_shoulder:.2f}",
                target_price=target,
                stop_loss=right_shoulder * 1.02,
                metadata={
                    'left_shoulder': left_shoulder,
                    'head': head,
                    'right_shoulder': right_shoulder,
                    'neckline': neckline
                }
            ))
    
    return patterns


def detect_inverse_head_and_shoulders(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    tolerance: float = 0.03
) -> List[Pattern]:
    """
    Detect Inverse Head and Shoulders pattern.
    """
    patterns = []
    _, troughs = find_local_extrema(low, window=5)
    
    if len(troughs) < 3:
        return patterns
    
    trough_indices = list(troughs.index)
    trough_values = list(troughs.values)
    
    for i in range(len(trough_indices) - 2):
        left_shoulder_idx, left_shoulder = trough_indices[i], trough_values[i]
        head_idx, head = trough_indices[i+1], trough_values[i+1]
        right_shoulder_idx, right_shoulder = trough_indices[i+2], trough_values[i+2]
        
        # Head must be lower than shoulders
        if head >= left_shoulder or head >= right_shoulder:
            continue
        
        avg_shoulder = (left_shoulder + right_shoulder) / 2
        if abs(left_shoulder - right_shoulder) / avg_shoulder > tolerance:
            continue
        
        pos_ls = low.index.get_loc(left_shoulder_idx)
        pos_h = low.index.get_loc(head_idx)
        pos_rs = low.index.get_loc(right_shoulder_idx)
        
        peak1 = high.iloc[pos_ls:pos_h+1].max()
        peak2 = high.iloc[pos_h:pos_rs+1].max()
        neckline = (peak1 + peak2) / 2
        
        current_price = close.iloc[-1]
        if current_price > neckline:
            target = neckline + (neckline - head)
            
            patterns.append(Pattern(
                pattern_type=PatternType.INVERSE_HEAD_AND_SHOULDERS,
                direction=PatternDirection.BULLISH,
                start_idx=pos_ls,
                end_idx=pos_rs,
                start_date=left_shoulder_idx,
                end_date=right_shoulder_idx,
                price_at_detection=current_price,
                confidence=80.0,
                reliability=PatternReliability.HIGH,
                description=f"Inverse H&S: Head {head:.2f}, Shoulders ~{avg_shoulder:.2f}",
                target_price=target,
                stop_loss=right_shoulder * 0.98,
                metadata={
                    'left_shoulder': left_shoulder,
                    'head': head,
                    'right_shoulder': right_shoulder,
                    'neckline': neckline
                }
            ))
    
    return patterns


def detect_triangle(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    min_points: int = 4
) -> List[Pattern]:
    """
    Detect Triangle patterns (Ascending, Descending, Symmetrical).
    """
    patterns = []
    
    peaks, troughs = find_local_extrema(high, window=5)
    _, low_troughs = find_local_extrema(low, window=5)
    
    if len(peaks) < 2 or len(troughs) < 2:
        return patterns
    
    # Analyze trend of highs and lows
    peak_values = peaks.values
    trough_values = low_troughs.values
    
    if len(peak_values) >= 2 and len(trough_values) >= 2:
        # High trend
        high_slope = (peak_values[-1] - peak_values[0]) / len(peak_values)
        # Low trend
        low_slope = (trough_values[-1] - trough_values[0]) / len(trough_values)
        
        current_price = close.iloc[-1]
        
        # Ascending Triangle: flat resistance, rising support
        if abs(high_slope) < 0.001 and low_slope > 0.001:
            resistance = peak_values.mean()
            support = trough_values[-1]
            
            if current_price > resistance * 0.98:
                patterns.append(Pattern(
                    pattern_type=PatternType.ASCENDING_TRIANGLE,
                    direction=PatternDirection.BULLISH,
                    start_idx=len(close) - len(peak_values) * 5,
                    end_idx=len(close) - 1,
                    start_date=peaks.index[0],
                    end_date=close.index[-1],
                    price_at_detection=current_price,
                    confidence=70.0,
                    reliability=PatternReliability.MEDIUM,
                    description=f"Ascending Triangle, Resistance ~{resistance:.2f}",
                    target_price=resistance + (resistance - support),
                    stop_loss=support * 0.98,
                    metadata={'resistance': resistance, 'support': support}
                ))
        
        # Descending Triangle: falling highs, flat support
        elif high_slope < -0.001 and abs(low_slope) < 0.001:
            resistance = peak_values[-1]
            support = trough_values.mean()
            
            if current_price < support * 1.02:
                patterns.append(Pattern(
                    pattern_type=PatternType.DESCENDING_TRIANGLE,
                    direction=PatternDirection.BEARISH,
                    start_idx=len(close) - len(peak_values) * 5,
                    end_idx=len(close) - 1,
                    start_date=peaks.index[0],
                    end_date=close.index[-1],
                    price_at_detection=current_price,
                    confidence=70.0,
                    reliability=PatternReliability.MEDIUM,
                    description=f"Descending Triangle, Support ~{support:.2f}",
                    target_price=support - (resistance - support),
                    stop_loss=resistance * 1.02,
                    metadata={'resistance': resistance, 'support': support}
                ))
        
        # Symmetrical Triangle: converging highs and lows
        elif high_slope < -0.0005 and low_slope > 0.0005:
            avg_price = (peak_values[-1] + trough_values[-1]) / 2
            triangle_height = peak_values[0] - trough_values[0]
            
            patterns.append(Pattern(
                pattern_type=PatternType.SYMMETRICAL_TRIANGLE,
                direction=PatternDirection.NEUTRAL,
                start_idx=len(close) - len(peak_values) * 5,
                end_idx=len(close) - 1,
                start_date=peaks.index[0] if len(peaks) > 0 else close.index[0],
                end_date=close.index[-1],
                price_at_detection=current_price,
                confidence=65.0,
                reliability=PatternReliability.MEDIUM,
                description="Symmetrical Triangle (breakout pending)",
                target_price=current_price + triangle_height if current_price > avg_price else current_price - triangle_height,
                metadata={'apex_price': avg_price, 'height': triangle_height}
            ))
    
    return patterns


# ============================================================================
# PATTERN SERVICE CLASS
# ============================================================================

class PatternService:
    """
    Pattern Recognition Service.
    
    Provides comprehensive candlestick and chart pattern detection.
    """
    
    def __init__(self):
        """Initialize the Pattern Service."""
        self._pattern_cache: Dict[str, List[Pattern]] = {}
        logger.info("PatternService initialized")
    
    def detect_candlestick_patterns(
        self,
        df: pd.DataFrame,
        lookback: int = 50
    ) -> List[Pattern]:
        """
        Detect candlestick patterns in recent price data.
        
        Args:
            df: OHLCV DataFrame
            lookback: Number of recent bars to analyze
            
        Returns:
            List of detected candlestick patterns
        """
        patterns = []
        
        # Handle DataFrame structure
        if isinstance(df.columns, pd.MultiIndex):
            open_prices = df['Open'].iloc[:, 0]
            high = df['High'].iloc[:, 0]
            low = df['Low'].iloc[:, 0]
            close = df['Close'].iloc[:, 0]
        else:
            open_prices = df['Open']
            high = df['High']
            low = df['Low']
            close = df['Close']
        
        # Analyze recent candles
        start_idx = max(0, len(df) - lookback)
        
        for i in range(start_idx + 3, len(df)):
            o, h, l, c = open_prices.iloc[i], high.iloc[i], low.iloc[i], close.iloc[i]
            prev_o, prev_h, prev_l, prev_c = (
                open_prices.iloc[i-1], high.iloc[i-1], low.iloc[i-1], close.iloc[i-1]
            )
            
            date = df.index[i]
            
            # Single candle patterns
            if detect_doji(o, h, l, c):
                patterns.append(Pattern(
                    pattern_type=PatternType.DOJI,
                    direction=PatternDirection.NEUTRAL,
                    start_idx=i, end_idx=i,
                    start_date=date, end_date=date,
                    price_at_detection=c,
                    confidence=60.0,
                    reliability=PatternReliability.MEDIUM,
                    description="Doji - Indecision candle, potential reversal"
                ))
            
            if detect_hammer(o, h, l, c):
                patterns.append(Pattern(
                    pattern_type=PatternType.HAMMER,
                    direction=PatternDirection.BULLISH,
                    start_idx=i, end_idx=i,
                    start_date=date, end_date=date,
                    price_at_detection=c,
                    confidence=65.0,
                    reliability=PatternReliability.MEDIUM,
                    description="Hammer - Bullish reversal signal"
                ))
            
            if detect_inverted_hammer(o, h, l, c):
                patterns.append(Pattern(
                    pattern_type=PatternType.INVERTED_HAMMER,
                    direction=PatternDirection.BULLISH,
                    start_idx=i, end_idx=i,
                    start_date=date, end_date=date,
                    price_at_detection=c,
                    confidence=60.0,
                    reliability=PatternReliability.MEDIUM,
                    description="Inverted Hammer - Potential bullish reversal"
                ))
            
            if detect_shooting_star(prev_c, o, h, l, c):
                patterns.append(Pattern(
                    pattern_type=PatternType.SHOOTING_STAR,
                    direction=PatternDirection.BEARISH,
                    start_idx=i, end_idx=i,
                    start_date=date, end_date=date,
                    price_at_detection=c,
                    confidence=65.0,
                    reliability=PatternReliability.MEDIUM,
                    description="Shooting Star - Bearish reversal after uptrend"
                ))
            
            if detect_hanging_man(prev_c, o, h, l, c):
                patterns.append(Pattern(
                    pattern_type=PatternType.HANGING_MAN,
                    direction=PatternDirection.BEARISH,
                    start_idx=i, end_idx=i,
                    start_date=date, end_date=date,
                    price_at_detection=c,
                    confidence=60.0,
                    reliability=PatternReliability.MEDIUM,
                    description="Hanging Man - Warning of potential top"
                ))
            
            is_marubozu, is_bullish = detect_marubozu(o, h, l, c)
            if is_marubozu:
                direction = PatternDirection.BULLISH if is_bullish else PatternDirection.BEARISH
                patterns.append(Pattern(
                    pattern_type=PatternType.MARUBOZU,
                    direction=direction,
                    start_idx=i, end_idx=i,
                    start_date=date, end_date=date,
                    price_at_detection=c,
                    confidence=70.0,
                    reliability=PatternReliability.MEDIUM,
                    description=f"{'Bullish' if is_bullish else 'Bearish'} Marubozu - Strong conviction"
                ))
            
            if detect_spinning_top(o, h, l, c):
                patterns.append(Pattern(
                    pattern_type=PatternType.SPINNING_TOP,
                    direction=PatternDirection.NEUTRAL,
                    start_idx=i, end_idx=i,
                    start_date=date, end_date=date,
                    price_at_detection=c,
                    confidence=55.0,
                    reliability=PatternReliability.LOW,
                    description="Spinning Top - Market indecision"
                ))
            
            # Two-candle patterns
            if detect_bullish_engulfing(prev_o, prev_c, o, c):
                patterns.append(Pattern(
                    pattern_type=PatternType.BULLISH_ENGULFING,
                    direction=PatternDirection.BULLISH,
                    start_idx=i-1, end_idx=i,
                    start_date=df.index[i-1], end_date=date,
                    price_at_detection=c,
                    confidence=75.0,
                    reliability=PatternReliability.HIGH,
                    description="Bullish Engulfing - Strong reversal signal"
                ))
            
            if detect_bearish_engulfing(prev_o, prev_c, o, c):
                patterns.append(Pattern(
                    pattern_type=PatternType.BEARISH_ENGULFING,
                    direction=PatternDirection.BEARISH,
                    start_idx=i-1, end_idx=i,
                    start_date=df.index[i-1], end_date=date,
                    price_at_detection=c,
                    confidence=75.0,
                    reliability=PatternReliability.HIGH,
                    description="Bearish Engulfing - Strong reversal signal"
                ))
            
            candle1 = (prev_o, prev_h, prev_l, prev_c)
            candle2 = (o, h, l, c)
            
            if detect_tweezer_top(candle1, candle2):
                patterns.append(Pattern(
                    pattern_type=PatternType.TWEEZER_TOP,
                    direction=PatternDirection.BEARISH,
                    start_idx=i-1, end_idx=i,
                    start_date=df.index[i-1], end_date=date,
                    price_at_detection=c,
                    confidence=65.0,
                    reliability=PatternReliability.MEDIUM,
                    description="Tweezer Top - Potential reversal at resistance"
                ))
            
            if detect_tweezer_bottom(candle1, candle2):
                patterns.append(Pattern(
                    pattern_type=PatternType.TWEEZER_BOTTOM,
                    direction=PatternDirection.BULLISH,
                    start_idx=i-1, end_idx=i,
                    start_date=df.index[i-1], end_date=date,
                    price_at_detection=c,
                    confidence=65.0,
                    reliability=PatternReliability.MEDIUM,
                    description="Tweezer Bottom - Potential reversal at support"
                ))
            
            if detect_piercing_line(candle1, candle2):
                patterns.append(Pattern(
                    pattern_type=PatternType.PIERCING_LINE,
                    direction=PatternDirection.BULLISH,
                    start_idx=i-1, end_idx=i,
                    start_date=df.index[i-1], end_date=date,
                    price_at_detection=c,
                    confidence=70.0,
                    reliability=PatternReliability.MEDIUM,
                    description="Piercing Line - Bullish reversal pattern"
                ))
            
            if detect_dark_cloud_cover(candle1, candle2):
                patterns.append(Pattern(
                    pattern_type=PatternType.DARK_CLOUD_COVER,
                    direction=PatternDirection.BEARISH,
                    start_idx=i-1, end_idx=i,
                    start_date=df.index[i-1], end_date=date,
                    price_at_detection=c,
                    confidence=70.0,
                    reliability=PatternReliability.MEDIUM,
                    description="Dark Cloud Cover - Bearish reversal pattern"
                ))
            
            # Three-candle patterns
            if i >= 2:
                candles = [
                    (open_prices.iloc[i-2], high.iloc[i-2], low.iloc[i-2], close.iloc[i-2]),
                    (open_prices.iloc[i-1], high.iloc[i-1], low.iloc[i-1], close.iloc[i-1]),
                    (o, h, l, c)
                ]
                
                if detect_morning_star(candles):
                    patterns.append(Pattern(
                        pattern_type=PatternType.MORNING_STAR,
                        direction=PatternDirection.BULLISH,
                        start_idx=i-2, end_idx=i,
                        start_date=df.index[i-2], end_date=date,
                        price_at_detection=c,
                        confidence=80.0,
                        reliability=PatternReliability.HIGH,
                        description="Morning Star - Strong bullish reversal"
                    ))
                
                if detect_evening_star(candles):
                    patterns.append(Pattern(
                        pattern_type=PatternType.EVENING_STAR,
                        direction=PatternDirection.BEARISH,
                        start_idx=i-2, end_idx=i,
                        start_date=df.index[i-2], end_date=date,
                        price_at_detection=c,
                        confidence=80.0,
                        reliability=PatternReliability.HIGH,
                        description="Evening Star - Strong bearish reversal"
                    ))
                
                if detect_three_white_soldiers(candles):
                    patterns.append(Pattern(
                        pattern_type=PatternType.THREE_WHITE_SOLDIERS,
                        direction=PatternDirection.BULLISH,
                        start_idx=i-2, end_idx=i,
                        start_date=df.index[i-2], end_date=date,
                        price_at_detection=c,
                        confidence=75.0,
                        reliability=PatternReliability.HIGH,
                        description="Three White Soldiers - Strong bullish continuation"
                    ))
                
                if detect_three_black_crows(candles):
                    patterns.append(Pattern(
                        pattern_type=PatternType.THREE_BLACK_CROWS,
                        direction=PatternDirection.BEARISH,
                        start_idx=i-2, end_idx=i,
                        start_date=df.index[i-2], end_date=date,
                        price_at_detection=c,
                        confidence=75.0,
                        reliability=PatternReliability.HIGH,
                        description="Three Black Crows - Strong bearish continuation"
                    ))
        
        return patterns
    
    def detect_chart_patterns(
        self,
        df: pd.DataFrame
    ) -> List[Pattern]:
        """
        Detect chart patterns in price data.
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            List of detected chart patterns
        """
        patterns = []
        
        # Handle DataFrame structure
        if isinstance(df.columns, pd.MultiIndex):
            high = df['High'].iloc[:, 0]
            low = df['Low'].iloc[:, 0]
            close = df['Close'].iloc[:, 0]
        else:
            high = df['High']
            low = df['Low']
            close = df['Close']
        
        if len(df) < 30:
            return patterns
        
        try:
            # Detect major chart patterns
            patterns.extend(detect_double_top(high, close))
            patterns.extend(detect_double_bottom(low, close))
            patterns.extend(detect_head_and_shoulders(high, low, close))
            patterns.extend(detect_inverse_head_and_shoulders(high, low, close))
            patterns.extend(detect_triangle(high, low, close))
        except Exception as e:
            logger.error(f"Error detecting chart patterns: {e}")
        
        return patterns
    
    def detect_all_patterns(
        self,
        df: pd.DataFrame,
        include_candlestick: bool = True,
        include_chart: bool = True,
        min_confidence: float = 0.0
    ) -> List[Pattern]:
        """
        Detect all patterns in price data.
        
        Args:
            df: OHLCV DataFrame
            include_candlestick: Include candlestick patterns
            include_chart: Include chart patterns
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of all detected patterns
        """
        all_patterns = []
        
        if include_candlestick:
            all_patterns.extend(self.detect_candlestick_patterns(df))
        
        if include_chart:
            all_patterns.extend(self.detect_chart_patterns(df))
        
        # Filter by confidence
        if min_confidence > 0:
            all_patterns = [p for p in all_patterns if p.confidence >= min_confidence]
        
        # Sort by date (most recent first)
        all_patterns.sort(key=lambda p: p.end_date, reverse=True)
        
        return all_patterns
    
    def get_recent_patterns(
        self,
        df: pd.DataFrame,
        days: int = 5
    ) -> List[Pattern]:
        """
        Get patterns detected in the last N days.
        
        Args:
            df: OHLCV DataFrame
            days: Number of recent days to check
            
        Returns:
            List of recent patterns
        """
        all_patterns = self.detect_all_patterns(df)
        
        if len(df) == 0:
            return []
        
        cutoff_idx = max(0, len(df) - days)
        recent = [p for p in all_patterns if p.end_idx >= cutoff_idx]
        
        return recent
    
    def get_pattern_summary(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Get a summary of pattern detection results.
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            Dictionary with pattern summary
        """
        patterns = self.detect_all_patterns(df)
        recent = self.get_recent_patterns(df, days=5)
        
        bullish = [p for p in patterns if p.direction == PatternDirection.BULLISH]
        bearish = [p for p in patterns if p.direction == PatternDirection.BEARISH]
        
        recent_bullish = [p for p in recent if p.direction == PatternDirection.BULLISH]
        recent_bearish = [p for p in recent if p.direction == PatternDirection.BEARISH]
        
        # Calculate bias score
        if recent:
            bullish_weight = sum(p.confidence for p in recent_bullish)
            bearish_weight = sum(p.confidence for p in recent_bearish)
            total_weight = bullish_weight + bearish_weight
            
            bias_score = ((bullish_weight - bearish_weight) / total_weight * 50 + 50) if total_weight > 0 else 50
        else:
            bias_score = 50
        
        return {
            'total_patterns': len(patterns),
            'recent_patterns': len(recent),
            'bullish_count': len(bullish),
            'bearish_count': len(bearish),
            'recent_bullish': len(recent_bullish),
            'recent_bearish': len(recent_bearish),
            'bias_score': bias_score,  # 0-100, >50 bullish, <50 bearish
            'recent_patterns_list': [p.to_dict() for p in recent[:10]],
            'pattern_types': list(set(p.pattern_type.value for p in patterns))
        }
    
    def clear_cache(self) -> None:
        """Clear the pattern cache."""
        self._pattern_cache.clear()
        logger.info("PatternService cache cleared")
