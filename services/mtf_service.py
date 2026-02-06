"""
Multi-Timeframe Analysis Service
=================================
Analyze signals across multiple timeframes

Provides:
- Multi-timeframe indicator analysis
- Timeframe alignment detection
- MTF signal confirmation
- Trend strength scoring

Author: Stock Risk App | Feb 2026
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import yfinance as yf
import logging

from .ta_service import (
    calculate_sma,
    calculate_ema,
    calculate_rsi,
    calculate_macd,
    calculate_adx
)

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

# Timeframe definitions
TIMEFRAMES = {
    '15m': {'period': '5d', 'interval': '15m', 'weight': 0.1, 'name': '15 Minutes'},
    '1h': {'period': '1mo', 'interval': '60m', 'weight': 0.15, 'name': '1 Hour'},
    '4h': {'period': '2mo', 'interval': '60m', 'weight': 0.20, 'name': '4 Hours'},  # Approximated
    '1d': {'period': '1y', 'interval': '1d', 'weight': 0.30, 'name': 'Daily'},
    '1w': {'period': '2y', 'interval': '1wk', 'weight': 0.25, 'name': 'Weekly'},
}


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class TrendDirection(Enum):
    """Trend direction."""
    STRONG_UP = "Strong Uptrend"
    UP = "Uptrend"
    NEUTRAL = "Neutral"
    DOWN = "Downtrend"
    STRONG_DOWN = "Strong Downtrend"


class TimeframeAlignment(Enum):
    """MTF alignment status."""
    FULLY_ALIGNED_BULLISH = "Fully Aligned Bullish"
    MOSTLY_ALIGNED_BULLISH = "Mostly Aligned Bullish"
    MIXED = "Mixed"
    MOSTLY_ALIGNED_BEARISH = "Mostly Aligned Bearish"
    FULLY_ALIGNED_BEARISH = "Fully Aligned Bearish"


@dataclass
class TimeframeAnalysis:
    """Analysis result for a single timeframe."""
    timeframe: str
    trend: TrendDirection
    rsi: float
    macd_histogram: float
    price_vs_sma20: float  # % above/below
    price_vs_sma50: float
    adx: float
    score: float  # -100 to 100
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timeframe': self.timeframe,
            'trend': self.trend.value,
            'rsi': self.rsi,
            'macd_histogram': self.macd_histogram,
            'price_vs_sma20': self.price_vs_sma20,
            'price_vs_sma50': self.price_vs_sma50,
            'adx': self.adx,
            'score': self.score,
            'timestamp': self.timestamp.isoformat() if hasattr(self.timestamp, 'isoformat') else str(self.timestamp)
        }


@dataclass
class MTFAnalysisResult:
    """Multi-timeframe analysis result."""
    symbol: str
    timeframe_analyses: Dict[str, TimeframeAnalysis]
    alignment: TimeframeAlignment
    composite_score: float  # -100 to 100
    trend_strength: float  # 0 to 100
    recommended_action: str
    confidence: float  # 0 to 100
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'timeframe_analyses': {k: v.to_dict() for k, v in self.timeframe_analyses.items()},
            'alignment': self.alignment.value,
            'composite_score': self.composite_score,
            'trend_strength': self.trend_strength,
            'recommended_action': self.recommended_action,
            'confidence': self.confidence,
            'metadata': self.metadata
        }


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_single_timeframe(
    df: pd.DataFrame,
    timeframe: str
) -> Optional[TimeframeAnalysis]:
    """
    Analyze a single timeframe.
    
    Args:
        df: OHLCV DataFrame
        timeframe: Timeframe name
        
    Returns:
        TimeframeAnalysis or None if insufficient data
    """
    if df is None or len(df) < 50:
        return None
    
    try:
        # Handle DataFrame structure
        if isinstance(df.columns, pd.MultiIndex):
            close = df['Close'].iloc[:, 0]
            high = df['High'].iloc[:, 0]
            low = df['Low'].iloc[:, 0]
        else:
            close = df['Close']
            high = df['High']
            low = df['Low']
        
        current_price = close.iloc[-1]
        
        # Calculate indicators
        sma_20 = calculate_sma(close, 20)
        sma_50 = calculate_sma(close, 50)
        rsi = calculate_rsi(close, 14)
        macd_line, signal_line, histogram = calculate_macd(close)
        adx, plus_di, minus_di = calculate_adx(high, low, close)
        
        # Get latest values
        latest_sma20 = sma_20.iloc[-1] if not pd.isna(sma_20.iloc[-1]) else current_price
        latest_sma50 = sma_50.iloc[-1] if not pd.isna(sma_50.iloc[-1]) else current_price
        latest_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
        latest_histogram = histogram.iloc[-1] if not pd.isna(histogram.iloc[-1]) else 0
        latest_adx = adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 20
        
        # Calculate relative position
        price_vs_sma20 = ((current_price - latest_sma20) / latest_sma20 * 100) if latest_sma20 > 0 else 0
        price_vs_sma50 = ((current_price - latest_sma50) / latest_sma50 * 100) if latest_sma50 > 0 else 0
        
        # Determine trend direction
        if price_vs_sma20 > 2 and price_vs_sma50 > 5:
            trend = TrendDirection.STRONG_UP
        elif price_vs_sma20 > 0 and price_vs_sma50 > 0:
            trend = TrendDirection.UP
        elif price_vs_sma20 < -2 and price_vs_sma50 < -5:
            trend = TrendDirection.STRONG_DOWN
        elif price_vs_sma20 < 0 and price_vs_sma50 < 0:
            trend = TrendDirection.DOWN
        else:
            trend = TrendDirection.NEUTRAL
        
        # Calculate score (-100 to 100)
        score = 0
        
        # RSI component (-30 to 30)
        if latest_rsi > 70:
            score -= (latest_rsi - 70) * 1.5
        elif latest_rsi < 30:
            score += (30 - latest_rsi) * 1.5
        else:
            score += (50 - latest_rsi) * -0.5
        
        # MACD component (-30 to 30)
        macd_score = np.clip(latest_histogram * 500, -30, 30)
        score += macd_score
        
        # Trend component (-40 to 40)
        if trend == TrendDirection.STRONG_UP:
            score += 40
        elif trend == TrendDirection.UP:
            score += 20
        elif trend == TrendDirection.STRONG_DOWN:
            score -= 40
        elif trend == TrendDirection.DOWN:
            score -= 20
        
        score = np.clip(score, -100, 100)
        
        return TimeframeAnalysis(
            timeframe=timeframe,
            trend=trend,
            rsi=float(latest_rsi),
            macd_histogram=float(latest_histogram),
            price_vs_sma20=float(price_vs_sma20),
            price_vs_sma50=float(price_vs_sma50),
            adx=float(latest_adx),
            score=float(score),
            timestamp=df.index[-1] if hasattr(df.index[-1], 'isoformat') else datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error analyzing timeframe {timeframe}: {e}")
        return None


def determine_alignment(
    analyses: Dict[str, TimeframeAnalysis]
) -> TimeframeAlignment:
    """
    Determine the alignment status across timeframes.
    
    Args:
        analyses: Dictionary of timeframe -> analysis
        
    Returns:
        TimeframeAlignment status
    """
    if not analyses:
        return TimeframeAlignment.MIXED
    
    bullish_count = 0
    bearish_count = 0
    
    for tf, analysis in analyses.items():
        if analysis.score > 20:
            bullish_count += 1
        elif analysis.score < -20:
            bearish_count += 1
    
    total = len(analyses)
    
    if bullish_count == total:
        return TimeframeAlignment.FULLY_ALIGNED_BULLISH
    elif bullish_count >= total * 0.7:
        return TimeframeAlignment.MOSTLY_ALIGNED_BULLISH
    elif bearish_count == total:
        return TimeframeAlignment.FULLY_ALIGNED_BEARISH
    elif bearish_count >= total * 0.7:
        return TimeframeAlignment.MOSTLY_ALIGNED_BEARISH
    else:
        return TimeframeAlignment.MIXED


def get_recommended_action(
    alignment: TimeframeAlignment,
    composite_score: float,
    confidence: float
) -> str:
    """
    Get recommended action based on MTF analysis.
    """
    if alignment == TimeframeAlignment.FULLY_ALIGNED_BULLISH:
        if composite_score > 50:
            return "STRONG BUY - All timeframes aligned bullish"
        return "BUY - Bullish alignment across timeframes"
    
    elif alignment == TimeframeAlignment.FULLY_ALIGNED_BEARISH:
        if composite_score < -50:
            return "STRONG SELL - All timeframes aligned bearish"
        return "SELL - Bearish alignment across timeframes"
    
    elif alignment == TimeframeAlignment.MOSTLY_ALIGNED_BULLISH:
        return "BUY - Most timeframes bullish"
    
    elif alignment == TimeframeAlignment.MOSTLY_ALIGNED_BEARISH:
        return "SELL - Most timeframes bearish"
    
    else:
        if abs(composite_score) < 20:
            return "HOLD - Mixed signals, wait for clearer direction"
        elif composite_score > 0:
            return "WEAK BUY - Slight bullish bias"
        else:
            return "WEAK SELL - Slight bearish bias"


# ============================================================================
# MTF SERVICE CLASS
# ============================================================================

class MTFService:
    """
    Multi-Timeframe Analysis Service.
    
    Provides comprehensive multi-timeframe analysis for trading decisions.
    """
    
    def __init__(self, timeframes: Optional[List[str]] = None):
        """
        Initialize the MTF Service.
        
        Args:
            timeframes: List of timeframes to analyze. Uses defaults if not provided.
        """
        self.timeframes = timeframes or ['1h', '4h', '1d', '1w']
        self._cache: Dict[str, MTFAnalysisResult] = {}
        logger.info(f"MTFService initialized with timeframes: {self.timeframes}")
    
    def fetch_timeframe_data(
        self,
        symbol: str,
        timeframe: str
    ) -> Optional[pd.DataFrame]:
        """
        Fetch data for a specific timeframe.
        
        Args:
            symbol: Stock symbol
            timeframe: Timeframe key
            
        Returns:
            DataFrame or None
        """
        if timeframe not in TIMEFRAMES:
            logger.warning(f"Unknown timeframe: {timeframe}")
            return None
        
        tf_config = TIMEFRAMES[timeframe]
        
        try:
            ticker = yf.Ticker(symbol)
            
            # Handle 4h special case (aggregate from 1h)
            if timeframe == '4h':
                df = ticker.history(period='2mo', interval='60m')
                if not df.empty:
                    # Resample to 4h
                    df = df.resample('4H').agg({
                        'Open': 'first',
                        'High': 'max',
                        'Low': 'min',
                        'Close': 'last',
                        'Volume': 'sum'
                    }).dropna()
            else:
                df = ticker.history(
                    period=tf_config['period'],
                    interval=tf_config['interval']
                )
            
            return df if not df.empty else None
            
        except Exception as e:
            logger.error(f"Error fetching {timeframe} data for {symbol}: {e}")
            return None
    
    def analyze_symbol(
        self,
        symbol: str,
        use_cache: bool = True,
        cache_ttl_minutes: int = 5
    ) -> MTFAnalysisResult:
        """
        Perform multi-timeframe analysis for a symbol.
        
        Args:
            symbol: Stock symbol
            use_cache: Whether to use cached results
            cache_ttl_minutes: Cache time-to-live in minutes
            
        Returns:
            MTFAnalysisResult
        """
        cache_key = f"{symbol}_{','.join(self.timeframes)}"
        
        # Check cache
        if use_cache and cache_key in self._cache:
            cached = self._cache[cache_key]
            # Check if still valid (simplified check)
            return cached
        
        analyses: Dict[str, TimeframeAnalysis] = {}
        
        for tf in self.timeframes:
            df = self.fetch_timeframe_data(symbol, tf)
            if df is not None:
                analysis = analyze_single_timeframe(df, tf)
                if analysis:
                    analyses[tf] = analysis
        
        if not analyses:
            return MTFAnalysisResult(
                symbol=symbol,
                timeframe_analyses={},
                alignment=TimeframeAlignment.MIXED,
                composite_score=0,
                trend_strength=0,
                recommended_action="HOLD - Insufficient data",
                confidence=0,
                metadata={'error': 'No data available'}
            )
        
        # Calculate composite score (weighted by timeframe importance)
        total_weight = 0
        weighted_score = 0
        
        for tf, analysis in analyses.items():
            weight = TIMEFRAMES.get(tf, {}).get('weight', 0.2)
            weighted_score += analysis.score * weight
            total_weight += weight
        
        composite_score = weighted_score / total_weight if total_weight > 0 else 0
        
        # Calculate trend strength (based on ADX and alignment)
        avg_adx = np.mean([a.adx for a in analyses.values()])
        alignment = determine_alignment(analyses)
        
        # adjustment for alignment
        alignment_factor = {
            TimeframeAlignment.FULLY_ALIGNED_BULLISH: 1.5,
            TimeframeAlignment.FULLY_ALIGNED_BEARISH: 1.5,
            TimeframeAlignment.MOSTLY_ALIGNED_BULLISH: 1.2,
            TimeframeAlignment.MOSTLY_ALIGNED_BEARISH: 1.2,
            TimeframeAlignment.MIXED: 0.8
        }
        
        trend_strength = min(100, avg_adx * alignment_factor.get(alignment, 1.0))
        
        # Calculate confidence
        confidence_factors = []
        
        # More timeframes = more confidence
        confidence_factors.append(len(analyses) / len(self.timeframes) * 30)
        
        # Better alignment = more confidence
        if alignment in [TimeframeAlignment.FULLY_ALIGNED_BULLISH, TimeframeAlignment.FULLY_ALIGNED_BEARISH]:
            confidence_factors.append(40)
        elif alignment in [TimeframeAlignment.MOSTLY_ALIGNED_BULLISH, TimeframeAlignment.MOSTLY_ALIGNED_BEARISH]:
            confidence_factors.append(25)
        else:
            confidence_factors.append(10)
        
        # Stronger ADX = more confidence
        confidence_factors.append(min(30, avg_adx))
        
        confidence = min(100, sum(confidence_factors))
        
        # Get recommendation
        recommended_action = get_recommended_action(alignment, composite_score, confidence)
        
        result = MTFAnalysisResult(
            symbol=symbol,
            timeframe_analyses=analyses,
            alignment=alignment,
            composite_score=composite_score,
            trend_strength=trend_strength,
            recommended_action=recommended_action,
            confidence=confidence,
            metadata={
                'analyzed_timeframes': list(analyses.keys()),
                'avg_adx': avg_adx
            }
        )
        
        # Cache result
        if use_cache:
            self._cache[cache_key] = result
        
        return result
    
    def get_alignment_heatmap_data(
        self,
        symbols: List[str]
    ) -> pd.DataFrame:
        """
        Get alignment heatmap data for multiple symbols.
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            DataFrame with scores for each symbol/timeframe
        """
        data = []
        
        for symbol in symbols:
            result = self.analyze_symbol(symbol)
            row = {'symbol': symbol}
            
            for tf, analysis in result.timeframe_analyses.items():
                row[tf] = analysis.score
            
            row['composite'] = result.composite_score
            row['alignment'] = result.alignment.value
            data.append(row)
        
        return pd.DataFrame(data)
    
    def find_aligned_opportunities(
        self,
        symbols: List[str],
        min_alignment: TimeframeAlignment = TimeframeAlignment.MOSTLY_ALIGNED_BULLISH,
        min_confidence: float = 60
    ) -> List[MTFAnalysisResult]:
        """
        Find symbols with good MTF alignment.
        
        Args:
            symbols: List of stock symbols
            min_alignment: Minimum alignment level
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of MTF analysis results meeting criteria
        """
        aligned_results = []
        
        bullish_alignments = [
            TimeframeAlignment.FULLY_ALIGNED_BULLISH,
            TimeframeAlignment.MOSTLY_ALIGNED_BULLISH
        ]
        bearish_alignments = [
            TimeframeAlignment.FULLY_ALIGNED_BEARISH,
            TimeframeAlignment.MOSTLY_ALIGNED_BEARISH
        ]
        
        for symbol in symbols:
            try:
                result = self.analyze_symbol(symbol)
                
                if result.confidence < min_confidence:
                    continue
                
                if min_alignment in bullish_alignments:
                    if result.alignment in bullish_alignments:
                        aligned_results.append(result)
                elif min_alignment in bearish_alignments:
                    if result.alignment in bearish_alignments:
                        aligned_results.append(result)
                else:
                    aligned_results.append(result)
                    
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                continue
        
        # Sort by confidence
        aligned_results.sort(key=lambda x: (x.confidence, abs(x.composite_score)), reverse=True)
        
        return aligned_results
    
    def get_timeframe_divergences(
        self,
        symbol: str
    ) -> Dict[str, Any]:
        """
        Identify divergences between timeframes.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with divergence information
        """
        result = self.analyze_symbol(symbol)
        
        divergences = {
            'has_divergence': False,
            'divergence_type': None,
            'details': [],
            'risk_level': 'Low'
        }
        
        if len(result.timeframe_analyses) < 2:
            return divergences
        
        analyses = list(result.timeframe_analyses.items())
        
        for i in range(len(analyses) - 1):
            tf1, a1 = analyses[i]
            tf2, a2 = analyses[i + 1]
            
            # Check for significant divergence
            if (a1.score > 30 and a2.score < -30) or (a1.score < -30 and a2.score > 30):
                divergences['has_divergence'] = True
                divergences['details'].append({
                    'timeframe1': tf1,
                    'timeframe2': tf2,
                    'score1': a1.score,
                    'score2': a2.score,
                    'trend1': a1.trend.value,
                    'trend2': a2.trend.value
                })
        
        if divergences['has_divergence']:
            divergences['risk_level'] = 'High'
            divergences['divergence_type'] = 'Timeframe Conflict'
        
        return divergences
    
    def clear_cache(self) -> None:
        """Clear the analysis cache."""
        self._cache.clear()
        logger.info("MTFService cache cleared")
