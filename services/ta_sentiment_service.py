"""
Social Sentiment Service
========================
Sentiment analysis for trading signals

Provides:
- News sentiment analysis
- Social media sentiment tracking
- Sentiment scoring and aggregation
- Sentiment-price correlation

Note: This service provides a framework for sentiment analysis.
In production, it would integrate with:
- Twitter/X API for social sentiment
- News APIs (NewsAPI, Alpha Vantage, etc.)
- Reddit API for retail sentiment
- StockTwits API

Author: Stock Risk App | Feb 2026
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import re
from collections import defaultdict

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS
# ============================================================================

class SentimentLevel(Enum):
    """Sentiment levels."""
    VERY_BEARISH = "very_bearish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    BULLISH = "bullish"
    VERY_BULLISH = "very_bullish"
    
    @classmethod
    def from_score(cls, score: float) -> 'SentimentLevel':
        """Convert numeric score (-1 to 1) to sentiment level."""
        if score <= -0.6:
            return cls.VERY_BEARISH
        elif score <= -0.2:
            return cls.BEARISH
        elif score <= 0.2:
            return cls.NEUTRAL
        elif score <= 0.6:
            return cls.BULLISH
        else:
            return cls.VERY_BULLISH


class SentimentSource(Enum):
    """Sources of sentiment data."""
    NEWS = "news"
    TWITTER = "twitter"
    REDDIT = "reddit"
    STOCKTWITS = "stocktwits"
    ANALYST = "analyst"
    OPTIONS_FLOW = "options_flow"
    INSIDER = "insider"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class SentimentItem:
    """
    Individual sentiment data point.
    """
    source: SentimentSource
    text: str
    score: float  # -1 (bearish) to 1 (bullish)
    confidence: float  # 0 to 1
    timestamp: datetime
    symbol: str
    url: Optional[str] = None
    author: Optional[str] = None
    engagement: int = 0  # Likes, retweets, etc.
    
    @property
    def level(self) -> SentimentLevel:
        return SentimentLevel.from_score(self.score)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'source': self.source.value,
            'text': self.text[:200],  # Truncate for display
            'score': self.score,
            'confidence': self.confidence,
            'level': self.level.value,
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'url': self.url,
            'author': self.author,
            'engagement': self.engagement
        }


@dataclass
class SentimentAggregate:
    """
    Aggregated sentiment for a symbol.
    """
    symbol: str
    overall_score: float
    overall_level: SentimentLevel
    confidence: float
    source_scores: Dict[str, float]  # Score by source
    score_history: List[Tuple[datetime, float]]  # Time series
    total_mentions: int
    bullish_ratio: float  # Percentage of bullish items
    sentiment_velocity: float  # Rate of sentiment change
    key_topics: List[str]
    top_items: List[SentimentItem]
    updated_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'overall_score': self.overall_score,
            'overall_level': self.overall_level.value,
            'confidence': self.confidence,
            'source_scores': self.source_scores,
            'score_history': [
                {'date': d.isoformat(), 'score': s} 
                for d, s in self.score_history[-30:]  # Last 30 points
            ],
            'total_mentions': self.total_mentions,
            'bullish_ratio': self.bullish_ratio,
            'sentiment_velocity': self.sentiment_velocity,
            'key_topics': self.key_topics,
            'top_items': [item.to_dict() for item in self.top_items[:5]],
            'updated_at': self.updated_at.isoformat()
        }


@dataclass
class SentimentSignal:
    """
    Trading signal based on sentiment.
    """
    symbol: str
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    strength: float  # 0 to 1
    sentiment_score: float
    sentiment_level: SentimentLevel
    reason: str
    confidence: float
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'signal_type': self.signal_type,
            'strength': self.strength,
            'sentiment_score': self.sentiment_score,
            'sentiment_level': self.sentiment_level.value,
            'reason': self.reason,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat()
        }


# ============================================================================
# SENTIMENT KEYWORDS
# ============================================================================

# Keywords for simple sentiment analysis
BULLISH_KEYWORDS = [
    'buy', 'bullish', 'long', 'moon', 'rocket', 'calls', 'upgrade',
    'breakout', 'rally', 'surge', 'soar', 'spike', 'beat', 'outperform',
    'growth', 'profit', 'revenue', 'earnings', 'strong', 'up', 'gain',
    'higher', 'momentum', 'accumulate', 'support', 'bounce', 'recovery',
    'opportunity', 'undervalued', 'cheap', 'dividend', 'innovation'
]

BEARISH_KEYWORDS = [
    'sell', 'bearish', 'short', 'puts', 'downgrade', 'crash', 'dump',
    'tank', 'plunge', 'fall', 'drop', 'miss', 'underperform', 'weak',
    'down', 'loss', 'decline', 'lower', 'resistance', 'breakdown',
    'overvalued', 'expensive', 'risk', 'warning', 'concern', 'lawsuit',
    'investigation', 'fraud', 'bankruptcy', 'recession', 'inflation'
]

INTENSITY_MODIFIERS = {
    'very': 1.5, 'extremely': 2.0, 'highly': 1.5, 'massive': 1.8,
    'huge': 1.7, 'incredible': 1.6, 'slightly': 0.5, 'somewhat': 0.6,
    'potentially': 0.7, 'likely': 0.8, 'definitely': 1.4
}

NEGATION_WORDS = ['not', 'no', "n't", 'never', 'neither', 'without', 'hardly']


# ============================================================================
# SENTIMENT SERVICE CLASS
# ============================================================================

class SentimentService:
    """
    Social Sentiment Service.
    
    Provides sentiment analysis and scoring for stocks based on
    news and social media data.
    """
    
    def __init__(self):
        """Initialize the Sentiment Service."""
        self._cache: Dict[str, SentimentAggregate] = {}
        self._history: Dict[str, List[SentimentItem]] = defaultdict(list)
        logger.info("SentimentService initialized")
    
    def analyze_text(
        self,
        text: str,
        symbol: Optional[str] = None
    ) -> Tuple[float, float]:
        """
        Analyze sentiment of text using keyword-based analysis.
        
        Args:
            text: Text to analyze
            symbol: Optional symbol to look for
            
        Returns:
            Tuple of (sentiment_score, confidence)
        """
        if not text:
            return 0.0, 0.0
        
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        bullish_count = 0
        bearish_count = 0
        intensity = 1.0
        negated = False
        
        for i, word in enumerate(words):
            # Check for negation in previous words
            if i > 0 and words[i-1] in NEGATION_WORDS:
                negated = True
            else:
                negated = False
            
            # Check for intensity modifiers
            if word in INTENSITY_MODIFIERS:
                intensity = INTENSITY_MODIFIERS[word]
                continue
            
            # Count bullish/bearish words
            if word in BULLISH_KEYWORDS:
                if negated:
                    bearish_count += intensity
                else:
                    bullish_count += intensity
            elif word in BEARISH_KEYWORDS:
                if negated:
                    bullish_count += intensity
                else:
                    bearish_count += intensity
            
            intensity = 1.0  # Reset intensity
        
        total = bullish_count + bearish_count
        if total == 0:
            return 0.0, 0.0
        
        # Calculate score (-1 to 1)
        score = (bullish_count - bearish_count) / (bullish_count + bearish_count)
        
        # Calculate confidence based on total matches and text length
        word_confidence = min(total / 10, 1.0)  # More matches = higher confidence
        length_confidence = min(len(words) / 50, 1.0)  # Longer text = higher confidence
        confidence = (word_confidence + length_confidence) / 2
        
        return score, confidence
    
    def add_sentiment_item(
        self,
        item: SentimentItem
    ) -> None:
        """Add a sentiment item to history."""
        self._history[item.symbol.upper()].append(item)
        
        # Keep only last 1000 items per symbol
        if len(self._history[item.symbol.upper()]) > 1000:
            self._history[item.symbol.upper()] = self._history[item.symbol.upper()][-1000:]
        
        # Invalidate cache
        if item.symbol.upper() in self._cache:
            del self._cache[item.symbol.upper()]
    
    def get_aggregate_sentiment(
        self,
        symbol: str,
        lookback_hours: int = 24
    ) -> Optional[SentimentAggregate]:
        """
        Get aggregated sentiment for a symbol.
        
        Args:
            symbol: Stock symbol
            lookback_hours: Hours to look back
            
        Returns:
            SentimentAggregate or None
        """
        symbol = symbol.upper()
        
        # Check cache
        cache_key = f"{symbol}_{lookback_hours}"
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            if (datetime.now() - cached.updated_at).seconds < 300:  # 5 min cache
                return cached
        
        items = self._history.get(symbol, [])
        if not items:
            return None
        
        # Filter by lookback period
        cutoff = datetime.now() - timedelta(hours=lookback_hours)
        recent_items = [i for i in items if i.timestamp >= cutoff]
        
        if not recent_items:
            return None
        
        # Calculate overall score (weighted by confidence and engagement)
        total_weight = 0
        weighted_score = 0
        
        for item in recent_items:
            weight = item.confidence * (1 + np.log1p(item.engagement) / 10)
            weighted_score += item.score * weight
            total_weight += weight
        
        overall_score = weighted_score / total_weight if total_weight > 0 else 0
        overall_level = SentimentLevel.from_score(overall_score)
        
        # Score by source
        source_scores = {}
        for source in SentimentSource:
            source_items = [i for i in recent_items if i.source == source]
            if source_items:
                source_scores[source.value] = np.mean([i.score for i in source_items])
        
        # Calculate score history (hourly)
        score_history = []
        for h in range(lookback_hours, 0, -1):
            hour_start = datetime.now() - timedelta(hours=h)
            hour_end = hour_start + timedelta(hours=1)
            hour_items = [i for i in recent_items if hour_start <= i.timestamp < hour_end]
            if hour_items:
                score_history.append((hour_start, np.mean([i.score for i in hour_items])))
        
        # Bullish ratio
        bullish_count = len([i for i in recent_items if i.score > 0.2])
        bullish_ratio = bullish_count / len(recent_items) * 100
        
        # Sentiment velocity (change rate)
        if len(score_history) >= 2:
            recent_scores = [s for _, s in score_history[-6:]]
            older_scores = [s for _, s in score_history[:-6]] if len(score_history) > 6 else [0]
            velocity = np.mean(recent_scores) - np.mean(older_scores)
        else:
            velocity = 0
        
        # Key topics (simplified - extract common words)
        all_text = ' '.join([i.text for i in recent_items])
        words = re.findall(r'\b[A-Za-z]{4,}\b', all_text.lower())
        word_freq = defaultdict(int)
        for word in words:
            if word not in BULLISH_KEYWORDS + BEARISH_KEYWORDS + ['that', 'this', 'with', 'from']:
                word_freq[word] += 1
        key_topics = [w for w, _ in sorted(word_freq.items(), key=lambda x: -x[1])[:10]]
        
        # Top items (highest engagement)
        top_items = sorted(recent_items, key=lambda x: x.engagement, reverse=True)[:10]
        
        # Overall confidence
        confidence = np.mean([i.confidence for i in recent_items])
        
        aggregate = SentimentAggregate(
            symbol=symbol,
            overall_score=overall_score,
            overall_level=overall_level,
            confidence=confidence,
            source_scores=source_scores,
            score_history=score_history,
            total_mentions=len(recent_items),
            bullish_ratio=bullish_ratio,
            sentiment_velocity=velocity,
            key_topics=key_topics,
            top_items=top_items,
            updated_at=datetime.now()
        )
        
        self._cache[cache_key] = aggregate
        return aggregate
    
    def generate_sentiment_signal(
        self,
        symbol: str,
        price_data: Optional[pd.DataFrame] = None
    ) -> Optional[SentimentSignal]:
        """
        Generate trading signal based on sentiment.
        
        Args:
            symbol: Stock symbol
            price_data: Optional OHLCV data for correlation
            
        Returns:
            SentimentSignal or None
        """
        aggregate = self.get_aggregate_sentiment(symbol)
        if not aggregate:
            return None
        
        score = aggregate.overall_score
        velocity = aggregate.sentiment_velocity
        confidence = aggregate.confidence
        mentions = aggregate.total_mentions
        
        # Determine signal
        signal_type = 'HOLD'
        strength = 0.0
        reason = "Neutral sentiment"
        
        # Strong bullish conditions
        if score > 0.4 and velocity > 0.1 and mentions >= 10:
            signal_type = 'BUY'
            strength = min(abs(score) * confidence, 1.0)
            reason = f"Strong bullish sentiment (score: {score:.2f}) with positive momentum"
        
        # Moderate bullish
        elif score > 0.2 and confidence > 0.5:
            signal_type = 'BUY'
            strength = min(abs(score) * confidence * 0.7, 0.7)
            reason = f"Moderately bullish sentiment (score: {score:.2f})"
        
        # Strong bearish conditions
        elif score < -0.4 and velocity < -0.1 and mentions >= 10:
            signal_type = 'SELL'
            strength = min(abs(score) * confidence, 1.0)
            reason = f"Strong bearish sentiment (score: {score:.2f}) with negative momentum"
        
        # Moderate bearish
        elif score < -0.2 and confidence > 0.5:
            signal_type = 'SELL'
            strength = min(abs(score) * confidence * 0.7, 0.7)
            reason = f"Moderately bearish sentiment (score: {score:.2f})"
        
        # Sentiment reversal
        elif abs(velocity) > 0.3:
            if velocity > 0:
                signal_type = 'BUY'
                reason = "Sentiment reversing to bullish"
            else:
                signal_type = 'SELL'
                reason = "Sentiment reversing to bearish"
            strength = min(abs(velocity) * confidence, 0.6)
        
        return SentimentSignal(
            symbol=symbol,
            signal_type=signal_type,
            strength=strength,
            sentiment_score=score,
            sentiment_level=aggregate.overall_level,
            reason=reason,
            confidence=confidence,
            timestamp=datetime.now()
        )
    
    def calculate_sentiment_price_correlation(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        lookback_days: int = 30
    ) -> Dict[str, float]:
        """
        Calculate correlation between sentiment and price movements.
        
        Args:
            symbol: Stock symbol
            price_data: OHLCV DataFrame
            lookback_days: Days to analyze
            
        Returns:
            Dict with correlation metrics
        """
        symbol = symbol.upper()
        items = self._history.get(symbol, [])
        
        if not items or len(price_data) < 5:
            return {'correlation': 0, 'lead_lag': 0, 'r_squared': 0}
        
        # Handle MultiIndex columns
        if isinstance(price_data.columns, pd.MultiIndex):
            close = price_data['Close'].iloc[:, 0]
        else:
            close = price_data['Close']
        
        # Create daily sentiment scores
        cutoff = datetime.now() - timedelta(days=lookback_days)
        recent_items = [i for i in items if i.timestamp >= cutoff]
        
        daily_sentiment = {}
        for item in recent_items:
            date = item.timestamp.date()
            if date not in daily_sentiment:
                daily_sentiment[date] = []
            daily_sentiment[date].append(item.score)
        
        daily_scores = {d: np.mean(scores) for d, scores in daily_sentiment.items()}
        
        if len(daily_scores) < 5:
            return {'correlation': 0, 'lead_lag': 0, 'r_squared': 0}
        
        # Create aligned series
        sentiment_series = pd.Series(daily_scores)
        price_returns = close.pct_change()
        
        # Align indices
        common_dates = sentiment_series.index.intersection(price_returns.index)
        if len(common_dates) < 5:
            return {'correlation': 0, 'lead_lag': 0, 'r_squared': 0}
        
        sent = sentiment_series.loc[common_dates]
        ret = price_returns.loc[common_dates]
        
        # Same-day correlation
        correlation = sent.corr(ret)
        
        # Lead-lag analysis (sentiment leading price)
        lead_corr = sent.shift(1).corr(ret)
        lag_corr = sent.corr(ret.shift(1))
        
        if abs(lead_corr) > abs(lag_corr):
            lead_lag = lead_corr
        else:
            lead_lag = -lag_corr
        
        # R-squared
        r_squared = correlation ** 2 if not np.isnan(correlation) else 0
        
        return {
            'correlation': float(correlation) if not np.isnan(correlation) else 0,
            'lead_lag': float(lead_lag) if not np.isnan(lead_lag) else 0,
            'r_squared': float(r_squared)
        }
    
    def get_sentiment_heatmap_data(
        self,
        symbols: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get sentiment data for multiple symbols (for heatmap display).
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dict mapping symbol to sentiment data
        """
        result = {}
        
        for symbol in symbols:
            aggregate = self.get_aggregate_sentiment(symbol.upper())
            if aggregate:
                result[symbol.upper()] = {
                    'score': aggregate.overall_score,
                    'level': aggregate.overall_level.value,
                    'mentions': aggregate.total_mentions,
                    'velocity': aggregate.sentiment_velocity,
                    'bullish_ratio': aggregate.bullish_ratio
                }
            else:
                result[symbol.upper()] = {
                    'score': 0,
                    'level': 'neutral',
                    'mentions': 0,
                    'velocity': 0,
                    'bullish_ratio': 50
                }
        
        return result
    
    def simulate_news_sentiment(
        self,
        symbol: str,
        n_items: int = 50,
        bias: float = 0.0
    ) -> None:
        """
        Simulate news sentiment items for testing.
        
        Args:
            symbol: Stock symbol
            n_items: Number of items to generate
            bias: Sentiment bias (-1 to 1)
        """
        symbol = symbol.upper()
        
        headlines = [
            f"{symbol} reports strong quarterly earnings",
            f"{symbol} stock surges on analyst upgrade",
            f"{symbol} faces regulatory investigation",
            f"{symbol} announces major partnership",
            f"{symbol} misses revenue expectations",
            f"{symbol} CEO discusses growth strategy",
            f"Analysts bullish on {symbol} outlook",
            f"{symbol} shares drop on weak guidance",
            f"{symbol} expands into new markets",
            f"Institutional investors accumulate {symbol}",
        ]
        
        for i in range(n_items):
            text = np.random.choice(headlines)
            base_score, confidence = self.analyze_text(text, symbol)
            
            # Add some randomness and bias
            score = np.clip(base_score + bias + np.random.normal(0, 0.2), -1, 1)
            
            item = SentimentItem(
                source=np.random.choice(list(SentimentSource)),
                text=text,
                score=score,
                confidence=confidence + np.random.uniform(0, 0.3),
                timestamp=datetime.now() - timedelta(hours=np.random.randint(0, 72)),
                symbol=symbol,
                engagement=np.random.randint(0, 1000)
            )
            
            self.add_sentiment_item(item)
        
        logger.info(f"Simulated {n_items} sentiment items for {symbol}")
    
    def get_sentiment_summary(
        self,
        symbol: str
    ) -> Dict[str, Any]:
        """
        Get a summary of sentiment for display.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Summary dictionary
        """
        aggregate = self.get_aggregate_sentiment(symbol.upper())
        signal = self.generate_sentiment_signal(symbol)
        
        if not aggregate:
            return {
                'symbol': symbol.upper(),
                'has_data': False,
                'score': 0,
                'level': 'neutral',
                'signal': 'HOLD',
                'message': 'No sentiment data available'
            }
        
        return {
            'symbol': symbol.upper(),
            'has_data': True,
            'score': aggregate.overall_score,
            'level': aggregate.overall_level.value,
            'confidence': aggregate.confidence,
            'mentions': aggregate.total_mentions,
            'bullish_ratio': aggregate.bullish_ratio,
            'velocity': aggregate.sentiment_velocity,
            'signal': signal.signal_type if signal else 'HOLD',
            'signal_strength': signal.strength if signal else 0,
            'signal_reason': signal.reason if signal else 'Insufficient data',
            'key_topics': aggregate.key_topics[:5],
            'source_breakdown': aggregate.source_scores
        }
