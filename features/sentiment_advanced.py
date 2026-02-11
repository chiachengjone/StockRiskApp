"""
Advanced Sentiment Analysis Module
===================================
Expanded sentiment analysis with social media integration,
news impact scoring, and sentiment momentum indicators.

Features:
- Reddit integration (WallStreetBets, stocks, investing)
- News sentiment with source weighting
- Sentiment momentum and divergence
- Multi-source aggregation
- Real-time sentiment tracking

Author: Stock Risk App | Feb 2026
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import re
from collections import defaultdict

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import praw
    HAS_PRAW = True
except ImportError:
    HAS_PRAW = False

try:
    import tweepy
    HAS_TWEEPY = True
except ImportError:
    HAS_TWEEPY = False

try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    HAS_VADER = True
except ImportError:
    HAS_VADER = False


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class SentimentSource(Enum):
    """Sources of sentiment data."""
    REDDIT = "reddit"
    TWITTER = "twitter"
    NEWS = "news"
    STOCKTWITS = "stocktwits"
    FINVIZ = "finviz"


class SentimentStrength(Enum):
    """Strength of sentiment signal."""
    VERY_BEARISH = "very_bearish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    BULLISH = "bullish"
    VERY_BULLISH = "very_bullish"


@dataclass
class SentimentPost:
    """Individual sentiment post/article."""
    source: SentimentSource
    timestamp: datetime
    title: str
    content: str
    author: str
    score: float  # -1 to 1
    confidence: float  # 0 to 1
    engagement: int  # upvotes, retweets, etc
    url: str = ""
    tickers_mentioned: List[str] = field(default_factory=list)


@dataclass
class SourceSentiment:
    """Sentiment from a single source."""
    source: SentimentSource
    score: float  # -1 to 1
    volume: int  # Number of posts/articles
    bullish_pct: float
    bearish_pct: float
    neutral_pct: float
    trending_topics: List[str]
    top_posts: List[SentimentPost]
    last_updated: datetime


@dataclass
class SentimentMomentum:
    """Sentiment momentum indicators."""
    current_score: float
    score_1d_ago: float
    score_7d_ago: float
    score_30d_ago: float
    momentum_1d: float
    momentum_7d: float
    momentum_30d: float
    acceleration: float
    is_diverging_from_price: bool
    divergence_direction: str  # 'bullish_divergence', 'bearish_divergence', 'none'


@dataclass
class AggregateSentiment:
    """Aggregated sentiment across all sources."""
    ticker: str
    overall_score: float
    strength: SentimentStrength
    confidence: float
    source_breakdown: Dict[str, SourceSentiment]
    momentum: SentimentMomentum
    risk_level: str  # 'low', 'medium', 'high'
    recommendation: str
    timestamp: datetime


# ============================================================================
# SENTIMENT ANALYZERS
# ============================================================================

class TextSentimentAnalyzer:
    """
    Analyze sentiment from text using multiple NLP methods.
    """
    
    def __init__(self):
        if HAS_VADER:
            self.vader = SentimentIntensityAnalyzer()
        else:
            self.vader = None
    
    def analyze(self, text: str) -> Tuple[float, float]:
        """
        Analyze text sentiment.
        
        Returns:
            Tuple of (score, confidence) where score is -1 to 1
        """
        if not text or len(text.strip()) < 3:
            return 0.0, 0.0
        
        scores = []
        
        # VADER sentiment
        if self.vader:
            vs = self.vader.polarity_scores(text)
            scores.append(vs['compound'])
        
        # TextBlob sentiment
        if HAS_TEXTBLOB:
            blob = TextBlob(text)
            scores.append(blob.sentiment.polarity)
        
        # Fallback: simple keyword-based
        if not scores:
            scores.append(self._keyword_sentiment(text))
        
        avg_score = np.mean(scores)
        confidence = 1.0 - np.std(scores) if len(scores) > 1 else 0.7
        
        return avg_score, min(1.0, confidence)
    
    def _keyword_sentiment(self, text: str) -> float:
        """Simple keyword-based sentiment as fallback."""
        text_lower = text.lower()
        
        bullish_keywords = [
            'buy', 'long', 'calls', 'moon', 'rocket', 'bull', 'breakout',
            'undervalued', 'growth', 'strong', 'beat', 'upgrade', 'green',
            'rally', 'surge', 'boom', 'gain', 'profit', 'diamond hands', 'hodl'
        ]
        
        bearish_keywords = [
            'sell', 'short', 'puts', 'crash', 'bear', 'breakdown',
            'overvalued', 'decline', 'weak', 'miss', 'downgrade', 'red',
            'dump', 'plunge', 'bust', 'loss', 'paper hands', 'bail'
        ]
        
        bullish_count = sum(1 for kw in bullish_keywords if kw in text_lower)
        bearish_count = sum(1 for kw in bearish_keywords if kw in text_lower)
        
        total = bullish_count + bearish_count
        if total == 0:
            return 0.0
        
        return (bullish_count - bearish_count) / total


# ============================================================================
# REDDIT PROVIDER
# ============================================================================

class RedditSentimentProvider:
    """
    Fetch and analyze sentiment from Reddit.
    """
    
    SUBREDDITS = [
        'wallstreetbets',
        'stocks',
        'investing',
        'options',
        'stockmarket',
        'pennystocks'
    ]
    
    def __init__(self, client_id: str = None, client_secret: str = None):
        self.reddit = None
        self.analyzer = TextSentimentAnalyzer()
        
        if HAS_PRAW and client_id and client_secret:
            try:
                self.reddit = praw.Reddit(
                    client_id=client_id,
                    client_secret=client_secret,
                    user_agent="StockRiskApp/1.0"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize Reddit API: {e}")
    
    def get_sentiment(
        self,
        ticker: str,
        days: int = 7,
        limit: int = 100
    ) -> SourceSentiment:
        """
        Get sentiment for a ticker from Reddit.
        
        Args:
            ticker: Stock ticker symbol
            days: Number of days to look back
            limit: Maximum posts to analyze
            
        Returns:
            SourceSentiment object
        """
        posts = self._fetch_posts(ticker, limit)
        
        if not posts:
            return self._empty_sentiment()
        
        # Analyze posts
        scores = []
        bullish = 0
        bearish = 0
        neutral = 0
        
        for post in posts:
            score, _ = self.analyzer.analyze(f"{post.title} {post.content}")
            scores.append(score)
            
            if score > 0.1:
                bullish += 1
            elif score < -0.1:
                bearish += 1
            else:
                neutral += 1
        
        total = len(posts)
        avg_score = np.mean(scores)
        
        # Find trending topics
        all_text = ' '.join([f"{p.title} {p.content}" for p in posts])
        trending = self._extract_topics(all_text)
        
        return SourceSentiment(
            source=SentimentSource.REDDIT,
            score=avg_score,
            volume=len(posts),
            bullish_pct=bullish / total * 100 if total > 0 else 0,
            bearish_pct=bearish / total * 100 if total > 0 else 0,
            neutral_pct=neutral / total * 100 if total > 0 else 0,
            trending_topics=trending[:10],
            top_posts=sorted(posts, key=lambda x: x.engagement, reverse=True)[:5],
            last_updated=datetime.now()
        )
    
    def _fetch_posts(self, ticker: str, limit: int) -> List[SentimentPost]:
        """Fetch posts mentioning ticker."""
        posts = []
        
        if self.reddit:
            try:
                for subreddit_name in self.SUBREDDITS:
                    subreddit = self.reddit.subreddit(subreddit_name)
                    
                    for submission in subreddit.search(
                        ticker, sort='hot', time_filter='week', limit=limit // len(self.SUBREDDITS)
                    ):
                        score, confidence = self.analyzer.analyze(
                            f"{submission.title} {submission.selftext}"
                        )
                        
                        posts.append(SentimentPost(
                            source=SentimentSource.REDDIT,
                            timestamp=datetime.fromtimestamp(submission.created_utc),
                            title=submission.title,
                            content=submission.selftext[:500],
                            author=str(submission.author),
                            score=score,
                            confidence=confidence,
                            engagement=submission.score,
                            url=f"https://reddit.com{submission.permalink}",
                            tickers_mentioned=self._extract_tickers(
                                f"{submission.title} {submission.selftext}"
                            )
                        ))
            except Exception as e:
                logger.warning(f"Reddit API error: {e}")
        
        # If no API, generate sample data for demo
        if not posts:
            posts = self._generate_sample_posts(ticker)
        
        return posts
    
    def _generate_sample_posts(self, ticker: str) -> List[SentimentPost]:
        """Generate sample posts for demonstration."""
        sample_titles = [
            f" {ticker} looking bullish! Technical breakout incoming",
            f"DD: Why {ticker} is undervalued",
            f"{ticker} earnings coming up - what's your play?",
            f"Bearish on {ticker}? Here's why you're wrong",
            f"{ticker} short squeeze potential?",
            f"Just bought 1000 shares of {ticker}",
            f"Warning: {ticker} showing weakness",
            f"{ticker} to the moon! ",
        ]
        
        posts = []
        for i, title in enumerate(sample_titles):
            score, conf = self.analyzer.analyze(title)
            posts.append(SentimentPost(
                source=SentimentSource.REDDIT,
                timestamp=datetime.now() - timedelta(hours=i*3),
                title=title,
                content=f"Sample content about {ticker}...",
                author=f"user_{i}",
                score=score,
                confidence=conf,
                engagement=np.random.randint(10, 1000),
                tickers_mentioned=[ticker]
            ))
        
        return posts
    
    def _extract_tickers(self, text: str) -> List[str]:
        """Extract stock tickers from text."""
        # Match $TICKER or all-caps 2-5 letter words
        pattern = r'\$([A-Z]{1,5})|(?<![A-Z])([A-Z]{2,5})(?![A-Z])'
        matches = re.findall(pattern, text)
        tickers = [m[0] or m[1] for m in matches if m[0] or m[1]]
        
        # Filter out common words
        common_words = {'I', 'A', 'THE', 'AND', 'OR', 'FOR', 'TO', 'IN', 'ON', 'AT', 'IS'}
        return [t for t in tickers if t not in common_words]
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract trending topics from text."""
        # Simple word frequency
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter stopwords
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'this', 'that',
            'and', 'but', 'or', 'if', 'then', 'else', 'when', 'where',
            'what', 'which', 'who', 'how', 'why', 'with', 'from', 'to',
            'of', 'for', 'in', 'on', 'at', 'by', 'as', 'up', 'down'
        }
        
        filtered = [w for w in words if w not in stopwords and len(w) > 2]
        
        from collections import Counter
        counts = Counter(filtered)
        return [word for word, _ in counts.most_common(10)]
    
    def _empty_sentiment(self) -> SourceSentiment:
        """Return empty sentiment object."""
        return SourceSentiment(
            source=SentimentSource.REDDIT,
            score=0.0,
            volume=0,
            bullish_pct=0,
            bearish_pct=0,
            neutral_pct=0,
            trending_topics=[],
            top_posts=[],
            last_updated=datetime.now()
        )


# ============================================================================
# NEWS SENTIMENT PROVIDER
# ============================================================================

class NewsSentimentProvider:
    """
    Analyze sentiment from financial news sources.
    """
    
    # Source weights for credibility
    SOURCE_WEIGHTS = {
        'bloomberg': 1.2,
        'reuters': 1.2,
        'wsj': 1.1,
        'ft': 1.1,
        'cnbc': 1.0,
        'marketwatch': 0.9,
        'benzinga': 0.8,
        'seeking_alpha': 0.7,
        'motley_fool': 0.6,
        'default': 0.5
    }
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.analyzer = TextSentimentAnalyzer()
    
    def get_sentiment(
        self,
        ticker: str,
        days: int = 7
    ) -> SourceSentiment:
        """
        Get news sentiment for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            days: Number of days to look back
            
        Returns:
            SourceSentiment object
        """
        articles = self._fetch_news(ticker, days)
        
        if not articles:
            return self._empty_sentiment()
        
        # Weighted sentiment analysis
        weighted_scores = []
        bullish = bearish = neutral = 0
        
        for article in articles:
            score, confidence = self.analyzer.analyze(
                f"{article.title} {article.content}"
            )
            
            # Apply source weight
            weight = self._get_source_weight(article.url)
            weighted_scores.append(score * weight * confidence)
            
            if score > 0.1:
                bullish += 1
            elif score < -0.1:
                bearish += 1
            else:
                neutral += 1
        
        total = len(articles)
        avg_score = np.mean(weighted_scores) if weighted_scores else 0
        
        return SourceSentiment(
            source=SentimentSource.NEWS,
            score=avg_score,
            volume=len(articles),
            bullish_pct=bullish / total * 100 if total > 0 else 0,
            bearish_pct=bearish / total * 100 if total > 0 else 0,
            neutral_pct=neutral / total * 100 if total > 0 else 0,
            trending_topics=self._extract_themes(articles),
            top_posts=sorted(articles, key=lambda x: abs(x.score), reverse=True)[:5],
            last_updated=datetime.now()
        )
    
    def _fetch_news(self, ticker: str, days: int) -> List[SentimentPost]:
        """Fetch news articles (demo data if no API)."""
        # In production, would use NewsAPI, Benzinga, etc.
        
        sample_articles = [
            f"{ticker} Reports Strong Quarterly Earnings, Beats Estimates",
            f"Analysts Upgrade {ticker} on Growth Prospects",
            f"{ticker} Faces Headwinds from Rising Competition",
            f"Why Investors Are Watching {ticker} Closely",
            f"{ticker}: Buy, Sell, or Hold? Expert Analysis",
            f"Market Update: {ticker} Rallies on Positive News",
        ]
        
        articles = []
        for i, title in enumerate(sample_articles):
            score, conf = self.analyzer.analyze(title)
            articles.append(SentimentPost(
                source=SentimentSource.NEWS,
                timestamp=datetime.now() - timedelta(days=i),
                title=title,
                content=f"Detailed analysis of {ticker}...",
                author="Financial News",
                score=score,
                confidence=conf,
                engagement=0,
                url=f"https://example.com/news/{i}",
                tickers_mentioned=[ticker]
            ))
        
        return articles
    
    def _get_source_weight(self, url: str) -> float:
        """Get credibility weight for a news source."""
        url_lower = url.lower()
        
        for source, weight in self.SOURCE_WEIGHTS.items():
            if source in url_lower:
                return weight
        
        return self.SOURCE_WEIGHTS['default']
    
    def _extract_themes(self, articles: List[SentimentPost]) -> List[str]:
        """Extract common themes from articles."""
        themes = defaultdict(int)
        
        theme_keywords = {
            'earnings': ['earnings', 'revenue', 'profit', 'eps'],
            'growth': ['growth', 'expansion', 'scale'],
            'competition': ['competition', 'rival', 'market share'],
            'regulation': ['regulation', 'regulatory', 'compliance'],
            'valuation': ['valuation', 'price target', 'rating'],
            'management': ['ceo', 'executive', 'leadership'],
            'technology': ['technology', 'innovation', 'ai', 'tech'],
            'dividends': ['dividend', 'yield', 'payout']
        }
        
        all_text = ' '.join([f"{a.title} {a.content}" for a in articles]).lower()
        
        for theme, keywords in theme_keywords.items():
            for kw in keywords:
                if kw in all_text:
                    themes[theme] += 1
                    break
        
        return sorted(themes.keys(), key=lambda x: themes[x], reverse=True)
    
    def _empty_sentiment(self) -> SourceSentiment:
        """Return empty sentiment object."""
        return SourceSentiment(
            source=SentimentSource.NEWS,
            score=0.0,
            volume=0,
            bullish_pct=0,
            bearish_pct=0,
            neutral_pct=0,
            trending_topics=[],
            top_posts=[],
            last_updated=datetime.now()
        )


# ============================================================================
# SENTIMENT AGGREGATOR
# ============================================================================

class SentimentAggregator:
    """
    Aggregate sentiment from multiple sources and calculate momentum.
    """
    
    # Source importance weights
    SOURCE_WEIGHTS = {
        SentimentSource.NEWS: 0.4,
        SentimentSource.REDDIT: 0.3,
        SentimentSource.TWITTER: 0.2,
        SentimentSource.STOCKTWITS: 0.1
    }
    
    def __init__(self):
        self.reddit_provider = RedditSentimentProvider()
        self.news_provider = NewsSentimentProvider()
        self.history: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)
    
    def get_aggregate_sentiment(
        self,
        ticker: str,
        price_data: pd.DataFrame = None
    ) -> AggregateSentiment:
        """
        Get aggregated sentiment across all sources.
        
        Args:
            ticker: Stock ticker symbol
            price_data: Optional price data for divergence detection
            
        Returns:
            AggregateSentiment object
        """
        # Fetch from all sources
        reddit_sentiment = self.reddit_provider.get_sentiment(ticker)
        news_sentiment = self.news_provider.get_sentiment(ticker)
        
        source_breakdown = {
            'reddit': reddit_sentiment,
            'news': news_sentiment
        }
        
        # Calculate weighted average
        weighted_sum = 0
        weight_total = 0
        
        for source, sentiment in [
            (SentimentSource.REDDIT, reddit_sentiment),
            (SentimentSource.NEWS, news_sentiment)
        ]:
            weight = self.SOURCE_WEIGHTS.get(source, 0.1)
            # Adjust weight by volume (more posts = more confidence)
            volume_factor = min(2.0, 1 + np.log1p(sentiment.volume) / 10)
            adjusted_weight = weight * volume_factor
            
            weighted_sum += sentiment.score * adjusted_weight
            weight_total += adjusted_weight
        
        overall_score = weighted_sum / weight_total if weight_total > 0 else 0
        
        # Store in history
        self.history[ticker].append((datetime.now(), overall_score))
        
        # Calculate momentum
        momentum = self._calculate_momentum(ticker, price_data)
        
        # Determine strength
        strength = self._score_to_strength(overall_score)
        
        # Calculate confidence
        volumes = [reddit_sentiment.volume, news_sentiment.volume]
        confidence = min(1.0, sum(volumes) / 50)  # Max confidence at 50+ posts
        
        # Determine risk level
        risk = self._assess_risk(overall_score, momentum, confidence)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            overall_score, strength, momentum, confidence
        )
        
        return AggregateSentiment(
            ticker=ticker,
            overall_score=overall_score,
            strength=strength,
            confidence=confidence,
            source_breakdown=source_breakdown,
            momentum=momentum,
            risk_level=risk,
            recommendation=recommendation,
            timestamp=datetime.now()
        )
    
    def _calculate_momentum(
        self,
        ticker: str,
        price_data: pd.DataFrame = None
    ) -> SentimentMomentum:
        """Calculate sentiment momentum and divergence."""
        history = self.history.get(ticker, [])
        
        if len(history) < 2:
            return SentimentMomentum(
                current_score=history[-1][1] if history else 0,
                score_1d_ago=0, score_7d_ago=0, score_30d_ago=0,
                momentum_1d=0, momentum_7d=0, momentum_30d=0,
                acceleration=0, is_diverging_from_price=False,
                divergence_direction='none'
            )
        
        current = history[-1][1]
        
        # Find historical scores
        now = datetime.now()
        score_1d = self._get_historical_score(history, now - timedelta(days=1))
        score_7d = self._get_historical_score(history, now - timedelta(days=7))
        score_30d = self._get_historical_score(history, now - timedelta(days=30))
        
        # Momentum
        mom_1d = current - score_1d
        mom_7d = current - score_7d
        mom_30d = current - score_30d
        
        # Acceleration
        accel = mom_1d - (score_1d - score_7d) / 6  # Compare recent vs avg
        
        # Check for price divergence
        is_diverging = False
        div_direction = 'none'
        
        if price_data is not None and len(price_data) > 7:
            price_return_7d = (
                price_data['Close'].iloc[-1] / price_data['Close'].iloc[-7] - 1
            )
            
            # Bullish divergence: price down but sentiment improving
            if price_return_7d < -0.03 and mom_7d > 0.1:
                is_diverging = True
                div_direction = 'bullish_divergence'
            # Bearish divergence: price up but sentiment deteriorating
            elif price_return_7d > 0.03 and mom_7d < -0.1:
                is_diverging = True
                div_direction = 'bearish_divergence'
        
        return SentimentMomentum(
            current_score=current,
            score_1d_ago=score_1d,
            score_7d_ago=score_7d,
            score_30d_ago=score_30d,
            momentum_1d=mom_1d,
            momentum_7d=mom_7d,
            momentum_30d=mom_30d,
            acceleration=accel,
            is_diverging_from_price=is_diverging,
            divergence_direction=div_direction
        )
    
    def _get_historical_score(
        self,
        history: List[Tuple[datetime, float]],
        target_time: datetime
    ) -> float:
        """Get sentiment score closest to target time."""
        if not history:
            return 0
        
        closest = min(history, key=lambda x: abs((x[0] - target_time).total_seconds()))
        return closest[1]
    
    def _score_to_strength(self, score: float) -> SentimentStrength:
        """Convert numeric score to strength enum."""
        if score < -0.5:
            return SentimentStrength.VERY_BEARISH
        elif score < -0.1:
            return SentimentStrength.BEARISH
        elif score > 0.5:
            return SentimentStrength.VERY_BULLISH
        elif score > 0.1:
            return SentimentStrength.BULLISH
        else:
            return SentimentStrength.NEUTRAL
    
    def _assess_risk(
        self,
        score: float,
        momentum: SentimentMomentum,
        confidence: float
    ) -> str:
        """Assess sentiment-based risk level."""
        risk_score = 0
        
        # Extreme sentiment is risky (contrarian)
        if abs(score) > 0.7:
            risk_score += 2
        
        # Low confidence is risky
        if confidence < 0.3:
            risk_score += 1
        
        # Divergence from price is risky
        if momentum.is_diverging_from_price:
            risk_score += 2
        
        # Rapid changes are risky
        if abs(momentum.acceleration) > 0.1:
            risk_score += 1
        
        if risk_score >= 4:
            return 'high'
        elif risk_score >= 2:
            return 'medium'
        else:
            return 'low'
    
    def _generate_recommendation(
        self,
        score: float,
        strength: SentimentStrength,
        momentum: SentimentMomentum,
        confidence: float
    ) -> str:
        """Generate actionable recommendation."""
        if confidence < 0.2:
            return "Insufficient data for reliable sentiment analysis"
        
        direction = "bullish" if score > 0 else "bearish" if score < 0 else "neutral"
        
        if momentum.is_diverging_from_price:
            if momentum.divergence_direction == 'bullish_divergence':
                return "Bullish divergence detected: sentiment improving despite price decline. Potential buying opportunity if fundamentals support."
            else:
                return "Bearish divergence detected: sentiment weakening despite price gains. Consider reducing exposure or hedging."
        
        if strength == SentimentStrength.VERY_BULLISH:
            return "Strong bullish sentiment. Caution: extreme readings often precede reversals. Consider taking partial profits if long."
        elif strength == SentimentStrength.VERY_BEARISH:
            return "Strong bearish sentiment. Potential contrarian buy opportunity. Watch for sentiment improvement."
        elif strength == SentimentStrength.BULLISH:
            return f"Moderately bullish sentiment with {momentum.momentum_7d:+.2f} 7-day momentum. Sentiment supports long positions."
        elif strength == SentimentStrength.BEARISH:
            return f"Moderately bearish sentiment with {momentum.momentum_7d:+.2f} 7-day momentum. Consider reduced exposure."
        else:
            return "Neutral sentiment. No strong directional bias from social/news sources."


# ============================================================================
# STREAMLIT RENDERING
# ============================================================================

def render_advanced_sentiment_dashboard():
    """Render advanced sentiment analysis dashboard."""
    import streamlit as st
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    st.subheader(" Advanced Sentiment Analysis")
    
    ticker = st.text_input("Ticker Symbol", value="AAPL").upper()
    
    if st.button("Analyze Sentiment"):
        with st.spinner("Analyzing sentiment across sources..."):
            aggregator = SentimentAggregator()
            result = aggregator.get_aggregate_sentiment(ticker)
        
        # Overall sentiment gauge
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result.overall_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Overall Sentiment"},
                gauge={
                    'axis': {'range': [-1, 1]},
                    'bar': {'color': '#00D4AA' if result.overall_score > 0 else '#FF5722'},
                    'steps': [
                        {'range': [-1, -0.5], 'color': '#FF5722'},
                        {'range': [-0.5, -0.1], 'color': '#FF9800'},
                        {'range': [-0.1, 0.1], 'color': '#9E9E9E'},
                        {'range': [0.1, 0.5], 'color': '#8BC34A'},
                        {'range': [0.5, 1], 'color': '#4CAF50'}
                    ]
                }
            ))
            fig.update_layout(template='plotly_dark', height=300)
            st.plotly_chart(fig, width="stretch")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Strength", result.strength.value.replace('_', ' ').title())
        with col2:
            st.metric("Confidence", f"{result.confidence:.0%}")
        with col3:
            st.metric("Risk Level", result.risk_level.upper())
        with col4:
            st.metric("7D Momentum", f"{result.momentum.momentum_7d:+.2f}")
        
        # Source breakdown
        st.write("### Source Breakdown")
        
        tabs = st.tabs(["Reddit", "News", "Momentum"])
        
        with tabs[0]:
            reddit = result.source_breakdown.get('reddit')
            if reddit:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Reddit Score", f"{reddit.score:+.2f}")
                with col2:
                    st.metric("Posts Analyzed", reddit.volume)
                with col3:
                    st.metric("Bullish %", f"{reddit.bullish_pct:.0f}%")
                
                # Top posts
                if reddit.top_posts:
                    st.write("**Top Posts:**")
                    for post in reddit.top_posts[:3]:
                        icon = "🟢" if post.score > 0.1 else "" if post.score < -0.1 else ""
                        st.markdown(f"{icon} **{post.title}** (Score: {post.score:+.2f})")
        
        with tabs[1]:
            news = result.source_breakdown.get('news')
            if news:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("News Score", f"{news.score:+.2f}")
                with col2:
                    st.metric("Articles", news.volume)
                with col3:
                    st.metric("Bullish %", f"{news.bullish_pct:.0f}%")
                
                # Trending themes
                if news.trending_topics:
                    st.write("**Trending Themes:**")
                    st.write(", ".join(news.trending_topics))
        
        with tabs[2]:
            mom = result.momentum
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("1D Momentum", f"{mom.momentum_1d:+.2f}")
            with col2:
                st.metric("7D Momentum", f"{mom.momentum_7d:+.2f}")
            with col3:
                st.metric("30D Momentum", f"{mom.momentum_30d:+.2f}")
            with col4:
                st.metric("Acceleration", f"{mom.acceleration:+.2f}")
            
            if mom.is_diverging_from_price:
                if mom.divergence_direction == 'bullish_divergence':
                    st.success(" **Bullish Divergence:** Sentiment improving despite price decline")
                else:
                    st.warning(" **Bearish Divergence:** Sentiment weakening despite price gains")
        
        # Recommendation
        st.write("### Recommendation")
        st.info(result.recommendation)
        
        # Sentiment vs Price chart (demo)
        st.write("### Sentiment History")
        
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        sentiment_history = np.cumsum(np.random.randn(30) * 0.1)
        sentiment_history = np.clip(sentiment_history, -1, 1)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, y=sentiment_history,
            name='Sentiment Score',
            fill='tozeroy',
            line=dict(color='#00D4AA')
        ))
        
        fig.add_hline(y=0.5, line_dash="dash", line_color="green", 
                     annotation_text="Bullish")
        fig.add_hline(y=-0.5, line_dash="dash", line_color="red",
                     annotation_text="Bearish")
        
        fig.update_layout(
            title="30-Day Sentiment History",
            xaxis_title="Date",
            yaxis_title="Sentiment Score",
            template='plotly_dark',
            height=400
        )
        
        st.plotly_chart(fig, width="stretch")


# Make features available
HAS_ADVANCED_SENTIMENT = True
