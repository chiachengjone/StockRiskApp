"""
Sentiment Service - News & Social Media Sentiment Analysis
============================================================
NLP-based sentiment scoring for market sentiment analysis.

Features:
- News article scraping and aggregation
- Sentiment scoring using TextBlob/VADER
- Social media sentiment (Twitter/Reddit)
- Whale tracking and institutional activity
- Sentiment-adjusted risk metrics

Author: Stock Risk App | Feb 2026
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
import re
import json
from collections import defaultdict

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Sentiment thresholds
SENTIMENT_THRESHOLDS = {
    'very_negative': -0.6,
    'negative': -0.2,
    'neutral_low': -0.05,
    'neutral_high': 0.05,
    'positive': 0.2,
    'very_positive': 0.6
}

# Whale tracking thresholds
WHALE_THRESHOLDS = {
    'large_trade_usd': 1_000_000,  # $1M+
    'institutional_holding_pct': 0.01,  # 1% ownership
    'insider_trade_usd': 100_000  # $100K+
}

# Sentiment decay (older news has less impact)
SENTIMENT_DECAY_DAYS = 7
SENTIMENT_DECAY_RATE = 0.3  # Per day decay rate


# ============================================================================
# DATA CLASSES
# ============================================================================

class SentimentLabel(Enum):
    """Sentiment classification labels."""
    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"


@dataclass
class NewsArticle:
    """Represents a news article with sentiment."""
    title: str
    source: str
    published_at: datetime
    url: str = ""
    summary: str = ""
    sentiment_score: float = 0.0
    sentiment_label: SentimentLabel = SentimentLabel.NEUTRAL
    tickers: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    impact_score: float = 0.0  # Estimated market impact
    
    def to_dict(self) -> Dict:
        return {
            'title': self.title,
            'source': self.source,
            'published_at': self.published_at.isoformat(),
            'url': self.url,
            'sentiment_score': self.sentiment_score,
            'sentiment_label': self.sentiment_label.value,
            'tickers': self.tickers,
            'impact_score': self.impact_score
        }


@dataclass
class WhaleActivity:
    """Represents large institutional/whale trading activity."""
    ticker: str
    activity_type: str  # 'buy', 'sell', 'filing'
    value_usd: float
    shares: int = 0
    entity_name: str = ""
    filing_date: datetime = None
    source: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'ticker': self.ticker,
            'activity_type': self.activity_type,
            'value_usd': self.value_usd,
            'shares': self.shares,
            'entity_name': self.entity_name,
            'filing_date': self.filing_date.isoformat() if self.filing_date else None,
            'source': self.source
        }


@dataclass
class SentimentResult:
    """Aggregated sentiment analysis result."""
    ticker: str
    overall_score: float
    overall_label: SentimentLabel
    news_count: int
    positive_count: int
    negative_count: int
    neutral_count: int
    trending_score: float  # Momentum in sentiment
    whale_activity_score: float
    sentiment_var_adjustment: float  # Risk adjustment factor
    articles: List[NewsArticle] = field(default_factory=list)
    whale_activities: List[WhaleActivity] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'ticker': self.ticker,
            'overall_score': self.overall_score,
            'overall_label': self.overall_label.value,
            'news_count': self.news_count,
            'positive_count': self.positive_count,
            'negative_count': self.negative_count,
            'neutral_count': self.neutral_count,
            'trending_score': self.trending_score,
            'whale_activity_score': self.whale_activity_score,
            'sentiment_var_adjustment': self.sentiment_var_adjustment
        }


# ============================================================================
# SENTIMENT ANALYZERS
# ============================================================================

class TextBlobAnalyzer:
    """Sentiment analysis using TextBlob."""
    
    def __init__(self):
        self._textblob = None
        self._available = self._check_availability()
    
    def _check_availability(self) -> bool:
        try:
            from textblob import TextBlob
            self._textblob = TextBlob
            return True
        except ImportError:
            logger.warning("TextBlob not installed. Install with: pip install textblob")
            return False
    
    def analyze(self, text: str) -> Tuple[float, float]:
        """
        Analyze text sentiment.
        
        Returns:
            Tuple of (polarity, subjectivity)
            Polarity: -1 (negative) to 1 (positive)
            Subjectivity: 0 (objective) to 1 (subjective)
        """
        if not self._available:
            return 0.0, 0.5
        
        try:
            blob = self._textblob(text)
            return blob.sentiment.polarity, blob.sentiment.subjectivity
        except Exception as e:
            logger.error(f"TextBlob analysis error: {e}")
            return 0.0, 0.5
    
    @property
    def is_available(self) -> bool:
        return self._available


class VADERAnalyzer:
    """Sentiment analysis using VADER (Valence Aware Dictionary)."""
    
    def __init__(self):
        self._vader = None
        self._available = self._check_availability()
    
    def _check_availability(self) -> bool:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._vader = SentimentIntensityAnalyzer()
            return True
        except ImportError:
            logger.warning("VADER not installed. Install with: pip install vaderSentiment")
            return False
    
    def analyze(self, text: str) -> Dict[str, float]:
        """
        Analyze text sentiment using VADER.
        
        Returns:
            Dictionary with 'neg', 'neu', 'pos', 'compound' scores
            Compound: -1 (negative) to 1 (positive)
        """
        if not self._available:
            return {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
        
        try:
            return self._vader.polarity_scores(text)
        except Exception as e:
            logger.error(f"VADER analysis error: {e}")
            return {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    
    @property
    def is_available(self) -> bool:
        return self._available


class FinancialSentimentAnalyzer:
    """
    Financial domain-specific sentiment analyzer.
    
    Uses custom lexicon for financial terms and
    combines multiple analyzers for robust scoring.
    """
    
    # Financial-specific positive words
    POSITIVE_WORDS = {
        'upgrade', 'outperform', 'beat', 'exceeds', 'growth', 'profit',
        'bullish', 'rally', 'surge', 'gain', 'breakout', 'upside',
        'momentum', 'strong', 'solid', 'robust', 'innovative', 'leader',
        'dividend', 'buyback', 'acquisition', 'partnership', 'expansion',
        'optimistic', 'confident', 'record', 'breakthrough', 'positive'
    }
    
    # Financial-specific negative words
    NEGATIVE_WORDS = {
        'downgrade', 'underperform', 'miss', 'decline', 'loss', 'drop',
        'bearish', 'crash', 'plunge', 'sell-off', 'downside', 'risk',
        'warning', 'weak', 'struggling', 'lawsuit', 'investigation',
        'recall', 'bankruptcy', 'debt', 'layoffs', 'restructuring',
        'disappointing', 'concern', 'fraud', 'scandal', 'negative'
    }
    
    # Intensity modifiers
    INTENSIFIERS = {'very', 'extremely', 'significantly', 'massively', 'hugely'}
    DIMINISHERS = {'slightly', 'somewhat', 'marginally', 'modestly'}
    
    def __init__(self):
        self.textblob = TextBlobAnalyzer()
        self.vader = VADERAnalyzer()
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze text with financial domain knowledge.
        
        Returns comprehensive sentiment analysis with multiple scores.
        """
        text_lower = text.lower()
        words = set(re.findall(r'\b\w+\b', text_lower))
        
        # Count domain-specific words
        positive_matches = words & self.POSITIVE_WORDS
        negative_matches = words & self.NEGATIVE_WORDS
        
        # Calculate domain score
        pos_count = len(positive_matches)
        neg_count = len(negative_matches)
        total_matches = pos_count + neg_count
        
        if total_matches > 0:
            domain_score = (pos_count - neg_count) / total_matches
        else:
            domain_score = 0.0
        
        # Check for intensifiers/diminishers
        has_intensifier = bool(words & self.INTENSIFIERS)
        has_diminisher = bool(words & self.DIMINISHERS)
        
        if has_intensifier:
            domain_score *= 1.3
        if has_diminisher:
            domain_score *= 0.7
        
        # Get general NLP scores
        tb_polarity, tb_subjectivity = self.textblob.analyze(text)
        vader_scores = self.vader.analyze(text)
        
        # Combine scores with weighting
        # Domain-specific knowledge gets higher weight for financial text
        combined_score = (
            0.4 * domain_score +
            0.3 * tb_polarity +
            0.3 * vader_scores['compound']
        )
        
        # Clamp to [-1, 1]
        combined_score = max(-1.0, min(1.0, combined_score))
        
        # Determine label
        label = self._score_to_label(combined_score)
        
        return {
            'score': combined_score,
            'label': label,
            'domain_score': domain_score,
            'textblob_score': tb_polarity,
            'vader_score': vader_scores['compound'],
            'positive_words': list(positive_matches),
            'negative_words': list(negative_matches),
            'subjectivity': tb_subjectivity
        }
    
    def _score_to_label(self, score: float) -> SentimentLabel:
        """Convert numeric score to sentiment label."""
        if score <= SENTIMENT_THRESHOLDS['very_negative']:
            return SentimentLabel.VERY_NEGATIVE
        elif score <= SENTIMENT_THRESHOLDS['negative']:
            return SentimentLabel.NEGATIVE
        elif score <= SENTIMENT_THRESHOLDS['neutral_high']:
            return SentimentLabel.NEUTRAL
        elif score <= SENTIMENT_THRESHOLDS['positive']:
            return SentimentLabel.POSITIVE
        else:
            return SentimentLabel.VERY_POSITIVE


# ============================================================================
# NEWS FETCHERS
# ============================================================================

class NewsFetcher:
    """
    Aggregates news from multiple sources.
    
    Supported sources:
    - Polygon.io (if configured)
    - Alpaca (if configured)
    - Yahoo Finance RSS
    """
    
    def __init__(
        self,
        polygon_key: str = "",
        alpaca_key: str = "",
        alpaca_secret: str = ""
    ):
        self.polygon_key = polygon_key
        self.alpaca_key = alpaca_key
        self.alpaca_secret = alpaca_secret
        self.logger = logging.getLogger(__name__)
    
    def fetch_news(
        self,
        ticker: str = None,
        days_back: int = 7,
        limit: int = 50
    ) -> List[Dict]:
        """
        Fetch news from all available sources.
        
        Args:
            ticker: Stock symbol (optional)
            days_back: Number of days to look back
            limit: Maximum articles per source
        
        Returns:
            List of news article dictionaries
        """
        all_news = []
        
        # Try Polygon
        if self.polygon_key:
            try:
                from data_sources.polygon_provider import PolygonProvider
                polygon = PolygonProvider(api_key=self.polygon_key)
                
                start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
                news = polygon.get_news(
                    ticker=ticker,
                    limit=limit,
                    published_after=start_date
                )
                
                for article in news:
                    all_news.append({
                        'title': article.get('title', ''),
                        'summary': article.get('description', ''),
                        'source': article.get('source', 'polygon'),
                        'published_at': article.get('published_utc', ''),
                        'url': article.get('article_url', ''),
                        'tickers': article.get('tickers', []),
                        'provider': 'polygon'
                    })
                    
            except Exception as e:
                self.logger.warning(f"Polygon news fetch failed: {e}")
        
        # Try Alpaca
        if self.alpaca_key and self.alpaca_secret:
            try:
                from data_sources.alpaca_provider import AlpacaProvider
                alpaca = AlpacaProvider(
                    api_key=self.alpaca_key,
                    api_secret=self.alpaca_secret
                )
                
                symbols = [ticker] if ticker else None
                start = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
                news = alpaca.get_news(
                    symbols=symbols,
                    limit=limit,
                    start=start
                )
                
                for article in news:
                    all_news.append({
                        'title': article.get('headline', ''),
                        'summary': article.get('summary', ''),
                        'source': article.get('source', 'alpaca'),
                        'published_at': article.get('created_at', ''),
                        'url': article.get('url', ''),
                        'tickers': article.get('symbols', []),
                        'provider': 'alpaca'
                    })
                    
            except Exception as e:
                self.logger.warning(f"Alpaca news fetch failed: {e}")
        
        # Yahoo Finance fallback (basic RSS)
        if not all_news:
            try:
                all_news.extend(self._fetch_yahoo_news(ticker, limit))
            except Exception as e:
                self.logger.warning(f"Yahoo news fetch failed: {e}")
        
        return all_news
    
    def _fetch_yahoo_news(self, ticker: str, limit: int = 10) -> List[Dict]:
        """Fetch news from Yahoo Finance."""
        import yfinance as yf
        
        try:
            stock = yf.Ticker(ticker)
            news = stock.news[:limit] if hasattr(stock, 'news') else []
            
            return [{
                'title': item.get('title', ''),
                'summary': '',
                'source': item.get('publisher', 'yahoo'),
                'published_at': datetime.fromtimestamp(
                    item.get('providerPublishTime', 0)
                ).isoformat(),
                'url': item.get('link', ''),
                'tickers': [ticker],
                'provider': 'yahoo'
            } for item in news]
            
        except Exception as e:
            self.logger.warning(f"Yahoo news error: {e}")
            return []


# ============================================================================
# WHALE TRACKER
# ============================================================================

class WhaleTracker:
    """
    Track institutional and whale activity.
    
    Monitors:
    - Large block trades
    - 13F filings
    - Insider transactions
    - Options flow
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def get_institutional_holdings(self, ticker: str) -> Dict:
        """Get institutional holdings for a ticker."""
        import yfinance as yf
        
        try:
            stock = yf.Ticker(ticker)
            holders = stock.institutional_holders
            
            if holders is None or holders.empty:
                return {'ticker': ticker, 'holders': [], 'total_pct': 0}
            
            holders_list = []
            total_pct = 0
            
            for _, row in holders.head(10).iterrows():
                pct = row.get('% Out', row.get('pctHeld', 0))
                if isinstance(pct, str):
                    pct = float(pct.replace('%', '')) / 100
                
                holders_list.append({
                    'name': row.get('Holder', ''),
                    'shares': int(row.get('Shares', 0)),
                    'value': float(row.get('Value', 0)),
                    'pct_out': float(pct) if pct else 0
                })
                total_pct += float(pct) if pct else 0
            
            return {
                'ticker': ticker,
                'holders': holders_list,
                'total_pct': total_pct,
                'top_holder': holders_list[0] if holders_list else None
            }
            
        except Exception as e:
            self.logger.warning(f"Institutional holdings error for {ticker}: {e}")
            return {'ticker': ticker, 'holders': [], 'total_pct': 0}
    
    def get_insider_transactions(self, ticker: str) -> List[Dict]:
        """Get recent insider transactions."""
        import yfinance as yf
        
        try:
            stock = yf.Ticker(ticker)
            insiders = stock.insider_transactions
            
            if insiders is None or insiders.empty:
                return []
            
            transactions = []
            for _, row in insiders.head(20).iterrows():
                transactions.append({
                    'insider': row.get('Insider Trading', row.get('insiderName', '')),
                    'relation': row.get('Relationship', row.get('insiderRelation', '')),
                    'transaction': row.get('Transaction', row.get('transactionType', '')),
                    'shares': int(row.get('Shares', row.get('shares', 0))),
                    'value': float(row.get('Value', row.get('value', 0))),
                    'date': str(row.get('Date', row.get('transactionDate', '')))
                })
            
            return transactions
            
        except Exception as e:
            self.logger.warning(f"Insider transactions error for {ticker}: {e}")
            return []
    
    def calculate_whale_score(
        self,
        ticker: str,
        institutional_data: Dict = None,
        insider_data: List[Dict] = None
    ) -> float:
        """
        Calculate whale activity score.
        
        Higher scores indicate more bullish institutional activity.
        
        Returns:
            Score from -1 (bearish) to 1 (bullish)
        """
        score = 0.0
        
        # Get data if not provided
        if institutional_data is None:
            institutional_data = self.get_institutional_holdings(ticker)
        if insider_data is None:
            insider_data = self.get_insider_transactions(ticker)
        
        # Institutional holdings factor
        total_inst_pct = institutional_data.get('total_pct', 0)
        if total_inst_pct > 0.7:
            score += 0.3  # High institutional ownership is generally positive
        elif total_inst_pct > 0.5:
            score += 0.15
        
        # Insider transaction factor
        buy_value = 0
        sell_value = 0
        
        for tx in insider_data[:10]:  # Recent transactions
            tx_type = tx.get('transaction', '').lower()
            value = tx.get('value', 0)
            
            if 'buy' in tx_type or 'purchase' in tx_type:
                buy_value += value
            elif 'sell' in tx_type or 'sale' in tx_type:
                sell_value += value
        
        total_value = buy_value + sell_value
        if total_value > 0:
            net_ratio = (buy_value - sell_value) / total_value
            score += net_ratio * 0.5  # Weight insider activity heavily
        
        # Clamp to [-1, 1]
        return max(-1.0, min(1.0, score))


# ============================================================================
# SENTIMENT SERVICE (MAIN CLASS)
# ============================================================================

class SentimentService:
    """
    Main service for sentiment analysis and risk adjustment.
    
    Usage:
        service = SentimentService(polygon_key='...')
        
        # Get sentiment for a ticker
        result = service.analyze_ticker('AAPL')
        
        # Get sentiment VaR adjustment
        var_adj = service.get_sentiment_var_adjustment('AAPL', base_var=0.02)
    """
    
    def __init__(
        self,
        polygon_key: str = "",
        alpaca_key: str = "",
        alpaca_secret: str = ""
    ):
        self.analyzer = FinancialSentimentAnalyzer()
        self.news_fetcher = NewsFetcher(
            polygon_key=polygon_key,
            alpaca_key=alpaca_key,
            alpaca_secret=alpaca_secret
        )
        self.whale_tracker = WhaleTracker()
        self.logger = logging.getLogger(__name__)
        
        # Cache for sentiment results
        self._cache = {}
        self._cache_ttl = timedelta(minutes=30)
    
    def analyze_ticker(
        self,
        ticker: str,
        days_back: int = 7,
        include_whale_data: bool = True
    ) -> SentimentResult:
        """
        Perform comprehensive sentiment analysis for a ticker.
        
        Args:
            ticker: Stock symbol
            days_back: Days of news to analyze
            include_whale_data: Include institutional/insider data
        
        Returns:
            SentimentResult with aggregated analysis
        """
        # Check cache
        cache_key = f"{ticker}_{days_back}"
        if cache_key in self._cache:
            cached, timestamp = self._cache[cache_key]
            if datetime.now() - timestamp < self._cache_ttl:
                return cached
        
        # Fetch news
        news_items = self.news_fetcher.fetch_news(
            ticker=ticker,
            days_back=days_back,
            limit=50
        )
        
        # Analyze each article
        articles = []
        sentiment_scores = []
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        now = datetime.now()
        
        for item in news_items:
            # Combine title and summary for analysis
            text = f"{item.get('title', '')} {item.get('summary', '')}"
            
            if not text.strip():
                continue
            
            # Analyze sentiment
            analysis = self.analyzer.analyze(text)
            score = analysis['score']
            label = analysis['label']
            
            # Apply time decay
            try:
                pub_date = datetime.fromisoformat(
                    item.get('published_at', '').replace('Z', '+00:00')
                )
                days_old = (now - pub_date.replace(tzinfo=None)).days
            except (ValueError, TypeError):
                days_old = days_back
            
            decay_factor = np.exp(-SENTIMENT_DECAY_RATE * days_old)
            weighted_score = score * decay_factor
            
            # Create article object
            article = NewsArticle(
                title=item.get('title', ''),
                source=item.get('source', ''),
                published_at=now - timedelta(days=days_old),
                url=item.get('url', ''),
                summary=item.get('summary', '')[:500],
                sentiment_score=score,
                sentiment_label=label,
                tickers=item.get('tickers', [ticker]),
                impact_score=abs(weighted_score)
            )
            articles.append(article)
            sentiment_scores.append(weighted_score)
            
            # Count by sentiment
            if score > SENTIMENT_THRESHOLDS['neutral_high']:
                positive_count += 1
            elif score < SENTIMENT_THRESHOLDS['neutral_low']:
                negative_count += 1
            else:
                neutral_count += 1
        
        # Calculate overall sentiment
        if sentiment_scores:
            overall_score = float(np.mean(sentiment_scores))
        else:
            overall_score = 0.0
        
        overall_label = self.analyzer._score_to_label(overall_score)
        
        # Calculate trending (sentiment momentum)
        if len(sentiment_scores) >= 2:
            recent = sentiment_scores[:len(sentiment_scores)//2]
            older = sentiment_scores[len(sentiment_scores)//2:]
            trending_score = float(np.mean(recent) - np.mean(older))
        else:
            trending_score = 0.0
        
        # Whale activity
        whale_activities = []
        whale_score = 0.0
        
        if include_whale_data:
            whale_score = self.whale_tracker.calculate_whale_score(ticker)
            
            # Get recent insider transactions as activities
            insider_txs = self.whale_tracker.get_insider_transactions(ticker)
            for tx in insider_txs[:5]:
                activity = WhaleActivity(
                    ticker=ticker,
                    activity_type='buy' if 'buy' in tx.get('transaction', '').lower() else 'sell',
                    value_usd=tx.get('value', 0),
                    shares=tx.get('shares', 0),
                    entity_name=tx.get('insider', ''),
                    source='insider'
                )
                whale_activities.append(activity)
        
        # Calculate VaR adjustment
        var_adjustment = self._calculate_var_adjustment(
            overall_score, trending_score, whale_score
        )
        
        # Create result
        result = SentimentResult(
            ticker=ticker,
            overall_score=overall_score,
            overall_label=overall_label,
            news_count=len(articles),
            positive_count=positive_count,
            negative_count=negative_count,
            neutral_count=neutral_count,
            trending_score=trending_score,
            whale_activity_score=whale_score,
            sentiment_var_adjustment=var_adjustment,
            articles=articles,
            whale_activities=whale_activities
        )
        
        # Cache result
        self._cache[cache_key] = (result, datetime.now())
        
        return result
    
    def _calculate_var_adjustment(
        self,
        sentiment_score: float,
        trending_score: float,
        whale_score: float
    ) -> float:
        """
        Calculate VaR adjustment factor based on sentiment.
        
        Negative sentiment increases VaR, positive decreases it.
        
        Returns:
            Multiplier for VaR (e.g., 1.2 means 20% higher VaR)
        """
        # Base adjustment from sentiment (max ±20%)
        sentiment_adj = -sentiment_score * 0.2
        
        # Trending adjustment (max ±10%)
        # Worsening sentiment adds more risk
        trend_adj = -trending_score * 0.1
        
        # Whale adjustment (max ±10%)
        whale_adj = -whale_score * 0.1
        
        # Combine adjustments
        total_adj = sentiment_adj + trend_adj + whale_adj
        
        # Convert to multiplier (centered at 1.0)
        multiplier = 1.0 + total_adj
        
        # Clamp to reasonable range [0.7, 1.5]
        return max(0.7, min(1.5, multiplier))
    
    def get_sentiment_var_adjustment(
        self,
        ticker: str,
        base_var: float,
        confidence: float = 0.95
    ) -> Dict:
        """
        Get sentiment-adjusted VaR.
        
        Args:
            ticker: Stock symbol
            base_var: Base VaR value (as decimal, e.g., 0.02 for 2%)
            confidence: VaR confidence level
        
        Returns:
            Dictionary with adjusted VaR and components
        """
        sentiment = self.analyze_ticker(ticker)
        
        adjusted_var = base_var * sentiment.sentiment_var_adjustment
        
        return {
            'ticker': ticker,
            'base_var': base_var,
            'adjusted_var': adjusted_var,
            'adjustment_factor': sentiment.sentiment_var_adjustment,
            'sentiment_score': sentiment.overall_score,
            'sentiment_label': sentiment.overall_label.value,
            'confidence': confidence,
            'news_count': sentiment.news_count,
            'whale_score': sentiment.whale_activity_score
        }
    
    def analyze_portfolio(
        self,
        tickers: List[str],
        weights: Dict[str, float] = None
    ) -> Dict:
        """
        Analyze sentiment for entire portfolio.
        
        Args:
            tickers: List of ticker symbols
            weights: Optional portfolio weights
        
        Returns:
            Portfolio-level sentiment analysis
        """
        if weights is None:
            weights = {t: 1.0/len(tickers) for t in tickers}
        
        results = {}
        weighted_score = 0.0
        
        for ticker in tickers:
            result = self.analyze_ticker(ticker)
            results[ticker] = result
            weighted_score += result.overall_score * weights.get(ticker, 0)
        
        # Portfolio-level metrics
        overall_label = self.analyzer._score_to_label(weighted_score)
        
        # Aggregate news
        all_articles = []
        for result in results.values():
            all_articles.extend(result.articles[:5])
        
        # Sort by impact
        all_articles.sort(key=lambda x: x.impact_score, reverse=True)
        
        return {
            'portfolio_sentiment': weighted_score,
            'portfolio_label': overall_label.value,
            'ticker_sentiments': {t: r.to_dict() for t, r in results.items()},
            'top_articles': [a.to_dict() for a in all_articles[:10]],
            'bullish_tickers': [t for t, r in results.items() if r.overall_score > 0.2],
            'bearish_tickers': [t for t, r in results.items() if r.overall_score < -0.2],
            'portfolio_var_adjustment': np.mean([
                r.sentiment_var_adjustment * weights.get(t, 0)
                for t, r in results.items()
            ])
        }
    
    def get_trending_sentiment(
        self,
        ticker: str,
        lookback_days: int = 30,
        window: int = 7
    ) -> pd.DataFrame:
        """
        Calculate rolling sentiment over time.
        
        Args:
            ticker: Stock symbol
            lookback_days: Total days to analyze
            window: Rolling window size
        
        Returns:
            DataFrame with daily sentiment scores
        """
        # Fetch extended news history
        news_items = self.news_fetcher.fetch_news(
            ticker=ticker,
            days_back=lookback_days,
            limit=200
        )
        
        # Group by date
        daily_scores = defaultdict(list)
        now = datetime.now()
        
        for item in news_items:
            text = f"{item.get('title', '')} {item.get('summary', '')}"
            analysis = self.analyzer.analyze(text)
            
            try:
                pub_date = datetime.fromisoformat(
                    item.get('published_at', '').replace('Z', '+00:00')
                ).date()
            except (ValueError, TypeError):
                continue
            
            daily_scores[pub_date].append(analysis['score'])
        
        # Create time series
        dates = pd.date_range(
            end=now.date(),
            periods=lookback_days,
            freq='D'
        )
        
        scores = []
        for date in dates:
            day_scores = daily_scores.get(date.date(), [])
            if day_scores:
                scores.append(np.mean(day_scores))
            else:
                scores.append(np.nan)
        
        df = pd.DataFrame({
            'date': dates,
            'sentiment': scores
        }).set_index('date')
        
        # Forward fill missing values
        df['sentiment'] = df['sentiment'].fillna(method='ffill')
        
        # Calculate rolling average
        df['sentiment_ma'] = df['sentiment'].rolling(window=window).mean()
        
        return df


# ============================================================================
# STREAMLIT COMPONENTS
# ============================================================================

def display_sentiment_dashboard(
    service: SentimentService,
    ticker: str,
    container=None
):
    """
    Display sentiment dashboard in Streamlit.
    
    Args:
        service: SentimentService instance
        ticker: Stock symbol
        container: Optional Streamlit container
    """
    import streamlit as st
    
    result = service.analyze_ticker(ticker)
    
    with container if container else st:
        # Header metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            color = "[+]" if result.overall_score > 0 else "[-]" if result.overall_score < 0 else "[~]"
            st.metric(
                "Overall Sentiment",
                f"{color} {result.overall_label.value.replace('_', ' ').title()}"
            )
        
        with col2:
            st.metric("Score", f"{result.overall_score:.2f}")
        
        with col3:
            trend_icon = "UP" if result.trending_score > 0 else "DOWN" if result.trending_score < 0 else "NEUTRAL"
            st.metric("Trend", f"{trend_icon} {result.trending_score:.2f}")
        
        with col4:
            st.metric("VaR Adjustment", f"{(result.sentiment_var_adjustment-1)*100:+.1f}%")
        
        # News breakdown
        st.subheader("News Sentiment")
        cols = st.columns(3)
        cols[0].metric("Positive", result.positive_count, delta_color="normal")
        cols[1].metric("Neutral", result.neutral_count)
        cols[2].metric("Negative", result.negative_count, delta_color="inverse")
        
        # Top articles
        if result.articles:
            st.subheader("Recent News")
            for article in result.articles[:5]:
                sentiment_emoji = "[+]" if article.sentiment_score > 0.2 else "[-]" if article.sentiment_score < -0.2 else "[~]"
                st.markdown(f"{sentiment_emoji} **{article.title}**")
                st.caption(f"{article.source} • Score: {article.sentiment_score:.2f}")
