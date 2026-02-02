"""
Sentiment Analysis Service
==========================
Sentiment analysis and Sentiment VaR calculations.

Provides sentiment scoring and VaR adjustments based on market sentiment.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SentimentLabel(str, Enum):
    """Sentiment classification labels."""
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"


class SentimentAnalyzer:
    """
    Sentiment analysis using momentum and returns as proxy.
    
    For full NLP sentiment, configure external APIs.
    """
    
    def __init__(self):
        pass
    
    def analyze_returns_sentiment(
        self,
        returns: pd.Series,
        prices: pd.Series = None
    ) -> Dict[str, Any]:
        """
        Calculate sentiment proxy based on price momentum.
        
        Args:
            returns: Daily returns series
            prices: Price series (optional)
        
        Returns:
            Sentiment analysis results
        """
        if len(returns) < 20:
            return {
                'overall_score': 0.0,
                'overall_label': SentimentLabel.NEUTRAL.value,
                'news_count': 0,
                'trending_score': 0.0,
                'whale_activity_score': 0.0,
                'sentiment_var_adjustment': 1.0
            }
        
        # Calculate momentum-based sentiment
        short_ma = returns.rolling(5).mean()
        long_ma = returns.rolling(20).mean()
        sentiment_momentum = float(short_ma.iloc[-1] - long_ma.iloc[-1]) * 100
        
        # Recent trend (annualized)
        recent_ret = float(returns.tail(5).mean()) * 252
        
        # Volatility regime
        recent_vol = float(returns.tail(20).std()) * np.sqrt(252)
        avg_vol = float(returns.std()) * np.sqrt(252)
        vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 1
        
        # Overall sentiment score (-1 to 1)
        overall_score = np.tanh(sentiment_momentum * 10)  # Scale and bound
        
        # Classify sentiment
        if overall_score > 0.3:
            label = SentimentLabel.VERY_POSITIVE
        elif overall_score > 0.1:
            label = SentimentLabel.POSITIVE
        elif overall_score > -0.1:
            label = SentimentLabel.NEUTRAL
        elif overall_score > -0.3:
            label = SentimentLabel.NEGATIVE
        else:
            label = SentimentLabel.VERY_NEGATIVE
        
        # Calculate VaR adjustment
        # Negative sentiment = higher VaR, positive sentiment = slightly lower VaR
        var_adjustment = 1.0
        if overall_score < -0.2:
            var_adjustment = 1.0 + abs(overall_score) * 0.5  # Up to 1.25x
        elif overall_score > 0.2:
            var_adjustment = 1.0 - overall_score * 0.2  # Down to 0.90x
        
        # Add volatility regime adjustment
        if vol_ratio > 1.3:
            var_adjustment *= 1.1  # Elevated vol increases VaR
        elif vol_ratio < 0.7:
            var_adjustment *= 0.95  # Low vol slightly decreases VaR
        
        return {
            'overall_score': float(overall_score),
            'overall_label': label.value,
            'news_count': 0,  # No external API
            'trending_score': float(sentiment_momentum),
            'whale_activity_score': 0.0,
            'sentiment_var_adjustment': float(var_adjustment),
            'momentum_5d': float(short_ma.iloc[-1]) if not pd.isna(short_ma.iloc[-1]) else 0,
            'momentum_20d': float(long_ma.iloc[-1]) if not pd.isna(long_ma.iloc[-1]) else 0,
            'recent_annualized_return': float(recent_ret),
            'volatility_ratio': float(vol_ratio)
        }


class SentimentVaR:
    """
    Calculate Sentiment-adjusted Value at Risk.
    
    Combines traditional VaR with sentiment analysis for forward-looking risk.
    """
    
    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()
    
    def calculate(
        self,
        returns: pd.Series,
        ticker: str,
        confidence: float = 0.95,
        horizon: int = 1,
        base_var_method: str = 'historical'
    ) -> Dict:
        """
        Calculate sentiment-adjusted VaR.
        
        Args:
            returns: Historical returns series
            ticker: Stock symbol for sentiment lookup
            confidence: VaR confidence level
            horizon: VaR horizon in days
            base_var_method: Method for base VaR calculation
        
        Returns:
            Dictionary with VaR metrics
        """
        # Calculate base VaR
        if base_var_method == 'historical':
            base_var = float(np.percentile(returns, 100 * (1 - confidence)))
        elif base_var_method == 'parametric':
            from scipy.stats import norm
            mu = float(returns.mean()) * horizon
            sigma = float(returns.std()) * np.sqrt(horizon)
            base_var = float(norm.ppf(1 - confidence, mu, sigma))
        else:
            base_var = float(np.percentile(returns, 100 * (1 - confidence)))
        
        # Get sentiment adjustment
        sentiment = self.sentiment_analyzer.analyze_returns_sentiment(returns)
        adjustment_factor = sentiment.get('sentiment_var_adjustment', 1.0)
        
        # Apply adjustment
        adjusted_var = base_var * adjustment_factor
        
        # Calculate CVaR
        base_cvar = float(returns[returns <= base_var].mean())
        adjusted_cvar = base_cvar * adjustment_factor
        
        return {
            'ticker': ticker,
            'confidence': confidence,
            'horizon': horizon,
            'base_var': abs(float(base_var)),
            'adjusted_var': abs(float(adjusted_var)),
            'base_cvar': abs(float(base_cvar)),
            'adjusted_cvar': abs(float(adjusted_cvar)),
            'adjustment_factor': adjustment_factor,
            'sentiment': {
                'score': sentiment.get('overall_score', 0),
                'label': sentiment.get('overall_label', 'neutral'),
                'trending': sentiment.get('trending_score', 0),
                'whale_score': sentiment.get('whale_activity_score', 0)
            },
            'method': base_var_method
        }
    
    def calculate_portfolio(
        self,
        returns_df: pd.DataFrame,
        weights: Dict[str, float],
        confidence: float = 0.95
    ) -> Dict:
        """
        Calculate sentiment-adjusted portfolio VaR.
        
        Args:
            returns_df: DataFrame of asset returns
            weights: Portfolio weights
            confidence: VaR confidence level
        
        Returns:
            Portfolio-level VaR with sentiment adjustment
        """
        # Calculate portfolio returns
        weight_array = np.array([weights.get(col, 0) for col in returns_df.columns])
        portfolio_returns = (returns_df * weight_array).sum(axis=1)
        
        # Base portfolio VaR
        base_var = float(np.percentile(portfolio_returns, 100 * (1 - confidence)))
        
        # Get per-asset sentiment and calculate weighted adjustment
        adjustments = []
        sentiment_details = {}
        
        for ticker in returns_df.columns:
            sentiment = self.sentiment_analyzer.analyze_returns_sentiment(returns_df[ticker])
            adj = sentiment.get('sentiment_var_adjustment', 1.0)
            adjustments.append(adj * weights.get(ticker, 0))
            sentiment_details[ticker] = {
                'score': sentiment.get('overall_score', 0),
                'label': sentiment.get('overall_label', 'neutral'),
                'adjustment': adj
            }
        
        # Portfolio-level adjustment
        portfolio_adjustment = sum(adjustments) if adjustments else 1.0
        adjusted_var = base_var * portfolio_adjustment
        
        return {
            'base_var': abs(float(base_var)),
            'adjusted_var': abs(float(adjusted_var)),
            'adjustment_factor': float(portfolio_adjustment),
            'per_asset_sentiment': sentiment_details,
            'confidence': confidence
        }


# Singleton instances
_sentiment_analyzer = None
_sentiment_var = None


def get_sentiment_analyzer() -> SentimentAnalyzer:
    """Get singleton SentimentAnalyzer instance."""
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        _sentiment_analyzer = SentimentAnalyzer()
    return _sentiment_analyzer


def get_sentiment_var() -> SentimentVaR:
    """Get singleton SentimentVaR instance."""
    global _sentiment_var
    if _sentiment_var is None:
        _sentiment_var = SentimentVaR()
    return _sentiment_var
