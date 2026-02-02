"""
Sentiment Analysis API Routes
=============================
Endpoints for sentiment analysis and sentiment-adjusted VaR.
"""

from fastapi import APIRouter, HTTPException
from typing import Optional
import logging

from app.schemas.sentiment import (
    SentimentRequest, SentimentResponse,
    SentimentVaRRequest, SentimentVaRResponse
)
from app.services.data_service import get_data_service
from app.services.sentiment_service import (
    get_sentiment_analyzer,
    get_sentiment_var
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/analyze", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    """
    Analyze price-based market sentiment for a stock.
    
    Uses momentum indicators, volume analysis, and technical patterns
    to derive a sentiment score (-1 to +1).
    
    Components:
    - Short-term momentum (5-day returns)
    - Medium-term momentum (20-day returns)
    - Long-term momentum (60-day returns)
    - Relative performance vs benchmark
    """
    data_service = get_data_service()
    
    # Fetch stock data
    prices, returns = data_service.get_returns(
        request.ticker, request.start_date, request.end_date
    )
    
    if returns.empty:
        raise HTTPException(status_code=404, detail=f"No data for {request.ticker}")
    
    # Fetch benchmark if provided
    bench_returns = None
    if request.benchmark:
        try:
            _, bench_returns = data_service.get_returns(
                request.benchmark, request.start_date, request.end_date
            )
        except:
            pass
    
    analyzer = get_sentiment_analyzer()
    result = analyzer.analyze_returns_sentiment(returns, bench_returns)
    
    return SentimentResponse(
        ticker=request.ticker,
        sentiment_score=result['sentiment_score'],
        sentiment_label=result['sentiment_label'],
        confidence=result['confidence'],
        momentum_signals=result['momentum_signals'],
        components=result['components']
    )


@router.get("/analyze/{ticker}")
async def quick_sentiment(
    ticker: str,
    days: int = 60,
    benchmark: str = "^GSPC"
):
    """Quick sentiment analysis with defaults."""
    from datetime import datetime, timedelta
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    request = SentimentRequest(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        benchmark=benchmark
    )
    
    return await analyze_sentiment(request)


@router.post("/var", response_model=SentimentVaRResponse)
async def sentiment_adjusted_var(request: SentimentVaRRequest):
    """
    Calculate sentiment-adjusted Value at Risk.
    
    Adjusts traditional VaR based on current market sentiment:
    - Positive sentiment: VaR may be reduced (market confidence)
    - Negative sentiment: VaR is increased (higher risk)
    
    Adjustment range: ±20% of base VaR.
    """
    data_service = get_data_service()
    
    # Fetch stock data
    prices, returns = data_service.get_returns(
        request.ticker, request.start_date, request.end_date
    )
    
    if returns.empty:
        raise HTTPException(status_code=404, detail=f"No data for {request.ticker}")
    
    sentiment_var = get_sentiment_var()
    result = sentiment_var.calculate(
        returns=returns,
        confidence_level=request.confidence_level
    )
    
    return SentimentVaRResponse(
        ticker=request.ticker,
        base_var=result['base_var'],
        sentiment_score=result['sentiment_score'],
        adjustment_factor=result['adjustment_factor'],
        adjusted_var=result['adjusted_var'],
        confidence_level=request.confidence_level,
        var_reduction_pct=result['var_reduction_pct'],
        interpretation=result['interpretation']
    )


@router.post("/portfolio-var")
async def portfolio_sentiment_var(
    tickers: list[str],
    weights: list[float],
    start_date: str,
    end_date: str,
    confidence_level: float = 0.95
):
    """
    Calculate sentiment-adjusted VaR for a portfolio.
    
    Aggregates sentiment across holdings weighted by position size.
    """
    import pandas as pd
    
    data_service = get_data_service()
    sentiment_var = get_sentiment_var()
    
    # Validate inputs
    if len(tickers) != len(weights):
        raise HTTPException(status_code=400, detail="Tickers and weights must have same length")
    
    if abs(sum(weights) - 1.0) > 0.01:
        raise HTTPException(status_code=400, detail="Weights must sum to 1.0")
    
    # Fetch all returns
    returns_dict = {}
    for ticker in tickers:
        try:
            _, returns = data_service.get_returns(ticker, start_date, end_date)
            if not returns.empty:
                returns_dict[ticker] = returns
        except Exception as e:
            logger.warning(f"Could not fetch {ticker}: {e}")
    
    if not returns_dict:
        raise HTTPException(status_code=404, detail="No valid data for any tickers")
    
    # Build returns DataFrame
    returns_df = pd.DataFrame(returns_dict).dropna()
    
    result = sentiment_var.calculate_portfolio(
        returns_df=returns_df,
        weights=dict(zip(tickers, weights)),
        confidence_level=confidence_level
    )
    
    return {
        'tickers': tickers,
        'weights': weights,
        **result
    }


@router.get("/signals/{ticker}")
async def momentum_signals(ticker: str, days: int = 60):
    """
    Get momentum-based trading signals.
    
    Returns buy/sell signals based on momentum indicators.
    """
    from datetime import datetime, timedelta
    
    data_service = get_data_service()
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    prices, returns = data_service.get_returns(ticker, start_date, end_date)
    
    if returns.empty:
        raise HTTPException(status_code=404, detail=f"No data for {ticker}")
    
    analyzer = get_sentiment_analyzer()
    result = analyzer.analyze_returns_sentiment(returns)
    
    # Generate signals
    score = result['sentiment_score']
    momentum = result['momentum_signals']
    
    signals = []
    
    # Short-term signal
    if momentum['short_term'] == 'bullish':
        signals.append({'type': 'short_term', 'signal': 'BUY', 'strength': 'moderate'})
    elif momentum['short_term'] == 'bearish':
        signals.append({'type': 'short_term', 'signal': 'SELL', 'strength': 'moderate'})
    
    # Medium-term signal
    if momentum['medium_term'] == 'bullish':
        signals.append({'type': 'medium_term', 'signal': 'BUY', 'strength': 'strong'})
    elif momentum['medium_term'] == 'bearish':
        signals.append({'type': 'medium_term', 'signal': 'SELL', 'strength': 'strong'})
    
    # Trend alignment
    if momentum['trend_alignment'] == 'aligned':
        overall_signal = 'STRONG_BUY' if score > 0.3 else 'STRONG_SELL' if score < -0.3 else 'HOLD'
    else:
        overall_signal = 'HOLD' if abs(score) < 0.5 else ('BUY' if score > 0 else 'SELL')
    
    return {
        'ticker': ticker,
        'sentiment_score': score,
        'signals': signals,
        'overall_signal': overall_signal,
        'momentum': momentum,
        'confidence': result['confidence']
    }
