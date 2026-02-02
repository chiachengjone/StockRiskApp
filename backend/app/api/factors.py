"""
Factor Analysis API Routes
==========================
Endpoints for Fama-French, Kelly Criterion, ESG, and Style Factors.
"""

from fastapi import APIRouter, HTTPException
from typing import Optional
import logging

from app.schemas.factors import (
    FamaFrenchRequest, FamaFrenchResponse,
    KellyRequest, KellyResponse,
    ESGRequest, ESGResponse,
    StyleFactorRequest, StyleFactorResponse
)
from app.services.data_service import get_data_service
from app.services.factors_service import (
    get_factor_analyzer,
    get_style_analyzer
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/fama-french", response_model=FamaFrenchResponse)
async def fama_french_analysis(request: FamaFrenchRequest):
    """
    Fama-French 5-Factor Regression.
    
    Calculates factor loadings, alpha, and R² for the stock.
    
    Factors:
    - Mkt-RF: Market excess return
    - SMB: Small Minus Big (size factor)
    - HML: High Minus Low (value factor)
    - RMW: Robust Minus Weak (profitability)
    - CMA: Conservative Minus Aggressive (investment)
    """
    data_service = get_data_service()
    
    # Fetch stock returns
    prices, returns = data_service.get_returns(
        request.ticker, request.start_date, request.end_date
    )
    
    if returns.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {request.ticker}")
    
    # Fetch benchmark returns if provided
    bench_returns = None
    if request.benchmark:
        try:
            bench_prices, bench_returns = data_service.get_returns(
                request.benchmark, request.start_date, request.end_date
            )
        except:
            pass
    
    # Run factor analysis
    analyzer = get_factor_analyzer()
    result = analyzer.fama_french_regression(returns, bench_returns)
    
    if 'error' in result and result.get('r_squared', 0) == 0:
        raise HTTPException(status_code=400, detail=result['error'])
    
    return FamaFrenchResponse(
        ticker=request.ticker,
        alpha=result['alpha'],
        r_squared=result['r_squared'],
        loadings=result['loadings'],
        n_observations=result.get('n_observations', 0),
        t_stats=result.get('t_stats')
    )


@router.post("/kelly", response_model=KellyResponse)
async def kelly_criterion(request: KellyRequest):
    """
    Kelly Criterion optimal position sizing.
    
    Calculates the theoretically optimal position size based on
    win rate and win/loss ratio.
    
    Recommendation: Use half-Kelly (fraction=0.5) for safety.
    """
    data_service = get_data_service()
    prices, returns = data_service.get_returns(
        request.ticker, request.start_date, request.end_date
    )
    
    if returns.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {request.ticker}")
    
    analyzer = get_factor_analyzer()
    result = analyzer.kelly_criterion(returns, request.fraction)
    
    return KellyResponse(
        ticker=request.ticker,
        kelly_pct=result['kelly_pct'],
        full_kelly=result['full_kelly'],
        win_rate=result['win_rate'],
        win_loss_ratio=result['win_loss_ratio'],
        edge_per_trade=result['edge_per_trade'],
        fraction_used=result['fraction_used']
    )


@router.get("/esg/{ticker}", response_model=ESGResponse)
async def esg_rating(ticker: str):
    """
    Get ESG (Environmental, Social, Governance) rating.
    
    Returns sustainability scores from Yahoo Finance.
    
    Rating scale: AAA, AA, A, BBB, BB, B, CCC
    Lower scores are better.
    """
    analyzer = get_factor_analyzer()
    result = analyzer.get_esg_rating(ticker)
    
    return ESGResponse(
        ticker=ticker,
        rating=result['rating'],
        total_esg=result.get('total_esg'),
        environment_score=result.get('environment_score'),
        social_score=result.get('social_score'),
        governance_score=result.get('governance_score'),
        source=result.get('source', 'Yahoo Finance')
    )


@router.post("/style", response_model=StyleFactorResponse)
async def style_factor_analysis(request: StyleFactorRequest):
    """
    Multi-Factor Style Analysis.
    
    Analyzes:
    - Momentum: Price momentum, RSI, trend strength
    - Value: P/E, P/B, dividend yield
    - Quality: ROE, profit margins, debt ratios
    - Relative Strength: Performance vs benchmark
    
    Returns scores 0-100 and style classification.
    """
    data_service = get_data_service()
    
    # Fetch stock data
    prices, returns = data_service.get_returns(
        request.ticker, request.start_date, request.end_date
    )
    
    if returns.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {request.ticker}")
    
    # Fetch benchmark
    bench_returns = None
    if request.benchmark:
        try:
            _, bench_returns = data_service.get_returns(
                request.benchmark, request.start_date, request.end_date
            )
        except:
            pass
    
    # Fetch fundamentals
    fundamentals = data_service.get_fundamentals(request.ticker)
    
    # Run analysis
    analyzer = get_style_analyzer()
    result = analyzer.analyze_style_factors(
        returns=returns,
        prices=prices,
        fundamentals=fundamentals,
        benchmark_returns=bench_returns
    )
    
    if 'error' in result:
        raise HTTPException(status_code=400, detail=result['error'])
    
    return StyleFactorResponse(
        ticker=request.ticker,
        overall_score=result['overall_score'],
        scores=result['scores'],
        style_label=result['style_label'],
        style_classification=result['style_classification'],
        momentum_details=result['momentum_details'],
        value_details=result['value_details'],
        quality_details=result['quality_details'],
        radar_data=result['radar_data']
    )


@router.get("/style/{ticker}")
async def quick_style_analysis(
    ticker: str,
    days: int = 252,
    benchmark: str = "^GSPC"
):
    """Quick style factor analysis with defaults."""
    from datetime import datetime, timedelta
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    request = StyleFactorRequest(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        benchmark=benchmark
    )
    
    return await style_factor_analysis(request)
