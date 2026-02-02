"""
Fundamentals Analysis API Routes
================================
Endpoints for fundamental analysis, DCF valuation, and quality scoring.
"""

from fastapi import APIRouter, HTTPException
from typing import List, Optional
import logging

from app.schemas.fundamentals import (
    FundamentalsRequest, FundamentalsResponse,
    DCFRequest, DCFResponse,
    PeerComparisonRequest, PeerComparisonResponse,
    QualityScoreRequest, QualityScoreResponse,
    ValuationMetrics, ProfitabilityMetrics,
    FinancialHealthMetrics, GrowthMetrics
)
from app.services.data_service import get_data_service
from app.services.fundamentals_service import get_fundamental_analyzer

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/{ticker}", response_model=FundamentalsResponse)
async def get_fundamentals(ticker: str):
    """
    Get comprehensive fundamental analysis for a stock.
    
    Returns valuation ratios, profitability metrics, financial health,
    and growth indicators with assessments.
    """
    data_service = get_data_service()
    info = data_service.get_fundamentals(ticker)
    
    if not info or 'error' in info:
        raise HTTPException(status_code=404, detail=f"No fundamental data for {ticker}")
    
    analyzer = get_fundamental_analyzer()
    result = analyzer.analyze_fundamentals(info)
    
    if 'error' in result:
        raise HTTPException(status_code=404, detail=result['error'])
    
    return FundamentalsResponse(
        ticker=ticker,
        name=result.get('name'),
        sector=result.get('sector'),
        industry=result.get('industry'),
        valuation=ValuationMetrics(**result['valuation']),
        profitability=ProfitabilityMetrics(**result['profitability']),
        financial_health=FinancialHealthMetrics(**result['financial_health']),
        growth=GrowthMetrics(**result['growth']),
        quality_score=result['quality_score']
    )


@router.post("/analyze", response_model=FundamentalsResponse)
async def analyze_fundamentals(request: FundamentalsRequest):
    """Analyze fundamentals for a given ticker."""
    return await get_fundamentals(request.ticker)


@router.post("/dcf", response_model=DCFResponse)
async def dcf_valuation(request: DCFRequest):
    """
    Discounted Cash Flow valuation.
    
    Projects future cash flows and discounts to present value
    to estimate intrinsic value per share.
    
    Key assumptions:
    - growth_rate: Expected FCF growth during projection period
    - terminal_growth: Long-term growth rate (should be ≤ GDP growth)
    - discount_rate: WACC or required return
    """
    data_service = get_data_service()
    info = data_service.get_fundamentals(request.ticker)
    
    if not info:
        raise HTTPException(status_code=404, detail=f"No data for {request.ticker}")
    
    analyzer = get_fundamental_analyzer()
    result = analyzer.dcf_valuation(
        info=info,
        growth_rate=request.growth_rate,
        terminal_growth=request.terminal_growth,
        discount_rate=request.discount_rate,
        projection_years=request.projection_years
    )
    
    if 'error' in result:
        raise HTTPException(status_code=400, detail=result['error'])
    
    return DCFResponse(
        ticker=request.ticker,
        fair_value=result['fair_value'],
        current_price=result['current_price'],
        upside_potential=result['upside_potential'],
        intrinsic_value=result['intrinsic_value'],
        terminal_value=result['terminal_value'],
        present_value_cf=result['present_value_cf'],
        present_value_terminal=result['present_value_terminal'],
        assumptions=result['assumptions'],
        sensitivity=result['sensitivity']
    )


@router.post("/peer-comparison", response_model=PeerComparisonResponse)
async def peer_comparison(request: PeerComparisonRequest):
    """
    Compare company against peers on key metrics.
    
    Provides rankings and percentiles for:
    - Valuation (P/E, P/B)
    - Profitability (ROE, margins)
    - Growth rates
    """
    data_service = get_data_service()
    
    # Get subject company info
    info = data_service.get_fundamentals(request.ticker)
    if not info:
        raise HTTPException(status_code=404, detail=f"No data for {request.ticker}")
    
    # Determine peers
    peers = request.peers
    if not peers:
        # Auto-select peers from same sector (simplified)
        sector = info.get('sector', '')
        industry = info.get('industry', '')
        
        # Default tech peers as fallback
        default_peers = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA'],
            'Financial Services': ['JPM', 'BAC', 'WFC', 'GS', 'MS'],
            'Healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK'],
            'Consumer Cyclical': ['AMZN', 'TSLA', 'HD', 'NKE', 'SBUX']
        }
        peers = default_peers.get(sector, ['SPY', 'QQQ', 'IWM'])[:5]
        peers = [p for p in peers if p != request.ticker]
    
    # Fetch peer data
    peer_infos = {}
    for peer in peers:
        try:
            peer_info = data_service.get_fundamentals(peer)
            if peer_info:
                peer_infos[peer] = peer_info
        except:
            continue
    
    if not peer_infos:
        raise HTTPException(status_code=400, detail="Could not fetch peer data")
    
    analyzer = get_fundamental_analyzer()
    result = analyzer.peer_comparison(request.ticker, info, peer_infos)
    
    return PeerComparisonResponse(**result)


@router.get("/quality-score/{ticker}", response_model=QualityScoreResponse)
async def quality_score(ticker: str):
    """
    Calculate overall quality score for a stock.
    
    Combines valuation, profitability, financial health,
    and growth into a single 0-100 score with letter grade.
    """
    data_service = get_data_service()
    info = data_service.get_fundamentals(ticker)
    
    if not info:
        raise HTTPException(status_code=404, detail=f"No data for {ticker}")
    
    analyzer = get_fundamental_analyzer()
    result = analyzer.analyze_fundamentals(info)
    
    overall = result.get('quality_score', 0)
    grade = analyzer.get_quality_grade(overall)
    
    # Generate recommendation
    if overall >= 75:
        recommendation = "Strong fundamentals - consider for long-term portfolio"
    elif overall >= 50:
        recommendation = "Mixed fundamentals - further research recommended"
    else:
        recommendation = "Weak fundamentals - exercise caution"
    
    return QualityScoreResponse(
        ticker=ticker,
        overall_score=overall,
        valuation_score=result['valuation']['score'] * 12.5,
        profitability_score=result['profitability']['score'] * 12.5,
        financial_health_score=result['financial_health']['score'] * 12.5,
        growth_score=result['growth']['score'] * 12.5,
        grade=grade,
        recommendation=recommendation
    )


@router.get("/quick/{ticker}")
async def quick_fundamentals(ticker: str):
    """
    Quick fundamental snapshot.
    
    Returns key metrics without detailed analysis.
    """
    data_service = get_data_service()
    info = data_service.get_fundamentals(ticker)
    
    if not info:
        raise HTTPException(status_code=404, detail=f"No data for {ticker}")
    
    return {
        'ticker': ticker,
        'name': info.get('shortName') or info.get('longName'),
        'sector': info.get('sector'),
        'industry': info.get('industry'),
        'market_cap': info.get('marketCap'),
        'pe_ratio': info.get('trailingPE'),
        'forward_pe': info.get('forwardPE'),
        'price_to_book': info.get('priceToBook'),
        'dividend_yield': info.get('dividendYield'),
        'roe': info.get('returnOnEquity'),
        'profit_margin': info.get('profitMargins'),
        'debt_to_equity': info.get('debtToEquity'),
        'current_price': info.get('currentPrice') or info.get('regularMarketPrice'),
        '52_week_high': info.get('fiftyTwoWeekHigh'),
        '52_week_low': info.get('fiftyTwoWeekLow')
    }
