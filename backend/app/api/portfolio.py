"""
Portfolio API Routes
====================
Endpoints for portfolio optimization and risk analysis.
"""

from fastapi import APIRouter, HTTPException
from typing import List, Optional
import numpy as np
import pandas as pd
import logging

from app.schemas.portfolio import (
    RiskParityRequest, RiskParityResponse,
    HRPRequest, HRPResponse,
    BlackLittermanRequest, BlackLittermanResponse,
    EfficientFrontierRequest, EfficientFrontierResponse,
    TransactionCostRequest, TransactionCostResponse,
    RebalanceAnalysisRequest, RebalanceAnalysisResponse,
    ThresholdRebalanceRequest, ThresholdRebalanceResponse,
    TaxLossHarvestingRequest, TaxLossHarvestingResponse,
    PortfolioVaRRequest, PortfolioVaRResponse,
    MarginalVaRRequest, MarginalVaRResponse,
    CorrelationAnalysisResponse
)
from app.services.data_service import get_data_service
from app.services.portfolio_service import (
    calculate_risk_parity, calculate_hrp,
    calculate_black_litterman, calculate_efficient_frontier,
    calculate_transaction_costs, analyze_rebalancing,
    check_threshold_rebalance, find_tax_loss_opportunities,
    calculate_portfolio_var, calculate_marginal_var,
    analyze_correlations
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/risk-parity", response_model=RiskParityResponse)
async def risk_parity_optimization(request: RiskParityRequest):
    """
    Calculate Risk Parity portfolio weights.
    
    Equal risk contribution across all assets.
    """
    data_service = get_data_service()
    df = data_service.fetch_multiple(
        request.tickers, request.start_date, request.end_date
    )
    
    if df.empty:
        raise HTTPException(status_code=404, detail="No data found for the requested tickers")
    
    returns = np.log(df / df.shift(1)).dropna()
    
    result = calculate_risk_parity(
        returns,
        target_risk=request.target_risk,
        max_weight=request.max_weight,
        min_weight=request.min_weight
    )
    
    return RiskParityResponse(**result)


@router.post("/hrp", response_model=HRPResponse)
async def hrp_optimization(request: HRPRequest):
    """
    Hierarchical Risk Parity optimization.
    
    Uses clustering for more robust allocation.
    """
    data_service = get_data_service()
    df = data_service.fetch_multiple(
        request.tickers, request.start_date, request.end_date
    )
    
    if df.empty:
        raise HTTPException(status_code=404, detail="No data found for the requested tickers")
    
    returns = np.log(df / df.shift(1)).dropna()
    
    result = calculate_hrp(returns, method=request.method)
    
    return HRPResponse(**result)


@router.post("/black-litterman", response_model=BlackLittermanResponse)
async def black_litterman_optimization(request: BlackLittermanRequest):
    """
    Black-Litterman portfolio optimization.
    
    Combines market equilibrium with investor views.
    """
    data_service = get_data_service()
    df = data_service.fetch_multiple(
        request.tickers, request.start_date, request.end_date
    )
    
    if df.empty:
        raise HTTPException(status_code=404, detail="No data found for the requested tickers")
    
    returns = np.log(df / df.shift(1)).dropna()
    
    # Convert views to expected format
    views = []
    for view in request.views:
        if view.asset:
            views.append({
                'asset': view.asset,
                'return': view.expected_return
            })
        elif view.long_asset and view.short_asset:
            views.append({
                'long': view.long_asset,
                'short': view.short_asset,
                'return': view.expected_return
            })
    
    result = calculate_black_litterman(
        returns,
        request.market_caps,
        views,
        risk_aversion=request.risk_aversion,
        tau=request.tau
    )
    
    return BlackLittermanResponse(**result)


@router.post("/efficient-frontier", response_model=EfficientFrontierResponse)
async def efficient_frontier(request: EfficientFrontierRequest):
    """
    Calculate efficient frontier points.
    """
    data_service = get_data_service()
    df = data_service.fetch_multiple(
        request.tickers, request.start_date, request.end_date
    )
    
    if df.empty:
        raise HTTPException(status_code=404, detail="No data found for the requested tickers")
    
    returns = np.log(df / df.shift(1)).dropna()
    
    result = calculate_efficient_frontier(returns, request.n_points)
    
    return EfficientFrontierResponse(**result)


@router.post("/transaction-costs", response_model=TransactionCostResponse)
async def analyze_transaction_costs(request: TransactionCostRequest):
    """
    Analyze transaction costs for rebalancing.
    """
    result = calculate_transaction_costs(
        current_weights=request.current_weights,
        target_weights=request.target_weights,
        portfolio_value=request.portfolio_value,
        prices=request.prices,
        volumes=request.volumes,
        spread_bps=request.spread_bps,
        commission_per_share=request.commission_per_share,
        market_impact_bps=request.market_impact_bps
    )
    
    return TransactionCostResponse(**result)


@router.post("/rebalance-analysis", response_model=RebalanceAnalysisResponse)
async def rebalance_frequency_analysis(request: RebalanceAnalysisRequest):
    """
    Analyze optimal rebalancing frequency.
    """
    data_service = get_data_service()
    df = data_service.fetch_multiple(
        request.tickers, request.start_date, request.end_date
    )
    
    if df.empty:
        raise HTTPException(status_code=404, detail="No data found for the requested tickers")
    
    returns = np.log(df / df.shift(1)).dropna()
    
    result = analyze_rebalancing(
        returns,
        request.target_weights,
        request.cost_per_rebalance
    )
    
    return RebalanceAnalysisResponse(**result)


@router.post("/threshold-rebalance", response_model=ThresholdRebalanceResponse)
async def threshold_rebalance_check(request: ThresholdRebalanceRequest):
    """
    Check if rebalancing is needed based on drift threshold.
    """
    result = check_threshold_rebalance(
        request.current_weights,
        request.target_weights,
        request.threshold
    )
    
    return ThresholdRebalanceResponse(**result)


@router.post("/tax-loss-harvesting", response_model=TaxLossHarvestingResponse)
async def tax_loss_harvesting(request: TaxLossHarvestingRequest):
    """
    Identify tax-loss harvesting opportunities.
    """
    result = find_tax_loss_opportunities(
        request.positions,
        request.short_term_rate,
        request.long_term_rate
    )
    
    return TaxLossHarvestingResponse(**result)


@router.post("/portfolio-var", response_model=PortfolioVaRResponse)
async def portfolio_var(request: PortfolioVaRRequest):
    """
    Calculate portfolio VaR using covariance matrix.
    """
    data_service = get_data_service()
    df = data_service.fetch_multiple(
        request.tickers, request.start_date, request.end_date
    )
    
    if df.empty:
        raise HTTPException(status_code=404, detail="No data found for the requested tickers")
    
    returns = np.log(df / df.shift(1)).dropna()
    weights = np.array(request.weights)
    
    result = calculate_portfolio_var(returns, weights, request.confidence)
    
    return PortfolioVaRResponse(
        portfolio_var=result['portfolio_var'],
        confidence=request.confidence,
        portfolio_volatility=result['portfolio_volatility'],
        interpretation=f"At {int(request.confidence*100)}% confidence, daily portfolio loss will not exceed {abs(result['portfolio_var'])*100:.2f}%"
    )


@router.post("/marginal-var", response_model=MarginalVaRResponse)
async def marginal_var_contribution(request: MarginalVaRRequest):
    """
    Calculate marginal VaR contribution by asset.
    """
    data_service = get_data_service()
    df = data_service.fetch_multiple(
        request.tickers, request.start_date, request.end_date
    )
    
    if df.empty:
        raise HTTPException(status_code=404, detail="No data found for the requested tickers")
    
    returns = np.log(df / df.shift(1)).dropna()
    weights = np.array(request.weights)
    
    result = calculate_marginal_var(returns, weights, request.confidence)
    
    return MarginalVaRResponse(
        contributions=result['contributions'],
        portfolio_var=result['portfolio_var'],
        largest_contributor=result['largest_contributor'],
        smallest_contributor=result['smallest_contributor']
    )


@router.post("/correlation-analysis", response_model=CorrelationAnalysisResponse)
async def correlation_analysis(request: HRPRequest):
    """
    Analyze correlations including crash vs normal periods.
    """
    data_service = get_data_service()
    df = data_service.fetch_multiple(
        request.tickers, request.start_date, request.end_date
    )
    
    if df.empty:
        raise HTTPException(status_code=404, detail="No data found for the requested tickers")
    
    returns = np.log(df / df.shift(1)).dropna()
    
    result = analyze_correlations(returns)
    
    return CorrelationAnalysisResponse(**result)
