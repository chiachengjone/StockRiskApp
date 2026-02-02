"""
Risk Analysis API Routes
========================
Endpoints for VaR, GARCH, EVT, Monte Carlo, and stress testing.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
import logging

from app.schemas.risk import (
    RiskMetricsRequest, RiskMetricsResponse,
    VaRRequest, VaRResponse,
    GARCHRequest, GARCHResponse,
    EVTRequest, EVTResponse,
    MonteCarloRequest, MonteCarloResponse,
    StressTestRequest, StressTestResponse,
    RollingMetricsRequest, RollingMetricsResponse,
    StressScenariosListResponse
)
from app.services.data_service import get_data_service
from app.services.risk_service import (
    compute_returns, compute_metrics, calculate_all_var,
    fit_garch, evt_tail_risk, monte_carlo_simulation,
    run_stress_test, get_stress_scenarios_list,
    calculate_rolling_metrics, parametric_var, historical_var, cvar
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/metrics", response_model=RiskMetricsResponse)
async def get_risk_metrics(request: RiskMetricsRequest):
    """
    Calculate comprehensive risk metrics for a stock.
    
    Returns annualized return, volatility, Sharpe, Sortino, max drawdown, etc.
    """
    data_service = get_data_service()
    prices, returns = data_service.get_returns(
        request.ticker, request.start_date, request.end_date
    )
    
    if returns.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {request.ticker}")
    
    metrics = compute_metrics(returns)
    
    return RiskMetricsResponse(
        ticker=request.ticker,
        period={"start": request.start_date, "end": request.end_date},
        metrics=metrics
    )


@router.post("/var", response_model=VaRResponse)
async def calculate_var(request: VaRRequest):
    """
    Calculate Value-at-Risk using multiple methods.
    
    Methods:
    - Parametric (Normal distribution assumption)
    - Historical (Empirical percentile)
    - CVaR (Expected Shortfall)
    """
    data_service = get_data_service()
    prices, returns = data_service.get_returns(
        request.ticker, request.start_date, request.end_date
    )
    
    if returns.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {request.ticker}")
    
    result = calculate_all_var(returns, request.confidence)
    
    var_pct = abs(result['parametric']) * 100
    interpretation = f"At {int(request.confidence*100)}% confidence, daily loss will not exceed {var_pct:.2f}%"
    
    return VaRResponse(
        ticker=request.ticker,
        confidence=request.confidence,
        var_parametric=result['parametric'],
        var_historical=result['historical'],
        cvar=result['cvar'],
        var_interpretation=interpretation
    )


@router.get("/var/{ticker}")
async def quick_var(
    ticker: str,
    days: int = Query(252, ge=30, le=1000),
    confidence: float = Query(0.95, ge=0.9, le=0.99)
):
    """Quick VaR calculation with default parameters."""
    from datetime import datetime, timedelta
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    data_service = get_data_service()
    prices, returns = data_service.get_returns(ticker, start_date, end_date)
    
    if returns.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {ticker}")
    
    return calculate_all_var(returns, confidence)


@router.post("/garch", response_model=GARCHResponse)
async def calculate_garch(request: GARCHRequest):
    """
    Fit GARCH(p,q) model and forecast volatility.
    
    GARCH models capture volatility clustering in financial returns.
    """
    data_service = get_data_service()
    prices, returns = data_service.get_returns(
        request.ticker, request.start_date, request.end_date
    )
    
    if returns.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {request.ticker}")
    
    result = fit_garch(returns, request.p, request.q)
    
    if 'error' in result:
        raise HTTPException(status_code=400, detail=result['error'])
    
    return GARCHResponse(
        ticker=request.ticker,
        **result
    )


@router.post("/evt", response_model=EVTResponse)
async def calculate_evt(request: EVTRequest):
    """
    Extreme Value Theory analysis for tail risk.
    
    Fits Generalized Pareto Distribution to extreme losses.
    """
    data_service = get_data_service()
    prices, returns = data_service.get_returns(
        request.ticker, request.start_date, request.end_date
    )
    
    if returns.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {request.ticker}")
    
    result = evt_tail_risk(returns, request.threshold_percentile)
    
    if 'error' in result:
        raise HTTPException(status_code=400, detail=result['error'])
    
    return EVTResponse(
        ticker=request.ticker,
        threshold=result['threshold'],
        shape_parameter=result['shape_parameter'],
        evt_var=result['evt_var'],
        excesses_count=result['excesses_count'],
        tail_index=result.get('tail_index')
    )


@router.post("/monte-carlo", response_model=MonteCarloResponse)
async def run_monte_carlo(request: MonteCarloRequest):
    """
    Monte Carlo simulation for future returns.
    
    Simulates multiple return paths to estimate risk distribution.
    """
    data_service = get_data_service()
    prices, returns = data_service.get_returns(
        request.ticker, request.start_date, request.end_date
    )
    
    if returns.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {request.ticker}")
    
    result = monte_carlo_simulation(returns, request.n_simulations, request.horizon_days)
    
    return MonteCarloResponse(
        ticker=request.ticker,
        **result
    )


@router.post("/stress-test", response_model=StressTestResponse)
async def stress_test(request: StressTestRequest):
    """
    Run stress test with predefined or custom scenario.
    
    Estimates portfolio impact under adverse market conditions.
    """
    data_service = get_data_service()
    prices, returns = data_service.get_returns(
        request.ticker, request.start_date, request.end_date
    )
    
    if returns.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {request.ticker}")
    
    # Get sector for sector-specific stress
    info = data_service.fetch_info(request.ticker)
    sector = info.get('sector')
    
    result = run_stress_test(
        returns,
        scenario_name=request.scenario,
        custom_shock=request.custom_shock,
        ticker_sector=sector
    )
    
    if 'error' in result:
        raise HTTPException(status_code=400, detail=result['error'])
    
    return StressTestResponse(
        ticker=request.ticker,
        **result
    )


@router.get("/stress-scenarios", response_model=StressScenariosListResponse)
async def list_stress_scenarios():
    """Get list of available predefined stress scenarios."""
    scenarios = get_stress_scenarios_list()
    return StressScenariosListResponse(scenarios=scenarios)


@router.post("/rolling", response_model=RollingMetricsResponse)
async def get_rolling_metrics(request: RollingMetricsRequest):
    """
    Calculate rolling risk metrics over time.
    
    Useful for visualizing how risk evolves.
    """
    data_service = get_data_service()
    prices, returns = data_service.get_returns(
        request.ticker, request.start_date, request.end_date
    )
    
    if returns.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {request.ticker}")
    
    result = calculate_rolling_metrics(returns, request.window_days)
    
    return RollingMetricsResponse(
        ticker=request.ticker,
        window_days=request.window_days,
        metrics={
            "rolling_volatility": result["rolling_volatility"],
            "rolling_sharpe": result["rolling_sharpe"],
            "rolling_var": result["rolling_var"],
            "rolling_max_drawdown": result["rolling_max_drawdown"]
        },
        dates=result["dates"]
    )
