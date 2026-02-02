"""
Digital Twin and What-If Analysis API Routes
=============================================
Endpoints for portfolio simulation and scenario analysis.
"""

from fastapi import APIRouter, HTTPException
from typing import List, Optional, Dict
import logging

from app.schemas.sentiment import (
    DigitalTwinRequest, DigitalTwinResponse,
    WhatIfRequest, WhatIfResponse,
    ScenarioResult, PortfolioHealthScore
)
from app.services.data_service import get_data_service
from app.services.digital_twin_service import (
    create_digital_twin,
    create_what_if_analyzer
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/scenario", response_model=DigitalTwinResponse)
async def run_scenario(request: DigitalTwinRequest):
    """
    Run a portfolio simulation scenario.
    
    Strategies:
    - buy_and_hold: No rebalancing
    - monthly_rebalance: Rebalance to target weights monthly
    - quarterly_rebalance: Rebalance to target weights quarterly
    - tax_optimized: Rebalance with tax-loss harvesting
    """
    import pandas as pd
    from datetime import datetime, timedelta
    
    data_service = get_data_service()
    
    # Determine date range
    end_date = datetime.now().strftime('%Y-%m-%d')
    years = request.years or 1
    start_date = (datetime.now() - timedelta(days=365*years)).strftime('%Y-%m-%d')
    
    # Fetch returns for all tickers
    returns_dict = {}
    for ticker in request.tickers:
        try:
            _, returns = data_service.get_returns(ticker, start_date, end_date)
            if not returns.empty:
                returns_dict[ticker] = returns
        except Exception as e:
            logger.warning(f"Could not fetch {ticker}: {e}")
    
    if not returns_dict:
        raise HTTPException(status_code=404, detail="No valid data for any tickers")
    
    returns_df = pd.DataFrame(returns_dict).dropna()
    weights = dict(zip(request.tickers, request.weights))
    
    engine = create_digital_twin()
    result = engine.run_scenario(
        returns_df=returns_df,
        weights=weights,
        strategy=request.strategy,
        initial_value=request.initial_value
    )
    
    scenario_result = ScenarioResult(**result['metrics'])
    
    return DigitalTwinResponse(
        scenario_name=request.strategy,
        result=scenario_result,
        portfolio_values=result['portfolio_values'],
        drawdown_series=result.get('drawdown_series')
    )


@router.post("/compare-scenarios")
async def compare_scenarios(
    tickers: List[str],
    weights: List[float],
    strategies: List[str] = ["buy_and_hold", "monthly_rebalance", "quarterly_rebalance"],
    years: int = 1,
    initial_value: float = 100000
):
    """
    Compare multiple portfolio strategies side by side.
    
    Returns performance metrics for each strategy.
    """
    import pandas as pd
    from datetime import datetime, timedelta
    
    data_service = get_data_service()
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365*years)).strftime('%Y-%m-%d')
    
    # Fetch returns
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
    
    returns_df = pd.DataFrame(returns_dict).dropna()
    weights_dict = dict(zip(tickers, weights))
    
    engine = create_digital_twin()
    comparison = engine.compare_scenarios(
        returns_df=returns_df,
        weights=weights_dict,
        strategies=strategies,
        initial_value=initial_value
    )
    
    return {
        'tickers': tickers,
        'weights': weights,
        'period': {'start': start_date, 'end': end_date},
        'comparison': comparison
    }


@router.post("/health-score", response_model=PortfolioHealthScore)
async def calculate_health_score(
    tickers: List[str],
    weights: List[float],
    days: int = 252
):
    """
    Calculate overall portfolio health score.
    
    Combines:
    - Return metrics
    - Risk metrics
    - Diversification
    - Drawdown analysis
    """
    import pandas as pd
    from datetime import datetime, timedelta
    
    data_service = get_data_service()
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    returns_dict = {}
    for ticker in tickers:
        try:
            _, returns = data_service.get_returns(ticker, start_date, end_date)
            if not returns.empty:
                returns_dict[ticker] = returns
        except:
            pass
    
    if not returns_dict:
        raise HTTPException(status_code=404, detail="No valid data")
    
    returns_df = pd.DataFrame(returns_dict).dropna()
    weights_dict = dict(zip(tickers, weights))
    
    engine = create_digital_twin()
    result = engine.calculate_health_score(returns_df, weights_dict)
    
    return PortfolioHealthScore(**result)


@router.post("/what-if", response_model=WhatIfResponse)
async def what_if_analysis(request: WhatIfRequest):
    """
    Analyze impact of portfolio weight changes.
    
    Compares current allocation vs proposed allocation.
    """
    import pandas as pd
    from datetime import datetime, timedelta
    
    data_service = get_data_service()
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=252)).strftime('%Y-%m-%d')
    
    # Determine all tickers
    all_tickers = set(request.current_weights.keys()) | set(request.proposed_weights.keys())
    
    returns_dict = {}
    for ticker in all_tickers:
        try:
            _, returns = data_service.get_returns(ticker, start_date, end_date)
            if not returns.empty:
                returns_dict[ticker] = returns
        except:
            pass
    
    if not returns_dict:
        raise HTTPException(status_code=404, detail="No valid data")
    
    returns_df = pd.DataFrame(returns_dict).dropna()
    
    analyzer = create_what_if_analyzer()
    result = analyzer.analyze_scenario(
        returns_df=returns_df,
        current_weights=request.current_weights,
        proposed_weights=request.proposed_weights
    )
    
    return WhatIfResponse(**result)


@router.post("/rebalance-trades")
async def calculate_rebalance_trades(
    tickers: List[str],
    current_weights: List[float],
    target_weights: List[float],
    portfolio_value: float,
    min_trade_pct: float = 0.01
):
    """
    Calculate trades needed to rebalance portfolio.
    
    Returns buy/sell trades with dollar amounts.
    """
    if len(tickers) != len(current_weights) or len(tickers) != len(target_weights):
        raise HTTPException(status_code=400, detail="Lists must have same length")
    
    current = dict(zip(tickers, current_weights))
    target = dict(zip(tickers, target_weights))
    
    analyzer = create_what_if_analyzer()
    trades = analyzer.calculate_rebalance_trades(
        current_weights=current,
        target_weights=target,
        portfolio_value=portfolio_value,
        min_trade_pct=min_trade_pct
    )
    
    return {
        'portfolio_value': portfolio_value,
        'trades': trades,
        'total_turnover': sum(abs(t['amount']) for t in trades) / 2
    }


@router.post("/optimize-target")
async def optimize_for_target(
    tickers: List[str],
    current_weights: List[float],
    target: str = "max_sharpe",  # max_sharpe, min_volatility, target_return
    target_value: Optional[float] = None,
    constraints: Optional[Dict] = None
):
    """
    Find optimal weights for a given target.
    
    Targets:
    - max_sharpe: Maximize Sharpe ratio
    - min_volatility: Minimize portfolio volatility
    - target_return: Hit specific return with min risk
    """
    import pandas as pd
    from datetime import datetime, timedelta
    
    data_service = get_data_service()
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=252)).strftime('%Y-%m-%d')
    
    returns_dict = {}
    for ticker in tickers:
        try:
            _, returns = data_service.get_returns(ticker, start_date, end_date)
            if not returns.empty:
                returns_dict[ticker] = returns
        except:
            pass
    
    if not returns_dict:
        raise HTTPException(status_code=404, detail="No valid data")
    
    returns_df = pd.DataFrame(returns_dict).dropna()
    current = dict(zip(tickers, current_weights))
    
    analyzer = create_what_if_analyzer()
    result = analyzer.optimize_for_target(
        returns_df=returns_df,
        current_weights=current,
        target=target,
        target_value=target_value,
        constraints=constraints or {}
    )
    
    return {
        'tickers': tickers,
        'current_weights': current_weights,
        'optimal_weights': [result['optimal_weights'].get(t, 0) for t in tickers],
        'current_metrics': result['current_metrics'],
        'optimal_metrics': result['optimal_metrics'],
        'improvement': result['improvement']
    }


@router.get("/stress-test/{scenario}")
async def quick_stress_test(
    scenario: str,
    tickers: str,  # Comma-separated
    weights: str   # Comma-separated
):
    """
    Quick stress test for common scenarios.
    
    Scenarios:
    - market_crash: -30% market decline
    - rate_hike: +2% interest rate increase
    - volatility_spike: 2x volatility increase
    """
    import pandas as pd
    from datetime import datetime, timedelta
    
    ticker_list = [t.strip() for t in tickers.split(',')]
    weight_list = [float(w.strip()) for w in weights.split(',')]
    
    if len(ticker_list) != len(weight_list):
        raise HTTPException(status_code=400, detail="Tickers and weights must match")
    
    data_service = get_data_service()
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=252)).strftime('%Y-%m-%d')
    
    returns_dict = {}
    for ticker in ticker_list:
        try:
            _, returns = data_service.get_returns(ticker, start_date, end_date)
            if not returns.empty:
                returns_dict[ticker] = returns
        except:
            pass
    
    if not returns_dict:
        raise HTTPException(status_code=404, detail="No valid data")
    
    returns_df = pd.DataFrame(returns_dict).dropna()
    weights_dict = dict(zip(ticker_list, weight_list))
    
    # Calculate base metrics
    analyzer = create_what_if_analyzer()
    base_stats = analyzer.calculate_stats(returns_df, weights_dict)
    
    # Apply scenario shocks
    scenarios = {
        'market_crash': {'shock': -0.30, 'description': '30% market decline'},
        'rate_hike': {'shock': -0.10, 'description': '10% equity decline from rate hike'},
        'volatility_spike': {'vol_mult': 2.0, 'description': '2x volatility increase'},
        'recovery': {'shock': 0.20, 'description': '20% market recovery'}
    }
    
    if scenario not in scenarios:
        raise HTTPException(status_code=400, detail=f"Unknown scenario. Available: {list(scenarios.keys())}")
    
    scenario_params = scenarios[scenario]
    
    if 'shock' in scenario_params:
        stressed_return = base_stats['annualized_return'] + scenario_params['shock']
        stressed_volatility = base_stats['volatility']
    else:
        stressed_return = base_stats['annualized_return']
        stressed_volatility = base_stats['volatility'] * scenario_params['vol_mult']
    
    # Recalculate VaR under stress
    stressed_var_95 = abs(stressed_return / 252 - 1.645 * stressed_volatility / (252**0.5))
    
    return {
        'scenario': scenario,
        'description': scenario_params['description'],
        'base_metrics': base_stats,
        'stressed_metrics': {
            'annualized_return': stressed_return,
            'volatility': stressed_volatility,
            'var_95': stressed_var_95,
            'expected_loss_pct': scenario_params.get('shock', 0) * 100
        },
        'impact': {
            'return_change': stressed_return - base_stats['annualized_return'],
            'volatility_change': stressed_volatility - base_stats['volatility']
        }
    }
