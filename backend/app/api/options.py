"""
Options Analytics API Routes
============================
Endpoints for Black-Scholes pricing, Greeks, and implied volatility.
"""

from fastapi import APIRouter, HTTPException
from typing import List
import logging

from app.schemas.options import (
    BlackScholesRequest, BlackScholesResponse,
    GreeksRequest, GreeksResponse,
    ImpliedVolRequest, ImpliedVolResponse,
    OptionAnalysisRequest, OptionAnalysisResponse,
    VolatilitySurfaceRequest, VolatilitySurfaceResponse
)
from app.services.options_service import get_options_analytics

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/black-scholes", response_model=BlackScholesResponse)
async def calculate_black_scholes(request: BlackScholesRequest):
    """
    Calculate option price using Black-Scholes model.
    
    The Black-Scholes model assumes:
    - European-style options
    - No dividends
    - Constant volatility
    - Log-normal price distribution
    """
    analytics = get_options_analytics()
    
    price = analytics.black_scholes(
        S=request.spot_price,
        K=request.strike_price,
        T=request.time_to_expiry,
        r=request.risk_free_rate,
        sigma=request.volatility,
        option_type=request.option_type
    )
    
    # Calculate intrinsic and time value
    if request.option_type.lower() == 'call':
        intrinsic = max(0, request.spot_price - request.strike_price)
    else:
        intrinsic = max(0, request.strike_price - request.spot_price)
    
    time_value = price - intrinsic
    
    # Moneyness
    ratio = request.spot_price / request.strike_price
    if request.option_type.lower() == 'call':
        moneyness = 'ITM' if ratio > 1.02 else 'OTM' if ratio < 0.98 else 'ATM'
    else:
        moneyness = 'ITM' if ratio < 0.98 else 'OTM' if ratio > 1.02 else 'ATM'
    
    return BlackScholesResponse(
        price=price,
        option_type=request.option_type,
        intrinsic_value=intrinsic,
        time_value=time_value,
        moneyness=moneyness
    )


@router.post("/greeks", response_model=GreeksResponse)
async def calculate_greeks(request: GreeksRequest):
    """
    Calculate option Greeks.
    
    Greeks measure option sensitivity to various factors:
    - Delta: Price change per $1 underlying move
    - Gamma: Delta change per $1 underlying move
    - Theta: Daily time decay
    - Vega: Price change per 1% volatility change
    - Rho: Price change per 1% rate change
    """
    analytics = get_options_analytics()
    
    greeks = analytics.calculate_greeks(
        S=request.spot_price,
        K=request.strike_price,
        T=request.time_to_expiry,
        r=request.risk_free_rate,
        sigma=request.volatility,
        option_type=request.option_type
    )
    
    return GreeksResponse(**greeks)


@router.post("/implied-volatility", response_model=ImpliedVolResponse)
async def calculate_implied_vol(request: ImpliedVolRequest):
    """
    Calculate implied volatility from market price.
    
    Uses Newton-Raphson iteration to find the volatility
    that matches the observed market price.
    """
    analytics = get_options_analytics()
    
    iv = analytics.implied_volatility(
        market_price=request.market_price,
        S=request.spot_price,
        K=request.strike_price,
        T=request.time_to_expiry,
        r=request.risk_free_rate,
        option_type=request.option_type
    )
    
    if iv is None:
        raise HTTPException(
            status_code=400,
            detail="Could not calculate implied volatility - price may be outside valid range"
        )
    
    return ImpliedVolResponse(
        implied_volatility=iv,
        annualized_iv=iv,
        converged=True
    )


@router.post("/analyze", response_model=OptionAnalysisResponse)
async def analyze_option(request: OptionAnalysisRequest):
    """
    Comprehensive option analysis.
    
    Returns pricing, Greeks, breakeven, and risk metrics.
    """
    analytics = get_options_analytics()
    
    result = analytics.analyze_option(
        S=request.spot_price,
        K=request.strike_price,
        T=request.time_to_expiry,
        r=request.risk_free_rate,
        sigma=request.volatility,
        option_type=request.option_type,
        position_size=request.position_size
    )
    
    greeks = GreeksResponse(**result['greeks'])
    
    return OptionAnalysisResponse(
        ticker=request.ticker,
        option_type=request.option_type,
        price=result['price'],
        greeks=greeks,
        breakeven=result['breakeven'],
        max_profit=result['max_profit'],
        max_loss=result['max_loss'],
        probability_itm=result['probability_itm'],
        position_value=result['position_value'],
        risk_reward_ratio=None if result['max_profit'] is None else result['max_profit'] / result['max_loss']
    )


@router.post("/volatility-surface", response_model=VolatilitySurfaceResponse)
async def generate_volatility_surface(request: VolatilitySurfaceRequest):
    """
    Generate volatility surface data.
    
    Creates a grid of implied volatilities across strikes and expiries.
    Uses a simple smile/skew model for demonstration.
    """
    analytics = get_options_analytics()
    
    surface_data = analytics.generate_volatility_surface(
        S=request.spot_price,
        strikes=request.strikes,
        expiries=request.expiries
    )
    
    # Calculate ATM vol and skew
    atm_strike = min(request.strikes, key=lambda x: abs(x - request.spot_price))
    atm_data = [d for d in surface_data if d['strike'] == atm_strike]
    atm_vol = atm_data[0]['implied_vol'] if atm_data else 0.25
    
    # 25-delta skew approximation
    otm_put_strike = request.spot_price * 0.95
    otm_call_strike = request.spot_price * 1.05
    put_vol = next((d['implied_vol'] for d in surface_data if d['strike'] <= otm_put_strike), atm_vol)
    call_vol = next((d['implied_vol'] for d in surface_data if d['strike'] >= otm_call_strike), atm_vol)
    vol_skew = put_vol - call_vol
    
    return VolatilitySurfaceResponse(
        ticker=request.ticker,
        surface_data=surface_data,
        atm_vol=atm_vol,
        vol_skew=vol_skew
    )


@router.get("/quick/{ticker}")
async def quick_option_analysis(
    ticker: str,
    strike_pct: float = 1.0,  # Strike as % of spot (1.0 = ATM)
    expiry_days: int = 30,
    option_type: str = "call"
):
    """
    Quick option analysis using current market data.
    
    Fetches current price and calculates option metrics.
    """
    from app.services.data_service import get_data_service
    
    data_service = get_data_service()
    info = data_service.get_fundamentals(ticker)
    
    spot_price = info.get('currentPrice') or info.get('regularMarketPrice', 100)
    strike_price = spot_price * strike_pct
    time_to_expiry = expiry_days / 365
    
    # Estimate volatility from historical data
    from datetime import datetime, timedelta
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
    
    try:
        _, returns = data_service.get_returns(ticker, start_date, end_date)
        volatility = float(returns.std()) * (252 ** 0.5)
    except:
        volatility = 0.30  # Default 30%
    
    analytics = get_options_analytics()
    result = analytics.analyze_option(
        S=spot_price,
        K=strike_price,
        T=time_to_expiry,
        r=0.05,
        sigma=volatility,
        option_type=option_type
    )
    
    return {
        'ticker': ticker,
        'spot_price': spot_price,
        'strike_price': strike_price,
        'expiry_days': expiry_days,
        'estimated_volatility': volatility,
        **result
    }
