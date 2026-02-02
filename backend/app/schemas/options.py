"""
Options Analytics Pydantic Schemas
==================================
Request/Response models for options endpoints.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any


# ============================================================================
# REQUEST SCHEMAS
# ============================================================================

class BlackScholesRequest(BaseModel):
    """Request for Black-Scholes option pricing."""
    spot_price: float = Field(..., gt=0, description="Current stock price")
    strike_price: float = Field(..., gt=0, description="Option strike price")
    time_to_expiry: float = Field(..., gt=0, description="Time to expiration in years")
    risk_free_rate: float = Field(0.05, description="Risk-free interest rate")
    volatility: float = Field(..., gt=0, le=5, description="Annualized volatility")
    option_type: str = Field("call", description="Option type: 'call' or 'put'")


class GreeksRequest(BaseModel):
    """Request for option Greeks calculation."""
    spot_price: float = Field(..., gt=0, description="Current stock price")
    strike_price: float = Field(..., gt=0, description="Option strike price")
    time_to_expiry: float = Field(..., gt=0, description="Time to expiration in years")
    risk_free_rate: float = Field(0.05, description="Risk-free interest rate")
    volatility: float = Field(..., gt=0, le=5, description="Annualized volatility")
    option_type: str = Field("call", description="Option type: 'call' or 'put'")


class ImpliedVolRequest(BaseModel):
    """Request for implied volatility calculation."""
    market_price: float = Field(..., gt=0, description="Market price of the option")
    spot_price: float = Field(..., gt=0, description="Current stock price")
    strike_price: float = Field(..., gt=0, description="Option strike price")
    time_to_expiry: float = Field(..., gt=0, description="Time to expiration in years")
    risk_free_rate: float = Field(0.05, description="Risk-free interest rate")
    option_type: str = Field("call", description="Option type: 'call' or 'put'")


class OptionAnalysisRequest(BaseModel):
    """Request for comprehensive option analysis."""
    ticker: str = Field(..., description="Underlying stock ticker")
    spot_price: float = Field(..., gt=0, description="Current stock price")
    strike_price: float = Field(..., gt=0, description="Option strike price")
    time_to_expiry: float = Field(..., gt=0, description="Time to expiration in years")
    risk_free_rate: float = Field(0.05, description="Risk-free interest rate")
    volatility: float = Field(..., gt=0, le=5, description="Annualized volatility")
    option_type: str = Field("call", description="Option type: 'call' or 'put'")
    position_size: int = Field(1, ge=1, description="Number of contracts")


class VolatilitySurfaceRequest(BaseModel):
    """Request for volatility surface data."""
    ticker: str = Field(..., description="Stock ticker")
    strikes: List[float] = Field(..., description="List of strike prices")
    expiries: List[float] = Field(..., description="List of expiry times in years")
    spot_price: float = Field(..., gt=0, description="Current spot price")


# ============================================================================
# RESPONSE SCHEMAS
# ============================================================================

class BlackScholesResponse(BaseModel):
    """Response with option price."""
    price: float
    option_type: str
    intrinsic_value: float
    time_value: float
    moneyness: str  # ITM, ATM, OTM


class GreeksResponse(BaseModel):
    """Response with option Greeks."""
    delta: float = Field(..., description="Price sensitivity to underlying")
    gamma: float = Field(..., description="Delta sensitivity to underlying")
    theta: float = Field(..., description="Time decay per day")
    vega: float = Field(..., description="Sensitivity to 1% vol change")
    rho: float = Field(..., description="Sensitivity to 1% rate change")


class ImpliedVolResponse(BaseModel):
    """Response with implied volatility."""
    implied_volatility: float
    annualized_iv: float
    converged: bool


class OptionAnalysisResponse(BaseModel):
    """Comprehensive option analysis response."""
    ticker: str
    option_type: str
    price: float
    greeks: GreeksResponse
    breakeven: float
    max_profit: Optional[float]
    max_loss: float
    probability_itm: float
    position_value: float
    risk_reward_ratio: Optional[float]


class VolatilitySurfaceResponse(BaseModel):
    """Response with volatility surface data."""
    ticker: str
    surface_data: List[Dict[str, float]]  # [{strike, expiry, iv}, ...]
    atm_vol: float
    vol_skew: float  # 25-delta risk reversal
