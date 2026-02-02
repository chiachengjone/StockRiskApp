"""
Risk Analysis Pydantic Schemas
==============================
Request/Response models for risk endpoints.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import date


# ============================================================================
# REQUEST SCHEMAS
# ============================================================================

class RiskMetricsRequest(BaseModel):
    """Request for basic risk metrics calculation."""
    ticker: str = Field(..., description="Stock ticker symbol")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")


class VaRRequest(BaseModel):
    """Request for VaR calculation."""
    ticker: str = Field(..., description="Stock ticker symbol")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    confidence: float = Field(0.95, ge=0.9, le=0.99, description="Confidence level")
    method: str = Field("all", description="VaR method: parametric, historical, cvar, or all")


class GARCHRequest(BaseModel):
    """Request for GARCH volatility modeling."""
    ticker: str = Field(..., description="Stock ticker symbol")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    p: int = Field(1, ge=1, le=3, description="GARCH p parameter")
    q: int = Field(1, ge=1, le=3, description="GARCH q parameter")


class EVTRequest(BaseModel):
    """Request for Extreme Value Theory analysis."""
    ticker: str = Field(..., description="Stock ticker symbol")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    threshold_percentile: float = Field(0.95, ge=0.9, le=0.99)


class MonteCarloRequest(BaseModel):
    """Request for Monte Carlo simulation."""
    ticker: str = Field(..., description="Stock ticker symbol")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    n_simulations: int = Field(10000, ge=1000, le=100000)
    horizon_days: int = Field(10, ge=1, le=252)


class StressTestRequest(BaseModel):
    """Request for stress testing."""
    ticker: str = Field(..., description="Stock ticker symbol")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    scenario: Optional[str] = Field(None, description="Predefined scenario name")
    custom_shock: Optional[float] = Field(None, description="Custom market shock (-0.5 to 0.5)")


class RollingMetricsRequest(BaseModel):
    """Request for rolling risk metrics."""
    ticker: str = Field(..., description="Stock ticker symbol")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    window_days: int = Field(21, ge=5, le=252, description="Rolling window in trading days")


# ============================================================================
# RESPONSE SCHEMAS
# ============================================================================

class RiskMetricsResponse(BaseModel):
    """Response with basic risk metrics."""
    ticker: str
    period: Dict[str, str]
    metrics: Dict[str, Optional[float]]
    
    class Config:
        json_schema_extra = {
            "example": {
                "ticker": "AAPL",
                "period": {"start": "2023-01-01", "end": "2024-01-01"},
                "metrics": {
                    "annualized_return": 0.25,
                    "annualized_volatility": 0.22,
                    "sharpe_ratio": 1.14,
                    "sortino_ratio": 1.45,
                    "max_drawdown": -0.15,
                    "calmar_ratio": 1.67,
                    "skewness": -0.3,
                    "kurtosis": 4.2
                }
            }
        }


class VaRResponse(BaseModel):
    """Response with VaR calculations."""
    ticker: str
    confidence: float
    var_parametric: Optional[float] = None
    var_historical: Optional[float] = None
    cvar: Optional[float] = None
    var_interpretation: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "ticker": "AAPL",
                "confidence": 0.95,
                "var_parametric": -0.0245,
                "var_historical": -0.0232,
                "cvar": -0.0312,
                "var_interpretation": "At 95% confidence, daily loss will not exceed 2.45%"
            }
        }


class GARCHResponse(BaseModel):
    """Response with GARCH model results."""
    ticker: str
    model_type: str
    omega: float
    alpha: float
    beta: float
    persistence: float
    current_volatility: float
    forecast_volatility: float
    annualized_volatility: float
    conditional_var: float


class EVTResponse(BaseModel):
    """Response with EVT analysis."""
    ticker: str
    threshold: float
    shape_parameter: float
    evt_var: float
    excesses_count: int
    tail_index: Optional[float] = None


class MonteCarloResponse(BaseModel):
    """Response with Monte Carlo simulation results."""
    ticker: str
    n_simulations: int
    horizon_days: int
    mean_return: float
    std_return: float
    var_95: float
    var_99: float
    worst_case: float
    best_case: float
    prob_loss_10pct: float
    percentiles: Dict[str, float]


class StressTestResponse(BaseModel):
    """Response with stress test results."""
    ticker: str
    scenario_name: str
    market_shock: float
    estimated_loss: float
    beta: float
    stressed_volatility: float
    scenario_description: Optional[str] = None


class RollingMetricsResponse(BaseModel):
    """Response with rolling metrics time series."""
    ticker: str
    window_days: int
    metrics: Dict[str, List[Optional[float]]]
    dates: List[str]


class StressScenariosListResponse(BaseModel):
    """List of available stress scenarios."""
    scenarios: List[Dict[str, Any]]
