"""
Portfolio Pydantic Schemas
==========================
Request/Response models for portfolio endpoints.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any


# ============================================================================
# REQUEST SCHEMAS
# ============================================================================

class PortfolioWeights(BaseModel):
    """Portfolio weights specification."""
    weights: Dict[str, float] = Field(..., description="Ticker to weight mapping")


class RiskParityRequest(BaseModel):
    """Request for Risk Parity optimization."""
    tickers: List[str] = Field(..., min_length=2, max_length=20)
    start_date: str
    end_date: str
    target_risk: float = Field(0.10, ge=0.01, le=0.50)
    max_weight: float = Field(0.40, ge=0.10, le=1.0)
    min_weight: float = Field(0.02, ge=0.0, le=0.20)


class HRPRequest(BaseModel):
    """Request for Hierarchical Risk Parity."""
    tickers: List[str] = Field(..., min_length=2, max_length=20)
    start_date: str
    end_date: str
    method: str = Field("complete", description="Linkage method: single, complete, average, ward")


class BlackLittermanView(BaseModel):
    """Single Black-Litterman view."""
    asset: Optional[str] = Field(None, description="Asset for absolute view")
    long_asset: Optional[str] = Field(None, description="Long asset for relative view")
    short_asset: Optional[str] = Field(None, description="Short asset for relative view")
    expected_return: float = Field(..., description="Expected return (e.g., 0.10 for 10%)")


class BlackLittermanRequest(BaseModel):
    """Request for Black-Litterman optimization."""
    tickers: List[str] = Field(..., min_length=2, max_length=20)
    start_date: str
    end_date: str
    market_caps: Dict[str, float] = Field(..., description="Market caps by ticker")
    views: List[BlackLittermanView] = Field(default_factory=list)
    risk_aversion: float = Field(2.5, ge=0.5, le=10.0)
    tau: float = Field(0.05, ge=0.01, le=0.5)


class EfficientFrontierRequest(BaseModel):
    """Request for efficient frontier calculation."""
    tickers: List[str] = Field(..., min_length=2, max_length=20)
    start_date: str
    end_date: str
    n_points: int = Field(50, ge=10, le=200)


class TransactionCostRequest(BaseModel):
    """Request for transaction cost analysis."""
    current_weights: Dict[str, float]
    target_weights: Dict[str, float]
    portfolio_value: float = Field(..., gt=0)
    prices: Dict[str, float]
    volumes: Optional[Dict[str, float]] = None
    spread_bps: float = Field(5.0, ge=0)
    commission_per_share: float = Field(0.005, ge=0)
    market_impact_bps: float = Field(10.0, ge=0)


class RebalanceAnalysisRequest(BaseModel):
    """Request for rebalancing analysis."""
    tickers: List[str] = Field(..., min_length=2, max_length=20)
    start_date: str
    end_date: str
    target_weights: Dict[str, float]
    cost_per_rebalance: float = Field(0.001, ge=0, le=0.05)


class ThresholdRebalanceRequest(BaseModel):
    """Request for threshold rebalancing check."""
    current_weights: Dict[str, float]
    target_weights: Dict[str, float]
    threshold: float = Field(0.05, ge=0.01, le=0.20)


class TaxLossHarvestingRequest(BaseModel):
    """Request for tax-loss harvesting analysis."""
    positions: Dict[str, Dict[str, Any]] = Field(
        ..., 
        description="Positions with cost_basis, current_value, purchase_date"
    )
    short_term_rate: float = Field(0.37, ge=0, le=0.60)
    long_term_rate: float = Field(0.20, ge=0, le=0.40)


class PortfolioVaRRequest(BaseModel):
    """Request for portfolio VaR calculation."""
    tickers: List[str] = Field(..., min_length=2, max_length=20)
    weights: List[float] = Field(..., min_length=2, max_length=20)
    start_date: str
    end_date: str
    confidence: float = Field(0.95, ge=0.9, le=0.99)


class MarginalVaRRequest(BaseModel):
    """Request for marginal VaR contribution."""
    tickers: List[str] = Field(..., min_length=2, max_length=20)
    weights: List[float] = Field(..., min_length=2, max_length=20)
    start_date: str
    end_date: str
    confidence: float = Field(0.95, ge=0.9, le=0.99)


# ============================================================================
# RESPONSE SCHEMAS
# ============================================================================

class RiskParityResponse(BaseModel):
    """Response with Risk Parity results."""
    weights: Dict[str, float]
    portfolio_volatility: float
    risk_contributions: Dict[str, float]
    leverage_for_target: float
    levered_weights: Dict[str, float]
    diversification_ratio: float
    success: bool


class HRPResponse(BaseModel):
    """Response with HRP results."""
    weights: Dict[str, float]
    sorted_order: List[str]
    portfolio_volatility: float
    method: str
    n_assets: int


class BlackLittermanResponse(BaseModel):
    """Response with Black-Litterman results."""
    weights: Dict[str, float]
    expected_returns: Dict[str, float]
    equilibrium_returns: Dict[str, float]
    market_weights: Dict[str, float]
    portfolio_return: float
    portfolio_volatility: float
    views_applied: int
    success: bool


class EfficientFrontierResponse(BaseModel):
    """Response with efficient frontier points."""
    frontier: List[Dict[str, float]]
    optimal_portfolio: Dict[str, Any]


class TransactionCostResponse(BaseModel):
    """Response with transaction cost analysis."""
    trades: List[Dict[str, Any]]
    total_cost: float
    total_cost_bps: float
    turnover: float
    turnover_pct: float
    n_trades: int
    net_of_costs_value: float


class RebalanceAnalysisResponse(BaseModel):
    """Response with rebalancing analysis."""
    optimal_frequency: str
    results: Dict[str, Dict[str, float]]
    sharpe_improvement: float


class ThresholdRebalanceResponse(BaseModel):
    """Response with threshold rebalance decision."""
    needs_rebalance: bool
    max_drift: float
    threshold: float
    drifts: Dict[str, float]
    assets_to_trade: List[str]
    avg_drift: float


class TaxLossHarvestingResponse(BaseModel):
    """Response with tax-loss harvesting opportunities."""
    opportunities: List[Dict[str, Any]]
    total_harvestable_loss: float
    potential_tax_savings: float
    n_opportunities: int
    max_annual_deduction: float


class PortfolioVaRResponse(BaseModel):
    """Response with portfolio VaR."""
    portfolio_var: float
    confidence: float
    portfolio_volatility: float
    interpretation: str


class MarginalVaRResponse(BaseModel):
    """Response with marginal VaR contributions."""
    contributions: Dict[str, float]
    portfolio_var: float
    largest_contributor: str
    smallest_contributor: str


class CorrelationAnalysisResponse(BaseModel):
    """Response with correlation analysis."""
    correlation_matrix: Dict[str, Dict[str, float]]
    avg_normal_correlation: float
    avg_crash_correlation: float
    crash_days_count: int
    total_days: int
