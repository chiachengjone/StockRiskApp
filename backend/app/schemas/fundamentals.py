"""
Fundamentals Analysis Pydantic Schemas
======================================
Request/Response models for fundamental analysis endpoints.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any


# ============================================================================
# REQUEST SCHEMAS
# ============================================================================

class FundamentalsRequest(BaseModel):
    """Request for fundamental analysis."""
    ticker: str = Field(..., description="Stock ticker symbol")


class DCFRequest(BaseModel):
    """Request for DCF valuation."""
    ticker: str = Field(..., description="Stock ticker symbol")
    growth_rate: float = Field(0.10, ge=0, le=0.5, description="Expected growth rate")
    terminal_growth: float = Field(0.03, ge=0, le=0.05, description="Terminal growth rate")
    discount_rate: float = Field(0.10, ge=0.05, le=0.20, description="Discount rate (WACC)")
    projection_years: int = Field(5, ge=3, le=10, description="Years to project")


class PeerComparisonRequest(BaseModel):
    """Request for peer comparison."""
    ticker: str = Field(..., description="Stock ticker symbol")
    peers: Optional[List[str]] = Field(None, description="List of peer tickers (auto-select if not provided)")


class QualityScoreRequest(BaseModel):
    """Request for quality score calculation."""
    ticker: str = Field(..., description="Stock ticker symbol")


# ============================================================================
# RESPONSE SCHEMAS
# ============================================================================

class ValuationMetrics(BaseModel):
    """Valuation metrics subset."""
    pe_ratio: Optional[float] = None
    forward_pe: Optional[float] = None
    price_to_book: Optional[float] = None
    price_to_sales: Optional[float] = None
    peg_ratio: Optional[float] = None
    ev_to_ebitda: Optional[float] = None
    score: int = Field(default=0, description="Valuation score 0-10")
    assessments: List[str] = Field(default_factory=list)


class ProfitabilityMetrics(BaseModel):
    """Profitability metrics subset."""
    roe: Optional[float] = None
    roa: Optional[float] = None
    profit_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    score: int = Field(default=0, description="Profitability score 0-10")
    assessments: List[str] = Field(default_factory=list)


class FinancialHealthMetrics(BaseModel):
    """Financial health metrics subset."""
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    score: int = Field(default=0, description="Financial health score 0-10")
    assessments: List[str] = Field(default_factory=list)


class GrowthMetrics(BaseModel):
    """Growth metrics subset."""
    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None
    score: int = Field(default=0, description="Growth score 0-10")
    assessments: List[str] = Field(default_factory=list)


class FundamentalsResponse(BaseModel):
    """Comprehensive fundamentals response."""
    ticker: str
    name: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    valuation: ValuationMetrics
    profitability: ProfitabilityMetrics
    financial_health: FinancialHealthMetrics
    growth: GrowthMetrics
    quality_score: float = Field(..., description="Overall quality score 0-100")


class DCFResponse(BaseModel):
    """DCF valuation response."""
    ticker: str
    fair_value: float = Field(..., description="Estimated fair value per share")
    current_price: float
    upside_potential: float = Field(..., description="Percentage upside/downside")
    intrinsic_value: float
    terminal_value: float
    present_value_cf: float
    present_value_terminal: float
    assumptions: Dict[str, float]
    sensitivity: Dict[str, float]  # Fair value at different discount rates


class PeerComparisonResponse(BaseModel):
    """Peer comparison response."""
    ticker: str
    peers: List[str]
    comparison: Dict[str, Dict[str, Optional[float]]]  # {ticker: {metric: value}}
    rankings: Dict[str, int]  # {metric: rank}
    percentiles: Dict[str, float]  # {metric: percentile}


class QualityScoreResponse(BaseModel):
    """Quality score response."""
    ticker: str
    overall_score: float = Field(..., description="Quality score 0-100")
    valuation_score: float
    profitability_score: float
    financial_health_score: float
    growth_score: float
    grade: str = Field(..., description="Letter grade A-F")
    recommendation: str
