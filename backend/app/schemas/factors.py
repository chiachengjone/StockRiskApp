"""
Factor Analysis Pydantic Schemas
================================
Request/Response models for factor analysis endpoints.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any


# ============================================================================
# REQUEST SCHEMAS
# ============================================================================

class FamaFrenchRequest(BaseModel):
    """Request for Fama-French factor regression."""
    ticker: str = Field(..., description="Stock ticker symbol")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    benchmark: Optional[str] = Field("^GSPC", description="Benchmark ticker for beta calculation")


class KellyRequest(BaseModel):
    """Request for Kelly Criterion calculation."""
    ticker: str = Field(..., description="Stock ticker symbol")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    fraction: float = Field(0.5, ge=0.1, le=1.0, description="Kelly fraction (0.5 = half Kelly)")


class ESGRequest(BaseModel):
    """Request for ESG rating."""
    ticker: str = Field(..., description="Stock ticker symbol")


class StyleFactorRequest(BaseModel):
    """Request for style factor analysis."""
    ticker: str = Field(..., description="Stock ticker symbol")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    benchmark: Optional[str] = Field("^GSPC", description="Benchmark ticker")


# ============================================================================
# RESPONSE SCHEMAS
# ============================================================================

class FamaFrenchResponse(BaseModel):
    """Response with Fama-French factor analysis results."""
    ticker: str
    alpha: float = Field(..., description="Annualized alpha")
    r_squared: float = Field(..., description="Model R-squared")
    loadings: Dict[str, float] = Field(..., description="Factor loadings")
    n_observations: int
    t_stats: Optional[Dict[str, float]] = None


class KellyResponse(BaseModel):
    """Response with Kelly Criterion results."""
    ticker: str
    kelly_pct: float = Field(..., description="Optimal position size")
    full_kelly: float = Field(..., description="Full Kelly position")
    win_rate: float
    win_loss_ratio: float
    edge_per_trade: float
    fraction_used: float


class ESGResponse(BaseModel):
    """Response with ESG rating."""
    ticker: str
    rating: str = Field(..., description="Overall ESG rating (AAA, AA, A, BBB, BB, B, CCC)")
    total_esg: Optional[float] = None
    environment_score: Optional[float] = None
    social_score: Optional[float] = None
    governance_score: Optional[float] = None
    source: str = Field(default="Yahoo Finance")


class StyleFactorResponse(BaseModel):
    """Response with style factor analysis."""
    ticker: str
    overall_score: float
    scores: Dict[str, float]
    style_label: str
    style_classification: Dict[str, str]
    momentum_details: Dict[str, Any]
    value_details: Dict[str, Any]
    quality_details: Dict[str, Any]
    radar_data: Dict[str, Any]
