"""
Sentiment & Digital Twin Pydantic Schemas
=========================================
Request/Response models for sentiment and simulation endpoints.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from enum import Enum


class SentimentLabel(str, Enum):
    """Sentiment classification labels."""
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"


# ============================================================================
# SENTIMENT REQUEST SCHEMAS
# ============================================================================

class SentimentRequest(BaseModel):
    """Request for sentiment analysis."""
    ticker: str = Field(..., description="Stock ticker symbol")
    include_news: bool = Field(True, description="Include news sentiment")
    include_social: bool = Field(False, description="Include social media sentiment")


class SentimentVaRRequest(BaseModel):
    """Request for sentiment-adjusted VaR."""
    ticker: str = Field(..., description="Stock ticker symbol")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    confidence: float = Field(0.95, ge=0.9, le=0.99, description="Confidence level")
    base_method: str = Field("historical", description="Base VaR method")


class PortfolioSentimentRequest(BaseModel):
    """Request for portfolio sentiment analysis."""
    tickers: List[str] = Field(..., min_length=1, description="List of tickers")
    weights: Dict[str, float] = Field(..., description="Portfolio weights")


# ============================================================================
# DIGITAL TWIN REQUEST SCHEMAS
# ============================================================================

class DigitalTwinRequest(BaseModel):
    """Request for digital twin scenario comparison."""
    tickers: List[str] = Field(..., min_length=2, description="List of tickers")
    weights: Dict[str, float] = Field(..., description="Current portfolio weights")
    start_date: str = Field(..., description="Historical data start date")
    end_date: str = Field(..., description="Historical data end date")
    initial_capital: float = Field(100000, gt=0, description="Initial portfolio value")
    horizon_days: int = Field(252, ge=21, le=504, description="Simulation horizon")


class ScenarioConfig(BaseModel):
    """Configuration for a single scenario."""
    name: str
    rebalance_frequency: str = Field("monthly", description="none, monthly, quarterly, annually")
    enable_tax_loss_harvesting: bool = False
    transaction_cost_bps: float = Field(10, ge=0, le=100)
    rebalance_threshold: float = Field(0.05, ge=0, le=0.5)


class CustomScenarioRequest(BaseModel):
    """Request for custom scenario simulation."""
    tickers: List[str] = Field(..., min_length=2)
    weights: Dict[str, float]
    start_date: str
    end_date: str
    initial_capital: float = 100000
    horizon_days: int = 252
    scenario: ScenarioConfig


# ============================================================================
# WHAT-IF REQUEST SCHEMAS
# ============================================================================

class WhatIfRequest(BaseModel):
    """Request for what-if analysis."""
    tickers: List[str] = Field(..., min_length=2)
    current_weights: Dict[str, float]
    new_weights: Dict[str, float]
    start_date: str
    end_date: str
    risk_free_rate: float = Field(0.05, ge=0, le=0.15)
    transaction_cost_bps: float = Field(10, ge=0, le=100)


class RebalanceTradesRequest(BaseModel):
    """Request for rebalance trade calculation."""
    tickers: List[str]
    current_weights: Dict[str, float]
    target_weights: Dict[str, float]
    portfolio_value: float = Field(100000, gt=0)
    transaction_cost_bps: float = Field(10, ge=0, le=100)


class OptimizeTargetRequest(BaseModel):
    """Request to optimize for a target metric."""
    tickers: List[str]
    current_weights: Dict[str, float]
    start_date: str
    end_date: str
    target_return: Optional[float] = None
    target_volatility: Optional[float] = None
    target_sharpe: Optional[float] = None
    min_weight: float = Field(0.0, ge=0)
    max_weight: float = Field(1.0, le=1)


# ============================================================================
# SENTIMENT RESPONSE SCHEMAS
# ============================================================================

class NewsArticle(BaseModel):
    """Individual news article sentiment."""
    title: str
    source: str
    published: str
    sentiment_score: float
    sentiment_label: SentimentLabel


class SentimentResponse(BaseModel):
    """Sentiment analysis response."""
    ticker: str
    overall_score: float = Field(..., ge=-1, le=1, description="Overall sentiment -1 to 1")
    overall_label: SentimentLabel
    news_count: int
    trending_score: float
    whale_activity_score: float
    sentiment_var_adjustment: float = Field(..., description="VaR adjustment multiplier")
    recent_news: List[NewsArticle] = Field(default_factory=list)


class SentimentVaRResponse(BaseModel):
    """Sentiment-adjusted VaR response."""
    ticker: str
    confidence: float
    horizon: int
    base_var: float
    adjusted_var: float
    base_cvar: float
    adjusted_cvar: float
    adjustment_factor: float
    sentiment: Dict[str, Any]
    method: str


class PortfolioSentimentResponse(BaseModel):
    """Portfolio sentiment response."""
    base_var: float
    adjusted_var: float
    adjustment_factor: float
    per_asset_sentiment: Dict[str, Dict[str, Any]]
    confidence: float


# ============================================================================
# DIGITAL TWIN RESPONSE SCHEMAS
# ============================================================================

class ScenarioResult(BaseModel):
    """Result from a single scenario simulation."""
    name: str
    mean_return: float
    std_return: float
    sharpe_ratio: float
    var_95: float
    cvar_95: float
    max_drawdown: float
    transaction_costs: float
    tax_benefits: float
    final_value_median: float
    final_value_5th: float
    final_value_95th: float
    rebalance_frequency: str


class DigitalTwinResponse(BaseModel):
    """Digital twin comparison response."""
    scenarios: Dict[str, ScenarioResult]
    best_scenario: str
    recommendation: str
    portfolio_health_score: float


class PortfolioHealthScore(BaseModel):
    """Portfolio health assessment."""
    overall_score: float = Field(..., ge=0, le=100)
    diversification_score: float
    risk_efficiency_score: float
    momentum_score: float
    correlation_score: float
    recommendations: List[str]


# ============================================================================
# WHAT-IF RESPONSE SCHEMAS
# ============================================================================

class WhatIfScenario(BaseModel):
    """What-if scenario result."""
    name: str
    weights: Dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    var_95: float
    max_drawdown: float
    turnover: float


class WhatIfResponse(BaseModel):
    """What-if analysis response."""
    current: WhatIfScenario
    proposed: WhatIfScenario
    delta: Dict[str, float]
    improvement: Dict[str, bool]


class RebalanceTradeResponse(BaseModel):
    """Rebalance trade details."""
    asset: str
    current_weight: float
    new_weight: float
    change: float
    trade_value: float
    action: str  # BUY or SELL
    transaction_cost: float


class RebalanceTradesResponse(BaseModel):
    """Full rebalance response."""
    trades: List[RebalanceTradeResponse]
    total_transaction_cost: float
    total_turnover: float
    portfolio_value: float


class OptimizeTargetResponse(BaseModel):
    """Optimization result."""
    optimal_weights: Dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    target_achieved: bool
    constraint_binding: Optional[str] = None
