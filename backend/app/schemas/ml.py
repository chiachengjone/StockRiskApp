"""
Machine Learning Pydantic Schemas
=================================
Request/Response models for ML endpoints.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any


# ============================================================================
# REQUEST SCHEMAS
# ============================================================================

class MLPredictionRequest(BaseModel):
    """Request for ML VaR prediction."""
    ticker: str = Field(..., description="Stock ticker symbol")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    model_type: str = Field("auto", description="Model type: xgboost, gradientboosting, or auto")


class EnsemblePredictionRequest(BaseModel):
    """Request for ensemble VaR prediction."""
    ticker: str = Field(..., description="Stock ticker symbol")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    models: Optional[List[str]] = Field(
        None, 
        description="Models to include: ml, garch, historical, parametric, ewma"
    )


class BacktestRequest(BaseModel):
    """Request for VaR backtest."""
    ticker: str = Field(..., description="Stock ticker symbol")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    window: int = Field(252, ge=60, le=504, description="Rolling window for VaR estimation")
    confidence: float = Field(0.95, ge=0.9, le=0.99)


class VolatilityForecastRequest(BaseModel):
    """Request for volatility forecast."""
    ticker: str = Field(..., description="Stock ticker symbol")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    horizon: int = Field(10, ge=1, le=60, description="Forecast horizon in days")


class BootstrapConfidenceRequest(BaseModel):
    """Request for bootstrap confidence intervals."""
    ticker: str = Field(..., description="Stock ticker symbol")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    n_bootstrap: int = Field(100, ge=50, le=500)


# ============================================================================
# RESPONSE SCHEMAS
# ============================================================================

class MLPredictionResponse(BaseModel):
    """Response with ML prediction results."""
    ticker: str
    model_type: str
    predicted_var: float
    training_score: float
    test_score: float
    feature_importance: Dict[str, float]
    features_used: List[str]
    garch_approximation: Optional[float] = None


class EnsemblePredictionResponse(BaseModel):
    """Response with ensemble prediction results."""
    ticker: str
    ensemble_var: float
    simple_average: float
    median_var: float
    individual_predictions: Dict[str, float]
    weights: Dict[str, float]
    n_models: int
    spread: float
    best_model: str
    worst_model: str


class BacktestResponse(BaseModel):
    """Response with VaR backtest results."""
    ticker: str
    window: int
    confidence: float
    violations: int
    total_days: int
    violation_rate: float
    expected_rate: float
    ratio: float
    status: str
    interpretation: str


class VolatilityForecastResponse(BaseModel):
    """Response with volatility forecasts."""
    ticker: str
    horizon: int
    current_volatility: float
    forecasts: Dict[str, List[float]]
    ensemble_forecast: List[float]
    vol_trend: str


class BootstrapConfidenceResponse(BaseModel):
    """Response with bootstrap confidence intervals."""
    ticker: str
    mean_var: float
    median_var: float
    std_var: float
    ci_lower: float
    ci_upper: float
    ci_5: float
    ci_95: float
    n_bootstrap: int
    confidence_width: float


class FeatureImportanceResponse(BaseModel):
    """Response with feature importance data."""
    ticker: str
    features: List[str]
    importance: List[float]
    descriptions: List[str]
    total_features: int
    top_feature: Optional[str]
    top_importance: Optional[float]


class ModelComparisonResponse(BaseModel):
    """Response comparing multiple VaR methods."""
    ticker: str
    methods: List[str]
    var_estimates: List[float]
    descriptions: List[str]
