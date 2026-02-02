"""
Machine Learning API Routes
===========================
Endpoints for ML-based predictions and ensemble models.
"""

from fastapi import APIRouter, HTTPException
from typing import List, Optional
import logging

from app.schemas.ml import (
    MLPredictionRequest, MLPredictionResponse,
    EnsemblePredictionRequest, EnsemblePredictionResponse,
    BacktestRequest, BacktestResponse,
    VolatilityForecastRequest, VolatilityForecastResponse,
    BootstrapConfidenceRequest, BootstrapConfidenceResponse,
    FeatureImportanceResponse, ModelComparisonResponse
)
from app.services.data_service import get_data_service
from app.services.ml_service import get_ml_predictor

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/predict", response_model=MLPredictionResponse)
async def ml_prediction(request: MLPredictionRequest):
    """
    ML-based VaR prediction using XGBoost or GradientBoosting.
    
    Uses engineered features including volatility, momentum, and RSI.
    """
    data_service = get_data_service()
    prices, returns = data_service.get_returns(
        request.ticker, request.start_date, request.end_date
    )
    
    if returns.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {request.ticker}")
    
    predictor = get_ml_predictor()
    result = predictor.train_predict(returns, prices)
    
    if 'error' in result:
        raise HTTPException(status_code=400, detail=result['error'])
    
    return MLPredictionResponse(
        ticker=request.ticker,
        model_type=result['model_type'],
        predicted_var=result['predicted_var'],
        training_score=result['training_score'],
        test_score=result['test_score'],
        feature_importance=result['feature_importance'],
        features_used=result['features_used'],
        garch_approximation=result.get('garch_approximation')
    )


@router.post("/ensemble", response_model=EnsemblePredictionResponse)
async def ensemble_prediction(request: EnsemblePredictionRequest):
    """
    Ensemble VaR prediction combining multiple models.
    
    Combines ML, GARCH, Historical, Parametric, and EWMA predictions.
    """
    data_service = get_data_service()
    prices, returns = data_service.get_returns(
        request.ticker, request.start_date, request.end_date
    )
    
    if returns.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {request.ticker}")
    
    predictor = get_ml_predictor()
    result = predictor.ensemble_predict(returns, prices, request.models)
    
    if 'error' in result:
        raise HTTPException(status_code=400, detail=result['error'])
    
    return EnsemblePredictionResponse(
        ticker=request.ticker,
        ensemble_var=result['ensemble_var'],
        simple_average=result['simple_average'],
        median_var=result['median_var'],
        individual_predictions=result['individual_predictions'],
        weights=result['weights'],
        n_models=result['n_models'],
        spread=result['spread'],
        best_model=result['best_model'],
        worst_model=result['worst_model']
    )


@router.post("/backtest", response_model=BacktestResponse)
async def backtest_var(request: BacktestRequest):
    """
    Backtest VaR predictions to validate model accuracy.
    
    Counts violations and compares to expected rate.
    """
    data_service = get_data_service()
    prices, returns = data_service.get_returns(
        request.ticker, request.start_date, request.end_date
    )
    
    if returns.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {request.ticker}")
    
    predictor = get_ml_predictor()
    result = predictor.backtest_var(returns, request.window, request.confidence)
    
    if 'error' in result:
        raise HTTPException(status_code=400, detail=result['error'])
    
    return BacktestResponse(
        ticker=request.ticker,
        window=request.window,
        confidence=request.confidence,
        violations=result['violations'],
        total_days=result['total_days'],
        violation_rate=result['violation_rate'],
        expected_rate=result['expected_rate'],
        ratio=result['ratio'],
        status=result['status'],
        interpretation=result['interpretation']
    )


@router.post("/volatility-forecast", response_model=VolatilityForecastResponse)
async def volatility_forecast(request: VolatilityForecastRequest):
    """
    Multi-model volatility forecast.
    
    Combines GARCH, EWMA, and historical volatility forecasts.
    """
    data_service = get_data_service()
    prices, returns = data_service.get_returns(
        request.ticker, request.start_date, request.end_date
    )
    
    if returns.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {request.ticker}")
    
    predictor = get_ml_predictor()
    result = predictor.volatility_forecast(returns, request.horizon)
    
    return VolatilityForecastResponse(
        ticker=request.ticker,
        horizon=result['horizon'],
        current_volatility=result['current_volatility'],
        forecasts=result['forecasts'],
        ensemble_forecast=result['ensemble_forecast'],
        vol_trend=result['vol_trend']
    )


@router.post("/bootstrap", response_model=BootstrapConfidenceResponse)
async def bootstrap_confidence(request: BootstrapConfidenceRequest):
    """
    Bootstrap confidence intervals for VaR prediction.
    
    Resamples data to quantify prediction uncertainty.
    """
    data_service = get_data_service()
    prices, returns = data_service.get_returns(
        request.ticker, request.start_date, request.end_date
    )
    
    if returns.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {request.ticker}")
    
    predictor = get_ml_predictor()
    result = predictor.predict_with_confidence(returns, request.n_bootstrap)
    
    if 'error' in result:
        raise HTTPException(status_code=400, detail=result['error'])
    
    return BootstrapConfidenceResponse(
        ticker=request.ticker,
        mean_var=result['mean_var'],
        median_var=result['median_var'],
        std_var=result['std_var'],
        ci_lower=result['ci_lower'],
        ci_upper=result['ci_upper'],
        ci_5=result['ci_5'],
        ci_95=result['ci_95'],
        n_bootstrap=result['n_bootstrap'],
        confidence_width=result['confidence_width']
    )


@router.post("/feature-importance", response_model=FeatureImportanceResponse)
async def get_feature_importance(request: MLPredictionRequest):
    """
    Get feature importance from the ML model.
    
    Useful for understanding what drives risk predictions.
    """
    data_service = get_data_service()
    prices, returns = data_service.get_returns(
        request.ticker, request.start_date, request.end_date
    )
    
    if returns.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {request.ticker}")
    
    predictor = get_ml_predictor()
    result = predictor.get_feature_importance(returns, prices)
    
    if 'error' in result:
        raise HTTPException(status_code=400, detail=result['error'])
    
    return FeatureImportanceResponse(
        ticker=request.ticker,
        features=result['features'],
        importance=result['importance'],
        descriptions=result['descriptions'],
        total_features=result['total_features'],
        top_feature=result['top_feature'],
        top_importance=result['top_importance']
    )
