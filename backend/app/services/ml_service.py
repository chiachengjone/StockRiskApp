"""
Machine Learning Service
========================
ML-based VaR prediction and ensemble methods.

Ported from ml_predictor.py with identical calculations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from scipy.stats import norm
import logging

logger = logging.getLogger(__name__)

# Try to import XGBoost, fall back to GradientBoosting
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.warning("XGBoost not available, using GradientBoosting")

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


class MLPredictor:
    """
    Machine Learning VaR Predictor.
    
    Features:
    - XGBoost/GradientBoosting for VaR prediction
    - Feature engineering (RSI, volatility, momentum)
    - Ensemble predictions
    - Bootstrap confidence intervals
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
        self.model_type = "XGBoost" if HAS_XGBOOST else "GradientBoosting"
    
    def engineer_features(
        self,
        returns: pd.Series,
        prices: pd.Series
    ) -> pd.DataFrame:
        """
        Engineer features for ML model.
        
        Features include:
        - Rolling volatility (5, 20, 60 day)
        - Volatility ratio
        - RSI
        - Momentum
        - Return statistics
        """
        df = pd.DataFrame(index=returns.index)
        
        # Volatility features
        df['vol_5'] = returns.rolling(5).std()
        df['vol_20'] = returns.rolling(20).std()
        df['vol_60'] = returns.rolling(60).std()
        df['vol_ratio'] = df['vol_5'] / df['vol_20']
        
        # RSI calculation
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Momentum
        df['momentum_5'] = prices.pct_change(5)
        df['momentum_20'] = prices.pct_change(20)
        
        # Return statistics
        df['max_return_20'] = returns.rolling(20).max()
        df['min_return_20'] = returns.rolling(20).min()
        df['skew_20'] = returns.rolling(20).skew()
        df['kurt_20'] = returns.rolling(20).kurt()
        df['return_zscore'] = (returns - returns.rolling(20).mean()) / returns.rolling(20).std()
        
        # Lagged returns
        df['return_lag_1'] = returns.shift(1)
        df['return_lag_3'] = returns.shift(3)
        df['return_lag_5'] = returns.shift(5)
        df['return_lag_10'] = returns.shift(10)
        
        return df
    
    def train_predict(
        self,
        returns: pd.Series,
        prices: pd.Series
    ) -> Dict[str, Any]:
        """
        Train ML model and predict VaR.
        
        Uses time series cross-validation to avoid look-ahead bias.
        """
        if len(returns) < 200:
            return {"error": "Insufficient data for ML model (need 200+ observations)"}
        
        # Engineer features
        features = self.engineer_features(returns, prices)
        
        # Target: forward 5-day volatility (proxy for VaR)
        target = returns.rolling(5).std().shift(-5)
        
        # Align and clean
        data = pd.concat([features, target.rename('target')], axis=1).dropna()
        
        if len(data) < 100:
            return {"error": "Insufficient clean data after feature engineering"}
        
        X = data.drop('target', axis=1)
        y = data['target']
        
        self.feature_names = X.columns.tolist()
        
        # Time series split
        train_size = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        if HAS_XGBOOST:
            self.model = XGBRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42
            )
            self.model_type = "XGBoost"
        else:
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42
            )
            self.model_type = "GradientBoosting"
        
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        # Predict current VaR
        X_last = X.iloc[[-1]]
        X_last_scaled = self.scaler.transform(X_last)
        predicted_vol = float(self.model.predict(X_last_scaled)[0])
        predicted_var = predicted_vol * 1.645  # 95% VaR
        
        # Feature importance
        importance = dict(zip(self.feature_names, self.model.feature_importances_))
        
        # GARCH approximation for comparison
        try:
            from arch import arch_model
            returns_scaled = returns * 100
            garch = arch_model(returns_scaled, vol='Garch', p=1, q=1, mean='Zero')
            fitted = garch.fit(disp='off')
            forecast = fitted.forecast(horizon=1)
            garch_vol = np.sqrt(forecast.variance.iloc[-1].values[0]) / 100
        except:
            garch_vol = float(returns.std())
        
        return {
            "model_type": self.model_type,
            "predicted_var": predicted_var,
            "predicted_volatility": predicted_vol,
            "training_score": float(train_score),
            "test_score": float(test_score),
            "feature_importance": importance,
            "features_used": self.feature_names,
            "garch_approximation": float(garch_vol * 1.645)
        }
    
    def ensemble_predict(
        self,
        returns: pd.Series,
        prices: pd.Series,
        models: List[str] = None
    ) -> Dict[str, Any]:
        """
        Ensemble VaR prediction combining multiple models.
        """
        all_models = ['ml', 'garch', 'historical', 'parametric', 'ewma']
        models = models or all_models
        
        predictions = {}
        weights = {}
        
        # 1. ML Model
        if 'ml' in models:
            try:
                ml_result = self.train_predict(returns, prices)
                if 'error' not in ml_result:
                    predictions['ml'] = abs(ml_result['predicted_var'])
                    weights['ml'] = 0.30
            except Exception as e:
                logger.warning(f"ML prediction failed: {e}")
        
        # 2. GARCH
        if 'garch' in models:
            try:
                from arch import arch_model
                returns_scaled = returns * 100
                model = arch_model(returns_scaled, vol='Garch', p=1, q=1, mean='Zero')
                fitted = model.fit(disp='off')
                forecast = fitted.forecast(horizon=1)
                garch_vol = np.sqrt(forecast.variance.iloc[-1].values[0]) / 100
                predictions['garch'] = garch_vol * 1.645
                weights['garch'] = 0.25
            except Exception as e:
                logger.warning(f"GARCH failed: {e}")
        
        # 3. Historical
        if 'historical' in models:
            predictions['historical'] = abs(float(np.percentile(returns, 5)))
            weights['historical'] = 0.20
        
        # 4. Parametric
        if 'parametric' in models:
            mu = float(returns.mean())
            sigma = float(returns.std())
            predictions['parametric'] = abs(norm.ppf(0.05, mu, sigma))
            weights['parametric'] = 0.15
        
        # 5. EWMA
        if 'ewma' in models:
            ewma_vol = float(returns.ewm(span=20).std().iloc[-1])
            predictions['ewma'] = ewma_vol * 1.645
            weights['ewma'] = 0.10
        
        if not predictions:
            return {"error": "No models produced predictions"}
        
        # Normalize weights
        total_weight = sum(weights[m] for m in predictions.keys())
        normalized_weights = {m: weights[m] / total_weight for m in predictions.keys()}
        
        # Calculate ensemble
        ensemble_var = sum(predictions[m] * normalized_weights[m] for m in predictions.keys())
        simple_avg = np.mean(list(predictions.values()))
        median_var = np.median(list(predictions.values()))
        
        return {
            "ensemble_var": float(ensemble_var),
            "simple_average": float(simple_avg),
            "median_var": float(median_var),
            "individual_predictions": predictions,
            "weights": normalized_weights,
            "n_models": len(predictions),
            "spread": float(max(predictions.values()) - min(predictions.values())),
            "best_model": min(predictions.keys(), key=lambda m: predictions[m]),
            "worst_model": max(predictions.keys(), key=lambda m: predictions[m])
        }
    
    def backtest_var(
        self,
        returns: pd.Series,
        window: int = 252,
        confidence: float = 0.95
    ) -> Dict[str, Any]:
        """
        Backtest VaR predictions.
        
        Counts violations (days when loss exceeded VaR).
        """
        if len(returns) < window + 50:
            return {"error": "Insufficient data for backtest"}
        
        violations = 0
        predictions = []
        actuals = []
        
        for i in range(window, len(returns) - 1):
            train_rets = returns.iloc[i-window:i]
            
            # Simple VaR prediction
            var_pred = abs(float(np.percentile(train_rets, 100 * (1 - confidence))))
            actual_loss = abs(float(returns.iloc[i + 1]))
            
            predictions.append(var_pred)
            actuals.append(actual_loss)
            
            if actual_loss > var_pred:
                violations += 1
        
        n_days = len(predictions)
        violation_rate = violations / n_days if n_days > 0 else 0
        expected_rate = 1 - confidence
        ratio = violation_rate / expected_rate if expected_rate > 0 else 0
        
        # Determine status
        if 0.8 <= ratio <= 1.2:
            status = "Good"
            interpretation = "VaR model is well-calibrated"
        elif ratio < 0.8:
            status = "Conservative"
            interpretation = "VaR model is too conservative (fewer violations than expected)"
        else:
            status = "Aggressive"
            interpretation = "VaR model underestimates risk (more violations than expected)"
        
        return {
            "violations": violations,
            "total_days": n_days,
            "violation_rate": violation_rate,
            "expected_rate": expected_rate,
            "ratio": ratio,
            "status": status,
            "interpretation": interpretation
        }
    
    def predict_with_confidence(
        self,
        returns: pd.Series,
        n_bootstrap: int = 100
    ) -> Dict[str, Any]:
        """
        Bootstrap confidence intervals for VaR prediction.
        """
        if len(returns) < 100:
            return {"error": "Insufficient data for bootstrap"}
        
        predictions = []
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(len(returns), size=len(returns), replace=True)
            indices = np.sort(indices)
            sample_returns = returns.iloc[indices].reset_index(drop=True)
            
            try:
                var_pred = abs(float(np.percentile(sample_returns, 5)))
                predictions.append(var_pred)
            except:
                continue
        
        if len(predictions) < 10:
            return {"error": "Bootstrap failed"}
        
        predictions = np.array(predictions)
        
        return {
            "mean_var": float(np.mean(predictions)),
            "median_var": float(np.median(predictions)),
            "std_var": float(np.std(predictions)),
            "ci_5": float(np.percentile(predictions, 5)),
            "ci_95": float(np.percentile(predictions, 95)),
            "ci_lower": float(np.percentile(predictions, 2.5)),
            "ci_upper": float(np.percentile(predictions, 97.5)),
            "n_bootstrap": len(predictions),
            "confidence_width": float(np.percentile(predictions, 97.5) - np.percentile(predictions, 2.5))
        }
    
    def get_feature_importance(
        self,
        returns: pd.Series,
        prices: pd.Series,
        top_n: int = 10
    ) -> Dict[str, Any]:
        """
        Get feature importance data for visualization.
        """
        if not self.is_trained:
            result = self.train_predict(returns, prices)
            if 'error' in result:
                return result
        
        if self.model is None:
            return {"error": "Model not trained"}
        
        importance = dict(zip(self.feature_names, self.model.feature_importances_))
        sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        
        top_features = dict(list(sorted_importance.items())[:top_n])
        
        # Feature descriptions
        feature_descriptions = {
            'vol_5': '5-day volatility',
            'vol_20': '20-day volatility',
            'vol_60': '60-day volatility',
            'vol_ratio': 'Short/Long term vol ratio',
            'rsi_14': 'RSI (14-day)',
            'momentum_5': '5-day momentum',
            'momentum_20': '20-day momentum',
            'max_return_20': '20-day max return',
            'min_return_20': '20-day min return',
            'skew_20': '20-day skewness',
            'kurt_20': '20-day kurtosis',
            'return_zscore': 'Return z-score',
            'return_lag_1': 'Yesterday return',
            'return_lag_3': '3-day lagged return',
            'return_lag_5': '5-day lagged return',
            'return_lag_10': '10-day lagged return'
        }
        
        return {
            "features": list(top_features.keys()),
            "importance": list(top_features.values()),
            "descriptions": [feature_descriptions.get(f, f) for f in top_features.keys()],
            "total_features": len(self.feature_names),
            "top_feature": list(top_features.keys())[0] if top_features else None,
            "top_importance": list(top_features.values())[0] if top_features else 0
        }
    
    def volatility_forecast(
        self,
        returns: pd.Series,
        horizon: int = 10
    ) -> Dict[str, Any]:
        """
        Multi-model volatility forecast.
        """
        forecasts = {}
        
        # 1. GARCH forecast
        try:
            from arch import arch_model
            returns_scaled = returns * 100
            model = arch_model(returns_scaled, vol='Garch', p=1, q=1, mean='Zero')
            fitted = model.fit(disp='off')
            garch_forecast = fitted.forecast(horizon=horizon)
            garch_vol = np.sqrt(garch_forecast.variance.iloc[-1].values) / 100 * np.sqrt(252)
            forecasts['garch'] = garch_vol.tolist()
        except:
            forecasts['garch'] = None
        
        # 2. EWMA forecast
        ewma_vol = float(returns.ewm(span=20).std().iloc[-1]) * np.sqrt(252)
        forecasts['ewma'] = [ewma_vol] * horizon
        
        # 3. Historical forecast
        hist_vol = float(returns.std()) * np.sqrt(252)
        forecasts['historical'] = [hist_vol] * horizon
        
        # Average across models
        valid_forecasts = [f for f in [forecasts['garch'], forecasts['ewma'], forecasts['historical']] if f is not None]
        
        if valid_forecasts:
            avg_forecast = np.mean(valid_forecasts, axis=0).tolist()
        else:
            avg_forecast = forecasts['historical']
        
        return {
            "horizon": horizon,
            "forecasts": forecasts,
            "ensemble_forecast": avg_forecast,
            "current_volatility": float(returns.std() * np.sqrt(252)),
            "vol_trend": "increasing" if avg_forecast[-1] > avg_forecast[0] else "decreasing"
        }


# Singleton instance
_ml_predictor = None


def get_ml_predictor() -> MLPredictor:
    """Get singleton MLPredictor instance."""
    global _ml_predictor
    if _ml_predictor is None:
        _ml_predictor = MLPredictor()
    return _ml_predictor
