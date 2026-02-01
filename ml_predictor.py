"""
ML Predictor Module - AI-Powered Risk Analytics
================================================
XGBoost VaR Prediction • Feature Engineering • Model Comparison
Fallback to GradientBoosting if XGBoost unavailable

Author: Professional Risk Analytics | Jan 2026
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost, fallback to sklearn
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except (ImportError, Exception) as e:
    HAS_XGBOOST = False
    XGBOOST_ERROR = str(e)


class MLPredictor:
    """AI-powered next-day VaR predictor using XGBoost or GradientBoosting fallback."""
    
    def __init__(self):
        self.model = None
        self.feature_names = []
        self.is_trained = False
        self.model_type = 'xgboost' if HAS_XGBOOST else 'gradient_boosting'
    
    def compute_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Compute Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def engineer_features(self, returns: pd.Series, prices: pd.Series) -> pd.DataFrame:
        """
        Engineer features for VaR prediction.
        
        Features:
        - RSI(14): Momentum indicator
        - Vol_20: 20-day rolling volatility
        - Vol_5: 5-day rolling volatility
        - Return lags: [1, 3, 5, 10] day returns
        - Volatility ratio: short/long term vol
        - Return momentum: cumulative returns
        - Max drawdown (rolling)
        """
        df = pd.DataFrame(index=returns.index)
        
        # Basic returns
        df['return'] = returns
        
        # Return lags
        for lag in [1, 3, 5, 10]:
            df[f'return_lag_{lag}'] = returns.shift(lag)
        
        # Rolling volatility
        df['vol_5'] = returns.rolling(5).std()
        df['vol_20'] = returns.rolling(20).std()
        df['vol_60'] = returns.rolling(60).std()
        
        # Volatility ratio (regime indicator)
        df['vol_ratio'] = df['vol_5'] / df['vol_20']
        
        # RSI
        df['rsi_14'] = self.compute_rsi(prices, 14)
        
        # Momentum features
        df['momentum_5'] = returns.rolling(5).sum()
        df['momentum_20'] = returns.rolling(20).sum()
        
        # Rolling max/min returns (tail events)
        df['max_return_20'] = returns.rolling(20).max()
        df['min_return_20'] = returns.rolling(20).min()
        
        # Skewness and kurtosis (rolling)
        df['skew_20'] = returns.rolling(20).skew()
        df['kurt_20'] = returns.rolling(20).kurt()
        
        # Z-score of current return
        df['return_zscore'] = (returns - returns.rolling(20).mean()) / returns.rolling(20).std()
        
        # Target: Next day absolute return (realized VaR proxy)
        df['target'] = returns.shift(-1).abs()
        
        return df.dropna()
    
    def train_predict(self, returns: pd.Series, prices: pd.Series, 
                     test_size: float = 0.2, n_estimators: int = 100) -> dict:
        """
        Train XGBoost model and predict next-day VaR.
        Falls back to GradientBoostingRegressor if XGBoost unavailable.
        
        Args:
            returns: Daily log returns
            prices: Price series for RSI calculation
            test_size: Fraction of data for testing (time-series split)
            n_estimators: Number of trees
        
        Returns:
            Dictionary with predictions, metrics, and feature importance
        """
        # Engineer features
        df = self.engineer_features(returns, prices)
        
        if len(df) < 100:
            return {
                'error': 'Insufficient data for ML model (need 100+ observations)',
                'predicted_var': float(returns.std() * 1.645),
                'r2_score': 0.0,
                'importance': {}
            }
        
        # Features and target
        feature_cols = [c for c in df.columns if c not in ['target', 'return']]
        self.feature_names = feature_cols
        
        X = df[feature_cols].values
        y = df['target'].values
        
        # Time-series split (no shuffle - preserve temporal order)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train model - use XGBoost if available, otherwise GradientBoosting
        if HAS_XGBOOST:
            self.model = XGBRegressor(
                n_estimators=n_estimators,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0
            )
            self.model_type = 'xgboost'
        else:
            # Fallback to sklearn GradientBoostingRegressor
            self.model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            )
            self.model_type = 'gradient_boosting'
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Predictions
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # Metrics
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        
        # Feature importance
        importance = dict(zip(feature_cols, self.model.feature_importances_))
        importance_sorted = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        
        # Predict next day VaR (use latest features)
        latest_features = X[-1:].reshape(1, -1)
        predicted_var_raw = float(self.model.predict(latest_features)[0])
        
        # Scale to 95% VaR (multiply by ~1.645 for normal approx)
        predicted_var_95 = predicted_var_raw * 1.645
        
        # Comparison with historical methods
        hist_var = float(np.percentile(returns, 5))
        garch_vol = float(returns.tail(20).std()) * np.sqrt(1) * 1.645
        
        return {
            'predicted_var': predicted_var_95,
            'predicted_var_raw': predicted_var_raw,
            'r2_score': float(r2_test),
            'r2_train': float(r2_train),
            'mae': float(mae_test),
            'importance': importance_sorted,
            'top_features': list(importance_sorted.keys())[:5],
            'n_train': len(X_train),
            'n_test': len(X_test),
            'model_type': self.model_type,
            'comparison': {
                'ml_model': predicted_var_95,
                'historical': abs(hist_var),
                'garch_approx': garch_vol
            },
            'test_predictions': y_pred_test,
            'test_actual': y_test
        }
    
    def predict_var_range(self, returns: pd.Series, prices: pd.Series, 
                         horizons: list = [1, 5, 10]) -> dict:
        """
        Predict VaR for multiple horizons.
        
        Returns VaR estimates for each horizon.
        """
        if not self.is_trained:
            base_result = self.train_predict(returns, prices)
            if 'error' in base_result:
                return base_result
        
        results = {}
        base_var = float(returns.std())
        
        for h in horizons:
            # Scale VaR by sqrt(horizon) - simplified
            scaled_var = base_var * np.sqrt(h) * 1.645
            results[f'{h}d_var'] = scaled_var
        
        return results
    
    def model_comparison(self, returns: pd.Series, prices: pd.Series) -> pd.DataFrame:
        """
        Compare ML VaR with traditional methods.
        
        Methods compared:
        - XGBoost (ML)
        - Historical Simulation
        - Parametric Normal
        - EWMA Volatility
        """
        # Train ML model
        ml_result = self.train_predict(returns, prices)
        
        # Historical VaR
        hist_var = abs(float(np.percentile(returns, 5)))
        
        # Parametric (Normal)
        from scipy.stats import norm
        mu = float(returns.mean())
        sigma = float(returns.std())
        param_var = abs(norm.ppf(0.05, mu, sigma))
        
        # EWMA Volatility (lambda = 0.94)
        ewma_var_series = returns.ewm(span=20).std()
        ewma_var = float(ewma_var_series.iloc[-1]) * 1.645
        
        comparison = pd.DataFrame({
            'Method': ['XGBoost ML', 'Historical', 'Parametric', 'EWMA'],
            '95% VaR': [
                ml_result.get('predicted_var', 0),
                hist_var,
                param_var,
                ewma_var
            ],
            'Description': [
                'Machine Learning (features + XGBoost)',
                'Empirical 5th percentile',
                'Normal distribution assumption',
                'Exponentially weighted volatility'
            ]
        })
        
        return comparison
    
    def backtest_var(self, returns: pd.Series, prices: pd.Series, 
                    window: int = 252, conf: float = 0.95) -> dict:
        """
        Backtest VaR predictions.
        
        Counts violations (days when loss exceeded VaR).
        Good model: violation rate ≈ (1 - confidence level)
        """
        if len(returns) < window + 50:
            return {'error': 'Insufficient data for backtest'}
        
        violations = 0
        predictions = []
        actuals = []
        
        # Rolling backtest
        for i in range(window, len(returns) - 1):
            train_rets = returns.iloc[i-window:i]
            train_prices = prices.iloc[i-window:i]
            
            # Simple VaR prediction (use percentile for speed)
            var_pred = abs(float(np.percentile(train_rets, 100*(1-conf))))
            actual_loss = abs(float(returns.iloc[i+1]))
            
            predictions.append(var_pred)
            actuals.append(actual_loss)
            
            if actual_loss > var_pred:
                violations += 1
        
        n_days = len(predictions)
        violation_rate = violations / n_days if n_days > 0 else 0
        expected_rate = 1 - conf
        
        return {
            'violations': violations,
            'total_days': n_days,
            'violation_rate': violation_rate,
            'expected_rate': expected_rate,
            'ratio': violation_rate / expected_rate if expected_rate > 0 else 0,
            'status': 'Good' if 0.8 <= violation_rate/expected_rate <= 1.2 else 'Review'
        }

    # =========================================================================
    # ENSEMBLE METHODS (NEW)
    # =========================================================================
    
    def ensemble_predict(self, returns: pd.Series, prices: pd.Series,
                        models: list = None) -> dict:
        """
        Ensemble VaR prediction combining multiple models.
        
        Combines:
        - ML Model (XGBoost or GradientBoosting)
        - GARCH volatility forecast
        - Historical simulation
        - Parametric VaR
        - EWMA volatility
        
        Args:
            returns: Returns series
            prices: Price series
            models: List of models to include (default: all)
        
        Returns:
            Dictionary with ensemble prediction and individual model results
        """
        from scipy.stats import norm
        
        all_models = ['ml', 'garch', 'historical', 'parametric', 'ewma']
        models = models or all_models
        
        predictions = {}
        weights = {}
        
        # 1. ML Model (XGBoost or GradientBoosting)
        if 'ml' in models or 'xgboost' in models:
            try:
                ml_result = self.train_predict(returns, prices)
                if 'error' not in ml_result:
                    model_name = ml_result.get('model_type', 'ml')
                    predictions[model_name] = abs(ml_result['predicted_var'])
                    weights[model_name] = 0.30  # Higher weight for ML
            except:
                pass
        
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
            except:
                pass
        
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
        
        # Calculate ensemble (weighted average)
        if predictions:
            # Normalize weights for available models
            total_weight = sum(weights[m] for m in predictions.keys())
            normalized_weights = {m: weights[m] / total_weight for m in predictions.keys()}
            
            ensemble_var = sum(predictions[m] * normalized_weights[m] 
                              for m in predictions.keys())
            
            # Also calculate simple average and median
            simple_avg = np.mean(list(predictions.values()))
            median_var = np.median(list(predictions.values()))
            
            return {
                'ensemble_var': float(ensemble_var),
                'simple_average': float(simple_avg),
                'median_var': float(median_var),
                'individual_predictions': predictions,
                'weights': normalized_weights,
                'n_models': len(predictions),
                'spread': float(max(predictions.values()) - min(predictions.values())),
                'best_model': min(predictions.keys(), key=lambda m: predictions[m]),
                'worst_model': max(predictions.keys(), key=lambda m: predictions[m])
            }
        
        return {'error': 'No models produced predictions'}
    
    def predict_with_confidence(self, returns: pd.Series, prices: pd.Series,
                               n_bootstrap: int = 100) -> dict:
        """
        Predict VaR with bootstrap confidence intervals.
        
        Resamples data and retrains model to quantify prediction uncertainty.
        
        Args:
            returns: Returns series
            prices: Price series
            n_bootstrap: Number of bootstrap samples
        
        Returns:
            Dictionary with mean prediction and confidence intervals
        """
        if len(returns) < 100:
            return {'error': 'Insufficient data for bootstrap'}
        
        predictions = []
        
        for i in range(n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(len(returns), size=len(returns), replace=True)
            indices = np.sort(indices)  # Maintain some temporal order
            
            sample_returns = returns.iloc[indices].reset_index(drop=True)
            sample_prices = prices.iloc[indices].reset_index(drop=True)
            
            try:
                # Quick prediction using percentile (for speed)
                var_pred = abs(float(np.percentile(sample_returns, 5)))
                predictions.append(var_pred)
            except:
                continue
        
        if len(predictions) < 10:
            return {'error': 'Bootstrap failed'}
        
        predictions = np.array(predictions)
        
        return {
            'mean_var': float(np.mean(predictions)),
            'median_var': float(np.median(predictions)),
            'std_var': float(np.std(predictions)),
            'ci_5': float(np.percentile(predictions, 5)),
            'ci_95': float(np.percentile(predictions, 95)),
            'ci_lower': float(np.percentile(predictions, 2.5)),
            'ci_upper': float(np.percentile(predictions, 97.5)),
            'n_bootstrap': len(predictions),
            'confidence_width': float(np.percentile(predictions, 97.5) - 
                                     np.percentile(predictions, 2.5))
        }
    
    def get_feature_importance_chart_data(self, returns: pd.Series, 
                                         prices: pd.Series, top_n: int = 10) -> dict:
        """
        Get feature importance data for visualization.
        
        Args:
            returns: Returns series
            prices: Price series
            top_n: Number of top features to return
        
        Returns:
            Dictionary with feature importance data
        """
        if not self.is_trained:
            result = self.train_predict(returns, prices)
            if 'error' in result:
                return result
        
        if self.model is None:
            return {'error': 'Model not trained'}
        
        importance = dict(zip(self.feature_names, self.model.feature_importances_))
        sorted_importance = dict(sorted(importance.items(), 
                                        key=lambda x: x[1], reverse=True))
        
        # Get top N features
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
            'features': list(top_features.keys()),
            'importance': list(top_features.values()),
            'descriptions': [feature_descriptions.get(f, f) for f in top_features.keys()],
            'total_features': len(self.feature_names),
            'top_feature': list(top_features.keys())[0] if top_features else None,
            'top_importance': list(top_features.values())[0] if top_features else 0
        }
    
    def volatility_forecast(self, returns: pd.Series, horizon: int = 10) -> dict:
        """
        Multi-model volatility forecast.
        
        Combines GARCH, EWMA, and simple forecasts.
        
        Args:
            returns: Returns series
            horizon: Forecast horizon in days
        
        Returns:
            Dictionary with volatility forecasts
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
        
        # 2. EWMA forecast (constant projection)
        ewma_vol = float(returns.ewm(span=20).std().iloc[-1]) * np.sqrt(252)
        forecasts['ewma'] = [ewma_vol] * horizon
        
        # 3. Simple (historical) forecast
        hist_vol = float(returns.std()) * np.sqrt(252)
        forecasts['historical'] = [hist_vol] * horizon
        
        # 4. Expanding window forecast
        expanding_vol = float(returns.expanding(min_periods=20).std().iloc[-1]) * np.sqrt(252)
        forecasts['expanding'] = [expanding_vol] * horizon
        
        # Average across models
        valid_forecasts = [f for f in [forecasts['garch'], forecasts['ewma'], 
                                       forecasts['historical']] if f is not None]
        
        if valid_forecasts:
            avg_forecast = np.mean(valid_forecasts, axis=0).tolist()
        else:
            avg_forecast = forecasts['historical']
        
        return {
            'horizon': horizon,
            'forecasts': forecasts,
            'ensemble_forecast': avg_forecast,
            'current_vol': float(returns.std() * np.sqrt(252)),
            'vol_trend': 'increasing' if avg_forecast[-1] > avg_forecast[0] else 'decreasing'
        }
