"""
ML Predictor Module - AI-Powered Risk Analytics
================================================
XGBoost VaR Prediction • Feature Engineering • Model Comparison

Author: Professional Risk Analytics | Jan 2026
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


class MLPredictor:
    """AI-powered next-day VaR predictor using XGBoost."""
    
    def __init__(self):
        self.model = None
        self.feature_names = []
        self.is_trained = False
    
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
        
        Args:
            returns: Daily log returns
            prices: Price series for RSI calculation
            test_size: Fraction of data for testing (time-series split)
            n_estimators: Number of trees in XGBoost
        
        Returns:
            Dictionary with predictions, metrics, and feature importance
        """
        try:
            from xgboost import XGBRegressor
        except ImportError:
            return {
                'error': 'XGBoost not installed. Run: pip install xgboost',
                'predicted_var': 0.0,
                'r2_score': 0.0,
                'importance': {}
            }
        
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
        
        # Train XGBoost
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        )
        
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
            'comparison': {
                'xgboost': predicted_var_95,
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
