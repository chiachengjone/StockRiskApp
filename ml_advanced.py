"""
Advanced Machine Learning Predictor Module
============================================
Enhanced ML capabilities with deep learning models,
hyperparameter optimization, explainability, and ensemble methods.

Features:
- LSTM/GRU neural networks for time series
- Transformer-based sequence models
- Optuna hyperparameter optimization
- SHAP feature importance explanations
- Ensemble methods (stacking, voting)
- Cross-validation with proper time series splits

Author: Stock Risk App | Feb 2026
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import logging
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import (
    GradientBoostingRegressor, 
    RandomForestRegressor,
    VotingRegressor,
    StackingRegressor
)
from sklearn.linear_model import Ridge, Lasso, ElasticNet
import joblib

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# Try to import optional deep learning dependencies
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        LSTM, GRU, Dense, Dropout, BatchNormalization,
        MultiHeadAttention, LayerNormalization, Flatten, Input
    )
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ModelConfig:
    """Configuration for ML model training."""
    model_type: str = 'xgboost'  # 'xgboost', 'lstm', 'gru', 'transformer', 'ensemble'
    lookback: int = 60  # Days of history to use
    forecast_horizon: int = 5  # Days ahead to predict
    n_features: int = 10
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    hidden_units: List[int] = field(default_factory=lambda: [64, 32])
    dropout: float = 0.2
    use_attention: bool = True
    optimize_hyperparams: bool = False
    n_trials: int = 50


@dataclass
class PredictionResult:
    """Result of a prediction."""
    ticker: str
    prediction: float
    confidence_interval: Tuple[float, float]
    direction: str  # 'up', 'down', 'neutral'
    probability: float
    features_used: List[str]
    feature_importance: Dict[str, float]
    model_type: str
    training_metrics: Dict[str, float]
    timestamp: datetime


@dataclass
class ModelEvaluation:
    """Model evaluation metrics."""
    mse: float
    rmse: float
    mae: float
    mape: float
    r2: float
    directional_accuracy: float
    sharpe_of_predictions: float
    information_ratio: float


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

class AdvancedFeatureEngineer:
    """
    Create advanced features for ML models.
    """
    
    @staticmethod
    def create_features(data: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive feature set.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        df = data.copy()
        
        # Price-based features
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Multi-period returns
        for period in [5, 10, 20, 60]:
            df[f'returns_{period}d'] = df['Close'].pct_change(period)
            df[f'volatility_{period}d'] = df['returns'].rolling(period).std()
        
        # Moving averages and crossovers
        for ma in [5, 10, 20, 50, 200]:
            df[f'sma_{ma}'] = df['Close'].rolling(ma).mean()
            df[f'ema_{ma}'] = df['Close'].ewm(span=ma).mean()
            df[f'price_vs_sma_{ma}'] = df['Close'] / df[f'sma_{ma}'] - 1
        
        # MA crossovers
        df['sma_5_10_cross'] = (df['sma_5'] > df['sma_10']).astype(int)
        df['sma_10_20_cross'] = (df['sma_10'] > df['sma_20']).astype(int)
        df['sma_50_200_cross'] = (df['sma_50'] > df['sma_200']).astype(int)
        
        # Momentum indicators
        df['rsi_14'] = AdvancedFeatureEngineer._rsi(df['Close'], 14)
        df['rsi_7'] = AdvancedFeatureEngineer._rsi(df['Close'], 7)
        
        # MACD
        ema12 = df['Close'].ewm(span=12).mean()
        ema26 = df['Close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        sma20 = df['Close'].rolling(20).mean()
        std20 = df['Close'].rolling(20).std()
        df['bb_upper'] = sma20 + 2 * std20
        df['bb_lower'] = sma20 - 2 * std20
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume features
        if 'Volume' in df.columns:
            df['volume_sma_20'] = df['Volume'].rolling(20).mean()
            df['volume_ratio'] = df['Volume'] / df['volume_sma_20']
            df['volume_trend'] = df['Volume'].pct_change(5)
            
            # OBV
            df['obv'] = (np.sign(df['returns']) * df['Volume']).cumsum()
            df['obv_ema'] = df['obv'].ewm(span=20).mean()
        
        # Price patterns
        df['higher_high'] = (
            (df['High'] > df['High'].shift(1)) & 
            (df['Low'] > df['Low'].shift(1))
        ).astype(int)
        
        df['lower_low'] = (
            (df['High'] < df['High'].shift(1)) & 
            (df['Low'] < df['Low'].shift(1))
        ).astype(int)
        
        # ATR
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift(1))
        low_close = abs(df['Low'] - df['Close'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(14).mean()
        df['atr_ratio'] = tr / df['atr_14']
        
        # Calendar features
        if isinstance(df.index, pd.DatetimeIndex):
            df['day_of_week'] = df.index.dayofweek
            df['day_of_month'] = df.index.day
            df['month'] = df.index.month
            df['quarter'] = df.index.quarter
            df['is_month_end'] = df.index.is_month_end.astype(int)
            df['is_month_start'] = df.index.is_month_start.astype(int)
        
        # Target variable (next day return)
        df['target'] = df['returns'].shift(-1)
        
        # Drop NaN
        df = df.dropna()
        
        return df
    
    @staticmethod
    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))


# ============================================================================
# LSTM/GRU MODEL
# ============================================================================

class LSTMPredictor:
    """
    LSTM/GRU model for time series prediction.
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.history = None
    
    def build_model(self, n_features: int, model_type: str = 'lstm') -> Any:
        """Build LSTM or GRU model."""
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow required for LSTM/GRU models")
        
        model = Sequential()
        
        # First recurrent layer
        if model_type == 'lstm':
            model.add(LSTM(
                self.config.hidden_units[0],
                return_sequences=len(self.config.hidden_units) > 1,
                input_shape=(self.config.lookback, n_features)
            ))
        else:
            model.add(GRU(
                self.config.hidden_units[0],
                return_sequences=len(self.config.hidden_units) > 1,
                input_shape=(self.config.lookback, n_features)
            ))
        
        model.add(BatchNormalization())
        model.add(Dropout(self.config.dropout))
        
        # Additional layers
        for i, units in enumerate(self.config.hidden_units[1:]):
            is_last = i == len(self.config.hidden_units) - 2
            
            if model_type == 'lstm':
                model.add(LSTM(units, return_sequences=not is_last))
            else:
                model.add(GRU(units, return_sequences=not is_last))
            
            model.add(BatchNormalization())
            model.add(Dropout(self.config.dropout))
        
        # Output layer
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1))
        
        model.compile(
            optimizer=Adam(learning_rate=self.config.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        return model
    
    def prepare_sequences(
        self,
        data: np.ndarray,
        target: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM input."""
        X, y = [], []
        
        for i in range(self.config.lookback, len(data)):
            X.append(data[i - self.config.lookback:i])
            if target is not None:
                y.append(target[i])
        
        return np.array(X), np.array(y) if target is not None else None
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None
    ) -> Dict[str, Any]:
        """Train the model."""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=10,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5
            )
        ]
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        self.history = self.model.fit(
            X_train, y_train,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=0
        )
        
        return {
            'final_loss': self.history.history['loss'][-1],
            'epochs_trained': len(self.history.history['loss'])
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        return self.model.predict(X, verbose=0).flatten()


# ============================================================================
# TRANSFORMER MODEL
# ============================================================================

class TransformerPredictor:
    """
    Transformer-based model for time series prediction.
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
    
    def build_model(self, n_features: int) -> Any:
        """Build transformer model."""
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow required for Transformer models")
        
        inputs = Input(shape=(self.config.lookback, n_features))
        
        # Positional encoding (simplified)
        x = Dense(64)(inputs)
        
        # Multi-head attention layers
        for _ in range(2):
            # Self-attention
            attention = MultiHeadAttention(
                num_heads=4,
                key_dim=16
            )(x, x)
            attention = Dropout(self.config.dropout)(attention)
            x = LayerNormalization()(x + attention)
            
            # Feed-forward
            ff = Dense(64, activation='relu')(x)
            ff = Dense(64)(ff)
            ff = Dropout(self.config.dropout)(ff)
            x = LayerNormalization()(x + ff)
        
        # Global average pooling
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        
        # Dense layers
        x = Dense(32, activation='relu')(x)
        x = Dropout(self.config.dropout)(x)
        outputs = Dense(1)(x)
        
        self.model = keras.Model(inputs, outputs)
        self.model.compile(
            optimizer=Adam(learning_rate=self.config.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return self.model
    
    def prepare_sequences(self, data: np.ndarray, target: np.ndarray = None):
        """Same as LSTM."""
        X, y = [], []
        for i in range(self.config.lookback, len(data)):
            X.append(data[i - self.config.lookback:i])
            if target is not None:
                y.append(target[i])
        return np.array(X), np.array(y) if target is not None else None
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train transformer model."""
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        self.model.fit(
            X_train, y_train,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_data=(X_val, y_val) if X_val is not None else None,
            callbacks=callbacks,
            verbose=0
        )
    
    def predict(self, X):
        return self.model.predict(X, verbose=0).flatten()


# ============================================================================
# HYPERPARAMETER OPTIMIZATION
# ============================================================================

class HyperparameterOptimizer:
    """
    Optuna-based hyperparameter optimization.
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.best_params = None
        self.study = None
    
    def optimize_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict[str, Any]:
        """Optimize XGBoost hyperparameters."""
        if not HAS_OPTUNA:
            logger.warning("Optuna not installed. Using default parameters.")
            return {}
        
        if not HAS_XGB:
            raise ImportError("XGBoost required for optimization")
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True)
            }
            
            model = xgb.XGBRegressor(**params, random_state=42)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            preds = model.predict(X_val)
            return mean_squared_error(y_val, preds)
        
        self.study = optuna.create_study(direction='minimize')
        self.study.optimize(objective, n_trials=self.config.n_trials, show_progress_bar=False)
        
        self.best_params = self.study.best_params
        return self.best_params
    
    def optimize_lstm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict[str, Any]:
        """Optimize LSTM hyperparameters."""
        if not HAS_OPTUNA or not HAS_TENSORFLOW:
            return {}
        
        def objective(trial):
            config = ModelConfig(
                hidden_units=[
                    trial.suggest_int('units_1', 32, 128),
                    trial.suggest_int('units_2', 16, 64)
                ],
                dropout=trial.suggest_float('dropout', 0.1, 0.5),
                learning_rate=trial.suggest_float('lr', 1e-4, 1e-2, log=True),
                batch_size=trial.suggest_categorical('batch_size', [16, 32, 64]),
                epochs=50,
                lookback=self.config.lookback
            )
            
            predictor = LSTMPredictor(config)
            predictor.build_model(X_train.shape[2], 'lstm')
            predictor.train(X_train, y_train, X_val, y_val)
            
            preds = predictor.predict(X_val)
            return mean_squared_error(y_val, preds)
        
        self.study = optuna.create_study(direction='minimize')
        self.study.optimize(objective, n_trials=min(20, self.config.n_trials))
        
        self.best_params = self.study.best_params
        return self.best_params


# ============================================================================
# FEATURE IMPORTANCE (SHAP)
# ============================================================================

class FeatureExplainer:
    """
    SHAP-based feature importance and explanation.
    """
    
    def __init__(self, model: Any, X_train: np.ndarray, feature_names: List[str]):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        self.shap_values = None
        self.explainer = None
    
    def compute_shap_values(
        self,
        X_explain: np.ndarray = None,
        max_samples: int = 100
    ) -> Dict[str, float]:
        """
        Compute SHAP values for feature importance.
        
        Returns:
            Dictionary of feature name to importance score
        """
        if not HAS_SHAP:
            logger.warning("SHAP not installed. Using model feature importance.")
            return self._fallback_importance()
        
        if X_explain is None:
            X_explain = self.X_train[:max_samples]
        
        try:
            # Create explainer based on model type
            if hasattr(self.model, 'feature_importances_'):
                # Tree-based model
                self.explainer = shap.TreeExplainer(self.model)
            else:
                # Use KernelExplainer for other models
                background = shap.kmeans(self.X_train, 10)
                self.explainer = shap.KernelExplainer(self.model.predict, background)
            
            self.shap_values = self.explainer.shap_values(X_explain)
            
            # Calculate mean absolute SHAP values
            if isinstance(self.shap_values, list):
                self.shap_values = self.shap_values[0]
            
            importance = np.abs(self.shap_values).mean(axis=0)
            
            return dict(zip(self.feature_names, importance))
            
        except Exception as e:
            logger.warning(f"SHAP computation failed: {e}")
            return self._fallback_importance()
    
    def _fallback_importance(self) -> Dict[str, float]:
        """Fallback to model's native feature importance."""
        if hasattr(self.model, 'feature_importances_'):
            return dict(zip(self.feature_names, self.model.feature_importances_))
        elif hasattr(self.model, 'coef_'):
            return dict(zip(self.feature_names, np.abs(self.model.coef_)))
        else:
            return {name: 1.0 / len(self.feature_names) for name in self.feature_names}
    
    def get_top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get top N most important features."""
        importance = self.compute_shap_values()
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        return sorted_features[:n]


# ============================================================================
# ENSEMBLE METHODS
# ============================================================================

class EnsemblePredictor:
    """
    Ensemble methods combining multiple models.
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.models = {}
        self.ensemble = None
        self.meta_model = None
    
    def build_voting_ensemble(self) -> VotingRegressor:
        """Build voting ensemble of diverse models."""
        estimators = []
        
        # XGBoost
        if HAS_XGB:
            estimators.append(('xgb', xgb.XGBRegressor(
                n_estimators=100, max_depth=5, random_state=42
            )))
        
        # Random Forest
        estimators.append(('rf', RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42
        )))
        
        # Gradient Boosting
        estimators.append(('gbm', GradientBoostingRegressor(
            n_estimators=100, max_depth=5, random_state=42
        )))
        
        # Ridge
        estimators.append(('ridge', Ridge(alpha=1.0)))
        
        self.ensemble = VotingRegressor(estimators)
        return self.ensemble
    
    def build_stacking_ensemble(self) -> StackingRegressor:
        """Build stacking ensemble with meta-learner."""
        estimators = []
        
        if HAS_XGB:
            estimators.append(('xgb', xgb.XGBRegressor(
                n_estimators=100, max_depth=5, random_state=42
            )))
        
        estimators.append(('rf', RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42
        )))
        
        estimators.append(('gbm', GradientBoostingRegressor(
            n_estimators=100, max_depth=5, random_state=42
        )))
        
        # Meta-learner
        self.ensemble = StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge(alpha=1.0),
            cv=TimeSeriesSplit(n_splits=3)
        )
        
        return self.ensemble
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train ensemble."""
        if self.ensemble is None:
            self.build_voting_ensemble()
        
        self.ensemble.fit(X_train, y_train)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate ensemble predictions."""
        return self.ensemble.predict(X)
    
    def get_model_weights(self) -> Dict[str, float]:
        """Get individual model contributions (for voting ensemble)."""
        if isinstance(self.ensemble, VotingRegressor):
            # All models have equal weight in default voting
            n_models = len(self.ensemble.estimators_)
            return {name: 1.0 / n_models for name, _ in self.ensemble.estimators}
        return {}


# ============================================================================
# MAIN ADVANCED ML PREDICTOR
# ============================================================================

class AdvancedMLPredictor:
    """
    Main class for advanced ML predictions.
    """
    
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.feature_engineer = AdvancedFeatureEngineer()
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = []
        self.explainer = None
    
    def prepare_data(
        self,
        data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare data with feature engineering.
        
        Returns:
            X, y, feature_names
        """
        # Engineer features
        df = self.feature_engineer.create_features(data)
        
        # Select features
        exclude = ['target', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
        feature_cols = [c for c in df.columns if c not in exclude]
        
        X = df[feature_cols].values
        y = df['target'].values
        
        self.feature_names = feature_cols
        
        return X, y, feature_cols
    
    def train(
        self,
        data: pd.DataFrame,
        val_split: float = 0.2
    ) -> Dict[str, Any]:
        """
        Train the model on historical data.
        
        Args:
            data: OHLCV DataFrame
            val_split: Validation split ratio
            
        Returns:
            Training metrics
        """
        X, y, features = self.prepare_data(data)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train/validation split (time-aware)
        split_idx = int(len(X) * (1 - val_split))
        X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Hyperparameter optimization
        if self.config.optimize_hyperparams and HAS_OPTUNA:
            optimizer = HyperparameterOptimizer(self.config)
            
            if self.config.model_type in ['lstm', 'gru']:
                # Prepare sequences for optimization
                predictor = LSTMPredictor(self.config)
                X_seq_train, y_seq_train = predictor.prepare_sequences(X_train, y_train)
                X_seq_val, y_seq_val = predictor.prepare_sequences(X_val, y_val)
                best_params = optimizer.optimize_lstm(
                    X_seq_train, y_seq_train, X_seq_val, y_seq_val
                )
            else:
                best_params = optimizer.optimize_xgboost(X_train, y_train, X_val, y_val)
        
        # Train model based on type
        if self.config.model_type == 'lstm':
            predictor = LSTMPredictor(self.config)
            predictor.build_model(len(features))
            X_seq_train, y_seq_train = predictor.prepare_sequences(X_train, y_train)
            X_seq_val, y_seq_val = predictor.prepare_sequences(X_val, y_val)
            result = predictor.train(X_seq_train, y_seq_train, X_seq_val, y_seq_val)
            self.model = predictor
            
        elif self.config.model_type == 'gru':
            predictor = LSTMPredictor(self.config)
            predictor.build_model(len(features), 'gru')
            X_seq_train, y_seq_train = predictor.prepare_sequences(X_train, y_train)
            X_seq_val, y_seq_val = predictor.prepare_sequences(X_val, y_val)
            result = predictor.train(X_seq_train, y_seq_train, X_seq_val, y_seq_val)
            self.model = predictor
            
        elif self.config.model_type == 'transformer':
            predictor = TransformerPredictor(self.config)
            predictor.build_model(len(features))
            X_seq_train, y_seq_train = predictor.prepare_sequences(X_train, y_train)
            X_seq_val, y_seq_val = predictor.prepare_sequences(X_val, y_val)
            predictor.train(X_seq_train, y_seq_train, X_seq_val, y_seq_val)
            self.model = predictor
            
        elif self.config.model_type == 'ensemble':
            ensemble = EnsemblePredictor(self.config)
            ensemble.build_stacking_ensemble()
            ensemble.train(X_train, y_train)
            self.model = ensemble
            
        else:  # Default to XGBoost
            if HAS_XGB:
                self.model = xgb.XGBRegressor(
                    n_estimators=200,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42
                )
            else:
                self.model = GradientBoostingRegressor(
                    n_estimators=200,
                    max_depth=5,
                    random_state=42
                )
            self.model.fit(X_train, y_train)
        
        # Evaluate
        if hasattr(self.model, 'predict'):
            preds = self.model.predict(X_val)
        else:
            X_seq_val, _ = self.model.prepare_sequences(X_val, y_val)
            preds = self.model.predict(X_seq_val)
            y_val = y_val[self.config.lookback:]
        
        metrics = self._calculate_metrics(y_val, preds)
        
        # Setup explainer
        if hasattr(self.model, 'predict') and not isinstance(self.model, (LSTMPredictor, TransformerPredictor, EnsemblePredictor)):
            self.explainer = FeatureExplainer(self.model, X_train, self.feature_names)
        
        return metrics
    
    def predict(self, data: pd.DataFrame) -> PredictionResult:
        """
        Generate prediction for latest data.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            PredictionResult object
        """
        X, _, features = self.prepare_data(data)
        X_scaled = self.scaler.transform(X)
        
        # Get latest window
        if isinstance(self.model, (LSTMPredictor, TransformerPredictor)):
            X_seq, _ = self.model.prepare_sequences(X_scaled[-self.config.lookback-1:])
            pred = self.model.predict(X_seq)[0]
        else:
            pred = self.model.predict(X_scaled[-1:].reshape(1, -1))[0]
        
        # Get feature importance
        feature_importance = {}
        if self.explainer:
            feature_importance = self.explainer.compute_shap_values()
        elif hasattr(self.model, 'feature_importances_'):
            feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
        
        # Direction and probability
        direction = 'up' if pred > 0 else 'down' if pred < 0 else 'neutral'
        probability = min(0.99, 0.5 + abs(pred) * 10)  # Simple probability estimate
        
        # Confidence interval (simplified)
        ci_width = abs(pred) * 0.5
        ci = (pred - ci_width, pred + ci_width)
        
        return PredictionResult(
            ticker=data.name if hasattr(data, 'name') else 'UNKNOWN',
            prediction=pred * 100,  # As percentage
            confidence_interval=(ci[0] * 100, ci[1] * 100),
            direction=direction,
            probability=probability,
            features_used=self.feature_names,
            feature_importance=feature_importance,
            model_type=self.config.model_type,
            training_metrics={},
            timestamp=datetime.now()
        )
    
    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        
        # MAPE (avoiding division by zero)
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else 0
        
        r2 = r2_score(y_true, y_pred)
        
        # Directional accuracy
        dir_acc = np.mean(np.sign(y_true) == np.sign(y_pred)) * 100
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2,
            'directional_accuracy': dir_acc
        }


# ============================================================================
# STREAMLIT RENDERING
# ============================================================================

def render_advanced_ml_dashboard():
    """Render advanced ML dashboard."""
    import streamlit as st
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    st.subheader(" Advanced Machine Learning Predictor")
    
    # Model configuration
    st.write("### Model Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        model_type = st.selectbox(
            "Model Type",
            ["XGBoost", "LSTM", "GRU", "Transformer", "Ensemble"]
        ).lower()
    
    with col2:
        lookback = st.slider("Lookback Period", 20, 120, 60)
    
    with col3:
        optimize = st.checkbox("Optimize Hyperparameters")
    
    # Feature availability
    st.write("### Available Features")
    
    features_info = {
        'TensorFlow/Keras': HAS_TENSORFLOW,
        'Optuna (Optimization)': HAS_OPTUNA,
        'SHAP (Explainability)': HAS_SHAP,
        'XGBoost': HAS_XGB
    }
    
    cols = st.columns(4)
    for i, (name, available) in enumerate(features_info.items()):
        with cols[i]:
            status = "" if available else ""
            st.write(f"{status} {name}")
    
    # Model comparison
    st.write("### Model Comparison")
    
    models_df = pd.DataFrame({
        'Model': ['XGBoost', 'LSTM', 'GRU', 'Transformer', 'Ensemble'],
        'Type': ['Tree-based', 'Recurrent', 'Recurrent', 'Attention', 'Meta'],
        'Best For': [
            'Tabular features',
            'Sequence patterns',
            'Faster training',
            'Long sequences',
            'Robustness'
        ],
        'Complexity': ['Medium', 'High', 'High', 'Very High', 'High']
    })
    
    st.dataframe(models_df, hide_index=True)
    
    # Demo prediction
    st.write("### Demo Prediction")
    
    if st.button("Run Demo with Sample Data"):
        with st.spinner("Training model..."):
            # Generate sample data
            np.random.seed(42)
            dates = pd.date_range('2023-01-01', periods=500, freq='B')
            
            close = 100 * (1 + np.random.randn(500).cumsum() * 0.01)
            high = close * (1 + abs(np.random.randn(500)) * 0.01)
            low = close * (1 - abs(np.random.randn(500)) * 0.01)
            open_ = close + np.random.randn(500) * 0.5
            volume = np.random.randint(1000000, 5000000, 500)
            
            data = pd.DataFrame({
                'Open': open_,
                'High': high,
                'Low': low,
                'Close': close,
                'Volume': volume
            }, index=dates)
            
            # Train model
            config = ModelConfig(
                model_type='xgboost',  # Use XGBoost for demo (fastest)
                lookback=lookback,
                optimize_hyperparams=False
            )
            
            predictor = AdvancedMLPredictor(config)
            metrics = predictor.train(data)
        
        st.success("Model trained successfully!")
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("RMSE", f"{metrics['rmse']:.4f}")
        with col2:
            st.metric("MAE", f"{metrics['mae']:.4f}")
        with col3:
            st.metric("R²", f"{metrics['r2']:.4f}")
        with col4:
            st.metric("Direction Acc", f"{metrics['directional_accuracy']:.1f}%")
        
        # Feature importance
        if predictor.explainer:
            st.write("### Feature Importance (SHAP)")
            importance = predictor.explainer.get_top_features(10)
            
            fig = go.Figure(go.Bar(
                x=[imp for _, imp in importance],
                y=[name for name, _ in importance],
                orientation='h',
                marker_color='#00D4AA'
            ))
            
            fig.update_layout(
                title="Top 10 Features",
                xaxis_title="Importance",
                yaxis_title="Feature",
                template='plotly_dark',
                height=400
            )
            
            st.plotly_chart(fig, width="stretch")
    
    # Information boxes
    st.write("### Model Information")
    
    with st.expander("LSTM/GRU Models"):
        st.markdown("""
        **Long Short-Term Memory (LSTM)** and **Gated Recurrent Unit (GRU)** are 
        recurrent neural networks designed for sequential data:
        
        - Handle long-term dependencies in time series
        - Learn complex temporal patterns
        - Require more data and training time
        - Best for capturing momentum and regime changes
        
        **Requirements:** TensorFlow/Keras
        """)
    
    with st.expander("Transformer Models"):
        st.markdown("""
        **Transformer** models use self-attention mechanisms:
        
        - Capture long-range dependencies effectively
        - Parallel processing for faster training
        - State-of-the-art for many sequence tasks
        - Can model multiple factors simultaneously
        
        **Requirements:** TensorFlow/Keras
        """)
    
    with st.expander("Ensemble Methods"):
        st.markdown("""
        **Ensemble** methods combine multiple models:
        
        - **Voting:** Average predictions from diverse models
        - **Stacking:** Use a meta-learner on base model outputs
        - More robust than single models
        - Reduces overfitting
        
        **Included:** XGBoost, Random Forest, Gradient Boosting, Ridge
        """)


# Make features available
HAS_ADVANCED_ML = True
