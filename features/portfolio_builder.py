"""
Portfolio Builder Module
========================
Risk Budget Allocation | Factor-Based Portfolio | What-If Presets

Advanced portfolio construction tools with constraint-based optimization.

Author: Stock Risk App | Feb 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize, LinearConstraint
from scipy.stats import norm
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

class FactorType(Enum):
    """Factor types for portfolio construction."""
    VALUE = "Value"
    MOMENTUM = "Momentum"
    QUALITY = "Quality"
    SIZE = "Size"
    LOW_VOLATILITY = "Low Volatility"
    DIVIDEND_YIELD = "Dividend Yield"


@dataclass
class RiskBudgetResult:
    """Result of risk budget optimization."""
    weights: Dict[str, float]
    risk_contributions: Dict[str, float]
    pct_risk_contributions: Dict[str, float]
    portfolio_var: float
    portfolio_volatility: float
    target_var: float
    constraint_satisfied: bool
    utilization: float  # % of risk budget used
    slack: float  # Remaining risk budget
    iterations: int
    success: bool


@dataclass
class FactorPortfolioResult:
    """Result of factor-based portfolio construction."""
    weights: Dict[str, float]
    factor_exposures: Dict[str, float]
    target_exposures: Dict[str, float]
    tracking_error: float
    expected_return: float
    volatility: float
    factor_contribution: Dict[str, float]  # % return from each factor
    alpha_estimate: float
    success: bool


@dataclass
class WhatIfPreset:
    """A what-if analysis preset configuration."""
    name: str
    description: str
    target_volatility: Optional[float]
    target_return: Optional[float]
    method: str  # 'equal_weight', 'risk_parity', 'max_sharpe', 'min_vol', 'target_vol'
    constraints: Dict[str, Any]
    icon: str


# =============================================================================
# WHAT-IF PRESETS
# =============================================================================

WHAT_IF_PRESETS = {
    'conservative': WhatIfPreset(
        name='Conservative',
        description='Low-risk allocation targeting 8% volatility',
        target_volatility=0.08,
        target_return=None,
        method='target_vol',
        constraints={'max_weight': 0.25, 'min_weight': 0.02},
        icon='[Shield]'
    ),
    'balanced': WhatIfPreset(
        name='Balanced',
        description='Moderate risk-return profile (12% vol target)',
        target_volatility=0.12,
        target_return=None,
        method='target_vol',
        constraints={'max_weight': 0.30, 'min_weight': 0.02},
        icon='[Balance]'
    ),
    'growth': WhatIfPreset(
        name='Growth',
        description='Higher risk for higher returns (18% vol target)',
        target_volatility=0.18,
        target_return=None,
        method='target_vol',
        constraints={'max_weight': 0.40, 'min_weight': 0.02},
        icon='[Trending Up]'
    ),
    'aggressive': WhatIfPreset(
        name='Aggressive',
        description='Maximum return focus with 25% volatility',
        target_volatility=0.25,
        target_return=None,
        method='target_vol',
        constraints={'max_weight': 0.50, 'min_weight': 0.01},
        icon='[Rocket]'
    ),
    'max_sharpe': WhatIfPreset(
        name='Max Sharpe Ratio',
        description='Optimize for best risk-adjusted return',
        target_volatility=None,
        target_return=None,
        method='max_sharpe',
        constraints={'max_weight': 0.40, 'min_weight': 0.02},
        icon='[Target]'
    ),
    'min_volatility': WhatIfPreset(
        name='Minimum Volatility',
        description='Lowest possible portfolio volatility',
        target_volatility=None,
        target_return=None,
        method='min_vol',
        constraints={'max_weight': 0.30, 'min_weight': 0.02},
        icon='[Shield Check]'
    ),
    'equal_weight': WhatIfPreset(
        name='Equal Weight',
        description='Simple 1/N allocation across all assets',
        target_volatility=None,
        target_return=None,
        method='equal_weight',
        constraints={},
        icon='[Equals]'
    ),
    'risk_parity': WhatIfPreset(
        name='Risk Parity',
        description='Equal risk contribution from each asset',
        target_volatility=None,
        target_return=None,
        method='risk_parity',
        constraints={'max_weight': 0.40, 'min_weight': 0.02},
        icon='[Pie Chart]'
    ),
    'income_focus': WhatIfPreset(
        name='Income Focus',
        description='Emphasize dividend-paying assets',
        target_volatility=0.10,
        target_return=None,
        method='income',
        constraints={'max_weight': 0.25, 'min_weight': 0.02, 'dividend_tilt': 0.5},
        icon='[Dollar Sign]'
    ),
    'momentum_tilt': WhatIfPreset(
        name='Momentum Tilt',
        description='Overweight recent winners',
        target_volatility=0.15,
        target_return=None,
        method='momentum',
        constraints={'max_weight': 0.35, 'min_weight': 0.02},
        icon='[Activity]'
    )
}


# =============================================================================
# FACTOR SCORES (Simplified estimation from returns)
# =============================================================================

def estimate_factor_scores(
    returns: pd.DataFrame,
    prices: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Estimate factor scores for each asset from return data.
    
    Factors estimated:
    - Value: Inverse of recent return (contrarian)
    - Momentum: 12-month return momentum
    - Quality: Consistency of returns (lower vol = higher quality)
    - Size: Use volatility as proxy (higher vol = smaller)
    - Low Volatility: Inverse of volatility
    - Dividend Yield: Estimated from stability
    
    Args:
        returns: DataFrame of asset returns
        prices: Optional DataFrame of prices
    
    Returns:
        DataFrame with factor scores (z-scores) for each asset
    """
    n_assets = len(returns.columns)
    
    # Calculate raw factor metrics
    metrics = pd.DataFrame(index=returns.columns)
    
    # Momentum: 12-month return (or available period)
    lookback = min(252, len(returns))
    if lookback > 20:
        metrics['momentum_raw'] = (1 + returns.tail(lookback)).prod() - 1
    else:
        metrics['momentum_raw'] = returns.mean() * 252
    
    # Value: Inverse of momentum (contrarian)
    metrics['value_raw'] = -metrics['momentum_raw']
    
    # Volatility
    metrics['volatility'] = returns.std() * np.sqrt(252)
    
    # Quality: Stability of returns (Sharpe approximation)
    metrics['quality_raw'] = returns.mean() / returns.std()
    
    # Size: Higher vol = smaller cap proxy
    metrics['size_raw'] = -metrics['volatility']
    
    # Low Volatility: Inverse of volatility
    metrics['low_vol_raw'] = -metrics['volatility']
    
    # Dividend Yield: Stability proxy (low vol + positive skew)
    skewness = returns.skew()
    metrics['dividend_raw'] = -metrics['volatility'] + (skewness.clip(-2, 2) * 0.1)
    
    # Convert to z-scores
    factor_scores = pd.DataFrame(index=returns.columns)
    
    for factor, raw_col in [
        ('Value', 'value_raw'),
        ('Momentum', 'momentum_raw'),
        ('Quality', 'quality_raw'),
        ('Size', 'size_raw'),
        ('Low Volatility', 'low_vol_raw'),
        ('Dividend Yield', 'dividend_raw')
    ]:
        raw = metrics[raw_col]
        if raw.std() > 0:
            factor_scores[factor] = (raw - raw.mean()) / raw.std()
        else:
            factor_scores[factor] = 0
    
    return factor_scores


# =============================================================================
# RISK BUDGET OPTIMIZER
# =============================================================================

class RiskBudgetOptimizer:
    """
    Optimizer for risk budget constrained portfolios.
    
    Finds optimal weights such that portfolio VaR stays within budget.
    """
    
    def __init__(
        self,
        returns: pd.DataFrame,
        risk_free_rate: float = 0.05,
        conf_level: float = 0.95
    ):
        self.returns = returns
        self.assets = list(returns.columns)
        self.n_assets = len(self.assets)
        self.rf_rate = risk_free_rate
        self.conf_level = conf_level
        
        # Pre-compute statistics
        self.mean_returns = returns.mean().values * 252
        self.cov_matrix = returns.cov().values * 252
        self.z_score = norm.ppf(conf_level)
    
    def _calculate_portfolio_var(self, weights: np.ndarray) -> float:
        """Calculate portfolio VaR (annualized)."""
        port_return = np.dot(weights, self.mean_returns)
        port_vol = np.sqrt(weights.T @ self.cov_matrix @ weights)
        # Parametric VaR (as positive number for loss)
        var = -(port_return / 252 - self.z_score * port_vol / np.sqrt(252))
        return var * 252  # Annualized
    
    def _calculate_risk_contribution(self, weights: np.ndarray) -> np.ndarray:
        """Calculate risk contribution of each asset."""
        port_vol = np.sqrt(weights.T @ self.cov_matrix @ weights)
        if port_vol > 0:
            mcr = (self.cov_matrix @ weights) / port_vol
            return weights * mcr
        return np.zeros(self.n_assets)
    
    def optimize(
        self,
        target_var: float,
        max_weight: float = 0.40,
        min_weight: float = 0.02,
        method: str = 'max_return'
    ) -> RiskBudgetResult:
        """
        Optimize portfolio subject to risk budget constraint.
        
        Args:
            target_var: Maximum acceptable VaR (annualized, e.g., 0.15 for 15%)
            max_weight: Maximum weight per asset
            min_weight: Minimum weight per asset
            method: Optimization method ('max_return', 'max_sharpe', 'min_turnover')
        
        Returns:
            RiskBudgetResult with optimal weights
        """
        # Objective function
        if method == 'max_return':
            def objective(w):
                return -np.dot(w, self.mean_returns)
        elif method == 'max_sharpe':
            def objective(w):
                ret = np.dot(w, self.mean_returns)
                vol = np.sqrt(w.T @ self.cov_matrix @ w)
                return -(ret - self.rf_rate) / vol if vol > 0 else 0
        else:  # min_turnover - equal weight reference
            equal_w = np.ones(self.n_assets) / self.n_assets
            def objective(w):
                return np.sum((w - equal_w) ** 2)
        
        # VaR constraint
        def var_constraint(w):
            port_vol = np.sqrt(w.T @ self.cov_matrix @ w)
            port_ret = np.dot(w, self.mean_returns)
            var = self.z_score * port_vol / np.sqrt(252) - port_ret / 252
            return target_var - var * 252  # Must be >= 0
        
        # Sum to 1 constraint
        def sum_constraint(w):
            return np.sum(w) - 1.0
        
        constraints = [
            {'type': 'eq', 'fun': sum_constraint},
            {'type': 'ineq', 'fun': var_constraint}
        ]
        
        bounds = [(min_weight, max_weight) for _ in range(self.n_assets)]
        
        # Initial guess: equal weight
        x0 = np.ones(self.n_assets) / self.n_assets
        
        # Optimize
        result = minimize(
            objective, x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500}
        )
        
        if result.success:
            weights = result.x / result.x.sum()  # Normalize
        else:
            # Fallback to equal weight
            weights = np.ones(self.n_assets) / self.n_assets
        
        # Calculate results
        portfolio_var = self._calculate_portfolio_var(weights)
        portfolio_vol = np.sqrt(weights.T @ self.cov_matrix @ weights)
        risk_contrib = self._calculate_risk_contribution(weights)
        total_risk = np.sum(risk_contrib)
        pct_contrib = risk_contrib / total_risk if total_risk > 0 else risk_contrib
        
        return RiskBudgetResult(
            weights=dict(zip(self.assets, weights)),
            risk_contributions=dict(zip(self.assets, risk_contrib)),
            pct_risk_contributions=dict(zip(self.assets, pct_contrib)),
            portfolio_var=float(portfolio_var),
            portfolio_volatility=float(portfolio_vol),
            target_var=target_var,
            constraint_satisfied=portfolio_var <= target_var * 1.01,  # 1% tolerance
            utilization=min(100, portfolio_var / target_var * 100) if target_var > 0 else 100,
            slack=max(0, target_var - portfolio_var),
            iterations=result.nit if hasattr(result, 'nit') else 0,
            success=result.success
        )


# =============================================================================
# FACTOR PORTFOLIO BUILDER
# =============================================================================

class FactorPortfolioBuilder:
    """
    Build portfolios with target factor exposures.
    
    Allows tilting portfolio toward specific factors while
    maintaining risk constraints.
    """
    
    def __init__(
        self,
        returns: pd.DataFrame,
        factor_scores: pd.DataFrame = None,
        risk_free_rate: float = 0.05
    ):
        self.returns = returns
        self.assets = list(returns.columns)
        self.n_assets = len(self.assets)
        self.rf_rate = risk_free_rate
        
        # Get or estimate factor scores
        if factor_scores is None:
            self.factor_scores = estimate_factor_scores(returns)
        else:
            self.factor_scores = factor_scores
        
        self.factors = list(self.factor_scores.columns)
        
        # Pre-compute statistics
        self.mean_returns = returns.mean().values * 252
        self.cov_matrix = returns.cov().values * 252
    
    def build(
        self,
        target_exposures: Dict[str, float],
        max_weight: float = 0.30,
        min_weight: float = 0.02,
        max_tracking_error: float = 0.10
    ) -> FactorPortfolioResult:
        """
        Build portfolio with target factor exposures.
        
        Args:
            target_exposures: Dict of factor name -> target z-score exposure
            max_weight: Maximum weight per asset
            min_weight: Minimum weight per asset
            max_tracking_error: Maximum tracking error vs equal weight
        
        Returns:
            FactorPortfolioResult with optimized weights
        """
        # Objective: minimize deviation from target exposures
        def objective(w):
            total_deviation = 0
            for factor, target in target_exposures.items():
                if factor in self.factors:
                    exposure = np.dot(w, self.factor_scores[factor].values)
                    total_deviation += (exposure - target) ** 2
            return total_deviation
        
        # Constraints
        def sum_constraint(w):
            return np.sum(w) - 1.0
        
        constraints = [
            {'type': 'eq', 'fun': sum_constraint}
        ]
        
        bounds = [(min_weight, max_weight) for _ in range(self.n_assets)]
        
        # Initial guess: equal weight
        x0 = np.ones(self.n_assets) / self.n_assets
        
        # Optimize
        result = minimize(
            objective, x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500}
        )
        
        if result.success:
            weights = result.x / result.x.sum()
        else:
            weights = np.ones(self.n_assets) / self.n_assets
        
        # Calculate achieved exposures
        achieved_exposures = {}
        for factor in self.factors:
            achieved_exposures[factor] = float(np.dot(weights, self.factor_scores[factor].values))
        
        # Calculate portfolio stats
        expected_return = np.dot(weights, self.mean_returns)
        volatility = np.sqrt(weights.T @ self.cov_matrix @ weights)
        
        # Tracking error vs equal weight
        equal_w = np.ones(self.n_assets) / self.n_assets
        diff = weights - equal_w
        tracking_var = diff.T @ self.cov_matrix @ diff
        tracking_error = np.sqrt(tracking_var)
        
        # Estimate factor contribution to return
        factor_contribution = {}
        for factor in self.factors:
            exposure = achieved_exposures[factor]
            # Simplified: assume each unit of factor exposure contributes 2% return
            factor_contribution[factor] = exposure * 0.02
        
        total_factor_return = sum(factor_contribution.values())
        alpha_estimate = expected_return - total_factor_return - self.rf_rate
        
        return FactorPortfolioResult(
            weights=dict(zip(self.assets, weights)),
            factor_exposures=achieved_exposures,
            target_exposures=target_exposures,
            tracking_error=float(tracking_error),
            expected_return=float(expected_return),
            volatility=float(volatility),
            factor_contribution=factor_contribution,
            alpha_estimate=float(alpha_estimate),
            success=result.success
        )


# =============================================================================
# PRESET OPTIMIZER
# =============================================================================

class PresetOptimizer:
    """
    Apply what-if presets to portfolio.
    
    Provides quick one-click optimization to preset configurations.
    """
    
    def __init__(
        self,
        returns: pd.DataFrame,
        risk_free_rate: float = 0.05
    ):
        self.returns = returns
        self.assets = list(returns.columns)
        self.n_assets = len(self.assets)
        self.rf_rate = risk_free_rate
        
        # Pre-compute statistics
        self.mean_returns = returns.mean().values * 252
        self.cov_matrix = returns.cov().values * 252
    
    def apply_preset(
        self,
        preset_name: str,
        current_weights: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """
        Apply a preset configuration to the portfolio.
        
        Args:
            preset_name: Name of preset from WHAT_IF_PRESETS
            current_weights: Current portfolio weights (optional)
        
        Returns:
            Dictionary with new weights and metrics
        """
        if preset_name not in WHAT_IF_PRESETS:
            raise ValueError(f"Unknown preset: {preset_name}")
        
        preset = WHAT_IF_PRESETS[preset_name]
        method = preset.method
        constraints = preset.constraints
        
        max_weight = constraints.get('max_weight', 0.40)
        min_weight = constraints.get('min_weight', 0.02)
        
        # Apply optimization based on method
        if method == 'equal_weight':
            weights = np.ones(self.n_assets) / self.n_assets
        
        elif method == 'risk_parity':
            weights = self._risk_parity_optimize(max_weight, min_weight)
        
        elif method == 'max_sharpe':
            weights = self._max_sharpe_optimize(max_weight, min_weight)
        
        elif method == 'min_vol':
            weights = self._min_vol_optimize(max_weight, min_weight)
        
        elif method == 'target_vol':
            target_vol = preset.target_volatility or 0.12
            weights = self._target_vol_optimize(target_vol, max_weight, min_weight)
        
        elif method == 'momentum':
            weights = self._momentum_optimize(max_weight, min_weight)
        
        elif method == 'income':
            weights = self._income_optimize(max_weight, min_weight)
        
        else:
            weights = np.ones(self.n_assets) / self.n_assets
        
        # Calculate metrics
        expected_return = np.dot(weights, self.mean_returns)
        volatility = np.sqrt(weights.T @ self.cov_matrix @ weights)
        sharpe = (expected_return - self.rf_rate) / volatility if volatility > 0 else 0
        
        # Calculate turnover if current weights provided
        turnover = 0
        if current_weights:
            for i, asset in enumerate(self.assets):
                current = current_weights.get(asset, 0)
                turnover += abs(weights[i] - current)
        
        return {
            'weights': dict(zip(self.assets, weights)),
            'expected_return': float(expected_return),
            'volatility': float(volatility),
            'sharpe_ratio': float(sharpe),
            'turnover': float(turnover),
            'preset': preset,
            'method': method
        }
    
    def _risk_parity_optimize(
        self,
        max_weight: float,
        min_weight: float
    ) -> np.ndarray:
        """Risk parity optimization."""
        def objective(w):
            port_vol = np.sqrt(w.T @ self.cov_matrix @ w)
            if port_vol <= 0:
                return 0
            mcr = (self.cov_matrix @ w) / port_vol
            rc = w * mcr
            target_rc = port_vol / self.n_assets
            return np.sum((rc - target_rc) ** 2)
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        bounds = [(min_weight, max_weight) for _ in range(self.n_assets)]
        x0 = np.ones(self.n_assets) / self.n_assets
        
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                         constraints=constraints, options={'maxiter': 500})
        
        return result.x / result.x.sum() if result.success else x0
    
    def _max_sharpe_optimize(
        self,
        max_weight: float,
        min_weight: float
    ) -> np.ndarray:
        """Maximize Sharpe ratio."""
        def objective(w):
            ret = np.dot(w, self.mean_returns)
            vol = np.sqrt(w.T @ self.cov_matrix @ w)
            return -(ret - self.rf_rate) / vol if vol > 0 else 0
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        bounds = [(min_weight, max_weight) for _ in range(self.n_assets)]
        x0 = np.ones(self.n_assets) / self.n_assets
        
        result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                         constraints=constraints, options={'maxiter': 500})
        
        return result.x / result.x.sum() if result.success else x0
    
    def _min_vol_optimize(
        self,
        max_weight: float,
        min_weight: float
    ) -> np.ndarray:
        """Minimize volatility."""
        def objective(w):
            return np.sqrt(w.T @ self.cov_matrix @ w)
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        bounds = [(min_weight, max_weight) for _ in range(self.n_assets)]
        x0 = np.ones(self.n_assets) / self.n_assets
        
        result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                         constraints=constraints, options={'maxiter': 500})
        
        return result.x / result.x.sum() if result.success else x0
    
    def _target_vol_optimize(
        self,
        target_vol: float,
        max_weight: float,
        min_weight: float
    ) -> np.ndarray:
        """Optimize for target volatility."""
        # First get min vol portfolio
        min_vol_weights = self._min_vol_optimize(max_weight, min_weight)
        min_vol = np.sqrt(min_vol_weights.T @ self.cov_matrix @ min_vol_weights)
        
        # Then get max sharpe
        max_sharpe_weights = self._max_sharpe_optimize(max_weight, min_weight)
        max_sharpe_vol = np.sqrt(max_sharpe_weights.T @ self.cov_matrix @ max_sharpe_weights)
        
        # Interpolate between min vol and max sharpe
        if target_vol <= min_vol:
            return min_vol_weights
        elif target_vol >= max_sharpe_vol:
            return max_sharpe_weights
        else:
            # Linear interpolation
            t = (target_vol - min_vol) / (max_sharpe_vol - min_vol)
            weights = (1 - t) * min_vol_weights + t * max_sharpe_weights
            return weights / weights.sum()
    
    def _momentum_optimize(
        self,
        max_weight: float,
        min_weight: float
    ) -> np.ndarray:
        """Momentum-tilted optimization."""
        # Calculate momentum scores
        momentum = (1 + self.returns.tail(252)).prod() - 1
        momentum_z = (momentum - momentum.mean()) / momentum.std()
        
        # Objective: maximize expected return with momentum tilt
        def objective(w):
            ret = np.dot(w, self.mean_returns)
            mom_tilt = np.dot(w, momentum_z.values)
            return -(ret + 0.02 * mom_tilt)  # Add momentum bonus
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        bounds = [(min_weight, max_weight) for _ in range(self.n_assets)]
        x0 = np.ones(self.n_assets) / self.n_assets
        
        result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                         constraints=constraints, options={'maxiter': 500})
        
        return result.x / result.x.sum() if result.success else x0
    
    def _income_optimize(
        self,
        max_weight: float,
        min_weight: float
    ) -> np.ndarray:
        """Income-focused optimization (low vol proxy)."""
        # Prefer low volatility, stable assets
        vols = self.returns.std() * np.sqrt(252)
        stability = 1 / (vols + 0.01)  # Inverse volatility
        stability_z = (stability - stability.mean()) / stability.std()
        
        # Objective: maximize stability while maintaining return
        def objective(w):
            ret = np.dot(w, self.mean_returns)
            stab_tilt = np.dot(w, stability_z.values)
            vol = np.sqrt(w.T @ self.cov_matrix @ w)
            return -stab_tilt - 0.5 * ret + 0.5 * vol  # Balance stability and return
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        bounds = [(min_weight, max_weight) for _ in range(self.n_assets)]
        x0 = np.ones(self.n_assets) / self.n_assets
        
        result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                         constraints=constraints, options={'maxiter': 500})
        
        return result.x / result.x.sum() if result.success else x0


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_risk_budget_chart(
    result: RiskBudgetResult,
    title: str = "Risk Budget Utilization"
) -> go.Figure:
    """Create risk budget visualization."""
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "pie"}, {"type": "indicator"}]],
        column_widths=[0.6, 0.4]
    )
    
    # Risk contribution pie chart
    fig.add_trace(
        go.Pie(
            labels=list(result.pct_risk_contributions.keys()),
            values=[v * 100 for v in result.pct_risk_contributions.values()],
            hole=0.4,
            textinfo='label+percent',
            textposition='outside',
            marker=dict(colors=px.colors.qualitative.Set2)
        ),
        row=1, col=1
    )
    
    # Utilization gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=result.utilization,
            title={'text': "Budget Utilization"},
            delta={'reference': 100, 'relative': True},
            gauge={
                'axis': {'range': [0, 120]},
                'bar': {'color': "#2196F3"},
                'steps': [
                    {'range': [0, 80], 'color': "#E8F5E9"},
                    {'range': [80, 100], 'color': "#FFF9C4"},
                    {'range': [100, 120], 'color': "#FFEBEE"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 100
                }
            }
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        height=400,
        showlegend=False
    )
    
    return fig


def create_factor_exposure_radar(
    exposures: Dict[str, float],
    targets: Dict[str, float] = None,
    title: str = "Factor Exposures"
) -> go.Figure:
    """Create radar chart for factor exposures."""
    factors = list(exposures.keys())
    values = list(exposures.values())
    
    # Close the polygon
    factors_closed = factors + [factors[0]]
    values_closed = values + [values[0]]
    
    fig = go.Figure()
    
    # Actual exposures
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=factors_closed,
        fill='toself',
        name='Actual',
        fillcolor='rgba(33, 150, 243, 0.3)',
        line=dict(color='#2196F3', width=2)
    ))
    
    # Target exposures if provided
    if targets:
        target_values = [targets.get(f, 0) for f in factors]
        target_closed = target_values + [target_values[0]]
        
        fig.add_trace(go.Scatterpolar(
            r=target_closed,
            theta=factors_closed,
            fill='toself',
            name='Target',
            fillcolor='rgba(255, 152, 0, 0.2)',
            line=dict(color='#FF9800', width=2, dash='dash')
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[-2, 2])
        ),
        title=dict(text=title, x=0.5),
        showlegend=True,
        height=450
    )
    
    return fig


def create_preset_comparison_chart(
    returns: pd.DataFrame,
    current_weights: Dict[str, float],
    presets: List[str] = None
) -> go.Figure:
    """Create comparison chart for different presets."""
    optimizer = PresetOptimizer(returns)
    
    if presets is None:
        presets = ['conservative', 'balanced', 'growth', 'max_sharpe', 'min_volatility']
    
    results = []
    for preset_name in presets:
        try:
            result = optimizer.apply_preset(preset_name, current_weights)
            results.append({
                'Preset': WHAT_IF_PRESETS[preset_name].name,
                'Return': result['expected_return'] * 100,
                'Volatility': result['volatility'] * 100,
                'Sharpe': result['sharpe_ratio'],
                'Turnover': result['turnover'] * 100
            })
        except:
            pass
    
    if not results:
        return go.Figure()
    
    df = pd.DataFrame(results)
    
    # Risk-return scatter
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['Volatility'],
        y=df['Return'],
        mode='markers+text',
        text=df['Preset'],
        textposition='top center',
        marker=dict(
            size=df['Sharpe'] * 20 + 10,  # Size by Sharpe
            color=df['Turnover'],
            colorscale='Viridis',
            colorbar=dict(title='Turnover %'),
            showscale=True
        ),
        hovertemplate=(
            '<b>%{text}</b><br>' +
            'Return: %{y:.1f}%<br>' +
            'Volatility: %{x:.1f}%<br>' +
            'Sharpe: %{marker.size:.2f}<extra></extra>'
        )
    ))
    
    fig.update_layout(
        title=dict(text="Preset Portfolio Comparison", x=0.5),
        xaxis_title="Volatility (%)",
        yaxis_title="Expected Return (%)",
        height=450,
        showlegend=False
    )
    
    return fig


# =============================================================================
# STREAMLIT RENDER FUNCTIONS
# =============================================================================

def render_risk_budget_tab(
    returns: pd.DataFrame,
    current_weights: Dict[str, float],
    portfolio_value: float = 100000
):
    """Render the Risk Budget Allocation tab."""
    st.markdown("### Risk Budget Optimization")
    st.markdown("*Optimize portfolio weights subject to VaR constraint*")
    
    # Risk budget input
    col1, col2, col3 = st.columns(3)
    
    with col1:
        target_var_pct = st.slider(
            "Maximum VaR (%)",
            min_value=1.0,
            max_value=30.0,
            value=10.0,
            step=0.5,
            help="Maximum acceptable Value at Risk (annualized)"
        )
        target_var = target_var_pct / 100
    
    with col2:
        opt_method = st.selectbox(
            "Optimization Goal",
            options=['max_return', 'max_sharpe', 'min_turnover'],
            format_func=lambda x: {
                'max_return': 'Maximize Return',
                'max_sharpe': 'Maximize Sharpe',
                'min_turnover': 'Minimize Turnover'
            }[x]
        )
    
    with col3:
        max_weight = st.slider(
            "Max Weight per Asset",
            min_value=0.1,
            max_value=0.6,
            value=0.4,
            step=0.05
        )
    
    # Optimize
    if st.button("Optimize Risk Budget", type="primary", use_container_width=True):
        with st.spinner("Optimizing..."):
            optimizer = RiskBudgetOptimizer(returns)
            result = optimizer.optimize(
                target_var=target_var,
                max_weight=max_weight,
                method=opt_method
            )
        
        # Display results
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Portfolio VaR", f"{result.portfolio_var:.1%}")
        col2.metric("Target VaR", f"{result.target_var:.1%}")
        col3.metric("Budget Utilization", f"{result.utilization:.0f}%")
        col4.metric("Volatility", f"{result.portfolio_volatility:.1%}")
        
        # Status
        if result.constraint_satisfied:
            st.success(f"VaR constraint satisfied with {result.slack:.2%} slack")
        else:
            st.warning("VaR constraint could not be fully satisfied")
        
        # Charts
        fig = create_risk_budget_chart(result)
        st.plotly_chart(fig, use_container_width=True)
        
        # Weights table
        st.markdown("#### Optimized Weights")
        weights_df = pd.DataFrame({
            'Asset': list(result.weights.keys()),
            'Weight': [f"{v:.1%}" for v in result.weights.values()],
            'Risk Contribution': [f"{v:.4f}" for v in result.risk_contributions.values()],
            '% of Total Risk': [f"{v:.1%}" for v in result.pct_risk_contributions.values()]
        })
        st.dataframe(weights_df, use_container_width=True, hide_index=True)


def render_factor_builder_tab(
    returns: pd.DataFrame,
    current_weights: Dict[str, float]
):
    """Render the Factor Portfolio Builder tab."""
    st.markdown("### Factor-Based Portfolio Builder")
    st.markdown("*Construct portfolios with target factor exposures*")
    
    # Factor score preview
    factor_scores = estimate_factor_scores(returns)
    
    with st.expander("View Factor Scores by Asset"):
        st.dataframe(
            factor_scores.style.format("{:.2f}").background_gradient(
                cmap='RdYlGn', axis=0, vmin=-2, vmax=2
            ),
            use_container_width=True
        )
    
    st.markdown("#### Target Factor Exposures")
    st.caption("Set desired exposure to each factor (-2 to +2 z-score)")
    
    # Factor sliders
    target_exposures = {}
    cols = st.columns(3)
    
    factors = ['Value', 'Momentum', 'Quality', 'Size', 'Low Volatility', 'Dividend Yield']
    for i, factor in enumerate(factors):
        with cols[i % 3]:
            target_exposures[factor] = st.slider(
                factor,
                min_value=-2.0,
                max_value=2.0,
                value=0.0,
                step=0.1,
                key=f"factor_{factor}"
            )
    
    # Constraints
    col1, col2 = st.columns(2)
    with col1:
        max_weight = st.slider("Max Weight", 0.1, 0.5, 0.3, 0.05, key="fb_max_weight")
    with col2:
        min_weight = st.slider("Min Weight", 0.0, 0.1, 0.02, 0.01, key="fb_min_weight")
    
    # Build portfolio
    if st.button("Build Factor Portfolio", type="primary", use_container_width=True):
        with st.spinner("Building portfolio..."):
            builder = FactorPortfolioBuilder(returns, factor_scores)
            result = builder.build(
                target_exposures=target_exposures,
                max_weight=max_weight,
                min_weight=min_weight
            )
        
        # Results
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Expected Return", f"{result.expected_return:.1%}")
        col2.metric("Volatility", f"{result.volatility:.1%}")
        col3.metric("Tracking Error", f"{result.tracking_error:.1%}")
        col4.metric("Alpha Estimate", f"{result.alpha_estimate:.1%}")
        
        # Radar chart
        fig = create_factor_exposure_radar(
            result.factor_exposures,
            target_exposures,
            "Factor Exposure: Actual vs Target"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Weights and exposures
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Portfolio Weights")
            weights_df = pd.DataFrame({
                'Asset': list(result.weights.keys()),
                'Weight': [f"{v:.1%}" for v in result.weights.values()]
            })
            st.dataframe(weights_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("#### Factor Exposure Comparison")
            exp_df = pd.DataFrame({
                'Factor': list(result.factor_exposures.keys()),
                'Target': [f"{target_exposures.get(f, 0):.2f}" for f in result.factor_exposures.keys()],
                'Achieved': [f"{v:.2f}" for v in result.factor_exposures.values()],
                'Return Contrib': [f"{result.factor_contribution.get(f, 0):.2%}" for f in result.factor_exposures.keys()]
            })
            st.dataframe(exp_df, use_container_width=True, hide_index=True)


def render_presets_tab(
    returns: pd.DataFrame,
    current_weights: Dict[str, float]
):
    """Render the What-If Presets tab."""
    st.markdown("### Quick Portfolio Presets")
    st.markdown("*One-click optimization to preset configurations*")
    
    # Preset grid
    presets_list = list(WHAT_IF_PRESETS.keys())
    n_cols = 4
    
    for row_start in range(0, len(presets_list), n_cols):
        cols = st.columns(n_cols)
        for i, col in enumerate(cols):
            idx = row_start + i
            if idx < len(presets_list):
                preset_key = presets_list[idx]
                preset = WHAT_IF_PRESETS[preset_key]
                
                with col:
                    if st.button(
                        f"{preset.icon} {preset.name}",
                        key=f"preset_{preset_key}",
                        use_container_width=True
                    ):
                        st.session_state['selected_preset'] = preset_key
    
    st.markdown("---")
    
    # Show selected preset details
    if 'selected_preset' in st.session_state:
        preset_key = st.session_state['selected_preset']
        preset = WHAT_IF_PRESETS[preset_key]
        
        st.markdown(f"#### {preset.icon} {preset.name}")
        st.caption(preset.description)
        
        with st.spinner("Calculating..."):
            optimizer = PresetOptimizer(returns)
            result = optimizer.apply_preset(preset_key, current_weights)
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Expected Return", f"{result['expected_return']:.1%}")
        col2.metric("Volatility", f"{result['volatility']:.1%}")
        col3.metric("Sharpe Ratio", f"{result['sharpe_ratio']:.2f}")
        col4.metric("Turnover", f"{result['turnover']:.1%}")
        
        # Comparison chart
        st.markdown("#### Weight Comparison")
        weights_df = pd.DataFrame({
            'Asset': list(result['weights'].keys()),
            'Current': [current_weights.get(a, 0) * 100 for a in result['weights'].keys()],
            'Proposed': [v * 100 for v in result['weights'].values()]
        })
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Current', x=weights_df['Asset'], y=weights_df['Current']))
        fig.add_trace(go.Bar(name='Proposed', x=weights_df['Asset'], y=weights_df['Proposed']))
        fig.update_layout(
            barmode='group',
            title="Current vs Proposed Weights",
            yaxis_title="Weight (%)",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Comparison chart of all presets
    st.markdown("---")
    st.markdown("#### All Presets Comparison")
    
    fig = create_preset_comparison_chart(returns, current_weights)
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Data classes
    'RiskBudgetResult',
    'FactorPortfolioResult',
    'WhatIfPreset',
    'FactorType',
    
    # Constants
    'WHAT_IF_PRESETS',
    
    # Classes
    'RiskBudgetOptimizer',
    'FactorPortfolioBuilder',
    'PresetOptimizer',
    
    # Functions
    'estimate_factor_scores',
    'create_risk_budget_chart',
    'create_factor_exposure_radar',
    'create_preset_comparison_chart',
    
    # Render functions
    'render_risk_budget_tab',
    'render_factor_builder_tab',
    'render_presets_tab',
]
