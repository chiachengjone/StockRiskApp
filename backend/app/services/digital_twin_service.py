"""
Digital Twin & What-If Analysis Service
========================================
Portfolio scenario simulation and what-if analysis.

Provides Digital Twin comparison and interactive portfolio analysis.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ScenarioConfig:
    """Configuration for a portfolio scenario."""
    name: str
    rebalance_frequency: str = 'monthly'  # none, monthly, quarterly, annually
    enable_tax_loss_harvesting: bool = False
    transaction_cost_bps: float = 10
    rebalance_threshold: float = 0.05


class DigitalTwinEngine:
    """
    Portfolio Digital Twin for scenario analysis.
    
    Creates simulated versions of the portfolio under different
    management strategies and compares outcomes.
    """
    
    def __init__(
        self,
        returns: pd.DataFrame,
        current_weights: Dict[str, float],
        initial_capital: float = 100000,
        horizon_days: int = 252
    ):
        self.returns = returns
        self.current_weights = current_weights
        self.initial_capital = initial_capital
        self.horizon_days = horizon_days
        self.assets = list(returns.columns)
        
        # Statistics
        self.mean_returns = returns.mean().values
        self.cov_matrix = returns.cov().values
    
    def run_scenario(
        self,
        config: ScenarioConfig,
        n_simulations: int = 500
    ) -> Dict[str, Any]:
        """
        Run a single scenario simulation.
        
        Returns simulation statistics.
        """
        weights = np.array([self.current_weights.get(a, 0) for a in self.assets])
        
        # Determine rebalance frequency
        rebal_freq = {
            'none': float('inf'),
            'monthly': 21,
            'quarterly': 63,
            'annually': 252
        }.get(config.rebalance_frequency, float('inf'))
        
        final_values = []
        transaction_costs_total = []
        max_drawdowns = []
        
        for sim in range(n_simulations):
            np.random.seed(sim)
            
            # Simulate returns path
            sim_returns = np.random.multivariate_normal(
                self.mean_returns,
                self.cov_matrix,
                size=self.horizon_days
            )
            
            portfolio_value = self.initial_capital
            current_weights = weights.copy()
            tx_costs = 0
            peak_value = portfolio_value
            max_dd = 0
            
            for day in range(self.horizon_days):
                # Apply daily returns
                daily_ret = np.dot(current_weights, sim_returns[day])
                portfolio_value *= (1 + daily_ret)
                
                # Update peak and drawdown
                peak_value = max(peak_value, portfolio_value)
                dd = (portfolio_value - peak_value) / peak_value
                max_dd = min(max_dd, dd)
                
                # Drift weights
                weight_factors = 1 + sim_returns[day]
                current_weights = current_weights * weight_factors
                current_weights = current_weights / current_weights.sum()
                
                # Check for rebalance
                if (day + 1) % rebal_freq == 0:
                    drift = np.abs(current_weights - weights).sum()
                    
                    if drift > config.rebalance_threshold:
                        # Rebalance cost
                        turnover = np.abs(current_weights - weights).sum() / 2
                        cost = turnover * portfolio_value * (config.transaction_cost_bps / 10000)
                        tx_costs += cost
                        current_weights = weights.copy()
            
            final_values.append(portfolio_value)
            transaction_costs_total.append(tx_costs)
            max_drawdowns.append(max_dd)
        
        final_values = np.array(final_values)
        tx_costs = np.array(transaction_costs_total)
        max_drawdowns = np.array(max_drawdowns)
        
        # Calculate statistics
        total_returns = (final_values / self.initial_capital) - 1
        mean_return = float(np.mean(total_returns))
        std_return = float(np.std(total_returns))
        sharpe = mean_return / std_return if std_return > 0 else 0
        
        var_95 = float(np.percentile(total_returns, 5))
        cvar_95 = float(np.mean(total_returns[total_returns <= var_95]))
        
        return {
            'name': config.name,
            'mean_return': mean_return,
            'std_return': std_return,
            'sharpe_ratio': sharpe,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'max_drawdown': float(np.mean(max_drawdowns)),
            'transaction_costs': float(np.mean(tx_costs)),
            'tax_benefits': 0.0,  # Placeholder
            'final_value_median': float(np.percentile(final_values, 50)),
            'final_value_5th': float(np.percentile(final_values, 5)),
            'final_value_95th': float(np.percentile(final_values, 95)),
            'rebalance_frequency': config.rebalance_frequency
        }
    
    def compare_scenarios(self) -> Dict[str, Dict]:
        """Compare standard scenarios."""
        scenarios = {
            'buy_and_hold': ScenarioConfig(
                name='Buy & Hold',
                rebalance_frequency='none',
                transaction_cost_bps=0
            ),
            'monthly_rebalance': ScenarioConfig(
                name='Monthly Rebalance',
                rebalance_frequency='monthly',
                transaction_cost_bps=10
            ),
            'quarterly_rebalance': ScenarioConfig(
                name='Quarterly Rebalance',
                rebalance_frequency='quarterly',
                transaction_cost_bps=10
            ),
            'tax_optimized': ScenarioConfig(
                name='Tax-Optimized',
                rebalance_frequency='quarterly',
                enable_tax_loss_harvesting=True,
                transaction_cost_bps=10
            )
        }
        
        results = {}
        for key, config in scenarios.items():
            results[key] = self.run_scenario(config)
        
        return results
    
    def get_best_scenario(self, results: Dict[str, Dict]) -> str:
        """Determine the best scenario based on Sharpe ratio."""
        best = max(results.items(), key=lambda x: x[1]['sharpe_ratio'])
        return best[0]
    
    def calculate_health_score(self) -> Dict[str, float]:
        """Calculate portfolio health score."""
        weights = np.array([self.current_weights.get(a, 0) for a in self.assets])
        
        # Diversification score (inverse of concentration)
        hhi = np.sum(weights ** 2)
        diversification = (1 - hhi) * 100
        
        # Risk efficiency (Sharpe approximation)
        port_return = np.dot(weights, self.mean_returns) * 252
        port_vol = np.sqrt(weights @ self.cov_matrix @ weights) * np.sqrt(252)
        risk_efficiency = min(100, max(0, (port_return / port_vol + 0.5) * 40))
        
        # Correlation score
        corr = pd.DataFrame(self.cov_matrix).corr()
        avg_corr = corr.values[np.triu_indices_from(corr.values, 1)].mean()
        correlation_score = (1 - avg_corr) * 100
        
        # Overall score
        overall = (diversification * 0.4 + risk_efficiency * 0.4 + correlation_score * 0.2)
        
        recommendations = []
        if diversification < 60:
            recommendations.append("Consider diversifying across more assets")
        if risk_efficiency < 40:
            recommendations.append("Risk-adjusted returns could be improved")
        if correlation_score < 50:
            recommendations.append("Assets are highly correlated - consider uncorrelated assets")
        
        return {
            'overall_score': float(overall),
            'diversification_score': float(diversification),
            'risk_efficiency_score': float(risk_efficiency),
            'momentum_score': 50.0,  # Placeholder
            'correlation_score': float(correlation_score),
            'recommendations': recommendations
        }


class WhatIfAnalyzer:
    """
    Engine for what-if portfolio analysis.
    
    Provides real-time calculation of portfolio metrics as weights are adjusted.
    """
    
    def __init__(
        self,
        returns: pd.DataFrame,
        current_weights: Dict[str, float],
        risk_free_rate: float = 0.05,
        transaction_cost_bps: float = 10
    ):
        self.returns = returns
        self.current_weights = current_weights
        self.risk_free_rate = risk_free_rate
        self.transaction_cost_bps = transaction_cost_bps
        
        self.assets = list(returns.columns)
        self.n_assets = len(self.assets)
        
        # Pre-calculate statistics
        self.mean_returns = returns.mean().values * 252
        self.cov_matrix = returns.cov().values * 252
    
    def _weights_to_array(self, weights: Dict[str, float]) -> np.ndarray:
        """Convert weight dict to array."""
        return np.array([weights.get(a, 0) for a in self.assets])
    
    def calculate_stats(self, weights: Dict[str, float]) -> Dict:
        """Calculate portfolio statistics for given weights."""
        w = self._weights_to_array(weights)
        
        # Basic stats
        expected_return = float(np.dot(w, self.mean_returns))
        volatility = float(np.sqrt(w.T @ self.cov_matrix @ w))
        sharpe = (expected_return - self.risk_free_rate) / volatility if volatility > 0 else 0
        
        # VaR (parametric)
        var_95 = -(expected_return - 1.645 * volatility)
        
        # Max drawdown approximation
        max_dd = volatility * 2.5
        
        return {
            'expected_return': expected_return,
            'volatility': volatility,
            'sharpe_ratio': float(sharpe),
            'var_95': float(var_95),
            'max_drawdown': float(max_dd)
        }
    
    def analyze_scenario(
        self,
        new_weights: Dict[str, float],
        scenario_name: str = "What-If"
    ) -> Dict:
        """Analyze a what-if scenario."""
        stats = self.calculate_stats(new_weights)
        
        # Calculate turnover
        turnover = sum(
            abs(new_weights.get(a, 0) - self.current_weights.get(a, 0))
            for a in self.assets
        )
        
        return {
            'name': scenario_name,
            'weights': new_weights,
            **stats,
            'turnover': float(turnover)
        }
    
    def compare_scenarios(
        self,
        new_weights: Dict[str, float]
    ) -> Dict:
        """Compare current vs proposed weights."""
        current = self.analyze_scenario(self.current_weights, "Current")
        proposed = self.analyze_scenario(new_weights, "Proposed")
        
        delta = {
            'expected_return': proposed['expected_return'] - current['expected_return'],
            'volatility': proposed['volatility'] - current['volatility'],
            'sharpe_ratio': proposed['sharpe_ratio'] - current['sharpe_ratio'],
            'var_95': proposed['var_95'] - current['var_95']
        }
        
        improvement = {
            'expected_return': delta['expected_return'] > 0,
            'volatility': delta['volatility'] < 0,
            'sharpe_ratio': delta['sharpe_ratio'] > 0,
            'var_95': delta['var_95'] < 0  # Lower VaR is better
        }
        
        return {
            'current': current,
            'proposed': proposed,
            'delta': delta,
            'improvement': improvement
        }
    
    def calculate_rebalance_trades(
        self,
        new_weights: Dict[str, float],
        portfolio_value: float = 100000
    ) -> List[Dict]:
        """Calculate trades needed to rebalance."""
        trades = []
        
        for asset in self.assets:
            current = self.current_weights.get(asset, 0)
            new = new_weights.get(asset, 0)
            diff = new - current
            
            if abs(diff) > 0.001:
                trade_value = diff * portfolio_value
                cost = abs(trade_value) * (self.transaction_cost_bps / 10000)
                
                trades.append({
                    'asset': asset,
                    'current_weight': float(current),
                    'new_weight': float(new),
                    'change': float(diff),
                    'trade_value': float(trade_value),
                    'action': 'BUY' if diff > 0 else 'SELL',
                    'transaction_cost': float(cost)
                })
        
        return sorted(trades, key=lambda x: abs(x['trade_value']), reverse=True)
    
    def optimize_for_target(
        self,
        target_return: float = None,
        target_volatility: float = None,
        min_weight: float = 0.0,
        max_weight: float = 1.0
    ) -> Dict:
        """Optimize portfolio for a target metric."""
        x0 = self._weights_to_array(self.current_weights)
        bounds = [(min_weight, max_weight) for _ in range(self.n_assets)]
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        if target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda w: np.dot(w, self.mean_returns) - target_return
            })
            
            def objective(w):
                return np.sqrt(w.T @ self.cov_matrix @ w)
        
        elif target_volatility is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda w: np.sqrt(w.T @ self.cov_matrix @ w) - target_volatility
            })
            
            def objective(w):
                return -np.dot(w, self.mean_returns)  # Maximize return
        
        else:
            # Maximize Sharpe
            def objective(w):
                ret = np.dot(w, self.mean_returns)
                vol = np.sqrt(w.T @ self.cov_matrix @ w)
                return -(ret - self.risk_free_rate) / vol if vol > 0 else 0
        
        result = minimize(
            objective, x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        optimal_weights = {self.assets[i]: float(result.x[i]) for i in range(self.n_assets)}
        stats = self.calculate_stats(optimal_weights)
        
        return {
            'optimal_weights': optimal_weights,
            **stats,
            'target_achieved': result.success,
            'constraint_binding': None
        }


# Factory functions
def create_digital_twin(
    returns: pd.DataFrame,
    weights: Dict[str, float],
    initial_capital: float = 100000,
    horizon_days: int = 252
) -> DigitalTwinEngine:
    """Create a DigitalTwinEngine instance."""
    return DigitalTwinEngine(returns, weights, initial_capital, horizon_days)


def create_what_if_analyzer(
    returns: pd.DataFrame,
    weights: Dict[str, float],
    risk_free_rate: float = 0.05,
    transaction_cost_bps: float = 10
) -> WhatIfAnalyzer:
    """Create a WhatIfAnalyzer instance."""
    return WhatIfAnalyzer(returns, weights, risk_free_rate, transaction_cost_bps)
