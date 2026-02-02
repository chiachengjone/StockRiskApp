"""
Portfolio Enhancement Module

Advanced portfolio optimization features:
- Risk parity allocation
- Black-Litterman model
- Transaction cost analysis
- Rebalancing strategies
- Tax-loss harvesting signals
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from scipy.optimize import minimize, LinearConstraint
from scipy.stats import norm
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# RISK PARITY
# =============================================================================

def risk_parity_weights(
    returns: pd.DataFrame,
    target_risk: float = 0.10,
    max_weight: float = 0.4,
    min_weight: float = 0.02
) -> Dict:
    """
    Calculate risk parity portfolio weights.
    
    Risk parity allocates capital so each asset contributes equally to 
    portfolio risk. This provides better diversification than equal-weight
    or market-cap weighted approaches.
    
    Args:
        returns: DataFrame of asset returns (columns = assets)
        target_risk: Target annualized volatility
        max_weight: Maximum weight per asset
        min_weight: Minimum weight per asset
    
    Returns:
        Dictionary with weights and risk metrics
    """
    n_assets = len(returns.columns)
    cov_matrix = returns.cov() * 252
    
    # Risk contribution function
    def risk_contribution(weights, cov):
        portfolio_vol = np.sqrt(weights @ cov @ weights)
        marginal_contrib = (cov @ weights) / portfolio_vol
        risk_contrib = weights * marginal_contrib
        return risk_contrib
    
    # Objective: equal risk contribution
    def objective(weights):
        cov = cov_matrix.values
        rc = risk_contribution(weights, cov)
        target_rc = 1.0 / n_assets
        return np.sum((rc - target_rc) ** 2)
    
    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    ]
    
    bounds = [(min_weight, max_weight) for _ in range(n_assets)]
    
    # Initial guess: equal weights
    x0 = np.ones(n_assets) / n_assets
    
    # Optimize
    result = minimize(
        objective, x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000}
    )
    
    if not result.success:
        # Fallback to equal weights
        weights = np.ones(n_assets) / n_assets
    else:
        weights = result.x / result.x.sum()  # Normalize
    
    # Calculate risk metrics
    cov = cov_matrix.values
    portfolio_vol = np.sqrt(weights @ cov @ weights)
    risk_contrib = risk_contribution(weights, cov)
    
    # Leverage to hit target risk
    leverage = target_risk / portfolio_vol if portfolio_vol > 0 else 1.0
    
    return {
        'weights': dict(zip(returns.columns, weights)),
        'portfolio_volatility': float(portfolio_vol),
        'risk_contributions': dict(zip(returns.columns, risk_contrib)),
        'leverage_for_target': float(leverage),
        'levered_weights': dict(zip(returns.columns, weights * leverage)),
        'diversification_ratio': float(np.sum(np.sqrt(np.diag(cov)) * weights) / portfolio_vol),
        'success': result.success if hasattr(result, 'success') else True
    }


def hierarchical_risk_parity(
    returns: pd.DataFrame,
    method: str = 'complete'
) -> Dict:
    """
    Hierarchical Risk Parity (HRP) - clustering-based allocation.
    
    Uses hierarchical clustering to group similar assets and allocates
    risk more efficiently than traditional optimization methods.
    
    Args:
        returns: DataFrame of asset returns
        method: Clustering linkage method ('single', 'complete', 'average', 'ward')
    
    Returns:
        Dictionary with HRP weights and cluster info
    """
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform
    
    # Correlation matrix
    corr = returns.corr()
    
    # Distance matrix (1 - corr for similarity to distance)
    dist = ((1 - corr) / 2) ** 0.5
    
    # Hierarchical clustering
    condensed_dist = squareform(dist.values, checks=False)
    link = linkage(condensed_dist, method=method)
    
    # Get sorted order
    sort_idx = leaves_list(link)
    sorted_assets = [returns.columns[i] for i in sort_idx]
    
    # Calculate inverse variance weights with clustering structure
    def get_cluster_var(cov, assets):
        """Get variance of equal-weight cluster portfolio."""
        n = len(assets)
        weights = np.ones(n) / n
        return weights @ cov.loc[assets, assets].values @ weights
    
    # Recursive bisection
    def bisect_weights(assets, cov, weights):
        if len(assets) == 1:
            return
        
        mid = len(assets) // 2
        left_assets = assets[:mid]
        right_assets = assets[mid:]
        
        # Variance of each cluster
        left_var = get_cluster_var(cov, left_assets)
        right_var = get_cluster_var(cov, right_assets)
        
        # Inverse variance allocation
        total_var = left_var + right_var
        left_weight = 1 - left_var / total_var if total_var > 0 else 0.5
        right_weight = 1 - left_weight
        
        # Update weights
        for asset in left_assets:
            weights[asset] *= left_weight
        for asset in right_assets:
            weights[asset] *= right_weight
        
        # Recurse
        bisect_weights(left_assets, cov, weights)
        bisect_weights(right_assets, cov, weights)
    
    # Initialize weights
    cov = returns.cov() * 252
    weights = {asset: 1.0 for asset in sorted_assets}
    bisect_weights(sorted_assets, cov, weights)
    
    # Normalize
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}
    
    # Portfolio metrics
    w = np.array([weights[col] for col in returns.columns])
    portfolio_vol = np.sqrt(w @ cov.values @ w)
    
    return {
        'weights': weights,
        'sorted_order': sorted_assets,
        'portfolio_volatility': float(portfolio_vol),
        'method': method,
        'n_assets': len(returns.columns)
    }


# =============================================================================
# BLACK-LITTERMAN MODEL
# =============================================================================

def black_litterman_optimization(
    returns: pd.DataFrame,
    market_caps: Dict[str, float],
    views: List[Dict],
    risk_aversion: float = 2.5,
    tau: float = 0.05
) -> Dict:
    """
    Black-Litterman model combining market equilibrium with investor views.
    
    The Black-Litterman model solves the problem of unintuitive weights from
    mean-variance optimization by starting with market equilibrium and
    adjusting based on investor views.
    
    Args:
        returns: DataFrame of asset returns
        market_caps: Dict of market capitalizations {ticker: cap}
        views: List of view dictionaries:
            - {'asset': 'AAPL', 'return': 0.10}  # Absolute view
            - {'long': 'AAPL', 'short': 'MSFT', 'return': 0.02}  # Relative view
        risk_aversion: Risk aversion coefficient (default 2.5)
        tau: Scalar indicating uncertainty in equilibrium (default 0.05)
    
    Returns:
        Dictionary with optimal weights and posterior expected returns
    """
    assets = list(returns.columns)
    n_assets = len(assets)
    
    # Market weights from market caps
    total_cap = sum(market_caps.get(a, 1e9) for a in assets)
    market_weights = np.array([market_caps.get(a, 1e9) / total_cap for a in assets])
    
    # Covariance matrix
    cov = returns.cov().values * 252
    
    # Equilibrium returns (reverse optimization)
    pi = risk_aversion * cov @ market_weights
    
    if not views:
        # No views - return market weights
        return {
            'weights': dict(zip(assets, market_weights)),
            'expected_returns': dict(zip(assets, pi)),
            'equilibrium_returns': dict(zip(assets, pi)),
            'portfolio_return': float(market_weights @ pi),
            'portfolio_volatility': float(np.sqrt(market_weights @ cov @ market_weights)),
            'views_applied': 0
        }
    
    # Construct views matrix P and vector Q
    n_views = len(views)
    P = np.zeros((n_views, n_assets))
    Q = np.zeros(n_views)
    omega_diag = []  # View uncertainty
    
    for i, view in enumerate(views):
        if 'asset' in view:
            # Absolute view
            asset_idx = assets.index(view['asset'])
            P[i, asset_idx] = 1
            Q[i] = view['return']
        else:
            # Relative view
            long_idx = assets.index(view['long'])
            short_idx = assets.index(view['short'])
            P[i, long_idx] = 1
            P[i, short_idx] = -1
            Q[i] = view['return']
        
        # View uncertainty (proportional to variance of view portfolio)
        omega_diag.append(tau * P[i] @ cov @ P[i])
    
    Omega = np.diag(omega_diag)
    
    # Black-Litterman posterior
    tau_cov = tau * cov
    
    # Posterior covariance
    M1 = np.linalg.inv(tau_cov)
    M2 = P.T @ np.linalg.inv(Omega) @ P
    posterior_cov = np.linalg.inv(M1 + M2)
    
    # Posterior expected returns
    posterior_returns = posterior_cov @ (M1 @ pi + P.T @ np.linalg.inv(Omega) @ Q)
    
    # Optimal weights (mean-variance with posterior)
    def objective(w):
        ret = w @ posterior_returns
        risk = w @ cov @ w
        return -ret + 0.5 * risk_aversion * risk
    
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0, 0.4) for _ in range(n_assets)]
    x0 = market_weights
    
    result = minimize(
        objective, x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    optimal_weights = result.x if result.success else market_weights
    
    return {
        'weights': dict(zip(assets, optimal_weights)),
        'expected_returns': dict(zip(assets, posterior_returns)),
        'equilibrium_returns': dict(zip(assets, pi)),
        'market_weights': dict(zip(assets, market_weights)),
        'portfolio_return': float(optimal_weights @ posterior_returns),
        'portfolio_volatility': float(np.sqrt(optimal_weights @ cov @ optimal_weights)),
        'views_applied': n_views,
        'success': result.success
    }


# =============================================================================
# TRANSACTION COSTS
# =============================================================================

@dataclass
class TransactionCostModel:
    """Model for estimating transaction costs."""
    spread_bps: float = 5.0  # Bid-ask spread in basis points
    commission_per_share: float = 0.005  # Commission per share
    market_impact_bps: float = 10.0  # Market impact in basis points
    min_commission: float = 1.0  # Minimum commission per trade
    
    def estimate_cost(
        self,
        trade_value: float,
        shares: int,
        avg_daily_volume: float = None
    ) -> Dict:
        """
        Estimate total transaction cost.
        
        Args:
            trade_value: Dollar value of trade
            shares: Number of shares
            avg_daily_volume: Average daily volume for impact estimate
        
        Returns:
            Dictionary with cost breakdown
        """
        # Spread cost
        spread_cost = trade_value * (self.spread_bps / 10000)
        
        # Commission
        commission = max(shares * self.commission_per_share, self.min_commission)
        
        # Market impact (higher for larger trades relative to volume)
        if avg_daily_volume and avg_daily_volume > 0:
            participation_rate = shares / avg_daily_volume
            impact_multiplier = 1 + participation_rate * 10  # Nonlinear impact
        else:
            impact_multiplier = 1
        
        market_impact = trade_value * (self.market_impact_bps / 10000) * impact_multiplier
        
        total_cost = spread_cost + commission + market_impact
        
        return {
            'spread_cost': spread_cost,
            'commission': commission,
            'market_impact': market_impact,
            'total_cost': total_cost,
            'cost_bps': (total_cost / trade_value) * 10000 if trade_value > 0 else 0,
            'trade_value': trade_value
        }


def calculate_rebalance_costs(
    current_weights: Dict[str, float],
    target_weights: Dict[str, float],
    portfolio_value: float,
    prices: Dict[str, float],
    volumes: Dict[str, float] = None,
    cost_model: TransactionCostModel = None
) -> Dict:
    """
    Calculate costs of rebalancing to target weights.
    
    Args:
        current_weights: Current portfolio weights
        target_weights: Target portfolio weights
        portfolio_value: Total portfolio value
        prices: Current prices per asset
        volumes: Average daily volumes per asset
        cost_model: Transaction cost model
    
    Returns:
        Dictionary with rebalancing costs and trade list
    """
    if cost_model is None:
        cost_model = TransactionCostModel()
    
    volumes = volumes or {}
    
    all_assets = set(current_weights.keys()) | set(target_weights.keys())
    
    trades = []
    total_cost = 0
    turnover = 0
    
    for asset in all_assets:
        current_w = current_weights.get(asset, 0)
        target_w = target_weights.get(asset, 0)
        weight_diff = target_w - current_w
        
        if abs(weight_diff) < 0.001:  # Skip small changes
            continue
        
        trade_value = abs(weight_diff * portfolio_value)
        price = prices.get(asset, 100)
        shares = int(trade_value / price)
        volume = volumes.get(asset)
        
        cost_info = cost_model.estimate_cost(trade_value, shares, volume)
        
        trades.append({
            'asset': asset,
            'action': 'BUY' if weight_diff > 0 else 'SELL',
            'weight_change': weight_diff,
            'trade_value': trade_value,
            'shares': shares,
            'cost': cost_info['total_cost'],
            'cost_bps': cost_info['cost_bps']
        })
        
        total_cost += cost_info['total_cost']
        turnover += trade_value
    
    return {
        'trades': trades,
        'total_cost': total_cost,
        'total_cost_bps': (total_cost / portfolio_value) * 10000 if portfolio_value > 0 else 0,
        'turnover': turnover,
        'turnover_pct': (turnover / portfolio_value) * 100 if portfolio_value > 0 else 0,
        'n_trades': len(trades),
        'net_of_costs_value': portfolio_value - total_cost
    }


# =============================================================================
# REBALANCING STRATEGIES
# =============================================================================

def optimal_rebalance_frequency(
    returns: pd.DataFrame,
    target_weights: Dict[str, float],
    cost_per_rebalance: float = 0.001,
    frequencies: List[str] = None
) -> Dict:
    """
    Find optimal rebalancing frequency by backtesting.
    
    Tests different rebalancing frequencies and finds the one that
    maximizes risk-adjusted returns net of transaction costs.
    
    Args:
        returns: DataFrame of historical returns
        target_weights: Target portfolio weights
        cost_per_rebalance: Cost as fraction of portfolio per rebalance
        frequencies: List of frequencies to test
    
    Returns:
        Dictionary with optimal frequency and comparison metrics
    """
    if frequencies is None:
        frequencies = ['daily', 'weekly', 'monthly', 'quarterly', 'annually']
    
    freq_map = {
        'daily': 1,
        'weekly': 5,
        'monthly': 21,
        'quarterly': 63,
        'annually': 252
    }
    
    results = {}
    weights = np.array([target_weights.get(col, 0) for col in returns.columns])
    
    for freq in frequencies:
        days = freq_map.get(freq, 21)
        
        # Simulate portfolio with rebalancing
        portfolio_values = [1.0]
        current_weights = weights.copy()
        n_rebalances = 0
        
        for i, (date, row) in enumerate(returns.iterrows()):
            # Update weights based on returns
            returns_today = row.values
            new_values = current_weights * (1 + returns_today)
            current_weights = new_values / new_values.sum()
            
            # Portfolio return
            port_return = np.sum(current_weights * returns_today)
            
            # Rebalance?
            if (i + 1) % days == 0:
                # Apply rebalancing cost
                portfolio_values.append(
                    portfolio_values[-1] * (1 + port_return) * (1 - cost_per_rebalance)
                )
                current_weights = weights.copy()
                n_rebalances += 1
            else:
                portfolio_values.append(portfolio_values[-1] * (1 + port_return))
        
        portfolio_values = np.array(portfolio_values)
        daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        results[freq] = {
            'total_return': float(portfolio_values[-1] - 1),
            'annualized_return': float(np.mean(daily_returns) * 252),
            'volatility': float(np.std(daily_returns) * np.sqrt(252)),
            'sharpe_ratio': float(np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)),
            'n_rebalances': n_rebalances,
            'total_costs': float(n_rebalances * cost_per_rebalance)
        }
    
    # Find optimal
    optimal = max(results.keys(), key=lambda k: results[k]['sharpe_ratio'])
    
    return {
        'optimal_frequency': optimal,
        'results': results,
        'sharpe_improvement': results[optimal]['sharpe_ratio'] - results['monthly']['sharpe_ratio']
    }


def threshold_rebalancing(
    current_weights: Dict[str, float],
    target_weights: Dict[str, float],
    threshold: float = 0.05
) -> Dict:
    """
    Determine if rebalancing is needed based on drift threshold.
    
    Only rebalance when weights drift beyond threshold, reducing
    unnecessary trading costs.
    
    Args:
        current_weights: Current portfolio weights
        target_weights: Target portfolio weights
        threshold: Maximum allowed drift (e.g., 0.05 = 5%)
    
    Returns:
        Dictionary with rebalance decision and drift metrics
    """
    drifts = {}
    max_drift = 0
    
    all_assets = set(current_weights.keys()) | set(target_weights.keys())
    
    for asset in all_assets:
        current = current_weights.get(asset, 0)
        target = target_weights.get(asset, 0)
        drift = abs(current - target)
        drifts[asset] = drift
        max_drift = max(max_drift, drift)
    
    needs_rebalance = max_drift > threshold
    
    # Identify assets to trade
    assets_to_trade = [a for a, d in drifts.items() if d > threshold / 2]
    
    return {
        'needs_rebalance': needs_rebalance,
        'max_drift': float(max_drift),
        'threshold': threshold,
        'drifts': drifts,
        'assets_to_trade': assets_to_trade,
        'avg_drift': float(np.mean(list(drifts.values())))
    }


# =============================================================================
# TAX-LOSS HARVESTING
# =============================================================================

def tax_loss_harvesting_opportunities(
    positions: Dict[str, Dict],
    short_term_rate: float = 0.37,
    long_term_rate: float = 0.20,
    wash_sale_window: int = 30
) -> Dict:
    """
    Identify tax-loss harvesting opportunities.
    
    Tax-loss harvesting allows realizing losses to offset gains,
    reducing tax liability while maintaining market exposure.
    
    Args:
        positions: Dict of positions with 'cost_basis', 'current_value', 'purchase_date'
        short_term_rate: Short-term capital gains tax rate
        long_term_rate: Long-term capital gains tax rate
        wash_sale_window: Days to wait before repurchasing (30 for IRS rules)
    
    Returns:
        Dictionary with harvesting opportunities and tax savings
    """
    from datetime import datetime, timedelta
    
    today = datetime.now()
    opportunities = []
    total_harvestable_loss = 0
    potential_tax_savings = 0
    
    for ticker, pos in positions.items():
        cost_basis = pos.get('cost_basis', 0)
        current_value = pos.get('current_value', 0)
        gain_loss = current_value - cost_basis
        
        # Only interested in losses
        if gain_loss >= 0:
            continue
        
        # Determine holding period
        purchase_date = pos.get('purchase_date')
        if purchase_date:
            if isinstance(purchase_date, str):
                purchase_date = datetime.strptime(purchase_date, '%Y-%m-%d')
            holding_days = (today - purchase_date).days
            is_long_term = holding_days > 365
            tax_rate = long_term_rate if is_long_term else short_term_rate
        else:
            holding_days = None
            is_long_term = None
            tax_rate = short_term_rate  # Assume short-term for safety
        
        loss_amount = abs(gain_loss)
        tax_savings = loss_amount * tax_rate
        
        opportunities.append({
            'ticker': ticker,
            'loss_amount': loss_amount,
            'cost_basis': cost_basis,
            'current_value': current_value,
            'holding_days': holding_days,
            'is_long_term': is_long_term,
            'tax_rate': tax_rate,
            'tax_savings': tax_savings,
            'repurchase_date': (today + timedelta(days=wash_sale_window)).strftime('%Y-%m-%d')
        })
        
        total_harvestable_loss += loss_amount
        potential_tax_savings += tax_savings
    
    # Sort by tax savings
    opportunities.sort(key=lambda x: x['tax_savings'], reverse=True)
    
    return {
        'opportunities': opportunities,
        'total_harvestable_loss': total_harvestable_loss,
        'potential_tax_savings': potential_tax_savings,
        'n_opportunities': len(opportunities),
        'wash_sale_window': wash_sale_window,
        'max_annual_deduction': 3000  # IRS limit for net capital losses
    }


# =============================================================================
# PORTFOLIO OPTIMIZER CLASS
# =============================================================================

class PortfolioOptimizer:
    """
    Unified interface for portfolio optimization features.
    
    Combines risk parity, Black-Litterman, transaction costs,
    and rebalancing analysis.
    """
    
    def __init__(
        self,
        returns: pd.DataFrame,
        market_caps: Dict[str, float] = None,
        prices: Dict[str, float] = None,
        volumes: Dict[str, float] = None
    ):
        """
        Initialize portfolio optimizer.
        
        Args:
            returns: DataFrame of asset returns
            market_caps: Market capitalizations
            prices: Current prices
            volumes: Average daily volumes
        """
        self.returns = returns
        self.assets = list(returns.columns)
        self.n_assets = len(self.assets)
        self.market_caps = market_caps or {a: 1e9 for a in self.assets}
        self.prices = prices or {a: 100 for a in self.assets}
        self.volumes = volumes or {}
        self.cov_matrix = returns.cov() * 252
        self.cost_model = TransactionCostModel()
    
    def optimize(
        self,
        method: str = 'risk_parity',
        views: List[Dict] = None,
        **kwargs
    ) -> Dict:
        """
        Run portfolio optimization.
        
        Args:
            method: 'risk_parity', 'hrp', 'black_litterman', 'mean_variance'
            views: Views for Black-Litterman (optional)
            **kwargs: Additional method-specific arguments
        
        Returns:
            Optimization results
        """
        if method == 'risk_parity':
            return risk_parity_weights(self.returns, **kwargs)
        elif method == 'hrp':
            return hierarchical_risk_parity(self.returns, **kwargs)
        elif method == 'black_litterman':
            return black_litterman_optimization(
                self.returns, self.market_caps, views or [], **kwargs
            )
        elif method == 'mean_variance':
            return self._mean_variance_optimize(**kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _mean_variance_optimize(
        self,
        target_return: float = None,
        risk_aversion: float = 2.5
    ) -> Dict:
        """Standard mean-variance optimization."""
        from scipy.optimize import minimize
        
        mean_returns = self.returns.mean().values * 252
        cov = self.cov_matrix.values
        
        def objective(w):
            ret = w @ mean_returns
            risk = w @ cov @ w
            return -ret + 0.5 * risk_aversion * risk
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(0, 0.4) for _ in range(self.n_assets)]
        x0 = np.ones(self.n_assets) / self.n_assets
        
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        weights = result.x if result.success else x0
        
        return {
            'weights': dict(zip(self.assets, weights)),
            'expected_return': float(weights @ mean_returns),
            'volatility': float(np.sqrt(weights @ cov @ weights)),
            'sharpe_ratio': float((weights @ mean_returns) / np.sqrt(weights @ cov @ weights))
        }
    
    def analyze_rebalance(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        portfolio_value: float = 100000
    ) -> Dict:
        """
        Analyze rebalancing from current to target weights.
        
        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            portfolio_value: Total portfolio value
        
        Returns:
            Rebalancing analysis
        """
        # Transaction costs
        cost_analysis = calculate_rebalance_costs(
            current_weights, target_weights,
            portfolio_value, self.prices, self.volumes, self.cost_model
        )
        
        # Threshold analysis
        threshold_5 = threshold_rebalancing(current_weights, target_weights, 0.05)
        threshold_10 = threshold_rebalancing(current_weights, target_weights, 0.10)
        
        return {
            'costs': cost_analysis,
            'threshold_5pct': threshold_5,
            'threshold_10pct': threshold_10,
            'recommendation': 'rebalance' if threshold_5['needs_rebalance'] else 'hold'
        }
    
    def compare_methods(self) -> Dict:
        """
        Compare different optimization methods.
        
        Returns:
            Comparison of methods with metrics
        """
        methods = ['risk_parity', 'hrp', 'mean_variance']
        results = {}
        
        for method in methods:
            try:
                opt_result = self.optimize(method)
                weights = np.array([opt_result['weights'].get(a, 0) for a in self.assets])
                vol = np.sqrt(weights @ self.cov_matrix.values @ weights)
                ret = weights @ (self.returns.mean().values * 252)
                
                results[method] = {
                    'weights': opt_result['weights'],
                    'volatility': float(vol),
                    'expected_return': float(ret),
                    'sharpe': float(ret / vol) if vol > 0 else 0,
                    'max_weight': float(max(opt_result['weights'].values())),
                    'min_weight': float(min(opt_result['weights'].values()))
                }
            except Exception as e:
                results[method] = {'error': str(e)}
        
        return results
    
    def full_analysis(
        self,
        current_weights: Dict[str, float] = None,
        portfolio_value: float = 100000,
        views: List[Dict] = None
    ) -> Dict:
        """
        Complete portfolio analysis.
        
        Args:
            current_weights: Current weights (for rebalancing analysis)
            portfolio_value: Portfolio value
            views: Views for Black-Litterman
        
        Returns:
            Comprehensive analysis
        """
        # Method comparison
        comparison = self.compare_methods()
        
        # Risk parity (primary recommendation)
        risk_parity = self.optimize('risk_parity')
        
        # HRP
        hrp = self.optimize('hrp')
        
        # Black-Litterman if views provided
        if views:
            bl = self.optimize('black_litterman', views=views)
        else:
            bl = None
        
        # Rebalancing if current weights provided
        if current_weights:
            rebalance = self.analyze_rebalance(
                current_weights, risk_parity['weights'], portfolio_value
            )
        else:
            rebalance = None
        
        return {
            'risk_parity': risk_parity,
            'hierarchical_risk_parity': hrp,
            'black_litterman': bl,
            'method_comparison': comparison,
            'rebalancing': rebalance,
            'best_method': min(comparison.keys(), 
                              key=lambda m: comparison[m].get('volatility', float('inf'))
                              if 'error' not in comparison[m] else float('inf'))
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_portfolio_optimize(
    tickers: List[str],
    returns_df: pd.DataFrame,
    method: str = 'risk_parity'
) -> Dict:
    """
    Quick portfolio optimization.
    
    Args:
        tickers: List of ticker symbols
        returns_df: DataFrame with returns
        method: Optimization method
    
    Returns:
        Optimization results
    """
    optimizer = PortfolioOptimizer(returns_df)
    return optimizer.optimize(method)


def portfolio_risk_decomposition(
    weights: Dict[str, float],
    returns: pd.DataFrame
) -> Dict:
    """
    Decompose portfolio risk by asset.
    
    Args:
        weights: Portfolio weights
        returns: Asset returns
    
    Returns:
        Risk decomposition
    """
    cov = returns.cov().values * 252
    w = np.array([weights.get(col, 0) for col in returns.columns])
    
    portfolio_var = w @ cov @ w
    portfolio_vol = np.sqrt(portfolio_var)
    
    # Marginal risk contribution
    marginal_contrib = (cov @ w) / portfolio_vol
    
    # Risk contribution
    risk_contrib = w * marginal_contrib
    
    # Percentage contribution
    pct_contrib = risk_contrib / portfolio_vol * 100
    
    return {
        'portfolio_volatility': float(portfolio_vol),
        'risk_contributions': dict(zip(returns.columns, risk_contrib)),
        'pct_contributions': dict(zip(returns.columns, pct_contrib)),
        'marginal_contributions': dict(zip(returns.columns, marginal_contrib)),
        'concentration': float(np.sum(pct_contrib ** 2))  # HHI-style
    }


# =============================================================================
# PORTFOLIO SIMULATION ENGINE
# =============================================================================

@dataclass
class SimulationConfig:
    """Configuration for portfolio simulation."""
    initial_capital: float = 100000
    n_simulations: int = 1000
    horizon_days: int = 252
    rebalance_frequency: str = 'monthly'  # 'daily', 'weekly', 'monthly', 'quarterly', 'none'
    rebalance_threshold: float = 0.05  # 5% drift threshold
    transaction_cost_bps: float = 10  # 10 basis points
    tax_rate_short: float = 0.35  # Short-term capital gains
    tax_rate_long: float = 0.15  # Long-term capital gains
    enable_tax_loss_harvesting: bool = True
    tax_loss_threshold: float = -0.10  # 10% loss threshold
    dividend_yield: float = 0.02  # 2% annual dividend yield
    inflation_rate: float = 0.025  # 2.5% annual inflation


class PortfolioSimulator:
    """
    Monte Carlo simulation engine for portfolio analysis.
    
    Simulates:
    - Dynamic rebalancing strategies
    - Tax-loss harvesting
    - Transaction cost decay
    - Dividend reinvestment
    - Inflation-adjusted returns
    
    Usage:
        simulator = PortfolioSimulator(returns_df, weights, config)
        results = simulator.run_simulation()
    """
    
    def __init__(
        self,
        returns: pd.DataFrame,
        weights: Dict[str, float],
        config: SimulationConfig = None
    ):
        self.returns = returns
        self.weights = weights
        self.assets = list(returns.columns)
        self.n_assets = len(self.assets)
        self.config = config or SimulationConfig()
        
        # Calculate return statistics
        self.mean_returns = returns.mean().values
        self.cov_matrix = returns.cov().values
        self.cholesky = np.linalg.cholesky(self.cov_matrix)
        
        # Initialize weight array
        self.weight_array = np.array([weights.get(a, 0) for a in self.assets])
    
    def _generate_returns(self, n_days: int, n_sims: int) -> np.ndarray:
        """Generate correlated random returns using Cholesky decomposition."""
        # Generate uncorrelated random returns
        z = np.random.standard_normal((n_sims, n_days, self.n_assets))
        
        # Apply Cholesky to induce correlation
        correlated_z = np.zeros_like(z)
        for i in range(n_sims):
            for j in range(n_days):
                correlated_z[i, j] = self.cholesky @ z[i, j]
        
        # Scale to actual return distribution
        returns = self.mean_returns + correlated_z * np.sqrt(np.diag(self.cov_matrix))
        
        return returns
    
    def _rebalance_weights(
        self,
        current_weights: np.ndarray,
        target_weights: np.ndarray,
        portfolio_value: float
    ) -> Tuple[np.ndarray, float]:
        """
        Rebalance portfolio and calculate transaction costs.
        
        Returns:
            Tuple of (new_weights, transaction_cost)
        """
        # Calculate turnover
        turnover = np.sum(np.abs(current_weights - target_weights))
        
        # Transaction cost (proportional to turnover)
        cost = portfolio_value * turnover * (self.config.transaction_cost_bps / 10000)
        
        return target_weights.copy(), cost
    
    def _apply_tax_loss_harvesting(
        self,
        positions: np.ndarray,
        cost_basis: np.ndarray,
        current_prices: np.ndarray,
        days_held: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        Simulate tax-loss harvesting.
        
        Returns:
            Tuple of (tax_benefit, updated_cost_basis)
        """
        if not self.config.enable_tax_loss_harvesting:
            return 0.0, cost_basis
        
        # Calculate unrealized gains/losses
        gains = (current_prices - cost_basis) / cost_basis
        
        # Find positions with losses exceeding threshold
        harvest_mask = gains < self.config.tax_loss_threshold
        
        tax_benefit = 0.0
        new_cost_basis = cost_basis.copy()
        
        for i in range(len(positions)):
            if harvest_mask[i] and positions[i] > 0:
                loss = positions[i] * (cost_basis[i] - current_prices[i])
                
                # Apply appropriate tax rate
                if days_held[i] > 365:
                    tax_rate = self.config.tax_rate_long
                else:
                    tax_rate = self.config.tax_rate_short
                
                tax_benefit += loss * tax_rate
                
                # Reset cost basis (simulate selling and rebuying)
                new_cost_basis[i] = current_prices[i]
        
        return tax_benefit, new_cost_basis
    
    def _get_rebalance_days(self, n_days: int) -> List[int]:
        """Get days when rebalancing should occur."""
        if self.config.rebalance_frequency == 'none':
            return []
        elif self.config.rebalance_frequency == 'daily':
            return list(range(1, n_days))
        elif self.config.rebalance_frequency == 'weekly':
            return list(range(5, n_days, 5))
        elif self.config.rebalance_frequency == 'monthly':
            return list(range(21, n_days, 21))
        elif self.config.rebalance_frequency == 'quarterly':
            return list(range(63, n_days, 63))
        else:
            return list(range(21, n_days, 21))  # Default to monthly
    
    def run_simulation(self) -> Dict:
        """
        Run full Monte Carlo simulation.
        
        Returns:
            Dictionary with simulation results
        """
        n_sims = self.config.n_simulations
        n_days = self.config.horizon_days
        initial_capital = self.config.initial_capital
        
        # Generate random returns
        sim_returns = self._generate_returns(n_days, n_sims)
        
        # Initialize tracking arrays
        portfolio_values = np.zeros((n_sims, n_days + 1))
        portfolio_values[:, 0] = initial_capital
        
        # Track costs and taxes
        total_tx_costs = np.zeros(n_sims)
        total_tax_benefits = np.zeros(n_sims)
        total_dividends = np.zeros(n_sims)
        
        # Rebalance days
        rebalance_days = set(self._get_rebalance_days(n_days))
        
        # Daily dividend rate
        daily_div_rate = self.config.dividend_yield / 252
        
        # Run simulations
        for sim in range(n_sims):
            current_weights = self.weight_array.copy()
            current_value = initial_capital
            cost_basis = np.ones(self.n_assets) * 100  # Arbitrary starting price
            days_held = np.zeros(self.n_assets)
            
            for day in range(n_days):
                # Apply daily returns
                daily_ret = sim_returns[sim, day]
                
                # Update weights based on returns
                new_weights = current_weights * (1 + daily_ret)
                new_weights = new_weights / np.sum(new_weights)
                
                # Update portfolio value
                portfolio_return = np.sum(current_weights * daily_ret)
                current_value = current_value * (1 + portfolio_return)
                
                # Add dividends
                dividend = current_value * daily_div_rate
                current_value += dividend
                total_dividends[sim] += dividend
                
                # Increment days held
                days_held += 1
                
                # Check for threshold-based rebalancing
                needs_rebalance = False
                if day + 1 in rebalance_days:
                    drift = np.max(np.abs(new_weights - self.weight_array))
                    if drift > self.config.rebalance_threshold:
                        needs_rebalance = True
                
                # Rebalance if needed
                if needs_rebalance:
                    new_weights, tx_cost = self._rebalance_weights(
                        new_weights, self.weight_array, current_value
                    )
                    current_value -= tx_cost
                    total_tx_costs[sim] += tx_cost
                    
                    # Tax-loss harvesting opportunity
                    current_prices = cost_basis * (1 + daily_ret)
                    tax_benefit, cost_basis = self._apply_tax_loss_harvesting(
                        new_weights * current_value / current_prices,
                        cost_basis,
                        current_prices,
                        days_held
                    )
                    total_tax_benefits[sim] += tax_benefit
                    days_held = np.zeros(self.n_assets)  # Reset for harvested positions
                
                current_weights = new_weights
                portfolio_values[sim, day + 1] = current_value
        
        # Calculate results
        final_values = portfolio_values[:, -1]
        total_returns = (final_values - initial_capital) / initial_capital
        
        # Inflation adjustment
        inflation_factor = (1 + self.config.inflation_rate) ** (n_days / 252)
        real_returns = total_returns - (inflation_factor - 1)
        
        # Statistics
        return {
            'mean_return': float(np.mean(total_returns)),
            'median_return': float(np.median(total_returns)),
            'std_return': float(np.std(total_returns)),
            'mean_real_return': float(np.mean(real_returns)),
            'var_95': float(np.percentile(total_returns, 5)),
            'cvar_95': float(np.mean(total_returns[total_returns <= np.percentile(total_returns, 5)])),
            'max_return': float(np.max(total_returns)),
            'min_return': float(np.min(total_returns)),
            'sharpe_ratio': float(np.mean(total_returns) / np.std(total_returns)) if np.std(total_returns) > 0 else 0,
            'prob_loss': float(np.mean(total_returns < 0)),
            'prob_gain_10': float(np.mean(total_returns > 0.10)),
            'mean_tx_costs': float(np.mean(total_tx_costs)),
            'mean_tax_benefits': float(np.mean(total_tax_benefits)),
            'mean_dividends': float(np.mean(total_dividends)),
            'net_costs': float(np.mean(total_tx_costs) - np.mean(total_tax_benefits)),
            'portfolio_values': portfolio_values,
            'final_values': final_values,
            'percentiles': {
                '5': float(np.percentile(final_values, 5)),
                '25': float(np.percentile(final_values, 25)),
                '50': float(np.percentile(final_values, 50)),
                '75': float(np.percentile(final_values, 75)),
                '95': float(np.percentile(final_values, 95))
            }
        }
    
    def compare_strategies(self) -> Dict:
        """
        Compare different rebalancing strategies.
        
        Returns:
            Comparison of rebalancing strategies
        """
        strategies = {
            'no_rebalance': 'none',
            'daily': 'daily',
            'weekly': 'weekly',
            'monthly': 'monthly',
            'quarterly': 'quarterly'
        }
        
        results = {}
        original_freq = self.config.rebalance_frequency
        
        for name, freq in strategies.items():
            self.config.rebalance_frequency = freq
            sim_result = self.run_simulation()
            
            results[name] = {
                'mean_return': sim_result['mean_return'],
                'std_return': sim_result['std_return'],
                'sharpe_ratio': sim_result['sharpe_ratio'],
                'var_95': sim_result['var_95'],
                'mean_tx_costs': sim_result['mean_tx_costs'],
                'net_return': sim_result['mean_return'] - sim_result['mean_tx_costs'] / self.config.initial_capital
            }
        
        self.config.rebalance_frequency = original_freq
        return results


class DynamicRebalancer:
    """
    Dynamic rebalancing engine with multiple strategies.
    
    Strategies:
    - Calendar-based (fixed schedule)
    - Threshold-based (drift triggers)
    - Volatility-adjusted (rebalance more in high vol)
    - Momentum-aware (let winners run)
    """
    
    def __init__(
        self,
        target_weights: Dict[str, float],
        returns: pd.DataFrame,
        cost_model: TransactionCostModel = None
    ):
        self.target_weights = target_weights
        self.returns = returns
        self.cost_model = cost_model or TransactionCostModel()
        self.assets = list(target_weights.keys())
    
    def calendar_rebalance(
        self,
        current_weights: Dict[str, float],
        frequency: str = 'monthly'
    ) -> Dict:
        """Calendar-based rebalancing."""
        return {
            'rebalance': True,
            'new_weights': self.target_weights.copy(),
            'strategy': 'calendar',
            'frequency': frequency
        }
    
    def threshold_rebalance(
        self,
        current_weights: Dict[str, float],
        threshold: float = 0.05
    ) -> Dict:
        """Threshold-based rebalancing."""
        max_drift = max(
            abs(current_weights.get(a, 0) - self.target_weights.get(a, 0))
            for a in self.assets
        )
        
        if max_drift > threshold:
            return {
                'rebalance': True,
                'new_weights': self.target_weights.copy(),
                'strategy': 'threshold',
                'max_drift': max_drift
            }
        else:
            return {
                'rebalance': False,
                'new_weights': current_weights,
                'strategy': 'threshold',
                'max_drift': max_drift
            }
    
    def volatility_adjusted_rebalance(
        self,
        current_weights: Dict[str, float],
        lookback: int = 20
    ) -> Dict:
        """Volatility-adjusted rebalancing."""
        # Calculate recent volatility
        recent_vol = self.returns.tail(lookback).std().mean() * np.sqrt(252)
        historical_vol = self.returns.std().mean() * np.sqrt(252)
        
        vol_ratio = recent_vol / historical_vol if historical_vol > 0 else 1.0
        
        # More frequent rebalancing in high volatility
        threshold = 0.05 / vol_ratio  # Lower threshold = more rebalancing
        
        return self.threshold_rebalance(current_weights, min(0.10, threshold))
    
    def momentum_aware_rebalance(
        self,
        current_weights: Dict[str, float],
        momentum_lookback: int = 60,
        momentum_threshold: float = 0.05
    ) -> Dict:
        """Momentum-aware rebalancing (let winners run)."""
        # Calculate momentum
        momentum = self.returns.tail(momentum_lookback).sum()
        
        # Adjust target weights based on momentum
        adjusted_weights = {}
        total_adj = 0
        
        for asset in self.assets:
            base_weight = self.target_weights.get(asset, 0)
            asset_momentum = momentum.get(asset, 0)
            
            if asset_momentum > momentum_threshold:
                # Let winners run - reduce rebalancing pressure
                adj_factor = 1.1
            elif asset_momentum < -momentum_threshold:
                # Cut losers faster
                adj_factor = 0.9
            else:
                adj_factor = 1.0
            
            adjusted_weights[asset] = base_weight * adj_factor
            total_adj += adjusted_weights[asset]
        
        # Normalize
        for asset in adjusted_weights:
            adjusted_weights[asset] /= total_adj
        
        return {
            'rebalance': True,
            'new_weights': adjusted_weights,
            'strategy': 'momentum_aware',
            'momentum': dict(momentum)
        }


# =============================================================================
# CORRELATION CONVERGENCE ALERTS
# =============================================================================

class CorrelationMonitor:
    """
    Monitor correlation changes between assets.
    
    Alerts when normally uncorrelated assets begin moving together,
    which often signals market stress.
    """
    
    def __init__(
        self,
        returns: pd.DataFrame,
        baseline_window: int = 252,
        alert_window: int = 21,
        alert_threshold: float = 0.3
    ):
        self.returns = returns
        self.baseline_window = baseline_window
        self.alert_window = alert_window
        self.alert_threshold = alert_threshold
        self.assets = list(returns.columns)
    
    def calculate_baseline_correlation(self) -> pd.DataFrame:
        """Calculate baseline correlation matrix."""
        if len(self.returns) < self.baseline_window:
            return self.returns.corr()
        return self.returns.tail(self.baseline_window).corr()
    
    def calculate_recent_correlation(self) -> pd.DataFrame:
        """Calculate recent correlation matrix."""
        return self.returns.tail(self.alert_window).corr()
    
    def detect_convergence(self) -> Dict:
        """
        Detect correlation convergence/divergence.
        
        Returns:
            Dictionary with alerts and details
        """
        baseline = self.calculate_baseline_correlation()
        recent = self.calculate_recent_correlation()
        
        # Calculate correlation changes
        change = recent - baseline
        
        alerts = []
        
        for i, asset1 in enumerate(self.assets):
            for j, asset2 in enumerate(self.assets):
                if i >= j:
                    continue
                
                baseline_corr = baseline.loc[asset1, asset2]
                recent_corr = recent.loc[asset1, asset2]
                corr_change = change.loc[asset1, asset2]
                
                # Alert if correlation increased significantly
                if abs(corr_change) > self.alert_threshold:
                    alert_type = 'convergence' if corr_change > 0 else 'divergence'
                    
                    alerts.append({
                        'asset1': asset1,
                        'asset2': asset2,
                        'baseline_correlation': float(baseline_corr),
                        'recent_correlation': float(recent_corr),
                        'change': float(corr_change),
                        'alert_type': alert_type,
                        'severity': 'high' if abs(corr_change) > 0.5 else 'medium'
                    })
        
        # Calculate average correlation (market stress indicator)
        baseline_avg = baseline.values[np.triu_indices_from(baseline.values, 1)].mean()
        recent_avg = recent.values[np.triu_indices_from(recent.values, 1)].mean()
        
        return {
            'alerts': alerts,
            'n_alerts': len(alerts),
            'baseline_avg_correlation': float(baseline_avg),
            'recent_avg_correlation': float(recent_avg),
            'correlation_regime': 'stressed' if recent_avg > baseline_avg + 0.2 else 'normal',
            'baseline_matrix': baseline,
            'recent_matrix': recent,
            'change_matrix': change
        }
    
    def get_rolling_average_correlation(
        self,
        window: int = 21
    ) -> pd.Series:
        """Calculate rolling average pairwise correlation."""
        rolling_corr = []
        dates = []
        
        for i in range(window, len(self.returns)):
            subset = self.returns.iloc[i-window:i]
            corr = subset.corr()
            avg_corr = corr.values[np.triu_indices_from(corr.values, 1)].mean()
            rolling_corr.append(avg_corr)
            dates.append(self.returns.index[i])
        
        return pd.Series(rolling_corr, index=dates, name='avg_correlation')


# =============================================================================
# LIQUIDITY RISK ENGINE (v4.3 Professional)
# =============================================================================

@dataclass
class LiquidityMetrics:
    """Container for liquidity risk metrics."""
    time_to_liquidate_days: float
    market_impact_bps: float
    liquidity_score: float  # 0-100
    liquidity_grade: str  # A, B, C, D, F
    warning_level: str  # none, low, medium, high, critical
    participation_rate: float
    slippage_estimate_pct: float


def calculate_liquidity_risk(
    positions: Dict[str, float],
    volumes: Dict[str, float],
    prices: Dict[str, float] = None,
    participation_rate: float = 0.10,
    urgency: str = 'normal'
) -> Dict[str, Any]:
    """
    Calculate comprehensive liquidity risk metrics.
    
    Computes Time to Liquidate (TTL), Market Impact, and slippage estimates
    based on Average Daily Volume (ADV) and position sizes.
    
    Args:
        positions: Dict of {ticker: position_value_or_shares}
        volumes: Dict of {ticker: average_daily_volume}
        prices: Dict of {ticker: current_price} (optional, for dollar calculations)
        participation_rate: Max % of ADV to trade (default 10%)
        urgency: 'low', 'normal', 'high', 'urgent' - affects impact estimates
    
    Returns:
        Dictionary with liquidity metrics and warnings
    """
    if not positions or not volumes:
        return {'error': 'Missing positions or volume data'}
    
    # Urgency multipliers for market impact
    urgency_factors = {
        'low': 0.5,      # Patient trading, minimal impact
        'normal': 1.0,   # Standard execution
        'high': 1.5,     # Need to execute faster
        'urgent': 2.5    # Liquidation pressure, maximum impact
    }
    urgency_mult = urgency_factors.get(urgency, 1.0)
    
    results = {}
    total_value = 0
    total_ttl_weighted = 0
    total_impact_weighted = 0
    warnings = []
    
    for ticker, position in positions.items():
        adv = volumes.get(ticker, 0)
        price = prices.get(ticker, 100) if prices else 100
        
        # Convert position to shares if value provided
        if isinstance(position, float) and position > 1000:
            # Assume it's a dollar value
            shares = position / price
            position_value = position
        else:
            shares = position
            position_value = shares * price
        
        total_value += position_value
        
        # Skip if no volume data
        if adv <= 0:
            results[ticker] = {
                'error': 'No volume data',
                'ttl_days': float('inf'),
                'liquidity_score': 0
            }
            warnings.append({
                'ticker': ticker,
                'level': 'critical',
                'message': 'No volume data available'
            })
            continue
        
        # Time to Liquidate (days)
        daily_tradeable = adv * participation_rate
        ttl_days = shares / daily_tradeable if daily_tradeable > 0 else float('inf')
        
        # Market Impact Estimate (basis points)
        # Using square-root market impact model: Impact ≈ k * sqrt(shares / ADV)
        impact_coefficient = 10.0  # Base impact in bps
        raw_impact = impact_coefficient * np.sqrt(shares / adv) * 100
        market_impact_bps = raw_impact * urgency_mult
        
        # Slippage estimate (percentage)
        slippage_pct = market_impact_bps / 100
        
        # Liquidity score (0-100, higher = more liquid)
        # Based on TTL and relative position size
        position_pct_of_adv = (shares / adv) * 100
        if ttl_days <= 0.5:
            liquidity_score = 95
        elif ttl_days <= 1:
            liquidity_score = 85
        elif ttl_days <= 3:
            liquidity_score = 70
        elif ttl_days <= 5:
            liquidity_score = 50
        elif ttl_days <= 10:
            liquidity_score = 30
        else:
            liquidity_score = max(0, 20 - ttl_days)
        
        # Liquidity grade
        if liquidity_score >= 80:
            grade = 'A'
        elif liquidity_score >= 60:
            grade = 'B'
        elif liquidity_score >= 40:
            grade = 'C'
        elif liquidity_score >= 20:
            grade = 'D'
        else:
            grade = 'F'
        
        # Warning level
        if ttl_days > 5:
            warning_level = 'critical'
            warnings.append({
                'ticker': ticker,
                'level': 'critical',
                'message': f'Liquidation takes >{ttl_days:.1f} days (exceeds 5-day threshold)'
            })
        elif ttl_days > 3:
            warning_level = 'high'
            warnings.append({
                'ticker': ticker,
                'level': 'high',
                'message': f'Liquidation takes {ttl_days:.1f} days (>3 days)'
            })
        elif ttl_days > 1:
            warning_level = 'medium'
        elif ttl_days > 0.5:
            warning_level = 'low'
        else:
            warning_level = 'none'
        
        results[ticker] = {
            'position_value': float(position_value),
            'shares': float(shares),
            'adv': float(adv),
            'ttl_days': float(ttl_days),
            'market_impact_bps': float(market_impact_bps),
            'slippage_pct': float(slippage_pct),
            'liquidity_score': float(liquidity_score),
            'grade': grade,
            'warning_level': warning_level,
            'pct_of_adv': float(position_pct_of_adv)
        }
        
        # Weighted aggregates
        total_ttl_weighted += ttl_days * position_value
        total_impact_weighted += market_impact_bps * position_value
    
    # Portfolio-level metrics
    if total_value > 0:
        portfolio_ttl = total_ttl_weighted / total_value
        portfolio_impact = total_impact_weighted / total_value
    else:
        portfolio_ttl = 0
        portfolio_impact = 0
    
    # Portfolio liquidity score
    portfolio_score = np.mean([r['liquidity_score'] for r in results.values() if 'liquidity_score' in r])
    
    # Portfolio warning
    if portfolio_ttl > 5:
        portfolio_warning = 'critical'
    elif portfolio_ttl > 3:
        portfolio_warning = 'high'
    elif portfolio_ttl > 1:
        portfolio_warning = 'medium'
    else:
        portfolio_warning = 'low'
    
    return {
        'positions': results,
        'portfolio': {
            'total_value': float(total_value),
            'weighted_ttl_days': float(portfolio_ttl),
            'weighted_impact_bps': float(portfolio_impact),
            'liquidity_score': float(portfolio_score),
            'warning_level': portfolio_warning,
            'participation_rate': participation_rate,
            'urgency': urgency
        },
        'warnings': warnings,
        'n_warnings': len(warnings),
        'requires_attention': portfolio_ttl > 3 or len([w for w in warnings if w['level'] == 'critical']) > 0
    }


def estimate_execution_cost(
    position_value: float,
    adv_value: float,
    volatility: float = 0.02,
    urgency: str = 'normal',
    spread_bps: float = 5.0
) -> Dict[str, float]:
    """
    Estimate total execution cost for a trade.
    
    Uses Almgren-Chriss inspired market impact model.
    
    Args:
        position_value: Dollar value to trade
        adv_value: Average daily volume in dollars
        volatility: Daily volatility of the asset
        urgency: Trading urgency level
        spread_bps: Bid-ask spread in basis points
    
    Returns:
        Dictionary with cost breakdown
    """
    # Participation rate based on urgency
    urgency_participation = {
        'low': 0.05,
        'normal': 0.10,
        'high': 0.20,
        'urgent': 0.30
    }
    participation = urgency_participation.get(urgency, 0.10)
    
    # Time to execute
    execution_days = position_value / (adv_value * participation)
    
    # Permanent impact (price moves against you permanently)
    # γ * σ * sqrt(X/V) where X=shares, V=ADV
    permanent_impact_coef = 0.1
    permanent_impact = permanent_impact_coef * volatility * np.sqrt(position_value / adv_value)
    
    # Temporary impact (decays after trade)
    # η * σ * (X/V)^0.6
    temporary_impact_coef = 0.05
    temporary_impact = temporary_impact_coef * volatility * (position_value / adv_value) ** 0.6
    
    # Timing risk (variance of execution price)
    timing_risk = volatility * np.sqrt(execution_days) * position_value
    
    # Spread cost
    spread_cost = position_value * (spread_bps / 10000)
    
    # Total cost
    total_impact = (permanent_impact + temporary_impact) * position_value
    total_cost = total_impact + spread_cost
    total_cost_bps = (total_cost / position_value) * 10000
    
    return {
        'execution_days': float(execution_days),
        'permanent_impact_pct': float(permanent_impact * 100),
        'temporary_impact_pct': float(temporary_impact * 100),
        'spread_cost': float(spread_cost),
        'timing_risk': float(timing_risk),
        'total_impact_cost': float(total_impact),
        'total_cost': float(total_cost),
        'total_cost_bps': float(total_cost_bps),
        'participation_rate': participation
    }
