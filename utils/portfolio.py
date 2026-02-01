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
from typing import Dict, List, Tuple, Optional, Union
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
