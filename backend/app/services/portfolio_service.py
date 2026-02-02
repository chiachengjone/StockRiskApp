"""
Portfolio Service
=================
Portfolio optimization and analysis functions.

Ported from utils/portfolio.py with identical calculations.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# RISK PARITY
# ============================================================================

def calculate_risk_parity(
    returns: pd.DataFrame,
    target_risk: float = 0.10,
    max_weight: float = 0.4,
    min_weight: float = 0.02
) -> Dict[str, Any]:
    """
    Calculate Risk Parity portfolio weights.
    
    Equal risk contribution across all assets.
    """
    n_assets = len(returns.columns)
    cov_matrix = returns.cov() * 252
    
    def risk_contribution(weights, cov):
        portfolio_vol = np.sqrt(weights @ cov @ weights)
        marginal_contrib = (cov @ weights) / portfolio_vol
        return weights * marginal_contrib
    
    def objective(weights):
        cov = cov_matrix.values
        rc = risk_contribution(weights, cov)
        target_rc = 1.0 / n_assets
        return np.sum((rc - target_rc) ** 2)
    
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    bounds = [(min_weight, max_weight) for _ in range(n_assets)]
    x0 = np.ones(n_assets) / n_assets
    
    result = minimize(
        objective, x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000}
    )
    
    if not result.success:
        weights = np.ones(n_assets) / n_assets
    else:
        weights = result.x / result.x.sum()
    
    cov = cov_matrix.values
    portfolio_vol = np.sqrt(weights @ cov @ weights)
    risk_contrib = risk_contribution(weights, cov)
    leverage = target_risk / portfolio_vol if portfolio_vol > 0 else 1.0
    
    return {
        'weights': dict(zip(returns.columns, weights.tolist())),
        'portfolio_volatility': float(portfolio_vol),
        'risk_contributions': dict(zip(returns.columns, risk_contrib.tolist())),
        'leverage_for_target': float(leverage),
        'levered_weights': dict(zip(returns.columns, (weights * leverage).tolist())),
        'diversification_ratio': float(np.sum(np.sqrt(np.diag(cov)) * weights) / portfolio_vol),
        'success': result.success if hasattr(result, 'success') else True
    }


def calculate_hrp(
    returns: pd.DataFrame,
    method: str = 'complete'
) -> Dict[str, Any]:
    """
    Hierarchical Risk Parity optimization.
    """
    corr = returns.corr()
    dist = ((1 - corr) / 2) ** 0.5
    
    condensed_dist = squareform(dist.values, checks=False)
    link = linkage(condensed_dist, method=method)
    sort_idx = leaves_list(link)
    sorted_assets = [returns.columns[i] for i in sort_idx]
    
    def get_cluster_var(cov, assets):
        n = len(assets)
        w = np.ones(n) / n
        return w @ cov.loc[assets, assets].values @ w
    
    def bisect_weights(assets, cov, weights):
        if len(assets) == 1:
            return
        
        mid = len(assets) // 2
        left_assets = assets[:mid]
        right_assets = assets[mid:]
        
        left_var = get_cluster_var(cov, left_assets)
        right_var = get_cluster_var(cov, right_assets)
        
        total_var = left_var + right_var
        left_weight = 1 - left_var / total_var if total_var > 0 else 0.5
        right_weight = 1 - left_weight
        
        for asset in left_assets:
            weights[asset] *= left_weight
        for asset in right_assets:
            weights[asset] *= right_weight
        
        bisect_weights(left_assets, cov, weights)
        bisect_weights(right_assets, cov, weights)
    
    cov = returns.cov() * 252
    weights = {asset: 1.0 for asset in sorted_assets}
    bisect_weights(sorted_assets, cov, weights)
    
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}
    
    w = np.array([weights[col] for col in returns.columns])
    portfolio_vol = np.sqrt(w @ cov.values @ w)
    
    return {
        'weights': weights,
        'sorted_order': sorted_assets,
        'portfolio_volatility': float(portfolio_vol),
        'method': method,
        'n_assets': len(returns.columns)
    }


# ============================================================================
# BLACK-LITTERMAN
# ============================================================================

def calculate_black_litterman(
    returns: pd.DataFrame,
    market_caps: Dict[str, float],
    views: List[Dict],
    risk_aversion: float = 2.5,
    tau: float = 0.05
) -> Dict[str, Any]:
    """
    Black-Litterman portfolio optimization.
    """
    assets = list(returns.columns)
    n_assets = len(assets)
    
    total_cap = sum(market_caps.get(a, 1e9) for a in assets)
    market_weights = np.array([market_caps.get(a, 1e9) / total_cap for a in assets])
    
    cov = returns.cov().values * 252
    pi = risk_aversion * cov @ market_weights
    
    if not views:
        return {
            'weights': dict(zip(assets, market_weights.tolist())),
            'expected_returns': dict(zip(assets, pi.tolist())),
            'equilibrium_returns': dict(zip(assets, pi.tolist())),
            'market_weights': dict(zip(assets, market_weights.tolist())),
            'portfolio_return': float(market_weights @ pi),
            'portfolio_volatility': float(np.sqrt(market_weights @ cov @ market_weights)),
            'views_applied': 0,
            'success': True
        }
    
    n_views = len(views)
    P = np.zeros((n_views, n_assets))
    Q = np.zeros(n_views)
    omega_diag = []
    
    for i, view in enumerate(views):
        if 'asset' in view:
            asset_idx = assets.index(view['asset'])
            P[i, asset_idx] = 1
            Q[i] = view['return']
        else:
            long_idx = assets.index(view['long'])
            short_idx = assets.index(view['short'])
            P[i, long_idx] = 1
            P[i, short_idx] = -1
            Q[i] = view['return']
        
        omega_diag.append(tau * P[i] @ cov @ P[i])
    
    Omega = np.diag(omega_diag)
    tau_cov = tau * cov
    
    M1 = np.linalg.inv(tau_cov)
    M2 = P.T @ np.linalg.inv(Omega) @ P
    posterior_cov = np.linalg.inv(M1 + M2)
    posterior_returns = posterior_cov @ (M1 @ pi + P.T @ np.linalg.inv(Omega) @ Q)
    
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
        'weights': dict(zip(assets, optimal_weights.tolist())),
        'expected_returns': dict(zip(assets, posterior_returns.tolist())),
        'equilibrium_returns': dict(zip(assets, pi.tolist())),
        'market_weights': dict(zip(assets, market_weights.tolist())),
        'portfolio_return': float(optimal_weights @ posterior_returns),
        'portfolio_volatility': float(np.sqrt(optimal_weights @ cov @ optimal_weights)),
        'views_applied': n_views,
        'success': result.success
    }


# ============================================================================
# EFFICIENT FRONTIER
# ============================================================================

def calculate_efficient_frontier(
    returns: pd.DataFrame,
    n_points: int = 50
) -> Dict[str, Any]:
    """
    Calculate efficient frontier points.
    """
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    n_assets = len(returns.columns)
    
    results = []
    target_returns = np.linspace(float(mean_returns.min()), float(mean_returns.max()), n_points)
    
    for target in target_returns:
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w, t=target: w @ mean_returns.values - t}
        ]
        bounds = [(0, 1) for _ in range(n_assets)]
        initial = np.ones(n_assets) / n_assets
        
        def port_vol(weights):
            return np.sqrt(weights @ cov_matrix.values @ weights)
        
        try:
            result = minimize(port_vol, initial, method='SLSQP', bounds=bounds, constraints=constraints)
            if result.success:
                results.append({
                    'return': float(target),
                    'volatility': float(result.fun),
                    'sharpe': float(target / result.fun) if result.fun > 0 else 0
                })
        except:
            pass
    
    # Find optimal (max Sharpe)
    if results:
        optimal = max(results, key=lambda x: x['sharpe'])
    else:
        optimal = {'return': 0, 'volatility': 0, 'sharpe': 0}
    
    return {
        'frontier': results,
        'optimal_portfolio': optimal
    }


# ============================================================================
# TRANSACTION COSTS
# ============================================================================

def calculate_transaction_costs(
    current_weights: Dict[str, float],
    target_weights: Dict[str, float],
    portfolio_value: float,
    prices: Dict[str, float],
    volumes: Dict[str, float] = None,
    spread_bps: float = 5.0,
    commission_per_share: float = 0.005,
    market_impact_bps: float = 10.0
) -> Dict[str, Any]:
    """
    Calculate transaction costs for rebalancing.
    """
    volumes = volumes or {}
    all_assets = set(current_weights.keys()) | set(target_weights.keys())
    
    trades = []
    total_cost = 0
    turnover = 0
    
    for asset in all_assets:
        current_w = current_weights.get(asset, 0)
        target_w = target_weights.get(asset, 0)
        weight_diff = target_w - current_w
        
        if abs(weight_diff) < 0.001:
            continue
        
        trade_value = abs(weight_diff * portfolio_value)
        price = prices.get(asset, 100)
        shares = int(trade_value / price)
        
        # Calculate costs
        spread_cost = trade_value * (spread_bps / 10000)
        commission = max(shares * commission_per_share, 1.0)
        
        volume = volumes.get(asset)
        if volume and volume > 0:
            participation_rate = shares / volume
            impact_multiplier = 1 + participation_rate * 10
        else:
            impact_multiplier = 1
        
        market_impact = trade_value * (market_impact_bps / 10000) * impact_multiplier
        cost = spread_cost + commission + market_impact
        
        trades.append({
            'asset': asset,
            'action': 'BUY' if weight_diff > 0 else 'SELL',
            'weight_change': float(weight_diff),
            'trade_value': float(trade_value),
            'shares': shares,
            'cost': float(cost),
            'cost_bps': float((cost / trade_value) * 10000) if trade_value > 0 else 0
        })
        
        total_cost += cost
        turnover += trade_value
    
    return {
        'trades': trades,
        'total_cost': float(total_cost),
        'total_cost_bps': float((total_cost / portfolio_value) * 10000) if portfolio_value > 0 else 0,
        'turnover': float(turnover),
        'turnover_pct': float((turnover / portfolio_value) * 100) if portfolio_value > 0 else 0,
        'n_trades': len(trades),
        'net_of_costs_value': float(portfolio_value - total_cost)
    }


# ============================================================================
# REBALANCING
# ============================================================================

def analyze_rebalancing(
    returns: pd.DataFrame,
    target_weights: Dict[str, float],
    cost_per_rebalance: float = 0.001
) -> Dict[str, Any]:
    """
    Analyze optimal rebalancing frequency.
    """
    frequencies = ['daily', 'weekly', 'monthly', 'quarterly', 'annually']
    freq_map = {'daily': 1, 'weekly': 5, 'monthly': 21, 'quarterly': 63, 'annually': 252}
    
    results = {}
    weights = np.array([target_weights.get(col, 0) for col in returns.columns])
    
    for freq in frequencies:
        days = freq_map.get(freq, 21)
        
        portfolio_values = [1.0]
        current_weights = weights.copy()
        n_rebalances = 0
        
        for i, (date, row) in enumerate(returns.iterrows()):
            returns_today = row.values
            new_values = current_weights * (1 + returns_today)
            current_weights = new_values / new_values.sum()
            
            port_return = np.sum(current_weights * returns_today)
            
            if (i + 1) % days == 0:
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
            'sharpe_ratio': float(np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)) if np.std(daily_returns) > 0 else 0,
            'n_rebalances': n_rebalances,
            'total_costs': float(n_rebalances * cost_per_rebalance)
        }
    
    optimal = max(results.keys(), key=lambda k: results[k]['sharpe_ratio'])
    
    return {
        'optimal_frequency': optimal,
        'results': results,
        'sharpe_improvement': float(results[optimal]['sharpe_ratio'] - results['monthly']['sharpe_ratio'])
    }


def check_threshold_rebalance(
    current_weights: Dict[str, float],
    target_weights: Dict[str, float],
    threshold: float = 0.05
) -> Dict[str, Any]:
    """
    Check if rebalancing is needed based on drift threshold.
    """
    drifts = {}
    max_drift = 0
    
    all_assets = set(current_weights.keys()) | set(target_weights.keys())
    
    for asset in all_assets:
        current = current_weights.get(asset, 0)
        target = target_weights.get(asset, 0)
        drift = abs(current - target)
        drifts[asset] = float(drift)
        max_drift = max(max_drift, drift)
    
    needs_rebalance = max_drift > threshold
    assets_to_trade = [a for a, d in drifts.items() if d > threshold / 2]
    
    return {
        'needs_rebalance': needs_rebalance,
        'max_drift': float(max_drift),
        'threshold': threshold,
        'drifts': drifts,
        'assets_to_trade': assets_to_trade,
        'avg_drift': float(np.mean(list(drifts.values())))
    }


# ============================================================================
# TAX-LOSS HARVESTING
# ============================================================================

def find_tax_loss_opportunities(
    positions: Dict[str, Dict],
    short_term_rate: float = 0.37,
    long_term_rate: float = 0.20
) -> Dict[str, Any]:
    """
    Identify tax-loss harvesting opportunities.
    """
    today = datetime.now()
    opportunities = []
    total_harvestable_loss = 0
    potential_tax_savings = 0
    
    for ticker, pos in positions.items():
        cost_basis = pos.get('cost_basis', 0)
        current_value = pos.get('current_value', 0)
        gain_loss = current_value - cost_basis
        
        if gain_loss >= 0:
            continue
        
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
            tax_rate = short_term_rate
        
        loss_amount = abs(gain_loss)
        tax_savings = loss_amount * tax_rate
        
        opportunities.append({
            'ticker': ticker,
            'loss_amount': float(loss_amount),
            'cost_basis': float(cost_basis),
            'current_value': float(current_value),
            'holding_days': holding_days,
            'is_long_term': is_long_term,
            'tax_rate': tax_rate,
            'tax_savings': float(tax_savings),
            'repurchase_date': (today + timedelta(days=30)).strftime('%Y-%m-%d')
        })
        
        total_harvestable_loss += loss_amount
        potential_tax_savings += tax_savings
    
    opportunities.sort(key=lambda x: x['tax_savings'], reverse=True)
    
    return {
        'opportunities': opportunities,
        'total_harvestable_loss': float(total_harvestable_loss),
        'potential_tax_savings': float(potential_tax_savings),
        'n_opportunities': len(opportunities),
        'max_annual_deduction': 3000
    }


# ============================================================================
# PORTFOLIO VAR
# ============================================================================

def calculate_portfolio_var(
    returns: pd.DataFrame,
    weights: np.ndarray,
    confidence: float = 0.95
) -> Dict[str, float]:
    """
    Calculate portfolio VaR using covariance matrix.
    """
    cov_matrix = returns.cov() * 252
    port_vol = np.sqrt(weights.T @ cov_matrix.values @ weights)
    var = norm.ppf(1 - confidence) * port_vol / np.sqrt(252)
    
    return {
        'portfolio_var': float(var),
        'portfolio_volatility': float(port_vol)
    }


def calculate_marginal_var(
    returns: pd.DataFrame,
    weights: np.ndarray,
    confidence: float = 0.95
) -> Dict[str, Any]:
    """
    Calculate marginal VaR contribution by asset.
    """
    cov_matrix = returns.cov()
    port_var = np.sqrt(weights.T @ cov_matrix.values @ weights)
    marginal = (cov_matrix.values @ weights) / port_var
    contrib = weights * marginal
    normalized = contrib / contrib.sum()
    
    contributions = dict(zip(returns.columns, normalized.tolist()))
    
    return {
        'contributions': contributions,
        'portfolio_var': float(port_var),
        'largest_contributor': max(contributions.keys(), key=lambda k: contributions[k]),
        'smallest_contributor': min(contributions.keys(), key=lambda k: contributions[k])
    }


# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================

def analyze_correlations(
    returns: pd.DataFrame,
    crash_threshold: float = -0.02
) -> Dict[str, Any]:
    """
    Analyze correlations including crash vs normal periods.
    """
    market_proxy = returns.mean(axis=1)
    crash_days = market_proxy < crash_threshold
    
    normal_corr = returns[~crash_days].corr()
    crash_corr = returns[crash_days].corr() if crash_days.sum() > 10 else normal_corr
    
    n = len(returns.columns)
    
    def avg_corr(corr_matrix):
        total = 0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total += corr_matrix.iloc[i, j]
                count += 1
        return total / count if count > 0 else 0
    
    return {
        'correlation_matrix': {col: normal_corr[col].to_dict() for col in normal_corr.columns},
        'avg_normal_correlation': float(avg_corr(normal_corr)),
        'avg_crash_correlation': float(avg_corr(crash_corr)),
        'crash_days_count': int(crash_days.sum()),
        'total_days': len(returns)
    }
