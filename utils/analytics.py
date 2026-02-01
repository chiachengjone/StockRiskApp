"""
Enhanced Analytics Module
=========================
VaR Backtesting | Regime Detection | Time-Varying Beta | CoVaR | Rolling Analysis

Author: Stock Risk App | Feb 2026
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm, chi2
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture
from typing import Dict, List, Optional, Tuple, Any
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# ============================================================================
# VAR BACKTESTING
# ============================================================================

def backtest_var_kupiec(
    returns: pd.Series,
    var_series: pd.Series = None,
    conf_level: float = 0.95,
    window: int = 252
) -> Dict[str, Any]:
    """
    Kupiec POF (Proportion of Failures) Test for VaR.
    
    Tests if VaR violations are consistent with confidence level.
    
    H0: VaR model is correctly specified (violation rate = 1 - conf_level)
    
    Args:
        returns: Actual returns series
        var_series: Pre-computed VaR (negative values). If None, computed internally.
        conf_level: Confidence level (e.g., 0.95)
        window: Rolling window for VaR calculation
    
    Returns:
        Dictionary with test results, statistics, and interpretation
    """
    result = {
        'test': 'Kupiec POF Test',
        'conf_level': conf_level,
        'window': window,
        'violations': 0,
        'total_observations': 0,
        'violation_rate': 0.0,
        'expected_rate': 1 - conf_level,
        'lr_statistic': 0.0,
        'p_value': 0.0,
        'conclusion': '',
        'model_status': '',
        'violation_dates': [],
        'var_predictions': None,
        'actual_returns': None
    }
    
    if len(returns) < window + 20:
        result['conclusion'] = 'Insufficient data for backtest'
        return result
    
    # Calculate rolling VaR if not provided
    if var_series is None:
        var_predictions = []
        
        for i in range(window, len(returns)):
            historical_window = returns.iloc[i-window:i]
            var = np.percentile(historical_window, (1 - conf_level) * 100)
            var_predictions.append(var)
        
        var_series = pd.Series(var_predictions, index=returns.index[window:])
    
    # Align returns and VaR
    common_idx = returns.index.intersection(var_series.index)
    aligned_returns = returns.loc[common_idx]
    aligned_var = var_series.loc[common_idx]
    
    # Count violations (when actual return < VaR)
    violations = aligned_returns < aligned_var
    n_violations = violations.sum()
    n_total = len(aligned_returns)
    
    result['violations'] = int(n_violations)
    result['total_observations'] = n_total
    result['violation_rate'] = n_violations / n_total if n_total > 0 else 0
    result['violation_dates'] = list(aligned_returns[violations].index)
    result['var_predictions'] = aligned_var
    result['actual_returns'] = aligned_returns
    
    # Kupiec LR test statistic
    p_expected = 1 - conf_level  # Expected violation rate
    p_actual = result['violation_rate']
    x = n_violations
    n = n_total
    
    # Handle edge cases
    if p_actual == 0:
        p_actual = 0.0001
    if p_actual == 1:
        p_actual = 0.9999
    
    # Log-likelihood ratio
    try:
        lr_num = (1 - p_expected) ** (n - x) * p_expected ** x
        lr_den = (1 - p_actual) ** (n - x) * p_actual ** x
        
        if lr_num > 0 and lr_den > 0:
            lr_stat = -2 * np.log(lr_num / lr_den)
        else:
            lr_stat = 0
        
        result['lr_statistic'] = float(lr_stat)
        result['p_value'] = float(1 - chi2.cdf(lr_stat, df=1))
        
    except Exception as e:
        logger.error(f"LR calculation error: {e}")
        result['lr_statistic'] = 0
        result['p_value'] = 1
    
    # Interpretation
    alpha = 0.05  # Test significance level
    
    if result['p_value'] < alpha:
        result['conclusion'] = f"Reject H0 (p={result['p_value']:.4f} < {alpha})"
        result['model_status'] = '❌ VaR model inadequate'
    else:
        result['conclusion'] = f"Fail to reject H0 (p={result['p_value']:.4f} >= {alpha})"
        result['model_status'] = '✓ VaR model adequate'
    
    # Additional assessment based on violation ratio
    ratio = result['violation_rate'] / result['expected_rate'] if result['expected_rate'] > 0 else 0
    
    if ratio < 0.8:
        result['assessment'] = 'Conservative (too few violations)'
    elif ratio > 1.2:
        result['assessment'] = 'Aggressive (too many violations)'
    else:
        result['assessment'] = 'Well-calibrated'
    
    return result

def backtest_var_christoffersen(
    returns: pd.Series,
    var_series: pd.Series,
    conf_level: float = 0.95
) -> Dict[str, Any]:
    """
    Christoffersen Conditional Coverage Test.
    
    Tests both unconditional coverage (Kupiec) AND independence of violations.
    
    Args:
        returns: Actual returns
        var_series: VaR predictions
        conf_level: Confidence level
    
    Returns:
        Dictionary with test results
    """
    # Align data
    common_idx = returns.index.intersection(var_series.index)
    aligned_returns = returns.loc[common_idx]
    aligned_var = var_series.loc[common_idx]
    
    # Violation indicator (1 = violation, 0 = no violation)
    violations = (aligned_returns < aligned_var).astype(int).values
    
    n = len(violations)
    
    # Count transitions
    n00 = n01 = n10 = n11 = 0
    
    for i in range(1, n):
        if violations[i-1] == 0 and violations[i] == 0:
            n00 += 1
        elif violations[i-1] == 0 and violations[i] == 1:
            n01 += 1
        elif violations[i-1] == 1 and violations[i] == 0:
            n10 += 1
        else:
            n11 += 1
    
    # Calculate probabilities
    p01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    p11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    p = (n01 + n11) / n  # Overall violation rate
    
    # Independence test statistic
    try:
        # Likelihood under independence (violations are IID)
        if p > 0 and p < 1:
            l_ind = (1 - p) ** (n00 + n10) * p ** (n01 + n11)
        else:
            l_ind = 1e-10
        
        # Likelihood under dependence (Markov process)
        if 0 < p01 < 1 and 0 < p11 < 1:
            l_dep = ((1 - p01) ** n00 * p01 ** n01 * 
                    (1 - p11) ** n10 * p11 ** n11)
        else:
            l_dep = 1e-10
        
        lr_ind = -2 * np.log(l_ind / l_dep) if l_ind > 0 and l_dep > 0 else 0
        p_value_ind = 1 - chi2.cdf(lr_ind, df=1)
        
    except Exception as e:
        logger.error(f"Independence test error: {e}")
        lr_ind = 0
        p_value_ind = 1
    
    # Unconditional coverage (Kupiec)
    kupiec = backtest_var_kupiec(returns, var_series, conf_level)
    
    # Combined test (conditional coverage)
    lr_cc = kupiec['lr_statistic'] + lr_ind
    p_value_cc = 1 - chi2.cdf(lr_cc, df=2)
    
    return {
        'test': 'Christoffersen Conditional Coverage',
        'unconditional_lr': kupiec['lr_statistic'],
        'unconditional_p': kupiec['p_value'],
        'independence_lr': float(lr_ind),
        'independence_p': float(p_value_ind),
        'conditional_lr': float(lr_cc),
        'conditional_p': float(p_value_cc),
        'p01': float(p01),  # P(violation | no previous violation)
        'p11': float(p11),  # P(violation | previous violation)
        'independence_status': '✓ Independent' if p_value_ind >= 0.05 else '❌ Dependent (clustering)',
        'overall_status': '✓ Model adequate' if p_value_cc >= 0.05 else '❌ Model inadequate'
    }

# ============================================================================
# REGIME DETECTION
# ============================================================================

def regime_detection(
    returns: pd.Series,
    n_regimes: int = 3,
    features: List[str] = None
) -> Dict[str, Any]:
    """
    Detect market regimes using Gaussian Mixture Model.
    
    Identifies Bull, Bear, and Sideways regimes based on:
    - Return levels
    - Volatility levels
    - Trend characteristics
    
    Args:
        returns: Returns series
        n_regimes: Number of regimes to detect (2-4)
        features: Additional features to include
    
    Returns:
        Dictionary with regime classification and characteristics
    """
    result = {
        'n_regimes': n_regimes,
        'current_regime': None,
        'regime_labels': [],
        'regime_probs': None,
        'regime_characteristics': {},
        'regime_series': None,
        'transition_matrix': None
    }
    
    if len(returns) < 100:
        result['error'] = 'Insufficient data for regime detection'
        return result
    
    # Build feature matrix
    vol = returns.rolling(20).std() * np.sqrt(252)  # Annualized vol
    momentum = returns.rolling(20).sum()  # 20-day momentum
    
    # Align and create feature matrix
    features_df = pd.DataFrame({
        'returns': returns,
        'volatility': vol,
        'momentum': momentum
    }).dropna()
    
    X = features_df.values
    
    # Fit Gaussian Mixture Model
    gmm = GaussianMixture(
        n_components=n_regimes,
        covariance_type='full',
        random_state=42,
        n_init=10
    )
    
    try:
        regimes = gmm.fit_predict(X)
        probs = gmm.predict_proba(X)
    except Exception as e:
        result['error'] = f'GMM fitting failed: {e}'
        return result
    
    # Analyze regime characteristics
    regime_stats = {}
    for i in range(n_regimes):
        mask = regimes == i
        regime_returns = features_df.loc[mask, 'returns']
        regime_vol = features_df.loc[mask, 'volatility']
        
        regime_stats[i] = {
            'count': int(mask.sum()),
            'pct_time': float(mask.sum() / len(regimes)),
            'avg_return': float(regime_returns.mean() * 252),  # Annualized
            'avg_volatility': float(regime_vol.mean()),
            'sharpe': float(regime_returns.mean() * 252 / regime_vol.mean()) if regime_vol.mean() > 0 else 0
        }
    
    # Label regimes based on characteristics
    sorted_regimes = sorted(regime_stats.keys(), 
                           key=lambda x: regime_stats[x]['avg_return'])
    
    if n_regimes == 2:
        regime_names = {sorted_regimes[0]: 'Bear', sorted_regimes[1]: 'Bull'}
    elif n_regimes == 3:
        regime_names = {
            sorted_regimes[0]: 'Bear',
            sorted_regimes[1]: 'Sideways',
            sorted_regimes[2]: 'Bull'
        }
    else:
        regime_names = {i: f'Regime_{i+1}' for i in range(n_regimes)}
    
    # Map regimes to names
    regime_labels = [regime_names[r] for r in regimes]
    
    # Current regime
    current_regime = regime_names[regimes[-1]]
    current_prob = probs[-1, regimes[-1]]
    
    # Calculate transition matrix
    transition_counts = np.zeros((n_regimes, n_regimes))
    for i in range(1, len(regimes)):
        transition_counts[regimes[i-1], regimes[i]] += 1
    
    # Normalize to probabilities
    row_sums = transition_counts.sum(axis=1, keepdims=True)
    transition_matrix = np.divide(transition_counts, row_sums, 
                                 where=row_sums != 0,
                                 out=np.zeros_like(transition_counts))
    
    # Build result
    result['current_regime'] = current_regime
    result['current_regime_prob'] = float(current_prob)
    result['regime_labels'] = regime_labels
    result['regime_probs'] = probs
    result['regime_series'] = pd.Series(regime_labels, index=features_df.index)
    result['regime_characteristics'] = {
        regime_names[i]: regime_stats[i] for i in range(n_regimes)
    }
    result['transition_matrix'] = pd.DataFrame(
        transition_matrix,
        index=[regime_names[i] for i in range(n_regimes)],
        columns=[regime_names[i] for i in range(n_regimes)]
    )
    
    # Regime persistence (expected duration)
    for i in range(n_regimes):
        name = regime_names[i]
        p_stay = transition_matrix[i, i]
        expected_duration = 1 / (1 - p_stay) if p_stay < 1 else float('inf')
        result['regime_characteristics'][name]['expected_duration_days'] = expected_duration
    
    return result

# ============================================================================
# TIME-VARYING BETA
# ============================================================================

def time_varying_beta(
    stock_returns: pd.Series,
    benchmark_returns: pd.Series,
    method: str = 'rolling',
    window: int = 63,
    decay: float = 0.94
) -> Dict[str, Any]:
    """
    Calculate time-varying beta using different methods.
    
    Methods:
    - rolling: Simple rolling window OLS
    - ewma: Exponentially weighted
    - kalman: Kalman filter (simplified)
    
    Args:
        stock_returns: Stock returns
        benchmark_returns: Benchmark returns
        method: Estimation method
        window: Window size for rolling methods
        decay: Decay factor for EWMA
    
    Returns:
        Dictionary with beta series and statistics
    """
    # Align returns
    aligned = pd.DataFrame({
        'stock': stock_returns,
        'bench': benchmark_returns
    }).dropna()
    
    if len(aligned) < window + 10:
        return {'error': 'Insufficient data for time-varying beta'}
    
    betas = []
    r_squared = []
    dates = []
    
    if method == 'rolling':
        # Simple rolling OLS
        for i in range(window, len(aligned)):
            window_data = aligned.iloc[i-window:i]
            X = window_data['bench'].values.reshape(-1, 1)
            y = window_data['stock'].values
            
            reg = LinearRegression().fit(X, y)
            betas.append(reg.coef_[0])
            r_squared.append(reg.score(X, y))
            dates.append(aligned.index[i])
    
    elif method == 'ewma':
        # Exponentially weighted covariance/variance
        cov = aligned['stock'].ewm(span=window).cov(aligned['bench'])
        var = aligned['bench'].ewm(span=window).var()
        
        ewma_beta = cov / var
        ewma_beta = ewma_beta.dropna()
        
        betas = ewma_beta.values.tolist()
        dates = ewma_beta.index.tolist()
        r_squared = [None] * len(betas)
    
    elif method == 'kalman':
        # Simplified Kalman filter (random walk)
        beta_t = 1.0  # Initial beta
        P = 1.0  # Initial variance
        Q = 0.01  # Process noise
        R = 0.1  # Observation noise
        
        for i in range(len(aligned)):
            y = aligned['stock'].iloc[i]
            x = aligned['bench'].iloc[i]
            
            # Prediction step
            beta_pred = beta_t
            P_pred = P + Q
            
            # Update step
            K = P_pred * x / (x**2 * P_pred + R)
            beta_t = beta_pred + K * (y - x * beta_pred)
            P = (1 - K * x) * P_pred
            
            betas.append(beta_t)
            dates.append(aligned.index[i])
        
        r_squared = [None] * len(betas)
    
    beta_series = pd.Series(betas, index=dates)
    
    # Statistics
    current_beta = betas[-1]
    avg_beta = np.mean(betas)
    std_beta = np.std(betas)
    min_beta = np.min(betas)
    max_beta = np.max(betas)
    
    # Beta regime
    if current_beta > avg_beta + std_beta:
        beta_regime = 'High (aggressive)'
    elif current_beta < avg_beta - std_beta:
        beta_regime = 'Low (defensive)'
    else:
        beta_regime = 'Normal'
    
    return {
        'method': method,
        'window': window,
        'beta_series': beta_series,
        'r_squared_series': pd.Series(r_squared, index=dates) if r_squared[0] is not None else None,
        'current_beta': float(current_beta),
        'average_beta': float(avg_beta),
        'std_beta': float(std_beta),
        'min_beta': float(min_beta),
        'max_beta': float(max_beta),
        'beta_regime': beta_regime,
        'beta_change_30d': float(betas[-1] - betas[-31]) if len(betas) > 30 else None
    }

# ============================================================================
# COVAR (CONDITIONAL VAR)
# ============================================================================

def covar_systemic_risk(
    asset_returns: pd.Series,
    system_returns: pd.Series,
    quantile: float = 0.05
) -> Dict[str, Any]:
    """
    Calculate Conditional VaR (CoVaR) for systemic risk.
    
    CoVaR measures the VaR of the system conditional on an asset being in distress.
    ΔCoVaR = CoVaR - VaR (marginal contribution to systemic risk)
    
    Args:
        asset_returns: Individual asset returns
        system_returns: System/market returns (e.g., S&P 500)
        quantile: Distress quantile (e.g., 0.05 for bottom 5%)
    
    Returns:
        Dictionary with CoVaR metrics and interpretation
    """
    # Align data
    aligned = pd.DataFrame({
        'asset': asset_returns,
        'system': system_returns
    }).dropna()
    
    if len(aligned) < 100:
        return {'error': 'Insufficient data for CoVaR calculation'}
    
    # Find when asset is in distress (below quantile)
    asset_threshold = aligned['asset'].quantile(quantile)
    distress_mask = aligned['asset'] <= asset_threshold
    normal_mask = ~distress_mask
    
    # Unconditional system VaR
    system_var = aligned['system'].quantile(quantile)
    
    # Conditional VaR (system VaR when asset in distress)
    system_distress = aligned.loc[distress_mask, 'system']
    covar = system_distress.quantile(quantile) if len(system_distress) > 10 else system_var
    
    # Conditional VaR when asset is normal (for comparison)
    system_normal = aligned.loc[normal_mask, 'system']
    covar_normal = system_normal.quantile(quantile) if len(system_normal) > 10 else system_var
    
    # Delta CoVaR (contribution to systemic risk)
    delta_covar = covar - system_var
    
    # Systemic risk contribution ratio
    covar_ratio = abs(covar / system_var) if system_var != 0 else 1
    
    # Correlation during stress vs normal times
    stress_corr = aligned.loc[distress_mask, 'asset'].corr(aligned.loc[distress_mask, 'system'])
    normal_corr = aligned.loc[normal_mask, 'asset'].corr(aligned.loc[normal_mask, 'system'])
    
    # Interpretation
    if abs(delta_covar) < abs(system_var) * 0.1:
        risk_contribution = 'Low (minimal systemic impact)'
    elif abs(delta_covar) < abs(system_var) * 0.3:
        risk_contribution = 'Moderate (some systemic impact)'
    else:
        risk_contribution = 'High (significant systemic risk)'
    
    return {
        'quantile': quantile,
        'asset_distress_threshold': float(asset_threshold),
        'system_var': float(system_var),
        'covar': float(covar),
        'covar_normal': float(covar_normal),
        'delta_covar': float(delta_covar),
        'covar_ratio': float(covar_ratio),
        'stress_observations': int(distress_mask.sum()),
        'stress_correlation': float(stress_corr) if not np.isnan(stress_corr) else None,
        'normal_correlation': float(normal_corr) if not np.isnan(normal_corr) else None,
        'correlation_increase': float(stress_corr - normal_corr) if not np.isnan(stress_corr) else None,
        'risk_contribution': risk_contribution
    }

# ============================================================================
# ROLLING CORRELATION ANALYSIS
# ============================================================================

def rolling_correlation_breakdown(
    returns_df: pd.DataFrame,
    window: int = 63,
    crisis_threshold: float = -0.02
) -> Dict[str, Any]:
    """
    Analyze rolling correlations and crisis behavior.
    
    Args:
        returns_df: DataFrame with asset returns
        window: Rolling window for correlation
        crisis_threshold: Daily return threshold for crisis detection
    
    Returns:
        Dictionary with correlation analysis
    """
    result = {
        'rolling_correlations': {},
        'crisis_vs_normal': {},
        'average_correlation': None,
        'max_correlation_pair': None,
        'min_correlation_pair': None
    }
    
    tickers = returns_df.columns.tolist()
    n_assets = len(tickers)
    
    if n_assets < 2:
        result['error'] = 'Need at least 2 assets for correlation analysis'
        return result
    
    # Calculate rolling correlations for each pair
    all_correlations = []
    
    for i, t1 in enumerate(tickers):
        for j, t2 in enumerate(tickers):
            if i >= j:  # Skip diagonal and duplicates
                continue
            
            pair_key = f"{t1}_{t2}"
            rolling_corr = returns_df[t1].rolling(window).corr(returns_df[t2])
            result['rolling_correlations'][pair_key] = rolling_corr.dropna()
            
            avg_corr = rolling_corr.mean()
            all_correlations.append((pair_key, avg_corr))
    
    # Find max/min correlation pairs
    if all_correlations:
        sorted_corr = sorted(all_correlations, key=lambda x: x[1])
        result['min_correlation_pair'] = {
            'pair': sorted_corr[0][0],
            'correlation': float(sorted_corr[0][1])
        }
        result['max_correlation_pair'] = {
            'pair': sorted_corr[-1][0],
            'correlation': float(sorted_corr[-1][1])
        }
        result['average_correlation'] = float(np.mean([c[1] for c in all_correlations]))
    
    # Crisis vs normal correlation
    market_proxy = returns_df.mean(axis=1)
    crisis_days = market_proxy < crisis_threshold
    
    if crisis_days.sum() > 10:
        crisis_corr = returns_df[crisis_days].corr()
        normal_corr = returns_df[~crisis_days].corr()
        
        # Extract upper triangle averages
        def avg_upper_triangle(corr_matrix):
            n = len(corr_matrix)
            total = sum(corr_matrix.iloc[i, j] 
                       for i in range(n) for j in range(i+1, n))
            count = n * (n - 1) / 2
            return total / count if count > 0 else 0
        
        result['crisis_vs_normal'] = {
            'crisis_days': int(crisis_days.sum()),
            'normal_days': int((~crisis_days).sum()),
            'crisis_avg_correlation': float(avg_upper_triangle(crisis_corr)),
            'normal_avg_correlation': float(avg_upper_triangle(normal_corr)),
            'crisis_correlation_matrix': crisis_corr,
            'normal_correlation_matrix': normal_corr
        }
        
        # Correlation increase during crisis
        corr_increase = (result['crisis_vs_normal']['crisis_avg_correlation'] - 
                        result['crisis_vs_normal']['normal_avg_correlation'])
        result['crisis_vs_normal']['correlation_increase'] = float(corr_increase)
        
        if corr_increase > 0.1:
            result['crisis_vs_normal']['interpretation'] = \
                '⚠️ Correlations increase significantly during crises (diversification breakdown)'
        else:
            result['crisis_vs_normal']['interpretation'] = \
                '✓ Correlations stable during crises (diversification holds)'
    
    return result

# ============================================================================
# ENHANCED STRESS TESTING
# ============================================================================

def enhanced_stress_test(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    beta: float,
    scenarios: Dict[str, Dict] = None
) -> Dict[str, Any]:
    """
    Enhanced stress testing with multiple scenarios and Monte Carlo.
    
    Args:
        returns: Asset returns
        benchmark_returns: Benchmark returns
        beta: Asset beta
        scenarios: Custom scenarios (optional)
    
    Returns:
        Dictionary with stress test results
    """
    default_scenarios = {
        'Black Monday 1987': {'market_shock': -0.20, 'vol_mult': 3.0},
        'GFC 2008': {'market_shock': -0.50, 'vol_mult': 4.0},
        'COVID 2020': {'market_shock': -0.35, 'vol_mult': 5.0},
        'Tech Crash 2000': {'market_shock': -0.45, 'vol_mult': 2.0},
        'Mild Correction': {'market_shock': -0.10, 'vol_mult': 1.5},
        'Flash Crash': {'market_shock': -0.08, 'vol_mult': 6.0},
        'Interest Rate Shock': {'market_shock': -0.15, 'vol_mult': 2.0},
        'Stagflation': {'market_shock': -0.25, 'vol_mult': 2.5}
    }
    
    scenarios = scenarios or default_scenarios
    
    idio_vol = float(returns.std())
    
    results = {
        'scenarios': {},
        'summary': {
            'worst_case': None,
            'worst_scenario': None,
            'average_loss': None
        }
    }
    
    worst_loss = 0
    worst_scenario = None
    total_loss = 0
    
    for name, params in scenarios.items():
        market_shock = params['market_shock']
        vol_mult = params['vol_mult']
        
        # Systematic component
        systematic_loss = beta * market_shock
        
        # Idiosyncratic component (Monte Carlo)
        n_sims = 1000
        idio_shocks = np.random.normal(0, idio_vol * vol_mult, n_sims)
        total_returns = systematic_loss + idio_shocks
        
        # Statistics
        expected_loss = systematic_loss  # Expected = systematic (idio averages to 0)
        worst_mc = np.percentile(total_returns, 1)  # 1% worst case
        best_mc = np.percentile(total_returns, 99)  # 1% best case
        
        results['scenarios'][name] = {
            'market_shock': market_shock,
            'vol_multiplier': vol_mult,
            'systematic_impact': float(systematic_loss),
            'expected_return': float(expected_loss),
            'var_99': float(worst_mc),
            'best_case': float(best_mc),
            'std': float(np.std(total_returns))
        }
        
        if expected_loss < worst_loss:
            worst_loss = expected_loss
            worst_scenario = name
        
        total_loss += expected_loss
    
    results['summary']['worst_case'] = float(worst_loss)
    results['summary']['worst_scenario'] = worst_scenario
    results['summary']['average_loss'] = float(total_loss / len(scenarios))
    
    return results

# ============================================================================
# ANALYTICS ENGINE CLASS
# ============================================================================

class AnalyticsEngine:
    """
    Comprehensive analytics engine combining all enhanced features.
    
    Usage:
        engine = AnalyticsEngine()
        
        # VaR backtesting
        backtest = engine.backtest_var(returns)
        
        # Regime detection
        regimes = engine.detect_regimes(returns)
        
        # Time-varying beta
        beta = engine.rolling_beta(stock_returns, benchmark_returns)
    """
    
    def __init__(self):
        self.cache = {}
    
    def backtest_var(self, returns: pd.Series, conf_level: float = 0.95,
                    window: int = 252) -> Dict:
        """Run VaR backtest with Kupiec test."""
        return backtest_var_kupiec(returns, conf_level=conf_level, window=window)
    
    def backtest_var_full(self, returns: pd.Series, var_series: pd.Series,
                         conf_level: float = 0.95) -> Dict:
        """Run full VaR backtest with Christoffersen test."""
        return backtest_var_christoffersen(returns, var_series, conf_level)
    
    def detect_regimes(self, returns: pd.Series, n_regimes: int = 3) -> Dict:
        """Detect market regimes."""
        return regime_detection(returns, n_regimes)
    
    def rolling_beta(self, stock_returns: pd.Series, benchmark_returns: pd.Series,
                    method: str = 'rolling', window: int = 63) -> Dict:
        """Calculate time-varying beta."""
        return time_varying_beta(stock_returns, benchmark_returns, method, window)
    
    def systemic_risk(self, asset_returns: pd.Series, 
                     system_returns: pd.Series) -> Dict:
        """Calculate CoVaR systemic risk metrics."""
        return covar_systemic_risk(asset_returns, system_returns)
    
    def correlation_analysis(self, returns_df: pd.DataFrame, 
                            window: int = 63) -> Dict:
        """Analyze rolling correlations."""
        return rolling_correlation_breakdown(returns_df, window)
    
    def stress_test(self, returns: pd.Series, benchmark_returns: pd.Series,
                   beta: float) -> Dict:
        """Run enhanced stress tests."""
        return enhanced_stress_test(returns, benchmark_returns, beta)
    
    def full_analysis(self, returns: pd.Series, benchmark_returns: pd.Series) -> Dict:
        """Run comprehensive analysis."""
        from sklearn.linear_model import LinearRegression
        
        # Calculate beta first
        aligned = pd.DataFrame({
            'stock': returns,
            'bench': benchmark_returns
        }).dropna()
        
        if len(aligned) < 60:
            return {'error': 'Insufficient data'}
        
        X = aligned['bench'].values.reshape(-1, 1)
        y = aligned['stock'].values
        beta = LinearRegression().fit(X, y).coef_[0]
        
        return {
            'var_backtest': self.backtest_var(returns),
            'regimes': self.detect_regimes(returns),
            'time_varying_beta': self.rolling_beta(returns, benchmark_returns),
            'systemic_risk': self.systemic_risk(returns, benchmark_returns),
            'stress_test': self.stress_test(returns, benchmark_returns, beta)
        }
