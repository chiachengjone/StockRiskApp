"""
Risk Calculation Service
========================
Core risk calculations ported from risk_engine.py

All mathematical operations maintain bit-for-bit parity with original.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, genpareto
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# STRESS SCENARIOS (from risk_engine.py)
# ============================================================================

STRESS_SCENARIOS = {
    # Historical Crises
    "2008 Financial Crisis": {"market_shock": -0.40, "vol_multiplier": 2.5},
    "2020 COVID Crash": {"market_shock": -0.34, "vol_multiplier": 3.0},
    "2000 Dot-com Bust": {"market_shock": -0.49, "vol_multiplier": 1.8},
    "1987 Black Monday": {"market_shock": -0.22, "vol_multiplier": 4.0},
    "2011 Euro Crisis": {"market_shock": -0.19, "vol_multiplier": 1.5},
    "2015 China Devaluation": {"market_shock": -0.12, "vol_multiplier": 1.3},
    "2018 Q4 Selloff": {"market_shock": -0.20, "vol_multiplier": 1.6},
    "2022 Rate Shock": {"market_shock": -0.25, "vol_multiplier": 1.4},
    
    # Macro-driven (with sector betas)
    "Inflation Surge (+3%)": {
        "market_shock": -0.08,
        "vol_multiplier": 1.3,
        "sector_betas": {
            "Technology": 1.4, "Healthcare": 0.8, "Financials": 1.1,
            "Consumer Discretionary": 1.2, "Consumer Staples": 0.7,
            "Energy": 0.9, "Utilities": 0.6, "Real Estate": 1.3,
            "Materials": 1.0, "Industrials": 1.1, "Communication Services": 1.2
        }
    },
    "Rate Hike (+100bp)": {
        "market_shock": -0.05,
        "vol_multiplier": 1.2,
        "sector_betas": {
            "Technology": 1.5, "Healthcare": 0.9, "Financials": 0.7,
            "Consumer Discretionary": 1.1, "Consumer Staples": 0.8,
            "Energy": 0.6, "Utilities": 1.4, "Real Estate": 1.6,
            "Materials": 0.8, "Industrials": 0.9, "Communication Services": 1.3
        }
    },
    "USD Strengthening (+10%)": {
        "market_shock": -0.06,
        "vol_multiplier": 1.1,
        "sector_betas": {
            "Technology": 1.3, "Healthcare": 1.0, "Financials": 0.9,
            "Consumer Discretionary": 1.1, "Consumer Staples": 0.8,
            "Energy": 0.7, "Utilities": 0.5, "Real Estate": 0.6,
            "Materials": 1.2, "Industrials": 1.1, "Communication Services": 1.0
        }
    },
    "Oil Spike (+50%)": {
        "market_shock": -0.04,
        "vol_multiplier": 1.4,
        "sector_betas": {
            "Technology": 0.6, "Healthcare": 0.7, "Financials": 0.8,
            "Consumer Discretionary": 1.3, "Consumer Staples": 0.9,
            "Energy": -1.5, "Utilities": 1.1, "Real Estate": 0.8,
            "Materials": 1.0, "Industrials": 1.2, "Communication Services": 0.7
        }
    }
}


# ============================================================================
# RETURN CALCULATIONS
# ============================================================================

def compute_returns(prices: pd.Series) -> pd.Series:
    """Compute daily log returns."""
    return np.log(prices / prices.shift(1)).dropna()


def compute_metrics(returns: pd.Series, rf: float = 0.0) -> Dict[str, float]:
    """
    Compute comprehensive risk metrics.
    
    Returns:
        Dictionary with annualized return, volatility, Sharpe, Sortino,
        max drawdown, Calmar, skewness, kurtosis
    """
    ann_ret = float(returns.mean() * 252)
    ann_vol = float(returns.std() * np.sqrt(252))
    
    # Sharpe ratio
    sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else 0.0
    
    # Sortino ratio (downside deviation)
    downside = returns[returns < 0]
    downside_std = float(downside.std() * np.sqrt(252)) if len(downside) > 0 else ann_vol
    sortino = (ann_ret - rf) / downside_std if downside_std > 0 else 0.0
    
    # Max drawdown
    cum_ret = (1 + returns).cumprod()
    running_max = cum_ret.expanding().max()
    drawdowns = (cum_ret - running_max) / running_max
    max_dd = float(drawdowns.min())
    
    # Calmar ratio
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0.0
    
    return {
        "annualized_return": ann_ret,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "max_drawdown": max_dd,
        "calmar_ratio": float(calmar),
        "skewness": float(returns.skew()),
        "kurtosis": float(returns.kurtosis())
    }


# ============================================================================
# VAR CALCULATIONS
# ============================================================================

def parametric_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Parametric VaR assuming normal distribution.
    
    VaR = μ + σ * Z(α)
    """
    mu = float(returns.mean())
    sigma = float(returns.std())
    z_score = norm.ppf(1 - confidence)
    return float(mu + sigma * z_score)


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Historical VaR using empirical distribution.
    
    Simply returns the (1-confidence) percentile of returns.
    """
    percentile = 100 * (1 - confidence)
    return float(np.percentile(returns, percentile))


def cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Conditional VaR (Expected Shortfall).
    
    Average of returns below VaR threshold.
    """
    var = historical_var(returns, confidence)
    tail_returns = returns[returns <= var]
    return float(tail_returns.mean()) if len(tail_returns) > 0 else var


def calculate_all_var(returns: pd.Series, confidence: float = 0.95) -> Dict[str, float]:
    """Calculate all VaR metrics."""
    return {
        "parametric": parametric_var(returns, confidence),
        "historical": historical_var(returns, confidence),
        "cvar": cvar(returns, confidence),
        "confidence": confidence
    }


# ============================================================================
# GARCH MODEL
# ============================================================================

def fit_garch(returns: pd.Series, p: int = 1, q: int = 1) -> Dict[str, Any]:
    """
    Fit GARCH(p,q) model and forecast volatility.
    
    Returns model parameters and volatility forecast.
    """
    try:
        from arch import arch_model
        
        # Scale returns to percentages for numerical stability
        returns_scaled = returns * 100
        
        model = arch_model(returns_scaled, vol='Garch', p=p, q=q, mean='Zero')
        fitted = model.fit(disp='off')
        
        # Extract parameters
        omega = float(fitted.params.get('omega', 0))
        alpha = float(fitted.params.get('alpha[1]', 0))
        beta = float(fitted.params.get('beta[1]', 0))
        
        # Forecast
        forecast = fitted.forecast(horizon=1)
        forecast_var = forecast.variance.iloc[-1].values[0]
        forecast_vol = np.sqrt(forecast_var) / 100  # Scale back
        
        # Current conditional volatility
        current_vol = np.sqrt(fitted.conditional_volatility.iloc[-1]) / 100
        
        # Annualized volatility
        ann_vol = forecast_vol * np.sqrt(252)
        
        return {
            "model_type": f"GARCH({p},{q})",
            "omega": omega,
            "alpha": alpha,
            "beta": beta,
            "persistence": alpha + beta,
            "current_volatility": float(current_vol),
            "forecast_volatility": float(forecast_vol),
            "annualized_volatility": float(ann_vol),
            "conditional_var": float(-forecast_vol * 1.645)
        }
    except Exception as e:
        logger.error(f"GARCH fitting failed: {e}")
        # Fallback to simple volatility estimate
        vol = float(returns.std())
        return {
            "model_type": "Simple",
            "omega": 0,
            "alpha": 0,
            "beta": 0,
            "persistence": 0,
            "current_volatility": vol,
            "forecast_volatility": vol,
            "annualized_volatility": vol * np.sqrt(252),
            "conditional_var": float(-vol * 1.645),
            "error": str(e)
        }


# ============================================================================
# EXTREME VALUE THEORY
# ============================================================================

def evt_tail_risk(returns: pd.Series, threshold_percentile: float = 0.95) -> Dict[str, Any]:
    """
    Extreme Value Theory analysis using GPD for tail risk.
    
    Fits Generalized Pareto Distribution to tail excesses.
    """
    try:
        losses = -returns
        threshold = float(np.percentile(losses, threshold_percentile * 100))
        excesses = losses[losses > threshold] - threshold
        
        if len(excesses) < 10:
            return {"error": "Insufficient tail observations for EVT"}
        
        # Fit GPD
        shape, loc, scale = genpareto.fit(excesses, floc=0)
        
        # EVT VaR at 99%
        n = len(losses)
        n_exceed = len(excesses)
        conf = 0.99
        
        tail_var = threshold + (scale / shape) * (
            ((n / n_exceed) * (1 - conf)) ** (-shape) - 1
        )
        
        return {
            "threshold": threshold,
            "shape_parameter": float(shape),
            "scale": float(scale),
            "evt_var": float(-tail_var),
            "excesses_count": len(excesses),
            "tail_index": float(-1 / shape) if shape != 0 else None
        }
    except Exception as e:
        logger.error(f"EVT fitting failed: {e}")
        return {"error": str(e)}


# ============================================================================
# MONTE CARLO SIMULATION
# ============================================================================

def monte_carlo_simulation(
    returns: pd.Series,
    n_simulations: int = 10000,
    horizon_days: int = 10
) -> Dict[str, Any]:
    """
    Monte Carlo simulation for portfolio returns.
    
    Simulates future returns using historical mean and volatility.
    """
    mu = float(returns.mean())
    sigma = float(returns.std())
    
    # Simulate daily returns over horizon
    sim_daily = np.random.normal(mu, sigma, (n_simulations, horizon_days))
    
    # Cumulative returns over horizon
    sim_cumulative = sim_daily.sum(axis=1)
    
    return {
        "n_simulations": n_simulations,
        "horizon_days": horizon_days,
        "mean_return": float(np.mean(sim_cumulative)),
        "std_return": float(np.std(sim_cumulative)),
        "var_95": float(np.percentile(sim_cumulative, 5)),
        "var_99": float(np.percentile(sim_cumulative, 1)),
        "worst_case": float(np.min(sim_cumulative)),
        "best_case": float(np.max(sim_cumulative)),
        "prob_loss_10pct": float(np.mean(sim_cumulative < -0.10)),
        "percentiles": {
            "1%": float(np.percentile(sim_cumulative, 1)),
            "5%": float(np.percentile(sim_cumulative, 5)),
            "10%": float(np.percentile(sim_cumulative, 10)),
            "25%": float(np.percentile(sim_cumulative, 25)),
            "50%": float(np.percentile(sim_cumulative, 50)),
            "75%": float(np.percentile(sim_cumulative, 75)),
            "90%": float(np.percentile(sim_cumulative, 90)),
            "95%": float(np.percentile(sim_cumulative, 95)),
            "99%": float(np.percentile(sim_cumulative, 99))
        }
    }


# ============================================================================
# STRESS TESTING
# ============================================================================

def run_stress_test(
    returns: pd.Series,
    scenario_name: str = None,
    custom_shock: float = None,
    ticker_sector: str = None
) -> Dict[str, Any]:
    """
    Run stress test using predefined or custom scenario.
    """
    # Use custom shock or get from scenario
    if custom_shock is not None:
        market_shock = custom_shock
        vol_multiplier = 1.5
        scenario_desc = f"Custom shock: {custom_shock*100:.1f}%"
    elif scenario_name and scenario_name in STRESS_SCENARIOS:
        scenario = STRESS_SCENARIOS[scenario_name]
        market_shock = scenario["market_shock"]
        vol_multiplier = scenario["vol_multiplier"]
        scenario_desc = scenario_name
        
        # Apply sector beta if available
        if "sector_betas" in scenario and ticker_sector:
            sector_beta = scenario["sector_betas"].get(ticker_sector, 1.0)
            market_shock *= sector_beta
    else:
        return {"error": f"Unknown scenario: {scenario_name}"}
    
    # Calculate beta (assume market proxy = historical returns)
    beta = 1.0  # Default beta
    
    # Stressed volatility
    historical_vol = float(returns.std())
    stressed_vol = historical_vol * vol_multiplier
    
    # Estimated loss
    estimated_loss = market_shock
    
    return {
        "scenario_name": scenario_desc,
        "market_shock": market_shock,
        "estimated_loss": estimated_loss,
        "beta": beta,
        "stressed_volatility": float(stressed_vol),
        "annualized_stressed_vol": float(stressed_vol * np.sqrt(252)),
        "scenario_description": scenario_desc
    }


def get_stress_scenarios_list() -> List[Dict[str, Any]]:
    """Get list of available stress scenarios."""
    scenarios = []
    for name, params in STRESS_SCENARIOS.items():
        scenarios.append({
            "name": name,
            "market_shock": params["market_shock"],
            "vol_multiplier": params["vol_multiplier"],
            "has_sector_betas": "sector_betas" in params
        })
    return scenarios


# ============================================================================
# ROLLING METRICS
# ============================================================================

def calculate_rolling_metrics(
    returns: pd.Series,
    window: int = 21
) -> Dict[str, List[Optional[float]]]:
    """
    Calculate rolling risk metrics over time.
    """
    n = len(returns)
    dates = returns.index.strftime('%Y-%m-%d').tolist()
    
    # Rolling volatility
    rolling_vol = returns.rolling(window).std() * np.sqrt(252)
    
    # Rolling Sharpe
    rolling_ret = returns.rolling(window).mean() * 252
    rolling_sharpe = rolling_ret / rolling_vol
    
    # Rolling VaR
    rolling_var = returns.rolling(window).apply(
        lambda x: np.percentile(x, 5), raw=True
    )
    
    # Rolling max drawdown
    cum_ret = (1 + returns).cumprod()
    
    def calc_dd(window_prices):
        running_max = np.maximum.accumulate(window_prices)
        drawdowns = (window_prices - running_max) / running_max
        return np.min(drawdowns)
    
    rolling_dd = cum_ret.rolling(window).apply(calc_dd, raw=True)
    
    return {
        "dates": dates,
        "rolling_volatility": [
            None if pd.isna(x) else float(x) for x in rolling_vol
        ],
        "rolling_sharpe": [
            None if pd.isna(x) else float(x) for x in rolling_sharpe
        ],
        "rolling_var": [
            None if pd.isna(x) else float(x) for x in rolling_var
        ],
        "rolling_max_drawdown": [
            None if pd.isna(x) else float(x) for x in rolling_dd
        ]
    }


# ============================================================================
# BETA CALCULATION
# ============================================================================

def compute_beta(
    stock_returns: pd.Series,
    benchmark_returns: pd.Series
) -> Tuple[float, float]:
    """
    Calculate CAPM beta and alpha.
    """
    aligned = pd.DataFrame({
        'stock': stock_returns,
        'bench': benchmark_returns
    }).dropna()
    
    if len(aligned) < 30:
        return np.nan, np.nan
    
    X = aligned['bench'].values.reshape(-1, 1)
    y = aligned['stock'].values
    
    reg = LinearRegression().fit(X, y)
    
    return float(reg.coef_[0]), float(reg.intercept_)
