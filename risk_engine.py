"""
Risk Engine - Classical Risk Models
====================================
GARCH(1,1) • EVT • Monte Carlo • Portfolio Optimization • Stress Testing

Enterprise-grade risk calculations extracted for modularity.
Author: Professional Risk Analytics | Jan 2026
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm, t, genpareto
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')

# Import new modules (optional - graceful fallback)
try:
    from data_sources import DataAggregator, YahooProvider, AlphaVantageProvider
    HAS_DATA_LAYER = True
except ImportError:
    HAS_DATA_LAYER = False

try:
    from config.settings import STRESS_SCENARIOS as CONFIG_SCENARIOS
except ImportError:
    CONFIG_SCENARIOS = None

# ============================================================================
# STRESS SCENARIOS
# ============================================================================
# Use config if available, otherwise fall back to defaults
if CONFIG_SCENARIOS:
    STRESS_SCENARIOS = CONFIG_SCENARIOS
else:
    STRESS_SCENARIOS = {
        "Black Monday 1987": {"market_shock": -0.20, "vol_multiplier": 3.0, "description": "Oct 19, 1987: -20% single day"},
        "Dot-com Crash 2000": {"market_shock": -0.45, "vol_multiplier": 2.0, "description": "2000-2002: Tech bubble burst"},
        "GFC 2008": {"market_shock": -0.50, "vol_multiplier": 4.0, "description": "2008: Lehman collapse, -50% peak-trough"},
        "COVID Crash 2020": {"market_shock": -0.35, "vol_multiplier": 5.0, "description": "Mar 2020: Fastest 30% drop ever"},
        "Mild Correction": {"market_shock": -0.10, "vol_multiplier": 1.5, "description": "Typical 10% pullback"},
        "Severe Bear": {"market_shock": -0.30, "vol_multiplier": 2.5, "description": "Extended bear market"},
    }

# ============================================================================
# DATA FETCHERS
# ============================================================================
@st.cache_data(ttl=300)
def fetch_data(ticker: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    """Yahoo Finance data fetcher."""
    try:
        data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False, interval=interval)
        if data.empty:
            return pd.DataFrame()
        return data
    except:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_info(ticker: str) -> dict:
    """Stock fundamentals."""
    try:
        info = yf.Ticker(ticker).info
        return info if info else {}
    except:
        return {}

@st.cache_data(ttl=300)
def fetch_multiple(tickers: list, start: str, end: str) -> pd.DataFrame:
    """Fetch multiple tickers at once."""
    try:
        data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
        return data
    except:
        return pd.DataFrame()

def validate_ticker(ticker: str) -> bool:
    """Check if ticker is valid."""
    try:
        info = yf.Ticker(ticker).info
        return info.get('regularMarketPrice') is not None or info.get('previousClose') is not None
    except:
        return False

# ============================================================================
# RETURN CALCULATIONS
# ============================================================================
def compute_returns(prices: pd.Series) -> pd.Series:
    """Compute log returns."""
    if isinstance(prices, pd.DataFrame):
        prices = prices.iloc[:, 0]
    return np.log(prices / prices.shift(1)).dropna()

def compute_metrics(returns: pd.Series, rf: float = 0.0) -> dict:
    """Comprehensive metrics suite."""
    if len(returns) < 20:
        return {}
    
    ann_ret = float(returns.mean() * 252)
    ann_vol = float(returns.std() * np.sqrt(252))
    
    # Drawdown
    cum_ret = (1 + returns).cumprod()
    rolling_max = cum_ret.expanding().max()
    drawdown = (cum_ret - rolling_max) / rolling_max
    max_dd = float(drawdown.min())
    
    # Risk-adjusted metrics
    sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else 0
    
    # Sortino (downside deviation)
    downside_returns = returns[returns < 0]
    downside_std = float(downside_returns.std() * np.sqrt(252)) if len(downside_returns) > 0 else ann_vol
    sortino = (ann_ret - rf) / downside_std if downside_std > 0 else 0
    
    # Calmar
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0
    
    return {
        'ann_ret': ann_ret,
        'ann_vol': ann_vol,
        'max_dd': max_dd,
        'sharpe': float(sharpe),
        'sortino': float(sortino),
        'calmar': float(calmar),
        'skew': float(returns.skew()),
        'kurtosis': float(returns.kurtosis()),
        'drawdown_series': drawdown,
        'returns': returns
    }

# ============================================================================
# VAR CALCULATIONS
# ============================================================================
def parametric_var(returns: pd.Series, horizon: int = 1, conf: float = 0.95, dist: str = 'normal') -> float:
    """Parametric VaR."""
    mu = float(returns.mean()) * horizon
    sigma = float(returns.std()) * np.sqrt(horizon)
    if dist == 'normal':
        return float(norm.ppf(1-conf, mu, sigma))
    else:
        df, loc, scale = t.fit(returns)
        return float(t.ppf(1-conf, df, loc * horizon, scale * np.sqrt(horizon)))

def historical_var(returns: pd.Series, horizon: int = 1, conf: float = 0.95) -> float:
    """Historical simulation VaR."""
    if horizon > 1:
        sim_rets = returns.rolling(horizon).sum().dropna()
    else:
        sim_rets = returns
    return float(np.percentile(sim_rets, 100*(1-conf)))

def cvar(returns: pd.Series, conf: float = 0.95) -> float:
    """Conditional VaR / Expected Shortfall."""
    var_threshold = np.percentile(returns, 100*(1-conf))
    tail_losses = returns[returns <= var_threshold]
    return float(tail_losses.mean()) if len(tail_losses) > 0 else 0.0

# ============================================================================
# ADVANCED MODELS
# ============================================================================
@st.cache_data
def fit_garch(_returns: pd.Series) -> tuple:
    """GARCH(1,1) model."""
    try:
        returns_scaled = _returns * 100
        model = arch_model(returns_scaled, vol='Garch', p=1, q=1, mean='Zero')
        fitted = model.fit(disp='off')
        forecast = fitted.forecast(horizon=10)
        cond_vol = fitted.conditional_volatility / 100
        garch_vol_forecast = np.sqrt(forecast.variance.iloc[-1].values) / 100 * np.sqrt(252)
        return fitted, cond_vol, garch_vol_forecast
    except:
        return None, pd.Series(), np.array([])

def evt_tail_risk(returns: pd.Series, threshold_pct: float = 0.05, conf: float = 0.99) -> dict:
    """Extreme Value Theory."""
    threshold = np.percentile(returns, 100*threshold_pct)
    losses = -returns[returns <= threshold]
    excesses = losses - (-threshold)
    
    if len(excesses) < 20:
        return {"error": "Insufficient tail data"}
    
    try:
        shape, loc, scale = genpareto.fit(excesses)
        gpd = genpareto(shape, loc, scale)
        tail_var = threshold + gpd.ppf(conf)
        
        return {
            'threshold': float(threshold),
            'shape': float(shape),
            'evt_var': float(-tail_var),
            'excesses_count': len(excesses)
        }
    except:
        return {"error": "EVT fitting failed"}

def compute_beta(stock_rets: pd.Series, bench_rets: pd.Series) -> tuple:
    """CAPM beta and alpha."""
    aligned = pd.DataFrame({'stock': stock_rets, 'bench': bench_rets}).dropna()
    if len(aligned) < 30:
        return np.nan, np.nan
    
    X = aligned['bench'].values.reshape(-1, 1)
    y = aligned['stock'].values
    reg = LinearRegression().fit(X, y)
    return float(reg.coef_[0]), float(reg.intercept_)

def mc_simulation(returns: pd.Series, n_sims: int = 10000, horizon: int = 10) -> np.ndarray:
    """Monte Carlo simulation."""
    mu = float(returns.mean())
    sigma = float(returns.std())
    sim_rets = np.random.normal(mu, sigma, (n_sims, horizon))
    return sim_rets.sum(axis=1)

# ============================================================================
# PORTFOLIO FUNCTIONS
# ============================================================================
def portfolio_returns(returns_df: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """Compute weighted portfolio returns."""
    return (returns_df * weights).sum(axis=1)

def portfolio_var(returns_df: pd.DataFrame, weights: np.ndarray, conf: float = 0.95) -> float:
    """Portfolio VaR using covariance matrix."""
    cov_matrix = returns_df.cov() * 252
    port_vol = np.sqrt(weights.T @ cov_matrix @ weights)
    return float(norm.ppf(1-conf) * port_vol / np.sqrt(252))

def marginal_var_contribution(returns_df: pd.DataFrame, weights: np.ndarray, conf: float = 0.95) -> pd.Series:
    """Risk contribution by asset."""
    cov_matrix = returns_df.cov()
    port_var = np.sqrt(weights.T @ cov_matrix @ weights)
    marginal = (cov_matrix @ weights) / port_var
    contrib = weights * marginal
    return pd.Series(contrib / contrib.sum(), index=returns_df.columns)

def optimize_portfolio(returns_df: pd.DataFrame, target_return: float = None) -> dict:
    """Mean-Variance Optimization."""
    n_assets = len(returns_df.columns)
    mean_returns = returns_df.mean() * 252
    cov_matrix = returns_df.cov() * 252
    
    def neg_sharpe(weights):
        port_ret = np.dot(weights, mean_returns)
        port_vol = np.sqrt(weights.T @ cov_matrix @ weights)
        return -port_ret / port_vol if port_vol > 0 else 0
    
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = tuple((0, 1) for _ in range(n_assets))
    initial = np.array([1/n_assets] * n_assets)
    
    result = minimize(neg_sharpe, initial, method='SLSQP', bounds=bounds, constraints=constraints)
    
    opt_weights = result.x
    opt_ret = np.dot(opt_weights, mean_returns)
    opt_vol = np.sqrt(opt_weights.T @ cov_matrix @ opt_weights)
    
    return {
        'weights': opt_weights,
        'return': float(opt_ret),
        'volatility': float(opt_vol),
        'sharpe': float(opt_ret / opt_vol) if opt_vol > 0 else 0
    }

def efficient_frontier(returns_df: pd.DataFrame, n_points: int = 50) -> pd.DataFrame:
    """Generate efficient frontier."""
    mean_returns = returns_df.mean() * 252
    cov_matrix = returns_df.cov() * 252
    n_assets = len(returns_df.columns)
    
    results = []
    target_returns = np.linspace(float(mean_returns.min()), float(mean_returns.max()), n_points)
    
    for target in target_returns:
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w, t=target: np.dot(w, mean_returns) - t}
        ]
        bounds = tuple((0, 1) for _ in range(n_assets))
        initial = np.array([1/n_assets] * n_assets)
        
        def port_vol(weights):
            return np.sqrt(weights.T @ cov_matrix @ weights)
        
        try:
            result = minimize(port_vol, initial, method='SLSQP', bounds=bounds, constraints=constraints)
            if result.success:
                results.append({'return': target, 'volatility': result.fun, 'sharpe': target / result.fun})
        except:
            pass
    
    return pd.DataFrame(results)

# ============================================================================
# STRESS TESTING
# ============================================================================
def stress_test_portfolio(returns_df: pd.DataFrame, weights: np.ndarray, betas: dict, 
                         market_shock: float, vol_multiplier: float = 1.0) -> dict:
    """Apply stress scenario to portfolio."""
    stressed_returns = {}
    for ticker in returns_df.columns:
        beta = betas.get(ticker, 1.0)
        idio_vol = float(returns_df[ticker].std())
        stressed_ret = beta * market_shock + idio_vol * vol_multiplier * np.random.randn()
        stressed_returns[ticker] = stressed_ret
    
    port_stressed_ret = sum(weights[i] * stressed_returns[list(returns_df.columns)[i]] 
                           for i in range(len(weights)))
    
    return {
        'individual': stressed_returns,
        'portfolio': port_stressed_ret
    }

# ============================================================================
# BACKTESTING
# ============================================================================
def backtest_strategy(returns: pd.Series, benchmark_returns: pd.Series) -> dict:
    """Simple buy-and-hold backtest vs benchmark."""
    aligned = pd.DataFrame({'strategy': returns, 'benchmark': benchmark_returns}).dropna()
    
    strat_cum = (1 + aligned['strategy']).cumprod()
    bench_cum = (1 + aligned['benchmark']).cumprod()
    
    strat_total = float(strat_cum.iloc[-1] - 1)
    bench_total = float(bench_cum.iloc[-1] - 1)
    
    strat_dd = (strat_cum / strat_cum.expanding().max() - 1).min()
    bench_dd = (bench_cum / bench_cum.expanding().max() - 1).min()
    
    excess_return = strat_total - bench_total
    
    return {
        'strategy_return': strat_total,
        'benchmark_return': bench_total,
        'excess_return': excess_return,
        'strategy_dd': float(strat_dd),
        'benchmark_dd': float(bench_dd),
        'strategy_cum': strat_cum,
        'benchmark_cum': bench_cum
    }

# ============================================================================
# ROLLING METRICS (NEW)
# ============================================================================
def rolling_volatility(returns: pd.Series, window: int = 21) -> pd.Series:
    """Rolling annualized volatility."""
    return returns.rolling(window).std() * np.sqrt(252)

def rolling_sharpe(returns: pd.Series, window: int = 63, rf: float = 0.0) -> pd.Series:
    """Rolling Sharpe ratio (default: quarterly window)."""
    rolling_ret = returns.rolling(window).mean() * 252
    rolling_vol = returns.rolling(window).std() * np.sqrt(252)
    return (rolling_ret - rf) / rolling_vol

def rolling_var(returns: pd.Series, window: int = 252, conf: float = 0.95) -> pd.Series:
    """Rolling historical VaR."""
    return returns.rolling(window).apply(lambda x: np.percentile(x, 100*(1-conf)), raw=True)

def rolling_beta(stock_rets: pd.Series, bench_rets: pd.Series, window: int = 63) -> pd.Series:
    """Rolling beta coefficient."""
    aligned = pd.DataFrame({'stock': stock_rets, 'bench': bench_rets}).dropna()
    
    def calc_beta(window_data):
        if len(window_data) < 20:
            return np.nan
        cov = np.cov(window_data['stock'], window_data['bench'])[0, 1]
        var = np.var(window_data['bench'])
        return cov / var if var > 0 else 0
    
    # Calculate rolling beta manually
    betas = []
    for i in range(len(aligned)):
        if i < window:
            betas.append(np.nan)
        else:
            window_data = aligned.iloc[i-window:i]
            cov = np.cov(window_data['stock'], window_data['bench'])[0, 1]
            var = np.var(window_data['bench'])
            betas.append(cov / var if var > 0 else 0)
    
    return pd.Series(betas, index=aligned.index)

def rolling_max_drawdown(returns: pd.Series, window: int = 252) -> pd.Series:
    """Rolling maximum drawdown."""
    cum_ret = (1 + returns).cumprod()
    
    def calc_dd(window_prices):
        running_max = np.maximum.accumulate(window_prices)
        drawdowns = (window_prices - running_max) / running_max
        return np.min(drawdowns)
    
    return cum_ret.rolling(window).apply(calc_dd, raw=True)

def get_rolling_metrics_df(returns: pd.Series, bench_returns: pd.Series = None) -> pd.DataFrame:
    """Get all rolling metrics as DataFrame."""
    df = pd.DataFrame(index=returns.index)
    
    df['rolling_vol_21d'] = rolling_volatility(returns, 21)
    df['rolling_vol_63d'] = rolling_volatility(returns, 63)
    df['rolling_sharpe'] = rolling_sharpe(returns, 63)
    df['rolling_var'] = rolling_var(returns, 252)
    df['rolling_max_dd'] = rolling_max_drawdown(returns, 252)
    
    if bench_returns is not None:
        df['rolling_beta'] = rolling_beta(returns, bench_returns, 63)
    
    return df

# ============================================================================
# ENHANCED STRESS TESTING (NEW)
# ============================================================================
def monte_carlo_stress(returns: pd.Series, n_sims: int = 10000, 
                       stress_factor: float = 2.0) -> dict:
    """Monte Carlo with stressed parameters."""
    mu = float(returns.mean())
    sigma = float(returns.std()) * stress_factor  # Stressed volatility
    
    # Simulate 10-day returns
    sim_rets = np.random.normal(mu, sigma, (n_sims, 10))
    cum_rets = sim_rets.sum(axis=1)
    
    return {
        'mean': float(np.mean(cum_rets)),
        'std': float(np.std(cum_rets)),
        'var_95': float(np.percentile(cum_rets, 5)),
        'var_99': float(np.percentile(cum_rets, 1)),
        'worst_case': float(np.min(cum_rets)),
        'best_case': float(np.max(cum_rets)),
        'prob_loss_10pct': float(np.mean(cum_rets < -0.10)),
        'distribution': cum_rets
    }

def custom_stress_scenario(returns_df: pd.DataFrame, weights: np.ndarray,
                          shocks: dict) -> dict:
    """Apply custom shocks to each asset."""
    results = {}
    portfolio_impact = 0
    
    for i, ticker in enumerate(returns_df.columns):
        shock = shocks.get(ticker, 0)
        results[ticker] = {
            'shock': shock,
            'weighted_impact': shock * weights[i]
        }
        portfolio_impact += shock * weights[i]
    
    return {
        'individual': results,
        'portfolio_impact': portfolio_impact,
        'portfolio_loss': portfolio_impact if portfolio_impact < 0 else 0
    }

def sector_stress_test(returns_df: pd.DataFrame, weights: np.ndarray,
                       sector_mapping: dict, sector_shocks: dict) -> dict:
    """Apply sector-based stress scenarios."""
    results = {}
    portfolio_impact = 0
    
    for i, ticker in enumerate(returns_df.columns):
        sector = sector_mapping.get(ticker, 'Unknown')
        shock = sector_shocks.get(sector, 0)
        
        results[ticker] = {
            'sector': sector,
            'shock': shock,
            'weighted_impact': shock * weights[i]
        }
        portfolio_impact += shock * weights[i]
    
    return {
        'by_asset': results,
        'portfolio_impact': portfolio_impact
    }

# ============================================================================
# CORRELATION ANALYSIS (NEW)
# ============================================================================
def rolling_correlation(returns_df: pd.DataFrame, window: int = 63) -> dict:
    """Rolling correlation between assets."""
    correlations = {}
    tickers = returns_df.columns.tolist()
    
    for i, t1 in enumerate(tickers):
        for t2 in tickers[i+1:]:
            key = f"{t1}_{t2}"
            correlations[key] = returns_df[t1].rolling(window).corr(returns_df[t2])
    
    return correlations

def correlation_breakdown(returns_df: pd.DataFrame, 
                         crash_threshold: float = -0.02) -> dict:
    """Compare correlations during crashes vs normal times."""
    # Identify crash days (market down more than threshold)
    market_proxy = returns_df.mean(axis=1)  # Simple average
    crash_days = market_proxy < crash_threshold
    
    normal_corr = returns_df[~crash_days].corr()
    crash_corr = returns_df[crash_days].corr()
    
    # Average correlation
    n = len(returns_df.columns)
    
    def avg_corr(corr_matrix):
        total = 0
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                total += corr_matrix.iloc[i, j]
                count += 1
        return total / count if count > 0 else 0
    
    return {
        'normal_correlation_matrix': normal_corr,
        'crash_correlation_matrix': crash_corr,
        'avg_normal_correlation': avg_corr(normal_corr),
        'avg_crash_correlation': avg_corr(crash_corr),
        'crash_days_count': int(crash_days.sum()),
        'total_days': len(returns_df)
    }

# ============================================================================
# DATA LAYER INTEGRATION (NEW)
# ============================================================================
def get_data_aggregator():
    """Get DataAggregator instance with fallback."""
    if HAS_DATA_LAYER:
        return DataAggregator()
    return None

def fetch_with_fallback(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch data using new data layer with fallback to direct yfinance."""
    aggregator = get_data_aggregator()
    
    if aggregator:
        return aggregator.fetch_historical(ticker, start, end)
    else:
        return fetch_data(ticker, start, end)

def fetch_fundamentals_enhanced(ticker: str) -> dict:
    """Fetch enhanced fundamentals using data layer."""
    aggregator = get_data_aggregator()
    
    if aggregator:
        return aggregator.fetch_info(ticker)
    else:
        return fetch_info(ticker)

