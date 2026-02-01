"""
Performance Optimization Module
===============================
Caching | Lazy Loading | Progress Tracking | Performance Monitoring

Author: Stock Risk App | Feb 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import functools
from typing import Any, Callable, Optional, Dict, List
from datetime import datetime, timedelta
import hashlib
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# ENHANCED CACHING WITH TTL
# ============================================================================

def _hash_dataframe(df: pd.DataFrame) -> str:
    """Create hash of DataFrame for caching."""
    return hashlib.md5(pd.util.hash_pandas_object(df).values.tobytes()).hexdigest()

def _hash_series(s: pd.Series) -> str:
    """Create hash of Series for caching."""
    return hashlib.md5(pd.util.hash_pandas_object(s).values.tobytes()).hexdigest()

@st.cache_data(ttl=300, show_spinner=False)
def cached_fetch_data(ticker: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    """
    Cached data fetcher with 5-minute TTL.
    
    Significantly reduces API calls for repeated queries.
    """
    import yfinance as yf
    try:
        data = yf.download(ticker, start=start, end=end, 
                          auto_adjust=True, progress=False, interval=interval)
        if data.empty:
            logger.warning(f"No data returned for {ticker}")
            return pd.DataFrame()
        return data
    except Exception as e:
        logger.error(f"Error fetching {ticker}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def cached_fetch_info(ticker: str) -> dict:
    """
    Cached stock info fetcher with 1-hour TTL.
    
    Fundamentals don't change frequently.
    """
    import yfinance as yf
    try:
        info = yf.Ticker(ticker).info
        return info if info else {}
    except Exception as e:
        logger.error(f"Error fetching info for {ticker}: {e}")
        return {}

@st.cache_data(ttl=300, show_spinner=False)
def cached_fetch_multiple(tickers: tuple, start: str, end: str) -> pd.DataFrame:
    """
    Cached multi-ticker fetcher with 5-minute TTL.
    
    Note: tickers must be tuple for caching (lists aren't hashable).
    """
    import yfinance as yf
    try:
        data = yf.download(list(tickers), start=start, end=end, 
                          auto_adjust=True, progress=False)
        return data
    except Exception as e:
        logger.error(f"Error fetching multiple tickers: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=600, show_spinner=False)
def cached_garch(returns_hash: str, _returns: pd.Series) -> tuple:
    """
    Cached GARCH fitting with 10-minute TTL.
    
    GARCH fitting is expensive (~5-15 seconds), cache significantly improves UX.
    
    Args:
        returns_hash: Hash of returns for cache key
        _returns: Returns series (underscore prefix means Streamlit won't hash it)
    """
    from arch import arch_model
    try:
        returns_scaled = _returns * 100
        model = arch_model(returns_scaled, vol='Garch', p=1, q=1, mean='Zero')
        fitted = model.fit(disp='off')
        forecast = fitted.forecast(horizon=10)
        cond_vol = fitted.conditional_volatility / 100
        garch_vol_forecast = np.sqrt(forecast.variance.iloc[-1].values) / 100 * np.sqrt(252)
        return fitted, cond_vol, garch_vol_forecast
    except Exception as e:
        logger.error(f"GARCH fitting failed: {e}")
        return None, pd.Series(), np.array([])

def fit_garch_cached(returns: pd.Series) -> tuple:
    """Wrapper to call cached_garch with proper hashing."""
    returns_hash = _hash_series(returns)
    return cached_garch(returns_hash, returns)

@st.cache_data(ttl=300, show_spinner=False)
def cached_monte_carlo(returns_hash: str, mu: float, sigma: float, 
                       n_sims: int, horizon: int, seed: int = 42) -> np.ndarray:
    """
    Cached Monte Carlo simulation.
    
    Args:
        returns_hash: Hash for cache key
        mu: Mean return
        sigma: Standard deviation
        n_sims: Number of simulations
        horizon: Simulation horizon
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)
    sim_rets = np.random.normal(mu, sigma, (n_sims, horizon))
    return sim_rets.sum(axis=1)

def run_monte_carlo_cached(returns: pd.Series, n_sims: int = 10000, 
                           horizon: int = 10) -> np.ndarray:
    """Wrapper to call cached Monte Carlo."""
    returns_hash = _hash_series(returns)
    mu = float(returns.mean())
    sigma = float(returns.std())
    return cached_monte_carlo(returns_hash, mu, sigma, n_sims, horizon)

@st.cache_data(ttl=600, show_spinner=False)
def cached_optimization(returns_hash: str, _returns_df: pd.DataFrame) -> dict:
    """
    Cached portfolio optimization.
    
    Mean-variance optimization is computationally intensive.
    """
    from scipy.optimize import minimize
    from scipy.stats import norm
    
    n_assets = len(_returns_df.columns)
    mean_returns = _returns_df.mean() * 252
    cov_matrix = _returns_df.cov() * 252
    
    def neg_sharpe(weights):
        port_ret = np.dot(weights, mean_returns)
        port_vol = np.sqrt(weights.T @ cov_matrix @ weights)
        return -port_ret / port_vol if port_vol > 0 else 0
    
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = tuple((0, 1) for _ in range(n_assets))
    initial = np.array([1/n_assets] * n_assets)
    
    result = minimize(neg_sharpe, initial, method='SLSQP', 
                     bounds=bounds, constraints=constraints)
    
    opt_weights = result.x
    opt_ret = np.dot(opt_weights, mean_returns)
    opt_vol = np.sqrt(opt_weights.T @ cov_matrix @ opt_weights)
    
    return {
        'weights': opt_weights,
        'return': float(opt_ret),
        'volatility': float(opt_vol),
        'sharpe': float(opt_ret / opt_vol) if opt_vol > 0 else 0,
        'tickers': list(_returns_df.columns)
    }

def optimize_portfolio_cached(returns_df: pd.DataFrame) -> dict:
    """Wrapper for cached optimization."""
    returns_hash = _hash_dataframe(returns_df)
    return cached_optimization(returns_hash, returns_df)

# ============================================================================
# PROGRESS TRACKING
# ============================================================================

class ProgressTracker:
    """
    Enhanced progress tracking for long operations.
    
    Usage:
        tracker = ProgressTracker(total_steps=5, description="Analyzing portfolio")
        tracker.start()
        
        for ticker in tickers:
            # do work
            tracker.update(f"Processing {ticker}")
        
        tracker.complete()
    """
    
    def __init__(self, total_steps: int, description: str = "Processing"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.progress_bar = None
        self.status_text = None
        self.start_time = None
        
    def start(self):
        """Initialize progress tracking."""
        self.start_time = time.time()
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()
        self.status_text.text(f"{self.description}...")
        
    def update(self, message: str = None, increment: int = 1):
        """Update progress."""
        self.current_step += increment
        progress = min(self.current_step / self.total_steps, 1.0)
        
        if self.progress_bar:
            self.progress_bar.progress(progress)
        
        if self.status_text and message:
            elapsed = time.time() - self.start_time
            eta = (elapsed / self.current_step) * (self.total_steps - self.current_step) \
                  if self.current_step > 0 else 0
            self.status_text.text(f"{message} ({self.current_step}/{self.total_steps}) - ETA: {eta:.1f}s")
    
    def complete(self, message: str = "Complete!"):
        """Finish progress tracking."""
        if self.progress_bar:
            self.progress_bar.progress(1.0)
        if self.status_text:
            elapsed = time.time() - self.start_time
            self.status_text.text(f"{message} (took {elapsed:.1f}s)")
            time.sleep(0.5)
            self.status_text.empty()
            self.progress_bar.empty()

# ============================================================================
# LAZY LOADING
# ============================================================================

def lazy_load(func: Callable) -> Callable:
    """
    Decorator for lazy loading expensive computations.
    
    Only runs function when result is accessed in session state.
    
    Usage:
        @lazy_load
        def expensive_calculation():
            return heavy_computation()
        
        # In tab - only runs when tab is viewed
        with tab:
            if st.button("Run Analysis"):
                result = expensive_calculation()
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        cache_key = f"lazy_{func.__name__}_{hash(str(args))}_{hash(str(kwargs))}"
        
        if cache_key not in st.session_state:
            st.session_state[cache_key] = func(*args, **kwargs)
        
        return st.session_state[cache_key]
    
    return wrapper

def lazy_tab_content(tab_key: str, compute_func: Callable, *args, **kwargs):
    """
    Helper for lazy tab content loading.
    
    Args:
        tab_key: Unique key for the tab
        compute_func: Function to compute when tab is first viewed
        *args, **kwargs: Arguments to pass to compute_func
    
    Usage:
        with tab_advanced:
            result = lazy_tab_content("advanced_analysis", run_garch, returns)
    """
    state_key = f"tab_content_{tab_key}"
    
    if state_key not in st.session_state:
        with st.spinner("Computing..."):
            st.session_state[state_key] = compute_func(*args, **kwargs)
    
    return st.session_state[state_key]

# ============================================================================
# PERFORMANCE MONITORING
# ============================================================================

class PerformanceMonitor:
    """
    Track execution times for functions.
    
    Usage:
        monitor = PerformanceMonitor()
        
        with monitor.track("data_fetch"):
            data = fetch_data(ticker)
        
        monitor.report()
    """
    
    def __init__(self):
        self.timings: Dict[str, List[float]] = {}
        
    class TrackContext:
        def __init__(self, monitor: 'PerformanceMonitor', operation: str):
            self.monitor = monitor
            self.operation = operation
            self.start_time = None
            
        def __enter__(self):
            self.start_time = time.time()
            return self
            
        def __exit__(self, *args):
            elapsed = time.time() - self.start_time
            if self.operation not in self.monitor.timings:
                self.monitor.timings[self.operation] = []
            self.monitor.timings[self.operation].append(elapsed)
    
    def track(self, operation: str):
        """Context manager to track operation timing."""
        return self.TrackContext(self, operation)
    
    def report(self) -> pd.DataFrame:
        """Generate performance report."""
        data = []
        for op, times in self.timings.items():
            data.append({
                'Operation': op,
                'Calls': len(times),
                'Total (s)': sum(times),
                'Avg (s)': np.mean(times),
                'Max (s)': max(times),
                'Min (s)': min(times)
            })
        return pd.DataFrame(data)
    
    def clear(self):
        """Reset timings."""
        self.timings = {}

# Create singleton monitor
performance_monitor = PerformanceMonitor()

def timed(func: Callable) -> Callable:
    """
    Decorator to automatically time function execution.
    
    Usage:
        @timed
        def my_function():
            pass
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        
        if func.__name__ not in performance_monitor.timings:
            performance_monitor.timings[func.__name__] = []
        performance_monitor.timings[func.__name__].append(elapsed)
        
        return result
    return wrapper

# ============================================================================
# BATCH PROCESSING
# ============================================================================

def batch_fetch_data(tickers: List[str], start: str, end: str, 
                     max_parallel: int = 5) -> Dict[str, pd.DataFrame]:
    """
    Fetch data for multiple tickers with progress tracking.
    
    Args:
        tickers: List of ticker symbols
        start: Start date
        end: End date
        max_parallel: Max concurrent requests (not used currently, for future)
    
    Returns:
        Dictionary mapping ticker to DataFrame
    """
    results = {}
    tracker = ProgressTracker(len(tickers), "Fetching data")
    tracker.start()
    
    for ticker in tickers:
        try:
            data = cached_fetch_data(ticker, start, end)
            results[ticker] = data
        except Exception as e:
            logger.error(f"Failed to fetch {ticker}: {e}")
            results[ticker] = pd.DataFrame()
        
        tracker.update(f"Fetched {ticker}")
    
    tracker.complete(f"Fetched {len(results)} tickers")
    return results

# ============================================================================
# SESSION STATE HELPERS
# ============================================================================

def get_or_compute(key: str, compute_func: Callable, *args, **kwargs):
    """
    Get value from session state or compute and store it.
    
    Args:
        key: Session state key
        compute_func: Function to compute value if not cached
        *args, **kwargs: Arguments to pass to compute_func
    
    Returns:
        Cached or computed value
    """
    if key not in st.session_state:
        st.session_state[key] = compute_func(*args, **kwargs)
    return st.session_state[key]

def invalidate_cache(prefix: str = None):
    """
    Invalidate cached values in session state.
    
    Args:
        prefix: Only invalidate keys starting with this prefix
    """
    keys_to_delete = []
    for key in st.session_state:
        if prefix is None or key.startswith(prefix):
            keys_to_delete.append(key)
    
    for key in keys_to_delete:
        del st.session_state[key]
    
    logger.info(f"Invalidated {len(keys_to_delete)} cached values")
