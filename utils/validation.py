"""
Data Validation Module
======================
Ticker Validation | Data Quality | Error Handling | Retry Logic

Author: Stock Risk App | Feb 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
from functools import wraps

logger = logging.getLogger(__name__)

# ============================================================================
# TICKER VALIDATION
# ============================================================================

def validate_ticker_robust(ticker: str) -> Dict[str, Any]:
    """
    Comprehensive ticker validation with detailed feedback.
    
    Returns:
        dict with keys:
        - valid: bool - whether ticker is valid
        - error: str or None - error message if invalid
        - info: dict - ticker info if valid
        - warnings: list - any warnings about the ticker
    """
    import yfinance as yf
    
    result = {
        'valid': False,
        'error': None,
        'info': {},
        'warnings': []
    }
    
    # Clean ticker
    ticker = ticker.strip().upper()
    
    if not ticker:
        result['error'] = "Ticker symbol is empty"
        return result
    
    # Check for invalid characters
    if not all(c.isalnum() or c in '-^.' for c in ticker):
        result['error'] = f"Invalid characters in ticker: {ticker}"
        return result
    
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Check if we got valid data
        if not info:
            result['error'] = f"No data found for {ticker}"
            return result
        
        # Check for essential fields
        price_fields = ['regularMarketPrice', 'currentPrice', 'previousClose', 'open']
        has_price = any(info.get(field) is not None for field in price_fields)
        
        if not has_price:
            result['error'] = f"{ticker} appears to be invalid or delisted"
            return result
        
        # Check quote type
        quote_type = info.get('quoteType', 'UNKNOWN')
        if quote_type == 'NONE':
            result['error'] = f"{ticker} is not a valid security"
            return result
        
        # Add warnings for edge cases
        if info.get('marketState') == 'CLOSED':
            result['warnings'].append("Market is currently closed")
        
        if quote_type == 'CRYPTOCURRENCY':
            result['warnings'].append("Cryptocurrency - trades 24/7")
        
        if info.get('delisted'):
            result['error'] = f"{ticker} has been delisted"
            return result
        
        # Check for low volume (potentially illiquid)
        avg_volume = info.get('averageVolume', 0)
        if avg_volume and avg_volume < 10000:
            result['warnings'].append(f"Low liquidity (avg volume: {avg_volume:,})")
        
        result['valid'] = True
        result['info'] = info
        
    except Exception as e:
        result['error'] = f"Error validating {ticker}: {str(e)}"
        logger.error(f"Ticker validation error: {e}")
    
    return result

def validate_multiple_tickers(tickers: List[str]) -> Dict[str, Dict]:
    """
    Validate multiple tickers at once.
    
    Returns:
        Dictionary mapping ticker to validation result
    """
    results = {}
    for ticker in tickers:
        results[ticker] = validate_ticker_robust(ticker)
    return results

# ============================================================================
# RETRY LOGIC
# ============================================================================

def fetch_with_retry(
    fetch_func: Callable,
    *args,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    **kwargs
) -> Any:
    """
    Fetch data with exponential backoff retry.
    
    Args:
        fetch_func: Function to call
        *args: Arguments to pass to function
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries (seconds)
        backoff_factor: Multiplier for delay after each retry
        **kwargs: Keyword arguments to pass to function
    
    Returns:
        Result from fetch_func or empty DataFrame on failure
    """
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            result = fetch_func(*args, **kwargs)
            
            # Check if result is valid (not empty)
            if isinstance(result, pd.DataFrame) and result.empty:
                if attempt < max_retries - 1:
                    delay = initial_delay * (backoff_factor ** attempt)
                    logger.warning(f"Empty result, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue
            
            return result
            
        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                delay = initial_delay * (backoff_factor ** attempt)
                logger.warning(f"Fetch failed: {e}. Retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                logger.error(f"Fetch failed after {max_retries} attempts: {e}")
    
    # Return empty DataFrame on complete failure
    if last_exception:
        st.error(f"Failed after {max_retries} attempts: {last_exception}")
    
    return pd.DataFrame()

def retry_decorator(max_retries: int = 3, delay: float = 1.0):
    """
    Decorator for adding retry logic to functions.
    
    Usage:
        @retry_decorator(max_retries=3)
        def fetch_data(ticker):
            return yf.download(ticker)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return fetch_with_retry(func, *args, max_retries=max_retries, 
                                   initial_delay=delay, **kwargs)
        return wrapper
    return decorator

# ============================================================================
# DATA QUALITY VALIDATION
# ============================================================================

def validate_returns_data(returns: pd.Series, min_observations: int = 30) -> Dict[str, Any]:
    """
    Validate returns data for analysis suitability.
    
    Args:
        returns: Returns series
        min_observations: Minimum required observations
    
    Returns:
        Dictionary with validation results and warnings
    """
    result = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    # Check length
    if len(returns) < min_observations:
        result['errors'].append(f"Insufficient data: {len(returns)} observations (need {min_observations}+)")
        result['valid'] = False
    
    # Check for NaN/Inf
    nan_count = returns.isna().sum()
    inf_count = np.isinf(returns).sum()
    
    if nan_count > 0:
        nan_pct = nan_count / len(returns) * 100
        if nan_pct > 10:
            result['warnings'].append(f"{nan_pct:.1f}% missing values")
        if nan_pct > 50:
            result['errors'].append(f"Too many missing values: {nan_pct:.1f}%")
            result['valid'] = False
    
    if inf_count > 0:
        result['warnings'].append(f"{inf_count} infinite values detected")
    
    # Statistical checks
    clean_returns = returns.dropna()
    if len(clean_returns) > 0:
        skewness = clean_returns.skew()
        kurtosis = clean_returns.kurtosis()
        
        result['stats'] = {
            'n_observations': len(returns),
            'n_valid': len(clean_returns),
            'mean': float(clean_returns.mean()),
            'std': float(clean_returns.std()),
            'skewness': float(skewness),
            'kurtosis': float(kurtosis),
            'min': float(clean_returns.min()),
            'max': float(clean_returns.max())
        }
        
        # Check for extreme skewness
        if abs(skewness) > 5:
            result['warnings'].append(f"Extreme skewness: {skewness:.2f}")
        
        # Check for extreme kurtosis (fat tails)
        if kurtosis > 10:
            result['warnings'].append(f"Heavy tails detected (kurtosis: {kurtosis:.2f})")
        
        # Check for suspiciously low variance
        if clean_returns.std() < 0.0001:
            result['warnings'].append("Extremely low variance - check data quality")
    
    return result

def validate_portfolio_weights(weights: Dict[str, float]) -> Dict[str, Any]:
    """
    Validate portfolio weights.
    
    Args:
        weights: Dictionary of ticker -> weight (percentage)
    
    Returns:
        Dictionary with validation results
    """
    result = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'normalized_weights': {}
    }
    
    if not weights:
        result['errors'].append("No weights provided")
        result['valid'] = False
        return result
    
    total_weight = sum(weights.values())
    
    # Check total weight
    if abs(total_weight - 100) > 0.1:
        result['warnings'].append(f"Weights sum to {total_weight:.1f}% (should be 100%)")
    
    # Check individual weights
    for ticker, weight in weights.items():
        if weight < 0:
            result['errors'].append(f"Negative weight for {ticker}: {weight}%")
            result['valid'] = False
        elif weight > 100:
            result['errors'].append(f"Weight exceeds 100% for {ticker}: {weight}%")
            result['valid'] = False
        elif weight == 0:
            result['warnings'].append(f"Zero weight for {ticker}")
    
    # Normalize weights
    if total_weight > 0:
        result['normalized_weights'] = {t: w / total_weight for t, w in weights.items()}
    
    return result

# ============================================================================
# DATA CLEANING
# ============================================================================

def handle_missing_data(
    prices: pd.DataFrame,
    method: str = 'ffill',
    max_gap: int = 3,
    warn: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Handle missing data in price series.
    
    Args:
        prices: DataFrame with price data
        method: Fill method ('ffill', 'bfill', 'interpolate', 'drop')
        max_gap: Maximum gap to fill (in days)
        warn: Whether to show warnings
    
    Returns:
        Tuple of (cleaned DataFrame, report dictionary)
    """
    report = {
        'original_rows': len(prices),
        'missing_by_column': {},
        'filled_values': 0,
        'dropped_rows': 0
    }
    
    # Calculate missing data by column
    for col in prices.columns:
        missing = prices[col].isna().sum()
        if missing > 0:
            pct = missing / len(prices) * 100
            report['missing_by_column'][col] = {'count': missing, 'pct': pct}
            if warn and pct > 5:
                st.warning(f"{col}: {pct:.1f}% missing data")
    
    # Apply filling strategy
    if method == 'ffill':
        filled = prices.fillna(method='ffill', limit=max_gap)
    elif method == 'bfill':
        filled = prices.fillna(method='bfill', limit=max_gap)
    elif method == 'interpolate':
        filled = prices.interpolate(method='linear', limit=max_gap)
    else:
        filled = prices.copy()
    
    # Count filled values
    report['filled_values'] = int((prices.isna().sum() - filled.isna().sum()).sum())
    
    # Drop remaining NaN rows
    original_len = len(filled)
    filled = filled.dropna()
    report['dropped_rows'] = original_len - len(filled)
    report['final_rows'] = len(filled)
    
    return filled, report

def detect_outliers(
    returns: pd.Series,
    method: str = 'zscore',
    threshold: float = 5.0,
    show_details: bool = True
) -> Tuple[pd.Series, Dict]:
    """
    Detect and optionally winsorize outliers.
    
    Args:
        returns: Returns series
        method: Detection method ('zscore', 'iqr', 'mad')
        threshold: Threshold for outlier detection
        show_details: Whether to display outlier details
    
    Returns:
        Tuple of (returns with outlier flags, outlier info)
    """
    outlier_info = {
        'method': method,
        'threshold': threshold,
        'outliers': [],
        'outlier_count': 0,
        'outlier_dates': []
    }
    
    clean_returns = returns.dropna()
    
    if method == 'zscore':
        z_scores = np.abs((clean_returns - clean_returns.mean()) / clean_returns.std())
        outlier_mask = z_scores > threshold
    elif method == 'iqr':
        q1 = clean_returns.quantile(0.25)
        q3 = clean_returns.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        outlier_mask = (clean_returns < lower) | (clean_returns > upper)
    elif method == 'mad':
        # Median Absolute Deviation
        median = clean_returns.median()
        mad = np.median(np.abs(clean_returns - median))
        outlier_mask = np.abs(clean_returns - median) / mad > threshold
    else:
        outlier_mask = pd.Series(False, index=clean_returns.index)
    
    outliers = clean_returns[outlier_mask]
    outlier_info['outlier_count'] = len(outliers)
    outlier_info['outlier_dates'] = list(outliers.index)
    outlier_info['outliers'] = list(outliers.values)
    
    if show_details and len(outliers) > 0:
        st.warning(f"{len(outliers)} outlier(s) detected:")
        for date, ret in outliers.head(5).items():
            st.caption(f"  {date.strftime('%Y-%m-%d')}: {ret:.2%}")
        if len(outliers) > 5:
            st.caption(f"  ... and {len(outliers) - 5} more")
    
    return outlier_mask, outlier_info

def winsorize_returns(
    returns: pd.Series,
    lower_pct: float = 0.01,
    upper_pct: float = 0.99
) -> pd.Series:
    """
    Winsorize returns by clipping extreme values.
    
    Args:
        returns: Returns series
        lower_pct: Lower percentile for clipping
        upper_pct: Upper percentile for clipping
    
    Returns:
        Winsorized returns
    """
    lower = returns.quantile(lower_pct)
    upper = returns.quantile(upper_pct)
    return returns.clip(lower=lower, upper=upper)

# ============================================================================
# DATA FRESHNESS & ADJUSTMENTS
# ============================================================================

def verify_data_adjustments(ticker: str, prices: pd.Series) -> Dict[str, Any]:
    """
    Check for potential stock splits or dividend issues.
    
    Args:
        ticker: Ticker symbol
        prices: Price series
    
    Returns:
        Dictionary with adjustment analysis
    """
    result = {
        'potential_splits': [],
        'large_moves': [],
        'adjustment_needed': False
    }
    
    if len(prices) < 2:
        return result
    
    returns = prices.pct_change().dropna()
    
    # Detect potential splits (>50% single-day moves)
    large_moves = returns[abs(returns) > 0.5]
    
    for date, ret in large_moves.items():
        move_info = {
            'date': date,
            'return': float(ret),
            'type': 'split' if abs(ret) > 0.9 else 'large_move'
        }
        result['large_moves'].append(move_info)
        
        if abs(ret) > 0.9:
            result['potential_splits'].append(move_info)
    
    result['adjustment_needed'] = len(result['potential_splits']) > 0
    
    return result

def check_data_freshness(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Check how fresh the data is.
    
    Args:
        data: DataFrame with DatetimeIndex
    
    Returns:
        Dictionary with freshness information
    """
    result = {
        'last_date': None,
        'days_old': None,
        'status': 'unknown',
        'message': ''
    }
    
    if data.empty:
        result['message'] = "No data available"
        return result
    
    last_date = data.index[-1]
    result['last_date'] = last_date
    
    # Calculate days since last data point
    now = datetime.now()
    if hasattr(last_date, 'to_pydatetime'):
        last_date = last_date.to_pydatetime()
    
    # Handle timezone-aware datetimes
    if hasattr(last_date, 'tzinfo') and last_date.tzinfo:
        last_date = last_date.replace(tzinfo=None)
    
    days_old = (now - last_date).days
    result['days_old'] = days_old
    
    # Determine status
    if days_old <= 1:
        result['status'] = 'current'
        result['message'] = f"✓ Data current (last: {last_date.strftime('%Y-%m-%d')})"
    elif days_old <= 3:
        result['status'] = 'recent'
        result['message'] = f"Data is {days_old} days old (weekends/holidays)"
    elif days_old <= 7:
        result['status'] = 'stale'
        result['message'] = f"Data is {days_old} days old"
    else:
        result['status'] = 'old'
        result['message'] = f"Data is {days_old} days old - may be outdated"
    
    return result

# ============================================================================
# DATAVALIDATOR CLASS
# ============================================================================

class DataValidator:
    """
    Comprehensive data validation utility class.
    
    Usage:
        validator = DataValidator()
        
        # Validate ticker
        is_valid, info = validator.validate_ticker("AAPL")
        
        # Validate and clean data
        clean_data, report = validator.validate_and_clean(df)
    """
    
    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode
        self.validation_history = []
    
    def validate_ticker(self, ticker: str) -> Tuple[bool, Dict]:
        """Validate a single ticker."""
        result = validate_ticker_robust(ticker)
        self.validation_history.append({
            'type': 'ticker',
            'input': ticker,
            'result': result,
            'timestamp': datetime.now()
        })
        return result['valid'], result
    
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, Dict]:
        """Validate price/returns data."""
        report = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'freshness': None,
            'missing_data': None
        }
        
        if data.empty:
            report['valid'] = False
            report['errors'].append("Empty dataset")
            return False, report
        
        # Check freshness
        report['freshness'] = check_data_freshness(data)
        if report['freshness']['status'] == 'old':
            report['warnings'].append(report['freshness']['message'])
        
        # Check for missing data
        missing_pct = data.isna().sum() / len(data) * 100
        report['missing_data'] = missing_pct.to_dict() if hasattr(missing_pct, 'to_dict') else {}
        
        for col, pct in report['missing_data'].items():
            if pct > 10:
                report['warnings'].append(f"{col}: {pct:.1f}% missing")
            if pct > 50:
                report['errors'].append(f"{col}: {pct:.1f}% missing (too high)")
                report['valid'] = False
        
        self.validation_history.append({
            'type': 'data',
            'shape': data.shape,
            'result': report,
            'timestamp': datetime.now()
        })
        
        return report['valid'], report
    
    def validate_and_clean(
        self,
        data: pd.DataFrame,
        fill_method: str = 'ffill',
        remove_outliers: bool = False
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Validate data and clean if necessary.
        
        Args:
            data: Input DataFrame
            fill_method: Method for filling missing values
            remove_outliers: Whether to winsorize outliers
        
        Returns:
            Tuple of (cleaned DataFrame, full report)
        """
        full_report = {
            'validation': {},
            'cleaning': {},
            'outliers': {}
        }
        
        # Validate
        is_valid, val_report = self.validate_data(data)
        full_report['validation'] = val_report
        
        if not is_valid and self.strict_mode:
            return data, full_report
        
        # Clean missing data
        cleaned, clean_report = handle_missing_data(data, method=fill_method, warn=False)
        full_report['cleaning'] = clean_report
        
        # Handle outliers if requested
        if remove_outliers and len(cleaned) > 0:
            # For each numeric column, winsorize
            for col in cleaned.select_dtypes(include=[np.number]).columns:
                cleaned[col] = winsorize_returns(cleaned[col])
            full_report['outliers']['applied'] = True
        
        return cleaned, full_report
    
    def get_validation_summary(self) -> pd.DataFrame:
        """Get summary of all validations performed."""
        if not self.validation_history:
            return pd.DataFrame()
        
        data = []
        for v in self.validation_history:
            data.append({
                'Type': v['type'],
                'Timestamp': v['timestamp'],
                'Valid': v['result'].get('valid', True),
                'Errors': len(v['result'].get('errors', [])),
                'Warnings': len(v['result'].get('warnings', []))
            })
        
        return pd.DataFrame(data)
