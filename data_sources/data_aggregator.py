"""
Data Aggregator - Multi-Source Data Layer
==========================================
Combines multiple data providers with fallback, validation, and caching.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime, timedelta
import hashlib
import json
import os

from .base_provider import BaseDataProvider
from .yahoo_provider import YahooProvider
from .alpha_vantage_provider import AlphaVantageProvider


class DataAggregator:
    """
    Unified data layer that combines multiple providers with:
    - Automatic fallback when primary source fails
    - Data validation and cross-checking
    - Caching to reduce API calls
    - Rate limit management
    """
    
    def __init__(self, alpha_vantage_key: str = ""):
        self.logger = logging.getLogger(__name__)
        
        # Initialize providers
        self.providers = {
            'yahoo': YahooProvider(),
            'alpha_vantage': AlphaVantageProvider(api_key=alpha_vantage_key)
        }
        
        # Default provider order (primary first)
        self.provider_order = ['yahoo', 'alpha_vantage']
        
        # Cache settings
        self.cache_enabled = True
        self.cache_dir = os.path.join(os.path.dirname(__file__), '..', 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_ttl_minutes = 60
        
        # Statistics
        self.stats = {
            'requests': 0,
            'cache_hits': 0,
            'fallbacks': 0,
            'failures': 0
        }
    
    def _get_cache_key(self, ticker: str, start: str, end: str, data_type: str = "historical") -> str:
        """Generate cache key."""
        key_str = f"{ticker}_{start}_{end}_{data_type}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> str:
        """Get cache file path."""
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")
    
    def _read_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Read from cache if valid."""
        if not self.cache_enabled:
            return None
        
        cache_path = self._get_cache_path(cache_key)
        meta_path = cache_path + ".meta"
        
        if not os.path.exists(cache_path) or not os.path.exists(meta_path):
            return None
        
        try:
            # Check cache age
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            
            cached_time = datetime.fromisoformat(meta['timestamp'])
            if datetime.now() - cached_time > timedelta(minutes=self.cache_ttl_minutes):
                return None
            
            # Read cached data
            df = pd.read_pickle(cache_path)
            self.stats['cache_hits'] += 1
            return df
            
        except Exception as e:
            self.logger.warning(f"Cache read error: {e}")
            return None
    
    def _write_cache(self, cache_key: str, data: pd.DataFrame):
        """Write to cache."""
        if not self.cache_enabled or data.empty:
            return
        
        try:
            cache_path = self._get_cache_path(cache_key)
            meta_path = cache_path + ".meta"
            
            data.to_pickle(cache_path)
            
            with open(meta_path, 'w') as f:
                json.dump({'timestamp': datetime.now().isoformat()}, f)
                
        except Exception as e:
            self.logger.warning(f"Cache write error: {e}")
    
    def fetch_historical(
        self, 
        ticker: str, 
        start: str, 
        end: str, 
        interval: str = "1d",
        preferred_provider: str = None
    ) -> Tuple[pd.DataFrame, str]:
        """
        Fetch historical data with fallback.
        
        Args:
            ticker: Stock symbol
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            interval: Data interval
            preferred_provider: Force specific provider (optional)
        
        Returns:
            Tuple of (DataFrame, source_name)
        """
        self.stats['requests'] += 1
        
        # Check cache first
        cache_key = self._get_cache_key(ticker, start, end)
        cached_data = self._read_cache(cache_key)
        if cached_data is not None:
            return cached_data, 'cache'
        
        # Determine provider order
        if preferred_provider and preferred_provider in self.providers:
            providers_to_try = [preferred_provider] + [p for p in self.provider_order if p != preferred_provider]
        else:
            providers_to_try = self.provider_order
        
        # Try each provider
        for provider_name in providers_to_try:
            provider = self.providers.get(provider_name)
            if not provider:
                continue
            
            try:
                # Skip Alpha Vantage if no API key
                if provider_name == 'alpha_vantage' and not provider.api_key:
                    continue
                
                data = provider.fetch_historical(ticker, start, end, interval)
                
                if not data.empty:
                    self._write_cache(cache_key, data)
                    
                    if provider_name != providers_to_try[0]:
                        self.stats['fallbacks'] += 1
                        self.logger.info(f"Fallback to {provider_name} for {ticker}")
                    
                    return data, provider_name
                    
            except Exception as e:
                self.logger.warning(f"{provider_name} failed for {ticker}: {e}")
                continue
        
        self.stats['failures'] += 1
        self.logger.error(f"All providers failed for {ticker}")
        return pd.DataFrame(), 'none'
    
    def fetch_with_validation(
        self, 
        ticker: str, 
        start: str, 
        end: str,
        tolerance: float = 0.02
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Fetch from multiple sources and validate data consistency.
        
        Compares prices across sources and flags discrepancies.
        
        Args:
            ticker: Stock symbol
            start: Start date
            end: End date
            tolerance: Maximum allowed price difference (2% default)
        
        Returns:
            Tuple of (best_data, validation_report)
        """
        results = {}
        
        for provider_name, provider in self.providers.items():
            if provider_name == 'alpha_vantage' and not provider.api_key:
                continue
            
            try:
                data = provider.fetch_historical(ticker, start, end)
                if not data.empty:
                    results[provider_name] = data
            except Exception:
                continue
        
        if not results:
            return pd.DataFrame(), {'error': 'No data from any provider'}
        
        if len(results) == 1:
            provider_name, data = list(results.items())[0]
            return data, {'sources': [provider_name], 'validated': False}
        
        # Cross-validate if multiple sources
        validation_report = {
            'sources': list(results.keys()),
            'validated': True,
            'discrepancies': []
        }
        
        # Compare close prices
        provider_names = list(results.keys())
        df1 = results[provider_names[0]]
        df2 = results[provider_names[1]]
        
        # Align dates
        common_dates = df1.index.intersection(df2.index)
        
        if len(common_dates) > 0:
            close1 = df1.loc[common_dates, 'Close']
            close2 = df2.loc[common_dates, 'Close']
            
            # Calculate percentage difference
            pct_diff = abs((close1 - close2) / close1)
            avg_diff = pct_diff.mean()
            max_diff = pct_diff.max()
            
            validation_report['avg_difference'] = float(avg_diff)
            validation_report['max_difference'] = float(max_diff)
            
            if avg_diff > tolerance:
                validation_report['discrepancies'].append({
                    'type': 'price_mismatch',
                    'message': f"Average price difference {avg_diff:.2%} exceeds tolerance {tolerance:.2%}",
                    'sources': provider_names
                })
        
        # Return primary source data
        return results[provider_names[0]], validation_report
    
    def fetch_multiple(
        self, 
        tickers: List[str], 
        start: str, 
        end: str, 
        interval: str = "1d"
    ) -> Tuple[pd.DataFrame, str]:
        """
        Fetch data for multiple tickers.
        
        Uses batch fetching when available.
        """
        # Try Yahoo first (supports batch)
        yahoo = self.providers.get('yahoo')
        if yahoo:
            try:
                data = yahoo.fetch_multiple(tickers, start, end, interval)
                if not data.empty:
                    return data, 'yahoo'
            except Exception as e:
                self.logger.warning(f"Yahoo batch fetch failed: {e}")
        
        # Fallback to individual fetches
        all_data = {}
        source = 'mixed'
        
        for ticker in tickers:
            df, src = self.fetch_historical(ticker, start, end, interval)
            if not df.empty:
                all_data[ticker] = df
        
        if not all_data:
            return pd.DataFrame(), 'none'
        
        # Combine into multi-level columns
        combined = pd.concat(all_data, axis=1)
        return combined, source
    
    def fetch_info(self, ticker: str, merge_sources: bool = True) -> Dict:
        """
        Fetch company info, optionally merging from multiple sources.
        
        Args:
            ticker: Stock symbol
            merge_sources: If True, combine data from multiple providers
        
        Returns:
            Dictionary with company information
        """
        info_results = {}
        
        for provider_name in self.provider_order:
            provider = self.providers.get(provider_name)
            if not provider:
                continue
            
            if provider_name == 'alpha_vantage' and not provider.api_key:
                continue
            
            try:
                info = provider.fetch_info(ticker)
                if info and 'error' not in info:
                    info_results[provider_name] = info
                    if not merge_sources:
                        return info
            except Exception as e:
                self.logger.warning(f"{provider_name} info failed for {ticker}: {e}")
                continue
        
        if not info_results:
            return {'ticker': ticker, 'error': 'No info available'}
        
        if not merge_sources or len(info_results) == 1:
            return list(info_results.values())[0]
        
        # Merge info from multiple sources
        merged = {}
        for provider_name, info in info_results.items():
            for key, value in info.items():
                # Prefer non-None values
                if key not in merged or merged[key] is None:
                    merged[key] = value
        
        merged['sources'] = list(info_results.keys())
        return merged
    
    def validate_ticker(self, ticker: str) -> bool:
        """Check if ticker is valid in any provider."""
        for provider_name, provider in self.providers.items():
            if provider_name == 'alpha_vantage' and not provider.api_key:
                continue
            
            try:
                if provider.validate_ticker(ticker):
                    return True
            except Exception:
                continue
        
        return False
    
    def get_available_providers(self) -> List[str]:
        """Get list of currently available providers."""
        available = []
        for name, provider in self.providers.items():
            if name == 'alpha_vantage' and not provider.api_key:
                continue
            try:
                if provider.is_available():
                    available.append(name)
            except Exception:
                continue
        return available
    
    def get_stats(self) -> Dict:
        """Get data fetching statistics."""
        return {
            **self.stats,
            'cache_hit_rate': self.stats['cache_hits'] / max(1, self.stats['requests']),
            'fallback_rate': self.stats['fallbacks'] / max(1, self.stats['requests']),
            'failure_rate': self.stats['failures'] / max(1, self.stats['requests'])
        }
    
    def clear_cache(self):
        """Clear all cached data."""
        import glob
        cache_files = glob.glob(os.path.join(self.cache_dir, "*"))
        for f in cache_files:
            try:
                os.remove(f)
            except Exception:
                pass
        self.logger.info("Cache cleared")
    
    def set_provider_order(self, order: List[str]):
        """Set provider priority order."""
        valid_order = [p for p in order if p in self.providers]
        if valid_order:
            self.provider_order = valid_order
