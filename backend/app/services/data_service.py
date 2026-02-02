"""
Data Fetching Service
=====================
Multi-provider data fetching with caching.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
import hashlib
import json
import os
import time
from functools import lru_cache

from app.core.config import settings

logger = logging.getLogger(__name__)


class DataService:
    """
    Unified data service with caching and fallback.
    """
    
    def __init__(self):
        self.cache_dir = settings.CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_ttl = settings.CACHE_TTL_MINUTES * 60
    
    def _get_cache_key(self, ticker: str, start: str, end: str, data_type: str) -> str:
        """Generate cache key."""
        key_str = f"{ticker}_{start}_{end}_{data_type}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> str:
        """Get cache file path."""
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")
    
    def _is_cache_valid(self, cache_path: str) -> bool:
        """Check if cache is still valid."""
        if not os.path.exists(cache_path):
            return False
        
        file_time = os.path.getmtime(cache_path)
        return (datetime.now().timestamp() - file_time) < self.cache_ttl
    
    def _fetch_yahoo_direct(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch data directly from Yahoo Finance API (fallback for yfinance issues).
        """
        try:
            # Convert dates to Unix timestamps
            start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
            end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())
            
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
            params = {
                "period1": start_ts,
                "period2": end_ts,
                "interval": "1d",
                "events": "history"
            }
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            result = data.get("chart", {}).get("result", [])
            
            if not result:
                return pd.DataFrame()
            
            chart = result[0]
            timestamps = chart.get("timestamp", [])
            indicators = chart.get("indicators", {})
            quote = indicators.get("quote", [{}])[0]
            
            if not timestamps:
                return pd.DataFrame()
            
            df = pd.DataFrame({
                "Open": quote.get("open", []),
                "High": quote.get("high", []),
                "Low": quote.get("low", []),
                "Close": quote.get("close", []),
                "Volume": quote.get("volume", [])
            }, index=pd.to_datetime(timestamps, unit='s'))
            
            df.index.name = "Date"
            return df
            
        except Exception as e:
            logger.error(f"Direct Yahoo API fetch failed for {ticker}: {e}")
            return pd.DataFrame()
    
    def fetch_historical(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data.
        
        Args:
            ticker: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            use_cache: Whether to use cached data
        
        Returns:
            DataFrame with OHLCV columns
        """
        cache_key = self._get_cache_key(ticker, start_date, end_date, "historical")
        cache_path = self._get_cache_path(cache_key)
        
        # Check cache
        if use_cache and self._is_cache_valid(cache_path):
            try:
                df = pd.read_pickle(cache_path)
                logger.info(f"Cache hit for {ticker}")
                return df
            except Exception as e:
                logger.warning(f"Cache read failed: {e}")
        
        # First try yfinance
        try:
            ticker_obj = yf.Ticker(ticker)
            df = ticker_obj.history(start=start_date, end=end_date)
            
            if not df.empty:
                # Standardize column names
                df.columns = [c.title() for c in df.columns]
                
                # Handle timezone-aware index
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                
                # Cache the data
                if use_cache:
                    df.to_pickle(cache_path)
                    logger.info(f"Cached data for {ticker}")
                
                return df
        except Exception as e:
            logger.warning(f"yfinance failed for {ticker}: {e}")
        
        # Fallback to direct Yahoo API
        logger.info(f"Trying direct Yahoo API for {ticker}")
        df = self._fetch_yahoo_direct(ticker, start_date, end_date)
        
        if not df.empty:
            # Cache the data
            if use_cache:
                df.to_pickle(cache_path)
                logger.info(f"Cached data for {ticker} (via direct API)")
            return df
        
        logger.error(f"All fetch methods failed for {ticker}")
        return pd.DataFrame()
    
    def fetch_multiple(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetch data for multiple tickers.
        
        Returns DataFrame with MultiIndex columns (Price, Ticker).
        """
        all_data = {}
        
        for ticker in tickers:
            df = self.fetch_historical(ticker, start_date, end_date)
            if not df.empty and 'Close' in df.columns:
                all_data[ticker] = df['Close']
        
        if not all_data:
            return pd.DataFrame()
        
        return pd.DataFrame(all_data)
    
    def fetch_info(self, ticker: str) -> Dict[str, Any]:
        """
        Fetch company info and fundamentals.
        """
        cache_key = self._get_cache_key(ticker, "", "", "info")
        cache_path = self._get_cache_path(cache_key)
        
        # Check cache
        if self._is_cache_valid(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        
        try:
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info
            
            # Standardize the info dict
            result = {
                "ticker": ticker,
                "name": info.get("longName", info.get("shortName", ticker)),
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "price_to_book": info.get("priceToBook"),
                "price_to_sales": info.get("priceToSalesTrailing12Months"),
                "dividend_yield": info.get("dividendYield"),
                "beta": info.get("beta"),
                "52_week_high": info.get("fiftyTwoWeekHigh"),
                "52_week_low": info.get("fiftyTwoWeekLow"),
                "avg_volume": info.get("averageVolume"),
                "current_price": info.get("currentPrice", info.get("regularMarketPrice")),
                "currency": info.get("currency", "USD")
            }
            
            # Cache
            with open(cache_path, 'w') as f:
                json.dump(result, f)
            
            return result
        except Exception as e:
            logger.error(f"Failed to fetch info for {ticker}: {e}")
            return {"ticker": ticker, "error": str(e)}
    
    def get_returns(
        self,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Get price series and returns for a ticker.
        
        Returns:
            Tuple of (prices, returns)
        """
        df = self.fetch_historical(ticker, start_date, end_date)
        
        if df.empty or 'Close' not in df.columns:
            return pd.Series(), pd.Series()
        
        prices = df['Close']
        returns = np.log(prices / prices.shift(1)).dropna()
        
        return prices, returns
    
    def get_benchmark_returns(
        self,
        start_date: str,
        end_date: str,
        benchmark: str = "SPY"
    ) -> pd.Series:
        """Get benchmark returns for beta calculation."""
        _, returns = self.get_returns(benchmark, start_date, end_date)
        return returns
    
    def get_real_time_quote(self, ticker: str) -> Dict[str, Any]:
        """Get real-time quote (best effort)."""
        try:
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info
            
            return {
                "ticker": ticker,
                "price": info.get("currentPrice", info.get("regularMarketPrice")),
                "change": info.get("regularMarketChange"),
                "change_pct": info.get("regularMarketChangePercent"),
                "volume": info.get("regularMarketVolume"),
                "market_cap": info.get("marketCap"),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {"ticker": ticker, "error": str(e)}
    
    def get_fundamentals(self, ticker: str) -> Dict[str, Any]:
        """
        Get comprehensive fundamental data for a ticker.
        
        Returns full yfinance info dict for fundamental analysis.
        """
        cache_key = self._get_cache_key(ticker, "", "", "fundamentals")
        cache_path = self._get_cache_path(cache_key)
        
        # Check cache
        if self._is_cache_valid(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        
        try:
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info
            
            if not info or 'symbol' not in info:
                return {"ticker": ticker, "error": "No data available"}
            
            # Cache the full info
            try:
                # Convert any non-serializable values
                serializable_info = {}
                for k, v in info.items():
                    try:
                        json.dumps({k: v})
                        serializable_info[k] = v
                    except (TypeError, ValueError):
                        serializable_info[k] = str(v)
                
                with open(cache_path, 'w') as f:
                    json.dump(serializable_info, f)
            except Exception as e:
                logger.warning(f"Could not cache fundamentals: {e}")
            
            return info
        except Exception as e:
            logger.error(f"Failed to fetch fundamentals for {ticker}: {e}")
            return {"ticker": ticker, "error": str(e)}


# Singleton instance
_data_service = None


def get_data_service() -> DataService:
    """Get singleton DataService instance."""
    global _data_service
    if _data_service is None:
        _data_service = DataService()
    return _data_service
