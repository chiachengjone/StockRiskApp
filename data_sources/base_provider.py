"""
Base Data Provider - Abstract Interface
========================================
Defines the contract for all data providers.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, List
import pandas as pd
from datetime import datetime


class BaseDataProvider(ABC):
    """Abstract base class for all data providers."""
    
    def __init__(self, name: str):
        self.name = name
        self._last_request_time = None
        self._request_count = 0
    
    @abstractmethod
    def fetch_historical(
        self, 
        ticker: str, 
        start: str, 
        end: str, 
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data.
        
        Args:
            ticker: Stock symbol (e.g., 'AAPL')
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            interval: Data interval ('1d', '1h', etc.)
        
        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume
            Index should be DatetimeIndex
        """
        pass
    
    @abstractmethod
    def fetch_info(self, ticker: str) -> Dict:
        """
        Fetch stock fundamentals/info.
        
        Args:
            ticker: Stock symbol
        
        Returns:
            Dictionary with company info (name, sector, market cap, etc.)
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the data provider is available and responding.
        
        Returns:
            True if provider is operational
        """
        pass
    
    def fetch_multiple(
        self, 
        tickers: List[str], 
        start: str, 
        end: str, 
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch historical data for multiple tickers.
        
        Default implementation fetches one by one.
        Subclasses can override for batch fetching.
        
        Returns:
            DataFrame with MultiIndex columns (ticker, OHLCV)
        """
        all_data = {}
        for ticker in tickers:
            try:
                data = self.fetch_historical(ticker, start, end, interval)
                if not data.empty:
                    all_data[ticker] = data
            except Exception:
                continue
        
        if not all_data:
            return pd.DataFrame()
        
        # Combine into multi-level columns
        combined = pd.concat(all_data, axis=1)
        return combined
    
    def validate_ticker(self, ticker: str) -> bool:
        """
        Check if a ticker symbol is valid.
        
        Args:
            ticker: Stock symbol to validate
        
        Returns:
            True if ticker exists
        """
        try:
            data = self.fetch_historical(
                ticker, 
                (datetime.now().strftime('%Y-%m-%d')),
                (datetime.now().strftime('%Y-%m-%d'))
            )
            return not data.empty
        except Exception:
            return False
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure consistent column naming."""
        column_mapping = {
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume',
            'adj close': 'Adj Close',
            'adjusted close': 'Adj Close'
        }
        
        df.columns = [column_mapping.get(c.lower(), c) for c in df.columns]
        return df
    
    def __repr__(self):
        return f"<{self.__class__.__name__}(name={self.name})>"
