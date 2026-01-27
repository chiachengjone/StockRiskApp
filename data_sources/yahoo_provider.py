"""
Yahoo Finance Data Provider
============================
Primary data source using yfinance library.
"""

import pandas as pd
import yfinance as yf
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging

from .base_provider import BaseDataProvider


class YahooProvider(BaseDataProvider):
    """Yahoo Finance data provider using yfinance."""
    
    def __init__(self):
        super().__init__(name="yahoo")
        self.logger = logging.getLogger(__name__)
        self._cache = {}
    
    def fetch_historical(
        self, 
        ticker: str, 
        start: str, 
        end: str, 
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data from Yahoo Finance.
        
        Args:
            ticker: Stock symbol (e.g., 'AAPL', '^GSPC')
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            interval: Data interval ('1d', '1h', '5m', etc.)
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(start=start, end=end, interval=interval)
            
            if data.empty:
                self.logger.warning(f"No data returned for {ticker}")
                return pd.DataFrame()
            
            # Standardize columns
            data = self._standardize_columns(data)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Yahoo Finance error for {ticker}: {e}")
            return pd.DataFrame()
    
    def fetch_info(self, ticker: str) -> Dict:
        """
        Fetch stock fundamentals from Yahoo Finance.
        
        Returns comprehensive company information including:
        - Company name, sector, industry
        - Market cap, P/E, EPS
        - Dividend yield
        - ESG scores (if available)
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Extract key fields with defaults
            return {
                'ticker': ticker,
                'name': info.get('longName', info.get('shortName', ticker)),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', None),
                'forward_pe': info.get('forwardPE', None),
                'eps': info.get('trailingEps', None),
                'dividend_yield': info.get('dividendYield', 0),
                'beta': info.get('beta', None),
                'price': info.get('currentPrice', info.get('regularMarketPrice', None)),
                '52w_high': info.get('fiftyTwoWeekHigh', None),
                '52w_low': info.get('fiftyTwoWeekLow', None),
                'avg_volume': info.get('averageVolume', None),
                'description': info.get('longBusinessSummary', ''),
                'website': info.get('website', ''),
                'employees': info.get('fullTimeEmployees', None),
                'country': info.get('country', 'Unknown'),
                'currency': info.get('currency', 'USD'),
                # Valuation metrics
                'price_to_book': info.get('priceToBook', None),
                'price_to_sales': info.get('priceToSalesTrailing12Months', None),
                'enterprise_value': info.get('enterpriseValue', None),
                'profit_margin': info.get('profitMargins', None),
                'operating_margin': info.get('operatingMargins', None),
                'roe': info.get('returnOnEquity', None),
                'roa': info.get('returnOnAssets', None),
                'revenue': info.get('totalRevenue', None),
                'revenue_growth': info.get('revenueGrowth', None),
                'earnings_growth': info.get('earningsGrowth', None),
                'debt_to_equity': info.get('debtToEquity', None),
                'current_ratio': info.get('currentRatio', None),
                'quick_ratio': info.get('quickRatio', None),
                'free_cash_flow': info.get('freeCashflow', None),
                # ESG (if available)
                'esg_score': info.get('esgScores', {}).get('totalEsg', None) if isinstance(info.get('esgScores'), dict) else None,
                'source': 'yahoo'
            }
            
        except Exception as e:
            self.logger.error(f"Yahoo Finance info error for {ticker}: {e}")
            return {'ticker': ticker, 'error': str(e), 'source': 'yahoo'}
    
    def fetch_multiple(
        self, 
        tickers: List[str], 
        start: str, 
        end: str, 
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch data for multiple tickers efficiently using yfinance batch download.
        """
        try:
            data = yf.download(
                tickers, 
                start=start, 
                end=end, 
                interval=interval,
                group_by='ticker',
                auto_adjust=True,
                progress=False
            )
            
            if data.empty:
                return pd.DataFrame()
            
            return data
            
        except Exception as e:
            self.logger.error(f"Yahoo Finance batch download error: {e}")
            # Fallback to individual downloads
            return super().fetch_multiple(tickers, start, end, interval)
    
    def is_available(self) -> bool:
        """Check if Yahoo Finance is responding."""
        try:
            # Quick test with SPY
            stock = yf.Ticker("SPY")
            hist = stock.history(period="1d")
            return not hist.empty
        except Exception:
            return False
    
    def get_options_chain(self, ticker: str, expiration: str = None) -> Dict:
        """
        Fetch options chain for a ticker.
        
        Args:
            ticker: Stock symbol
            expiration: Expiration date (YYYY-MM-DD), or None for nearest
        
        Returns:
            Dictionary with 'calls' and 'puts' DataFrames
        """
        try:
            stock = yf.Ticker(ticker)
            
            # Get available expirations
            expirations = stock.options
            
            if not expirations:
                return {'error': 'No options available', 'calls': pd.DataFrame(), 'puts': pd.DataFrame()}
            
            # Use provided expiration or first available
            exp_date = expiration if expiration in expirations else expirations[0]
            
            opt = stock.option_chain(exp_date)
            
            return {
                'expiration': exp_date,
                'expirations_available': list(expirations),
                'calls': opt.calls,
                'puts': opt.puts
            }
            
        except Exception as e:
            self.logger.error(f"Options chain error for {ticker}: {e}")
            return {'error': str(e), 'calls': pd.DataFrame(), 'puts': pd.DataFrame()}
    
    def get_financials(self, ticker: str) -> Dict:
        """
        Fetch financial statements.
        
        Returns:
            Dictionary with income_statement, balance_sheet, cash_flow
        """
        try:
            stock = yf.Ticker(ticker)
            
            return {
                'income_statement': stock.income_stmt,
                'balance_sheet': stock.balance_sheet,
                'cash_flow': stock.cashflow,
                'quarterly_income': stock.quarterly_income_stmt,
                'quarterly_balance': stock.quarterly_balance_sheet,
                'quarterly_cashflow': stock.quarterly_cashflow
            }
            
        except Exception as e:
            self.logger.error(f"Financials error for {ticker}: {e}")
            return {'error': str(e)}
    
    def get_analyst_recommendations(self, ticker: str) -> pd.DataFrame:
        """Fetch analyst recommendations."""
        try:
            stock = yf.Ticker(ticker)
            return stock.recommendations
        except Exception:
            return pd.DataFrame()
    
    def get_institutional_holders(self, ticker: str) -> pd.DataFrame:
        """Fetch institutional holders."""
        try:
            stock = yf.Ticker(ticker)
            return stock.institutional_holders
        except Exception:
            return pd.DataFrame()
