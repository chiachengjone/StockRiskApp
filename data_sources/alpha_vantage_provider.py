"""
Alpha Vantage Data Provider
============================
Secondary data source for fundamentals and backup historical data.
Free tier: 5 API calls/minute, 500 calls/day
"""

import pandas as pd
import requests
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging
import time

from .base_provider import BaseDataProvider


class AlphaVantageProvider(BaseDataProvider):
    """Alpha Vantage data provider."""
    
    BASE_URL = "https://www.alphavantage.co/query"
    
    def __init__(self, api_key: str = ""):
        super().__init__(name="alpha_vantage")
        self.api_key = api_key
        self.logger = logging.getLogger(__name__)
        self._last_call_time = 0
        self._min_interval = 12  # seconds between calls (5/min limit)
    
    def _rate_limit(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self._last_call_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call_time = time.time()
    
    def _make_request(self, params: Dict) -> Dict:
        """Make API request with rate limiting."""
        if not self.api_key:
            self.logger.warning("Alpha Vantage API key not set")
            return {'error': 'API key not configured'}
        
        self._rate_limit()
        
        params['apikey'] = self.api_key
        
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Check for API error messages
            if 'Error Message' in data:
                self.logger.error(f"Alpha Vantage API error: {data['Error Message']}")
                return {'error': data['Error Message']}
            
            if 'Note' in data:
                self.logger.warning(f"Alpha Vantage rate limit: {data['Note']}")
                return {'error': 'Rate limit reached'}
            
            return data
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Alpha Vantage request error: {e}")
            return {'error': str(e)}
    
    def fetch_historical(
        self, 
        ticker: str, 
        start: str, 
        end: str, 
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch historical data from Alpha Vantage.
        
        Note: Alpha Vantage returns full history, we filter by date range.
        """
        # Map interval to Alpha Vantage function
        if interval in ['1d', 'daily']:
            params = {
                'function': 'TIME_SERIES_DAILY_ADJUSTED',
                'symbol': ticker,
                'outputsize': 'full'
            }
            time_series_key = 'Time Series (Daily)'
        elif interval in ['1wk', 'weekly']:
            params = {
                'function': 'TIME_SERIES_WEEKLY_ADJUSTED',
                'symbol': ticker
            }
            time_series_key = 'Weekly Adjusted Time Series'
        else:
            # Default to daily
            params = {
                'function': 'TIME_SERIES_DAILY_ADJUSTED',
                'symbol': ticker,
                'outputsize': 'full'
            }
            time_series_key = 'Time Series (Daily)'
        
        data = self._make_request(params)
        
        if 'error' in data or time_series_key not in data:
            return pd.DataFrame()
        
        try:
            # Parse time series data
            ts_data = data[time_series_key]
            
            records = []
            for date_str, values in ts_data.items():
                records.append({
                    'Date': pd.to_datetime(date_str),
                    'Open': float(values.get('1. open', 0)),
                    'High': float(values.get('2. high', 0)),
                    'Low': float(values.get('3. low', 0)),
                    'Close': float(values.get('4. close', 0)),
                    'Adj Close': float(values.get('5. adjusted close', values.get('4. close', 0))),
                    'Volume': int(float(values.get('6. volume', values.get('5. volume', 0))))
                })
            
            df = pd.DataFrame(records)
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
            
            # Filter by date range
            start_date = pd.to_datetime(start)
            end_date = pd.to_datetime(end)
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            
            return df
            
        except Exception as e:
            self.logger.error(f"Alpha Vantage parse error: {e}")
            return pd.DataFrame()
    
    def fetch_info(self, ticker: str) -> Dict:
        """
        Fetch company overview from Alpha Vantage.
        
        This provides extensive fundamental data.
        """
        params = {
            'function': 'OVERVIEW',
            'symbol': ticker
        }
        
        data = self._make_request(params)
        
        if 'error' in data or not data:
            return {'ticker': ticker, 'error': data.get('error', 'No data'), 'source': 'alpha_vantage'}
        
        return {
            'ticker': ticker,
            'name': data.get('Name', ticker),
            'description': data.get('Description', ''),
            'sector': data.get('Sector', 'Unknown'),
            'industry': data.get('Industry', 'Unknown'),
            'market_cap': self._safe_float(data.get('MarketCapitalization')),
            'pe_ratio': self._safe_float(data.get('PERatio')),
            'forward_pe': self._safe_float(data.get('ForwardPE')),
            'peg_ratio': self._safe_float(data.get('PEGRatio')),
            'eps': self._safe_float(data.get('EPS')),
            'dividend_yield': self._safe_float(data.get('DividendYield')),
            'dividend_per_share': self._safe_float(data.get('DividendPerShare')),
            'beta': self._safe_float(data.get('Beta')),
            'price': self._safe_float(data.get('AnalystTargetPrice')),
            '52w_high': self._safe_float(data.get('52WeekHigh')),
            '52w_low': self._safe_float(data.get('52WeekLow')),
            '50d_ma': self._safe_float(data.get('50DayMovingAverage')),
            '200d_ma': self._safe_float(data.get('200DayMovingAverage')),
            'shares_outstanding': self._safe_float(data.get('SharesOutstanding')),
            'book_value': self._safe_float(data.get('BookValue')),
            'price_to_book': self._safe_float(data.get('PriceToBookRatio')),
            'price_to_sales': self._safe_float(data.get('PriceToSalesRatioTTM')),
            'enterprise_value': self._safe_float(data.get('EnterpriseValue')),
            'profit_margin': self._safe_float(data.get('ProfitMargin')),
            'operating_margin': self._safe_float(data.get('OperatingMarginTTM')),
            'roe': self._safe_float(data.get('ReturnOnEquityTTM')),
            'roa': self._safe_float(data.get('ReturnOnAssetsTTM')),
            'revenue': self._safe_float(data.get('RevenueTTM')),
            'revenue_per_share': self._safe_float(data.get('RevenuePerShareTTM')),
            'gross_profit': self._safe_float(data.get('GrossProfitTTM')),
            'ebitda': self._safe_float(data.get('EBITDA')),
            'quarterly_earnings_growth': self._safe_float(data.get('QuarterlyEarningsGrowthYOY')),
            'quarterly_revenue_growth': self._safe_float(data.get('QuarterlyRevenueGrowthYOY')),
            'analyst_target': self._safe_float(data.get('AnalystTargetPrice')),
            'trailing_pe': self._safe_float(data.get('TrailingPE')),
            'ev_to_revenue': self._safe_float(data.get('EVToRevenue')),
            'ev_to_ebitda': self._safe_float(data.get('EVToEBITDA')),
            'exchange': data.get('Exchange', ''),
            'currency': data.get('Currency', 'USD'),
            'country': data.get('Country', 'Unknown'),
            'fiscal_year_end': data.get('FiscalYearEnd', ''),
            'latest_quarter': data.get('LatestQuarter', ''),
            'source': 'alpha_vantage'
        }
    
    def _safe_float(self, value) -> Optional[float]:
        """Safely convert to float."""
        if value is None or value == 'None' or value == '':
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def is_available(self) -> bool:
        """Check if Alpha Vantage is available."""
        if not self.api_key:
            return False
        
        # Simple test - get quote for IBM (test symbol)
        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': 'IBM'
        }
        
        data = self._make_request(params)
        return 'Global Quote' in data
    
    def fetch_income_statement(self, ticker: str) -> pd.DataFrame:
        """Fetch annual income statement."""
        params = {
            'function': 'INCOME_STATEMENT',
            'symbol': ticker
        }
        
        data = self._make_request(params)
        
        if 'error' in data or 'annualReports' not in data:
            return pd.DataFrame()
        
        return pd.DataFrame(data['annualReports'])
    
    def fetch_balance_sheet(self, ticker: str) -> pd.DataFrame:
        """Fetch annual balance sheet."""
        params = {
            'function': 'BALANCE_SHEET',
            'symbol': ticker
        }
        
        data = self._make_request(params)
        
        if 'error' in data or 'annualReports' not in data:
            return pd.DataFrame()
        
        return pd.DataFrame(data['annualReports'])
    
    def fetch_cash_flow(self, ticker: str) -> pd.DataFrame:
        """Fetch annual cash flow statement."""
        params = {
            'function': 'CASH_FLOW',
            'symbol': ticker
        }
        
        data = self._make_request(params)
        
        if 'error' in data or 'annualReports' not in data:
            return pd.DataFrame()
        
        return pd.DataFrame(data['annualReports'])
    
    def fetch_earnings(self, ticker: str) -> Dict:
        """Fetch earnings data (annual and quarterly)."""
        params = {
            'function': 'EARNINGS',
            'symbol': ticker
        }
        
        data = self._make_request(params)
        
        if 'error' in data:
            return {'error': data['error']}
        
        return {
            'annual': pd.DataFrame(data.get('annualEarnings', [])),
            'quarterly': pd.DataFrame(data.get('quarterlyEarnings', []))
        }
    
    def fetch_quote(self, ticker: str) -> Dict:
        """Fetch real-time quote."""
        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': ticker
        }
        
        data = self._make_request(params)
        
        if 'error' in data or 'Global Quote' not in data:
            return {'error': data.get('error', 'No data')}
        
        quote = data['Global Quote']
        return {
            'symbol': quote.get('01. symbol'),
            'open': self._safe_float(quote.get('02. open')),
            'high': self._safe_float(quote.get('03. high')),
            'low': self._safe_float(quote.get('04. low')),
            'price': self._safe_float(quote.get('05. price')),
            'volume': self._safe_float(quote.get('06. volume')),
            'latest_trading_day': quote.get('07. latest trading day'),
            'previous_close': self._safe_float(quote.get('08. previous close')),
            'change': self._safe_float(quote.get('09. change')),
            'change_percent': quote.get('10. change percent', '').replace('%', '')
        }
    
    def fetch_sma(self, ticker: str, period: int = 50) -> pd.DataFrame:
        """Fetch Simple Moving Average."""
        params = {
            'function': 'SMA',
            'symbol': ticker,
            'interval': 'daily',
            'time_period': period,
            'series_type': 'close'
        }
        
        data = self._make_request(params)
        
        if 'error' in data or 'Technical Analysis: SMA' not in data:
            return pd.DataFrame()
        
        sma_data = data['Technical Analysis: SMA']
        records = [{'Date': k, 'SMA': float(v['SMA'])} for k, v in sma_data.items()]
        df = pd.DataFrame(records)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        return df.sort_index()
    
    def fetch_rsi(self, ticker: str, period: int = 14) -> pd.DataFrame:
        """Fetch Relative Strength Index."""
        params = {
            'function': 'RSI',
            'symbol': ticker,
            'interval': 'daily',
            'time_period': period,
            'series_type': 'close'
        }
        
        data = self._make_request(params)
        
        if 'error' in data or 'Technical Analysis: RSI' not in data:
            return pd.DataFrame()
        
        rsi_data = data['Technical Analysis: RSI']
        records = [{'Date': k, 'RSI': float(v['RSI'])} for k, v in rsi_data.items()]
        df = pd.DataFrame(records)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        return df.sort_index()
