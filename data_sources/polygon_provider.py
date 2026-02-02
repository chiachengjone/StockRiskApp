"""
Polygon.io Data Provider
=========================
Professional market data provider with sub-second updates.

Features:
- Historical OHLCV data
- Real-time quotes and trades
- Options data
- Cryptocurrency support
- News and sentiment data

Author: Stock Risk App | Feb 2026
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
import time
import requests
from dataclasses import dataclass

from .base_provider import BaseDataProvider

logger = logging.getLogger(__name__)


@dataclass
class PolygonConfig:
    """Configuration for Polygon.io API."""
    base_url: str = "https://api.polygon.io"
    ws_url: str = "wss://socket.polygon.io"
    rate_limit_per_minute: int = 5  # Free tier
    timeout: int = 10
    retries: int = 3


class PolygonProvider(BaseDataProvider):
    """
    Polygon.io data provider for professional market data.
    
    Supports:
    - Stocks, Options, Forex, Crypto
    - Real-time and historical data
    - Aggregated bars (minute, hour, day)
    - Trades and quotes
    """
    
    def __init__(self, api_key: str = ""):
        super().__init__(name="polygon")
        self.api_key = api_key
        self.config = PolygonConfig()
        self.logger = logging.getLogger(__name__)
        
        # Rate limiting
        self._last_request_time = 0
        self._request_count = 0
        self._minute_start = time.time()
        
        # Cache for reducing API calls
        self._cache = {}
        self._cache_ttl = 60  # seconds
    
    def _check_rate_limit(self) -> bool:
        """Check and enforce rate limiting."""
        current_time = time.time()
        
        # Reset counter every minute
        if current_time - self._minute_start >= 60:
            self._request_count = 0
            self._minute_start = current_time
        
        if self._request_count >= self.config.rate_limit_per_minute:
            wait_time = 60 - (current_time - self._minute_start)
            if wait_time > 0:
                self.logger.warning(f"Rate limit reached. Waiting {wait_time:.1f}s")
                time.sleep(wait_time)
                self._request_count = 0
                self._minute_start = time.time()
        
        self._request_count += 1
        self._last_request_time = current_time
        return True
    
    def _make_request(
        self, 
        endpoint: str, 
        params: Dict = None
    ) -> Optional[Dict]:
        """Make authenticated request to Polygon API."""
        if not self.api_key:
            self.logger.error("No Polygon API key configured")
            return None
        
        self._check_rate_limit()
        
        url = f"{self.config.base_url}{endpoint}"
        params = params or {}
        params['apiKey'] = self.api_key
        
        for attempt in range(self.config.retries):
            try:
                response = requests.get(
                    url, 
                    params=params, 
                    timeout=self.config.timeout
                )
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    # Rate limited - wait and retry
                    wait_time = 60
                    self.logger.warning(f"Rate limited. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                elif response.status_code == 403:
                    self.logger.error("Polygon API key invalid or insufficient permissions")
                    return None
                else:
                    self.logger.warning(f"Polygon API error: {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < self.config.retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        return None
    
    def fetch_historical(
        self, 
        ticker: str, 
        start: str, 
        end: str, 
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data from Polygon.io.
        
        Args:
            ticker: Stock symbol (e.g., 'AAPL')
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            interval: Data interval ('1m', '5m', '1h', '1d', '1w')
        
        Returns:
            DataFrame with OHLCV data
        """
        # Map interval to Polygon timespan/multiplier
        interval_map = {
            '1m': ('minute', 1),
            '5m': ('minute', 5),
            '15m': ('minute', 15),
            '30m': ('minute', 30),
            '1h': ('hour', 1),
            '4h': ('hour', 4),
            '1d': ('day', 1),
            '1w': ('week', 1),
            '1mo': ('month', 1)
        }
        
        timespan, multiplier = interval_map.get(interval, ('day', 1))
        
        endpoint = f"/v2/aggs/ticker/{ticker.upper()}/range/{multiplier}/{timespan}/{start}/{end}"
        
        params = {
            'adjusted': 'true',
            'sort': 'asc',
            'limit': 50000
        }
        
        data = self._make_request(endpoint, params)
        
        if not data or 'results' not in data:
            self.logger.warning(f"No data returned for {ticker}")
            return pd.DataFrame()
        
        try:
            results = data['results']
            
            df = pd.DataFrame(results)
            
            # Rename columns to standard format
            column_map = {
                't': 'timestamp',
                'o': 'Open',
                'h': 'High',
                'l': 'Low',
                'c': 'Close',
                'v': 'Volume',
                'vw': 'VWAP',
                'n': 'Transactions'
            }
            df.rename(columns=column_map, inplace=True)
            
            # Convert timestamp (milliseconds) to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Ensure required columns exist
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col not in df.columns:
                    df[col] = np.nan
            
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
        except Exception as e:
            self.logger.error(f"Error parsing Polygon data: {e}")
            return pd.DataFrame()
    
    def fetch_info(self, ticker: str) -> Dict:
        """
        Fetch stock details from Polygon.io.
        
        Returns company information, market status, and reference data.
        """
        endpoint = f"/v3/reference/tickers/{ticker.upper()}"
        
        data = self._make_request(endpoint)
        
        if not data or 'results' not in data:
            return {'ticker': ticker, 'error': 'No data', 'source': 'polygon'}
        
        try:
            result = data['results']
            
            return {
                'ticker': ticker,
                'name': result.get('name', ticker),
                'description': result.get('description', ''),
                'sector': result.get('sic_description', 'Unknown'),
                'industry': result.get('sic_description', 'Unknown'),
                'market': result.get('market', 'stocks'),
                'locale': result.get('locale', 'us'),
                'type': result.get('type', ''),
                'currency': result.get('currency_name', 'USD'),
                'cik': result.get('cik', ''),
                'composite_figi': result.get('composite_figi', ''),
                'share_class_figi': result.get('share_class_figi', ''),
                'homepage_url': result.get('homepage_url', ''),
                'phone_number': result.get('phone_number', ''),
                'address': result.get('address', {}),
                'market_cap': result.get('market_cap', None),
                'share_class_shares_outstanding': result.get('share_class_shares_outstanding', None),
                'weighted_shares_outstanding': result.get('weighted_shares_outstanding', None),
                'source': 'polygon'
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing Polygon info: {e}")
            return {'ticker': ticker, 'error': str(e), 'source': 'polygon'}
    
    def get_real_time_quote(self, ticker: str) -> Dict[str, Any]:
        """
        Get real-time quote snapshot.
        
        Returns:
            Dictionary with current price, bid, ask, and volume
        """
        endpoint = f"/v2/snapshot/locale/us/markets/stocks/tickers/{ticker.upper()}"
        
        data = self._make_request(endpoint)
        
        if not data or 'ticker' not in data:
            return {'ticker': ticker, 'error': 'No data'}
        
        try:
            result = data['ticker']
            day = result.get('day', {})
            prev_day = result.get('prevDay', {})
            last_quote = result.get('lastQuote', {})
            last_trade = result.get('lastTrade', {})
            
            current_price = last_trade.get('p', day.get('c', 0))
            prev_close = prev_day.get('c', 0)
            
            change = current_price - prev_close if prev_close else 0
            change_pct = (change / prev_close * 100) if prev_close else 0
            
            return {
                'ticker': ticker,
                'price': current_price,
                'change': change,
                'change_pct': change_pct,
                'bid': last_quote.get('p', None),
                'bid_size': last_quote.get('s', 0),
                'ask': last_quote.get('P', None),
                'ask_size': last_quote.get('S', 0),
                'volume': day.get('v', 0),
                'vwap': day.get('vw', 0),
                'open': day.get('o', 0),
                'high': day.get('h', 0),
                'low': day.get('l', 0),
                'prev_close': prev_close,
                'last_trade_time': last_trade.get('t', None),
                'source': 'polygon'
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing real-time quote: {e}")
            return {'ticker': ticker, 'error': str(e)}
    
    def get_news(
        self, 
        ticker: str = None, 
        limit: int = 10,
        published_after: str = None
    ) -> List[Dict]:
        """
        Fetch news articles for a ticker or market.
        
        Args:
            ticker: Stock symbol (optional, for ticker-specific news)
            limit: Maximum number of articles
            published_after: Filter by date (YYYY-MM-DD)
        
        Returns:
            List of news article dictionaries
        """
        endpoint = "/v2/reference/news"
        
        params = {
            'limit': limit,
            'order': 'desc'
        }
        
        if ticker:
            params['ticker'] = ticker.upper()
        
        if published_after:
            params['published_utc.gte'] = published_after
        
        data = self._make_request(endpoint, params)
        
        if not data or 'results' not in data:
            return []
        
        articles = []
        for article in data['results']:
            articles.append({
                'id': article.get('id', ''),
                'title': article.get('title', ''),
                'description': article.get('description', ''),
                'author': article.get('author', ''),
                'published_utc': article.get('published_utc', ''),
                'article_url': article.get('article_url', ''),
                'tickers': article.get('tickers', []),
                'image_url': article.get('image_url', ''),
                'keywords': article.get('keywords', []),
                'source': 'polygon'
            })
        
        return articles
    
    def get_options_chain(
        self, 
        underlying: str,
        expiration_date: str = None,
        strike_price: float = None,
        contract_type: str = None
    ) -> pd.DataFrame:
        """
        Fetch options chain data.
        
        Args:
            underlying: Underlying stock symbol
            expiration_date: Filter by expiration (YYYY-MM-DD)
            strike_price: Filter by strike price
            contract_type: 'call' or 'put'
        
        Returns:
            DataFrame with options contracts
        """
        endpoint = "/v3/reference/options/contracts"
        
        params = {
            'underlying_ticker': underlying.upper(),
            'limit': 250
        }
        
        if expiration_date:
            params['expiration_date'] = expiration_date
        
        if strike_price:
            params['strike_price'] = strike_price
        
        if contract_type:
            params['contract_type'] = contract_type
        
        data = self._make_request(endpoint, params)
        
        if not data or 'results' not in data:
            return pd.DataFrame()
        
        return pd.DataFrame(data['results'])
    
    def is_available(self) -> bool:
        """Check if Polygon.io is responding."""
        if not self.api_key:
            return False
        
        try:
            endpoint = "/v1/marketstatus/now"
            data = self._make_request(endpoint)
            return data is not None
        except Exception:
            return False
    
    def get_market_status(self) -> Dict[str, Any]:
        """Get current market status."""
        endpoint = "/v1/marketstatus/now"
        
        data = self._make_request(endpoint)
        
        if not data:
            return {'error': 'Unable to fetch market status'}
        
        return {
            'market': data.get('market', 'unknown'),
            'exchanges': data.get('exchanges', {}),
            'currencies': data.get('currencies', {}),
            'serverTime': data.get('serverTime', ''),
            'source': 'polygon'
        }
    
    def get_grouped_daily(self, date: str) -> pd.DataFrame:
        """
        Get all tickers' daily bars for a specific date.
        
        Useful for market-wide analysis.
        
        Args:
            date: Date in YYYY-MM-DD format
        
        Returns:
            DataFrame with all tickers' OHLCV data
        """
        endpoint = f"/v2/aggs/grouped/locale/us/market/stocks/{date}"
        
        params = {'adjusted': 'true'}
        
        data = self._make_request(endpoint, params)
        
        if not data or 'results' not in data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data['results'])
        
        column_map = {
            'T': 'Ticker',
            'o': 'Open',
            'h': 'High',
            'l': 'Low',
            'c': 'Close',
            'v': 'Volume',
            'vw': 'VWAP',
            'n': 'Transactions'
        }
        df.rename(columns=column_map, inplace=True)
        
        return df


# ============================================================================
# WEBSOCKET STREAMING
# ============================================================================

class PolygonWebSocket:
    """
    WebSocket client for real-time Polygon.io data streams.
    
    Supports:
    - Trades (T.*)
    - Quotes (Q.*)
    - Aggregates (A.*, AM.*)
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.ws_url = "wss://socket.polygon.io/stocks"
        self.ws = None
        self.subscribed_channels = set()
        self.callbacks = {}
        self.logger = logging.getLogger(__name__)
        self._running = False
    
    def connect(self) -> bool:
        """Establish WebSocket connection."""
        try:
            import websocket
            
            self.ws = websocket.create_connection(self.ws_url)
            
            # Authenticate
            auth_msg = {"action": "auth", "params": self.api_key}
            self.ws.send(str(auth_msg).replace("'", '"'))
            
            response = self.ws.recv()
            self.logger.info(f"WebSocket auth response: {response}")
            
            self._running = True
            return True
            
        except ImportError:
            self.logger.error("websocket-client not installed")
            return False
        except Exception as e:
            self.logger.error(f"WebSocket connection failed: {e}")
            return False
    
    def subscribe(
        self, 
        tickers: List[str], 
        channels: List[str] = None,
        callback: callable = None
    ):
        """
        Subscribe to real-time data streams.
        
        Args:
            tickers: List of ticker symbols
            channels: Channel types ('T' for trades, 'Q' for quotes, 'A' for aggregates)
            callback: Function to call with received data
        """
        if not self.ws:
            self.logger.error("Not connected. Call connect() first.")
            return
        
        channels = channels or ['T', 'Q', 'A']
        
        subscriptions = []
        for ticker in tickers:
            for channel in channels:
                sub = f"{channel}.{ticker.upper()}"
                subscriptions.append(sub)
                self.subscribed_channels.add(sub)
        
        if callback:
            for sub in subscriptions:
                self.callbacks[sub] = callback
        
        msg = {"action": "subscribe", "params": ",".join(subscriptions)}
        self.ws.send(str(msg).replace("'", '"'))
        self.logger.info(f"Subscribed to: {subscriptions}")
    
    def unsubscribe(self, tickers: List[str], channels: List[str] = None):
        """Unsubscribe from data streams."""
        if not self.ws:
            return
        
        channels = channels or ['T', 'Q', 'A']
        
        unsubscriptions = []
        for ticker in tickers:
            for channel in channels:
                sub = f"{channel}.{ticker.upper()}"
                unsubscriptions.append(sub)
                self.subscribed_channels.discard(sub)
                self.callbacks.pop(sub, None)
        
        msg = {"action": "unsubscribe", "params": ",".join(unsubscriptions)}
        self.ws.send(str(msg).replace("'", '"'))
    
    def listen(self):
        """Listen for incoming messages (blocking)."""
        import json
        
        while self._running and self.ws:
            try:
                message = self.ws.recv()
                data = json.loads(message)
                
                if isinstance(data, list):
                    for item in data:
                        self._handle_message(item)
                else:
                    self._handle_message(data)
                    
            except Exception as e:
                self.logger.error(f"WebSocket error: {e}")
                break
    
    def _handle_message(self, data: Dict):
        """Process incoming message."""
        ev = data.get('ev', '')
        sym = data.get('sym', '')
        
        channel_key = f"{ev}.{sym}"
        
        if channel_key in self.callbacks:
            try:
                self.callbacks[channel_key](data)
            except Exception as e:
                self.logger.error(f"Callback error: {e}")
    
    def close(self):
        """Close WebSocket connection."""
        self._running = False
        if self.ws:
            self.ws.close()
            self.ws = None
        self.logger.info("WebSocket connection closed")
