"""
Alpaca Markets Data Provider
=============================
Commission-free trading API with real-time market data.

Features:
- Historical bars (minute, hour, day)
- Real-time quotes and trades
- News API integration
- Account and portfolio data
- Paper trading support

Author: Stock Risk App | Feb 2026
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import logging
import time
import requests
from dataclasses import dataclass
from enum import Enum

from .base_provider import BaseDataProvider

logger = logging.getLogger(__name__)


class AlpacaEnvironment(Enum):
    """Alpaca API environments."""
    PAPER = "paper"
    LIVE = "live"


@dataclass
class AlpacaConfig:
    """Configuration for Alpaca API."""
    base_url_paper: str = "https://paper-api.alpaca.markets"
    base_url_live: str = "https://api.alpaca.markets"
    data_url: str = "https://data.alpaca.markets"
    stream_url: str = "wss://stream.data.alpaca.markets"
    rate_limit_per_minute: int = 200
    timeout: int = 10
    retries: int = 3


class AlpacaProvider(BaseDataProvider):
    """
    Alpaca Markets data provider for stocks and crypto.
    
    Supports:
    - Historical bars with multiple timeframes
    - Real-time quotes and trades
    - News and market data
    - Account management (for trading)
    """
    
    def __init__(
        self, 
        api_key: str = "", 
        api_secret: str = "",
        environment: AlpacaEnvironment = AlpacaEnvironment.PAPER
    ):
        super().__init__(name="alpaca")
        self.api_key = api_key
        self.api_secret = api_secret
        self.environment = environment
        self.config = AlpacaConfig()
        self.logger = logging.getLogger(__name__)
        
        # Rate limiting
        self._last_request_time = 0
        self._request_count = 0
        self._minute_start = time.time()
        
        # Set base URLs based on environment
        if environment == AlpacaEnvironment.LIVE:
            self.trading_url = self.config.base_url_live
        else:
            self.trading_url = self.config.base_url_paper
        
        self.data_url = self.config.data_url
    
    def _get_headers(self) -> Dict[str, str]:
        """Get authentication headers."""
        return {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.api_secret,
            'Content-Type': 'application/json'
        }
    
    def _check_rate_limit(self) -> bool:
        """Check and enforce rate limiting."""
        current_time = time.time()
        
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
        base_url: str = None,
        params: Dict = None,
        method: str = 'GET'
    ) -> Optional[Union[Dict, List]]:
        """Make authenticated request to Alpaca API."""
        if not self.api_key or not self.api_secret:
            self.logger.error("No Alpaca API credentials configured")
            return None
        
        self._check_rate_limit()
        
        base_url = base_url or self.data_url
        url = f"{base_url}{endpoint}"
        
        for attempt in range(self.config.retries):
            try:
                if method == 'GET':
                    response = requests.get(
                        url,
                        headers=self._get_headers(),
                        params=params,
                        timeout=self.config.timeout
                    )
                else:
                    response = requests.post(
                        url,
                        headers=self._get_headers(),
                        json=params,
                        timeout=self.config.timeout
                    )
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    wait_time = int(response.headers.get('Retry-After', 60))
                    self.logger.warning(f"Rate limited. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                elif response.status_code == 401:
                    self.logger.error("Alpaca API credentials invalid")
                    return None
                elif response.status_code == 403:
                    self.logger.error("Alpaca API access forbidden")
                    return None
                else:
                    self.logger.warning(f"Alpaca API error: {response.status_code} - {response.text}")
                    
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < self.config.retries - 1:
                    time.sleep(2 ** attempt)
        
        return None
    
    def fetch_historical(
        self, 
        ticker: str, 
        start: str, 
        end: str, 
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch historical bar data from Alpaca.
        
        Args:
            ticker: Stock symbol (e.g., 'AAPL')
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            interval: Data interval ('1m', '5m', '15m', '1h', '1d')
        
        Returns:
            DataFrame with OHLCV data
        """
        # Map interval to Alpaca timeframe
        interval_map = {
            '1m': '1Min',
            '5m': '5Min',
            '15m': '15Min',
            '30m': '30Min',
            '1h': '1Hour',
            '4h': '4Hour',
            '1d': '1Day',
            '1w': '1Week',
            '1mo': '1Month'
        }
        
        timeframe = interval_map.get(interval, '1Day')
        
        # Use stocks or crypto endpoint based on ticker
        if '-USD' in ticker or ticker.endswith('USD'):
            endpoint = f"/v1beta3/crypto/us/bars"
            params = {
                'symbols': ticker.upper(),
                'timeframe': timeframe,
                'start': f"{start}T00:00:00Z",
                'end': f"{end}T23:59:59Z"
            }
        else:
            endpoint = f"/v2/stocks/{ticker.upper()}/bars"
            params = {
                'timeframe': timeframe,
                'start': f"{start}T00:00:00Z",
                'end': f"{end}T23:59:59Z",
                'adjustment': 'all',
                'limit': 10000
            }
        
        all_bars = []
        next_page_token = None
        
        while True:
            if next_page_token:
                params['page_token'] = next_page_token
            
            data = self._make_request(endpoint, params=params)
            
            if not data:
                break
            
            # Handle stock vs crypto response format
            if 'bars' in data:
                bars = data['bars']
                if isinstance(bars, dict):
                    # Crypto format
                    bars = bars.get(ticker.upper(), [])
                all_bars.extend(bars)
            
            next_page_token = data.get('next_page_token')
            if not next_page_token:
                break
        
        if not all_bars:
            self.logger.warning(f"No data returned for {ticker}")
            return pd.DataFrame()
        
        try:
            df = pd.DataFrame(all_bars)
            
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
            
            # Parse timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Ensure required columns
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col not in df.columns:
                    df[col] = np.nan
            
            return df[['Open', 'High', 'Low', 'Close', 'Volume']].sort_index()
            
        except Exception as e:
            self.logger.error(f"Error parsing Alpaca data: {e}")
            return pd.DataFrame()
    
    def fetch_info(self, ticker: str) -> Dict:
        """
        Fetch asset information from Alpaca.
        
        Returns basic asset information and trading status.
        """
        endpoint = f"/v2/assets/{ticker.upper()}"
        
        data = self._make_request(endpoint, base_url=self.trading_url)
        
        if not data:
            return {'ticker': ticker, 'error': 'No data', 'source': 'alpaca'}
        
        try:
            return {
                'ticker': ticker,
                'name': data.get('name', ticker),
                'asset_class': data.get('class', 'us_equity'),
                'exchange': data.get('exchange', ''),
                'status': data.get('status', ''),
                'tradable': data.get('tradable', False),
                'marginable': data.get('marginable', False),
                'shortable': data.get('shortable', False),
                'easy_to_borrow': data.get('easy_to_borrow', False),
                'fractionable': data.get('fractionable', False),
                'min_order_size': data.get('min_order_size', None),
                'min_trade_increment': data.get('min_trade_increment', None),
                'source': 'alpaca'
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing Alpaca info: {e}")
            return {'ticker': ticker, 'error': str(e), 'source': 'alpaca'}
    
    def get_real_time_quote(self, ticker: str) -> Dict[str, Any]:
        """
        Get latest quote for a ticker.
        
        Returns:
            Dictionary with bid, ask, and latest trade
        """
        endpoint = f"/v2/stocks/{ticker.upper()}/quotes/latest"
        
        data = self._make_request(endpoint)
        
        if not data or 'quote' not in data:
            return {'ticker': ticker, 'error': 'No data'}
        
        try:
            quote = data['quote']
            
            return {
                'ticker': ticker,
                'bid': quote.get('bp', None),
                'bid_size': quote.get('bs', 0),
                'ask': quote.get('ap', None),
                'ask_size': quote.get('as', 0),
                'timestamp': quote.get('t', ''),
                'conditions': quote.get('c', []),
                'source': 'alpaca'
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing quote: {e}")
            return {'ticker': ticker, 'error': str(e)}
    
    def get_latest_trade(self, ticker: str) -> Dict[str, Any]:
        """Get latest trade for a ticker."""
        endpoint = f"/v2/stocks/{ticker.upper()}/trades/latest"
        
        data = self._make_request(endpoint)
        
        if not data or 'trade' not in data:
            return {'ticker': ticker, 'error': 'No data'}
        
        try:
            trade = data['trade']
            
            return {
                'ticker': ticker,
                'price': trade.get('p', None),
                'size': trade.get('s', 0),
                'timestamp': trade.get('t', ''),
                'exchange': trade.get('x', ''),
                'conditions': trade.get('c', []),
                'source': 'alpaca'
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing trade: {e}")
            return {'ticker': ticker, 'error': str(e)}
    
    def get_news(
        self, 
        symbols: List[str] = None,
        limit: int = 10,
        start: str = None,
        end: str = None
    ) -> List[Dict]:
        """
        Fetch news articles from Alpaca.
        
        Args:
            symbols: List of ticker symbols (optional)
            limit: Maximum number of articles
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
        
        Returns:
            List of news article dictionaries
        """
        endpoint = "/v1beta1/news"
        
        params = {'limit': limit}
        
        if symbols:
            params['symbols'] = ','.join([s.upper() for s in symbols])
        
        if start:
            params['start'] = f"{start}T00:00:00Z"
        
        if end:
            params['end'] = f"{end}T23:59:59Z"
        
        data = self._make_request(endpoint, params=params)
        
        if not data or 'news' not in data:
            return []
        
        articles = []
        for article in data['news']:
            articles.append({
                'id': article.get('id', ''),
                'headline': article.get('headline', ''),
                'summary': article.get('summary', ''),
                'author': article.get('author', ''),
                'created_at': article.get('created_at', ''),
                'updated_at': article.get('updated_at', ''),
                'url': article.get('url', ''),
                'symbols': article.get('symbols', []),
                'images': article.get('images', []),
                'source': 'alpaca'
            })
        
        return articles
    
    def get_snapshot(self, ticker: str) -> Dict[str, Any]:
        """
        Get comprehensive snapshot including quote, trade, and bar.
        
        Returns all current market data for a ticker in one call.
        """
        endpoint = f"/v2/stocks/{ticker.upper()}/snapshot"
        
        data = self._make_request(endpoint)
        
        if not data:
            return {'ticker': ticker, 'error': 'No data'}
        
        try:
            latest_trade = data.get('latestTrade', {})
            latest_quote = data.get('latestQuote', {})
            minute_bar = data.get('minuteBar', {})
            daily_bar = data.get('dailyBar', {})
            prev_daily_bar = data.get('prevDailyBar', {})
            
            current_price = latest_trade.get('p', daily_bar.get('c', 0))
            prev_close = prev_daily_bar.get('c', 0)
            
            change = current_price - prev_close if prev_close else 0
            change_pct = (change / prev_close * 100) if prev_close else 0
            
            return {
                'ticker': ticker,
                'price': current_price,
                'change': change,
                'change_pct': change_pct,
                'bid': latest_quote.get('bp'),
                'ask': latest_quote.get('ap'),
                'volume': daily_bar.get('v', 0),
                'vwap': daily_bar.get('vw', 0),
                'open': daily_bar.get('o', 0),
                'high': daily_bar.get('h', 0),
                'low': daily_bar.get('l', 0),
                'prev_close': prev_close,
                'minute_close': minute_bar.get('c', 0),
                'minute_volume': minute_bar.get('v', 0),
                'source': 'alpaca'
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing snapshot: {e}")
            return {'ticker': ticker, 'error': str(e)}
    
    def get_multi_snapshot(self, tickers: List[str]) -> Dict[str, Dict]:
        """Get snapshots for multiple tickers in one call."""
        endpoint = "/v2/stocks/snapshots"
        
        params = {'symbols': ','.join([t.upper() for t in tickers])}
        
        data = self._make_request(endpoint, params=params)
        
        if not data:
            return {}
        
        snapshots = {}
        for ticker, snapshot in data.items():
            snapshots[ticker] = self._parse_snapshot(ticker, snapshot)
        
        return snapshots
    
    def _parse_snapshot(self, ticker: str, data: Dict) -> Dict:
        """Parse individual snapshot data."""
        try:
            latest_trade = data.get('latestTrade', {})
            latest_quote = data.get('latestQuote', {})
            daily_bar = data.get('dailyBar', {})
            prev_daily_bar = data.get('prevDailyBar', {})
            
            current_price = latest_trade.get('p', daily_bar.get('c', 0))
            prev_close = prev_daily_bar.get('c', 0)
            
            return {
                'ticker': ticker,
                'price': current_price,
                'change': current_price - prev_close if prev_close else 0,
                'change_pct': ((current_price - prev_close) / prev_close * 100) if prev_close else 0,
                'bid': latest_quote.get('bp'),
                'ask': latest_quote.get('ap'),
                'volume': daily_bar.get('v', 0),
                'prev_close': prev_close,
                'source': 'alpaca'
            }
        except Exception:
            return {'ticker': ticker, 'error': 'Parse error'}
    
    def is_available(self) -> bool:
        """Check if Alpaca is responding."""
        if not self.api_key or not self.api_secret:
            return False
        
        try:
            endpoint = "/v2/account"
            data = self._make_request(endpoint, base_url=self.trading_url)
            return data is not None
        except Exception:
            return False
    
    def get_account(self) -> Dict[str, Any]:
        """Get account information (requires trading permissions)."""
        endpoint = "/v2/account"
        
        data = self._make_request(endpoint, base_url=self.trading_url)
        
        if not data:
            return {'error': 'Unable to fetch account'}
        
        return {
            'id': data.get('id', ''),
            'status': data.get('status', ''),
            'currency': data.get('currency', 'USD'),
            'buying_power': float(data.get('buying_power', 0)),
            'cash': float(data.get('cash', 0)),
            'portfolio_value': float(data.get('portfolio_value', 0)),
            'pattern_day_trader': data.get('pattern_day_trader', False),
            'trading_blocked': data.get('trading_blocked', False),
            'created_at': data.get('created_at', ''),
            'source': 'alpaca'
        }
    
    def get_positions(self) -> List[Dict]:
        """Get current positions (requires trading permissions)."""
        endpoint = "/v2/positions"
        
        data = self._make_request(endpoint, base_url=self.trading_url)
        
        if not data:
            return []
        
        positions = []
        for pos in data:
            positions.append({
                'symbol': pos.get('symbol', ''),
                'qty': float(pos.get('qty', 0)),
                'avg_entry_price': float(pos.get('avg_entry_price', 0)),
                'market_value': float(pos.get('market_value', 0)),
                'cost_basis': float(pos.get('cost_basis', 0)),
                'unrealized_pl': float(pos.get('unrealized_pl', 0)),
                'unrealized_plpc': float(pos.get('unrealized_plpc', 0)),
                'current_price': float(pos.get('current_price', 0)),
                'change_today': float(pos.get('change_today', 0)),
                'source': 'alpaca'
            })
        
        return positions


# ============================================================================
# WEBSOCKET STREAMING
# ============================================================================

class AlpacaWebSocket:
    """
    WebSocket client for real-time Alpaca data streams.
    
    Supports:
    - Trades
    - Quotes
    - Bars (aggregates)
    """
    
    def __init__(self, api_key: str, api_secret: str, feed: str = 'iex'):
        """
        Initialize Alpaca WebSocket client.
        
        Args:
            api_key: Alpaca API key
            api_secret: Alpaca API secret
            feed: Data feed ('iex' for free, 'sip' for paid)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.feed = feed
        self.ws_url = f"wss://stream.data.alpaca.markets/v2/{feed}"
        self.ws = None
        self.subscribed_channels = {'trades': set(), 'quotes': set(), 'bars': set()}
        self.callbacks = {'trades': {}, 'quotes': {}, 'bars': {}}
        self.logger = logging.getLogger(__name__)
        self._running = False
        self._authenticated = False
    
    def connect(self) -> bool:
        """Establish WebSocket connection and authenticate."""
        try:
            import websocket
            import json
            
            self.ws = websocket.create_connection(self.ws_url)
            
            # Wait for connection message
            msg = self.ws.recv()
            data = json.loads(msg)
            self.logger.info(f"Connection message: {data}")
            
            # Authenticate
            auth_msg = {
                "action": "auth",
                "key": self.api_key,
                "secret": self.api_secret
            }
            self.ws.send(json.dumps(auth_msg))
            
            # Wait for auth response
            msg = self.ws.recv()
            data = json.loads(msg)
            
            if data[0].get('T') == 'success' and data[0].get('msg') == 'authenticated':
                self._authenticated = True
                self._running = True
                self.logger.info("WebSocket authenticated successfully")
                return True
            else:
                self.logger.error(f"Authentication failed: {data}")
                return False
            
        except ImportError:
            self.logger.error("websocket-client not installed")
            return False
        except Exception as e:
            self.logger.error(f"WebSocket connection failed: {e}")
            return False
    
    def subscribe(
        self, 
        tickers: List[str],
        trades: bool = True,
        quotes: bool = True,
        bars: bool = False,
        callbacks: Dict[str, callable] = None
    ):
        """
        Subscribe to real-time data streams.
        
        Args:
            tickers: List of ticker symbols
            trades: Subscribe to trades
            quotes: Subscribe to quotes
            bars: Subscribe to minute bars
            callbacks: Dict of callbacks {'trades': func, 'quotes': func, 'bars': func}
        """
        import json
        
        if not self.ws or not self._authenticated:
            self.logger.error("Not connected or authenticated")
            return
        
        tickers = [t.upper() for t in tickers]
        
        msg = {"action": "subscribe"}
        
        if trades:
            msg['trades'] = tickers
            self.subscribed_channels['trades'].update(tickers)
        
        if quotes:
            msg['quotes'] = tickers
            self.subscribed_channels['quotes'].update(tickers)
        
        if bars:
            msg['bars'] = tickers
            self.subscribed_channels['bars'].update(tickers)
        
        # Store callbacks
        if callbacks:
            for channel, callback in callbacks.items():
                if channel in self.callbacks:
                    for ticker in tickers:
                        self.callbacks[channel][ticker] = callback
        
        self.ws.send(json.dumps(msg))
        self.logger.info(f"Subscribed to: {msg}")
    
    def unsubscribe(
        self, 
        tickers: List[str],
        trades: bool = True,
        quotes: bool = True,
        bars: bool = True
    ):
        """Unsubscribe from data streams."""
        import json
        
        if not self.ws:
            return
        
        tickers = [t.upper() for t in tickers]
        
        msg = {"action": "unsubscribe"}
        
        if trades:
            msg['trades'] = tickers
            self.subscribed_channels['trades'].difference_update(tickers)
        
        if quotes:
            msg['quotes'] = tickers
            self.subscribed_channels['quotes'].difference_update(tickers)
        
        if bars:
            msg['bars'] = tickers
            self.subscribed_channels['bars'].difference_update(tickers)
        
        self.ws.send(json.dumps(msg))
    
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
        msg_type = data.get('T', '')
        symbol = data.get('S', '')
        
        if msg_type == 't':  # Trade
            channel = 'trades'
        elif msg_type == 'q':  # Quote
            channel = 'quotes'
        elif msg_type == 'b':  # Bar
            channel = 'bars'
        else:
            return
        
        if symbol in self.callbacks[channel]:
            try:
                self.callbacks[channel][symbol](data)
            except Exception as e:
                self.logger.error(f"Callback error: {e}")
    
    def close(self):
        """Close WebSocket connection."""
        self._running = False
        self._authenticated = False
        if self.ws:
            self.ws.close()
            self.ws = None
        self.logger.info("WebSocket connection closed")
