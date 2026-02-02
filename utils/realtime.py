"""
Real-Time Features Module
=========================
Live Quotes | Market Hours | P&L Tracking | Intraday Data

Author: Stock Risk App | Feb 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time as dt_time
from typing import Dict, List, Optional, Any, Callable
import logging
import threading
import queue

logger = logging.getLogger(__name__)

# Try to import timezone support
try:
    import pytz
    HAS_PYTZ = True
except ImportError:
    HAS_PYTZ = False
    logger.warning("pytz not installed - timezone features limited")

# ============================================================================
# MARKET HOURS
# ============================================================================

MARKET_SCHEDULES = {
    'NYSE': {
        'timezone': 'US/Eastern',
        'open': dt_time(9, 30),
        'close': dt_time(16, 0),
        'pre_market_open': dt_time(4, 0),
        'after_hours_close': dt_time(20, 0),
        'trading_days': [0, 1, 2, 3, 4]  # Monday=0
    },
    'NASDAQ': {
        'timezone': 'US/Eastern',
        'open': dt_time(9, 30),
        'close': dt_time(16, 0),
        'pre_market_open': dt_time(4, 0),
        'after_hours_close': dt_time(20, 0),
        'trading_days': [0, 1, 2, 3, 4]
    },
    'LSE': {
        'timezone': 'Europe/London',
        'open': dt_time(8, 0),
        'close': dt_time(16, 30),
        'pre_market_open': None,
        'after_hours_close': None,
        'trading_days': [0, 1, 2, 3, 4]
    },
    'CRYPTO': {
        'timezone': 'UTC',
        'open': dt_time(0, 0),
        'close': dt_time(23, 59),
        'pre_market_open': None,
        'after_hours_close': None,
        'trading_days': [0, 1, 2, 3, 4, 5, 6]  # 24/7
    }
}

# US Market holidays (simplified - 2024-2026)
US_HOLIDAYS = [
    # 2025
    '2025-01-01', '2025-01-20', '2025-02-17', '2025-04-18', '2025-05-26',
    '2025-06-19', '2025-07-04', '2025-09-01', '2025-11-27', '2025-12-25',
    # 2026
    '2026-01-01', '2026-01-19', '2026-02-16', '2026-04-03', '2026-05-25',
    '2026-06-19', '2026-07-03', '2026-09-07', '2026-11-26', '2026-12-25',
]

def get_market_timezone(exchange: str = 'NYSE'):
    """Get timezone for exchange."""
    if not HAS_PYTZ:
        return None
    
    schedule = MARKET_SCHEDULES.get(exchange, MARKET_SCHEDULES['NYSE'])
    return pytz.timezone(schedule['timezone'])

def is_market_open(exchange: str = 'NYSE', include_extended: bool = False) -> Dict[str, Any]:
    """
    Check if market is currently open.
    
    Args:
        exchange: Exchange to check
        include_extended: Include pre-market and after-hours
    
    Returns:
        Dictionary with market status details
    """
    result = {
        'is_open': False,
        'session': 'closed',
        'exchange': exchange,
        'message': '',
        'next_open': None,
        'time_until_open': None,
        'time_until_close': None
    }
    
    schedule = MARKET_SCHEDULES.get(exchange, MARKET_SCHEDULES['NYSE'])
    
    # Get current time in exchange timezone
    if HAS_PYTZ:
        tz = pytz.timezone(schedule['timezone'])
        now = datetime.now(tz)
    else:
        now = datetime.now()
    
    current_time = now.time()
    current_day = now.weekday()
    current_date = now.strftime('%Y-%m-%d')
    
    # Check if it's a trading day
    if current_day not in schedule['trading_days']:
        result['message'] = f"🔴 Market CLOSED (Weekend)"
        result['session'] = 'weekend'
        return result
    
    # Check holidays (US markets only)
    if exchange in ['NYSE', 'NASDAQ'] and current_date in US_HOLIDAYS:
        result['message'] = f"🔴 Market CLOSED (Holiday)"
        result['session'] = 'holiday'
        return result
    
    # Crypto is always open
    if exchange == 'CRYPTO':
        result['is_open'] = True
        result['session'] = 'regular'
        result['message'] = "🟢 CRYPTO Market OPEN 24/7"
        return result
    
    # Check trading hours
    market_open = schedule['open']
    market_close = schedule['close']
    
    if market_open <= current_time < market_close:
        result['is_open'] = True
        result['session'] = 'regular'
        result['message'] = f"🟢 Market OPEN (Regular hours)"
        
        # Calculate time until close
        close_dt = datetime.combine(now.date(), market_close)
        result['time_until_close'] = str(close_dt - now.replace(tzinfo=None))[:8]
        
    elif include_extended and schedule.get('pre_market_open'):
        pre_open = schedule['pre_market_open']
        after_close = schedule['after_hours_close']
        
        if pre_open <= current_time < market_open:
            result['is_open'] = True
            result['session'] = 'pre_market'
            result['message'] = "🟡 PRE-MARKET Open"
        elif market_close <= current_time < after_close:
            result['is_open'] = True
            result['session'] = 'after_hours'
            result['message'] = "🟡 AFTER-HOURS Open"
        else:
            result['message'] = f"🔴 Market CLOSED"
    else:
        result['message'] = f"🔴 Market CLOSED"
    
    return result

def get_next_market_open(exchange: str = 'NYSE') -> datetime:
    """Get the next market open time."""
    schedule = MARKET_SCHEDULES.get(exchange, MARKET_SCHEDULES['NYSE'])
    
    if HAS_PYTZ:
        tz = pytz.timezone(schedule['timezone'])
        now = datetime.now(tz)
    else:
        now = datetime.now()
    
    # Start from tomorrow if market already closed today
    check_date = now.date()
    if now.time() >= schedule['close']:
        check_date = check_date + timedelta(days=1)
    
    # Find next trading day
    for i in range(7):
        test_date = check_date + timedelta(days=i)
        if test_date.weekday() in schedule['trading_days']:
            date_str = test_date.strftime('%Y-%m-%d')
            if exchange not in ['NYSE', 'NASDAQ'] or date_str not in US_HOLIDAYS:
                return datetime.combine(test_date, schedule['open'])
    
    return None

# ============================================================================
# LIVE QUOTES
# ============================================================================

def get_live_quote(ticker: str) -> Dict[str, Any]:
    """
    Get real-time quote for a ticker.
    
    Returns:
        Dictionary with current price, change, volume, bid/ask
    """
    import yfinance as yf
    
    result = {
        'ticker': ticker,
        'price': None,
        'change': None,
        'change_pct': None,
        'volume': None,
        'bid': None,
        'ask': None,
        'high': None,
        'low': None,
        'open': None,
        'prev_close': None,
        'market_cap': None,
        'last_updated': datetime.now(),
        'error': None
    }
    
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get price data
        result['price'] = info.get('currentPrice') or info.get('regularMarketPrice')
        result['prev_close'] = info.get('previousClose') or info.get('regularMarketPreviousClose')
        result['open'] = info.get('open') or info.get('regularMarketOpen')
        result['high'] = info.get('dayHigh') or info.get('regularMarketDayHigh')
        result['low'] = info.get('dayLow') or info.get('regularMarketDayLow')
        result['volume'] = info.get('volume') or info.get('regularMarketVolume')
        result['bid'] = info.get('bid')
        result['ask'] = info.get('ask')
        result['market_cap'] = info.get('marketCap')
        
        # Calculate change
        if result['price'] and result['prev_close']:
            result['change'] = result['price'] - result['prev_close']
            result['change_pct'] = (result['change'] / result['prev_close']) * 100
        
    except Exception as e:
        result['error'] = str(e)
        logger.error(f"Error getting live quote for {ticker}: {e}")
    
    return result

def get_multiple_quotes(tickers: List[str]) -> Dict[str, Dict]:
    """
    Get live quotes for multiple tickers.
    
    Args:
        tickers: List of ticker symbols
    
    Returns:
        Dictionary mapping ticker to quote data
    """
    quotes = {}
    for ticker in tickers:
        quotes[ticker] = get_live_quote(ticker)
    return quotes

def format_quote_display(quote: Dict[str, Any]) -> str:
    """Format quote for display."""
    if quote.get('error'):
        return f"Error: {quote['error']}"
    
    price = quote.get('price', 0)
    change = quote.get('change', 0)
    change_pct = quote.get('change_pct', 0)
    
    arrow = "↑" if change >= 0 else "↓"
    color = "green" if change >= 0 else "red"
    
    return f"${price:.2f} {arrow} {abs(change):.2f} ({abs(change_pct):.2f}%)"

# ============================================================================
# INTRADAY DATA
# ============================================================================

def get_intraday_data(
    ticker: str,
    period: str = '1d',
    interval: str = '1m'
) -> pd.DataFrame:
    """
    Get intraday price data.
    
    Args:
        ticker: Ticker symbol
        period: Time period ('1d', '5d', '1mo')
        interval: Data interval ('1m', '5m', '15m', '1h')
    
    Returns:
        DataFrame with OHLCV data
    """
    import yfinance as yf
    
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval=interval)
        return data
    except Exception as e:
        logger.error(f"Error getting intraday data for {ticker}: {e}")
        return pd.DataFrame()

def get_intraday_vwap(data: pd.DataFrame) -> pd.Series:
    """
    Calculate Volume-Weighted Average Price (VWAP).
    
    Args:
        data: DataFrame with High, Low, Close, Volume columns
    
    Returns:
        VWAP series
    """
    if data.empty:
        return pd.Series()
    
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    vwap = (typical_price * data['Volume']).cumsum() / data['Volume'].cumsum()
    return vwap

# ============================================================================
# P&L TRACKING
# ============================================================================

def calculate_live_pnl(
    positions: Dict[str, Dict],
    live_prices: Dict[str, float] = None
) -> Dict[str, Any]:
    """
    Calculate real-time P&L for portfolio positions.
    
    Args:
        positions: Dictionary with position details
            Format: {'AAPL': {'shares': 100, 'cost_basis': 150.0}}
        live_prices: Optional pre-fetched prices (fetched if not provided)
    
    Returns:
        Dictionary with P&L breakdown and totals
    """
    result = {
        'positions': {},
        'total_cost': 0,
        'total_value': 0,
        'total_pnl': 0,
        'total_pnl_pct': 0,
        'best_performer': None,
        'worst_performer': None,
        'updated_at': datetime.now()
    }
    
    if not positions:
        return result
    
    # Fetch live prices if not provided
    if live_prices is None:
        live_prices = {}
        for ticker in positions:
            quote = get_live_quote(ticker)
            if quote['price']:
                live_prices[ticker] = quote['price']
    
    best_pnl_pct = float('-inf')
    worst_pnl_pct = float('inf')
    
    for ticker, pos in positions.items():
        shares = pos.get('shares', 0)
        cost_basis = pos.get('cost_basis', 0)
        current_price = live_prices.get(ticker, cost_basis)
        
        cost = shares * cost_basis
        value = shares * current_price
        pnl = value - cost
        pnl_pct = ((current_price / cost_basis) - 1) * 100 if cost_basis > 0 else 0
        
        result['positions'][ticker] = {
            'shares': shares,
            'cost_basis': cost_basis,
            'current_price': current_price,
            'cost': cost,
            'value': value,
            'pnl': pnl,
            'pnl_pct': pnl_pct
        }
        
        result['total_cost'] += cost
        result['total_value'] += value
        result['total_pnl'] += pnl
        
        if pnl_pct > best_pnl_pct:
            best_pnl_pct = pnl_pct
            result['best_performer'] = ticker
        if pnl_pct < worst_pnl_pct:
            worst_pnl_pct = pnl_pct
            result['worst_performer'] = ticker
    
    if result['total_cost'] > 0:
        result['total_pnl_pct'] = ((result['total_value'] / result['total_cost']) - 1) * 100
    
    return result

def format_pnl_table(pnl_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Format P&L data as displayable DataFrame.
    
    Args:
        pnl_data: Output from calculate_live_pnl()
    
    Returns:
        Formatted DataFrame
    """
    rows = []
    for ticker, data in pnl_data['positions'].items():
        rows.append({
            'Ticker': ticker,
            'Shares': data['shares'],
            'Cost Basis': f"${data['cost_basis']:.2f}",
            'Current': f"${data['current_price']:.2f}",
            'Cost': f"${data['cost']:,.2f}",
            'Value': f"${data['value']:,.2f}",
            'P&L': f"${data['pnl']:,.2f}",
            'P&L %': f"{data['pnl_pct']:+.2f}%"
        })
    
    df = pd.DataFrame(rows)
    
    # Add totals row
    if len(rows) > 0:
        totals = pd.DataFrame([{
            'Ticker': 'TOTAL',
            'Shares': '',
            'Cost Basis': '',
            'Current': '',
            'Cost': f"${pnl_data['total_cost']:,.2f}",
            'Value': f"${pnl_data['total_value']:,.2f}",
            'P&L': f"${pnl_data['total_pnl']:,.2f}",
            'P&L %': f"{pnl_data['total_pnl_pct']:+.2f}%"
        }])
        df = pd.concat([df, totals], ignore_index=True)
    
    return df

# ============================================================================
# STREAMING & AUTO-REFRESH
# ============================================================================

class PriceStream:
    """
    Price streaming utility for real-time updates.
    
    Usage:
        stream = PriceStream(['AAPL', 'MSFT'])
        stream.start()
        
        prices = stream.get_latest()
        
        stream.stop()
    """
    
    def __init__(self, tickers: List[str], interval: int = 10):
        self.tickers = tickers
        self.interval = interval  # seconds
        self.prices = {}
        self._running = False
        self._thread = None
        self._queue = queue.Queue()
    
    def _update_loop(self):
        """Background update loop."""
        while self._running:
            for ticker in self.tickers:
                try:
                    quote = get_live_quote(ticker)
                    self.prices[ticker] = quote
                    self._queue.put((ticker, quote))
                except Exception as e:
                    logger.error(f"Stream error for {ticker}: {e}")
            
            threading.Event().wait(self.interval)
    
    def start(self):
        """Start price streaming."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()
        logger.info(f"Started price stream for {self.tickers}")
    
    def stop(self):
        """Stop price streaming."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        logger.info("Stopped price stream")
    
    def get_latest(self) -> Dict[str, Dict]:
        """Get latest prices."""
        return self.prices.copy()
    
    def get_updates(self) -> List[tuple]:
        """Get queued updates."""
        updates = []
        while not self._queue.empty():
            try:
                updates.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return updates

def stream_price_updates(tickers: List[str], container, interval: int = 30):
    """
    Stream price updates to a Streamlit container.
    
    Args:
        tickers: List of ticker symbols
        container: Streamlit container for updates
        interval: Update interval in seconds
    """
    import time
    
    while True:
        quotes = get_multiple_quotes(tickers)
        
        with container:
            cols = st.columns(len(tickers))
            for i, ticker in enumerate(tickers):
                quote = quotes[ticker]
                if quote['price']:
                    delta = f"{quote['change_pct']:.2f}%" if quote['change_pct'] else None
                    cols[i].metric(
                        ticker, 
                        f"${quote['price']:.2f}",
                        delta=delta
                    )
        
        time.sleep(interval)

# ============================================================================
# REALTIME ENGINE CLASS
# ============================================================================

class RealtimeEngine:
    """
    Comprehensive real-time data engine.
    
    Usage:
        engine = RealtimeEngine()
        
        # Check market status
        status = engine.market_status()
        
        # Get live quote
        quote = engine.quote('AAPL')
        
        # Track P&L
        pnl = engine.calculate_pnl(positions)
    """
    
    def __init__(self, exchange: str = 'NYSE'):
        self.exchange = exchange
        self._streams = {}
    
    def market_status(self, include_extended: bool = False) -> Dict:
        """Get market status."""
        return is_market_open(self.exchange, include_extended)
    
    def quote(self, ticker: str) -> Dict:
        """Get live quote."""
        return get_live_quote(ticker)
    
    def quotes(self, tickers: List[str]) -> Dict[str, Dict]:
        """Get multiple quotes."""
        return get_multiple_quotes(tickers)
    
    def intraday(self, ticker: str, period: str = '1d', interval: str = '5m') -> pd.DataFrame:
        """Get intraday data."""
        return get_intraday_data(ticker, period, interval)
    
    def calculate_pnl(self, positions: Dict) -> Dict:
        """Calculate P&L for positions."""
        return calculate_live_pnl(positions)
    
    def pnl_table(self, positions: Dict) -> pd.DataFrame:
        """Get formatted P&L table."""
        pnl_data = self.calculate_pnl(positions)
        return format_pnl_table(pnl_data)
    
    def start_stream(self, tickers: List[str], interval: int = 10) -> PriceStream:
        """Start price streaming."""
        stream_key = tuple(sorted(tickers))
        if stream_key not in self._streams:
            self._streams[stream_key] = PriceStream(tickers, interval)
            self._streams[stream_key].start()
        return self._streams[stream_key]
    
    def stop_all_streams(self):
        """Stop all price streams."""
        for stream in self._streams.values():
            stream.stop()
        self._streams = {}
    
    def display_market_banner(self):
        """Display market status banner in Streamlit."""
        status = self.market_status(include_extended=True)
        
        if status['is_open']:
            if status['session'] == 'regular':
                st.success(status['message'])
            else:
                st.warning(status['message'])
        else:
            st.info(status['message'])
    
    def display_live_quotes(self, tickers: List[str]):
        """Display live quotes in Streamlit."""
        quotes = self.quotes(tickers)
        
        cols = st.columns(len(tickers))
        for i, ticker in enumerate(tickers):
            quote = quotes[ticker]
            if quote['price']:
                delta = f"{quote['change_pct']:.2f}%" if quote['change_pct'] else None
                cols[i].metric(
                    ticker,
                    f"${quote['price']:.2f}",
                    delta=delta
                )


# ============================================================================
# WEBSOCKET STREAMING FOR SUB-SECOND UPDATES
# ============================================================================

class WebSocketPriceStream:
    """
    WebSocket-based price streaming for sub-second updates.
    
    Supports multiple providers:
    - Polygon.io WebSocket
    - Alpaca WebSocket
    
    Usage:
        stream = WebSocketPriceStream(
            provider='polygon',
            api_key='your_api_key'
        )
        stream.connect()
        stream.subscribe(['AAPL', 'MSFT'], callback=handle_update)
        stream.start()
    """
    
    def __init__(
        self, 
        provider: str = 'polygon',
        api_key: str = "",
        api_secret: str = "",
        feed: str = 'iex'
    ):
        self.provider = provider
        self.api_key = api_key
        self.api_secret = api_secret
        self.feed = feed
        self.ws_client = None
        self.logger = logging.getLogger(__name__)
        self._running = False
        self._thread = None
        self._callbacks = {}
        self._latest_prices = {}
        self._update_queue = queue.Queue(maxsize=1000)
        
        # Statistics
        self.stats = {
            'messages_received': 0,
            'errors': 0,
            'reconnects': 0,
            'connected_since': None
        }
    
    def connect(self) -> bool:
        """Establish WebSocket connection."""
        try:
            if self.provider == 'polygon':
                from data_sources.polygon_provider import PolygonWebSocket
                self.ws_client = PolygonWebSocket(api_key=self.api_key)
            elif self.provider == 'alpaca':
                from data_sources.alpaca_provider import AlpacaWebSocket
                self.ws_client = AlpacaWebSocket(
                    api_key=self.api_key,
                    api_secret=self.api_secret,
                    feed=self.feed
                )
            else:
                self.logger.error(f"Unknown provider: {self.provider}")
                return False
            
            success = self.ws_client.connect()
            if success:
                self.stats['connected_since'] = datetime.now()
                self.logger.info(f"Connected to {self.provider} WebSocket")
            return success
            
        except ImportError as e:
            self.logger.error(f"Provider not available: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Connection failed: {e}")
            return False
    
    def subscribe(
        self, 
        tickers: List[str],
        channels: List[str] = None,
        callback: Callable = None
    ):
        """
        Subscribe to real-time data streams.
        
        Args:
            tickers: List of ticker symbols
            channels: Channel types (provider-specific)
            callback: Function to call with price updates
        """
        if not self.ws_client:
            self.logger.error("Not connected. Call connect() first.")
            return
        
        # Store callback
        if callback:
            for ticker in tickers:
                self._callbacks[ticker.upper()] = callback
        
        # Subscribe based on provider
        if self.provider == 'polygon':
            channels = channels or ['T', 'Q', 'A']
            self.ws_client.subscribe(tickers, channels, callback=self._handle_update)
        elif self.provider == 'alpaca':
            self.ws_client.subscribe(
                tickers,
                trades=True,
                quotes=True,
                bars=False,
                callbacks={
                    'trades': self._handle_update,
                    'quotes': self._handle_update
                }
            )
    
    def _handle_update(self, data: Dict):
        """Process incoming price update."""
        try:
            self.stats['messages_received'] += 1
            
            # Parse based on provider format
            if self.provider == 'polygon':
                ticker = data.get('sym', '')
                price = data.get('p', data.get('c'))  # Trade price or close
                volume = data.get('s', data.get('v', 0))  # Size or volume
            elif self.provider == 'alpaca':
                ticker = data.get('S', '')
                price = data.get('p', data.get('c'))
                volume = data.get('s', data.get('v', 0))
            else:
                return
            
            if ticker and price:
                update = {
                    'ticker': ticker,
                    'price': float(price),
                    'volume': volume,
                    'timestamp': datetime.now(),
                    'source': self.provider
                }
                
                # Update latest prices
                self._latest_prices[ticker] = update
                
                # Queue for external consumption
                try:
                    self._update_queue.put_nowait(update)
                except queue.Full:
                    pass
                
                # Call registered callback
                if ticker in self._callbacks:
                    self._callbacks[ticker](update)
                    
        except Exception as e:
            self.stats['errors'] += 1
            self.logger.error(f"Update processing error: {e}")
    
    def start(self, blocking: bool = False):
        """
        Start listening for updates.
        
        Args:
            blocking: If True, blocks current thread. If False, runs in background.
        """
        if not self.ws_client:
            self.logger.error("Not connected. Call connect() first.")
            return
        
        self._running = True
        
        if blocking:
            self.ws_client.listen()
        else:
            self._thread = threading.Thread(target=self.ws_client.listen, daemon=True)
            self._thread.start()
            self.logger.info("WebSocket streaming started in background")
    
    def stop(self):
        """Stop the WebSocket stream."""
        self._running = False
        if self.ws_client:
            self.ws_client.close()
        self.logger.info("WebSocket stream stopped")
    
    def get_latest_prices(self) -> Dict[str, Dict]:
        """Get latest prices for all subscribed tickers."""
        return self._latest_prices.copy()
    
    def get_latest_price(self, ticker: str) -> Optional[Dict]:
        """Get latest price for a specific ticker."""
        return self._latest_prices.get(ticker.upper())
    
    def get_updates(self, max_count: int = 100) -> List[Dict]:
        """Get queued price updates."""
        updates = []
        count = 0
        while not self._update_queue.empty() and count < max_count:
            try:
                updates.append(self._update_queue.get_nowait())
                count += 1
            except queue.Empty:
                break
        return updates
    
    def get_stats(self) -> Dict:
        """Get streaming statistics."""
        return {
            **self.stats,
            'tickers_subscribed': len(self._callbacks),
            'is_running': self._running,
            'provider': self.provider
        }


class RealTimePriceAggregator:
    """
    Aggregates real-time prices from multiple sources with fallback.
    
    Combines WebSocket streams with polling for comprehensive coverage.
    
    Usage:
        aggregator = RealTimePriceAggregator()
        aggregator.add_websocket_source('polygon', api_key='...')
        aggregator.add_polling_source(['AAPL', 'MSFT'], interval=5)
        
        aggregator.start()
        
        # Get prices
        prices = aggregator.get_prices(['AAPL', 'MSFT'])
    """
    
    def __init__(self):
        self.ws_streams = {}
        self.polling_streams = {}
        self.prices = {}
        self.logger = logging.getLogger(__name__)
        self._running = False
    
    def add_websocket_source(
        self,
        name: str,
        provider: str,
        api_key: str,
        api_secret: str = "",
        tickers: List[str] = None
    ):
        """Add a WebSocket data source."""
        stream = WebSocketPriceStream(
            provider=provider,
            api_key=api_key,
            api_secret=api_secret
        )
        self.ws_streams[name] = {
            'stream': stream,
            'tickers': tickers or []
        }
    
    def add_polling_source(
        self,
        tickers: List[str],
        interval: int = 10
    ):
        """Add a polling-based data source."""
        stream = PriceStream(tickers, interval)
        self.polling_streams[tuple(tickers)] = stream
    
    def start(self):
        """Start all data sources."""
        self._running = True
        
        # Start WebSocket streams
        for name, ws_info in self.ws_streams.items():
            stream = ws_info['stream']
            tickers = ws_info['tickers']
            
            if stream.connect():
                if tickers:
                    stream.subscribe(tickers, callback=self._handle_update)
                stream.start(blocking=False)
                self.logger.info(f"Started WebSocket stream: {name}")
        
        # Start polling streams
        for tickers, stream in self.polling_streams.items():
            stream.start()
            self.logger.info(f"Started polling stream for: {tickers}")
    
    def stop(self):
        """Stop all data sources."""
        self._running = False
        
        for ws_info in self.ws_streams.values():
            ws_info['stream'].stop()
        
        for stream in self.polling_streams.values():
            stream.stop()
    
    def _handle_update(self, update: Dict):
        """Handle price update from any source."""
        ticker = update.get('ticker', '')
        if ticker:
            self.prices[ticker] = {
                'price': update.get('price'),
                'volume': update.get('volume'),
                'timestamp': update.get('timestamp', datetime.now()),
                'source': update.get('source', 'unknown')
            }
    
    def get_prices(self, tickers: List[str] = None) -> Dict[str, Dict]:
        """
        Get current prices.
        
        Args:
            tickers: Optional list of tickers (returns all if None)
        
        Returns:
            Dictionary of prices
        """
        if tickers is None:
            return self.prices.copy()
        
        result = {}
        for ticker in tickers:
            ticker_upper = ticker.upper()
            if ticker_upper in self.prices:
                result[ticker_upper] = self.prices[ticker_upper]
            else:
                # Fallback to direct quote
                quote = get_live_quote(ticker)
                result[ticker_upper] = {
                    'price': quote.get('price'),
                    'volume': quote.get('volume'),
                    'timestamp': datetime.now(),
                    'source': 'fallback'
                }
        
        return result
    
    def get_price(self, ticker: str) -> Optional[float]:
        """Get current price for a single ticker."""
        prices = self.get_prices([ticker])
        return prices.get(ticker.upper(), {}).get('price')


# ============================================================================
# STREAMLIT INTEGRATION
# ============================================================================

def create_realtime_dashboard(
    tickers: List[str],
    container,
    refresh_interval: int = 5
):
    """
    Create a real-time price dashboard in Streamlit.
    
    Args:
        tickers: List of ticker symbols
        container: Streamlit container
        refresh_interval: Refresh interval in seconds
    """
    import time
    
    # Initialize session state
    if 'realtime_prices' not in st.session_state:
        st.session_state.realtime_prices = {}
    
    while True:
        prices = {}
        for ticker in tickers:
            quote = get_live_quote(ticker)
            prices[ticker] = quote
        
        st.session_state.realtime_prices = prices
        
        with container:
            cols = st.columns(len(tickers))
            for i, ticker in enumerate(tickers):
                quote = prices[ticker]
                price = quote.get('price', 0)
                change_pct = quote.get('change_pct', 0)
                
                delta_color = "normal" if change_pct >= 0 else "inverse"
                
                cols[i].metric(
                    label=ticker,
                    value=f"${price:.2f}" if price else "N/A",
                    delta=f"{change_pct:.2f}%" if change_pct else None,
                    delta_color=delta_color
                )
        
        time.sleep(refresh_interval)


def display_websocket_status(stream: WebSocketPriceStream):
    """Display WebSocket connection status in Streamlit."""
    stats = stream.get_stats()
    
    if stats.get('connected_since'):
        uptime = datetime.now() - stats['connected_since']
        st.success(f"🟢 Connected ({uptime.seconds}s)")
    else:
        st.error("🔴 Disconnected")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Messages", stats['messages_received'])
    col2.metric("Errors", stats['errors'])
    col3.metric("Subscribed", stats['tickers_subscribed'])
