# Data Sources Module
from .data_aggregator import DataAggregator
from .yahoo_provider import YahooProvider
from .alpha_vantage_provider import AlphaVantageProvider
from .base_provider import BaseDataProvider

# Import new professional providers (optional)
try:
    from .polygon_provider import PolygonProvider, PolygonWebSocket
    HAS_POLYGON = True
except ImportError:
    PolygonProvider = None
    PolygonWebSocket = None
    HAS_POLYGON = False

try:
    from .alpaca_provider import AlpacaProvider, AlpacaWebSocket
    HAS_ALPACA = True
except ImportError:
    AlpacaProvider = None
    AlpacaWebSocket = None
    HAS_ALPACA = False

__all__ = [
    'DataAggregator', 
    'YahooProvider', 
    'AlphaVantageProvider',
    'BaseDataProvider',
    'PolygonProvider',
    'PolygonWebSocket',
    'AlpacaProvider',
    'AlpacaWebSocket',
    'HAS_POLYGON',
    'HAS_ALPACA'
]
