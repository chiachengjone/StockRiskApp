"""
Instrument Metadata Handler (v4.3 Professional)
================================================
Unified metadata structures for ETFs, Options, and Complex Instruments.

Provides standardized interfaces for:
- ETF Duration & Yield attributes
- Options Greeks integration
- Fixed Income durations
- Complex instrument decomposition
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from datetime import datetime, date
import logging
import numpy as np


class AssetClass(Enum):
    """Asset classification taxonomy."""
    EQUITY = "equity"
    ETF = "etf"
    FIXED_INCOME = "fixed_income"
    OPTION = "option"
    FUTURE = "future"
    FOREX = "forex"
    CRYPTO = "crypto"
    COMMODITY = "commodity"
    REIT = "reit"
    HYBRID = "hybrid"


class ETFCategory(Enum):
    """ETF classification by underlying exposure."""
    BROAD_EQUITY = "broad_equity"
    SECTOR = "sector"
    STYLE = "style"  # Growth, Value, etc.
    SIZE = "size"    # Large, Mid, Small cap
    FIXED_INCOME = "fixed_income"
    COMMODITY = "commodity"
    CURRENCY = "currency"
    LEVERAGED = "leveraged"
    INVERSE = "inverse"
    THEMATIC = "thematic"
    INTERNATIONAL = "international"
    ALTERNATIVE = "alternative"


@dataclass
class ETFMetadata:
    """
    Standardized ETF attributes for risk analysis.
    
    Captures duration, yield, leverage, and underlying exposure.
    """
    ticker: str
    name: str
    category: ETFCategory
    
    # Core metrics
    expense_ratio: float = 0.0
    aum: float = 0.0  # Assets under management (millions)
    avg_daily_volume: float = 0.0
    
    # Duration attributes (primarily for bond ETFs)
    effective_duration: Optional[float] = None  # Interest rate sensitivity
    modified_duration: Optional[float] = None
    option_adjusted_duration: Optional[float] = None
    
    # Yield attributes
    distribution_yield: Optional[float] = None  # 12-month yield
    sec_yield: Optional[float] = None  # 30-day SEC yield
    yield_to_maturity: Optional[float] = None  # For bond ETFs
    
    # Credit attributes (bond ETFs)
    average_credit_quality: Optional[str] = None  # AAA, AA, A, BBB, etc.
    credit_spread_duration: Optional[float] = None
    
    # Leverage/Inverse attributes
    leverage_factor: float = 1.0  # 2x, 3x, -1x, -2x, -3x
    is_leveraged: bool = False
    is_inverse: bool = False
    
    # Underlying exposure
    underlying_index: Optional[str] = None
    sector_weights: Dict[str, float] = field(default_factory=dict)
    country_weights: Dict[str, float] = field(default_factory=dict)
    top_holdings: List[Dict[str, Any]] = field(default_factory=list)
    
    # Risk attributes
    beta_to_benchmark: Optional[float] = None
    tracking_error: Optional[float] = None
    
    # Timestamps
    data_date: Optional[date] = None
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'ticker': self.ticker,
            'name': self.name,
            'asset_class': 'ETF',
            'category': self.category.value,
            'expense_ratio': self.expense_ratio,
            'aum_millions': self.aum,
            'avg_daily_volume': self.avg_daily_volume,
            'duration': {
                'effective': self.effective_duration,
                'modified': self.modified_duration,
                'option_adjusted': self.option_adjusted_duration,
                'credit_spread': self.credit_spread_duration
            },
            'yield': {
                'distribution': self.distribution_yield,
                'sec_30day': self.sec_yield,
                'ytm': self.yield_to_maturity
            },
            'credit': {
                'average_quality': self.average_credit_quality
            },
            'leverage': {
                'factor': self.leverage_factor,
                'is_leveraged': self.is_leveraged,
                'is_inverse': self.is_inverse
            },
            'exposure': {
                'underlying_index': self.underlying_index,
                'sector_weights': self.sector_weights,
                'country_weights': self.country_weights,
                'top_holdings': self.top_holdings
            },
            'risk': {
                'beta': self.beta_to_benchmark,
                'tracking_error': self.tracking_error
            },
            'data_date': self.data_date.isoformat() if self.data_date else None,
            'last_updated': self.last_updated.isoformat()
        }
    
    @property
    def risk_adjusted_duration(self) -> Optional[float]:
        """Duration adjusted for leverage factor."""
        if self.effective_duration is not None:
            return self.effective_duration * abs(self.leverage_factor)
        return None
    
    @property
    def is_bond_etf(self) -> bool:
        """Check if this is a fixed income ETF."""
        return self.category == ETFCategory.FIXED_INCOME


@dataclass
class OptionMetadata:
    """
    Standardized option attributes with Greeks integration.
    
    Provides unified interface for options risk analysis.
    """
    underlying_ticker: str
    option_symbol: str
    option_type: str  # 'call' or 'put'
    
    # Contract specs
    strike_price: float = 0.0
    expiration_date: Optional[date] = None
    contract_size: int = 100
    
    # Pricing
    last_price: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    implied_volatility: float = 0.0
    
    # Greeks (pre-calculated or calculated on demand)
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    rho: Optional[float] = None
    
    # Additional Greeks
    vanna: Optional[float] = None  # d(delta)/d(vol)
    charm: Optional[float] = None  # d(delta)/d(time)
    vomma: Optional[float] = None  # d(vega)/d(vol)
    
    # Volume & OI
    volume: int = 0
    open_interest: int = 0
    
    # Underlying context
    underlying_price: float = 0.0
    days_to_expiry: int = 0
    
    # Timestamps
    data_time: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'underlying': self.underlying_ticker,
            'symbol': self.option_symbol,
            'asset_class': 'OPTION',
            'type': self.option_type,
            'strike': self.strike_price,
            'expiration': self.expiration_date.isoformat() if self.expiration_date else None,
            'contract_size': self.contract_size,
            'pricing': {
                'last': self.last_price,
                'bid': self.bid,
                'ask': self.ask,
                'mid': (self.bid + self.ask) / 2 if self.bid and self.ask else self.last_price,
                'iv': self.implied_volatility
            },
            'greeks': {
                'delta': self.delta,
                'gamma': self.gamma,
                'theta': self.theta,
                'vega': self.vega,
                'rho': self.rho,
                'vanna': self.vanna,
                'charm': self.charm,
                'vomma': self.vomma
            },
            'liquidity': {
                'volume': self.volume,
                'open_interest': self.open_interest,
                'bid_ask_spread': (self.ask - self.bid) if self.bid and self.ask else None
            },
            'context': {
                'underlying_price': self.underlying_price,
                'days_to_expiry': self.days_to_expiry,
                'moneyness': self._calculate_moneyness()
            },
            'data_time': self.data_time.isoformat()
        }
    
    def _calculate_moneyness(self) -> str:
        """Determine if option is ITM, ATM, or OTM."""
        if self.underlying_price <= 0 or self.strike_price <= 0:
            return "unknown"
        
        ratio = self.underlying_price / self.strike_price
        
        if self.option_type.lower() == 'call':
            if ratio > 1.02:
                return "ITM"
            elif ratio < 0.98:
                return "OTM"
            else:
                return "ATM"
        else:  # put
            if ratio < 0.98:
                return "ITM"
            elif ratio > 1.02:
                return "OTM"
            else:
                return "ATM"
    
    @property
    def dollar_delta(self) -> Optional[float]:
        """Position delta in dollar terms."""
        if self.delta is not None and self.underlying_price > 0:
            return self.delta * self.underlying_price * self.contract_size
        return None
    
    @property
    def notional_value(self) -> float:
        """Notional exposure of the option position."""
        return self.underlying_price * self.contract_size


@dataclass
class EquityMetadata:
    """Standardized equity/stock metadata."""
    ticker: str
    name: str
    sector: Optional[str] = None
    industry: Optional[str] = None
    
    # Market data
    market_cap: float = 0.0
    shares_outstanding: float = 0.0
    avg_daily_volume: float = 0.0
    float_shares: float = 0.0
    
    # Fundamentals
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    eps: Optional[float] = None
    revenue: Optional[float] = None
    
    # Risk metrics
    beta: Optional[float] = None
    volatility_252d: Optional[float] = None
    
    # Short interest
    short_interest: Optional[float] = None
    days_to_cover: Optional[float] = None
    
    # Timestamps
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'ticker': self.ticker,
            'name': self.name,
            'asset_class': 'EQUITY',
            'classification': {
                'sector': self.sector,
                'industry': self.industry
            },
            'market_data': {
                'market_cap': self.market_cap,
                'shares_outstanding': self.shares_outstanding,
                'avg_daily_volume': self.avg_daily_volume,
                'float_shares': self.float_shares
            },
            'fundamentals': {
                'pe_ratio': self.pe_ratio,
                'pb_ratio': self.pb_ratio,
                'dividend_yield': self.dividend_yield,
                'eps': self.eps,
                'revenue': self.revenue
            },
            'risk': {
                'beta': self.beta,
                'volatility_252d': self.volatility_252d
            },
            'short_interest': {
                'short_interest': self.short_interest,
                'days_to_cover': self.days_to_cover
            },
            'last_updated': self.last_updated.isoformat()
        }


# =============================================================================
# ETF METADATA DATABASE (Common ETFs with pre-populated attributes)
# =============================================================================

ETF_METADATA_DB: Dict[str, ETFMetadata] = {
    # Broad Market ETFs
    'SPY': ETFMetadata(
        ticker='SPY',
        name='SPDR S&P 500 ETF Trust',
        category=ETFCategory.BROAD_EQUITY,
        expense_ratio=0.0945,
        underlying_index='S&P 500'
    ),
    'QQQ': ETFMetadata(
        ticker='QQQ',
        name='Invesco QQQ Trust',
        category=ETFCategory.BROAD_EQUITY,
        expense_ratio=0.20,
        underlying_index='NASDAQ-100'
    ),
    'IWM': ETFMetadata(
        ticker='IWM',
        name='iShares Russell 2000 ETF',
        category=ETFCategory.SIZE,
        expense_ratio=0.19,
        underlying_index='Russell 2000'
    ),
    'DIA': ETFMetadata(
        ticker='DIA',
        name='SPDR Dow Jones Industrial Average ETF',
        category=ETFCategory.BROAD_EQUITY,
        expense_ratio=0.16,
        underlying_index='Dow Jones Industrial Average'
    ),
    
    # Sector ETFs
    'XLF': ETFMetadata(
        ticker='XLF',
        name='Financial Select Sector SPDR',
        category=ETFCategory.SECTOR,
        expense_ratio=0.10,
        sector_weights={'Financials': 1.0}
    ),
    'XLK': ETFMetadata(
        ticker='XLK',
        name='Technology Select Sector SPDR',
        category=ETFCategory.SECTOR,
        expense_ratio=0.10,
        sector_weights={'Technology': 1.0}
    ),
    'XLE': ETFMetadata(
        ticker='XLE',
        name='Energy Select Sector SPDR',
        category=ETFCategory.SECTOR,
        expense_ratio=0.10,
        sector_weights={'Energy': 1.0}
    ),
    'XLV': ETFMetadata(
        ticker='XLV',
        name='Health Care Select Sector SPDR',
        category=ETFCategory.SECTOR,
        expense_ratio=0.10,
        sector_weights={'Healthcare': 1.0}
    ),
    
    # Bond ETFs with Duration
    'TLT': ETFMetadata(
        ticker='TLT',
        name='iShares 20+ Year Treasury Bond ETF',
        category=ETFCategory.FIXED_INCOME,
        expense_ratio=0.15,
        effective_duration=17.5,
        yield_to_maturity=4.5,
        average_credit_quality='AAA'
    ),
    'IEF': ETFMetadata(
        ticker='IEF',
        name='iShares 7-10 Year Treasury Bond ETF',
        category=ETFCategory.FIXED_INCOME,
        expense_ratio=0.15,
        effective_duration=7.5,
        yield_to_maturity=4.2,
        average_credit_quality='AAA'
    ),
    'SHY': ETFMetadata(
        ticker='SHY',
        name='iShares 1-3 Year Treasury Bond ETF',
        category=ETFCategory.FIXED_INCOME,
        expense_ratio=0.15,
        effective_duration=1.9,
        yield_to_maturity=4.8,
        average_credit_quality='AAA'
    ),
    'LQD': ETFMetadata(
        ticker='LQD',
        name='iShares iBoxx Investment Grade Corporate Bond ETF',
        category=ETFCategory.FIXED_INCOME,
        expense_ratio=0.14,
        effective_duration=8.5,
        yield_to_maturity=5.2,
        average_credit_quality='A',
        credit_spread_duration=8.0
    ),
    'HYG': ETFMetadata(
        ticker='HYG',
        name='iShares iBoxx High Yield Corporate Bond ETF',
        category=ETFCategory.FIXED_INCOME,
        expense_ratio=0.48,
        effective_duration=3.8,
        yield_to_maturity=7.5,
        average_credit_quality='BB',
        credit_spread_duration=3.5
    ),
    'AGG': ETFMetadata(
        ticker='AGG',
        name='iShares Core U.S. Aggregate Bond ETF',
        category=ETFCategory.FIXED_INCOME,
        expense_ratio=0.03,
        effective_duration=6.2,
        yield_to_maturity=4.8,
        average_credit_quality='AA'
    ),
    
    # Leveraged & Inverse ETFs
    'TQQQ': ETFMetadata(
        ticker='TQQQ',
        name='ProShares UltraPro QQQ',
        category=ETFCategory.LEVERAGED,
        expense_ratio=0.86,
        leverage_factor=3.0,
        is_leveraged=True,
        underlying_index='NASDAQ-100'
    ),
    'SQQQ': ETFMetadata(
        ticker='SQQQ',
        name='ProShares UltraPro Short QQQ',
        category=ETFCategory.INVERSE,
        expense_ratio=0.95,
        leverage_factor=-3.0,
        is_leveraged=True,
        is_inverse=True,
        underlying_index='NASDAQ-100'
    ),
    'SPXU': ETFMetadata(
        ticker='SPXU',
        name='ProShares UltraPro Short S&P500',
        category=ETFCategory.INVERSE,
        expense_ratio=0.90,
        leverage_factor=-3.0,
        is_leveraged=True,
        is_inverse=True,
        underlying_index='S&P 500'
    ),
    'TMF': ETFMetadata(
        ticker='TMF',
        name='Direxion Daily 20+ Year Treasury Bull 3X',
        category=ETFCategory.LEVERAGED,
        expense_ratio=1.00,
        leverage_factor=3.0,
        is_leveraged=True,
        effective_duration=52.5,  # 3x TLT duration
        underlying_index='ICE U.S. Treasury 20+ Year Bond'
    ),
    
    # International ETFs
    'EFA': ETFMetadata(
        ticker='EFA',
        name='iShares MSCI EAFE ETF',
        category=ETFCategory.INTERNATIONAL,
        expense_ratio=0.32,
        underlying_index='MSCI EAFE',
        country_weights={'Japan': 0.22, 'UK': 0.15, 'France': 0.11, 'Switzerland': 0.10}
    ),
    'EEM': ETFMetadata(
        ticker='EEM',
        name='iShares MSCI Emerging Markets ETF',
        category=ETFCategory.INTERNATIONAL,
        expense_ratio=0.68,
        underlying_index='MSCI Emerging Markets',
        country_weights={'China': 0.30, 'Taiwan': 0.16, 'India': 0.14, 'S. Korea': 0.12}
    ),
    
    # Commodity ETFs
    'GLD': ETFMetadata(
        ticker='GLD',
        name='SPDR Gold Shares',
        category=ETFCategory.COMMODITY,
        expense_ratio=0.40
    ),
    'USO': ETFMetadata(
        ticker='USO',
        name='United States Oil Fund',
        category=ETFCategory.COMMODITY,
        expense_ratio=0.73
    )
}


class InstrumentMetadataHandler:
    """
    Unified handler for all instrument metadata.
    
    Provides consistent interface for retrieving and standardizing
    metadata across different asset classes.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._etf_cache: Dict[str, ETFMetadata] = ETF_METADATA_DB.copy()
        self._equity_cache: Dict[str, EquityMetadata] = {}
        self._option_cache: Dict[str, OptionMetadata] = {}
    
    def get_etf_metadata(self, ticker: str) -> Optional[ETFMetadata]:
        """
        Retrieve ETF metadata by ticker.
        
        Args:
            ticker: ETF ticker symbol
        
        Returns:
            ETFMetadata object or None if not found
        """
        ticker = ticker.upper()
        if ticker in self._etf_cache:
            return self._etf_cache[ticker]
        return None
    
    def get_instrument_metadata(
        self,
        ticker: str,
        asset_class: Optional[AssetClass] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get unified metadata for any instrument.
        
        Args:
            ticker: Instrument ticker/symbol
            asset_class: Optional asset class hint
        
        Returns:
            Dictionary with standardized metadata
        """
        ticker = ticker.upper()
        
        # Check ETF database first
        if ticker in self._etf_cache:
            return self._etf_cache[ticker].to_dict()
        
        # Check equity cache
        if ticker in self._equity_cache:
            return self._equity_cache[ticker].to_dict()
        
        # Return basic structure if not found
        return {
            'ticker': ticker,
            'asset_class': asset_class.value if asset_class else 'unknown',
            'metadata_available': False
        }
    
    def enrich_with_option_greeks(
        self,
        option_meta: OptionMetadata,
        greeks: Dict[str, float]
    ) -> OptionMetadata:
        """
        Enrich option metadata with calculated Greeks.
        
        Args:
            option_meta: Base option metadata
            greeks: Dictionary of Greek values from options analytics
        
        Returns:
            Updated OptionMetadata with Greeks
        """
        option_meta.delta = greeks.get('delta')
        option_meta.gamma = greeks.get('gamma')
        option_meta.theta = greeks.get('theta')
        option_meta.vega = greeks.get('vega')
        option_meta.rho = greeks.get('rho')
        return option_meta
    
    def get_etf_duration_risk(
        self,
        ticker: str,
        rate_change_bps: float = 100
    ) -> Dict[str, Any]:
        """
        Calculate interest rate risk for a bond ETF.
        
        Args:
            ticker: ETF ticker
            rate_change_bps: Rate change in basis points (default 100bp = 1%)
        
        Returns:
            Dictionary with duration-based risk estimates
        """
        etf = self.get_etf_metadata(ticker)
        if not etf or not etf.is_bond_etf:
            return {'error': f'{ticker} is not a bond ETF or not found'}
        
        duration = etf.effective_duration or 0
        rate_change_pct = rate_change_bps / 100
        
        # Price change estimate: ΔP ≈ -Duration × ΔY
        price_change_pct = -duration * rate_change_pct
        
        # Convexity adjustment (approximation)
        convexity_adj = 0.5 * (duration ** 2) * (rate_change_pct ** 2)
        adjusted_price_change = price_change_pct + convexity_adj
        
        return {
            'ticker': ticker,
            'effective_duration': duration,
            'rate_change_bps': rate_change_bps,
            'estimated_price_change_pct': adjusted_price_change,
            'leverage_adjusted': adjusted_price_change * abs(etf.leverage_factor),
            'credit_spread_duration': etf.credit_spread_duration,
            'average_credit_quality': etf.average_credit_quality
        }
    
    def classify_instrument(self, ticker: str) -> AssetClass:
        """
        Attempt to classify an instrument by ticker.
        
        Args:
            ticker: Instrument ticker
        
        Returns:
            Best-guess AssetClass
        """
        ticker = ticker.upper()
        
        # Check if it's a known ETF
        if ticker in self._etf_cache:
            return AssetClass.ETF
        
        # Common crypto tickers
        crypto_tickers = {'BTC', 'ETH', 'SOL', 'DOGE', 'ADA', 'XRP', 'BNB'}
        if ticker in crypto_tickers or ticker.endswith('-USD'):
            return AssetClass.CRYPTO
        
        # Forex patterns
        if len(ticker) == 6 and ticker[:3].isalpha() and ticker[3:].isalpha():
            return AssetClass.FOREX
        
        # Default to equity
        return AssetClass.EQUITY
    
    def register_etf(self, etf_meta: ETFMetadata) -> None:
        """Register or update an ETF in the cache."""
        self._etf_cache[etf_meta.ticker.upper()] = etf_meta
    
    def register_equity(self, equity_meta: EquityMetadata) -> None:
        """Register or update an equity in the cache."""
        self._equity_cache[equity_meta.ticker.upper()] = equity_meta
    
    def get_all_etf_tickers(self) -> List[str]:
        """Get list of all registered ETF tickers."""
        return list(self._etf_cache.keys())
    
    def get_leveraged_etfs(self) -> List[ETFMetadata]:
        """Get all leveraged/inverse ETFs."""
        return [
            etf for etf in self._etf_cache.values()
            if etf.is_leveraged or etf.is_inverse
        ]
    
    def get_bond_etfs(self) -> List[ETFMetadata]:
        """Get all fixed income ETFs."""
        return [
            etf for etf in self._etf_cache.values()
            if etf.is_bond_etf
        ]
    
    def get_portfolio_duration(
        self,
        holdings: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Calculate weighted portfolio duration for bond ETF holdings.
        
        Args:
            holdings: Dict of {ticker: weight}
        
        Returns:
            Portfolio duration metrics
        """
        total_duration = 0.0
        total_weight = 0.0
        duration_contributions = {}
        
        for ticker, weight in holdings.items():
            etf = self.get_etf_metadata(ticker)
            if etf and etf.effective_duration is not None:
                contrib = weight * etf.effective_duration
                duration_contributions[ticker] = {
                    'weight': weight,
                    'duration': etf.effective_duration,
                    'contribution': contrib
                }
                total_duration += contrib
                total_weight += weight
        
        return {
            'portfolio_duration': total_duration,
            'weighted_coverage': total_weight,
            'contributions': duration_contributions,
            'rate_sensitivity_1pct': -total_duration * 1.0  # Price change for 1% rate rise
        }


# Singleton instance for easy access
_metadata_handler: Optional[InstrumentMetadataHandler] = None


def get_metadata_handler() -> InstrumentMetadataHandler:
    """Get the singleton metadata handler instance."""
    global _metadata_handler
    if _metadata_handler is None:
        _metadata_handler = InstrumentMetadataHandler()
    return _metadata_handler
