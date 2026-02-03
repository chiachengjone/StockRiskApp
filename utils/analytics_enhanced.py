"""
Enhanced Analytics Module v2.0
==============================
Risk Score | Historical Scenarios | Sector Exposure | VaR Backtesting Dashboard
Performance Attribution | Advanced Analytics

Author: Stock Risk App | Feb 2026
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm, chi2
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class RiskScoreResult:
    """Result of unified risk score calculation."""
    total_score: float  # 0-100, higher = riskier
    risk_level: str  # Low, Moderate, High, Very High, Extreme
    components: Dict[str, float]  # Individual component scores
    weights: Dict[str, float]  # Weights used
    recommendations: List[str]  # Actionable recommendations
    grade: str  # A, B, C, D, F grade
    color: str  # Color code for UI


@dataclass
class ScenarioResult:
    """Result of historical scenario replay."""
    scenario_name: str
    start_date: str
    end_date: str
    market_return: float
    portfolio_return: float
    max_drawdown: float
    recovery_days: int
    var_breach: bool
    description: str
    severity: str  # Low, Medium, High, Extreme


@dataclass
class SectorExposure:
    """Sector exposure analysis result."""
    sector_weights: Dict[str, float]
    sector_risk_contribution: Dict[str, float]
    sector_correlation: pd.DataFrame
    concentration_index: float  # HHI
    diversification_score: float
    dominant_sector: str


@dataclass
class PerformanceAttribution:
    """Performance attribution breakdown."""
    total_return: float
    market_contribution: float
    factor_contributions: Dict[str, float]
    alpha: float
    selection_effect: float
    allocation_effect: float
    interaction_effect: float
    residual: float


@dataclass
class VaRBacktestResult:
    """VaR backtesting result for dashboard."""
    model_name: str
    violations: int
    total_observations: int
    violation_rate: float
    expected_rate: float
    kupiec_stat: float
    kupiec_pvalue: float
    christoffersen_stat: float
    christoffersen_pvalue: float
    model_adequate: bool
    assessment: str
    violation_dates: List[str]
    var_series: pd.Series
    return_series: pd.Series


# =============================================================================
# HISTORICAL SCENARIO LIBRARY
# =============================================================================

HISTORICAL_SCENARIOS = {
    'COVID-19 Crash': {
        'start': '2020-02-19',
        'end': '2020-03-23',
        'description': 'Pandemic-driven global market crash',
        'market_return': -0.34,
        'sectors_hit': ['Travel', 'Energy', 'Financials'],
        'sectors_safe': ['Tech', 'Healthcare', 'Consumer Staples'],
        'severity': 'Extreme'
    },
    '2008 Financial Crisis': {
        'start': '2007-10-09',
        'end': '2009-03-09',
        'description': 'Global financial crisis triggered by subprime mortgages',
        'market_return': -0.57,
        'sectors_hit': ['Financials', 'Real Estate', 'Consumer Discretionary'],
        'sectors_safe': ['Consumer Staples', 'Healthcare', 'Utilities'],
        'severity': 'Extreme'
    },
    'Dot-Com Bubble': {
        'start': '2000-03-10',
        'end': '2002-10-09',
        'description': 'Tech bubble burst and prolonged bear market',
        'market_return': -0.49,
        'sectors_hit': ['Technology', 'Telecom', 'Internet'],
        'sectors_safe': ['Consumer Staples', 'Utilities', 'Healthcare'],
        'severity': 'Extreme'
    },
    'Black Monday 1987': {
        'start': '1987-10-14',
        'end': '1987-10-19',
        'description': 'Single-day market crash of 22%+',
        'market_return': -0.22,
        'sectors_hit': ['All sectors'],
        'sectors_safe': [],
        'severity': 'High'
    },
    'Tech Wreck 2022': {
        'start': '2021-11-19',
        'end': '2022-10-13',
        'description': 'Rate hike driven tech selloff',
        'market_return': -0.27,
        'sectors_hit': ['Technology', 'Growth', 'Crypto'],
        'sectors_safe': ['Energy', 'Utilities', 'Value'],
        'severity': 'High'
    },
    'Flash Crash 2010': {
        'start': '2010-05-06',
        'end': '2010-05-06',
        'description': 'Intraday crash and recovery',
        'market_return': -0.09,
        'sectors_hit': ['All sectors'],
        'sectors_safe': [],
        'severity': 'Medium'
    },
    'Greek Debt Crisis': {
        'start': '2011-07-22',
        'end': '2011-10-03',
        'description': 'European sovereign debt crisis',
        'market_return': -0.19,
        'sectors_hit': ['Financials', 'European Exposure'],
        'sectors_safe': ['US Domestic', 'Healthcare'],
        'severity': 'Medium'
    },
    'China Crash 2015': {
        'start': '2015-06-12',
        'end': '2015-08-25',
        'description': 'Chinese stock market bubble burst',
        'market_return': -0.12,
        'sectors_hit': ['Emerging Markets', 'Materials', 'Industrials'],
        'sectors_safe': ['US Domestic', 'Utilities'],
        'severity': 'Medium'
    },
    'Brexit Vote': {
        'start': '2016-06-23',
        'end': '2016-06-27',
        'description': 'UK referendum shock',
        'market_return': -0.05,
        'sectors_hit': ['UK Exposure', 'Financials'],
        'sectors_safe': ['US Domestic', 'Gold'],
        'severity': 'Low'
    },
    'COVID Delta Wave': {
        'start': '2021-09-02',
        'end': '2021-10-04',
        'description': 'Delta variant concerns',
        'market_return': -0.05,
        'sectors_hit': ['Travel', 'Reopening Plays'],
        'sectors_safe': ['Tech', 'Stay-at-Home'],
        'severity': 'Low'
    },
    'SVB Bank Crisis': {
        'start': '2023-03-08',
        'end': '2023-03-13',
        'description': 'Regional banking crisis',
        'market_return': -0.04,
        'sectors_hit': ['Regional Banks', 'Financials'],
        'sectors_safe': ['Large-Cap Tech', 'Utilities'],
        'severity': 'Medium'
    },
    'Volcker Shock': {
        'start': '1981-01-01',
        'end': '1982-08-12',
        'description': 'Fed rate hikes to combat inflation',
        'market_return': -0.27,
        'sectors_hit': ['Real Estate', 'Industrials', 'Financials'],
        'sectors_safe': ['Energy', 'Commodities'],
        'severity': 'High'
    }
}


# Sector mapping for common tickers
SECTOR_MAP = {
    # Technology
    'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'GOOG': 'Technology',
    'META': 'Technology', 'NVDA': 'Technology', 'AMD': 'Technology', 'INTC': 'Technology',
    'CRM': 'Technology', 'ADBE': 'Technology', 'ORCL': 'Technology', 'CSCO': 'Technology',
    'IBM': 'Technology', 'AVGO': 'Technology', 'QCOM': 'Technology', 'TXN': 'Technology',
    'NOW': 'Technology', 'AMAT': 'Technology', 'MU': 'Technology', 'LRCX': 'Technology',
    'NFLX': 'Technology', 'PYPL': 'Technology', 'SQ': 'Technology', 'SHOP': 'Technology',
    
    # Consumer Discretionary
    'AMZN': 'Consumer Discretionary', 'TSLA': 'Consumer Discretionary', 'HD': 'Consumer Discretionary',
    'MCD': 'Consumer Discretionary', 'NKE': 'Consumer Discretionary', 'SBUX': 'Consumer Discretionary',
    'LOW': 'Consumer Discretionary', 'TGT': 'Consumer Discretionary', 'BKNG': 'Consumer Discretionary',
    'MAR': 'Consumer Discretionary', 'GM': 'Consumer Discretionary', 'F': 'Consumer Discretionary',
    
    # Healthcare
    'JNJ': 'Healthcare', 'UNH': 'Healthcare', 'PFE': 'Healthcare', 'ABBV': 'Healthcare',
    'MRK': 'Healthcare', 'LLY': 'Healthcare', 'TMO': 'Healthcare', 'ABT': 'Healthcare',
    'DHR': 'Healthcare', 'BMY': 'Healthcare', 'AMGN': 'Healthcare', 'GILD': 'Healthcare',
    'MDT': 'Healthcare', 'CVS': 'Healthcare', 'ISRG': 'Healthcare', 'VRTX': 'Healthcare',
    'MRNA': 'Healthcare', 'REGN': 'Healthcare', 'ZTS': 'Healthcare', 'BIIB': 'Healthcare',
    
    # Financials
    'JPM': 'Financials', 'BAC': 'Financials', 'WFC': 'Financials', 'GS': 'Financials',
    'MS': 'Financials', 'C': 'Financials', 'BLK': 'Financials', 'SCHW': 'Financials',
    'AXP': 'Financials', 'SPGI': 'Financials', 'CME': 'Financials', 'ICE': 'Financials',
    'V': 'Financials', 'MA': 'Financials', 'COF': 'Financials', 'USB': 'Financials',
    'PNC': 'Financials', 'TFC': 'Financials', 'BK': 'Financials', 'STT': 'Financials',
    
    # Consumer Staples
    'PG': 'Consumer Staples', 'KO': 'Consumer Staples', 'PEP': 'Consumer Staples',
    'COST': 'Consumer Staples', 'WMT': 'Consumer Staples', 'PM': 'Consumer Staples',
    'MO': 'Consumer Staples', 'MDLZ': 'Consumer Staples', 'CL': 'Consumer Staples',
    'EL': 'Consumer Staples', 'KMB': 'Consumer Staples', 'GIS': 'Consumer Staples',
    'K': 'Consumer Staples', 'HSY': 'Consumer Staples', 'STZ': 'Consumer Staples',
    
    # Energy
    'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy', 'SLB': 'Energy',
    'EOG': 'Energy', 'MPC': 'Energy', 'PSX': 'Energy', 'VLO': 'Energy',
    'OXY': 'Energy', 'PXD': 'Energy', 'DVN': 'Energy', 'HAL': 'Energy',
    
    # Industrials
    'BA': 'Industrials', 'UNP': 'Industrials', 'HON': 'Industrials', 'UPS': 'Industrials',
    'CAT': 'Industrials', 'GE': 'Industrials', 'MMM': 'Industrials', 'LMT': 'Industrials',
    'RTX': 'Industrials', 'DE': 'Industrials', 'FDX': 'Industrials', 'NOC': 'Industrials',
    'WM': 'Industrials', 'EMR': 'Industrials', 'ETN': 'Industrials', 'ITW': 'Industrials',
    
    # Materials
    'LIN': 'Materials', 'APD': 'Materials', 'SHW': 'Materials', 'ECL': 'Materials',
    'FCX': 'Materials', 'NEM': 'Materials', 'NUE': 'Materials', 'VMC': 'Materials',
    'DOW': 'Materials', 'DD': 'Materials', 'PPG': 'Materials', 'ALB': 'Materials',
    
    # Utilities
    'NEE': 'Utilities', 'DUK': 'Utilities', 'SO': 'Utilities', 'D': 'Utilities',
    'AEP': 'Utilities', 'EXC': 'Utilities', 'SRE': 'Utilities', 'XEL': 'Utilities',
    'ED': 'Utilities', 'WEC': 'Utilities', 'ES': 'Utilities', 'PEG': 'Utilities',
    
    # Real Estate
    'PLD': 'Real Estate', 'AMT': 'Real Estate', 'CCI': 'Real Estate', 'EQIX': 'Real Estate',
    'PSA': 'Real Estate', 'SPG': 'Real Estate', 'O': 'Real Estate', 'WELL': 'Real Estate',
    'DLR': 'Real Estate', 'AVB': 'Real Estate', 'EQR': 'Real Estate', 'VTR': 'Real Estate',
    
    # Communication Services
    'DIS': 'Communication Services', 'CMCSA': 'Communication Services', 'VZ': 'Communication Services',
    'T': 'Communication Services', 'TMUS': 'Communication Services', 'CHTR': 'Communication Services',
    'ATVI': 'Communication Services', 'EA': 'Communication Services', 'TTWO': 'Communication Services',
    
    # ETFs
    'SPY': 'ETF - Broad Market', 'QQQ': 'ETF - Technology', 'IWM': 'ETF - Small Cap',
    'DIA': 'ETF - Dow', 'VTI': 'ETF - Total Market', 'VOO': 'ETF - S&P 500',
    'XLF': 'ETF - Financials', 'XLK': 'ETF - Technology', 'XLE': 'ETF - Energy',
    'XLV': 'ETF - Healthcare', 'XLY': 'ETF - Consumer Discretionary', 'XLP': 'ETF - Consumer Staples',
    'XLI': 'ETF - Industrials', 'XLB': 'ETF - Materials', 'XLU': 'ETF - Utilities',
    'XLRE': 'ETF - Real Estate', 'XLC': 'ETF - Communication Services',
    'GLD': 'ETF - Gold', 'SLV': 'ETF - Silver', 'TLT': 'ETF - Treasury Bonds',
    'BND': 'ETF - Bonds', 'AGG': 'ETF - Aggregate Bonds',
    'EEM': 'ETF - Emerging Markets', 'VWO': 'ETF - Emerging Markets',
    'EFA': 'ETF - Developed Markets', 'VEA': 'ETF - Developed Markets',
}


# =============================================================================
# RISK SCORE CALCULATION
# =============================================================================

def calculate_unified_risk_score(
    var_pct: float,
    sharpe_ratio: float,
    max_drawdown: float,
    volatility: float,
    correlation_avg: float = 0.3,
    sentiment_score: float = 50,
    custom_weights: Dict[str, float] = None
) -> RiskScoreResult:
    """
    Calculate unified risk score (0-100) combining multiple risk factors.
    
    Components:
    - VaR Level (30%): Higher VaR = higher risk score
    - Sharpe Ratio (25%): Lower Sharpe = higher risk score  
    - Max Drawdown (20%): Higher drawdown = higher risk score
    - Volatility (15%): Higher volatility = higher risk score
    - Correlation (5%): Higher correlation = higher risk score
    - Sentiment (5%): Lower sentiment = higher risk score
    
    Args:
        var_pct: Value at Risk as percentage (e.g., 0.05 for 5%)
        sharpe_ratio: Sharpe ratio
        max_drawdown: Maximum drawdown as negative percentage (e.g., -0.15)
        volatility: Annualized volatility (e.g., 0.20 for 20%)
        correlation_avg: Average correlation in portfolio (0-1)
        sentiment_score: Sentiment score (0-100, 50 = neutral)
        custom_weights: Override default weights
    
    Returns:
        RiskScoreResult with score and breakdown
    """
    # Default weights
    weights = custom_weights or {
        'var': 0.30,
        'sharpe': 0.25,
        'drawdown': 0.20,
        'volatility': 0.15,
        'correlation': 0.05,
        'sentiment': 0.05
    }
    
    # Normalize weights
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}
    
    # Calculate component scores (0-100, higher = riskier)
    components = {}
    
    # VaR score: 0% VaR = 0, 10%+ VaR = 100
    var_abs = abs(var_pct)
    components['var'] = min(100, var_abs * 1000)  # 0.10 (10%) -> 100
    
    # Sharpe score: 3+ Sharpe = 0 (low risk), -1 Sharpe = 100 (high risk)
    sharpe_clamped = max(-1, min(3, sharpe_ratio))
    components['sharpe'] = 100 - ((sharpe_clamped + 1) / 4) * 100
    
    # Drawdown score: 0% DD = 0, -50%+ DD = 100
    dd_abs = abs(max_drawdown)
    components['drawdown'] = min(100, dd_abs * 200)  # 0.50 (50%) -> 100
    
    # Volatility score: 0% vol = 0, 50%+ vol = 100
    components['volatility'] = min(100, volatility * 200)  # 0.50 (50%) -> 100
    
    # Correlation score: 0 corr = 0 (diversified), 1 corr = 100 (concentrated)
    components['correlation'] = correlation_avg * 100
    
    # Sentiment score: 100 sentiment = 0 (bullish = low risk), 0 sentiment = 100 (bearish = high risk)
    components['sentiment'] = 100 - sentiment_score
    
    # Calculate weighted total
    total_score = sum(components[k] * weights[k] for k in weights.keys())
    total_score = max(0, min(100, total_score))
    
    # Determine risk level and grade
    if total_score <= 20:
        risk_level = 'Low'
        grade = 'A'
        color = '#00C853'  # Green
    elif total_score <= 40:
        risk_level = 'Moderate'
        grade = 'B'
        color = '#2196F3'  # Blue
    elif total_score <= 60:
        risk_level = 'High'
        grade = 'C'
        color = '#FFC107'  # Yellow
    elif total_score <= 80:
        risk_level = 'Very High'
        grade = 'D'
        color = '#FF9800'  # Orange
    else:
        risk_level = 'Extreme'
        grade = 'F'
        color = '#FF1744'  # Red
    
    # Generate recommendations
    recommendations = []
    
    if components['var'] > 60:
        recommendations.append("Consider reducing position sizes to lower VaR")
    
    if components['sharpe'] > 60:
        recommendations.append("Risk-adjusted returns are poor - review asset selection")
    
    if components['drawdown'] > 60:
        recommendations.append("Historical drawdowns are severe - add defensive assets")
    
    if components['volatility'] > 60:
        recommendations.append("Portfolio is highly volatile - consider low-vol alternatives")
    
    if components['correlation'] > 60:
        recommendations.append("Assets are highly correlated - diversify across sectors")
    
    if components['sentiment'] > 60:
        recommendations.append("Market sentiment is bearish - maintain cash buffer")
    
    if not recommendations:
        recommendations.append("Portfolio risk profile is healthy")
    
    return RiskScoreResult(
        total_score=round(total_score, 1),
        risk_level=risk_level,
        components={k: round(v, 1) for k, v in components.items()},
        weights=weights,
        recommendations=recommendations,
        grade=grade,
        color=color
    )


# =============================================================================
# HISTORICAL SCENARIO REPLAY
# =============================================================================

def replay_historical_scenario(
    returns: pd.DataFrame,
    weights: Dict[str, float],
    scenario_name: str,
    portfolio_value: float = 100000
) -> ScenarioResult:
    """
    Replay a historical market scenario on current portfolio.
    
    Args:
        returns: DataFrame of asset returns
        weights: Portfolio weights
        scenario_name: Name of scenario from HISTORICAL_SCENARIOS
        portfolio_value: Current portfolio value
    
    Returns:
        ScenarioResult with impact analysis
    """
    if scenario_name not in HISTORICAL_SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario_name}")
    
    scenario = HISTORICAL_SCENARIOS[scenario_name]
    
    # Calculate portfolio beta and volatility
    weights_array = np.array([weights.get(t, 0) for t in returns.columns])
    weights_array = weights_array / weights_array.sum()  # Normalize
    
    # Portfolio volatility
    cov_matrix = returns.cov() * 252
    port_vol = np.sqrt(weights_array @ cov_matrix.values @ weights_array)
    
    # Estimate portfolio beta (assuming market correlation of 0.7)
    market_vol = 0.16  # Approximate market volatility
    market_corr = 0.7  # Approximate correlation with market
    port_beta = (port_vol * market_corr) / market_vol
    
    # Estimate portfolio return during scenario
    market_return = scenario['market_return']
    portfolio_return = market_return * port_beta
    
    # Adjust for sector exposure
    sectors_hit = scenario.get('sectors_hit', [])
    sector_exposure = _get_sector_exposure(list(weights.keys()))
    
    # Additional hit if exposed to affected sectors
    hit_exposure = sum(
        sector_exposure.get(s, 0) for s in sectors_hit 
        if s in sector_exposure
    )
    if hit_exposure > 0.3:  # More than 30% in hit sectors
        portfolio_return *= 1.1  # 10% worse
    
    # Calculate max drawdown during scenario
    max_drawdown = portfolio_return * 1.2  # Assume intra-period DD is worse
    
    # Estimate recovery days (based on severity)
    severity_recovery = {
        'Low': 30,
        'Medium': 90,
        'High': 180,
        'Extreme': 365
    }
    recovery_days = severity_recovery.get(scenario['severity'], 180)
    
    # Check if VaR would be breached
    # Assuming 95% VaR is around 2% daily
    daily_return = portfolio_return  # For multi-day scenarios
    var_95 = port_vol * 1.645 / np.sqrt(252)  # Daily VaR
    var_breach = abs(daily_return) > var_95
    
    return ScenarioResult(
        scenario_name=scenario_name,
        start_date=scenario['start'],
        end_date=scenario['end'],
        market_return=market_return,
        portfolio_return=round(portfolio_return, 4),
        max_drawdown=round(max_drawdown, 4),
        recovery_days=recovery_days,
        var_breach=var_breach,
        description=scenario['description'],
        severity=scenario['severity']
    )


def replay_all_scenarios(
    returns: pd.DataFrame,
    weights: Dict[str, float]
) -> List[ScenarioResult]:
    """Replay all historical scenarios on portfolio."""
    results = []
    for scenario_name in HISTORICAL_SCENARIOS:
        try:
            result = replay_historical_scenario(returns, weights, scenario_name)
            results.append(result)
        except Exception as e:
            logger.warning(f"Failed to replay {scenario_name}: {e}")
    
    # Sort by severity and impact
    severity_order = {'Extreme': 0, 'High': 1, 'Medium': 2, 'Low': 3}
    results.sort(key=lambda x: (severity_order.get(x.severity, 2), x.portfolio_return))
    
    return results


# =============================================================================
# SECTOR EXPOSURE ANALYSIS
# =============================================================================

def _get_sector_exposure(tickers: List[str]) -> Dict[str, float]:
    """Get sector exposure for a list of tickers (equal weight)."""
    sectors = {}
    for ticker in tickers:
        sector = SECTOR_MAP.get(ticker.upper(), 'Other')
        sectors[sector] = sectors.get(sector, 0) + 1
    
    total = len(tickers)
    return {k: v / total for k, v in sectors.items()}


def analyze_sector_exposure(
    returns: pd.DataFrame,
    weights: Dict[str, float]
) -> SectorExposure:
    """
    Analyze portfolio sector exposure and risk contribution.
    
    Args:
        returns: DataFrame of asset returns
        weights: Portfolio weights
    
    Returns:
        SectorExposure with detailed breakdown
    """
    tickers = list(returns.columns)
    
    # Map tickers to sectors
    ticker_sectors = {t: SECTOR_MAP.get(t.upper(), 'Other') for t in tickers}
    
    # Calculate sector weights
    sector_weights = {}
    for ticker, weight in weights.items():
        sector = ticker_sectors.get(ticker, 'Other')
        sector_weights[sector] = sector_weights.get(sector, 0) + weight
    
    # Normalize
    total_weight = sum(sector_weights.values())
    if total_weight > 0:
        sector_weights = {k: v / total_weight for k, v in sector_weights.items()}
    
    # Calculate sector risk contribution
    cov_matrix = returns.cov() * 252
    weights_array = np.array([weights.get(t, 0) for t in tickers])
    weights_array = weights_array / weights_array.sum()
    
    port_var = weights_array @ cov_matrix.values @ weights_array
    port_vol = np.sqrt(port_var)
    
    # Marginal contribution to risk
    mcr = (cov_matrix.values @ weights_array) / port_vol
    component_risk = weights_array * mcr
    
    # Aggregate by sector
    sector_risk = {}
    for i, ticker in enumerate(tickers):
        sector = ticker_sectors.get(ticker, 'Other')
        sector_risk[sector] = sector_risk.get(sector, 0) + component_risk[i]
    
    # Normalize risk contribution
    total_risk = sum(abs(r) for r in sector_risk.values())
    if total_risk > 0:
        sector_risk = {k: abs(v) / total_risk for k, v in sector_risk.items()}
    
    # Create sector-level correlation matrix
    unique_sectors = list(set(ticker_sectors.values()))
    sector_returns = pd.DataFrame()
    
    for sector in unique_sectors:
        sector_tickers = [t for t in tickers if ticker_sectors[t] == sector]
        if sector_tickers:
            sector_weights_norm = [weights.get(t, 0) for t in sector_tickers]
            total = sum(sector_weights_norm)
            if total > 0:
                sector_weights_norm = [w / total for w in sector_weights_norm]
            sector_returns[sector] = (returns[sector_tickers] * sector_weights_norm).sum(axis=1)
    
    sector_corr = sector_returns.corr() if len(sector_returns.columns) > 1 else pd.DataFrame()
    
    # Calculate concentration (HHI)
    hhi = sum(w ** 2 for w in sector_weights.values())
    
    # Diversification score (1 - normalized HHI)
    n_sectors = len(sector_weights)
    if n_sectors > 1:
        min_hhi = 1 / n_sectors
        diversification_score = 1 - (hhi - min_hhi) / (1 - min_hhi)
    else:
        diversification_score = 0
    
    # Dominant sector
    dominant_sector = max(sector_weights, key=sector_weights.get)
    
    return SectorExposure(
        sector_weights=sector_weights,
        sector_risk_contribution=sector_risk,
        sector_correlation=sector_corr,
        concentration_index=round(hhi, 4),
        diversification_score=round(diversification_score, 4),
        dominant_sector=dominant_sector
    )


# =============================================================================
# PERFORMANCE ATTRIBUTION
# =============================================================================

def calculate_performance_attribution(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    factor_returns: pd.DataFrame = None,
    weights: Dict[str, float] = None,
    asset_returns: pd.DataFrame = None
) -> PerformanceAttribution:
    """
    Calculate performance attribution breakdown.
    
    Decomposes portfolio return into:
    - Market contribution (beta * market return)
    - Factor contributions (factor exposures * factor returns)
    - Alpha (residual unexplained return)
    - Selection effect (stock picking within sectors)
    - Allocation effect (over/underweight sectors)
    - Interaction effect (cross effects)
    
    Args:
        portfolio_returns: Portfolio return series
        benchmark_returns: Benchmark return series
        factor_returns: DataFrame of factor returns (optional)
        weights: Portfolio weights (optional)
        asset_returns: Individual asset returns (optional)
    
    Returns:
        PerformanceAttribution breakdown
    """
    # Align data
    common_idx = portfolio_returns.index.intersection(benchmark_returns.index)
    port_ret = portfolio_returns.loc[common_idx]
    bench_ret = benchmark_returns.loc[common_idx]
    
    # Total return
    total_return = (1 + port_ret).prod() - 1
    bench_total = (1 + bench_ret).prod() - 1
    
    # Calculate beta
    cov = np.cov(port_ret.values, bench_ret.values)[0, 1]
    var = np.var(bench_ret.values)
    beta = cov / var if var > 0 else 1.0
    
    # Market contribution
    market_contribution = beta * bench_total
    
    # Factor contributions (if factor returns provided)
    factor_contributions = {}
    factors_total = 0
    
    if factor_returns is not None:
        aligned_factors = factor_returns.loc[common_idx]
        
        for factor in aligned_factors.columns:
            # Calculate factor loading
            factor_cov = np.cov(port_ret.values, aligned_factors[factor].values)[0, 1]
            factor_var = np.var(aligned_factors[factor].values)
            factor_beta = factor_cov / factor_var if factor_var > 0 else 0
            
            # Factor return contribution
            factor_return = (1 + aligned_factors[factor]).prod() - 1
            contrib = factor_beta * factor_return
            factor_contributions[factor] = contrib
            factors_total += contrib
    
    # Alpha (residual)
    alpha = total_return - market_contribution - factors_total
    
    # Brinson attribution (if weights and asset returns provided)
    selection_effect = 0
    allocation_effect = 0
    interaction_effect = 0
    
    if weights is not None and asset_returns is not None:
        # Simplified Brinson attribution
        # Would need sector weights and benchmark sector returns for full implementation
        # For now, estimate based on active returns
        active_return = total_return - bench_total
        
        # Split active return approximately
        selection_effect = active_return * 0.6  # Assume 60% from selection
        allocation_effect = active_return * 0.3  # 30% from allocation
        interaction_effect = active_return * 0.1  # 10% from interaction
    
    # Residual
    residual = total_return - market_contribution - factors_total - selection_effect - allocation_effect - interaction_effect
    
    return PerformanceAttribution(
        total_return=round(total_return, 4),
        market_contribution=round(market_contribution, 4),
        factor_contributions={k: round(v, 4) for k, v in factor_contributions.items()},
        alpha=round(alpha, 4),
        selection_effect=round(selection_effect, 4),
        allocation_effect=round(allocation_effect, 4),
        interaction_effect=round(interaction_effect, 4),
        residual=round(residual, 4)
    )


# =============================================================================
# VAR BACKTESTING DASHBOARD
# =============================================================================

def run_var_backtest(
    returns: pd.Series,
    conf_level: float = 0.95,
    methods: List[str] = None,
    window: int = 252
) -> List[VaRBacktestResult]:
    """
    Run comprehensive VaR backtesting for multiple methods.
    
    Args:
        returns: Return series
        conf_level: Confidence level (e.g., 0.95)
        methods: VaR methods to test ['parametric', 'historical', 'ewma']
        window: Lookback window for VaR calculation
    
    Returns:
        List of VaRBacktestResult for each method
    """
    if methods is None:
        methods = ['parametric', 'historical', 'ewma']
    
    results = []
    
    for method in methods:
        try:
            result = _backtest_var_method(returns, conf_level, method, window)
            results.append(result)
        except Exception as e:
            logger.warning(f"VaR backtest failed for {method}: {e}")
    
    return results


def _backtest_var_method(
    returns: pd.Series,
    conf_level: float,
    method: str,
    window: int
) -> VaRBacktestResult:
    """Backtest a single VaR method."""
    
    # Calculate rolling VaR
    var_series = []
    dates = []
    
    for i in range(window, len(returns)):
        hist_window = returns.iloc[i-window:i]
        
        if method == 'parametric':
            # Normal distribution VaR
            mu = hist_window.mean()
            sigma = hist_window.std()
            z = stats.norm.ppf(1 - conf_level)
            var = -(mu + z * sigma)
        
        elif method == 'historical':
            # Historical VaR
            var = -np.percentile(hist_window, (1 - conf_level) * 100)
        
        elif method == 'ewma':
            # EWMA VaR
            lambda_param = 0.94
            ewma_var = hist_window.ewm(alpha=1-lambda_param).var().iloc[-1]
            z = stats.norm.ppf(1 - conf_level)
            var = -(hist_window.mean() + z * np.sqrt(ewma_var))
        
        else:
            var = -np.percentile(hist_window, (1 - conf_level) * 100)
        
        var_series.append(-abs(var))  # VaR as negative number
        dates.append(returns.index[i])
    
    var_series = pd.Series(var_series, index=dates)
    actual_returns = returns.loc[dates]
    
    # Count violations
    violations = actual_returns < var_series
    n_violations = violations.sum()
    n_total = len(violations)
    violation_rate = n_violations / n_total if n_total > 0 else 0
    expected_rate = 1 - conf_level
    
    # Kupiec test
    kupiec_result = _kupiec_test(n_violations, n_total, expected_rate)
    
    # Christoffersen test
    chris_result = _christoffersen_test(violations.astype(int).values)
    
    # Determine if model is adequate
    model_adequate = kupiec_result['p_value'] >= 0.05 and chris_result['p_value'] >= 0.05
    
    # Assessment
    if violation_rate < expected_rate * 0.8:
        assessment = 'Conservative (too few violations)'
    elif violation_rate > expected_rate * 1.2:
        assessment = 'Aggressive (too many violations)'
    else:
        assessment = 'Well-calibrated'
    
    return VaRBacktestResult(
        model_name=method.capitalize(),
        violations=int(n_violations),
        total_observations=n_total,
        violation_rate=round(violation_rate, 4),
        expected_rate=expected_rate,
        kupiec_stat=round(kupiec_result['statistic'], 4),
        kupiec_pvalue=round(kupiec_result['p_value'], 4),
        christoffersen_stat=round(chris_result['statistic'], 4),
        christoffersen_pvalue=round(chris_result['p_value'], 4),
        model_adequate=model_adequate,
        assessment=assessment,
        violation_dates=[str(d) for d in actual_returns[violations].index[:10]],  # First 10
        var_series=var_series,
        return_series=actual_returns
    )


def _kupiec_test(violations: int, total: int, expected_rate: float) -> Dict:
    """Kupiec POF test."""
    if total == 0:
        return {'statistic': 0, 'p_value': 1}
    
    observed_rate = violations / total
    
    # Handle edge cases
    if observed_rate == 0:
        observed_rate = 0.0001
    if observed_rate == 1:
        observed_rate = 0.9999
    
    # Log-likelihood ratio
    try:
        lr_num = (1 - expected_rate) ** (total - violations) * expected_rate ** violations
        lr_den = (1 - observed_rate) ** (total - violations) * observed_rate ** violations
        
        if lr_num > 0 and lr_den > 0:
            lr_stat = -2 * np.log(lr_num / lr_den)
        else:
            lr_stat = 0
        
        p_value = 1 - chi2.cdf(lr_stat, df=1)
    except:
        lr_stat = 0
        p_value = 1
    
    return {'statistic': lr_stat, 'p_value': p_value}


def _christoffersen_test(violations: np.ndarray) -> Dict:
    """Christoffersen independence test."""
    n = len(violations)
    if n < 2:
        return {'statistic': 0, 'p_value': 1}
    
    # Count transitions
    n00 = n01 = n10 = n11 = 0
    
    for i in range(1, n):
        if violations[i-1] == 0 and violations[i] == 0:
            n00 += 1
        elif violations[i-1] == 0 and violations[i] == 1:
            n01 += 1
        elif violations[i-1] == 1 and violations[i] == 0:
            n10 += 1
        else:
            n11 += 1
    
    # Calculate test statistic
    try:
        p01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
        p11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
        p = (n01 + n11) / (n - 1)
        
        if p > 0 and p < 1 and 0 < p01 < 1 and 0 < p11 < 1:
            l_ind = (1 - p) ** (n00 + n10) * p ** (n01 + n11)
            l_dep = ((1 - p01) ** n00 * p01 ** n01 * 
                    (1 - p11) ** n10 * p11 ** n11)
            
            lr_stat = -2 * np.log(l_ind / l_dep) if l_ind > 0 and l_dep > 0 else 0
            p_value = 1 - chi2.cdf(lr_stat, df=1)
        else:
            lr_stat = 0
            p_value = 1
    except:
        lr_stat = 0
        p_value = 1
    
    return {'statistic': lr_stat, 'p_value': p_value}


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Data classes
    'RiskScoreResult',
    'ScenarioResult', 
    'SectorExposure',
    'PerformanceAttribution',
    'VaRBacktestResult',
    
    # Constants
    'HISTORICAL_SCENARIOS',
    'SECTOR_MAP',
    
    # Functions
    'calculate_unified_risk_score',
    'replay_historical_scenario',
    'replay_all_scenarios',
    'analyze_sector_exposure',
    'calculate_performance_attribution',
    'run_var_backtest',
]
