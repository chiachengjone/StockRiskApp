"""
Macro Indicators Integration
=============================
Track economic indicators and their impact on markets.

Features:
- Fed funds rate and yield curve tracking
- Economic indicators (GDP, unemployment, CPI)
- Macro regime detection (expansion/recession)
- Cross-asset correlation analysis
- Central bank policy tracker

Author: Stock Risk App | Feb 2026
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import requests

logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class EconomicIndicator:
    """Single economic indicator data point."""
    name: str
    value: float
    previous: float
    change: float
    date: datetime
    frequency: str  # 'daily', 'weekly', 'monthly', 'quarterly'
    trend: str  # 'rising', 'falling', 'stable'
    signal: str  # 'bullish', 'bearish', 'neutral'


@dataclass
class YieldCurveData:
    """Yield curve snapshot."""
    date: datetime
    rates: Dict[str, float]  # tenor -> rate
    is_inverted: bool
    spread_10y_2y: float
    spread_10y_3m: float
    inversion_depth: float  # How inverted (negative spread)
    recession_probability: float


@dataclass
class MacroRegime:
    """Current macroeconomic regime."""
    regime: str  # 'expansion', 'peak', 'contraction', 'trough'
    confidence: float
    leading_indicators: Dict[str, str]
    risk_level: str  # 'low', 'moderate', 'high', 'extreme'
    recommended_positioning: str


@dataclass
class CentralBankPolicy:
    """Central bank policy summary."""
    bank: str  # 'Fed', 'ECB', 'BOJ', etc.
    current_rate: float
    last_change: float
    last_change_date: datetime
    next_meeting: Optional[datetime]
    expected_action: str  # 'hike', 'cut', 'hold'
    dot_plot_median: Optional[float]
    market_expectations: Dict[str, float]


# ============================================================================
# FRED DATA PROVIDER
# ============================================================================

class FREDProvider:
    """
    Federal Reserve Economic Data (FRED) provider.
    Uses free FRED API for economic data.
    """
    
    BASE_URL = "https://api.stlouisfed.org/fred"
    
    # Key economic indicators and their FRED series IDs
    INDICATORS = {
        # Interest Rates
        'fed_funds_rate': 'FEDFUNDS',
        'treasury_10y': 'DGS10',
        'treasury_2y': 'DGS2',
        'treasury_3m': 'DTB3',
        'treasury_30y': 'DGS30',
        'treasury_5y': 'DGS5',
        'treasury_1y': 'DGS1',
        
        # Yield Spreads
        'spread_10y_2y': 'T10Y2Y',
        'spread_10y_3m': 'T10Y3M',
        
        # Economic Activity
        'gdp': 'GDP',
        'gdp_growth': 'A191RL1Q225SBEA',
        'industrial_production': 'INDPRO',
        'unemployment_rate': 'UNRATE',
        'nonfarm_payrolls': 'PAYEMS',
        'initial_claims': 'ICSA',
        
        # Inflation
        'cpi': 'CPIAUCSL',
        'core_cpi': 'CPILFESL',
        'pce': 'PCE',
        'core_pce': 'PCEPILFE',
        'inflation_expectation_5y': 'T5YIE',
        'inflation_expectation_10y': 'T10YIE',
        
        # Consumer
        'consumer_sentiment': 'UMCSENT',
        'retail_sales': 'RSXFS',
        'personal_income': 'PI',
        
        # Housing
        'housing_starts': 'HOUST',
        'existing_home_sales': 'EXHOSLUSM495S',
        'case_shiller': 'CSUSHPISA',
        
        # Manufacturing
        'ism_manufacturing': 'MANEMP',
        'durable_goods': 'DGORDER',
        
        # Financial Conditions
        'vix': 'VIXCLS',
        'credit_spread': 'BAMLH0A0HYM2',
        'financial_stress': 'STLFSI4',
        
        # Money Supply
        'm2': 'M2SL',
        
        # Leading Indicators
        'lei': 'USSLIND',  # Leading Economic Index
    }
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self._cache = {}
        self._cache_time = {}
        self._cache_ttl = 3600  # 1 hour
    
    def get_series(
        self,
        series_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 100
    ) -> pd.Series:
        """
        Fetch a FRED series.
        
        Args:
            series_id: FRED series ID
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            limit: Maximum observations
            
        Returns:
            pandas Series with the data
        """
        # Check cache
        cache_key = f"{series_id}_{start_date}_{end_date}"
        if cache_key in self._cache:
            cache_time = self._cache_time.get(cache_key, 0)
            if datetime.now().timestamp() - cache_time < self._cache_ttl:
                return self._cache[cache_key]
        
        # Try pandas-datareader first
        try:
            import pandas_datareader as pdr
            
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            data = pdr.get_data_fred(series_id, start=start_date, end=end_date)
            
            if isinstance(data, pd.DataFrame):
                data = data.iloc[:, 0]
            
            self._cache[cache_key] = data
            self._cache_time[cache_key] = datetime.now().timestamp()
            
            return data
            
        except Exception as e:
            logger.warning(f"pandas-datareader failed for {series_id}: {e}")
        
        # Fallback: Try direct FRED API
        if self.api_key:
            try:
                params = {
                    'series_id': series_id,
                    'api_key': self.api_key,
                    'file_type': 'json',
                    'limit': limit,
                    'sort_order': 'desc'
                }
                
                if start_date:
                    params['observation_start'] = start_date
                if end_date:
                    params['observation_end'] = end_date
                
                response = requests.get(
                    f"{self.BASE_URL}/series/observations",
                    params=params,
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    observations = data.get('observations', [])
                    
                    dates = [obs['date'] for obs in observations]
                    values = [float(obs['value']) if obs['value'] != '.' else np.nan 
                             for obs in observations]
                    
                    series = pd.Series(values, index=pd.to_datetime(dates), name=series_id)
                    series = series.sort_index()
                    
                    self._cache[cache_key] = series
                    self._cache_time[cache_key] = datetime.now().timestamp()
                    
                    return series
                    
            except Exception as e:
                logger.error(f"FRED API error: {e}")
        
        # Return empty series if all fails
        return pd.Series(dtype=float, name=series_id)
    
    def get_indicator(self, indicator_name: str) -> Optional[EconomicIndicator]:
        """Get a specific economic indicator with analysis."""
        series_id = self.INDICATORS.get(indicator_name)
        if not series_id:
            return None
        
        data = self.get_series(series_id, limit=30)
        
        if data.empty:
            return None
        
        current = data.iloc[-1]
        previous = data.iloc[-2] if len(data) > 1 else current
        change = current - previous
        change_pct = (change / previous * 100) if previous != 0 else 0
        
        # Determine trend
        if len(data) >= 3:
            recent = data.tail(3)
            if recent.is_monotonic_increasing:
                trend = 'rising'
            elif recent.is_monotonic_decreasing:
                trend = 'falling'
            else:
                trend = 'stable'
        else:
            trend = 'stable'
        
        # Determine signal based on indicator type
        signal = self._determine_signal(indicator_name, current, change_pct, trend)
        
        return EconomicIndicator(
            name=indicator_name,
            value=current,
            previous=previous,
            change=change_pct,
            date=data.index[-1],
            frequency=self._get_frequency(indicator_name),
            trend=trend,
            signal=signal
        )
    
    def _determine_signal(
        self,
        indicator_name: str,
        value: float,
        change_pct: float,
        trend: str
    ) -> str:
        """Determine bullish/bearish signal for an indicator."""
        # Indicators where higher = bearish
        bearish_when_high = [
            'unemployment_rate', 'initial_claims', 'vix', 
            'credit_spread', 'cpi', 'core_cpi', 'fed_funds_rate'
        ]
        
        # Indicators where higher = bullish
        bullish_when_high = [
            'gdp', 'gdp_growth', 'industrial_production', 'nonfarm_payrolls',
            'consumer_sentiment', 'retail_sales', 'lei', 'housing_starts'
        ]
        
        if indicator_name in bearish_when_high:
            if trend == 'rising':
                return 'bearish'
            elif trend == 'falling':
                return 'bullish'
        elif indicator_name in bullish_when_high:
            if trend == 'rising':
                return 'bullish'
            elif trend == 'falling':
                return 'bearish'
        
        return 'neutral'
    
    def _get_frequency(self, indicator_name: str) -> str:
        """Get the frequency of an indicator."""
        daily = ['treasury_10y', 'treasury_2y', 'treasury_3m', 'vix', 'credit_spread']
        weekly = ['initial_claims']
        monthly = ['unemployment_rate', 'cpi', 'core_cpi', 'retail_sales', 'housing_starts']
        quarterly = ['gdp', 'gdp_growth']
        
        if indicator_name in daily:
            return 'daily'
        elif indicator_name in weekly:
            return 'weekly'
        elif indicator_name in monthly:
            return 'monthly'
        elif indicator_name in quarterly:
            return 'quarterly'
        return 'unknown'


# ============================================================================
# MACRO ANALYZER
# ============================================================================

class MacroAnalyzer:
    """
    Analyze macroeconomic conditions and their impact on markets.
    """
    
    def __init__(self, fred_provider: Optional[FREDProvider] = None):
        self.fred = fred_provider or FREDProvider()
    
    def get_yield_curve(self) -> YieldCurveData:
        """Get current yield curve data and analysis."""
        tenors = {
            '3M': 'treasury_3m',
            '1Y': 'treasury_1y',
            '2Y': 'treasury_2y',
            '5Y': 'treasury_5y',
            '10Y': 'treasury_10y',
            '30Y': 'treasury_30y'
        }
        
        rates = {}
        for tenor, indicator in tenors.items():
            data = self.fred.get_series(self.fred.INDICATORS.get(indicator, ''), limit=5)
            if not data.empty:
                rates[tenor] = data.iloc[-1]
        
        # Calculate spreads
        spread_10y_2y = rates.get('10Y', 0) - rates.get('2Y', 0)
        spread_10y_3m = rates.get('10Y', 0) - rates.get('3M', 0)
        
        # Check for inversion
        is_inverted = spread_10y_2y < 0 or spread_10y_3m < 0
        inversion_depth = min(spread_10y_2y, spread_10y_3m) if is_inverted else 0
        
        # Simple recession probability model based on yield curve
        # Historical: Inverted curve has preceded all recessions since 1970s
        if spread_10y_3m < -0.5:
            recession_prob = 0.7
        elif spread_10y_3m < 0:
            recession_prob = 0.5
        elif spread_10y_3m < 0.5:
            recession_prob = 0.3
        else:
            recession_prob = 0.1
        
        return YieldCurveData(
            date=datetime.now(),
            rates=rates,
            is_inverted=is_inverted,
            spread_10y_2y=spread_10y_2y,
            spread_10y_3m=spread_10y_3m,
            inversion_depth=inversion_depth,
            recession_probability=recession_prob
        )
    
    def detect_macro_regime(self) -> MacroRegime:
        """
        Detect current macroeconomic regime.
        
        Uses composite of leading indicators to classify:
        - Expansion: Growth accelerating
        - Peak: Growth slowing but positive
        - Contraction: Growth declining
        - Trough: Growth bottoming
        """
        scores = {}
        
        # Check key indicators
        indicators_to_check = [
            ('lei', 'leading'),
            ('unemployment_rate', 'lagging'),
            ('gdp_growth', 'coincident'),
            ('consumer_sentiment', 'leading'),
            ('initial_claims', 'leading'),
            ('industrial_production', 'coincident')
        ]
        
        for indicator_name, category in indicators_to_check:
            indicator = self.fred.get_indicator(indicator_name)
            if indicator:
                scores[indicator_name] = {
                    'trend': indicator.trend,
                    'signal': indicator.signal,
                    'category': category
                }
        
        # Aggregate signals
        bullish_count = sum(1 for s in scores.values() if s['signal'] == 'bullish')
        bearish_count = sum(1 for s in scores.values() if s['signal'] == 'bearish')
        total = len(scores)
        
        # Determine regime
        if total == 0:
            regime = 'unknown'
            confidence = 0
        elif bullish_count > bearish_count * 2:
            regime = 'expansion'
            confidence = bullish_count / total
        elif bearish_count > bullish_count * 2:
            regime = 'contraction'
            confidence = bearish_count / total
        elif bullish_count > bearish_count:
            regime = 'peak'  # Growth positive but slowing
            confidence = 0.5
        else:
            regime = 'trough'  # Growth negative but stabilizing
            confidence = 0.5
        
        # Risk level
        yield_curve = self.get_yield_curve()
        if yield_curve.recession_probability > 0.5:
            risk_level = 'high'
        elif yield_curve.recession_probability > 0.3:
            risk_level = 'moderate'
        else:
            risk_level = 'low'
        
        # Positioning recommendation
        if regime == 'expansion':
            positioning = "Risk-on: Favor equities, cyclicals, small caps"
        elif regime == 'peak':
            positioning = "Defensive: Reduce risk, increase quality, add bonds"
        elif regime == 'contraction':
            positioning = "Risk-off: Underweight equities, favor bonds, cash, gold"
        elif regime == 'trough':
            positioning = "Early cycle: Begin adding risk, favor value, commodities"
        else:
            positioning = "Neutral: Balanced allocation"
        
        return MacroRegime(
            regime=regime,
            confidence=confidence,
            leading_indicators={k: v['signal'] for k, v in scores.items()},
            risk_level=risk_level,
            recommended_positioning=positioning
        )
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get all macro data for dashboard display."""
        # Key indicators
        key_indicators = [
            'fed_funds_rate', 'treasury_10y', 'unemployment_rate',
            'cpi', 'gdp_growth', 'consumer_sentiment', 'vix'
        ]
        
        indicators = {}
        for ind in key_indicators:
            data = self.fred.get_indicator(ind)
            if data:
                indicators[ind] = {
                    'value': data.value,
                    'change': data.change,
                    'trend': data.trend,
                    'signal': data.signal,
                    'date': data.date.strftime('%Y-%m-%d') if data.date else None
                }
        
        yield_curve = self.get_yield_curve()
        regime = self.detect_macro_regime()
        
        return {
            'indicators': indicators,
            'yield_curve': {
                'rates': yield_curve.rates,
                'is_inverted': yield_curve.is_inverted,
                'spread_10y_2y': yield_curve.spread_10y_2y,
                'recession_probability': yield_curve.recession_probability
            },
            'regime': {
                'name': regime.regime,
                'confidence': regime.confidence,
                'risk_level': regime.risk_level,
                'positioning': regime.recommended_positioning
            },
            'generated_at': datetime.now().isoformat()
        }
    
    def calculate_market_correlation(
        self,
        ticker: str,
        indicators: List[str] = None
    ) -> Dict[str, float]:
        """
        Calculate correlation between a stock/index and macro indicators.
        
        Args:
            ticker: Stock or index symbol
            indicators: List of indicator names to correlate
            
        Returns:
            Dictionary of correlations
        """
        if indicators is None:
            indicators = ['treasury_10y', 'vix', 'spread_10y_2y', 'cpi']
        
        try:
            import yfinance as yf
            
            # Get stock data
            stock = yf.download(ticker, period='2y', progress=False)
            if stock.empty:
                return {}
            
            stock_returns = stock['Close'].pct_change().dropna()
            
            correlations = {}
            for ind_name in indicators:
                series_id = self.fred.INDICATORS.get(ind_name)
                if not series_id:
                    continue
                
                ind_data = self.fred.get_series(series_id)
                if ind_data.empty:
                    continue
                
                # Align dates
                aligned = pd.concat([stock_returns, ind_data], axis=1).dropna()
                if len(aligned) > 30:
                    corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
                    correlations[ind_name] = corr
            
            return correlations
            
        except Exception as e:
            logger.error(f"Correlation calculation error: {e}")
            return {}


# ============================================================================
# STREAMLIT RENDERING
# ============================================================================

def render_macro_dashboard():
    """Render the macro indicators dashboard in Streamlit."""
    import streamlit as st
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    st.subheader(" Macro Indicators Dashboard")
    
    analyzer = MacroAnalyzer()
    
    # Get all data
    with st.spinner("Loading macro data..."):
        data = analyzer.get_dashboard_data()
    
    # Regime indicator
    regime = data['regime']
    regime_colors = {
        'expansion': '#00C853',
        'peak': '#FFC107',
        'contraction': '#FF5722',
        'trough': '#2196F3',
        'unknown': '#9E9E9E'
    }
    
    st.markdown(f"""
    <div style="padding: 20px; border-radius: 10px; background: {regime_colors.get(regime['name'], '#9E9E9E')}22; 
                border: 2px solid {regime_colors.get(regime['name'], '#9E9E9E')};">
        <h3 style="margin: 0; color: {regime_colors.get(regime['name'], '#9E9E9E')};">
            Current Regime: {regime['name'].upper()}
        </h3>
        <p style="margin: 5px 0;">Confidence: {regime['confidence']*100:.0f}% | Risk Level: {regime['risk_level'].upper()}</p>
        <p style="margin: 5px 0;"><strong>Positioning:</strong> {regime['positioning']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("")
    
    # Key indicators
    st.write("### Key Economic Indicators")
    
    indicator_display = {
        'fed_funds_rate': ('Fed Funds Rate', '%'),
        'treasury_10y': ('10Y Treasury', '%'),
        'unemployment_rate': ('Unemployment', '%'),
        'cpi': ('CPI (YoY)', 'index'),
        'gdp_growth': ('GDP Growth', '%'),
        'consumer_sentiment': ('Consumer Sentiment', 'index'),
        'vix': ('VIX', 'index')
    }
    
    cols = st.columns(4)
    for idx, (key, (name, unit)) in enumerate(indicator_display.items()):
        ind = data['indicators'].get(key, {})
        with cols[idx % 4]:
            value = ind.get('value', 'N/A')
            change = ind.get('change', 0)
            signal = ind.get('signal', 'neutral')
            
            signal_icon = '' if signal == 'bullish' else '' if signal == 'bearish' else ''
            
            if isinstance(value, (int, float)):
                st.metric(
                    f"{name} {signal_icon}",
                    f"{value:.2f}" if unit == '%' else f"{value:.0f}",
                    f"{change:+.2f}%" if change else None
                )
            else:
                st.metric(name, value)
    
    # Yield Curve
    st.write("### Yield Curve")
    
    yc = data['yield_curve']
    
    col1, col2, col3 = st.columns(3)
    with col1:
        color = "inverse" if yc['is_inverted'] else "normal"
        st.metric("Curve Status", "INVERTED " if yc['is_inverted'] else "Normal")
    with col2:
        st.metric("10Y-2Y Spread", f"{yc['spread_10y_2y']:.2f}%")
    with col3:
        st.metric("Recession Prob.", f"{yc['recession_probability']*100:.0f}%")
    
    # Yield curve chart
    if yc['rates']:
        tenors_order = ['3M', '1Y', '2Y', '5Y', '10Y', '30Y']
        x_vals = [t for t in tenors_order if t in yc['rates']]
        y_vals = [yc['rates'][t] for t in x_vals]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='lines+markers',
            line=dict(color='#00D4AA' if not yc['is_inverted'] else '#FF5722', width=3),
            marker=dict(size=10)
        ))
        
        fig.update_layout(
            title="Current Yield Curve",
            xaxis_title="Tenor",
            yaxis_title="Yield (%)",
            template='plotly_dark',
            height=300
        )
        
        st.plotly_chart(fig, width="stretch")
    
    # Market correlation section
    st.write("### Market-Macro Correlation")
    
    ticker = st.text_input("Enter ticker for correlation analysis:", value="SPY")
    
    if ticker:
        with st.spinner("Calculating correlations..."):
            correlations = analyzer.calculate_market_correlation(ticker)
        
        if correlations:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=list(correlations.keys()),
                y=list(correlations.values()),
                marker_color=['#00C853' if v > 0 else '#FF5722' for v in correlations.values()]
            ))
            
            fig.update_layout(
                title=f"{ticker} Correlation with Macro Indicators",
                xaxis_title="Indicator",
                yaxis_title="Correlation",
                template='plotly_dark',
                height=300
            )
            
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Could not calculate correlations. Check if ticker is valid.")


# Make features available
HAS_MACRO = True
