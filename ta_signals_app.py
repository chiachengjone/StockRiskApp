"""
TA SIGNALS DASHBOARD
====================
Technical Analysis Signals Extension for Stock Risk App

Features:
- Technical Indicators (SMA, EMA, RSI, MACD, BB, ADX, Stochastic, ATR)
- Signal Generation (MA Crossover, RSI, MACD, Bollinger)
- Risk-Filtered Signals
- Interactive Candlestick Charts
- Signal Dashboard and History
- Screener Mode
- Backtesting Module
- Portfolio Signals

Author: Stock Risk App | Feb 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
import warnings
warnings.filterwarnings('ignore')

# Import services
from services.ta_service import (
    TAService,
    get_all_indicators,
    SMA_PERIODS,
    EMA_PERIODS,
    RSI_OVERBOUGHT,
    RSI_OVERSOLD
)
from services.signals_service import (
    SignalsService,
    Signal,
    SignalType,
    RiskMetrics,
    BacktestResult,
    calculate_risk_metrics,
    generate_ma_crossover_signals,
    generate_rsi_signals,
    generate_macd_signals,
    generate_bollinger_signals,
    calculate_combined_signal_score,
    MAX_VOLATILITY,
    MAX_BETA,
    MIN_SHARPE
)

# Import new services for enhanced features
from services.pattern_service import PatternService, PatternType, PatternDirection
from services.mtf_service import MTFService, TrendDirection, TimeframeAlignment
from services.regime_service import RegimeService, MarketRegime, VolatilityRegime
from services.divergence_service import DivergenceService, DivergenceType
from services.strategy_service import StrategyService
from services.advanced_backtest_service import AdvancedBacktestService
from services.alerts_service import AlertsService, AlertType, AlertPriority, AlertStatus
from services.ta_sentiment_service import SentimentService, SentimentLevel
from services.reporting_service import ReportingService, SignalOutcome

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================

# Color palette (consistent with main app)
COLORS = {
    'primary': '#007AFF',
    'secondary': '#5856D6',
    'success': '#34C759',
    'warning': '#FF9500',
    'danger': '#FF3B30',
    'gray': '#8E8E93',
    'light': '#F2F2F7',
    'dark': '#1C1C1E',
    'buy': '#34C759',
    'sell': '#FF3B30',
    'hold': '#FF9500',
    'bullish': '#00C853',
    'bearish': '#FF1744',
    'neutral': '#9E9E9E'
}

# Timeframe options
TIMEFRAMES = {
    '1d': '1 Day',
    '5d': '5 Days',
    '1mo': '1 Month',
    '3mo': '3 Months',
    '6mo': '6 Months',
    '1y': '1 Year',
    '2y': '2 Years',
    '5y': '5 Years'
}

# Popular stocks for screener
POPULAR_STOCKS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B',
    'UNH', 'JNJ', 'JPM', 'V', 'PG', 'XOM', 'HD', 'CVX', 'MA', 'ABBV',
    'MRK', 'PEP', 'KO', 'COST', 'AVGO', 'LLY', 'WMT', 'TMO', 'MCD', 'CSCO',
    'ACN', 'ABT', 'DHR', 'NEE', 'VZ', 'ADBE', 'CRM', 'NKE', 'INTC', 'AMD'
]

# Sector mapping
SECTOR_MAPPING = {
    'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
    'AMZN': 'Consumer Cyclical', 'NVDA': 'Technology', 'META': 'Technology',
    'TSLA': 'Consumer Cyclical', 'BRK-B': 'Financial', 'UNH': 'Healthcare',
    'JNJ': 'Healthcare', 'JPM': 'Financial', 'V': 'Financial',
    'PG': 'Consumer Defensive', 'XOM': 'Energy', 'HD': 'Consumer Cyclical',
    'CVX': 'Energy', 'MA': 'Financial', 'ABBV': 'Healthcare',
    'MRK': 'Healthcare', 'PEP': 'Consumer Defensive', 'KO': 'Consumer Defensive',
    'COST': 'Consumer Defensive', 'AVGO': 'Technology', 'LLY': 'Healthcare',
    'WMT': 'Consumer Defensive', 'TMO': 'Healthcare', 'MCD': 'Consumer Cyclical',
    'CSCO': 'Technology', 'ACN': 'Technology', 'ABT': 'Healthcare',
    'DHR': 'Healthcare', 'NEE': 'Utilities', 'VZ': 'Communication',
    'ADBE': 'Technology', 'CRM': 'Technology', 'NKE': 'Consumer Cyclical',
    'INTC': 'Technology', 'AMD': 'Technology'
}

# Chart template
CHART_CONFIG = {
    'displayModeBar': True,
    'displaylogo': False,
    'modeBarButtonsToRemove': ['lasso2d', 'select2d']
}


# ============================================================================
# DATA FUNCTIONS
# ============================================================================

@st.cache_data(ttl=300)
def fetch_stock_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """
    Fetch stock data from Yahoo Finance.
    
    Args:
        ticker: Stock ticker symbol
        period: Data period
        
    Returns:
        DataFrame with OHLCV data
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        
        if df.empty:
            logger.warning(f"No data returned for {ticker}")
            return pd.DataFrame()
        
        return df
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def fetch_stock_info(ticker: str) -> Dict[str, Any]:
    """
    Fetch stock information.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dictionary with stock info
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return info if info else {}
    except Exception as e:
        logger.error(f"Error fetching info for {ticker}: {e}")
        return {}


@st.cache_data(ttl=300)
def fetch_multiple_stocks(tickers: List[str], period: str = "1y") -> Dict[str, pd.DataFrame]:
    """
    Fetch data for multiple stocks.
    
    Args:
        tickers: List of ticker symbols
        period: Data period
        
    Returns:
        Dictionary of {ticker: DataFrame}
    """
    results = {}
    for ticker in tickers:
        df = fetch_stock_data(ticker, period)
        if not df.empty:
            results[ticker] = df
    return results


# ============================================================================
# CHART FUNCTIONS
# ============================================================================

def create_candlestick_chart(
    df: pd.DataFrame,
    indicators: Dict[str, pd.Series],
    signals: List[Signal],
    title: str = "Price Chart"
) -> go.Figure:
    """
    Create a candlestick chart with indicators and signals.
    
    Args:
        df: OHLCV DataFrame
        indicators: Dictionary of indicator series
        signals: List of Signal objects
        title: Chart title
        
    Returns:
        Plotly Figure
    """
    # Create subplots: Price, RSI, MACD, Volume
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.15, 0.15, 0.2],
        subplot_titles=(title, 'RSI', 'MACD', 'Volume')
    )
    
    # Handle DataFrame structure
    if isinstance(df.columns, pd.MultiIndex):
        open_prices = df['Open'].iloc[:, 0] if isinstance(df['Open'], pd.DataFrame) else df['Open']
        high = df['High'].iloc[:, 0] if isinstance(df['High'], pd.DataFrame) else df['High']
        low = df['Low'].iloc[:, 0] if isinstance(df['Low'], pd.DataFrame) else df['Low']
        close = df['Close'].iloc[:, 0] if isinstance(df['Close'], pd.DataFrame) else df['Close']
        volume = df['Volume'].iloc[:, 0] if isinstance(df['Volume'], pd.DataFrame) else df['Volume']
    else:
        open_prices = df['Open']
        high = df['High']
        low = df['Low']
        close = df['Close']
        volume = df['Volume']
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=open_prices,
            high=high,
            low=low,
            close=close,
            name='Price',
            increasing_line_color=COLORS['success'],
            decreasing_line_color=COLORS['danger']
        ),
        row=1, col=1
    )
    
    # Add SMAs
    for period in SMA_PERIODS:
        key = f'SMA_{period}'
        if key in indicators:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=indicators[key],
                    name=f'SMA {period}',
                    line=dict(width=1.5),
                    opacity=0.8
                ),
                row=1, col=1
            )
    
    # Add EMAs
    for period in EMA_PERIODS:
        key = f'EMA_{period}'
        if key in indicators:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=indicators[key],
                    name=f'EMA {period}',
                    line=dict(width=1.5, dash='dash'),
                    opacity=0.8
                ),
                row=1, col=1
            )
    
    # Bollinger Bands
    if all(k in indicators for k in ['BB_Upper', 'BB_Middle', 'BB_Lower']):
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=indicators['BB_Upper'],
                name='BB Upper',
                line=dict(color=COLORS['gray'], width=1, dash='dot'),
                opacity=0.6
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=indicators['BB_Lower'],
                name='BB Lower',
                line=dict(color=COLORS['gray'], width=1, dash='dot'),
                fill='tonexty',
                fillcolor='rgba(142, 142, 147, 0.1)',
                opacity=0.6
            ),
            row=1, col=1
        )
    
    # Add buy/sell markers
    buy_signals = [s for s in signals if s.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]]
    sell_signals = [s for s in signals if s.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]]
    
    if buy_signals:
        buy_dates = [s.timestamp for s in buy_signals]
        buy_prices = [s.price for s in buy_signals]
        fig.add_trace(
            go.Scatter(
                x=buy_dates,
                y=buy_prices,
                mode='markers',
                name='Buy Signal',
                marker=dict(
                    symbol='triangle-up',
                    size=12,
                    color=COLORS['buy'],
                    line=dict(width=1, color='white')
                )
            ),
            row=1, col=1
        )
    
    if sell_signals:
        sell_dates = [s.timestamp for s in sell_signals]
        sell_prices = [s.price for s in sell_signals]
        fig.add_trace(
            go.Scatter(
                x=sell_dates,
                y=sell_prices,
                mode='markers',
                name='Sell Signal',
                marker=dict(
                    symbol='triangle-down',
                    size=12,
                    color=COLORS['sell'],
                    line=dict(width=1, color='white')
                )
            ),
            row=1, col=1
        )
    
    # RSI
    if 'RSI' in indicators:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=indicators['RSI'],
                name='RSI',
                line=dict(color=COLORS['secondary'], width=1.5)
            ),
            row=2, col=1
        )
        
        # Overbought/Oversold lines
        fig.add_hline(y=RSI_OVERBOUGHT, line_dash="dash", line_color=COLORS['danger'], 
                      opacity=0.5, row=2, col=1)
        fig.add_hline(y=RSI_OVERSOLD, line_dash="dash", line_color=COLORS['success'], 
                      opacity=0.5, row=2, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color=COLORS['gray'], 
                      opacity=0.3, row=2, col=1)
    
    # MACD
    if all(k in indicators for k in ['MACD', 'MACD_Signal', 'MACD_Histogram']):
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=indicators['MACD'],
                name='MACD',
                line=dict(color=COLORS['primary'], width=1.5)
            ),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=indicators['MACD_Signal'],
                name='Signal',
                line=dict(color=COLORS['warning'], width=1.5)
            ),
            row=3, col=1
        )
        
        # Histogram with conditional colors
        histogram = indicators['MACD_Histogram']
        colors = [COLORS['success'] if v >= 0 else COLORS['danger'] for v in histogram]
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=histogram,
                name='Histogram',
                marker_color=colors,
                opacity=0.6
            ),
            row=3, col=1
        )
    
    # Volume
    volume_colors = [COLORS['success'] if close.iloc[i] >= open_prices.iloc[i] 
                    else COLORS['danger'] for i in range(len(close))]
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=volume,
            name='Volume',
            marker_color=volume_colors,
            opacity=0.7
        ),
        row=4, col=1
    )
    
    # Layout
    fig.update_layout(
        height=900,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10)
        ),
        xaxis_rangeslider_visible=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(
            family='-apple-system, BlinkMacSystemFont, SF Pro Display, sans-serif',
            color='#fafafa'
        ),
        margin=dict(t=60, b=40, l=60, r=40)
    )
    
    # Update axes
    for i in range(1, 5):
        fig.update_xaxes(
            gridcolor='rgba(128,128,128,0.1)',
            zerolinecolor='rgba(128,128,128,0.2)',
            row=i, col=1
        )
        fig.update_yaxes(
            gridcolor='rgba(128,128,128,0.1)',
            zerolinecolor='rgba(128,128,128,0.2)',
            row=i, col=1
        )
    
    # Set RSI y-axis range
    fig.update_yaxes(range=[0, 100], row=2, col=1)
    
    return fig


def create_equity_curve_chart(equity_curve: pd.Series, title: str = "Equity Curve") -> go.Figure:
    """
    Create an equity curve chart for backtesting.
    
    Args:
        equity_curve: Series with equity values
        title: Chart title
        
    Returns:
        Plotly Figure
    """
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=equity_curve.index,
            y=equity_curve.values,
            mode='lines',
            name='Equity',
            line=dict(color=COLORS['primary'], width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 122, 255, 0.1)'
        )
    )
    
    # Add horizontal line for initial capital
    initial = equity_curve.iloc[0]
    fig.add_hline(
        y=initial,
        line_dash="dash",
        line_color=COLORS['gray'],
        opacity=0.5,
        annotation_text=f"Initial: ${initial:,.0f}"
    )
    
    fig.update_layout(
        title=title,
        height=400,
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(
            family='-apple-system, BlinkMacSystemFont, SF Pro Display, sans-serif',
            color='#fafafa'
        ),
        xaxis=dict(
            gridcolor='rgba(128,128,128,0.1)',
            zerolinecolor='rgba(128,128,128,0.2)'
        ),
        yaxis=dict(
            gridcolor='rgba(128,128,128,0.1)',
            zerolinecolor='rgba(128,128,128,0.2)',
            tickformat='$,.0f'
        ),
        margin=dict(t=60, b=40, l=80, r=40)
    )
    
    return fig


def create_signal_gauge(score: float, signal_type: SignalType) -> go.Figure:
    """
    Create a gauge chart for signal score.
    
    Args:
        score: Signal score (0-100)
        signal_type: Type of signal
        
    Returns:
        Plotly Figure
    """
    # Determine color based on score
    if score >= 70:
        color = COLORS['success']
    elif score >= 55:
        color = COLORS['bullish']
    elif score <= 30:
        color = COLORS['danger']
    elif score <= 45:
        color = COLORS['bearish']
    else:
        color = COLORS['warning']
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': signal_type.value, 'font': {'size': 16}},
        number={'suffix': '', 'font': {'size': 32}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': color},
            'bgcolor': 'rgba(0,0,0,0)',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 30], 'color': 'rgba(255, 59, 48, 0.2)'},
                {'range': [30, 45], 'color': 'rgba(255, 23, 68, 0.2)'},
                {'range': [45, 55], 'color': 'rgba(255, 149, 0, 0.2)'},
                {'range': [55, 70], 'color': 'rgba(0, 200, 83, 0.2)'},
                {'range': [70, 100], 'color': 'rgba(52, 199, 89, 0.2)'}
            ],
            'threshold': {
                'line': {'color': 'white', 'width': 2},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#fafafa'),
        margin=dict(t=40, b=20, l=40, r=40)
    )
    
    return fig


# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_signal_badge(signal_type: SignalType, score: float) -> str:
    """
    Render a signal badge as HTML.
    
    Args:
        signal_type: Type of signal
        score: Signal score
        
    Returns:
        HTML string
    """
    colors_map = {
        SignalType.STRONG_BUY: COLORS['success'],
        SignalType.BUY: COLORS['bullish'],
        SignalType.HOLD: COLORS['warning'],
        SignalType.SELL: COLORS['bearish'],
        SignalType.STRONG_SELL: COLORS['danger']
    }
    
    color = colors_map.get(signal_type, COLORS['gray'])
    
    return f"""
    <span style="
        background-color: {color};
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 500;
    ">{signal_type.value} ({score:.0f})</span>
    """


def render_risk_badge(risk_score: float) -> str:
    """
    Render a risk score badge as HTML.
    
    Args:
        risk_score: Risk score (0-100, lower is better)
        
    Returns:
        HTML string
    """
    if risk_score < 30:
        color = COLORS['success']
        label = "Low Risk"
    elif risk_score < 60:
        color = COLORS['warning']
        label = "Medium Risk"
    else:
        color = COLORS['danger']
        label = "High Risk"
    
    return f"""
    <span style="
        background-color: {color};
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 500;
    ">{label} ({risk_score:.0f})</span>
    """


def render_indicator_card(name: str, value: float, interpretation: str) -> None:
    """
    Render an indicator card.
    
    Args:
        name: Indicator name
        value: Current value
        interpretation: Text interpretation
    """
    # Determine color based on interpretation
    if 'Overbought' in interpretation or 'Bearish' in interpretation or 'downtrend' in interpretation.lower():
        color = COLORS['danger']
    elif 'Oversold' in interpretation or 'Bullish' in interpretation or 'uptrend' in interpretation.lower():
        color = COLORS['success']
    else:
        color = COLORS['gray']
    
    st.markdown(f"""
    <div style="
        background: rgba(255,255,255,0.05);
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 8px;
        border-left: 3px solid {color};
    ">
        <div style="font-size: 12px; color: #a0a0a0; text-transform: uppercase; letter-spacing: 0.05em;">
            {name}
        </div>
        <div style="font-size: 24px; font-weight: 500; color: #fafafa; margin: 4px 0;">
            {value:.2f}
        </div>
        <div style="font-size: 13px; color: {color};">
            {interpretation}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# MAIN TABS
# ============================================================================

def render_single_stock_analysis(
    ta_service: TAService, 
    signals_service: SignalsService,
    pattern_service: PatternService = None,
    mtf_service: MTFService = None,
    divergence_service: DivergenceService = None
) -> None:
    """Render single stock analysis tab with extended features."""
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        ticker = st.text_input("Enter Ticker Symbol", value="AAPL", key="single_ticker").upper()
    
    with col2:
        timeframe = st.selectbox(
            "Timeframe",
            options=list(TIMEFRAMES.keys()),
            format_func=lambda x: TIMEFRAMES[x],
            index=5,  # Default to 1y
            key="single_timeframe"
        )
    
    with col3:
        analyze_btn = st.button("Analyze", type="primary", use_container_width=True, key="analyze_btn")
    
    if analyze_btn or ticker:
        with st.spinner("Fetching data..."):
            df = fetch_stock_data(ticker, timeframe)
            
            if df.empty:
                st.error(f"Could not fetch data for {ticker}. Please check the symbol.")
                return
            
            # Calculate indicators
            indicators = ta_service.calculate_all(df)
            
            # Get signals
            signals = signals_service.generate_all_signals(ticker, df)
            
            # Get current signal score
            score, signal_type, reason = signals_service.get_current_signal(ticker, df)
        
        # Display current signal
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            # Signal Gauge
            fig_gauge = create_signal_gauge(score, signal_type)
            st.plotly_chart(fig_gauge, use_container_width=True, config=CHART_CONFIG)
        
        with col2:
            st.markdown("### Current Signal Analysis")
            st.markdown(f"**Signal:** {signal_type.value}")
            st.markdown(f"**Score:** {score:.1f}/100")
            st.markdown(f"**Reason:** {reason}")
            
            # Stock info
            info = fetch_stock_info(ticker)
            if info:
                st.markdown("---")
                mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                with mcol1:
                    current_price = info.get('regularMarketPrice', info.get('previousClose', 0))
                    st.metric("Price", f"${current_price:.2f}")
                with mcol2:
                    change = info.get('regularMarketChangePercent', 0)
                    st.metric("Change", f"{change:.2f}%")
                with mcol3:
                    volume = info.get('regularMarketVolume', 0)
                    st.metric("Volume", f"{volume/1e6:.1f}M")
                with mcol4:
                    market_cap = info.get('marketCap', 0)
                    st.metric("Market Cap", f"${market_cap/1e9:.1f}B")
        
        with col3:
            # Risk metrics
            close = df['Close'] if not isinstance(df['Close'], pd.DataFrame) else df['Close'].iloc[:, 0]
            risk = calculate_risk_metrics(close)
            
            st.markdown("### Risk Metrics")
            st.markdown(f"**Volatility:** {risk.volatility*100:.1f}%")
            st.markdown(f"**Beta:** {risk.beta:.2f}")
            st.markdown(f"**Sharpe:** {risk.sharpe:.2f}")
            st.markdown(f"**Max DD:** {risk.max_drawdown*100:.1f}%")
            
            # Risk filter status
            if risk.passes_filter():
                st.success("Passes risk filter")
            else:
                st.warning("Fails risk filter")
        
        # Chart
        st.markdown("---")
        st.markdown("### Price Chart with Indicators")
        
        fig = create_candlestick_chart(df, indicators, signals, title=f"{ticker} Technical Analysis")
        st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)
        
        # Indicator Summary
        st.markdown("---")
        st.markdown("### Indicator Summary")
        
        summary = ta_service.get_indicator_summary(df)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            rsi_val = summary['indicators'].get('RSI', 50)
            rsi_interp = summary['interpretations'].get('RSI', 'Neutral')
            render_indicator_card("RSI (14)", rsi_val, rsi_interp)
        
        with col2:
            macd_val = summary['indicators'].get('MACD', 0)
            macd_interp = summary['interpretations'].get('MACD', 'Neutral')
            render_indicator_card("MACD", macd_val, macd_interp)
        
        with col3:
            adx_val = summary['indicators'].get('ADX', 0)
            adx_interp = summary['interpretations'].get('ADX', 'Neutral')
            render_indicator_card("ADX (14)", adx_val, adx_interp)
        
        with col4:
            bb_interp = summary['interpretations'].get('BB', 'Within Bands')
            close_val = summary['indicators'].get('Close', 0)
            render_indicator_card("Price Position", close_val, bb_interp)
        
        # Recent Signals Table
        if signals:
            st.markdown("---")
            st.markdown("### Recent Signals")
            
            # Convert to DataFrame
            signals_data = []
            for s in sorted(signals, key=lambda x: x.timestamp, reverse=True)[:20]:
                signals_data.append({
                    'Timestamp': s.timestamp.strftime('%Y-%m-%d') if hasattr(s.timestamp, 'strftime') else str(s.timestamp)[:10],
                    'Type': s.signal_type.value,
                    'Price': f"${s.price:.2f}",
                    'Reason': s.reason,
                    'Score': f"{s.score:.0f}"
                })
            
            if signals_data:
                signals_df = pd.DataFrame(signals_data)
                st.dataframe(signals_df, use_container_width=True, hide_index=True)


def render_screener_mode(ta_service: TAService, signals_service: SignalsService) -> None:
    """Render screener mode tab."""
    
    st.markdown("### Stock Screener")
    st.markdown("Scan multiple stocks for trading signals")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        custom_tickers = st.text_input(
            "Custom Tickers (comma-separated)",
            placeholder="AAPL, MSFT, GOOGL",
            key="screener_custom"
        )
    
    with col2:
        signal_filter = st.multiselect(
            "Signal Filter",
            options=["BUY", "SELL", "STRONG_BUY", "STRONG_SELL", "HOLD"],
            default=["BUY", "STRONG_BUY"],
            key="screener_signal_filter"
        )
    
    with col3:
        sector_filter = st.selectbox(
            "Sector Filter",
            options=["All"] + list(set(SECTOR_MAPPING.values())),
            key="screener_sector"
        )
    
    col1, col2 = st.columns([1, 1])
    with col1:
        use_popular = st.checkbox("Include Popular Stocks", value=True, key="use_popular")
    with col2:
        min_score = st.slider("Minimum Signal Score", 0, 100, 50, key="min_score")
    
    scan_btn = st.button("Scan Stocks", type="primary", key="scan_btn")
    
    if scan_btn:
        # Build ticker list
        tickers = []
        
        if custom_tickers:
            tickers.extend([t.strip().upper() for t in custom_tickers.split(",")])
        
        if use_popular:
            # Filter by sector if selected
            if sector_filter != "All":
                tickers.extend([t for t, s in SECTOR_MAPPING.items() if s == sector_filter])
            else:
                tickers.extend(POPULAR_STOCKS[:20])
        
        tickers = list(set(tickers))  # Remove duplicates
        
        if not tickers:
            st.warning("Please enter tickers or enable Popular Stocks")
            return
        
        # Progress
        progress = st.progress(0)
        status = st.empty()
        
        results = []
        
        for i, ticker in enumerate(tickers):
            status.text(f"Scanning {ticker}... ({i+1}/{len(tickers)})")
            progress.progress((i + 1) / len(tickers))
            
            try:
                df = fetch_stock_data(ticker, "1y")
                
                if df.empty or len(df) < 50:
                    continue
                
                # Get current signal
                score, signal_type, reason = signals_service.get_current_signal(ticker, df)
                
                # Apply filters
                if signal_type.value not in signal_filter:
                    continue
                
                if score < min_score:
                    continue
                
                # Calculate risk
                close = df['Close'] if not isinstance(df['Close'], pd.DataFrame) else df['Close'].iloc[:, 0]
                risk = calculate_risk_metrics(close)
                
                # Get price info
                current_price = float(close.iloc[-1])
                prev_price = float(close.iloc[-2]) if len(close) > 1 else current_price
                change_pct = ((current_price - prev_price) / prev_price) * 100
                
                results.append({
                    'Symbol': ticker,
                    'Price': f"${current_price:.2f}",
                    'Change': f"{change_pct:+.2f}%",
                    'Signal': signal_type.value,
                    'Score': score,
                    'Reason': reason[:50] + "..." if len(reason) > 50 else reason,
                    'Volatility': f"{risk.volatility*100:.1f}%",
                    'Sharpe': f"{risk.sharpe:.2f}",
                    'Sector': SECTOR_MAPPING.get(ticker, 'Unknown')
                })
                
            except Exception as e:
                logger.error(f"Error scanning {ticker}: {e}")
                continue
        
        progress.empty()
        status.empty()
        
        if results:
            st.success(f"Found {len(results)} stocks matching criteria")
            
            # Sort by score
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values('Score', ascending=False)
            
            st.dataframe(
                results_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Score': st.column_config.ProgressColumn(
                        'Score',
                        min_value=0,
                        max_value=100,
                        format='%.0f'
                    )
                }
            )
        else:
            st.info("No stocks found matching the criteria")


def render_backtest_module(ta_service: TAService, signals_service: SignalsService) -> None:
    """Render backtest module tab."""
    
    st.markdown("### Backtest Signals")
    st.markdown("Test signal performance on historical data")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        ticker = st.text_input("Ticker Symbol", value="AAPL", key="backtest_ticker").upper()
    
    with col2:
        period = st.selectbox(
            "Backtest Period",
            options=['1y', '2y', '5y'],
            index=1,
            key="backtest_period"
        )
    
    with col3:
        initial_capital = st.number_input(
            "Initial Capital ($)",
            min_value=1000,
            max_value=10000000,
            value=100000,
            step=10000,
            key="backtest_capital"
        )
    
    with col4:
        position_size = st.slider(
            "Position Size (%)",
            min_value=5,
            max_value=100,
            value=10,
            step=5,
            key="backtest_position"
        ) / 100
    
    backtest_btn = st.button("Run Backtest", type="primary", key="backtest_btn")
    
    if backtest_btn:
        with st.spinner("Running backtest..."):
            df = fetch_stock_data(ticker, period)
            
            if df.empty:
                st.error(f"Could not fetch data for {ticker}")
                return
            
            # Run backtest
            result = signals_service.backtest_symbol(
                ticker,
                df,
                initial_capital=initial_capital,
                position_size=position_size
            )
        
        if result.total_trades == 0:
            st.warning("No trades were executed during the backtest period")
            return
        
        # Results summary
        st.markdown("---")
        st.markdown("### Backtest Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Trades", result.total_trades)
            st.metric("Winning Trades", result.winning_trades)
        
        with col2:
            color = "normal" if result.win_rate >= 50 else "inverse"
            st.metric("Win Rate", f"{result.win_rate:.1f}%")
            st.metric("Losing Trades", result.losing_trades)
        
        with col3:
            color = "normal" if result.total_return >= 0 else "inverse"
            st.metric("Total Return", f"{result.total_return:.2f}%")
            st.metric("Profit Factor", f"{result.profit_factor:.2f}")
        
        with col4:
            st.metric("Max Drawdown", f"{result.max_drawdown:.2f}%")
            st.metric("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")
        
        # Equity curve
        st.markdown("---")
        st.markdown("### Equity Curve")
        
        fig = create_equity_curve_chart(result.equity_curve, f"{ticker} Backtest Equity Curve")
        st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)
        
        # Trades table
        if result.trades:
            st.markdown("---")
            st.markdown("### Trade History")
            
            trades_data = []
            for t in result.trades:
                trades_data.append({
                    'Date': t['date'].strftime('%Y-%m-%d') if hasattr(t['date'], 'strftime') else str(t['date'])[:10],
                    'Type': t['type'],
                    'Price': f"${t['price']:.2f}",
                    'Shares': f"{t['shares']:.2f}",
                    'Value': f"${t['value']:.2f}",
                    'P/L': f"${t.get('pnl', 0):.2f}" if 'pnl' in t else "-",
                    'P/L %': f"{t.get('pnl_pct', 0):.2f}%" if 'pnl_pct' in t else "-",
                    'Reason': t.get('reason', '')[:40]
                })
            
            trades_df = pd.DataFrame(trades_data)
            st.dataframe(trades_df, use_container_width=True, hide_index=True)


def render_portfolio_signals(ta_service: TAService, signals_service: SignalsService) -> None:
    """Render portfolio signals tab."""
    
    st.markdown("### Portfolio Signal Analysis")
    st.markdown("Aggregate signals for portfolio holdings")
    
    # Portfolio input
    col1, col2 = st.columns([2, 1])
    
    with col1:
        portfolio_input = st.text_area(
            "Enter Portfolio (Symbol: Weight)",
            value="AAPL: 25\nMSFT: 20\nGOOGL: 15\nAMZN: 15\nNVDA: 10\nMETA: 10\nTSLA: 5",
            height=200,
            key="portfolio_input"
        )
    
    with col2:
        st.markdown("**Format:**")
        st.markdown("- One holding per line")
        st.markdown("- Format: SYMBOL: WEIGHT")
        st.markdown("- Weights in percentages")
        st.markdown("- Example: AAPL: 25")
    
    analyze_portfolio_btn = st.button("Analyze Portfolio", type="primary", key="analyze_portfolio_btn")
    
    if analyze_portfolio_btn:
        # Parse portfolio
        portfolio = {}
        for line in portfolio_input.strip().split('\n'):
            if ':' in line:
                parts = line.split(':')
                symbol = parts[0].strip().upper()
                try:
                    weight = float(parts[1].strip())
                    portfolio[symbol] = weight
                except ValueError:
                    continue
        
        if not portfolio:
            st.error("Could not parse portfolio. Please check the format.")
            return
        
        # Normalize weights
        total_weight = sum(portfolio.values())
        portfolio = {k: v/total_weight*100 for k, v in portfolio.items()}
        
        with st.spinner("Analyzing portfolio..."):
            result = signals_service.get_portfolio_signals(portfolio, period="1y")
        
        if not result['symbols']:
            st.error("Could not analyze any symbols in the portfolio")
            return
        
        # Portfolio aggregate signal
        st.markdown("---")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            agg_score = result['aggregate_score']
            agg_signal = result['aggregate_signal']
            
            fig = create_signal_gauge(agg_score, agg_signal)
            st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)
        
        with col2:
            st.markdown("### Portfolio Signal Summary")
            st.markdown(f"**Aggregate Signal:** {agg_signal.value}")
            st.markdown(f"**Aggregate Score:** {agg_score:.1f}/100")
            
            # Signal interpretation
            if agg_score >= 70:
                st.success("Portfolio shows strong bullish signals across holdings")
            elif agg_score >= 55:
                st.info("Portfolio shows moderate bullish bias")
            elif agg_score <= 30:
                st.error("Portfolio shows strong bearish signals - consider reducing exposure")
            elif agg_score <= 45:
                st.warning("Portfolio shows moderate bearish bias")
            else:
                st.info("Portfolio signals are mixed - hold current positions")
        
        # Individual holdings
        st.markdown("---")
        st.markdown("### Individual Holdings")
        
        holdings_data = []
        for symbol, data in result['symbols'].items():
            holdings_data.append({
                'Symbol': symbol,
                'Weight': f"{data['weight']*100:.1f}%",
                'Signal': data['signal'],
                'Score': data['score'],
                'Reason': data['reason'][:50] + "..." if len(data['reason']) > 50 else data['reason']
            })
        
        holdings_df = pd.DataFrame(holdings_data)
        holdings_df = holdings_df.sort_values('Score', ascending=False)
        
        st.dataframe(
            holdings_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                'Score': st.column_config.ProgressColumn(
                    'Score',
                    min_value=0,
                    max_value=100,
                    format='%.0f'
                )
            }
        )
        
        # Signal distribution chart
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Signal Distribution")
            
            signal_counts = {}
            for data in result['symbols'].values():
                sig = data['signal']
                signal_counts[sig] = signal_counts.get(sig, 0) + 1
            
            colors_list = [COLORS['success'], COLORS['bullish'], COLORS['warning'], 
                          COLORS['bearish'], COLORS['danger']]
            
            fig = go.Figure(data=[go.Pie(
                labels=list(signal_counts.keys()),
                values=list(signal_counts.values()),
                marker_colors=colors_list[:len(signal_counts)],
                hole=0.4
            )])
            
            fig.update_layout(
                height=300,
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#fafafa'),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2),
                margin=dict(t=20, b=60, l=20, r=20)
            )
            
            st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)
        
        with col2:
            st.markdown("### Score by Holding")
            
            symbols = list(result['symbols'].keys())
            scores = [result['symbols'][s]['score'] for s in symbols]
            weights = [result['symbols'][s]['weight'] * 100 for s in symbols]
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=symbols,
                y=scores,
                name='Signal Score',
                marker_color=[COLORS['success'] if s >= 60 else 
                             COLORS['warning'] if s >= 40 else 
                             COLORS['danger'] for s in scores]
            ))
            
            fig.update_layout(
                height=300,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#fafafa'),
                xaxis=dict(gridcolor='rgba(128,128,128,0.1)'),
                yaxis=dict(
                    gridcolor='rgba(128,128,128,0.1)',
                    range=[0, 100],
                    title='Score'
                ),
                margin=dict(t=20, b=40, l=60, r=20)
            )
            
            st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)


def render_signal_dashboard(ta_service: TAService, signals_service: SignalsService) -> None:
    """Render signal dashboard tab."""
    
    st.markdown("### Active Signals Dashboard")
    
    # Quick scan of popular stocks
    col1, col2 = st.columns([2, 1])
    
    with col1:
        scan_tickers = st.text_input(
            "Tickers to Monitor (comma-separated)",
            value="AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA",
            key="dashboard_tickers"
        )
    
    with col2:
        refresh_btn = st.button("Refresh Dashboard", type="primary", key="refresh_dashboard")
    
    tickers = [t.strip().upper() for t in scan_tickers.split(",")]
    
    if refresh_btn or 'dashboard_data' not in st.session_state:
        with st.spinner("Loading dashboard..."):
            dashboard_data = []
            
            for ticker in tickers:
                try:
                    df = fetch_stock_data(ticker, "6mo")
                    
                    if df.empty or len(df) < 50:
                        continue
                    
                    # Get current signal
                    score, signal_type, reason = signals_service.get_current_signal(ticker, df)
                    
                    # Get recent signals
                    recent_signals = signals_service.generate_all_signals(ticker, df)
                    recent_count = len([s for s in recent_signals 
                                       if hasattr(s.timestamp, 'date') and 
                                       (datetime.now().date() - s.timestamp.date()).days <= 5])
                    
                    # Calculate risk
                    close = df['Close'] if not isinstance(df['Close'], pd.DataFrame) else df['Close'].iloc[:, 0]
                    risk = calculate_risk_metrics(close)
                    
                    current_price = float(close.iloc[-1])
                    
                    dashboard_data.append({
                        'ticker': ticker,
                        'price': current_price,
                        'signal': signal_type,
                        'score': score,
                        'reason': reason,
                        'risk_score': 30 if risk.passes_filter() else 70,
                        'recent_signals': recent_count,
                        'volatility': risk.volatility,
                        'sharpe': risk.sharpe
                    })
                    
                except Exception as e:
                    logger.error(f"Error loading {ticker}: {e}")
                    continue
            
            st.session_state.dashboard_data = dashboard_data
    
    dashboard_data = st.session_state.get('dashboard_data', [])
    
    if not dashboard_data:
        st.info("No data available. Click Refresh Dashboard to load.")
        return
    
    # Display cards
    st.markdown("---")
    
    # Group by signal type
    buy_signals = [d for d in dashboard_data if d['signal'] in [SignalType.BUY, SignalType.STRONG_BUY]]
    sell_signals = [d for d in dashboard_data if d['signal'] in [SignalType.SELL, SignalType.STRONG_SELL]]
    hold_signals = [d for d in dashboard_data if d['signal'] == SignalType.HOLD]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"### Buy Signals ({len(buy_signals)})")
        for d in sorted(buy_signals, key=lambda x: x['score'], reverse=True):
            st.markdown(f"""
            <div style="
                background: rgba(52, 199, 89, 0.1);
                border-radius: 8px;
                padding: 12px;
                margin-bottom: 8px;
                border-left: 3px solid {COLORS['success']};
            ">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 600; font-size: 16px;">{d['ticker']}</span>
                    <span style="color: {COLORS['success']}; font-weight: 500;">{d['signal'].value}</span>
                </div>
                <div style="color: #a0a0a0; font-size: 13px; margin-top: 4px;">
                    ${d['price']:.2f} | Score: {d['score']:.0f}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"### Hold Signals ({len(hold_signals)})")
        for d in hold_signals:
            st.markdown(f"""
            <div style="
                background: rgba(255, 149, 0, 0.1);
                border-radius: 8px;
                padding: 12px;
                margin-bottom: 8px;
                border-left: 3px solid {COLORS['warning']};
            ">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 600; font-size: 16px;">{d['ticker']}</span>
                    <span style="color: {COLORS['warning']}; font-weight: 500;">{d['signal'].value}</span>
                </div>
                <div style="color: #a0a0a0; font-size: 13px; margin-top: 4px;">
                    ${d['price']:.2f} | Score: {d['score']:.0f}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"### Sell Signals ({len(sell_signals)})")
        for d in sorted(sell_signals, key=lambda x: x['score']):
            st.markdown(f"""
            <div style="
                background: rgba(255, 59, 48, 0.1);
                border-radius: 8px;
                padding: 12px;
                margin-bottom: 8px;
                border-left: 3px solid {COLORS['danger']};
            ">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 600; font-size: 16px;">{d['ticker']}</span>
                    <span style="color: {COLORS['danger']}; font-weight: 500;">{d['signal'].value}</span>
                </div>
                <div style="color: #a0a0a0; font-size: 13px; margin-top: 4px;">
                    ${d['price']:.2f} | Score: {d['score']:.0f}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Summary table
    st.markdown("---")
    st.markdown("### All Signals Summary")
    
    summary_data = []
    for d in dashboard_data:
        summary_data.append({
            'Symbol': d['ticker'],
            'Price': f"${d['price']:.2f}",
            'Signal': d['signal'].value,
            'Score': d['score'],
            'Risk': 'Low' if d['risk_score'] < 50 else 'High',
            'Volatility': f"{d['volatility']*100:.1f}%",
            'Sharpe': f"{d['sharpe']:.2f}",
            'Recent Signals': d['recent_signals']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Score', ascending=False)
    
    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Score': st.column_config.ProgressColumn(
                'Score',
                min_value=0,
                max_value=100,
                format='%.0f'
            )
        }
    )


# ============================================================================
# NEW FEATURE RENDER FUNCTIONS
# ============================================================================

def render_pattern_analysis(pattern_service: PatternService) -> None:
    """Render pattern recognition analysis tab."""
    
    st.markdown("### 🕯️ Pattern Recognition")
    st.markdown("Detect candlestick and chart patterns")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        ticker = st.text_input("Enter Ticker Symbol", value="AAPL", key="pattern_ticker").upper()
    
    with col2:
        analyze_btn = st.button("Detect Patterns", type="primary", key="pattern_analyze")
    
    if analyze_btn or ticker:
        with st.spinner("Analyzing patterns..."):
            df = fetch_stock_data(ticker, "6mo")
            
            if df.empty:
                st.error(f"Could not fetch data for {ticker}")
                return
            
            # Detect candlestick patterns
            candlestick_patterns = pattern_service.detect_candlestick_patterns(df)
            
            # Detect chart patterns
            chart_patterns = pattern_service.detect_chart_patterns(df)
            
            # Get summary
            summary = pattern_service.get_pattern_summary(df)
        
        # Display summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Patterns", summary.get('total_patterns', 0))
        with col2:
            st.metric("Bullish Signals", summary.get('bullish_count', 0))
        with col3:
            st.metric("Bearish Signals", summary.get('bearish_count', 0))
        with col4:
            bias = summary.get('overall_bias', 'Neutral')
            st.metric("Overall Bias", bias)
        
        st.markdown("---")
        
        # Recent candlestick patterns
        st.markdown("#### Recent Candlestick Patterns")
        
        if candlestick_patterns:
            recent = candlestick_patterns[-10:]  # Last 10 patterns
            pattern_data = []
            for p in recent:
                pattern_data.append({
                    'Date': p.end_date.strftime('%Y-%m-%d') if hasattr(p.end_date, 'strftime') else str(p.end_date)[:10],
                    'Pattern': p.pattern_type.value.replace('_', ' ').title(),
                    'Direction': '🟢 Bullish' if p.direction == PatternDirection.BULLISH else ('🔴 Bearish' if p.direction == PatternDirection.BEARISH else '⚪ Neutral'),
                    'Reliability': f"{p.reliability.value.title()}",
                    'Confidence': f"{p.confidence:.0f}%"
                })
            
            st.dataframe(pd.DataFrame(pattern_data), use_container_width=True, hide_index=True)
        else:
            st.info("No candlestick patterns detected in recent data")
        
        # Chart patterns
        st.markdown("#### Chart Patterns")
        
        if chart_patterns:
            for p in chart_patterns[-5:]:  # Last 5 chart patterns
                direction_emoji = '🟢' if p.direction == PatternDirection.BULLISH else ('🔴' if p.direction == PatternDirection.BEARISH else '⚪')
                st.markdown(f"""
                <div style="
                    background: rgba(30, 35, 45, 0.6);
                    padding: 12px 16px;
                    border-radius: 8px;
                    margin-bottom: 8px;
                    border-left: 3px solid {'#34C759' if p.direction == PatternDirection.BULLISH else '#FF3B30'};
                ">
                    <div style="font-weight: 600;">{direction_emoji} {p.pattern_type.value.replace('_', ' ').title()}</div>
                    <div style="color: #a0a0a0; font-size: 13px; margin-top: 4px;">
                        Detected at ${p.price_at_detection:.2f} | Reliability: {p.reliability.value.title()}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No chart patterns detected in recent data")


def render_market_regime(regime_service: RegimeService) -> None:
    """Render market regime detection tab."""
    
    st.markdown("### 🌡️ Market Regime Detection")
    st.markdown("Identify current market conditions and volatility regime")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        ticker = st.text_input("Enter Ticker Symbol", value="SPY", key="regime_ticker").upper()
    
    with col2:
        analyze_btn = st.button("Analyze Regime", type="primary", key="regime_analyze")
    
    if analyze_btn or ticker:
        with st.spinner("Analyzing market regime..."):
            df = fetch_stock_data(ticker, "2y")
            
            if df.empty:
                st.error(f"Could not fetch data for {ticker}")
                return
            
            # Analyze regime
            result = regime_service.analyze_regime(ticker, df)
        
        if not result:
            st.error("Could not analyze regime")
            return
        
        # Display current regime
        st.markdown("---")
        st.markdown("#### Current Market Conditions")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            regime = result.market_regime
            regime_colors = {
                MarketRegime.BULL: '#34C759',
                MarketRegime.BEAR: '#FF3B30',
                MarketRegime.SIDEWAYS: '#FF9500',
                MarketRegime.CRASH: '#FF1744',
                MarketRegime.RECOVERY: '#00C853'
            }
            color = regime_colors.get(regime, '#8E8E93')
            st.markdown(f"""
            <div style="text-align: center;">
                <div style="font-size: 14px; color: #a0a0a0; text-transform: uppercase;">Market Regime</div>
                <div style="font-size: 24px; font-weight: 600; color: {color}; margin-top: 8px;">
                    {regime.value.replace('_', ' ').title()}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            vol_regime = result.volatility_regime
            vol_colors = {
                VolatilityRegime.LOW: '#34C759',
                VolatilityRegime.NORMAL: '#007AFF',
                VolatilityRegime.HIGH: '#FF9500',
                VolatilityRegime.EXTREME: '#FF3B30'
            }
            vol_color = vol_colors.get(vol_regime, '#8E8E93')
            st.markdown(f"""
            <div style="text-align: center;">
                <div style="font-size: 14px; color: #a0a0a0; text-transform: uppercase;">Volatility</div>
                <div style="font-size: 24px; font-weight: 600; color: {vol_color}; margin-top: 8px;">
                    {vol_regime.value.title()}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            trend = result.trend_strength
            st.metric("Trend Strength", f"{trend:.0f}%")
        
        with col4:
            confidence = result.regime_probability
            st.metric("Confidence", f"{confidence:.0f}%")
        
        # Regime indicators
        st.markdown("---")
        st.markdown("#### Key Indicators")
        
        indicators = result.indicators
        col1, col2 = st.columns(2)
        
        with col1:
            for key, value in list(indicators.items())[:4]:
                if value is not None:
                    if isinstance(value, float):
                        st.markdown(f"**{key.replace('_', ' ').title()}:** {value:.2f}")
                    else:
                        st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")
        
        with col2:
            for key, value in list(indicators.items())[4:]:
                if value is not None:
                    if isinstance(value, float):
                        st.markdown(f"**{key.replace('_', ' ').title()}:** {value:.2f}")
                    else:
                        st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")
        
        # Trading implications
        st.markdown("---")
        st.markdown("#### Trading Implications")
        
        implications = {
            MarketRegime.BULL: "🟢 Favor long positions. Consider trend-following strategies with trailing stops.",
            MarketRegime.BEAR: "🔴 Exercise caution with long positions. Consider hedging or short opportunities.",
            MarketRegime.SIDEWAYS: "⚪ Range-bound market. Consider mean-reversion strategies.",
            MarketRegime.CRASH: "🔴 High risk environment. Reduce exposure and wait for stabilization.",
            MarketRegime.RECOVERY: "🟢 Potential reversal. Look for early entry opportunities with tight stops."
        }
        
        st.info(implications.get(result.market_regime, "Monitor market conditions closely."))


def render_alerts_manager(alerts_service: AlertsService) -> None:
    """Render alerts management tab."""
    
    st.markdown("### 🔔 Smart Alerts Manager")
    st.markdown("Create and manage technical analysis alerts")
    
    # Tabs for create/view
    alert_tab1, alert_tab2, alert_tab3 = st.tabs(["Create Alert", "Active Alerts", "History"])
    
    with alert_tab1:
        st.markdown("#### Create New Alert")
        
        col1, col2 = st.columns(2)
        
        with col1:
            alert_symbol = st.text_input("Symbol", value="AAPL", key="alert_symbol").upper()
            alert_name = st.text_input("Alert Name", f"My {alert_symbol} Alert", key="alert_name")
        
        with col2:
            alert_type = st.selectbox(
                "Alert Type",
                options=[
                    "Price Above", "Price Below", "RSI Oversold", "RSI Overbought",
                    "MACD Bullish", "MACD Bearish", "Golden Cross", "Volume Spike"
                ],
                key="alert_type"
            )
            
            alert_priority = st.selectbox(
                "Priority",
                options=["Low", "Medium", "High", "Critical"],
                index=1,
                key="alert_priority"
            )
        
        # Threshold based on alert type
        if "Price" in alert_type:
            threshold = st.number_input("Price Threshold ($)", value=150.0, key="alert_threshold")
        elif "RSI" in alert_type:
            default_val = 30.0 if "Oversold" in alert_type else 70.0
            threshold = st.number_input("RSI Threshold", value=default_val, min_value=0.0, max_value=100.0, key="alert_threshold")
        elif "Volume" in alert_type:
            threshold = st.number_input("Volume Multiplier", value=2.0, min_value=1.0, key="alert_threshold")
        else:
            threshold = 0.0
        
        if st.button("Create Alert", type="primary", key="create_alert_btn"):
            # Map alert type string to enum
            type_mapping = {
                "Price Above": AlertType.PRICE_ABOVE,
                "Price Below": AlertType.PRICE_BELOW,
                "RSI Oversold": AlertType.RSI_OVERSOLD,
                "RSI Overbought": AlertType.RSI_OVERBOUGHT,
                "MACD Bullish": AlertType.MACD_BULLISH,
                "MACD Bearish": AlertType.MACD_BEARISH,
                "Golden Cross": AlertType.GOLDEN_CROSS,
                "Volume Spike": AlertType.VOLUME_SPIKE
            }
            
            priority_mapping = {
                "Low": AlertPriority.LOW,
                "Medium": AlertPriority.MEDIUM,
                "High": AlertPriority.HIGH,
                "Critical": AlertPriority.CRITICAL
            }
            
            from services.alerts_service import AlertCondition
            
            condition = AlertCondition(
                alert_type=type_mapping[alert_type],
                symbol=alert_symbol,
                threshold=threshold
            )
            
            alert = alerts_service.create_alert(
                name=alert_name,
                conditions=[condition],
                priority=priority_mapping[alert_priority]
            )
            
            st.success(f"Alert created: {alert.name} (ID: {alert.id})")
    
    with alert_tab2:
        st.markdown("#### Active Alerts")
        
        active_alerts = alerts_service.get_alerts(status=AlertStatus.ACTIVE)
        
        if active_alerts:
            for alert in active_alerts[:20]:
                priority_colors = {
                    AlertPriority.LOW: '#8E8E93',
                    AlertPriority.MEDIUM: '#FF9500',
                    AlertPriority.HIGH: '#FF3B30',
                    AlertPriority.CRITICAL: '#FF1744'
                }
                color = priority_colors.get(alert.priority, '#8E8E93')
                
                st.markdown(f"""
                <div style="
                    background: rgba(30, 35, 45, 0.6);
                    padding: 12px 16px;
                    border-radius: 8px;
                    margin-bottom: 8px;
                    border-left: 3px solid {color};
                ">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: 600;">{alert.name}</span>
                        <span style="color: {color}; font-size: 12px; text-transform: uppercase;">{alert.priority.value}</span>
                    </div>
                    <div style="color: #a0a0a0; font-size: 13px; margin-top: 4px;">
                        ID: {alert.id} | Triggers: {alert.trigger_count}/{alert.max_triggers if alert.max_triggers > 0 else '∞'}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No active alerts. Create one in the 'Create Alert' tab.")
    
    with alert_tab3:
        st.markdown("#### Alert History")
        
        history = alerts_service.get_trigger_history(limit=50)
        
        if history:
            history_data = []
            for trigger in history:
                history_data.append({
                    'Time': trigger.triggered_at.strftime('%Y-%m-%d %H:%M'),
                    'Symbol': trigger.symbol,
                    'Alert': trigger.alert_name,
                    'Type': trigger.condition_type.value.replace('_', ' ').title(),
                    'Price': f"${trigger.price_at_trigger:.2f}",
                    'Priority': trigger.priority.value.title()
                })
            
            st.dataframe(pd.DataFrame(history_data), use_container_width=True, hide_index=True)
        else:
            st.info("No triggered alerts yet.")
        
        # Statistics
        stats = alerts_service.get_alert_statistics()
        
        st.markdown("---")
        st.markdown("#### Alert Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Alerts", stats['total_alerts'])
        with col2:
            st.metric("Active", stats['active'])
        with col3:
            st.metric("Triggered", stats['triggered'])
        with col4:
            st.metric("Total Triggers", stats['total_triggers'])


def render_performance_reports(reporting_service: ReportingService) -> None:
    """Render performance reporting tab."""
    
    st.markdown("### 📋 Performance Reports")
    st.markdown("Track signal performance and generate reports")
    
    # Tabs for different views
    report_tab1, report_tab2, report_tab3, report_tab4 = st.tabs([
        "Record Signal", "Signal History", "Report Card", "Export"
    ])
    
    with report_tab1:
        st.markdown("#### Record New Signal")
        
        col1, col2 = st.columns(2)
        
        with col1:
            symbol = st.text_input("Symbol", value="AAPL", key="report_symbol").upper()
            signal_type = st.selectbox("Signal Type", ["BUY", "SELL"], key="report_signal_type")
            entry_price = st.number_input("Entry Price ($)", value=150.0, min_value=0.01, key="report_entry")
        
        with col2:
            signal_source = st.selectbox(
                "Signal Source",
                ["RSI_Oversold", "RSI_Overbought", "MACD_Cross", "MA_Crossover", 
                 "Pattern", "Bollinger", "Manual"],
                key="report_source"
            )
            target_price = st.number_input("Target Price ($)", value=160.0, min_value=0.01, key="report_target")
            stop_loss = st.number_input("Stop Loss ($)", value=145.0, min_value=0.01, key="report_stop")
        
        confidence = st.slider("Confidence", 0.0, 1.0, 0.7, 0.1, key="report_confidence")
        notes = st.text_area("Notes", "", key="report_notes")
        
        if st.button("Record Signal", type="primary", key="record_signal_btn"):
            signal = reporting_service.record_signal(
                symbol=symbol,
                signal_type=signal_type,
                signal_source=signal_source,
                entry_price=entry_price,
                target_price=target_price,
                stop_loss=stop_loss,
                confidence=confidence,
                notes=notes
            )
            st.success(f"Signal recorded: {signal.signal_type} {signal.symbol} (ID: {signal.id})")
    
    with report_tab2:
        st.markdown("#### Signal History")
        
        # Filters
        col1, col2 = st.columns(2)
        with col1:
            filter_symbol = st.text_input("Filter by Symbol (optional)", "", key="history_filter_symbol")
        with col2:
            filter_outcome = st.selectbox(
                "Filter by Outcome",
                ["All", "Pending", "Win", "Loss", "Breakeven"],
                key="history_filter_outcome"
            )
        
        # Get signals
        outcome_map = {
            "All": None,
            "Pending": SignalOutcome.PENDING,
            "Win": SignalOutcome.WIN,
            "Loss": SignalOutcome.LOSS,
            "Breakeven": SignalOutcome.BREAKEVEN
        }
        
        signals = reporting_service.get_signal_history(
            symbol=filter_symbol if filter_symbol else None,
            outcome=outcome_map[filter_outcome]
        )
        
        if signals:
            signal_data = []
            for s in signals:
                outcome_emoji = {
                    SignalOutcome.WIN: '🟢',
                    SignalOutcome.LOSS: '🔴',
                    SignalOutcome.BREAKEVEN: '⚪',
                    SignalOutcome.PENDING: '⏳',
                    SignalOutcome.EXPIRED: '⌛'
                }
                
                signal_data.append({
                    'ID': s.id,
                    'Symbol': s.symbol,
                    'Type': s.signal_type,
                    'Source': s.signal_source,
                    'Entry': f"${s.entry_price:.2f}",
                    'Exit': f"${s.exit_price:.2f}" if s.exit_price else '-',
                    'P&L': f"{s.pnl_pct:+.1f}%" if s.pnl_pct != 0 else '-',
                    'Outcome': f"{outcome_emoji.get(s.outcome, '')} {s.outcome.value.title()}",
                    'Date': s.entry_date.strftime('%Y-%m-%d')
                })
            
            st.dataframe(pd.DataFrame(signal_data), use_container_width=True, hide_index=True)
            
            # Close signal form
            st.markdown("---")
            st.markdown("#### Close an Open Signal")
            
            open_signals = reporting_service.get_open_signals()
            if open_signals:
                signal_options = {f"{s.id} - {s.symbol} {s.signal_type}": s.id for s in open_signals}
                selected = st.selectbox("Select Signal", list(signal_options.keys()), key="close_signal_select")
                exit_price = st.number_input("Exit Price ($)", value=150.0, min_value=0.01, key="close_exit_price")
                
                if st.button("Close Signal", key="close_signal_btn"):
                    closed = reporting_service.close_signal(signal_options[selected], exit_price)
                    if closed:
                        st.success(f"Signal closed: {closed.outcome.value} ({closed.pnl_pct:+.1f}%)")
                        st.rerun()
        else:
            st.info("No signals recorded yet. Record a signal in the 'Record Signal' tab.")
    
    with report_tab3:
        st.markdown("#### Performance Report Card")
        
        if st.button("Generate Report", type="primary", key="generate_report_btn"):
            report = reporting_service.generate_report_card()
            
            # Overall metrics
            st.markdown("##### Overall Performance")
            metrics = report.overall_metrics
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Signals", metrics.total_signals)
            with col2:
                st.metric("Win Rate", f"{metrics.win_rate:.1f}%")
            with col3:
                st.metric("Profit Factor", f"{metrics.profit_factor:.2f}")
            with col4:
                st.metric("Total Return", f"{metrics.total_return_pct:+.1f}%")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Winning Trades", metrics.winning_signals)
            with col2:
                st.metric("Losing Trades", metrics.losing_signals)
            with col3:
                st.metric("Avg Win", f"{metrics.avg_win_pct:+.1f}%")
            with col4:
                st.metric("Avg Loss", f"{metrics.avg_loss_pct:.1f}%")
            
            # Recommendations
            st.markdown("---")
            st.markdown("##### Recommendations")
            
            for rec in report.recommendations:
                st.markdown(f"- {rec}")
            
            # By source
            if report.by_source:
                st.markdown("---")
                st.markdown("##### Performance by Signal Source")
                
                source_data = []
                for source, m in report.by_source.items():
                    source_data.append({
                        'Source': source,
                        'Signals': m.total_signals,
                        'Win Rate': f"{m.win_rate:.1f}%",
                        'Return': f"{m.total_return_pct:+.1f}%",
                        'Profit Factor': f"{m.profit_factor:.2f}"
                    })
                
                st.dataframe(pd.DataFrame(source_data), use_container_width=True, hide_index=True)
    
    with report_tab4:
        st.markdown("#### Export Signals")
        
        export_format = st.selectbox("Export Format", ["CSV", "JSON"], key="export_format")
        
        if st.button("Export", type="primary", key="export_btn"):
            if export_format == "CSV":
                csv_data = reporting_service.export_to_csv()
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name="signal_history.csv",
                    mime="text/csv"
                )
            else:
                json_data = reporting_service.export_to_json()
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name="signal_history.json",
                    mime="application/json"
                )
        
        # Summary stats
        stats = reporting_service.get_summary_stats()
        
        st.markdown("---")
        st.markdown("#### Quick Stats")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**All Time**")
            st.markdown(f"- Signals: {stats['all_time']['total_signals']}")
            st.markdown(f"- Win Rate: {stats['all_time']['win_rate']:.1f}%")
            st.markdown(f"- Return: {stats['all_time']['total_return']:+.1f}%")
        
        with col2:
            st.markdown("**Last 30 Days**")
            st.markdown(f"- Signals: {stats['last_30_days']['total_signals']}")
            st.markdown(f"- Win Rate: {stats['last_30_days']['win_rate']:.1f}%")
            st.markdown(f"- Return: {stats['last_30_days']['total_return']:+.1f}%")


# ============================================================================
# MAIN RENDER FUNCTION
# ============================================================================

def render_ta_signals() -> None:
    """
    Main render function for TA Signals Dashboard.
    
    This function can be called from stock_risk_app.py to integrate
    the TA Signals extension.
    """
    
    # Initialize services
    ta_service = TAService()
    signals_service = SignalsService()
    pattern_service = PatternService()
    mtf_service = MTFService()
    regime_service = RegimeService()
    divergence_service = DivergenceService()
    strategy_service = StrategyService()
    alerts_service = AlertsService()
    reporting_service = ReportingService()
    
    # Header
    st.markdown("## Technical Analysis Signals")
    st.markdown("Advanced technical indicators and signal generation")
    
    # Tabs - Extended with new features
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "📈 Single Stock",
        "📊 Dashboard",
        "🔍 Screener",
        "⏱️ Backtest",
        "💼 Portfolio",
        "🕯️ Patterns",
        "🌡️ Regime",
        "🔔 Alerts",
        "📋 Reports"
    ])
    
    with tab1:
        render_single_stock_analysis(ta_service, signals_service, pattern_service, 
                                     mtf_service, divergence_service)
    
    with tab2:
        render_signal_dashboard(ta_service, signals_service)
    
    with tab3:
        render_screener_mode(ta_service, signals_service)
    
    with tab4:
        render_backtest_module(ta_service, signals_service)
    
    with tab5:
        render_portfolio_signals(ta_service, signals_service)
    
    with tab6:
        render_pattern_analysis(pattern_service)
    
    with tab7:
        render_market_regime(regime_service)
    
    with tab8:
        render_alerts_manager(alerts_service)
    
    with tab9:
        render_performance_reports(reporting_service)


# ============================================================================
# STANDALONE ENTRY POINT
# ============================================================================

def main() -> None:
    """Main entry point for standalone execution."""
    
    # Page config
    st.set_page_config(
        page_title="TA Signals",
        page_icon="",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS (matching main app)
    st.markdown("""
    <style>
        .stApp {
            font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', Roboto, sans-serif;
            background-color: #0e1117 !important;
            color: #fafafa !important;
        }
        
        .main {
            background-color: #0e1117 !important;
        }
        
        h1, h2, h3 {
            font-weight: 500 !important;
            letter-spacing: -0.02em;
            color: #fafafa !important;
        }
        
        [data-testid="stMetricValue"] {
            font-size: 1.4rem !important;
            font-weight: 500 !important;
            color: #fafafa !important;
        }
        
        [data-testid="stMetricLabel"] {
            font-size: 0.75rem !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            opacity: 0.7;
            color: #a0a0a0 !important;
        }
        
        hr {
            border: none;
            height: 1px;
            background: rgba(255, 255, 255, 0.1);
            margin: 1.5rem 0;
        }
        
        .stButton > button {
            border-radius: 8px;
            font-weight: 500;
            letter-spacing: 0.02em;
        }
        
        .stTabs [data-baseweb="tab-list"] {
            gap: 2rem;
            background-color: transparent !important;
        }
        
        .stTabs [data-baseweb="tab"] {
            font-size: 0.9rem;
            font-weight: 500;
            color: #a0a0a0 !important;
        }
        
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            color: #fafafa !important;
        }
        
        [data-testid="stSidebar"] {
            background: #1a1d24 !important;
        }
        
        .stTextInput input, .stNumberInput input, .stSelectbox select {
            background-color: #262730 !important;
            color: #fafafa !important;
            border-color: rgba(255, 255, 255, 0.1) !important;
        }
        
        .block-container {
            padding-top: 2rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Render dashboard
    render_ta_signals()


if __name__ == "__main__":
    main()
