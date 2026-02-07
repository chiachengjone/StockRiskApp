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
from services.ta_sentiment_service import SentimentService, SentimentLevel

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

# Popular stocks for screener - organized by sector
POPULAR_STOCKS = {
    'Technology': [
        'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'AVGO', 'CSCO', 'ACN',
        'ADBE', 'CRM', 'INTC', 'AMD', 'ORCL', 'IBM', 'QCOM', 'TXN',
        'NOW', 'AMAT', 'MU', 'LRCX', 'KLAC', 'SNPS', 'CDNS', 'PANW'
    ],
    'Healthcare': [
        'UNH', 'JNJ', 'ABBV', 'MRK', 'LLY', 'TMO', 'ABT', 'DHR',
        'PFE', 'BMY', 'AMGN', 'GILD', 'VRTX', 'REGN', 'ISRG', 'MDT',
        'SYK', 'BSX', 'ZTS', 'CI', 'HUM', 'CVS', 'MCK', 'ELV'
    ],
    'Financial': [
        'BRK-B', 'JPM', 'V', 'MA', 'BAC', 'WFC', 'GS', 'MS',
        'AXP', 'C', 'SPGI', 'BLK', 'SCHW', 'CB', 'MMC', 'AON',
        'PGR', 'TRV', 'USB', 'PNC', 'CME', 'ICE', 'COF', 'AIG'
    ],
    'Consumer Cyclical': [
        'AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'LOW', 'SBUX', 'TJX',
        'BKNG', 'MAR', 'GM', 'F', 'ABNB', 'CMG', 'ORLY', 'AZO',
        'ROST', 'DHI', 'LEN', 'PHM', 'POOL', 'ULTA', 'LULU', 'YUM'
    ],
    'Consumer Defensive': [
        'PG', 'PEP', 'KO', 'COST', 'WMT', 'PM', 'MO', 'CL',
        'MDLZ', 'GIS', 'K', 'KHC', 'STZ', 'SYY', 'KR', 'TGT',
        'DG', 'DLTR', 'EL', 'HSY', 'TSN', 'HRL', 'CAG', 'MKC'
    ],
    'Energy': [
        'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'MPC', 'PSX', 'VLO',
        'OXY', 'PXD', 'DVN', 'HES', 'HAL', 'BKR', 'FANG', 'KMI',
        'WMB', 'OKE', 'TRGP', 'LNG', 'MRO', 'APA', 'EQT', 'CTRA'
    ],
    'Industrials': [
        'CAT', 'DE', 'RTX', 'HON', 'UNP', 'BA', 'LMT', 'GE',
        'UPS', 'FDX', 'MMM', 'CSX', 'NSC', 'ITW', 'EMR', 'ETN',
        'GD', 'NOC', 'WM', 'RSG', 'CMI', 'PH', 'ROK', 'IR'
    ],
    'Utilities': [
        'NEE', 'DUK', 'SO', 'D', 'AEP', 'SRE', 'EXC', 'XEL',
        'ED', 'PEG', 'WEC', 'ES', 'AWK', 'DTE', 'ETR', 'FE',
        'PPL', 'CMS', 'AEE', 'ATO', 'NI', 'LNT', 'EVRG', 'PNW'
    ],
    'Communication': [
        'VZ', 'T', 'TMUS', 'CMCSA', 'NFLX', 'DIS', 'CHTR', 'WBD',
        'EA', 'TTWO', 'ATVI', 'PARA', 'FOX', 'FOXA', 'NWS', 'NWSA',
        'LUMN', 'DISH', 'IPG', 'OMC', 'LYV', 'MTCH', 'ZG', 'Z'
    ],
    'Real Estate': [
        'AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'SPG', 'O', 'WELL',
        'DLR', 'AVB', 'EQR', 'VTR', 'ARE', 'MAA', 'UDR', 'ESS',
        'SUI', 'HST', 'PEAK', 'INVH', 'CPT', 'IRM', 'KIM', 'REG'
    ],
    'Materials': [
        'LIN', 'APD', 'SHW', 'ECL', 'FCX', 'NEM', 'NUE', 'DD',
        'DOW', 'PPG', 'VMC', 'MLM', 'ALB', 'CTVA', 'IFF', 'BALL',
        'PKG', 'IP', 'CE', 'EMN', 'FMC', 'MOS', 'CF', 'LYB'
    ]
}

# Flat list of all popular stocks
ALL_POPULAR_STOCKS = [ticker for tickers in POPULAR_STOCKS.values() for ticker in tickers]

# Sector mapping (reverse lookup)
SECTOR_MAPPING = {ticker: sector for sector, tickers in POPULAR_STOCKS.items() for ticker in tickers}

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
    """Render enhanced screener mode tab."""
    
    st.markdown("### 🔍 Stock Screener")
    st.markdown("Scan stocks for trading signals by ticker, sector, or from popular lists")
    
    # Stock source selection
    st.markdown("#### Select Stock Source")
    
    source_tab1, source_tab2, source_tab3 = st.tabs([
        "📝 Custom Tickers", 
        "🏢 By Sector", 
        "⭐ Popular Lists"
    ])
    
    selected_tickers = []
    
    with source_tab1:
        st.markdown("Enter any stock ticker(s) to scan:")
        custom_input = st.text_area(
            "Tickers (comma or space separated)",
            placeholder="AAPL, MSFT, GOOGL, AMZN, TSLA...",
            height=100,
            key="screener_custom_tickers"
        )
        if custom_input:
            # Parse tickers - handle both comma and space separated
            raw_tickers = custom_input.replace(',', ' ').split()
            selected_tickers = [t.strip().upper() for t in raw_tickers if t.strip()]
    
    with source_tab2:
        st.markdown("Select stocks by sector:")
        
        # Show sector checkboxes with stock counts
        col1, col2 = st.columns(2)
        selected_sectors = []
        
        sectors = list(POPULAR_STOCKS.keys())
        half = len(sectors) // 2
        
        with col1:
            for sector in sectors[:half]:
                count = len(POPULAR_STOCKS[sector])
                if st.checkbox(f"{sector} ({count} stocks)", key=f"sector_{sector}"):
                    selected_sectors.append(sector)
        
        with col2:
            for sector in sectors[half:]:
                count = len(POPULAR_STOCKS[sector])
                if st.checkbox(f"{sector} ({count} stocks)", key=f"sector_{sector}"):
                    selected_sectors.append(sector)
        
        # Add "Select All" option
        if st.checkbox("Select All Sectors", key="select_all_sectors"):
            selected_sectors = list(POPULAR_STOCKS.keys())
        
        # Build ticker list from selected sectors
        for sector in selected_sectors:
            selected_tickers.extend(POPULAR_STOCKS.get(sector, []))
    
    with source_tab3:
        st.markdown("Quick select from popular stock lists:")
        
        preset_options = {
            "Top 20 Overall": ALL_POPULAR_STOCKS[:20],
            "Top 50 Overall": ALL_POPULAR_STOCKS[:50],
            "Magnificent 7": ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA'],
            "FAANG+": ['META', 'AAPL', 'AMZN', 'NFLX', 'GOOGL', 'MSFT', 'NVDA'],
            "Top Tech": POPULAR_STOCKS.get('Technology', [])[:15],
            "Top Healthcare": POPULAR_STOCKS.get('Healthcare', [])[:15],
            "Top Financial": POPULAR_STOCKS.get('Financial', [])[:15],
            "Top Consumer": POPULAR_STOCKS.get('Consumer Cyclical', [])[:10] + POPULAR_STOCKS.get('Consumer Defensive', [])[:10],
            "Top Energy": POPULAR_STOCKS.get('Energy', [])[:15],
            "Top Industrials": POPULAR_STOCKS.get('Industrials', [])[:15],
            "Dividend Aristocrats": ['JNJ', 'PG', 'KO', 'PEP', 'MMM', 'MCD', 'ABT', 'XOM', 'CVX', 'WMT'],
            "Growth Stocks": ['NVDA', 'TSLA', 'AMD', 'CRM', 'NOW', 'ADBE', 'PANW', 'SNOW', 'NET', 'DDOG'],
            "All Stocks": ALL_POPULAR_STOCKS
        }
        
        selected_preset = st.selectbox(
            "Choose a preset list",
            options=list(preset_options.keys()),
            key="screener_preset"
        )
        
        if selected_preset:
            preset_tickers = preset_options[selected_preset]
            st.info(f"📊 {len(preset_tickers)} stocks in '{selected_preset}'")
            
            # Show preview
            with st.expander("Preview stocks in this list"):
                st.write(", ".join(preset_tickers[:50]))
                if len(preset_tickers) > 50:
                    st.write(f"... and {len(preset_tickers) - 50} more")
            
            if st.checkbox("Use this list", value=True, key="use_preset"):
                selected_tickers = preset_tickers
    
    # Remove duplicates
    selected_tickers = list(dict.fromkeys(selected_tickers))
    
    st.markdown("---")
    
    # Signal and filter options
    st.markdown("#### Filter Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        signal_filter = st.multiselect(
            "Signal Types",
            options=["BUY", "SELL", "STRONG_BUY", "STRONG_SELL", "HOLD"],
            default=["BUY", "STRONG_BUY"],
            key="screener_signal_filter"
        )
    
    with col2:
        min_score = st.slider("Minimum Score", 0, 100, 50, key="min_score")
    
    with col3:
        max_stocks = st.number_input(
            "Max Stocks to Scan",
            min_value=1,
            max_value=300,
            value=min(100, len(selected_tickers)) if selected_tickers else 50,
            key="max_stocks"
        )
    
    # Risk filters
    with st.expander("Risk Filters (Optional)"):
        col1, col2, col3 = st.columns(3)
        with col1:
            max_volatility = st.slider("Max Volatility %", 0, 100, 50, key="max_vol")
        with col2:
            max_beta = st.slider("Max Beta", 0.0, 3.0, 2.0, 0.1, key="max_beta")
        with col3:
            min_sharpe = st.slider("Min Sharpe Ratio", -2.0, 3.0, 0.0, 0.1, key="min_sharpe")
    
    # Show current selection count
    if selected_tickers:
        st.info(f"📋 Ready to scan {min(len(selected_tickers), int(max_stocks))} stocks")
    
    # Scan button
    scan_btn = st.button("🔍 Scan Stocks", type="primary", key="scan_btn", use_container_width=True)
    
    if scan_btn:
        if not selected_tickers:
            st.warning("Please select stocks to scan using one of the tabs above")
            return
        
        if not signal_filter:
            st.warning("Please select at least one signal type to filter")
            return
        
        # Limit tickers
        tickers_to_scan = selected_tickers[:int(max_stocks)]
        
        # Progress
        progress = st.progress(0)
        status = st.empty()
        
        results = []
        errors = 0
        
        for i, ticker in enumerate(tickers_to_scan):
            status.text(f"Scanning {ticker}... ({i+1}/{len(tickers_to_scan)})")
            progress.progress((i + 1) / len(tickers_to_scan))
            
            try:
                df = fetch_stock_data(ticker, "1y")
                
                if df.empty or len(df) < 50:
                    errors += 1
                    continue
                
                # Get current signal
                score, signal_type, reason = signals_service.get_current_signal(ticker, df)
                
                # Apply signal filter
                if signal_type.value not in signal_filter:
                    continue
                
                # Apply score filter
                if score < min_score:
                    continue
                
                # Calculate risk
                close = df['Close'] if not isinstance(df['Close'], pd.DataFrame) else df['Close'].iloc[:, 0]
                risk = calculate_risk_metrics(close)
                
                # Apply risk filters
                if risk.volatility * 100 > max_volatility:
                    continue
                if hasattr(risk, 'beta') and risk.beta > max_beta:
                    continue
                if risk.sharpe < min_sharpe:
                    continue
                
                # Get price info
                current_price = float(close.iloc[-1])
                prev_price = float(close.iloc[-2]) if len(close) > 1 else current_price
                change_pct = ((current_price - prev_price) / prev_price) * 100
                
                results.append({
                    'Symbol': ticker,
                    'Sector': SECTOR_MAPPING.get(ticker, 'Other'),
                    'Price': f"${current_price:.2f}",
                    'Change': f"{change_pct:+.2f}%",
                    'Signal': signal_type.value,
                    'Score': score,
                    'Reason': reason[:40] + "..." if len(reason) > 40 else reason,
                    'Volatility': f"{risk.volatility*100:.1f}%",
                    'Sharpe': f"{risk.sharpe:.2f}"
                })
                
            except Exception as e:
                logger.error(f"Error scanning {ticker}: {e}")
                errors += 1
                continue
        
        progress.empty()
        status.empty()
        
        if results:
            st.success(f"Found {len(results)} stocks matching criteria")
            if errors > 0:
                st.caption(f"({errors} stocks skipped due to insufficient data)")
            
            # Sort by score
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values('Score', ascending=False)
            
            # Signal color mapping
            def signal_color(val):
                colors = {
                    'STRONG_BUY': 'background-color: #00C853; color: white',
                    'BUY': 'background-color: #34C759; color: white',
                    'HOLD': 'background-color: #FF9500; color: white',
                    'SELL': 'background-color: #FF3B30; color: white',
                    'STRONG_SELL': 'background-color: #FF1744; color: white'
                }
                return colors.get(val, '')
            
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
                    ),
                    'Symbol': st.column_config.TextColumn('Symbol', width='small'),
                    'Sector': st.column_config.TextColumn('Sector', width='medium'),
                    'Signal': st.column_config.TextColumn('Signal', width='small')
                }
            )
            
            # Export option
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results (CSV)",
                data=csv,
                file_name="screener_results.csv",
                mime="text/csv"
            )
        else:
            st.info("No stocks found matching the criteria. Try adjusting your filters.")


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
    
    # Header
    st.markdown("## Technical Analysis Signals")
    st.markdown("Advanced technical indicators and signal generation")
    
    # Tabs - Core features
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📈 Single Stock",
        "📊 Dashboard",
        "🔍 Screener",
        "⏱️ Backtest",
        "💼 Portfolio",
        "🕯️ Patterns",
        "🌡️ Regime"
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
