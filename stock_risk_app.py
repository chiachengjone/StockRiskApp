"""
STOCK RISK MODELLING APP v4.2
==============================
Streamlit + yfinance + GARCH + EVT + Monte Carlo + Portfolio Mode + Stress Testing
+ Fama-French Factors + Kelly Criterion + ESG + XGBoost AI VaR
+ Options Analytics + Fundamentals + Comparison + Alerts + PDF Reports
+ Enhanced Analytics + Risk Parity + Black-Litterman + Real-time Features
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from risk_engine import (
    fetch_data, fetch_info, fetch_multiple, validate_ticker,
    compute_returns, compute_metrics, parametric_var, historical_var, cvar,
    fit_garch, evt_tail_risk, compute_beta, mc_simulation,
    portfolio_returns, portfolio_var, marginal_var_contribution,
    optimize_portfolio, efficient_frontier, stress_test_portfolio,
    backtest_strategy, STRESS_SCENARIOS,
    # New functions
    rolling_volatility, rolling_sharpe, rolling_var, rolling_beta, 
    rolling_max_drawdown, get_rolling_metrics_df, monte_carlo_stress,
    correlation_breakdown
)
from factors import FactorAnalyzer
from ml_predictor import MLPredictor

# Import new modules (graceful fallback)
try:
    from features import AlertManager, ReportGenerator, OptionsAnalytics, FundamentalAnalyzer, StockComparison
    from storage import PortfolioStore
    HAS_FEATURES = True
except ImportError:
    HAS_FEATURES = False

# Import new v4.2 features (sentiment, digital twin, what-if)
try:
    from features import (
        render_sentiment_tab, render_portfolio_sentiment, SentimentVaR,
        HAS_SENTIMENT_FEATURE
    )
except ImportError:
    HAS_SENTIMENT_FEATURE = False

try:
    from features import (
        render_digital_twin_tab, render_quick_scenario_widget,
        DigitalTwinEngine, HAS_DIGITAL_TWIN
    )
except ImportError:
    HAS_DIGITAL_TWIN = False

try:
    from features import (
        render_what_if_tab, render_scenario_builder,
        WhatIfAnalyzer, HAS_WHAT_IF
    )
except ImportError:
    HAS_WHAT_IF = False

# Import sentiment service
try:
    from services import SentimentService, HAS_SENTIMENT
except ImportError:
    HAS_SENTIMENT = False

try:
    from config.settings import COLORS as CONFIG_COLORS, POPULAR_STOCKS as CONFIG_STOCKS
except ImportError:
    CONFIG_COLORS = None
    CONFIG_STOCKS = None

# Import enhanced utilities (v4.1)
try:
    from utils import (
        # Performance
        ProgressTracker, performance_monitor,
        # Validation
        validate_ticker_robust, validate_returns_data, DataValidator,
        fetch_with_retry, handle_missing_data, detect_outliers,
        # Analytics
        backtest_var_kupiec, backtest_var_christoffersen,
        regime_detection, time_varying_beta, covar_systemic_risk,
        rolling_correlation_breakdown, enhanced_stress_test, AnalyticsEngine,
        # Portfolio
        risk_parity_weights, hierarchical_risk_parity,
        black_litterman_optimization, PortfolioOptimizer,
        threshold_rebalancing, calculate_rebalance_costs,
        tax_loss_harvesting_opportunities, portfolio_risk_decomposition,
        # New: Simulation & Correlation
        PortfolioSimulator, SimulationConfig, DynamicRebalancer, CorrelationMonitor,
        # Visualization
        interactive_correlation_heatmap, rolling_correlation_chart,
        volatility_surface_3d, var_cone_chart, var_comparison_chart,
        risk_contribution_chart, cumulative_returns_chart,
        performance_attribution_chart, rolling_performance_chart,
        factor_exposure_chart, regime_chart, VisualizationEngine,
        make_chart_downloadable, apply_dark_theme,
        # New: Enhanced Visualization
        enhanced_volatility_surface_3d, volatility_smile_cross_sections,
        term_structure_chart, efficient_frontier_chart,
        interactive_weight_slider_chart, weight_allocation_chart,
        # Realtime
        get_live_quote, is_market_open, calculate_live_pnl, RealtimeEngine,
        # New: WebSocket Realtime
        WebSocketPriceStream, RealTimePriceAggregator,
        create_realtime_dashboard, display_websocket_status
    )
    HAS_ENHANCED_UTILS = True
except ImportError as e:
    HAS_ENHANCED_UTILS = False
    print(f"Enhanced utils not available: {e}")

# Import TA Signals Extension
try:
    from ta_signals_app import render_ta_signals
    HAS_TA_SIGNALS = True
except ImportError as e:
    HAS_TA_SIGNALS = False
    print(f"TA Signals not available: {e}")

# Import v4.3 Professional Features
try:
    from data_sources import DataAggregator
    from factors import StyleFactorAnalyzer
    from utils.portfolio import calculate_liquidity_risk, estimate_execution_cost
    from utils.instrument_metadata import (
        get_metadata_handler, InstrumentMetadataHandler,
        ETFMetadata, OptionMetadata, AssetClass
    )
    HAS_PRO_FEATURES = True
except ImportError as e:
    HAS_PRO_FEATURES = False
    print(f"v4.3 Pro features not available: {e}")

# ============================================================================
# PAGE CONFIG & STYLING
# ============================================================================
st.set_page_config(
    page_title="Risk Model", 
    page_icon="", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean, minimal design with dark theme
st.markdown("""
<style>
    /* Dark theme for entire app */
    .stApp {
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', Roboto, sans-serif;
        background-color: #0e1117 !important;
        color: #fafafa !important;
    }
    
    /* Main content background */
    .main {
        background-color: #0e1117 !important;
    }
    
    /* Subtle headers */
    h1, h2, h3 {
        font-weight: 500 !important;
        letter-spacing: -0.02em;
        color: #fafafa !important;
    }
    
    /* Clean metric cards */
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
    
    /* Subtle dividers */
    hr {
        border: none;
        height: 1px;
        background: rgba(255, 255, 255, 0.1);
        margin: 1.5rem 0;
    }
    
    /* Clean buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        letter-spacing: 0.02em;
    }
    
    /* Subtle tabs */
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
    
    /* Clean dataframes */
    .stDataFrame {
        border-radius: 8px;
    }
    
    /* Dark sidebar styling */
    [data-testid="stSidebar"] {
        background: #1a1d24 !important;
    }
    
    [data-testid="stSidebar"] * {
        color: #fafafa !important;
    }
    
    /* Sidebar widgets */
    [data-testid="stSidebar"] .stMarkdown {
        color: #fafafa !important;
    }
    
    /* Expander in sidebar */
    [data-testid="stSidebar"] .streamlit-expanderHeader {
        background-color: rgba(255, 255, 255, 0.05) !important;
        color: #fafafa !important;
    }
    
    /* Input fields dark theme */
    .stTextInput input, .stNumberInput input, .stSelectbox select {
        background-color: #262730 !important;
        color: #fafafa !important;
        border-color: rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Slider */
    .stSlider {
        color: #fafafa !important;
    }
    
    /* Remove excessive padding */
    .block-container {
        padding-top: 2rem;
    }
    
    /* Caption text */
    .caption, .stCaption {
        color: #a0a0a0 !important;
    }
    
    /* Radio buttons */
    .stRadio label {
        color: #fafafa !important;
    }
    
    /* Checkbox/Toggle */
    .stCheckbox label, .stToggle label {
        color: #fafafa !important;
    }
</style>
""", unsafe_allow_html=True)

# Chart template
CHART_TEMPLATE = {
    'layout': {
        'font': {'family': '-apple-system, BlinkMacSystemFont, SF Pro Display, sans-serif', 'size': 12},
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'margin': {'t': 40, 'b': 40, 'l': 40, 'r': 20},
        'xaxis': {'gridcolor': 'rgba(128,128,128,0.1)', 'zerolinecolor': 'rgba(128,128,128,0.2)'},
        'yaxis': {'gridcolor': 'rgba(128,128,128,0.1)', 'zerolinecolor': 'rgba(128,128,128,0.2)'},
    }
}

# Minimal color palette
COLORS = {
    'primary': '#007AFF',
    'secondary': '#5856D6', 
    'success': '#34C759',
    'warning': '#FF9500',
    'danger': '#FF3B30',
    'gray': '#8E8E93',
    'light': '#F2F2F7',
    'dark': '#1C1C1E'
}

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("### Settings")
    
    with st.expander("Metrics Reference", expanded=False):
        st.markdown("""
        **VaR** — Maximum expected loss at confidence level
        
        **CVaR/ES** — Average loss beyond VaR threshold
        
        **Sharpe** — Risk-adjusted return
        
        **Sortino** — Downside risk-adjusted return
        
        **Calmar** — Return / Max Drawdown
        
        **Beta** — Sensitivity to benchmark
        
        **GARCH** — Volatility clustering model
        
        **EVT** — Extreme Value Theory for tails
        """)
    
    with st.expander("How to Use", expanded=False):
        st.markdown("""
        **Single Stock**
        1. Select asset and benchmark
        2. Configure timeframe
        3. Run analysis
        
        **Portfolio**
        1. Enter tickers (comma-separated)
        2. Set weights (sum to 100%)
        3. Run analysis
        """)
    
    st.divider()
    
    rf_rate = st.number_input("Risk-Free Rate (%)", 0.0, 10.0, 4.5, 0.1, key="sidebar_rf_rate") / 100
    theme_dark = st.toggle("Dark Charts", value=True, key="sidebar_theme_dark")
    auto_refresh = st.toggle("Auto-Refresh (5min)", value=False, key="sidebar_auto_refresh")
    
    # New: Alerts Section
    if HAS_FEATURES:
        st.divider()
        alert_manager = AlertManager()
        alert_summary = alert_manager.get_summary()
        st.markdown(f"**Alerts:** {alert_summary['active_alerts']} active")
        if alert_summary['triggered_today'] > 0:
            st.warning(f"{alert_summary['triggered_today']} alerts triggered today")
    
    # Market Status (v4.1)
    if HAS_ENHANCED_UTILS:
        st.divider()
        market_status = is_market_open()
        if market_status.get('is_open', False):
            st.success("Market Open")
        else:
            st.info("Market Closed")
    
    # Enable all features by default (no toggles)
    st.session_state.enable_sentiment = HAS_SENTIMENT_FEATURE and HAS_SENTIMENT
    st.session_state.enable_digital_twin = HAS_DIGITAL_TWIN
    st.session_state.enable_what_if = HAS_WHAT_IF
    st.session_state.enable_websocket = HAS_ENHANCED_UTILS
    
    st.divider()
    
    # App Mode Switcher (at bottom of sidebar)
    st.markdown("### App Mode")
    if HAS_TA_SIGNALS:
        app_mode = st.radio(
            "Select Mode",
            ["Risk Analysis", "TA Signals"],
            index=0,
            key="app_mode_selector",
            help="Switch between Risk Analysis and Technical Analysis Signals"
        )
    else:
        app_mode = "Risk Analysis"
        st.info("TA Signals extension not available")
    
    st.divider()
    st.caption("v4.3 Enterprise" + (" | TA" if HAS_TA_SIGNALS else ""))

# ============================================================================
# TA SIGNALS MODE
# ============================================================================
if app_mode == "TA Signals" and HAS_TA_SIGNALS:
    render_ta_signals()
    st.stop()  # Don't render the rest of the app

# ============================================================================
# SESSION STATE INITIALIZATION - State Controller
# ============================================================================
def init_session_state():
    """Initialize all session state variables with defaults."""
    defaults = {
        # Analysis flags
        'single_analyzed': False,
        'port_analyzed': False,
        
        # Single stock inputs (synced with widgets)
        'single_asset_type': 'Stocks',
        'single_stock_selection': 'Apple (AAPL)',
        'single_custom_ticker': 'AAPL',
        'single_bench_selection': 'S&P 500',
        'single_custom_benchmark': '^GSPC',
        'single_days_back': 756,
        'single_conf_level': 0.95,
        'single_var_horizon': 1,
        
        # Portfolio inputs
        'port_tickers_input': 'AAPL, MSFT, GOOGL, AMZN, NVDA',
        'port_days_back': 756,
        'port_conf_level': 0.95,
        
        # Feature flags
        'enable_sentiment': False,
        'enable_digital_twin': False,
        'enable_what_if': False,
        'enable_websocket': False,
        
        # Reactive parameters (outside form, updated live)
        'reactive_conf_level': 0.95,
        'reactive_var_horizon': 1,
        'reactive_port_conf_level': 0.95,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Initialize state on first run
init_session_state()

# Fixed tab list - NEVER changes to prevent index shifting
SINGLE_STOCK_TABS = [
    "Overview", "VaR Analysis", "Monte Carlo", "Stress Test", 
    "Sentiment", "Factors", "AI Risk", "Options", "Fundamentals", "Export"
]

PORTFOLIO_TABS = [
    "Summary", "Monte Carlo", "Correlation", "Stress Test", 
    "Optimization", "Digital Twin", "What-If", "Charts", "Export"
]

# ============================================================================
# ASSET DEFINITIONS
# ============================================================================
POPULAR_STOCKS = {
    "Apple (AAPL)": "AAPL", "Microsoft (MSFT)": "MSFT", "Google (GOOGL)": "GOOGL",
    "Amazon (AMZN)": "AMZN", "Tesla (TSLA)": "TSLA", "NVIDIA (NVDA)": "NVDA",
    "Meta (META)": "META", "Netflix (NFLX)": "NFLX", "JPMorgan (JPM)": "JPM",
    "Berkshire (BRK-B)": "BRK-B", "Johnson & Johnson (JNJ)": "JNJ", "Visa (V)": "V",
    "Walmart (WMT)": "WMT", "Mastercard (MA)": "MA", "Disney (DIS)": "DIS",
    "AMD (AMD)": "AMD", "Intel (INTC)": "INTC", "Coca-Cola (KO)": "KO",
    "PepsiCo (PEP)": "PEP", "Pfizer (PFE)": "PFE", "Custom...": "CUSTOM"
}

POPULAR_BENCHMARKS = {
    "S&P 500": "^GSPC", "Dow Jones": "^DJI", "NASDAQ 100": "^NDX",
    "NASDAQ Composite": "^IXIC", "Russell 2000": "^RUT", "VIX": "^VIX",
    "FTSE 100": "^FTSE", "DAX": "^GDAXI", "Nikkei 225": "^N225",
    "Hang Seng": "^HSI", "Custom...": "CUSTOM"
}

CRYPTO_ASSETS = {
    "Bitcoin": "BTC-USD", "Ethereum": "ETH-USD", "Solana": "SOL-USD",
    "Cardano": "ADA-USD", "XRP": "XRP-USD"
}

ETF_ASSETS = {
    "SPY (S&P 500)": "SPY", "QQQ (NASDAQ)": "QQQ", "IWM (Russell 2000)": "IWM",
    "VTI (Total Market)": "VTI", "GLD (Gold)": "GLD", "TLT (20Y Treasury)": "TLT",
    "EEM (Emerging Markets)": "EEM"
}

# ============================================================================
# CACHED COMPUTATION HELPERS - Performance Optimization
# ============================================================================
@st.cache_data(ttl=300, show_spinner=False)
def cached_fetch_data(ticker: str, start_date: str, end_date: str):
    """Cache data fetching to avoid repeated API calls."""
    return fetch_data(ticker, start_date, end_date)

@st.cache_data(ttl=300, show_spinner=False)
def cached_fetch_multiple(tickers: tuple, start_date: str, end_date: str):
    """Cache multiple ticker data fetching."""
    return fetch_multiple(list(tickers), start_date, end_date)

@st.cache_data(ttl=600, show_spinner=False)
def cached_fetch_info(ticker: str):
    """Cache ticker info fetching."""
    return fetch_info(ticker)

@st.cache_data(ttl=60, show_spinner=False)
def cached_compute_metrics(rets_tuple: tuple, rf_rate: float):
    """Cache metrics computation. Pass returns as tuple for hashability."""
    import pandas as pd
    rets = pd.Series(rets_tuple)
    return compute_metrics(rets, rf_rate)

@st.cache_data(ttl=60, show_spinner=False)
def cached_fit_garch(rets_tuple: tuple):
    """Cache GARCH model fitting."""
    import pandas as pd
    rets = pd.Series(rets_tuple)
    return fit_garch(rets)

@st.cache_data(ttl=60, show_spinner=False)
def cached_mc_simulation(rets_tuple: tuple, n_sims: int, horizon: int):
    """Cache Monte Carlo simulation results."""
    import pandas as pd
    rets = pd.Series(rets_tuple)
    return mc_simulation(rets, n_sims, horizon)

@st.cache_data(ttl=300, show_spinner=False)
def cached_regime_detection(rets_tuple: tuple, n_regimes: int = 3):
    """Cache regime detection results."""
    import pandas as pd
    rets = pd.Series(rets_tuple)
    if HAS_ENHANCED_UTILS:
        return regime_detection(rets, n_regimes=n_regimes)
    return {'error': 'Enhanced utils not available'}

@st.cache_data(ttl=60, show_spinner=False)
def cached_optimize_portfolio(returns_df_json: str):
    """Cache portfolio optimization. Pass returns as JSON for hashability."""
    import pandas as pd
    returns_df = pd.read_json(returns_df_json)
    return optimize_portfolio(returns_df)

@st.cache_data(ttl=60, show_spinner=False)
def cached_efficient_frontier(returns_df_json: str):
    """Cache efficient frontier computation."""
    import pandas as pd
    returns_df = pd.read_json(returns_df_json)
    return efficient_frontier(returns_df)

# ============================================================================
# HELPER: Create Monte Carlo Chart
# ============================================================================
def create_mc_distribution_chart(mc_rets, var_95, cvar_95, theme_dark=True):
    """Create a clean Monte Carlo distribution chart."""
    fig = go.Figure()
    
    # Histogram
    fig.add_trace(go.Histogram(
        x=mc_rets * 100,
        nbinsx=80,
        marker_color=COLORS['primary'],
        opacity=0.7,
        name='Simulated Returns'
    ))
    
    # VaR line
    fig.add_vline(x=var_95 * 100, line_dash="solid", line_color=COLORS['warning'], line_width=2,
                  annotation_text=f"VaR 95%: {var_95:.2%}", annotation_position="top left")
    
    # CVaR line
    fig.add_vline(x=cvar_95 * 100, line_dash="dash", line_color=COLORS['danger'], line_width=2,
                  annotation_text=f"CVaR: {cvar_95:.2%}", annotation_position="top left")
    
    fig.update_layout(
        title="Monte Carlo Return Distribution",
        xaxis_title="Return (%)",
        yaxis_title="Frequency",
        template='plotly_dark' if theme_dark else 'plotly_white',
        showlegend=False,
        height=350
    )
    return fig


def create_mc_paths_chart(returns, n_paths=100, horizon=20, theme_dark=True):
    """Create Monte Carlo simulation paths chart."""
    mu = float(returns.mean())
    sigma = float(returns.std())
    
    fig = go.Figure()
    
    # Generate paths
    np.random.seed(42)
    for i in range(n_paths):
        path_rets = np.random.normal(mu, sigma, horizon)
        path = np.cumprod(1 + path_rets) * 100  # Start at 100
        color = COLORS['primary'] if i < n_paths - 10 else COLORS['danger']
        opacity = 0.15 if i < n_paths - 10 else 0.6
        fig.add_trace(go.Scatter(
            x=list(range(horizon)),
            y=path,
            mode='lines',
            line=dict(color=color, width=1),
            opacity=opacity,
            showlegend=False
        ))
    
    # Add median path
    all_paths = []
    for _ in range(1000):
        path_rets = np.random.normal(mu, sigma, horizon)
        path = np.cumprod(1 + path_rets) * 100
        all_paths.append(path)
    
    median_path = np.median(all_paths, axis=0)
    p5_path = np.percentile(all_paths, 5, axis=0)
    p95_path = np.percentile(all_paths, 95, axis=0)
    
    fig.add_trace(go.Scatter(
        x=list(range(horizon)), y=median_path,
        mode='lines', line=dict(color=COLORS['success'], width=2),
        name='Median'
    ))
    
    fig.update_layout(
        title=f"Monte Carlo Simulation Paths ({horizon} Days)",
        xaxis_title="Days",
        yaxis_title="Portfolio Value (Start = 100)",
        template='plotly_dark' if theme_dark else 'plotly_white',
        height=350,
        showlegend=True
    )
    return fig, p5_path[-1], p95_path[-1], median_path[-1]


def create_mc_cone_chart(returns, horizon=60, theme_dark=True):
    """Create Monte Carlo confidence cone chart."""
    mu = float(returns.mean())
    sigma = float(returns.std())
    
    # Simulate many paths
    np.random.seed(42)
    n_sims = 1000
    all_paths = np.zeros((n_sims, horizon))
    
    for i in range(n_sims):
        path_rets = np.random.normal(mu, sigma, horizon)
        all_paths[i] = np.cumprod(1 + path_rets) * 100
    
    # Calculate percentiles
    p5 = np.percentile(all_paths, 5, axis=0)
    p25 = np.percentile(all_paths, 25, axis=0)
    p50 = np.percentile(all_paths, 50, axis=0)
    p75 = np.percentile(all_paths, 75, axis=0)
    p95 = np.percentile(all_paths, 95, axis=0)
    
    days = list(range(horizon))
    
    fig = go.Figure()
    
    # 90% confidence band
    fig.add_trace(go.Scatter(
        x=days + days[::-1],
        y=list(p95) + list(p5[::-1]),
        fill='toself',
        fillcolor='rgba(0, 122, 255, 0.1)',
        line=dict(color='rgba(0,0,0,0)'),
        name='90% Confidence'
    ))
    
    # 50% confidence band
    fig.add_trace(go.Scatter(
        x=days + days[::-1],
        y=list(p75) + list(p25[::-1]),
        fill='toself',
        fillcolor='rgba(0, 122, 255, 0.2)',
        line=dict(color='rgba(0,0,0,0)'),
        name='50% Confidence'
    ))
    
    # Median line
    fig.add_trace(go.Scatter(
        x=days, y=p50,
        mode='lines',
        line=dict(color=COLORS['primary'], width=2),
        name='Median Path'
    ))
    
    fig.update_layout(
        title="Monte Carlo Confidence Cone",
        xaxis_title="Days Forward",
        yaxis_title="Portfolio Value (Start = 100)",
        template='plotly_dark' if theme_dark else 'plotly_white',
        height=350
    )
    
    return fig, p5[-1], p50[-1], p95[-1]

# ============================================================================
# MAIN UI
# ============================================================================
st.title("Stock Risk Model")
st.caption("Portfolio Analysis | Stress Testing | Factor Models | AI Risk")

mode = st.radio("Analysis Mode", ["Single Stock", "Portfolio"], horizontal=True, label_visibility="collapsed", key="main_analysis_mode")

# ============================================================================
# SINGLE STOCK MODE
# ============================================================================
if mode == "Single Stock":
    # Use form ONLY for data fetching inputs (ticker, benchmark, days)
    with st.form(key="single_stock_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            asset_type = st.selectbox(
                "Asset Type", 
                ["Stocks", "ETFs", "Crypto", "Custom"],
                key="single_asset_type_widget"
            )
            if asset_type == "Stocks":
                stock_selection = st.selectbox(
                    "Select Stock", 
                    options=list(POPULAR_STOCKS.keys()),
                    key="single_stock_selection_widget"
                )
                if stock_selection == "Custom...":
                    ticker = st.text_input("Custom Ticker", "AAPL", key="single_custom_ticker_widget").upper()
                else:
                    ticker = POPULAR_STOCKS[stock_selection]
            elif asset_type == "ETFs":
                etf_selection = st.selectbox(
                    "Select ETF", 
                    options=list(ETF_ASSETS.keys()),
                    key="single_etf_selection_widget"
                )
                ticker = ETF_ASSETS[etf_selection]
            elif asset_type == "Crypto":
                crypto_selection = st.selectbox(
                    "Select Crypto", 
                    options=list(CRYPTO_ASSETS.keys()),
                    key="single_crypto_selection_widget"
                )
                ticker = CRYPTO_ASSETS[crypto_selection]
            else:
                ticker = st.text_input("Enter Ticker", "AAPL", key="single_enter_ticker_widget").upper()
        
        with col2:
            bench_selection = st.selectbox(
                "Benchmark", 
                options=list(POPULAR_BENCHMARKS.keys()),
                key="single_bench_selection_widget"
            )
            if bench_selection == "Custom...":
                benchmark = st.text_input("Custom Benchmark", "^GSPC", key="single_custom_bench_widget").upper()
            else:
                benchmark = POPULAR_BENCHMARKS[bench_selection]
        
        with col3:
            days_back = st.slider("Days Back", 100, 2000, 756, key="single_days_back_widget")
        
        # Form submit button
        analyze_submitted = st.form_submit_button("Analyze Risk", type="primary", use_container_width=True)
    
    # REACTIVE SLIDERS - Outside form so they update VaR live after analysis
    if st.session_state.single_analyzed:
        st.markdown("#### Risk Parameters (Reactive)")
        col4, col5 = st.columns(2)
        with col4:
            conf_level = st.slider(
                "Confidence Level", 0.90, 0.999, 
                st.session_state.reactive_conf_level, 0.01, 
                key="reactive_conf_level_widget"
            )
            st.session_state.reactive_conf_level = conf_level
        with col5:
            var_horizon = st.slider(
                "VaR Horizon (days)", 1, 30, 
                st.session_state.reactive_var_horizon, 
                key="reactive_var_horizon_widget"
            )
            st.session_state.reactive_var_horizon = var_horizon
    else:
        # Default values before analysis
        conf_level = 0.95
        var_horizon = 1
    
    # Always enable advanced analysis
    advanced_mode = True

    # Process form submission - ONLY fetches data, no rendering
    if analyze_submitted:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        with st.spinner(f"Fetching {ticker} data..."):
            data = fetch_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            bench_data = fetch_data(benchmark, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            info = fetch_info(ticker)
        
        if data.empty:
            st.error(f"No data for {ticker}. Check ticker symbol.")
            st.stop()
        
        if bench_data.empty:
            st.warning(f"Benchmark {benchmark} not found. Using stock data only.")
            bench_data = data.copy()
        
        prices = data['Close'].iloc[:, 0] if isinstance(data['Close'], pd.DataFrame) else data['Close']
        rets = compute_returns(prices)
        
        bench_prices = bench_data['Close'].iloc[:, 0] if isinstance(bench_data['Close'], pd.DataFrame) else bench_data['Close']
        bench_rets = compute_returns(bench_prices)
        
        common_idx = rets.index.intersection(bench_rets.index)
        rets = rets.loc[common_idx]
        bench_rets = bench_rets.loc[common_idx]
        
        metrics = compute_metrics(rets, rf_rate)
        beta, alpha = compute_beta(rets, bench_rets)
        
        st.session_state.single_analyzed = True
        st.session_state.single_ticker = ticker
        st.session_state.single_benchmark = benchmark
        st.session_state.single_rets = rets
        st.session_state.single_bench_rets = bench_rets
        st.session_state.single_prices = prices
        st.session_state.single_metrics = metrics
        st.session_state.single_beta = beta
        st.session_state.single_alpha = alpha
        st.session_state.single_info = info
        st.session_state.single_advanced = advanced_mode
        st.session_state.single_rf_rate = rf_rate
        st.session_state.single_theme_dark = theme_dark
        st.session_state.single_days_back = days_back
        # Initialize reactive params on first analysis
        st.session_state.reactive_conf_level = 0.95
        st.session_state.reactive_var_horizon = 1
    
    # DASHBOARD RENDERING - Only uses session state data
    if st.session_state.single_analyzed:
        # Load all data from session state (immutable after analysis)
        ticker = st.session_state.single_ticker
        benchmark = st.session_state.single_benchmark
        rets = st.session_state.single_rets
        bench_rets = st.session_state.single_bench_rets
        prices = st.session_state.single_prices
        metrics = st.session_state.single_metrics
        beta = st.session_state.single_beta
        alpha = st.session_state.single_alpha
        info = st.session_state.single_info
        advanced_mode = st.session_state.single_advanced
        rf_rate = st.session_state.single_rf_rate
        theme_dark = st.session_state.single_theme_dark
        days_back = st.session_state.single_days_back
        
        # Use REACTIVE values from sliders (updated on every interaction)
        conf_level = st.session_state.reactive_conf_level
        var_horizon = st.session_state.reactive_var_horizon
        
        # Calculate VaR with current reactive parameters
        p_var = parametric_var(rets, var_horizon, conf_level)
        p_var_t = parametric_var(rets, var_horizon, conf_level, 't')
        h_var = historical_var(rets, var_horizon, conf_level)
        cv = cvar(rets, conf_level)
        
        # Use FIXED tab list to prevent index shifting
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs(SINGLE_STOCK_TABS)
        
        # TAB 1: OVERVIEW (includes real-time data and market regime)
        with tab1:
            st.subheader(f"{ticker} Risk Summary")
            
            long_term_var = parametric_var(rets, 1, 0.95)
            recent_var = parametric_var(rets.tail(60), 1, 0.95)
            if abs(recent_var) > abs(long_term_var) * 1.5:
                st.warning(f"VaR Alert: Recent VaR ({recent_var:.2%}) is 50%+ higher than historical ({long_term_var:.2%})")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Ann. Return", f"{metrics['ann_ret']:.1%}")
            col2.metric("Ann. Volatility", f"{metrics['ann_vol']:.1%}")
            col3.metric("Max Drawdown", f"{metrics['max_dd']:.1%}")
            col4.metric(f"Beta vs {benchmark}", f"{beta:.2f}" if not np.isnan(beta) else "N/A")
            
            col5, col6, col7, col8 = st.columns(4)
            col5.metric("Sharpe Ratio", f"{metrics['sharpe']:.2f}")
            col6.metric("Sortino Ratio", f"{metrics['sortino']:.2f}")
            col7.metric("Calmar Ratio", f"{metrics['calmar']:.2f}")
            col8.metric("Skewness", f"{metrics['skew']:.2f}")
            
            if info:
                st.markdown("---")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Sector", info.get("sector", "N/A"))
                col2.metric("Market Cap", f"${info.get('marketCap', 0)/1e9:.1f}B" if info.get('marketCap') else "N/A")
                col3.metric("P/E Ratio", f"{info.get('trailingPE', 'N/A'):.1f}" if isinstance(info.get('trailingPE'), (int, float)) else "N/A")
                col4.metric("Div Yield", f"{info.get('dividendYield', 0)*100:.2f}%" if info.get('dividendYield') else "N/A")
            
            # Price chart
            st.markdown("---")
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=prices.index, y=prices.values, name='Price', 
                                      line=dict(color=COLORS['primary'], width=1.5),
                                      fill='tozeroy', fillcolor='rgba(0, 122, 255, 0.1)'))
            fig1.update_layout(title=f"{ticker} Price History", height=300, 
                              template='plotly_dark' if theme_dark else 'plotly_white',
                              margin=dict(t=40, b=40, l=40, r=20))
            st.plotly_chart(fig1, use_container_width=True)
            
            # Rolling Metrics Section (NEW)
            st.markdown("---")
            st.markdown("#### Rolling Risk Metrics")
            col1, col2 = st.columns(2)
            
            with col1:
                rolling_vol = rolling_volatility(rets, 21)
                fig_rvol = go.Figure()
                fig_rvol.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol.values * 100,
                                              name='21-Day Rolling Vol', 
                                              line=dict(color=COLORS['primary'], width=1.5)))
                # Add long-term average
                avg_vol = metrics['ann_vol'] * 100
                fig_rvol.add_hline(y=avg_vol, line_dash="dash", line_color=COLORS['warning'],
                                   annotation_text=f"Avg: {avg_vol:.1f}%")
                fig_rvol.update_layout(title="Rolling Volatility (Annualized)", height=250,
                                       template='plotly_dark' if theme_dark else 'plotly_white',
                                       yaxis_title="Volatility (%)")
                st.plotly_chart(fig_rvol, use_container_width=True)
            
            with col2:
                rolling_sh = rolling_sharpe(rets, 63, rf_rate)
                fig_rsh = go.Figure()
                fig_rsh.add_trace(go.Scatter(x=rolling_sh.index, y=rolling_sh.values,
                                             name='63-Day Rolling Sharpe',
                                             line=dict(color=COLORS['success'], width=1.5)))
                fig_rsh.add_hline(y=0, line_dash="dash", line_color=COLORS['gray'])
                fig_rsh.update_layout(title="Rolling Sharpe Ratio (Quarterly)", height=250,
                                      template='plotly_dark' if theme_dark else 'plotly_white',
                                      yaxis_title="Sharpe Ratio")
                st.plotly_chart(fig_rsh, use_container_width=True)
            
            # Real-time Data & Market Regime Section
            st.markdown("---")
            st.markdown("### Real-time Data & Market Regime")
            
            rt_col1, rt_col2 = st.columns([2, 3])
            
            with rt_col1:
                # Live quote
                market_status = is_market_open()
                if market_status.get('is_open', False):
                    st.success("Market OPEN")
                else:
                    st.info("Market CLOSED")
                
                live = get_live_quote(ticker)
                if live and live.get('price'):
                    st.markdown(f"**{ticker} Live Quote**")
                    lq_col1, lq_col2 = st.columns(2)
                    lq_col1.metric("Price", f"${live['price']:.2f}")
                    change_val = live.get('change', 0) or 0
                    change_pct = live.get('change_pct', 0) or 0
                    lq_col2.metric("Change", f"${change_val:.2f}", delta=f"{change_pct:.2f}%")
                    
                    if live.get('volume'):
                        st.metric("Volume", f"{live['volume']:,.0f}")
                    st.caption(f"Updated: {live.get('last_updated', 'Unknown')}")
                else:
                    st.caption("Live quote not available")
            
            with rt_col2:
                # Regime Detection - Summary only
                st.markdown("**Market Regime Detection**")
                regime_result = regime_detection(rets, n_regimes=3)
                
                if 'error' not in regime_result:
                    reg_col1, reg_col2 = st.columns(2)
                    with reg_col1:
                        current_label = regime_result['current_regime']
                        st.metric("Current Regime", current_label)
                    with reg_col2:
                        if 'Bull' in str(current_label):
                            st.success("Bullish Environment")
                        elif 'Bear' in str(current_label):
                            st.error("Bearish Environment")
                        else:
                            st.warning("Neutral/Sideways")
                else:
                    st.info("Insufficient data for regime detection")
            
            # Regime Chart - Full Width Section (inside tab1)
            if 'error' not in regime_result:
                st.markdown("---")
                st.markdown("### Price Chart with Regime Overlay")
                regime_series = regime_result['regime_series']
                fig_regime = regime_chart(prices, regime_series, title=f"{ticker} Price with Market Regime Detection")
                fig_regime.update_layout(height=500)
                st.plotly_chart(fig_regime, use_container_width=True)
        
        # TAB 2: VAR ANALYSIS
        with tab2:
            st.subheader("Value at Risk Analysis")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Normal VaR", f"{p_var:.2%}", help="Assumes normal distribution")
            col2.metric("t-Distribution VaR", f"{p_var_t:.2%}", help="Fat-tailed t-distribution")
            col3.metric("Historical VaR", f"{h_var:.2%}", help="Based on actual returns")
            col4.metric("CVaR (Expected Shortfall)", f"{cv:.2%}", help="Average loss beyond VaR")
            
            st.caption(f"Expected losses at {conf_level:.0%} confidence over {var_horizon} day(s)")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                var_df = pd.DataFrame({
                    'Method': ['Normal', 't-Distribution', 'Historical', 'CVaR'],
                    'VaR (%)': [abs(p_var)*100, abs(p_var_t)*100, abs(h_var)*100, abs(cv)*100]
                })
                fig_var = px.bar(var_df, x='Method', y='VaR (%)', title="VaR Comparison by Method",
                                color_discrete_sequence=[COLORS['primary']])
                fig_var.update_layout(template='plotly_dark' if theme_dark else 'plotly_white', 
                                     showlegend=False, height=350)
                st.plotly_chart(fig_var, use_container_width=True)
            
            with col2:
                fig2 = px.histogram(x=rets.values, nbins=60, title="Return Distribution with VaR Threshold",
                                   color_discrete_sequence=[COLORS['gray']])
                fig2.add_vline(x=float(np.percentile(rets, 5)), line_dash="solid", 
                              line_color=COLORS['danger'], line_width=2,
                              annotation_text="5% VaR")
                fig2.add_vline(x=float(np.percentile(rets, 1)), line_dash="dash", 
                              line_color=COLORS['warning'], line_width=2,
                              annotation_text="1% VaR")
                fig2.update_layout(template='plotly_dark' if theme_dark else 'plotly_white', height=350)
                st.plotly_chart(fig2, use_container_width=True)
            
            # Advanced Risk Models Section
            st.markdown("---")
            st.markdown("### Advanced Risk Models")
            
            adv_col1, adv_col2 = st.columns(2)
            
            with adv_col1:
                st.markdown("#### GARCH(1,1) Volatility Model")
                garch_fitted, garch_cond_vol, garch_forecast = fit_garch(rets)
                
                if garch_fitted is not None:
                    garch_met1, garch_met2, garch_met3 = st.columns(3)
                    current_garch_vol = float(garch_cond_vol.iloc[-1]) * np.sqrt(252)
                    garch_met1.metric("Current GARCH Vol", f"{current_garch_vol:.1%}")
                    garch_met2.metric("10-Day Forecast", f"{garch_forecast[-1]:.1%}")
                    vol_regime = "High" if current_garch_vol > metrics['ann_vol'] * 1.2 else "Normal"
                    garch_met3.metric("Vol Regime", vol_regime)
                    
                    # GARCH volatility chart
                    fig_garch = go.Figure()
                    fig_garch.add_trace(go.Scatter(
                        x=garch_cond_vol.index, 
                        y=garch_cond_vol.values * np.sqrt(252) * 100,
                        name='GARCH Volatility',
                        line=dict(color=COLORS['primary'], width=1.5),
                        fill='tozeroy',
                        fillcolor='rgba(0, 150, 136, 0.2)'
                    ))
                    fig_garch.update_layout(
                        title="GARCH Conditional Volatility (Annualized)",
                        yaxis_title="Volatility (%)",
                        template='plotly_dark' if theme_dark else 'plotly_white',
                        height=280
                    )
                    st.plotly_chart(fig_garch, use_container_width=True)
                else:
                    st.info("Insufficient data for GARCH model")
            
            with adv_col2:
                st.markdown("#### EVT Tail Risk (Extreme Value Theory)")
                evt_results = evt_tail_risk(rets)
                if "error" not in evt_results:
                    evt_met1, evt_met2, evt_met3 = st.columns(3)
                    evt_met1.metric("Tail Threshold", f"{evt_results['threshold']:.2%}")
                    evt_met2.metric("GPD Shape", f"{evt_results['shape']:.3f}", help="Shape > 0 indicates heavy tails")
                    evt_met3.metric("EVT 99% VaR", f"{evt_results['evt_var']:.2%}")
                    
                    # EVT visualization - tail distribution
                    tail_rets = rets[rets < evt_results['threshold']]
                    fig_evt = go.Figure()
                    fig_evt.add_trace(go.Histogram(
                        x=tail_rets.values * 100,
                        nbinsx=30,
                        name='Tail Returns',
                        marker_color=COLORS['danger']
                    ))
                    fig_evt.add_vline(x=evt_results['evt_var'] * 100, line_dash="dash", 
                                     line_color=COLORS['warning'], line_width=2,
                                     annotation_text="EVT VaR")
                    fig_evt.update_layout(
                        title="Extreme Tail Distribution",
                        xaxis_title="Return (%)",
                        yaxis_title="Frequency",
                        template='plotly_dark' if theme_dark else 'plotly_white',
                        height=280
                    )
                    st.plotly_chart(fig_evt, use_container_width=True)
                else:
                    st.info(evt_results["error"])
            
            # VaR Backtesting Section
            if HAS_ENHANCED_UTILS:
                st.markdown("---")
                st.markdown("### VaR Model Backtesting")
                st.caption("Test how well VaR models predicted actual losses using Kupiec test")
                
                bt_col1, bt_col2 = st.columns(2)
                with bt_col1:
                    bt_confidence = st.selectbox("Confidence Level", [0.95, 0.99], index=0, key="var_bt_conf")
                with bt_col2:
                    bt_window = st.slider("Lookback Window", 60, 252, 126, key="var_bt_window")
                
                # Auto-run VaR backtest
                with st.spinner("Running VaR backtest..."):
                    # Calculate VaR series
                    var_series = rets.rolling(bt_window).apply(
                        lambda x: np.percentile(x, (1 - bt_confidence) * 100)
                    )
                    
                    # Kupiec test
                    kupiec = backtest_var_kupiec(rets, var_series, bt_confidence)
                    
                    bt_res1, bt_res2, bt_res3 = st.columns(3)
                    bt_res1.metric("Violations", kupiec['violations'])
                    bt_res2.metric("Expected Rate", f"{kupiec['expected_rate']*100:.1f}%")
                    bt_res3.metric("P-Value", f"{kupiec['p_value']:.4f}")
                    
                    if kupiec['p_value'] >= 0.05:
                        st.success(f"VaR model PASSED Kupiec test at {bt_confidence:.0%} confidence")
                    else:
                        st.error(f"VaR model FAILED Kupiec test - model may be miscalibrated")
                    
                    # Show violations chart
                    fig_bt = go.Figure()
                    fig_bt.add_trace(go.Scatter(
                        x=rets.index, y=rets.values * 100,
                        mode='lines', name='Returns', line=dict(color='#2196F3', width=1)
                    ))
                    fig_bt.add_trace(go.Scatter(
                        x=var_series.index, y=var_series.values * 100,
                        mode='lines', name=f'VaR {bt_confidence:.0%}', 
                        line=dict(color='#FF5722', width=2, dash='dash')
                    ))
                    fig_bt.update_layout(
                        title="Returns vs VaR Threshold",
                        xaxis_title="Date", yaxis_title="Return (%)",
                        template='plotly_dark' if theme_dark else 'plotly_white',
                        height=350
                    )
                    st.plotly_chart(fig_bt, use_container_width=True)
            
            # Strategy Backtest Section
            st.markdown("---")
            st.markdown("### Strategy Backtest: Buy & Hold vs Benchmark")
            
            bt_results = backtest_strategy(rets, bench_rets)
            
            strat_col1, strat_col2, strat_col3 = st.columns(3)
            strat_col1.metric(f"{ticker} Total Return", f"{bt_results['strategy_return']:.1%}")
            strat_col2.metric(f"{benchmark} Return", f"{bt_results['benchmark_return']:.1%}")
            strat_col3.metric("Excess Return", f"{bt_results['excess_return']:.1%}",
                             delta=f"{'Outperforming' if bt_results['excess_return'] > 0 else 'Underperforming'}")
            
            # Cumulative returns chart
            fig_strat = go.Figure()
            fig_strat.add_trace(go.Scatter(
                x=bt_results['strategy_cum'].index, y=bt_results['strategy_cum'].values,
                name=ticker, line=dict(color=COLORS['primary'], width=2),
                fill='tozeroy', fillcolor='rgba(0, 150, 136, 0.1)'
            ))
            fig_strat.add_trace(go.Scatter(
                x=bt_results['benchmark_cum'].index, y=bt_results['benchmark_cum'].values,
                name=benchmark, line=dict(color=COLORS['gray'], width=2, dash='dash')
            ))
            fig_strat.update_layout(
                title="Cumulative Returns Comparison", 
                xaxis_title="Date", yaxis_title="Cumulative Return",
                template='plotly_dark' if theme_dark else 'plotly_white', 
                height=350
            )
            st.plotly_chart(fig_strat, use_container_width=True)
        
        # TAB 3: MONTE CARLO
        with tab3:
            st.subheader("Monte Carlo Simulation")
            
            col1, col2 = st.columns([1, 3])
            with col1:
                mc_sims = st.selectbox("Simulations", [1000, 5000, 10000, 50000], index=2, key="single_mc_sims")
                mc_horizon = st.selectbox("Horizon (days)", [5, 10, 20, 60, 120], index=2, key="single_mc_horizon")
            
            with st.spinner("Running Monte Carlo simulation..."):
                mc_rets = mc_simulation(rets, mc_sims, mc_horizon)
                mc_var = np.percentile(mc_rets, 5)
                mc_cvar = mc_rets[mc_rets <= np.percentile(mc_rets, 5)].mean()
                mc_mean = mc_rets.mean()
                mc_median = np.median(mc_rets)
                mc_p95 = np.percentile(mc_rets, 95)
            
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("MC 95% VaR", f"{mc_var:.2%}")
            col2.metric("MC CVaR", f"{mc_cvar:.2%}")
            col3.metric("Expected Return", f"{mc_mean:.2%}")
            col4.metric("Median Return", f"{mc_median:.2%}")
            col5.metric("95th Percentile", f"{mc_p95:.2%}")
            
            st.markdown("---")
            
            # Monte Carlo Distribution
            fig_mc_dist = create_mc_distribution_chart(mc_rets, mc_var, mc_cvar, theme_dark)
            st.plotly_chart(fig_mc_dist, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Simulation Paths
                fig_paths, p5_end, p95_end, median_end = create_mc_paths_chart(rets, n_paths=100, horizon=mc_horizon, theme_dark=theme_dark)
                st.plotly_chart(fig_paths, use_container_width=True)
                st.caption(f"End values: 5th pctl = {p5_end:.1f}, Median = {median_end:.1f}, 95th pctl = {p95_end:.1f}")
            
            with col2:
                # Confidence Cone
                fig_cone, cone_p5, cone_p50, cone_p95 = create_mc_cone_chart(rets, horizon=mc_horizon, theme_dark=theme_dark)
                st.plotly_chart(fig_cone, use_container_width=True)
                st.caption(f"At horizon: 5% worst = {cone_p5:.1f}, Median = {cone_p50:.1f}, 95% best = {cone_p95:.1f}")
            
            # Advanced Visualizations Section
            if HAS_ENHANCED_UTILS:
                st.markdown("---")
                st.markdown("### Advanced Visualizations")
                
                # Controls row
                ctrl_col1, ctrl_col2 = st.columns(2)
                with ctrl_col1:
                    cone_horizon = st.slider("Projection Horizon (days)", 10, 90, 30, key="viz_horizon")
                with ctrl_col2:
                    perf_metric = st.selectbox("Rolling Metric", ["sharpe", "volatility", "return", "sortino"], key="viz_metric")
                
                # Stacked visualizations for better viewing
                st.markdown("#### VaR Cone Projection")
                fig_var_cone = var_cone_chart(rets, horizon=cone_horizon, title=f"VaR Cone Projection ({cone_horizon}d)")
                fig_var_cone.update_layout(height=450)
                st.plotly_chart(fig_var_cone, use_container_width=True)
                
                st.markdown("#### Rolling Performance")
                fig_perf = rolling_performance_chart(rets, metric=perf_metric, title=f"Rolling {perf_metric.title()}")
                fig_perf.update_layout(height=450)
                st.plotly_chart(fig_perf, use_container_width=True)
                
                st.markdown("#### Cumulative Returns Comparison")
                combined = pd.DataFrame({ticker: rets, benchmark: bench_rets})
                fig_cum = cumulative_returns_chart(combined, title="Cumulative Returns")
                fig_cum.update_layout(height=450)
                st.plotly_chart(fig_cum, use_container_width=True)
        
        # TAB 4: STRESS TEST
        with tab4:
            st.subheader("Stress Testing")
            st.markdown(f"Impact on {ticker} Position")
            
            col1, col2 = st.columns(2)
            with col1:
                scenario = st.selectbox("Historical Scenario", list(STRESS_SCENARIOS.keys()), key="single_scenario")
                scenario_params = STRESS_SCENARIOS[scenario]
                st.caption(scenario_params['description'])
                st.markdown("---")
                st.metric("Market Shock", f"{scenario_params['market_shock']:.0%}")
                st.metric("Volatility Multiplier", f"{scenario_params['vol_multiplier']:.1f}x")
            
            with col2:
                st.markdown("#### Custom Scenario")
                custom_shock = st.slider("Market Shock (%)", -80, 0, int(scenario_params['market_shock']*100), key="single_shock") / 100
                custom_vol = st.slider("Vol Multiplier", 1.0, 5.0, scenario_params['vol_multiplier'], 0.5, key="single_vol")
                use_custom = st.checkbox("Use Custom Parameters", key="single_custom")
            
            shock = custom_shock if use_custom else scenario_params['market_shock']
            vol_mult = custom_vol if use_custom else scenario_params['vol_multiplier']
            
            position_value = st.number_input("Position Value ($)", 1000, 10000000, 100000, 10000, key="position_val")
            
            idio_vol = float(rets.std())
            np.random.seed(hash(scenario) % 2**32)
            random_shock = np.random.randn()
            stressed_ret = beta * shock + idio_vol * vol_mult * random_shock
            stressed_pnl = position_value * stressed_ret
            
            st.markdown("---")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Stressed Return", f"{stressed_ret:.2%}")
            col2.metric("P&L Impact", f"${stressed_pnl:,.0f}")
            col3.metric("Remaining Value", f"${position_value + stressed_pnl:,.0f}")
            
            fig_stress = go.Figure()
            fig_stress.add_trace(go.Bar(
                x=['Normal Daily', 'Stressed'],
                y=[metrics['ann_ret']/252*100, stressed_ret*100],
                marker_color=[COLORS['success'] if metrics['ann_ret'] > 0 else COLORS['danger'], COLORS['danger']],
                text=[f"{metrics['ann_ret']/252:.2%}", f"{stressed_ret:.2%}"],
                textposition='auto'
            ))
            fig_stress.update_layout(title=f"Return Comparison: {scenario}", yaxis_title="Return (%)",
                                    template='plotly_dark' if theme_dark else 'plotly_white', height=350)
            st.plotly_chart(fig_stress, use_container_width=True)
            
            # Macro Stress Scenarios (v4.3 Professional)
            if HAS_PRO_FEATURES:
                st.markdown("---")
                st.markdown("### Macro Stress Scenarios (v4.3)")
                
                from risk_engine import macro_stress_test
                
                # Get sector info
                info = fetch_info(ticker)
                ticker_sector = info.get('sector', 'Unknown')
                
                # Filter for macro scenarios (those with sector_betas)
                macro_scenarios = {k: v for k, v in STRESS_SCENARIOS.items() if 'sector_betas' in v}
                
                if macro_scenarios:
                    selected_macro = st.selectbox(
                        "Select Macro Event",
                        list(macro_scenarios.keys()),
                        key="macro_scenario"
                    )
                    
                    macro_params = macro_scenarios[selected_macro]
                    st.caption(macro_params['description'])
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Base Market Shock", f"{macro_params['market_shock']:.0%}")
                    with col2:
                        sector_beta = macro_params['sector_betas'].get(ticker_sector, 1.0)
                        st.metric(f"Sector Beta ({ticker_sector})", f"{sector_beta:.1f}x")
                    with col3:
                        adjusted_shock = macro_params['market_shock'] * sector_beta
                        st.metric("Adjusted Impact", f"{adjusted_shock:.0%}")
                    
                    # Show all sector impacts
                    with st.expander("Sector Impact Matrix", expanded=False):
                        sector_df = pd.DataFrame([
                            {'Sector': s, 'Beta': b, 'Impact': f"{macro_params['market_shock']*b:.1%}"}
                            for s, b in macro_params['sector_betas'].items()
                        ])
                        st.dataframe(sector_df, use_container_width=True)
                else:
                    st.info("No macro scenarios with sector-specific impacts available")
        
        # TAB 5: SENTIMENT ANALYSIS
        with tab5:
            st.subheader("Sentiment Analysis")
            st.caption("NLP-based sentiment scoring and VaR adjustment")
            
            if HAS_SENTIMENT_FEATURE and HAS_SENTIMENT:
                try:
                    import os
                    # Create SentimentService instance
                    sentiment_service = SentimentService(
                        polygon_key=os.getenv('POLYGON_API_KEY', ''),
                        alpaca_key=os.getenv('ALPACA_API_KEY', ''),
                        alpaca_secret=os.getenv('ALPACA_API_SECRET', '')
                    )
                    
                    # Render full sentiment tab with SentimentService
                    render_sentiment_tab(sentiment_service, ticker, rets)
                    
                except Exception as e:
                    # Fallback: Simple sentiment display using returns-based proxy
                    st.warning(f"Full sentiment service unavailable: {e}")
                    st.markdown("### Returns-Based Sentiment Proxy")
                    st.caption("Using price momentum as a sentiment indicator")
                    
                    # Calculate sentiment proxy based on returns
                    short_ma = rets.rolling(5).mean()
                    long_ma = rets.rolling(20).mean()
                    sentiment_proxy = (short_ma - long_ma) * 100
                    
                    # Current sentiment
                    current_sentiment = sentiment_proxy.iloc[-1] if len(sentiment_proxy) > 0 else 0
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if current_sentiment > 0.1:
                            st.metric("Momentum Sentiment", "Positive", delta=f"{current_sentiment:.2f}")
                        elif current_sentiment < -0.1:
                            st.metric("Momentum Sentiment", "Negative", delta=f"{current_sentiment:.2f}")
                        else:
                            st.metric("Momentum Sentiment", "Neutral", delta=f"{current_sentiment:.2f}")
                    
                    with col2:
                        # Recent trend
                        recent_ret = rets.tail(5).mean() * 252  # Annualized
                        st.metric("5-Day Trend (Ann.)", f"{recent_ret:.1%}")
                    
                    with col3:
                        # Volatility regime
                        recent_vol = rets.tail(20).std() * np.sqrt(252)
                        avg_vol = rets.std() * np.sqrt(252)
                        vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 1
                        vol_label = "Elevated" if vol_ratio > 1.2 else "Normal" if vol_ratio > 0.8 else "Low"
                        st.metric("Volatility Regime", vol_label, delta=f"{(vol_ratio-1)*100:.0f}%")
                    
                    # Sentiment trend chart
                    st.markdown("---")
                    fig_sent = go.Figure()
                    fig_sent.add_trace(go.Scatter(
                        x=sentiment_proxy.index,
                        y=sentiment_proxy.values,
                        mode='lines',
                        name='Momentum Sentiment',
                        line=dict(color=COLORS['primary'], width=2),
                        fill='tozeroy',
                        fillcolor='rgba(0, 150, 136, 0.2)'
                    ))
                    fig_sent.add_hline(y=0, line_dash="dash", line_color="gray")
                    fig_sent.update_layout(
                        title="Momentum-Based Sentiment Indicator",
                        xaxis_title="Date",
                        yaxis_title="Sentiment Score",
                        template='plotly_dark' if theme_dark else 'plotly_white',
                        height=400
                    )
                    st.plotly_chart(fig_sent, use_container_width=True)
                    
                    st.info("For full sentiment analysis with news and whale tracking, configure API keys (POLYGON_API_KEY, ALPACA_API_KEY)")
            else:
                st.info("Sentiment analysis requires the sentiment_service module and TextBlob/VADER installed.")
        
        # TAB 6: FACTORS
        with tab6:
            st.subheader("Factor Analysis")
            
            fa = FactorAnalyzer()
            
            st.markdown("### Fama-French 5-Factor Model")
            with st.spinner("Running factor regression..."):
                ff_results = fa.fama_french_regression(rets, bench_rets)
            
            if 'error' not in ff_results or ff_results.get('r_squared', 0) > 0:
                col1, col2, col3 = st.columns(3)
                col1.metric("Alpha (Annual)", f"{ff_results['alpha']:.2%}")
                col2.metric("R-Squared", f"{ff_results['r_squared']:.2%}")
                col3.metric("Observations", f"{ff_results.get('n_observations', 'N/A')}")
                
                if ff_results.get('loadings'):
                    st.markdown("#### Factor Loadings")
                    loadings_df = pd.DataFrame([ff_results['loadings']]).T
                    loadings_df.columns = ['Loading']
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.dataframe(loadings_df.style.format("{:.4f}"), use_container_width=True)
                    with col2:
                        fig_loadings = px.bar(x=list(ff_results['loadings'].keys()), 
                                             y=list(ff_results['loadings'].values()),
                                             title="Factor Loadings",
                                             color_discrete_sequence=[COLORS['primary']])
                        fig_loadings.update_layout(template='plotly_dark' if theme_dark else 'plotly_white', 
                                                  showlegend=False, height=300)
                        st.plotly_chart(fig_loadings, use_container_width=True)
            else:
                st.info(ff_results.get('error', 'Factor analysis unavailable'))
            
            st.markdown("---")
            
            st.markdown("### Kelly Criterion Position Sizing")
            kelly_fraction = st.slider("Kelly Fraction (safety multiplier)", 0.25, 1.0, 0.5, 0.05, key="kelly_frac")
            kelly_results = fa.kelly_criterion(rets, fraction=kelly_fraction)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Optimal Position Size", f"{kelly_results['kelly_pct']:.1%}")
            col2.metric("Win Rate", f"{kelly_results['win_rate']:.1%}")
            col3.metric("Win/Loss Ratio", f"{kelly_results['win_loss_ratio']:.2f}")
            col4.metric("Edge per Trade", f"{kelly_results['edge_per_trade']*10000:.1f} bps")
            
            st.caption(f"Using {kelly_fraction:.0%} Kelly (Half-Kelly is generally recommended for safety)")
            
            st.markdown("---")
            
            st.markdown("### ESG Sustainability Rating")
            with st.spinner("Fetching ESG data..."):
                esg_data = fa.get_esg_rating(ticker)
            
            col1, col2 = st.columns([1, 2])
            with col1:
                rating = esg_data.get('rating', 'NR')
                st.metric("ESG Rating", rating)
                st.metric("Total ESG Score", f"{esg_data.get('total_esg', 'N/A')}")
                st.caption(f"Source: {esg_data.get('source', 'N/A')}")
            
            with col2:
                esg_scores = {
                    'Environment': esg_data.get('environment_score', 0),
                    'Social': esg_data.get('social_score', 0),
                    'Governance': esg_data.get('governance_score', 0)
                }
                fig_esg = px.bar(x=list(esg_scores.keys()), y=list(esg_scores.values()),
                                title="ESG Pillar Scores (Lower = Better)",
                                color_discrete_sequence=[COLORS['success'], COLORS['primary'], COLORS['secondary']])
                fig_esg.update_layout(template='plotly_dark' if theme_dark else 'plotly_white', 
                                     showlegend=False, height=300)
                st.plotly_chart(fig_esg, use_container_width=True)
            
            # Style Factor Analysis (v4.3 Professional)
            if HAS_PRO_FEATURES:
                st.markdown("---")
                st.markdown("### Style Factor Analysis (v4.3)")
                
                try:
                    style_analyzer = StyleFactorAnalyzer()
                    
                    # Get current info for fundamentals
                    info = fetch_info(ticker)
                    fundamentals = {
                        'trailingPE': info.get('trailingPE'),
                        'forwardPE': info.get('forwardPE'),
                        'priceToBook': info.get('priceToBook'),
                        'dividendYield': info.get('dividendYield', 0),
                        'returnOnEquity': info.get('returnOnEquity'),
                        'profitMargins': info.get('profitMargins'),
                        'grossMargins': info.get('grossMargins'),
                        'debtToEquity': info.get('debtToEquity')
                    }
                    
                    with st.spinner("Analyzing style factors..."):
                        # Pass returns, prices, fundamentals, and benchmark_returns correctly
                        style_results = style_analyzer.analyze_style_factors(
                            returns=rets,
                            prices=prices,
                            fundamentals=fundamentals,
                            benchmark_returns=bench_rets
                        )
                    
                    if 'error' not in style_results:
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.markdown("#### Factor Scores")
                            scores = style_results.get('scores', {})
                            for factor_name, score in scores.items():
                                score_val = score if isinstance(score, (int, float)) else 50
                                if score_val >= 70:
                                    st.success(f"**{factor_name}**: {score_val:.0f}/100")
                                elif score_val >= 40:
                                    st.info(f"**{factor_name}**: {score_val:.0f}/100")
                                else:
                                    st.warning(f"**{factor_name}**: {score_val:.0f}/100")
                            
                            # Overall style classification
                            style_label = style_results.get('style_label', 'Blend')
                            st.metric("Style Profile", style_label)
                        
                        with col2:
                            # Radar chart - handle nested structure
                            radar_data = style_results.get('radar_data', {})
                            if radar_data:
                                # Extract categories and scores from nested structure
                                categories = radar_data.get('categories', [])
                                values = radar_data.get('scores', [])
                                
                                if categories and values:
                                    # Close the radar
                                    categories_closed = categories + [categories[0]]
                                    values_closed = values + [values[0]]
                                    
                                    fig_radar = go.Figure()
                                    fig_radar.add_trace(go.Scatterpolar(
                                        r=values_closed,
                                        theta=categories_closed,
                                        fill='toself',
                                        name=ticker,
                                        fillcolor='rgba(0, 122, 255, 0.3)',
                                        line_color='#007AFF'
                                    ))
                                    fig_radar.update_layout(
                                        polar=dict(
                                            radialaxis=dict(
                                                visible=True,
                                                range=[0, 100]
                                            ),
                                            bgcolor='rgba(0,0,0,0)'
                                        ),
                                        showlegend=False,
                                        title="Style Factor Radar",
                                        template='plotly_dark' if theme_dark else 'plotly_white',
                                        height=350
                                    )
                                    st.plotly_chart(fig_radar, use_container_width=True)
                    else:
                        st.info(style_results.get('error', 'Style analysis unavailable'))
                except Exception as e:
                    st.warning(f"Style factor analysis not available: {e}")
        
        # TAB 7: AI RISK
        with tab7:
            st.subheader("AI-Powered Risk Prediction")
            
            ml = MLPredictor()
            
            # Show model type indicator
            model_status = "XGBoost" if ml.model_type == 'xgboost' else "GradientBoosting"
            st.info(f"Using: **{model_status}** (ML Model)")
            
            with st.spinner(f"Training {ml.model_type} model..."):
                ml_results = ml.train_predict(rets, prices)
            
            if 'error' not in ml_results:
                # Display model type used
                model_name = ml_results.get('model_type', 'ML').replace('_', ' ').title()
                st.success(f"Model trained successfully using **{model_name}**")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("AI Predicted VaR", f"{ml_results['predicted_var']:.2%}")
                col2.metric("Model R-Squared", f"{ml_results['r2_score']:.3f}")
                col3.metric("Training R-Squared", f"{ml_results['r2_train']:.3f}")
                col4.metric("Mean Absolute Error", f"{ml_results['mae']:.4f}")
                
                st.markdown("---")
                
                st.markdown("### VaR Method Comparison")
                model_display_name = ml_results.get('model_type', 'ML').replace('_', ' ').title()
                comparison_df = pd.DataFrame({
                    'Method': [f'{model_display_name} ML', 'GARCH', 'Historical', 'Parametric'],
                    '95% VaR': [
                        ml_results['predicted_var'],
                        ml_results['comparison'].get('garch_approx', 0),
                        ml_results['comparison'].get('historical', 0),
                        abs(p_var)
                    ]
                })
                
                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe(comparison_df.style.format({'95% VaR': '{:.2%}'}), use_container_width=True)
                
                with col2:
                    fig_compare = px.bar(comparison_df, x='Method', y='95% VaR', 
                                        title="AI vs Traditional Methods",
                                        color_discrete_sequence=[COLORS['primary']])
                    fig_compare.update_layout(template='plotly_dark' if theme_dark else 'plotly_white', 
                                             showlegend=False, height=300)
                    st.plotly_chart(fig_compare, use_container_width=True)
                
                st.markdown("---")
                
                st.markdown("### Feature Importance")
                importance = ml_results.get('importance', {})
                if importance:
                    top_n = 10
                    top_features = dict(list(importance.items())[:top_n])
                    
                    fig_imp = px.bar(x=list(top_features.values()), y=list(top_features.keys()),
                                    orientation='h', title=f"Top {top_n} Predictive Features",
                                    color_discrete_sequence=[COLORS['primary']])
                    fig_imp.update_layout(template='plotly_dark' if theme_dark else 'plotly_white', height=350)
                    st.plotly_chart(fig_imp, use_container_width=True)
                
                st.markdown("---")
                
                st.markdown("### Model Details")
                col1, col2, col3 = st.columns(3)
                col1.metric("Training Samples", ml_results.get('n_train', 'N/A'))
                col2.metric("Test Samples", ml_results.get('n_test', 'N/A'))
                col3.metric("Top Feature", ml_results.get('top_features', ['N/A'])[0])
                
                # Ensemble Predictions Section
                st.markdown("---")
                st.markdown("### Ensemble VaR Predictions")
                st.caption(f"Combine multiple models ({model_status}, GARCH, Historical, Parametric, EWMA) for robust VaR estimates")
                
                # Auto-generate ensemble prediction
                with st.spinner("Running ensemble models..."):
                    ensemble = ml.ensemble_predict(rets, prices)
                    
                    if 'error' not in ensemble:
                        ens_col1, ens_col2, ens_col3 = st.columns(3)
                        ens_col1.metric("Ensemble VaR", f"{ensemble['ensemble_var']:.2%}")
                        ens_col2.metric("Model Spread", f"{ensemble['spread']:.2%}")
                        ens_col3.metric("Models Used", ensemble['n_models'])
                        
                        # Individual predictions chart
                        fig_ens = var_comparison_chart(
                            ensemble['individual_predictions'],
                            title="VaR by Model (Ensemble)"
                        )
                        st.plotly_chart(fig_ens, use_container_width=True)
                        
                        # Confidence intervals
                        with st.spinner("Calculating confidence intervals..."):
                            ci_result = ml.predict_with_confidence(rets, prices, n_bootstrap=50)
                            
                            if ci_result and 'error' not in ci_result:
                                st.markdown("#### Bootstrap Confidence Intervals")
                                ci_col1, ci_col2, ci_col3 = st.columns(3)
                                ci_col1.metric("Mean VaR", f"{ci_result['mean_var']:.2%}")
                                ci_col2.metric("95% CI Lower", f"{ci_result['ci_5']:.2%}")
                                ci_col3.metric("95% CI Upper", f"{ci_result['ci_95']:.2%}")
                    else:
                        st.error(ensemble.get('error', 'Ensemble prediction failed'))
            else:
                st.error(ml_results.get('error', 'ML prediction failed'))
                st.info("Ensure XGBoost is installed: pip install xgboost")
        
        # TAB 8: OPTIONS ANALYTICS (NEW)
        with tab8:
            st.subheader("Options Analytics")
            
            if HAS_FEATURES:
                options = OptionsAnalytics()
                current_price = float(prices.iloc[-1]) if len(prices) > 0 else 100
                hist_vol = metrics['ann_vol']
                
                st.markdown("#### Black-Scholes Options Calculator")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    opt_type = st.selectbox("Option Type", ["Call", "Put"], key="opt_type_widget")
                    strike = st.number_input("Strike Price ($)", 
                                            value=float(round(current_price * 1.0)), 
                                            min_value=1.0, step=1.0, key="opt_strike_widget")
                with col2:
                    days_to_exp = st.slider("Days to Expiry", 1, 365, 30, key="opt_days_exp_widget")
                    T = days_to_exp / 365
                with col3:
                    vol_input = st.slider("Volatility (%)", 5, 100, int(hist_vol * 100), key="opt_vol_widget") / 100
                    rf_option = st.number_input("Risk-Free Rate (%)", 0.0, 10.0, 4.5, 0.1, key="opt_rf_widget") / 100
                with col4:
                    st.metric("Current Price", f"${current_price:.2f}")
                    st.metric("Historical Vol", f"{hist_vol:.1%}")
                
                # Calculate option metrics
                opt_analysis = options.analyze_option(
                    S=current_price, K=strike, T=T, r=rf_option, 
                    sigma=vol_input, option_type=opt_type.lower()
                )
                
                st.markdown("---")
                st.markdown("#### Option Pricing & Greeks")
                
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Option Price", f"${opt_analysis['price']:.2f}")
                col2.metric("Premium (100 shares)", f"${opt_analysis['premium']:.2f}")
                col3.metric("Breakeven", f"${opt_analysis['breakeven']:.2f}")
                col4.metric("Moneyness", opt_analysis['moneyness'])
                col5.metric("Intrinsic Value", f"${opt_analysis['intrinsic_value']:.2f}")
                
                st.markdown("---")
                greeks = opt_analysis['greeks']
                
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Delta (Δ)", f"{greeks['delta']:.4f}", 
                           help="Change in option price per $1 stock move")
                col2.metric("Gamma (Γ)", f"{greeks['gamma']:.4f}",
                           help="Rate of change of delta")
                col3.metric("Theta (Θ)", f"${greeks['theta']:.4f}",
                           help="Daily time decay in $")
                col4.metric("Vega (ν)", f"${greeks['vega']:.4f}",
                           help="Sensitivity to 1% vol change")
                col5.metric("Rho (ρ)", f"${greeks['rho']:.4f}",
                           help="Sensitivity to 1% rate change")
                
                # Payoff diagram
                st.markdown("---")
                st.markdown("#### Payoff Diagram at Expiration")
                
                prices_range, payoffs = options.calculate_payoff_diagram(
                    S=current_price, K=strike, premium=opt_analysis['price'],
                    option_type=opt_type.lower(), is_long=True, price_range=0.3
                )
                
                fig_payoff = go.Figure()
                fig_payoff.add_trace(go.Scatter(
                    x=prices_range, y=payoffs, name='Profit/Loss',
                    line=dict(color=COLORS['primary'], width=2),
                    fill='tozeroy', 
                    fillcolor='rgba(0, 122, 255, 0.1)'
                ))
                fig_payoff.add_vline(x=current_price, line_dash="dash", 
                                     line_color=COLORS['warning'],
                                     annotation_text=f"Current: ${current_price:.0f}")
                fig_payoff.add_vline(x=strike, line_dash="dot", 
                                     line_color=COLORS['gray'],
                                     annotation_text=f"Strike: ${strike:.0f}")
                fig_payoff.add_hline(y=0, line_color=COLORS['danger'], line_width=1)
                fig_payoff.update_layout(
                    title=f"{opt_type} Option Payoff at Expiration",
                    xaxis_title="Stock Price at Expiration ($)",
                    yaxis_title="Profit/Loss ($)",
                    template='plotly_dark' if theme_dark else 'plotly_white',
                    height=350
                )
                st.plotly_chart(fig_payoff, use_container_width=True)
                
                # Strategy Analysis
                st.markdown("---")
                st.markdown("#### Strategy Analysis")
                
                strategy = st.selectbox("Select Strategy", 
                                       ["Covered Call", "Protective Put", "Straddle"],
                                       key="opt_strategy_widget")
                
                if strategy == "Covered Call":
                    cc = options.covered_call_analysis(current_price, strike, T, rf_option, vol_input)
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Premium Received", f"${cc['premium_received']:.2f}")
                    col2.metric("Max Profit", f"${cc['max_profit']:.2f}")
                    col3.metric("Breakeven", f"${cc['breakeven']:.2f}")
                    st.caption(f"Return if called: {cc['return_if_called']:.1%} annualized")
                
                elif strategy == "Protective Put":
                    pp = options.protective_put_analysis(current_price, strike, T, rf_option, vol_input)
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Put Premium", f"${pp['premium_paid']:.2f}")
                    col2.metric("Max Loss", f"${pp['max_loss']:.2f}")
                    col3.metric("Protection Level", f"${pp['protection_level']:.2f}")
                    st.caption(f"Cost of protection: {pp['cost_of_protection']:.2%}")
                
                elif strategy == "Straddle":
                    strad = options.straddle_analysis(current_price, current_price, T, rf_option, vol_input)
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Cost", f"${strad['total_cost']:.2f}")
                    col2.metric("Upper Breakeven", f"${strad['upper_breakeven']:.2f}")
                    col3.metric("Lower Breakeven", f"${strad['lower_breakeven']:.2f}")
                    st.caption(f"Required move to breakeven: {strad['required_move']:.1%}")
            else:
                st.info("Options analytics module not available. Check installation.")
        
        # TAB 9: FUNDAMENTALS (NEW)
        with tab9:
            st.subheader("Fundamental Analysis")
            
            if HAS_FEATURES and info:
                fa = FundamentalAnalyzer()
                
                # Prepare info dict with proper field mappings
                fund_info = {
                    'ticker': ticker,
                    'name': info.get('shortName', info.get('longName', ticker)),
                    'sector': info.get('sector', 'Unknown'),
                    'industry': info.get('industry', 'Unknown'),
                    'pe_ratio': info.get('trailingPE'),
                    'forward_pe': info.get('forwardPE'),
                    'price_to_book': info.get('priceToBook'),
                    'price_to_sales': info.get('priceToSalesTrailing12Months'),
                    'peg_ratio': info.get('pegRatio'),
                    'ev_to_ebitda': info.get('enterpriseToEbitda'),
                    'roe': info.get('returnOnEquity'),
                    'roa': info.get('returnOnAssets'),
                    'profit_margin': info.get('profitMargins'),
                    'operating_margin': info.get('operatingMargins'),
                    'debt_to_equity': info.get('debtToEquity'),
                    'current_ratio': info.get('currentRatio'),
                    'quick_ratio': info.get('quickRatio'),
                    'revenue_growth': info.get('revenueGrowth'),
                    'earnings_growth': info.get('earningsGrowth'),
                    'price': info.get('currentPrice', info.get('regularMarketPrice')),
                    'eps': info.get('trailingEps'),
                    'book_value': info.get('bookValue'),
                    'free_cash_flow': info.get('freeCashflow'),
                    'shares_outstanding': info.get('sharesOutstanding')
                }
                
                analysis = fa.analyze_fundamentals(fund_info)
                
                # Quality Score
                if analysis.get('quality_score'):
                    qs = analysis['quality_score']
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Quality Grade", qs['grade'])
                    col2.metric("Score", f"{qs['total_score']}/{qs['max_score']}")
                    col3.metric("Sector", analysis.get('sector', 'N/A'))
                    col4.metric("Industry", analysis.get('industry', 'N/A')[:20])
                
                st.markdown("---")
                
                # Valuation Section
                st.markdown("#### Valuation Ratios")
                val = analysis.get('valuation', {})
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("P/E Ratio", f"{val.get('pe_ratio', 'N/A'):.1f}" if val.get('pe_ratio') else "N/A")
                col2.metric("Forward P/E", f"{val.get('forward_pe', 'N/A'):.1f}" if val.get('forward_pe') else "N/A")
                col3.metric("P/B Ratio", f"{val.get('price_to_book', 'N/A'):.1f}" if val.get('price_to_book') else "N/A")
                col4.metric("P/S Ratio", f"{val.get('price_to_sales', 'N/A'):.1f}" if val.get('price_to_sales') else "N/A")
                col5.metric("EV/EBITDA", f"{val.get('ev_to_ebitda', 'N/A'):.1f}" if val.get('ev_to_ebitda') else "N/A")
                
                # Profitability
                st.markdown("---")
                st.markdown("#### Profitability")
                prof = analysis.get('profitability', {})
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("ROE", f"{prof.get('roe', 0):.1%}" if prof.get('roe') else "N/A")
                col2.metric("ROA", f"{prof.get('roa', 0):.1%}" if prof.get('roa') else "N/A")
                col3.metric("Profit Margin", f"{prof.get('profit_margin', 0):.1%}" if prof.get('profit_margin') else "N/A")
                col4.metric("Operating Margin", f"{prof.get('operating_margin', 0):.1%}" if prof.get('operating_margin') else "N/A")
                
                # Financial Health
                st.markdown("---")
                st.markdown("#### Financial Health")
                health = analysis.get('financial_health', {})
                col1, col2, col3 = st.columns(3)
                col1.metric("Debt/Equity", f"{health.get('debt_to_equity', 'N/A'):.0f}" if health.get('debt_to_equity') else "N/A")
                col2.metric("Current Ratio", f"{health.get('current_ratio', 'N/A'):.2f}" if health.get('current_ratio') else "N/A")
                col3.metric("Quick Ratio", f"{health.get('quick_ratio', 'N/A'):.2f}" if health.get('quick_ratio') else "N/A")
                
                # Growth
                st.markdown("---")
                st.markdown("#### Growth Metrics")
                growth = analysis.get('growth', {})
                col1, col2 = st.columns(2)
                col1.metric("Revenue Growth", f"{growth.get('revenue_growth', 0):.1%}" if growth.get('revenue_growth') else "N/A")
                col2.metric("Earnings Growth", f"{growth.get('earnings_growth', 0):.1%}" if growth.get('earnings_growth') else "N/A")
                
                # Intrinsic Value Estimate
                st.markdown("---")
                st.markdown("#### Intrinsic Value Estimate")
                iv = fa.calculate_intrinsic_value(fund_info)
                if iv and 'methods' in iv:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Avg Intrinsic Value", f"${iv['average_intrinsic_value']:.2f}" if iv['average_intrinsic_value'] else "N/A")
                    col2.metric("Current Price", f"${iv['current_price']:.2f}" if iv['current_price'] else "N/A")
                    if iv.get('upside_potential'):
                        col3.metric("Upside Potential", f"{iv['upside_potential']:.1%}",
                                   delta=f"{'Undervalued' if iv['upside_potential'] > 0 else 'Overvalued'}")
                    
                    st.caption("Valuation methods: " + ", ".join(iv['methods'].keys()))
                else:
                    st.info("Insufficient data for intrinsic value calculation")
            else:
                st.info("Fundamental analysis requires company info data")
        
        # TAB 10: EXPORT DATA & REPORTS
        with tab10:
            st.subheader("Export Data & Reports")
            
            metrics_export = {
                'Ticker': ticker,
                'Period': f"{days_back} days",
                'Ann Return': f"{metrics['ann_ret']:.2%}",
                'Ann Volatility': f"{metrics['ann_vol']:.2%}",
                'Max Drawdown': f"{metrics['max_dd']:.2%}",
                'Sharpe': f"{metrics['sharpe']:.2f}",
                'Sortino': f"{metrics['sortino']:.2f}",
                'Calmar': f"{metrics['calmar']:.2f}",
                'Beta': f"{beta:.2f}" if not np.isnan(beta) else "N/A",
                f"VaR {conf_level:.0%}": f"{p_var:.2%}",
                'CVaR': f"{cv:.2%}"
            }
            
            metrics_df = pd.DataFrame([metrics_export])
            csv_metrics = metrics_df.to_csv(index=False)
            
            st.markdown("#### CSV Downloads")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.download_button("Download Metrics", csv_metrics, f"{ticker}_risk_metrics.csv", "text/csv")
            
            with col2:
                returns_export_df = pd.DataFrame({'Date': rets.index, 'Return': rets.values})
                csv_returns = returns_export_df.to_csv(index=False)
                st.download_button("Download Returns", csv_returns, f"{ticker}_returns.csv", "text/csv")
            
            with col3:
                prices_export_df = pd.DataFrame({'Date': prices.index, 'Price': prices.values})
                csv_prices = prices_export_df.to_csv(index=False)
                st.download_button("Download Prices", csv_prices, f"{ticker}_prices.csv", "text/csv")
            
            # PDF Report Generation
            st.markdown("---")
            st.markdown("#### PDF Risk Report")
            
            if HAS_FEATURES:
                report_gen = ReportGenerator()
                if report_gen.is_available():
                    if st.button("Generate PDF Report", type="primary", key="single_gen_pdf_btn"):
                        with st.spinner("Generating professional PDF report..."):
                            var_data = {
                                'var_95': abs(p_var),
                                'var_99': abs(parametric_var(rets, var_horizon, 0.99)),
                                'hist_var_95': abs(h_var),
                                'hist_var_99': abs(historical_var(rets, var_horizon, 0.99)),
                                'cvar': abs(cv),
                                'cvar_99': abs(cvar(rets, 0.99))
                            }
                            
                            pdf_bytes = report_gen.generate_single_stock_report(
                                ticker=ticker,
                                metrics=metrics,
                                var_data=var_data,
                                info=info
                            )
                            
                            if pdf_bytes:
                                st.download_button(
                                    "Download PDF Report",
                                    data=pdf_bytes,
                                    file_name=f"{ticker}_risk_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                                    mime="application/pdf"
                                )
                                st.success("PDF report generated successfully!")
                            else:
                                st.error("Failed to generate PDF report")
                else:
                    st.info("PDF generation requires fpdf2: `pip install fpdf2`")
            else:
                st.info("Report generation module not available")
            
            # Alerts Management
            st.markdown("---")
            st.markdown("#### Risk Alerts")
            
            if HAS_FEATURES:
                alert_manager = AlertManager()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Create New Alert**")
                    alert_metric = st.selectbox("Metric", ['var', 'volatility', 'max_drawdown', 'sharpe'], key="alert_metric")
                    alert_threshold = st.number_input("Threshold", value=0.05, step=0.01, key="alert_thresh")
                    alert_direction = st.selectbox("Trigger when", ['above', 'below'], key="alert_dir")
                    
                    if st.button("➕ Add Alert", key="single_add_alert_btn"):
                        alert_manager.add_alert(ticker, alert_metric, alert_threshold, alert_direction)
                        st.success(f"Alert created for {ticker}")
                
                with col2:
                    st.markdown("**Active Alerts**")
                    alerts = alert_manager.get_alerts(ticker)
                    if alerts:
                        for alert in alerts:
                            st.text(f"- {alert['name']}")
                    else:
                        st.caption("No active alerts for this ticker")
            else:
                st.info("Alerts module not available")

# ============================================================================
# PORTFOLIO MODE
# ============================================================================
else:
    st.markdown("### Portfolio Analysis")
    
    # Portfolio Save/Load Section (NEW)
    if HAS_FEATURES:
        portfolio_store = PortfolioStore()
        
        with st.expander("Saved Portfolios", expanded=False):
            saved_portfolios = portfolio_store.list_portfolios()
            
            if saved_portfolios:
                col1, col2 = st.columns([3, 1])
                with col1:
                    selected_portfolio = st.selectbox(
                        "Load Portfolio", 
                        [p['name'] for p in saved_portfolios],
                        key="load_portfolio_select"
                    )
                with col2:
                    if st.button("Load", key="load_portfolio_btn"):
                        loaded = portfolio_store.load_portfolio(selected_portfolio)
                        if loaded:
                            st.session_state['loaded_tickers'] = ",".join(loaded['tickers'])
                            st.session_state['loaded_weights'] = loaded['weights']
                            st.success(f"Loaded: {selected_portfolio}")
                            st.rerun()
            else:
                st.caption("No saved portfolios yet")
    
    # Initialize weights dict before form
    weights = {}
    weights_array = np.array([])
    
    # Use form to batch portfolio input changes
    with st.form(key="portfolio_form"):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            default_tickers = st.session_state.get('loaded_tickers', "AAPL, MSFT, GOOGL, AMZN, NVDA")
            tickers_input = st.text_input("Enter Tickers (comma-separated)", default_tickers, key="port_tickers_input_widget")
            tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        
        with col2:
            days_back = st.slider("Days", 100, 2000, 756, key="port_days_widget")
        
        if tickers:
            st.markdown("#### Portfolio Weights")
            weights = {}
            loaded_weights = st.session_state.get('loaded_weights', {})
            cols = st.columns(min(len(tickers), 5))  # Max 5 columns per row
            for i, ticker in enumerate(tickers):
                col_idx = i % 5
                with cols[col_idx]:
                    default_weight = int(loaded_weights.get(ticker, 100/len(tickers)))
                    weights[ticker] = st.number_input(f"{ticker} %", 0, 100, default_weight, key=f"port_w_{ticker}")
            
            total_weight = sum(weights.values())
            if total_weight != 100:
                st.warning(f"Weights sum to {total_weight}%. Should be 100%.")
            
            weights_array = np.array(list(weights.values())) / 100
        
        col1, col2 = st.columns(2)
        with col1:
            bench_selection = st.selectbox("Benchmark", list(POPULAR_BENCHMARKS.keys()), key="port_bench_widget")
            benchmark = POPULAR_BENCHMARKS[bench_selection] if bench_selection != "Custom..." else "^GSPC"
        with col2:
            st.empty()  # Placeholder - confidence slider moved outside
        
        # Form submit button
        portfolio_submitted = st.form_submit_button("Analyze Portfolio", type="primary", use_container_width=True)
    
    # REACTIVE SLIDER - Outside form so it updates VaR live
    if st.session_state.port_analyzed:
        conf_level = st.slider(
            "Confidence Level (Reactive)", 0.90, 0.999,
            st.session_state.reactive_port_conf_level, 0.01,
            key="reactive_port_conf_widget"
        )
        st.session_state.reactive_port_conf_level = conf_level
    else:
        conf_level = 0.95
    
    # Save Portfolio Button (outside form)
    if tickers and weights and HAS_FEATURES:
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            portfolio_name = st.text_input("Portfolio Name", "My Portfolio", key="save_port_name")
        with col2:
            if st.button("Save Portfolio", key="save_portfolio_btn"):
                portfolio_store.save_portfolio(portfolio_name, tickers, weights)
                st.success(f"Saved: {portfolio_name}")
    
    # Process form submission
    if portfolio_submitted and tickers:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        with st.spinner("Fetching portfolio data..."):
            portfolio_data = fetch_multiple(tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            bench_data = fetch_data(benchmark, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        
        if portfolio_data.empty:
            st.error("Could not fetch portfolio data.")
            st.stop()
        
        if 'Close' in portfolio_data.columns.get_level_values(0):
            prices_df = portfolio_data['Close']
        else:
            prices_df = portfolio_data
        
        returns_df = np.log(prices_df / prices_df.shift(1)).dropna()
        
        port_rets = portfolio_returns(returns_df, weights_array)
        port_metrics = compute_metrics(port_rets, rf_rate)
        
        bench_prices = bench_data['Close'].iloc[:, 0] if isinstance(bench_data['Close'], pd.DataFrame) else bench_data['Close']
        bench_rets = compute_returns(bench_prices)
        
        betas = {}
        for ticker in tickers:
            if ticker in returns_df.columns:
                b, _ = compute_beta(returns_df[ticker], bench_rets.reindex(returns_df.index).dropna())
                betas[ticker] = b if not np.isnan(b) else 1.0
        
        st.session_state.port_analyzed = True
        st.session_state.port_tickers = tickers
        st.session_state.port_weights_array = weights_array
        st.session_state.port_returns_df = returns_df
        st.session_state.port_rets = port_rets
        st.session_state.port_metrics = port_metrics
        st.session_state.port_bench_rets = bench_rets
        st.session_state.port_betas = betas
        st.session_state.port_rf_rate = rf_rate
        st.session_state.port_theme_dark = theme_dark
        st.session_state.port_prices_df = prices_df
        st.session_state.port_weights = weights
        st.session_state.port_benchmark = benchmark
        # Initialize reactive params on first analysis
        st.session_state.reactive_port_conf_level = 0.95
    
    # DASHBOARD RENDERING - Only uses session state data
    if st.session_state.port_analyzed:
        # Load all data from session state (immutable after analysis)
        tickers = st.session_state.port_tickers
        weights_array = st.session_state.port_weights_array
        returns_df = st.session_state.port_returns_df
        port_rets = st.session_state.port_rets
        port_metrics = st.session_state.port_metrics
        bench_rets = st.session_state.port_bench_rets
        betas = st.session_state.port_betas
        rf_rate = st.session_state.port_rf_rate
        theme_dark = st.session_state.port_theme_dark
        prices_df = st.session_state.port_prices_df
        weights = st.session_state.port_weights
        benchmark = st.session_state.get('port_benchmark', '^GSPC')
        
        # Use REACTIVE values from slider
        conf_level = st.session_state.reactive_port_conf_level
        
        # Use FIXED tab list to prevent index shifting
        tabs = st.tabs(PORTFOLIO_TABS)
        tab_idx = 0
        
        # TAB: Summary
        with tabs[tab_idx]:
            tab_idx += 1
            st.subheader("Portfolio Risk Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Ann. Return", f"{port_metrics['ann_ret']:.1%}")
            col2.metric("Ann. Volatility", f"{port_metrics['ann_vol']:.1%}")
            col3.metric("Max Drawdown", f"{port_metrics['max_dd']:.1%}")
            col4.metric("Sharpe Ratio", f"{port_metrics['sharpe']:.2f}")
            
            port_var = portfolio_var(returns_df, weights_array, conf_level)
            port_hvar = historical_var(port_rets, 1, conf_level)
            port_cvar = cvar(port_rets, conf_level)
            
            col5, col6, col7, col8 = st.columns(4)
            col5.metric("Parametric VaR", f"{port_var:.2%}")
            col6.metric("Historical VaR", f"{port_hvar:.2%}")
            col7.metric("CVaR", f"{port_cvar:.2%}")
            col8.metric("Sortino Ratio", f"{port_metrics['sortino']:.2f}")
            
            st.markdown("---")
            st.markdown("#### Individual Asset Metrics")
            
            individual_metrics = []
            for ticker in tickers:
                if ticker in returns_df.columns:
                    m = compute_metrics(returns_df[ticker], rf_rate)
                    individual_metrics.append({
                        'Ticker': ticker,
                        'Weight': f"{weights[ticker]:.0f}%",
                        'Ann Return': f"{m['ann_ret']:.1%}",
                        'Ann Vol': f"{m['ann_vol']:.1%}",
                        'Sharpe': f"{m['sharpe']:.2f}",
                        'Beta': f"{betas.get(ticker, 'N/A'):.2f}" if isinstance(betas.get(ticker), float) else "N/A"
                    })
            
            st.dataframe(pd.DataFrame(individual_metrics), use_container_width=True, hide_index=True)
            
            st.markdown("#### Risk Contribution by Asset")
            risk_contrib = marginal_var_contribution(returns_df, weights_array, conf_level)
            
            fig_contrib = px.pie(values=risk_contrib.values * 100, names=risk_contrib.index,
                                title="Risk Contribution (%)", 
                                color_discrete_sequence=px.colors.sequential.Blues_r)
            fig_contrib.update_layout(template='plotly_dark' if theme_dark else 'plotly_white', height=350)
            st.plotly_chart(fig_contrib, use_container_width=True)
            
            # Liquidity Risk Analysis (v4.3 Professional)
            if HAS_PRO_FEATURES:
                st.markdown("---")
                st.markdown("### Liquidity Risk Analysis (v4.3)")
                
                # Gather volume data
                volumes = {}
                prices_current = {}
                for ticker in tickers:
                    try:
                        info = fetch_info(ticker)
                        volumes[ticker] = info.get('averageVolume', 0) or info.get('volume', 0)
                        prices_current[ticker] = info.get('regularMarketPrice') or info.get('currentPrice', 100)
                    except:
                        volumes[ticker] = 0
                        prices_current[ticker] = 100
                
                # Calculate position values (assume $100k portfolio)
                portfolio_value = st.number_input(
                    "Portfolio Value ($)", 
                    min_value=10000, 
                    max_value=100000000, 
                    value=100000, 
                    step=10000,
                    key="liquidity_portfolio_value"
                )
                
                positions = {ticker: portfolio_value * weights[ticker] / 100 for ticker in tickers}
                
                # Calculate liquidity risk
                liquidity_results = calculate_liquidity_risk(
                    positions=positions,
                    volumes=volumes,
                    prices=prices_current,
                    participation_rate=0.10
                )
                
                if 'error' not in liquidity_results:
                    portfolio_liq = liquidity_results.get('portfolio', {})
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric(
                        "Time to Liquidate",
                        f"{portfolio_liq.get('weighted_ttl_days', 0):.1f} days"
                    )
                    col2.metric(
                        "Market Impact",
                        f"{portfolio_liq.get('weighted_impact_bps', 0):.0f} bps"
                    )
                    col3.metric(
                        "Liquidity Score",
                        f"{portfolio_liq.get('liquidity_score', 0):.0f}/100"
                    )
                    warning_level = portfolio_liq.get('warning_level', 'low')
                    col4.metric("Warning Level", warning_level.upper())
                    
                    # Show warnings if any
                    warnings = liquidity_results.get('warnings', [])
                    if warnings:
                        st.warning(f"{len(warnings)} liquidity warning(s) detected")
                        with st.expander("View Liquidity Warnings"):
                            for w in warnings:
                                if w['level'] == 'critical':
                                    st.error(f"**{w['ticker']}**: {w['message']}")
                                elif w['level'] == 'high':
                                    st.warning(f"**{w['ticker']}**: {w['message']}")
                    
                    # Detailed liquidity by asset
                    with st.expander("Detailed Liquidity Metrics by Asset"):
                        liq_df = []
                        for ticker, data in liquidity_results.get('positions', {}).items():
                            if 'error' not in data:
                                liq_df.append({
                                    'Ticker': ticker,
                                    'TTL (days)': f"{data.get('ttl_days', 0):.1f}",
                                    'Impact (bps)': f"{data.get('market_impact_bps', 0):.0f}",
                                    'Score': f"{data.get('liquidity_score', 0):.0f}",
                                    'Grade': data.get('grade', 'N/A'),
                                    '% of ADV': f"{data.get('pct_of_adv', 0):.1f}%"
                                })
                        if liq_df:
                            st.dataframe(pd.DataFrame(liq_df), use_container_width=True, hide_index=True)
                else:
                    st.info("Liquidity analysis requires volume data")
        
        # TAB 2: MONTE CARLO (NEW FOR PORTFOLIO)
        with tabs[tab_idx]:
            tab_idx += 1
            st.subheader("Portfolio Monte Carlo Simulation")
            
            col1, col2 = st.columns([1, 3])
            with col1:
                mc_sims = st.selectbox("Simulations", [1000, 5000, 10000, 50000], index=2, key="port_mc_sims")
                mc_horizon = st.selectbox("Horizon (days)", [5, 10, 20, 60, 120], index=2, key="port_mc_horizon")
            
            with st.spinner("Running Monte Carlo simulation..."):
                mc_rets = mc_simulation(port_rets, mc_sims, mc_horizon)
                mc_var = np.percentile(mc_rets, 5)
                mc_cvar = mc_rets[mc_rets <= np.percentile(mc_rets, 5)].mean()
                mc_mean = mc_rets.mean()
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("MC 95% VaR", f"{mc_var:.2%}")
            col2.metric("MC CVaR", f"{mc_cvar:.2%}")
            col3.metric("Expected Return", f"{mc_mean:.2%}")
            col4.metric("95th Percentile", f"{np.percentile(mc_rets, 95):.2%}")
            
            st.markdown("---")
            
            fig_mc_dist = create_mc_distribution_chart(mc_rets, mc_var, mc_cvar, theme_dark)
            st.plotly_chart(fig_mc_dist, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                fig_paths, _, _, _ = create_mc_paths_chart(port_rets, n_paths=100, horizon=mc_horizon, theme_dark=theme_dark)
                st.plotly_chart(fig_paths, use_container_width=True)
            
            with col2:
                fig_cone, _, _, _ = create_mc_cone_chart(port_rets, horizon=mc_horizon, theme_dark=theme_dark)
                st.plotly_chart(fig_cone, use_container_width=True)
        
        # TAB 3: CORRELATION
        with tabs[tab_idx]:
            tab_idx += 1
            st.subheader("Correlation Matrix")
            corr_matrix = returns_df.corr()
            fig_corr = px.imshow(corr_matrix, text_auto='.2f', aspect='auto',
                                title="Asset Correlation Heatmap",
                                color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
            fig_corr.update_layout(height=450, template='plotly_dark' if theme_dark else 'plotly_white')
            st.plotly_chart(fig_corr, use_container_width=True)
        
        # TAB 4: STRESS TESTING
        with tabs[tab_idx]:
            tab_idx += 1
            st.subheader("Stress Testing")
            
            col1, col2 = st.columns(2)
            with col1:
                scenario = st.selectbox("Select Scenario", list(STRESS_SCENARIOS.keys()), key="port_scenario_widget")
                scenario_params = STRESS_SCENARIOS[scenario]
                st.caption(scenario_params['description'])
            
            with col2:
                custom_shock = st.slider("Market Shock (%)", -80, 0, int(scenario_params['market_shock']*100), key="port_stress_shock") / 100
                custom_vol = st.slider("Vol Multiplier", 1.0, 5.0, scenario_params['vol_multiplier'], 0.5, key="port_stress_vol")
            
            use_custom = st.checkbox("Use Custom Parameters", key="port_stress_custom")
            shock = custom_shock if use_custom else scenario_params['market_shock']
            vol_mult = custom_vol if use_custom else scenario_params['vol_multiplier']
            
            stress_results = stress_test_portfolio(returns_df, weights_array, betas, shock, vol_mult)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Portfolio Impact", f"{stress_results['portfolio']:.1%}")
            col2.metric("Scenario", scenario)
            col3.metric("Applied Shock", f"{shock:.0%}")
            
            st.markdown("---")
            
            stress_df = pd.DataFrame([{'Asset': k, 'Stressed Return': f"{v:.2%}"} for k, v in stress_results['individual'].items()])
            st.dataframe(stress_df, use_container_width=True, hide_index=True)
        
        # TAB 5: OPTIMIZATION
        with tabs[tab_idx]:
            tab_idx += 1
            st.subheader("Portfolio Optimization (MPT)")
            
            with st.spinner("Optimizing..."):
                opt_result = optimize_portfolio(returns_df)
                ef = efficient_frontier(returns_df)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Expected Return", f"{opt_result['return']:.1%}")
            col2.metric("Expected Volatility", f"{opt_result['volatility']:.1%}")
            col3.metric("Sharpe Ratio", f"{opt_result['sharpe']:.2f}")
            
            opt_weights_df = pd.DataFrame({
                'Asset': tickers,
                'Current Weight': [f"{w:.1%}" for w in weights_array],
                'Optimal Weight': [f"{w:.1%}" for w in opt_result['weights']]
            })
            st.dataframe(opt_weights_df, use_container_width=True, hide_index=True)
            
            fig_weights = go.Figure()
            fig_weights.add_trace(go.Bar(name='Current', x=tickers, y=weights_array*100, marker_color=COLORS['gray']))
            fig_weights.add_trace(go.Bar(name='Optimal', x=tickers, y=opt_result['weights']*100, marker_color=COLORS['primary']))
            fig_weights.update_layout(barmode='group', title='Current vs Optimal Weights', yaxis_title='Weight (%)',
                                     template='plotly_dark' if theme_dark else 'plotly_white', height=350)
            st.plotly_chart(fig_weights, use_container_width=True)
            
            # Efficient Frontier
            if len(ef) > 0:
                fig_ef = px.scatter(ef, x='volatility', y='return', 
                                   title="Efficient Frontier",
                                   color='sharpe', color_continuous_scale='Blues')
                fig_ef.update_layout(template='plotly_dark' if theme_dark else 'plotly_white', height=350,
                                    xaxis_title='Volatility', yaxis_title='Expected Return')
                st.plotly_chart(fig_ef, use_container_width=True)
            
            # Advanced Portfolio Optimization Section (integrated from Enhanced tab)
            if HAS_ENHANCED_UTILS:
                st.markdown("---")
                st.markdown("### Advanced Portfolio Optimization")
                
                port_enhanced_subtab = st.radio(
                    "Select Analysis",
                    ["Risk Parity", "Black-Litterman", "Transaction Costs", 
                     "Rebalancing", "Risk Decomposition"],
                    horizontal=True,
                    key="port_enhanced_subtab"
                )
                
                st.markdown("---")
                
                if port_enhanced_subtab == "Risk Parity":
                    st.markdown("### Risk Parity Optimization")
                    st.info("Allocate capital so each asset contributes equally to portfolio risk.")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        target_vol = st.slider("Target Volatility", 0.05, 0.30, 0.15, 0.01, key="rp_target_vol")
                    with col2:
                        max_weight = st.slider("Max Weight per Asset", 0.20, 0.60, 0.40, 0.05, key="rp_max_weight")
                    
                    # Auto-run Risk Parity
                    with st.spinner("Optimizing..."):
                        rp_result = risk_parity_weights(
                            returns_df, 
                            target_risk=target_vol,
                            max_weight=max_weight
                        )
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Portfolio Volatility", f"{rp_result['portfolio_volatility']:.1%}")
                            st.metric("Diversification Ratio", f"{rp_result['diversification_ratio']:.2f}")
                        with col2:
                            st.metric("Leverage for Target", f"{rp_result['leverage_for_target']:.2f}x")
                        
                        # Weights comparison
                        st.markdown("#### Risk Parity Weights")
                        rp_df = pd.DataFrame({
                            'Asset': list(rp_result['weights'].keys()),
                            'Current Weight': [weights.get(t, 0) for t in rp_result['weights'].keys()],
                            'Risk Parity Weight': [v * 100 for v in rp_result['weights'].values()],
                            'Risk Contribution': [v * 100 for v in rp_result['risk_contributions'].values()]
                        })
                        st.dataframe(rp_df.style.format({
                            'Current Weight': '{:.1f}%',
                            'Risk Parity Weight': '{:.1f}%',
                            'Risk Contribution': '{:.1f}%'
                        }), use_container_width=True, hide_index=True)
                        
                        # Risk contribution chart
                        fig = risk_contribution_chart(
                            rp_result['risk_contributions'],
                            title="Risk Contribution by Asset"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                elif port_enhanced_subtab == "Black-Litterman":
                    st.markdown("### Black-Litterman Model")
                    st.info("Combine market equilibrium with your investment views.")
                    
                    st.markdown("#### Define Your Views")
                    st.caption("Express views on expected returns (e.g., 'AAPL will outperform by 5%')")
                    
                    # Simple view input
                    view_asset = st.selectbox("Asset with View", tickers, key="bl_view_asset")
                    view_return = st.slider("Expected Annual Return", -0.20, 0.50, 0.10, 0.01, key="bl_view_return")
                    
                    views = [{'asset': view_asset, 'return': view_return}]
                    
                    # Placeholder market caps (equal weight assumed)
                    market_caps = {t: 1e9 for t in tickers}
                    
                    # Auto-run Black-Litterman
                    with st.spinner("Optimizing with views..."):
                        bl_result = black_litterman_optimization(
                            returns_df, market_caps, views
                        )
                        
                        if 'error' not in bl_result:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Expected Return", f"{bl_result['portfolio_return']:.1%}")
                            with col2:
                                st.metric("Portfolio Volatility", f"{bl_result['portfolio_volatility']:.1%}")
                            
                            # Show adjusted returns
                            st.markdown("#### Posterior Expected Returns")
                            ret_df = pd.DataFrame({
                                'Asset': list(bl_result['expected_returns'].keys()),
                                'Equilibrium': [v*100 for v in bl_result['equilibrium_returns'].values()],
                                'Posterior': [v*100 for v in bl_result['expected_returns'].values()],
                                'Weight': [v*100 for v in bl_result['weights'].values()]
                            })
                            st.dataframe(ret_df.style.format({
                                'Equilibrium': '{:.1f}%',
                                'Posterior': '{:.1f}%',
                                'Weight': '{:.1f}%'
                            }), use_container_width=True, hide_index=True)
                        else:
                            st.error("Black-Litterman optimization failed")
                
                elif port_enhanced_subtab == "Transaction Costs":
                    st.markdown("### Transaction Cost Analysis")
                    st.info("Estimate costs of rebalancing to target weights.")
                    
                    portfolio_value = st.number_input("Portfolio Value ($)", 10000, 10000000, 100000, 10000, key="tc_portfolio_value")
                    
                    # Auto-run Transaction Cost Analysis
                    with st.spinner("Calculating costs..."):
                        # Use current weights vs equal weight as example
                        current_w = {t: weights[t]/100 for t in tickers}
                        target_w = {t: 1/len(tickers) for t in tickers}
                        
                        # Placeholder prices
                        prices_dict = {t: 100 for t in tickers}
                        
                        cost_result = calculate_rebalance_costs(
                            current_w, target_w, portfolio_value, prices_dict
                        )
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Cost", f"${cost_result['total_cost']:.2f}")
                        col2.metric("Cost (bps)", f"{cost_result['total_cost_bps']:.1f}")
                        col3.metric("Turnover", f"{cost_result['turnover_pct']:.1f}%")
                        
                        # Trade list
                        if cost_result.get('trades'):
                            st.markdown("#### Proposed Trades")
                            trades_df = pd.DataFrame(cost_result['trades'])
                            st.dataframe(trades_df, use_container_width=True, hide_index=True)
                
                elif port_enhanced_subtab == "Rebalancing":
                    st.markdown("### Rebalancing Analysis")
                    st.info("Check if portfolio drift exceeds rebalancing thresholds.")
                    
                    threshold = st.slider("Rebalancing Threshold", 0.01, 0.20, 0.05, 0.01, key="rebal_threshold_widget")
                    
                    # Simulate some drift
                    current_w = {t: weights[t]/100 for t in tickers}
                    target_w = {t: weights[t]/100 for t in tickers}  # Same for demo
                    
                    # Add some random drift for demo
                    import random
                    drifted_w = {t: w * (1 + random.uniform(-0.1, 0.1)) for t, w in current_w.items()}
                    total_d = sum(drifted_w.values())
                    drifted_w = {t: v/total_d for t, v in drifted_w.items()}
                    
                    rebal_result = threshold_rebalancing(drifted_w, target_w, threshold)
                    
                    if rebal_result['needs_rebalance']:
                        st.warning(f"Rebalancing RECOMMENDED - Max drift: {rebal_result['max_drift']:.1%}")
                    else:
                        st.success(f"No rebalancing needed - Max drift: {rebal_result['max_drift']:.1%}")
                    
                    # Show drifts
                    drift_df = pd.DataFrame({
                        'Asset': list(rebal_result['drifts'].keys()),
                        'Drift': [v*100 for v in rebal_result['drifts'].values()],
                        'Threshold': [threshold*100]*len(rebal_result['drifts'])
                    })
                    st.dataframe(drift_df.style.format({'Drift': '{:.2f}%', 'Threshold': '{:.1f}%'}),
                               use_container_width=True, hide_index=True)
                
                elif port_enhanced_subtab == "Risk Decomposition":
                    st.markdown("### Portfolio Risk Decomposition")
                    st.info("Understand how each asset contributes to total portfolio risk.")
                    
                    weights_dict = {t: weights[t]/100 for t in tickers}
                    decomp = portfolio_risk_decomposition(weights_dict, returns_df)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Portfolio Volatility", f"{decomp['portfolio_volatility']:.1%}")
                    with col2:
                        st.metric("Concentration (HHI)", f"{decomp['concentration']:.0f}")
                    
                    # Risk contribution chart
                    fig = risk_contribution_chart(
                        decomp['pct_contributions'],
                        title="Risk Contribution (%)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed breakdown
                    st.markdown("#### Detailed Breakdown")
                    decomp_df = pd.DataFrame({
                        'Asset': list(decomp['risk_contributions'].keys()),
                        'Risk Contribution': list(decomp['risk_contributions'].values()),
                        '% of Total': list(decomp['pct_contributions'].values()),
                        'Marginal Contribution': list(decomp['marginal_contributions'].values())
                    })
                    st.dataframe(decomp_df.style.format({
                        'Risk Contribution': '{:.4f}',
                        '% of Total': '{:.1f}%',
                        'Marginal Contribution': '{:.4f}'
                    }), use_container_width=True, hide_index=True)
        
        # TAB: DIGITAL TWIN (v4.3) - Always enabled
        with tabs[tab_idx]:
            tab_idx += 1
            if HAS_DIGITAL_TWIN:
                try:
                    render_digital_twin_tab(
                        returns=returns_df,
                        weights={t: w for t, w in zip(tickers, weights_array)},
                        initial_capital=100000
                    )
                except Exception as e:
                    st.error(f"Digital Twin error: {e}")
                    st.info("Check that the portfolio simulation engine is properly loaded.")
            else:
                st.info("Digital Twin requires the portfolio simulation module.")
        
        # TAB: WHAT-IF ANALYSIS (v4.3) - Always enabled
        with tabs[tab_idx]:
            tab_idx += 1
            if HAS_WHAT_IF:
                try:
                    render_what_if_tab(
                        returns=returns_df,
                        current_weights={t: w for t, w in zip(tickers, weights_array)},
                        portfolio_value=100000
                    )
                except Exception as e:
                    st.error(f"What-If Analysis error: {e}")
                    st.info("Check that the What-If analyzer is properly loaded.")
            else:
                st.info("What-If Analysis requires the What-If analysis module.")
        
        # TAB: CHARTS
        with tabs[tab_idx]:
            tab_idx += 1
            cum_rets = (1 + port_rets).cumprod()
            bench_cum = (1 + bench_rets.reindex(port_rets.index).fillna(0)).cumprod()
            
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=cum_rets.index, y=cum_rets.values, name='Portfolio',
                                      line=dict(color=COLORS['primary'], width=2)))
            fig1.add_trace(go.Scatter(x=bench_cum.index, y=bench_cum.values, name=benchmark, 
                                      line=dict(color=COLORS['gray'], width=2, dash='dash')))
            fig1.update_layout(title="Cumulative Returns: Portfolio vs Benchmark", height=350,
                              template='plotly_dark' if theme_dark else 'plotly_white')
            st.plotly_chart(fig1, use_container_width=True)
            
            fig3 = px.area(x=port_metrics['drawdown_series'].index, y=port_metrics['drawdown_series'].values,
                          title="Portfolio Drawdown", color_discrete_sequence=[COLORS['danger']])
            fig3.update_layout(template='plotly_dark' if theme_dark else 'plotly_white', height=300)
            st.plotly_chart(fig3, use_container_width=True)
            
            # Rolling volatility
            rolling_vol = port_rets.rolling(20).std() * np.sqrt(252)
            fig_vol = px.line(x=rolling_vol.index, y=rolling_vol.values * 100, 
                             title="20-Day Rolling Volatility (%)")
            fig_vol.update_layout(template='plotly_dark' if theme_dark else 'plotly_white', height=300)
            st.plotly_chart(fig_vol, use_container_width=True)
        
        # TAB: EXPORT
        with tabs[tab_idx]:
            tab_idx += 1
            st.subheader("Export Portfolio Data")
            port_var = portfolio_var(returns_df, weights_array, conf_level)
            port_cvar = cvar(port_rets, conf_level)
            
            summary_data = {
                'Portfolio': ['Current'],
                'Ann Return': [f"{port_metrics['ann_ret']:.2%}"],
                'Ann Vol': [f"{port_metrics['ann_vol']:.2%}"],
                'Sharpe': [f"{port_metrics['sharpe']:.2f}"],
                'Max DD': [f"{port_metrics['max_dd']:.2%}"],
                f"VaR {conf_level:.0%}": [f"{port_var:.2%}"],
                'CVaR': [f"{port_cvar:.2%}"]
            }
            summary_df = pd.DataFrame(summary_data)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.download_button("Download Summary", summary_df.to_csv(index=False), "portfolio_summary.csv", "text/csv")
            with col2:
                st.download_button("Download Correlation", corr_matrix.to_csv(), "correlation_matrix.csv", "text/csv")
            with col3:
                st.download_button("Download Returns", returns_df.to_csv(), "portfolio_returns.csv", "text/csv")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
v43_features = " | Sentiment Analysis | Digital Twin | What-If Scenarios" if (HAS_SENTIMENT_FEATURE or HAS_DIGITAL_TWIN or HAS_WHAT_IF) else ""
st.markdown(f"""
<div style='text-align: center; color: #8E8E93; font-size: 0.8rem;'>
    Stock Risk Model v4.3 Enterprise | Portfolio Analysis | Stress Testing | Factor Models | AI Risk<br>
    Options Analytics | Fundamentals | PDF Reports | Real-time Data | Risk Parity{v43_features}<br>
    Built with Streamlit, XGBoost, GARCH, Fama-French, NLP Sentiment | Local Analysis Only
</div>
""", unsafe_allow_html=True)

if auto_refresh:
    import time
    time.sleep(300)
    st.rerun()
