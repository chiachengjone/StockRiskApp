"""
STOCK RISK MODELLING APP
========================
Streamlit + yfinance + GARCH + EVT + Portfolio Mode + Stress Testing
+ Fama-French Factors + Kelly Criterion + ESG + XGBoost AI VaR
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
    backtest_strategy, STRESS_SCENARIOS
)
from factors import FactorAnalyzer
from ml_predictor import MLPredictor

# ============================================================================
# PAGE CONFIG & STYLING
# ============================================================================
st.set_page_config(
    page_title="Risk Model", 
    page_icon="", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apple-inspired minimal CSS
st.markdown("""
<style>
    /* Clean, minimal typography */
    .stApp {
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Subtle headers */
    h1, h2, h3 {
        font-weight: 500 !important;
        letter-spacing: -0.02em;
    }
    
    /* Clean metric cards */
    [data-testid="stMetricValue"] {
        font-size: 1.4rem !important;
        font-weight: 500 !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        opacity: 0.7;
    }
    
    /* Subtle dividers */
    hr {
        border: none;
        height: 1px;
        background: rgba(128, 128, 128, 0.2);
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
    }
    
    .stTabs [data-baseweb="tab"] {
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    /* Clean dataframes */
    .stDataFrame {
        border-radius: 8px;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: rgba(248, 248, 248, 0.95);
    }
    
    /* Remove excessive padding */
    .block-container {
        padding-top: 2rem;
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
    
    rf_rate = st.number_input("Risk-Free Rate (%)", 0.0, 10.0, 4.5, 0.1) / 100
    theme_dark = st.toggle("Dark Charts", value=True)
    auto_refresh = st.toggle("Auto-Refresh (5min)", value=False)
    
    st.divider()
    st.caption("v3.0")

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

mode = st.radio("Analysis Mode", ["Single Stock", "Portfolio"], horizontal=True, label_visibility="collapsed")

# ============================================================================
# SINGLE STOCK MODE
# ============================================================================
if mode == "Single Stock":
    col1, col2, col3 = st.columns(3)
    
    with col1:
        asset_type = st.selectbox("Asset Type", ["Stocks", "ETFs", "Crypto", "Custom"])
        if asset_type == "Stocks":
            stock_selection = st.selectbox("Select Stock", options=list(POPULAR_STOCKS.keys()))
            ticker = POPULAR_STOCKS[stock_selection] if stock_selection != "Custom..." else st.text_input("Custom Ticker", "AAPL").upper()
        elif asset_type == "ETFs":
            etf_selection = st.selectbox("Select ETF", options=list(ETF_ASSETS.keys()))
            ticker = ETF_ASSETS[etf_selection]
        elif asset_type == "Crypto":
            crypto_selection = st.selectbox("Select Crypto", options=list(CRYPTO_ASSETS.keys()))
            ticker = CRYPTO_ASSETS[crypto_selection]
        else:
            ticker = st.text_input("Enter Ticker", "AAPL").upper()
    
    with col2:
        bench_selection = st.selectbox("Benchmark", options=list(POPULAR_BENCHMARKS.keys()))
        benchmark = POPULAR_BENCHMARKS[bench_selection] if bench_selection != "Custom..." else st.text_input("Custom Benchmark", "^GSPC").upper()
    
    with col3:
        days_back = st.slider("Days Back", 100, 2000, 756)

    col4, col5, col6 = st.columns(3)
    with col4:
        conf_level = st.slider("Confidence Level", 0.90, 0.999, 0.95, 0.01)
    with col5:
        var_horizon = st.slider("VaR Horizon (days)", 1, 30, 1)
    with col6:
        advanced_mode = st.toggle("Advanced Analysis")

    if 'single_analyzed' not in st.session_state:
        st.session_state.single_analyzed = False
    
    if st.button("Analyze Risk", type="primary", use_container_width=True):
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
        st.session_state.single_conf_level = conf_level
        st.session_state.single_var_horizon = var_horizon
        st.session_state.single_advanced = advanced_mode
        st.session_state.single_rf_rate = rf_rate
        st.session_state.single_theme_dark = theme_dark
        st.session_state.single_days_back = days_back
    
    if st.session_state.single_analyzed:
        ticker = st.session_state.single_ticker
        benchmark = st.session_state.single_benchmark
        rets = st.session_state.single_rets
        bench_rets = st.session_state.single_bench_rets
        prices = st.session_state.single_prices
        metrics = st.session_state.single_metrics
        beta = st.session_state.single_beta
        alpha = st.session_state.single_alpha
        info = st.session_state.single_info
        conf_level = st.session_state.single_conf_level
        var_horizon = st.session_state.single_var_horizon
        advanced_mode = st.session_state.single_advanced
        rf_rate = st.session_state.single_rf_rate
        theme_dark = st.session_state.single_theme_dark
        days_back = st.session_state.single_days_back
        
        p_var = parametric_var(rets, var_horizon, conf_level)
        p_var_t = parametric_var(rets, var_horizon, conf_level, 't')
        h_var = historical_var(rets, var_horizon, conf_level)
        cv = cvar(rets, conf_level)
        
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "Overview", "VaR Analysis", "Monte Carlo", "Stress Test", 
            "Advanced", "Factors", "AI Risk", "Export"
        ])
        
        # TAB 1: OVERVIEW
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
        
        # TAB 3: MONTE CARLO
        with tab3:
            st.subheader("Monte Carlo Simulation")
            
            col1, col2 = st.columns([1, 3])
            with col1:
                mc_sims = st.selectbox("Simulations", [1000, 5000, 10000, 50000], index=2)
                mc_horizon = st.selectbox("Horizon (days)", [5, 10, 20, 60, 120], index=2)
            
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
        
        # TAB 5: ADVANCED
        with tab5:
            st.subheader("Advanced Risk Models")
            
            if advanced_mode:
                st.markdown("### GARCH(1,1) Volatility Model")
                garch_fitted, garch_cond_vol, garch_forecast = fit_garch(rets)
                
                if garch_fitted is not None:
                    col1, col2, col3 = st.columns(3)
                    current_garch_vol = float(garch_cond_vol.iloc[-1]) * np.sqrt(252)
                    col1.metric("Current GARCH Vol", f"{current_garch_vol:.1%}")
                    col2.metric("10-Day Forecast", f"{garch_forecast[-1]:.1%}")
                    vol_regime = "High" if current_garch_vol > metrics['ann_vol'] * 1.2 else "Normal"
                    col3.metric("Vol Regime", vol_regime)
                    
                    # GARCH volatility chart
                    fig_garch = go.Figure()
                    fig_garch.add_trace(go.Scatter(
                        x=garch_cond_vol.index, 
                        y=garch_cond_vol.values * np.sqrt(252) * 100,
                        name='GARCH Volatility',
                        line=dict(color=COLORS['primary'], width=1.5)
                    ))
                    fig_garch.update_layout(
                        title="GARCH Conditional Volatility (Annualized)",
                        yaxis_title="Volatility (%)",
                        template='plotly_dark' if theme_dark else 'plotly_white',
                        height=300
                    )
                    st.plotly_chart(fig_garch, use_container_width=True)
                
                st.markdown("---")
                st.markdown("### EVT Tail Risk (Extreme Value Theory)")
                evt_results = evt_tail_risk(rets)
                if "error" not in evt_results:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Tail Threshold", f"{evt_results['threshold']:.2%}")
                    col2.metric("GPD Shape Parameter", f"{evt_results['shape']:.3f}", help="Shape > 0 indicates heavy tails")
                    col3.metric("EVT 99% VaR", f"{evt_results['evt_var']:.2%}")
                else:
                    st.info(evt_results["error"])
                
                st.markdown("---")
                st.markdown("### Backtest: Buy & Hold vs Benchmark")
                bt_results = backtest_strategy(rets, bench_rets)
                
                col1, col2, col3 = st.columns(3)
                col1.metric(f"{ticker} Total Return", f"{bt_results['strategy_return']:.1%}")
                col2.metric(f"{benchmark} Return", f"{bt_results['benchmark_return']:.1%}")
                col3.metric("Excess Return", f"{bt_results['excess_return']:.1%}")
                
                # Cumulative returns chart
                fig_bt = go.Figure()
                fig_bt.add_trace(go.Scatter(x=bt_results['strategy_cum'].index, y=bt_results['strategy_cum'].values,
                                           name=ticker, line=dict(color=COLORS['primary'], width=2)))
                fig_bt.add_trace(go.Scatter(x=bt_results['benchmark_cum'].index, y=bt_results['benchmark_cum'].values,
                                           name=benchmark, line=dict(color=COLORS['gray'], width=2, dash='dash')))
                fig_bt.update_layout(title="Cumulative Returns Comparison", 
                                    template='plotly_dark' if theme_dark else 'plotly_white', height=350)
                st.plotly_chart(fig_bt, use_container_width=True)
            else:
                st.info("Enable 'Advanced Analysis' toggle to access GARCH, EVT, and Backtesting")
        
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
        
        # TAB 7: AI RISK
        with tab7:
            st.subheader("AI-Powered Risk Prediction")
            
            ml = MLPredictor()
            
            with st.spinner("Training XGBoost model..."):
                ml_results = ml.train_predict(rets, prices)
            
            if 'error' not in ml_results:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("AI Predicted VaR", f"{ml_results['predicted_var']:.2%}")
                col2.metric("Model R-Squared", f"{ml_results['r2_score']:.3f}")
                col3.metric("Training R-Squared", f"{ml_results['r2_train']:.3f}")
                col4.metric("Mean Absolute Error", f"{ml_results['mae']:.4f}")
                
                st.markdown("---")
                
                st.markdown("### VaR Method Comparison")
                comparison_df = pd.DataFrame({
                    'Method': ['XGBoost ML', 'GARCH', 'Historical', 'Parametric'],
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
            else:
                st.error(ml_results.get('error', 'ML prediction failed'))
                st.info("Ensure XGBoost is installed: pip install xgboost")
        
        # TAB 8: EXPORT
        with tab8:
            st.subheader("Export Data")
            
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

# ============================================================================
# PORTFOLIO MODE
# ============================================================================
else:
    st.markdown("### Portfolio Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        tickers_input = st.text_input("Enter Tickers (comma-separated)", "AAPL, MSFT, GOOGL, AMZN, NVDA")
        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    
    with col2:
        days_back = st.slider("Days", 100, 2000, 756, key="port_days")
    
    if tickers:
        st.markdown("#### Portfolio Weights")
        weights = {}
        cols = st.columns(len(tickers))
        for i, ticker in enumerate(tickers):
            with cols[i]:
                weights[ticker] = st.number_input(f"{ticker} %", 0, 100, int(100/len(tickers)), key=f"w_{ticker}")
        
        total_weight = sum(weights.values())
        if total_weight != 100:
            st.warning(f"Weights sum to {total_weight}%. Should be 100%.")
        
        weights_array = np.array(list(weights.values())) / 100
    
    col1, col2 = st.columns(2)
    with col1:
        bench_selection = st.selectbox("Benchmark", list(POPULAR_BENCHMARKS.keys()), key="port_bench")
        benchmark = POPULAR_BENCHMARKS[bench_selection] if bench_selection != "Custom..." else "^GSPC"
    with col2:
        conf_level = st.slider("Confidence Level", 0.90, 0.999, 0.95, 0.01, key="port_conf")
    
    if 'port_analyzed' not in st.session_state:
        st.session_state.port_analyzed = False
    
    if st.button("Analyze Portfolio", type="primary", use_container_width=True) and tickers:
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
        st.session_state.port_conf_level = conf_level
        st.session_state.port_rf_rate = rf_rate
        st.session_state.port_theme_dark = theme_dark
        st.session_state.port_prices_df = prices_df
        st.session_state.port_weights = weights
    
    if st.session_state.port_analyzed:
        tickers = st.session_state.port_tickers
        weights_array = st.session_state.port_weights_array
        returns_df = st.session_state.port_returns_df
        port_rets = st.session_state.port_rets
        port_metrics = st.session_state.port_metrics
        bench_rets = st.session_state.port_bench_rets
        betas = st.session_state.port_betas
        conf_level = st.session_state.port_conf_level
        rf_rate = st.session_state.port_rf_rate
        theme_dark = st.session_state.port_theme_dark
        prices_df = st.session_state.port_prices_df
        weights = st.session_state.port_weights
        
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "Summary", "Monte Carlo", "Correlation", "Stress Test", "Optimization", "Charts", "Export"
        ])
        
        with tab1:
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
        
        # TAB 2: MONTE CARLO (NEW FOR PORTFOLIO)
        with tab2:
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
        
        with tab3:
            st.subheader("Correlation Matrix")
            corr_matrix = returns_df.corr()
            fig_corr = px.imshow(corr_matrix, text_auto='.2f', aspect='auto',
                                title="Asset Correlation Heatmap",
                                color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
            fig_corr.update_layout(height=450, template='plotly_dark' if theme_dark else 'plotly_white')
            st.plotly_chart(fig_corr, use_container_width=True)
        
        with tab4:
            st.subheader("Stress Testing")
            
            col1, col2 = st.columns(2)
            with col1:
                scenario = st.selectbox("Select Scenario", list(STRESS_SCENARIOS.keys()))
                scenario_params = STRESS_SCENARIOS[scenario]
                st.caption(scenario_params['description'])
            
            with col2:
                custom_shock = st.slider("Market Shock (%)", -80, 0, int(scenario_params['market_shock']*100)) / 100
                custom_vol = st.slider("Vol Multiplier", 1.0, 5.0, scenario_params['vol_multiplier'], 0.5)
            
            use_custom = st.checkbox("Use Custom Parameters")
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
        
        with tab5:
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
        
        with tab6:
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
        
        with tab7:
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
st.markdown("""
<div style='text-align: center; color: #8E8E93; font-size: 0.8rem;'>
    Stock Risk Model v3.0 | Portfolio Analysis | Stress Testing | Factor Models | AI Risk<br>
    Built with Streamlit, XGBoost, GARCH, Fama-French | Local Analysis Only
</div>
""", unsafe_allow_html=True)

if auto_refresh:
    import time
    time.sleep(300)
    st.rerun()
