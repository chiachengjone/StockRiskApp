"""
Visualization Enhancement Module

Advanced visualization features:
- Interactive correlation heatmaps
- 3D volatility surfaces
- Animated price charts
- Downloadable chart exports
- Performance attribution charts
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Union
import base64
from io import BytesIO


# =============================================================================
# THEME CONFIGURATION
# =============================================================================

DARK_THEME = {
    'paper_bgcolor': 'rgba(0,0,0,0)',
    'plot_bgcolor': 'rgba(0,0,0,0)',
    'font_color': '#FAFAFA',
    'grid_color': 'rgba(128,128,128,0.2)',
    'colorscale': 'RdYlGn',
    'positive_color': '#00C853',
    'negative_color': '#FF1744',
    'neutral_color': '#2196F3',
    'warning_color': '#FFC107'
}


def apply_dark_theme(fig: go.Figure) -> go.Figure:
    """Apply consistent dark theme to plotly figure."""
    fig.update_layout(
        paper_bgcolor=DARK_THEME['paper_bgcolor'],
        plot_bgcolor=DARK_THEME['plot_bgcolor'],
        font=dict(color=DARK_THEME['font_color']),
        xaxis=dict(gridcolor=DARK_THEME['grid_color']),
        yaxis=dict(gridcolor=DARK_THEME['grid_color'])
    )
    return fig


# =============================================================================
# INTERACTIVE CORRELATION HEATMAP
# =============================================================================

def interactive_correlation_heatmap(
    returns: pd.DataFrame,
    title: str = "Correlation Matrix",
    show_values: bool = True,
    cluster: bool = True
) -> go.Figure:
    """
    Create interactive correlation heatmap with clustering.
    
    Features:
    - Hover tooltips with exact values
    - Optional hierarchical clustering
    - Color scale from -1 to 1
    - Click to zoom
    
    Args:
        returns: DataFrame of asset returns
        title: Chart title
        show_values: Show correlation values on cells
        cluster: Apply hierarchical clustering to sort assets
    
    Returns:
        Plotly figure
    """
    corr = returns.corr()
    
    # Optionally cluster
    if cluster and len(corr) > 2:
        try:
            from scipy.cluster.hierarchy import linkage, leaves_list
            from scipy.spatial.distance import squareform
            
            dist = 1 - corr.abs()
            condensed = squareform(dist.values, checks=False)
            link = linkage(condensed, method='average')
            order = leaves_list(link)
            
            # Reorder
            cols = [corr.columns[i] for i in order]
            corr = corr.loc[cols, cols]
        except:
            pass  # Use original order
    
    # Create heatmap
    text = corr.round(2).astype(str) if show_values else None
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale='RdBu_r',
        zmid=0,
        zmin=-1,
        zmax=1,
        text=text,
        texttemplate="%{text}" if show_values else None,
        textfont={"size": 10},
        hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>",
        colorbar=dict(
            title="Correlation",
            titleside="right"
        )
    ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis=dict(tickangle=45),
        height=500 + len(corr) * 20,
        width=500 + len(corr) * 20
    )
    
    return apply_dark_theme(fig)


def rolling_correlation_chart(
    returns: pd.DataFrame,
    asset1: str,
    asset2: str,
    windows: List[int] = None,
    title: str = None
) -> go.Figure:
    """
    Chart rolling correlation between two assets over multiple windows.
    
    Args:
        returns: DataFrame of returns
        asset1: First asset
        asset2: Second asset
        windows: List of rolling window sizes
        title: Chart title
    
    Returns:
        Plotly figure
    """
    if windows is None:
        windows = [20, 60, 120]
    
    fig = go.Figure()
    
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
    
    for i, window in enumerate(windows):
        rolling_corr = returns[asset1].rolling(window).corr(returns[asset2])
        
        fig.add_trace(go.Scatter(
            x=rolling_corr.index,
            y=rolling_corr.values,
            mode='lines',
            name=f'{window}-day',
            line=dict(color=colors[i % len(colors)], width=2)
        ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    # Add +/- 0.5 reference lines
    fig.add_hline(y=0.5, line_dash="dot", line_color="green", opacity=0.3)
    fig.add_hline(y=-0.5, line_dash="dot", line_color="red", opacity=0.3)
    
    title = title or f"Rolling Correlation: {asset1} vs {asset2}"
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Date",
        yaxis_title="Correlation",
        yaxis=dict(range=[-1.1, 1.1]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=400
    )
    
    return apply_dark_theme(fig)


# =============================================================================
# 3D VOLATILITY SURFACE
# =============================================================================

def volatility_surface_3d(
    strikes: np.ndarray,
    expirations: np.ndarray,
    implied_vols: np.ndarray,
    spot_price: float = None,
    title: str = "Implied Volatility Surface"
) -> go.Figure:
    """
    Create 3D implied volatility surface plot.
    
    Features:
    - Interactive rotation and zoom
    - Color gradient based on IV level
    - Spot price marker
    
    Args:
        strikes: Array of strike prices
        expirations: Array of expiration times (in years or days)
        implied_vols: 2D array of implied volatilities
        spot_price: Current spot price (for reference)
        title: Chart title
    
    Returns:
        Plotly figure
    """
    fig = go.Figure(data=[go.Surface(
        x=strikes,
        y=expirations,
        z=implied_vols,
        colorscale='Viridis',
        colorbar=dict(title="IV", titleside="right"),
        hovertemplate=(
            "Strike: %{x:.0f}<br>"
            "Expiry: %{y:.2f}<br>"
            "IV: %{z:.1%}<extra></extra>"
        )
    )])
    
    # Add spot price plane if provided
    if spot_price is not None:
        fig.add_trace(go.Scatter3d(
            x=[spot_price] * len(expirations),
            y=expirations,
            z=[implied_vols.max()] * len(expirations),
            mode='lines',
            line=dict(color='white', width=4, dash='dash'),
            name='Spot Price'
        ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        scene=dict(
            xaxis_title="Strike",
            yaxis_title="Time to Expiry",
            zaxis_title="Implied Volatility",
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=0.8)
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        height=600,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    return apply_dark_theme(fig)


def create_sample_vol_surface(
    spot: float = 100,
    atm_vol: float = 0.20,
    skew: float = 0.02,
    term_slope: float = 0.01
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create sample volatility surface data for demonstration.
    
    Args:
        spot: Spot price
        atm_vol: At-the-money volatility
        skew: Volatility skew (OTM puts higher)
        term_slope: Term structure slope
    
    Returns:
        Tuple of (strikes, expirations, implied_vols)
    """
    strikes = np.linspace(spot * 0.7, spot * 1.3, 20)
    expirations = np.array([0.08, 0.17, 0.25, 0.5, 1.0, 2.0])  # Years
    
    # Create surface with smile and term structure
    K, T = np.meshgrid(strikes, expirations)
    moneyness = np.log(K / spot)
    
    # Volatility smile/skew
    smile = atm_vol + skew * (moneyness ** 2) - 0.5 * skew * moneyness
    
    # Term structure
    term = term_slope * np.sqrt(T)
    
    implied_vols = smile + term
    
    return strikes, expirations, implied_vols


# =============================================================================
# ANIMATED PRICE CHARTS
# =============================================================================

def animated_price_chart(
    prices: pd.DataFrame,
    title: str = "Price Animation",
    frame_duration: int = 100
) -> go.Figure:
    """
    Create animated price chart showing price evolution.
    
    Features:
    - Play/pause button
    - Slider for manual control
    - Multiple assets support
    
    Args:
        prices: DataFrame of prices (columns = assets)
        title: Chart title
        frame_duration: Duration per frame in milliseconds
    
    Returns:
        Plotly figure with animation
    """
    # Normalize prices to start at 100
    normalized = (prices / prices.iloc[0]) * 100
    
    # Create frames
    frames = []
    for i in range(len(normalized)):
        frame_data = []
        for col in normalized.columns:
            frame_data.append(go.Scatter(
                x=normalized.index[:i+1],
                y=normalized[col].values[:i+1],
                mode='lines',
                name=col
            ))
        frames.append(go.Frame(data=frame_data, name=str(i)))
    
    # Initial data (full lines, but we'll animate)
    fig = go.Figure(
        data=[go.Scatter(
            x=[normalized.index[0]],
            y=[100],
            mode='lines',
            name=col
        ) for col in normalized.columns],
        frames=frames
    )
    
    # Animation controls
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Date",
        yaxis_title="Normalized Price (100 = Start)",
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=1.15,
                x=0.5,
                xanchor="center",
                buttons=[
                    dict(
                        label="▶ Play",
                        method="animate",
                        args=[None, {
                            "frame": {"duration": frame_duration},
                            "fromcurrent": True,
                            "transition": {"duration": 0}
                        }]
                    ),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[[None], {
                            "frame": {"duration": 0},
                            "mode": "immediate",
                            "transition": {"duration": 0}
                        }]
                    )
                ]
            )
        ],
        sliders=[{
            "active": 0,
            "steps": [
                {"args": [[f.name], {"frame": {"duration": 0}, "mode": "immediate"}],
                 "label": str(i), "method": "animate"}
                for i, f in enumerate(frames)
            ],
            "x": 0.1,
            "len": 0.8,
            "xanchor": "left",
            "y": -0.05,
            "yanchor": "top",
            "currentvalue": {
                "prefix": "Day: ",
                "visible": True,
                "xanchor": "center"
            }
        }],
        height=500
    )
    
    return apply_dark_theme(fig)


def cumulative_returns_chart(
    returns: pd.DataFrame,
    title: str = "Cumulative Returns",
    show_drawdown: bool = True
) -> go.Figure:
    """
    Create cumulative returns chart with optional drawdown overlay.
    
    Args:
        returns: DataFrame of returns
        title: Chart title
        show_drawdown: Show drawdown as shaded area
    
    Returns:
        Plotly figure
    """
    cum_returns = (1 + returns).cumprod() - 1
    
    if show_drawdown:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           vertical_spacing=0.05,
                           row_heights=[0.7, 0.3])
    else:
        fig = go.Figure()
        
    colors = px.colors.qualitative.Set2
    
    for i, col in enumerate(cum_returns.columns):
        fig.add_trace(go.Scatter(
            x=cum_returns.index,
            y=cum_returns[col] * 100,
            mode='lines',
            name=col,
            line=dict(color=colors[i % len(colors)], width=2)
        ), row=1 if show_drawdown else None, col=1 if show_drawdown else None)
    
    if show_drawdown:
        # Calculate portfolio drawdown (equal weight)
        portfolio_cum = cum_returns.mean(axis=1)
        rolling_max = portfolio_cum.expanding().max()
        drawdown = (portfolio_cum - rolling_max) / (1 + rolling_max) * 100
        
        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            fill='tozeroy',
            fillcolor='rgba(255, 23, 68, 0.3)',
            line=dict(color='#FF1744', width=1),
            name='Drawdown',
            showlegend=True
        ), row=2, col=1)
        
        fig.update_yaxes(title_text="Cumulative Return (%)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    else:
        fig.update_yaxes(title_text="Cumulative Return (%)")
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Date",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=500 if not show_drawdown else 600,
        hovermode='x unified'
    )
    
    return apply_dark_theme(fig)


# =============================================================================
# RISK VISUALIZATION
# =============================================================================

def var_cone_chart(
    returns: pd.Series,
    horizon: int = 30,
    confidence_levels: List[float] = None,
    simulations: int = 1000,
    title: str = "VaR Cone Projection"
) -> go.Figure:
    """
    Create VaR cone projection showing potential future price paths.
    
    Args:
        returns: Historical returns
        horizon: Forecast horizon in days
        confidence_levels: VaR confidence levels
        simulations: Number of Monte Carlo simulations
        title: Chart title
    
    Returns:
        Plotly figure
    """
    if confidence_levels is None:
        confidence_levels = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    
    mu = returns.mean()
    sigma = returns.std()
    
    # Generate paths
    np.random.seed(42)
    paths = np.zeros((simulations, horizon + 1))
    paths[:, 0] = 100  # Start at 100
    
    for t in range(1, horizon + 1):
        shocks = np.random.normal(mu, sigma, simulations)
        paths[:, t] = paths[:, t-1] * (1 + shocks)
    
    # Calculate percentiles
    percentiles = np.percentile(paths, [q * 100 for q in confidence_levels], axis=0)
    
    fig = go.Figure()
    
    # Add cone bands
    colors = ['rgba(255,23,68,0.3)', 'rgba(255,152,0,0.3)', 'rgba(76,175,80,0.3)',
              'rgba(76,175,80,0.3)', 'rgba(255,152,0,0.3)', 'rgba(255,23,68,0.3)']
    
    x_range = list(range(horizon + 1))
    
    # Plot symmetric bands
    for i in range(len(confidence_levels) // 2):
        upper = percentiles[-(i+1)]
        lower = percentiles[i]
        
        fig.add_trace(go.Scatter(
            x=x_range + x_range[::-1],
            y=list(upper) + list(lower[::-1]),
            fill='toself',
            fillcolor=colors[i],
            line=dict(width=0),
            name=f'{int((1-confidence_levels[i])*100)}% CI',
            hoverinfo='skip'
        ))
    
    # Median path
    median_idx = len(confidence_levels) // 2
    fig.add_trace(go.Scatter(
        x=x_range,
        y=percentiles[median_idx],
        mode='lines',
        name='Median',
        line=dict(color='white', width=2)
    ))
    
    # Some sample paths
    for i in range(5):
        fig.add_trace(go.Scatter(
            x=x_range,
            y=paths[i * (simulations // 5)],
            mode='lines',
            line=dict(width=0.5, color='gray'),
            opacity=0.5,
            showlegend=False,
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Days",
        yaxis_title="Value",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    
    return apply_dark_theme(fig)


def risk_contribution_chart(
    contributions: Dict[str, float],
    title: str = "Risk Contribution by Asset"
) -> go.Figure:
    """
    Create pie/donut chart showing risk contributions.
    
    Args:
        contributions: Dict of asset -> risk contribution
        title: Chart title
    
    Returns:
        Plotly figure
    """
    assets = list(contributions.keys())
    values = [abs(v) for v in contributions.values()]
    
    colors = px.colors.qualitative.Set2
    
    fig = go.Figure(data=[go.Pie(
        labels=assets,
        values=values,
        hole=0.4,
        marker=dict(colors=colors[:len(assets)]),
        textinfo='label+percent',
        textposition='outside',
        hovertemplate="<b>%{label}</b><br>Risk Contribution: %{value:.2%}<extra></extra>"
    )])
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        annotations=[dict(text='Risk', x=0.5, y=0.5, font_size=20, showarrow=False)],
        height=450,
        showlegend=False
    )
    
    return apply_dark_theme(fig)


def var_comparison_chart(
    var_estimates: Dict[str, float],
    title: str = "VaR Comparison by Method"
) -> go.Figure:
    """
    Create bar chart comparing VaR estimates across methods.
    
    Args:
        var_estimates: Dict of method -> VaR estimate
        title: Chart title
    
    Returns:
        Plotly figure
    """
    methods = list(var_estimates.keys())
    values = [abs(v) * 100 for v in var_estimates.values()]  # Convert to percent
    
    # Color based on value
    mean_val = np.mean(values)
    colors = ['#00C853' if v < mean_val else '#FF1744' for v in values]
    
    fig = go.Figure(data=[go.Bar(
        x=methods,
        y=values,
        marker_color=colors,
        text=[f'{v:.2f}%' for v in values],
        textposition='outside',
        hovertemplate="<b>%{x}</b><br>VaR: %{y:.2f}%<extra></extra>"
    )])
    
    # Add mean line
    fig.add_hline(y=mean_val, line_dash="dash", line_color="white", 
                  annotation_text=f"Mean: {mean_val:.2f}%")
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Method",
        yaxis_title="VaR (%)",
        height=400,
        showlegend=False
    )
    
    return apply_dark_theme(fig)


# =============================================================================
# PERFORMANCE ATTRIBUTION
# =============================================================================

def performance_attribution_chart(
    returns: pd.DataFrame,
    weights: Dict[str, float],
    benchmark_returns: pd.Series = None,
    title: str = "Performance Attribution"
) -> go.Figure:
    """
    Create performance attribution waterfall chart.
    
    Args:
        returns: Asset returns DataFrame
        weights: Portfolio weights
        benchmark_returns: Optional benchmark returns
        title: Chart title
    
    Returns:
        Plotly figure
    """
    # Calculate contributions
    contributions = {}
    for col in returns.columns:
        w = weights.get(col, 0)
        asset_return = (1 + returns[col]).prod() - 1
        contributions[col] = w * asset_return
    
    portfolio_return = sum(contributions.values())
    
    # Sort by contribution
    sorted_contrib = dict(sorted(contributions.items(), key=lambda x: x[1], reverse=True))
    
    # Waterfall data
    assets = list(sorted_contrib.keys())
    values = list(sorted_contrib.values())
    
    # Add total
    assets.append('Total')
    
    fig = go.Figure(go.Waterfall(
        name="Attribution",
        orientation="v",
        x=assets,
        y=values + [None],
        measure=['relative'] * len(values) + ['total'],
        text=[f'{v*100:.2f}%' for v in values] + [f'{portfolio_return*100:.2f}%'],
        textposition="outside",
        connector={"line": {"color": "rgba(128,128,128,0.5)"}},
        increasing={"marker": {"color": "#00C853"}},
        decreasing={"marker": {"color": "#FF1744"}},
        totals={"marker": {"color": "#2196F3"}}
    ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Asset",
        yaxis_title="Contribution to Return (%)",
        yaxis=dict(tickformat='.1%'),
        height=450,
        showlegend=False
    )
    
    return apply_dark_theme(fig)


def rolling_performance_chart(
    returns: pd.Series,
    windows: List[int] = None,
    metric: str = 'sharpe',
    title: str = None
) -> go.Figure:
    """
    Chart rolling performance metrics over time.
    
    Args:
        returns: Returns series
        windows: Rolling window sizes
        metric: 'sharpe', 'return', 'volatility', 'sortino'
        title: Chart title
    
    Returns:
        Plotly figure
    """
    if windows is None:
        windows = [21, 63, 126, 252]
    
    fig = go.Figure()
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
    
    for i, window in enumerate(windows):
        if metric == 'sharpe':
            rolling = returns.rolling(window).apply(
                lambda x: np.mean(x) / np.std(x) * np.sqrt(252) if np.std(x) > 0 else 0
            )
            y_label = "Sharpe Ratio"
        elif metric == 'return':
            rolling = returns.rolling(window).apply(lambda x: (1 + x).prod() - 1) * 100
            y_label = "Return (%)"
        elif metric == 'volatility':
            rolling = returns.rolling(window).std() * np.sqrt(252) * 100
            y_label = "Volatility (%)"
        elif metric == 'sortino':
            def sortino(r):
                downside = r[r < 0]
                if len(downside) == 0 or np.std(downside) == 0:
                    return 0
                return np.mean(r) / np.std(downside) * np.sqrt(252)
            rolling = returns.rolling(window).apply(sortino)
            y_label = "Sortino Ratio"
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        fig.add_trace(go.Scatter(
            x=rolling.index,
            y=rolling.values,
            mode='lines',
            name=f'{window}-day',
            line=dict(color=colors[i % len(colors)], width=2)
        ))
    
    # Add zero line for ratios
    if metric in ['sharpe', 'sortino']:
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_hline(y=1, line_dash="dot", line_color="green", opacity=0.3,
                     annotation_text="Good")
        fig.add_hline(y=2, line_dash="dot", line_color="green", opacity=0.3,
                     annotation_text="Excellent")
    
    title = title or f"Rolling {metric.title()}"
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Date",
        yaxis_title=y_label,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=400
    )
    
    return apply_dark_theme(fig)


# =============================================================================
# CHART EXPORT UTILITIES
# =============================================================================

def make_chart_downloadable(fig: go.Figure, filename: str = "chart") -> str:
    """
    Convert plotly figure to downloadable HTML.
    
    Args:
        fig: Plotly figure
        filename: Base filename
    
    Returns:
        HTML string that can be downloaded
    """
    # Full HTML with plotly.js included
    html = fig.to_html(include_plotlyjs='cdn', full_html=True)
    return html


def figure_to_png_base64(fig: go.Figure, width: int = 1200, height: int = 700) -> str:
    """
    Convert plotly figure to base64-encoded PNG.
    
    Args:
        fig: Plotly figure
        width: Image width
        height: Image height
    
    Returns:
        Base64-encoded PNG string
    """
    try:
        img_bytes = fig.to_image(format='png', width=width, height=height)
        b64 = base64.b64encode(img_bytes).decode()
        return f"data:image/png;base64,{b64}"
    except Exception as e:
        return f"Error: {str(e)}"


def get_download_link(html_content: str, filename: str = "chart.html") -> str:
    """
    Create download link for HTML content.
    
    Args:
        html_content: HTML string
        filename: Download filename
    
    Returns:
        HTML anchor tag for download
    """
    b64 = base64.b64encode(html_content.encode()).decode()
    return f'<a href="data:text/html;base64,{b64}" download="{filename}">📥 Download Chart</a>'


# =============================================================================
# FACTOR VISUALIZATION
# =============================================================================

def factor_exposure_chart(
    exposures: Dict[str, float],
    title: str = "Factor Exposures (Betas)"
) -> go.Figure:
    """
    Create horizontal bar chart showing factor exposures.
    
    Args:
        exposures: Dict of factor -> exposure value
        title: Chart title
    
    Returns:
        Plotly figure
    """
    factors = list(exposures.keys())
    values = list(exposures.values())
    
    colors = ['#00C853' if v > 0 else '#FF1744' for v in values]
    
    fig = go.Figure(data=[go.Bar(
        x=values,
        y=factors,
        orientation='h',
        marker_color=colors,
        text=[f'{v:.2f}' for v in values],
        textposition='outside',
        hovertemplate="<b>%{y}</b><br>Exposure: %{x:.3f}<extra></extra>"
    )])
    
    fig.add_vline(x=0, line_color="white", line_width=1)
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Beta (Exposure)",
        yaxis_title="Factor",
        height=max(300, len(factors) * 40),
        showlegend=False
    )
    
    return apply_dark_theme(fig)


def regime_chart(
    prices: pd.Series,
    regimes: pd.Series,
    title: str = "Price Chart with Regime Detection"
) -> go.Figure:
    """
    Create price chart with regime overlay.
    
    Args:
        prices: Price series
        regimes: Regime labels series
        title: Chart title
    
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Price line
    fig.add_trace(go.Scatter(
        x=prices.index,
        y=prices.values,
        mode='lines',
        name='Price',
        line=dict(color='white', width=2)
    ))
    
    # Regime coloring
    regime_colors = {
        'Bull': 'rgba(0, 200, 83, 0.2)',
        'Bear': 'rgba(255, 23, 68, 0.2)',
        'Sideways': 'rgba(33, 150, 243, 0.2)',
        0: 'rgba(0, 200, 83, 0.2)',
        1: 'rgba(255, 23, 68, 0.2)',
        2: 'rgba(33, 150, 243, 0.2)'
    }
    
    # Add regime background
    unique_regimes = regimes.unique()
    for regime in unique_regimes:
        mask = regimes == regime
        if mask.any():
            regime_periods = []
            start = None
            for i, (date, is_regime) in enumerate(mask.items()):
                if is_regime and start is None:
                    start = date
                elif not is_regime and start is not None:
                    regime_periods.append((start, mask.index[i-1]))
                    start = None
            if start is not None:
                regime_periods.append((start, mask.index[-1]))
            
            for start, end in regime_periods:
                fig.add_vrect(
                    x0=start, x1=end,
                    fillcolor=regime_colors.get(regime, 'rgba(128,128,128,0.1)'),
                    line_width=0,
                    annotation_text=str(regime) if start == regime_periods[0][0] else None,
                    annotation_position="top left"
                )
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Date",
        yaxis_title="Price",
        height=500
    )
    
    return apply_dark_theme(fig)


# =============================================================================
# VISUALIZATION ENGINE CLASS
# =============================================================================

class VisualizationEngine:
    """
    Unified interface for all visualization features.
    """
    
    def __init__(self, dark_mode: bool = True):
        """
        Initialize visualization engine.
        
        Args:
            dark_mode: Use dark theme
        """
        self.dark_mode = dark_mode
        self.theme = DARK_THEME if dark_mode else {}
    
    def correlation_heatmap(self, returns: pd.DataFrame, **kwargs) -> go.Figure:
        """Create correlation heatmap."""
        return interactive_correlation_heatmap(returns, **kwargs)
    
    def vol_surface(self, strikes, expirations, ivs, **kwargs) -> go.Figure:
        """Create 3D volatility surface."""
        return volatility_surface_3d(strikes, expirations, ivs, **kwargs)
    
    def animated_prices(self, prices: pd.DataFrame, **kwargs) -> go.Figure:
        """Create animated price chart."""
        return animated_price_chart(prices, **kwargs)
    
    def cumulative_returns(self, returns: pd.DataFrame, **kwargs) -> go.Figure:
        """Create cumulative returns chart."""
        return cumulative_returns_chart(returns, **kwargs)
    
    def var_cone(self, returns: pd.Series, **kwargs) -> go.Figure:
        """Create VaR cone projection."""
        return var_cone_chart(returns, **kwargs)
    
    def risk_contribution(self, contributions: Dict, **kwargs) -> go.Figure:
        """Create risk contribution chart."""
        return risk_contribution_chart(contributions, **kwargs)
    
    def var_comparison(self, estimates: Dict, **kwargs) -> go.Figure:
        """Create VaR comparison chart."""
        return var_comparison_chart(estimates, **kwargs)
    
    def performance_attribution(self, returns, weights, **kwargs) -> go.Figure:
        """Create performance attribution chart."""
        return performance_attribution_chart(returns, weights, **kwargs)
    
    def rolling_performance(self, returns: pd.Series, **kwargs) -> go.Figure:
        """Create rolling performance chart."""
        return rolling_performance_chart(returns, **kwargs)
    
    def factor_exposure(self, exposures: Dict, **kwargs) -> go.Figure:
        """Create factor exposure chart."""
        return factor_exposure_chart(exposures, **kwargs)
    
    def regime_detection(self, prices: pd.Series, regimes: pd.Series, **kwargs) -> go.Figure:
        """Create regime detection chart."""
        return regime_chart(prices, regimes, **kwargs)
    
    def export_chart(self, fig: go.Figure, format: str = 'html') -> str:
        """
        Export chart to specified format.
        
        Args:
            fig: Plotly figure
            format: 'html', 'png', or 'json'
        
        Returns:
            Exported content
        """
        if format == 'html':
            return make_chart_downloadable(fig)
        elif format == 'png':
            return figure_to_png_base64(fig)
        elif format == 'json':
            return fig.to_json()
        else:
            raise ValueError(f"Unknown format: {format}")
