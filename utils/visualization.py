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
                        label="Play",
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
    return f'<a href="data:text/html;base64,{b64}" download="{filename}">Download Chart</a>'


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
    
    # Regime coloring with proper labels
    regime_colors = {
        'Bull': 'rgba(0, 200, 83, 0.2)',
        'Bear': 'rgba(255, 23, 68, 0.2)',
        'Sideways': 'rgba(33, 150, 243, 0.2)',
        0: 'rgba(0, 200, 83, 0.2)',
        1: 'rgba(255, 23, 68, 0.2)',
        2: 'rgba(33, 150, 243, 0.2)'
    }
    
    # Get proper regime names for legend
    regime_display_names = {
        'Bull': 'Bull Market',
        'Bear': 'Bear Market',
        'Sideways': 'Sideways',
        0: 'Regime 1',
        1: 'Regime 2',
        2: 'Regime 3'
    }
    
    # Add regime background - without annotation text to avoid the new_text issue
    unique_regimes = regimes.unique()
    legend_added = set()
    
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
            
            for j, (start, end) in enumerate(regime_periods):
                fig.add_vrect(
                    x0=start, x1=end,
                    fillcolor=regime_colors.get(regime, 'rgba(128,128,128,0.1)'),
                    line_width=0
                )
    
    # Add legend entries for regimes
    for regime in unique_regimes:
        display_name = regime_display_names.get(regime, str(regime))
        color = regime_colors.get(regime, 'rgba(128,128,128,0.3)')
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=15, color=color.replace('0.2', '0.8')),
            name=display_name,
            showlegend=True
        ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Date",
        yaxis_title="Price",
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return apply_dark_theme(fig)


# =============================================================================
# VISUALIZATION ENGINE CLASS
# =============================================================================

# =============================================================================
# ENHANCED 3D VOLATILITY SURFACES WITH SLICING
# =============================================================================

def enhanced_volatility_surface_3d(
    strikes: np.ndarray,
    expirations: np.ndarray,
    implied_vols: np.ndarray,
    spot_price: float = None,
    title: str = "Enhanced Implied Volatility Surface",
    show_slices: bool = True,
    slice_expiry: float = None
) -> go.Figure:
    """
    Create enhanced 3D implied volatility surface with interactive features.
    
    Features:
    - Interactive rotation, zoom, pan
    - Time horizon cross-sections
    - Volatility smile slices
    - ATM volatility highlighting
    - Contour projections
    
    Args:
        strikes: Array of strike prices
        expirations: Array of expiration times (in years)
        implied_vols: 2D array of implied volatilities
        spot_price: Current spot price
        title: Chart title
        show_slices: Show cross-section slices
        slice_expiry: Specific expiry for slice highlight
    
    Returns:
        Plotly figure
    """
    # Create meshgrid for surface
    K, T = np.meshgrid(strikes, expirations)
    
    fig = go.Figure()
    
    # Main surface
    fig.add_trace(go.Surface(
        x=K,
        y=T,
        z=implied_vols,
        colorscale='Viridis',
        opacity=0.9,
        colorbar=dict(
            title="IV",
            titleside="right",
            x=1.02
        ),
        contours=dict(
            x=dict(show=True, color='white', width=1),
            y=dict(show=True, color='white', width=1),
            z=dict(show=True, color='white', width=1)
        ),
        hovertemplate=(
            "Strike: %{x:.0f}<br>"
            "Expiry: %{y:.3f}y<br>"
            "IV: %{z:.1%}<extra></extra>"
        ),
        name='IV Surface'
    ))
    
    # Add ATM line if spot provided
    if spot_price is not None:
        # ATM volatilities along term structure
        atm_idx = np.argmin(np.abs(strikes - spot_price))
        atm_vols = implied_vols[:, atm_idx]
        
        fig.add_trace(go.Scatter3d(
            x=[spot_price] * len(expirations),
            y=expirations,
            z=atm_vols,
            mode='lines+markers',
            line=dict(color='#FF5722', width=6),
            marker=dict(size=4, color='#FF5722'),
            name='ATM Term Structure'
        ))
        
        # Add vertical plane at spot
        plane_y, plane_z = np.meshgrid(expirations, np.linspace(0, implied_vols.max() * 1.1, 10))
        plane_x = np.ones_like(plane_y) * spot_price
        
        fig.add_trace(go.Surface(
            x=plane_x,
            y=plane_y,
            z=plane_z,
            colorscale=[[0, 'rgba(255,87,34,0.1)'], [1, 'rgba(255,87,34,0.1)']],
            showscale=False,
            opacity=0.3,
            name='Spot Plane',
            showlegend=False
        ))
    
    # Add time slice
    if show_slices and slice_expiry is not None:
        exp_idx = np.argmin(np.abs(expirations - slice_expiry))
        slice_vols = implied_vols[exp_idx, :]
        
        fig.add_trace(go.Scatter3d(
            x=strikes,
            y=[expirations[exp_idx]] * len(strikes),
            z=slice_vols,
            mode='lines+markers',
            line=dict(color='#4CAF50', width=5),
            marker=dict(size=3, color='#4CAF50'),
            name=f'{expirations[exp_idx]:.2f}y Smile'
        ))
    
    # Camera angles for different views
    camera_views = {
        'default': dict(eye=dict(x=1.5, y=1.5, z=0.8)),
        'top': dict(eye=dict(x=0, y=0, z=2.5)),
        'front': dict(eye=dict(x=0, y=2.5, z=0.5)),
        'side': dict(eye=dict(x=2.5, y=0, z=0.5))
    }
    
    # Add view buttons
    buttons = []
    for view_name, camera in camera_views.items():
        buttons.append(dict(
            args=[{'scene.camera': camera}],
            label=view_name.title(),
            method='relayout'
        ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        scene=dict(
            xaxis_title="Strike",
            yaxis_title="Time to Expiry (Years)",
            zaxis_title="Implied Volatility",
            camera=camera_views['default'],
            bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                backgroundcolor="rgba(30,30,30,0.9)",
                gridcolor="rgba(100,100,100,0.5)",
                showbackground=True
            ),
            yaxis=dict(
                backgroundcolor="rgba(30,30,30,0.9)",
                gridcolor="rgba(100,100,100,0.5)",
                showbackground=True
            ),
            zaxis=dict(
                backgroundcolor="rgba(30,30,30,0.9)",
                gridcolor="rgba(100,100,100,0.5)",
                showbackground=True,
                tickformat='.0%'
            ),
            aspectratio=dict(x=1.2, y=1, z=0.7)
        ),
        updatemenus=[dict(
            type='buttons',
            showactive=True,
            y=0.8,
            x=1.15,
            buttons=buttons
        )],
        height=700,
        margin=dict(l=0, r=100, t=50, b=0)
    )
    
    return apply_dark_theme(fig)


def volatility_smile_cross_sections(
    strikes: np.ndarray,
    expirations: np.ndarray,
    implied_vols: np.ndarray,
    spot_price: float = None,
    title: str = "Volatility Smile Cross-Sections"
) -> go.Figure:
    """
    Create 2D chart showing volatility smiles at different expirations.
    
    Args:
        strikes: Array of strike prices
        expirations: Array of expiration times
        implied_vols: 2D array of implied volatilities
        spot_price: Current spot price
        title: Chart title
    
    Returns:
        Plotly figure with multiple smile curves
    """
    fig = go.Figure()
    
    # Color gradient for different expirations
    colors = px.colors.sequential.Viridis
    n_exp = len(expirations)
    
    for i, exp in enumerate(expirations):
        color_idx = int(i * (len(colors) - 1) / max(n_exp - 1, 1))
        
        # Convert to moneyness if spot provided
        if spot_price:
            x_axis = (strikes / spot_price - 1) * 100  # % moneyness
            x_title = "Moneyness (%)"
        else:
            x_axis = strikes
            x_title = "Strike"
        
        fig.add_trace(go.Scatter(
            x=x_axis,
            y=implied_vols[i, :] * 100,  # Convert to %
            mode='lines+markers',
            name=f'{exp:.2f}y',
            line=dict(color=colors[color_idx], width=2),
            marker=dict(size=4)
        ))
    
    # Add ATM reference line
    if spot_price:
        fig.add_vline(x=0, line_dash='dash', line_color='white', opacity=0.5)
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title=x_title if spot_price else 'Strike',
        yaxis_title="Implied Volatility (%)",
        legend=dict(
            title="Expiration",
            orientation='v',
            yanchor='top',
            y=1,
            xanchor='left',
            x=1.02
        ),
        height=450,
        hovermode='x unified'
    )
    
    return apply_dark_theme(fig)


def term_structure_chart(
    expirations: np.ndarray,
    atm_vols: np.ndarray,
    forward_vols: np.ndarray = None,
    title: str = "Volatility Term Structure"
) -> go.Figure:
    """
    Create term structure chart showing ATM volatility vs expiration.
    
    Args:
        expirations: Array of expiration times
        atm_vols: ATM implied volatilities
        forward_vols: Optional forward volatilities
        title: Chart title
    
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # ATM volatility curve
    fig.add_trace(go.Scatter(
        x=expirations,
        y=atm_vols * 100,
        mode='lines+markers',
        name='ATM IV',
        line=dict(color='#2196F3', width=3),
        marker=dict(size=8),
        fill='tozeroy',
        fillcolor='rgba(33,150,243,0.1)'
    ))
    
    # Forward volatility if provided
    if forward_vols is not None:
        fig.add_trace(go.Scatter(
            x=expirations[:-1],
            y=forward_vols * 100,
            mode='lines+markers',
            name='Forward Vol',
            line=dict(color='#FF9800', width=2, dash='dash'),
            marker=dict(size=6)
        ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Time to Expiry (Years)",
        yaxis_title="Volatility (%)",
        height=400,
        hovermode='x unified'
    )
    
    return apply_dark_theme(fig)


# =============================================================================
# EFFICIENT FRONTIER VISUALIZATION
# =============================================================================

def efficient_frontier_chart(
    expected_returns: np.ndarray,
    volatilities: np.ndarray,
    sharpe_ratios: np.ndarray = None,
    asset_names: List[str] = None,
    asset_returns: np.ndarray = None,
    asset_vols: np.ndarray = None,
    current_portfolio: Tuple[float, float] = None,
    optimal_portfolio: Tuple[float, float] = None,
    title: str = "Efficient Frontier"
) -> go.Figure:
    """
    Create efficient frontier visualization with asset points.
    
    Args:
        expected_returns: Array of portfolio returns on frontier
        volatilities: Array of portfolio volatilities on frontier
        sharpe_ratios: Optional Sharpe ratios for coloring
        asset_names: Individual asset names
        asset_returns: Individual asset returns
        asset_vols: Individual asset volatilities
        current_portfolio: Current portfolio (vol, return)
        optimal_portfolio: Optimal portfolio (vol, return)
        title: Chart title
    
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Efficient frontier line
    if sharpe_ratios is not None:
        # Color by Sharpe ratio
        fig.add_trace(go.Scatter(
            x=volatilities * 100,
            y=expected_returns * 100,
            mode='lines+markers',
            marker=dict(
                size=8,
                color=sharpe_ratios,
                colorscale='RdYlGn',
                colorbar=dict(title="Sharpe"),
                showscale=True
            ),
            line=dict(width=3, color='rgba(100,100,100,0.3)'),
            name='Efficient Frontier',
            hovertemplate=(
                "Volatility: %{x:.1f}%<br>"
                "Return: %{y:.1f}%<br>"
                "Sharpe: %{marker.color:.2f}<extra></extra>"
            )
        ))
    else:
        fig.add_trace(go.Scatter(
            x=volatilities * 100,
            y=expected_returns * 100,
            mode='lines',
            line=dict(width=3, color='#4CAF50'),
            name='Efficient Frontier',
            hovertemplate="Vol: %{x:.1f}% | Return: %{y:.1f}%<extra></extra>"
        ))
    
    # Individual assets
    if asset_names and asset_returns is not None and asset_vols is not None:
        fig.add_trace(go.Scatter(
            x=asset_vols * 100,
            y=asset_returns * 100,
            mode='markers+text',
            marker=dict(size=12, color='#FF9800', symbol='diamond'),
            text=asset_names,
            textposition='top center',
            textfont=dict(size=10),
            name='Assets',
            hovertemplate="<b>%{text}</b><br>Vol: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>"
        ))
    
    # Current portfolio
    if current_portfolio:
        fig.add_trace(go.Scatter(
            x=[current_portfolio[0] * 100],
            y=[current_portfolio[1] * 100],
            mode='markers',
            marker=dict(size=15, color='#F44336', symbol='circle'),
            name='Current Portfolio',
            hovertemplate="Current<br>Vol: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>"
        ))
    
    # Optimal portfolio
    if optimal_portfolio:
        fig.add_trace(go.Scatter(
            x=[optimal_portfolio[0] * 100],
            y=[optimal_portfolio[1] * 100],
            mode='markers',
            marker=dict(size=15, color='#4CAF50', symbol='star'),
            name='Optimal Portfolio',
            hovertemplate="Optimal<br>Vol: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>"
        ))
        
        # Draw line from current to optimal
        if current_portfolio:
            fig.add_trace(go.Scatter(
                x=[current_portfolio[0] * 100, optimal_portfolio[0] * 100],
                y=[current_portfolio[1] * 100, optimal_portfolio[1] * 100],
                mode='lines',
                line=dict(color='white', width=1, dash='dot'),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # Capital Market Line (if we have optimal and risk-free info)
    if optimal_portfolio:
        rf_rate = 0.05  # Assume 5% risk-free
        sharpe = (optimal_portfolio[1] - rf_rate) / optimal_portfolio[0]
        
        # Extend CML
        max_vol = max(volatilities) * 1.2
        cml_vols = np.linspace(0, max_vol, 100)
        cml_returns = rf_rate + sharpe * cml_vols
        
        fig.add_trace(go.Scatter(
            x=cml_vols * 100,
            y=cml_returns * 100,
            mode='lines',
            line=dict(color='#2196F3', width=2, dash='dash'),
            name='Capital Market Line',
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Volatility (%)",
        yaxis_title="Expected Return (%)",
        height=550,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        hovermode='closest'
    )
    
    return apply_dark_theme(fig)


def interactive_weight_slider_chart(
    assets: List[str],
    current_weights: Dict[str, float],
    returns: pd.DataFrame,
    n_frontier_points: int = 50
) -> Tuple[go.Figure, Dict]:
    """
    Create efficient frontier with weight visualization for interactive adjustment.
    
    Note: This returns the figure and data needed for Streamlit sliders.
    The actual interactivity is handled in the Streamlit layer.
    
    Args:
        assets: List of asset names
        current_weights: Current portfolio weights
        returns: Historical returns DataFrame
        n_frontier_points: Number of points on frontier
    
    Returns:
        Tuple of (figure, frontier_data dict)
    """
    # Calculate frontier
    from scipy.optimize import minimize
    
    mean_returns = returns.mean().values * 252
    cov_matrix = returns.cov().values * 252
    n_assets = len(assets)
    
    def portfolio_stats(weights):
        ret = np.dot(weights, mean_returns)
        vol = np.sqrt(weights.T @ cov_matrix @ weights)
        return ret, vol
    
    def min_volatility(target_return):
        def volatility(w):
            return np.sqrt(w.T @ cov_matrix @ w)
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: np.dot(w, mean_returns) - target_return}
        ]
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        result = minimize(
            volatility,
            np.ones(n_assets) / n_assets,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        return result.x if result.success else None
    
    # Generate frontier points
    target_returns = np.linspace(mean_returns.min(), mean_returns.max(), n_frontier_points)
    frontier_vols = []
    frontier_rets = []
    frontier_weights = []
    
    for target in target_returns:
        weights = min_volatility(target)
        if weights is not None:
            ret, vol = portfolio_stats(weights)
            frontier_rets.append(ret)
            frontier_vols.append(vol)
            frontier_weights.append(dict(zip(assets, weights)))
    
    # Calculate current portfolio position
    current_w = np.array([current_weights.get(a, 0) for a in assets])
    current_ret, current_vol = portfolio_stats(current_w)
    
    # Create chart
    sharpe_ratios = np.array(frontier_rets) / np.array(frontier_vols)
    
    fig = efficient_frontier_chart(
        expected_returns=np.array(frontier_rets),
        volatilities=np.array(frontier_vols),
        sharpe_ratios=sharpe_ratios,
        asset_names=assets,
        asset_returns=mean_returns,
        asset_vols=np.sqrt(np.diag(cov_matrix)),
        current_portfolio=(current_vol, current_ret)
    )
    
    frontier_data = {
        'returns': frontier_rets,
        'volatilities': frontier_vols,
        'weights': frontier_weights,
        'sharpe_ratios': sharpe_ratios.tolist(),
        'current': {'return': current_ret, 'volatility': current_vol}
    }
    
    return fig, frontier_data


def weight_allocation_chart(
    weights: Dict[str, float],
    title: str = "Portfolio Allocation"
) -> go.Figure:
    """
    Create horizontal bar chart showing portfolio weights.
    
    Args:
        weights: Portfolio weights
        title: Chart title
    
    Returns:
        Plotly figure
    """
    # Sort by weight
    sorted_weights = dict(sorted(weights.items(), key=lambda x: x[1], reverse=True))
    
    assets = list(sorted_weights.keys())
    values = [v * 100 for v in sorted_weights.values()]
    
    # Colors based on size
    colors = ['#4CAF50' if v > 15 else '#2196F3' if v > 5 else '#9E9E9E' for v in values]
    
    fig = go.Figure(go.Bar(
        x=values,
        y=assets,
        orientation='h',
        marker_color=colors,
        text=[f'{v:.1f}%' for v in values],
        textposition='outside',
        hovertemplate="<b>%{y}</b><br>Weight: %{x:.1f}%<extra></extra>"
    ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Weight (%)",
        yaxis_title="Asset",
        height=max(300, len(assets) * 35),
        showlegend=False,
        xaxis=dict(range=[0, max(values) * 1.2])
    )
    
    return apply_dark_theme(fig)


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
