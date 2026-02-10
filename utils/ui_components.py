"""
UI Components - Enhanced User Interface Helpers
================================================
Progressive disclosure, visual hierarchy, insights, and responsive layouts.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class MetricLevel(Enum):
    """Classification for metric status."""
    GOOD = "good"
    WARNING = "warning"
    DANGER = "danger"
    NEUTRAL = "neutral"


@dataclass
class MetricThresholds:
    """Thresholds for metric classification."""
    sharpe_good: float = 1.0
    sharpe_warning: float = 0.5
    volatility_warning: float = 0.25
    volatility_danger: float = 0.40
    drawdown_warning: float = -0.10
    drawdown_danger: float = -0.20
    var_warning: float = 0.03
    var_danger: float = 0.05
    beta_warning: float = 1.3
    beta_danger: float = 1.8


# Color palette for UI
UI_COLORS = {
    'good': '#34C759',
    'warning': '#FF9500',
    'danger': '#FF3B30',
    'neutral': '#8E8E93',
    'primary': '#007AFF',
    'secondary': '#5856D6',
    'background': '#161b22',
    'card': '#21262d',
    'border': '#30363d',
    'text': '#c9d1d9',
    'text_muted': '#8b949e'
}


def init_ui_state():
    """Initialize UI state variables."""
    defaults = {
        'show_advanced': False,
        'compact_mode': False,
        'show_insights': True,
        'expanded_sections': set()
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def classify_metric(metric_name: str, value: float, thresholds: MetricThresholds = None) -> MetricLevel:
    """
    Classify a metric value as good, warning, danger, or neutral.
    
    Args:
        metric_name: Name of the metric (e.g., 'sharpe', 'volatility')
        value: The metric value
        thresholds: Optional custom thresholds
    
    Returns:
        MetricLevel classification
    """
    if thresholds is None:
        thresholds = MetricThresholds()
    
    if metric_name == 'sharpe':
        if value >= thresholds.sharpe_good:
            return MetricLevel.GOOD
        elif value >= thresholds.sharpe_warning:
            return MetricLevel.WARNING
        else:
            return MetricLevel.DANGER
    
    elif metric_name == 'volatility':
        if value <= thresholds.volatility_warning:
            return MetricLevel.GOOD
        elif value <= thresholds.volatility_danger:
            return MetricLevel.WARNING
        else:
            return MetricLevel.DANGER
    
    elif metric_name == 'max_drawdown':
        if value >= thresholds.drawdown_warning:
            return MetricLevel.GOOD
        elif value >= thresholds.drawdown_danger:
            return MetricLevel.WARNING
        else:
            return MetricLevel.DANGER
    
    elif metric_name == 'var':
        if abs(value) <= thresholds.var_warning:
            return MetricLevel.GOOD
        elif abs(value) <= thresholds.var_danger:
            return MetricLevel.WARNING
        else:
            return MetricLevel.DANGER
    
    elif metric_name == 'beta':
        if abs(value) <= 1.0:
            return MetricLevel.GOOD
        elif abs(value) <= thresholds.beta_warning:
            return MetricLevel.WARNING
        else:
            return MetricLevel.DANGER
    
    return MetricLevel.NEUTRAL


def get_metric_color(level: MetricLevel) -> str:
    """Get color for metric level."""
    return UI_COLORS.get(level.value, UI_COLORS['neutral'])


def render_metric_card(
    label: str,
    value: str,
    level: MetricLevel = MetricLevel.NEUTRAL,
    delta: str = None,
    delta_direction: str = None,
    tooltip: str = None,
    size: str = "normal"  # "normal", "large", "small"
) -> None:
    """
    Render an enhanced metric card with color coding.
    
    Args:
        label: Metric label
        value: Display value
        level: MetricLevel for color coding
        delta: Optional delta value
        delta_direction: "up" or "down"
        tooltip: Optional tooltip text
        size: Card size ("normal", "large", "small")
    """
    color = get_metric_color(level)
    
    # Size configurations
    sizes = {
        "large": {"value_size": "2rem", "label_size": "0.8rem", "padding": "1.2rem"},
        "normal": {"value_size": "1.3rem", "label_size": "0.7rem", "padding": "0.8rem"},
        "small": {"value_size": "1rem", "label_size": "0.6rem", "padding": "0.5rem"}
    }
    
    config = sizes.get(size, sizes["normal"])
    
    # Build card HTML
    delta_html = ""
    if delta:
        delta_color = UI_COLORS['good'] if delta_direction == "up" else UI_COLORS['danger']
        arrow = "↑" if delta_direction == "up" else "↓"
        delta_html = f'<div style="font-size: 0.75rem; color: {delta_color};">{arrow} {delta}</div>'
    
    tooltip_attr = f'title="{tooltip}"' if tooltip else ""
    
    html = f"""
    <div {tooltip_attr} style="
        background: {UI_COLORS['card']};
        border: 1px solid {UI_COLORS['border']};
        border-left: 3px solid {color};
        border-radius: 8px;
        padding: {config['padding']};
        text-align: center;
        margin-bottom: 0.5rem;
    ">
        <div style="font-size: {config['label_size']}; color: {UI_COLORS['text_muted']}; 
                    text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.3rem;">
            {label}
        </div>
        <div style="font-size: {config['value_size']}; font-weight: 600; color: {color};
                    font-family: 'SF Mono', monospace;">
            {value}
        </div>
        {delta_html}
    </div>
    """
    
    st.markdown(html, unsafe_allow_html=True)


def render_primary_metrics(metrics: Dict, var_value: float = None, benchmark_name: str = "SPY"):
    """
    Render primary metrics in large format with visual hierarchy.
    
    Args:
        metrics: Dictionary containing ann_ret, ann_vol, sharpe, max_dd
        var_value: Optional VaR value
        benchmark_name: Name of benchmark for comparison
    """
    st.markdown("#### Key Risk Metrics")
    
    # Primary metrics - large cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        sharpe_level = classify_metric('sharpe', metrics.get('sharpe', 0))
        render_metric_card(
            "Sharpe Ratio",
            f"{metrics.get('sharpe', 0):.2f}",
            level=sharpe_level,
            tooltip="Risk-adjusted return. >1 is good, >2 is excellent",
            size="large"
        )
    
    with col2:
        var_display = var_value or metrics.get('var', 0)
        var_level = classify_metric('var', var_display)
        render_metric_card(
            "VaR (95%)",
            f"{abs(var_display):.2%}",
            level=var_level,
            tooltip="Maximum expected daily loss at 95% confidence",
            size="large"
        )
    
    with col3:
        dd_level = classify_metric('max_drawdown', metrics.get('max_dd', 0))
        render_metric_card(
            "Max Drawdown",
            f"{metrics.get('max_dd', 0):.1%}",
            level=dd_level,
            tooltip="Largest peak-to-trough decline",
            size="large"
        )
    
    with col4:
        vol_level = classify_metric('volatility', metrics.get('ann_vol', 0))
        render_metric_card(
            "Volatility",
            f"{metrics.get('ann_vol', 0):.1%}",
            level=vol_level,
            tooltip="Annualized standard deviation of returns",
            size="large"
        )


def render_secondary_metrics(metrics: Dict, beta: float = None, alpha: float = None):
    """
    Render secondary metrics in smaller format.
    
    Args:
        metrics: Dictionary containing sortino, calmar, skew
        beta: Portfolio beta
        alpha: Portfolio alpha
    """
    with st.expander("📊 Additional Metrics", expanded=False):
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            render_metric_card(
                "Sortino",
                f"{metrics.get('sortino', 0):.2f}",
                level=classify_metric('sharpe', metrics.get('sortino', 0)),
                size="small"
            )
        
        with col2:
            render_metric_card(
                "Calmar",
                f"{metrics.get('calmar', 0):.2f}",
                size="small"
            )
        
        with col3:
            render_metric_card(
                "Ann. Return",
                f"{metrics.get('ann_ret', 0):.1%}",
                level=MetricLevel.GOOD if metrics.get('ann_ret', 0) > 0 else MetricLevel.DANGER,
                size="small"
            )
        
        with col4:
            if beta is not None:
                beta_level = classify_metric('beta', beta)
                render_metric_card("Beta", f"{beta:.2f}", level=beta_level, size="small")
        
        with col5:
            render_metric_card(
                "Skewness",
                f"{metrics.get('skew', 0):.2f}",
                level=MetricLevel.GOOD if metrics.get('skew', 0) > 0 else MetricLevel.WARNING,
                size="small"
            )


def render_insight_box(
    message: str,
    insight_type: str = "info",  # "info", "warning", "success", "danger", "tip"
    icon: str = None
) -> None:
    """
    Render an insight/interpretation box.
    
    Args:
        message: The insight message
        insight_type: Type of insight for styling
        icon: Optional custom icon
    """
    icons = {
        "info": "💡",
        "warning": "⚠️",
        "success": "✅",
        "danger": "🚨",
        "tip": "💎"
    }
    
    colors = {
        "info": (UI_COLORS['primary'], f"{UI_COLORS['primary']}15"),
        "warning": (UI_COLORS['warning'], f"{UI_COLORS['warning']}15"),
        "success": (UI_COLORS['good'], f"{UI_COLORS['good']}15"),
        "danger": (UI_COLORS['danger'], f"{UI_COLORS['danger']}15"),
        "tip": (UI_COLORS['secondary'], f"{UI_COLORS['secondary']}15")
    }
    
    icon_char = icon or icons.get(insight_type, "💡")
    border_color, bg_color = colors.get(insight_type, colors["info"])
    
    html = f"""
    <div style="
        background: {bg_color};
        border-left: 3px solid {border_color};
        border-radius: 6px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.85rem;
        color: {UI_COLORS['text']};
    ">
        <span style="margin-right: 0.5rem;">{icon_char}</span>
        {message}
    </div>
    """
    
    st.markdown(html, unsafe_allow_html=True)


def generate_risk_insights(metrics: Dict, var_value: float, beta: float = None) -> List[Tuple[str, str]]:
    """
    Generate contextual insights based on metrics.
    
    Args:
        metrics: Dictionary of risk metrics
        var_value: VaR value
        beta: Portfolio beta
    
    Returns:
        List of (message, insight_type) tuples
    """
    insights = []
    
    # Sharpe ratio insight
    sharpe = metrics.get('sharpe', 0)
    if sharpe >= 2.0:
        insights.append(("Excellent risk-adjusted returns. Sharpe ratio above 2.0 indicates strong performance relative to risk.", "success"))
    elif sharpe >= 1.0:
        insights.append(("Good risk-adjusted returns. Portfolio is generating positive alpha.", "success"))
    elif sharpe >= 0.5:
        insights.append(("Moderate risk-adjusted returns. Consider reducing volatility or improving returns.", "info"))
    else:
        insights.append(("Poor risk-adjusted returns. The risk taken isn't being adequately compensated.", "warning"))
    
    # Volatility insight
    vol = metrics.get('ann_vol', 0)
    if vol > 0.40:
        insights.append((f"High volatility ({vol:.0%}). Consider hedging or diversification to reduce risk.", "danger"))
    elif vol > 0.25:
        insights.append((f"Elevated volatility ({vol:.0%}). Monitor position sizing carefully.", "warning"))
    
    # Drawdown insight
    max_dd = metrics.get('max_dd', 0)
    if max_dd < -0.30:
        insights.append((f"Severe historical drawdown ({max_dd:.0%}). Recovery may take significant time.", "danger"))
    elif max_dd < -0.20:
        insights.append((f"Significant drawdown history ({max_dd:.0%}). Ensure proper risk management.", "warning"))
    
    # VaR insight
    if abs(var_value) > 0.05:
        insights.append((f"VaR suggests potential daily loss exceeding 5%. Size positions accordingly.", "warning"))
    
    # Beta insight
    if beta is not None:
        if beta > 1.5:
            insights.append((f"High beta ({beta:.2f}) amplifies market moves. Expect larger swings than benchmark.", "warning"))
        elif beta < 0.5:
            insights.append((f"Low beta ({beta:.2f}) provides defensive characteristics.", "info"))
    
    return insights


def render_chart_with_context(
    chart_func,
    title: str,
    interpretation: str,
    chart_args: Dict = None,
    show_interpretation: bool = True
):
    """
    Render a chart with contextual interpretation.
    
    Args:
        chart_func: Function that returns a plotly figure
        title: Chart title
        interpretation: Text explaining what the chart means
        chart_args: Arguments to pass to chart function
        show_interpretation: Whether to show interpretation
    """
    if show_interpretation and st.session_state.get('show_insights', True):
        render_insight_box(interpretation, "info")
    
    fig = chart_func(**(chart_args or {}))
    st.plotly_chart(fig, use_container_width=True)


def render_compact_tabs(tab_names: List[str], group_advanced: bool = True) -> Tuple:
    """
    Render tabs with progressive disclosure for advanced features.
    
    Args:
        tab_names: List of tab names
        group_advanced: Whether to group advanced tabs under expandable section
    
    Returns:
        Tuple of tab objects
    """
    if not group_advanced or len(tab_names) <= 6:
        return st.tabs(tab_names)
    
    # Split into core and advanced
    core_tabs = tab_names[:5]
    advanced_tabs = tab_names[5:]
    
    # Add "More..." to core tabs
    core_tabs.append("More...")
    tabs = st.tabs(core_tabs)
    
    # In the "More..." tab, show advanced options
    with tabs[-1]:
        advanced_tab_selection = st.selectbox(
            "Select Analysis",
            advanced_tabs,
            key="advanced_tab_select"
        )
        st.session_state['selected_advanced_tab'] = advanced_tab_selection
    
    return tabs


def render_alert_badge(count: int, alert_type: str = "warning") -> None:
    """
    Render an alert badge indicator.
    
    Args:
        count: Number of alerts
        alert_type: Type of alert for styling
    """
    if count == 0:
        return
    
    color = UI_COLORS['danger'] if alert_type == "danger" else UI_COLORS['warning']
    
    html = f"""
    <span style="
        background: {color};
        color: white;
        font-size: 0.7rem;
        padding: 2px 6px;
        border-radius: 10px;
        font-weight: bold;
        margin-left: 0.3rem;
    ">{count}</span>
    """
    
    st.markdown(html, unsafe_allow_html=True)


def toggle_section(section_name: str, default_expanded: bool = False) -> bool:
    """
    Create a toggle for section visibility.
    
    Args:
        section_name: Unique name for the section
        default_expanded: Whether section is expanded by default
    
    Returns:
        Whether section should be shown
    """
    key = f"section_{section_name}"
    if key not in st.session_state:
        st.session_state[key] = default_expanded
    
    return st.session_state[key]


def render_responsive_columns(items: List[Dict], max_cols: int = 4) -> None:
    """
    Render items in responsive columns.
    
    Args:
        items: List of dicts with 'label', 'value', 'level' keys
        max_cols: Maximum number of columns
    """
    compact = st.session_state.get('compact_mode', False)
    num_cols = 2 if compact else min(max_cols, len(items))
    
    for i in range(0, len(items), num_cols):
        cols = st.columns(num_cols)
        for j, col in enumerate(cols):
            if i + j < len(items):
                item = items[i + j]
                with col:
                    render_metric_card(
                        item.get('label', ''),
                        item.get('value', ''),
                        item.get('level', MetricLevel.NEUTRAL),
                        size="small" if compact else "normal"
                    )


def render_help_modal():
    """Render a help modal with documentation."""
    with st.expander("❓ Quick Reference", expanded=False):
        st.markdown("""
        ### Key Metrics Explained
        
        | Metric | Description | Good Value |
        |--------|-------------|------------|
        | **Sharpe** | Risk-adjusted return | > 1.0 |
        | **VaR 95%** | Max expected daily loss | < 3% |
        | **Volatility** | Return variability | < 25% |
        | **Max Drawdown** | Worst peak-to-trough | > -20% |
        | **Beta** | Market sensitivity | 0.8 - 1.2 |
        
        ### Quick Actions
        - **Single Stock**: Analyze one asset's risk profile
        - **Portfolio**: Multi-asset risk assessment
        - **Stress Test**: Historical scenario analysis
        - **Optimization**: Find efficient allocations
        """)


def render_quick_actions(ticker: str = None, tickers: List[str] = None):
    """
    Render quick action buttons.
    
    Args:
        ticker: Single stock ticker
        tickers: List of portfolio tickers
    """
    st.markdown("#### Quick Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📊 Full Report", use_container_width=True, help="Download comprehensive PDF report"):
            st.session_state['generate_report'] = True
    
    with col2:
        if st.button("🔔 Set Alerts", use_container_width=True, help="Configure risk alerts"):
            st.session_state['show_alerts'] = True
    
    with col3:
        if st.button("📈 Compare", use_container_width=True, help="Compare with benchmarks"):
            st.session_state['show_comparison'] = True
    
    with col4:
        if st.button("⚙️ Settings", use_container_width=True, help="Adjust analysis parameters"):
            st.session_state['show_settings'] = True


def render_rebalance_recommendations(
    current_weights: Dict[str, float],
    target_weights: Dict[str, float],
    portfolio_value: float = 100000,
    transaction_cost_pct: float = 0.001
) -> pd.DataFrame:
    """
    Generate and display rebalancing trade recommendations.
    
    Args:
        current_weights: Current portfolio weights {ticker: weight}
        target_weights: Target portfolio weights {ticker: weight}
        portfolio_value: Total portfolio value
        transaction_cost_pct: Transaction cost as percentage
    
    Returns:
        DataFrame with trade recommendations
    """
    trades = []
    
    all_tickers = set(current_weights.keys()) | set(target_weights.keys())
    
    for ticker in all_tickers:
        current = current_weights.get(ticker, 0)
        target = target_weights.get(ticker, 0)
        diff = target - current
        
        if abs(diff) > 0.001:  # Threshold of 0.1%
            trade_value = diff * portfolio_value
            shares = abs(trade_value) / 100  # Placeholder price
            cost = abs(trade_value) * transaction_cost_pct
            
            trades.append({
                'Ticker': ticker,
                'Action': 'BUY' if diff > 0 else 'SELL',
                'Current %': f"{current:.1%}",
                'Target %': f"{target:.1%}",
                'Trade Value': f"${abs(trade_value):,.0f}",
                'Est. Cost': f"${cost:.2f}"
            })
    
    if trades:
        df = pd.DataFrame(trades)
        
        st.markdown("#### 📋 Rebalancing Trade List")
        
        # Summary
        total_trades = len(trades)
        total_value = sum(abs(float(t['Trade Value'].replace('$', '').replace(',', ''))) for t in trades)
        total_cost = sum(float(t['Est. Cost'].replace('$', '')) for t in trades)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Trades Required", total_trades)
        col2.metric("Total Trade Value", f"${total_value:,.0f}")
        col3.metric("Est. Transaction Cost", f"${total_cost:.2f}")
        
        # Trade table with color coding
        st.dataframe(
            df.style.apply(
                lambda x: ['background-color: #1a3d1a' if v == 'BUY' else 'background-color: #3d1a1a' if v == 'SELL' else '' for v in x],
                subset=['Action']
            ),
            use_container_width=True,
            hide_index=True
        )
        
        return df
    else:
        st.info("Portfolio is already at target weights. No rebalancing needed.")
        return pd.DataFrame()


def render_performance_attribution_summary(
    portfolio_return: float,
    benchmark_return: float,
    asset_contributions: Dict[str, float]
) -> None:
    """
    Render a visual performance attribution summary.
    
    Args:
        portfolio_return: Total portfolio return
        benchmark_return: Benchmark return
        asset_contributions: Dict of {ticker: contribution}
    """
    st.markdown("#### Performance Attribution")
    
    excess_return = portfolio_return - benchmark_return
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        render_metric_card(
            "Portfolio Return",
            f"{portfolio_return:.2%}",
            level=MetricLevel.GOOD if portfolio_return > 0 else MetricLevel.DANGER,
            size="normal"
        )
    
    with col2:
        render_metric_card(
            "Benchmark Return",
            f"{benchmark_return:.2%}",
            size="normal"
        )
    
    with col3:
        render_metric_card(
            "Excess Return (Alpha)",
            f"{excess_return:+.2%}",
            level=MetricLevel.GOOD if excess_return > 0 else MetricLevel.DANGER,
            size="normal"
        )
    
    # Top contributors
    if asset_contributions:
        sorted_contrib = sorted(asset_contributions.items(), key=lambda x: x[1], reverse=True)
        
        st.markdown("##### Top Contributors")
        for ticker, contrib in sorted_contrib[:3]:
            pct = contrib * 100
            bar_width = min(abs(pct) * 10, 100)
            color = UI_COLORS['good'] if contrib > 0 else UI_COLORS['danger']
            
            st.markdown(f"""
            <div style="margin-bottom: 0.5rem;">
                <span style="font-weight: 500; width: 60px; display: inline-block;">{ticker}</span>
                <span style="color: {color}; font-family: monospace;">{contrib:+.2%}</span>
                <div style="height: 6px; background: {UI_COLORS['card']}; border-radius: 3px; margin-top: 2px;">
                    <div style="height: 100%; width: {bar_width}%; background: {color}; border-radius: 3px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Detractors
        if len(sorted_contrib) > 3:
            st.markdown("##### Top Detractors")
            for ticker, contrib in sorted_contrib[-3:]:
                if contrib < 0:
                    pct = contrib * 100
                    bar_width = min(abs(pct) * 10, 100)
                    
                    st.markdown(f"""
                    <div style="margin-bottom: 0.5rem;">
                        <span style="font-weight: 500; width: 60px; display: inline-block;">{ticker}</span>
                        <span style="color: {UI_COLORS['danger']}; font-family: monospace;">{contrib:+.2%}</span>
                        <div style="height: 6px; background: {UI_COLORS['card']}; border-radius: 3px; margin-top: 2px;">
                            <div style="height: 100%; width: {bar_width}%; background: {UI_COLORS['danger']}; border-radius: 3px;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)


# Export all components
__all__ = [
    'MetricLevel',
    'MetricThresholds',
    'UI_COLORS',
    'init_ui_state',
    'classify_metric',
    'get_metric_color',
    'render_metric_card',
    'render_primary_metrics',
    'render_secondary_metrics',
    'render_insight_box',
    'generate_risk_insights',
    'render_chart_with_context',
    'render_compact_tabs',
    'render_alert_badge',
    'toggle_section',
    'render_responsive_columns',
    'render_help_modal',
    'render_quick_actions',
    'render_rebalance_recommendations',
    'render_performance_attribution_summary'
]
