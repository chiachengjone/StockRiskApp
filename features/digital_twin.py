"""
Digital Twin Feature for Portfolio Analysis.

Provides simulation-based scenario comparison between current portfolio
and automated rebalancing strategies. Enables forward-looking analysis
of portfolio performance under different management approaches.

Features:
- Scenario comparison dashboard
- Buy-and-hold vs active management comparison
- Tax-loss harvesting impact analysis
- Correlation convergence alerts visualization
- Real-time portfolio health monitoring
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta

# Import simulation components
try:
    from utils.portfolio import (
        PortfolioSimulator,
        SimulationConfig,
        DynamicRebalancer,
        CorrelationMonitor,
        TransactionCostModel
    )
    HAS_SIMULATION = True
except ImportError:
    HAS_SIMULATION = False


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ScenarioResult:
    """Result from a portfolio scenario simulation."""
    name: str
    mean_return: float
    std_return: float
    sharpe_ratio: float
    var_95: float
    cvar_95: float
    max_drawdown: float
    transaction_costs: float
    tax_benefits: float
    final_value_median: float
    final_value_5th: float
    final_value_95th: float
    rebalance_frequency: str
    portfolio_values: Optional[np.ndarray] = None


@dataclass
class PortfolioHealthScore:
    """Comprehensive portfolio health assessment."""
    overall_score: float  # 0-100
    diversification_score: float
    risk_efficiency_score: float
    momentum_score: float
    correlation_score: float
    recommendations: List[str] = field(default_factory=list)


# =============================================================================
# DIGITAL TWIN ENGINE
# =============================================================================

class DigitalTwinEngine:
    """
    Portfolio Digital Twin for scenario analysis.
    
    Creates simulated versions of the portfolio under different
    management strategies and compares outcomes.
    """
    
    def __init__(
        self,
        returns: pd.DataFrame,
        current_weights: Dict[str, float],
        initial_capital: float = 100000,
        horizon_days: int = 252
    ):
        self.returns = returns
        self.current_weights = current_weights
        self.initial_capital = initial_capital
        self.horizon_days = horizon_days
        self.assets = list(returns.columns)
        
        # Initialize components
        self.correlation_monitor = CorrelationMonitor(returns) if HAS_SIMULATION else None
    
    def run_scenario(
        self,
        name: str,
        rebalance_frequency: str = 'monthly',
        enable_tax_loss_harvesting: bool = False,
        transaction_cost_bps: float = 10,
        rebalance_threshold: float = 0.05
    ) -> ScenarioResult:
        """Run a single scenario simulation."""
        if not HAS_SIMULATION:
            # Return dummy result if simulation not available
            return ScenarioResult(
                name=name,
                mean_return=0.08,
                std_return=0.16,
                sharpe_ratio=0.5,
                var_95=-0.15,
                cvar_95=-0.20,
                max_drawdown=-0.25,
                transaction_costs=100,
                tax_benefits=0,
                final_value_median=self.initial_capital * 1.08,
                final_value_5th=self.initial_capital * 0.85,
                final_value_95th=self.initial_capital * 1.35,
                rebalance_frequency=rebalance_frequency
            )
        
        config = SimulationConfig(
            initial_capital=self.initial_capital,
            n_simulations=500,  # Reduced for speed
            horizon_days=self.horizon_days,
            rebalance_frequency=rebalance_frequency,
            rebalance_threshold=rebalance_threshold,
            transaction_cost_bps=transaction_cost_bps,
            enable_tax_loss_harvesting=enable_tax_loss_harvesting
        )
        
        simulator = PortfolioSimulator(
            self.returns,
            self.current_weights,
            config
        )
        
        results = simulator.run_simulation()
        
        # Calculate max drawdown from portfolio values
        portfolio_values = results['portfolio_values']
        max_drawdown = self._calculate_max_drawdown(portfolio_values)
        
        return ScenarioResult(
            name=name,
            mean_return=results['mean_return'],
            std_return=results['std_return'],
            sharpe_ratio=results['sharpe_ratio'],
            var_95=results['var_95'],
            cvar_95=results['cvar_95'],
            max_drawdown=max_drawdown,
            transaction_costs=results['mean_tx_costs'],
            tax_benefits=results['mean_tax_benefits'],
            final_value_median=results['percentiles']['50'],
            final_value_5th=results['percentiles']['5'],
            final_value_95th=results['percentiles']['95'],
            rebalance_frequency=rebalance_frequency,
            portfolio_values=portfolio_values
        )
    
    def _calculate_max_drawdown(self, portfolio_values: np.ndarray) -> float:
        """Calculate average maximum drawdown across simulations."""
        max_drawdowns = []
        
        for sim in range(portfolio_values.shape[0]):
            values = portfolio_values[sim]
            running_max = np.maximum.accumulate(values)
            drawdowns = (values - running_max) / running_max
            max_drawdowns.append(np.min(drawdowns))
        
        return float(np.mean(max_drawdowns))
    
    def compare_scenarios(self) -> Dict[str, ScenarioResult]:
        """Compare all standard scenarios."""
        scenarios = {
            'buy_and_hold': {
                'name': 'Buy & Hold',
                'rebalance_frequency': 'none',
                'enable_tax_loss_harvesting': False,
                'transaction_cost_bps': 0,
                'rebalance_threshold': 1.0  # Never triggers
            },
            'monthly_rebalance': {
                'name': 'Monthly Rebalance',
                'rebalance_frequency': 'monthly',
                'enable_tax_loss_harvesting': False,
                'transaction_cost_bps': 10,
                'rebalance_threshold': 0.05
            },
            'quarterly_rebalance': {
                'name': 'Quarterly Rebalance',
                'rebalance_frequency': 'quarterly',
                'enable_tax_loss_harvesting': False,
                'transaction_cost_bps': 10,
                'rebalance_threshold': 0.05
            },
            'threshold_5pct': {
                'name': '5% Threshold',
                'rebalance_frequency': 'daily',
                'enable_tax_loss_harvesting': False,
                'transaction_cost_bps': 10,
                'rebalance_threshold': 0.05
            },
            'tax_optimized': {
                'name': 'Tax Optimized',
                'rebalance_frequency': 'monthly',
                'enable_tax_loss_harvesting': True,
                'transaction_cost_bps': 10,
                'rebalance_threshold': 0.05
            }
        }
        
        results = {}
        for key, params in scenarios.items():
            results[key] = self.run_scenario(**params)
        
        return results
    
    def calculate_health_score(self) -> PortfolioHealthScore:
        """Calculate comprehensive portfolio health score."""
        recommendations = []
        
        # 1. Diversification Score (based on weight concentration)
        weights = np.array(list(self.current_weights.values()))
        hhi = np.sum(weights ** 2)
        diversification_score = max(0, 100 * (1 - hhi))
        
        if hhi > 0.3:
            recommendations.append("Consider diversifying - portfolio is concentrated")
        
        # 2. Risk Efficiency Score (based on Sharpe-like metric)
        mean_ret = self.returns.mean() @ weights * 252
        cov = self.returns.cov().values * 252
        port_vol = np.sqrt(weights @ cov @ weights)
        risk_efficiency = mean_ret / port_vol if port_vol > 0 else 0
        risk_efficiency_score = min(100, max(0, risk_efficiency * 100))
        
        if risk_efficiency < 0.3:
            recommendations.append("Risk-adjusted returns are low - consider optimization")
        
        # 3. Momentum Score (recent performance)
        recent_returns = self.returns.tail(21).mean() @ weights * 252
        historical_returns = self.returns.mean() @ weights * 252
        momentum_ratio = recent_returns / historical_returns if historical_returns != 0 else 1
        momentum_score = min(100, max(0, 50 + momentum_ratio * 50))
        
        if momentum_ratio < 0.5:
            recommendations.append("Negative momentum detected - review positions")
        
        # 4. Correlation Score (lower average correlation is better)
        corr_matrix = self.returns.corr()
        avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, 1)].mean()
        correlation_score = max(0, 100 * (1 - avg_corr))
        
        if avg_corr > 0.7:
            recommendations.append("High correlation between assets - diversification benefits limited")
        
        # Check correlation convergence
        if self.correlation_monitor:
            convergence = self.correlation_monitor.detect_convergence()
            if convergence['n_alerts'] > 0:
                recommendations.append(f"{convergence['n_alerts']} correlation convergence alerts detected")
        
        # Overall Score (weighted average)
        overall_score = (
            diversification_score * 0.25 +
            risk_efficiency_score * 0.30 +
            momentum_score * 0.20 +
            correlation_score * 0.25
        )
        
        return PortfolioHealthScore(
            overall_score=overall_score,
            diversification_score=diversification_score,
            risk_efficiency_score=risk_efficiency_score,
            momentum_score=momentum_score,
            correlation_score=correlation_score,
            recommendations=recommendations
        )


# =============================================================================
# VISUALIZATION COMPONENTS
# =============================================================================

def create_scenario_comparison_chart(
    scenarios: Dict[str, ScenarioResult]
) -> go.Figure:
    """Create bar chart comparing scenario metrics."""
    names = [s.name for s in scenarios.values()]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Expected Return', 'Sharpe Ratio', 'VaR (95%)', 'Net Costs'),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    colors = px.colors.qualitative.Set2
    
    # Expected Return
    fig.add_trace(
        go.Bar(
            x=names,
            y=[s.mean_return * 100 for s in scenarios.values()],
            marker_color=colors[:len(names)],
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Sharpe Ratio
    fig.add_trace(
        go.Bar(
            x=names,
            y=[s.sharpe_ratio for s in scenarios.values()],
            marker_color=colors[:len(names)],
            showlegend=False
        ),
        row=1, col=2
    )
    
    # VaR
    fig.add_trace(
        go.Bar(
            x=names,
            y=[s.var_95 * 100 for s in scenarios.values()],
            marker_color=colors[:len(names)],
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Net Costs
    fig.add_trace(
        go.Bar(
            x=names,
            y=[s.transaction_costs - s.tax_benefits for s in scenarios.values()],
            marker_color=colors[:len(names)],
            showlegend=False
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=500,
        template='plotly_dark',
        title='Scenario Comparison',
        font=dict(size=12)
    )
    
    fig.update_yaxes(title_text='Return %', row=1, col=1)
    fig.update_yaxes(title_text='Ratio', row=1, col=2)
    fig.update_yaxes(title_text='VaR %', row=2, col=1)
    fig.update_yaxes(title_text='$ Cost', row=2, col=2)
    
    return fig


def create_fan_chart(
    scenarios: Dict[str, ScenarioResult],
    horizon_days: int = 252,
    initial_capital: float = 100000
) -> go.Figure:
    """Create fan chart showing portfolio value distributions."""
    fig = go.Figure()
    
    colors = {
        'buy_and_hold': '#636EFA',
        'monthly_rebalance': '#00CC96',
        'quarterly_rebalance': '#AB63FA',
        'threshold_5pct': '#FFA15A',
        'tax_optimized': '#EF553B'
    }
    
    x = list(range(horizon_days + 1))
    
    for key, scenario in scenarios.items():
        color = colors.get(key, '#636EFA')
        
        if scenario.portfolio_values is not None:
            values = scenario.portfolio_values
            p5 = np.percentile(values, 5, axis=0)
            p25 = np.percentile(values, 25, axis=0)
            p50 = np.percentile(values, 50, axis=0)
            p75 = np.percentile(values, 75, axis=0)
            p95 = np.percentile(values, 95, axis=0)
            
            # 5-95 percentile band
            fig.add_trace(go.Scatter(
                x=x + x[::-1],
                y=list(p95) + list(p5)[::-1],
                fill='toself',
                fillcolor=f'rgba{tuple(int(color[1:][i:i+2], 16) for i in (0, 2, 4)) + (0.1,)}',
                line=dict(width=0),
                name=f'{scenario.name} (5-95%)',
                showlegend=False
            ))
            
            # 25-75 percentile band
            fig.add_trace(go.Scatter(
                x=x + x[::-1],
                y=list(p75) + list(p25)[::-1],
                fill='toself',
                fillcolor=f'rgba{tuple(int(color[1:][i:i+2], 16) for i in (0, 2, 4)) + (0.2,)}',
                line=dict(width=0),
                name=f'{scenario.name} (25-75%)',
                showlegend=False
            ))
            
            # Median line
            fig.add_trace(go.Scatter(
                x=x,
                y=p50,
                mode='lines',
                line=dict(color=color, width=2),
                name=scenario.name
            ))
    
    fig.update_layout(
        height=450,
        template='plotly_dark',
        title='Portfolio Value Fan Chart',
        xaxis_title='Days',
        yaxis_title='Portfolio Value ($)',
        legend=dict(orientation='h', y=-0.15)
    )
    
    return fig


def create_health_gauge(score: float, title: str = 'Portfolio Health') -> go.Figure:
    """Create a gauge chart for health score."""
    # Determine color based on score
    if score >= 70:
        color = '#00CC96'
    elif score >= 40:
        color = '#FFA15A'
    else:
        color = '#EF553B'
    
    fig = go.Figure(go.Indicator(
        mode='gauge+number',
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 16, 'color': 'white'}},
        number={'suffix': '', 'font': {'size': 32, 'color': 'white'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': 'white'},
            'bar': {'color': color},
            'bgcolor': '#1a1a2e',
            'borderwidth': 2,
            'bordercolor': '#3d3d5c',
            'steps': [
                {'range': [0, 40], 'color': '#2d1f1f'},
                {'range': [40, 70], 'color': '#2d2d1f'},
                {'range': [70, 100], 'color': '#1f2d1f'}
            ]
        }
    ))
    
    fig.update_layout(
        height=250,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def create_correlation_heatmap(
    corr_monitor: CorrelationMonitor
) -> Tuple[go.Figure, go.Figure]:
    """Create baseline and recent correlation heatmaps."""
    baseline = corr_monitor.calculate_baseline_correlation()
    recent = corr_monitor.calculate_recent_correlation()
    
    fig_baseline = go.Figure(data=go.Heatmap(
        z=baseline.values,
        x=baseline.columns,
        y=baseline.index,
        colorscale='RdBu',
        zmid=0,
        text=np.round(baseline.values, 2),
        texttemplate='%{text}',
        showscale=True
    ))
    
    fig_baseline.update_layout(
        title='Baseline Correlation (252d)',
        height=400,
        template='plotly_dark'
    )
    
    fig_recent = go.Figure(data=go.Heatmap(
        z=recent.values,
        x=recent.columns,
        y=recent.index,
        colorscale='RdBu',
        zmid=0,
        text=np.round(recent.values, 2),
        texttemplate='%{text}',
        showscale=True
    ))
    
    fig_recent.update_layout(
        title='Recent Correlation (21d)',
        height=400,
        template='plotly_dark'
    )
    
    return fig_baseline, fig_recent


def create_correlation_change_chart(
    corr_monitor: CorrelationMonitor
) -> go.Figure:
    """Create chart showing correlation changes over time."""
    rolling_corr = corr_monitor.get_rolling_average_correlation()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=rolling_corr.index,
        y=rolling_corr.values,
        mode='lines',
        fill='tozeroy',
        line=dict(color='#636EFA'),
        name='Avg Correlation'
    ))
    
    # Add warning zone
    fig.add_hline(y=0.6, line_dash='dash', line_color='#FFA15A',
                  annotation_text='Warning Zone', annotation_position='right')
    fig.add_hline(y=0.8, line_dash='dash', line_color='#EF553B',
                  annotation_text='Danger Zone', annotation_position='right')
    
    fig.update_layout(
        height=350,
        template='plotly_dark',
        title='Rolling Average Pairwise Correlation',
        xaxis_title='Date',
        yaxis_title='Average Correlation',
        yaxis=dict(range=[0, 1])
    )
    
    return fig


# =============================================================================
# STREAMLIT COMPONENTS
# =============================================================================

def render_digital_twin_tab(
    returns: pd.DataFrame,
    weights: Dict[str, float],
    initial_capital: float = 100000
):
    """Render the Digital Twin dashboard tab."""
    st.markdown("### Portfolio Digital Twin")
    st.markdown("*Compare your portfolio under different management scenarios*")
    
    # Configuration
    with st.expander("⚙️ Simulation Configuration", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            horizon_months = st.slider(
                "Simulation Horizon (months)",
                min_value=3,
                max_value=36,
                value=12
            )
        
        with col2:
            n_simulations = st.selectbox(
                "Number of Simulations",
                [100, 250, 500, 1000],
                index=1
            )
        
        with col3:
            tx_cost_bps = st.slider(
                "Transaction Cost (bps)",
                min_value=0,
                max_value=50,
                value=10
            )
    
    # Initialize engine
    horizon_days = horizon_months * 21  # Approximate trading days
    
    engine = DigitalTwinEngine(
        returns=returns,
        current_weights=weights,
        initial_capital=initial_capital,
        horizon_days=horizon_days
    )
    
    # Run simulations
    with st.spinner("Running scenario simulations..."):
        scenarios = engine.compare_scenarios()
    
    # Health Score
    health = engine.calculate_health_score()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.plotly_chart(
            create_health_gauge(health.overall_score),
            use_container_width=True
        )
        
        # Component scores
        st.markdown("**Component Scores:**")
        st.progress(health.diversification_score / 100, text=f"Diversification: {health.diversification_score:.0f}")
        st.progress(health.risk_efficiency_score / 100, text=f"Risk Efficiency: {health.risk_efficiency_score:.0f}")
        st.progress(health.momentum_score / 100, text=f"Momentum: {health.momentum_score:.0f}")
        st.progress(health.correlation_score / 100, text=f"Correlation: {health.correlation_score:.0f}")
    
    with col2:
        # Recommendations
        st.markdown("**Recommendations:**")
        if health.recommendations:
            for rec in health.recommendations:
                st.info(rec)
        else:
            st.success("Portfolio looks healthy! No immediate action needed.")
    
    st.divider()
    
    # Scenario Comparison
    st.markdown("### Scenario Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Comparison", "Fan Chart", "Details"])
    
    with tab1:
        st.plotly_chart(
            create_scenario_comparison_chart(scenarios),
            use_container_width=True
        )
    
    with tab2:
        st.plotly_chart(
            create_fan_chart(scenarios, horizon_days, initial_capital),
            use_container_width=True
        )
    
    with tab3:
        # Detailed table
        scenario_data = []
        for key, s in scenarios.items():
            scenario_data.append({
                'Strategy': s.name,
                'Mean Return': f"{s.mean_return * 100:.1f}%",
                'Volatility': f"{s.std_return * 100:.1f}%",
                'Sharpe Ratio': f"{s.sharpe_ratio:.2f}",
                'VaR (95%)': f"{s.var_95 * 100:.1f}%",
                'CVaR (95%)': f"{s.cvar_95 * 100:.1f}%",
                'Max Drawdown': f"{s.max_drawdown * 100:.1f}%",
                'Tx Costs': f"${s.transaction_costs:,.0f}",
                'Tax Benefits': f"${s.tax_benefits:,.0f}",
                'Median Final': f"${s.final_value_median:,.0f}"
            })
        
        df = pd.DataFrame(scenario_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Best scenario
        best_sharpe = max(scenarios.values(), key=lambda x: x.sharpe_ratio)
        best_return = max(scenarios.values(), key=lambda x: x.mean_return)
        lowest_risk = min(scenarios.values(), key=lambda x: abs(x.var_95))
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Best Risk-Adjusted", best_sharpe.name, f"Sharpe: {best_sharpe.sharpe_ratio:.2f}")
        col2.metric("Highest Return", best_return.name, f"{best_return.mean_return * 100:.1f}%")
        col3.metric("Lowest Risk", lowest_risk.name, f"VaR: {lowest_risk.var_95 * 100:.1f}%")
    
    st.divider()
    
    # Correlation Monitoring
    st.markdown("### 🔗 Correlation Monitoring")
    
    if engine.correlation_monitor:
        convergence = engine.correlation_monitor.detect_convergence()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Avg Baseline Correlation",
                f"{convergence['baseline_avg_correlation']:.2f}"
            )
        
        with col2:
            st.metric(
                "Avg Recent Correlation",
                f"{convergence['recent_avg_correlation']:.2f}",
                delta=f"{convergence['recent_avg_correlation'] - convergence['baseline_avg_correlation']:.2f}"
            )
        
        # Regime indicator
        if convergence['correlation_regime'] == 'stressed':
            st.warning("Elevated correlation detected - diversification benefits may be reduced")
        else:
            st.success("Correlation levels are normal")
        
        # Alerts
        if convergence['n_alerts'] > 0:
            st.markdown("**Correlation Alerts:**")
            for alert in convergence['alerts'][:5]:
                severity_emoji = "" if alert['severity'] == 'high' else ""
                st.warning(
                    f"{severity_emoji} {alert['asset1']} ↔ {alert['asset2']}: "
                    f"Correlation {alert['alert_type']} from {alert['baseline_correlation']:.2f} "
                    f"to {alert['recent_correlation']:.2f}"
                )
        
        # Correlation heatmaps
        with st.expander("Correlation Matrices", expanded=False):
            fig_baseline, fig_recent = create_correlation_heatmap(engine.correlation_monitor)
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig_baseline, use_container_width=True)
            with col2:
                st.plotly_chart(fig_recent, use_container_width=True)
        
        # Rolling correlation chart
        st.plotly_chart(
            create_correlation_change_chart(engine.correlation_monitor),
            use_container_width=True
        )


def render_quick_scenario_widget(
    returns: pd.DataFrame,
    weights: Dict[str, float]
):
    """Render a compact scenario comparison widget for sidebar."""
    st.markdown("#### Quick Scenario Check")
    
    engine = DigitalTwinEngine(
        returns=returns,
        current_weights=weights,
        initial_capital=100000,
        horizon_days=252
    )
    
    # Run quick scenarios
    scenarios = {
        'hold': engine.run_scenario('Buy & Hold', 'none', False, 0),
        'rebal': engine.run_scenario('Monthly Rebal', 'monthly', False, 10)
    }
    
    # Compare
    hold_ret = scenarios['hold'].mean_return * 100
    rebal_ret = scenarios['rebal'].mean_return * 100
    
    col1, col2 = st.columns(2)
    col1.metric("Buy & Hold", f"{hold_ret:.1f}%")
    col2.metric("Rebalance", f"{rebal_ret:.1f}%", f"{rebal_ret - hold_ret:+.1f}%")
    
    # Health
    health = engine.calculate_health_score()
    st.progress(health.overall_score / 100, text=f"Health: {health.overall_score:.0f}/100")
