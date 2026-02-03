"""
Interactive What-If Analysis Feature.

Provides drag-and-drop portfolio optimization interface with
real-time metric updates and scenario comparisons.

Features:
- Interactive weight adjustment sliders
- Real-time risk/return recalculation
- Efficient frontier overlay
- Constraint visualization
- Rebalancing trade recommendations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from scipy.optimize import minimize
import copy


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PortfolioConstraints:
    """Constraints for portfolio optimization."""
    min_weight: float = 0.0
    max_weight: float = 1.0
    asset_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    sector_limits: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    max_positions: int = None
    min_positions: int = None


@dataclass
class WhatIfScenario:
    """A what-if scenario definition."""
    name: str
    weights: Dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    var_95: float
    max_drawdown: float
    turnover: float = 0.0


# =============================================================================
# WHAT-IF ANALYSIS ENGINE
# =============================================================================

class WhatIfAnalyzer:
    """
    Engine for what-if portfolio analysis.
    
    Provides real-time calculation of portfolio metrics
    as weights are adjusted.
    """
    
    def __init__(
        self,
        returns: pd.DataFrame,
        current_weights: Dict[str, float],
        risk_free_rate: float = 0.05,
        transaction_cost_bps: float = 10
    ):
        self.returns = returns
        self.current_weights = current_weights
        self.risk_free_rate = risk_free_rate
        self.transaction_cost_bps = transaction_cost_bps
        
        self.assets = list(returns.columns)
        self.n_assets = len(self.assets)
        
        # Pre-calculate statistics
        self.mean_returns = returns.mean().values * 252
        self.cov_matrix = returns.cov().values * 252
        
        # Current portfolio stats
        self.current_stats = self._calculate_stats(current_weights)
    
    def _weights_to_array(self, weights: Dict[str, float]) -> np.ndarray:
        """Convert weight dict to array."""
        return np.array([weights.get(a, 0) for a in self.assets])
    
    def _calculate_stats(self, weights: Dict[str, float]) -> Dict:
        """Calculate portfolio statistics for given weights."""
        w = self._weights_to_array(weights)
        
        # Basic stats
        expected_return = np.dot(w, self.mean_returns)
        volatility = np.sqrt(w.T @ self.cov_matrix @ w)
        sharpe = (expected_return - self.risk_free_rate) / volatility if volatility > 0 else 0
        
        # VaR calculation (parametric)
        z_score = 1.645  # 95% confidence
        var_95 = -(expected_return - z_score * volatility)
        
        # Component VaR
        marginal_var = self.cov_matrix @ w / volatility if volatility > 0 else np.zeros(self.n_assets)
        component_var = w * marginal_var
        
        # Diversification ratio
        weighted_vol = np.sum(w * np.sqrt(np.diag(self.cov_matrix)))
        diversification_ratio = weighted_vol / volatility if volatility > 0 else 1
        
        return {
            'expected_return': expected_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'var_95': var_95,
            'component_var': dict(zip(self.assets, component_var)),
            'diversification_ratio': diversification_ratio,
            'concentration': np.sum(w ** 2)  # HHI
        }
    
    def analyze_scenario(
        self,
        new_weights: Dict[str, float],
        scenario_name: str = "What-If"
    ) -> WhatIfScenario:
        """Analyze a what-if scenario."""
        stats = self._calculate_stats(new_weights)
        
        # Calculate turnover from current
        turnover = sum(
            abs(new_weights.get(a, 0) - self.current_weights.get(a, 0))
            for a in self.assets
        )
        
        return WhatIfScenario(
            name=scenario_name,
            weights=new_weights,
            expected_return=stats['expected_return'],
            volatility=stats['volatility'],
            sharpe_ratio=stats['sharpe_ratio'],
            var_95=stats['var_95'],
            max_drawdown=stats['volatility'] * 2.5,  # Rough estimate
            turnover=turnover
        )
    
    def calculate_rebalance_trades(
        self,
        new_weights: Dict[str, float],
        portfolio_value: float = 100000
    ) -> List[Dict]:
        """Calculate trades needed to rebalance."""
        trades = []
        
        for asset in self.assets:
            current = self.current_weights.get(asset, 0)
            new = new_weights.get(asset, 0)
            diff = new - current
            
            if abs(diff) > 0.001:  # 0.1% threshold
                trade_value = diff * portfolio_value
                cost = abs(trade_value) * (self.transaction_cost_bps / 10000)
                
                trades.append({
                    'asset': asset,
                    'current_weight': current,
                    'new_weight': new,
                    'change': diff,
                    'trade_value': trade_value,
                    'action': 'BUY' if diff > 0 else 'SELL',
                    'transaction_cost': cost
                })
        
        return sorted(trades, key=lambda x: abs(x['trade_value']), reverse=True)
    
    def optimize_for_target(
        self,
        target_return: float = None,
        target_volatility: float = None,
        target_sharpe: float = None,
        constraints: PortfolioConstraints = None
    ) -> Dict[str, float]:
        """Optimize portfolio for a target metric."""
        constraints = constraints or PortfolioConstraints()
        
        # Initial weights
        x0 = self._weights_to_array(self.current_weights)
        
        # Bounds
        bounds = []
        for asset in self.assets:
            if asset in constraints.asset_bounds:
                bounds.append(constraints.asset_bounds[asset])
            else:
                bounds.append((constraints.min_weight, constraints.max_weight))
        
        # Constraints
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        if target_return is not None:
            cons.append({
                'type': 'eq',
                'fun': lambda w: np.dot(w, self.mean_returns) - target_return
            })
            
            # Minimize volatility for target return
            def objective(w):
                return np.sqrt(w.T @ self.cov_matrix @ w)
        
        elif target_volatility is not None:
            cons.append({
                'type': 'eq',
                'fun': lambda w: np.sqrt(w.T @ self.cov_matrix @ w) - target_volatility
            })
            
            # Maximize return for target volatility
            def objective(w):
                return -np.dot(w, self.mean_returns)
        
        else:
            # Maximize Sharpe ratio
            def objective(w):
                ret = np.dot(w, self.mean_returns)
                vol = np.sqrt(w.T @ self.cov_matrix @ w)
                return -(ret - self.risk_free_rate) / vol if vol > 0 else 0
        
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons
        )
        
        if result.success:
            return dict(zip(self.assets, result.x))
        else:
            return self.current_weights
    
    def generate_efficient_frontier(
        self,
        n_points: int = 50
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """Generate efficient frontier data."""
        target_returns = np.linspace(
            self.mean_returns.min(),
            self.mean_returns.max(),
            n_points
        )
        
        frontier_vols = []
        frontier_rets = []
        frontier_weights = []
        
        for target in target_returns:
            try:
                weights = self.optimize_for_target(target_return=target)
                stats = self._calculate_stats(weights)
                frontier_rets.append(stats['expected_return'])
                frontier_vols.append(stats['volatility'])
                frontier_weights.append(weights)
            except:
                continue
        
        return np.array(frontier_rets), np.array(frontier_vols), frontier_weights


# =============================================================================
# VISUALIZATION COMPONENTS
# =============================================================================

def create_metrics_comparison_chart(
    current: WhatIfScenario,
    proposed: WhatIfScenario
) -> go.Figure:
    """Create radar chart comparing current vs proposed portfolio."""
    categories = ['Return', 'Sharpe', 'Diversification', 'Risk Control', 'Efficiency']
    
    # Normalize metrics to 0-100 scale
    def normalize(val, min_val, max_val):
        return (val - min_val) / (max_val - min_val) * 100 if max_val > min_val else 50
    
    current_values = [
        normalize(current.expected_return, -0.1, 0.3),
        normalize(current.sharpe_ratio, 0, 2),
        100 - normalize(current.turnover, 0, 2),  # Lower turnover = better
        100 - normalize(current.var_95, 0, 0.3),  # Lower VaR = better
        normalize(current.sharpe_ratio / max(current.volatility, 0.01), 0, 10)
    ]
    
    proposed_values = [
        normalize(proposed.expected_return, -0.1, 0.3),
        normalize(proposed.sharpe_ratio, 0, 2),
        100 - normalize(proposed.turnover, 0, 2),
        100 - normalize(proposed.var_95, 0, 0.3),
        normalize(proposed.sharpe_ratio / max(proposed.volatility, 0.01), 0, 10)
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=current_values + [current_values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(255,87,34,0.2)',
        line=dict(color='#FF5722', width=2),
        name='Current'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=proposed_values + [proposed_values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(76,175,80,0.2)',
        line=dict(color='#4CAF50', width=2),
        name='Proposed'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showticklabels=False
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        showlegend=True,
        legend=dict(orientation='h', y=-0.1),
        height=350,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def create_weight_comparison_chart(
    current: Dict[str, float],
    proposed: Dict[str, float]
) -> go.Figure:
    """Create grouped bar chart comparing weights."""
    assets = sorted(set(current.keys()) | set(proposed.keys()))
    
    current_vals = [current.get(a, 0) * 100 for a in assets]
    proposed_vals = [proposed.get(a, 0) * 100 for a in assets]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=assets,
        y=current_vals,
        name='Current',
        marker_color='#FF5722',
        text=[f'{v:.1f}%' for v in current_vals],
        textposition='outside'
    ))
    
    fig.add_trace(go.Bar(
        x=assets,
        y=proposed_vals,
        name='Proposed',
        marker_color='#4CAF50',
        text=[f'{v:.1f}%' for v in proposed_vals],
        textposition='outside'
    ))
    
    fig.update_layout(
        barmode='group',
        title='Weight Comparison',
        xaxis_title='Asset',
        yaxis_title='Weight (%)',
        height=350,
        template='plotly_dark',
        legend=dict(orientation='h', y=1.1)
    )
    
    return fig


def create_trade_table_chart(trades: List[Dict]) -> go.Figure:
    """Create visual trade recommendation table."""
    if not trades:
        return go.Figure()
    
    assets = [t['asset'] for t in trades]
    changes = [t['change'] * 100 for t in trades]
    values = [t['trade_value'] for t in trades]
    
    colors = ['#4CAF50' if c > 0 else '#F44336' for c in changes]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=assets,
        x=changes,
        orientation='h',
        marker_color=colors,
        text=[f'{c:+.1f}%' for c in changes],
        textposition='outside',
        hovertemplate="<b>%{y}</b><br>Change: %{x:+.1f}%<br>Value: $%{customdata:,.0f}<extra></extra>",
        customdata=values
    ))
    
    fig.add_vline(x=0, line_color='white', line_width=1)
    
    fig.update_layout(
        title='Recommended Trades',
        xaxis_title='Weight Change (%)',
        yaxis_title='Asset',
        height=max(250, len(trades) * 40),
        template='plotly_dark',
        showlegend=False
    )
    
    return fig


def create_frontier_with_scenarios(
    frontier_rets: np.ndarray,
    frontier_vols: np.ndarray,
    scenarios: List[WhatIfScenario]
) -> go.Figure:
    """Create efficient frontier with scenario points."""
    fig = go.Figure()
    
    # Frontier line
    fig.add_trace(go.Scatter(
        x=frontier_vols * 100,
        y=frontier_rets * 100,
        mode='lines',
        line=dict(color='#2196F3', width=3),
        name='Efficient Frontier',
        hovertemplate="Vol: %{x:.1f}% | Return: %{y:.1f}%<extra></extra>"
    ))
    
    # Scenario points
    colors = ['#FF5722', '#4CAF50', '#9C27B0', '#FFC107', '#00BCD4']
    symbols = ['circle', 'star', 'diamond', 'square', 'cross']
    
    for i, scenario in enumerate(scenarios):
        fig.add_trace(go.Scatter(
            x=[scenario.volatility * 100],
            y=[scenario.expected_return * 100],
            mode='markers+text',
            marker=dict(
                size=15,
                color=colors[i % len(colors)],
                symbol=symbols[i % len(symbols)]
            ),
            text=[scenario.name],
            textposition='top center',
            name=scenario.name,
            hovertemplate=(
                f"<b>{scenario.name}</b><br>"
                f"Return: {scenario.expected_return*100:.1f}%<br>"
                f"Volatility: {scenario.volatility*100:.1f}%<br>"
                f"Sharpe: {scenario.sharpe_ratio:.2f}<extra></extra>"
            )
        ))
    
    fig.update_layout(
        title='Scenarios on Efficient Frontier',
        xaxis_title='Volatility (%)',
        yaxis_title='Expected Return (%)',
        height=450,
        template='plotly_dark',
        legend=dict(orientation='h', y=-0.15),
        hovermode='closest'
    )
    
    return fig


# =============================================================================
# STREAMLIT COMPONENTS
# =============================================================================

def render_what_if_tab(
    returns: pd.DataFrame,
    current_weights: Dict[str, float],
    portfolio_value: float = 100000
):
    """Render the What-If Analysis tab."""
    st.markdown("### What-If Portfolio Analysis")
    st.markdown("*Adjust weights interactively and see real-time impact*")
    
    # Initialize analyzer
    analyzer = WhatIfAnalyzer(
        returns=returns,
        current_weights=current_weights
    )
    
    # Get current scenario
    current_scenario = analyzer.analyze_scenario(current_weights, "Current")
    
    # Initialize session state for weights (store as percentages 0-100 for better slider UX)
    if 'what_if_weights' not in st.session_state:
        st.session_state.what_if_weights = {k: v * 100 for k, v in current_weights.items()}
    
    # Weight adjustment UI
    st.markdown("#### Adjust Portfolio Weights")
    
    # Auto-normalize toggle
    col1, col2 = st.columns([2, 1])
    with col2:
        auto_normalize = st.checkbox("Auto-normalize to 100%", value=True, key="auto_norm")
    
    # Weight sliders - display as percentages (0-100)
    new_weights_pct = {}
    n_assets = len(analyzer.assets)
    n_cols = min(4, n_assets)
    cols = st.columns(n_cols)
    
    for i, asset in enumerate(analyzer.assets):
        col_idx = i % n_cols
        with cols[col_idx]:
            current_pct = current_weights.get(asset, 0) * 100
            stored_pct = st.session_state.what_if_weights.get(asset, current_pct)
            
            new_pct = st.slider(
                f"{asset}",
                min_value=0.0,
                max_value=100.0,
                value=float(stored_pct),
                step=1.0,
                format="%.0f%%",
                key=f"weight_slider_{asset}",
                help=f"Current: {current_pct:.1f}%"
            )
            new_weights_pct[asset] = new_pct
    
    # Update session state with new slider values
    st.session_state.what_if_weights = new_weights_pct.copy()
    
    # Calculate total and normalize
    total_pct = sum(new_weights_pct.values())
    
    # Convert to decimal weights
    if auto_normalize and total_pct > 0:
        normalized_weights = {k: v / total_pct for k, v in new_weights_pct.items()}
    else:
        normalized_weights = {k: v / 100 for k, v in new_weights_pct.items()}
    
    # Weight sum indicator and actions
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if abs(total_pct - 100.0) < 1.0:
            st.success(f"Total: {total_pct:.1f}%")
        elif total_pct > 100:
            st.warning(f"Total: {total_pct:.1f}% (over-allocated)")
        else:
            st.error(f"Total: {total_pct:.1f}% (under-allocated)")
    
    with col2:
        if st.button("Reset to Current", key="reset_btn"):
            st.session_state.what_if_weights = {k: v * 100 for k, v in current_weights.items()}
            st.rerun()
    
    with col3:
        if st.button("Equal Weight", key="equal_btn"):
            eq_weight = 100.0 / n_assets
            st.session_state.what_if_weights = {a: eq_weight for a in analyzer.assets}
            st.rerun()
    
    with col4:
        if total_pct != 100 and not auto_normalize:
            if st.button("Normalize Weights", key="normalize_btn"):
                if total_pct > 0:
                    st.session_state.what_if_weights = {k: (v / total_pct) * 100 for k, v in new_weights_pct.items()}
                    st.rerun()
    
    st.divider()
    
    # Analyze proposed scenario
    proposed_scenario = analyzer.analyze_scenario(normalized_weights, "Proposed")
    
    # Side-by-side comparison
    st.markdown("#### Impact Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Current Portfolio**")
        st.metric("Expected Return", f"{current_scenario.expected_return * 100:.1f}%")
        st.metric("Volatility", f"{current_scenario.volatility * 100:.1f}%")
        st.metric("Sharpe Ratio", f"{current_scenario.sharpe_ratio:.2f}")
        st.metric("VaR (95%)", f"{current_scenario.var_95 * 100:.1f}%")
    
    with col2:
        st.markdown("**Proposed Portfolio**")
        ret_delta = (proposed_scenario.expected_return - current_scenario.expected_return) * 100
        st.metric("Expected Return", f"{proposed_scenario.expected_return * 100:.1f}%", 
                  delta=f"{ret_delta:+.1f}%")
        
        vol_delta = (proposed_scenario.volatility - current_scenario.volatility) * 100
        st.metric("Volatility", f"{proposed_scenario.volatility * 100:.1f}%",
                  delta=f"{vol_delta:+.1f}%", delta_color="inverse")
        
        sharpe_delta = proposed_scenario.sharpe_ratio - current_scenario.sharpe_ratio
        st.metric("Sharpe Ratio", f"{proposed_scenario.sharpe_ratio:.2f}",
                  delta=f"{sharpe_delta:+.2f}")
        
        var_delta = (proposed_scenario.var_95 - current_scenario.var_95) * 100
        st.metric("VaR (95%)", f"{proposed_scenario.var_95 * 100:.1f}%",
                  delta=f"{var_delta:+.1f}%", delta_color="inverse")
    
    # Visualizations
    tab1, tab2, tab3 = st.tabs(["Comparison", "Frontier", "Trades"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(
                create_metrics_comparison_chart(current_scenario, proposed_scenario),
                use_container_width=True
            )
        
        with col2:
            st.plotly_chart(
                create_weight_comparison_chart(current_weights, normalized_weights),
                use_container_width=True
            )
    
    with tab2:
        # Generate frontier
        with st.spinner("Calculating efficient frontier..."):
            frontier_rets, frontier_vols, _ = analyzer.generate_efficient_frontier(30)
        
        scenarios = [current_scenario, proposed_scenario]
        
        # Add optimization targets
        if st.checkbox("Show optimal portfolios"):
            # Max Sharpe
            max_sharpe_weights = analyzer.optimize_for_target()
            max_sharpe = analyzer.analyze_scenario(max_sharpe_weights, "Max Sharpe")
            scenarios.append(max_sharpe)
            
            # Min Vol
            min_vol_weights = analyzer.optimize_for_target(target_return=analyzer.mean_returns.min())
            min_vol = analyzer.analyze_scenario(min_vol_weights, "Min Volatility")
            scenarios.append(min_vol)
        
        st.plotly_chart(
            create_frontier_with_scenarios(frontier_rets, frontier_vols, scenarios),
            use_container_width=True
        )
    
    with tab3:
        trades = analyzer.calculate_rebalance_trades(normalized_weights, portfolio_value)
        
        if trades:
            total_cost = sum(t['transaction_cost'] for t in trades)
            total_turnover = proposed_scenario.turnover / 2  # One-way
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Turnover", f"{total_turnover * 100:.1f}%")
            col2.metric("# of Trades", len(trades))
            col3.metric("Est. Transaction Cost", f"${total_cost:,.0f}")
            
            st.plotly_chart(
                create_trade_table_chart(trades),
                use_container_width=True
            )
            
            # Detailed trade list
            with st.expander("Detailed Trade List"):
                df = pd.DataFrame(trades)
                df['current_weight'] = df['current_weight'].apply(lambda x: f"{x*100:.1f}%")
                df['new_weight'] = df['new_weight'].apply(lambda x: f"{x*100:.1f}%")
                df['change'] = df['change'].apply(lambda x: f"{x*100:+.1f}%")
                df['trade_value'] = df['trade_value'].apply(lambda x: f"${x:,.0f}")
                df['transaction_cost'] = df['transaction_cost'].apply(lambda x: f"${x:,.2f}")
                
                st.dataframe(df[['asset', 'action', 'current_weight', 'new_weight', 
                               'change', 'trade_value', 'transaction_cost']], 
                           use_container_width=True, hide_index=True)
        else:
            st.info("No trades needed - proposed weights match current allocation.")
    
    # Optimization helpers
    st.divider()
    st.markdown("#### Quick Optimization")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Max Sharpe Ratio**")
        st.caption("Optimize for best risk-adjusted return")
        if st.button("Maximize Sharpe", key="max_sharpe_btn"):
            opt_weights = analyzer.optimize_for_target()
            # Convert to percentages for session state
            st.session_state.what_if_weights = {k: v * 100 for k, v in opt_weights.items()}
            st.rerun()
    
    with col2:
        st.markdown("**Target Return**")
        target_ret = st.slider("Select Target", 0, 30, 10, format="%d%%", key="target_ret_slider")
        if st.button("Optimize for Return", key="target_ret_btn"):
            opt_weights = analyzer.optimize_for_target(target_return=target_ret / 100)
            st.session_state.what_if_weights = {k: v * 100 for k, v in opt_weights.items()}
            st.rerun()
    
    with col3:
        st.markdown("**Target Volatility**")
        target_vol = st.slider("Select Target", 5, 30, 15, format="%d%%", key="target_vol_slider")
        if st.button("Optimize for Vol", key="target_vol_btn"):
            opt_weights = analyzer.optimize_for_target(target_volatility=target_vol / 100)
            st.session_state.what_if_weights = {k: v * 100 for k, v in opt_weights.items()}
            st.rerun()


def render_scenario_builder():
    """Render scenario builder in sidebar."""
    st.sidebar.markdown("### Saved Scenarios")
    
    if 'saved_scenarios' not in st.session_state:
        st.session_state.saved_scenarios = []
    
    # Save current scenario
    scenario_name = st.sidebar.text_input("Scenario Name")
    if st.sidebar.button("Save Current") and scenario_name:
        if 'what_if_weights' in st.session_state:
            st.session_state.saved_scenarios.append({
                'name': scenario_name,
                'weights': st.session_state.what_if_weights.copy()
            })
            st.sidebar.success(f"Saved: {scenario_name}")
    
    # Load saved scenarios
    if st.session_state.saved_scenarios:
        for i, scenario in enumerate(st.session_state.saved_scenarios):
            col1, col2 = st.sidebar.columns([3, 1])
            with col1:
                if st.button(f"{scenario['name']}", key=f"load_{i}"):
                    st.session_state.what_if_weights = scenario['weights'].copy()
                    st.rerun()
            with col2:
                if st.button("X", key=f"delete_{i}"):
                    st.session_state.saved_scenarios.pop(i)
                    st.rerun()
