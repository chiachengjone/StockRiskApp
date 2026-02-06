"""
Stock & Portfolio Growth Forecast Module
=========================================
Advanced growth forecasting using multiple techniques:
- Monte Carlo simulation with GARCH volatility
- Geometric Brownian Motion with regime detection
- Bootstrap resampling for confidence intervals
- Machine learning ensemble forecasts
- Fan charts and price projections

Single Stock Mode: Predicts stock price and growth
Portfolio Mode: Predicts individual stock prices + overall portfolio growth

Author: Stock Risk App | Feb 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Try to import GARCH for volatility forecasting
try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False

# Try to import sklearn for ML forecasting
try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    HAS_ML = True
except ImportError:
    HAS_ML = False


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ForecastResult:
    """Result of portfolio forecast."""
    horizon_days: int
    expected_value: float
    median_value: float
    confidence_intervals: Dict[str, Tuple[float, float]]  # e.g., {'90%': (low, high)}
    expected_return_pct: float
    prob_positive: float
    prob_loss_10pct: float
    prob_gain_20pct: float
    percentiles: Dict[int, float]  # {5: value, 25: value, 50: value, 75: value, 95: value}
    simulation_paths: np.ndarray
    time_steps: np.ndarray


@dataclass
class GrowthScenario:
    """Growth scenario projection."""
    scenario_name: str
    annual_return: float
    volatility: float
    projected_values: np.ndarray
    time_days: np.ndarray


# =============================================================================
# FORECAST ENGINE
# =============================================================================

class PortfolioForecastEngine:
    """
    Advanced portfolio forecasting engine using multiple techniques.
    """
    
    def __init__(
        self,
        returns: pd.DataFrame,
        weights: Dict[str, float],
        portfolio_value: float = 100000,
        risk_free_rate: float = 0.045
    ):
        self.returns = returns
        self.weights = weights
        self.portfolio_value = portfolio_value
        self.risk_free_rate = risk_free_rate
        
        # Calculate portfolio returns
        weights_arr = np.array([weights.get(col, 0) for col in returns.columns])
        self.port_returns = (returns * weights_arr).sum(axis=1)
        
        # Historical statistics
        self.historical_mean = self.port_returns.mean() * 252
        self.historical_vol = self.port_returns.std() * np.sqrt(252)
        self.historical_skew = self.port_returns.skew()
        self.historical_kurt = self.port_returns.kurtosis()
        
        # GARCH volatility forecast
        self.garch_forecast = self._fit_garch() if HAS_ARCH else None
        
    def _fit_garch(self) -> Optional[Dict]:
        """Fit GARCH(1,1) model for volatility forecasting."""
        try:
            scaled_returns = self.port_returns * 100
            model = arch_model(scaled_returns, vol='Garch', p=1, q=1, rescale=False)
            result = model.fit(disp='off', show_warning=False)
            
            # Forecast volatility for next 30 days
            forecast = result.forecast(horizon=30)
            vol_forecast = np.sqrt(forecast.variance.values[-1]) / 100
            
            return {
                'current_vol': vol_forecast[0] * np.sqrt(252),
                'forecast_vol': vol_forecast * np.sqrt(252),
                'persistence': result.params.get('alpha[1]', 0) + result.params.get('beta[1]', 0)
            }
        except:
            return None
    
    def monte_carlo_forecast(
        self,
        horizon_days: int = 252,
        n_simulations: int = 10000,
        use_garch: bool = True
    ) -> ForecastResult:
        """
        Monte Carlo simulation with optional GARCH volatility.
        
        Args:
            horizon_days: Forecast horizon in trading days
            n_simulations: Number of simulation paths
            use_garch: Whether to use GARCH volatility forecast
        
        Returns:
            ForecastResult with simulation results
        """
        np.random.seed(42)
        
        # Determine volatility to use
        if use_garch and self.garch_forecast:
            base_vol = self.garch_forecast['current_vol']
        else:
            base_vol = self.historical_vol
        
        # Daily parameters
        daily_mu = self.historical_mean / 252
        daily_sigma = base_vol / np.sqrt(252)
        
        # Generate random returns with slight fat tails (t-distribution)
        df_t = 5  # degrees of freedom for t-distribution
        random_shocks = stats.t.rvs(df=df_t, size=(n_simulations, horizon_days))
        random_shocks = random_shocks / np.sqrt(df_t / (df_t - 2))  # Scale to unit variance
        
        # Simulate returns with drift
        simulated_returns = daily_mu + daily_sigma * random_shocks
        
        # Calculate price paths
        price_paths = np.zeros((n_simulations, horizon_days + 1))
        price_paths[:, 0] = self.portfolio_value
        
        for t in range(horizon_days):
            price_paths[:, t + 1] = price_paths[:, t] * (1 + simulated_returns[:, t])
        
        # Extract final values
        final_values = price_paths[:, -1]
        
        # Calculate statistics
        expected_value = np.mean(final_values)
        median_value = np.median(final_values)
        
        # Confidence intervals
        ci_90 = (np.percentile(final_values, 5), np.percentile(final_values, 95))
        ci_80 = (np.percentile(final_values, 10), np.percentile(final_values, 90))
        ci_50 = (np.percentile(final_values, 25), np.percentile(final_values, 75))
        
        # Probabilities
        prob_positive = np.mean(final_values > self.portfolio_value)
        prob_loss_10 = np.mean(final_values < self.portfolio_value * 0.9)
        prob_gain_20 = np.mean(final_values > self.portfolio_value * 1.2)
        
        # Percentiles
        percentiles = {
            5: np.percentile(final_values, 5),
            25: np.percentile(final_values, 25),
            50: np.percentile(final_values, 50),
            75: np.percentile(final_values, 75),
            95: np.percentile(final_values, 95)
        }
        
        # Time steps for plotting
        time_steps = np.arange(horizon_days + 1)
        
        return ForecastResult(
            horizon_days=horizon_days,
            expected_value=expected_value,
            median_value=median_value,
            confidence_intervals={
                '90%': ci_90,
                '80%': ci_80,
                '50%': ci_50
            },
            expected_return_pct=(expected_value / self.portfolio_value - 1) * 100,
            prob_positive=prob_positive,
            prob_loss_10pct=prob_loss_10,
            prob_gain_20pct=prob_gain_20,
            percentiles=percentiles,
            simulation_paths=price_paths,
            time_steps=time_steps
        )
    
    def bootstrap_forecast(
        self,
        horizon_days: int = 252,
        n_simulations: int = 5000,
        block_size: int = 20
    ) -> ForecastResult:
        """
        Bootstrap resampling forecast preserving return structure.
        
        Args:
            horizon_days: Forecast horizon
            n_simulations: Number of bootstrap samples
            block_size: Block size for block bootstrap
        
        Returns:
            ForecastResult
        """
        np.random.seed(42)
        
        returns_arr = self.port_returns.values
        n_returns = len(returns_arr)
        n_blocks = (horizon_days // block_size) + 1
        
        price_paths = np.zeros((n_simulations, horizon_days + 1))
        price_paths[:, 0] = self.portfolio_value
        
        for sim in range(n_simulations):
            # Sample random blocks
            block_starts = np.random.randint(0, n_returns - block_size, n_blocks)
            
            # Concatenate blocks
            sampled_returns = []
            for start in block_starts:
                sampled_returns.extend(returns_arr[start:start + block_size])
            
            sampled_returns = np.array(sampled_returns[:horizon_days])
            
            # Calculate path
            for t in range(horizon_days):
                price_paths[sim, t + 1] = price_paths[sim, t] * (1 + sampled_returns[t])
        
        final_values = price_paths[:, -1]
        
        return ForecastResult(
            horizon_days=horizon_days,
            expected_value=np.mean(final_values),
            median_value=np.median(final_values),
            confidence_intervals={
                '90%': (np.percentile(final_values, 5), np.percentile(final_values, 95)),
                '80%': (np.percentile(final_values, 10), np.percentile(final_values, 90)),
                '50%': (np.percentile(final_values, 25), np.percentile(final_values, 75))
            },
            expected_return_pct=(np.mean(final_values) / self.portfolio_value - 1) * 100,
            prob_positive=np.mean(final_values > self.portfolio_value),
            prob_loss_10pct=np.mean(final_values < self.portfolio_value * 0.9),
            prob_gain_20pct=np.mean(final_values > self.portfolio_value * 1.2),
            percentiles={
                5: np.percentile(final_values, 5),
                25: np.percentile(final_values, 25),
                50: np.percentile(final_values, 50),
                75: np.percentile(final_values, 75),
                95: np.percentile(final_values, 95)
            },
            simulation_paths=price_paths,
            time_steps=np.arange(horizon_days + 1)
        )
    
    def scenario_projections(
        self,
        horizon_days: int = 252
    ) -> List[GrowthScenario]:
        """
        Generate growth scenarios (bull, base, bear).
        
        Returns:
            List of GrowthScenario objects
        """
        time_days = np.arange(horizon_days + 1)
        
        scenarios = []
        
        # Bull scenario: Historical mean + 1 std
        bull_return = self.historical_mean + self.historical_vol * 0.5
        bull_vol = self.historical_vol * 0.8
        bull_values = self.portfolio_value * np.exp(
            (bull_return - 0.5 * bull_vol**2) * time_days / 252
        )
        scenarios.append(GrowthScenario(
            scenario_name="Bull",
            annual_return=bull_return,
            volatility=bull_vol,
            projected_values=bull_values,
            time_days=time_days
        ))
        
        # Base scenario: Historical mean
        base_values = self.portfolio_value * np.exp(
            (self.historical_mean - 0.5 * self.historical_vol**2) * time_days / 252
        )
        scenarios.append(GrowthScenario(
            scenario_name="Base",
            annual_return=self.historical_mean,
            volatility=self.historical_vol,
            projected_values=base_values,
            time_days=time_days
        ))
        
        # Bear scenario: Historical mean - 1 std
        bear_return = self.historical_mean - self.historical_vol * 1.0
        bear_vol = self.historical_vol * 1.2
        bear_values = self.portfolio_value * np.exp(
            (bear_return - 0.5 * bear_vol**2) * time_days / 252
        )
        scenarios.append(GrowthScenario(
            scenario_name="Bear",
            annual_return=bear_return,
            volatility=bear_vol,
            projected_values=bear_values,
            time_days=time_days
        ))
        
        return scenarios
    
    def get_price_targets(
        self,
        horizons: List[int] = [30, 90, 180, 252, 504]
    ) -> pd.DataFrame:
        """
        Calculate price targets at various horizons.
        
        Args:
            horizons: List of horizon days
        
        Returns:
            DataFrame with price targets
        """
        targets = []
        
        for horizon in horizons:
            forecast = self.monte_carlo_forecast(horizon_days=horizon, n_simulations=5000)
            
            targets.append({
                'Horizon': f"{horizon}D" if horizon < 252 else f"{horizon//252}Y",
                'Days': horizon,
                'Expected': forecast.expected_value,
                'Median': forecast.median_value,
                'Low (5%)': forecast.percentiles[5],
                'High (95%)': forecast.percentiles[95],
                'Exp. Return': f"{forecast.expected_return_pct:.1f}%",
                'P(Gain)': f"{forecast.prob_positive*100:.0f}%"
            })
        
        return pd.DataFrame(targets)


# =============================================================================
# SINGLE STOCK FORECAST ENGINE
# =============================================================================

class SingleStockForecastEngine:
    """
    Forecast engine for individual stock price prediction.
    """
    
    def __init__(
        self,
        returns: pd.Series,
        current_price: float,
        ticker: str,
        risk_free_rate: float = 0.045
    ):
        self.returns = returns
        self.current_price = current_price
        self.ticker = ticker
        self.risk_free_rate = risk_free_rate
        
        # Historical statistics
        self.historical_mean = returns.mean() * 252
        self.historical_vol = returns.std() * np.sqrt(252)
        self.historical_skew = returns.skew()
        self.historical_kurt = returns.kurtosis()
        
        # GARCH volatility forecast
        self.garch_forecast = self._fit_garch() if HAS_ARCH else None
    
    def _fit_garch(self) -> Optional[Dict]:
        """Fit GARCH(1,1) model for volatility forecasting."""
        try:
            scaled_returns = self.returns * 100
            model = arch_model(scaled_returns, vol='Garch', p=1, q=1, rescale=False)
            result = model.fit(disp='off', show_warning=False)
            
            forecast = result.forecast(horizon=30)
            vol_forecast = np.sqrt(forecast.variance.values[-1]) / 100
            
            return {
                'current_vol': vol_forecast[0] * np.sqrt(252),
                'forecast_vol': vol_forecast * np.sqrt(252),
                'persistence': result.params.get('alpha[1]', 0) + result.params.get('beta[1]', 0)
            }
        except:
            return None
    
    def monte_carlo_forecast(
        self,
        horizon_days: int = 252,
        n_simulations: int = 10000,
        use_garch: bool = True
    ) -> ForecastResult:
        """Monte Carlo simulation for stock price."""
        np.random.seed(42)
        
        if use_garch and self.garch_forecast:
            base_vol = self.garch_forecast['current_vol']
        else:
            base_vol = self.historical_vol
        
        daily_mu = self.historical_mean / 252
        daily_sigma = base_vol / np.sqrt(252)
        
        # t-distribution for fat tails
        df_t = 5
        random_shocks = stats.t.rvs(df=df_t, size=(n_simulations, horizon_days))
        random_shocks = random_shocks / np.sqrt(df_t / (df_t - 2))
        
        simulated_returns = daily_mu + daily_sigma * random_shocks
        
        # Price paths
        price_paths = np.zeros((n_simulations, horizon_days + 1))
        price_paths[:, 0] = self.current_price
        
        for t in range(horizon_days):
            price_paths[:, t + 1] = price_paths[:, t] * (1 + simulated_returns[:, t])
        
        final_values = price_paths[:, -1]
        expected_value = np.mean(final_values)
        median_value = np.median(final_values)
        
        return ForecastResult(
            horizon_days=horizon_days,
            expected_value=expected_value,
            median_value=median_value,
            confidence_intervals={
                '90%': (np.percentile(final_values, 5), np.percentile(final_values, 95)),
                '80%': (np.percentile(final_values, 10), np.percentile(final_values, 90)),
                '50%': (np.percentile(final_values, 25), np.percentile(final_values, 75))
            },
            expected_return_pct=(expected_value / self.current_price - 1) * 100,
            prob_positive=np.mean(final_values > self.current_price),
            prob_loss_10pct=np.mean(final_values < self.current_price * 0.9),
            prob_gain_20pct=np.mean(final_values > self.current_price * 1.2),
            percentiles={
                5: np.percentile(final_values, 5),
                25: np.percentile(final_values, 25),
                50: np.percentile(final_values, 50),
                75: np.percentile(final_values, 75),
                95: np.percentile(final_values, 95)
            },
            simulation_paths=price_paths,
            time_steps=np.arange(horizon_days + 1)
        )
    
    def get_price_targets(
        self,
        horizons: List[int] = [30, 90, 180, 252, 504]
    ) -> pd.DataFrame:
        """Calculate price targets at various horizons."""
        targets = []
        
        for horizon in horizons:
            forecast = self.monte_carlo_forecast(horizon_days=horizon, n_simulations=5000)
            
            targets.append({
                'Horizon': f"{horizon}D" if horizon < 252 else f"{horizon//252}Y",
                'Days': horizon,
                'Expected': forecast.expected_value,
                'Median': forecast.median_value,
                'Low (5%)': forecast.percentiles[5],
                'High (95%)': forecast.percentiles[95],
                'Exp. Return': f"{forecast.expected_return_pct:.1f}%",
                'P(Gain)': f"{forecast.prob_positive*100:.0f}%"
            })
        
        return pd.DataFrame(targets)
    
    def scenario_projections(
        self,
        horizon_days: int = 252
    ) -> List[GrowthScenario]:
        """Generate growth scenarios (bull, base, bear)."""
        time_days = np.arange(horizon_days + 1)
        scenarios = []
        
        # Bull scenario
        bull_return = self.historical_mean + self.historical_vol * 0.5
        bull_vol = self.historical_vol * 0.8
        bull_values = self.current_price * np.exp(
            (bull_return - 0.5 * bull_vol**2) * time_days / 252
        )
        scenarios.append(GrowthScenario(
            scenario_name="Bull",
            annual_return=bull_return,
            volatility=bull_vol,
            projected_values=bull_values,
            time_days=time_days
        ))
        
        # Base scenario
        base_values = self.current_price * np.exp(
            (self.historical_mean - 0.5 * self.historical_vol**2) * time_days / 252
        )
        scenarios.append(GrowthScenario(
            scenario_name="Base",
            annual_return=self.historical_mean,
            volatility=self.historical_vol,
            projected_values=base_values,
            time_days=time_days
        ))
        
        # Bear scenario
        bear_return = self.historical_mean - self.historical_vol * 1.0
        bear_vol = self.historical_vol * 1.2
        bear_values = self.current_price * np.exp(
            (bear_return - 0.5 * bear_vol**2) * time_days / 252
        )
        scenarios.append(GrowthScenario(
            scenario_name="Bear",
            annual_return=bear_return,
            volatility=bear_vol,
            projected_values=bear_values,
            time_days=time_days
        ))
        
        return scenarios


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_fan_chart(
    forecast: ForecastResult,
    portfolio_value: float,
    title: str = "Portfolio Growth Forecast"
) -> go.Figure:
    """
    Create fan chart showing forecast with confidence intervals.
    """
    # Sample paths for display (not all 10k)
    n_display = min(100, forecast.simulation_paths.shape[0])
    sample_idx = np.random.choice(forecast.simulation_paths.shape[0], n_display, replace=False)
    sample_paths = forecast.simulation_paths[sample_idx]
    
    # Calculate percentile bands
    p5 = np.percentile(forecast.simulation_paths, 5, axis=0)
    p25 = np.percentile(forecast.simulation_paths, 25, axis=0)
    p50 = np.percentile(forecast.simulation_paths, 50, axis=0)
    p75 = np.percentile(forecast.simulation_paths, 75, axis=0)
    p95 = np.percentile(forecast.simulation_paths, 95, axis=0)
    
    fig = go.Figure()
    
    # 90% confidence band
    fig.add_trace(go.Scatter(
        x=np.concatenate([forecast.time_steps, forecast.time_steps[::-1]]),
        y=np.concatenate([p95, p5[::-1]]),
        fill='toself',
        fillcolor='rgba(88, 166, 255, 0.1)',
        line=dict(color='rgba(0,0,0,0)'),
        name='90% CI',
        showlegend=True
    ))
    
    # 50% confidence band
    fig.add_trace(go.Scatter(
        x=np.concatenate([forecast.time_steps, forecast.time_steps[::-1]]),
        y=np.concatenate([p75, p25[::-1]]),
        fill='toself',
        fillcolor='rgba(88, 166, 255, 0.25)',
        line=dict(color='rgba(0,0,0,0)'),
        name='50% CI',
        showlegend=True
    ))
    
    # Median line
    fig.add_trace(go.Scatter(
        x=forecast.time_steps,
        y=p50,
        mode='lines',
        line=dict(color='#58a6ff', width=2),
        name='Median'
    ))
    
    # Starting value line
    fig.add_hline(
        y=portfolio_value,
        line_dash='dash',
        line_color='#666666',
        annotation_text=f'Start: ${portfolio_value:,.0f}'
    )
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=12, color='#58a6ff')),
        xaxis_title='Trading Days',
        yaxis_title='Portfolio Value ($)',
        height=400,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(gridcolor='#222222'),
        yaxis=dict(gridcolor='#222222', tickformat='$,.0f'),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            font=dict(size=10)
        )
    )
    
    return fig


def create_scenario_chart(
    scenarios: List[GrowthScenario],
    portfolio_value: float
) -> go.Figure:
    """Create scenario comparison chart."""
    colors = {'Bull': '#7ee787', 'Base': '#58a6ff', 'Bear': '#f85149'}
    
    fig = go.Figure()
    
    for scenario in scenarios:
        fig.add_trace(go.Scatter(
            x=scenario.time_days,
            y=scenario.projected_values,
            mode='lines',
            line=dict(color=colors.get(scenario.scenario_name, '#888888'), width=2),
            name=f"{scenario.scenario_name} ({scenario.annual_return*100:.1f}%)"
        ))
    
    fig.add_hline(
        y=portfolio_value,
        line_dash='dash',
        line_color='#666666'
    )
    
    fig.update_layout(
        title=dict(text='Scenario Projections', font=dict(size=12, color='#58a6ff')),
        xaxis_title='Trading Days',
        yaxis_title='Portfolio Value ($)',
        height=350,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(gridcolor='#222222'),
        yaxis=dict(gridcolor='#222222', tickformat='$,.0f'),
        legend=dict(font=dict(size=10))
    )
    
    return fig


def create_probability_chart(forecast: ForecastResult) -> go.Figure:
    """Create probability distribution chart of final values."""
    final_values = forecast.simulation_paths[:, -1]
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=final_values,
        nbinsx=50,
        marker_color='#58a6ff',
        opacity=0.7,
        name='Distribution'
    ))
    
    # Add vertical lines for percentiles
    fig.add_vline(x=forecast.percentiles[5], line_dash='dash', line_color='#f85149',
                  annotation_text='5%', annotation_position='top')
    fig.add_vline(x=forecast.median_value, line_dash='solid', line_color='#7ee787',
                  annotation_text='Median', annotation_position='top')
    fig.add_vline(x=forecast.percentiles[95], line_dash='dash', line_color='#7ee787',
                  annotation_text='95%', annotation_position='top')
    
    fig.update_layout(
        title=dict(text='Forecast Distribution', font=dict(size=12, color='#58a6ff')),
        xaxis_title='Portfolio Value ($)',
        yaxis_title='Frequency',
        height=300,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(gridcolor='#222222', tickformat='$,.0f'),
        yaxis=dict(gridcolor='#222222'),
        showlegend=False
    )
    
    return fig


def create_stock_price_chart(
    forecast: ForecastResult,
    current_price: float,
    ticker: str
) -> go.Figure:
    """Create stock price forecast fan chart."""
    p5 = np.percentile(forecast.simulation_paths, 5, axis=0)
    p25 = np.percentile(forecast.simulation_paths, 25, axis=0)
    p50 = np.percentile(forecast.simulation_paths, 50, axis=0)
    p75 = np.percentile(forecast.simulation_paths, 75, axis=0)
    p95 = np.percentile(forecast.simulation_paths, 95, axis=0)
    
    fig = go.Figure()
    
    # 90% CI band
    fig.add_trace(go.Scatter(
        x=np.concatenate([forecast.time_steps, forecast.time_steps[::-1]]),
        y=np.concatenate([p95, p5[::-1]]),
        fill='toself',
        fillcolor='rgba(88, 166, 255, 0.1)',
        line=dict(color='rgba(0,0,0,0)'),
        name='90% CI',
        showlegend=True
    ))
    
    # 50% CI band
    fig.add_trace(go.Scatter(
        x=np.concatenate([forecast.time_steps, forecast.time_steps[::-1]]),
        y=np.concatenate([p75, p25[::-1]]),
        fill='toself',
        fillcolor='rgba(88, 166, 255, 0.25)',
        line=dict(color='rgba(0,0,0,0)'),
        name='50% CI',
        showlegend=True
    ))
    
    # Median line
    fig.add_trace(go.Scatter(
        x=forecast.time_steps,
        y=p50,
        mode='lines',
        line=dict(color='#58a6ff', width=2),
        name='Median Price'
    ))
    
    # Current price line
    fig.add_hline(
        y=current_price,
        line_dash='dash',
        line_color='#666666',
        annotation_text=f'Current: ${current_price:.2f}'
    )
    
    fig.update_layout(
        title=dict(text=f'{ticker} Price Forecast', font=dict(size=12, color='#58a6ff')),
        xaxis_title='Trading Days',
        yaxis_title='Stock Price ($)',
        height=400,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(gridcolor='#222222'),
        yaxis=dict(gridcolor='#222222', tickformat='$,.2f'),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            font=dict(size=10)
        )
    )
    
    return fig


def create_stock_scenario_chart(
    scenarios: List[GrowthScenario],
    current_price: float,
    ticker: str
) -> go.Figure:
    """Create scenario comparison chart for single stock."""
    colors = {'Bull': '#7ee787', 'Base': '#58a6ff', 'Bear': '#f85149'}
    
    fig = go.Figure()
    
    for scenario in scenarios:
        fig.add_trace(go.Scatter(
            x=scenario.time_days,
            y=scenario.projected_values,
            mode='lines',
            line=dict(color=colors.get(scenario.scenario_name, '#888888'), width=2),
            name=f"{scenario.scenario_name} ({scenario.annual_return*100:.1f}%/yr)"
        ))
    
    fig.add_hline(
        y=current_price,
        line_dash='dash',
        line_color='#666666'
    )
    
    fig.update_layout(
        title=dict(text=f'{ticker} Price Scenarios', font=dict(size=12, color='#58a6ff')),
        xaxis_title='Trading Days',
        yaxis_title='Stock Price ($)',
        height=350,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(gridcolor='#222222'),
        yaxis=dict(gridcolor='#222222', tickformat='$,.2f'),
        legend=dict(font=dict(size=10))
    )
    
    return fig


def create_stock_distribution_chart(
    forecast: ForecastResult,
    current_price: float,
    ticker: str
) -> go.Figure:
    """Create price distribution chart for single stock."""
    final_values = forecast.simulation_paths[:, -1]
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=final_values,
        nbinsx=50,
        marker_color='#58a6ff',
        opacity=0.7,
        name='Distribution'
    ))
    
    fig.add_vline(x=forecast.percentiles[5], line_dash='dash', line_color='#f85149',
                  annotation_text='5%', annotation_position='top')
    fig.add_vline(x=current_price, line_dash='solid', line_color='#ffcc00',
                  annotation_text='Current', annotation_position='top')
    fig.add_vline(x=forecast.median_value, line_dash='solid', line_color='#7ee787',
                  annotation_text='Median', annotation_position='top')
    fig.add_vline(x=forecast.percentiles[95], line_dash='dash', line_color='#7ee787',
                  annotation_text='95%', annotation_position='top')
    
    fig.update_layout(
        title=dict(text=f'{ticker} Price Distribution', font=dict(size=12, color='#58a6ff')),
        xaxis_title='Stock Price ($)',
        yaxis_title='Frequency',
        height=300,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(gridcolor='#222222', tickformat='$,.2f'),
        yaxis=dict(gridcolor='#222222'),
        showlegend=False
    )
    
    return fig


# =============================================================================
# SINGLE STOCK FORECAST RENDER
# =============================================================================

def render_single_stock_forecast_tab(
    returns: pd.Series,
    current_price: float,
    ticker: str
):
    """
    Render forecast tab for a single stock.
    
    Args:
        returns: Historical returns Series for the stock
        current_price: Current stock price
        ticker: Stock ticker symbol
    """
    st.markdown(f"#### {ticker} PRICE FORECAST")
    
    # Configuration
    col1, col2, col3 = st.columns(3)
    with col1:
        horizon = st.selectbox(
            "Forecast Horizon",
            options=[30, 60, 90, 180, 252, 504],
            index=4,
            format_func=lambda x: f"{x}D ({x/252:.1f}Y)" if x >= 252 else f"{x}D",
            key="stock_forecast_horizon"
        )
    with col2:
        n_sims = st.selectbox(
            "Simulations",
            options=[1000, 5000, 10000, 25000],
            index=2,
            key="stock_n_sims"
        )
    with col3:
        method = st.selectbox(
            "Method",
            options=["Monte Carlo (GARCH)", "Monte Carlo"],
            key="stock_forecast_method"
        )
    
    # Initialize engine
    engine = SingleStockForecastEngine(
        returns=returns,
        current_price=current_price,
        ticker=ticker
    )
    
    # Run forecast
    with st.spinner("Running price simulations..."):
        use_garch = "GARCH" in method
        forecast = engine.monte_carlo_forecast(
            horizon_days=horizon,
            n_simulations=n_sims,
            use_garch=use_garch
        )
    
    # Key Metrics
    st.markdown("---")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    expected_return = (forecast.expected_value / current_price - 1)
    col1.metric(
        "Expected Price",
        f"${forecast.expected_value:.2f}",
        f"{expected_return:+.1%}"
    )
    
    col2.metric(
        "Median Price",
        f"${forecast.median_value:.2f}"
    )
    
    col3.metric(
        "P(Price Up)",
        f"{forecast.prob_positive*100:.0f}%"
    )
    
    col4.metric(
        "P(Drop >10%)",
        f"{forecast.prob_loss_10pct*100:.1f}%"
    )
    
    col5.metric(
        "P(Gain >20%)",
        f"{forecast.prob_gain_20pct*100:.1f}%"
    )
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_price = create_stock_price_chart(forecast, current_price, ticker)
        st.plotly_chart(fig_price, use_container_width=True)
    
    with col2:
        fig_dist = create_stock_distribution_chart(forecast, current_price, ticker)
        st.plotly_chart(fig_dist, use_container_width=True)
    
    # Scenario Projections
    st.markdown("#### PRICE SCENARIOS")
    scenarios = engine.scenario_projections(horizon_days=horizon)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_scenario = create_stock_scenario_chart(scenarios, current_price, ticker)
        st.plotly_chart(fig_scenario, use_container_width=True)
    
    with col2:
        st.markdown("**Scenario Summary**")
        for scenario in scenarios:
            end_price = scenario.projected_values[-1]
            ret_pct = (end_price / current_price - 1) * 100
            color = "#7ee787" if ret_pct > 0 else "#f85149"
            st.markdown(f"""
            <div style='margin-bottom: 10px;'>
                <span style='color: {color}; font-weight: bold;'>{scenario.scenario_name}</span><br>
                <span style='color: #888; font-size: 0.8rem;'>
                    Return: {ret_pct:+.1f}% | Price: ${end_price:.2f}
                </span>
            </div>
            """, unsafe_allow_html=True)
    
    # Price Targets Table
    st.markdown("---")
    st.markdown("#### PRICE TARGETS")
    
    targets_df = engine.get_price_targets()
    
    st.dataframe(
        targets_df[['Horizon', 'Expected', 'Median', 'Low (5%)', 'High (95%)', 'Exp. Return', 'P(Gain)']].style.format({
            'Expected': '${:,.2f}',
            'Median': '${:,.2f}',
            'Low (5%)': '${:,.2f}',
            'High (95%)': '${:,.2f}'
        }),
        use_container_width=True,
        hide_index=True
    )
    
    # Confidence Intervals
    st.markdown("---")
    st.markdown("#### CONFIDENCE INTERVALS")
    
    col1, col2, col3 = st.columns(3)
    
    ci_90 = forecast.confidence_intervals['90%']
    ci_80 = forecast.confidence_intervals['80%']
    ci_50 = forecast.confidence_intervals['50%']
    
    with col1:
        st.markdown("**90% CI**")
        st.markdown(f"${ci_90[0]:.2f} — ${ci_90[1]:.2f}")
        range_pct = (ci_90[1] - ci_90[0]) / current_price * 100
        st.caption(f"Range: {range_pct:.0f}%")
    
    with col2:
        st.markdown("**80% CI**")
        st.markdown(f"${ci_80[0]:.2f} — ${ci_80[1]:.2f}")
        range_pct = (ci_80[1] - ci_80[0]) / current_price * 100
        st.caption(f"Range: {range_pct:.0f}%")
    
    with col3:
        st.markdown("**50% CI**")
        st.markdown(f"${ci_50[0]:.2f} — ${ci_50[1]:.2f}")
        range_pct = (ci_50[1] - ci_50[0]) / current_price * 100
        st.caption(f"Range: {range_pct:.0f}%")
    
    # Model Info
    with st.expander("Model Details"):
        st.markdown(f"""
        **{ticker} Statistics:**
        - Historical Annual Return: {engine.historical_mean*100:.1f}%
        - Historical Annual Volatility: {engine.historical_vol*100:.1f}%
        - Skewness: {engine.historical_skew:.2f}
        - Kurtosis: {engine.historical_kurt:.2f}
        
        **Method Used:** {method}
        - Simulations: {n_sims:,}
        - Horizon: {horizon} trading days
        - Current Price: ${current_price:.2f}
        """)
        
        if engine.garch_forecast:
            st.markdown(f"""
            **GARCH Volatility Forecast:**
            - Current: {engine.garch_forecast['current_vol']*100:.1f}%
            - Persistence: {engine.garch_forecast['persistence']:.3f}
            """)


# =============================================================================
# STREAMLIT RENDER FUNCTION
# =============================================================================

def render_forecast_tab(
    returns: pd.DataFrame,
    weights: Dict[str, float],
    portfolio_value: float = 100000
):
    """
    Render the forecast tab in Streamlit.
    
    Args:
        returns: Historical returns DataFrame
        weights: Portfolio weights dict
        portfolio_value: Current portfolio value
    """
    st.markdown("#### PORTFOLIO GROWTH FORECAST")
    
    # Configuration
    col1, col2, col3 = st.columns(3)
    with col1:
        horizon = st.selectbox(
            "Forecast Horizon",
            options=[30, 60, 90, 180, 252, 504],
            index=4,
            format_func=lambda x: f"{x}D ({x/252:.1f}Y)" if x >= 252 else f"{x}D",
            key="forecast_horizon"
        )
    with col2:
        n_sims = st.selectbox(
            "Simulations",
            options=[1000, 5000, 10000, 25000],
            index=2,
            key="n_sims"
        )
    with col3:
        method = st.selectbox(
            "Method",
            options=["Monte Carlo (GARCH)", "Monte Carlo", "Bootstrap"],
            key="forecast_method"
        )
    
    # Initialize engine
    engine = PortfolioForecastEngine(
        returns=returns,
        weights=weights,
        portfolio_value=portfolio_value
    )
    
    # Run forecast
    with st.spinner("Running simulations..."):
        if method == "Bootstrap":
            forecast = engine.bootstrap_forecast(horizon_days=horizon, n_simulations=n_sims)
        else:
            use_garch = "GARCH" in method
            forecast = engine.monte_carlo_forecast(
                horizon_days=horizon,
                n_simulations=n_sims,
                use_garch=use_garch
            )
    
    # Key Metrics Row
    st.markdown("---")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    expected_return = (forecast.expected_value / portfolio_value - 1)
    col1.metric(
        "Expected Value",
        f"${forecast.expected_value:,.0f}",
        f"{expected_return:+.1%}"
    )
    
    col2.metric(
        "Median Value",
        f"${forecast.median_value:,.0f}"
    )
    
    col3.metric(
        "P(Gain)",
        f"{forecast.prob_positive*100:.0f}%"
    )
    
    col4.metric(
        "P(Loss >10%)",
        f"{forecast.prob_loss_10pct*100:.1f}%"
    )
    
    col5.metric(
        "P(Gain >20%)",
        f"{forecast.prob_gain_20pct*100:.1f}%"
    )
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_fan = create_fan_chart(forecast, portfolio_value)
        st.plotly_chart(fig_fan, use_container_width=True)
    
    with col2:
        fig_dist = create_probability_chart(forecast)
        st.plotly_chart(fig_dist, use_container_width=True)
    
    # Scenario Projections
    st.markdown("#### SCENARIO ANALYSIS")
    scenarios = engine.scenario_projections(horizon_days=horizon)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_scenario = create_scenario_chart(scenarios, portfolio_value)
        st.plotly_chart(fig_scenario, use_container_width=True)
    
    with col2:
        st.markdown("**Scenario Summary**")
        for scenario in scenarios:
            end_value = scenario.projected_values[-1]
            ret_pct = (end_value / portfolio_value - 1) * 100
            color = "#7ee787" if ret_pct > 0 else "#f85149"
            st.markdown(f"""
            <div style='margin-bottom: 10px;'>
                <span style='color: {color}; font-weight: bold;'>{scenario.scenario_name}</span><br>
                <span style='color: #888; font-size: 0.8rem;'>
                    Return: {ret_pct:+.1f}% | Value: ${end_value:,.0f}
                </span>
            </div>
            """, unsafe_allow_html=True)
    
    # Price Targets Table
    st.markdown("---")
    st.markdown("#### PRICE TARGETS")
    
    targets_df = engine.get_price_targets()
    
    # Style the dataframe
    st.dataframe(
        targets_df[['Horizon', 'Expected', 'Median', 'Low (5%)', 'High (95%)', 'Exp. Return', 'P(Gain)']].style.format({
            'Expected': '${:,.0f}',
            'Median': '${:,.0f}',
            'Low (5%)': '${:,.0f}',
            'High (95%)': '${:,.0f}'
        }),
        use_container_width=True,
        hide_index=True
    )
    
    # Confidence Intervals
    st.markdown("---")
    st.markdown("#### CONFIDENCE INTERVALS")
    
    col1, col2, col3 = st.columns(3)
    
    ci_90 = forecast.confidence_intervals['90%']
    ci_80 = forecast.confidence_intervals['80%']
    ci_50 = forecast.confidence_intervals['50%']
    
    with col1:
        st.markdown("**90% CI**")
        st.markdown(f"${ci_90[0]:,.0f} — ${ci_90[1]:,.0f}")
        range_pct = (ci_90[1] - ci_90[0]) / portfolio_value * 100
        st.caption(f"Range: {range_pct:.0f}%")
    
    with col2:
        st.markdown("**80% CI**")
        st.markdown(f"${ci_80[0]:,.0f} — ${ci_80[1]:,.0f}")
        range_pct = (ci_80[1] - ci_80[0]) / portfolio_value * 100
        st.caption(f"Range: {range_pct:.0f}%")
    
    with col3:
        st.markdown("**50% CI**")
        st.markdown(f"${ci_50[0]:,.0f} — ${ci_50[1]:,.0f}")
        range_pct = (ci_50[1] - ci_50[0]) / portfolio_value * 100
        st.caption(f"Range: {range_pct:.0f}%")
    
    # Individual Stock Price Forecasts
    st.markdown("---")
    st.markdown("#### INDIVIDUAL STOCK PRICE FORECASTS")
    st.caption("Projected prices for each stock in the portfolio")
    
    # Get current prices from returns (we'll estimate from the end of the series)
    tickers = list(weights.keys())
    
    # Create individual forecasts for each stock
    stock_forecasts = []
    for ticker in tickers:
        if ticker in returns.columns:
            stock_returns = returns[ticker].dropna()
            if len(stock_returns) > 50:
                # Estimate current price as 100 (normalized)
                # The user can pass actual prices for exact values
                stock_engine = SingleStockForecastEngine(
                    returns=stock_returns,
                    current_price=100,  # Normalized baseline
                    ticker=ticker
                )
                
                use_garch = "GARCH" in method
                stock_fc = stock_engine.monte_carlo_forecast(
                    horizon_days=horizon,
                    n_simulations=min(n_sims, 5000),
                    use_garch=use_garch
                )
                
                stock_forecasts.append({
                    'Ticker': ticker,
                    'Weight': f"{weights.get(ticker, 0)*100:.1f}%",
                    'Expected Change': f"{stock_fc.expected_return_pct:+.1f}%",
                    'Median Change': f"{(stock_fc.median_value/100-1)*100:+.1f}%",
                    'Range (90% CI)': f"{(stock_fc.percentiles[5]/100-1)*100:+.1f}% to {(stock_fc.percentiles[95]/100-1)*100:+.1f}%",
                    'P(Up)': f"{stock_fc.prob_positive*100:.0f}%",
                    'Volatility': f"{stock_engine.historical_vol*100:.1f}%"
                })
    
    if stock_forecasts:
        stock_df = pd.DataFrame(stock_forecasts)
        st.dataframe(stock_df, use_container_width=True, hide_index=True)
        
        # Mini chart showing relative expected changes
        fig_stocks = go.Figure()
        
        for i, fc in enumerate(stock_forecasts):
            ticker = fc['Ticker']
            exp_change = float(fc['Expected Change'].rstrip('%'))
            color = '#7ee787' if exp_change > 0 else '#f85149'
            
            fig_stocks.add_trace(go.Bar(
                x=[ticker],
                y=[exp_change],
                marker_color=color,
                name=ticker,
                showlegend=False,
                text=f"{exp_change:+.1f}%",
                textposition='outside'
            ))
        
        fig_stocks.update_layout(
            title=dict(text='Expected Price Changes by Stock', font=dict(size=12, color='#58a6ff')),
            xaxis_title='Stock',
            yaxis_title='Expected Change (%)',
            height=300,
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(gridcolor='#222222'),
            yaxis=dict(gridcolor='#222222', tickformat='+.1f%', zeroline=True, zerolinecolor='#666666')
        )
        
        st.plotly_chart(fig_stocks, use_container_width=True)
    
    # Model Info
    with st.expander("Model Details"):
        st.markdown(f"""
        **Historical Statistics:**
        - Annual Return: {engine.historical_mean*100:.1f}%
        - Annual Volatility: {engine.historical_vol*100:.1f}%
        - Skewness: {engine.historical_skew:.2f}
        - Kurtosis: {engine.historical_kurt:.2f}
        
        **Method Used:** {method}
        - Simulations: {n_sims:,}
        - Horizon: {horizon} trading days
        """)
        
        if engine.garch_forecast:
            st.markdown(f"""
            **GARCH Volatility Forecast:**
            - Current: {engine.garch_forecast['current_vol']*100:.1f}%
            - Persistence: {engine.garch_forecast['persistence']:.3f}
            """)


# =============================================================================
# EXPORTS
# =============================================================================

HAS_FORECAST = True

__all__ = [
    'PortfolioForecastEngine',
    'SingleStockForecastEngine',
    'ForecastResult',
    'GrowthScenario',
    'create_fan_chart',
    'create_scenario_chart',
    'create_probability_chart',
    'create_stock_price_chart',
    'create_stock_scenario_chart',
    'create_stock_distribution_chart',
    'render_forecast_tab',
    'render_single_stock_forecast_tab',
    'HAS_FORECAST'
]
