"""
Utils Module - Enhanced Features for Stock Risk App
====================================================
Performance | Validation | Analytics | Portfolio | Visualization | Realtime
"""

from .performance import (
    cached_fetch_data, cached_fetch_info, cached_fetch_multiple,
    cached_garch, cached_monte_carlo, cached_optimization,
    ProgressTracker, performance_monitor, lazy_load
)

from .validation import (
    validate_ticker_robust, validate_returns_data, validate_portfolio_weights,
    fetch_with_retry, handle_missing_data, detect_outliers,
    verify_data_adjustments, check_data_freshness, DataValidator
)

from .analytics import (
    backtest_var_kupiec, backtest_var_christoffersen,
    regime_detection, time_varying_beta,
    covar_systemic_risk, rolling_correlation_breakdown,
    enhanced_stress_test, AnalyticsEngine
)

from .portfolio import (
    risk_parity_weights, hierarchical_risk_parity,
    black_litterman_optimization, TransactionCostModel,
    calculate_rebalance_costs, optimal_rebalance_frequency,
    threshold_rebalancing, tax_loss_harvesting_opportunities,
    PortfolioOptimizer, quick_portfolio_optimize,
    portfolio_risk_decomposition
)

from .visualization import (
    interactive_correlation_heatmap, rolling_correlation_chart,
    volatility_surface_3d, create_sample_vol_surface,
    animated_price_chart, cumulative_returns_chart,
    var_cone_chart, risk_contribution_chart, var_comparison_chart,
    performance_attribution_chart, rolling_performance_chart,
    make_chart_downloadable, figure_to_png_base64, get_download_link,
    factor_exposure_chart, regime_chart, VisualizationEngine,
    apply_dark_theme, DARK_THEME
)

from .realtime import (
    get_live_quote, is_market_open, calculate_live_pnl,
    PriceStream, RealtimeEngine, MARKET_SCHEDULES
)

__all__ = [
    # Performance
    'cached_fetch_data', 'cached_fetch_info', 'cached_fetch_multiple',
    'cached_garch', 'cached_monte_carlo', 'cached_optimization',
    'ProgressTracker', 'performance_monitor', 'lazy_load',
    
    # Validation
    'validate_ticker_robust', 'validate_returns_data', 'validate_portfolio_weights',
    'fetch_with_retry', 'handle_missing_data', 'detect_outliers',
    'verify_data_adjustments', 'check_data_freshness', 'DataValidator',
    
    # Analytics
    'backtest_var_kupiec', 'backtest_var_christoffersen',
    'regime_detection', 'time_varying_beta',
    'covar_systemic_risk', 'rolling_correlation_breakdown',
    'enhanced_stress_test', 'AnalyticsEngine',
    
    # Portfolio
    'risk_parity_weights', 'hierarchical_risk_parity',
    'black_litterman_optimization', 'TransactionCostModel',
    'calculate_rebalance_costs', 'optimal_rebalance_frequency',
    'threshold_rebalancing', 'tax_loss_harvesting_opportunities',
    'PortfolioOptimizer', 'quick_portfolio_optimize',
    'portfolio_risk_decomposition',
    
    # Visualization
    'interactive_correlation_heatmap', 'rolling_correlation_chart',
    'volatility_surface_3d', 'create_sample_vol_surface',
    'animated_price_chart', 'cumulative_returns_chart',
    'var_cone_chart', 'risk_contribution_chart', 'var_comparison_chart',
    'performance_attribution_chart', 'rolling_performance_chart',
    'make_chart_downloadable', 'figure_to_png_base64', 'get_download_link',
    'factor_exposure_chart', 'regime_chart', 'VisualizationEngine',
    'apply_dark_theme', 'DARK_THEME',
    
    # Realtime
    'get_live_quote', 'is_market_open', 'calculate_live_pnl',
    'PriceStream', 'RealtimeEngine', 'MARKET_SCHEDULES',
]

