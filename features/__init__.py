# Features Module
from .alerts import AlertManager
from .reports import ReportGenerator
from .options import OptionsAnalytics
from .fundamentals import FundamentalAnalyzer
from .comparison import StockComparison

# Sentiment Analysis (optional)
try:
    from .sentiment_analysis import (
        SentimentVaR,
        render_sentiment_tab,
        render_portfolio_sentiment,
        create_sentiment_gauge,
        create_sentiment_trend_chart,
        create_news_sentiment_breakdown,
        create_var_comparison_chart as create_sentiment_var_chart,
        create_whale_activity_chart,
        create_portfolio_sentiment_heatmap
    )
    HAS_SENTIMENT_FEATURE = True
except ImportError:
    HAS_SENTIMENT_FEATURE = False

# Digital Twin (optional)
try:
    from .digital_twin import (
        DigitalTwinEngine,
        ScenarioResult,
        PortfolioHealthScore,
        render_digital_twin_tab,
        render_quick_scenario_widget,
        create_scenario_comparison_chart,
        create_fan_chart,
        create_health_gauge,
        create_correlation_heatmap,
        create_correlation_change_chart
    )
    HAS_DIGITAL_TWIN = True
except ImportError:
    HAS_DIGITAL_TWIN = False

# What-If Analysis (optional)
try:
    from .what_if import (
        WhatIfAnalyzer,
        WhatIfScenario,
        PortfolioConstraints,
        render_what_if_tab,
        render_scenario_builder,
        create_metrics_comparison_chart,
        create_weight_comparison_chart,
        create_trade_table_chart,
        create_frontier_with_scenarios
    )
    HAS_WHAT_IF = True
except ImportError:
    HAS_WHAT_IF = False

__all__ = [
    'AlertManager', 
    'ReportGenerator', 
    'OptionsAnalytics', 
    'FundamentalAnalyzer',
    'StockComparison',
    # Sentiment
    'SentimentVaR',
    'render_sentiment_tab',
    'render_portfolio_sentiment',
    'HAS_SENTIMENT_FEATURE',
    # Digital Twin
    'DigitalTwinEngine',
    'ScenarioResult',
    'PortfolioHealthScore',
    'render_digital_twin_tab',
    'render_quick_scenario_widget',
    'HAS_DIGITAL_TWIN',
    # What-If
    'WhatIfAnalyzer',
    'WhatIfScenario',
    'render_what_if_tab',
    'render_scenario_builder',
    'HAS_WHAT_IF'
]
