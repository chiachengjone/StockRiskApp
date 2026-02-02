"""
Services Module
===============
Technical Analysis and Signal Generation Services

Author: Stock Risk App | Feb 2026
"""

from .ta_service import (
    TAService,
    calculate_sma,
    calculate_ema,
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_adx,
    calculate_stochastic,
    calculate_atr,
    get_all_indicators
)

from .signals_service import (
    SignalsService,
    Signal,
    SignalType,
    generate_ma_crossover_signals,
    generate_rsi_signals,
    generate_macd_signals,
    generate_bollinger_signals,
    calculate_combined_signal_score,
    filter_signals_by_risk
)

# Sentiment service (optional)
try:
    from .sentiment_service import (
        SentimentService,
        SentimentResult,
        NewsArticle,
        WhaleActivity,
        TextBlobAnalyzer,
        VADERAnalyzer,
        FinancialSentimentAnalyzer,
        NewsFetcher,
        WhaleTracker
    )
    HAS_SENTIMENT = True
except ImportError:
    HAS_SENTIMENT = False

__all__ = [
    # TA Service
    'TAService',
    'calculate_sma',
    'calculate_ema',
    'calculate_rsi',
    'calculate_macd',
    'calculate_bollinger_bands',
    'calculate_adx',
    'calculate_stochastic',
    'calculate_atr',
    'get_all_indicators',
    # Signals Service
    'SignalsService',
    'Signal',
    'SignalType',
    'generate_ma_crossover_signals',
    'generate_rsi_signals',
    'generate_macd_signals',
    'generate_bollinger_signals',
    'calculate_combined_signal_score',
    'filter_signals_by_risk',
    # Sentiment Service
    'SentimentService',
    'SentimentResult',
    'NewsArticle',
    'WhaleActivity',
    'HAS_SENTIMENT'
]
