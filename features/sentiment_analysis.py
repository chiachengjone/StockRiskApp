"""
Sentiment Analysis Feature
===========================
Streamlit UI components for sentiment analysis and Sentiment VaR.

Features:
- Sentiment dashboard with news feed
- Sentiment-adjusted VaR calculations
- Whale tracking visualization
- Portfolio sentiment heatmap
- Trend analysis charts

Author: Stock Risk App | Feb 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# Try to import sentiment service
try:
    from services.sentiment_service import (
        SentimentService, 
        SentimentResult,
        SentimentLabel,
        NewsArticle,
        WhaleActivity
    )
    HAS_SENTIMENT = True
except ImportError:
    HAS_SENTIMENT = False
    logger.warning("Sentiment service not available")


# ============================================================================
# CONFIGURATION
# ============================================================================

SENTIMENT_COLORS = {
    'very_positive': '#00C853',
    'positive': '#4CAF50',
    'neutral': '#9E9E9E',
    'negative': '#FF5722',
    'very_negative': '#D50000'
}

CHART_TEMPLATE = 'plotly_dark'


# ============================================================================
# SENTIMENT VAR CALCULATOR
# ============================================================================

class SentimentVaR:
    """
    Calculate Sentiment-adjusted Value at Risk.
    
    Combines traditional VaR with sentiment analysis to provide
    a more forward-looking risk estimate.
    """
    
    def __init__(self, sentiment_service: 'SentimentService' = None):
        self.sentiment_service = sentiment_service
    
    def calculate(
        self,
        returns: pd.Series,
        ticker: str,
        confidence: float = 0.95,
        horizon: int = 1,
        base_var_method: str = 'historical'
    ) -> Dict:
        """
        Calculate sentiment-adjusted VaR.
        
        Args:
            returns: Historical returns series
            ticker: Stock symbol for sentiment lookup
            confidence: VaR confidence level
            horizon: VaR horizon in days
            base_var_method: Method for base VaR calculation
        
        Returns:
            Dictionary with VaR metrics
        """
        # Calculate base VaR
        if base_var_method == 'historical':
            base_var = float(np.percentile(returns, 100 * (1 - confidence)))
        elif base_var_method == 'parametric':
            from scipy.stats import norm
            mu = float(returns.mean()) * horizon
            sigma = float(returns.std()) * np.sqrt(horizon)
            base_var = float(norm.ppf(1 - confidence, mu, sigma))
        else:
            base_var = float(np.percentile(returns, 100 * (1 - confidence)))
        
        # Get sentiment adjustment
        adjustment_factor = 1.0
        sentiment_data = {}
        
        if self.sentiment_service:
            try:
                sentiment = self.sentiment_service.analyze_ticker(ticker)
                adjustment_factor = sentiment.sentiment_var_adjustment
                sentiment_data = {
                    'score': sentiment.overall_score,
                    'label': sentiment.overall_label.value,
                    'news_count': sentiment.news_count,
                    'trending': sentiment.trending_score,
                    'whale_score': sentiment.whale_activity_score
                }
            except Exception as e:
                logger.warning(f"Sentiment analysis failed: {e}")
        
        # Apply adjustment
        adjusted_var = base_var * adjustment_factor
        
        # Calculate CVaR (Expected Shortfall)
        base_cvar = float(returns[returns <= base_var].mean())
        adjusted_cvar = base_cvar * adjustment_factor
        
        return {
            'ticker': ticker,
            'confidence': confidence,
            'horizon': horizon,
            'base_var': abs(base_var),
            'adjusted_var': abs(adjusted_var),
            'base_cvar': abs(base_cvar),
            'adjusted_cvar': abs(adjusted_cvar),
            'adjustment_factor': adjustment_factor,
            'sentiment': sentiment_data,
            'method': base_var_method
        }
    
    def calculate_portfolio_var(
        self,
        returns_df: pd.DataFrame,
        weights: Dict[str, float],
        confidence: float = 0.95
    ) -> Dict:
        """
        Calculate sentiment-adjusted portfolio VaR.
        
        Args:
            returns_df: DataFrame of asset returns
            weights: Portfolio weights
            confidence: VaR confidence level
        
        Returns:
            Portfolio-level VaR with sentiment adjustment
        """
        # Calculate portfolio returns
        weight_array = np.array([weights.get(col, 0) for col in returns_df.columns])
        portfolio_returns = (returns_df * weight_array).sum(axis=1)
        
        # Base portfolio VaR
        base_var = float(np.percentile(portfolio_returns, 100 * (1 - confidence)))
        
        # Get per-asset sentiment and calculate weighted adjustment
        adjustments = []
        sentiment_details = {}
        
        if self.sentiment_service:
            for ticker in returns_df.columns:
                try:
                    sentiment = self.sentiment_service.analyze_ticker(ticker)
                    adjustments.append(sentiment.sentiment_var_adjustment * weights.get(ticker, 0))
                    sentiment_details[ticker] = {
                        'score': sentiment.overall_score,
                        'label': sentiment.overall_label.value,
                        'adjustment': sentiment.sentiment_var_adjustment
                    }
                except Exception:
                    adjustments.append(1.0 * weights.get(ticker, 0))
        
        # Portfolio-level adjustment
        if adjustments:
            portfolio_adjustment = sum(adjustments)
        else:
            portfolio_adjustment = 1.0
        
        adjusted_var = base_var * portfolio_adjustment
        
        return {
            'base_var': abs(base_var),
            'adjusted_var': abs(adjusted_var),
            'adjustment_factor': portfolio_adjustment,
            'per_asset_sentiment': sentiment_details,
            'confidence': confidence
        }


# ============================================================================
# VISUALIZATION COMPONENTS
# ============================================================================

def create_sentiment_gauge(score: float, label: str) -> go.Figure:
    """Create a gauge chart for sentiment score."""
    
    # Determine color based on score
    if score > 0.3:
        color = SENTIMENT_COLORS['very_positive']
    elif score > 0:
        color = SENTIMENT_COLORS['positive']
    elif score > -0.3:
        color = SENTIMENT_COLORS['negative']
    else:
        color = SENTIMENT_COLORS['very_negative']
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Sentiment Score", 'font': {'size': 16}},
        gauge={
            'axis': {'range': [-1, 1], 'tickwidth': 1},
            'bar': {'color': color},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "rgba(255,255,255,0.3)",
            'steps': [
                {'range': [-1, -0.3], 'color': 'rgba(213, 0, 0, 0.3)'},
                {'range': [-0.3, 0], 'color': 'rgba(255, 87, 34, 0.3)'},
                {'range': [0, 0.3], 'color': 'rgba(76, 175, 80, 0.3)'},
                {'range': [0.3, 1], 'color': 'rgba(0, 200, 83, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        height=250
    )
    
    return fig


def create_sentiment_trend_chart(
    sentiment_df: pd.DataFrame,
    ticker: str
) -> go.Figure:
    """Create sentiment trend chart over time."""
    
    fig = go.Figure()
    
    # Add sentiment line
    fig.add_trace(go.Scatter(
        x=sentiment_df.index,
        y=sentiment_df['sentiment'],
        mode='lines+markers',
        name='Daily Sentiment',
        line=dict(color='#2196F3', width=2),
        marker=dict(size=6)
    ))
    
    # Add moving average
    if 'sentiment_ma' in sentiment_df.columns:
        fig.add_trace(go.Scatter(
            x=sentiment_df.index,
            y=sentiment_df['sentiment_ma'],
            mode='lines',
            name='7-Day MA',
            line=dict(color='#FF9800', width=2, dash='dash')
        ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    # Add colored regions
    fig.add_hrect(
        y0=0, y1=1, 
        fillcolor="rgba(76, 175, 80, 0.1)", 
        line_width=0,
        layer="below"
    )
    fig.add_hrect(
        y0=-1, y1=0, 
        fillcolor="rgba(244, 67, 54, 0.1)", 
        line_width=0,
        layer="below"
    )
    
    fig.update_layout(
        title=f"Sentiment Trend: {ticker}",
        xaxis_title="Date",
        yaxis_title="Sentiment Score",
        yaxis=dict(range=[-1.1, 1.1]),
        template=CHART_TEMPLATE,
        height=350,
        legend=dict(orientation="h", y=1.1)
    )
    
    return fig


def create_news_sentiment_breakdown(result: 'SentimentResult') -> go.Figure:
    """Create pie chart of news sentiment breakdown."""
    
    labels = ['Positive', 'Neutral', 'Negative']
    values = [result.positive_count, result.neutral_count, result.negative_count]
    colors = [SENTIMENT_COLORS['positive'], SENTIMENT_COLORS['neutral'], SENTIMENT_COLORS['negative']]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker_colors=colors,
        textinfo='label+percent',
        textposition='outside'
    )])
    
    fig.update_layout(
        title="News Sentiment Distribution",
        template=CHART_TEMPLATE,
        height=300,
        showlegend=False
    )
    
    return fig


def create_var_comparison_chart(var_data: Dict) -> go.Figure:
    """Create bar chart comparing base vs adjusted VaR."""
    
    fig = go.Figure()
    
    categories = ['VaR', 'CVaR']
    base_values = [var_data['base_var'] * 100, var_data['base_cvar'] * 100]
    adjusted_values = [var_data['adjusted_var'] * 100, var_data['adjusted_cvar'] * 100]
    
    fig.add_trace(go.Bar(
        name='Base (Traditional)',
        x=categories,
        y=base_values,
        marker_color='#2196F3'
    ))
    
    fig.add_trace(go.Bar(
        name='Sentiment-Adjusted',
        x=categories,
        y=adjusted_values,
        marker_color='#FF9800'
    ))
    
    fig.update_layout(
        title="VaR Comparison: Base vs Sentiment-Adjusted",
        yaxis_title="Risk (%)",
        barmode='group',
        template=CHART_TEMPLATE,
        height=300
    )
    
    return fig


def create_whale_activity_chart(activities: List['WhaleActivity']) -> go.Figure:
    """Create chart showing whale/insider activity."""
    
    if not activities:
        fig = go.Figure()
        fig.add_annotation(
            text="No recent whale activity detected",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            template=CHART_TEMPLATE,
            height=250
        )
        return fig
    
    # Prepare data
    dates = []
    values = []
    colors = []
    labels = []
    
    for activity in activities:
        dates.append(activity.filing_date or datetime.now())
        values.append(activity.value_usd / 1_000_000)  # In millions
        colors.append('#00C853' if activity.activity_type == 'buy' else '#FF1744')
        labels.append(f"{activity.entity_name}: ${activity.value_usd:,.0f}")
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=list(range(len(activities))),
        y=values,
        marker_color=colors,
        text=[a.activity_type.upper() for a in activities],
        textposition='outside',
        hovertemplate="%{text}<br>$%{y:.2f}M<extra></extra>"
    ))
    
    fig.update_layout(
        title="Recent Whale/Insider Activity",
        yaxis_title="Value ($M)",
        xaxis_title="Recent Transactions",
        template=CHART_TEMPLATE,
        height=250,
        showlegend=False
    )
    
    return fig


def create_portfolio_sentiment_heatmap(
    portfolio_sentiment: Dict
) -> go.Figure:
    """Create heatmap of portfolio sentiment scores."""
    
    ticker_data = portfolio_sentiment.get('ticker_sentiments', {})
    
    if not ticker_data:
        fig = go.Figure()
        fig.add_annotation(
            text="No sentiment data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    tickers = list(ticker_data.keys())
    scores = [ticker_data[t]['overall_score'] for t in tickers]
    labels = [ticker_data[t]['overall_label'] for t in tickers]
    
    fig = go.Figure(data=go.Heatmap(
        z=[scores],
        x=tickers,
        y=['Sentiment'],
        colorscale='RdYlGn',
        zmid=0,
        zmin=-1,
        zmax=1,
        text=[[f"{s:.2f}" for s in scores]],
        texttemplate="%{text}",
        textfont={"size": 12},
        hovertemplate="<b>%{x}</b><br>Score: %{z:.3f}<extra></extra>",
        colorbar=dict(title="Score")
    ))
    
    fig.update_layout(
        title="Portfolio Sentiment Heatmap",
        template=CHART_TEMPLATE,
        height=200
    )
    
    return fig


# ============================================================================
# STREAMLIT COMPONENTS
# ============================================================================

def render_sentiment_tab(
    service: 'SentimentService',
    ticker: str,
    returns: pd.Series = None
):
    """
    Render the sentiment analysis tab in Streamlit.
    
    Args:
        service: SentimentService instance
        ticker: Stock symbol
        returns: Historical returns for VaR calculation
    """
    st.subheader(f"📊 Sentiment Analysis: {ticker}")
    
    # Analyze sentiment
    with st.spinner("Analyzing sentiment..."):
        result = service.analyze_ticker(ticker, days_back=7)
    
    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        emoji = "🟢" if result.overall_score > 0.2 else "🔴" if result.overall_score < -0.2 else "🟡"
        st.metric(
            "Sentiment",
            f"{emoji} {result.overall_label.value.replace('_', ' ').title()}"
        )
    
    with col2:
        st.metric("Score", f"{result.overall_score:.3f}")
    
    with col3:
        trend_icon = "📈" if result.trending_score > 0 else "📉" if result.trending_score < 0 else "➡️"
        st.metric("Trend", f"{trend_icon} {result.trending_score:+.3f}")
    
    with col4:
        adj_pct = (result.sentiment_var_adjustment - 1) * 100
        color = "normal" if adj_pct < 0 else "inverse"
        st.metric(
            "VaR Adjustment",
            f"{adj_pct:+.1f}%",
            delta=f"{'Lower' if adj_pct < 0 else 'Higher'} Risk"
        )
    
    st.divider()
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Sentiment Overview",
        "📰 News Feed",
        "🐋 Whale Tracking",
        "⚠️ Sentiment VaR"
    ])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Gauge chart
            gauge_fig = create_sentiment_gauge(
                result.overall_score,
                result.overall_label.value
            )
            st.plotly_chart(gauge_fig, use_container_width=True)
        
        with col2:
            # News breakdown pie
            breakdown_fig = create_news_sentiment_breakdown(result)
            st.plotly_chart(breakdown_fig, use_container_width=True)
        
        # Trend chart
        try:
            trend_df = service.get_trending_sentiment(ticker, lookback_days=30)
            trend_fig = create_sentiment_trend_chart(trend_df, ticker)
            st.plotly_chart(trend_fig, use_container_width=True)
        except Exception as e:
            st.info("Sentiment trend data not available")
    
    with tab2:
        st.subheader(f"Recent News ({result.news_count} articles)")
        
        if result.articles:
            for article in result.articles[:10]:
                # Sentiment indicator
                if article.sentiment_score > 0.2:
                    emoji = "🟢"
                    bg_color = "rgba(0, 200, 83, 0.1)"
                elif article.sentiment_score < -0.2:
                    emoji = "🔴"
                    bg_color = "rgba(213, 0, 0, 0.1)"
                else:
                    emoji = "🟡"
                    bg_color = "rgba(158, 158, 158, 0.1)"
                
                with st.container():
                    col1, col2 = st.columns([10, 2])
                    
                    with col1:
                        st.markdown(f"**{emoji} {article.title}**")
                        st.caption(
                            f"Source: {article.source} • "
                            f"Score: {article.sentiment_score:.2f} • "
                            f"Impact: {article.impact_score:.2f}"
                        )
                    
                    with col2:
                        if article.url:
                            st.link_button("Read", article.url)
                
                st.divider()
        else:
            st.info("No recent news articles found")
    
    with tab3:
        st.subheader("🐋 Whale & Insider Activity")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Whale score metric
            whale_score = result.whale_activity_score
            if whale_score > 0.3:
                whale_label = "Bullish Accumulation"
                whale_emoji = "🟢"
            elif whale_score < -0.3:
                whale_label = "Distribution"
                whale_emoji = "🔴"
            else:
                whale_label = "Neutral"
                whale_emoji = "🟡"
            
            st.metric(
                "Whale Activity Score",
                f"{whale_emoji} {whale_label}",
                delta=f"{whale_score:+.2f}"
            )
        
        with col2:
            # Activity summary
            buy_count = sum(1 for a in result.whale_activities if a.activity_type == 'buy')
            sell_count = sum(1 for a in result.whale_activities if a.activity_type == 'sell')
            st.metric("Recent Transactions", f"{buy_count} Buy / {sell_count} Sell")
        
        # Activity chart
        whale_fig = create_whale_activity_chart(result.whale_activities)
        st.plotly_chart(whale_fig, use_container_width=True)
        
        # Activity table
        if result.whale_activities:
            activity_data = [{
                'Type': a.activity_type.upper(),
                'Entity': a.entity_name[:30],
                'Value': f"${a.value_usd:,.0f}",
                'Shares': f"{a.shares:,}" if a.shares else "N/A"
            } for a in result.whale_activities]
            
            st.dataframe(pd.DataFrame(activity_data), use_container_width=True)
    
    with tab4:
        st.subheader("⚠️ Sentiment-Adjusted VaR")
        
        if returns is not None and len(returns) > 50:
            # Calculate Sentiment VaR
            sentiment_var = SentimentVaR(service)
            var_result = sentiment_var.calculate(
                returns=returns,
                ticker=ticker,
                confidence=0.95
            )
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Base VaR (95%)",
                    f"{var_result['base_var']*100:.2f}%"
                )
            
            with col2:
                st.metric(
                    "Sentiment VaR (95%)",
                    f"{var_result['adjusted_var']*100:.2f}%",
                    delta=f"{(var_result['adjusted_var']/var_result['base_var']-1)*100:+.1f}%"
                )
            
            with col3:
                st.metric(
                    "Adjustment Factor",
                    f"{var_result['adjustment_factor']:.3f}"
                )
            
            # Comparison chart
            var_fig = create_var_comparison_chart(var_result)
            st.plotly_chart(var_fig, use_container_width=True)
            
            # Explanation
            with st.expander("📖 What is Sentiment VaR?"):
                st.markdown("""
                **Sentiment VaR** adjusts traditional Value at Risk based on current market sentiment:
                
                - **Negative sentiment** → Increases VaR (higher expected losses)
                - **Positive sentiment** → Decreases VaR (lower expected losses)
                - **Whale activity** → Institutional buying/selling patterns influence risk
                
                The adjustment factor combines:
                - News sentiment score (-20% to +20% impact)
                - Sentiment trend (worsening/improving, ±10% impact)
                - Whale activity score (institutional signal, ±10% impact)
                
                **Formula:** `Sentiment VaR = Base VaR × Adjustment Factor`
                """)
        else:
            st.warning("Insufficient return data for VaR calculation. Need at least 50 observations.")


def render_portfolio_sentiment(
    service: 'SentimentService',
    tickers: List[str],
    weights: Dict[str, float] = None
):
    """
    Render portfolio-level sentiment analysis.
    
    Args:
        service: SentimentService instance
        tickers: List of portfolio tickers
        weights: Optional portfolio weights
    """
    st.subheader("📊 Portfolio Sentiment")
    
    with st.spinner("Analyzing portfolio sentiment..."):
        portfolio_result = service.analyze_portfolio(tickers, weights)
    
    # Portfolio-level metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        score = portfolio_result['portfolio_sentiment']
        emoji = "🟢" if score > 0.2 else "🔴" if score < -0.2 else "🟡"
        st.metric("Portfolio Sentiment", f"{emoji} {score:.3f}")
    
    with col2:
        st.metric(
            "Sentiment Label",
            portfolio_result['portfolio_label'].replace('_', ' ').title()
        )
    
    with col3:
        bullish = len(portfolio_result.get('bullish_tickers', []))
        st.metric("Bullish Positions", bullish)
    
    with col4:
        bearish = len(portfolio_result.get('bearish_tickers', []))
        st.metric("Bearish Positions", bearish)
    
    # Heatmap
    heatmap_fig = create_portfolio_sentiment_heatmap(portfolio_result)
    st.plotly_chart(heatmap_fig, use_container_width=True)
    
    # Per-asset breakdown
    st.subheader("Per-Asset Sentiment")
    
    cols = st.columns(min(len(tickers), 4))
    for i, ticker in enumerate(tickers):
        sentiment_data = portfolio_result['ticker_sentiments'].get(ticker, {})
        col_idx = i % len(cols)
        
        with cols[col_idx]:
            score = sentiment_data.get('overall_score', 0)
            label = sentiment_data.get('overall_label', 'neutral')
            
            emoji = "🟢" if score > 0.2 else "🔴" if score < -0.2 else "🟡"
            
            st.markdown(f"**{ticker}**")
            st.caption(f"{emoji} {label.replace('_', ' ').title()}")
            st.progress((score + 1) / 2)  # Normalize to 0-1
    
    # Top news
    st.subheader("📰 Top Portfolio News")
    
    articles = portfolio_result.get('top_articles', [])
    if articles:
        for article in articles[:5]:
            emoji = "🟢" if article.get('sentiment_score', 0) > 0.2 else "🔴" if article.get('sentiment_score', 0) < -0.2 else "🟡"
            tickers_str = ", ".join(article.get('tickers', [])[:3])
            
            st.markdown(f"{emoji} **{article.get('title', '')}**")
            st.caption(f"Tickers: {tickers_str} • Score: {article.get('sentiment_score', 0):.2f}")
            st.divider()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def create_sentiment_service_from_config() -> Optional['SentimentService']:
    """Create SentimentService from environment configuration."""
    import os
    
    try:
        from services.sentiment_service import SentimentService
        
        return SentimentService(
            polygon_key=os.getenv('POLYGON_API_KEY', ''),
            alpaca_key=os.getenv('ALPACA_API_KEY', ''),
            alpaca_secret=os.getenv('ALPACA_API_SECRET', '')
        )
    except ImportError:
        return None


def main():
    """Test the sentiment analysis feature."""
    st.set_page_config(page_title="Sentiment Analysis", layout="wide")
    st.title("Sentiment Analysis Test")
    
    service = create_sentiment_service_from_config()
    
    if service:
        ticker = st.text_input("Enter Ticker", "AAPL")
        
        if st.button("Analyze"):
            render_sentiment_tab(service, ticker)
    else:
        st.error("Sentiment service not configured. Check API keys.")


if __name__ == "__main__":
    main()
