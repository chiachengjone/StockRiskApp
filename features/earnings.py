"""
Earnings Calendar & Analysis Feature
=====================================
Track earnings dates, analyze historical surprises, and predict price reactions.

Features:
- Upcoming earnings calendar
- Historical earnings surprise analysis
- Pre/post earnings price reaction patterns
- Options volume spikes detection
- Earnings sentiment from transcripts

Author: Stock Risk App | Feb 2026
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import requests
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class EarningsEvent:
    """Single earnings event data."""
    ticker: str
    date: datetime
    time: str  # 'BMO' (Before Market Open), 'AMC' (After Market Close), 'Unknown'
    eps_estimate: Optional[float] = None
    eps_actual: Optional[float] = None
    revenue_estimate: Optional[float] = None
    revenue_actual: Optional[float] = None
    surprise_pct: Optional[float] = None
    price_reaction_1d: Optional[float] = None
    price_reaction_5d: Optional[float] = None
    iv_before: Optional[float] = None  # Implied volatility before earnings
    iv_after: Optional[float] = None   # Implied volatility after earnings
    
    @property
    def beat_estimate(self) -> Optional[bool]:
        """Check if earnings beat estimate."""
        if self.eps_actual is not None and self.eps_estimate is not None:
            return self.eps_actual > self.eps_estimate
        return None


@dataclass
class EarningsSurpriseAnalysis:
    """Analysis of historical earnings surprises."""
    ticker: str
    total_quarters: int
    beat_count: int
    miss_count: int
    meet_count: int
    avg_surprise_pct: float
    avg_reaction_beat: float
    avg_reaction_miss: float
    consistency_score: float  # 0-100
    predictability_score: float  # Based on how consistent reactions are
    historical_events: List[EarningsEvent] = field(default_factory=list)


@dataclass
class EarningsCalendarEntry:
    """Calendar entry for upcoming earnings."""
    ticker: str
    company_name: str
    date: datetime
    time: str
    eps_estimate: Optional[float]
    revenue_estimate: Optional[float]
    market_cap: Optional[float]
    sector: Optional[str]
    expected_move: Optional[float]  # Based on options pricing


# ============================================================================
# EARNINGS DATA PROVIDER
# ============================================================================

class EarningsDataProvider:
    """
    Provides earnings data from multiple sources.
    Falls back gracefully if APIs are unavailable.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self._cache = {}
        self._cache_ttl = 3600  # 1 hour
    
    def get_earnings_calendar(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        tickers: Optional[List[str]] = None
    ) -> List[EarningsCalendarEntry]:
        """
        Get upcoming earnings calendar.
        
        Args:
            start_date: Start date for calendar (default: today)
            end_date: End date for calendar (default: 2 weeks ahead)
            tickers: Optional list of tickers to filter
            
        Returns:
            List of EarningsCalendarEntry objects
        """
        if start_date is None:
            start_date = datetime.now()
        if end_date is None:
            end_date = start_date + timedelta(days=14)
        
        calendar = []
        
        # Try yfinance for earnings dates
        try:
            import yfinance as yf
            
            target_tickers = tickers or self._get_major_tickers()
            
            for ticker in target_tickers[:50]:  # Limit to avoid rate limiting
                try:
                    stock = yf.Ticker(ticker)
                    cal = stock.calendar
                    
                    if cal is not None and not cal.empty:
                        # Handle different yfinance calendar formats
                        earnings_date = None
                        if hasattr(cal, 'loc'):
                            if 'Earnings Date' in cal.index:
                                earnings_date = cal.loc['Earnings Date']
                            elif len(cal.columns) > 0:
                                earnings_date = cal.iloc[0, 0] if len(cal) > 0 else None
                        
                        if earnings_date is not None:
                            if isinstance(earnings_date, (pd.Timestamp, datetime)):
                                if start_date <= earnings_date <= end_date:
                                    info = stock.info
                                    calendar.append(EarningsCalendarEntry(
                                        ticker=ticker,
                                        company_name=info.get('shortName', ticker),
                                        date=earnings_date,
                                        time='Unknown',
                                        eps_estimate=info.get('epsForward'),
                                        revenue_estimate=info.get('revenueEstimate'),
                                        market_cap=info.get('marketCap'),
                                        sector=info.get('sector'),
                                        expected_move=None
                                    ))
                except Exception as e:
                    logger.debug(f"Error fetching calendar for {ticker}: {e}")
                    continue
                    
        except ImportError:
            logger.warning("yfinance not available for earnings calendar")
        
        # Sort by date
        calendar.sort(key=lambda x: x.date)
        return calendar
    
    def get_historical_earnings(
        self,
        ticker: str,
        num_quarters: int = 12
    ) -> List[EarningsEvent]:
        """
        Get historical earnings data for a ticker.
        
        Args:
            ticker: Stock symbol
            num_quarters: Number of quarters to fetch
            
        Returns:
            List of EarningsEvent objects
        """
        events = []
        
        try:
            import yfinance as yf
            
            stock = yf.Ticker(ticker)
            
            # Get earnings history
            earnings = stock.earnings_history
            if earnings is not None and not earnings.empty:
                for idx, row in earnings.tail(num_quarters).iterrows():
                    eps_actual = row.get('epsActual')
                    eps_estimate = row.get('epsEstimate')
                    
                    surprise_pct = None
                    if eps_estimate and eps_estimate != 0:
                        surprise_pct = ((eps_actual - eps_estimate) / abs(eps_estimate)) * 100
                    
                    events.append(EarningsEvent(
                        ticker=ticker,
                        date=idx if isinstance(idx, datetime) else datetime.now(),
                        time='Unknown',
                        eps_estimate=eps_estimate,
                        eps_actual=eps_actual,
                        surprise_pct=surprise_pct
                    ))
            
            # Get quarterly earnings (alternative source)
            quarterly = stock.quarterly_earnings
            if quarterly is not None and not quarterly.empty and len(events) == 0:
                for idx, row in quarterly.tail(num_quarters).iterrows():
                    events.append(EarningsEvent(
                        ticker=ticker,
                        date=idx if isinstance(idx, datetime) else datetime.now(),
                        time='Unknown',
                        eps_actual=row.get('Earnings'),
                        revenue_actual=row.get('Revenue')
                    ))
                    
        except Exception as e:
            logger.error(f"Error fetching earnings history for {ticker}: {e}")
        
        return events
    
    def _get_major_tickers(self) -> List[str]:
        """Get list of major tickers to scan for earnings."""
        return [
            # Tech
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 'INTC', 'CRM',
            # Finance
            'JPM', 'BAC', 'GS', 'MS', 'WFC', 'C', 'BLK', 'SCHW',
            # Healthcare
            'JNJ', 'UNH', 'PFE', 'MRK', 'ABBV', 'LLY', 'TMO',
            # Consumer
            'WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'COST',
            # Industrial
            'CAT', 'BA', 'GE', 'HON', 'UPS', 'RTX',
            # Energy
            'XOM', 'CVX', 'COP', 'SLB',
            # Other
            'DIS', 'NFLX', 'V', 'MA', 'PYPL'
        ]


# ============================================================================
# EARNINGS ANALYZER
# ============================================================================

class EarningsAnalyzer:
    """
    Analyze earnings history and predict price reactions.
    """
    
    def __init__(self, data_provider: Optional[EarningsDataProvider] = None):
        self.data_provider = data_provider or EarningsDataProvider()
    
    def analyze_historical_surprises(
        self,
        ticker: str,
        price_data: pd.DataFrame,
        num_quarters: int = 12
    ) -> EarningsSurpriseAnalysis:
        """
        Analyze historical earnings surprises and price reactions.
        
        Args:
            ticker: Stock symbol
            price_data: OHLCV DataFrame with DatetimeIndex
            num_quarters: Number of quarters to analyze
            
        Returns:
            EarningsSurpriseAnalysis object
        """
        events = self.data_provider.get_historical_earnings(ticker, num_quarters)
        
        if not events:
            return EarningsSurpriseAnalysis(
                ticker=ticker,
                total_quarters=0,
                beat_count=0,
                miss_count=0,
                meet_count=0,
                avg_surprise_pct=0,
                avg_reaction_beat=0,
                avg_reaction_miss=0,
                consistency_score=0,
                predictability_score=0
            )
        
        # Calculate price reactions for each event
        events_with_reactions = []
        for event in events:
            event_with_reaction = self._calculate_price_reaction(event, price_data)
            events_with_reactions.append(event_with_reaction)
        
        # Categorize results
        beats = [e for e in events_with_reactions if e.beat_estimate == True]
        misses = [e for e in events_with_reactions if e.beat_estimate == False]
        meets = [e for e in events_with_reactions if e.beat_estimate is None or 
                 (e.surprise_pct is not None and abs(e.surprise_pct) < 1)]
        
        # Calculate averages
        surprises = [e.surprise_pct for e in events_with_reactions if e.surprise_pct is not None]
        avg_surprise = np.mean(surprises) if surprises else 0
        
        beat_reactions = [e.price_reaction_1d for e in beats if e.price_reaction_1d is not None]
        miss_reactions = [e.price_reaction_1d for e in misses if e.price_reaction_1d is not None]
        
        avg_reaction_beat = np.mean(beat_reactions) if beat_reactions else 0
        avg_reaction_miss = np.mean(miss_reactions) if miss_reactions else 0
        
        # Consistency score (how often they beat/miss consistently)
        if len(events_with_reactions) > 0:
            beat_rate = len(beats) / len(events_with_reactions)
            consistency_score = max(beat_rate, 1 - beat_rate) * 100
        else:
            consistency_score = 50
        
        # Predictability score (how consistent are reactions)
        all_reactions = beat_reactions + miss_reactions
        if all_reactions:
            # Lower variance = more predictable
            reaction_std = np.std(all_reactions)
            predictability_score = max(0, 100 - reaction_std * 10)
        else:
            predictability_score = 0
        
        return EarningsSurpriseAnalysis(
            ticker=ticker,
            total_quarters=len(events_with_reactions),
            beat_count=len(beats),
            miss_count=len(misses),
            meet_count=len(meets),
            avg_surprise_pct=avg_surprise,
            avg_reaction_beat=avg_reaction_beat,
            avg_reaction_miss=avg_reaction_miss,
            consistency_score=consistency_score,
            predictability_score=predictability_score,
            historical_events=events_with_reactions
        )
    
    def _calculate_price_reaction(
        self,
        event: EarningsEvent,
        price_data: pd.DataFrame
    ) -> EarningsEvent:
        """Calculate price reaction to an earnings event."""
        try:
            # Find the event date in price data
            event_date = event.date if isinstance(event.date, datetime) else pd.to_datetime(event.date)
            
            # Get price data around the event
            if 'Close' not in price_data.columns:
                return event
            
            # Find nearest trading day
            idx = price_data.index.get_indexer([event_date], method='nearest')[0]
            
            if idx < 0 or idx >= len(price_data) - 5:
                return event
            
            close_before = price_data['Close'].iloc[idx]
            
            # 1-day reaction
            if idx + 1 < len(price_data):
                close_1d = price_data['Close'].iloc[idx + 1]
                event.price_reaction_1d = ((close_1d - close_before) / close_before) * 100
            
            # 5-day reaction
            if idx + 5 < len(price_data):
                close_5d = price_data['Close'].iloc[idx + 5]
                event.price_reaction_5d = ((close_5d - close_before) / close_before) * 100
                
        except Exception as e:
            logger.debug(f"Error calculating price reaction: {e}")
        
        return event
    
    def get_expected_move(
        self,
        ticker: str,
        analysis: Optional[EarningsSurpriseAnalysis] = None
    ) -> Dict[str, float]:
        """
        Calculate expected move around earnings.
        
        Uses historical reactions and options-implied move if available.
        """
        if analysis is None:
            analysis = self.analyze_historical_surprises(ticker, pd.DataFrame())
        
        # Use historical reactions
        reactions = [e.price_reaction_1d for e in analysis.historical_events 
                    if e.price_reaction_1d is not None]
        
        if not reactions:
            return {
                'expected_move': 3.0,  # Default 3%
                'move_std': 2.0,
                'confidence': 'low',
                'method': 'default'
            }
        
        expected = np.mean(np.abs(reactions))
        std = np.std(reactions)
        
        return {
            'expected_move': expected,
            'move_std': std,
            'move_up_prob': len([r for r in reactions if r > 0]) / len(reactions),
            'move_down_prob': len([r for r in reactions if r < 0]) / len(reactions),
            'max_up': max(reactions),
            'max_down': min(reactions),
            'confidence': 'high' if len(reactions) >= 8 else 'medium' if len(reactions) >= 4 else 'low',
            'method': 'historical',
            'num_quarters': len(reactions)
        }
    
    def detect_option_volume_spike(
        self,
        ticker: str,
        days_before: int = 5
    ) -> Dict[str, Any]:
        """
        Detect unusual options volume before earnings.
        
        High options volume before earnings can indicate expected volatility
        or unusual activity (potential information).
        """
        try:
            import yfinance as yf
            
            stock = yf.Ticker(ticker)
            
            # Get options data
            expirations = stock.options
            if not expirations:
                return {'has_spike': False, 'message': 'No options data available'}
            
            # Get nearest expiration
            options = stock.option_chain(expirations[0])
            
            calls = options.calls
            puts = options.puts
            
            if calls.empty and puts.empty:
                return {'has_spike': False, 'message': 'No options data'}
            
            # Calculate total volume
            total_call_volume = calls['volume'].sum() if 'volume' in calls.columns else 0
            total_put_volume = puts['volume'].sum() if 'volume' in puts.columns else 0
            total_volume = total_call_volume + total_put_volume
            
            # Get open interest for comparison
            total_call_oi = calls['openInterest'].sum() if 'openInterest' in calls.columns else 1
            total_put_oi = puts['openInterest'].sum() if 'openInterest' in puts.columns else 1
            
            # Volume/OI ratio > 0.5 suggests unusual activity
            volume_oi_ratio = total_volume / max(total_call_oi + total_put_oi, 1)
            
            # Put/Call ratio
            put_call_ratio = total_put_volume / max(total_call_volume, 1)
            
            return {
                'has_spike': volume_oi_ratio > 0.5,
                'total_volume': total_volume,
                'call_volume': total_call_volume,
                'put_volume': total_put_volume,
                'volume_oi_ratio': volume_oi_ratio,
                'put_call_ratio': put_call_ratio,
                'sentiment': 'bearish' if put_call_ratio > 1.5 else 'bullish' if put_call_ratio < 0.7 else 'neutral',
                'expiration': expirations[0]
            }
            
        except Exception as e:
            logger.error(f"Error detecting options spike: {e}")
            return {'has_spike': False, 'error': str(e)}
    
    def generate_earnings_report(
        self,
        ticker: str,
        price_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Generate comprehensive earnings analysis report.
        """
        analysis = self.analyze_historical_surprises(ticker, price_data)
        expected_move = self.get_expected_move(ticker, analysis)
        options_spike = self.detect_option_volume_spike(ticker)
        
        # Generate insights
        insights = []
        
        if analysis.beat_count > analysis.miss_count * 2:
            insights.append(f"Strong track record: {analysis.beat_count}/{analysis.total_quarters} earnings beats")
        elif analysis.miss_count > analysis.beat_count:
            insights.append(f"Caution: More misses ({analysis.miss_count}) than beats ({analysis.beat_count})")
        
        if analysis.avg_reaction_beat > 3:
            insights.append(f"Large positive reactions on beats (avg +{analysis.avg_reaction_beat:.1f}%)")
        if analysis.avg_reaction_miss < -3:
            insights.append(f"Significant drops on misses (avg {analysis.avg_reaction_miss:.1f}%)")
        
        if analysis.predictability_score > 70:
            insights.append("Highly predictable earnings reactions")
        
        if options_spike.get('has_spike'):
            insights.append(f"Unusual options activity detected ({options_spike.get('sentiment')} bias)")
        
        return {
            'ticker': ticker,
            'summary': analysis,
            'expected_move': expected_move,
            'options_activity': options_spike,
            'insights': insights,
            'recommendation': self._generate_recommendation(analysis, expected_move, options_spike),
            'generated_at': datetime.now().isoformat()
        }
    
    def _generate_recommendation(
        self,
        analysis: EarningsSurpriseAnalysis,
        expected_move: Dict,
        options_spike: Dict
    ) -> str:
        """Generate trading recommendation based on analysis."""
        # Score-based approach
        score = 50  # Neutral starting point
        
        # Beat/miss history
        if analysis.total_quarters > 0:
            beat_rate = analysis.beat_count / analysis.total_quarters
            score += (beat_rate - 0.5) * 30
        
        # Reaction consistency
        score += (analysis.predictability_score - 50) * 0.3
        
        # Options sentiment
        if options_spike.get('sentiment') == 'bullish':
            score += 10
        elif options_spike.get('sentiment') == 'bearish':
            score -= 10
        
        if score > 60:
            return "Bullish bias - Historical beats and positive reactions suggest upside potential"
        elif score < 40:
            return "Bearish bias - History of misses or negative reactions suggest downside risk"
        else:
            return "Neutral - Consider straddle strategy to capture expected volatility"


# ============================================================================
# STREAMLIT RENDERING FUNCTIONS
# ============================================================================

def render_earnings_dashboard(ticker: Optional[str] = None):
    """Render the earnings analysis dashboard in Streamlit."""
    import streamlit as st
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    st.subheader(" Earnings Calendar & Analysis")
    
    provider = EarningsDataProvider()
    analyzer = EarningsAnalyzer(provider)
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs([" Calendar", " Analysis", " Options Activity"])
    
    with tab1:
        st.write("### Upcoming Earnings")
        
        # Date range selector
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("From", datetime.now())
        with col2:
            end_date = st.date_input("To", datetime.now() + timedelta(days=14))
        
        calendar = provider.get_earnings_calendar(
            start_date=datetime.combine(start_date, datetime.min.time()),
            end_date=datetime.combine(end_date, datetime.max.time())
        )
        
        if calendar:
            df = pd.DataFrame([{
                'Ticker': e.ticker,
                'Company': e.company_name,
                'Date': e.date.strftime('%Y-%m-%d') if e.date else 'N/A',
                'Time': e.time,
                'EPS Est': f"${e.eps_estimate:.2f}" if e.eps_estimate else 'N/A',
                'Sector': e.sector or 'N/A',
                'Market Cap': f"${e.market_cap/1e9:.1f}B" if e.market_cap else 'N/A'
            } for e in calendar])
            st.dataframe(df, width="stretch")
        else:
            st.info("No upcoming earnings found in the selected date range.")
    
    with tab2:
        if ticker:
            st.write(f"### Historical Earnings Analysis: {ticker}")
            
            # Get price data
            try:
                import yfinance as yf
                price_data = yf.download(ticker, period='2y', progress=False)
                
                report = analyzer.generate_earnings_report(ticker, price_data)
                
                # Summary metrics
                summary = report['summary']
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Quarters", summary.total_quarters)
                with col2:
                    beat_pct = (summary.beat_count / summary.total_quarters * 100) if summary.total_quarters > 0 else 0
                    st.metric("Beat Rate", f"{beat_pct:.0f}%")
                with col3:
                    st.metric("Avg Surprise", f"{summary.avg_surprise_pct:.1f}%")
                with col4:
                    st.metric("Predictability", f"{summary.predictability_score:.0f}/100")
                
                # Expected move
                st.write("#### Expected Move")
                em = report['expected_move']
                st.info(f"Expected Move: ±{em['expected_move']:.1f}% (Confidence: {em['confidence']})")
                
                # Insights
                if report['insights']:
                    st.write("#### Key Insights")
                    for insight in report['insights']:
                        st.write(f"• {insight}")
                
                # Recommendation
                st.write("#### Trading Consideration")
                st.success(report['recommendation'])
                
                # Historical chart
                if summary.historical_events:
                    st.write("#### Historical Earnings Reactions")
                    
                    events_df = pd.DataFrame([{
                        'Date': e.date,
                        'EPS Actual': e.eps_actual,
                        'EPS Estimate': e.eps_estimate,
                        'Surprise %': e.surprise_pct,
                        '1D Reaction %': e.price_reaction_1d,
                        '5D Reaction %': e.price_reaction_5d,
                        'Beat': '' if e.beat_estimate else '' if e.beat_estimate == False else ''
                    } for e in summary.historical_events if e.eps_actual is not None])
                    
                    if not events_df.empty:
                        st.dataframe(events_df, width="stretch")
                        
                        # Reaction chart
                        fig = go.Figure()
                        reactions = [e.price_reaction_1d for e in summary.historical_events 
                                   if e.price_reaction_1d is not None]
                        dates = [e.date for e in summary.historical_events 
                                if e.price_reaction_1d is not None]
                        
                        colors = ['green' if r > 0 else 'red' for r in reactions]
                        
                        fig.add_trace(go.Bar(
                            x=dates,
                            y=reactions,
                            marker_color=colors,
                            name='1-Day Reaction'
                        ))
                        
                        fig.update_layout(
                            title="Post-Earnings Price Reactions",
                            xaxis_title="Earnings Date",
                            yaxis_title="Return (%)",
                            template='plotly_dark'
                        )
                        
                        st.plotly_chart(fig, width="stretch")
                        
            except Exception as e:
                st.error(f"Error analyzing earnings: {e}")
        else:
            st.info("Enter a ticker symbol to analyze historical earnings.")
    
    with tab3:
        if ticker:
            st.write(f"### Options Activity: {ticker}")
            
            options_data = analyzer.detect_option_volume_spike(ticker)
            
            if 'error' not in options_data:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    color = 'normal' if not options_data['has_spike'] else 'off'
                    st.metric("Volume Spike", "Yes" if options_data['has_spike'] else "No")
                with col2:
                    st.metric("Put/Call Ratio", f"{options_data['put_call_ratio']:.2f}")
                with col3:
                    st.metric("Sentiment", options_data['sentiment'].title())
                
                st.write(f"**Total Option Volume:** {options_data['total_volume']:,.0f}")
                st.write(f"**Call Volume:** {options_data['call_volume']:,.0f}")
                st.write(f"**Put Volume:** {options_data['put_volume']:,.0f}")
                st.write(f"**Volume/OI Ratio:** {options_data['volume_oi_ratio']:.2f}")
                
                if options_data['has_spike']:
                    st.warning(" Unusual options activity detected - may indicate expected volatility or informed trading.")
            else:
                st.error(f"Could not fetch options data: {options_data.get('error', 'Unknown error')}")
        else:
            st.info("Enter a ticker symbol to view options activity.")


# Make features available
HAS_EARNINGS = True
