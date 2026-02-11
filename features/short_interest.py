"""
Short Interest & Squeeze Analysis
==================================
Track short interest and predict potential short squeezes.

Features:
- Short interest ratio tracking
- Days to cover calculation
- Borrow fee analysis
- Squeeze probability scoring
- Historical squeeze patterns

Author: Stock Risk App | Feb 2026
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class SqueezeRisk(Enum):
    """Risk levels for short squeeze."""
    EXTREME = "extreme"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    MINIMAL = "minimal"


@dataclass
class ShortInterestData:
    """Short interest data for a stock."""
    ticker: str
    date: datetime
    short_interest: int  # Number of shares shorted
    shares_outstanding: int
    short_percent_of_float: float
    short_percent_of_shares: float
    days_to_cover: float
    average_volume: int
    previous_short_interest: Optional[int] = None
    change_percent: Optional[float] = None


@dataclass
class BorrowData:
    """Stock borrow/lending data."""
    ticker: str
    borrow_rate: float  # Annual rate
    available_shares: int
    utilization: float  # % of lendable shares borrowed
    is_hard_to_borrow: bool
    cost_to_borrow_30d: float  # Estimated cost for 30 days


@dataclass
class SqueezeAnalysis:
    """Complete squeeze analysis for a stock."""
    ticker: str
    squeeze_score: float  # 0-100
    squeeze_risk: SqueezeRisk
    short_interest_data: Optional[ShortInterestData]
    borrow_data: Optional[BorrowData]
    technical_factors: Dict[str, Any]
    catalysts: List[str]
    historical_squeezes: List[Dict]
    price_targets_if_squeeze: Dict[str, float]
    risk_factors: List[str]
    recommendation: str


@dataclass
class SqueezeEvent:
    """Historical squeeze event."""
    ticker: str
    start_date: datetime
    peak_date: datetime
    end_date: datetime
    price_before: float
    price_peak: float
    price_after: float
    gain_percent: float
    duration_days: int
    short_interest_before: float
    volume_spike: float


# ============================================================================
# SHORT DATA PROVIDER
# ============================================================================

class ShortDataProvider:
    """
    Provides short interest and borrow data.
    Uses multiple sources with graceful fallbacks.
    """
    
    def __init__(self):
        self._cache = {}
        self._cache_ttl = 3600
    
    def get_short_interest(self, ticker: str) -> Optional[ShortInterestData]:
        """
        Get short interest data for a ticker.
        
        Note: Actual short interest is reported bi-monthly by FINRA.
        This provides estimates based on available data.
        """
        try:
            import yfinance as yf
            
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Get short interest data from yfinance
            short_interest = info.get('sharesShort', 0) or 0
            shares_outstanding = info.get('sharesOutstanding', 1) or 1
            float_shares = info.get('floatShares', shares_outstanding) or shares_outstanding
            avg_volume = info.get('averageVolume', 1) or 1
            prior_short = info.get('sharesShortPriorMonth', 0) or 0
            
            # Calculate metrics
            short_pct_float = (short_interest / float_shares * 100) if float_shares > 0 else 0
            short_pct_shares = (short_interest / shares_outstanding * 100) if shares_outstanding > 0 else 0
            days_to_cover = short_interest / avg_volume if avg_volume > 0 else 0
            
            change_pct = None
            if prior_short > 0:
                change_pct = ((short_interest - prior_short) / prior_short) * 100
            
            return ShortInterestData(
                ticker=ticker,
                date=datetime.now(),
                short_interest=short_interest,
                shares_outstanding=shares_outstanding,
                short_percent_of_float=short_pct_float,
                short_percent_of_shares=short_pct_shares,
                days_to_cover=days_to_cover,
                average_volume=avg_volume,
                previous_short_interest=prior_short,
                change_percent=change_pct
            )
            
        except Exception as e:
            logger.warning(f"Error fetching short interest for {ticker}: {e}")
            return None
    
    def estimate_borrow_data(
        self,
        ticker: str,
        short_data: Optional[ShortInterestData] = None
    ) -> BorrowData:
        """
        Estimate borrow data based on short interest.
        
        Note: Actual borrow rates require prime broker data.
        This provides estimates based on public data.
        """
        if short_data is None:
            short_data = self.get_short_interest(ticker)
        
        if short_data is None:
            return BorrowData(
                ticker=ticker,
                borrow_rate=1.0,  # Base rate
                available_shares=0,
                utilization=0,
                is_hard_to_borrow=False,
                cost_to_borrow_30d=0
            )
        
        # Estimate borrow rate based on short interest
        # Higher short interest = higher borrow rate
        short_pct = short_data.short_percent_of_float
        
        if short_pct > 50:
            borrow_rate = 50 + (short_pct - 50) * 2  # Very expensive
        elif short_pct > 30:
            borrow_rate = 20 + (short_pct - 30)
        elif short_pct > 20:
            borrow_rate = 10 + (short_pct - 20) * 0.5
        elif short_pct > 10:
            borrow_rate = 3 + (short_pct - 10) * 0.3
        else:
            borrow_rate = 1 + short_pct * 0.2
        
        # Estimate utilization
        utilization = min(95, short_pct * 1.5)
        
        # Hard to borrow threshold
        is_hard = short_pct > 20 or borrow_rate > 20
        
        # Cost to borrow for 30 days (annualized rate / 12)
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            price = stock.info.get('currentPrice', 100)
        except:
            price = 100
        
        cost_30d = (price * short_data.short_interest * borrow_rate / 100) / 12
        
        return BorrowData(
            ticker=ticker,
            borrow_rate=borrow_rate,
            available_shares=max(0, short_data.shares_outstanding - short_data.short_interest),
            utilization=utilization,
            is_hard_to_borrow=is_hard,
            cost_to_borrow_30d=cost_30d
        )


# ============================================================================
# SQUEEZE ANALYZER
# ============================================================================

class SqueezeAnalyzer:
    """
    Analyze short squeeze potential and risk.
    """
    
    # Historical squeeze thresholds (based on GME, AMC, etc.)
    SQUEEZE_THRESHOLDS = {
        'short_pct_extreme': 50,  # GME was ~140%
        'short_pct_high': 30,
        'short_pct_moderate': 20,
        'days_to_cover_high': 5,
        'days_to_cover_extreme': 10
    }
    
    def __init__(self, provider: Optional[ShortDataProvider] = None):
        self.provider = provider or ShortDataProvider()
    
    def analyze_squeeze_potential(
        self,
        ticker: str,
        price_data: Optional[pd.DataFrame] = None
    ) -> SqueezeAnalysis:
        """
        Comprehensive squeeze potential analysis.
        
        Args:
            ticker: Stock symbol
            price_data: Optional OHLCV data for technical analysis
            
        Returns:
            SqueezeAnalysis object
        """
        # Get short interest data
        short_data = self.provider.get_short_interest(ticker)
        borrow_data = self.provider.estimate_borrow_data(ticker, short_data)
        
        # Calculate squeeze score
        score, risk_level, factors = self._calculate_squeeze_score(
            short_data, borrow_data, price_data
        )
        
        # Identify catalysts
        catalysts = self._identify_catalysts(ticker, short_data)
        
        # Find historical squeezes in price data
        historical = self._find_historical_squeezes(price_data) if price_data is not None else []
        
        # Calculate price targets
        targets = self._calculate_squeeze_targets(ticker, short_data, price_data)
        
        # Risk factors
        risks = self._identify_risk_factors(short_data, borrow_data)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(score, risk_level, catalysts, risks)
        
        return SqueezeAnalysis(
            ticker=ticker,
            squeeze_score=score,
            squeeze_risk=risk_level,
            short_interest_data=short_data,
            borrow_data=borrow_data,
            technical_factors=factors,
            catalysts=catalysts,
            historical_squeezes=historical,
            price_targets_if_squeeze=targets,
            risk_factors=risks,
            recommendation=recommendation
        )
    
    def _calculate_squeeze_score(
        self,
        short_data: Optional[ShortInterestData],
        borrow_data: Optional[BorrowData],
        price_data: Optional[pd.DataFrame]
    ) -> Tuple[float, SqueezeRisk, Dict[str, Any]]:
        """Calculate squeeze probability score (0-100)."""
        score = 0
        factors = {}
        
        if short_data is None:
            return 0, SqueezeRisk.MINIMAL, {'error': 'No short data available'}
        
        # Short interest score (0-40 points)
        short_pct = short_data.short_percent_of_float
        if short_pct >= 50:
            score += 40
            factors['short_interest'] = f"Extreme ({short_pct:.1f}%)"
        elif short_pct >= 30:
            score += 30
            factors['short_interest'] = f"Very High ({short_pct:.1f}%)"
        elif short_pct >= 20:
            score += 20
            factors['short_interest'] = f"High ({short_pct:.1f}%)"
        elif short_pct >= 10:
            score += 10
            factors['short_interest'] = f"Moderate ({short_pct:.1f}%)"
        else:
            factors['short_interest'] = f"Low ({short_pct:.1f}%)"
        
        # Days to cover score (0-20 points)
        dtc = short_data.days_to_cover
        if dtc >= 10:
            score += 20
            factors['days_to_cover'] = f"Extreme ({dtc:.1f} days)"
        elif dtc >= 5:
            score += 15
            factors['days_to_cover'] = f"High ({dtc:.1f} days)"
        elif dtc >= 3:
            score += 10
            factors['days_to_cover'] = f"Moderate ({dtc:.1f} days)"
        else:
            factors['days_to_cover'] = f"Low ({dtc:.1f} days)"
        
        # Borrow data score (0-20 points)
        if borrow_data:
            if borrow_data.is_hard_to_borrow:
                score += 15
                factors['borrow_status'] = f"Hard to borrow (rate: {borrow_data.borrow_rate:.1f}%)"
            elif borrow_data.borrow_rate > 10:
                score += 10
                factors['borrow_status'] = f"Elevated borrow rate ({borrow_data.borrow_rate:.1f}%)"
            else:
                factors['borrow_status'] = f"Normal ({borrow_data.borrow_rate:.1f}%)"
            
            if borrow_data.utilization > 80:
                score += 5
                factors['utilization'] = f"High utilization ({borrow_data.utilization:.0f}%)"
        
        # Short interest change (0-10 points)
        if short_data.change_percent is not None:
            if short_data.change_percent > 20:
                score += 10
                factors['si_change'] = f"Rising rapidly (+{short_data.change_percent:.1f}%)"
            elif short_data.change_percent > 5:
                score += 5
                factors['si_change'] = f"Increasing (+{short_data.change_percent:.1f}%)"
            elif short_data.change_percent < -20:
                score -= 10
                factors['si_change'] = f"Covering rapidly ({short_data.change_percent:.1f}%)"
        
        # Technical factors (0-10 points)
        if price_data is not None and not price_data.empty:
            tech_score, tech_factors = self._analyze_technical_factors(price_data)
            score += tech_score
            factors.update(tech_factors)
        
        # Clamp score
        score = max(0, min(100, score))
        
        # Determine risk level
        if score >= 70:
            risk_level = SqueezeRisk.EXTREME
        elif score >= 50:
            risk_level = SqueezeRisk.HIGH
        elif score >= 30:
            risk_level = SqueezeRisk.MODERATE
        elif score >= 15:
            risk_level = SqueezeRisk.LOW
        else:
            risk_level = SqueezeRisk.MINIMAL
        
        return score, risk_level, factors
    
    def _analyze_technical_factors(
        self,
        price_data: pd.DataFrame
    ) -> Tuple[float, Dict[str, Any]]:
        """Analyze technical factors for squeeze potential."""
        score = 0
        factors = {}
        
        try:
            # Recent momentum
            returns = price_data['Close'].pct_change().dropna()
            recent_return = returns.tail(5).sum() * 100
            
            if recent_return > 20:
                score += 5
                factors['momentum'] = f"Strong upward ({recent_return:.1f}%)"
            elif recent_return > 10:
                score += 3
                factors['momentum'] = f"Positive ({recent_return:.1f}%)"
            
            # Volume surge
            if 'Volume' in price_data.columns:
                avg_vol = price_data['Volume'].tail(20).mean()
                recent_vol = price_data['Volume'].tail(5).mean()
                vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 1
                
                if vol_ratio > 3:
                    score += 5
                    factors['volume'] = f"Massive spike ({vol_ratio:.1f}x average)"
                elif vol_ratio > 2:
                    score += 3
                    factors['volume'] = f"Elevated ({vol_ratio:.1f}x average)"
            
            # Volatility
            volatility = returns.tail(20).std() * np.sqrt(252) * 100
            if volatility > 100:
                factors['volatility'] = f"Extreme ({volatility:.0f}% annualized)"
            elif volatility > 50:
                factors['volatility'] = f"High ({volatility:.0f}% annualized)"
            
        except Exception as e:
            logger.debug(f"Technical analysis error: {e}")
        
        return score, factors
    
    def _identify_catalysts(
        self,
        ticker: str,
        short_data: Optional[ShortInterestData]
    ) -> List[str]:
        """Identify potential squeeze catalysts."""
        catalysts = []
        
        if short_data is None:
            return catalysts
        
        if short_data.short_percent_of_float > 30:
            catalysts.append("High short interest creates covering pressure")
        
        if short_data.days_to_cover > 5:
            catalysts.append("Extended days to cover makes quick exits difficult")
        
        if short_data.change_percent is not None and short_data.change_percent < -10:
            catalysts.append("Early covering activity may trigger chain reaction")
        
        # Check for earnings (potential catalyst)
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            cal = stock.calendar
            if cal is not None and not cal.empty:
                catalysts.append("Upcoming earnings could trigger volatility")
        except:
            pass
        
        if not catalysts:
            catalysts.append("No significant catalysts identified")
        
        return catalysts
    
    def _find_historical_squeezes(
        self,
        price_data: pd.DataFrame,
        threshold: float = 50
    ) -> List[Dict]:
        """Find historical squeeze-like events in price data."""
        squeezes = []
        
        if price_data is None or price_data.empty:
            return squeezes
        
        try:
            # Look for rapid price increases
            returns = price_data['Close'].pct_change()
            
            # Find periods with >50% gain in 10 days or less
            for i in range(10, len(price_data)):
                window_return = (price_data['Close'].iloc[i] / price_data['Close'].iloc[i-10] - 1) * 100
                
                if window_return > threshold:
                    squeezes.append({
                        'start_date': price_data.index[i-10].strftime('%Y-%m-%d'),
                        'end_date': price_data.index[i].strftime('%Y-%m-%d'),
                        'gain_percent': window_return,
                        'price_start': price_data['Close'].iloc[i-10],
                        'price_end': price_data['Close'].iloc[i]
                    })
            
            # Deduplicate overlapping events
            if squeezes:
                squeezes = squeezes[:5]  # Keep top 5
                
        except Exception as e:
            logger.debug(f"Historical squeeze detection error: {e}")
        
        return squeezes
    
    def _calculate_squeeze_targets(
        self,
        ticker: str,
        short_data: Optional[ShortInterestData],
        price_data: Optional[pd.DataFrame]
    ) -> Dict[str, float]:
        """Calculate potential price targets if squeeze occurs."""
        targets = {}
        
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            current_price = stock.info.get('currentPrice', 0)
            
            if current_price == 0 and price_data is not None and not price_data.empty:
                current_price = price_data['Close'].iloc[-1]
            
            if current_price > 0:
                # Conservative: 25% squeeze
                targets['conservative'] = current_price * 1.25
                
                # Moderate: 50% squeeze
                targets['moderate'] = current_price * 1.5
                
                # Aggressive: 100% squeeze
                targets['aggressive'] = current_price * 2.0
                
                # Extreme (GME-style): 500%+
                if short_data and short_data.short_percent_of_float > 50:
                    targets['extreme'] = current_price * 5.0
                    
        except Exception as e:
            logger.debug(f"Target calculation error: {e}")
        
        return targets
    
    def _identify_risk_factors(
        self,
        short_data: Optional[ShortInterestData],
        borrow_data: Optional[BorrowData]
    ) -> List[str]:
        """Identify risk factors for squeeze play."""
        risks = []
        
        risks.append("Short squeezes are rare events - most heavily shorted stocks decline")
        risks.append("Extreme volatility can result in rapid gains AND losses")
        risks.append("Liquidity may dry up at critical moments")
        
        if short_data:
            if short_data.short_percent_of_float < 20:
                risks.append("Short interest may not be high enough for sustained squeeze")
            
            if short_data.days_to_cover < 2:
                risks.append("Low days to cover allows shorts to exit quickly")
        
        if borrow_data:
            if not borrow_data.is_hard_to_borrow:
                risks.append("Easy to borrow shares reduces squeeze pressure")
        
        return risks
    
    def _generate_recommendation(
        self,
        score: float,
        risk_level: SqueezeRisk,
        catalysts: List[str],
        risks: List[str]
    ) -> str:
        """Generate trading recommendation."""
        if risk_level == SqueezeRisk.EXTREME:
            return "EXTREME CAUTION: Very high squeeze potential but also extreme risk. Only for experienced traders with strict risk management."
        elif risk_level == SqueezeRisk.HIGH:
            return "HIGH ALERT: Significant squeeze conditions present. Consider small position with defined risk. Watch for catalyst triggers."
        elif risk_level == SqueezeRisk.MODERATE:
            return "MONITOR: Some squeeze characteristics present. Wait for additional catalysts before considering entry."
        elif risk_level == SqueezeRisk.LOW:
            return "LOW PROBABILITY: Limited squeeze potential. Short interest not elevated enough for significant short covering."
        else:
            return "MINIMAL: No significant squeeze indicators. Trade based on fundamentals/technicals, not squeeze potential."


# ============================================================================
# STREAMLIT RENDERING
# ============================================================================

def render_short_squeeze_dashboard(ticker: Optional[str] = None):
    """Render the short squeeze analysis dashboard in Streamlit."""
    import streamlit as st
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    st.subheader(" Short Interest & Squeeze Analysis")
    
    analyzer = SqueezeAnalyzer()
    
    if not ticker:
        ticker = st.text_input("Enter ticker symbol:", value="GME")
    
    if ticker:
        with st.spinner(f"Analyzing squeeze potential for {ticker}..."):
            # Get price data
            try:
                import yfinance as yf
                price_data = yf.download(ticker, period='1y', progress=False)
            except:
                price_data = pd.DataFrame()
            
            analysis = analyzer.analyze_squeeze_potential(ticker, price_data)
        
        # Squeeze score gauge
        risk_colors = {
            SqueezeRisk.EXTREME: '#FF0000',
            SqueezeRisk.HIGH: '#FF5722',
            SqueezeRisk.MODERATE: '#FFC107',
            SqueezeRisk.LOW: '#4CAF50',
            SqueezeRisk.MINIMAL: '#9E9E9E'
        }
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=analysis.squeeze_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Squeeze Score"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': risk_colors[analysis.squeeze_risk]},
                    'steps': [
                        {'range': [0, 15], 'color': '#9E9E9E22'},
                        {'range': [15, 30], 'color': '#4CAF5022'},
                        {'range': [30, 50], 'color': '#FFC10722'},
                        {'range': [50, 70], 'color': '#FF572222'},
                        {'range': [70, 100], 'color': '#FF000022'}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': analysis.squeeze_score
                    }
                }
            ))
            fig.update_layout(height=250, template='plotly_dark')
            st.plotly_chart(fig, width="stretch")
        
        with col2:
            st.markdown(f"""
            <div style="padding: 20px; border-radius: 10px; background: {risk_colors[analysis.squeeze_risk]}22; 
                        border: 2px solid {risk_colors[analysis.squeeze_risk]};">
                <h3 style="margin: 0; color: {risk_colors[analysis.squeeze_risk]};">
                    Risk Level: {analysis.squeeze_risk.value.upper()}
                </h3>
                <p style="margin: 10px 0;"><strong>Recommendation:</strong></p>
                <p style="margin: 5px 0;">{analysis.recommendation}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Short interest metrics
        st.write("### Short Interest Data")
        
        if analysis.short_interest_data:
            si = analysis.short_interest_data
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Short % of Float", f"{si.short_percent_of_float:.1f}%")
            with col2:
                st.metric("Days to Cover", f"{si.days_to_cover:.1f}")
            with col3:
                st.metric("Short Interest", f"{si.short_interest/1e6:.2f}M shares")
            with col4:
                if si.change_percent is not None:
                    st.metric("Change (MoM)", f"{si.change_percent:+.1f}%")
                else:
                    st.metric("Change (MoM)", "N/A")
        else:
            st.warning("Short interest data not available for this ticker")
        
        # Borrow data
        if analysis.borrow_data:
            st.write("### Borrow Data (Estimated)")
            bd = analysis.borrow_data
            
            col1, col2, col3 = st.columns(3)
            with col1:
                color = "inverse" if bd.is_hard_to_borrow else "normal"
                st.metric("Borrow Rate", f"{bd.borrow_rate:.1f}%")
            with col2:
                st.metric("Utilization", f"{bd.utilization:.0f}%")
            with col3:
                status = " Hard to Borrow" if bd.is_hard_to_borrow else " Available"
                st.metric("Status", status)
        
        # Technical factors
        if analysis.technical_factors:
            st.write("### Contributing Factors")
            for factor, value in analysis.technical_factors.items():
                if factor != 'error':
                    st.write(f"• **{factor.replace('_', ' ').title()}**: {value}")
        
        # Catalysts
        st.write("### Potential Catalysts")
        for catalyst in analysis.catalysts:
            st.write(f" {catalyst}")
        
        # Price targets
        if analysis.price_targets_if_squeeze:
            st.write("### Price Targets (If Squeeze)")
            
            try:
                import yfinance as yf
                current = yf.Ticker(ticker).info.get('currentPrice', 0)
            except:
                current = 0
            
            targets = analysis.price_targets_if_squeeze
            
            cols = st.columns(len(targets))
            for idx, (scenario, target) in enumerate(targets.items()):
                pct = ((target - current) / current * 100) if current > 0 else 0
                with cols[idx]:
                    st.metric(scenario.title(), f"${target:.2f}", f"+{pct:.0f}%")
        
        # Historical squeezes
        if analysis.historical_squeezes:
            st.write("### Historical Squeeze-Like Events")
            
            for event in analysis.historical_squeezes:
                st.markdown(f"""
                <div style="padding: 10px; margin: 5px 0; border-radius: 5px; background: #00C85322;">
                     <strong>{event['start_date']} to {event['end_date']}</strong>: 
                    +{event['gain_percent']:.0f}% (${event['price_start']:.2f} → ${event['price_end']:.2f})
                </div>
                """, unsafe_allow_html=True)
        
        # Risk factors
        st.write("### Risk Factors ")
        for risk in analysis.risk_factors:
            st.write(f"• {risk}")
        
        # Price chart with short interest overlay
        if not price_data.empty:
            st.write("### Price History")
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                              vertical_spacing=0.05,
                              row_heights=[0.7, 0.3])
            
            # Price chart
            fig.add_trace(go.Candlestick(
                x=price_data.index,
                open=price_data['Open'],
                high=price_data['High'],
                low=price_data['Low'],
                close=price_data['Close'],
                name='Price'
            ), row=1, col=1)
            
            # Volume
            colors = ['green' if price_data['Close'].iloc[i] >= price_data['Open'].iloc[i] 
                     else 'red' for i in range(len(price_data))]
            fig.add_trace(go.Bar(
                x=price_data.index,
                y=price_data['Volume'],
                marker_color=colors,
                name='Volume',
                opacity=0.7
            ), row=2, col=1)
            
            fig.update_layout(
                title=f"{ticker} Price & Volume",
                template='plotly_dark',
                height=500,
                xaxis_rangeslider_visible=False
            )
            
            st.plotly_chart(fig, width="stretch")


# Make features available
HAS_SHORT_SQUEEZE = True
