"""
Insider Trading Tracker
========================
Track SEC Form 4 filings and analyze insider trading patterns.

Features:
- SEC Form 4 filings parser
- Insider buying/selling patterns
- Cluster detection (multiple insiders)
- Historical accuracy scoring
- Alert on significant insider activity

Author: Stock Risk App | Feb 2026
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import requests

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class TransactionType(Enum):
    """Types of insider transactions."""
    BUY = "P"  # Open market purchase
    SELL = "S"  # Open market sale
    GRANT = "A"  # Award/grant
    EXERCISE = "M"  # Exercise of options
    CONVERSION = "C"  # Conversion
    GIFT = "G"  # Gift
    OTHER = "X"


class InsiderRole(Enum):
    """Insider roles."""
    CEO = "CEO"
    CFO = "CFO"
    COO = "COO"
    DIRECTOR = "Director"
    EVP = "EVP"
    SVP = "SVP"
    VP = "VP"
    OFFICER = "Officer"
    TEN_PERCENT = "10% Owner"
    BOARD = "Board Member"
    OTHER = "Other"


@dataclass
class InsiderTransaction:
    """Single insider transaction."""
    ticker: str
    insider_name: str
    insider_title: str
    transaction_type: TransactionType
    transaction_date: datetime
    filing_date: datetime
    shares: int
    price: float
    value: float
    shares_owned_after: int
    is_direct: bool  # Direct vs indirect ownership
    
    @property
    def is_open_market(self) -> bool:
        """Check if this is an open market transaction (most significant)."""
        return self.transaction_type in [TransactionType.BUY, TransactionType.SELL]
    
    @property
    def is_buy(self) -> bool:
        return self.transaction_type == TransactionType.BUY
    
    @property
    def is_sell(self) -> bool:
        return self.transaction_type == TransactionType.SELL


@dataclass
class InsiderCluster:
    """Cluster of insider transactions (multiple insiders trading similarly)."""
    ticker: str
    start_date: datetime
    end_date: datetime
    direction: str  # 'buying', 'selling', 'mixed'
    num_insiders: int
    total_value: float
    transactions: List[InsiderTransaction]
    significance_score: float  # 0-100


@dataclass
class InsiderAnalysis:
    """Complete insider trading analysis for a stock."""
    ticker: str
    total_buys_30d: int
    total_sells_30d: int
    net_shares_30d: int
    net_value_30d: float
    buy_sell_ratio: float
    recent_clusters: List[InsiderCluster]
    notable_transactions: List[InsiderTransaction]
    historical_accuracy: float  # How predictive insider trades have been
    signal: str  # 'bullish', 'bearish', 'neutral'
    signal_strength: float  # 0-100
    insights: List[str]


# ============================================================================
# SEC FILINGS PROVIDER
# ============================================================================

class SECFilingsProvider:
    """
    Fetch and parse SEC Form 4 filings.
    Uses SEC EDGAR and yfinance as data sources.
    """
    
    SEC_BASE_URL = "https://www.sec.gov"
    
    def __init__(self):
        self._cache = {}
        self._cache_ttl = 3600
    
    def get_insider_transactions(
        self,
        ticker: str,
        days: int = 90
    ) -> List[InsiderTransaction]:
        """
        Get insider transactions for a ticker.
        
        Args:
            ticker: Stock symbol
            days: Number of days to look back
            
        Returns:
            List of InsiderTransaction objects
        """
        transactions = []
        
        # Try yfinance first (easiest integration)
        try:
            import yfinance as yf
            
            stock = yf.Ticker(ticker)
            
            # Get insider transactions
            insiders = stock.insider_transactions
            
            if insiders is not None and not insiders.empty:
                for _, row in insiders.iterrows():
                    try:
                        # Parse transaction type
                        trans_text = str(row.get('Transaction', '')).lower()
                        if 'buy' in trans_text or 'purchase' in trans_text:
                            trans_type = TransactionType.BUY
                        elif 'sell' in trans_text or 'sale' in trans_text:
                            trans_type = TransactionType.SELL
                        elif 'option' in trans_text or 'exercise' in trans_text:
                            trans_type = TransactionType.EXERCISE
                        elif 'grant' in trans_text or 'award' in trans_text:
                            trans_type = TransactionType.GRANT
                        else:
                            trans_type = TransactionType.OTHER
                        
                        # Parse date
                        date_val = row.get('Start Date') or row.get('Date')
                        if pd.isna(date_val):
                            continue
                        trans_date = pd.to_datetime(date_val)
                        
                        # Skip if too old
                        if (datetime.now() - trans_date).days > days:
                            continue
                        
                        shares = abs(int(row.get('Shares', 0) or 0))
                        value = abs(float(row.get('Value', 0) or 0))
                        price = value / shares if shares > 0 else 0
                        
                        transactions.append(InsiderTransaction(
                            ticker=ticker,
                            insider_name=str(row.get('Insider', 'Unknown')),
                            insider_title=str(row.get('Position', 'Unknown')),
                            transaction_type=trans_type,
                            transaction_date=trans_date,
                            filing_date=trans_date,  # yfinance doesn't separate these
                            shares=shares,
                            price=price,
                            value=value,
                            shares_owned_after=0,  # Not available from yfinance
                            is_direct=True
                        ))
                    except Exception as e:
                        logger.debug(f"Error parsing transaction: {e}")
                        continue
                        
        except Exception as e:
            logger.warning(f"yfinance insider data error: {e}")
        
        # Sort by date
        transactions.sort(key=lambda x: x.transaction_date, reverse=True)
        
        return transactions
    
    def get_insider_holders(self, ticker: str) -> Dict[str, Any]:
        """Get information about insider holders."""
        try:
            import yfinance as yf
            
            stock = yf.Ticker(ticker)
            
            # Get insider holders
            holders = stock.insider_holders
            
            if holders is not None and not holders.empty:
                return {
                    'total_insiders': len(holders),
                    'holders': holders.to_dict('records'),
                    'total_shares_held': holders['Shares'].sum() if 'Shares' in holders.columns else 0
                }
                
        except Exception as e:
            logger.warning(f"Error fetching insider holders: {e}")
        
        return {'total_insiders': 0, 'holders': [], 'total_shares_held': 0}


# ============================================================================
# INSIDER ANALYZER
# ============================================================================

class InsiderAnalyzer:
    """
    Analyze insider trading patterns and generate signals.
    """
    
    def __init__(self, provider: Optional[SECFilingsProvider] = None):
        self.provider = provider or SECFilingsProvider()
    
    def analyze_insider_activity(
        self,
        ticker: str,
        price_data: Optional[pd.DataFrame] = None,
        days: int = 90
    ) -> InsiderAnalysis:
        """
        Comprehensive insider trading analysis.
        
        Args:
            ticker: Stock symbol
            price_data: Optional price data for accuracy calculation
            days: Days of history to analyze
            
        Returns:
            InsiderAnalysis object
        """
        transactions = self.provider.get_insider_transactions(ticker, days)
        
        if not transactions:
            return InsiderAnalysis(
                ticker=ticker,
                total_buys_30d=0,
                total_sells_30d=0,
                net_shares_30d=0,
                net_value_30d=0,
                buy_sell_ratio=1.0,
                recent_clusters=[],
                notable_transactions=[],
                historical_accuracy=0.5,
                signal='neutral',
                signal_strength=0,
                insights=["No insider transactions found in the past 90 days"]
            )
        
        # Filter to open market transactions (most significant)
        open_market = [t for t in transactions if t.is_open_market]
        
        # 30-day metrics
        thirty_days_ago = datetime.now() - timedelta(days=30)
        recent = [t for t in open_market if t.transaction_date >= thirty_days_ago]
        
        buys_30d = [t for t in recent if t.is_buy]
        sells_30d = [t for t in recent if t.is_sell]
        
        total_buys = len(buys_30d)
        total_sells = len(sells_30d)
        
        buy_shares = sum(t.shares for t in buys_30d)
        sell_shares = sum(t.shares for t in sells_30d)
        net_shares = buy_shares - sell_shares
        
        buy_value = sum(t.value for t in buys_30d)
        sell_value = sum(t.value for t in sells_30d)
        net_value = buy_value - sell_value
        
        # Buy/sell ratio
        if total_sells > 0:
            buy_sell_ratio = total_buys / total_sells
        elif total_buys > 0:
            buy_sell_ratio = float('inf')
        else:
            buy_sell_ratio = 1.0
        
        # Detect clusters
        clusters = self._detect_clusters(open_market)
        
        # Find notable transactions (large or from important insiders)
        notable = self._find_notable_transactions(open_market)
        
        # Calculate historical accuracy
        accuracy = self._calculate_historical_accuracy(transactions, price_data) if price_data is not None else 0.5
        
        # Generate signal
        signal, strength, insights = self._generate_signal(
            total_buys, total_sells, net_value, clusters, notable, accuracy
        )
        
        return InsiderAnalysis(
            ticker=ticker,
            total_buys_30d=total_buys,
            total_sells_30d=total_sells,
            net_shares_30d=net_shares,
            net_value_30d=net_value,
            buy_sell_ratio=buy_sell_ratio if buy_sell_ratio != float('inf') else 999,
            recent_clusters=clusters,
            notable_transactions=notable,
            historical_accuracy=accuracy,
            signal=signal,
            signal_strength=strength,
            insights=insights
        )
    
    def _detect_clusters(
        self,
        transactions: List[InsiderTransaction],
        window_days: int = 14
    ) -> List[InsiderCluster]:
        """Detect clusters of insider activity (multiple insiders trading similarly)."""
        if not transactions:
            return []
        
        clusters = []
        
        # Group transactions by date window
        transactions = sorted(transactions, key=lambda x: x.transaction_date)
        
        i = 0
        while i < len(transactions):
            window_start = transactions[i].transaction_date
            window_end = window_start + timedelta(days=window_days)
            
            # Collect all transactions in window
            window_trans = []
            j = i
            while j < len(transactions) and transactions[j].transaction_date <= window_end:
                window_trans.append(transactions[j])
                j += 1
            
            # Check if cluster (multiple unique insiders)
            unique_insiders = set(t.insider_name for t in window_trans)
            
            if len(unique_insiders) >= 2:
                buys = [t for t in window_trans if t.is_buy]
                sells = [t for t in window_trans if t.is_sell]
                
                if len(buys) > len(sells) * 2:
                    direction = 'buying'
                elif len(sells) > len(buys) * 2:
                    direction = 'selling'
                else:
                    direction = 'mixed'
                
                total_value = sum(t.value for t in window_trans)
                
                # Calculate significance
                significance = min(100, len(unique_insiders) * 20 + (total_value / 100000))
                
                clusters.append(InsiderCluster(
                    ticker=window_trans[0].ticker,
                    start_date=window_start,
                    end_date=window_trans[-1].transaction_date,
                    direction=direction,
                    num_insiders=len(unique_insiders),
                    total_value=total_value,
                    transactions=window_trans,
                    significance_score=significance
                ))
            
            i = j if j > i else i + 1
        
        return clusters
    
    def _find_notable_transactions(
        self,
        transactions: List[InsiderTransaction],
        value_threshold: float = 100000
    ) -> List[InsiderTransaction]:
        """Find notable transactions (large value or C-suite)."""
        notable = []
        
        important_titles = ['CEO', 'CFO', 'COO', 'President', 'Chairman', 'CTO']
        
        for t in transactions:
            is_important = any(title.lower() in t.insider_title.lower() for title in important_titles)
            is_large = t.value >= value_threshold
            
            if is_important or is_large:
                notable.append(t)
        
        return notable[:10]  # Top 10
    
    def _calculate_historical_accuracy(
        self,
        transactions: List[InsiderTransaction],
        price_data: pd.DataFrame
    ) -> float:
        """
        Calculate how accurate insider trades have been historically.
        
        Returns accuracy score 0-1 based on whether price moved in
        the direction of the trade.
        """
        if price_data.empty or 'Close' not in price_data.columns:
            return 0.5
        
        correct = 0
        total = 0
        
        for t in transactions:
            if not t.is_open_market:
                continue
            
            try:
                # Find price at transaction
                trans_date = t.transaction_date
                
                # Get price 30 days later
                future_date = trans_date + timedelta(days=30)
                
                # Find nearest dates in price data
                trans_idx = price_data.index.get_indexer([trans_date], method='nearest')[0]
                future_idx = price_data.index.get_indexer([future_date], method='nearest')[0]
                
                if trans_idx < 0 or future_idx < 0 or trans_idx >= len(price_data) or future_idx >= len(price_data):
                    continue
                
                price_at_trans = price_data['Close'].iloc[trans_idx]
                price_future = price_data['Close'].iloc[future_idx]
                
                # Check if trade was correct direction
                price_went_up = price_future > price_at_trans
                
                if (t.is_buy and price_went_up) or (t.is_sell and not price_went_up):
                    correct += 1
                
                total += 1
                
            except Exception:
                continue
        
        return correct / total if total > 0 else 0.5
    
    def _generate_signal(
        self,
        total_buys: int,
        total_sells: int,
        net_value: float,
        clusters: List[InsiderCluster],
        notable: List[InsiderTransaction],
        accuracy: float
    ) -> Tuple[str, float, List[str]]:
        """Generate trading signal based on insider activity."""
        insights = []
        score = 50  # Neutral starting point
        
        # Buy/sell analysis
        if total_buys > total_sells * 2:
            score += 20
            insights.append(f"Strong buying: {total_buys} buys vs {total_sells} sells in last 30 days")
        elif total_buys > total_sells:
            score += 10
            insights.append(f"Net buying activity: {total_buys} buys vs {total_sells} sells")
        elif total_sells > total_buys * 2:
            score -= 20
            insights.append(f"Heavy selling: {total_sells} sells vs {total_buys} buys in last 30 days")
        elif total_sells > total_buys:
            score -= 10
            insights.append(f"Net selling activity: {total_sells} sells vs {total_buys} buys")
        
        # Net value
        if net_value > 1000000:
            score += 15
            insights.append(f"Large net buying: ${net_value/1e6:.1f}M")
        elif net_value > 100000:
            score += 5
        elif net_value < -1000000:
            score -= 15
            insights.append(f"Large net selling: ${abs(net_value)/1e6:.1f}M")
        elif net_value < -100000:
            score -= 5
        
        # Cluster analysis
        for cluster in clusters:
            if cluster.direction == 'buying' and cluster.significance_score > 50:
                score += 10
                insights.append(f"Insider buying cluster: {cluster.num_insiders} insiders bought ${cluster.total_value/1e6:.2f}M")
            elif cluster.direction == 'selling' and cluster.significance_score > 50:
                score -= 10
                insights.append(f"Insider selling cluster: {cluster.num_insiders} insiders sold ${cluster.total_value/1e6:.2f}M")
        
        # Notable transactions
        for t in notable[:3]:
            if t.is_buy and t.value > 500000:
                score += 5
                insights.append(f"Major buy: {t.insider_name} ({t.insider_title}) bought ${t.value/1e6:.2f}M")
            elif t.is_sell and t.value > 500000:
                # CEO/CFO large sells are more concerning
                if any(x in t.insider_title.upper() for x in ['CEO', 'CFO']):
                    score -= 10
                    insights.append(f"Concerning: {t.insider_title} sold ${t.value/1e6:.2f}M")
        
        # Historical accuracy adjustment
        if accuracy > 0.65:
            insights.append(f"High historical accuracy: {accuracy*100:.0f}% of trades were correct")
        elif accuracy < 0.35:
            insights.append(f"Poor historical accuracy: {accuracy*100:.0f}% - may be contrarian indicator")
            score = 100 - score  # Flip signal
        
        # Determine signal
        if score >= 65:
            signal = 'bullish'
        elif score <= 35:
            signal = 'bearish'
        else:
            signal = 'neutral'
        
        strength = abs(score - 50) * 2  # 0-100 scale
        
        if not insights:
            insights.append("Limited insider activity to analyze")
        
        return signal, strength, insights


# ============================================================================
# STREAMLIT RENDERING
# ============================================================================

def render_insider_dashboard(ticker: Optional[str] = None):
    """Render the insider trading dashboard in Streamlit."""
    import streamlit as st
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    st.subheader(" Insider Trading Tracker")
    
    analyzer = InsiderAnalyzer()
    
    if not ticker:
        ticker = st.text_input("Enter ticker symbol:", value="AAPL")
    
    if ticker:
        with st.spinner(f"Analyzing insider activity for {ticker}..."):
            # Get price data for accuracy calculation
            try:
                import yfinance as yf
                price_data = yf.download(ticker, period='1y', progress=False)
            except:
                price_data = pd.DataFrame()
            
            analysis = analyzer.analyze_insider_activity(ticker, price_data)
        
        # Signal header
        signal_colors = {
            'bullish': '#00C853',
            'bearish': '#FF5722',
            'neutral': '#9E9E9E'
        }
        
        st.markdown(f"""
        <div style="padding: 20px; border-radius: 10px; background: {signal_colors[analysis.signal]}22; 
                    border: 2px solid {signal_colors[analysis.signal]};">
            <h3 style="margin: 0; color: {signal_colors[analysis.signal]};">
                Insider Signal: {analysis.signal.upper()}
            </h3>
            <p style="margin: 5px 0;">Signal Strength: {analysis.signal_strength:.0f}/100</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Buys (30d)", analysis.total_buys_30d)
        with col2:
            st.metric("Sells (30d)", analysis.total_sells_30d)
        with col3:
            net_str = f"${analysis.net_value_30d/1e6:.2f}M" if abs(analysis.net_value_30d) > 1e6 else f"${analysis.net_value_30d/1e3:.0f}K"
            st.metric("Net Value", net_str)
        with col4:
            st.metric("Buy/Sell Ratio", f"{min(analysis.buy_sell_ratio, 99.9):.1f}")
        
        # Insights
        st.write("### Key Insights")
        for insight in analysis.insights:
            icon = "🟢" if 'buy' in insight.lower() else "" if 'sell' in insight.lower() else ""
            st.write(f"{icon} {insight}")
        
        # Notable transactions table
        if analysis.notable_transactions:
            st.write("### Notable Transactions")
            
            notable_df = pd.DataFrame([{
                'Date': t.transaction_date.strftime('%Y-%m-%d'),
                'Insider': t.insider_name,
                'Title': t.insider_title,
                'Type': 'BUY 🟢' if t.is_buy else 'SELL ',
                'Shares': f"{t.shares:,}",
                'Value': f"${t.value:,.0f}"
            } for t in analysis.notable_transactions])
            
            st.dataframe(notable_df, width="stretch")
        
        # Cluster visualization
        if analysis.recent_clusters:
            st.write("### Insider Clusters (Coordinated Activity)")
            
            for cluster in analysis.recent_clusters:
                st.markdown(f"""
                <div style="padding: 10px; margin: 5px 0; border-radius: 5px; 
                            background: {'#00C85322' if cluster.direction == 'buying' else '#FF572222' if cluster.direction == 'selling' else '#9E9E9E22'};">
                    <strong>{cluster.direction.title()}</strong> cluster: 
                    {cluster.num_insiders} insiders, ${cluster.total_value/1e6:.2f}M total
                    ({cluster.start_date.strftime('%Y-%m-%d')} to {cluster.end_date.strftime('%Y-%m-%d')})
                </div>
                """, unsafe_allow_html=True)
        
        # Transaction timeline chart
        transactions = analyzer.provider.get_insider_transactions(ticker)
        if transactions:
            st.write("### Transaction Timeline")
            
            buys = [t for t in transactions if t.is_buy]
            sells = [t for t in transactions if t.is_sell]
            
            fig = go.Figure()
            
            if buys:
                fig.add_trace(go.Scatter(
                    x=[t.transaction_date for t in buys],
                    y=[t.value for t in buys],
                    mode='markers',
                    name='Buys',
                    marker=dict(size=10, color='#00C853'),
                    hovertemplate='%{text}<br>$%{y:,.0f}<extra></extra>',
                    text=[f"{t.insider_name}" for t in buys]
                ))
            
            if sells:
                fig.add_trace(go.Scatter(
                    x=[t.transaction_date for t in sells],
                    y=[t.value for t in sells],
                    mode='markers',
                    name='Sells',
                    marker=dict(size=10, color='#FF5722'),
                    hovertemplate='%{text}<br>$%{y:,.0f}<extra></extra>',
                    text=[f"{t.insider_name}" for t in sells]
                ))
            
            fig.update_layout(
                title="Insider Transactions Timeline",
                xaxis_title="Date",
                yaxis_title="Transaction Value ($)",
                template='plotly_dark',
                height=400
            )
            
            st.plotly_chart(fig, width="stretch")


# Make features available
HAS_INSIDER_TRADING = True
