"""
Advanced Options Strategies
============================
Complex options strategies and flow analysis.

Features:
- Iron Condor, Iron Butterfly
- Calendar Spreads, Diagonal Spreads
- Ratio Spreads, Backspreads
- Options flow analysis (unusual activity)
- IV Rank/Percentile
- Strategy recommendations

Author: Stock Risk App | Feb 2026
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from scipy.stats import norm
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class OptionStrategy(Enum):
    """Option strategy types."""
    IRON_CONDOR = "iron_condor"
    IRON_BUTTERFLY = "iron_butterfly"
    CALENDAR_SPREAD = "calendar_spread"
    DIAGONAL_SPREAD = "diagonal_spread"
    BUTTERFLY_SPREAD = "butterfly_spread"
    RATIO_SPREAD = "ratio_spread"
    JADE_LIZARD = "jade_lizard"
    STRANGLE = "strangle"
    COLLAR = "collar"


class MarketOutlook(Enum):
    """Market outlook for strategy selection."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


@dataclass
class StrategyLeg:
    """Single leg of an options strategy."""
    option_type: str  # 'call' or 'put'
    strike: float
    expiration_days: int
    position: str  # 'long' or 'short'
    quantity: int
    price: float
    delta: float
    gamma: float
    theta: float
    vega: float


@dataclass
class StrategyAnalysis:
    """Complete analysis of an options strategy."""
    name: str
    legs: List[StrategyLeg]
    max_profit: float
    max_loss: float
    breakeven_points: List[float]
    probability_of_profit: float
    expected_return: float
    risk_reward_ratio: float
    net_debit_credit: float  # Negative = credit
    greeks: Dict[str, float]
    payoff_data: Dict[str, Any]


@dataclass
class IVAnalysis:
    """Implied volatility analysis."""
    ticker: str
    current_iv: float
    iv_rank: float  # 0-100, where in the 52-week range
    iv_percentile: float  # 0-100, % of days below current
    iv_52w_high: float
    iv_52w_low: float
    hv_20: float  # 20-day historical volatility
    iv_hv_ratio: float  # IV / HV ratio
    term_structure: Dict[str, float]
    skew: Dict[str, float]
    recommendation: str


@dataclass
class OptionsFlowData:
    """Unusual options activity data."""
    ticker: str
    total_call_volume: int
    total_put_volume: int
    put_call_ratio: float
    unusual_trades: List[Dict]
    large_trades: List[Dict]
    implied_sentiment: str
    confidence: float


# ============================================================================
# ADVANCED OPTIONS CALCULATOR
# ============================================================================

class AdvancedOptionsCalculator:
    """
    Calculate metrics for advanced options strategies.
    """
    
    def __init__(self):
        self.r = 0.045  # Default risk-free rate
    
    @staticmethod
    def black_scholes(S: float, K: float, T: float, r: float, sigma: float, 
                      option_type: str = 'call') -> float:
        """Black-Scholes option pricing."""
        if T <= 0:
            if option_type.lower() == 'call':
                return max(0, S - K)
            return max(0, K - S)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type.lower() == 'call':
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    @staticmethod
    def calculate_greeks(S: float, K: float, T: float, r: float, sigma: float,
                        option_type: str = 'call') -> Dict[str, float]:
        """Calculate Greeks for an option."""
        if T <= 0:
            in_money = (option_type == 'call' and S > K) or (option_type == 'put' and S < K)
            return {
                'delta': 1.0 if in_money and option_type == 'call' else -1.0 if in_money else 0,
                'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0
            }
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            delta = norm.cdf(d1)
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                    - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:
            delta = norm.cdf(d1) - 1
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                    + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100
        
        return {
            'delta': float(delta),
            'gamma': float(gamma),
            'theta': float(theta),
            'vega': float(vega),
            'rho': float(rho)
        }
    
    def iron_condor(
        self,
        S: float,
        put_long_K: float,
        put_short_K: float,
        call_short_K: float,
        call_long_K: float,
        T: float,
        sigma: float
    ) -> StrategyAnalysis:
        """
        Analyze Iron Condor strategy.
        
        Iron Condor: Sell OTM put spread + Sell OTM call spread
        - Long put at put_long_K (lowest)
        - Short put at put_short_K
        - Short call at call_short_K
        - Long call at call_long_K (highest)
        
        Best for: Neutral outlook, expecting range-bound price action
        """
        r = self.r
        
        # Calculate option prices
        long_put = self.black_scholes(S, put_long_K, T, r, sigma, 'put')
        short_put = self.black_scholes(S, put_short_K, T, r, sigma, 'put')
        short_call = self.black_scholes(S, call_short_K, T, r, sigma, 'call')
        long_call = self.black_scholes(S, call_long_K, T, r, sigma, 'call')
        
        # Net credit received
        net_credit = (short_put - long_put) + (short_call - long_call)
        
        # Max profit = net credit received
        max_profit = net_credit * 100
        
        # Max loss = width of spread - net credit
        put_spread_width = put_short_K - put_long_K
        call_spread_width = call_long_K - call_short_K
        max_loss = (max(put_spread_width, call_spread_width) - net_credit) * 100
        
        # Breakevens
        lower_breakeven = put_short_K - net_credit
        upper_breakeven = call_short_K + net_credit
        
        # Calculate Greeks
        greeks = {
            'delta': 0,  # Approximately delta-neutral
            'gamma': 0,
            'theta': 0,
            'vega': 0
        }
        
        for k, opt_type, mult in [
            (put_long_K, 'put', 1), (put_short_K, 'put', -1),
            (call_short_K, 'call', -1), (call_long_K, 'call', 1)
        ]:
            g = self.calculate_greeks(S, k, T, r, sigma, opt_type)
            for greek in greeks:
                greeks[greek] += g[greek] * mult
        
        # Probability of profit (simplified - requires prob of staying between strikes)
        d1_lower = (np.log(S / put_short_K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d1_upper = (np.log(S / call_short_K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        pop = norm.cdf(d1_upper) - norm.cdf(d1_lower)
        
        # Payoff diagram data
        prices = np.linspace(put_long_K * 0.8, call_long_K * 1.2, 100)
        payoffs = []
        for p in prices:
            # Long put
            pnl = max(0, put_long_K - p) - long_put
            # Short put
            pnl -= max(0, put_short_K - p) - short_put
            # Short call
            pnl -= max(0, p - call_short_K) - short_call
            # Long call
            pnl += max(0, p - call_long_K) - long_call
            payoffs.append(pnl * 100)
        
        return StrategyAnalysis(
            name="Iron Condor",
            legs=[
                StrategyLeg('put', put_long_K, int(T*365), 'long', 1, long_put, 
                           **self.calculate_greeks(S, put_long_K, T, r, sigma, 'put')),
                StrategyLeg('put', put_short_K, int(T*365), 'short', 1, short_put,
                           **self.calculate_greeks(S, put_short_K, T, r, sigma, 'put')),
                StrategyLeg('call', call_short_K, int(T*365), 'short', 1, short_call,
                           **self.calculate_greeks(S, call_short_K, T, r, sigma, 'call')),
                StrategyLeg('call', call_long_K, int(T*365), 'long', 1, long_call,
                           **self.calculate_greeks(S, call_long_K, T, r, sigma, 'call'))
            ],
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven_points=[lower_breakeven, upper_breakeven],
            probability_of_profit=pop,
            expected_return=pop * max_profit - (1-pop) * max_loss,
            risk_reward_ratio=max_profit / max_loss if max_loss > 0 else 0,
            net_debit_credit=-net_credit * 100,  # Negative = credit
            greeks=greeks,
            payoff_data={'prices': prices.tolist(), 'payoffs': payoffs}
        )
    
    def iron_butterfly(
        self,
        S: float,
        lower_K: float,
        middle_K: float,
        upper_K: float,
        T: float,
        sigma: float
    ) -> StrategyAnalysis:
        """
        Analyze Iron Butterfly strategy.
        
        Iron Butterfly: ATM short straddle + OTM wings
        - Long put at lower_K
        - Short put at middle_K
        - Short call at middle_K
        - Long call at upper_K
        
        Best for: Very neutral outlook, high premium collection
        """
        r = self.r
        
        long_put = self.black_scholes(S, lower_K, T, r, sigma, 'put')
        short_put = self.black_scholes(S, middle_K, T, r, sigma, 'put')
        short_call = self.black_scholes(S, middle_K, T, r, sigma, 'call')
        long_call = self.black_scholes(S, upper_K, T, r, sigma, 'call')
        
        net_credit = (short_put + short_call) - (long_put + long_call)
        max_profit = net_credit * 100
        
        wing_width = upper_K - middle_K
        max_loss = (wing_width - net_credit) * 100
        
        lower_breakeven = middle_K - net_credit
        upper_breakeven = middle_K + net_credit
        
        # Probability of profit
        d1_lower = (np.log(S / lower_breakeven) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d1_upper = (np.log(S / upper_breakeven) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        pop = norm.cdf(d1_upper) - norm.cdf(d1_lower)
        
        prices = np.linspace(lower_K * 0.85, upper_K * 1.15, 100)
        payoffs = []
        for p in prices:
            pnl = max(0, lower_K - p) - long_put
            pnl -= max(0, middle_K - p) - short_put
            pnl -= max(0, p - middle_K) - short_call
            pnl += max(0, p - upper_K) - long_call
            payoffs.append(pnl * 100)
        
        return StrategyAnalysis(
            name="Iron Butterfly",
            legs=[
                StrategyLeg('put', lower_K, int(T*365), 'long', 1, long_put,
                           **self.calculate_greeks(S, lower_K, T, r, sigma, 'put')),
                StrategyLeg('put', middle_K, int(T*365), 'short', 1, short_put,
                           **self.calculate_greeks(S, middle_K, T, r, sigma, 'put')),
                StrategyLeg('call', middle_K, int(T*365), 'short', 1, short_call,
                           **self.calculate_greeks(S, middle_K, T, r, sigma, 'call')),
                StrategyLeg('call', upper_K, int(T*365), 'long', 1, long_call,
                           **self.calculate_greeks(S, upper_K, T, r, sigma, 'call'))
            ],
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven_points=[lower_breakeven, upper_breakeven],
            probability_of_profit=pop,
            expected_return=pop * max_profit - (1-pop) * max_loss,
            risk_reward_ratio=max_profit / max_loss if max_loss > 0 else 0,
            net_debit_credit=-net_credit * 100,
            greeks={'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0},
            payoff_data={'prices': prices.tolist(), 'payoffs': payoffs}
        )
    
    def butterfly_spread(
        self,
        S: float,
        lower_K: float,
        middle_K: float,
        upper_K: float,
        T: float,
        sigma: float,
        option_type: str = 'call'
    ) -> StrategyAnalysis:
        """
        Analyze Butterfly Spread.
        
        Long Butterfly: Buy 1 lower, Sell 2 middle, Buy 1 upper
        Best for: Targeting specific price at expiration
        """
        r = self.r
        
        buy_lower = self.black_scholes(S, lower_K, T, r, sigma, option_type)
        sell_middle = self.black_scholes(S, middle_K, T, r, sigma, option_type)
        buy_upper = self.black_scholes(S, upper_K, T, r, sigma, option_type)
        
        net_debit = buy_lower - 2 * sell_middle + buy_upper
        
        wing_width = middle_K - lower_K
        max_profit = (wing_width - net_debit) * 100
        max_loss = net_debit * 100
        
        lower_breakeven = lower_K + net_debit
        upper_breakeven = upper_K - net_debit
        
        prices = np.linspace(lower_K * 0.9, upper_K * 1.1, 100)
        payoffs = []
        for p in prices:
            if option_type == 'call':
                pnl = max(0, p - lower_K) - 2*max(0, p - middle_K) + max(0, p - upper_K)
            else:
                pnl = max(0, lower_K - p) - 2*max(0, middle_K - p) + max(0, upper_K - p)
            pnl -= net_debit
            payoffs.append(pnl * 100)
        
        return StrategyAnalysis(
            name=f"{option_type.title()} Butterfly",
            legs=[
                StrategyLeg(option_type, lower_K, int(T*365), 'long', 1, buy_lower,
                           **self.calculate_greeks(S, lower_K, T, r, sigma, option_type)),
                StrategyLeg(option_type, middle_K, int(T*365), 'short', 2, sell_middle,
                           **self.calculate_greeks(S, middle_K, T, r, sigma, option_type)),
                StrategyLeg(option_type, upper_K, int(T*365), 'long', 1, buy_upper,
                           **self.calculate_greeks(S, upper_K, T, r, sigma, option_type))
            ],
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven_points=[lower_breakeven, upper_breakeven],
            probability_of_profit=0.3,  # Simplified
            expected_return=max_profit * 0.3 - max_loss * 0.7,
            risk_reward_ratio=max_profit / max_loss if max_loss > 0 else 0,
            net_debit_credit=net_debit * 100,
            greeks={'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0},
            payoff_data={'prices': prices.tolist(), 'payoffs': payoffs}
        )
    
    def calendar_spread(
        self,
        S: float,
        K: float,
        T_front: float,
        T_back: float,
        sigma_front: float,
        sigma_back: float,
        option_type: str = 'call'
    ) -> StrategyAnalysis:
        """
        Analyze Calendar (Time) Spread.
        
        Sell near-term, Buy far-term at same strike
        Best for: Neutral with expected IV increase
        """
        r = self.r
        
        sell_front = self.black_scholes(S, K, T_front, r, sigma_front, option_type)
        buy_back = self.black_scholes(S, K, T_back, r, sigma_back, option_type)
        
        net_debit = buy_back - sell_front
        
        # Max profit is achieved if price is at strike at front expiration
        # This is complex to calculate exactly
        max_profit = buy_back * 0.5 * 100  # Approximate
        max_loss = net_debit * 100  # If stock moves significantly
        
        return StrategyAnalysis(
            name=f"{option_type.title()} Calendar Spread",
            legs=[
                StrategyLeg(option_type, K, int(T_front*365), 'short', 1, sell_front,
                           **self.calculate_greeks(S, K, T_front, r, sigma_front, option_type)),
                StrategyLeg(option_type, K, int(T_back*365), 'long', 1, buy_back,
                           **self.calculate_greeks(S, K, T_back, r, sigma_back, option_type))
            ],
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven_points=[K * 0.95, K * 1.05],  # Approximate
            probability_of_profit=0.4,
            expected_return=0,  # Complex
            risk_reward_ratio=max_profit / max_loss if max_loss > 0 else 0,
            net_debit_credit=net_debit * 100,
            greeks={'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0},
            payoff_data={'prices': [], 'payoffs': []}  # Complex to calculate
        )
    
    def strangle(
        self,
        S: float,
        put_K: float,
        call_K: float,
        T: float,
        sigma: float
    ) -> StrategyAnalysis:
        """
        Analyze Long Strangle.
        
        Long OTM put + Long OTM call
        Best for: Expecting big move, direction unknown
        """
        r = self.r
        
        put_price = self.black_scholes(S, put_K, T, r, sigma, 'put')
        call_price = self.black_scholes(S, call_K, T, r, sigma, 'call')
        
        total_premium = put_price + call_price
        
        lower_breakeven = put_K - total_premium
        upper_breakeven = call_K + total_premium
        
        max_loss = total_premium * 100
        
        prices = np.linspace(put_K * 0.7, call_K * 1.3, 100)
        payoffs = []
        for p in prices:
            pnl = max(0, put_K - p) + max(0, p - call_K) - total_premium
            payoffs.append(pnl * 100)
        
        return StrategyAnalysis(
            name="Long Strangle",
            legs=[
                StrategyLeg('put', put_K, int(T*365), 'long', 1, put_price,
                           **self.calculate_greeks(S, put_K, T, r, sigma, 'put')),
                StrategyLeg('call', call_K, int(T*365), 'long', 1, call_price,
                           **self.calculate_greeks(S, call_K, T, r, sigma, 'call'))
            ],
            max_profit=float('inf'),
            max_loss=max_loss,
            breakeven_points=[lower_breakeven, upper_breakeven],
            probability_of_profit=0.35,  # Needs significant move
            expected_return=0,
            risk_reward_ratio=float('inf'),
            net_debit_credit=total_premium * 100,
            greeks={
                'delta': 0,
                'gamma': sum(self.calculate_greeks(S, k, T, r, sigma, t)['gamma'] 
                           for k, t in [(put_K, 'put'), (call_K, 'call')]),
                'theta': sum(self.calculate_greeks(S, k, T, r, sigma, t)['theta'] 
                           for k, t in [(put_K, 'put'), (call_K, 'call')]),
                'vega': sum(self.calculate_greeks(S, k, T, r, sigma, t)['vega'] 
                          for k, t in [(put_K, 'put'), (call_K, 'call')])
            },
            payoff_data={'prices': prices.tolist(), 'payoffs': payoffs}
        )


# ============================================================================
# IV ANALYZER
# ============================================================================

class IVAnalyzer:
    """
    Analyze implied volatility for trading decisions.
    """
    
    def analyze(
        self,
        ticker: str,
        current_iv: Optional[float] = None
    ) -> IVAnalysis:
        """
        Perform comprehensive IV analysis.
        
        Args:
            ticker: Stock symbol
            current_iv: Current IV (will fetch if not provided)
            
        Returns:
            IVAnalysis object
        """
        try:
            import yfinance as yf
            
            stock = yf.Ticker(ticker)
            hist = stock.history(period='1y')
            
            # Calculate historical volatility
            returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
            hv_20 = returns.tail(20).std() * np.sqrt(252) * 100
            hv_60 = returns.tail(60).std() * np.sqrt(252) * 100
            
            # Get options chain for IV
            expirations = stock.options
            if expirations:
                chain = stock.option_chain(expirations[0])
                calls = chain.calls
                
                # Get ATM IV
                if 'impliedVolatility' in calls.columns:
                    current_price = hist['Close'].iloc[-1]
                    atm_idx = (calls['strike'] - current_price).abs().idxmin()
                    current_iv = calls.loc[atm_idx, 'impliedVolatility'] * 100
                else:
                    current_iv = current_iv or hv_20 * 1.2
            else:
                current_iv = current_iv or hv_20 * 1.2
            
            # Calculate IV rank (simplified - would need historical IV data)
            # Using HV as proxy
            hv_series = returns.rolling(20).std() * np.sqrt(252) * 100
            iv_52w_high = hv_series.max() * 1.3  # IV typically higher than HV
            iv_52w_low = hv_series.min() * 0.9
            
            iv_rank = ((current_iv - iv_52w_low) / (iv_52w_high - iv_52w_low) * 100) if iv_52w_high > iv_52w_low else 50
            iv_rank = max(0, min(100, iv_rank))
            
            # IV percentile
            iv_percentile = (hv_series < current_iv / 1.2).sum() / len(hv_series) * 100
            
            # IV/HV ratio
            iv_hv_ratio = current_iv / hv_20 if hv_20 > 0 else 1
            
            # Generate recommendation
            if iv_rank > 70:
                recommendation = "HIGH IV RANK: Consider selling premium (iron condors, covered calls, cash-secured puts)"
            elif iv_rank < 30:
                recommendation = "LOW IV RANK: Consider buying premium (long straddles, long calls/puts)"
            else:
                recommendation = "NEUTRAL IV: Consider defined-risk strategies or wait for better opportunity"
            
            return IVAnalysis(
                ticker=ticker,
                current_iv=current_iv,
                iv_rank=iv_rank,
                iv_percentile=iv_percentile,
                iv_52w_high=iv_52w_high,
                iv_52w_low=iv_52w_low,
                hv_20=hv_20,
                iv_hv_ratio=iv_hv_ratio,
                term_structure={},  # Would need multiple expirations
                skew={},  # Would need OTM options
                recommendation=recommendation
            )
            
        except Exception as e:
            logger.error(f"IV analysis error: {e}")
            return IVAnalysis(
                ticker=ticker,
                current_iv=current_iv or 30,
                iv_rank=50,
                iv_percentile=50,
                iv_52w_high=60,
                iv_52w_low=20,
                hv_20=25,
                iv_hv_ratio=1.2,
                term_structure={},
                skew={},
                recommendation="Unable to fetch data - using defaults"
            )


# ============================================================================
# OPTIONS FLOW ANALYZER
# ============================================================================

class OptionsFlowAnalyzer:
    """
    Analyze unusual options activity.
    """
    
    def analyze_flow(self, ticker: str) -> OptionsFlowData:
        """
        Analyze options flow for unusual activity.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            OptionsFlowData object
        """
        try:
            import yfinance as yf
            
            stock = yf.Ticker(ticker)
            expirations = stock.options
            
            if not expirations:
                return OptionsFlowData(
                    ticker=ticker,
                    total_call_volume=0,
                    total_put_volume=0,
                    put_call_ratio=1.0,
                    unusual_trades=[],
                    large_trades=[],
                    implied_sentiment='neutral',
                    confidence=0
                )
            
            total_call_vol = 0
            total_put_vol = 0
            total_call_oi = 0
            total_put_oi = 0
            unusual_trades = []
            large_trades = []
            
            # Analyze first few expirations
            for exp in expirations[:3]:
                chain = stock.option_chain(exp)
                
                # Calls
                calls = chain.calls
                if 'volume' in calls.columns and 'openInterest' in calls.columns:
                    for _, row in calls.iterrows():
                        vol = row.get('volume', 0) or 0
                        oi = row.get('openInterest', 1) or 1
                        
                        total_call_vol += vol
                        total_call_oi += oi
                        
                        # Unusual if volume > 2x OI
                        if vol > oi * 2 and vol > 1000:
                            unusual_trades.append({
                                'type': 'call',
                                'strike': row['strike'],
                                'expiration': exp,
                                'volume': vol,
                                'open_interest': oi,
                                'ratio': vol / oi if oi > 0 else 0
                            })
                        
                        # Large trades
                        if vol > 5000:
                            large_trades.append({
                                'type': 'call',
                                'strike': row['strike'],
                                'expiration': exp,
                                'volume': vol
                            })
                
                # Puts
                puts = chain.puts
                if 'volume' in puts.columns and 'openInterest' in puts.columns:
                    for _, row in puts.iterrows():
                        vol = row.get('volume', 0) or 0
                        oi = row.get('openInterest', 1) or 1
                        
                        total_put_vol += vol
                        total_put_oi += oi
                        
                        if vol > oi * 2 and vol > 1000:
                            unusual_trades.append({
                                'type': 'put',
                                'strike': row['strike'],
                                'expiration': exp,
                                'volume': vol,
                                'open_interest': oi,
                                'ratio': vol / oi if oi > 0 else 0
                            })
                        
                        if vol > 5000:
                            large_trades.append({
                                'type': 'put',
                                'strike': row['strike'],
                                'expiration': exp,
                                'volume': vol
                            })
            
            # Put/call ratio
            pcr = total_put_vol / total_call_vol if total_call_vol > 0 else 1
            
            # Determine sentiment
            if pcr > 1.5:
                sentiment = 'bearish'
                confidence = min(100, (pcr - 1) * 50)
            elif pcr < 0.7:
                sentiment = 'bullish'
                confidence = min(100, (1 - pcr) * 100)
            else:
                sentiment = 'neutral'
                confidence = 30
            
            # Adjust for unusual activity
            if unusual_trades:
                call_unusual = len([t for t in unusual_trades if t['type'] == 'call'])
                put_unusual = len([t for t in unusual_trades if t['type'] == 'put'])
                
                if call_unusual > put_unusual * 2:
                    sentiment = 'bullish'
                    confidence = min(100, confidence + 20)
                elif put_unusual > call_unusual * 2:
                    sentiment = 'bearish'
                    confidence = min(100, confidence + 20)
            
            return OptionsFlowData(
                ticker=ticker,
                total_call_volume=total_call_vol,
                total_put_volume=total_put_vol,
                put_call_ratio=pcr,
                unusual_trades=sorted(unusual_trades, key=lambda x: x['volume'], reverse=True)[:10],
                large_trades=sorted(large_trades, key=lambda x: x['volume'], reverse=True)[:10],
                implied_sentiment=sentiment,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Options flow analysis error: {e}")
            return OptionsFlowData(
                ticker=ticker,
                total_call_volume=0,
                total_put_volume=0,
                put_call_ratio=1.0,
                unusual_trades=[],
                large_trades=[],
                implied_sentiment='neutral',
                confidence=0
            )


# ============================================================================
# STRATEGY RECOMMENDER
# ============================================================================

class StrategyRecommender:
    """
    Recommend options strategies based on outlook and conditions.
    """
    
    def recommend(
        self,
        outlook: MarketOutlook,
        iv_rank: float,
        days_to_earnings: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get strategy recommendations based on conditions.
        
        Args:
            outlook: Market outlook
            iv_rank: Current IV rank (0-100)
            days_to_earnings: Days until earnings (if applicable)
            
        Returns:
            List of recommended strategies
        """
        recommendations = []
        
        # High IV strategies (sell premium)
        if iv_rank > 60:
            if outlook == MarketOutlook.NEUTRAL:
                recommendations.append({
                    'strategy': 'Iron Condor',
                    'description': 'Sell OTM put spread and call spread',
                    'best_for': 'Range-bound markets, high IV',
                    'risk_level': 'Defined',
                    'expected_win_rate': '60-70%'
                })
                recommendations.append({
                    'strategy': 'Iron Butterfly',
                    'description': 'ATM short straddle with wings',
                    'best_for': 'Very neutral, maximum premium',
                    'risk_level': 'Defined',
                    'expected_win_rate': '40-50%'
                })
            elif outlook == MarketOutlook.BULLISH:
                recommendations.append({
                    'strategy': 'Short Put Spread',
                    'description': 'Sell put spread for credit',
                    'best_for': 'Bullish with high IV',
                    'risk_level': 'Defined',
                    'expected_win_rate': '65-75%'
                })
            elif outlook == MarketOutlook.BEARISH:
                recommendations.append({
                    'strategy': 'Short Call Spread',
                    'description': 'Sell call spread for credit',
                    'best_for': 'Bearish with high IV',
                    'risk_level': 'Defined',
                    'expected_win_rate': '65-75%'
                })
        
        # Low IV strategies (buy premium)
        if iv_rank < 40:
            if outlook == MarketOutlook.HIGH_VOLATILITY:
                recommendations.append({
                    'strategy': 'Long Straddle',
                    'description': 'Buy ATM call and put',
                    'best_for': 'Expecting big move, direction unknown',
                    'risk_level': 'Defined (premium paid)',
                    'expected_win_rate': '30-40%'
                })
                recommendations.append({
                    'strategy': 'Long Strangle',
                    'description': 'Buy OTM call and put',
                    'best_for': 'Cheaper volatility play',
                    'risk_level': 'Defined',
                    'expected_win_rate': '25-35%'
                })
            elif outlook == MarketOutlook.BULLISH:
                recommendations.append({
                    'strategy': 'Long Call Spread',
                    'description': 'Buy call, sell higher strike call',
                    'best_for': 'Bullish with limited risk',
                    'risk_level': 'Defined',
                    'expected_win_rate': '40-50%'
                })
        
        # Earnings plays
        if days_to_earnings is not None and days_to_earnings < 14:
            if iv_rank > 70:
                recommendations.append({
                    'strategy': 'Iron Condor (pre-earnings)',
                    'description': 'Sell premium before IV crush',
                    'best_for': 'Capturing pre-earnings IV',
                    'risk_level': 'High (earnings risk)',
                    'expected_win_rate': '50-60%'
                })
            else:
                recommendations.append({
                    'strategy': 'Long Straddle (earnings)',
                    'description': 'Buy volatility before announcement',
                    'best_for': 'Playing earnings move',
                    'risk_level': 'High (need big move)',
                    'expected_win_rate': '35-45%'
                })
        
        # Time decay strategies
        if outlook == MarketOutlook.LOW_VOLATILITY:
            recommendations.append({
                'strategy': 'Calendar Spread',
                'description': 'Short front month, long back month',
                'best_for': 'Expecting IV increase, price stability',
                'risk_level': 'Moderate',
                'expected_win_rate': '40-50%'
            })
        
        if not recommendations:
            recommendations.append({
                'strategy': 'Wait',
                'description': 'No clear edge identified',
                'best_for': 'Capital preservation',
                'risk_level': 'None',
                'expected_win_rate': 'N/A'
            })
        
        return recommendations


# ============================================================================
# STREAMLIT RENDERING
# ============================================================================

def render_advanced_options_dashboard(ticker: Optional[str] = None):
    """Render advanced options analysis dashboard."""
    import streamlit as st
    import plotly.graph_objects as go
    
    st.subheader(" Advanced Options Strategies")
    
    if not ticker:
        ticker = st.text_input("Enter ticker:", value="SPY")
    
    if ticker:
        # IV Analysis
        st.write("### Implied Volatility Analysis")
        
        iv_analyzer = IVAnalyzer()
        iv_data = iv_analyzer.analyze(ticker)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current IV", f"{iv_data.current_iv:.1f}%")
        with col2:
            color = "inverse" if iv_data.iv_rank > 60 else "normal"
            st.metric("IV Rank", f"{iv_data.iv_rank:.0f}")
        with col3:
            st.metric("IV Percentile", f"{iv_data.iv_percentile:.0f}")
        with col4:
            st.metric("IV/HV Ratio", f"{iv_data.iv_hv_ratio:.2f}")
        
        st.info(f" {iv_data.recommendation}")
        
        # Options Flow
        st.write("### Options Flow Analysis")
        
        flow_analyzer = OptionsFlowAnalyzer()
        flow = flow_analyzer.analyze_flow(ticker)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Call Volume", f"{flow.total_call_volume:,}")
        with col2:
            st.metric("Put Volume", f"{flow.total_put_volume:,}")
        with col3:
            sentiment_color = {'bullish': '🟢', 'bearish': '', 'neutral': ''}
            st.metric("Sentiment", f"{sentiment_color.get(flow.implied_sentiment, '')} {flow.implied_sentiment.title()}")
        
        if flow.unusual_trades:
            st.write("#### Unusual Activity")
            for trade in flow.unusual_trades[:5]:
                st.write(f"• {trade['type'].upper()} ${trade['strike']} | Vol: {trade['volume']:,} | Exp: {trade['expiration']}")
        
        # Strategy Calculator
        st.write("### Strategy Calculator")
        
        calc = AdvancedOptionsCalculator()
        
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            S = stock.info.get('currentPrice', 100)
        except:
            S = 100
        
        strategy = st.selectbox(
            "Select Strategy",
            ["Iron Condor", "Iron Butterfly", "Butterfly Spread", "Strangle", "Calendar Spread"]
        )
        
        sigma = iv_data.current_iv / 100
        T = st.slider("Days to Expiration", 7, 90, 30) / 365
        
        if strategy == "Iron Condor":
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                put_long_K = st.number_input("Long Put Strike", value=round(S * 0.90, 0))
            with col2:
                put_short_K = st.number_input("Short Put Strike", value=round(S * 0.95, 0))
            with col3:
                call_short_K = st.number_input("Short Call Strike", value=round(S * 1.05, 0))
            with col4:
                call_long_K = st.number_input("Long Call Strike", value=round(S * 1.10, 0))
            
            analysis = calc.iron_condor(S, put_long_K, put_short_K, call_short_K, call_long_K, T, sigma)
            
        elif strategy == "Iron Butterfly":
            col1, col2, col3 = st.columns(3)
            with col1:
                lower_K = st.number_input("Lower Strike", value=round(S * 0.95, 0))
            with col2:
                middle_K = st.number_input("Middle Strike", value=round(S, 0))
            with col3:
                upper_K = st.number_input("Upper Strike", value=round(S * 1.05, 0))
            
            analysis = calc.iron_butterfly(S, lower_K, middle_K, upper_K, T, sigma)
            
        elif strategy == "Butterfly Spread":
            col1, col2, col3 = st.columns(3)
            with col1:
                lower_K = st.number_input("Lower Strike", value=round(S * 0.95, 0))
            with col2:
                middle_K = st.number_input("Middle Strike", value=round(S, 0))
            with col3:
                upper_K = st.number_input("Upper Strike", value=round(S * 1.05, 0))
            
            analysis = calc.butterfly_spread(S, lower_K, middle_K, upper_K, T, sigma)
            
        elif strategy == "Strangle":
            col1, col2 = st.columns(2)
            with col1:
                put_K = st.number_input("Put Strike", value=round(S * 0.95, 0))
            with col2:
                call_K = st.number_input("Call Strike", value=round(S * 1.05, 0))
            
            analysis = calc.strangle(S, put_K, call_K, T, sigma)
        else:
            st.info("Calendar spread calculator requires two expirations")
            analysis = None
        
        if analysis:
            # Display results
            st.write(f"### {analysis.name} Analysis")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Max Profit", f"${analysis.max_profit:.2f}")
            with col2:
                loss_str = f"${analysis.max_loss:.2f}" if analysis.max_loss < float('inf') else "Unlimited"
                st.metric("Max Loss", loss_str)
            with col3:
                st.metric("Risk/Reward", f"{analysis.risk_reward_ratio:.2f}" if analysis.risk_reward_ratio < 100 else "∞")
            with col4:
                st.metric("P.O.P.", f"{analysis.probability_of_profit*100:.0f}%")
            
            # Breakevens
            be_str = " / ".join([f"${be:.2f}" for be in analysis.breakeven_points])
            st.write(f"**Breakeven Points:** {be_str}")
            
            # Credit/Debit
            if analysis.net_debit_credit < 0:
                st.success(f"**Net Credit Received:** ${abs(analysis.net_debit_credit):.2f}")
            else:
                st.warning(f"**Net Debit Paid:** ${analysis.net_debit_credit:.2f}")
            
            # Payoff diagram
            if analysis.payoff_data.get('prices'):
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=analysis.payoff_data['prices'],
                    y=analysis.payoff_data['payoffs'],
                    mode='lines',
                    name='P&L at Expiration',
                    line=dict(color='#00D4AA', width=2)
                ))
                
                # Add zero line
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                
                # Add current price line
                fig.add_vline(x=S, line_dash="dash", line_color="yellow", 
                             annotation_text="Current Price")
                
                fig.update_layout(
                    title="Payoff Diagram at Expiration",
                    xaxis_title="Stock Price",
                    yaxis_title="Profit/Loss ($)",
                    template='plotly_dark',
                    height=400
                )
                
                st.plotly_chart(fig, width="stretch")
        
        # Strategy recommendations
        st.write("### Strategy Recommendations")
        
        outlook = st.selectbox(
            "Your Market Outlook",
            [o.value.title() for o in MarketOutlook]
        )
        
        recommender = StrategyRecommender()
        outlook_enum = MarketOutlook(outlook.lower().replace(' ', '_'))
        recommendations = recommender.recommend(outlook_enum, iv_data.iv_rank)
        
        for rec in recommendations:
            st.markdown(f"""
            <div style="padding: 10px; margin: 5px 0; border-radius: 5px; background: #21262d;">
                <strong>{rec['strategy']}</strong><br>
                {rec['description']}<br>
                <small>Best for: {rec['best_for']} | Risk: {rec['risk_level']} | Win Rate: {rec['expected_win_rate']}</small>
            </div>
            """, unsafe_allow_html=True)


# Make features available
HAS_ADVANCED_OPTIONS = True
