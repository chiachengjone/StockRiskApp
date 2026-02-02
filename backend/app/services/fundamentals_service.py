"""
Fundamentals Analysis Service
=============================
Fundamental analysis calculations ported from features/fundamentals.py

Provides company fundamentals, ratios, DCF valuation, and quality scoring.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class FundamentalAnalyzer:
    """
    Analyze company fundamentals and valuation.
    
    Features:
    - Financial ratios analysis
    - Peer comparison
    - DCF valuation
    - Quality scoring
    """
    
    def __init__(self):
        pass
    
    def analyze_fundamentals(self, info: Dict) -> Dict:
        """
        Comprehensive fundamental analysis from company info.
        
        Args:
            info: Company info dictionary from data provider
        
        Returns:
            Dictionary with analysis results
        """
        if not info or 'error' in info:
            return {'error': 'No fundamental data available'}
        
        # Calculate derived metrics
        analysis = {
            'ticker': info.get('ticker', info.get('symbol', 'Unknown')),
            'name': info.get('name', info.get('shortName', 'Unknown')),
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            
            # Valuation Ratios
            'valuation': self._analyze_valuation(info),
            
            # Profitability
            'profitability': self._analyze_profitability(info),
            
            # Financial Health
            'financial_health': self._analyze_financial_health(info),
            
            # Growth
            'growth': self._analyze_growth(info),
            
            # Overall Score
            'quality_score': None
        }
        
        # Calculate quality score
        analysis['quality_score'] = self._calculate_quality_score(analysis)
        
        return analysis
    
    def _analyze_valuation(self, info: Dict) -> Dict:
        """Analyze valuation metrics."""
        pe = info.get('trailingPE') or info.get('pe_ratio')
        forward_pe = info.get('forwardPE') or info.get('forward_pe')
        pb = info.get('priceToBook') or info.get('price_to_book')
        ps = info.get('priceToSalesTrailing12Months') or info.get('price_to_sales')
        
        valuation_score = 0
        assessments = []
        
        if pe:
            if pe < 15:
                valuation_score += 2
                assessments.append('Low P/E (potentially undervalued)')
            elif pe < 25:
                valuation_score += 1
                assessments.append('Moderate P/E')
            else:
                assessments.append('High P/E (growth expectations priced in)')
        
        if pb:
            if pb < 1:
                valuation_score += 2
                assessments.append('Trading below book value')
            elif pb < 3:
                valuation_score += 1
        
        return {
            'pe_ratio': pe,
            'forward_pe': forward_pe,
            'price_to_book': pb,
            'price_to_sales': ps,
            'peg_ratio': info.get('pegRatio') or info.get('peg_ratio'),
            'ev_to_ebitda': info.get('enterpriseToEbitda') or info.get('ev_to_ebitda'),
            'score': valuation_score,
            'assessments': assessments
        }
    
    def _analyze_profitability(self, info: Dict) -> Dict:
        """Analyze profitability metrics."""
        roe = info.get('returnOnEquity') or info.get('roe')
        roa = info.get('returnOnAssets') or info.get('roa')
        profit_margin = info.get('profitMargins') or info.get('profit_margin')
        operating_margin = info.get('operatingMargins') or info.get('operating_margin')
        
        score = 0
        assessments = []
        
        if roe:
            if roe > 0.20:
                score += 2
                assessments.append('Excellent ROE (>20%)')
            elif roe > 0.10:
                score += 1
                assessments.append('Good ROE (10-20%)')
        
        if profit_margin:
            if profit_margin > 0.20:
                score += 2
                assessments.append('High profit margins')
            elif profit_margin > 0.10:
                score += 1
        
        return {
            'roe': roe,
            'roa': roa,
            'profit_margin': profit_margin,
            'operating_margin': operating_margin,
            'score': score,
            'assessments': assessments
        }
    
    def _analyze_financial_health(self, info: Dict) -> Dict:
        """Analyze financial health metrics."""
        debt_to_equity = info.get('debtToEquity') or info.get('debt_to_equity')
        current_ratio = info.get('currentRatio') or info.get('current_ratio')
        quick_ratio = info.get('quickRatio') or info.get('quick_ratio')
        
        score = 0
        assessments = []
        
        if debt_to_equity is not None:
            if debt_to_equity < 50:
                score += 2
                assessments.append('Low debt levels')
            elif debt_to_equity < 100:
                score += 1
            else:
                assessments.append('High debt load')
        
        if current_ratio:
            if current_ratio > 2:
                score += 2
                assessments.append('Strong liquidity')
            elif current_ratio > 1:
                score += 1
            else:
                assessments.append('Liquidity concerns')
        
        return {
            'debt_to_equity': debt_to_equity,
            'current_ratio': current_ratio,
            'quick_ratio': quick_ratio,
            'score': score,
            'assessments': assessments
        }
    
    def _analyze_growth(self, info: Dict) -> Dict:
        """Analyze growth metrics."""
        revenue_growth = info.get('revenueGrowth') or info.get('revenue_growth')
        earnings_growth = info.get('earningsGrowth') or info.get('earnings_growth')
        
        score = 0
        assessments = []
        
        if revenue_growth:
            if revenue_growth > 0.20:
                score += 2
                assessments.append('Strong revenue growth (>20%)')
            elif revenue_growth > 0.10:
                score += 1
            elif revenue_growth < 0:
                assessments.append('Revenue declining')
        
        if earnings_growth:
            if earnings_growth > 0.20:
                score += 2
                assessments.append('Strong earnings growth')
            elif earnings_growth > 0:
                score += 1
        
        return {
            'revenue_growth': revenue_growth,
            'earnings_growth': earnings_growth,
            'score': score,
            'assessments': assessments
        }
    
    def _calculate_quality_score(self, analysis: Dict) -> float:
        """Calculate overall quality score 0-100."""
        valuation_score = analysis['valuation'].get('score', 0)
        profitability_score = analysis['profitability'].get('score', 0)
        health_score = analysis['financial_health'].get('score', 0)
        growth_score = analysis['growth'].get('score', 0)
        
        # Max possible: 4 + 4 + 4 + 4 = 16
        raw_score = valuation_score + profitability_score + health_score + growth_score
        quality_score = (raw_score / 16) * 100
        
        return float(quality_score)
    
    def dcf_valuation(
        self,
        info: Dict,
        growth_rate: float = 0.10,
        terminal_growth: float = 0.03,
        discount_rate: float = 0.10,
        projection_years: int = 5
    ) -> Dict:
        """
        Discounted Cash Flow valuation.
        
        Args:
            info: Company info with financial data
            growth_rate: Expected revenue/FCF growth rate
            terminal_growth: Terminal growth rate
            discount_rate: WACC/discount rate
            projection_years: Years to project
        
        Returns:
            DCF valuation results
        """
        # Get FCF or estimate from earnings
        fcf = info.get('freeCashflow') or info.get('free_cashflow')
        earnings = info.get('netIncome') or info.get('earnings')
        shares = info.get('sharesOutstanding') or info.get('shares_outstanding', 1)
        current_price = info.get('currentPrice') or info.get('regularMarketPrice', 100)
        
        if fcf is None and earnings is not None:
            fcf = earnings * 0.7  # Rough FCF proxy
        elif fcf is None:
            return {'error': 'Insufficient data for DCF'}
        
        # Project cash flows
        cash_flows = []
        cf = fcf
        for year in range(1, projection_years + 1):
            cf = cf * (1 + growth_rate)
            discounted_cf = cf / ((1 + discount_rate) ** year)
            cash_flows.append(discounted_cf)
        
        # Terminal value
        terminal_cf = cf * (1 + terminal_growth)
        terminal_value = terminal_cf / (discount_rate - terminal_growth)
        discounted_terminal = terminal_value / ((1 + discount_rate) ** projection_years)
        
        # Enterprise value
        present_value_cf = sum(cash_flows)
        intrinsic_value = present_value_cf + discounted_terminal
        
        # Equity value per share
        debt = info.get('totalDebt', 0) or 0
        cash = info.get('totalCash', 0) or 0
        equity_value = intrinsic_value - debt + cash
        fair_value = equity_value / shares if shares > 0 else 0
        
        # Upside potential
        upside = (fair_value / current_price - 1) * 100 if current_price > 0 else 0
        
        # Sensitivity analysis
        sensitivity = {}
        for rate in [discount_rate - 0.02, discount_rate, discount_rate + 0.02]:
            disc_term = terminal_value / ((1 + rate) ** projection_years)
            sens_value = (present_value_cf + disc_term - debt + cash) / shares
            sensitivity[f'{rate:.1%}'] = float(sens_value)
        
        return {
            'fair_value': float(fair_value),
            'current_price': float(current_price),
            'upside_potential': float(upside),
            'intrinsic_value': float(intrinsic_value),
            'terminal_value': float(terminal_value),
            'present_value_cf': float(present_value_cf),
            'present_value_terminal': float(discounted_terminal),
            'assumptions': {
                'growth_rate': growth_rate,
                'terminal_growth': terminal_growth,
                'discount_rate': discount_rate,
                'projection_years': projection_years
            },
            'sensitivity': sensitivity
        }
    
    def peer_comparison(
        self,
        ticker: str,
        info: Dict,
        peer_infos: Dict[str, Dict]
    ) -> Dict:
        """
        Compare company against peers.
        
        Args:
            ticker: Subject company ticker
            info: Subject company info
            peer_infos: Dictionary of peer tickers to their info
        
        Returns:
            Comparison results with rankings
        """
        all_tickers = [ticker] + list(peer_infos.keys())
        all_infos = {ticker: info, **peer_infos}
        
        metrics = ['trailingPE', 'priceToBook', 'returnOnEquity', 'profitMargins', 'revenueGrowth']
        
        comparison = {}
        for t in all_tickers:
            company_info = all_infos.get(t, {})
            comparison[t] = {
                metric: company_info.get(metric)
                for metric in metrics
            }
        
        # Calculate rankings
        rankings = {}
        for metric in metrics:
            values = [(t, comparison[t].get(metric)) for t in all_tickers if comparison[t].get(metric) is not None]
            values.sort(key=lambda x: x[1], reverse=(metric != 'trailingPE'))  # Lower P/E is better
            
            for rank, (t, _) in enumerate(values, 1):
                if t == ticker:
                    rankings[metric] = rank
        
        # Calculate percentiles
        percentiles = {}
        for metric, rank in rankings.items():
            total = len([t for t in all_tickers if comparison[t].get(metric) is not None])
            percentiles[metric] = ((total - rank + 1) / total) * 100 if total > 0 else 50
        
        return {
            'ticker': ticker,
            'peers': list(peer_infos.keys()),
            'comparison': comparison,
            'rankings': rankings,
            'percentiles': percentiles
        }
    
    def get_quality_grade(self, score: float) -> str:
        """Convert quality score to letter grade."""
        if score >= 90:
            return 'A+'
        elif score >= 80:
            return 'A'
        elif score >= 70:
            return 'B+'
        elif score >= 60:
            return 'B'
        elif score >= 50:
            return 'C'
        elif score >= 40:
            return 'D'
        else:
            return 'F'


# Singleton instance
_fundamental_analyzer = None


def get_fundamental_analyzer() -> FundamentalAnalyzer:
    """Get singleton FundamentalAnalyzer instance."""
    global _fundamental_analyzer
    if _fundamental_analyzer is None:
        _fundamental_analyzer = FundamentalAnalyzer()
    return _fundamental_analyzer
