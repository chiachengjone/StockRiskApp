"""
Fundamental Analysis - Financial Metrics & Valuation
=====================================================
Company fundamentals, ratios, and DCF valuation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging


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
        self.logger = logging.getLogger(__name__)
    
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
            'ticker': info.get('ticker', 'Unknown'),
            'name': info.get('name', 'Unknown'),
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
            'quality_score': None  # Calculated below
        }
        
        # Calculate quality score
        analysis['quality_score'] = self._calculate_quality_score(analysis)
        
        return analysis
    
    def _analyze_valuation(self, info: Dict) -> Dict:
        """Analyze valuation metrics."""
        pe = info.get('pe_ratio')
        forward_pe = info.get('forward_pe')
        pb = info.get('price_to_book')
        ps = info.get('price_to_sales')
        
        # Simple valuation assessment
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
            'peg_ratio': info.get('peg_ratio'),
            'ev_to_ebitda': info.get('ev_to_ebitda'),
            'score': valuation_score,
            'assessments': assessments
        }
    
    def _analyze_profitability(self, info: Dict) -> Dict:
        """Analyze profitability metrics."""
        roe = info.get('roe')
        roa = info.get('roa')
        profit_margin = info.get('profit_margin')
        operating_margin = info.get('operating_margin')
        
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
        debt_to_equity = info.get('debt_to_equity')
        current_ratio = info.get('current_ratio')
        quick_ratio = info.get('quick_ratio')
        
        score = 0
        assessments = []
        
        if debt_to_equity is not None:
            if debt_to_equity < 50:  # Usually expressed as percentage
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
        revenue_growth = info.get('revenue_growth')
        earnings_growth = info.get('earnings_growth')
        
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
            'quarterly_earnings_growth': info.get('quarterly_earnings_growth'),
            'quarterly_revenue_growth': info.get('quarterly_revenue_growth'),
            'score': score,
            'assessments': assessments
        }
    
    def _calculate_quality_score(self, analysis: Dict) -> Dict:
        """Calculate overall quality score."""
        scores = {
            'valuation': analysis['valuation'].get('score', 0),
            'profitability': analysis['profitability'].get('score', 0),
            'financial_health': analysis['financial_health'].get('score', 0),
            'growth': analysis['growth'].get('score', 0)
        }
        
        total_score = sum(scores.values())
        max_score = 16  # 4 categories × 4 max points each
        
        # Letter grade
        pct = total_score / max_score
        if pct >= 0.8:
            grade = 'A'
        elif pct >= 0.6:
            grade = 'B'
        elif pct >= 0.4:
            grade = 'C'
        elif pct >= 0.2:
            grade = 'D'
        else:
            grade = 'F'
        
        return {
            'total_score': total_score,
            'max_score': max_score,
            'percentage': pct,
            'grade': grade,
            'breakdown': scores
        }
    
    def dcf_valuation(
        self,
        free_cash_flow: float,
        growth_rate: float = 0.05,
        terminal_growth: float = 0.02,
        discount_rate: float = 0.10,
        projection_years: int = 5,
        shares_outstanding: float = None
    ) -> Dict:
        """
        Discounted Cash Flow valuation.
        
        Args:
            free_cash_flow: Current free cash flow
            growth_rate: Expected growth rate for projection period
            terminal_growth: Perpetual growth rate after projection period
            discount_rate: Required rate of return (WACC)
            projection_years: Number of years to project
            shares_outstanding: For per-share calculation
        
        Returns:
            Dictionary with DCF valuation results
        """
        if free_cash_flow <= 0:
            return {'error': 'Negative or zero free cash flow'}
        
        # Project cash flows
        projected_fcf = []
        current_fcf = free_cash_flow
        
        for year in range(1, projection_years + 1):
            current_fcf *= (1 + growth_rate)
            projected_fcf.append({
                'year': year,
                'fcf': current_fcf,
                'discount_factor': 1 / (1 + discount_rate) ** year,
                'pv': current_fcf / (1 + discount_rate) ** year
            })
        
        pv_projection = sum(p['pv'] for p in projected_fcf)
        
        # Terminal value
        terminal_fcf = projected_fcf[-1]['fcf'] * (1 + terminal_growth)
        terminal_value = terminal_fcf / (discount_rate - terminal_growth)
        pv_terminal = terminal_value / (1 + discount_rate) ** projection_years
        
        # Enterprise value
        enterprise_value = pv_projection + pv_terminal
        
        result = {
            'free_cash_flow': free_cash_flow,
            'growth_rate': growth_rate,
            'terminal_growth': terminal_growth,
            'discount_rate': discount_rate,
            'projected_fcf': projected_fcf,
            'pv_projection_period': pv_projection,
            'terminal_value': terminal_value,
            'pv_terminal_value': pv_terminal,
            'enterprise_value': enterprise_value
        }
        
        if shares_outstanding:
            result['intrinsic_value_per_share'] = enterprise_value / shares_outstanding
        
        return result
    
    def peer_comparison(
        self,
        target_info: Dict,
        peer_info_list: List[Dict]
    ) -> pd.DataFrame:
        """
        Compare target company against peers.
        
        Args:
            target_info: Target company info
            peer_info_list: List of peer company info dicts
        
        Returns:
            DataFrame with comparison metrics
        """
        all_companies = [target_info] + peer_info_list
        
        metrics = ['pe_ratio', 'price_to_book', 'roe', 'profit_margin', 
                   'debt_to_equity', 'revenue_growth']
        
        data = []
        for info in all_companies:
            row = {
                'Ticker': info.get('ticker', 'Unknown'),
                'Name': info.get('name', 'Unknown')[:20]
            }
            for metric in metrics:
                value = info.get(metric)
                if value is not None:
                    if metric in ['roe', 'profit_margin', 'revenue_growth']:
                        row[metric] = f"{value:.1%}" if isinstance(value, float) else value
                    else:
                        row[metric] = f"{value:.2f}" if isinstance(value, float) else value
                else:
                    row[metric] = 'N/A'
            data.append(row)
        
        df = pd.DataFrame(data)
        return df
    
    def get_financial_ratios_table(self, info: Dict) -> pd.DataFrame:
        """
        Get formatted financial ratios table for display.
        
        Returns:
            DataFrame with ratio categories and values
        """
        ratios = []
        
        # Valuation
        ratios.extend([
            ('Valuation', 'P/E Ratio', self._format_value(info.get('pe_ratio'))),
            ('Valuation', 'Forward P/E', self._format_value(info.get('forward_pe'))),
            ('Valuation', 'Price/Book', self._format_value(info.get('price_to_book'))),
            ('Valuation', 'Price/Sales', self._format_value(info.get('price_to_sales'))),
            ('Valuation', 'EV/EBITDA', self._format_value(info.get('ev_to_ebitda'))),
        ])
        
        # Profitability
        ratios.extend([
            ('Profitability', 'ROE', self._format_percent(info.get('roe'))),
            ('Profitability', 'ROA', self._format_percent(info.get('roa'))),
            ('Profitability', 'Profit Margin', self._format_percent(info.get('profit_margin'))),
            ('Profitability', 'Operating Margin', self._format_percent(info.get('operating_margin'))),
        ])
        
        # Financial Health
        ratios.extend([
            ('Financial Health', 'Debt/Equity', self._format_value(info.get('debt_to_equity'))),
            ('Financial Health', 'Current Ratio', self._format_value(info.get('current_ratio'))),
            ('Financial Health', 'Quick Ratio', self._format_value(info.get('quick_ratio'))),
        ])
        
        # Growth
        ratios.extend([
            ('Growth', 'Revenue Growth', self._format_percent(info.get('revenue_growth'))),
            ('Growth', 'Earnings Growth', self._format_percent(info.get('earnings_growth'))),
        ])
        
        df = pd.DataFrame(ratios, columns=['Category', 'Metric', 'Value'])
        return df
    
    def _format_value(self, value) -> str:
        """Format numeric value for display."""
        if value is None:
            return 'N/A'
        try:
            return f"{float(value):.2f}"
        except (ValueError, TypeError):
            return str(value)
    
    def _format_percent(self, value) -> str:
        """Format percentage value for display."""
        if value is None:
            return 'N/A'
        try:
            return f"{float(value):.1%}"
        except (ValueError, TypeError):
            return str(value)
    
    def calculate_intrinsic_value(
        self,
        info: Dict,
        growth_assumption: float = 0.10,
        discount_rate: float = 0.10
    ) -> Optional[Dict]:
        """
        Calculate intrinsic value using multiple methods.
        
        Returns various valuation estimates.
        """
        eps = info.get('eps')
        book_value = info.get('book_value')
        fcf = info.get('free_cash_flow')
        shares = info.get('shares_outstanding')
        current_price = info.get('price')
        
        valuations = {}
        
        # Graham Number (conservative intrinsic value)
        if eps and book_value and eps > 0 and book_value > 0:
            graham_number = np.sqrt(22.5 * eps * book_value)
            valuations['graham_number'] = graham_number
        
        # Earnings Power Value
        if eps:
            epv = eps / discount_rate
            valuations['earnings_power_value'] = epv
        
        # Peter Lynch PEG-based
        if info.get('pe_ratio') and info.get('peg_ratio'):
            # Fair value when PEG = 1
            fair_pe = growth_assumption * 100  # Convert to points
            if info.get('pe_ratio') > 0:
                peg_fair_value = current_price * (fair_pe / info['pe_ratio'])
                valuations['peg_fair_value'] = peg_fair_value
        
        # DCF if we have FCF
        if fcf and shares and fcf > 0:
            dcf_result = self.dcf_valuation(
                fcf, 
                growth_assumption, 
                0.02, 
                discount_rate,
                shares_outstanding=shares
            )
            if 'intrinsic_value_per_share' in dcf_result:
                valuations['dcf_value'] = dcf_result['intrinsic_value_per_share']
        
        if not valuations:
            return None
        
        # Average of available methods
        avg_value = np.mean(list(valuations.values()))
        
        return {
            'methods': valuations,
            'average_intrinsic_value': avg_value,
            'current_price': current_price,
            'upside_potential': (avg_value - current_price) / current_price if current_price else None,
            'margin_of_safety': (avg_value - current_price) / avg_value if avg_value else None
        }
