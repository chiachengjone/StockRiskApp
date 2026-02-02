"""
Reports Service
===============
PDF report generation for risk analysis.
"""

import io
from datetime import datetime
from typing import Dict, List, Optional
import logging

try:
    from fpdf import FPDF
    HAS_FPDF = True
except ImportError:
    HAS_FPDF = False

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generate professional PDF risk reports.
    
    Features:
    - Executive summary
    - Risk metrics tables
    - Charts and visualizations
    - Appendix with methodology
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def is_available(self) -> bool:
        """Check if PDF generation is available."""
        return HAS_FPDF
    
    def generate_single_stock_report(
        self,
        ticker: str,
        metrics: Dict,
        var_data: Dict,
        info: Dict = None,
    ) -> Optional[bytes]:
        """
        Generate a comprehensive single stock risk report.
        
        Args:
            ticker: Stock symbol
            metrics: Risk metrics dictionary
            var_data: VaR analysis data
            info: Company info dictionary (optional)
        
        Returns:
            PDF bytes
        """
        if not HAS_FPDF:
            self.logger.error("fpdf2 not installed. Run: pip install fpdf2")
            return None
        
        pdf = RiskReportPDF(ticker)
        
        # Title page
        pdf.add_title_page(ticker, info)
        
        # Executive Summary
        pdf.add_page()
        pdf.add_executive_summary(ticker, metrics, var_data)
        
        # Risk Metrics Detail
        pdf.add_page()
        pdf.add_risk_metrics_section(metrics)
        
        # VaR Analysis
        pdf.add_page()
        pdf.add_var_section(var_data)
        
        # Methodology appendix
        pdf.add_page()
        pdf.add_methodology_appendix()
        
        # Disclaimer
        pdf.add_disclaimer()
        
        return bytes(pdf.output())
    
    def generate_portfolio_report(
        self,
        portfolio_name: str,
        tickers: List[str],
        weights: List[float],
        metrics: Dict,
        var_data: Dict,
    ) -> Optional[bytes]:
        """
        Generate a comprehensive portfolio risk report.
        """
        if not HAS_FPDF:
            self.logger.error("fpdf2 not installed. Run: pip install fpdf2")
            return None
        
        pdf = RiskReportPDF(portfolio_name, is_portfolio=True)
        
        # Title page
        pdf.add_title_page(portfolio_name, {'type': 'portfolio', 'assets': len(tickers)})
        
        # Portfolio Summary
        pdf.add_page()
        pdf.add_portfolio_summary(tickers, weights, metrics)
        
        # Risk Metrics
        pdf.add_page()
        pdf.add_risk_metrics_section(metrics)
        
        # VaR Analysis
        pdf.add_page()
        pdf.add_var_section(var_data)
        
        # Methodology
        pdf.add_page()
        pdf.add_methodology_appendix()
        
        # Disclaimer
        pdf.add_disclaimer()
        
        return bytes(pdf.output())
    
    def generate_comparison_report(
        self,
        tickers: List[str],
        metrics_list: List[Dict],
    ) -> Optional[bytes]:
        """Generate a comparison report for multiple stocks."""
        if not HAS_FPDF:
            return None
        
        pdf = RiskReportPDF("Stock Comparison")
        
        # Title
        pdf.add_title_page("Stock Comparison", {'tickers': tickers})
        
        # Comparison table
        pdf.add_page()
        pdf.add_comparison_table(tickers, metrics_list)
        
        # Disclaimer
        pdf.add_disclaimer()
        
        return bytes(pdf.output())


class RiskReportPDF(FPDF if HAS_FPDF else object):
    """Custom PDF class for risk reports."""
    
    def __init__(self, title: str, is_portfolio: bool = False):
        if not HAS_FPDF:
            return
        
        super().__init__()
        self.title = title
        self.is_portfolio = is_portfolio
        self.set_auto_page_break(auto=True, margin=15)
    
    def header(self):
        """Page header."""
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Risk Analysis Report - {self.title}', 0, 1, 'L')
        self.line(10, 18, 200, 18)
        self.ln(5)
    
    def footer(self):
        """Page footer."""
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()} | Generated {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 0, 'C')
    
    def add_title_page(self, title: str, info: Dict = None):
        """Add title page."""
        self.add_page()
        self.set_font('Helvetica', 'B', 28)
        self.set_text_color(0, 0, 0)
        
        self.ln(60)
        self.cell(0, 15, 'Risk Analysis Report', 0, 1, 'C')
        
        self.set_font('Helvetica', 'B', 24)
        self.set_text_color(0, 122, 255)
        self.cell(0, 15, title, 0, 1, 'C')
        
        self.set_font('Helvetica', '', 12)
        self.set_text_color(100, 100, 100)
        self.ln(10)
        self.cell(0, 8, f'Generated: {datetime.now().strftime("%B %d, %Y")}', 0, 1, 'C')
        
        if info:
            if info.get('name'):
                self.cell(0, 8, info['name'], 0, 1, 'C')
            if info.get('sector'):
                self.cell(0, 8, f"Sector: {info['sector']}", 0, 1, 'C')
        
        self.ln(40)
        self.set_font('Helvetica', 'I', 10)
        self.cell(0, 8, 'Stock Risk Model v4.3', 0, 1, 'C')
    
    def add_executive_summary(self, ticker: str, metrics: Dict, var_data: Dict):
        """Add executive summary section."""
        self.set_font('Helvetica', 'B', 16)
        self.set_text_color(0, 0, 0)
        self.cell(0, 10, 'Executive Summary', 0, 1, 'L')
        self.ln(5)
        
        self.set_font('Helvetica', '', 11)
        
        ann_vol = metrics.get('ann_vol', metrics.get('volatility', 0))
        risk_level = 'High' if abs(ann_vol) > 0.30 else 'Moderate' if abs(ann_vol) > 0.20 else 'Low'
        
        summary = f"""This report provides a comprehensive risk analysis for {ticker}. 

Key findings from the analysis:

- Annualized Return: {metrics.get('ann_ret', metrics.get('annualized_return', 0)):.2%}
- Annualized Volatility: {ann_vol:.2%}
- Maximum Drawdown: {metrics.get('max_dd', metrics.get('max_drawdown', 0)):.2%}
- Sharpe Ratio: {metrics.get('sharpe', metrics.get('sharpe_ratio', 0)):.2f}
- Sortino Ratio: {metrics.get('sortino', metrics.get('sortino_ratio', 0)):.2f}
- 95% Value at Risk: {var_data.get('var_95', 0):.2%}
- Expected Shortfall (CVaR): {var_data.get('cvar', 0):.2%}

Risk Assessment: {risk_level} volatility profile."""
        
        self.multi_cell(0, 6, summary.strip())
    
    def add_risk_metrics_section(self, metrics: Dict):
        """Add detailed risk metrics table."""
        self.set_font('Helvetica', 'B', 16)
        self.cell(0, 10, 'Risk Metrics Detail', 0, 1, 'L')
        self.ln(5)
        
        # Create table
        self.set_font('Helvetica', 'B', 10)
        self.set_fill_color(240, 240, 240)
        self.cell(90, 8, 'Metric', 1, 0, 'C', True)
        self.cell(90, 8, 'Value', 1, 1, 'C', True)
        
        self.set_font('Helvetica', '', 10)
        
        metrics_display = [
            ('Annualized Return', f"{metrics.get('ann_ret', metrics.get('annualized_return', 0)):.2%}"),
            ('Annualized Volatility', f"{metrics.get('ann_vol', metrics.get('volatility', 0)):.2%}"),
            ('Maximum Drawdown', f"{metrics.get('max_dd', metrics.get('max_drawdown', 0)):.2%}"),
            ('Sharpe Ratio', f"{metrics.get('sharpe', metrics.get('sharpe_ratio', 0)):.2f}"),
            ('Sortino Ratio', f"{metrics.get('sortino', metrics.get('sortino_ratio', 0)):.2f}"),
            ('Calmar Ratio', f"{metrics.get('calmar', metrics.get('calmar_ratio', 0)):.2f}"),
            ('Skewness', f"{metrics.get('skew', metrics.get('skewness', 0)):.2f}"),
            ('Kurtosis', f"{metrics.get('kurtosis', 0):.2f}"),
        ]
        
        for name, value in metrics_display:
            self.cell(90, 7, name, 1, 0, 'L')
            self.cell(90, 7, value, 1, 1, 'C')
    
    def add_var_section(self, var_data: Dict):
        """Add VaR analysis section."""
        self.set_font('Helvetica', 'B', 16)
        self.cell(0, 10, 'Value at Risk Analysis', 0, 1, 'L')
        self.ln(5)
        
        self.set_font('Helvetica', '', 11)
        self.multi_cell(0, 6, """Value at Risk (VaR) represents the maximum expected loss at a given confidence level. Multiple methodologies are used to capture different aspects of tail risk.""")
        self.ln(5)
        
        # VaR table
        self.set_font('Helvetica', 'B', 10)
        self.set_fill_color(240, 240, 240)
        self.cell(60, 8, 'Method', 1, 0, 'C', True)
        self.cell(60, 8, '95% VaR', 1, 0, 'C', True)
        self.cell(60, 8, '99% VaR', 1, 1, 'C', True)
        
        self.set_font('Helvetica', '', 10)
        
        methods = [
            ('Parametric (Normal)', var_data.get('var_95', 0), var_data.get('var_99', 0)),
            ('Historical', var_data.get('hist_var_95', 0), var_data.get('hist_var_99', 0)),
            ('CVaR/ES', var_data.get('cvar', 0), var_data.get('cvar_99', 0)),
        ]
        
        for method, var95, var99 in methods:
            self.cell(60, 7, method, 1, 0, 'L')
            self.cell(60, 7, f"{var95:.2%}" if var95 else "N/A", 1, 0, 'C')
            self.cell(60, 7, f"{var99:.2%}" if var99 else "N/A", 1, 1, 'C')
    
    def add_portfolio_summary(self, tickers: List[str], weights: List[float], metrics: Dict):
        """Add portfolio composition summary."""
        self.set_font('Helvetica', 'B', 16)
        self.cell(0, 10, 'Portfolio Composition', 0, 1, 'L')
        self.ln(5)
        
        # Allocation table
        self.set_font('Helvetica', 'B', 10)
        self.set_fill_color(240, 240, 240)
        self.cell(90, 8, 'Asset', 1, 0, 'C', True)
        self.cell(90, 8, 'Weight', 1, 1, 'C', True)
        
        self.set_font('Helvetica', '', 10)
        
        for ticker, weight in zip(tickers, weights):
            self.cell(90, 7, ticker, 1, 0, 'L')
            self.cell(90, 7, f"{weight:.1%}", 1, 1, 'C')
        
        # Portfolio metrics
        self.ln(10)
        self.set_font('Helvetica', 'B', 14)
        self.cell(0, 10, 'Portfolio Metrics', 0, 1, 'L')
        self.ln(3)
        
        self.set_font('Helvetica', '', 10)
        
        portfolio_metrics = [
            ('Expected Return', f"{metrics.get('expected_return', 0):.2%}"),
            ('Portfolio Volatility', f"{metrics.get('volatility', 0):.2%}"),
            ('Sharpe Ratio', f"{metrics.get('sharpe_ratio', 0):.2f}"),
            ('Diversification Ratio', f"{metrics.get('diversification_ratio', 0):.2f}"),
        ]
        
        for name, value in portfolio_metrics:
            self.cell(90, 7, name, 0, 0, 'L')
            self.cell(90, 7, value, 0, 1, 'R')
    
    def add_comparison_table(self, tickers: List[str], metrics_list: List[Dict]):
        """Add stock comparison table."""
        self.set_font('Helvetica', 'B', 16)
        self.cell(0, 10, 'Stock Comparison', 0, 1, 'L')
        self.ln(5)
        
        # Header
        col_width = 180 / (len(tickers) + 1)
        self.set_font('Helvetica', 'B', 9)
        self.set_fill_color(240, 240, 240)
        self.cell(col_width, 8, 'Metric', 1, 0, 'C', True)
        for ticker in tickers:
            self.cell(col_width, 8, ticker, 1, 0, 'C', True)
        self.ln()
        
        # Metrics to compare
        comparison_metrics = [
            ('Return', 'ann_ret', '{:.2%}'),
            ('Volatility', 'ann_vol', '{:.2%}'),
            ('Sharpe', 'sharpe', '{:.2f}'),
            ('Max DD', 'max_dd', '{:.2%}'),
            ('VaR 95%', 'var_95', '{:.2%}'),
        ]
        
        self.set_font('Helvetica', '', 9)
        for display_name, key, fmt in comparison_metrics:
            self.cell(col_width, 7, display_name, 1, 0, 'L')
            for metrics in metrics_list:
                value = metrics.get(key, 0)
                self.cell(col_width, 7, fmt.format(value), 1, 0, 'C')
            self.ln()
    
    def add_methodology_appendix(self):
        """Add methodology appendix."""
        self.set_font('Helvetica', 'B', 16)
        self.cell(0, 10, 'Methodology', 0, 1, 'L')
        self.ln(5)
        
        self.set_font('Helvetica', '', 10)
        
        methodology_text = """
Value at Risk (VaR)
-------------------
VaR estimates the maximum expected loss at a given confidence level over a specific time horizon.

- Parametric VaR: Assumes normal distribution, uses mean and standard deviation
- Historical VaR: Uses actual historical return percentiles
- Conditional VaR (CVaR): Average loss beyond VaR threshold

GARCH Volatility
----------------
GARCH(1,1) models capture volatility clustering and time-varying risk, providing more accurate short-term volatility forecasts.

Extreme Value Theory (EVT)
--------------------------
EVT focuses on tail risk modeling using the Generalized Pareto Distribution (GPD) to estimate extreme loss probabilities.

Monte Carlo Simulation
----------------------
Monte Carlo simulation generates thousands of potential price paths to estimate risk metrics and their distributions.

Portfolio Optimization
----------------------
- Risk Parity: Equalizes risk contribution from each asset
- Hierarchical Risk Parity (HRP): Uses clustering to build diversified portfolios
- Black-Litterman: Combines market equilibrium with investor views
"""
        
        self.multi_cell(0, 5, methodology_text.strip())
    
    def add_disclaimer(self):
        """Add disclaimer."""
        self.ln(10)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(100, 100, 100)
        
        disclaimer = """
DISCLAIMER: This report is for informational purposes only and does not constitute investment advice. Past performance is not indicative of future results. Risk metrics are estimates based on historical data and statistical models. Actual results may differ materially. The user assumes all responsibility for investment decisions.
"""
        self.multi_cell(0, 4, disclaimer.strip())


# Singleton instance
_report_generator = None


def get_report_generator() -> ReportGenerator:
    """Get singleton report generator instance."""
    global _report_generator
    if _report_generator is None:
        _report_generator = ReportGenerator()
    return _report_generator
