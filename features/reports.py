"""
Report Generator - Professional PDF Reports
=============================================
Generate institutional-quality risk reports.
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

try:
    import plotly.io as pio
    HAS_KALEIDO = True
except ImportError:
    HAS_KALEIDO = False


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
        charts: Dict = None,
        info: Dict = None,
        output_path: str = None
    ) -> Optional[bytes]:
        """
        Generate a comprehensive single stock risk report.
        
        Args:
            ticker: Stock symbol
            metrics: Risk metrics dictionary
            var_data: VaR analysis data
            charts: Dictionary of plotly figures (optional)
            info: Company info dictionary (optional)
            output_path: Save path (optional, returns bytes if not provided)
        
        Returns:
            PDF bytes if output_path not provided
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
        
        # Charts (if available and kaleido installed)
        if charts and HAS_KALEIDO:
            for chart_name, fig in charts.items():
                try:
                    pdf.add_page()
                    pdf.add_chart(fig, chart_name)
                except Exception as e:
                    self.logger.warning(f"Could not add chart {chart_name}: {e}")
        
        # Methodology appendix
        pdf.add_page()
        pdf.add_methodology_appendix()
        
        # Disclaimer
        pdf.add_disclaimer()
        
        if output_path:
            pdf.output(output_path)
            return None
        else:
            return bytes(pdf.output())
    
    def generate_portfolio_report(
        self,
        portfolio_name: str,
        tickers: List[str],
        weights: List[float],
        metrics: Dict,
        var_data: Dict,
        correlation_matrix: 'pd.DataFrame' = None,
        charts: Dict = None,
        output_path: str = None
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
        
        if output_path:
            pdf.output(output_path)
            return None
        else:
            return bytes(pdf.output())
    
    def generate_comparison_report(
        self,
        tickers: List[str],
        metrics_list: List[Dict],
        output_path: str = None
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
        
        if output_path:
            pdf.output(output_path)
            return None
        else:
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
        self.cell(0, 8, 'Stock Risk Model v4.0', 0, 1, 'C')
    
    def add_executive_summary(self, ticker: str, metrics: Dict, var_data: Dict):
        """Add executive summary section."""
        self.set_font('Helvetica', 'B', 16)
        self.set_text_color(0, 0, 0)
        self.cell(0, 10, 'Executive Summary', 0, 1, 'L')
        self.ln(5)
        
        self.set_font('Helvetica', '', 11)
        
        summary = f"""
This report provides a comprehensive risk analysis for {ticker}. 

Key findings from the analysis:

- Annualized Return: {metrics.get('ann_ret', 0):.2%}
- Annualized Volatility: {metrics.get('ann_vol', 0):.2%}
- Maximum Drawdown: {metrics.get('max_dd', 0):.2%}
- Sharpe Ratio: {metrics.get('sharpe', 0):.2f}
- Sortino Ratio: {metrics.get('sortino', 0):.2f}
- 95% Value at Risk: {var_data.get('var_95', 0):.2%}
- Expected Shortfall (CVaR): {var_data.get('cvar', 0):.2%}

Risk Assessment: {'High' if abs(metrics.get('ann_vol', 0)) > 0.30 else 'Moderate' if abs(metrics.get('ann_vol', 0)) > 0.20 else 'Low'} volatility profile.
        """
        
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
            ('Annualized Return', f"{metrics.get('ann_ret', 0):.2%}"),
            ('Annualized Volatility', f"{metrics.get('ann_vol', 0):.2%}"),
            ('Maximum Drawdown', f"{metrics.get('max_dd', 0):.2%}"),
            ('Sharpe Ratio', f"{metrics.get('sharpe', 0):.2f}"),
            ('Sortino Ratio', f"{metrics.get('sortino', 0):.2f}"),
            ('Calmar Ratio', f"{metrics.get('calmar', 0):.2f}"),
            ('Skewness', f"{metrics.get('skew', 0):.2f}"),
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
        self.multi_cell(0, 6, """
Value at Risk (VaR) represents the maximum expected loss at a given confidence level. 
Multiple methodologies are used to capture different aspects of tail risk.
        """.strip())
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
    
    def add_comparison_table(self, tickers: List[str], metrics_list: List[Dict]):
        """Add stock comparison table."""
        self.set_font('Helvetica', 'B', 16)
        self.cell(0, 10, 'Stock Comparison', 0, 1, 'L')
        self.ln(5)
        
        # Headers
        col_width = 180 / (len(tickers) + 1)
        
        self.set_font('Helvetica', 'B', 9)
        self.set_fill_color(240, 240, 240)
        self.cell(col_width, 8, 'Metric', 1, 0, 'C', True)
        for ticker in tickers:
            self.cell(col_width, 8, ticker, 1, 0, 'C', True)
        self.ln()
        
        # Rows
        self.set_font('Helvetica', '', 9)
        metrics_to_compare = ['ann_ret', 'ann_vol', 'max_dd', 'sharpe', 'sortino']
        labels = ['Ann. Return', 'Volatility', 'Max DD', 'Sharpe', 'Sortino']
        
        for label, metric in zip(labels, metrics_to_compare):
            self.cell(col_width, 7, label, 1, 0, 'L')
            for metrics in metrics_list:
                value = metrics.get(metric, 0)
                if metric in ['ann_ret', 'ann_vol', 'max_dd']:
                    self.cell(col_width, 7, f"{value:.2%}", 1, 0, 'C')
                else:
                    self.cell(col_width, 7, f"{value:.2f}", 1, 0, 'C')
            self.ln()
    
    def add_chart(self, fig, title: str):
        """Add a plotly chart to the PDF."""
        self.set_font('Helvetica', 'B', 14)
        self.cell(0, 10, title, 0, 1, 'L')
        
        try:
            import tempfile
            import os
            
            # Convert plotly figure to image
            img_bytes = pio.to_image(fig, format='png', width=700, height=400)
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                tmp.write(img_bytes)
                tmp_path = tmp.name
            
            # Add to PDF
            self.image(tmp_path, x=15, w=180)
            
            # Clean up
            os.unlink(tmp_path)
            
        except Exception as e:
            self.set_font('Helvetica', 'I', 10)
            self.cell(0, 10, f'[Chart could not be rendered: {str(e)}]', 0, 1, 'C')
    
    def add_methodology_appendix(self):
        """Add methodology explanation."""
        self.set_font('Helvetica', 'B', 16)
        self.cell(0, 10, 'Methodology', 0, 1, 'L')
        self.ln(5)
        
        self.set_font('Helvetica', '', 10)
        
        methodology = """
VALUE AT RISK (VaR)
VaR represents the maximum expected loss at a specified confidence level over a given time horizon.

Parametric VaR: VaR = mu + sigma x Z_alpha x sqrt(t)
Assumes returns follow a normal distribution.

Historical VaR: Based on the empirical distribution of historical returns.
The alpha-quantile of the return distribution.

CONDITIONAL VALUE AT RISK (CVaR)
Also known as Expected Shortfall, CVaR measures the average loss beyond the VaR threshold.
Provides a more complete picture of tail risk.

SHARPE RATIO
Measures risk-adjusted return: (R_p - R_f) / sigma_p
Higher values indicate better risk-adjusted performance.

SORTINO RATIO
Similar to Sharpe but uses downside deviation instead of total volatility.
Focuses on harmful volatility only.

MAXIMUM DRAWDOWN
Largest peak-to-trough decline in portfolio value.
Measures the worst-case scenario for an investor.
        """
        
        self.multi_cell(0, 5, methodology.strip())
    
    def add_disclaimer(self):
        """Add disclaimer at the end."""
        self.add_page()
        self.set_font('Helvetica', 'B', 14)
        self.cell(0, 10, 'Important Disclaimer', 0, 1, 'L')
        self.ln(5)
        
        self.set_font('Helvetica', '', 9)
        self.set_text_color(100, 100, 100)
        
        disclaimer = """
This report is for informational and educational purposes only and should not be considered as financial advice. 
The analysis and models presented have inherent limitations and are based on historical data, which may not be indicative of future performance.

Key limitations:
- All models assume certain statistical properties that may not hold in practice
- Past performance does not guarantee future results
- Market conditions can change rapidly and unpredictably
- This analysis does not account for transaction costs, taxes, or other real-world factors

Always consult with a qualified financial advisor before making investment decisions. 
Use this information at your own risk.

Generated by Stock Risk Model v4.0
        """
        
        self.multi_cell(0, 5, disclaimer.strip())
