"""
Reports API Routes
==================
Endpoints for PDF report generation.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import List, Optional
from pydantic import BaseModel
import io
import logging

from app.services.reports_service import get_report_generator
from app.services.data_service import get_data_service
from app.services.risk_service import (
    compute_metrics,
    parametric_var,
    historical_var,
    cvar
)

router = APIRouter()
logger = logging.getLogger(__name__)


class SingleStockReportRequest(BaseModel):
    """Request for single stock report."""
    ticker: str
    start_date: str
    end_date: str


class PortfolioReportRequest(BaseModel):
    """Request for portfolio report."""
    portfolio_name: str
    tickers: List[str]
    weights: List[float]
    start_date: str
    end_date: str


class ComparisonReportRequest(BaseModel):
    """Request for comparison report."""
    tickers: List[str]
    start_date: str
    end_date: str


@router.get("/available")
async def check_availability():
    """Check if PDF generation is available."""
    generator = get_report_generator()
    return {
        "available": generator.is_available(),
        "message": "PDF generation available" if generator.is_available() else "Install fpdf2: pip install fpdf2"
    }


@router.post("/single-stock")
async def generate_single_stock_report(request: SingleStockReportRequest):
    """
    Generate a comprehensive PDF risk report for a single stock.
    
    Returns a downloadable PDF file.
    """
    generator = get_report_generator()
    
    if not generator.is_available():
        raise HTTPException(
            status_code=503,
            detail="PDF generation unavailable. Install: pip install fpdf2"
        )
    
    data_service = get_data_service()
    
    # Fetch stock data
    prices, returns = data_service.get_returns(
        request.ticker, request.start_date, request.end_date
    )
    
    if returns.empty:
        raise HTTPException(status_code=404, detail=f"No data for {request.ticker}")
    
    # Calculate risk metrics
    metrics = compute_metrics(returns)
    
    # Calculate VaR data
    var_data = {
        'var_95': parametric_var(returns, 0.95),
        'var_99': parametric_var(returns, 0.99),
        'hist_var_95': historical_var(returns, 0.95),
        'hist_var_99': historical_var(returns, 0.99),
        'cvar': cvar(returns, 0.95),
        'cvar_99': cvar(returns, 0.99),
    }
    
    # Get company info
    info = data_service.get_fundamentals(request.ticker)
    
    # Generate PDF
    pdf_bytes = generator.generate_single_stock_report(
        ticker=request.ticker,
        metrics=metrics,
        var_data=var_data,
        info=info
    )
    
    if pdf_bytes is None:
        raise HTTPException(status_code=500, detail="Failed to generate PDF")
    
    # Return as streaming response
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename={request.ticker}_risk_report.pdf"
        }
    )


@router.post("/portfolio")
async def generate_portfolio_report(request: PortfolioReportRequest):
    """
    Generate a comprehensive PDF risk report for a portfolio.
    
    Returns a downloadable PDF file.
    """
    import pandas as pd
    import numpy as np
    
    generator = get_report_generator()
    
    if not generator.is_available():
        raise HTTPException(
            status_code=503,
            detail="PDF generation unavailable. Install: pip install fpdf2"
        )
    
    # Validate weights
    if len(request.tickers) != len(request.weights):
        raise HTTPException(status_code=400, detail="Tickers and weights must match")
    
    if abs(sum(request.weights) - 1.0) > 0.01:
        raise HTTPException(status_code=400, detail="Weights must sum to 1.0")
    
    data_service = get_data_service()
    
    # Fetch returns for all tickers
    returns_dict = {}
    for ticker in request.tickers:
        try:
            _, returns = data_service.get_returns(
                ticker, request.start_date, request.end_date
            )
            if not returns.empty:
                returns_dict[ticker] = returns
        except Exception as e:
            logger.warning(f"Could not fetch {ticker}: {e}")
    
    if not returns_dict:
        raise HTTPException(status_code=404, detail="No valid data for any tickers")
    
    returns_df = pd.DataFrame(returns_dict).dropna()
    weights = np.array(request.weights[:len(returns_df.columns)])
    weights = weights / weights.sum()  # Normalize
    
    # Calculate portfolio returns
    portfolio_returns = returns_df @ weights
    
    # Calculate metrics
    metrics = compute_metrics(portfolio_returns)
    
    # Calculate VaR data
    var_data = {
        'var_95': parametric_var(portfolio_returns, 0.95),
        'var_99': parametric_var(portfolio_returns, 0.99),
        'hist_var_95': historical_var(portfolio_returns, 0.95),
        'hist_var_99': historical_var(portfolio_returns, 0.99),
        'cvar': cvar(portfolio_returns, 0.95),
        'cvar_99': cvar(portfolio_returns, 0.99),
    }
    
    # Generate PDF
    pdf_bytes = generator.generate_portfolio_report(
        portfolio_name=request.portfolio_name,
        tickers=request.tickers,
        weights=request.weights,
        metrics=metrics,
        var_data=var_data
    )
    
    if pdf_bytes is None:
        raise HTTPException(status_code=500, detail="Failed to generate PDF")
    
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename={request.portfolio_name}_portfolio_report.pdf"
        }
    )


@router.post("/comparison")
async def generate_comparison_report(request: ComparisonReportRequest):
    """
    Generate a comparison PDF report for multiple stocks.
    
    Returns a downloadable PDF file.
    """
    generator = get_report_generator()
    
    if not generator.is_available():
        raise HTTPException(
            status_code=503,
            detail="PDF generation unavailable. Install: pip install fpdf2"
        )
    
    data_service = get_data_service()
    
    metrics_list = []
    valid_tickers = []
    
    for ticker in request.tickers:
        try:
            _, returns = data_service.get_returns(
                ticker, request.start_date, request.end_date
            )
            if not returns.empty:
                metrics = compute_metrics(returns)
                metrics['var_95'] = parametric_var(returns, 0.95)
                metrics_list.append(metrics)
                valid_tickers.append(ticker)
        except Exception as e:
            logger.warning(f"Could not fetch {ticker}: {e}")
    
    if not valid_tickers:
        raise HTTPException(status_code=404, detail="No valid data for any tickers")
    
    # Generate PDF
    pdf_bytes = generator.generate_comparison_report(
        tickers=valid_tickers,
        metrics_list=metrics_list
    )
    
    if pdf_bytes is None:
        raise HTTPException(status_code=500, detail="Failed to generate PDF")
    
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={
            "Content-Disposition": "attachment; filename=stock_comparison_report.pdf"
        }
    )


@router.get("/quick/{ticker}")
async def quick_single_stock_report(ticker: str, days: int = 252):
    """
    Quick PDF report generation with default date range.
    """
    from datetime import datetime, timedelta
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    request = SingleStockReportRequest(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date
    )
    
    return await generate_single_stock_report(request)
