"""
Data API Routes
===============
Endpoints for fetching stock data and company info.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from datetime import datetime, timedelta
import logging

from pydantic import BaseModel, Field

from app.services.data_service import get_data_service

router = APIRouter()
logger = logging.getLogger(__name__)


# ============================================================================
# SCHEMAS
# ============================================================================

class HistoricalDataRequest(BaseModel):
    """Request for historical data."""
    ticker: str
    start_date: str
    end_date: str


class PriceDataResponse(BaseModel):
    """Response with price data."""
    ticker: str
    dates: List[str]
    open: List[Optional[float]]
    high: List[Optional[float]]
    low: List[Optional[float]]
    close: List[Optional[float]]
    volume: List[Optional[int]]
    returns: List[Optional[float]]


class CompanyInfoResponse(BaseModel):
    """Response with company information."""
    ticker: str
    name: Optional[str]
    sector: Optional[str]
    industry: Optional[str]
    market_cap: Optional[float]
    pe_ratio: Optional[float]
    forward_pe: Optional[float]
    price_to_book: Optional[float]
    dividend_yield: Optional[float]
    beta: Optional[float]
    current_price: Optional[float]


class QuoteResponse(BaseModel):
    """Response with real-time quote."""
    ticker: str
    price: Optional[float]
    change: Optional[float]
    change_pct: Optional[float]
    volume: Optional[int]
    market_cap: Optional[float]
    timestamp: str


class MultiTickerRequest(BaseModel):
    """Request for multiple tickers."""
    tickers: List[str] = Field(..., min_length=1, max_length=20)
    start_date: str
    end_date: str


class ReturnsMatrixResponse(BaseModel):
    """Response with returns matrix for multiple tickers."""
    tickers: List[str]
    dates: List[str]
    returns: dict  # ticker -> list of returns
    correlation_matrix: dict


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post("/historical", response_model=PriceDataResponse)
async def get_historical_data(request: HistoricalDataRequest):
    """
    Fetch historical OHLCV data for a stock.
    """
    data_service = get_data_service()
    df = data_service.fetch_historical(
        request.ticker, request.start_date, request.end_date
    )
    
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {request.ticker}")
    
    # Calculate returns - replace NaN/Inf with None for JSON compatibility
    import numpy as np
    returns_series = np.log(df['Close'] / df['Close'].shift(1))
    returns = [None if (np.isnan(x) or np.isinf(x)) else float(x) for x in returns_series]
    
    # Helper function to clean values
    def clean_list(series):
        return [None if (np.isnan(x) or np.isinf(x)) else float(x) for x in series]
    
    return PriceDataResponse(
        ticker=request.ticker,
        dates=df.index.strftime('%Y-%m-%d').tolist(),
        open=clean_list(df['Open']) if 'Open' in df.columns else [],
        high=clean_list(df['High']) if 'High' in df.columns else [],
        low=clean_list(df['Low']) if 'Low' in df.columns else [],
        close=clean_list(df['Close']),
        volume=df['Volume'].astype(int).tolist() if 'Volume' in df.columns else [],
        returns=returns
    )


@router.get("/historical/{ticker}")
async def quick_historical(
    ticker: str,
    days: int = Query(252, ge=30, le=2000)
):
    """Quick historical data with default parameters."""
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    data_service = get_data_service()
    df = data_service.fetch_historical(ticker, start_date, end_date)
    
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {ticker}")
    
    import numpy as np
    returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
    
    return {
        "ticker": ticker,
        "dates": df.index.strftime('%Y-%m-%d').tolist(),
        "close": df['Close'].tolist(),
        "returns": returns.tolist()
    }


@router.get("/info/{ticker}", response_model=CompanyInfoResponse)
async def get_company_info(ticker: str):
    """
    Get company information and fundamentals.
    """
    data_service = get_data_service()
    info = data_service.fetch_info(ticker)
    
    if 'error' in info:
        raise HTTPException(status_code=404, detail=info['error'])
    
    return CompanyInfoResponse(
        ticker=ticker,
        name=info.get('name'),
        sector=info.get('sector'),
        industry=info.get('industry'),
        market_cap=info.get('market_cap'),
        pe_ratio=info.get('pe_ratio'),
        forward_pe=info.get('forward_pe'),
        price_to_book=info.get('price_to_book'),
        dividend_yield=info.get('dividend_yield'),
        beta=info.get('beta'),
        current_price=info.get('current_price')
    )


@router.get("/quote/{ticker}", response_model=QuoteResponse)
async def get_quote(ticker: str):
    """
    Get real-time quote for a stock.
    """
    data_service = get_data_service()
    quote = data_service.get_real_time_quote(ticker)
    
    if 'error' in quote:
        raise HTTPException(status_code=404, detail=quote['error'])
    
    return QuoteResponse(
        ticker=ticker,
        price=quote.get('price'),
        change=quote.get('change'),
        change_pct=quote.get('change_pct'),
        volume=quote.get('volume'),
        market_cap=quote.get('market_cap'),
        timestamp=quote.get('timestamp', datetime.now().isoformat())
    )


@router.post("/multiple", response_model=ReturnsMatrixResponse)
async def get_multiple_tickers(request: MultiTickerRequest):
    """
    Fetch data for multiple tickers and calculate correlation matrix.
    """
    data_service = get_data_service()
    df = data_service.fetch_multiple(
        request.tickers, request.start_date, request.end_date
    )
    
    if df.empty:
        raise HTTPException(status_code=404, detail="No data found for the requested tickers")
    
    import numpy as np
    
    # Calculate returns
    returns_df = np.log(df / df.shift(1)).dropna()
    
    # Correlation matrix
    corr = returns_df.corr()
    
    return ReturnsMatrixResponse(
        tickers=df.columns.tolist(),
        dates=returns_df.index.strftime('%Y-%m-%d').tolist(),
        returns={col: returns_df[col].tolist() for col in returns_df.columns},
        correlation_matrix={col: corr[col].to_dict() for col in corr.columns}
    )


@router.get("/search")
async def search_tickers(q: str = Query(..., min_length=1)):
    """
    Search for stock tickers.
    
    Note: Limited functionality without premium API.
    """
    import yfinance as yf
    
    try:
        ticker = yf.Ticker(q.upper())
        info = ticker.info
        
        if info.get('symbol'):
            return [{
                "symbol": info.get('symbol'),
                "name": info.get('longName', info.get('shortName', '')),
                "exchange": info.get('exchange', ''),
                "type": info.get('quoteType', '')
            }]
        return []
    except:
        return []
