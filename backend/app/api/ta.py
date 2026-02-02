"""
Technical Analysis API Routes
=============================
Endpoints for technical indicators.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import logging

from app.services.data_service import get_data_service
from app.services.ta_service import (
    calculate_sma, calculate_ema, calculate_rsi,
    calculate_macd, calculate_bollinger_bands,
    calculate_adx, calculate_stochastic, calculate_atr,
    get_all_indicators
)

router = APIRouter()
logger = logging.getLogger(__name__)


# ============================================================================
# SCHEMAS
# ============================================================================

class TARequest(BaseModel):
    """Request for technical analysis."""
    ticker: str
    start_date: str
    end_date: str


class IndicatorConfig(BaseModel):
    """Configuration for indicators."""
    sma_periods: List[int] = Field(default=[9, 21, 50])
    ema_periods: List[int] = Field(default=[12, 26])
    rsi_period: int = Field(default=14)
    macd_fast: int = Field(default=12)
    macd_slow: int = Field(default=26)
    macd_signal: int = Field(default=9)
    bb_period: int = Field(default=20)
    bb_std: float = Field(default=2.0)
    adx_period: int = Field(default=14)
    stoch_k: int = Field(default=14)
    stoch_d: int = Field(default=3)
    atr_period: int = Field(default=14)


class FullTARequest(BaseModel):
    """Request for full technical analysis."""
    ticker: str
    start_date: str
    end_date: str
    config: Optional[IndicatorConfig] = None


class TAResponse(BaseModel):
    """Response with technical indicators."""
    ticker: str
    dates: List[str]
    indicators: dict


class SignalResponse(BaseModel):
    """Response with trading signals."""
    ticker: str
    date: str
    signals: dict
    summary: str


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post("/full", response_model=TAResponse)
async def get_full_ta(request: FullTARequest):
    """
    Calculate all technical indicators.
    """
    data_service = get_data_service()
    df = data_service.fetch_historical(
        request.ticker, request.start_date, request.end_date
    )
    
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {request.ticker}")
    
    config = request.config or IndicatorConfig()
    
    indicators = get_all_indicators(
        df,
        sma_periods=config.sma_periods,
        ema_periods=config.ema_periods
    )
    
    # Convert to serializable format
    result = {}
    for key, series in indicators.items():
        result[key] = [None if isinstance(x, float) and (x != x) else float(x) if x is not None else None 
                      for x in series.tolist()]
    
    return TAResponse(
        ticker=request.ticker,
        dates=df.index.strftime('%Y-%m-%d').tolist(),
        indicators=result
    )


@router.get("/sma/{ticker}")
async def get_sma(
    ticker: str,
    days: int = Query(252, ge=30, le=1000),
    periods: str = Query("9,21,50")
):
    """Calculate Simple Moving Averages."""
    from datetime import datetime, timedelta
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    data_service = get_data_service()
    df = data_service.fetch_historical(ticker, start_date, end_date)
    
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {ticker}")
    
    period_list = [int(p.strip()) for p in periods.split(',')]
    
    result = {
        "ticker": ticker,
        "dates": df.index.strftime('%Y-%m-%d').tolist(),
        "close": df['Close'].tolist()
    }
    
    for period in period_list:
        sma = calculate_sma(df['Close'], period)
        result[f"SMA_{period}"] = [None if x != x else float(x) for x in sma.tolist()]
    
    return result


@router.get("/ema/{ticker}")
async def get_ema(
    ticker: str,
    days: int = Query(252, ge=30, le=1000),
    periods: str = Query("12,26")
):
    """Calculate Exponential Moving Averages."""
    from datetime import datetime, timedelta
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    data_service = get_data_service()
    df = data_service.fetch_historical(ticker, start_date, end_date)
    
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {ticker}")
    
    period_list = [int(p.strip()) for p in periods.split(',')]
    
    result = {
        "ticker": ticker,
        "dates": df.index.strftime('%Y-%m-%d').tolist(),
        "close": df['Close'].tolist()
    }
    
    for period in period_list:
        ema = calculate_ema(df['Close'], period)
        result[f"EMA_{period}"] = [None if x != x else float(x) for x in ema.tolist()]
    
    return result


@router.get("/rsi/{ticker}")
async def get_rsi(
    ticker: str,
    days: int = Query(252, ge=30, le=1000),
    period: int = Query(14, ge=5, le=50)
):
    """Calculate Relative Strength Index."""
    from datetime import datetime, timedelta
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    data_service = get_data_service()
    df = data_service.fetch_historical(ticker, start_date, end_date)
    
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {ticker}")
    
    rsi = calculate_rsi(df['Close'], period)
    
    return {
        "ticker": ticker,
        "period": period,
        "dates": df.index.strftime('%Y-%m-%d').tolist(),
        "close": df['Close'].tolist(),
        "rsi": [None if x != x else float(x) for x in rsi.tolist()],
        "current_rsi": float(rsi.iloc[-1]) if not (rsi.iloc[-1] != rsi.iloc[-1]) else None,
        "signal": "Overbought" if rsi.iloc[-1] > 70 else "Oversold" if rsi.iloc[-1] < 30 else "Neutral"
    }


@router.get("/macd/{ticker}")
async def get_macd(
    ticker: str,
    days: int = Query(252, ge=60, le=1000),
    fast: int = Query(12),
    slow: int = Query(26),
    signal: int = Query(9)
):
    """Calculate MACD indicator."""
    from datetime import datetime, timedelta
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    data_service = get_data_service()
    df = data_service.fetch_historical(ticker, start_date, end_date)
    
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {ticker}")
    
    macd_line, signal_line, histogram = calculate_macd(df['Close'], fast, slow, signal)
    
    return {
        "ticker": ticker,
        "dates": df.index.strftime('%Y-%m-%d').tolist(),
        "close": df['Close'].tolist(),
        "macd": [None if x != x else float(x) for x in macd_line.tolist()],
        "signal": [None if x != x else float(x) for x in signal_line.tolist()],
        "histogram": [None if x != x else float(x) for x in histogram.tolist()],
        "current_macd": float(macd_line.iloc[-1]) if not (macd_line.iloc[-1] != macd_line.iloc[-1]) else None,
        "trend": "Bullish" if histogram.iloc[-1] > 0 else "Bearish"
    }


@router.get("/bollinger/{ticker}")
async def get_bollinger_bands(
    ticker: str,
    days: int = Query(252, ge=30, le=1000),
    period: int = Query(20),
    std: float = Query(2.0)
):
    """Calculate Bollinger Bands."""
    from datetime import datetime, timedelta
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    data_service = get_data_service()
    df = data_service.fetch_historical(ticker, start_date, end_date)
    
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {ticker}")
    
    upper, middle, lower = calculate_bollinger_bands(df['Close'], period, std)
    
    # Calculate %B (position within bands)
    current_price = df['Close'].iloc[-1]
    band_width = upper.iloc[-1] - lower.iloc[-1]
    pct_b = (current_price - lower.iloc[-1]) / band_width if band_width > 0 else 0.5
    
    return {
        "ticker": ticker,
        "dates": df.index.strftime('%Y-%m-%d').tolist(),
        "close": df['Close'].tolist(),
        "upper": [None if x != x else float(x) for x in upper.tolist()],
        "middle": [None if x != x else float(x) for x in middle.tolist()],
        "lower": [None if x != x else float(x) for x in lower.tolist()],
        "percent_b": float(pct_b),
        "signal": "Near Upper" if pct_b > 0.8 else "Near Lower" if pct_b < 0.2 else "Within Bands"
    }


@router.get("/signals/{ticker}", response_model=SignalResponse)
async def get_trading_signals(ticker: str, days: int = Query(252, ge=60, le=1000)):
    """
    Get consolidated trading signals from multiple indicators.
    """
    from datetime import datetime, timedelta
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    data_service = get_data_service()
    df = data_service.fetch_historical(ticker, start_date, end_date)
    
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {ticker}")
    
    close = df['Close']
    
    # Calculate indicators
    rsi = calculate_rsi(close)
    macd_line, signal_line, histogram = calculate_macd(close)
    upper, middle, lower = calculate_bollinger_bands(close)
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    
    # Generate signals
    signals = {}
    bullish = 0
    bearish = 0
    
    # RSI Signal
    current_rsi = float(rsi.iloc[-1])
    if current_rsi < 30:
        signals['rsi'] = {'value': current_rsi, 'signal': 'Bullish (Oversold)', 'score': 1}
        bullish += 1
    elif current_rsi > 70:
        signals['rsi'] = {'value': current_rsi, 'signal': 'Bearish (Overbought)', 'score': -1}
        bearish += 1
    else:
        signals['rsi'] = {'value': current_rsi, 'signal': 'Neutral', 'score': 0}
    
    # MACD Signal
    if histogram.iloc[-1] > 0 and histogram.iloc[-2] <= 0:
        signals['macd'] = {'value': float(histogram.iloc[-1]), 'signal': 'Bullish Crossover', 'score': 1}
        bullish += 1
    elif histogram.iloc[-1] < 0 and histogram.iloc[-2] >= 0:
        signals['macd'] = {'value': float(histogram.iloc[-1]), 'signal': 'Bearish Crossover', 'score': -1}
        bearish += 1
    elif histogram.iloc[-1] > 0:
        signals['macd'] = {'value': float(histogram.iloc[-1]), 'signal': 'Bullish', 'score': 0.5}
    else:
        signals['macd'] = {'value': float(histogram.iloc[-1]), 'signal': 'Bearish', 'score': -0.5}
    
    # Moving Average Signal
    current_price = float(close.iloc[-1])
    if current_price > sma_50.iloc[-1] > sma_200.iloc[-1]:
        signals['trend'] = {'signal': 'Bullish (Price > SMA50 > SMA200)', 'score': 1}
        bullish += 1
    elif current_price < sma_50.iloc[-1] < sma_200.iloc[-1]:
        signals['trend'] = {'signal': 'Bearish (Price < SMA50 < SMA200)', 'score': -1}
        bearish += 1
    else:
        signals['trend'] = {'signal': 'Mixed', 'score': 0}
    
    # Bollinger Band Signal
    pct_b = (current_price - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1])
    if pct_b < 0.2:
        signals['bollinger'] = {'percent_b': float(pct_b), 'signal': 'Near Support', 'score': 0.5}
    elif pct_b > 0.8:
        signals['bollinger'] = {'percent_b': float(pct_b), 'signal': 'Near Resistance', 'score': -0.5}
    else:
        signals['bollinger'] = {'percent_b': float(pct_b), 'signal': 'Within Range', 'score': 0}
    
    # Summary
    total_score = sum(s.get('score', 0) for s in signals.values())
    if total_score >= 2:
        summary = "Strong Bullish Signal"
    elif total_score >= 1:
        summary = "Bullish Signal"
    elif total_score <= -2:
        summary = "Strong Bearish Signal"
    elif total_score <= -1:
        summary = "Bearish Signal"
    else:
        summary = "Neutral / Mixed Signals"
    
    return SignalResponse(
        ticker=ticker,
        date=df.index[-1].strftime('%Y-%m-%d'),
        signals=signals,
        summary=summary
    )
