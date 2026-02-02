"""
FastAPI Application Entry Point
================================
Stock Risk Modelling API v1.0
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.api import risk, ml, portfolio, data, ta
from app.core.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management."""
    logger.info("Starting Stock Risk API...")
    yield
    logger.info("Shutting down Stock Risk API...")


# Create FastAPI application
app = FastAPI(
    title="Stock Risk Modelling API",
    description="""
    Comprehensive stock risk analysis and portfolio optimization API.
    
    ## Features
    
    * **Risk Analysis**: VaR, CVaR, GARCH, EVT, Monte Carlo
    * **Machine Learning**: XGBoost/GradientBoosting VaR prediction
    * **Portfolio Optimization**: Risk Parity, HRP, Black-Litterman
    * **Technical Analysis**: SMA, EMA, RSI, MACD, Bollinger Bands
    * **Data Aggregation**: Multi-provider with fallback
    """,
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(risk.router, prefix="/api/risk", tags=["Risk Analysis"])
app.include_router(ml.router, prefix="/api/ml", tags=["Machine Learning"])
app.include_router(portfolio.router, prefix="/api/portfolio", tags=["Portfolio"])
app.include_router(data.router, prefix="/api/data", tags=["Data"])
app.include_router(ta.router, prefix="/api/ta", tags=["Technical Analysis"])


@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "name": "Stock Risk Modelling API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
