"""
Application Settings & Configuration
=====================================
Centralized configuration for API keys, cache settings, and app constants.
"""

import os
from pathlib import Path

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# =============================================================================
# BASE PATHS
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = BASE_DIR / "cache"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# =============================================================================
# API KEYS
# =============================================================================
ALPHA_VANTAGE_KEY = os.getenv('ALPHA_VANTAGE_KEY', '')

# =============================================================================
# EMAIL SETTINGS (for alerts)
# =============================================================================
SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
SMTP_PORT = int(os.getenv('SMTP_PORT', 587))
EMAIL_ADDRESS = os.getenv('EMAIL_ADDRESS', '')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD', '')

# =============================================================================
# CACHE SETTINGS
# =============================================================================
CACHE_ENABLED = True
CACHE_TTL_MINUTES = 60  # 1 hour default cache

# =============================================================================
# DATA SOURCE PRIORITY
# =============================================================================
# Order determines fallback priority
DATA_SOURCE_PRIORITY = ['yahoo', 'alpha_vantage']

# =============================================================================
# RATE LIMITS (requests per minute)
# =============================================================================
RATE_LIMITS = {
    'yahoo': 2000,  # Generous limit
    'alpha_vantage': 5,  # Free tier: 5 calls/min, 500/day
}

# =============================================================================
# DATABASE
# =============================================================================
DATABASE_PATH = str(DATA_DIR / 'app_data.db')
ALERTS_CONFIG_PATH = str(DATA_DIR / 'alerts_config.json')

# =============================================================================
# APP CONSTANTS
# =============================================================================
APP_VERSION = "4.0"
APP_NAME = "Stock Risk Model"

# Trading days per year
TRADING_DAYS = 252

# Default risk-free rate
DEFAULT_RF_RATE = 0.045  # 4.5%

# Monte Carlo defaults
DEFAULT_MC_SIMULATIONS = 10000
MAX_MC_SIMULATIONS = 50000

# =============================================================================
# UI COLORS (iOS-inspired palette)
# =============================================================================
COLORS = {
    'primary': '#007AFF',
    'secondary': '#5856D6',
    'success': '#34C759',
    'warning': '#FF9500',
    'danger': '#FF3B30',
    'gray': '#8E8E93',
    'light': '#F2F2F7',
    'dark': '#1C1C1E',
    'background': '#0e1117',
    'surface': '#1a1d24',
    'text': '#fafafa',
    'text_secondary': '#a0a0a0'
}

# =============================================================================
# STRESS SCENARIOS
# =============================================================================
STRESS_SCENARIOS = {
    "Black Monday 1987": {
        "market_shock": -0.20,
        "vol_multiplier": 3.0,
        "description": "Oct 19, 1987: -20% single day"
    },
    "Dot-com Crash 2000": {
        "market_shock": -0.45,
        "vol_multiplier": 2.0,
        "description": "2000-2002: Tech bubble burst"
    },
    "GFC 2008": {
        "market_shock": -0.50,
        "vol_multiplier": 4.0,
        "description": "2008: Lehman collapse, -50% peak-trough"
    },
    "COVID Crash 2020": {
        "market_shock": -0.35,
        "vol_multiplier": 5.0,
        "description": "Mar 2020: Fastest 30% drop ever"
    },
    "Mild Correction": {
        "market_shock": -0.10,
        "vol_multiplier": 1.5,
        "description": "Typical 10% pullback"
    },
    "Severe Bear": {
        "market_shock": -0.30,
        "vol_multiplier": 2.5,
        "description": "Extended bear market"
    },
    "Flash Crash": {
        "market_shock": -0.09,
        "vol_multiplier": 6.0,
        "description": "May 6, 2010: Intraday -9%"
    },
    "Rate Shock": {
        "market_shock": -0.15,
        "vol_multiplier": 2.0,
        "description": "Sudden interest rate spike"
    }
}

# =============================================================================
# ASSET DEFINITIONS
# =============================================================================
POPULAR_STOCKS = {
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "Google (GOOGL)": "GOOGL",
    "Amazon (AMZN)": "AMZN",
    "Tesla (TSLA)": "TSLA",
    "NVIDIA (NVDA)": "NVDA",
    "Meta (META)": "META",
    "Netflix (NFLX)": "NFLX",
    "JPMorgan (JPM)": "JPM",
    "Berkshire (BRK-B)": "BRK-B",
    "Johnson & Johnson (JNJ)": "JNJ",
    "Visa (V)": "V",
    "Walmart (WMT)": "WMT",
    "Mastercard (MA)": "MA",
    "Disney (DIS)": "DIS",
    "AMD (AMD)": "AMD",
    "Intel (INTC)": "INTC",
    "Coca-Cola (KO)": "KO",
    "PepsiCo (PEP)": "PEP",
    "Pfizer (PFE)": "PFE",
    "Custom...": "CUSTOM"
}

POPULAR_BENCHMARKS = {
    "S&P 500": "^GSPC",
    "Dow Jones": "^DJI",
    "NASDAQ 100": "^NDX",
    "NASDAQ Composite": "^IXIC",
    "Russell 2000": "^RUT",
    "VIX": "^VIX",
    "FTSE 100": "^FTSE",
    "DAX": "^GDAXI",
    "Nikkei 225": "^N225",
    "Hang Seng": "^HSI",
    "Custom...": "CUSTOM"
}

CRYPTO_ASSETS = {
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "Solana": "SOL-USD",
    "Cardano": "ADA-USD",
    "XRP": "XRP-USD"
}

ETF_ASSETS = {
    "SPY (S&P 500)": "SPY",
    "QQQ (NASDAQ)": "QQQ",
    "IWM (Russell 2000)": "IWM",
    "VTI (Total Market)": "VTI",
    "GLD (Gold)": "GLD",
    "TLT (20Y Treasury)": "TLT",
    "EEM (Emerging Markets)": "EEM",
    "VNQ (Real Estate)": "VNQ",
    "XLF (Financials)": "XLF",
    "XLK (Technology)": "XLK"
}

# =============================================================================
# VALIDATION
# =============================================================================
def validate_config():
    """Validate critical configuration settings."""
    warnings = []
    
    if not ALPHA_VANTAGE_KEY:
        warnings.append("ALPHA_VANTAGE_KEY not set - Alpha Vantage will be disabled")
    
    if not EMAIL_ADDRESS:
        warnings.append("EMAIL_ADDRESS not set - Email alerts will be disabled")
    
    return warnings
