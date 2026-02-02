"""
Application Configuration
=========================
Environment-based settings for the API.
"""

from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    """Application settings loaded from environment."""
    
    # API Settings
    API_V1_PREFIX: str = "/api"
    DEBUG: bool = False
    
    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]
    
    # Data Provider Keys (optional)
    ALPHA_VANTAGE_KEY: str = ""
    POLYGON_KEY: str = ""
    ALPACA_KEY: str = ""
    ALPACA_SECRET: str = ""
    
    # Cache Settings
    CACHE_TTL_MINUTES: int = 60
    CACHE_DIR: str = "./cache"
    
    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./stockrisk.db"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


settings = Settings()
