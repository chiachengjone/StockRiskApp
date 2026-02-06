"""
Market Regime Detection Service
================================
Identify market regimes and optimize strategies accordingly

Provides:
- Market regime classification (Bull/Bear/Sideways)
- Volatility regime detection
- Regime change alerts
- Regime-based strategy recommendations

Author: Stock Risk App | Feb 2026
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging

try:
    from sklearn.mixture import GaussianMixture
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from .ta_service import calculate_sma, calculate_atr

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class MarketRegime(Enum):
    """Market regime types."""
    STRONG_BULL = "Strong Bull"
    BULL = "Bull"
    SIDEWAYS = "Sideways"
    BEAR = "Bear"
    STRONG_BEAR = "Strong Bear"
    CRASH = "Crash"
    RECOVERY = "Recovery"


class VolatilityRegime(Enum):
    """Volatility regime types."""
    VERY_LOW = "Very Low Volatility"
    LOW = "Low Volatility"
    NORMAL = "Normal Volatility"
    HIGH = "High Volatility"
    EXTREME = "Extreme Volatility"


@dataclass
class RegimeAnalysis:
    """Regime analysis result."""
    symbol: str
    market_regime: MarketRegime
    volatility_regime: VolatilityRegime
    regime_probability: float  # 0-100
    regime_duration_days: int
    trend_strength: float  # 0-100
    volatility_percentile: float
    recent_change: bool
    change_date: Optional[datetime]
    recommended_strategies: List[str]
    indicators: Dict[str, float]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'market_regime': self.market_regime.value,
            'volatility_regime': self.volatility_regime.value,
            'regime_probability': self.regime_probability,
            'regime_duration_days': self.regime_duration_days,
            'trend_strength': self.trend_strength,
            'volatility_percentile': self.volatility_percentile,
            'recent_change': self.recent_change,
            'change_date': self.change_date.isoformat() if self.change_date else None,
            'recommended_strategies': self.recommended_strategies,
            'indicators': self.indicators,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class RegimeChangeAlert:
    """Alert for regime change."""
    symbol: str
    old_regime: MarketRegime
    new_regime: MarketRegime
    change_date: datetime
    significance: str  # 'Major', 'Moderate', 'Minor'
    action_required: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'old_regime': self.old_regime.value,
            'new_regime': self.new_regime.value,
            'change_date': self.change_date.isoformat(),
            'significance': self.significance,
            'action_required': self.action_required
        }


# ============================================================================
# REGIME DETECTION FUNCTIONS
# ============================================================================

def calculate_returns_stats(
    returns: pd.Series,
    window: int = 21
) -> Dict[str, pd.Series]:
    """
    Calculate rolling return statistics.
    
    Args:
        returns: Returns series
        window: Rolling window
        
    Returns:
        Dictionary with rolling statistics
    """
    stats = {
        'rolling_mean': returns.rolling(window).mean() * 252,  # Annualized
        'rolling_std': returns.rolling(window).std() * np.sqrt(252),
        'rolling_skew': returns.rolling(window).skew(),
        'rolling_kurt': returns.rolling(window).apply(lambda x: x.kurtosis() if len(x) > 3 else 0)
    }
    return stats


def classify_market_regime_simple(
    close: pd.Series,
    returns: pd.Series,
    lookback: int = 50
) -> Tuple[MarketRegime, float]:
    """
    Simple market regime classification based on trend and returns.
    
    Args:
        close: Close price series
        returns: Returns series
        lookback: Days to look back
        
    Returns:
        Tuple of (MarketRegime, confidence)
    """
    if len(close) < lookback:
        return MarketRegime.SIDEWAYS, 50.0
    
    # Calculate metrics
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200) if len(close) >= 200 else calculate_sma(close, len(close) // 2)
    
    current_price = close.iloc[-1]
    
    # Get latest values
    latest_sma50 = sma_50.iloc[-1] if not pd.isna(sma_50.iloc[-1]) else current_price
    latest_sma200 = sma_200.iloc[-1] if not pd.isna(sma_200.iloc[-1]) else current_price
    
    # Recent return
    recent_return = (close.iloc[-1] / close.iloc[-lookback] - 1) * 100 if len(close) >= lookback else 0
    
    # Trend metrics
    price_vs_sma50 = (current_price / latest_sma50 - 1) * 100
    price_vs_sma200 = (current_price / latest_sma200 - 1) * 100
    sma_trend = (latest_sma50 / latest_sma200 - 1) * 100
    
    # Volatility
    recent_vol = returns.iloc[-lookback:].std() * np.sqrt(252) * 100
    
    # Classify
    if recent_return > 20 and price_vs_sma50 > 5 and price_vs_sma200 > 10:
        regime = MarketRegime.STRONG_BULL
        confidence = min(95, 70 + recent_return)
    elif recent_return > 5 and price_vs_sma50 > 0 and sma_trend > 0:
        regime = MarketRegime.BULL
        confidence = min(85, 60 + recent_return * 2)
    elif recent_return < -20 and price_vs_sma50 < -5 and price_vs_sma200 < -10:
        if recent_vol > 40:
            regime = MarketRegime.CRASH
            confidence = min(95, 70 + abs(recent_return))
        else:
            regime = MarketRegime.STRONG_BEAR
            confidence = min(90, 70 + abs(recent_return))
    elif recent_return < -5 and price_vs_sma50 < 0 and sma_trend < 0:
        regime = MarketRegime.BEAR
        confidence = min(80, 60 + abs(recent_return) * 2)
    elif abs(recent_return) < 5 and abs(price_vs_sma50) < 3:
        regime = MarketRegime.SIDEWAYS
        confidence = 70
    elif recent_return > 10 and price_vs_sma200 < 0 and price_vs_sma50 > 0:
        regime = MarketRegime.RECOVERY
        confidence = 65
    else:
        regime = MarketRegime.SIDEWAYS
        confidence = 55
    
    return regime, confidence


def classify_market_regime_gmm(
    returns: pd.Series,
    n_regimes: int = 3
) -> Tuple[MarketRegime, float, np.ndarray]:
    """
    Market regime classification using Gaussian Mixture Model.
    
    Args:
        returns: Returns series
        n_regimes: Number of regimes to detect
        
    Returns:
        Tuple of (current MarketRegime, probability, historical regimes)
    """
    if not HAS_SKLEARN:
        logger.warning("sklearn not available, using simple classification")
        return MarketRegime.SIDEWAYS, 50.0, np.array([])
    
    if len(returns) < 100:
        return MarketRegime.SIDEWAYS, 50.0, np.array([])
    
    # Prepare data
    returns_clean = returns.dropna().values.reshape(-1, 1)
    
    try:
        # Fit GMM
        gmm = GaussianMixture(n_components=n_regimes, random_state=42, n_init=5)
        gmm.fit(returns_clean)
        
        # Get regime labels
        regime_labels = gmm.predict(returns_clean)
        regime_probs = gmm.predict_proba(returns_clean)
        
        # Current regime
        current_regime_idx = regime_labels[-1]
        current_prob = regime_probs[-1, current_regime_idx] * 100
        
        # Characterize regimes by mean return
        regime_means = gmm.means_.flatten()
        regime_order = np.argsort(regime_means)
        
        # Map to MarketRegime
        if n_regimes == 3:
            regime_mapping = {
                regime_order[0]: MarketRegime.BEAR,
                regime_order[1]: MarketRegime.SIDEWAYS,
                regime_order[2]: MarketRegime.BULL
            }
        else:
            # For other n_regimes, use simple mapping
            regime_mapping = {i: MarketRegime.SIDEWAYS for i in range(n_regimes)}
        
        current_market_regime = regime_mapping.get(current_regime_idx, MarketRegime.SIDEWAYS)
        
        return current_market_regime, current_prob, regime_labels
        
    except Exception as e:
        logger.error(f"GMM fitting error: {e}")
        return MarketRegime.SIDEWAYS, 50.0, np.array([])


def classify_volatility_regime(
    returns: pd.Series,
    lookback: int = 252
) -> Tuple[VolatilityRegime, float]:
    """
    Classify volatility regime.
    
    Args:
        returns: Returns series
        lookback: Historical lookback for percentile calculation
        
    Returns:
        Tuple of (VolatilityRegime, percentile)
    """
    if len(returns) < 30:
        return VolatilityRegime.NORMAL, 50.0
    
    # Current volatility
    current_vol = returns.iloc[-21:].std() * np.sqrt(252) * 100  # Annualized %
    
    # Historical volatility distribution
    rolling_vol = returns.rolling(21).std() * np.sqrt(252) * 100
    rolling_vol = rolling_vol.dropna()
    
    if len(rolling_vol) < 10:
        return VolatilityRegime.NORMAL, 50.0
    
    # Calculate percentile
    percentile = (rolling_vol < current_vol).mean() * 100
    
    # Classify
    if percentile < 10:
        regime = VolatilityRegime.VERY_LOW
    elif percentile < 30:
        regime = VolatilityRegime.LOW
    elif percentile < 70:
        regime = VolatilityRegime.NORMAL
    elif percentile < 90:
        regime = VolatilityRegime.HIGH
    else:
        regime = VolatilityRegime.EXTREME
    
    return regime, percentile


def get_regime_strategies(
    market_regime: MarketRegime,
    volatility_regime: VolatilityRegime
) -> List[str]:
    """
    Get recommended strategies based on regimes.
    
    Args:
        market_regime: Current market regime
        volatility_regime: Current volatility regime
        
    Returns:
        List of recommended strategies
    """
    strategies = []
    
    # Market regime strategies
    if market_regime in [MarketRegime.STRONG_BULL, MarketRegime.BULL]:
        strategies.extend([
            "Trend following (long bias)",
            "Buy dips on pullbacks",
            "Use momentum indicators",
            "Wider stop losses"
        ])
    elif market_regime in [MarketRegime.STRONG_BEAR, MarketRegime.BEAR]:
        strategies.extend([
            "Trend following (short bias)",
            "Sell rallies",
            "Consider hedging positions",
            "Reduce position sizes"
        ])
    elif market_regime == MarketRegime.CRASH:
        strategies.extend([
            "Reduce exposure significantly",
            "Focus on capital preservation",
            "Look for oversold bounces only",
            "Consider hedging with options"
        ])
    elif market_regime == MarketRegime.RECOVERY:
        strategies.extend([
            "Gradually increase exposure",
            "Focus on quality stocks",
            "Buy breakouts with confirmation",
            "Use tight stops initially"
        ])
    else:  # Sideways
        strategies.extend([
            "Mean reversion strategies",
            "Range trading (buy support, sell resistance)",
            "Use oscillators (RSI, Stochastic)",
            "Reduce position sizes"
        ])
    
    # Volatility regime adjustments
    if volatility_regime in [VolatilityRegime.HIGH, VolatilityRegime.EXTREME]:
        strategies.append("Reduce position sizes due to high volatility")
        strategies.append("Use wider stops to avoid whipsaws")
    elif volatility_regime in [VolatilityRegime.VERY_LOW, VolatilityRegime.LOW]:
        strategies.append("Potential breakout approaching - prepare for volatility expansion")
        strategies.append("Consider volatility strategies (straddles/strangles)")
    
    return strategies


def detect_regime_change(
    regime_history: List[Tuple[datetime, MarketRegime]],
    lookback_days: int = 30
) -> Optional[RegimeChangeAlert]:
    """
    Detect regime changes in recent history.
    
    Args:
        regime_history: List of (date, regime) tuples
        lookback_days: Days to look back for changes
        
    Returns:
        RegimeChangeAlert if change detected, None otherwise
    """
    if len(regime_history) < 2:
        return None
    
    # Check for recent regime change
    for i in range(len(regime_history) - 1, 0, -1):
        current_date, current_regime = regime_history[i]
        prev_date, prev_regime = regime_history[i - 1]
        
        if current_regime != prev_regime:
            # Determine significance
            major_changes = [
                (MarketRegime.BULL, MarketRegime.BEAR),
                (MarketRegime.BEAR, MarketRegime.BULL),
                (MarketRegime.STRONG_BULL, MarketRegime.BEAR),
                (MarketRegime.BULL, MarketRegime.CRASH),
                (MarketRegime.SIDEWAYS, MarketRegime.CRASH),
            ]
            
            if (prev_regime, current_regime) in major_changes or (current_regime, prev_regime) in major_changes:
                significance = "Major"
                action = "Review all positions and adjust strategy immediately"
            elif prev_regime in [MarketRegime.BULL, MarketRegime.BEAR] or current_regime in [MarketRegime.BULL, MarketRegime.BEAR]:
                significance = "Moderate"
                action = "Consider adjusting position sizes and strategies"
            else:
                significance = "Minor"
                action = "Monitor for confirmation of regime change"
            
            return RegimeChangeAlert(
                symbol="",  # Will be set by caller
                old_regime=prev_regime,
                new_regime=current_regime,
                change_date=current_date,
                significance=significance,
                action_required=action
            )
    
    return None


# ============================================================================
# REGIME SERVICE CLASS
# ============================================================================

class RegimeService:
    """
    Market Regime Detection Service.
    
    Provides comprehensive market regime analysis and strategy recommendations.
    """
    
    def __init__(self, use_gmm: bool = True):
        """
        Initialize the Regime Service.
        
        Args:
            use_gmm: Whether to use GMM for regime detection (requires sklearn)
        """
        self.use_gmm = use_gmm and HAS_SKLEARN
        self._cache: Dict[str, RegimeAnalysis] = {}
        self._regime_history: Dict[str, List[Tuple[datetime, MarketRegime]]] = {}
        logger.info(f"RegimeService initialized (GMM: {self.use_gmm})")
    
    def analyze_regime(
        self,
        symbol: str,
        df: pd.DataFrame,
        use_cache: bool = True
    ) -> RegimeAnalysis:
        """
        Analyze market regime for a symbol.
        
        Args:
            symbol: Stock symbol
            df: OHLCV DataFrame
            use_cache: Whether to use cached results
            
        Returns:
            RegimeAnalysis
        """
        cache_key = f"{symbol}_{len(df)}"
        
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        # Handle DataFrame structure
        if isinstance(df.columns, pd.MultiIndex):
            close = df['Close'].iloc[:, 0]
            high = df['High'].iloc[:, 0]
            low = df['Low'].iloc[:, 0]
        else:
            close = df['Close']
            high = df['High']
            low = df['Low']
        
        # Calculate returns
        returns = close.pct_change().dropna()
        
        # Market regime classification
        if self.use_gmm and len(returns) >= 100:
            market_regime, regime_prob, _ = classify_market_regime_gmm(returns)
        else:
            market_regime, regime_prob = classify_market_regime_simple(close, returns)
        
        # Volatility regime
        vol_regime, vol_percentile = classify_volatility_regime(returns)
        
        # Calculate additional indicators
        sma_50 = calculate_sma(close, 50)
        sma_200 = calculate_sma(close, 200) if len(close) >= 200 else None
        atr = calculate_atr(high, low, close, 14)
        
        # Trend strength (simplified ADX-like measure)
        recent_returns = returns.iloc[-21:]
        trend_strength = min(100, abs(recent_returns.sum()) / recent_returns.std() * 10) if recent_returns.std() > 0 else 50
        
        # Check for regime change
        self._update_regime_history(symbol, market_regime, datetime.now())
        regime_change = detect_regime_change(self._regime_history.get(symbol, []))
        
        # Calculate regime duration
        regime_duration = self._calculate_regime_duration(symbol, market_regime)
        
        # Get strategy recommendations
        strategies = get_regime_strategies(market_regime, vol_regime)
        
        # Build indicators dict
        indicators = {
            'current_price': float(close.iloc[-1]),
            'sma_50': float(sma_50.iloc[-1]) if not pd.isna(sma_50.iloc[-1]) else None,
            'sma_200': float(sma_200.iloc[-1]) if sma_200 is not None and not pd.isna(sma_200.iloc[-1]) else None,
            'atr_14': float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else None,
            'return_1m': float((close.iloc[-1] / close.iloc[-21] - 1) * 100) if len(close) > 21 else 0,
            'return_3m': float((close.iloc[-1] / close.iloc[-63] - 1) * 100) if len(close) > 63 else 0,
            'volatility_21d': float(returns.iloc[-21:].std() * np.sqrt(252) * 100) if len(returns) > 21 else 0
        }
        
        result = RegimeAnalysis(
            symbol=symbol,
            market_regime=market_regime,
            volatility_regime=vol_regime,
            regime_probability=regime_prob,
            regime_duration_days=regime_duration,
            trend_strength=trend_strength,
            volatility_percentile=vol_percentile,
            recent_change=regime_change is not None,
            change_date=regime_change.change_date if regime_change else None,
            recommended_strategies=strategies,
            indicators=indicators,
            timestamp=datetime.now()
        )
        
        if use_cache:
            self._cache[cache_key] = result
        
        return result
    
    def _update_regime_history(
        self,
        symbol: str,
        regime: MarketRegime,
        date: datetime
    ) -> None:
        """Update regime history for a symbol."""
        if symbol not in self._regime_history:
            self._regime_history[symbol] = []
        
        history = self._regime_history[symbol]
        
        # Only add if different from last regime
        if not history or history[-1][1] != regime:
            history.append((date, regime))
            
            # Keep only last 100 entries
            if len(history) > 100:
                self._regime_history[symbol] = history[-100:]
    
    def _calculate_regime_duration(
        self,
        symbol: str,
        current_regime: MarketRegime
    ) -> int:
        """Calculate how long we've been in current regime."""
        history = self._regime_history.get(symbol, [])
        
        if not history:
            return 0
        
        duration = 0
        for i in range(len(history) - 1, -1, -1):
            if history[i][1] == current_regime:
                duration += 1
            else:
                break
        
        return duration
    
    def get_regime_history(
        self,
        symbol: str,
        df: pd.DataFrame,
        window: int = 21
    ) -> pd.DataFrame:
        """
        Calculate historical regime classification.
        
        Args:
            symbol: Stock symbol
            df: OHLCV DataFrame
            window: Assessment window
            
        Returns:
            DataFrame with regime history
        """
        if isinstance(df.columns, pd.MultiIndex):
            close = df['Close'].iloc[:, 0]
        else:
            close = df['Close']
        
        returns = close.pct_change().dropna()
        
        regimes = []
        dates = []
        
        for i in range(window, len(close)):
            subset_close = close.iloc[:i+1]
            subset_returns = returns.iloc[:i]
            
            regime, _ = classify_market_regime_simple(subset_close, subset_returns, lookback=min(50, i))
            regimes.append(regime.value)
            dates.append(close.index[i])
        
        return pd.DataFrame({
            'date': dates,
            'regime': regimes
        })
    
    def scan_for_regime_changes(
        self,
        symbols: List[str],
        dfs: Dict[str, pd.DataFrame]
    ) -> List[RegimeChangeAlert]:
        """
        Scan multiple symbols for recent regime changes.
        
        Args:
            symbols: List of stock symbols
            dfs: Dictionary of symbol -> DataFrame
            
        Returns:
            List of regime change alerts
        """
        alerts = []
        
        for symbol in symbols:
            if symbol not in dfs:
                continue
            
            try:
                analysis = self.analyze_regime(symbol, dfs[symbol])
                
                if analysis.recent_change:
                    change_alert = detect_regime_change(self._regime_history.get(symbol, []))
                    if change_alert:
                        change_alert.symbol = symbol
                        alerts.append(change_alert)
                        
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
                continue
        
        # Sort by significance
        significance_order = {'Major': 0, 'Moderate': 1, 'Minor': 2}
        alerts.sort(key=lambda x: significance_order.get(x.significance, 3))
        
        return alerts
    
    def get_market_overview(
        self,
        benchmark_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Get overall market regime overview.
        
        Args:
            benchmark_df: Benchmark (e.g., SPY) DataFrame
            
        Returns:
            Dictionary with market overview
        """
        analysis = self.analyze_regime("MARKET", benchmark_df)
        
        return {
            'regime': analysis.market_regime.value,
            'volatility': analysis.volatility_regime.value,
            'trend_strength': analysis.trend_strength,
            'probability': analysis.regime_probability,
            'strategies': analysis.recommended_strategies,
            'indicators': analysis.indicators
        }
    
    def clear_cache(self) -> None:
        """Clear the analysis cache."""
        self._cache.clear()
        logger.info("RegimeService cache cleared")
