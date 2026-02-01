"""
Signals Service
===============
Signal Generation, Filtering, and Backtesting

Provides:
- MA Crossover signals (Golden Cross/Death Cross)
- RSI Overbought/Oversold signals
- MACD Crossover signals
- Bollinger Band breakout signals
- Combined signal scoring
- Risk-based signal filtering
- Backtesting capabilities

Author: Stock Risk App | Feb 2026
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
import yfinance as yf

from .ta_service import (
    TAService, 
    calculate_sma, 
    calculate_ema, 
    calculate_rsi, 
    calculate_macd,
    calculate_bollinger_bands,
    RSI_OVERBOUGHT,
    RSI_OVERSOLD,
    get_all_indicators
)

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================

# Risk filter thresholds
MAX_VOLATILITY = 0.30  # 30%
MAX_BETA = 1.8
MIN_SHARPE = 0.3

# Signal scoring weights
WEIGHT_MA_CROSSOVER = 0.25
WEIGHT_RSI = 0.20
WEIGHT_MACD = 0.25
WEIGHT_BOLLINGER = 0.15
WEIGHT_ADX = 0.15

# Backtest defaults
DEFAULT_INITIAL_CAPITAL = 100000.0
DEFAULT_POSITION_SIZE = 0.1  # 10% of capital per trade
DEFAULT_STOP_LOSS = 0.02  # 2%
DEFAULT_TAKE_PROFIT = 0.05  # 5%


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class SignalType(Enum):
    """Types of trading signals."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    STRONG_BUY = "STRONG_BUY"
    STRONG_SELL = "STRONG_SELL"


class SignalReason(Enum):
    """Reasons for signal generation."""
    GOLDEN_CROSS = "Golden Cross (SMA 50 > SMA 21)"
    DEATH_CROSS = "Death Cross (SMA 50 < SMA 21)"
    RSI_OVERSOLD = "RSI Oversold (<30)"
    RSI_OVERBOUGHT = "RSI Overbought (>70)"
    MACD_BULLISH = "MACD Bullish Crossover"
    MACD_BEARISH = "MACD Bearish Crossover"
    BB_LOWER_TOUCH = "Price at Lower Bollinger Band"
    BB_UPPER_TOUCH = "Price at Upper Bollinger Band"
    COMBINED_BULLISH = "Multiple Bullish Indicators"
    COMBINED_BEARISH = "Multiple Bearish Indicators"


@dataclass
class Signal:
    """Represents a trading signal."""
    symbol: str
    signal_type: SignalType
    price: float
    reason: str
    score: float  # 0-100
    timestamp: datetime
    risk_score: float = 50.0
    indicators: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary."""
        return {
            'symbol': self.symbol,
            'signal_type': self.signal_type.value,
            'price': self.price,
            'reason': self.reason,
            'score': self.score,
            'timestamp': self.timestamp.isoformat(),
            'risk_score': self.risk_score,
            'indicators': self.indicators,
            'metadata': self.metadata
        }


@dataclass
class RiskMetrics:
    """Risk metrics for filtering signals."""
    volatility: float
    beta: float
    sharpe: float
    max_drawdown: float
    var_95: float
    
    def passes_filter(
        self,
        max_vol: float = MAX_VOLATILITY,
        max_beta: float = MAX_BETA,
        min_sharpe: float = MIN_SHARPE
    ) -> bool:
        """Check if metrics pass risk filter."""
        return (
            self.volatility < max_vol and
            self.beta < max_beta and
            self.sharpe > min_sharpe
        )


@dataclass
class BacktestResult:
    """Results from backtesting signals."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_return: float
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float
    equity_curve: pd.Series
    trades: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'total_return': self.total_return,
            'profit_factor': self.profit_factor,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio
        }


# ============================================================================
# SIGNAL GENERATION FUNCTIONS
# ============================================================================

def generate_ma_crossover_signals(
    prices: pd.Series,
    short_period: int = 21,
    long_period: int = 50
) -> List[Signal]:
    """
    Generate Moving Average crossover signals.
    
    Golden Cross: Short MA crosses above Long MA (Buy)
    Death Cross: Short MA crosses below Long MA (Sell)
    
    Args:
        prices: Close price series
        short_period: Short MA period (default: 21)
        long_period: Long MA period (default: 50)
        
    Returns:
        List of Signal objects
    """
    signals = []
    
    if len(prices) < long_period + 5:
        logger.warning("Insufficient data for MA crossover signals")
        return signals
    
    # Calculate SMAs
    short_ma = calculate_sma(prices, short_period)
    long_ma = calculate_sma(prices, long_period)
    
    # Find crossovers
    for i in range(1, len(prices)):
        if pd.isna(short_ma.iloc[i]) or pd.isna(long_ma.iloc[i]):
            continue
        if pd.isna(short_ma.iloc[i-1]) or pd.isna(long_ma.iloc[i-1]):
            continue
            
        prev_diff = short_ma.iloc[i-1] - long_ma.iloc[i-1]
        curr_diff = short_ma.iloc[i] - long_ma.iloc[i]
        
        # Golden Cross (bullish)
        if prev_diff <= 0 and curr_diff > 0:
            signals.append(Signal(
                symbol="",  # Will be set by caller
                signal_type=SignalType.BUY,
                price=float(prices.iloc[i]),
                reason=SignalReason.GOLDEN_CROSS.value,
                score=70.0,
                timestamp=prices.index[i] if hasattr(prices.index[i], 'to_pydatetime') 
                         else datetime.now(),
                indicators={
                    f'SMA_{short_period}': float(short_ma.iloc[i]),
                    f'SMA_{long_period}': float(long_ma.iloc[i])
                }
            ))
        
        # Death Cross (bearish)
        elif prev_diff >= 0 and curr_diff < 0:
            signals.append(Signal(
                symbol="",
                signal_type=SignalType.SELL,
                price=float(prices.iloc[i]),
                reason=SignalReason.DEATH_CROSS.value,
                score=70.0,
                timestamp=prices.index[i] if hasattr(prices.index[i], 'to_pydatetime')
                         else datetime.now(),
                indicators={
                    f'SMA_{short_period}': float(short_ma.iloc[i]),
                    f'SMA_{long_period}': float(long_ma.iloc[i])
                }
            ))
    
    return signals


def generate_rsi_signals(
    prices: pd.Series,
    period: int = 14,
    overbought: float = RSI_OVERBOUGHT,
    oversold: float = RSI_OVERSOLD
) -> List[Signal]:
    """
    Generate RSI overbought/oversold signals.
    
    Args:
        prices: Close price series
        period: RSI period (default: 14)
        overbought: Overbought threshold (default: 70)
        oversold: Oversold threshold (default: 30)
        
    Returns:
        List of Signal objects
    """
    signals = []
    
    if len(prices) < period + 5:
        logger.warning("Insufficient data for RSI signals")
        return signals
    
    rsi = calculate_rsi(prices, period)
    
    for i in range(1, len(prices)):
        if pd.isna(rsi.iloc[i]) or pd.isna(rsi.iloc[i-1]):
            continue
        
        prev_rsi = rsi.iloc[i-1]
        curr_rsi = rsi.iloc[i]
        
        # RSI crossing below oversold (potential buy when it recovers)
        if prev_rsi >= oversold and curr_rsi < oversold:
            signals.append(Signal(
                symbol="",
                signal_type=SignalType.BUY,
                price=float(prices.iloc[i]),
                reason=SignalReason.RSI_OVERSOLD.value,
                score=65.0,
                timestamp=prices.index[i] if hasattr(prices.index[i], 'to_pydatetime')
                         else datetime.now(),
                indicators={'RSI': float(curr_rsi)}
            ))
        
        # RSI crossing above overbought (potential sell when it reverses)
        elif prev_rsi <= overbought and curr_rsi > overbought:
            signals.append(Signal(
                symbol="",
                signal_type=SignalType.SELL,
                price=float(prices.iloc[i]),
                reason=SignalReason.RSI_OVERBOUGHT.value,
                score=65.0,
                timestamp=prices.index[i] if hasattr(prices.index[i], 'to_pydatetime')
                         else datetime.now(),
                indicators={'RSI': float(curr_rsi)}
            ))
    
    return signals


def generate_macd_signals(
    prices: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> List[Signal]:
    """
    Generate MACD crossover signals.
    
    Args:
        prices: Close price series
        fast_period: Fast EMA period (default: 12)
        slow_period: Slow EMA period (default: 26)
        signal_period: Signal line period (default: 9)
        
    Returns:
        List of Signal objects
    """
    signals = []
    
    min_data = slow_period + signal_period + 5
    if len(prices) < min_data:
        logger.warning("Insufficient data for MACD signals")
        return signals
    
    macd_line, signal_line, histogram = calculate_macd(
        prices, fast_period, slow_period, signal_period
    )
    
    for i in range(1, len(prices)):
        if pd.isna(macd_line.iloc[i]) or pd.isna(signal_line.iloc[i]):
            continue
        if pd.isna(macd_line.iloc[i-1]) or pd.isna(signal_line.iloc[i-1]):
            continue
        
        prev_diff = macd_line.iloc[i-1] - signal_line.iloc[i-1]
        curr_diff = macd_line.iloc[i] - signal_line.iloc[i]
        
        # Bullish crossover
        if prev_diff <= 0 and curr_diff > 0:
            signals.append(Signal(
                symbol="",
                signal_type=SignalType.BUY,
                price=float(prices.iloc[i]),
                reason=SignalReason.MACD_BULLISH.value,
                score=70.0,
                timestamp=prices.index[i] if hasattr(prices.index[i], 'to_pydatetime')
                         else datetime.now(),
                indicators={
                    'MACD': float(macd_line.iloc[i]),
                    'MACD_Signal': float(signal_line.iloc[i]),
                    'MACD_Histogram': float(histogram.iloc[i])
                }
            ))
        
        # Bearish crossover
        elif prev_diff >= 0 and curr_diff < 0:
            signals.append(Signal(
                symbol="",
                signal_type=SignalType.SELL,
                price=float(prices.iloc[i]),
                reason=SignalReason.MACD_BEARISH.value,
                score=70.0,
                timestamp=prices.index[i] if hasattr(prices.index[i], 'to_pydatetime')
                         else datetime.now(),
                indicators={
                    'MACD': float(macd_line.iloc[i]),
                    'MACD_Signal': float(signal_line.iloc[i]),
                    'MACD_Histogram': float(histogram.iloc[i])
                }
            ))
    
    return signals


def generate_bollinger_signals(
    prices: pd.Series,
    period: int = 20,
    num_std: float = 2.0
) -> List[Signal]:
    """
    Generate Bollinger Band breakout signals.
    
    Args:
        prices: Close price series
        period: BB period (default: 20)
        num_std: Number of standard deviations (default: 2)
        
    Returns:
        List of Signal objects
    """
    signals = []
    
    if len(prices) < period + 5:
        logger.warning("Insufficient data for Bollinger Band signals")
        return signals
    
    upper, middle, lower = calculate_bollinger_bands(prices, period, num_std)
    
    for i in range(1, len(prices)):
        if pd.isna(upper.iloc[i]) or pd.isna(lower.iloc[i]):
            continue
        
        price = prices.iloc[i]
        prev_price = prices.iloc[i-1]
        
        # Price touches or crosses lower band (potential buy)
        if prev_price > lower.iloc[i-1] and price <= lower.iloc[i]:
            signals.append(Signal(
                symbol="",
                signal_type=SignalType.BUY,
                price=float(price),
                reason=SignalReason.BB_LOWER_TOUCH.value,
                score=60.0,
                timestamp=prices.index[i] if hasattr(prices.index[i], 'to_pydatetime')
                         else datetime.now(),
                indicators={
                    'BB_Upper': float(upper.iloc[i]),
                    'BB_Middle': float(middle.iloc[i]),
                    'BB_Lower': float(lower.iloc[i])
                }
            ))
        
        # Price touches or crosses upper band (potential sell)
        elif prev_price < upper.iloc[i-1] and price >= upper.iloc[i]:
            signals.append(Signal(
                symbol="",
                signal_type=SignalType.SELL,
                price=float(price),
                reason=SignalReason.BB_UPPER_TOUCH.value,
                score=60.0,
                timestamp=prices.index[i] if hasattr(prices.index[i], 'to_pydatetime')
                         else datetime.now(),
                indicators={
                    'BB_Upper': float(upper.iloc[i]),
                    'BB_Middle': float(middle.iloc[i]),
                    'BB_Lower': float(lower.iloc[i])
                }
            ))
    
    return signals


def calculate_combined_signal_score(
    indicators: Dict[str, pd.Series],
    idx: int = -1
) -> Tuple[float, SignalType, str]:
    """
    Calculate a combined signal score (0-100) from multiple indicators.
    
    Args:
        indicators: Dictionary of indicator series
        idx: Index to evaluate (default: -1 for latest)
        
    Returns:
        Tuple of (score, signal_type, reason)
    """
    scores = []
    reasons = []
    
    # RSI component
    if 'RSI' in indicators:
        rsi = indicators['RSI'].iloc[idx]
        if not pd.isna(rsi):
            if rsi < RSI_OVERSOLD:
                scores.append((100, WEIGHT_RSI))  # Strong buy signal
                reasons.append("RSI oversold")
            elif rsi > RSI_OVERBOUGHT:
                scores.append((0, WEIGHT_RSI))  # Strong sell signal
                reasons.append("RSI overbought")
            else:
                # Linear scale: 30=100, 50=50, 70=0
                normalized = 100 - ((rsi - RSI_OVERSOLD) / (RSI_OVERBOUGHT - RSI_OVERSOLD) * 100)
                scores.append((normalized, WEIGHT_RSI))
    
    # MACD component
    if 'MACD' in indicators and 'MACD_Signal' in indicators:
        macd = indicators['MACD'].iloc[idx]
        signal = indicators['MACD_Signal'].iloc[idx]
        if not pd.isna(macd) and not pd.isna(signal):
            if macd > signal:
                # Bullish - score based on histogram strength
                histogram = indicators.get('MACD_Histogram', pd.Series([0])).iloc[idx]
                strength = min(abs(histogram) * 1000, 30)  # Cap contribution
                scores.append((60 + strength, WEIGHT_MACD))
                reasons.append("MACD bullish")
            else:
                histogram = indicators.get('MACD_Histogram', pd.Series([0])).iloc[idx]
                strength = min(abs(histogram) * 1000, 30)
                scores.append((40 - strength, WEIGHT_MACD))
                reasons.append("MACD bearish")
    
    # Moving Average trend
    if 'SMA_21' in indicators and 'SMA_50' in indicators:
        sma_short = indicators['SMA_21'].iloc[idx]
        sma_long = indicators['SMA_50'].iloc[idx]
        close = indicators.get('Close', pd.Series([0])).iloc[idx]
        
        if not any(pd.isna(x) for x in [sma_short, sma_long, close]):
            if close > sma_short > sma_long:
                scores.append((80, WEIGHT_MA_CROSSOVER))
                reasons.append("Price above MAs, uptrend")
            elif close < sma_short < sma_long:
                scores.append((20, WEIGHT_MA_CROSSOVER))
                reasons.append("Price below MAs, downtrend")
            elif sma_short > sma_long:
                scores.append((65, WEIGHT_MA_CROSSOVER))
                reasons.append("Golden cross pattern")
            else:
                scores.append((35, WEIGHT_MA_CROSSOVER))
                reasons.append("Death cross pattern")
    
    # Bollinger Bands component
    if all(k in indicators for k in ['BB_Upper', 'BB_Lower', 'Close']):
        close = indicators['Close'].iloc[idx]
        upper = indicators['BB_Upper'].iloc[idx]
        lower = indicators['BB_Lower'].iloc[idx]
        
        if not any(pd.isna(x) for x in [close, upper, lower]):
            bb_range = upper - lower
            if bb_range > 0:
                position = (close - lower) / bb_range  # 0 = lower band, 1 = upper band
                # Near lower band = buy opportunity, near upper = sell
                bb_score = 100 - (position * 100)
                scores.append((bb_score, WEIGHT_BOLLINGER))
                if position < 0.2:
                    reasons.append("Near lower Bollinger Band")
                elif position > 0.8:
                    reasons.append("Near upper Bollinger Band")
    
    # ADX component (trend strength)
    if 'ADX' in indicators:
        adx = indicators['ADX'].iloc[idx]
        plus_di = indicators.get('Plus_DI', pd.Series([0])).iloc[idx]
        minus_di = indicators.get('Minus_DI', pd.Series([0])).iloc[idx]
        
        if not any(pd.isna(x) for x in [adx, plus_di, minus_di]):
            if adx > 25:  # Strong trend
                if plus_di > minus_di:
                    scores.append((70, WEIGHT_ADX))
                    reasons.append("Strong uptrend (ADX)")
                else:
                    scores.append((30, WEIGHT_ADX))
                    reasons.append("Strong downtrend (ADX)")
            else:
                scores.append((50, WEIGHT_ADX))  # No strong trend
    
    # Calculate weighted average
    if not scores:
        return 50.0, SignalType.HOLD, "Insufficient data"
    
    total_weight = sum(w for _, w in scores)
    weighted_score = sum(s * w for s, w in scores) / total_weight if total_weight > 0 else 50
    
    # Determine signal type
    if weighted_score >= 75:
        signal_type = SignalType.STRONG_BUY
    elif weighted_score >= 60:
        signal_type = SignalType.BUY
    elif weighted_score <= 25:
        signal_type = SignalType.STRONG_SELL
    elif weighted_score <= 40:
        signal_type = SignalType.SELL
    else:
        signal_type = SignalType.HOLD
    
    reason = "; ".join(reasons) if reasons else "Mixed signals"
    
    return weighted_score, signal_type, reason


# ============================================================================
# RISK CALCULATIONS
# ============================================================================

def calculate_risk_metrics(
    prices: pd.Series,
    benchmark_prices: Optional[pd.Series] = None,
    rf_rate: float = 0.045
) -> RiskMetrics:
    """
    Calculate risk metrics for signal filtering.
    
    Args:
        prices: Close price series
        benchmark_prices: Benchmark price series (for beta calculation)
        rf_rate: Risk-free rate (default: 4.5%)
        
    Returns:
        RiskMetrics object
    """
    # Calculate returns
    returns = prices.pct_change().dropna()
    
    if len(returns) < 20:
        logger.warning("Insufficient data for risk metrics")
        return RiskMetrics(
            volatility=1.0,
            beta=1.0,
            sharpe=0.0,
            max_drawdown=0.0,
            var_95=0.0
        )
    
    # Annualized volatility
    volatility = returns.std() * np.sqrt(252)
    
    # Beta calculation
    beta = 1.0
    if benchmark_prices is not None and len(benchmark_prices) > 0:
        bench_returns = benchmark_prices.pct_change().dropna()
        
        # Align indices
        common_idx = returns.index.intersection(bench_returns.index)
        if len(common_idx) > 20:
            aligned_returns = returns.loc[common_idx]
            aligned_bench = bench_returns.loc[common_idx]
            
            cov = np.cov(aligned_returns, aligned_bench)[0, 1]
            var_bench = aligned_bench.var()
            if var_bench > 0:
                beta = cov / var_bench
    
    # Sharpe ratio
    excess_return = returns.mean() * 252 - rf_rate
    sharpe = excess_return / volatility if volatility > 0 else 0.0
    
    # Max drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdowns = (cumulative - running_max) / running_max
    max_drawdown = abs(drawdowns.min())
    
    # VaR 95%
    var_95 = np.percentile(returns, 5)
    
    return RiskMetrics(
        volatility=volatility,
        beta=beta,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        var_95=var_95
    )


def filter_signals_by_risk(
    signals: List[Signal],
    risk_metrics: RiskMetrics,
    max_volatility: float = MAX_VOLATILITY,
    max_beta: float = MAX_BETA,
    min_sharpe: float = MIN_SHARPE
) -> List[Signal]:
    """
    Filter signals based on risk criteria.
    
    Args:
        signals: List of signals to filter
        risk_metrics: Risk metrics for the asset
        max_volatility: Maximum allowed volatility
        max_beta: Maximum allowed beta
        min_sharpe: Minimum required Sharpe ratio
        
    Returns:
        Filtered list of signals
    """
    if not risk_metrics.passes_filter(max_volatility, max_beta, min_sharpe):
        logger.info(f"Asset filtered out due to risk criteria")
        return []
    
    # Calculate risk score for each signal (lower is better)
    filtered = []
    for signal in signals:
        # Risk score based on volatility, beta, and inverse Sharpe
        vol_score = min(risk_metrics.volatility / max_volatility * 50, 50)
        beta_score = min(risk_metrics.beta / max_beta * 30, 30)
        sharpe_score = max(0, 20 - (risk_metrics.sharpe / min_sharpe * 10))
        
        risk_score = vol_score + beta_score + sharpe_score
        
        # Update signal with risk score
        signal.risk_score = risk_score
        signal.metadata['volatility'] = risk_metrics.volatility
        signal.metadata['beta'] = risk_metrics.beta
        signal.metadata['sharpe'] = risk_metrics.sharpe
        
        filtered.append(signal)
    
    return filtered


# ============================================================================
# BACKTESTING
# ============================================================================

def backtest_signals(
    prices: pd.Series,
    signals: List[Signal],
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    position_size: float = DEFAULT_POSITION_SIZE,
    stop_loss: float = DEFAULT_STOP_LOSS,
    take_profit: float = DEFAULT_TAKE_PROFIT
) -> BacktestResult:
    """
    Backtest trading signals on historical data.
    
    Args:
        prices: Close price series
        signals: List of signals to backtest
        initial_capital: Starting capital
        position_size: Fraction of capital per trade
        stop_loss: Stop loss percentage
        take_profit: Take profit percentage
        
    Returns:
        BacktestResult object
    """
    if not signals or len(prices) < 10:
        return BacktestResult(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            total_return=0.0,
            profit_factor=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            equity_curve=pd.Series([initial_capital], index=[prices.index[0]]),
            trades=[]
        )
    
    # Sort signals by timestamp
    sorted_signals = sorted(signals, key=lambda s: s.timestamp)
    
    capital = initial_capital
    position = 0.0
    entry_price = 0.0
    trades = []
    equity_values = [initial_capital]
    equity_dates = [prices.index[0]]
    
    total_gains = 0.0
    total_losses = 0.0
    
    for i, signal in enumerate(sorted_signals):
        try:
            # Find the price at signal time
            signal_date = pd.Timestamp(signal.timestamp)
            
            # Find nearest date in prices
            date_diffs = abs(prices.index - signal_date)
            nearest_idx = date_diffs.argmin()
            current_price = prices.iloc[nearest_idx]
            
            if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                if position == 0:  # Not in a position
                    # Enter long position
                    trade_amount = capital * position_size
                    shares = trade_amount / current_price
                    position = shares
                    entry_price = current_price
                    
                    trades.append({
                        'type': 'BUY',
                        'date': signal_date,
                        'price': current_price,
                        'shares': shares,
                        'value': trade_amount,
                        'reason': signal.reason
                    })
            
            elif signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
                if position > 0:  # In a position
                    # Exit position
                    exit_value = position * current_price
                    entry_value = position * entry_price
                    pnl = exit_value - entry_value
                    
                    capital += pnl
                    
                    if pnl > 0:
                        total_gains += pnl
                    else:
                        total_losses += abs(pnl)
                    
                    trades.append({
                        'type': 'SELL',
                        'date': signal_date,
                        'price': current_price,
                        'shares': position,
                        'value': exit_value,
                        'pnl': pnl,
                        'pnl_pct': (pnl / entry_value) * 100,
                        'reason': signal.reason
                    })
                    
                    position = 0.0
                    entry_price = 0.0
            
            # Record equity
            current_equity = capital + (position * current_price if position > 0 else 0)
            equity_values.append(current_equity)
            equity_dates.append(signal_date)
            
        except Exception as e:
            logger.error(f"Error processing signal: {e}")
            continue
    
    # Close any remaining position at end
    if position > 0:
        final_price = prices.iloc[-1]
        final_value = position * final_price
        entry_value = position * entry_price
        pnl = final_value - entry_value
        capital += pnl
        
        if pnl > 0:
            total_gains += pnl
        else:
            total_losses += abs(pnl)
        
        trades.append({
            'type': 'CLOSE',
            'date': prices.index[-1],
            'price': final_price,
            'shares': position,
            'value': final_value,
            'pnl': pnl,
            'pnl_pct': (pnl / entry_value) * 100 if entry_value > 0 else 0,
            'reason': 'End of backtest'
        })
    
    # Calculate metrics
    equity_curve = pd.Series(equity_values, index=equity_dates)
    
    # Count winning/losing trades
    sell_trades = [t for t in trades if t['type'] in ['SELL', 'CLOSE'] and 'pnl' in t]
    winning = len([t for t in sell_trades if t.get('pnl', 0) > 0])
    losing = len([t for t in sell_trades if t.get('pnl', 0) <= 0])
    total_closed = winning + losing
    
    win_rate = (winning / total_closed * 100) if total_closed > 0 else 0.0
    total_return = ((capital - initial_capital) / initial_capital) * 100
    profit_factor = (total_gains / total_losses) if total_losses > 0 else float('inf')
    
    # Max drawdown
    running_max = equity_curve.expanding().max()
    drawdowns = (equity_curve - running_max) / running_max
    max_drawdown = abs(drawdowns.min()) * 100 if len(drawdowns) > 0 else 0.0
    
    # Sharpe ratio (simplified)
    returns = equity_curve.pct_change().dropna()
    if len(returns) > 1 and returns.std() > 0:
        sharpe = (returns.mean() * np.sqrt(252)) / returns.std()
    else:
        sharpe = 0.0
    
    return BacktestResult(
        total_trades=len(trades),
        winning_trades=winning,
        losing_trades=losing,
        win_rate=win_rate,
        total_return=total_return,
        profit_factor=profit_factor if profit_factor != float('inf') else 999.99,
        max_drawdown=max_drawdown,
        sharpe_ratio=sharpe,
        equity_curve=equity_curve,
        trades=trades
    )


# ============================================================================
# SIGNALS SERVICE CLASS
# ============================================================================

class SignalsService:
    """
    Signal Generation and Management Service.
    
    Provides comprehensive signal generation, filtering, and analysis.
    """
    
    def __init__(
        self,
        max_volatility: float = MAX_VOLATILITY,
        max_beta: float = MAX_BETA,
        min_sharpe: float = MIN_SHARPE
    ):
        """
        Initialize the Signals Service.
        
        Args:
            max_volatility: Maximum volatility for risk filter
            max_beta: Maximum beta for risk filter
            min_sharpe: Minimum Sharpe for risk filter
        """
        self.max_volatility = max_volatility
        self.max_beta = max_beta
        self.min_sharpe = min_sharpe
        self.ta_service = TAService()
        self._signals_cache: Dict[str, List[Signal]] = {}
        
        logger.info("SignalsService initialized")
    
    def generate_all_signals(
        self,
        symbol: str,
        df: pd.DataFrame,
        benchmark_df: Optional[pd.DataFrame] = None
    ) -> List[Signal]:
        """
        Generate all types of signals for a symbol.
        
        Args:
            symbol: Stock symbol
            df: OHLCV DataFrame
            benchmark_df: Benchmark DataFrame for risk calculations
            
        Returns:
            List of all generated signals
        """
        all_signals = []
        
        # Handle DataFrame structure
        if isinstance(df.columns, pd.MultiIndex):
            close = df['Close'].iloc[:, 0] if isinstance(df['Close'], pd.DataFrame) else df['Close']
        else:
            close = df['Close']
        
        try:
            # Generate signals from each source
            ma_signals = generate_ma_crossover_signals(close)
            rsi_signals = generate_rsi_signals(close)
            macd_signals = generate_macd_signals(close)
            bb_signals = generate_bollinger_signals(close)
            
            # Set symbol for all signals
            for sig_list in [ma_signals, rsi_signals, macd_signals, bb_signals]:
                for sig in sig_list:
                    sig.symbol = symbol
                all_signals.extend(sig_list)
            
            # Calculate risk metrics if benchmark provided
            if benchmark_df is not None:
                if isinstance(benchmark_df.columns, pd.MultiIndex):
                    bench_close = benchmark_df['Close'].iloc[:, 0]
                else:
                    bench_close = benchmark_df['Close']
                
                risk_metrics = calculate_risk_metrics(close, bench_close)
                all_signals = filter_signals_by_risk(
                    all_signals, 
                    risk_metrics,
                    self.max_volatility,
                    self.max_beta,
                    self.min_sharpe
                )
            
            logger.info(f"Generated {len(all_signals)} signals for {symbol}")
            return all_signals
            
        except Exception as e:
            logger.error(f"Error generating signals for {symbol}: {e}")
            return []
    
    def get_current_signal(
        self,
        symbol: str,
        df: pd.DataFrame
    ) -> Tuple[float, SignalType, str]:
        """
        Get the current combined signal for a symbol.
        
        Args:
            symbol: Stock symbol
            df: OHLCV DataFrame
            
        Returns:
            Tuple of (score, signal_type, reason)
        """
        indicators = self.ta_service.calculate_all(df)
        return calculate_combined_signal_score(indicators)
    
    def scan_symbols(
        self,
        symbols: List[str],
        period: str = "1y",
        signal_types: Optional[List[SignalType]] = None
    ) -> List[Signal]:
        """
        Scan multiple symbols for signals.
        
        Args:
            symbols: List of stock symbols
            period: Data period (e.g., "1y")
            signal_types: Filter by specific signal types
            
        Returns:
            List of signals found across all symbols
        """
        all_signals = []
        
        for symbol in symbols:
            try:
                # Fetch data
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period)
                
                if df.empty or len(df) < 50:
                    continue
                
                # Get benchmark for risk filtering
                spy = yf.Ticker("SPY")
                bench_df = spy.history(period=period)
                
                signals = self.generate_all_signals(symbol, df, bench_df)
                
                # Filter by signal type if specified
                if signal_types:
                    signals = [s for s in signals if s.signal_type in signal_types]
                
                all_signals.extend(signals)
                
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
                continue
        
        # Sort by timestamp (most recent first)
        all_signals.sort(key=lambda s: s.timestamp, reverse=True)
        
        return all_signals
    
    def backtest_symbol(
        self,
        symbol: str,
        df: pd.DataFrame,
        initial_capital: float = DEFAULT_INITIAL_CAPITAL,
        position_size: float = DEFAULT_POSITION_SIZE
    ) -> BacktestResult:
        """
        Backtest signals for a symbol.
        
        Args:
            symbol: Stock symbol
            df: OHLCV DataFrame
            initial_capital: Starting capital
            position_size: Position size as fraction
            
        Returns:
            BacktestResult object
        """
        # Generate signals
        signals = self.generate_all_signals(symbol, df)
        
        # Handle DataFrame structure
        if isinstance(df.columns, pd.MultiIndex):
            close = df['Close'].iloc[:, 0]
        else:
            close = df['Close']
        
        # Run backtest
        return backtest_signals(
            close,
            signals,
            initial_capital=initial_capital,
            position_size=position_size
        )
    
    def get_portfolio_signals(
        self,
        portfolio: Dict[str, float],
        period: str = "1y"
    ) -> Dict[str, Any]:
        """
        Get aggregated signals for a portfolio.
        
        Args:
            portfolio: Dictionary of {symbol: weight}
            period: Data period
            
        Returns:
            Dictionary with portfolio-level signal analysis
        """
        portfolio_signals = {
            'symbols': {},
            'aggregate_score': 0.0,
            'aggregate_signal': SignalType.HOLD,
            'weighted_signals': []
        }
        
        total_weight = sum(portfolio.values())
        weighted_scores = []
        
        for symbol, weight in portfolio.items():
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period)
                
                if df.empty or len(df) < 50:
                    continue
                
                # Get current signal
                score, signal_type, reason = self.get_current_signal(symbol, df)
                normalized_weight = weight / total_weight
                
                portfolio_signals['symbols'][symbol] = {
                    'score': score,
                    'signal': signal_type.value,
                    'reason': reason,
                    'weight': normalized_weight
                }
                
                weighted_scores.append(score * normalized_weight)
                
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                continue
        
        # Calculate aggregate score
        if weighted_scores:
            aggregate = sum(weighted_scores)
            portfolio_signals['aggregate_score'] = aggregate
            
            if aggregate >= 70:
                portfolio_signals['aggregate_signal'] = SignalType.STRONG_BUY
            elif aggregate >= 55:
                portfolio_signals['aggregate_signal'] = SignalType.BUY
            elif aggregate <= 30:
                portfolio_signals['aggregate_signal'] = SignalType.STRONG_SELL
            elif aggregate <= 45:
                portfolio_signals['aggregate_signal'] = SignalType.SELL
            else:
                portfolio_signals['aggregate_signal'] = SignalType.HOLD
        
        return portfolio_signals
    
    def clear_cache(self) -> None:
        """Clear the signals cache."""
        self._signals_cache.clear()
        self.ta_service.clear_cache()
        logger.info("SignalsService cache cleared")
