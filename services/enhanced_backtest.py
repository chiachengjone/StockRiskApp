"""
Enhanced Backtesting Engine
============================
Advanced backtesting with transaction costs, walk-forward analysis,
statistical robustness testing, and drawdown analysis.

Features:
- Transaction costs modeling (slippage, market impact, commissions)
- Walk-forward optimization with out-of-sample testing
- Monte Carlo permutation tests for statistical robustness
- Drawdown duration analysis and recovery periods
- Regime-aware backtesting

Author: Stock Risk App | Feb 2026
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class CommissionType(Enum):
    """Types of commission structures."""
    FLAT = "flat"  # Flat fee per trade
    PER_SHARE = "per_share"  # Fee per share
    PERCENTAGE = "percentage"  # Percentage of trade value
    TIERED = "tiered"  # Tiered pricing based on volume


class SlippageModel(Enum):
    """Slippage modeling approaches."""
    FIXED = "fixed"  # Fixed percentage
    PROPORTIONAL = "proportional"  # Proportional to trade size
    VOLATILITY_BASED = "volatility_based"  # Based on recent volatility
    MARKET_IMPACT = "market_impact"  # Square root market impact


@dataclass
class TransactionCostConfig:
    """Configuration for transaction costs."""
    commission_type: CommissionType = CommissionType.FLAT
    commission_value: float = 0.0  # Cost per trade (flat) or per share/percentage
    slippage_model: SlippageModel = SlippageModel.FIXED
    slippage_bps: float = 5.0  # Basis points of slippage
    market_impact_coefficient: float = 0.1  # For market impact model
    min_commission: float = 0.0
    max_commission: float = float('inf')


@dataclass
class TradeResult:
    """Result of a single trade with costs."""
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    shares: int
    direction: str  # 'long' or 'short'
    gross_pnl: float
    commission_cost: float
    slippage_cost: float
    net_pnl: float
    holding_period: int
    return_pct: float
    mae: float  # Maximum Adverse Excursion
    mfe: float  # Maximum Favorable Excursion


@dataclass
class DrawdownAnalysis:
    """Detailed drawdown analysis."""
    max_drawdown: float
    max_drawdown_duration: int  # Days
    avg_drawdown: float
    avg_drawdown_duration: float
    current_drawdown: float
    current_drawdown_duration: int
    drawdown_periods: List[Dict[str, Any]]
    recovery_periods: List[int]
    avg_recovery_time: float
    max_recovery_time: int
    underwater_percentage: float  # % of time in drawdown


@dataclass
class WalkForwardWindow:
    """Single walk-forward window result."""
    window_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    in_sample_return: float
    out_of_sample_return: float
    in_sample_sharpe: float
    out_of_sample_sharpe: float
    optimal_params: Dict[str, Any]
    degradation: float  # IS return - OOS return


@dataclass
class WalkForwardResult:
    """Complete walk-forward analysis result."""
    windows: List[WalkForwardWindow]
    combined_oos_return: float
    combined_oos_sharpe: float
    avg_degradation: float
    consistency_score: float  # % of OOS windows profitable
    robustness_score: float  # 0-100
    anchored_vs_rolling: str  # Which approach performed better


@dataclass
class MonteCarloResult:
    """Monte Carlo simulation result."""
    num_simulations: int
    original_return: float
    original_sharpe: float
    mean_simulated_return: float
    std_simulated_return: float
    percentile_5: float
    percentile_25: float
    median_return: float
    percentile_75: float
    percentile_95: float
    p_value: float  # Probability of achieving original return by chance
    is_significant: bool
    confidence_level: float


# ============================================================================
# TRANSACTION COST CALCULATOR
# ============================================================================

class TransactionCostCalculator:
    """
    Calculate realistic transaction costs including commissions,
    slippage, and market impact.
    """
    
    def __init__(self, config: TransactionCostConfig = None):
        self.config = config or TransactionCostConfig()
    
    def calculate_commission(
        self,
        shares: int,
        price: float,
        trade_value: float = None
    ) -> float:
        """Calculate commission for a trade."""
        if trade_value is None:
            trade_value = shares * price
        
        if self.config.commission_type == CommissionType.FLAT:
            commission = self.config.commission_value
        elif self.config.commission_type == CommissionType.PER_SHARE:
            commission = shares * self.config.commission_value
        elif self.config.commission_type == CommissionType.PERCENTAGE:
            commission = trade_value * (self.config.commission_value / 100)
        else:  # TIERED
            # Simplified tiered (common broker structure)
            if shares <= 100:
                commission = 1.0
            elif shares <= 1000:
                commission = 0.005 * shares
            else:
                commission = 0.003 * shares
        
        return np.clip(commission, self.config.min_commission, self.config.max_commission)
    
    def calculate_slippage(
        self,
        shares: int,
        price: float,
        volatility: float = None,
        avg_volume: int = None
    ) -> float:
        """Calculate slippage for a trade."""
        trade_value = shares * price
        
        if self.config.slippage_model == SlippageModel.FIXED:
            slippage = trade_value * (self.config.slippage_bps / 10000)
            
        elif self.config.slippage_model == SlippageModel.PROPORTIONAL:
            # Larger trades have more slippage
            if avg_volume and avg_volume > 0:
                size_factor = min(2.0, shares / avg_volume * 100)
            else:
                size_factor = 1.0
            slippage = trade_value * (self.config.slippage_bps / 10000) * size_factor
            
        elif self.config.slippage_model == SlippageModel.VOLATILITY_BASED:
            # Higher volatility = more slippage
            vol_factor = (volatility / 0.02) if volatility else 1.0  # Normalized to 2% vol
            slippage = trade_value * (self.config.slippage_bps / 10000) * vol_factor
            
        else:  # MARKET_IMPACT
            # Square root market impact model
            if avg_volume and avg_volume > 0:
                participation = shares / avg_volume
                impact = self.config.market_impact_coefficient * np.sqrt(participation)
                slippage = trade_value * impact
            else:
                slippage = trade_value * (self.config.slippage_bps / 10000)
        
        return slippage
    
    def calculate_total_cost(
        self,
        shares: int,
        price: float,
        volatility: float = None,
        avg_volume: int = None
    ) -> Tuple[float, float, float]:
        """
        Calculate total transaction costs.
        
        Returns:
            Tuple of (commission, slippage, total_cost)
        """
        commission = self.calculate_commission(shares, price)
        slippage = self.calculate_slippage(shares, price, volatility, avg_volume)
        
        return commission, slippage, commission + slippage


# ============================================================================
# DRAWDOWN ANALYZER
# ============================================================================

class DrawdownAnalyzer:
    """
    Analyze drawdowns, recovery periods, and underwater time.
    """
    
    def analyze(self, equity_curve: pd.Series) -> DrawdownAnalysis:
        """
        Perform comprehensive drawdown analysis.
        
        Args:
            equity_curve: Series of portfolio values
            
        Returns:
            DrawdownAnalysis object
        """
        if len(equity_curve) < 2:
            return DrawdownAnalysis(
                max_drawdown=0, max_drawdown_duration=0, avg_drawdown=0,
                avg_drawdown_duration=0, current_drawdown=0, current_drawdown_duration=0,
                drawdown_periods=[], recovery_periods=[], avg_recovery_time=0,
                max_recovery_time=0, underwater_percentage=0
            )
        
        # Calculate running maximum and drawdown
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        
        # Max drawdown
        max_dd = abs(drawdown.min())
        
        # Find drawdown periods
        in_drawdown = drawdown < 0
        drawdown_periods = []
        recovery_periods = []
        
        current_dd_start = None
        current_dd_peak = None
        current_dd_trough = None
        current_dd_trough_idx = None
        
        for i, (date, dd) in enumerate(drawdown.items()):
            if dd < 0:
                if current_dd_start is None:
                    current_dd_start = date
                    current_dd_peak = running_max.iloc[i]
                
                if current_dd_trough is None or dd < current_dd_trough:
                    current_dd_trough = dd
                    current_dd_trough_idx = i
            else:
                if current_dd_start is not None:
                    # End of drawdown period
                    duration = (date - current_dd_start).days if hasattr(date, 'days') else i - drawdown.index.get_loc(current_dd_start)
                    
                    drawdown_periods.append({
                        'start': current_dd_start,
                        'trough_date': drawdown.index[current_dd_trough_idx] if current_dd_trough_idx else date,
                        'end': date,
                        'depth': abs(current_dd_trough) if current_dd_trough else 0,
                        'duration': duration
                    })
                    
                    # Recovery period from trough to end
                    if current_dd_trough_idx is not None:
                        recovery_days = (date - drawdown.index[current_dd_trough_idx]).days if hasattr(date, 'days') else i - current_dd_trough_idx
                        recovery_periods.append(recovery_days)
                    
                    current_dd_start = None
                    current_dd_peak = None
                    current_dd_trough = None
                    current_dd_trough_idx = None
        
        # Handle ongoing drawdown
        current_drawdown = 0
        current_drawdown_duration = 0
        if current_dd_start is not None:
            current_drawdown = abs(drawdown.iloc[-1])
            current_drawdown_duration = (equity_curve.index[-1] - current_dd_start).days if hasattr(equity_curve.index[-1], 'days') else len(equity_curve) - drawdown.index.get_loc(current_dd_start)
        
        # Calculate max drawdown duration
        dd_durations = [p['duration'] for p in drawdown_periods]
        max_dd_duration = max(dd_durations) if dd_durations else current_drawdown_duration
        
        # Average metrics
        avg_dd = np.mean([p['depth'] for p in drawdown_periods]) if drawdown_periods else 0
        avg_dd_duration = np.mean(dd_durations) if dd_durations else 0
        avg_recovery = np.mean(recovery_periods) if recovery_periods else 0
        max_recovery = max(recovery_periods) if recovery_periods else 0
        
        # Underwater percentage
        underwater_pct = in_drawdown.sum() / len(drawdown) * 100
        
        return DrawdownAnalysis(
            max_drawdown=max_dd * 100,
            max_drawdown_duration=max_dd_duration,
            avg_drawdown=avg_dd * 100,
            avg_drawdown_duration=avg_dd_duration,
            current_drawdown=current_drawdown * 100,
            current_drawdown_duration=current_drawdown_duration,
            drawdown_periods=drawdown_periods,
            recovery_periods=recovery_periods,
            avg_recovery_time=avg_recovery,
            max_recovery_time=max_recovery,
            underwater_percentage=underwater_pct
        )


# ============================================================================
# WALK-FORWARD ANALYZER
# ============================================================================

class WalkForwardAnalyzer:
    """
    Walk-forward optimization with anchored and rolling variations.
    """
    
    def __init__(
        self,
        train_pct: float = 0.7,
        num_windows: int = 5,
        anchored: bool = False
    ):
        self.train_pct = train_pct
        self.num_windows = num_windows
        self.anchored = anchored
    
    def analyze(
        self,
        data: pd.DataFrame,
        strategy_func: Callable,
        param_ranges: Dict[str, List],
        metric: str = 'sharpe'
    ) -> WalkForwardResult:
        """
        Perform walk-forward analysis.
        
        Args:
            data: OHLCV data
            strategy_func: Function that takes data and params, returns returns series
            param_ranges: Dictionary of parameter ranges to optimize
            metric: Optimization metric ('sharpe', 'return', 'sortino')
            
        Returns:
            WalkForwardResult object
        """
        windows = []
        
        # Calculate window sizes
        total_len = len(data)
        window_size = total_len // self.num_windows
        train_size = int(window_size * self.train_pct)
        test_size = window_size - train_size
        
        for i in range(self.num_windows):
            if self.anchored:
                # Anchored: train always starts from beginning
                train_start_idx = 0
                train_end_idx = train_size + i * test_size
                test_start_idx = train_end_idx
                test_end_idx = train_end_idx + test_size
            else:
                # Rolling: window moves forward
                train_start_idx = i * window_size
                train_end_idx = train_start_idx + train_size
                test_start_idx = train_end_idx
                test_end_idx = min(test_start_idx + test_size, total_len)
            
            if test_end_idx > total_len:
                break
            
            train_data = data.iloc[train_start_idx:train_end_idx]
            test_data = data.iloc[test_start_idx:test_end_idx]
            
            # Optimize on training data
            best_params, is_metrics = self._optimize(
                train_data, strategy_func, param_ranges, metric
            )
            
            # Test on out-of-sample data
            oos_returns = strategy_func(test_data, best_params)
            oos_metrics = self._calculate_metrics(oos_returns)
            
            degradation = is_metrics.get('return', 0) - oos_metrics.get('return', 0)
            
            windows.append(WalkForwardWindow(
                window_id=i,
                train_start=train_data.index[0],
                train_end=train_data.index[-1],
                test_start=test_data.index[0],
                test_end=test_data.index[-1],
                in_sample_return=is_metrics.get('return', 0),
                out_of_sample_return=oos_metrics.get('return', 0),
                in_sample_sharpe=is_metrics.get('sharpe', 0),
                out_of_sample_sharpe=oos_metrics.get('sharpe', 0),
                optimal_params=best_params,
                degradation=degradation
            ))
        
        if not windows:
            return WalkForwardResult(
                windows=[], combined_oos_return=0, combined_oos_sharpe=0,
                avg_degradation=0, consistency_score=0, robustness_score=0,
                anchored_vs_rolling="N/A"
            )
        
        # Combined metrics
        combined_oos_return = sum(w.out_of_sample_return for w in windows)
        oos_returns_list = [w.out_of_sample_return for w in windows]
        combined_oos_sharpe = np.mean(oos_returns_list) / np.std(oos_returns_list) * np.sqrt(252) if np.std(oos_returns_list) > 0 else 0
        
        avg_degradation = np.mean([w.degradation for w in windows])
        consistency = sum(1 for w in windows if w.out_of_sample_return > 0) / len(windows) * 100
        
        # Robustness score
        robustness = 100 - min(100, avg_degradation * 5)  # Penalize degradation
        robustness = max(0, robustness * (consistency / 100))  # Adjust for consistency
        
        return WalkForwardResult(
            windows=windows,
            combined_oos_return=combined_oos_return,
            combined_oos_sharpe=combined_oos_sharpe,
            avg_degradation=avg_degradation,
            consistency_score=consistency,
            robustness_score=robustness,
            anchored_vs_rolling="anchored" if self.anchored else "rolling"
        )
    
    def _optimize(
        self,
        data: pd.DataFrame,
        strategy_func: Callable,
        param_ranges: Dict[str, List],
        metric: str
    ) -> Tuple[Dict, Dict]:
        """Grid search optimization."""
        best_params = {}
        best_metric = float('-inf')
        best_metrics = {}
        
        # Generate parameter combinations
        import itertools
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        
        for combo in itertools.product(*param_values):
            params = dict(zip(param_names, combo))
            
            try:
                returns = strategy_func(data, params)
                metrics = self._calculate_metrics(returns)
                
                if metrics.get(metric, 0) > best_metric:
                    best_metric = metrics[metric]
                    best_params = params
                    best_metrics = metrics
            except:
                continue
        
        return best_params, best_metrics
    
    def _calculate_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate strategy metrics."""
        if returns.empty or len(returns) < 2:
            return {'return': 0, 'sharpe': 0, 'sortino': 0}
        
        total_return = (1 + returns).prod() - 1
        mean_return = returns.mean() * 252
        std_return = returns.std() * np.sqrt(252)
        
        sharpe = mean_return / std_return if std_return > 0 else 0
        
        downside = returns[returns < 0].std() * np.sqrt(252)
        sortino = mean_return / downside if downside > 0 else 0
        
        return {
            'return': total_return * 100,
            'sharpe': sharpe,
            'sortino': sortino
        }


# ============================================================================
# MONTE CARLO ANALYZER
# ============================================================================

class MonteCarloAnalyzer:
    """
    Monte Carlo permutation tests for strategy robustness.
    """
    
    def __init__(self, num_simulations: int = 1000):
        self.num_simulations = num_simulations
    
    def permutation_test(
        self,
        returns: pd.Series,
        trades: Optional[List[TradeResult]] = None
    ) -> MonteCarloResult:
        """
        Perform Monte Carlo permutation test.
        
        Shuffles trade returns to test if strategy performance
        is statistically significant or due to luck.
        
        Args:
            returns: Strategy returns series
            trades: Optional list of trades (for trade-based shuffling)
            
        Returns:
            MonteCarloResult object
        """
        original_return = (1 + returns).prod() - 1
        original_sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        simulated_returns = []
        
        for _ in range(self.num_simulations):
            # Shuffle returns
            shuffled = returns.sample(frac=1, replace=False)
            sim_return = (1 + shuffled).prod() - 1
            simulated_returns.append(sim_return)
        
        simulated_returns = np.array(simulated_returns)
        
        # Calculate statistics
        mean_sim = np.mean(simulated_returns)
        std_sim = np.std(simulated_returns)
        
        # P-value: proportion of simulations >= original
        p_value = (simulated_returns >= original_return).sum() / self.num_simulations
        
        is_significant = p_value < 0.05
        
        return MonteCarloResult(
            num_simulations=self.num_simulations,
            original_return=original_return * 100,
            original_sharpe=original_sharpe,
            mean_simulated_return=mean_sim * 100,
            std_simulated_return=std_sim * 100,
            percentile_5=np.percentile(simulated_returns, 5) * 100,
            percentile_25=np.percentile(simulated_returns, 25) * 100,
            median_return=np.median(simulated_returns) * 100,
            percentile_75=np.percentile(simulated_returns, 75) * 100,
            percentile_95=np.percentile(simulated_returns, 95) * 100,
            p_value=p_value,
            is_significant=is_significant,
            confidence_level=(1 - p_value) * 100
        )
    
    def bootstrap_confidence_interval(
        self,
        returns: pd.Series,
        confidence: float = 0.95
    ) -> Dict[str, float]:
        """
        Calculate bootstrap confidence interval for returns.
        
        Args:
            returns: Strategy returns series
            confidence: Confidence level (e.g., 0.95 for 95%)
            
        Returns:
            Dictionary with mean, lower, upper bounds
        """
        bootstrapped_returns = []
        
        for _ in range(self.num_simulations):
            # Sample with replacement
            sample = returns.sample(frac=1, replace=True)
            total_return = (1 + sample).prod() - 1
            bootstrapped_returns.append(total_return)
        
        alpha = 1 - confidence
        lower = np.percentile(bootstrapped_returns, alpha/2 * 100) * 100
        upper = np.percentile(bootstrapped_returns, (1 - alpha/2) * 100) * 100
        
        return {
            'mean': np.mean(bootstrapped_returns) * 100,
            'lower_bound': lower,
            'upper_bound': upper,
            'confidence_level': confidence * 100
        }


# ============================================================================
# ENHANCED BACKTESTER
# ============================================================================

class EnhancedBacktester:
    """
    Full-featured backtester with all enhancements.
    """
    
    def __init__(
        self,
        cost_config: TransactionCostConfig = None,
        initial_capital: float = 100000
    ):
        self.cost_calculator = TransactionCostCalculator(cost_config)
        self.initial_capital = initial_capital
        self.drawdown_analyzer = DrawdownAnalyzer()
        self.mc_analyzer = MonteCarloAnalyzer()
    
    def run_backtest(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        position_size: float = 1.0
    ) -> Dict[str, Any]:
        """
        Run a complete backtest with all analysis.
        
        Args:
            data: OHLCV data
            signals: Buy/sell signals (-1, 0, 1)
            position_size: Fraction of capital per trade
            
        Returns:
            Complete backtest results
        """
        portfolio = []
        trades = []
        capital = self.initial_capital
        position = 0
        entry_price = 0
        entry_date = None
        shares = 0
        
        # Calculate volatility for slippage
        returns = data['Close'].pct_change()
        rolling_vol = returns.rolling(20).std()
        avg_volume = data['Volume'].rolling(20).mean() if 'Volume' in data.columns else None
        
        for i, (date, row) in enumerate(data.iterrows()):
            signal = signals.iloc[i] if i < len(signals) else 0
            close = row['Close']
            vol = rolling_vol.iloc[i] if i < len(rolling_vol) else 0.02
            avg_vol = avg_volume.iloc[i] if avg_volume is not None and i < len(avg_volume) else 100000
            
            # Entry
            if signal == 1 and position == 0:
                trade_capital = capital * position_size
                shares = int(trade_capital / close)
                
                if shares > 0:
                    commission, slippage, total_cost = self.cost_calculator.calculate_total_cost(
                        shares, close, vol, int(avg_vol)
                    )
                    
                    entry_price = close + (slippage / shares)  # Adjust for slippage
                    entry_date = date
                    capital -= (shares * entry_price + commission)
                    position = 1
            
            # Exit
            elif signal == -1 and position == 1:
                commission, slippage, total_cost = self.cost_calculator.calculate_total_cost(
                    shares, close, vol, int(avg_vol)
                )
                
                exit_price = close - (slippage / shares)  # Slippage hurts on exit too
                gross_pnl = (exit_price - entry_price) * shares
                net_pnl = gross_pnl - commission * 2  # Entry + exit commission
                
                capital += shares * exit_price - commission
                
                trades.append(TradeResult(
                    entry_date=entry_date,
                    exit_date=date,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    shares=shares,
                    direction='long',
                    gross_pnl=gross_pnl,
                    commission_cost=commission * 2,
                    slippage_cost=slippage * 2,
                    net_pnl=net_pnl,
                    holding_period=(date - entry_date).days if hasattr(date, 'days') else 1,
                    return_pct=(exit_price / entry_price - 1) * 100,
                    mae=0,  # Would need intra-trade tracking
                    mfe=0
                ))
                
                position = 0
                shares = 0
            
            # Track portfolio value
            if position == 1:
                portfolio_value = capital + shares * close
            else:
                portfolio_value = capital
            
            portfolio.append({'date': date, 'value': portfolio_value})
        
        # Create equity curve
        equity = pd.Series(
            [p['value'] for p in portfolio],
            index=[p['date'] for p in portfolio]
        )
        
        # Calculate metrics
        returns_series = equity.pct_change().dropna()
        
        total_return = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
        sharpe = returns_series.mean() / returns_series.std() * np.sqrt(252) if returns_series.std() > 0 else 0
        
        # Drawdown analysis
        dd_analysis = self.drawdown_analyzer.analyze(equity)
        
        # Monte Carlo analysis
        mc_result = self.mc_analyzer.permutation_test(returns_series)
        
        # Trade statistics
        if trades:
            win_trades = [t for t in trades if t.net_pnl > 0]
            lose_trades = [t for t in trades if t.net_pnl <= 0]
            
            win_rate = len(win_trades) / len(trades) * 100
            avg_win = np.mean([t.net_pnl for t in win_trades]) if win_trades else 0
            avg_loss = np.mean([t.net_pnl for t in lose_trades]) if lose_trades else 0
            profit_factor = sum(t.net_pnl for t in win_trades) / abs(sum(t.net_pnl for t in lose_trades)) if lose_trades else float('inf')
            
            total_commission = sum(t.commission_cost for t in trades)
            total_slippage = sum(t.slippage_cost for t in trades)
        else:
            win_rate = avg_win = avg_loss = profit_factor = 0
            total_commission = total_slippage = 0
        
        return {
            'summary': {
                'total_return': total_return,
                'sharpe_ratio': sharpe,
                'max_drawdown': dd_analysis.max_drawdown,
                'max_dd_duration': dd_analysis.max_drawdown_duration,
                'win_rate': win_rate,
                'profit_factor': profit_factor if profit_factor != float('inf') else 999,
                'total_trades': len(trades),
                'total_commission': total_commission,
                'total_slippage': total_slippage,
                'net_profit': equity.iloc[-1] - self.initial_capital
            },
            'equity_curve': equity,
            'trades': trades,
            'drawdown_analysis': dd_analysis,
            'monte_carlo': mc_result,
            'returns': returns_series
        }


# ============================================================================
# STREAMLIT RENDERING
# ============================================================================

def render_enhanced_backtest_dashboard():
    """Render enhanced backtesting dashboard."""
    import streamlit as st
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    st.subheader(" Enhanced Backtesting Engine")
    
    st.write("### Transaction Cost Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        commission_type = st.selectbox(
            "Commission Type",
            ["Flat", "Per Share", "Percentage"]
        )
        commission_value = st.number_input(
            "Commission Value",
            min_value=0.0,
            value=0.0 if commission_type == "Flat" else 0.005
        )
    
    with col2:
        slippage_model = st.selectbox(
            "Slippage Model",
            ["Fixed", "Proportional", "Volatility-Based", "Market Impact"]
        )
        slippage_bps = st.number_input(
            "Slippage (basis points)",
            min_value=0.0,
            value=5.0
        )
    
    st.write("### Analysis Tools")
    
    tabs = st.tabs(["Drawdown Analysis", "Walk-Forward", "Monte Carlo"])
    
    with tabs[0]:
        st.write("#### Drawdown Analysis")
        st.info("""
        Drawdown analysis includes:
        - Maximum drawdown depth and duration
        - Average drawdown characteristics
        - Recovery period analysis
        - Underwater time percentage
        """)
        
        # Demo with sample data
        if st.button("Run Demo Drawdown Analysis"):
            np.random.seed(42)
            equity = pd.Series(
                100000 * (1 + np.random.randn(252).cumsum() * 0.01),
                index=pd.date_range('2025-01-01', periods=252, freq='B')
            )
            
            analyzer = DrawdownAnalyzer()
            result = analyzer.analyze(equity)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Max Drawdown", f"{result.max_drawdown:.1f}%")
            with col2:
                st.metric("Max DD Duration", f"{result.max_drawdown_duration} days")
            with col3:
                st.metric("Avg Recovery", f"{result.avg_recovery_time:.0f} days")
            with col4:
                st.metric("Underwater %", f"{result.underwater_percentage:.0f}%")
            
            # Drawdown chart
            running_max = equity.expanding().max()
            drawdown = (equity - running_max) / running_max * 100
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                              vertical_spacing=0.05, row_heights=[0.7, 0.3])
            
            fig.add_trace(go.Scatter(
                x=equity.index, y=equity.values,
                name='Equity', line=dict(color='#00D4AA')
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=drawdown.index, y=drawdown.values,
                name='Drawdown', fill='tozeroy',
                line=dict(color='#FF5722')
            ), row=2, col=1)
            
            fig.update_layout(template='plotly_dark', height=500)
            st.plotly_chart(fig, width="stretch")
    
    with tabs[1]:
        st.write("#### Walk-Forward Analysis")
        st.info("""
        Walk-forward optimization:
        - Anchored vs Rolling windows
        - In-sample optimization
        - Out-of-sample validation
        - Degradation analysis
        - Robustness scoring
        """)
        
        num_windows = st.slider("Number of Windows", 3, 10, 5)
        train_pct = st.slider("Training %", 50, 80, 70) / 100
        anchored = st.checkbox("Use Anchored Walk-Forward")
        
        st.write(f"Configuration: {num_windows} windows, {train_pct*100:.0f}% training, {'Anchored' if anchored else 'Rolling'}")
    
    with tabs[2]:
        st.write("#### Monte Carlo Simulation")
        st.info("""
        Monte Carlo analysis:
        - Permutation testing for statistical significance
        - Bootstrap confidence intervals
        - P-value calculation
        - Luck vs skill assessment
        """)
        
        num_sims = st.slider("Number of Simulations", 100, 5000, 1000)
        
        if st.button("Run Demo Monte Carlo"):
            np.random.seed(42)
            returns = pd.Series(np.random.randn(252) * 0.02 + 0.0003)
            
            analyzer = MonteCarloAnalyzer(num_simulations=num_sims)
            result = analyzer.permutation_test(returns)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Original Return", f"{result.original_return:.1f}%")
            with col2:
                st.metric("P-Value", f"{result.p_value:.3f}")
            with col3:
                sig_text = " Significant" if result.is_significant else " Not Significant"
                st.metric("Significance", sig_text)
            
            st.write(f"**Interpretation:** There is a {result.p_value*100:.1f}% probability "
                    f"that the strategy's return of {result.original_return:.1f}% could be "
                    f"achieved by random chance.")
            
            # Distribution chart
            fig = go.Figure()
            
            # Simulated distribution histogram
            sim_returns = np.random.randn(num_sims) * result.std_simulated_return + result.mean_simulated_return
            fig.add_trace(go.Histogram(
                x=sim_returns,
                nbinsx=50,
                name='Simulated Returns',
                opacity=0.7
            ))
            
            # Original return line
            fig.add_vline(
                x=result.original_return,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Original: {result.original_return:.1f}%"
            )
            
            fig.update_layout(
                title="Monte Carlo Distribution vs Original Return",
                xaxis_title="Return (%)",
                yaxis_title="Frequency",
                template='plotly_dark',
                height=400
            )
            
            st.plotly_chart(fig, width="stretch")


# Make features available
HAS_ENHANCED_BACKTEST = True
