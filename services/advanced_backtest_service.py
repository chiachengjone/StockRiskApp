"""
Advanced Backtesting Service
=============================
Comprehensive backtesting with advanced analytics

Provides:
- Walk-forward analysis
- Monte Carlo simulation
- Parameter optimization
- Statistical significance testing
- Performance attribution

Author: Stock Risk App | Feb 2026
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class TradeRecord:
    """Record of a single trade."""
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    position_size: float
    pnl: float
    pnl_pct: float
    holding_period: int  # days
    trade_type: str  # 'LONG' or 'SHORT'
    exit_reason: str  # 'Signal', 'Stop Loss', 'Take Profit', 'Time Exit'
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'entry_date': self.entry_date.isoformat() if hasattr(self.entry_date, 'isoformat') else str(self.entry_date),
            'exit_date': self.exit_date.isoformat() if hasattr(self.exit_date, 'isoformat') else str(self.exit_date),
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'position_size': self.position_size,
            'pnl': self.pnl,
            'pnl_pct': self.pnl_pct,
            'holding_period': self.holding_period,
            'trade_type': self.trade_type,
            'exit_reason': self.exit_reason
        }


@dataclass
class BacktestMetrics:
    """Comprehensive backtest performance metrics."""
    # Return metrics
    total_return: float
    annualized_return: float
    benchmark_return: float
    alpha: float
    
    # Risk metrics
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int  # days
    
    # Trade metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    avg_trade: float
    
    # Risk-adjusted metrics
    expectancy: float
    kelly_criterion: float
    risk_reward_ratio: float
    
    # Time metrics
    avg_holding_period: float
    time_in_market: float  # percentage
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'benchmark_return': self.benchmark_return,
            'alpha': self.alpha,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_duration': self.max_drawdown_duration,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'profit_factor': self.profit_factor,
            'avg_trade': self.avg_trade,
            'expectancy': self.expectancy,
            'kelly_criterion': self.kelly_criterion,
            'risk_reward_ratio': self.risk_reward_ratio,
            'avg_holding_period': self.avg_holding_period,
            'time_in_market': self.time_in_market
        }


@dataclass
class WalkForwardResult:
    """Result from walk-forward analysis."""
    in_sample_metrics: List[BacktestMetrics]
    out_of_sample_metrics: List[BacktestMetrics]
    combined_equity: pd.Series
    robustness_score: float  # 0-100
    consistency_score: float  # 0-100
    period_results: List[Dict[str, Any]]


@dataclass
class MonteCarloResult:
    """Result from Monte Carlo simulation."""
    simulations: int
    mean_return: float
    median_return: float
    std_return: float
    percentile_5: float
    percentile_25: float
    percentile_75: float
    percentile_95: float
    probability_profit: float
    probability_drawdown_gt_20: float
    distribution: np.ndarray


@dataclass
class OptimizationResult:
    """Result from parameter optimization."""
    best_params: Dict[str, Any]
    best_metric_value: float
    optimization_metric: str
    all_results: List[Dict[str, Any]]
    parameter_sensitivity: Dict[str, float]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_detailed_metrics(
    equity_curve: pd.Series,
    trades: List[TradeRecord],
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.045
) -> BacktestMetrics:
    """
    Calculate comprehensive backtest metrics.
    
    Args:
        equity_curve: Equity curve series
        trades: List of trade records
        benchmark_returns: Benchmark returns for comparison
        risk_free_rate: Annual risk-free rate
        
    Returns:
        BacktestMetrics object
    """
    if len(equity_curve) < 2:
        return BacktestMetrics(
            total_return=0, annualized_return=0, benchmark_return=0, alpha=0,
            volatility=0, sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0,
            max_drawdown=0, max_drawdown_duration=0,
            total_trades=0, winning_trades=0, losing_trades=0, win_rate=0,
            avg_win=0, avg_loss=0, profit_factor=0, avg_trade=0,
            expectancy=0, kelly_criterion=0, risk_reward_ratio=0,
            avg_holding_period=0, time_in_market=0
        )
    
    # Returns
    returns = equity_curve.pct_change().dropna()
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
    
    # Annualized return (assuming daily data)
    n_days = len(equity_curve)
    years = n_days / 252
    annualized_return = ((1 + total_return/100) ** (1/years) - 1) * 100 if years > 0 else 0
    
    # Benchmark comparison
    benchmark_return = 0
    alpha = 0
    if benchmark_returns is not None and len(benchmark_returns) > 0:
        benchmark_return = (benchmark_returns + 1).prod() - 1
        benchmark_return *= 100
        alpha = annualized_return - benchmark_return
    
    # Volatility
    volatility = returns.std() * np.sqrt(252) * 100
    
    # Sharpe ratio
    excess_return = returns.mean() * 252 - risk_free_rate
    sharpe_ratio = excess_return / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
    
    # Sortino ratio (downside deviation)
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252)
    sortino_ratio = excess_return / downside_std if downside_std > 0 else 0
    
    # Max drawdown
    running_max = equity_curve.expanding().max()
    drawdowns = (equity_curve - running_max) / running_max * 100
    max_drawdown = abs(drawdowns.min())
    
    # Max drawdown duration
    is_in_drawdown = drawdowns < 0
    drawdown_periods = (is_in_drawdown != is_in_drawdown.shift()).cumsum()
    drawdown_lengths = is_in_drawdown.groupby(drawdown_periods).sum()
    max_drawdown_duration = int(drawdown_lengths.max()) if len(drawdown_lengths) > 0 else 0
    
    # Calmar ratio
    calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
    
    # Trade metrics
    total_trades = len(trades)
    winning_trades = len([t for t in trades if t.pnl > 0])
    losing_trades = len([t for t in trades if t.pnl <= 0])
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    wins = [t.pnl for t in trades if t.pnl > 0]
    losses = [abs(t.pnl) for t in trades if t.pnl <= 0]
    
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    
    total_wins = sum(wins)
    total_losses = sum(losses)
    profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
    
    avg_trade = np.mean([t.pnl for t in trades]) if trades else 0
    
    # Expectancy
    if total_trades > 0:
        expectancy = (win_rate/100 * avg_win) - ((1 - win_rate/100) * avg_loss)
    else:
        expectancy = 0
    
    # Kelly criterion
    if avg_loss > 0 and wins:
        win_prob = win_rate / 100
        win_loss_ratio = avg_win / avg_loss
        kelly_criterion = (win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio
        kelly_criterion = max(0, min(1, kelly_criterion))  # Clamp to 0-1
    else:
        kelly_criterion = 0
    
    # Risk/reward ratio
    risk_reward_ratio = avg_win / avg_loss if avg_loss > 0 else 0
    
    # Time metrics
    avg_holding_period = np.mean([t.holding_period for t in trades]) if trades else 0
    
    # Time in market (simplified)
    total_holding = sum(t.holding_period for t in trades)
    time_in_market = (total_holding / n_days * 100) if n_days > 0 else 0
    
    return BacktestMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        benchmark_return=benchmark_return,
        alpha=alpha,
        volatility=volatility,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        calmar_ratio=calmar_ratio,
        max_drawdown=max_drawdown,
        max_drawdown_duration=max_drawdown_duration,
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=profit_factor if profit_factor != float('inf') else 999.99,
        avg_trade=avg_trade,
        expectancy=expectancy,
        kelly_criterion=kelly_criterion,
        risk_reward_ratio=risk_reward_ratio,
        avg_holding_period=avg_holding_period,
        time_in_market=time_in_market
    )


def calculate_statistical_significance(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Calculate statistical significance of strategy performance.
    
    Args:
        strategy_returns: Strategy returns series
        benchmark_returns: Benchmark returns series
        alpha: Significance level
        
    Returns:
        Dictionary with statistical test results
    """
    from scipy import stats
    
    results = {}
    
    # Align series
    common_idx = strategy_returns.index.intersection(benchmark_returns.index)
    strat = strategy_returns.loc[common_idx]
    bench = benchmark_returns.loc[common_idx]
    
    if len(strat) < 30:
        return {'error': 'Insufficient data for statistical tests'}
    
    # T-test for mean difference
    t_stat, t_pval = stats.ttest_ind(strat, bench)
    results['t_test'] = {
        'statistic': float(t_stat),
        'p_value': float(t_pval),
        'significant': t_pval < alpha
    }
    
    # Sharpe ratio difference test (Jobson-Korkie)
    strat_sharpe = strat.mean() / strat.std() if strat.std() > 0 else 0
    bench_sharpe = bench.mean() / bench.std() if bench.std() > 0 else 0
    
    n = len(strat)
    se_diff = np.sqrt((2 - 2 * strat.corr(bench) + 0.5 * (strat_sharpe**2 + bench_sharpe**2 - strat_sharpe * bench_sharpe * strat.corr(bench))) / n)
    jk_stat = (strat_sharpe - bench_sharpe) / se_diff if se_diff > 0 else 0
    jk_pval = 2 * (1 - stats.norm.cdf(abs(jk_stat)))
    
    results['sharpe_difference'] = {
        'strategy_sharpe': float(strat_sharpe * np.sqrt(252)),
        'benchmark_sharpe': float(bench_sharpe * np.sqrt(252)),
        'difference': float((strat_sharpe - bench_sharpe) * np.sqrt(252)),
        'z_statistic': float(jk_stat),
        'p_value': float(jk_pval),
        'significant': jk_pval < alpha
    }
    
    # Kolmogorov-Smirnov test for distribution difference
    ks_stat, ks_pval = stats.ks_2samp(strat, bench)
    results['ks_test'] = {
        'statistic': float(ks_stat),
        'p_value': float(ks_pval),
        'significant': ks_pval < alpha
    }
    
    return results


# ============================================================================
# ADVANCED BACKTEST SERVICE CLASS
# ============================================================================

class AdvancedBacktestService:
    """
    Advanced Backtesting Service.
    
    Provides comprehensive backtesting with walk-forward analysis,
    Monte Carlo simulation, and parameter optimization.
    """
    
    def __init__(self):
        """Initialize the Advanced Backtest Service."""
        self._cache: Dict[str, Any] = {}
        logger.info("AdvancedBacktestService initialized")
    
    def walk_forward_analysis(
        self,
        df: pd.DataFrame,
        signal_generator: Callable,
        n_splits: int = 5,
        train_ratio: float = 0.7,
        initial_capital: float = 100000
    ) -> WalkForwardResult:
        """
        Perform walk-forward analysis.
        
        Args:
            df: OHLCV DataFrame
            signal_generator: Function that generates signals from data
            n_splits: Number of walk-forward periods
            train_ratio: Ratio of training to testing in each period
            initial_capital: Starting capital
            
        Returns:
            WalkForwardResult
        """
        n_samples = len(df)
        split_size = n_samples // n_splits
        
        in_sample_metrics = []
        out_of_sample_metrics = []
        period_results = []
        combined_equity = []
        
        for i in range(n_splits):
            start_idx = i * split_size
            end_idx = min((i + 1) * split_size, n_samples)
            
            period_data = df.iloc[start_idx:end_idx]
            train_size = int(len(period_data) * train_ratio)
            
            train_data = period_data.iloc[:train_size]
            test_data = period_data.iloc[train_size:]
            
            if len(train_data) < 50 or len(test_data) < 10:
                continue
            
            try:
                # Generate signals and backtest on training data
                train_signals = signal_generator(train_data)
                train_result = self._simple_backtest(train_data, train_signals, initial_capital)
                in_sample_metrics.append(train_result['metrics'])
                
                # Test on out-of-sample data
                test_signals = signal_generator(test_data)
                test_result = self._simple_backtest(test_data, test_signals, initial_capital)
                out_of_sample_metrics.append(test_result['metrics'])
                
                # Collect equity
                combined_equity.extend(test_result['equity'])
                
                period_results.append({
                    'period': i + 1,
                    'train_start': str(train_data.index[0]),
                    'train_end': str(train_data.index[-1]),
                    'test_start': str(test_data.index[0]),
                    'test_end': str(test_data.index[-1]),
                    'train_return': train_result['metrics'].total_return,
                    'test_return': test_result['metrics'].total_return,
                    'train_sharpe': train_result['metrics'].sharpe_ratio,
                    'test_sharpe': test_result['metrics'].sharpe_ratio
                })
                
            except Exception as e:
                logger.error(f"Walk-forward period {i} error: {e}")
                continue
        
        # Calculate robustness score
        if out_of_sample_metrics:
            positive_oos = len([m for m in out_of_sample_metrics if m.total_return > 0])
            robustness = (positive_oos / len(out_of_sample_metrics)) * 100
            
            # Consistency: compare IS vs OOS performance
            is_returns = [m.total_return for m in in_sample_metrics]
            oos_returns = [m.total_return for m in out_of_sample_metrics]
            
            if is_returns and oos_returns:
                avg_is = np.mean(is_returns)
                avg_oos = np.mean(oos_returns)
                consistency = max(0, 100 - abs(avg_is - avg_oos) * 2)
            else:
                consistency = 0
        else:
            robustness = 0
            consistency = 0
        
        return WalkForwardResult(
            in_sample_metrics=in_sample_metrics,
            out_of_sample_metrics=out_of_sample_metrics,
            combined_equity=pd.Series(combined_equity) if combined_equity else pd.Series([initial_capital]),
            robustness_score=robustness,
            consistency_score=consistency,
            period_results=period_results
        )
    
    def monte_carlo_simulation(
        self,
        trades: List[TradeRecord],
        n_simulations: int = 1000,
        initial_capital: float = 100000
    ) -> MonteCarloResult:
        """
        Perform Monte Carlo simulation on trade results.
        
        Args:
            trades: List of historical trades
            n_simulations: Number of simulations
            initial_capital: Starting capital
            
        Returns:
            MonteCarloResult
        """
        if not trades:
            return MonteCarloResult(
                simulations=0, mean_return=0, median_return=0, std_return=0,
                percentile_5=0, percentile_25=0, percentile_75=0, percentile_95=0,
                probability_profit=0, probability_drawdown_gt_20=0,
                distribution=np.array([])
            )
        
        # Extract trade returns
        trade_returns = [t.pnl_pct / 100 for t in trades]
        n_trades = len(trade_returns)
        
        final_returns = []
        max_drawdowns = []
        
        for _ in range(n_simulations):
            # Randomly sample trades with replacement
            sampled_returns = np.random.choice(trade_returns, size=n_trades, replace=True)
            
            # Calculate equity curve
            equity = [initial_capital]
            for ret in sampled_returns:
                equity.append(equity[-1] * (1 + ret))
            
            equity = np.array(equity)
            final_return = (equity[-1] / equity[0] - 1) * 100
            final_returns.append(final_return)
            
            # Calculate max drawdown
            running_max = np.maximum.accumulate(equity)
            drawdown = (equity - running_max) / running_max
            max_drawdowns.append(abs(drawdown.min()) * 100)
        
        final_returns = np.array(final_returns)
        max_drawdowns = np.array(max_drawdowns)
        
        return MonteCarloResult(
            simulations=n_simulations,
            mean_return=float(np.mean(final_returns)),
            median_return=float(np.median(final_returns)),
            std_return=float(np.std(final_returns)),
            percentile_5=float(np.percentile(final_returns, 5)),
            percentile_25=float(np.percentile(final_returns, 25)),
            percentile_75=float(np.percentile(final_returns, 75)),
            percentile_95=float(np.percentile(final_returns, 95)),
            probability_profit=float(np.mean(final_returns > 0) * 100),
            probability_drawdown_gt_20=float(np.mean(max_drawdowns > 20) * 100),
            distribution=final_returns
        )
    
    def optimize_parameters(
        self,
        df: pd.DataFrame,
        parameter_ranges: Dict[str, List[Any]],
        strategy_func: Callable,
        optimization_metric: str = 'sharpe_ratio',
        max_combinations: int = 100
    ) -> OptimizationResult:
        """
        Optimize strategy parameters using grid search.
        
        Args:
            df: OHLCV DataFrame
            parameter_ranges: Dict of parameter names to possible values
            strategy_func: Function(df, params) -> backtest result
            optimization_metric: Metric to optimize
            max_combinations: Maximum parameter combinations to test
            
        Returns:
            OptimizationResult
        """
        from itertools import product
        
        # Generate all combinations
        param_names = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())
        all_combinations = list(product(*param_values))
        
        # Limit combinations
        if len(all_combinations) > max_combinations:
            indices = np.random.choice(len(all_combinations), max_combinations, replace=False)
            all_combinations = [all_combinations[i] for i in indices]
        
        all_results = []
        best_value = float('-inf')
        best_params = None
        
        for combo in all_combinations:
            params = dict(zip(param_names, combo))
            
            try:
                result = strategy_func(df, params)
                metric_value = result.get(optimization_metric, 0)
                
                all_results.append({
                    'params': params,
                    'metric_value': metric_value,
                    **result
                })
                
                if metric_value > best_value:
                    best_value = metric_value
                    best_params = params
                    
            except Exception as e:
                logger.error(f"Optimization error with params {params}: {e}")
                continue
        
        # Calculate parameter sensitivity
        sensitivity = {}
        for param_name in param_names:
            param_results = {}
            for result in all_results:
                val = result['params'][param_name]
                if val not in param_results:
                    param_results[val] = []
                param_results[val].append(result['metric_value'])
            
            # Sensitivity = std of means across parameter values
            means = [np.mean(v) for v in param_results.values()]
            sensitivity[param_name] = float(np.std(means)) if means else 0
        
        return OptimizationResult(
            best_params=best_params or {},
            best_metric_value=best_value,
            optimization_metric=optimization_metric,
            all_results=all_results,
            parameter_sensitivity=sensitivity
        )
    
    def _simple_backtest(
        self,
        df: pd.DataFrame,
        signals: List[Dict[str, Any]],
        initial_capital: float
    ) -> Dict[str, Any]:
        """
        Simple backtest helper for walk-forward analysis.
        """
        if isinstance(df.columns, pd.MultiIndex):
            close = df['Close'].iloc[:, 0]
        else:
            close = df['Close']
        
        capital = initial_capital
        position = 0
        entry_price = 0
        trades = []
        equity = [initial_capital]
        
        for signal in signals:
            try:
                idx = signal.get('index', -1)
                action = signal.get('action', 'HOLD')
                
                if idx < 0 or idx >= len(close):
                    continue
                
                price = float(close.iloc[idx])
                date = close.index[idx]
                
                if action == 'BUY' and position == 0:
                    position = capital * 0.1 / price
                    entry_price = price
                    entry_date = date
                    
                elif action == 'SELL' and position > 0:
                    pnl = position * (price - entry_price)
                    capital += pnl
                    
                    trades.append(TradeRecord(
                        entry_date=entry_date,
                        exit_date=date,
                        entry_price=entry_price,
                        exit_price=price,
                        position_size=position,
                        pnl=pnl,
                        pnl_pct=(price / entry_price - 1) * 100,
                        holding_period=(date - entry_date).days if hasattr(date, 'days') else 1,
                        trade_type='LONG',
                        exit_reason='Signal'
                    ))
                    
                    position = 0
                
                equity.append(capital + (position * price if position > 0 else 0))
                
            except Exception as e:
                continue
        
        # Calculate metrics
        equity_series = pd.Series(equity)
        metrics = calculate_detailed_metrics(equity_series, trades)
        
        return {
            'metrics': metrics,
            'trades': trades,
            'equity': equity
        }
    
    def generate_performance_report(
        self,
        metrics: BacktestMetrics,
        trades: List[TradeRecord],
        equity_curve: pd.Series,
        monte_carlo: Optional[MonteCarloResult] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.
        
        Args:
            metrics: Backtest metrics
            trades: List of trades
            equity_curve: Equity curve
            monte_carlo: Monte Carlo results
            
        Returns:
            Performance report dictionary
        """
        report = {
            'summary': {
                'total_return': f"{metrics.total_return:.2f}%",
                'annualized_return': f"{metrics.annualized_return:.2f}%",
                'sharpe_ratio': f"{metrics.sharpe_ratio:.2f}",
                'max_drawdown': f"{metrics.max_drawdown:.2f}%",
                'win_rate': f"{metrics.win_rate:.1f}%",
                'profit_factor': f"{metrics.profit_factor:.2f}"
            },
            'risk_metrics': {
                'volatility': f"{metrics.volatility:.2f}%",
                'sortino_ratio': f"{metrics.sortino_ratio:.2f}",
                'calmar_ratio': f"{metrics.calmar_ratio:.2f}",
                'max_drawdown_duration': f"{metrics.max_drawdown_duration} days"
            },
            'trade_analysis': {
                'total_trades': metrics.total_trades,
                'winning_trades': metrics.winning_trades,
                'losing_trades': metrics.losing_trades,
                'avg_win': f"${metrics.avg_win:.2f}",
                'avg_loss': f"${metrics.avg_loss:.2f}",
                'avg_holding_period': f"{metrics.avg_holding_period:.1f} days",
                'expectancy': f"${metrics.expectancy:.2f}",
                'kelly_criterion': f"{metrics.kelly_criterion*100:.1f}%"
            },
            'monthly_returns': self._calculate_monthly_returns(equity_curve),
            'trade_distribution': self._analyze_trade_distribution(trades),
            'drawdown_analysis': self._analyze_drawdowns(equity_curve)
        }
        
        if monte_carlo:
            report['monte_carlo'] = {
                'simulations': monte_carlo.simulations,
                'mean_return': f"{monte_carlo.mean_return:.2f}%",
                'median_return': f"{monte_carlo.median_return:.2f}%",
                'probability_profit': f"{monte_carlo.probability_profit:.1f}%",
                'var_95': f"{monte_carlo.percentile_5:.2f}%",
                'best_case_95': f"{monte_carlo.percentile_95:.2f}%"
            }
        
        return report
    
    def _calculate_monthly_returns(
        self,
        equity_curve: pd.Series
    ) -> Dict[str, float]:
        """Calculate monthly returns from equity curve."""
        if not hasattr(equity_curve.index, 'to_period'):
            return {}
        
        try:
            monthly = equity_curve.resample('M').last()
            monthly_returns = monthly.pct_change().dropna() * 100
            
            return {
                str(date.strftime('%Y-%m')): float(ret)
                for date, ret in monthly_returns.items()
            }
        except:
            return {}
    
    def _analyze_trade_distribution(
        self,
        trades: List[TradeRecord]
    ) -> Dict[str, Any]:
        """Analyze trade distribution."""
        if not trades:
            return {}
        
        pnls = [t.pnl_pct for t in trades]
        
        return {
            'mean': float(np.mean(pnls)),
            'median': float(np.median(pnls)),
            'std': float(np.std(pnls)),
            'skewness': float(pd.Series(pnls).skew()),
            'kurtosis': float(pd.Series(pnls).kurtosis()),
            'max_win': float(max(pnls)),
            'max_loss': float(min(pnls)),
            'positive_trades_pct': float(len([p for p in pnls if p > 0]) / len(pnls) * 100)
        }
    
    def _analyze_drawdowns(
        self,
        equity_curve: pd.Series
    ) -> Dict[str, Any]:
        """Analyze drawdown periods."""
        running_max = equity_curve.expanding().max()
        drawdowns = (equity_curve - running_max) / running_max * 100
        
        # Find top 5 drawdown periods
        top_drawdowns = []
        
        # Simple approach: find local minima in drawdowns
        for i in range(1, len(drawdowns) - 1):
            if drawdowns.iloc[i] < drawdowns.iloc[i-1] and drawdowns.iloc[i] < drawdowns.iloc[i+1]:
                top_drawdowns.append({
                    'date': str(drawdowns.index[i]),
                    'drawdown': float(abs(drawdowns.iloc[i]))
                })
        
        top_drawdowns.sort(key=lambda x: x['drawdown'], reverse=True)
        
        return {
            'max_drawdown': float(abs(drawdowns.min())),
            'avg_drawdown': float(abs(drawdowns[drawdowns < 0].mean())) if len(drawdowns[drawdowns < 0]) > 0 else 0,
            'top_5_drawdowns': top_drawdowns[:5],
            'time_in_drawdown_pct': float(len(drawdowns[drawdowns < 0]) / len(drawdowns) * 100)
        }
