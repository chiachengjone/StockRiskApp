"""
Stock Comparison - Side-by-Side Analysis
=========================================
Compare multiple stocks on key metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging


class StockComparison:
    """
    Compare multiple stocks on risk and return metrics.
    
    Features:
    - Side-by-side metrics comparison
    - Radar chart data
    - Winner/loser determination
    - Ranking by metric
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def compare_metrics(
        self,
        tickers: List[str],
        metrics_list: List[Dict]
    ) -> pd.DataFrame:
        """
        Create comparison table for multiple stocks.
        
        Args:
            tickers: List of ticker symbols
            metrics_list: List of metrics dictionaries (same order as tickers)
        
        Returns:
            DataFrame with comparison
        """
        if len(tickers) != len(metrics_list):
            raise ValueError("Tickers and metrics lists must have same length")
        
        # Metrics to compare
        metric_definitions = [
            ('ann_ret', 'Annual Return', 'percent', 'higher'),
            ('ann_vol', 'Volatility', 'percent', 'lower'),
            ('max_dd', 'Max Drawdown', 'percent', 'higher'),  # Less negative is better
            ('sharpe', 'Sharpe Ratio', 'number', 'higher'),
            ('sortino', 'Sortino Ratio', 'number', 'higher'),
            ('calmar', 'Calmar Ratio', 'number', 'higher'),
            ('skew', 'Skewness', 'number', 'higher'),
            ('kurtosis', 'Kurtosis', 'number', 'lower'),  # Lower excess kurtosis preferred
        ]
        
        rows = []
        for key, label, fmt, better in metric_definitions:
            row = {'Metric': label}
            values = []
            
            for i, ticker in enumerate(tickers):
                value = metrics_list[i].get(key)
                values.append(value)
                
                if value is not None:
                    if fmt == 'percent':
                        row[ticker] = f"{value:.2%}"
                    else:
                        row[ticker] = f"{value:.2f}"
                else:
                    row[ticker] = 'N/A'
            
            # Determine winner
            valid_values = [(v, t) for v, t in zip(values, tickers) if v is not None]
            if valid_values:
                if better == 'higher':
                    winner = max(valid_values, key=lambda x: x[0])[1]
                else:
                    winner = min(valid_values, key=lambda x: x[0])[1]
                row['Best'] = winner
            else:
                row['Best'] = 'N/A'
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def get_radar_chart_data(
        self,
        tickers: List[str],
        metrics_list: List[Dict]
    ) -> Dict:
        """
        Prepare data for radar chart visualization.
        
        Normalizes metrics to 0-100 scale for comparison.
        
        Returns:
            Dictionary with categories and normalized values per ticker
        """
        # Metrics for radar chart (all normalized so higher = better)
        radar_metrics = [
            ('sharpe', 'Sharpe'),
            ('sortino', 'Sortino'),
            ('calmar', 'Calmar'),
            ('ann_ret', 'Return'),
            # Invert volatility and drawdown so higher = better
        ]
        
        # Get raw values
        raw_data = {ticker: [] for ticker in tickers}
        categories = []
        
        for key, label in radar_metrics:
            categories.append(label)
            for i, ticker in enumerate(tickers):
                value = metrics_list[i].get(key, 0) or 0
                raw_data[ticker].append(value)
        
        # Add inverted volatility (lower vol = higher score)
        categories.append('Low Vol')
        for i, ticker in enumerate(tickers):
            vol = metrics_list[i].get('ann_vol', 0.3) or 0.3
            # Invert: high vol -> low score
            raw_data[ticker].append(1 / vol if vol > 0 else 0)
        
        # Add inverted drawdown (less negative = higher score)
        categories.append('Stability')
        for i, ticker in enumerate(tickers):
            dd = metrics_list[i].get('max_dd', -0.5) or -0.5
            # Less negative is better
            raw_data[ticker].append(1 + dd if dd < 0 else 1)
        
        # Normalize to 0-100 scale
        normalized_data = {}
        for i, category in enumerate(categories):
            values = [raw_data[t][i] for t in tickers]
            min_val = min(values)
            max_val = max(values)
            range_val = max_val - min_val if max_val != min_val else 1
            
            for ticker in tickers:
                if ticker not in normalized_data:
                    normalized_data[ticker] = []
                normalized = (raw_data[ticker][i] - min_val) / range_val * 100
                normalized_data[ticker].append(normalized)
        
        return {
            'categories': categories,
            'data': normalized_data
        }
    
    def rank_stocks(
        self,
        tickers: List[str],
        metrics_list: List[Dict],
        ranking_metric: str = 'sharpe'
    ) -> List[Tuple[str, float]]:
        """
        Rank stocks by a specific metric.
        
        Returns:
            List of (ticker, value) tuples sorted by metric
        """
        rankings = []
        
        for i, ticker in enumerate(tickers):
            value = metrics_list[i].get(ranking_metric)
            if value is not None:
                rankings.append((ticker, value))
        
        # Sort descending for most metrics
        reverse = ranking_metric not in ['ann_vol', 'kurtosis']
        rankings.sort(key=lambda x: x[1], reverse=reverse)
        
        return rankings
    
    def calculate_composite_score(
        self,
        metrics: Dict,
        weights: Dict = None
    ) -> float:
        """
        Calculate a composite score for a stock.
        
        Args:
            metrics: Metrics dictionary
            weights: Optional custom weights for each metric
        
        Returns:
            Composite score (higher is better)
        """
        if weights is None:
            weights = {
                'sharpe': 0.25,
                'sortino': 0.20,
                'calmar': 0.15,
                'ann_ret': 0.15,
                'ann_vol': -0.15,  # Negative weight (lower is better)
                'max_dd': 0.10,    # Less negative is better
            }
        
        score = 0
        total_weight = 0
        
        for metric, weight in weights.items():
            value = metrics.get(metric)
            if value is not None:
                # Normalize based on typical ranges
                if metric == 'sharpe':
                    normalized = value / 3  # Assume 3 is excellent
                elif metric == 'sortino':
                    normalized = value / 4
                elif metric == 'calmar':
                    normalized = value / 5
                elif metric == 'ann_ret':
                    normalized = (value + 0.5) / 1  # -50% to +50% -> 0 to 1
                elif metric == 'ann_vol':
                    normalized = (0.5 - value) / 0.5  # 0% to 50% -> 1 to 0
                elif metric == 'max_dd':
                    normalized = (value + 0.5) / 0.5  # -50% to 0% -> 0 to 1
                else:
                    normalized = value
                
                score += normalized * abs(weight)
                total_weight += abs(weight)
        
        return score / total_weight if total_weight > 0 else 0
    
    def generate_comparison_summary(
        self,
        tickers: List[str],
        metrics_list: List[Dict]
    ) -> Dict:
        """
        Generate a comprehensive comparison summary.
        
        Returns:
            Dictionary with comparison insights
        """
        if len(tickers) < 2:
            return {'error': 'Need at least 2 stocks to compare'}
        
        # Calculate composite scores
        scores = []
        for i, ticker in enumerate(tickers):
            score = self.calculate_composite_score(metrics_list[i])
            scores.append((ticker, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Find winners for each category
        category_winners = {}
        
        metrics_to_check = [
            ('ann_ret', 'Highest Return', True),
            ('ann_vol', 'Lowest Volatility', False),
            ('sharpe', 'Best Risk-Adjusted', True),
            ('max_dd', 'Smallest Drawdown', True),  # Less negative
        ]
        
        for metric, label, higher_better in metrics_to_check:
            values = [(t, metrics_list[i].get(metric)) 
                     for i, t in enumerate(tickers) 
                     if metrics_list[i].get(metric) is not None]
            
            if values:
                if higher_better:
                    winner = max(values, key=lambda x: x[1])
                else:
                    winner = min(values, key=lambda x: x[1])
                category_winners[label] = winner[0]
        
        return {
            'overall_ranking': scores,
            'best_overall': scores[0][0],
            'worst_overall': scores[-1][0],
            'category_winners': category_winners,
            'score_difference': scores[0][1] - scores[-1][1],
            'tickers_analyzed': len(tickers)
        }
    
    def get_correlation_matrix(
        self,
        returns_dict: Dict[str, 'pd.Series']
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix between stocks.
        
        Args:
            returns_dict: Dictionary of ticker -> returns series
        
        Returns:
            Correlation matrix DataFrame
        """
        returns_df = pd.DataFrame(returns_dict)
        return returns_df.corr()
    
    def find_diversification_pairs(
        self,
        returns_dict: Dict[str, 'pd.Series'],
        threshold: float = 0.3
    ) -> List[Tuple[str, str, float]]:
        """
        Find stock pairs with low correlation for diversification.
        
        Args:
            returns_dict: Dictionary of ticker -> returns series
            threshold: Maximum correlation to consider "low"
        
        Returns:
            List of (ticker1, ticker2, correlation) tuples
        """
        corr_matrix = self.get_correlation_matrix(returns_dict)
        
        low_corr_pairs = []
        tickers = list(returns_dict.keys())
        
        for i, t1 in enumerate(tickers):
            for t2 in tickers[i+1:]:
                corr = corr_matrix.loc[t1, t2]
                if abs(corr) < threshold:
                    low_corr_pairs.append((t1, t2, corr))
        
        # Sort by correlation (lowest first)
        low_corr_pairs.sort(key=lambda x: abs(x[2]))
        
        return low_corr_pairs
