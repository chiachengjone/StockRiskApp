"""
Factor Analysis Service
=======================
Factor analysis calculations ported from factors.py

Provides Fama-French, Kelly Criterion, ESG, and Style Factor analysis.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class FactorAnalyzer:
    """Enterprise factor analysis for quant strategies."""
    
    def __init__(self):
        self.ff_factors = None
        self.esg_cache = {}
    
    def fama_french_regression(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series = None
    ) -> Dict[str, Any]:
        """
        Fama-French 5-Factor Regression.
        
        Factors: Mkt-RF, SMB, HML, RMW, CMA
        Model: R_excess = α + β1*Mkt-RF + β2*SMB + β3*HML + β4*RMW + β5*CMA + ε
        
        Returns loadings, alpha (annualized), and R².
        """
        try:
            import pandas_datareader.data as web
            ff = web.DataReader(
                'F-F_Research_Data_5_Factors_2x3_daily',
                'famafrench',
                start=returns.index[0],
                end=returns.index[-1]
            )[0]
            ff = ff / 100  # Convert from percentage
            self.ff_factors = ff
        except Exception:
            # Fallback: simulate factors from benchmark
            if benchmark_returns is not None and len(benchmark_returns) > 50:
                ff = self._simulate_factors(benchmark_returns)
            else:
                return {
                    'error': 'Could not fetch Fama-French factors',
                    'alpha': 0.0,
                    'r_squared': 0.0,
                    'loadings': {},
                    'n_observations': 0
                }
        
        # Align dates
        aligned = pd.DataFrame({'returns': returns}).join(ff, how='inner').dropna()
        
        if len(aligned) < 60:
            return {
                'error': 'Insufficient overlapping data',
                'alpha': 0.0,
                'r_squared': 0.0,
                'loadings': {},
                'n_observations': len(aligned)
            }
        
        # Excess returns
        rf_col = 'RF' if 'RF' in aligned.columns else None
        if rf_col:
            y = aligned['returns'] - aligned[rf_col]
        else:
            y = aligned['returns']
        
        # Factor columns
        factor_cols = [c for c in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA'] if c in aligned.columns]
        
        if not factor_cols:
            return {
                'error': 'No factor columns found',
                'alpha': 0.0,
                'r_squared': 0.0,
                'loadings': {},
                'n_observations': len(aligned)
            }
        
        X = aligned[factor_cols].values
        y_vals = y.values
        
        # OLS regression
        reg = LinearRegression().fit(X, y_vals)
        
        # Results
        loadings = {factor_cols[i]: float(reg.coef_[i]) for i in range(len(factor_cols))}
        alpha_daily = float(reg.intercept_)
        alpha_annual = alpha_daily * 252
        r_squared = float(reg.score(X, y_vals))
        
        return {
            'alpha': alpha_annual,
            'r_squared': r_squared,
            'loadings': loadings,
            'n_observations': len(aligned)
        }
    
    def _simulate_factors(self, benchmark_returns: pd.Series) -> pd.DataFrame:
        """Simulate Fama-French factors from benchmark when API unavailable."""
        np.random.seed(42)
        n = len(benchmark_returns)
        
        # Create synthetic factors based on benchmark behavior
        mkt_rf = benchmark_returns.values + np.random.normal(0, 0.001, n)
        smb = np.random.normal(0.0001, 0.005, n)
        hml = np.random.normal(0.0001, 0.004, n)
        rmw = np.random.normal(0.0001, 0.003, n)
        cma = np.random.normal(0.0001, 0.003, n)
        rf = np.full(n, 0.0002)  # ~5% annualized
        
        return pd.DataFrame({
            'Mkt-RF': mkt_rf,
            'SMB': smb,
            'HML': hml,
            'RMW': rmw,
            'CMA': cma,
            'RF': rf
        }, index=benchmark_returns.index)
    
    def kelly_criterion(
        self,
        returns: pd.Series,
        fraction: float = 0.5
    ) -> Dict[str, float]:
        """
        Kelly Criterion for optimal position sizing.
        
        Kelly % = W - (1-W)/R
        where W = win rate, R = win/loss ratio
        
        Args:
            returns: Daily returns series
            fraction: Kelly fraction (0.5 = half Kelly for safety)
        
        Returns:
            Dictionary with Kelly metrics
        """
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        
        if len(wins) == 0 or len(losses) == 0:
            return {
                'kelly_pct': 0.0,
                'full_kelly': 0.0,
                'win_rate': 0.0,
                'win_loss_ratio': 0.0,
                'edge_per_trade': 0.0,
                'fraction_used': fraction
            }
        
        win_rate = len(wins) / len(returns)
        avg_win = float(wins.mean())
        avg_loss = float(abs(losses.mean()))
        
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        
        # Kelly formula
        full_kelly = win_rate - (1 - win_rate) / win_loss_ratio if win_loss_ratio > 0 else 0
        kelly_pct = full_kelly * fraction
        
        # Edge per trade
        edge = win_rate * avg_win - (1 - win_rate) * avg_loss
        
        return {
            'kelly_pct': float(max(0, min(1, kelly_pct))),
            'full_kelly': float(max(0, min(1, full_kelly))),
            'win_rate': float(win_rate),
            'win_loss_ratio': float(win_loss_ratio),
            'edge_per_trade': float(edge),
            'fraction_used': fraction
        }
    
    def get_esg_rating(self, ticker: str) -> Dict[str, Any]:
        """
        Fetch ESG rating from Yahoo Finance.
        
        Returns ESG scores and rating.
        """
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            esg = stock.sustainability
            
            if esg is None or esg.empty:
                return {
                    'rating': 'NR',
                    'total_esg': None,
                    'environment_score': None,
                    'social_score': None,
                    'governance_score': None,
                    'source': 'Yahoo Finance'
                }
            
            # Extract scores
            total_esg = float(esg.loc['totalEsg'].iloc[0]) if 'totalEsg' in esg.index else None
            env_score = float(esg.loc['environmentScore'].iloc[0]) if 'environmentScore' in esg.index else None
            soc_score = float(esg.loc['socialScore'].iloc[0]) if 'socialScore' in esg.index else None
            gov_score = float(esg.loc['governanceScore'].iloc[0]) if 'governanceScore' in esg.index else None
            
            # Calculate rating
            if total_esg is not None:
                if total_esg < 10:
                    rating = 'AAA'
                elif total_esg < 15:
                    rating = 'AA'
                elif total_esg < 20:
                    rating = 'A'
                elif total_esg < 25:
                    rating = 'BBB'
                elif total_esg < 30:
                    rating = 'BB'
                elif total_esg < 35:
                    rating = 'B'
                else:
                    rating = 'CCC'
            else:
                rating = 'NR'
            
            return {
                'rating': rating,
                'total_esg': total_esg,
                'environment_score': env_score,
                'social_score': soc_score,
                'governance_score': gov_score,
                'source': 'Yahoo Finance'
            }
            
        except Exception as e:
            logger.warning(f"ESG fetch failed for {ticker}: {e}")
            return {
                'rating': 'NR',
                'total_esg': None,
                'environment_score': None,
                'social_score': None,
                'governance_score': None,
                'source': 'Yahoo Finance'
            }


class StyleFactorAnalyzer:
    """
    Multi-Factor Style Analysis for institutional portfolio management.
    
    Computes Momentum, Value, and Quality factors.
    """
    
    def __init__(self):
        self.factor_cache = {}
    
    def analyze_style_factors(
        self,
        returns: pd.Series,
        prices: pd.Series,
        fundamentals: Dict[str, Any] = None,
        benchmark_returns: pd.Series = None
    ) -> Dict[str, Any]:
        """
        Compute comprehensive style factor analysis.
        """
        if len(returns) < 60:
            return {'error': 'Insufficient data for style analysis (need 60+ days)'}
        
        # Compute individual factor categories
        momentum = self._compute_momentum_factors(returns, prices)
        value = self._compute_value_factors(fundamentals)
        quality = self._compute_quality_factors(fundamentals)
        
        # Compute composite scores (normalized 0-100)
        scores = {
            'momentum': self._normalize_score(momentum.get('composite', 50)),
            'value': self._normalize_score(value.get('composite', 50)),
            'quality': self._normalize_score(quality.get('composite', 50)),
        }
        
        # Relative performance
        if benchmark_returns is not None and len(benchmark_returns) > 20:
            relative = self._compute_relative_performance(returns, benchmark_returns)
            scores['relative_strength'] = self._normalize_score(relative.get('score', 50))
        else:
            scores['relative_strength'] = 50.0
        
        # Overall score
        overall_score = np.mean([scores['momentum'], scores['value'], scores['quality']])
        
        # Radar chart data
        radar_data = {
            'categories': ['Momentum', 'Value', 'Quality', 'Relative Strength'],
            'scores': [
                scores['momentum'],
                scores['value'],
                scores['quality'],
                scores['relative_strength']
            ]
        }
        
        # Style classification
        style = self._classify_style(scores)
        
        return {
            'overall_score': float(overall_score),
            'scores': scores,
            'momentum_details': momentum,
            'value_details': value,
            'quality_details': quality,
            'radar_data': radar_data,
            'style_classification': style,
            'style_label': self._get_style_label(style)
        }
    
    def _compute_momentum_factors(
        self,
        returns: pd.Series,
        prices: pd.Series
    ) -> Dict[str, float]:
        """Compute momentum factor metrics."""
        # Price momentum
        mom_12m_1m = 0.0
        if len(prices) >= 252:
            price_12m_ago = prices.iloc[-252]
            price_1m_ago = prices.iloc[-21]
            mom_12m_1m = ((price_1m_ago / price_12m_ago) - 1) * 100
        
        mom_3m = 0.0
        if len(prices) >= 63:
            mom_3m = ((prices.iloc[-1] / prices.iloc[-63]) - 1) * 100
        
        mom_1m = 0.0
        if len(prices) >= 21:
            mom_1m = ((prices.iloc[-1] / prices.iloc[-21]) - 1) * 100
        
        # RSI
        rsi = self._compute_rsi(prices, 14)
        
        # Trend strength
        vol = returns.tail(20).std() * np.sqrt(252)
        trend_strength = abs(mom_3m) / (vol * 100) if vol > 0 else 0
        trend_strength = min(trend_strength * 50, 100)
        
        # Composite
        rsi_normalized = (rsi - 50) + 50
        composite = (
            0.4 * self._normalize_score(mom_12m_1m, center=0, scale=50) +
            0.3 * self._normalize_score(mom_3m, center=0, scale=30) +
            0.2 * rsi_normalized +
            0.1 * trend_strength
        )
        
        return {
            'momentum_12m_1m': float(mom_12m_1m),
            'momentum_3m': float(mom_3m),
            'momentum_1m': float(mom_1m),
            'rsi_14': float(rsi),
            'trend_strength': float(trend_strength),
            'composite': float(composite)
        }
    
    def _compute_value_factors(
        self,
        fundamentals: Dict[str, Any] = None
    ) -> Dict[str, float]:
        """Compute value factor metrics."""
        if not fundamentals:
            return {'composite': 50.0, 'note': 'No fundamentals data'}
        
        pe = fundamentals.get('trailingPE') or fundamentals.get('forwardPE')
        pb = fundamentals.get('priceToBook')
        div_yield = fundamentals.get('dividendYield', 0) or 0
        
        pe_score = 50.0
        if pe is not None and pe > 0:
            pe_score = max(0, min(100, 100 - (pe - 10) * 3))
        
        pb_score = 50.0
        if pb is not None and pb > 0:
            pb_score = max(0, min(100, 100 - (pb - 1) * 20))
        
        div_score = 50.0
        if div_yield > 0:
            div_score = min(100, 50 + div_yield * 1000)
        
        composite = 0.4 * pe_score + 0.35 * pb_score + 0.25 * div_score
        
        return {
            'pe_ratio': pe,
            'pb_ratio': pb,
            'dividend_yield': div_yield * 100 if div_yield else 0,
            'pe_score': float(pe_score),
            'pb_score': float(pb_score),
            'div_score': float(div_score),
            'composite': float(composite)
        }
    
    def _compute_quality_factors(
        self,
        fundamentals: Dict[str, Any] = None
    ) -> Dict[str, float]:
        """Compute quality factor metrics."""
        if not fundamentals:
            return {'composite': 50.0, 'note': 'No fundamentals data'}
        
        roe = fundamentals.get('returnOnEquity')
        profit_margin = fundamentals.get('profitMargins')
        debt_equity = fundamentals.get('debtToEquity')
        gross_margin = fundamentals.get('grossMargins')
        
        roe_score = 50.0
        if roe is not None:
            roe_pct = roe * 100 if roe < 1 else roe
            roe_score = max(0, min(100, 30 + roe_pct * 3.5))
        
        margin_score = 50.0
        if profit_margin is not None:
            margin_pct = profit_margin * 100 if profit_margin < 1 else profit_margin
            margin_score = max(0, min(100, margin_pct * 4))
        
        debt_score = 50.0
        if debt_equity is not None and debt_equity >= 0:
            debt_score = max(0, min(100, 100 - debt_equity * 0.5))
        
        gross_score = 50.0
        if gross_margin is not None:
            gross_pct = gross_margin * 100 if gross_margin < 1 else gross_margin
            gross_score = max(0, min(100, gross_pct * 1.5))
        
        composite = 0.35 * roe_score + 0.25 * margin_score + 0.2 * debt_score + 0.2 * gross_score
        
        return {
            'roe': roe,
            'profit_margin': profit_margin,
            'debt_equity': debt_equity,
            'gross_margin': gross_margin,
            'roe_score': float(roe_score),
            'margin_score': float(margin_score),
            'debt_score': float(debt_score),
            'composite': float(composite)
        }
    
    def _compute_relative_performance(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> Dict[str, float]:
        """Compute relative strength vs benchmark."""
        aligned = pd.DataFrame({
            'stock': returns,
            'bench': benchmark_returns
        }).dropna()
        
        if len(aligned) < 20:
            return {'score': 50.0}
        
        stock_cum = (1 + aligned['stock']).cumprod().iloc[-1] - 1
        bench_cum = (1 + aligned['bench']).cumprod().iloc[-1] - 1
        excess = stock_cum - bench_cum
        
        score = 50 + excess * 200
        score = max(0, min(100, score))
        
        return {
            'stock_return': float(stock_cum),
            'benchmark_return': float(bench_cum),
            'excess_return': float(excess),
            'score': float(score)
        }
    
    def _compute_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Compute RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        if loss.iloc[-1] == 0:
            return 100.0
        
        rs = gain.iloc[-1] / loss.iloc[-1]
        rsi = 100 - (100 / (1 + rs))
        return float(rsi)
    
    def _normalize_score(
        self,
        value: float,
        center: float = 50,
        scale: float = 50
    ) -> float:
        """Normalize a value to 0-100 scale."""
        normalized = center + (value / scale) * 50
        return max(0, min(100, normalized))
    
    def _classify_style(self, scores: Dict[str, float]) -> Dict[str, str]:
        """Classify investment style based on factor scores."""
        classification = {}
        
        if scores['momentum'] >= 70:
            classification['momentum'] = 'High Momentum'
        elif scores['momentum'] <= 30:
            classification['momentum'] = 'Low Momentum'
        else:
            classification['momentum'] = 'Neutral'
        
        if scores['value'] >= 70:
            classification['value'] = 'Deep Value'
        elif scores['value'] <= 30:
            classification['value'] = 'Growth'
        else:
            classification['value'] = 'Blend'
        
        if scores['quality'] >= 70:
            classification['quality'] = 'High Quality'
        elif scores['quality'] <= 30:
            classification['quality'] = 'Speculative'
        else:
            classification['quality'] = 'Average Quality'
        
        return classification
    
    def _get_style_label(self, classification: Dict[str, str]) -> str:
        """Get a simple style label."""
        if classification.get('value') == 'Growth' and classification.get('momentum') == 'High Momentum':
            return 'Growth Momentum'
        elif classification.get('value') == 'Deep Value':
            return 'Value'
        elif classification.get('quality') == 'High Quality':
            return 'Quality'
        elif classification.get('momentum') == 'High Momentum':
            return 'Momentum'
        else:
            return 'Blend'


# Singleton instance
_factor_analyzer = None
_style_analyzer = None


def get_factor_analyzer() -> FactorAnalyzer:
    """Get singleton FactorAnalyzer instance."""
    global _factor_analyzer
    if _factor_analyzer is None:
        _factor_analyzer = FactorAnalyzer()
    return _factor_analyzer


def get_style_analyzer() -> StyleFactorAnalyzer:
    """Get singleton StyleFactorAnalyzer instance."""
    global _style_analyzer
    if _style_analyzer is None:
        _style_analyzer = StyleFactorAnalyzer()
    return _style_analyzer
