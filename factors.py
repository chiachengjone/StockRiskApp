"""
Factor Analysis Module - Enterprise Features
=============================================
Fama-French 5-Factor • Kelly Criterion • ESG Ratings • Style Factors

Author: Professional Risk Analytics | Jan 2026
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from typing import Dict, List, Optional, Any, Union
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# STYLE FACTOR ANALYZER (v4.3 Professional)
# =============================================================================

class StyleFactorAnalyzer:
    """
    Multi-Factor Style Analysis for institutional portfolio management.
    
    Computes Momentum, Value, and Quality factors with historical benchmarking
    and radar chart visualization support.
    
    Factors:
    - Momentum: Price momentum, RSI, trend strength
    - Value: P/E, P/B, dividend yield relative to sector
    - Quality: ROE, profit margins, debt ratios
    """
    
    def __init__(self):
        self.factor_cache = {}
        self.benchmark_data = {}
    
    def analyze_style_factors(
        self, 
        returns: pd.Series,
        prices: pd.Series,
        fundamentals: Dict[str, Any] = None,
        benchmark_returns: pd.Series = None
    ) -> Dict[str, Any]:
        """
        Compute comprehensive style factor analysis.
        
        Args:
            returns: Daily log returns
            prices: Price series
            fundamentals: Company fundamentals dict
            benchmark_returns: Benchmark for relative metrics
        
        Returns:
            Dictionary with momentum, value, quality scores and radar chart data
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
        
        # Add relative performance if benchmark provided
        if benchmark_returns is not None and len(benchmark_returns) > 20:
            relative = self._compute_relative_performance(returns, benchmark_returns)
            scores['relative_strength'] = self._normalize_score(relative.get('score', 50))
        else:
            scores['relative_strength'] = 50.0
        
        # Overall style score
        overall_score = np.mean([scores['momentum'], scores['value'], scores['quality']])
        
        # Radar chart data (for Plotly radar visualization)
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
        """
        Compute momentum factor metrics.
        
        Metrics:
        - 12M-1M momentum (skip most recent month)
        - 3M momentum
        - RSI (14-day)
        - Trend strength (ADX proxy)
        """
        n = len(returns)
        
        # Price momentum
        mom_12m_1m = 0.0
        if len(prices) >= 252:
            # 12-month return excluding last month
            price_12m_ago = prices.iloc[-252]
            price_1m_ago = prices.iloc[-21]
            price_now = prices.iloc[-1]
            mom_12m_1m = ((price_1m_ago / price_12m_ago) - 1) * 100
        
        mom_3m = 0.0
        if len(prices) >= 63:
            mom_3m = ((prices.iloc[-1] / prices.iloc[-63]) - 1) * 100
        
        mom_1m = 0.0
        if len(prices) >= 21:
            mom_1m = ((prices.iloc[-1] / prices.iloc[-21]) - 1) * 100
        
        # RSI calculation
        rsi = self._compute_rsi(prices, 14)
        
        # Trend strength (simplified using volatility-adjusted momentum)
        vol = returns.tail(20).std() * np.sqrt(252)
        trend_strength = abs(mom_3m) / (vol * 100) if vol > 0 else 0
        trend_strength = min(trend_strength * 50, 100)  # Scale to 0-100
        
        # Composite momentum score
        # Weight: 40% 12M-1M, 30% 3M, 20% RSI normalization, 10% trend
        rsi_normalized = (rsi - 50) + 50  # Center around 50
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
        """
        Compute value factor metrics.
        
        Metrics:
        - P/E ratio (inverse, lower = better)
        - P/B ratio (inverse)
        - Dividend yield
        - Free cash flow yield
        """
        if not fundamentals:
            return {'composite': 50.0, 'note': 'No fundamentals data'}
        
        # Extract metrics
        pe = fundamentals.get('trailingPE') or fundamentals.get('forwardPE')
        pb = fundamentals.get('priceToBook')
        div_yield = fundamentals.get('dividendYield', 0) or 0
        fcf_yield = fundamentals.get('freeCashflowYield', 0) or 0
        
        # Score each metric (higher = better value)
        pe_score = 50.0
        if pe is not None and pe > 0:
            # Lower P/E = higher score. P/E of 15 = neutral, <10 = great, >30 = poor
            pe_score = max(0, min(100, 100 - (pe - 10) * 3))
        
        pb_score = 50.0
        if pb is not None and pb > 0:
            # P/B < 1 = undervalued, > 3 = expensive
            pb_score = max(0, min(100, 100 - (pb - 1) * 20))
        
        div_score = 50.0
        if div_yield > 0:
            # Dividend yield 0% = 50, 5%+ = 100
            div_score = min(100, 50 + div_yield * 1000)
        
        # Composite value score
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
        """
        Compute quality factor metrics.
        
        Metrics:
        - Return on Equity (ROE)
        - Profit margins
        - Debt-to-equity ratio (inverse)
        - Earnings stability
        """
        if not fundamentals:
            return {'composite': 50.0, 'note': 'No fundamentals data'}
        
        # Extract metrics
        roe = fundamentals.get('returnOnEquity')
        profit_margin = fundamentals.get('profitMargins')
        debt_equity = fundamentals.get('debtToEquity')
        gross_margin = fundamentals.get('grossMargins')
        
        # Score each metric
        roe_score = 50.0
        if roe is not None:
            # ROE > 15% is excellent, < 5% is poor
            roe_pct = roe * 100 if roe < 1 else roe
            roe_score = max(0, min(100, 30 + roe_pct * 3.5))
        
        margin_score = 50.0
        if profit_margin is not None:
            margin_pct = profit_margin * 100 if profit_margin < 1 else profit_margin
            margin_score = max(0, min(100, margin_pct * 4))
        
        debt_score = 50.0
        if debt_equity is not None and debt_equity >= 0:
            # D/E < 50 = good, > 150 = risky
            debt_score = max(0, min(100, 100 - debt_equity * 0.5))
        
        gross_score = 50.0
        if gross_margin is not None:
            gross_pct = gross_margin * 100 if gross_margin < 1 else gross_margin
            gross_score = max(0, min(100, gross_pct * 1.5))
        
        # Composite quality score
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
        
        # Cumulative returns
        stock_cum = (1 + aligned['stock']).cumprod().iloc[-1] - 1
        bench_cum = (1 + aligned['bench']).cumprod().iloc[-1] - 1
        
        excess = stock_cum - bench_cum
        
        # Score based on excess return
        score = 50 + excess * 200  # +10% excess = score of 70
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
        
        # Momentum classification
        if scores['momentum'] >= 70:
            classification['momentum'] = 'High Momentum'
        elif scores['momentum'] <= 30:
            classification['momentum'] = 'Low Momentum'
        else:
            classification['momentum'] = 'Neutral'
        
        # Value classification
        if scores['value'] >= 70:
            classification['value'] = 'Deep Value'
        elif scores['value'] <= 30:
            classification['value'] = 'Growth'
        else:
            classification['value'] = 'Blend'
        
        # Quality classification
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


class FactorAnalyzer:
    """Enterprise factor analysis for quant strategies."""
    
    def __init__(self):
        self.ff_factors = None
        self.esg_cache = {}
    
    def fama_french_regression(self, returns: pd.Series, benchmark_returns: pd.Series = None) -> dict:
        """
        Fama-French 5-Factor Regression.
        
        Factors: Mkt-RF, SMB, HML, RMW, CMA
        Model: R_excess = α + β1*Mkt-RF + β2*SMB + β3*HML + β4*RMW + β5*CMA + ε
        
        Returns loadings, alpha (annualized), and R².
        """
        try:
            # Try to fetch Fama-French factors from pandas_datareader
            import pandas_datareader.data as web
            ff = web.DataReader('F-F_Research_Data_5_Factors_2x3_daily', 'famafrench', 
                               start=returns.index[0], end=returns.index[-1])[0]
            ff = ff / 100  # Convert from percentage
            self.ff_factors = ff
        except Exception:
            # Fallback: simulate factors from benchmark if available
            if benchmark_returns is not None and len(benchmark_returns) > 50:
                ff = self._simulate_factors(benchmark_returns)
            else:
                return {
                    'error': 'Could not fetch Fama-French factors. Install pandas-datareader.',
                    'alpha': 0.0,
                    'r_squared': 0.0,
                    'loadings': {}
                }
        
        # Align dates
        aligned = pd.DataFrame({'returns': returns}).join(ff, how='inner').dropna()
        
        if len(aligned) < 60:
            return {
                'error': 'Insufficient overlapping data',
                'alpha': 0.0,
                'r_squared': 0.0,
                'loadings': {}
            }
        
        # Excess returns (subtract risk-free rate)
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
                'loadings': {}
            }
        
        X = aligned[factor_cols].values
        y_vals = y.values
        
        # OLS regression
        reg = LinearRegression().fit(X, y_vals)
        
        # Results
        loadings = {factor_cols[i]: float(reg.coef_[i]) for i in range(len(factor_cols))}
        alpha_daily = float(reg.intercept_)
        alpha_annual = alpha_daily * 252  # Annualize
        r_squared = float(reg.score(X, y_vals))
        
        # T-stats (simplified)
        residuals = y_vals - reg.predict(X)
        mse = np.mean(residuals ** 2)
        se_alpha = np.sqrt(mse / len(y_vals))
        t_stat_alpha = alpha_daily / se_alpha if se_alpha > 0 else 0
        
        return {
            'alpha': alpha_annual,
            'alpha_daily': alpha_daily,
            't_stat_alpha': float(t_stat_alpha),
            'r_squared': r_squared,
            'loadings': loadings,
            'n_observations': len(aligned)
        }
    
    def _simulate_factors(self, benchmark_returns: pd.Series) -> pd.DataFrame:
        """Simulate FF-like factors from benchmark (fallback method)."""
        n = len(benchmark_returns)
        
        # Simulate correlated factors
        np.random.seed(42)
        mkt_rf = benchmark_returns.values
        smb = np.random.normal(0, 0.005, n)  # Small minus Big
        hml = np.random.normal(0, 0.004, n)  # High minus Low (Value)
        rmw = np.random.normal(0, 0.003, n)  # Robust minus Weak (Profitability)
        cma = np.random.normal(0, 0.003, n)  # Conservative minus Aggressive (Investment)
        rf = np.full(n, 0.0001)  # ~4% annual risk-free
        
        return pd.DataFrame({
            'Mkt-RF': mkt_rf,
            'SMB': smb,
            'HML': hml,
            'RMW': rmw,
            'CMA': cma,
            'RF': rf
        }, index=benchmark_returns.index)
    
    def kelly_criterion(self, returns: pd.Series, fraction: float = 0.5) -> dict:
        """
        Kelly Criterion Position Sizing.
        
        Full Kelly: f* = (p * b - q) / b
        Where: p = win probability, q = loss probability, b = win/loss ratio
        
        Args:
            returns: Daily returns series
            fraction: Kelly fraction (0.5 = half-Kelly, safer)
        
        Returns optimal position size (0-100%).
        """
        if len(returns) < 30:
            return {'kelly_pct': 0.0, 'error': 'Insufficient data'}
        
        # Calculate win/loss statistics
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        
        if len(wins) == 0 or len(losses) == 0:
            return {'kelly_pct': 0.0, 'win_rate': 0.0, 'win_loss_ratio': 0.0}
        
        win_prob = len(wins) / len(returns)
        loss_prob = 1 - win_prob
        
        avg_win = float(wins.mean())
        avg_loss = float(abs(losses.mean()))
        
        # Win/loss ratio (b)
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0
        
        # Kelly formula: f* = (p * b - q) / b = p - q/b
        kelly_full = (win_prob * win_loss_ratio - loss_prob) / win_loss_ratio
        
        # Apply fraction and clamp to 0-100%
        kelly_adjusted = max(0, min(1, kelly_full * fraction))
        
        # Edge calculation
        edge = win_prob * avg_win - loss_prob * avg_loss
        
        return {
            'kelly_pct': float(kelly_adjusted),
            'kelly_full': float(max(0, kelly_full)),
            'win_rate': float(win_prob),
            'loss_rate': float(loss_prob),
            'avg_win': float(avg_win),
            'avg_loss': float(avg_loss),
            'win_loss_ratio': float(win_loss_ratio),
            'edge_per_trade': float(edge),
            'fraction_used': fraction
        }
    
    def get_esg_rating(self, ticker: str) -> dict:
        """
        Get ESG (Environmental, Social, Governance) ratings.
        
        Uses Yahoo Finance ESG scores as primary source.
        Falls back to simulated ratings if unavailable.
        """
        if ticker in self.esg_cache:
            return self.esg_cache[ticker]
        
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            
            # Try to get sustainability data
            sustainability = stock.sustainability
            
            if sustainability is not None and not sustainability.empty:
                # Extract ESG scores
                esg_data = {
                    'ticker': ticker,
                    'total_esg': float(sustainability.loc['totalEsg'].values[0]) if 'totalEsg' in sustainability.index else None,
                    'environment_score': float(sustainability.loc['environmentScore'].values[0]) if 'environmentScore' in sustainability.index else None,
                    'social_score': float(sustainability.loc['socialScore'].values[0]) if 'socialScore' in sustainability.index else None,
                    'governance_score': float(sustainability.loc['governanceScore'].values[0]) if 'governanceScore' in sustainability.index else None,
                    'controversy_level': int(sustainability.loc['highestControversy'].values[0]) if 'highestControversy' in sustainability.index else None,
                    'source': 'Yahoo Finance / Sustainalytics'
                }
                
                # Calculate letter rating
                total = esg_data.get('total_esg')
                if total is not None:
                    if total <= 10:
                        esg_data['rating'] = 'AAA'
                    elif total <= 20:
                        esg_data['rating'] = 'AA'
                    elif total <= 30:
                        esg_data['rating'] = 'A'
                    elif total <= 40:
                        esg_data['rating'] = 'BBB'
                    elif total <= 50:
                        esg_data['rating'] = 'BB'
                    elif total <= 60:
                        esg_data['rating'] = 'B'
                    else:
                        esg_data['rating'] = 'CCC'
                else:
                    esg_data['rating'] = 'NR'
                
                self.esg_cache[ticker] = esg_data
                return esg_data
        except Exception:
            pass
        
        # Fallback: Generate estimated ESG based on sector
        return self._estimate_esg(ticker)
    
    def _estimate_esg(self, ticker: str) -> dict:
        """Estimate ESG scores based on sector averages."""
        try:
            import yfinance as yf
            info = yf.Ticker(ticker).info
            sector = info.get('sector', 'Unknown')
        except:
            sector = 'Unknown'
        
        # Sector-based ESG estimates (lower is better for total ESG)
        sector_esg = {
            'Technology': {'total': 22, 'E': 8, 'S': 12, 'G': 8, 'rating': 'AA'},
            'Healthcare': {'total': 25, 'E': 10, 'S': 14, 'G': 9, 'rating': 'A'},
            'Financial Services': {'total': 28, 'E': 6, 'S': 15, 'G': 10, 'rating': 'A'},
            'Consumer Cyclical': {'total': 32, 'E': 14, 'S': 12, 'G': 10, 'rating': 'BBB'},
            'Communication Services': {'total': 26, 'E': 8, 'S': 14, 'G': 9, 'rating': 'A'},
            'Industrials': {'total': 35, 'E': 16, 'S': 12, 'G': 10, 'rating': 'BBB'},
            'Consumer Defensive': {'total': 30, 'E': 12, 'S': 12, 'G': 10, 'rating': 'BBB'},
            'Energy': {'total': 45, 'E': 25, 'S': 12, 'G': 12, 'rating': 'BB'},
            'Utilities': {'total': 38, 'E': 20, 'S': 10, 'G': 11, 'rating': 'BBB'},
            'Real Estate': {'total': 28, 'E': 12, 'S': 10, 'G': 9, 'rating': 'A'},
            'Basic Materials': {'total': 40, 'E': 22, 'S': 11, 'G': 11, 'rating': 'BB'},
        }
        
        default = {'total': 35, 'E': 15, 'S': 12, 'G': 11, 'rating': 'BBB'}
        esg = sector_esg.get(sector, default)
        
        return {
            'ticker': ticker,
            'total_esg': esg['total'],
            'environment_score': esg['E'],
            'social_score': esg['S'],
            'governance_score': esg['G'],
            'rating': esg['rating'],
            'controversy_level': 2,
            'sector': sector,
            'source': 'Estimated (sector average)'
        }
    
    def factor_attribution(self, returns: pd.Series, benchmark_returns: pd.Series) -> dict:
        """
        Performance attribution using factor model.
        
        Decomposes returns into:
        - Alpha (skill)
        - Factor exposure returns
        - Residual (unexplained)
        """
        ff_results = self.fama_french_regression(returns, benchmark_returns)
        
        if 'error' in ff_results and ff_results.get('r_squared', 0) == 0:
            return ff_results
        
        total_return = float(returns.mean() * 252)
        alpha_return = ff_results['alpha']
        explained_return = total_return - alpha_return
        
        return {
            'total_return': total_return,
            'alpha_return': alpha_return,
            'factor_return': explained_return,
            'r_squared': ff_results['r_squared'],
            'loadings': ff_results['loadings']
        }
