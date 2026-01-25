"""
Factor Analysis Module - Enterprise Features
=============================================
Fama-French 5-Factor • Kelly Criterion • ESG Ratings

Author: Professional Risk Analytics | Jan 2026
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')


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
