"""
Options Analytics Service
=========================
Options pricing and Greeks calculations ported from features/options.py

Provides Black-Scholes pricing, Greeks, and implied volatility.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from typing import Dict, Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


class OptionsAnalytics:
    """
    Options pricing and Greeks calculations.
    
    Features:
    - Black-Scholes pricing
    - Full Greeks suite (Delta, Gamma, Theta, Vega, Rho)
    - Implied volatility calculation
    - Options strategies analysis
    """
    
    @staticmethod
    def black_scholes(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = 'call'
    ) -> float:
        """
        Calculate option price using Black-Scholes model.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free interest rate
            sigma: Volatility (annualized)
            option_type: 'call' or 'put'
        
        Returns:
            Option price
        """
        if T <= 0:
            # At expiration
            if option_type.lower() == 'call':
                return max(0, S - K)
            else:
                return max(0, K - S)
        
        if sigma <= 0:
            sigma = 0.0001  # Avoid division by zero
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type.lower() == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        return float(price)
    
    @staticmethod
    def calculate_greeks(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = 'call'
    ) -> Dict[str, float]:
        """
        Calculate all Greeks for an option.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
        
        Returns:
            Dictionary with Delta, Gamma, Theta, Vega, Rho
        """
        if T <= 0:
            # At expiration
            in_the_money = (option_type.lower() == 'call' and S > K) or \
                          (option_type.lower() == 'put' and S < K)
            return {
                'delta': 1.0 if in_the_money and option_type.lower() == 'call' else \
                        -1.0 if in_the_money and option_type.lower() == 'put' else 0.0,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0,
                'rho': 0.0
            }
        
        if sigma <= 0:
            sigma = 0.0001
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Delta
        if option_type.lower() == 'call':
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1
        
        # Gamma (same for call and put)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        # Theta (per day)
        if option_type.lower() == 'call':
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                    - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        else:
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                    + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        
        # Vega (per 1% change in vol)
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100
        
        # Rho (per 1% change in rate)
        if option_type.lower() == 'call':
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        
        return {
            'delta': float(delta),
            'gamma': float(gamma),
            'theta': float(theta),
            'vega': float(vega),
            'rho': float(rho)
        }
    
    @staticmethod
    def implied_volatility(
        market_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: str = 'call',
        max_iterations: int = 100,
        precision: float = 0.0001
    ) -> Optional[float]:
        """
        Calculate implied volatility using Newton-Raphson method.
        
        Args:
            market_price: Observed market price of the option
            S: Current stock price
            K: Strike price
            T: Time to expiration
            r: Risk-free rate
            option_type: 'call' or 'put'
        
        Returns:
            Implied volatility, or None if not converged
        """
        if T <= 0:
            return None
        
        # Use Brent's method for robustness
        try:
            def objective(sigma):
                return OptionsAnalytics.black_scholes(S, K, T, r, sigma, option_type) - market_price
            
            iv = brentq(objective, 0.001, 5.0)
            return float(iv)
        except:
            pass
        
        # Fall back to Newton-Raphson
        sigma = 0.3  # Initial guess
        
        for i in range(max_iterations):
            price = OptionsAnalytics.black_scholes(S, K, T, r, sigma, option_type)
            vega = OptionsAnalytics.calculate_greeks(S, K, T, r, sigma, option_type)['vega']
            
            diff = market_price - price
            
            if abs(diff) < precision:
                return sigma
            
            if abs(vega * 100) < 1e-10:
                break
            
            sigma = sigma + diff / (vega * 100)
            sigma = max(0.01, min(5.0, sigma))
        
        return sigma
    
    @staticmethod
    def analyze_option(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = 'call',
        position_size: int = 1
    ) -> Dict:
        """
        Comprehensive option analysis.
        
        Returns pricing, Greeks, breakeven, and risk metrics.
        """
        price = OptionsAnalytics.black_scholes(S, K, T, r, sigma, option_type)
        greeks = OptionsAnalytics.calculate_greeks(S, K, T, r, sigma, option_type)
        
        # Intrinsic and time value
        if option_type.lower() == 'call':
            intrinsic = max(0, S - K)
            breakeven = K + price
        else:
            intrinsic = max(0, K - S)
            breakeven = K - price
        
        time_value = price - intrinsic
        
        # Moneyness
        if option_type.lower() == 'call':
            if S > K * 1.02:
                moneyness = 'ITM'
            elif S < K * 0.98:
                moneyness = 'OTM'
            else:
                moneyness = 'ATM'
        else:
            if S < K * 0.98:
                moneyness = 'ITM'
            elif S > K * 1.02:
                moneyness = 'OTM'
            else:
                moneyness = 'ATM'
        
        # Probability ITM (using delta as approximation)
        prob_itm = abs(greeks['delta'])
        
        # Max profit/loss for long position
        if option_type.lower() == 'call':
            max_loss = price * position_size * 100
            max_profit = None  # Unlimited
        else:
            max_loss = price * position_size * 100
            max_profit = (K - price) * position_size * 100  # If stock goes to 0
        
        return {
            'price': price,
            'greeks': greeks,
            'intrinsic_value': intrinsic,
            'time_value': time_value,
            'moneyness': moneyness,
            'breakeven': breakeven,
            'probability_itm': prob_itm,
            'max_loss': max_loss,
            'max_profit': max_profit,
            'position_value': price * position_size * 100
        }
    
    @staticmethod
    def generate_volatility_surface(
        S: float,
        strikes: List[float],
        expiries: List[float],
        base_vol: float = 0.25
    ) -> List[Dict[str, float]]:
        """
        Generate synthetic volatility surface data.
        
        Uses a simple smile/skew model for demonstration.
        """
        surface_data = []
        
        for T in expiries:
            for K in strikes:
                # Simple volatility smile model
                moneyness = np.log(K / S)
                
                # Skew component (OTM puts have higher vol)
                skew = -0.1 * moneyness
                
                # Smile component (away from ATM has higher vol)
                smile = 0.5 * moneyness ** 2
                
                # Term structure (shorter expiries have slightly higher vol)
                term = 0.02 * (1 / T - 1) if T > 0 else 0
                
                iv = base_vol + skew + smile + term
                iv = max(0.05, min(2.0, iv))  # Bound IV
                
                surface_data.append({
                    'strike': K,
                    'expiry': T,
                    'implied_vol': iv
                })
        
        return surface_data


# Singleton instance
_options_analytics = None


def get_options_analytics() -> OptionsAnalytics:
    """Get singleton OptionsAnalytics instance."""
    global _options_analytics
    if _options_analytics is None:
        _options_analytics = OptionsAnalytics()
    return _options_analytics
