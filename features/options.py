"""
Options Analytics - Black-Scholes & Greeks
==========================================
Pricing models and risk metrics for options.
"""

import numpy as np
from scipy.stats import norm
from typing import Dict, List, Optional, Tuple
import logging


class OptionsAnalytics:
    """
    Options pricing and Greeks calculations.
    
    Features:
    - Black-Scholes pricing
    - Full Greeks suite (Delta, Gamma, Theta, Vega, Rho)
    - Implied volatility calculation
    - Options strategies analysis
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
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
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type.lower() == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        return price
    
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
            # At expiration - handle edge case
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
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Delta
        if option_type.lower() == 'call':
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1  # or -norm.cdf(-d1)
        
        # Gamma (same for call and put)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        # Theta
        if option_type.lower() == 'call':
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                    - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        else:
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                    + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        
        # Vega (same for call and put, expressed per 1% change in vol)
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100
        
        # Rho
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
        
        sigma = 0.3  # Initial guess (30% vol)
        
        for i in range(max_iterations):
            price = OptionsAnalytics.black_scholes(S, K, T, r, sigma, option_type)
            vega = OptionsAnalytics.calculate_greeks(S, K, T, r, sigma, option_type)['vega']
            
            diff = market_price - price
            
            if abs(diff) < precision:
                return sigma
            
            if abs(vega * 100) < 1e-10:
                # Vega too small, can't continue
                break
            
            sigma = sigma + diff / (vega * 100)
            
            # Keep sigma reasonable
            sigma = max(0.01, min(5.0, sigma))
        
        return sigma
    
    def analyze_option(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = 'call',
        position_size: int = 1
    ) -> Dict:
        """
        Complete option analysis.
        
        Returns:
            Dictionary with price, Greeks, breakeven, max profit/loss
        """
        price = self.black_scholes(S, K, T, r, sigma, option_type)
        greeks = self.calculate_greeks(S, K, T, r, sigma, option_type)
        
        premium = price * 100 * position_size  # Standard 100 shares per contract
        
        if option_type.lower() == 'call':
            breakeven = K + price
            max_loss = premium
            max_profit = float('inf')
            intrinsic = max(0, S - K)
        else:
            breakeven = K - price
            max_loss = premium
            max_profit = (K - price) * 100 * position_size
            intrinsic = max(0, K - S)
        
        extrinsic = price - intrinsic
        
        return {
            'price': price,
            'premium': premium,
            'greeks': greeks,
            'breakeven': breakeven,
            'max_loss': max_loss,
            'max_profit': max_profit if max_profit != float('inf') else 'Unlimited',
            'intrinsic_value': intrinsic,
            'extrinsic_value': extrinsic,
            'moneyness': 'ITM' if intrinsic > 0 else 'ATM' if abs(S - K) / S < 0.02 else 'OTM',
            'days_to_expiry': T * 365
        }
    
    def calculate_payoff_diagram(
        self,
        S: float,
        K: float,
        premium: float,
        option_type: str = 'call',
        is_long: bool = True,
        price_range: float = 0.3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate payoff diagram data.
        
        Args:
            S: Current stock price
            K: Strike price
            premium: Option premium paid/received
            option_type: 'call' or 'put'
            is_long: True for long position, False for short
            price_range: Range of prices as fraction of S
        
        Returns:
            Tuple of (price_array, payoff_array)
        """
        prices = np.linspace(S * (1 - price_range), S * (1 + price_range), 100)
        
        if option_type.lower() == 'call':
            payoffs = np.maximum(prices - K, 0) - premium
        else:
            payoffs = np.maximum(K - prices, 0) - premium
        
        if not is_long:
            payoffs = -payoffs
        
        return prices, payoffs * 100  # Per contract
    
    def covered_call_analysis(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        shares: int = 100
    ) -> Dict:
        """
        Analyze covered call strategy (long stock + short call).
        
        Returns:
            Strategy metrics including max profit, breakeven, etc.
        """
        call_price = self.black_scholes(S, K, T, r, sigma, 'call')
        premium_received = call_price * shares
        
        # Position cost
        stock_cost = S * shares
        net_cost = stock_cost - premium_received
        
        # Max profit (if assigned)
        max_profit = (K - S) * shares + premium_received
        
        # Breakeven
        breakeven = S - call_price
        
        # Max loss (stock goes to 0)
        max_loss = net_cost
        
        return {
            'stock_cost': stock_cost,
            'premium_received': premium_received,
            'net_cost': net_cost,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'breakeven': breakeven,
            'return_if_flat': premium_received / stock_cost * (365 / (T * 365)),
            'return_if_called': max_profit / stock_cost * (365 / (T * 365))
        }
    
    def protective_put_analysis(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        shares: int = 100
    ) -> Dict:
        """
        Analyze protective put strategy (long stock + long put).
        
        Returns:
            Strategy metrics including max loss, breakeven, etc.
        """
        put_price = self.black_scholes(S, K, T, r, sigma, 'put')
        premium_paid = put_price * shares
        
        stock_cost = S * shares
        total_cost = stock_cost + premium_paid
        
        # Max loss (stock below strike, put protects)
        max_loss = (S - K) * shares + premium_paid
        
        # Breakeven
        breakeven = S + put_price
        
        return {
            'stock_cost': stock_cost,
            'premium_paid': premium_paid,
            'total_cost': total_cost,
            'max_loss': max_loss,
            'max_profit': 'Unlimited',
            'breakeven': breakeven,
            'protection_level': K,
            'cost_of_protection': premium_paid / stock_cost
        }
    
    def straddle_analysis(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float
    ) -> Dict:
        """
        Analyze long straddle (long call + long put at same strike).
        
        Returns:
            Strategy metrics for betting on volatility
        """
        call_price = self.black_scholes(S, K, T, r, sigma, 'call')
        put_price = self.black_scholes(S, K, T, r, sigma, 'put')
        
        total_premium = call_price + put_price
        
        # Breakevens
        upper_breakeven = K + total_premium
        lower_breakeven = K - total_premium
        
        return {
            'call_price': call_price,
            'put_price': put_price,
            'total_premium': total_premium,
            'total_cost': total_premium * 100,
            'upper_breakeven': upper_breakeven,
            'lower_breakeven': lower_breakeven,
            'max_loss': total_premium * 100,
            'max_profit': 'Unlimited',
            'required_move': total_premium / S,  # % move needed to breakeven
            'greeks': {
                'delta': 0,  # Delta-neutral at the money
                'gamma': self.calculate_greeks(S, K, T, r, sigma, 'call')['gamma'] * 2,
                'vega': self.calculate_greeks(S, K, T, r, sigma, 'call')['vega'] * 2
            }
        }
