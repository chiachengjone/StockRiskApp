"""
Strategy Builder Service
=========================
Build and manage custom trading strategies

Provides:
- Visual strategy designer
- Condition-based rule engine
- Strategy backtesting
- Strategy persistence

Author: Stock Risk App | Feb 2026
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json
import os
import logging

from .ta_service import (
    calculate_sma, calculate_ema, calculate_rsi, calculate_macd,
    calculate_bollinger_bands, calculate_adx, calculate_stochastic, calculate_atr
)

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class ConditionOperator(Enum):
    """Comparison operators for conditions."""
    GREATER_THAN = ">"
    LESS_THAN = "<"
    GREATER_EQUAL = ">="
    LESS_EQUAL = "<="
    EQUALS = "=="
    NOT_EQUALS = "!="
    CROSSES_ABOVE = "crosses_above"
    CROSSES_BELOW = "crosses_below"


class LogicOperator(Enum):
    """Logic operators for combining conditions."""
    AND = "AND"
    OR = "OR"


class ActionType(Enum):
    """Action types for strategy rules."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE_LONG = "CLOSE_LONG"
    CLOSE_SHORT = "CLOSE_SHORT"


class IndicatorType(Enum):
    """Available indicators for conditions."""
    PRICE = "Price"
    SMA = "SMA"
    EMA = "EMA"
    RSI = "RSI"
    MACD = "MACD"
    MACD_SIGNAL = "MACD Signal"
    MACD_HISTOGRAM = "MACD Histogram"
    BOLLINGER_UPPER = "Bollinger Upper"
    BOLLINGER_MIDDLE = "Bollinger Middle"
    BOLLINGER_LOWER = "Bollinger Lower"
    ADX = "ADX"
    PLUS_DI = "+DI"
    MINUS_DI = "-DI"
    STOCH_K = "Stochastic %K"
    STOCH_D = "Stochastic %D"
    ATR = "ATR"
    VOLUME = "Volume"
    FIXED_VALUE = "Fixed Value"


@dataclass
class Condition:
    """A single condition in a strategy rule."""
    indicator: IndicatorType
    indicator_params: Dict[str, Any]  # e.g., {"period": 14} for RSI
    operator: ConditionOperator
    compare_to: IndicatorType
    compare_params: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'indicator': self.indicator.value,
            'indicator_params': self.indicator_params,
            'operator': self.operator.value,
            'compare_to': self.compare_to.value,
            'compare_params': self.compare_params
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Condition':
        return cls(
            indicator=IndicatorType(data['indicator']),
            indicator_params=data.get('indicator_params', {}),
            operator=ConditionOperator(data['operator']),
            compare_to=IndicatorType(data['compare_to']),
            compare_params=data.get('compare_params', {})
        )


@dataclass
class Rule:
    """A complete trading rule with conditions and action."""
    name: str
    conditions: List[Condition]
    logic: LogicOperator  # How to combine conditions
    action: ActionType
    priority: int = 1  # Higher priority rules execute first
    enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'conditions': [c.to_dict() for c in self.conditions],
            'logic': self.logic.value,
            'action': self.action.value,
            'priority': self.priority,
            'enabled': self.enabled
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Rule':
        conditions = [Condition.from_dict(c) for c in data.get('conditions', [])]
        return cls(
            name=data['name'],
            conditions=conditions,
            logic=LogicOperator(data.get('logic', 'AND')),
            action=ActionType(data['action']),
            priority=data.get('priority', 1),
            enabled=data.get('enabled', True)
        )


@dataclass
class Strategy:
    """A complete trading strategy."""
    name: str
    description: str
    rules: List[Rule]
    risk_params: Dict[str, float]  # stop_loss, take_profit, position_size
    created_at: datetime
    updated_at: datetime
    version: str = "1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'description': self.description,
            'rules': [r.to_dict() for r in self.rules],
            'risk_params': self.risk_params,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'version': self.version
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Strategy':
        rules = [Rule.from_dict(r) for r in data.get('rules', [])]
        return cls(
            name=data['name'],
            description=data.get('description', ''),
            rules=rules,
            risk_params=data.get('risk_params', {'stop_loss': 0.02, 'take_profit': 0.05, 'position_size': 0.1}),
            created_at=datetime.fromisoformat(data['created_at']) if isinstance(data['created_at'], str) else data['created_at'],
            updated_at=datetime.fromisoformat(data['updated_at']) if isinstance(data['updated_at'], str) else data['updated_at'],
            version=data.get('version', '1.0')
        )


@dataclass
class StrategySignal:
    """Signal generated by strategy evaluation."""
    rule_name: str
    action: ActionType
    confidence: float
    conditions_met: List[str]
    timestamp: datetime
    price: float
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# INDICATOR CALCULATION
# ============================================================================

def get_indicator_value(
    df: pd.DataFrame,
    indicator_type: IndicatorType,
    params: Dict[str, Any],
    idx: int = -1
) -> Optional[float]:
    """
    Get indicator value at specified index.
    
    Args:
        df: OHLCV DataFrame
        indicator_type: Type of indicator
        params: Indicator parameters
        idx: Index to get value at
        
    Returns:
        Indicator value or None
    """
    if isinstance(df.columns, pd.MultiIndex):
        close = df['Close'].iloc[:, 0]
        high = df['High'].iloc[:, 0]
        low = df['Low'].iloc[:, 0]
        volume = df['Volume'].iloc[:, 0] if 'Volume' in df.columns else pd.Series(index=df.index, dtype=float)
    else:
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume'] if 'Volume' in df.columns else pd.Series(index=df.index, dtype=float)
    
    try:
        if indicator_type == IndicatorType.PRICE:
            return float(close.iloc[idx])
        
        elif indicator_type == IndicatorType.FIXED_VALUE:
            return float(params.get('value', 0))
        
        elif indicator_type == IndicatorType.SMA:
            period = params.get('period', 20)
            sma = calculate_sma(close, period)
            return float(sma.iloc[idx]) if not pd.isna(sma.iloc[idx]) else None
        
        elif indicator_type == IndicatorType.EMA:
            period = params.get('period', 20)
            ema = calculate_ema(close, period)
            return float(ema.iloc[idx]) if not pd.isna(ema.iloc[idx]) else None
        
        elif indicator_type == IndicatorType.RSI:
            period = params.get('period', 14)
            rsi = calculate_rsi(close, period)
            return float(rsi.iloc[idx]) if not pd.isna(rsi.iloc[idx]) else None
        
        elif indicator_type == IndicatorType.MACD:
            macd_line, _, _ = calculate_macd(close)
            return float(macd_line.iloc[idx]) if not pd.isna(macd_line.iloc[idx]) else None
        
        elif indicator_type == IndicatorType.MACD_SIGNAL:
            _, signal_line, _ = calculate_macd(close)
            return float(signal_line.iloc[idx]) if not pd.isna(signal_line.iloc[idx]) else None
        
        elif indicator_type == IndicatorType.MACD_HISTOGRAM:
            _, _, histogram = calculate_macd(close)
            return float(histogram.iloc[idx]) if not pd.isna(histogram.iloc[idx]) else None
        
        elif indicator_type == IndicatorType.BOLLINGER_UPPER:
            period = params.get('period', 20)
            std = params.get('std', 2)
            upper, _, _ = calculate_bollinger_bands(close, period, std)
            return float(upper.iloc[idx]) if not pd.isna(upper.iloc[idx]) else None
        
        elif indicator_type == IndicatorType.BOLLINGER_MIDDLE:
            period = params.get('period', 20)
            std = params.get('std', 2)
            _, middle, _ = calculate_bollinger_bands(close, period, std)
            return float(middle.iloc[idx]) if not pd.isna(middle.iloc[idx]) else None
        
        elif indicator_type == IndicatorType.BOLLINGER_LOWER:
            period = params.get('period', 20)
            std = params.get('std', 2)
            _, _, lower = calculate_bollinger_bands(close, period, std)
            return float(lower.iloc[idx]) if not pd.isna(lower.iloc[idx]) else None
        
        elif indicator_type == IndicatorType.ADX:
            period = params.get('period', 14)
            adx, _, _ = calculate_adx(high, low, close, period)
            return float(adx.iloc[idx]) if not pd.isna(adx.iloc[idx]) else None
        
        elif indicator_type == IndicatorType.PLUS_DI:
            period = params.get('period', 14)
            _, plus_di, _ = calculate_adx(high, low, close, period)
            return float(plus_di.iloc[idx]) if not pd.isna(plus_di.iloc[idx]) else None
        
        elif indicator_type == IndicatorType.MINUS_DI:
            period = params.get('period', 14)
            _, _, minus_di = calculate_adx(high, low, close, period)
            return float(minus_di.iloc[idx]) if not pd.isna(minus_di.iloc[idx]) else None
        
        elif indicator_type == IndicatorType.STOCH_K:
            k_period = params.get('k_period', 14)
            d_period = params.get('d_period', 3)
            stoch_k, _ = calculate_stochastic(high, low, close, k_period, d_period)
            return float(stoch_k.iloc[idx]) if not pd.isna(stoch_k.iloc[idx]) else None
        
        elif indicator_type == IndicatorType.STOCH_D:
            k_period = params.get('k_period', 14)
            d_period = params.get('d_period', 3)
            _, stoch_d = calculate_stochastic(high, low, close, k_period, d_period)
            return float(stoch_d.iloc[idx]) if not pd.isna(stoch_d.iloc[idx]) else None
        
        elif indicator_type == IndicatorType.ATR:
            period = params.get('period', 14)
            atr = calculate_atr(high, low, close, period)
            return float(atr.iloc[idx]) if not pd.isna(atr.iloc[idx]) else None
        
        elif indicator_type == IndicatorType.VOLUME:
            return float(volume.iloc[idx]) if not pd.isna(volume.iloc[idx]) else None
        
        else:
            logger.warning(f"Unknown indicator type: {indicator_type}")
            return None
            
    except Exception as e:
        logger.error(f"Error calculating indicator {indicator_type}: {e}")
        return None


def check_crossover(
    df: pd.DataFrame,
    indicator1: IndicatorType,
    params1: Dict[str, Any],
    indicator2: IndicatorType,
    params2: Dict[str, Any],
    direction: str = "above"  # "above" or "below"
) -> bool:
    """
    Check if indicator1 crosses indicator2.
    
    Args:
        df: OHLCV DataFrame
        indicator1, indicator2: Indicators to compare
        params1, params2: Indicator parameters
        direction: "above" for crosses above, "below" for crosses below
        
    Returns:
        True if crossover detected
    """
    curr_val1 = get_indicator_value(df, indicator1, params1, -1)
    prev_val1 = get_indicator_value(df, indicator1, params1, -2)
    curr_val2 = get_indicator_value(df, indicator2, params2, -1)
    prev_val2 = get_indicator_value(df, indicator2, params2, -2)
    
    if any(v is None for v in [curr_val1, prev_val1, curr_val2, prev_val2]):
        return False
    
    if direction == "above":
        return prev_val1 <= prev_val2 and curr_val1 > curr_val2
    else:  # below
        return prev_val1 >= prev_val2 and curr_val1 < curr_val2


# ============================================================================
# PRE-BUILT STRATEGY TEMPLATES
# ============================================================================

STRATEGY_TEMPLATES = {
    "golden_cross": Strategy(
        name="Golden Cross",
        description="Buy when SMA 50 crosses above SMA 200, sell when crosses below",
        rules=[
            Rule(
                name="Golden Cross Buy",
                conditions=[
                    Condition(
                        indicator=IndicatorType.SMA,
                        indicator_params={"period": 50},
                        operator=ConditionOperator.CROSSES_ABOVE,
                        compare_to=IndicatorType.SMA,
                        compare_params={"period": 200}
                    )
                ],
                logic=LogicOperator.AND,
                action=ActionType.BUY,
                priority=1
            ),
            Rule(
                name="Death Cross Sell",
                conditions=[
                    Condition(
                        indicator=IndicatorType.SMA,
                        indicator_params={"period": 50},
                        operator=ConditionOperator.CROSSES_BELOW,
                        compare_to=IndicatorType.SMA,
                        compare_params={"period": 200}
                    )
                ],
                logic=LogicOperator.AND,
                action=ActionType.SELL,
                priority=1
            )
        ],
        risk_params={"stop_loss": 0.05, "take_profit": 0.15, "position_size": 0.2},
        created_at=datetime.now(),
        updated_at=datetime.now()
    ),
    
    "rsi_mean_reversion": Strategy(
        name="RSI Mean Reversion",
        description="Buy oversold, sell overbought based on RSI",
        rules=[
            Rule(
                name="RSI Oversold Buy",
                conditions=[
                    Condition(
                        indicator=IndicatorType.RSI,
                        indicator_params={"period": 14},
                        operator=ConditionOperator.LESS_THAN,
                        compare_to=IndicatorType.FIXED_VALUE,
                        compare_params={"value": 30}
                    )
                ],
                logic=LogicOperator.AND,
                action=ActionType.BUY,
                priority=1
            ),
            Rule(
                name="RSI Overbought Sell",
                conditions=[
                    Condition(
                        indicator=IndicatorType.RSI,
                        indicator_params={"period": 14},
                        operator=ConditionOperator.GREATER_THAN,
                        compare_to=IndicatorType.FIXED_VALUE,
                        compare_params={"value": 70}
                    )
                ],
                logic=LogicOperator.AND,
                action=ActionType.SELL,
                priority=1
            )
        ],
        risk_params={"stop_loss": 0.03, "take_profit": 0.08, "position_size": 0.1},
        created_at=datetime.now(),
        updated_at=datetime.now()
    ),
    
    "macd_crossover": Strategy(
        name="MACD Crossover",
        description="Trade MACD signal line crossovers",
        rules=[
            Rule(
                name="MACD Bullish Crossover",
                conditions=[
                    Condition(
                        indicator=IndicatorType.MACD,
                        indicator_params={},
                        operator=ConditionOperator.CROSSES_ABOVE,
                        compare_to=IndicatorType.MACD_SIGNAL,
                        compare_params={}
                    )
                ],
                logic=LogicOperator.AND,
                action=ActionType.BUY,
                priority=1
            ),
            Rule(
                name="MACD Bearish Crossover",
                conditions=[
                    Condition(
                        indicator=IndicatorType.MACD,
                        indicator_params={},
                        operator=ConditionOperator.CROSSES_BELOW,
                        compare_to=IndicatorType.MACD_SIGNAL,
                        compare_params={}
                    )
                ],
                logic=LogicOperator.AND,
                action=ActionType.SELL,
                priority=1
            )
        ],
        risk_params={"stop_loss": 0.03, "take_profit": 0.06, "position_size": 0.15},
        created_at=datetime.now(),
        updated_at=datetime.now()
    ),
    
    "bollinger_breakout": Strategy(
        name="Bollinger Band Breakout",
        description="Trade breakouts from Bollinger Bands",
        rules=[
            Rule(
                name="Lower Band Bounce",
                conditions=[
                    Condition(
                        indicator=IndicatorType.PRICE,
                        indicator_params={},
                        operator=ConditionOperator.LESS_EQUAL,
                        compare_to=IndicatorType.BOLLINGER_LOWER,
                        compare_params={"period": 20, "std": 2}
                    ),
                    Condition(
                        indicator=IndicatorType.RSI,
                        indicator_params={"period": 14},
                        operator=ConditionOperator.LESS_THAN,
                        compare_to=IndicatorType.FIXED_VALUE,
                        compare_params={"value": 40}
                    )
                ],
                logic=LogicOperator.AND,
                action=ActionType.BUY,
                priority=1
            ),
            Rule(
                name="Upper Band Rejection",
                conditions=[
                    Condition(
                        indicator=IndicatorType.PRICE,
                        indicator_params={},
                        operator=ConditionOperator.GREATER_EQUAL,
                        compare_to=IndicatorType.BOLLINGER_UPPER,
                        compare_params={"period": 20, "std": 2}
                    ),
                    Condition(
                        indicator=IndicatorType.RSI,
                        indicator_params={"period": 14},
                        operator=ConditionOperator.GREATER_THAN,
                        compare_to=IndicatorType.FIXED_VALUE,
                        compare_params={"value": 60}
                    )
                ],
                logic=LogicOperator.AND,
                action=ActionType.SELL,
                priority=1
            )
        ],
        risk_params={"stop_loss": 0.02, "take_profit": 0.04, "position_size": 0.1},
        created_at=datetime.now(),
        updated_at=datetime.now()
    ),
    
    "trend_following_adx": Strategy(
        name="ADX Trend Following",
        description="Trade with trend when ADX shows strong trend",
        rules=[
            Rule(
                name="Strong Uptrend Buy",
                conditions=[
                    Condition(
                        indicator=IndicatorType.ADX,
                        indicator_params={"period": 14},
                        operator=ConditionOperator.GREATER_THAN,
                        compare_to=IndicatorType.FIXED_VALUE,
                        compare_params={"value": 25}
                    ),
                    Condition(
                        indicator=IndicatorType.PLUS_DI,
                        indicator_params={"period": 14},
                        operator=ConditionOperator.GREATER_THAN,
                        compare_to=IndicatorType.MINUS_DI,
                        compare_params={"period": 14}
                    )
                ],
                logic=LogicOperator.AND,
                action=ActionType.BUY,
                priority=1
            ),
            Rule(
                name="Strong Downtrend Sell",
                conditions=[
                    Condition(
                        indicator=IndicatorType.ADX,
                        indicator_params={"period": 14},
                        operator=ConditionOperator.GREATER_THAN,
                        compare_to=IndicatorType.FIXED_VALUE,
                        compare_params={"value": 25}
                    ),
                    Condition(
                        indicator=IndicatorType.MINUS_DI,
                        indicator_params={"period": 14},
                        operator=ConditionOperator.GREATER_THAN,
                        compare_to=IndicatorType.PLUS_DI,
                        compare_params={"period": 14}
                    )
                ],
                logic=LogicOperator.AND,
                action=ActionType.SELL,
                priority=1
            )
        ],
        risk_params={"stop_loss": 0.04, "take_profit": 0.10, "position_size": 0.15},
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
}


# ============================================================================
# STRATEGY SERVICE CLASS
# ============================================================================

class StrategyService:
    """
    Strategy Builder Service.
    
    Provides strategy creation, management, and evaluation.
    """
    
    def __init__(self, storage_path: str = "data/strategies"):
        """
        Initialize the Strategy Service.
        
        Args:
            storage_path: Path to store saved strategies
        """
        self.storage_path = storage_path
        self._strategies: Dict[str, Strategy] = {}
        self._load_templates()
        self._load_saved_strategies()
        logger.info("StrategyService initialized")
    
    def _load_templates(self) -> None:
        """Load built-in strategy templates."""
        for name, strategy in STRATEGY_TEMPLATES.items():
            self._strategies[f"template_{name}"] = strategy
    
    def _load_saved_strategies(self) -> None:
        """Load saved strategies from storage."""
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path, exist_ok=True)
            return
        
        for filename in os.listdir(self.storage_path):
            if filename.endswith('.json'):
                try:
                    with open(os.path.join(self.storage_path, filename), 'r') as f:
                        data = json.load(f)
                        strategy = Strategy.from_dict(data)
                        self._strategies[strategy.name] = strategy
                except Exception as e:
                    logger.error(f"Error loading strategy {filename}: {e}")
    
    def create_strategy(
        self,
        name: str,
        description: str,
        rules: List[Rule],
        risk_params: Optional[Dict[str, float]] = None
    ) -> Strategy:
        """
        Create a new strategy.
        
        Args:
            name: Strategy name
            description: Strategy description
            rules: List of trading rules
            risk_params: Risk parameters
            
        Returns:
            Created Strategy
        """
        now = datetime.now()
        strategy = Strategy(
            name=name,
            description=description,
            rules=rules,
            risk_params=risk_params or {"stop_loss": 0.02, "take_profit": 0.05, "position_size": 0.1},
            created_at=now,
            updated_at=now
        )
        
        self._strategies[name] = strategy
        return strategy
    
    def save_strategy(self, strategy: Strategy) -> bool:
        """
        Save strategy to storage.
        
        Args:
            strategy: Strategy to save
            
        Returns:
            True if saved successfully
        """
        try:
            os.makedirs(self.storage_path, exist_ok=True)
            filepath = os.path.join(self.storage_path, f"{strategy.name.replace(' ', '_')}.json")
            
            with open(filepath, 'w') as f:
                json.dump(strategy.to_dict(), f, indent=2)
            
            self._strategies[strategy.name] = strategy
            logger.info(f"Strategy '{strategy.name}' saved")
            return True
            
        except Exception as e:
            logger.error(f"Error saving strategy: {e}")
            return False
    
    def get_strategy(self, name: str) -> Optional[Strategy]:
        """Get strategy by name."""
        return self._strategies.get(name)
    
    def list_strategies(self) -> List[Dict[str, Any]]:
        """List all available strategies."""
        return [
            {
                'name': s.name,
                'description': s.description,
                'rules_count': len(s.rules),
                'is_template': name.startswith('template_'),
                'updated_at': s.updated_at.isoformat()
            }
            for name, s in self._strategies.items()
        ]
    
    def delete_strategy(self, name: str) -> bool:
        """Delete a strategy."""
        if name.startswith('template_'):
            logger.warning("Cannot delete template strategies")
            return False
        
        if name in self._strategies:
            del self._strategies[name]
            
            # Remove from storage
            filepath = os.path.join(self.storage_path, f"{name.replace(' ', '_')}.json")
            if os.path.exists(filepath):
                os.remove(filepath)
            
            return True
        return False
    
    def evaluate_condition(
        self,
        df: pd.DataFrame,
        condition: Condition
    ) -> bool:
        """
        Evaluate a single condition.
        
        Args:
            df: OHLCV DataFrame
            condition: Condition to evaluate
            
        Returns:
            True if condition is met
        """
        # Handle crossover conditions
        if condition.operator == ConditionOperator.CROSSES_ABOVE:
            return check_crossover(
                df,
                condition.indicator, condition.indicator_params,
                condition.compare_to, condition.compare_params,
                "above"
            )
        elif condition.operator == ConditionOperator.CROSSES_BELOW:
            return check_crossover(
                df,
                condition.indicator, condition.indicator_params,
                condition.compare_to, condition.compare_params,
                "below"
            )
        
        # Get current values
        val1 = get_indicator_value(df, condition.indicator, condition.indicator_params)
        val2 = get_indicator_value(df, condition.compare_to, condition.compare_params)
        
        if val1 is None or val2 is None:
            return False
        
        # Compare
        if condition.operator == ConditionOperator.GREATER_THAN:
            return val1 > val2
        elif condition.operator == ConditionOperator.LESS_THAN:
            return val1 < val2
        elif condition.operator == ConditionOperator.GREATER_EQUAL:
            return val1 >= val2
        elif condition.operator == ConditionOperator.LESS_EQUAL:
            return val1 <= val2
        elif condition.operator == ConditionOperator.EQUALS:
            return abs(val1 - val2) < 0.0001
        elif condition.operator == ConditionOperator.NOT_EQUALS:
            return abs(val1 - val2) >= 0.0001
        
        return False
    
    def evaluate_rule(
        self,
        df: pd.DataFrame,
        rule: Rule
    ) -> Tuple[bool, List[str]]:
        """
        Evaluate a trading rule.
        
        Args:
            df: OHLCV DataFrame
            rule: Rule to evaluate
            
        Returns:
            Tuple of (rule_triggered, conditions_met)
        """
        if not rule.enabled:
            return False, []
        
        conditions_met = []
        
        for i, condition in enumerate(rule.conditions):
            if self.evaluate_condition(df, condition):
                conditions_met.append(f"{condition.indicator.value} {condition.operator.value} {condition.compare_to.value}")
        
        if rule.logic == LogicOperator.AND:
            triggered = len(conditions_met) == len(rule.conditions)
        else:  # OR
            triggered = len(conditions_met) > 0
        
        return triggered, conditions_met
    
    def evaluate_strategy(
        self,
        df: pd.DataFrame,
        strategy: Strategy
    ) -> Optional[StrategySignal]:
        """
        Evaluate a strategy and generate signal.
        
        Args:
            df: OHLCV DataFrame
            strategy: Strategy to evaluate
            
        Returns:
            StrategySignal if any rule triggered, None otherwise
        """
        if len(df) < 50:
            return None
        
        # Sort rules by priority
        sorted_rules = sorted(strategy.rules, key=lambda r: r.priority, reverse=True)
        
        for rule in sorted_rules:
            triggered, conditions_met = self.evaluate_rule(df, rule)
            
            if triggered:
                if isinstance(df.columns, pd.MultiIndex):
                    price = float(df['Close'].iloc[:, 0].iloc[-1])
                else:
                    price = float(df['Close'].iloc[-1])
                
                return StrategySignal(
                    rule_name=rule.name,
                    action=rule.action,
                    confidence=80 if len(conditions_met) > 1 else 65,
                    conditions_met=conditions_met,
                    timestamp=datetime.now(),
                    price=price,
                    metadata={
                        'strategy': strategy.name,
                        'rule_priority': rule.priority
                    }
                )
        
        return None
    
    def backtest_strategy(
        self,
        df: pd.DataFrame,
        strategy: Strategy,
        initial_capital: float = 100000
    ) -> Dict[str, Any]:
        """
        Backtest a strategy on historical data.
        
        Args:
            df: OHLCV DataFrame
            strategy: Strategy to backtest
            initial_capital: Starting capital
            
        Returns:
            Backtest results dictionary
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
        
        position_size = strategy.risk_params.get('position_size', 0.1)
        stop_loss = strategy.risk_params.get('stop_loss', 0.02)
        take_profit = strategy.risk_params.get('take_profit', 0.05)
        
        for i in range(50, len(df)):
            subset = df.iloc[:i+1]
            current_price = float(close.iloc[i])
            
            # Check stop loss / take profit if in position
            if position > 0:
                pnl_pct = (current_price - entry_price) / entry_price
                
                if pnl_pct <= -stop_loss or pnl_pct >= take_profit:
                    # Exit position
                    exit_value = position * current_price
                    pnl = exit_value - (position * entry_price)
                    capital += pnl
                    
                    trades.append({
                        'type': 'SELL',
                        'date': df.index[i],
                        'price': current_price,
                        'pnl': pnl,
                        'reason': 'Stop Loss' if pnl_pct <= -stop_loss else 'Take Profit'
                    })
                    
                    position = 0
                    entry_price = 0
            
            # Evaluate strategy
            signal = self.evaluate_strategy(subset, strategy)
            
            if signal:
                if signal.action == ActionType.BUY and position == 0:
                    # Enter long
                    trade_value = capital * position_size
                    position = trade_value / current_price
                    entry_price = current_price
                    
                    trades.append({
                        'type': 'BUY',
                        'date': df.index[i],
                        'price': current_price,
                        'rule': signal.rule_name
                    })
                
                elif signal.action == ActionType.SELL and position > 0:
                    # Exit long
                    exit_value = position * current_price
                    pnl = exit_value - (position * entry_price)
                    capital += pnl
                    
                    trades.append({
                        'type': 'SELL',
                        'date': df.index[i],
                        'price': current_price,
                        'pnl': pnl,
                        'rule': signal.rule_name
                    })
                    
                    position = 0
                    entry_price = 0
            
            # Record equity
            current_equity = capital + (position * current_price if position > 0 else 0)
            equity.append(current_equity)
        
        # Close remaining position
        if position > 0:
            final_price = float(close.iloc[-1])
            pnl = (position * final_price) - (position * entry_price)
            capital += pnl
            trades.append({
                'type': 'CLOSE',
                'date': df.index[-1],
                'price': final_price,
                'pnl': pnl,
                'reason': 'End of backtest'
            })
        
        # Calculate metrics
        equity_series = pd.Series(equity)
        returns = equity_series.pct_change().dropna()
        
        sell_trades = [t for t in trades if t['type'] in ['SELL', 'CLOSE'] and 'pnl' in t]
        winning = len([t for t in sell_trades if t.get('pnl', 0) > 0])
        losing = len([t for t in sell_trades if t.get('pnl', 0) <= 0])
        
        total_return = (capital - initial_capital) / initial_capital * 100
        win_rate = winning / (winning + losing) * 100 if (winning + losing) > 0 else 0
        
        # Max drawdown
        running_max = equity_series.expanding().max()
        drawdowns = (equity_series - running_max) / running_max
        max_drawdown = abs(drawdowns.min()) * 100
        
        # Sharpe
        sharpe = (returns.mean() * np.sqrt(252)) / returns.std() if returns.std() > 0 else 0
        
        return {
            'strategy_name': strategy.name,
            'initial_capital': initial_capital,
            'final_capital': capital,
            'total_return': total_return,
            'total_trades': len(trades),
            'winning_trades': winning,
            'losing_trades': losing,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'trades': trades,
            'equity_curve': equity
        }
    
    def get_template_names(self) -> List[str]:
        """Get list of template strategy names."""
        return [name.replace('template_', '') for name in self._strategies.keys() if name.startswith('template_')]
    
    def get_template(self, name: str) -> Optional[Strategy]:
        """Get a template strategy."""
        return self._strategies.get(f"template_{name}")
