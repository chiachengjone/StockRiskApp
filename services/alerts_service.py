"""
Smart Alerts Service
====================
Intelligent alert system for technical analysis signals

Provides:
- Price alerts (above/below/crossing levels)
- Indicator alerts (RSI, MACD, etc.)
- Pattern detection alerts
- Volume alerts
- Alert history and notifications
- Scheduled monitoring

Author: Stock Risk App | Feb 2026
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS
# ============================================================================

class AlertType(Enum):
    """Types of alerts."""
    PRICE_ABOVE = "price_above"
    PRICE_BELOW = "price_below"
    PRICE_CROSS_UP = "price_cross_up"
    PRICE_CROSS_DOWN = "price_cross_down"
    PERCENT_CHANGE = "percent_change"
    
    RSI_OVERBOUGHT = "rsi_overbought"
    RSI_OVERSOLD = "rsi_oversold"
    RSI_CROSS = "rsi_cross"
    
    MACD_BULLISH = "macd_bullish"
    MACD_BEARISH = "macd_bearish"
    MACD_HISTOGRAM = "macd_histogram"
    
    SMA_CROSS_UP = "sma_cross_up"
    SMA_CROSS_DOWN = "sma_cross_down"
    GOLDEN_CROSS = "golden_cross"
    DEATH_CROSS = "death_cross"
    
    BB_UPPER_TOUCH = "bb_upper_touch"
    BB_LOWER_TOUCH = "bb_lower_touch"
    BB_SQUEEZE = "bb_squeeze"
    
    VOLUME_SPIKE = "volume_spike"
    VOLUME_DROUGHT = "volume_drought"
    
    PATTERN_DETECTED = "pattern_detected"
    DIVERGENCE_DETECTED = "divergence_detected"
    REGIME_CHANGE = "regime_change"
    
    SUPPORT_BREAK = "support_break"
    RESISTANCE_BREAK = "resistance_break"
    
    CUSTOM = "custom"


class AlertPriority(Enum):
    """Alert priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status."""
    ACTIVE = "active"
    TRIGGERED = "triggered"
    EXPIRED = "expired"
    DISABLED = "disabled"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class AlertCondition:
    """
    Alert condition configuration.
    """
    alert_type: AlertType
    symbol: str
    threshold: float
    comparison: str = ">"  # >, <, >=, <=, ==, crosses_above, crosses_below
    secondary_threshold: Optional[float] = None  # For range alerts
    indicator_params: Dict[str, Any] = field(default_factory=dict)  # E.g., RSI period
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'alert_type': self.alert_type.value,
            'symbol': self.symbol,
            'threshold': self.threshold,
            'comparison': self.comparison,
            'secondary_threshold': self.secondary_threshold,
            'indicator_params': self.indicator_params
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AlertCondition':
        return cls(
            alert_type=AlertType(data['alert_type']),
            symbol=data['symbol'],
            threshold=data['threshold'],
            comparison=data.get('comparison', '>'),
            secondary_threshold=data.get('secondary_threshold'),
            indicator_params=data.get('indicator_params', {})
        )


@dataclass
class Alert:
    """
    Alert definition.
    """
    id: str
    name: str
    conditions: List[AlertCondition]
    priority: AlertPriority
    status: AlertStatus
    created_at: datetime
    expires_at: Optional[datetime] = None
    triggered_at: Optional[datetime] = None
    trigger_count: int = 0
    max_triggers: int = 1  # 0 for unlimited
    cooldown_minutes: int = 60  # Minimum time between triggers
    last_triggered: Optional[datetime] = None
    notification_message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'conditions': [c.to_dict() for c in self.conditions],
            'priority': self.priority.value,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'triggered_at': self.triggered_at.isoformat() if self.triggered_at else None,
            'trigger_count': self.trigger_count,
            'max_triggers': self.max_triggers,
            'cooldown_minutes': self.cooldown_minutes,
            'last_triggered': self.last_triggered.isoformat() if self.last_triggered else None,
            'notification_message': self.notification_message,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Alert':
        return cls(
            id=data['id'],
            name=data['name'],
            conditions=[AlertCondition.from_dict(c) for c in data['conditions']],
            priority=AlertPriority(data['priority']),
            status=AlertStatus(data['status']),
            created_at=datetime.fromisoformat(data['created_at']),
            expires_at=datetime.fromisoformat(data['expires_at']) if data.get('expires_at') else None,
            triggered_at=datetime.fromisoformat(data['triggered_at']) if data.get('triggered_at') else None,
            trigger_count=data.get('trigger_count', 0),
            max_triggers=data.get('max_triggers', 1),
            cooldown_minutes=data.get('cooldown_minutes', 60),
            last_triggered=datetime.fromisoformat(data['last_triggered']) if data.get('last_triggered') else None,
            notification_message=data.get('notification_message', ''),
            metadata=data.get('metadata', {})
        )


@dataclass
class AlertTrigger:
    """
    Record of an alert being triggered.
    """
    alert_id: str
    alert_name: str
    symbol: str
    triggered_at: datetime
    condition_type: AlertType
    trigger_value: float
    threshold: float
    price_at_trigger: float
    message: str
    priority: AlertPriority
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'alert_id': self.alert_id,
            'alert_name': self.alert_name,
            'symbol': self.symbol,
            'triggered_at': self.triggered_at.isoformat(),
            'condition_type': self.condition_type.value,
            'trigger_value': self.trigger_value,
            'threshold': self.threshold,
            'price_at_trigger': self.price_at_trigger,
            'message': self.message,
            'priority': self.priority.value
        }


# ============================================================================
# ALERT TEMPLATES
# ============================================================================

ALERT_TEMPLATES = {
    'rsi_oversold': {
        'name': 'RSI Oversold Alert',
        'description': 'Triggers when RSI drops below oversold level',
        'conditions': [
            {
                'alert_type': 'rsi_oversold',
                'threshold': 30,
                'comparison': '<',
                'indicator_params': {'period': 14}
            }
        ],
        'priority': 'high'
    },
    'rsi_overbought': {
        'name': 'RSI Overbought Alert',
        'description': 'Triggers when RSI rises above overbought level',
        'conditions': [
            {
                'alert_type': 'rsi_overbought',
                'threshold': 70,
                'comparison': '>',
                'indicator_params': {'period': 14}
            }
        ],
        'priority': 'high'
    },
    'golden_cross': {
        'name': 'Golden Cross Alert',
        'description': 'Triggers when 50-day SMA crosses above 200-day SMA',
        'conditions': [
            {
                'alert_type': 'golden_cross',
                'threshold': 0,
                'comparison': 'crosses_above',
                'indicator_params': {'fast_period': 50, 'slow_period': 200}
            }
        ],
        'priority': 'critical'
    },
    'death_cross': {
        'name': 'Death Cross Alert',
        'description': 'Triggers when 50-day SMA crosses below 200-day SMA',
        'conditions': [
            {
                'alert_type': 'death_cross',
                'threshold': 0,
                'comparison': 'crosses_below',
                'indicator_params': {'fast_period': 50, 'slow_period': 200}
            }
        ],
        'priority': 'critical'
    },
    'volume_spike': {
        'name': 'Volume Spike Alert',
        'description': 'Triggers when volume exceeds 2x average',
        'conditions': [
            {
                'alert_type': 'volume_spike',
                'threshold': 2.0,
                'comparison': '>',
                'indicator_params': {'period': 20}
            }
        ],
        'priority': 'medium'
    },
    'macd_bullish_cross': {
        'name': 'MACD Bullish Crossover',
        'description': 'Triggers when MACD line crosses above signal line',
        'conditions': [
            {
                'alert_type': 'macd_bullish',
                'threshold': 0,
                'comparison': 'crosses_above',
                'indicator_params': {'fast': 12, 'slow': 26, 'signal': 9}
            }
        ],
        'priority': 'high'
    },
    'bb_squeeze': {
        'name': 'Bollinger Band Squeeze',
        'description': 'Triggers when Bollinger Bands contract (low volatility)',
        'conditions': [
            {
                'alert_type': 'bb_squeeze',
                'threshold': 0.02,  # Band width threshold
                'comparison': '<',
                'indicator_params': {'period': 20, 'std_dev': 2}
            }
        ],
        'priority': 'medium'
    },
    'price_breakout': {
        'name': 'Price Breakout Alert',
        'description': 'Triggers on significant price movement',
        'conditions': [
            {
                'alert_type': 'percent_change',
                'threshold': 5.0,  # 5% move
                'comparison': '>',
                'indicator_params': {'period': 1}  # Daily change
            }
        ],
        'priority': 'high'
    }
}


# ============================================================================
# ALERT SERVICE CLASS
# ============================================================================

class AlertsService:
    """
    Smart Alerts Service.
    
    Manages technical analysis alerts including creation, monitoring,
    and triggering of various alert types.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize the Alerts Service.
        
        Args:
            storage_path: Path to store alerts configuration
        """
        self.storage_path = Path(storage_path) if storage_path else Path("data/alerts")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.alerts: Dict[str, Alert] = {}
        self.trigger_history: List[AlertTrigger] = []
        
        self._load_alerts()
        logger.info("AlertsService initialized")
    
    def create_alert(
        self,
        name: str,
        conditions: List[AlertCondition],
        priority: AlertPriority = AlertPriority.MEDIUM,
        expires_at: Optional[datetime] = None,
        max_triggers: int = 1,
        cooldown_minutes: int = 60,
        notification_message: str = ""
    ) -> Alert:
        """
        Create a new alert.
        
        Args:
            name: Alert name
            conditions: List of conditions
            priority: Alert priority
            expires_at: Expiration datetime
            max_triggers: Maximum times to trigger (0 = unlimited)
            cooldown_minutes: Minimum time between triggers
            notification_message: Custom notification message
            
        Returns:
            Created Alert
        """
        import uuid
        
        alert = Alert(
            id=str(uuid.uuid4())[:8],
            name=name,
            conditions=conditions,
            priority=priority,
            status=AlertStatus.ACTIVE,
            created_at=datetime.now(),
            expires_at=expires_at,
            max_triggers=max_triggers,
            cooldown_minutes=cooldown_minutes,
            notification_message=notification_message
        )
        
        self.alerts[alert.id] = alert
        self._save_alerts()
        
        logger.info(f"Created alert: {name} ({alert.id})")
        return alert
    
    def create_from_template(
        self,
        template_name: str,
        symbol: str,
        custom_threshold: Optional[float] = None,
        expires_at: Optional[datetime] = None
    ) -> Optional[Alert]:
        """
        Create an alert from a predefined template.
        
        Args:
            template_name: Name of the template
            symbol: Stock symbol
            custom_threshold: Override default threshold
            expires_at: Expiration datetime
            
        Returns:
            Created Alert or None
        """
        template = ALERT_TEMPLATES.get(template_name)
        if not template:
            logger.error(f"Template not found: {template_name}")
            return None
        
        conditions = []
        for cond_dict in template['conditions']:
            threshold = custom_threshold if custom_threshold is not None else cond_dict['threshold']
            conditions.append(AlertCondition(
                alert_type=AlertType(cond_dict['alert_type']),
                symbol=symbol,
                threshold=threshold,
                comparison=cond_dict.get('comparison', '>'),
                indicator_params=cond_dict.get('indicator_params', {})
            ))
        
        return self.create_alert(
            name=f"{template['name']} - {symbol}",
            conditions=conditions,
            priority=AlertPriority(template['priority']),
            expires_at=expires_at
        )
    
    def check_alerts(
        self,
        symbol: str,
        current_data: pd.DataFrame,
        indicators: Optional[Dict[str, Any]] = None
    ) -> List[AlertTrigger]:
        """
        Check all active alerts for a symbol.
        
        Args:
            symbol: Stock symbol
            current_data: Current OHLCV data
            indicators: Pre-calculated indicators
            
        Returns:
            List of triggered alerts
        """
        triggered = []
        now = datetime.now()
        
        for alert in self.alerts.values():
            # Skip non-active alerts
            if alert.status != AlertStatus.ACTIVE:
                continue
            
            # Check expiration
            if alert.expires_at and now > alert.expires_at:
                alert.status = AlertStatus.EXPIRED
                continue
            
            # Check max triggers
            if alert.max_triggers > 0 and alert.trigger_count >= alert.max_triggers:
                alert.status = AlertStatus.TRIGGERED
                continue
            
            # Check cooldown
            if alert.last_triggered:
                cooldown_end = alert.last_triggered + timedelta(minutes=alert.cooldown_minutes)
                if now < cooldown_end:
                    continue
            
            # Check each condition
            for condition in alert.conditions:
                if condition.symbol.upper() != symbol.upper():
                    continue
                
                trigger = self._check_condition(
                    condition, current_data, indicators
                )
                
                if trigger:
                    alert.trigger_count += 1
                    alert.last_triggered = now
                    alert.triggered_at = now
                    
                    alert_trigger = AlertTrigger(
                        alert_id=alert.id,
                        alert_name=alert.name,
                        symbol=symbol,
                        triggered_at=now,
                        condition_type=condition.alert_type,
                        trigger_value=trigger['value'],
                        threshold=condition.threshold,
                        price_at_trigger=trigger['price'],
                        message=self._generate_message(alert, condition, trigger),
                        priority=alert.priority
                    )
                    
                    triggered.append(alert_trigger)
                    self.trigger_history.append(alert_trigger)
                    
                    logger.info(f"Alert triggered: {alert.name}")
        
        self._save_alerts()
        return triggered
    
    def _check_condition(
        self,
        condition: AlertCondition,
        data: pd.DataFrame,
        indicators: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Check if a condition is met.
        
        Returns dict with 'value' and 'price' if triggered.
        """
        if len(data) < 2:
            return None
        
        # Handle MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            close = data['Close'].iloc[:, 0]
            volume = data['Volume'].iloc[:, 0] if 'Volume' in data.columns.get_level_values(0) else None
        else:
            close = data['Close']
            volume = data.get('Volume')
        
        current_price = float(close.iloc[-1])
        prev_price = float(close.iloc[-2])
        
        alert_type = condition.alert_type
        threshold = condition.threshold
        comparison = condition.comparison
        
        # Price alerts
        if alert_type == AlertType.PRICE_ABOVE:
            if current_price > threshold:
                return {'value': current_price, 'price': current_price}
                
        elif alert_type == AlertType.PRICE_BELOW:
            if current_price < threshold:
                return {'value': current_price, 'price': current_price}
                
        elif alert_type == AlertType.PRICE_CROSS_UP:
            if prev_price <= threshold < current_price:
                return {'value': current_price, 'price': current_price}
                
        elif alert_type == AlertType.PRICE_CROSS_DOWN:
            if prev_price >= threshold > current_price:
                return {'value': current_price, 'price': current_price}
                
        elif alert_type == AlertType.PERCENT_CHANGE:
            change = abs((current_price - prev_price) / prev_price * 100)
            if change > threshold:
                return {'value': change, 'price': current_price}
        
        # RSI alerts
        elif alert_type in [AlertType.RSI_OVERBOUGHT, AlertType.RSI_OVERSOLD, AlertType.RSI_CROSS]:
            rsi = self._calculate_rsi(close, condition.indicator_params.get('period', 14))
            if rsi is not None:
                if alert_type == AlertType.RSI_OVERBOUGHT and rsi > threshold:
                    return {'value': rsi, 'price': current_price}
                elif alert_type == AlertType.RSI_OVERSOLD and rsi < threshold:
                    return {'value': rsi, 'price': current_price}
        
        # MACD alerts
        elif alert_type in [AlertType.MACD_BULLISH, AlertType.MACD_BEARISH]:
            macd_data = self._calculate_macd(
                close,
                condition.indicator_params.get('fast', 12),
                condition.indicator_params.get('slow', 26),
                condition.indicator_params.get('signal', 9)
            )
            if macd_data:
                macd_line = macd_data['macd']
                signal_line = macd_data['signal']
                
                if len(macd_line) >= 2 and len(signal_line) >= 2:
                    if alert_type == AlertType.MACD_BULLISH:
                        if macd_line.iloc[-2] <= signal_line.iloc[-2] and macd_line.iloc[-1] > signal_line.iloc[-1]:
                            return {'value': float(macd_line.iloc[-1]), 'price': current_price}
                    elif alert_type == AlertType.MACD_BEARISH:
                        if macd_line.iloc[-2] >= signal_line.iloc[-2] and macd_line.iloc[-1] < signal_line.iloc[-1]:
                            return {'value': float(macd_line.iloc[-1]), 'price': current_price}
        
        # Moving average cross alerts
        elif alert_type in [AlertType.GOLDEN_CROSS, AlertType.DEATH_CROSS]:
            fast_period = condition.indicator_params.get('fast_period', 50)
            slow_period = condition.indicator_params.get('slow_period', 200)
            
            if len(close) >= slow_period:
                fast_ma = close.rolling(fast_period).mean()
                slow_ma = close.rolling(slow_period).mean()
                
                if alert_type == AlertType.GOLDEN_CROSS:
                    if fast_ma.iloc[-2] <= slow_ma.iloc[-2] and fast_ma.iloc[-1] > slow_ma.iloc[-1]:
                        return {'value': float(fast_ma.iloc[-1]), 'price': current_price}
                elif alert_type == AlertType.DEATH_CROSS:
                    if fast_ma.iloc[-2] >= slow_ma.iloc[-2] and fast_ma.iloc[-1] < slow_ma.iloc[-1]:
                        return {'value': float(fast_ma.iloc[-1]), 'price': current_price}
        
        # Volume alerts
        elif alert_type == AlertType.VOLUME_SPIKE:
            if volume is not None and len(volume) > 20:
                avg_volume = volume.rolling(20).mean().iloc[-1]
                current_volume = volume.iloc[-1]
                ratio = current_volume / avg_volume if avg_volume > 0 else 0
                
                if ratio > threshold:
                    return {'value': ratio, 'price': current_price}
        
        elif alert_type == AlertType.VOLUME_DROUGHT:
            if volume is not None and len(volume) > 20:
                avg_volume = volume.rolling(20).mean().iloc[-1]
                current_volume = volume.iloc[-1]
                ratio = current_volume / avg_volume if avg_volume > 0 else 0
                
                if ratio < threshold:
                    return {'value': ratio, 'price': current_price}
        
        # Bollinger Band alerts
        elif alert_type == AlertType.BB_SQUEEZE:
            period = condition.indicator_params.get('period', 20)
            std_dev = condition.indicator_params.get('std_dev', 2)
            
            if len(close) >= period:
                sma = close.rolling(period).mean()
                std = close.rolling(period).std()
                upper = sma + std_dev * std
                lower = sma - std_dev * std
                bandwidth = (upper - lower) / sma
                
                if bandwidth.iloc[-1] < threshold:
                    return {'value': float(bandwidth.iloc[-1]), 'price': current_price}
        
        return None
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> Optional[float]:
        """Calculate RSI."""
        if len(prices) < period + 1:
            return None
        
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None
    
    def _calculate_macd(
        self,
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Optional[Dict[str, pd.Series]]:
        """Calculate MACD."""
        if len(prices) < slow + signal:
            return None
        
        fast_ema = prices.ewm(span=fast, adjust=False).mean()
        slow_ema = prices.ewm(span=slow, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        
        return {'macd': macd_line, 'signal': signal_line}
    
    def _generate_message(
        self,
        alert: Alert,
        condition: AlertCondition,
        trigger: Dict[str, Any]
    ) -> str:
        """Generate alert notification message."""
        if alert.notification_message:
            return alert.notification_message.format(
                symbol=condition.symbol,
                value=trigger['value'],
                threshold=condition.threshold,
                price=trigger['price']
            )
        
        alert_type = condition.alert_type
        symbol = condition.symbol
        value = trigger['value']
        price = trigger['price']
        
        messages = {
            AlertType.PRICE_ABOVE: f"{symbol} price ${price:.2f} is above ${condition.threshold:.2f}",
            AlertType.PRICE_BELOW: f"{symbol} price ${price:.2f} is below ${condition.threshold:.2f}",
            AlertType.PRICE_CROSS_UP: f"{symbol} crossed above ${condition.threshold:.2f}",
            AlertType.PRICE_CROSS_DOWN: f"{symbol} crossed below ${condition.threshold:.2f}",
            AlertType.PERCENT_CHANGE: f"{symbol} moved {value:.1f}% (threshold: {condition.threshold}%)",
            AlertType.RSI_OVERBOUGHT: f"{symbol} RSI is overbought at {value:.1f}",
            AlertType.RSI_OVERSOLD: f"{symbol} RSI is oversold at {value:.1f}",
            AlertType.MACD_BULLISH: f"{symbol} MACD bullish crossover detected",
            AlertType.MACD_BEARISH: f"{symbol} MACD bearish crossover detected",
            AlertType.GOLDEN_CROSS: f"{symbol} Golden Cross detected - bullish signal",
            AlertType.DEATH_CROSS: f"{symbol} Death Cross detected - bearish signal",
            AlertType.VOLUME_SPIKE: f"{symbol} volume spike: {value:.1f}x average",
            AlertType.BB_SQUEEZE: f"{symbol} Bollinger Band squeeze detected",
        }
        
        return messages.get(alert_type, f"{alert.name} triggered for {symbol}")
    
    def get_alert(self, alert_id: str) -> Optional[Alert]:
        """Get alert by ID."""
        return self.alerts.get(alert_id)
    
    def get_alerts(
        self,
        symbol: Optional[str] = None,
        status: Optional[AlertStatus] = None
    ) -> List[Alert]:
        """Get alerts with optional filtering."""
        alerts = list(self.alerts.values())
        
        if symbol:
            symbol = symbol.upper()
            alerts = [a for a in alerts if any(c.symbol.upper() == symbol for c in a.conditions)]
        
        if status:
            alerts = [a for a in alerts if a.status == status]
        
        return sorted(alerts, key=lambda a: a.created_at, reverse=True)
    
    def update_alert(
        self,
        alert_id: str,
        **updates
    ) -> Optional[Alert]:
        """Update an existing alert."""
        alert = self.alerts.get(alert_id)
        if not alert:
            return None
        
        for key, value in updates.items():
            if hasattr(alert, key):
                setattr(alert, key, value)
        
        self._save_alerts()
        return alert
    
    def delete_alert(self, alert_id: str) -> bool:
        """Delete an alert."""
        if alert_id in self.alerts:
            del self.alerts[alert_id]
            self._save_alerts()
            return True
        return False
    
    def disable_alert(self, alert_id: str) -> Optional[Alert]:
        """Disable an alert."""
        return self.update_alert(alert_id, status=AlertStatus.DISABLED)
    
    def enable_alert(self, alert_id: str) -> Optional[Alert]:
        """Enable a disabled alert."""
        return self.update_alert(alert_id, status=AlertStatus.ACTIVE)
    
    def get_trigger_history(
        self,
        symbol: Optional[str] = None,
        limit: int = 50
    ) -> List[AlertTrigger]:
        """Get alert trigger history."""
        history = self.trigger_history
        
        if symbol:
            symbol = symbol.upper()
            history = [t for t in history if t.symbol.upper() == symbol]
        
        return sorted(history, key=lambda t: t.triggered_at, reverse=True)[:limit]
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics."""
        total = len(self.alerts)
        active = len([a for a in self.alerts.values() if a.status == AlertStatus.ACTIVE])
        triggered = len([a for a in self.alerts.values() if a.status == AlertStatus.TRIGGERED])
        expired = len([a for a in self.alerts.values() if a.status == AlertStatus.EXPIRED])
        
        # Priority breakdown
        by_priority = {
            p.value: len([a for a in self.alerts.values() if a.priority == p])
            for p in AlertPriority
        }
        
        # Most triggered symbols
        symbol_triggers = {}
        for trigger in self.trigger_history:
            symbol_triggers[trigger.symbol] = symbol_triggers.get(trigger.symbol, 0) + 1
        
        top_symbols = sorted(symbol_triggers.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'total_alerts': total,
            'active': active,
            'triggered': triggered,
            'expired': expired,
            'disabled': total - active - triggered - expired,
            'by_priority': by_priority,
            'total_triggers': len(self.trigger_history),
            'top_triggered_symbols': dict(top_symbols)
        }
    
    def _save_alerts(self):
        """Save alerts to storage."""
        try:
            alerts_file = self.storage_path / "alerts.json"
            data = {
                'alerts': [a.to_dict() for a in self.alerts.values()],
                'trigger_history': [t.to_dict() for t in self.trigger_history[-500:]]  # Keep last 500
            }
            
            with open(alerts_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving alerts: {e}")
    
    def _load_alerts(self):
        """Load alerts from storage."""
        try:
            alerts_file = self.storage_path / "alerts.json"
            
            if alerts_file.exists():
                with open(alerts_file, 'r') as f:
                    data = json.load(f)
                
                self.alerts = {
                    a['id']: Alert.from_dict(a) 
                    for a in data.get('alerts', [])
                }
                
                self.trigger_history = [
                    AlertTrigger(
                        alert_id=t['alert_id'],
                        alert_name=t['alert_name'],
                        symbol=t['symbol'],
                        triggered_at=datetime.fromisoformat(t['triggered_at']),
                        condition_type=AlertType(t['condition_type']),
                        trigger_value=t['trigger_value'],
                        threshold=t['threshold'],
                        price_at_trigger=t['price_at_trigger'],
                        message=t['message'],
                        priority=AlertPriority(t['priority'])
                    )
                    for t in data.get('trigger_history', [])
                ]
                
                logger.info(f"Loaded {len(self.alerts)} alerts")
                
        except Exception as e:
            logger.error(f"Error loading alerts: {e}")
            self.alerts = {}
            self.trigger_history = []
    
    @staticmethod
    def get_available_templates() -> Dict[str, Dict[str, Any]]:
        """Get list of available alert templates."""
        return {name: {'name': t['name'], 'description': t['description']} 
                for name, t in ALERT_TEMPLATES.items()}
