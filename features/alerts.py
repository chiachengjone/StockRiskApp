"""
Alert Manager - Risk Monitoring & Notifications
================================================
Set thresholds and get notified when risk metrics exceed limits.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


class AlertManager:
    """
    Manages risk alerts and notifications.
    
    Features:
    - Set alerts on VaR, volatility, drawdown, etc.
    - Check alerts against current metrics
    - Email notifications (optional)
    - Alert history tracking
    """
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            base_dir = os.path.dirname(os.path.dirname(__file__))
            data_dir = os.path.join(base_dir, 'data')
            os.makedirs(data_dir, exist_ok=True)
            config_path = os.path.join(data_dir, 'alerts_config.json')
        
        self.config_path = config_path
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        """Load alert configuration from file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.warning(f"Error loading alerts config: {e}")
        
        # Default config
        return {
            'email': '',
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'email_enabled': False,
            'alerts': [],
            'history': []
        }
    
    def _save_config(self):
        """Save alert configuration to file."""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving alerts config: {e}")
    
    def add_alert(
        self,
        ticker: str,
        metric: str,
        threshold: float,
        direction: str = 'above',
        name: str = None
    ) -> Dict:
        """
        Add a new alert.
        
        Args:
            ticker: Stock symbol (or 'PORTFOLIO' for portfolio alerts)
            metric: Metric to monitor ('var', 'volatility', 'max_drawdown', 'price', 'sharpe')
            threshold: Threshold value
            direction: 'above' or 'below'
            name: Optional custom name for the alert
        
        Returns:
            The created alert
        """
        alert = {
            'id': len(self.config['alerts']) + 1,
            'ticker': ticker.upper(),
            'metric': metric.lower(),
            'threshold': threshold,
            'direction': direction.lower(),
            'name': name or f"{ticker} {metric} {direction} {threshold}",
            'active': True,
            'created_at': datetime.now().isoformat(),
            'last_triggered': None,
            'trigger_count': 0
        }
        
        self.config['alerts'].append(alert)
        self._save_config()
        
        self.logger.info(f"Alert added: {alert['name']}")
        return alert
    
    def remove_alert(self, alert_id: int) -> bool:
        """Remove an alert by ID."""
        original_count = len(self.config['alerts'])
        self.config['alerts'] = [a for a in self.config['alerts'] if a['id'] != alert_id]
        
        if len(self.config['alerts']) < original_count:
            self._save_config()
            return True
        return False
    
    def toggle_alert(self, alert_id: int) -> bool:
        """Toggle an alert's active status."""
        for alert in self.config['alerts']:
            if alert['id'] == alert_id:
                alert['active'] = not alert['active']
                self._save_config()
                return True
        return False
    
    def get_alerts(self, ticker: str = None, active_only: bool = True) -> List[Dict]:
        """
        Get alerts, optionally filtered.
        
        Args:
            ticker: Filter by ticker (optional)
            active_only: Only return active alerts
        
        Returns:
            List of matching alerts
        """
        alerts = self.config['alerts']
        
        if active_only:
            alerts = [a for a in alerts if a['active']]
        
        if ticker:
            alerts = [a for a in alerts if a['ticker'] == ticker.upper()]
        
        return alerts
    
    def check_alerts(self, ticker: str, metrics: Dict) -> List[Dict]:
        """
        Check if any alerts are triggered for given metrics.
        
        Args:
            ticker: Stock symbol
            metrics: Dictionary of current metric values
                {
                    'var': 0.05,
                    'volatility': 0.25,
                    'max_drawdown': -0.15,
                    'price': 150.00,
                    'sharpe': 1.5
                }
        
        Returns:
            List of triggered alerts with current values
        """
        triggered = []
        
        for alert in self.config['alerts']:
            if not alert['active']:
                continue
            
            if alert['ticker'] != ticker.upper() and alert['ticker'] != 'PORTFOLIO':
                continue
            
            metric_key = alert['metric']
            current_value = metrics.get(metric_key)
            
            if current_value is None:
                continue
            
            # Check if threshold is breached
            threshold = alert['threshold']
            direction = alert['direction']
            
            is_triggered = False
            if direction == 'above' and current_value > threshold:
                is_triggered = True
            elif direction == 'below' and current_value < threshold:
                is_triggered = True
            
            if is_triggered:
                triggered_alert = {
                    **alert,
                    'current_value': current_value,
                    'triggered_at': datetime.now().isoformat()
                }
                triggered.append(triggered_alert)
                
                # Update alert stats
                alert['last_triggered'] = datetime.now().isoformat()
                alert['trigger_count'] += 1
                
                # Add to history
                self._add_to_history(triggered_alert)
        
        if triggered:
            self._save_config()
        
        return triggered
    
    def _add_to_history(self, triggered_alert: Dict):
        """Add triggered alert to history."""
        history_entry = {
            'alert_id': triggered_alert['id'],
            'ticker': triggered_alert['ticker'],
            'metric': triggered_alert['metric'],
            'threshold': triggered_alert['threshold'],
            'current_value': triggered_alert['current_value'],
            'triggered_at': triggered_alert['triggered_at']
        }
        
        self.config['history'].append(history_entry)
        
        # Keep only last 100 history entries
        if len(self.config['history']) > 100:
            self.config['history'] = self.config['history'][-100:]
    
    def get_history(self, limit: int = 20) -> List[Dict]:
        """Get recent alert history."""
        return self.config['history'][-limit:][::-1]
    
    def clear_history(self):
        """Clear alert history."""
        self.config['history'] = []
        self._save_config()
    
    def configure_email(
        self,
        email: str,
        smtp_server: str = 'smtp.gmail.com',
        smtp_port: int = 587,
        password: str = None
    ):
        """Configure email notifications."""
        self.config['email'] = email
        self.config['smtp_server'] = smtp_server
        self.config['smtp_port'] = smtp_port
        self.config['email_enabled'] = bool(email and password)
        
        if password:
            # Note: In production, use secure storage for password
            self.config['email_password'] = password
        
        self._save_config()
    
    def send_email_notification(self, triggered_alerts: List[Dict]) -> bool:
        """
        Send email notification for triggered alerts.
        
        Args:
            triggered_alerts: List of triggered alert dictionaries
        
        Returns:
            True if email sent successfully
        """
        if not self.config.get('email_enabled') or not self.config.get('email'):
            self.logger.warning("Email not configured")
            return False
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config['email']
            msg['To'] = self.config['email']
            msg['Subject'] = f" Risk Alert: {len(triggered_alerts)} Alert(s) Triggered"
            
            body = self._format_alert_email(triggered_alerts)
            msg.attach(MIMEText(body, 'html'))
            
            with smtplib.SMTP(self.config['smtp_server'], self.config['smtp_port']) as server:
                server.starttls()
                server.login(self.config['email'], self.config.get('email_password', ''))
                server.send_message(msg)
            
            self.logger.info(f"Alert email sent for {len(triggered_alerts)} alerts")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send email: {e}")
            return False
    
    def _format_alert_email(self, alerts: List[Dict]) -> str:
        """Format alerts as HTML email body."""
        html = """
        <html>
        <head>
            <style>
                body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; }
                .alert { 
                    padding: 15px; 
                    margin: 10px 0; 
                    border-left: 4px solid #FF3B30; 
                    background: #fff5f5;
                }
                .alert-title { font-weight: bold; color: #1C1C1E; }
                .alert-value { color: #FF3B30; font-size: 1.2em; }
                .alert-details { color: #666; font-size: 0.9em; }
            </style>
        </head>
        <body>
            <h2> Risk Alerts Triggered</h2>
            <p>The following risk thresholds have been breached:</p>
        """
        
        for alert in alerts:
            html += f"""
            <div class="alert">
                <div class="alert-title">{alert['ticker']} - {alert['metric'].upper()}</div>
                <div class="alert-value">
                    Current: {alert['current_value']:.2%} 
                    (Threshold: {alert['direction']} {alert['threshold']:.2%})
                </div>
                <div class="alert-details">
                    Triggered at: {alert['triggered_at']}
                </div>
            </div>
            """
        
        html += """
            <p style="color: #666; font-size: 0.8em;">
                This alert was generated by Stock Risk Model.
            </p>
        </body>
        </html>
        """
        
        return html
    
    def get_summary(self) -> Dict:
        """Get summary of alert system status."""
        alerts = self.config['alerts']
        
        return {
            'total_alerts': len(alerts),
            'active_alerts': len([a for a in alerts if a['active']]),
            'triggered_today': len([
                h for h in self.config['history']
                if h['triggered_at'].startswith(datetime.now().strftime('%Y-%m-%d'))
            ]),
            'email_enabled': self.config.get('email_enabled', False),
            'history_count': len(self.config['history'])
        }
    
    def create_default_alerts(self, ticker: str):
        """Create a set of default alerts for a ticker."""
        default_alerts = [
            ('var', 0.05, 'above', 'VaR exceeds 5%'),
            ('volatility', 0.40, 'above', 'Volatility exceeds 40%'),
            ('max_drawdown', -0.20, 'below', 'Drawdown exceeds 20%'),
            ('sharpe', 0.5, 'below', 'Sharpe below 0.5')
        ]
        
        for metric, threshold, direction, description in default_alerts:
            self.add_alert(
                ticker=ticker,
                metric=metric,
                threshold=threshold,
                direction=direction,
                name=f"{ticker}: {description}"
            )

# =============================================================================
# STREAMLIT UI INTEGRATION
# =============================================================================

def render_alerts_panel(
    alert_manager: AlertManager,
    ticker: str = None,
    current_metrics: Dict = None
):
    """
    Render the alerts management panel in Streamlit.
    
    Args:
        alert_manager: AlertManager instance
        ticker: Current ticker being analyzed
        current_metrics: Current metric values for checking
    """
    import streamlit as st
    
    # Colors for UI
    COLORS = {
        'danger': '#FF3B30',
        'warning': '#FF9500',
        'success': '#34C759',
        'primary': '#007AFF',
        'muted': '#8b949e'
    }
    
    st.markdown("###  Risk Alerts")
    
    # Get summary
    summary = alert_manager.get_summary()
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Active Alerts", summary['active_alerts'])
    col2.metric("Total Alerts", summary['total_alerts'])
    col3.metric("Triggered Today", summary['triggered_today'])
    col4.metric("Alert History", summary['history_count'])
    
    # Check current alerts if metrics provided
    if current_metrics and ticker:
        triggered = alert_manager.check_alerts(ticker, current_metrics)
        
        if triggered:
            st.markdown("---")
            st.markdown("####  Triggered Alerts")
            
            for alert in triggered:
                st.markdown(f"""
                <div style="
                    background: rgba(255, 59, 48, 0.1);
                    border-left: 3px solid {COLORS['danger']};
                    padding: 0.75rem 1rem;
                    margin-bottom: 0.5rem;
                    border-radius: 4px;
                ">
                    <strong style="color: {COLORS['danger']};">{alert['name']}</strong><br>
                    <span style="color: {COLORS['muted']};">
                        Current: {alert['current_value']:.2%} | Threshold: {alert['direction']} {alert['threshold']:.2%}
                    </span>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Alert management tabs
    alert_tab1, alert_tab2, alert_tab3 = st.tabs(["Active Alerts", "Add Alert", "History"])
    
    with alert_tab1:
        alerts = alert_manager.get_alerts(active_only=False)
        
        if alerts:
            for alert in alerts:
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    status_icon = "🟢" if alert['active'] else ""
                    st.markdown(f"{status_icon} **{alert['name']}**")
                    st.caption(f"{alert['metric'].upper()} {alert['direction']} {alert['threshold']:.2%}")
                
                with col2:
                    if st.button("Toggle", key=f"toggle_{alert['id']}"):
                        alert_manager.toggle_alert(alert['id'])
                        st.rerun()
                
                with col3:
                    if st.button("Delete", key=f"delete_{alert['id']}"):
                        alert_manager.remove_alert(alert['id'])
                        st.rerun()
        else:
            st.info("No alerts configured. Add an alert to get started.")
    
    with alert_tab2:
        st.markdown("#### Create New Alert")
        
        col1, col2 = st.columns(2)
        
        with col1:
            alert_ticker = st.text_input("Ticker", value=ticker or "AAPL", key="new_alert_ticker")
            
            alert_metric = st.selectbox(
                "Metric",
                ["var", "volatility", "max_drawdown", "sharpe", "price"],
                key="new_alert_metric"
            )
        
        with col2:
            alert_direction = st.selectbox("Direction", ["above", "below"], key="new_alert_direction")
            
            # Default thresholds based on metric
            default_thresholds = {
                "var": 0.05,
                "volatility": 0.30,
                "max_drawdown": -0.15,
                "sharpe": 0.5,
                "price": 100.0
            }
            
            alert_threshold = st.number_input(
                "Threshold",
                value=default_thresholds.get(alert_metric, 0.0),
                format="%.4f",
                key="new_alert_threshold"
            )
        
        alert_name = st.text_input(
            "Alert Name (optional)",
            value=f"{alert_ticker}: {alert_metric} {alert_direction} {alert_threshold}",
            key="new_alert_name"
        )
        
        if st.button("Create Alert", type="primary", key="create_alert_btn"):
            alert_manager.add_alert(
                ticker=alert_ticker,
                metric=alert_metric,
                threshold=alert_threshold,
                direction=alert_direction,
                name=alert_name
            )
            st.success(f"Alert created: {alert_name}")
            st.rerun()
        
        st.markdown("---")
        
        # Quick default alerts
        if ticker:
            if st.button(f"Create Default Alerts for {ticker}", key="create_defaults"):
                alert_manager.create_default_alerts(ticker)
                st.success(f"Default alerts created for {ticker}")
                st.rerun()
    
    with alert_tab3:
        history = alert_manager.get_history(limit=20)
        
        if history:
            for entry in history:
                st.markdown(f"""
                <div style="
                    background: rgba(255, 149, 0, 0.1);
                    border-left: 3px solid {COLORS['warning']};
                    padding: 0.5rem 0.75rem;
                    margin-bottom: 0.5rem;
                    border-radius: 4px;
                    font-size: 0.85rem;
                ">
                    <strong>{entry['ticker']}</strong> - {entry['metric'].upper()}<br>
                    <span style="color: {COLORS['muted']};">
                        Value: {entry['current_value']:.2%} | Threshold: {entry['threshold']:.2%}<br>
                        {entry['triggered_at'][:19]}
                    </span>
                </div>
                """, unsafe_allow_html=True)
            
            if st.button("Clear History", key="clear_history"):
                alert_manager.clear_history()
                st.rerun()
        else:
            st.info("No alert history yet.")


def render_alert_notification_badge(alert_manager: AlertManager) -> int:
    """
    Render a notification badge showing triggered alerts count.
    
    Args:
        alert_manager: AlertManager instance
        
    Returns:
        Number of alerts triggered today
    """
    import streamlit as st
    
    summary = alert_manager.get_summary()
    triggered_today = summary['triggered_today']
    
    if triggered_today > 0:
        st.markdown(f"""
        <span style="
            background: #FF3B30;
            color: white;
            font-size: 0.7rem;
            padding: 2px 8px;
            border-radius: 12px;
            font-weight: bold;
            animation: pulse 1s infinite;
        ">{triggered_today} Alert{'s' if triggered_today > 1 else ''}</span>
        <style>
            @keyframes pulse {{
                0%, 100% {{ opacity: 1; }}
                50% {{ opacity: 0.7; }}
            }}
        </style>
        """, unsafe_allow_html=True)
    
    return triggered_today