"""
Performance Reporting Service
==============================
Comprehensive performance tracking and reporting

Provides:
- Signal performance tracking
- Report card generation
- Win rate analysis
- Performance attribution
- Export functionality (CSV, PDF, JSON)

Author: Stock Risk App | Feb 2026
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
import csv
import io
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS
# ============================================================================

class SignalOutcome(Enum):
    """Outcome of a trading signal."""
    WIN = "win"
    LOSS = "loss"
    BREAKEVEN = "breakeven"
    PENDING = "pending"
    EXPIRED = "expired"


class TimeHorizon(Enum):
    """Time horizons for analysis."""
    INTRADAY = "intraday"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class SignalRecord:
    """
    Record of a trading signal.
    """
    id: str
    symbol: str
    signal_type: str  # 'BUY', 'SELL'
    signal_source: str  # e.g., 'RSI_Oversold', 'MACD_Cross', 'Pattern'
    entry_price: float
    entry_date: datetime
    exit_price: Optional[float] = None
    exit_date: Optional[datetime] = None
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    confidence: float = 0.5
    outcome: SignalOutcome = SignalOutcome.PENDING
    pnl_pct: float = 0.0
    pnl_amount: float = 0.0
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'symbol': self.symbol,
            'signal_type': self.signal_type,
            'signal_source': self.signal_source,
            'entry_price': self.entry_price,
            'entry_date': self.entry_date.isoformat(),
            'exit_price': self.exit_price,
            'exit_date': self.exit_date.isoformat() if self.exit_date else None,
            'target_price': self.target_price,
            'stop_loss': self.stop_loss,
            'confidence': self.confidence,
            'outcome': self.outcome.value,
            'pnl_pct': self.pnl_pct,
            'pnl_amount': self.pnl_amount,
            'notes': self.notes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SignalRecord':
        return cls(
            id=data['id'],
            symbol=data['symbol'],
            signal_type=data['signal_type'],
            signal_source=data['signal_source'],
            entry_price=data['entry_price'],
            entry_date=datetime.fromisoformat(data['entry_date']),
            exit_price=data.get('exit_price'),
            exit_date=datetime.fromisoformat(data['exit_date']) if data.get('exit_date') else None,
            target_price=data.get('target_price'),
            stop_loss=data.get('stop_loss'),
            confidence=data.get('confidence', 0.5),
            outcome=SignalOutcome(data.get('outcome', 'pending')),
            pnl_pct=data.get('pnl_pct', 0.0),
            pnl_amount=data.get('pnl_amount', 0.0),
            notes=data.get('notes', '')
        )


@dataclass
class PerformanceMetrics:
    """
    Performance metrics summary.
    """
    total_signals: int
    winning_signals: int
    losing_signals: int
    pending_signals: int
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float
    expectancy: float
    total_return_pct: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    avg_holding_period: float  # days
    best_signal: Optional[SignalRecord] = None
    worst_signal: Optional[SignalRecord] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_signals': self.total_signals,
            'winning_signals': self.winning_signals,
            'losing_signals': self.losing_signals,
            'pending_signals': self.pending_signals,
            'win_rate': self.win_rate,
            'avg_win_pct': self.avg_win_pct,
            'avg_loss_pct': self.avg_loss_pct,
            'profit_factor': self.profit_factor,
            'expectancy': self.expectancy,
            'total_return_pct': self.total_return_pct,
            'max_consecutive_wins': self.max_consecutive_wins,
            'max_consecutive_losses': self.max_consecutive_losses,
            'avg_holding_period': self.avg_holding_period,
            'best_signal': self.best_signal.to_dict() if self.best_signal else None,
            'worst_signal': self.worst_signal.to_dict() if self.worst_signal else None
        }


@dataclass
class ReportCard:
    """
    Signal performance report card.
    """
    period_start: datetime
    period_end: datetime
    overall_metrics: PerformanceMetrics
    by_symbol: Dict[str, PerformanceMetrics]
    by_source: Dict[str, PerformanceMetrics]
    by_signal_type: Dict[str, PerformanceMetrics]
    monthly_breakdown: Dict[str, PerformanceMetrics]
    risk_metrics: Dict[str, float]
    recommendations: List[str]
    generated_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'period': {
                'start': self.period_start.isoformat(),
                'end': self.period_end.isoformat()
            },
            'overall_metrics': self.overall_metrics.to_dict(),
            'by_symbol': {k: v.to_dict() for k, v in self.by_symbol.items()},
            'by_source': {k: v.to_dict() for k, v in self.by_source.items()},
            'by_signal_type': {k: v.to_dict() for k, v in self.by_signal_type.items()},
            'monthly_breakdown': {k: v.to_dict() for k, v in self.monthly_breakdown.items()},
            'risk_metrics': self.risk_metrics,
            'recommendations': self.recommendations,
            'generated_at': self.generated_at.isoformat()
        }


# ============================================================================
# REPORTING SERVICE CLASS  
# ============================================================================

class ReportingService:
    """
    Performance Reporting Service.
    
    Tracks signal performance and generates comprehensive reports.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize the Reporting Service.
        
        Args:
            storage_path: Path to store signal history
        """
        self.storage_path = Path(storage_path) if storage_path else Path("data/signals")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.signals: Dict[str, SignalRecord] = {}
        self._load_signals()
        
        logger.info("ReportingService initialized")
    
    def record_signal(
        self,
        symbol: str,
        signal_type: str,
        signal_source: str,
        entry_price: float,
        target_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        confidence: float = 0.5,
        notes: str = ""
    ) -> SignalRecord:
        """
        Record a new trading signal.
        
        Args:
            symbol: Stock symbol
            signal_type: 'BUY' or 'SELL'
            signal_source: Source of the signal
            entry_price: Entry price
            target_price: Target price
            stop_loss: Stop loss price
            confidence: Signal confidence
            notes: Additional notes
            
        Returns:
            SignalRecord
        """
        import uuid
        
        signal = SignalRecord(
            id=str(uuid.uuid4())[:8],
            symbol=symbol.upper(),
            signal_type=signal_type.upper(),
            signal_source=signal_source,
            entry_price=entry_price,
            entry_date=datetime.now(),
            target_price=target_price,
            stop_loss=stop_loss,
            confidence=confidence,
            notes=notes
        )
        
        self.signals[signal.id] = signal
        self._save_signals()
        
        logger.info(f"Recorded signal: {signal_type} {symbol} @ ${entry_price:.2f}")
        return signal
    
    def close_signal(
        self,
        signal_id: str,
        exit_price: float,
        notes: str = ""
    ) -> Optional[SignalRecord]:
        """
        Close an open signal.
        
        Args:
            signal_id: Signal ID
            exit_price: Exit price
            notes: Exit notes
            
        Returns:
            Updated SignalRecord
        """
        signal = self.signals.get(signal_id)
        if not signal:
            return None
        
        signal.exit_price = exit_price
        signal.exit_date = datetime.now()
        
        # Calculate P&L
        if signal.signal_type == 'BUY':
            signal.pnl_pct = (exit_price - signal.entry_price) / signal.entry_price * 100
        else:  # SELL
            signal.pnl_pct = (signal.entry_price - exit_price) / signal.entry_price * 100
        
        # Determine outcome
        if abs(signal.pnl_pct) < 0.1:
            signal.outcome = SignalOutcome.BREAKEVEN
        elif signal.pnl_pct > 0:
            signal.outcome = SignalOutcome.WIN
        else:
            signal.outcome = SignalOutcome.LOSS
        
        if notes:
            signal.notes = f"{signal.notes} | Exit: {notes}" if signal.notes else notes
        
        self._save_signals()
        
        logger.info(f"Closed signal {signal_id}: {signal.outcome.value} ({signal.pnl_pct:.2f}%)")
        return signal
    
    def update_signal_prices(
        self,
        price_data: Dict[str, float]
    ) -> List[SignalRecord]:
        """
        Update pending signals with current prices.
        
        Args:
            price_data: Dict of symbol -> current price
            
        Returns:
            List of signals that hit targets or stops
        """
        triggered = []
        
        for signal in self.signals.values():
            if signal.outcome != SignalOutcome.PENDING:
                continue
            
            current_price = price_data.get(signal.symbol)
            if not current_price:
                continue
            
            # Check if target hit
            if signal.target_price:
                if signal.signal_type == 'BUY' and current_price >= signal.target_price:
                    self.close_signal(signal.id, signal.target_price, "Target hit")
                    triggered.append(signal)
                    continue
                elif signal.signal_type == 'SELL' and current_price <= signal.target_price:
                    self.close_signal(signal.id, signal.target_price, "Target hit")
                    triggered.append(signal)
                    continue
            
            # Check if stop hit
            if signal.stop_loss:
                if signal.signal_type == 'BUY' and current_price <= signal.stop_loss:
                    self.close_signal(signal.id, signal.stop_loss, "Stop loss hit")
                    triggered.append(signal)
                    continue
                elif signal.signal_type == 'SELL' and current_price >= signal.stop_loss:
                    self.close_signal(signal.id, signal.stop_loss, "Stop loss hit")
                    triggered.append(signal)
                    continue
        
        return triggered
    
    def calculate_metrics(
        self,
        signals: List[SignalRecord]
    ) -> PerformanceMetrics:
        """
        Calculate performance metrics for a set of signals.
        
        Args:
            signals: List of signals
            
        Returns:
            PerformanceMetrics
        """
        if not signals:
            return PerformanceMetrics(
                total_signals=0, winning_signals=0, losing_signals=0,
                pending_signals=0, win_rate=0, avg_win_pct=0, avg_loss_pct=0,
                profit_factor=0, expectancy=0, total_return_pct=0,
                max_consecutive_wins=0, max_consecutive_losses=0,
                avg_holding_period=0
            )
        
        closed = [s for s in signals if s.outcome != SignalOutcome.PENDING]
        pending = [s for s in signals if s.outcome == SignalOutcome.PENDING]
        
        winners = [s for s in closed if s.outcome == SignalOutcome.WIN]
        losers = [s for s in closed if s.outcome == SignalOutcome.LOSS]
        
        total = len(signals)
        win_count = len(winners)
        loss_count = len(losers)
        
        win_rate = (win_count / len(closed) * 100) if closed else 0
        
        avg_win = np.mean([s.pnl_pct for s in winners]) if winners else 0
        avg_loss = abs(np.mean([s.pnl_pct for s in losers])) if losers else 0
        
        total_wins = sum(s.pnl_pct for s in winners) if winners else 0
        total_losses = abs(sum(s.pnl_pct for s in losers)) if losers else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else (999.99 if total_wins > 0 else 0)
        
        # Expectancy
        if closed:
            expectancy = (win_rate/100 * avg_win) - ((1 - win_rate/100) * avg_loss)
        else:
            expectancy = 0
        
        total_return = sum(s.pnl_pct for s in closed)
        
        # Consecutive wins/losses
        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0
        
        sorted_signals = sorted(closed, key=lambda s: s.entry_date)
        for s in sorted_signals:
            if s.outcome == SignalOutcome.WIN:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            elif s.outcome == SignalOutcome.LOSS:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
            else:
                current_wins = 0
                current_losses = 0
        
        # Average holding period
        holding_periods = []
        for s in closed:
            if s.exit_date and s.entry_date:
                days = (s.exit_date - s.entry_date).days
                holding_periods.append(max(days, 1))
        avg_holding = np.mean(holding_periods) if holding_periods else 0
        
        # Best and worst signals
        best = max(closed, key=lambda s: s.pnl_pct) if closed else None
        worst = min(closed, key=lambda s: s.pnl_pct) if closed else None
        
        return PerformanceMetrics(
            total_signals=total,
            winning_signals=win_count,
            losing_signals=loss_count,
            pending_signals=len(pending),
            win_rate=win_rate,
            avg_win_pct=avg_win,
            avg_loss_pct=avg_loss,
            profit_factor=min(profit_factor, 999.99),
            expectancy=expectancy,
            total_return_pct=total_return,
            max_consecutive_wins=max_wins,
            max_consecutive_losses=max_losses,
            avg_holding_period=avg_holding,
            best_signal=best,
            worst_signal=worst
        )
    
    def generate_report_card(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> ReportCard:
        """
        Generate comprehensive performance report card.
        
        Args:
            start_date: Report start date
            end_date: Report end date
            
        Returns:
            ReportCard
        """
        now = datetime.now()
        start = start_date or (now - timedelta(days=90))
        end = end_date or now
        
        # Filter signals by date
        signals = [
            s for s in self.signals.values()
            if start <= s.entry_date <= end
        ]
        
        # Overall metrics
        overall = self.calculate_metrics(signals)
        
        # By symbol
        by_symbol = {}
        symbol_groups = defaultdict(list)
        for s in signals:
            symbol_groups[s.symbol].append(s)
        for symbol, group in symbol_groups.items():
            by_symbol[symbol] = self.calculate_metrics(group)
        
        # By source
        by_source = {}
        source_groups = defaultdict(list)
        for s in signals:
            source_groups[s.signal_source].append(s)
        for source, group in source_groups.items():
            by_source[source] = self.calculate_metrics(group)
        
        # By signal type
        by_type = {}
        type_groups = defaultdict(list)
        for s in signals:
            type_groups[s.signal_type].append(s)
        for sig_type, group in type_groups.items():
            by_type[sig_type] = self.calculate_metrics(group)
        
        # Monthly breakdown
        monthly = {}
        month_groups = defaultdict(list)
        for s in signals:
            month_key = s.entry_date.strftime('%Y-%m')
            month_groups[month_key].append(s)
        for month, group in sorted(month_groups.items()):
            monthly[month] = self.calculate_metrics(group)
        
        # Risk metrics
        closed = [s for s in signals if s.outcome != SignalOutcome.PENDING]
        returns = [s.pnl_pct for s in closed]
        
        risk_metrics = {
            'volatility': float(np.std(returns)) if returns else 0,
            'sharpe_ratio': float(np.mean(returns) / np.std(returns)) if returns and np.std(returns) > 0 else 0,
            'max_drawdown': float(min(returns)) if returns else 0,
            'best_month': max(monthly.items(), key=lambda x: x[1].total_return_pct)[0] if monthly else 'N/A',
            'worst_month': min(monthly.items(), key=lambda x: x[1].total_return_pct)[0] if monthly else 'N/A'
        }
        
        # Generate recommendations
        recommendations = self._generate_recommendations(overall, by_source, by_symbol)
        
        return ReportCard(
            period_start=start,
            period_end=end,
            overall_metrics=overall,
            by_symbol=by_symbol,
            by_source=by_source,
            by_signal_type=by_type,
            monthly_breakdown=monthly,
            risk_metrics=risk_metrics,
            recommendations=recommendations,
            generated_at=now
        )
    
    def _generate_recommendations(
        self,
        overall: PerformanceMetrics,
        by_source: Dict[str, PerformanceMetrics],
        by_symbol: Dict[str, PerformanceMetrics]
    ) -> List[str]:
        """Generate actionable recommendations."""
        recs = []
        
        # Win rate recommendations
        if overall.win_rate < 40:
            recs.append("⚠️ Win rate is below 40%. Consider tightening entry criteria.")
        elif overall.win_rate > 60:
            recs.append("✅ Strong win rate. Consider sizing up on high-confidence signals.")
        
        # Profit factor
        if overall.profit_factor < 1.0:
            recs.append("⚠️ Profit factor below 1.0 indicates losses exceed wins. Review stop loss strategy.")
        elif overall.profit_factor > 2.0:
            recs.append("✅ Excellent profit factor. Strategy is working well.")
        
        # Best sources
        if by_source:
            best_source = max(by_source.items(), key=lambda x: x[1].win_rate)
            worst_source = min(by_source.items(), key=lambda x: x[1].win_rate)
            
            if best_source[1].win_rate > 60:
                recs.append(f"💡 {best_source[0]} signals have {best_source[1].win_rate:.1f}% win rate. Consider increasing allocation.")
            
            if worst_source[1].win_rate < 40:
                recs.append(f"💡 {worst_source[0]} signals have low {worst_source[1].win_rate:.1f}% win rate. Review or reduce usage.")
        
        # Best symbols
        if by_symbol:
            sorted_symbols = sorted(by_symbol.items(), key=lambda x: x[1].total_return_pct, reverse=True)
            if len(sorted_symbols) >= 3:
                top_3 = [s[0] for s in sorted_symbols[:3]]
                recs.append(f"💡 Top performing symbols: {', '.join(top_3)}")
        
        # Risk management
        if overall.max_consecutive_losses >= 5:
            recs.append("⚠️ Had 5+ consecutive losses. Consider implementing cooldown rules.")
        
        if overall.avg_loss_pct > overall.avg_win_pct:
            recs.append("⚠️ Average loss exceeds average win. Tighten stop losses or widen targets.")
        
        if not recs:
            recs.append("📊 Performance within normal parameters. Continue monitoring.")
        
        return recs
    
    def export_to_csv(
        self,
        signals: Optional[List[SignalRecord]] = None
    ) -> str:
        """
        Export signals to CSV format.
        
        Args:
            signals: Optional list of signals (defaults to all)
            
        Returns:
            CSV string
        """
        signals = signals or list(self.signals.values())
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Header
        writer.writerow([
            'ID', 'Symbol', 'Type', 'Source', 'Entry Date', 'Entry Price',
            'Exit Date', 'Exit Price', 'Target', 'Stop Loss', 'Confidence',
            'Outcome', 'P&L %', 'Notes'
        ])
        
        # Data
        for s in sorted(signals, key=lambda x: x.entry_date, reverse=True):
            writer.writerow([
                s.id, s.symbol, s.signal_type, s.signal_source,
                s.entry_date.strftime('%Y-%m-%d %H:%M'),
                f'{s.entry_price:.2f}',
                s.exit_date.strftime('%Y-%m-%d %H:%M') if s.exit_date else '',
                f'{s.exit_price:.2f}' if s.exit_price else '',
                f'{s.target_price:.2f}' if s.target_price else '',
                f'{s.stop_loss:.2f}' if s.stop_loss else '',
                f'{s.confidence:.2f}',
                s.outcome.value,
                f'{s.pnl_pct:.2f}',
                s.notes
            ])
        
        return output.getvalue()
    
    def export_to_json(
        self,
        signals: Optional[List[SignalRecord]] = None
    ) -> str:
        """
        Export signals to JSON format.
        
        Args:
            signals: Optional list of signals (defaults to all)
            
        Returns:
            JSON string
        """
        signals = signals or list(self.signals.values())
        data = {
            'exported_at': datetime.now().isoformat(),
            'total_signals': len(signals),
            'signals': [s.to_dict() for s in signals]
        }
        return json.dumps(data, indent=2)
    
    def get_signal_history(
        self,
        symbol: Optional[str] = None,
        signal_source: Optional[str] = None,
        outcome: Optional[SignalOutcome] = None,
        limit: int = 50
    ) -> List[SignalRecord]:
        """
        Get signal history with optional filters.
        
        Args:
            symbol: Filter by symbol
            signal_source: Filter by source
            outcome: Filter by outcome
            limit: Maximum results
            
        Returns:
            List of SignalRecords
        """
        signals = list(self.signals.values())
        
        if symbol:
            signals = [s for s in signals if s.symbol.upper() == symbol.upper()]
        
        if signal_source:
            signals = [s for s in signals if s.signal_source == signal_source]
        
        if outcome:
            signals = [s for s in signals if s.outcome == outcome]
        
        return sorted(signals, key=lambda s: s.entry_date, reverse=True)[:limit]
    
    def get_open_signals(self) -> List[SignalRecord]:
        """Get all open/pending signals."""
        return [s for s in self.signals.values() if s.outcome == SignalOutcome.PENDING]
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get quick summary statistics."""
        all_signals = list(self.signals.values())
        metrics = self.calculate_metrics(all_signals)
        
        # Recent performance (last 30 days)
        recent_cutoff = datetime.now() - timedelta(days=30)
        recent = [s for s in all_signals if s.entry_date >= recent_cutoff]
        recent_metrics = self.calculate_metrics(recent) if recent else None
        
        return {
            'all_time': {
                'total_signals': metrics.total_signals,
                'win_rate': metrics.win_rate,
                'total_return': metrics.total_return_pct,
                'profit_factor': metrics.profit_factor
            },
            'last_30_days': {
                'total_signals': recent_metrics.total_signals if recent_metrics else 0,
                'win_rate': recent_metrics.win_rate if recent_metrics else 0,
                'total_return': recent_metrics.total_return_pct if recent_metrics else 0,
                'profit_factor': recent_metrics.profit_factor if recent_metrics else 0
            },
            'open_positions': len(self.get_open_signals()),
            'best_source': max(
                self._get_source_performance().items(),
                key=lambda x: x[1]['win_rate'],
                default=('N/A', {'win_rate': 0})
            )[0]
        }
    
    def _get_source_performance(self) -> Dict[str, Dict[str, float]]:
        """Get performance by signal source."""
        source_groups = defaultdict(list)
        for s in self.signals.values():
            source_groups[s.signal_source].append(s)
        
        result = {}
        for source, signals in source_groups.items():
            closed = [s for s in signals if s.outcome != SignalOutcome.PENDING]
            if closed:
                winners = len([s for s in closed if s.outcome == SignalOutcome.WIN])
                result[source] = {
                    'total': len(signals),
                    'win_rate': winners / len(closed) * 100,
                    'avg_return': np.mean([s.pnl_pct for s in closed])
                }
        
        return result
    
    def _save_signals(self):
        """Save signals to storage."""
        try:
            signals_file = self.storage_path / "signal_history.json"
            data = {
                'signals': [s.to_dict() for s in self.signals.values()]
            }
            
            with open(signals_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving signals: {e}")
    
    def _load_signals(self):
        """Load signals from storage."""
        try:
            signals_file = self.storage_path / "signal_history.json"
            
            if signals_file.exists():
                with open(signals_file, 'r') as f:
                    data = json.load(f)
                
                self.signals = {
                    s['id']: SignalRecord.from_dict(s)
                    for s in data.get('signals', [])
                }
                
                logger.info(f"Loaded {len(self.signals)} signals")
                
        except Exception as e:
            logger.error(f"Error loading signals: {e}")
            self.signals = {}
