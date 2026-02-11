"""
Data Quality & Validation Module
==================================
Automated data quality checks, outlier detection, and validation.

Features:
- Automated outlier detection and correction
- Data completeness checks
- Data source comparison and cross-validation
- Market hours validation
- Missing data handling

Author: Stock Risk App | Feb 2026
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import warnings

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class DataQualityLevel(Enum):
    """Data quality levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    CRITICAL = "critical"


class OutlierMethod(Enum):
    """Methods for outlier detection."""
    ZSCORE = "zscore"
    IQR = "iqr"
    MAD = "mad"  # Median Absolute Deviation
    ISOLATION_FOREST = "isolation_forest"
    WINSORIZE = "winsorize"


@dataclass
class DataQualityReport:
    """Comprehensive data quality report."""
    ticker: str
    quality_level: DataQualityLevel
    quality_score: float  # 0-100
    total_records: int
    date_range: Tuple[datetime, datetime]
    issues: List[Dict[str, Any]]
    completeness: float  # 0-1
    missing_dates: List[datetime]
    outliers_detected: int
    stale_data: bool
    recommendations: List[str]


@dataclass
class OutlierReport:
    """Report on detected outliers."""
    total_outliers: int
    outlier_indices: List[int]
    outlier_dates: List[datetime]
    outlier_values: List[float]
    method_used: str
    threshold: float
    impact_on_metrics: Dict[str, float]


@dataclass
class DataComparisonResult:
    """Result of comparing data across sources."""
    source1: str
    source2: str
    records_compared: int
    matching_records: int
    discrepancy_rate: float
    significant_discrepancies: List[Dict]
    recommended_source: str


# ============================================================================
# DATA QUALITY CHECKER
# ============================================================================

class DataQualityChecker:
    """
    Comprehensive data quality validation and checking.
    """
    
    # Expected trading days per year (US markets)
    TRADING_DAYS_PER_YEAR = 252
    
    # Price change thresholds for anomaly detection
    MAX_DAILY_CHANGE = 0.50  # 50% daily change is suspicious
    MAX_GAP = 0.30  # 30% overnight gap is suspicious
    
    def __init__(self):
        self.issues_found = []
    
    def validate(
        self,
        data: pd.DataFrame,
        ticker: str,
        check_completeness: bool = True,
        check_outliers: bool = True,
        check_staleness: bool = True,
        check_consistency: bool = True
    ) -> DataQualityReport:
        """
        Perform comprehensive data quality validation.
        
        Args:
            data: OHLCV DataFrame
            ticker: Stock symbol
            check_*: Flags to enable/disable specific checks
            
        Returns:
            DataQualityReport object
        """
        self.issues_found = []
        
        if data.empty:
            return DataQualityReport(
                ticker=ticker,
                quality_level=DataQualityLevel.CRITICAL,
                quality_score=0,
                total_records=0,
                date_range=(datetime.now(), datetime.now()),
                issues=[{'type': 'no_data', 'message': 'No data available'}],
                completeness=0,
                missing_dates=[],
                outliers_detected=0,
                stale_data=True,
                recommendations=["Verify ticker symbol and data source"]
            )
        
        # Ensure datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            try:
                data.index = pd.to_datetime(data.index)
            except:
                self.issues_found.append({
                    'type': 'index_error',
                    'message': 'Could not convert index to datetime'
                })
        
        # Base metrics
        total_records = len(data)
        start_date = data.index.min()
        end_date = data.index.max()
        
        # Completeness check
        completeness = 1.0
        missing_dates = []
        if check_completeness:
            completeness, missing_dates = self._check_completeness(data)
        
        # Outlier detection
        outliers_detected = 0
        if check_outliers:
            outliers_detected = self._check_outliers(data)
        
        # Staleness check
        stale_data = False
        if check_staleness:
            stale_data = self._check_staleness(data)
        
        # Consistency check
        if check_consistency:
            self._check_consistency(data)
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(
            completeness, outliers_detected, total_records, stale_data
        )
        
        # Determine quality level
        if quality_score >= 90:
            quality_level = DataQualityLevel.EXCELLENT
        elif quality_score >= 75:
            quality_level = DataQualityLevel.GOOD
        elif quality_score >= 60:
            quality_level = DataQualityLevel.ACCEPTABLE
        elif quality_score >= 40:
            quality_level = DataQualityLevel.POOR
        else:
            quality_level = DataQualityLevel.CRITICAL
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            completeness, outliers_detected, stale_data, quality_level
        )
        
        return DataQualityReport(
            ticker=ticker,
            quality_level=quality_level,
            quality_score=quality_score,
            total_records=total_records,
            date_range=(start_date, end_date),
            issues=self.issues_found,
            completeness=completeness,
            missing_dates=missing_dates,
            outliers_detected=outliers_detected,
            stale_data=stale_data,
            recommendations=recommendations
        )
    
    def _check_completeness(
        self,
        data: pd.DataFrame
    ) -> Tuple[float, List[datetime]]:
        """Check data completeness for expected trading days."""
        try:
            # Generate expected trading days
            start = data.index.min()
            end = data.index.max()
            
            # Create business day range (rough approximation)
            expected_dates = pd.date_range(start=start, end=end, freq='B')
            
            # Account for holidays (approximately)
            expected_trading_days = len(expected_dates) * 0.96  # ~4% holidays
            
            actual_days = len(data)
            completeness = min(1.0, actual_days / expected_trading_days)
            
            # Find missing dates
            actual_dates = set(data.index.date)
            expected_date_set = set(expected_dates.date)
            missing = expected_date_set - actual_dates
            missing_dates = sorted([datetime.combine(d, datetime.min.time()) for d in missing])[:20]  # Top 20
            
            if completeness < 0.9:
                self.issues_found.append({
                    'type': 'incomplete_data',
                    'message': f'Data completeness is {completeness*100:.1f}%',
                    'severity': 'warning' if completeness > 0.7 else 'error'
                })
            
            return completeness, missing_dates
            
        except Exception as e:
            logger.warning(f"Completeness check error: {e}")
            return 1.0, []
    
    def _check_outliers(self, data: pd.DataFrame) -> int:
        """Check for price outliers using multiple methods."""
        outliers = 0
        
        if 'Close' not in data.columns:
            return 0
        
        # Calculate returns
        returns = data['Close'].pct_change().dropna()
        
        # Z-score method
        z_scores = np.abs((returns - returns.mean()) / returns.std())
        z_outliers = (z_scores > 4).sum()  # More than 4 std
        
        if z_outliers > 0:
            self.issues_found.append({
                'type': 'extreme_returns',
                'message': f'{z_outliers} days with returns > 4 standard deviations',
                'severity': 'warning'
            })
            outliers += z_outliers
        
        # Check for suspicious price changes
        large_changes = (returns.abs() > self.MAX_DAILY_CHANGE).sum()
        if large_changes > 0:
            self.issues_found.append({
                'type': 'large_price_changes',
                'message': f'{large_changes} days with >50% price changes',
                'severity': 'error'
            })
            outliers += large_changes
        
        # Check for OHLC consistency
        if all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
            # High should be >= Open, Close, Low
            invalid_high = (data['High'] < data[['Open', 'Close', 'Low']].max(axis=1)).sum()
            # Low should be <= Open, Close, High
            invalid_low = (data['Low'] > data[['Open', 'Close', 'High']].min(axis=1)).sum()
            
            if invalid_high > 0 or invalid_low > 0:
                self.issues_found.append({
                    'type': 'ohlc_inconsistency',
                    'message': f'{invalid_high + invalid_low} records with invalid OHLC relationships',
                    'severity': 'error'
                })
                outliers += invalid_high + invalid_low
        
        # Check for zero or negative prices
        if 'Close' in data.columns:
            invalid_prices = (data['Close'] <= 0).sum()
            if invalid_prices > 0:
                self.issues_found.append({
                    'type': 'invalid_prices',
                    'message': f'{invalid_prices} records with zero or negative prices',
                    'severity': 'critical'
                })
                outliers += invalid_prices
        
        return outliers
    
    def _check_staleness(self, data: pd.DataFrame) -> bool:
        """Check if data is stale (not recent)."""
        if data.empty:
            return True
        
        last_date = data.index.max()
        today = datetime.now()
        
        # Account for weekends
        days_since = (today - last_date).days
        
        # If today is Monday, data from Friday (2-3 days ago) is fine
        weekday = today.weekday()
        acceptable_delay = 1 if weekday < 5 else (3 if weekday == 0 else 2)
        
        is_stale = days_since > acceptable_delay + 1
        
        if is_stale:
            self.issues_found.append({
                'type': 'stale_data',
                'message': f'Data is {days_since} days old (last: {last_date.strftime("%Y-%m-%d")})',
                'severity': 'warning'
            })
        
        return is_stale
    
    def _check_consistency(self, data: pd.DataFrame):
        """Check for data consistency issues."""
        # Check for duplicate dates
        duplicates = data.index.duplicated().sum()
        if duplicates > 0:
            self.issues_found.append({
                'type': 'duplicate_dates',
                'message': f'{duplicates} duplicate date entries found',
                'severity': 'error'
            })
        
        # Check for chronological order
        if not data.index.is_monotonic_increasing:
            self.issues_found.append({
                'type': 'non_chronological',
                'message': 'Data is not in chronological order',
                'severity': 'warning'
            })
        
        # Check for NaN values
        nan_counts = data.isnull().sum()
        total_nans = nan_counts.sum()
        if total_nans > 0:
            self.issues_found.append({
                'type': 'missing_values',
                'message': f'{total_nans} missing values across {(nan_counts > 0).sum()} columns',
                'severity': 'warning'
            })
        
        # Check volume consistency
        if 'Volume' in data.columns:
            zero_volume = (data['Volume'] == 0).sum()
            if zero_volume > len(data) * 0.1:  # More than 10% zero volume
                self.issues_found.append({
                    'type': 'zero_volume',
                    'message': f'{zero_volume} days with zero volume ({zero_volume/len(data)*100:.1f}%)',
                    'severity': 'warning'
                })
    
    def _calculate_quality_score(
        self,
        completeness: float,
        outliers: int,
        total_records: int,
        stale: bool
    ) -> float:
        """Calculate overall quality score (0-100)."""
        score = 100.0
        
        # Penalize for incompleteness
        score -= (1 - completeness) * 30
        
        # Penalize for outliers
        outlier_rate = outliers / total_records if total_records > 0 else 0
        score -= min(30, outlier_rate * 100)
        
        # Penalize for staleness
        if stale:
            score -= 15
        
        # Penalize for issues
        for issue in self.issues_found:
            if issue.get('severity') == 'critical':
                score -= 20
            elif issue.get('severity') == 'error':
                score -= 10
            elif issue.get('severity') == 'warning':
                score -= 5
        
        return max(0, min(100, score))
    
    def _generate_recommendations(
        self,
        completeness: float,
        outliers: int,
        stale: bool,
        quality_level: DataQualityLevel
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        if completeness < 0.9:
            recommendations.append("Consider using a different data source for more complete data")
        
        if outliers > 0:
            recommendations.append("Review and potentially filter outliers before analysis")
        
        if stale:
            recommendations.append("Refresh data - current data may not reflect recent market activity")
        
        if quality_level in [DataQualityLevel.POOR, DataQualityLevel.CRITICAL]:
            recommendations.append("Data quality issues may significantly impact analysis accuracy")
            recommendations.append("Consider cross-validating with alternative data sources")
        
        if not recommendations:
            recommendations.append("Data quality is acceptable for analysis")
        
        return recommendations


# ============================================================================
# OUTLIER DETECTOR
# ============================================================================

class OutlierDetector:
    """
    Advanced outlier detection and handling.
    """
    
    def detect(
        self,
        data: Union[pd.Series, pd.DataFrame],
        method: OutlierMethod = OutlierMethod.ZSCORE,
        threshold: float = 3.0,
        column: str = 'Close'
    ) -> OutlierReport:
        """
        Detect outliers in data.
        
        Args:
            data: Data to analyze
            method: Detection method
            threshold: Threshold for outlier detection
            column: Column to analyze if DataFrame
            
        Returns:
            OutlierReport object
        """
        if isinstance(data, pd.DataFrame):
            if column not in data.columns:
                raise ValueError(f"Column {column} not found")
            series = data[column]
        else:
            series = data
        
        # Calculate returns for analysis
        returns = series.pct_change().dropna()
        
        if method == OutlierMethod.ZSCORE:
            outlier_mask = self._zscore_detection(returns, threshold)
        elif method == OutlierMethod.IQR:
            outlier_mask = self._iqr_detection(returns, threshold)
        elif method == OutlierMethod.MAD:
            outlier_mask = self._mad_detection(returns, threshold)
        else:
            outlier_mask = self._zscore_detection(returns, threshold)
        
        # Get outlier details
        outlier_indices = returns.index[outlier_mask].tolist()
        outlier_values = returns[outlier_mask].tolist()
        
        # Calculate impact
        impact = self._calculate_outlier_impact(returns, outlier_mask)
        
        return OutlierReport(
            total_outliers=outlier_mask.sum(),
            outlier_indices=list(range(len(outlier_indices))),
            outlier_dates=outlier_indices,
            outlier_values=outlier_values,
            method_used=method.value,
            threshold=threshold,
            impact_on_metrics=impact
        )
    
    def _zscore_detection(
        self,
        data: pd.Series,
        threshold: float
    ) -> pd.Series:
        """Z-score based outlier detection."""
        mean = data.mean()
        std = data.std()
        z_scores = np.abs((data - mean) / std)
        return z_scores > threshold
    
    def _iqr_detection(
        self,
        data: pd.Series,
        threshold: float = 1.5
    ) -> pd.Series:
        """IQR based outlier detection."""
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        return (data < lower) | (data > upper)
    
    def _mad_detection(
        self,
        data: pd.Series,
        threshold: float = 3.0
    ) -> pd.Series:
        """Median Absolute Deviation based outlier detection."""
        median = data.median()
        mad = np.median(np.abs(data - median))
        modified_z = 0.6745 * (data - median) / mad if mad > 0 else pd.Series(0, index=data.index)
        return np.abs(modified_z) > threshold
    
    def _calculate_outlier_impact(
        self,
        data: pd.Series,
        outlier_mask: pd.Series
    ) -> Dict[str, float]:
        """Calculate impact of outliers on key metrics."""
        clean_data = data[~outlier_mask]
        
        return {
            'mean_with_outliers': float(data.mean()),
            'mean_without_outliers': float(clean_data.mean()),
            'std_with_outliers': float(data.std()),
            'std_without_outliers': float(clean_data.std()),
            'variance_reduction': float(1 - clean_data.var() / data.var()) if data.var() > 0 else 0
        }
    
    def handle_outliers(
        self,
        data: pd.DataFrame,
        method: str = 'winsorize',
        column: str = 'Close',
        threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        Handle outliers by correcting/removing them.
        
        Args:
            data: DataFrame with outliers
            method: 'winsorize', 'remove', 'interpolate', 'clip'
            column: Column to fix
            threshold: Detection threshold
            
        Returns:
            Cleaned DataFrame
        """
        df = data.copy()
        
        if column not in df.columns:
            return df
        
        returns = df[column].pct_change()
        mean = returns.mean()
        std = returns.std()
        
        if method == 'winsorize':
            # Cap extreme values at threshold * std
            upper = mean + threshold * std
            lower = mean - threshold * std
            returns = returns.clip(lower=lower, upper=upper)
            # Reconstruct prices
            df[column] = (1 + returns.fillna(0)).cumprod() * df[column].iloc[0]
            
        elif method == 'remove':
            # Remove rows with outliers
            z_scores = np.abs((returns - mean) / std)
            df = df[z_scores <= threshold]
            
        elif method == 'interpolate':
            # Replace outliers with interpolated values
            z_scores = np.abs((returns - mean) / std)
            outlier_mask = z_scores > threshold
            df.loc[outlier_mask, column] = np.nan
            df[column] = df[column].interpolate(method='linear')
            
        elif method == 'clip':
            # Clip prices directly
            lower = df[column].quantile(0.01)
            upper = df[column].quantile(0.99)
            df[column] = df[column].clip(lower=lower, upper=upper)
        
        return df


# ============================================================================
# DATA SOURCE COMPARATOR
# ============================================================================

class DataSourceComparator:
    """
    Compare and validate data across multiple sources.
    """
    
    def compare(
        self,
        data1: pd.DataFrame,
        data2: pd.DataFrame,
        source1_name: str = "Source 1",
        source2_name: str = "Source 2",
        tolerance: float = 0.01
    ) -> DataComparisonResult:
        """
        Compare data from two sources.
        
        Args:
            data1, data2: DataFrames to compare
            source1_name, source2_name: Names for reporting
            tolerance: Acceptable difference threshold (1%)
            
        Returns:
            DataComparisonResult object
        """
        # Align dates
        common_dates = data1.index.intersection(data2.index)
        
        if len(common_dates) == 0:
            return DataComparisonResult(
                source1=source1_name,
                source2=source2_name,
                records_compared=0,
                matching_records=0,
                discrepancy_rate=1.0,
                significant_discrepancies=[],
                recommended_source="Cannot compare - no overlapping dates"
            )
        
        d1 = data1.loc[common_dates]
        d2 = data2.loc[common_dates]
        
        # Compare close prices
        if 'Close' in d1.columns and 'Close' in d2.columns:
            diff = (d1['Close'] - d2['Close']).abs() / d1['Close']
            matching = (diff <= tolerance).sum()
            discrepancies = diff[diff > tolerance]
            
            significant = []
            for date, disc in discrepancies.head(10).items():
                significant.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'source1_value': float(d1.loc[date, 'Close']),
                    'source2_value': float(d2.loc[date, 'Close']),
                    'difference_pct': float(disc * 100)
                })
            
            discrepancy_rate = 1 - (matching / len(common_dates))
            
            # Recommend source based on data quality
            # Prefer source with more data and less volatility in discrepancies
            if len(data1) > len(data2) * 1.05:
                recommended = source1_name
            elif len(data2) > len(data1) * 1.05:
                recommended = source2_name
            else:
                recommended = f"Both sources comparable"
                
        else:
            matching = 0
            discrepancy_rate = 1.0
            significant = []
            recommended = "Cannot compare - missing Close column"
        
        return DataComparisonResult(
            source1=source1_name,
            source2=source2_name,
            records_compared=len(common_dates),
            matching_records=matching,
            discrepancy_rate=discrepancy_rate,
            significant_discrepancies=significant,
            recommended_source=recommended
        )


# ============================================================================
# MISSING DATA HANDLER
# ============================================================================

class MissingDataHandler:
    """
    Handle missing data in financial time series.
    """
    
    def fill_missing(
        self,
        data: pd.DataFrame,
        method: str = 'ffill',
        max_gap: int = 5
    ) -> pd.DataFrame:
        """
        Fill missing values in data.
        
        Args:
            data: DataFrame with missing values
            method: 'ffill', 'bfill', 'interpolate', 'mean'
            max_gap: Maximum consecutive missing values to fill
            
        Returns:
            Filled DataFrame
        """
        df = data.copy()
        
        if method == 'ffill':
            df = df.fillna(method='ffill', limit=max_gap)
        elif method == 'bfill':
            df = df.fillna(method='bfill', limit=max_gap)
        elif method == 'interpolate':
            df = df.interpolate(method='time', limit=max_gap)
        elif method == 'mean':
            for col in df.columns:
                df[col] = df[col].fillna(df[col].rolling(max_gap, min_periods=1).mean())
        
        return df
    
    def identify_gaps(
        self,
        data: pd.DataFrame,
        expected_freq: str = 'B'
    ) -> List[Dict]:
        """
        Identify gaps in time series data.
        
        Args:
            data: DataFrame to analyze
            expected_freq: Expected frequency ('B' for business days)
            
        Returns:
            List of gap information dictionaries
        """
        gaps = []
        
        expected_index = pd.date_range(
            start=data.index.min(),
            end=data.index.max(),
            freq=expected_freq
        )
        
        actual_dates = set(data.index.date)
        
        current_gap_start = None
        current_gap_length = 0
        
        for date in expected_index:
            if date.date() not in actual_dates:
                if current_gap_start is None:
                    current_gap_start = date
                current_gap_length += 1
            else:
                if current_gap_start is not None:
                    gaps.append({
                        'start': current_gap_start,
                        'end': date - timedelta(days=1),
                        'length': current_gap_length
                    })
                    current_gap_start = None
                    current_gap_length = 0
        
        return gaps


# ============================================================================
# STREAMLIT RENDERING
# ============================================================================

def render_data_quality_dashboard(data: pd.DataFrame, ticker: str):
    """Render data quality dashboard in Streamlit."""
    import streamlit as st
    import plotly.graph_objects as go
    
    st.subheader(" Data Quality Report")
    
    checker = DataQualityChecker()
    report = checker.validate(data, ticker)
    
    # Quality score gauge
    quality_colors = {
        DataQualityLevel.EXCELLENT: '#00C853',
        DataQualityLevel.GOOD: '#4CAF50',
        DataQualityLevel.ACCEPTABLE: '#FFC107',
        DataQualityLevel.POOR: '#FF5722',
        DataQualityLevel.CRITICAL: '#FF0000'
    }
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=report.quality_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Quality Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': quality_colors[report.quality_level]},
                'steps': [
                    {'range': [0, 40], 'color': '#FF000022'},
                    {'range': [40, 60], 'color': '#FF572222'},
                    {'range': [60, 75], 'color': '#FFC10722'},
                    {'range': [75, 90], 'color': '#4CAF5022'},
                    {'range': [90, 100], 'color': '#00C85322'}
                ]
            }
        ))
        fig.update_layout(height=250, template='plotly_dark')
        st.plotly_chart(fig, width="stretch")
    
    with col2:
        st.markdown(f"""
        <div style="padding: 15px; border-radius: 10px; background: {quality_colors[report.quality_level]}22; 
                    border: 2px solid {quality_colors[report.quality_level]};">
            <h3 style="margin: 0;">Quality Level: {report.quality_level.value.upper()}</h3>
            <p>Records: {report.total_records} | 
               Completeness: {report.completeness*100:.1f}% | 
               Outliers: {report.outliers_detected}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Date range
        st.write(f"**Date Range:** {report.date_range[0].strftime('%Y-%m-%d')} to {report.date_range[1].strftime('%Y-%m-%d')}")
        
        if report.stale_data:
            st.warning(" Data appears to be stale")
    
    # Issues
    if report.issues:
        st.write("### Issues Detected")
        for issue in report.issues:
            severity_icon = {
                'critical': '',
                'error': '🟠',
                'warning': '🟡'
            }.get(issue.get('severity', 'warning'), '🟡')
            
            st.write(f"{severity_icon} **{issue['type']}**: {issue['message']}")
    
    # Recommendations
    if report.recommendations:
        st.write("### Recommendations")
        for rec in report.recommendations:
            st.write(f" {rec}")


# Make features available
HAS_DATA_QUALITY = True
