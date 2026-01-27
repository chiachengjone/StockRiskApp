"""
Portfolio Store - SQLite-based Portfolio Persistence
=====================================================
Save, load, and manage user portfolios.
"""

import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional
import logging
import os


class PortfolioStore:
    """
    SQLite-based portfolio storage system.
    
    Features:
    - Save/load portfolios with custom names
    - Store weights, metadata, and notes
    - Track creation and modification dates
    - Export/import functionality
    """
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            base_dir = os.path.dirname(os.path.dirname(__file__))
            data_dir = os.path.join(base_dir, 'data')
            os.makedirs(data_dir, exist_ok=True)
            db_path = os.path.join(data_dir, 'portfolios.db')
        
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Portfolios table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolios (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                tickers TEXT NOT NULL,
                weights TEXT NOT NULL,
                benchmark TEXT DEFAULT '^GSPC',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                notes TEXT,
                tags TEXT,
                is_favorite INTEGER DEFAULT 0
            )
        ''')
        
        # Portfolio snapshots (historical performance)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                portfolio_id INTEGER NOT NULL,
                snapshot_date TEXT NOT NULL,
                total_value REAL,
                daily_return REAL,
                var_95 REAL,
                sharpe REAL,
                metrics TEXT,
                FOREIGN KEY (portfolio_id) REFERENCES portfolios(id) ON DELETE CASCADE
            )
        ''')
        
        # Analysis history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                analysis_date TEXT NOT NULL,
                analysis_type TEXT NOT NULL,
                results TEXT,
                notes TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_portfolio(
        self, 
        name: str, 
        tickers: List[str], 
        weights: List[float],
        benchmark: str = "^GSPC",
        notes: str = "",
        tags: List[str] = None
    ) -> bool:
        """
        Save or update a portfolio.
        
        Args:
            name: Portfolio name (unique identifier)
            tickers: List of ticker symbols
            weights: List of weights (should sum to 1.0)
            benchmark: Benchmark ticker
            notes: Optional notes
            tags: Optional list of tags
        
        Returns:
            True if successful
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        now = datetime.now().isoformat()
        
        try:
            # Check if portfolio exists
            cursor.execute('SELECT id, created_at FROM portfolios WHERE name = ?', (name,))
            existing = cursor.fetchone()
            
            if existing:
                # Update existing
                cursor.execute('''
                    UPDATE portfolios 
                    SET tickers = ?, weights = ?, benchmark = ?, 
                        updated_at = ?, notes = ?, tags = ?
                    WHERE name = ?
                ''', (
                    json.dumps(tickers),
                    json.dumps(weights),
                    benchmark,
                    now,
                    notes,
                    json.dumps(tags) if tags else None,
                    name
                ))
            else:
                # Insert new
                cursor.execute('''
                    INSERT INTO portfolios 
                    (name, tickers, weights, benchmark, created_at, updated_at, notes, tags)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    name,
                    json.dumps(tickers),
                    json.dumps(weights),
                    benchmark,
                    now,
                    now,
                    notes,
                    json.dumps(tags) if tags else None
                ))
            
            conn.commit()
            self.logger.info(f"Portfolio '{name}' saved successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving portfolio: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def load_portfolio(self, name: str) -> Optional[Dict]:
        """
        Load a portfolio by name.
        
        Returns:
            Dictionary with portfolio data, or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT id, name, tickers, weights, benchmark, 
                       created_at, updated_at, notes, tags, is_favorite
                FROM portfolios WHERE name = ?
            ''', (name,))
            
            row = cursor.fetchone()
            
            if row:
                return {
                    'id': row[0],
                    'name': row[1],
                    'tickers': json.loads(row[2]),
                    'weights': json.loads(row[3]),
                    'benchmark': row[4],
                    'created_at': row[5],
                    'updated_at': row[6],
                    'notes': row[7],
                    'tags': json.loads(row[8]) if row[8] else [],
                    'is_favorite': bool(row[9])
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error loading portfolio: {e}")
            return None
        finally:
            conn.close()
    
    def list_portfolios(self, favorites_only: bool = False) -> List[Dict]:
        """
        Get list of all saved portfolios.
        
        Args:
            favorites_only: If True, return only favorited portfolios
        
        Returns:
            List of portfolio summaries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            query = '''
                SELECT id, name, tickers, benchmark, 
                       created_at, updated_at, is_favorite
                FROM portfolios
            '''
            
            if favorites_only:
                query += ' WHERE is_favorite = 1'
            
            query += ' ORDER BY updated_at DESC'
            
            cursor.execute(query)
            rows = cursor.fetchall()
            
            portfolios = []
            for row in rows:
                tickers = json.loads(row[2])
                portfolios.append({
                    'id': row[0],
                    'name': row[1],
                    'tickers': tickers,
                    'ticker_count': len(tickers),
                    'benchmark': row[3],
                    'created_at': row[4],
                    'updated_at': row[5],
                    'is_favorite': bool(row[6])
                })
            
            return portfolios
            
        except Exception as e:
            self.logger.error(f"Error listing portfolios: {e}")
            return []
        finally:
            conn.close()
    
    def delete_portfolio(self, name: str) -> bool:
        """Delete a portfolio by name."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('DELETE FROM portfolios WHERE name = ?', (name,))
            conn.commit()
            deleted = cursor.rowcount > 0
            
            if deleted:
                self.logger.info(f"Portfolio '{name}' deleted")
            
            return deleted
            
        except Exception as e:
            self.logger.error(f"Error deleting portfolio: {e}")
            return False
        finally:
            conn.close()
    
    def toggle_favorite(self, name: str) -> bool:
        """Toggle favorite status of a portfolio."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                UPDATE portfolios 
                SET is_favorite = CASE WHEN is_favorite = 1 THEN 0 ELSE 1 END
                WHERE name = ?
            ''', (name,))
            conn.commit()
            return cursor.rowcount > 0
            
        except Exception as e:
            self.logger.error(f"Error toggling favorite: {e}")
            return False
        finally:
            conn.close()
    
    def save_snapshot(
        self, 
        portfolio_name: str, 
        total_value: float = None,
        daily_return: float = None,
        var_95: float = None,
        sharpe: float = None,
        metrics: Dict = None
    ) -> bool:
        """
        Save a performance snapshot for a portfolio.
        
        Useful for tracking portfolio performance over time.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get portfolio ID
            cursor.execute('SELECT id FROM portfolios WHERE name = ?', (portfolio_name,))
            result = cursor.fetchone()
            
            if not result:
                return False
            
            portfolio_id = result[0]
            now = datetime.now().isoformat()
            
            cursor.execute('''
                INSERT INTO portfolio_snapshots
                (portfolio_id, snapshot_date, total_value, daily_return, var_95, sharpe, metrics)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                portfolio_id,
                now,
                total_value,
                daily_return,
                var_95,
                sharpe,
                json.dumps(metrics) if metrics else None
            ))
            
            conn.commit()
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving snapshot: {e}")
            return False
        finally:
            conn.close()
    
    def get_snapshots(self, portfolio_name: str, limit: int = 30) -> List[Dict]:
        """Get historical snapshots for a portfolio."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT s.snapshot_date, s.total_value, s.daily_return, 
                       s.var_95, s.sharpe, s.metrics
                FROM portfolio_snapshots s
                JOIN portfolios p ON s.portfolio_id = p.id
                WHERE p.name = ?
                ORDER BY s.snapshot_date DESC
                LIMIT ?
            ''', (portfolio_name, limit))
            
            rows = cursor.fetchall()
            
            return [{
                'date': row[0],
                'total_value': row[1],
                'daily_return': row[2],
                'var_95': row[3],
                'sharpe': row[4],
                'metrics': json.loads(row[5]) if row[5] else {}
            } for row in rows]
            
        except Exception as e:
            self.logger.error(f"Error getting snapshots: {e}")
            return []
        finally:
            conn.close()
    
    def save_analysis(
        self, 
        ticker: str, 
        analysis_type: str,
        results: Dict,
        notes: str = ""
    ) -> bool:
        """Save an analysis result for future reference."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            now = datetime.now().isoformat()
            
            cursor.execute('''
                INSERT INTO analysis_history
                (ticker, analysis_date, analysis_type, results, notes)
                VALUES (?, ?, ?, ?, ?)
            ''', (ticker, now, analysis_type, json.dumps(results), notes))
            
            conn.commit()
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving analysis: {e}")
            return False
        finally:
            conn.close()
    
    def get_analysis_history(self, ticker: str = None, limit: int = 50) -> List[Dict]:
        """Get analysis history, optionally filtered by ticker."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            if ticker:
                cursor.execute('''
                    SELECT ticker, analysis_date, analysis_type, results, notes
                    FROM analysis_history
                    WHERE ticker = ?
                    ORDER BY analysis_date DESC
                    LIMIT ?
                ''', (ticker, limit))
            else:
                cursor.execute('''
                    SELECT ticker, analysis_date, analysis_type, results, notes
                    FROM analysis_history
                    ORDER BY analysis_date DESC
                    LIMIT ?
                ''', (limit,))
            
            rows = cursor.fetchall()
            
            return [{
                'ticker': row[0],
                'date': row[1],
                'type': row[2],
                'results': json.loads(row[3]) if row[3] else {},
                'notes': row[4]
            } for row in rows]
            
        except Exception as e:
            self.logger.error(f"Error getting analysis history: {e}")
            return []
        finally:
            conn.close()
    
    def export_portfolio(self, name: str) -> Optional[Dict]:
        """
        Export portfolio to a shareable format.
        
        Returns:
            Dictionary that can be JSON-serialized
        """
        portfolio = self.load_portfolio(name)
        if not portfolio:
            return None
        
        # Remove internal fields
        export_data = {
            'name': portfolio['name'],
            'tickers': portfolio['tickers'],
            'weights': portfolio['weights'],
            'benchmark': portfolio['benchmark'],
            'notes': portfolio['notes'],
            'tags': portfolio['tags'],
            'exported_at': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        return export_data
    
    def import_portfolio(self, data: Dict, new_name: str = None) -> bool:
        """
        Import a portfolio from exported data.
        
        Args:
            data: Exported portfolio data
            new_name: Optional new name (uses original if not provided)
        
        Returns:
            True if successful
        """
        try:
            name = new_name or data.get('name', 'Imported Portfolio')
            
            return self.save_portfolio(
                name=name,
                tickers=data['tickers'],
                weights=data['weights'],
                benchmark=data.get('benchmark', '^GSPC'),
                notes=data.get('notes', f"Imported on {datetime.now().strftime('%Y-%m-%d')}"),
                tags=data.get('tags', [])
            )
            
        except Exception as e:
            self.logger.error(f"Error importing portfolio: {e}")
            return False
    
    def get_portfolio_names(self) -> List[str]:
        """Get just the names of all portfolios."""
        portfolios = self.list_portfolios()
        return [p['name'] for p in portfolios]
