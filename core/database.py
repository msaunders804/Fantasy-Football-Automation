import sqlite3
import json
import os
from typing import List, Dict, Any, Optional
from contextlib import contextmanager

class DatabaseManager:
    def __init__(self, db_path: str = None):
        if db_path is None:
            from config.settings import DATABASE_PATH
            db_path = DATABASE_PATH
            
        self.db_path = db_path
        self._ensure_db_directory()
        self._initialize_database()
    
    def _ensure_db_directory(self):
        """Create database directory if it doesn't exist"""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
    
    def _initialize_database(self):
        """Initialize database with schema"""
        schema_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'database_schema.sql')
        
        with open(schema_path, 'r') as f:
            schema_sql = f.read()
            
        with self.get_connection() as conn:
            conn.executescript(schema_sql)
            conn.commit()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
        finally:
            conn.close()
    
    def execute_query(self, query: str, params: tuple = ()) -> List[Dict]:
        """Execute SELECT query and return results"""
        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def execute_update(self, query: str, params: tuple = ()) -> int:
        """Execute INSERT/UPDATE/DELETE query and return affected rows"""
        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            conn.commit()
            return cursor.rowcount
    
    def execute_many(self, query: str, params_list: List[tuple]) -> int:
        """Execute query with multiple parameter sets"""
        with self.get_connection() as conn:
            cursor = conn.executemany(query, params_list)
            conn.commit()
            return cursor.rowcount
    
    # Player methods
    def upsert_player(self, player_id: str, name: str, position: str, team: str = None, bye_week: int = None):
        """Insert or update player"""
        query = """
        INSERT INTO players (player_id, name, position, team, bye_week) 
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(player_id) DO UPDATE SET
            name = excluded.name,
            position = excluded.position,
            team = excluded.team,
            bye_week = excluded.bye_week
        """
        return self.execute_update(query, (player_id, name, position, team, bye_week))
    
    def get_players(self, position: str = None) -> List[Dict]:
        """Get players, optionally filtered by position"""
        if position:
            query = "SELECT * FROM players WHERE position = ? ORDER BY name"
            return self.execute_query(query, (position,))
        else:
            query = "SELECT * FROM players ORDER BY position, name"
            return self.execute_query(query)
    
    # Projection methods
    def add_projection_source(self, name: str, weight: float = 1.0) -> int:
        """Add or update projection source"""
        query = """
        INSERT INTO projection_sources (name, weight) VALUES (?, ?)
        ON CONFLICT(name) DO UPDATE SET weight = excluded.weight
        """
        self.execute_update(query, (name, weight))
        
        # Return the source_id
        result = self.execute_query("SELECT source_id FROM projection_sources WHERE name = ?", (name,))
        return result[0]['source_id'] if result else None
    
    def bulk_insert_projections(self, projections: List[Dict]):
        """Bulk insert projections"""
        query = """
        INSERT INTO projections 
        (player_id, source_id, week, points, pass_yds, pass_tds, interceptions,
         rush_yds, rush_tds, receptions, rec_yds, rec_tds)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params_list = [
            (p['player_id'], p['source_id'], p.get('week', 0), p.get('points'),
             p.get('pass_yds', 0), p.get('pass_tds', 0), p.get('interceptions', 0),
             p.get('rush_yds', 0), p.get('rush_tds', 0), p.get('receptions', 0),
             p.get('rec_yds', 0), p.get('rec_tds', 0))
            for p in projections
        ]
        return self.execute_many(query, params_list)
    
    def get_projections(self, player_id: str = None, source_id: int = None) -> List[Dict]:
        """Get projections with optional filters"""
        query = """
        SELECT p.*, ps.name as source_name, pl.name as player_name, pl.position
        FROM projections p
        JOIN projection_sources ps ON p.source_id = ps.source_id
        JOIN players pl ON p.player_id = pl.player_id
        WHERE 1=1
        """
        params = []
        
        if player_id:
            query += " AND p.player_id = ?"
            params.append(player_id)
        if source_id:
            query += " AND p.source_id = ?"
            params.append(source_id)
            
        query += " ORDER BY p.points DESC"
        return self.execute_query(query, tuple(params))