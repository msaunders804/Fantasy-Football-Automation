# database_setup.py
"""
SQLite Database Schema and Setup for Fantasy Football Analytics
Creates normalized tables for efficient data storage and retrieval
"""

import sqlite3
import os
from datetime import datetime
import logging

class FantasyDatabase:
    """Handles all database operations for fantasy football data"""
    
    def __init__(self, db_path="fantasy_football.db"):
        self.db_path = db_path
        self.setup_logging()
        self.connection = None
        
    def setup_logging(self):
        """Configure logging for database operations"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('fantasy_db.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def connect(self):
        """Connect to SQLite database"""
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row  # Enable column access by name
            self.logger.info(f"Connected to database: {self.db_path}")
            return True
        except sqlite3.Error as e:
            self.logger.error(f"Database connection error: {e}")
            return False
    
    def disconnect(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            self.logger.info("Database connection closed")
    
    def create_tables(self):
        """Create all necessary tables for fantasy football data"""
        
        if not self.connection:
            self.logger.error("No database connection")
            return False
        
        cursor = self.connection.cursor()
        
        try:
            # Players table - Core player information
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS players (
                    player_id TEXT PRIMARY KEY,
                    sleeper_id TEXT UNIQUE,
                    first_name TEXT,
                    last_name TEXT,
                    full_name TEXT,
                    position TEXT,
                    team TEXT,
                    age INTEGER,
                    height TEXT,
                    weight INTEGER,
                    years_exp INTEGER,
                    college TEXT,
                    status TEXT,  -- Active, Inactive, IR, etc.
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Teams table - NFL team information
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS teams (
                    team_id TEXT PRIMARY KEY,
                    team_abbr TEXT UNIQUE,
                    team_name TEXT,
                    division TEXT,
                    conference TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Seasons table - Track different seasons
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS seasons (
                    season_id INTEGER PRIMARY KEY,
                    year INTEGER UNIQUE,
                    start_date DATE,
                    end_date DATE,
                    playoff_start_week INTEGER,
                    total_weeks INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Games table - Individual NFL games
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS games (
                    game_id TEXT PRIMARY KEY,
                    season_year INTEGER,
                    week INTEGER,
                    game_type TEXT,  -- REG, WC, DIV, CONF, SB
                    home_team TEXT,
                    away_team TEXT,
                    home_score INTEGER,
                    away_score INTEGER,
                    game_date DATE,
                    game_time TIME,
                    weather_conditions TEXT,
                    temperature REAL,
                    wind_speed REAL,
                    dome BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (season_year) REFERENCES seasons(year),
                    FOREIGN KEY (home_team) REFERENCES teams(team_abbr),
                    FOREIGN KEY (away_team) REFERENCES teams(team_abbr)
                )
            """)
            
            # Player stats table - Core statistical data
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS player_stats (
                    stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    player_id TEXT,
                    game_id TEXT,
                    season_year INTEGER,
                    week INTEGER,
                    team TEXT,
                    opponent TEXT,
                    is_home BOOLEAN,
                    
                    -- Passing stats
                    pass_att INTEGER DEFAULT 0,
                    pass_cmp INTEGER DEFAULT 0,
                    pass_yd INTEGER DEFAULT 0,
                    pass_td INTEGER DEFAULT 0,
                    pass_int INTEGER DEFAULT 0,
                    pass_sack INTEGER DEFAULT 0,
                    pass_sack_yd INTEGER DEFAULT 0,
                    pass_lng INTEGER DEFAULT 0,
                    pass_rating REAL DEFAULT 0,
                    
                    -- Rushing stats
                    rush_att INTEGER DEFAULT 0,
                    rush_yd INTEGER DEFAULT 0,
                    rush_td INTEGER DEFAULT 0,
                    rush_lng INTEGER DEFAULT 0,
                    rush_fum INTEGER DEFAULT 0,
                    
                    -- Receiving stats
                    rec_tgt INTEGER DEFAULT 0,
                    rec INTEGER DEFAULT 0,
                    rec_yd INTEGER DEFAULT 0,
                    rec_td INTEGER DEFAULT 0,
                    rec_lng INTEGER DEFAULT 0,
                    rec_fum INTEGER DEFAULT 0,
                    
                    -- Fantasy points (calculated)
                    fantasy_points_ppr REAL DEFAULT 0,
                    fantasy_points_half_ppr REAL DEFAULT 0,
                    fantasy_points_std REAL DEFAULT 0,
                    
                    -- Metadata
                    snap_count INTEGER,
                    snap_percentage REAL,
                    target_share REAL,
                    air_yards INTEGER,
                    
                    data_source TEXT DEFAULT 'sleeper',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    FOREIGN KEY (player_id) REFERENCES players(player_id),
                    FOREIGN KEY (game_id) REFERENCES games(game_id),
                    FOREIGN KEY (season_year) REFERENCES seasons(year),
                    UNIQUE(player_id, game_id)
                )
            """)
            
            # Advanced stats table - PFR and other advanced metrics
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS advanced_stats (
                    advanced_stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    player_id TEXT,
                    season_year INTEGER,
                    week INTEGER,
                    
                    -- Advanced passing
                    qb_rating REAL,
                    passer_rating REAL,
                    qbr REAL,
                    completion_pct_above_exp REAL,
                    air_yards_per_attempt REAL,
                    yards_after_catch_per_completion REAL,
                    pressure_rate REAL,
                    time_to_throw REAL,
                    
                    -- Advanced rushing
                    yards_before_contact_per_att REAL,
                    yards_after_contact_per_att REAL,
                    broken_tackles INTEGER,
                    rush_success_rate REAL,
                    
                    -- Advanced receiving
                    separation REAL,
                    catch_rate REAL,
                    drop_rate REAL,
                    yards_after_catch REAL,
                    contested_catch_rate REAL,
                    
                    -- Red zone stats
                    red_zone_targets INTEGER,
                    red_zone_receptions INTEGER,
                    red_zone_touchdowns INTEGER,
                    goal_line_carries INTEGER,
                    
                    data_source TEXT DEFAULT 'pfr',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    FOREIGN KEY (player_id) REFERENCES players(player_id),
                    UNIQUE(player_id, season_year, week, data_source)
                )
            """)
            
            # Team efficiency stats
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS team_stats (
                    team_stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    team_abbr TEXT,
                    season_year INTEGER,
                    week INTEGER,
                    
                    -- Offensive efficiency
                    points_per_game REAL,
                    yards_per_game REAL,
                    pass_yd_per_game REAL,
                    rush_yd_per_game REAL,
                    turnovers_per_game REAL,
                    red_zone_efficiency REAL,
                    third_down_conversion REAL,
                    
                    -- Defensive efficiency
                    points_allowed_per_game REAL,
                    yards_allowed_per_game REAL,
                    pass_yd_allowed_per_game REAL,
                    rush_yd_allowed_per_game REAL,
                    takeaways_per_game REAL,
                    red_zone_defense REAL,
                    third_down_defense REAL,
                    
                    -- Special teams
                    field_goal_pct REAL,
                    punt_return_avg REAL,
                    kickoff_return_avg REAL,
                    
                    data_source TEXT DEFAULT 'calculated',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    FOREIGN KEY (team_abbr) REFERENCES teams(team_abbr),
                    UNIQUE(team_abbr, season_year, week)
                )
            """)
            
            # Data collection log - Track API calls and data freshness
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS data_collection_log (
                    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    data_source TEXT,
                    collection_type TEXT,  -- 'full_season', 'weekly_update', 'player_update'
                    season_year INTEGER,
                    week INTEGER,
                    records_collected INTEGER,
                    success BOOLEAN,
                    error_message TEXT,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    duration_seconds REAL
                )
            """)
            
            # Create indexes for performance
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_player_stats_player_season ON player_stats(player_id, season_year)",
                "CREATE INDEX IF NOT EXISTS idx_player_stats_week ON player_stats(season_year, week)",
                "CREATE INDEX IF NOT EXISTS idx_player_stats_team ON player_stats(team, season_year)",
                "CREATE INDEX IF NOT EXISTS idx_players_position ON players(position)",
                "CREATE INDEX IF NOT EXISTS idx_players_team ON players(team)",
                "CREATE INDEX IF NOT EXISTS idx_games_season_week ON games(season_year, week)",
                "CREATE INDEX IF NOT EXISTS idx_advanced_stats_player ON advanced_stats(player_id, season_year)",
                "CREATE INDEX IF NOT EXISTS idx_data_log_source_date ON data_collection_log(data_source, started_at)"
            ]
            
            for index_sql in indexes:
                cursor.execute(index_sql)
            
            self.connection.commit()
            self.logger.info("All tables and indexes created successfully")
            return True
            
        except sqlite3.Error as e:
            self.logger.error(f"Error creating tables: {e}")
            self.connection.rollback()
            return False
    
    def insert_teams(self):
        """Insert NFL teams data"""
        nfl_teams = [
            ('ARI', 'Arizona Cardinals', 'NFC West', 'NFC'),
            ('ATL', 'Atlanta Falcons', 'NFC South', 'NFC'),
            ('BAL', 'Baltimore Ravens', 'AFC North', 'AFC'),
            ('BUF', 'Buffalo Bills', 'AFC East', 'AFC'),
            ('CAR', 'Carolina Panthers', 'NFC South', 'NFC'),
            ('CHI', 'Chicago Bears', 'NFC North', 'NFC'),
            ('CIN', 'Cincinnati Bengals', 'AFC North', 'AFC'),
            ('CLE', 'Cleveland Browns', 'AFC North', 'AFC'),
            ('DAL', 'Dallas Cowboys', 'NFC East', 'NFC'),
            ('DEN', 'Denver Broncos', 'AFC West', 'AFC'),
            ('DET', 'Detroit Lions', 'NFC North', 'NFC'),
            ('GB', 'Green Bay Packers', 'NFC North', 'NFC'),
            ('HOU', 'Houston Texans', 'AFC South', 'AFC'),
            ('IND', 'Indianapolis Colts', 'AFC South', 'AFC'),
            ('JAX', 'Jacksonville Jaguars', 'AFC South', 'AFC'),
            ('KC', 'Kansas City Chiefs', 'AFC West', 'AFC'),
            ('LV', 'Las Vegas Raiders', 'AFC West', 'AFC'),
            ('LAC', 'Los Angeles Chargers', 'AFC West', 'AFC'),
            ('LAR', 'Los Angeles Rams', 'NFC West', 'NFC'),
            ('MIA', 'Miami Dolphins', 'AFC East', 'AFC'),
            ('MIN', 'Minnesota Vikings', 'NFC North', 'NFC'),
            ('NE', 'New England Patriots', 'AFC East', 'AFC'),
            ('NO', 'New Orleans Saints', 'NFC South', 'NFC'),
            ('NYG', 'New York Giants', 'NFC East', 'NFC'),
            ('NYJ', 'New York Jets', 'AFC East', 'AFC'),
            ('PHI', 'Philadelphia Eagles', 'NFC East', 'NFC'),
            ('PIT', 'Pittsburgh Steelers', 'AFC North', 'AFC'),
            ('SF', 'San Francisco 49ers', 'NFC West', 'NFC'),
            ('SEA', 'Seattle Seahawks', 'NFC West', 'NFC'),
            ('TB', 'Tampa Bay Buccaneers', 'NFC South', 'NFC'),
            ('TEN', 'Tennessee Titans', 'AFC South', 'AFC'),
            ('WAS', 'Washington Commanders', 'NFC East', 'NFC')
        ]
        
        cursor = self.connection.cursor()
        try:
            cursor.executemany("""
                INSERT OR REPLACE INTO teams (team_abbr, team_name, division, conference, team_id)
                VALUES (?, ?, ?, ?, ?)
            """, [(abbr, name, div, conf, abbr) for abbr, name, div, conf in nfl_teams])
            
            self.connection.commit()
            self.logger.info(f"Inserted {len(nfl_teams)} NFL teams")
            return True
            
        except sqlite3.Error as e:
            self.logger.error(f"Error inserting teams: {e}")
            return False
    
    def insert_seasons(self):
        """Insert season data"""
        seasons = [
            (2021, '2021-09-09', '2022-02-13', 18, 21),
            (2022, '2022-09-08', '2023-02-12', 18, 22),
            (2023, '2023-09-07', '2024-02-11', 18, 22),
            (2024, '2024-09-05', '2025-02-09', 18, 22)
        ]
        
        cursor = self.connection.cursor()
        try:
            cursor.executemany("""
                INSERT OR REPLACE INTO seasons (year, start_date, end_date, playoff_start_week, total_weeks)
                VALUES (?, ?, ?, ?, ?)
            """, seasons)
            
            self.connection.commit()
            self.logger.info(f"Inserted {len(seasons)} seasons")
            return True
            
        except sqlite3.Error as e:
            self.logger.error(f"Error inserting seasons: {e}")
            return False
    
    def log_data_collection(self, data_source, collection_type, season_year=None, week=None, 
                           records_collected=0, success=True, error_message=None, 
                           started_at=None, completed_at=None):
        """Log data collection attempts"""
        
        if started_at and completed_at:
            duration = (completed_at - started_at).total_seconds()
        else:
            duration = None
            
        cursor = self.connection.cursor()
        try:
            cursor.execute("""
                INSERT INTO data_collection_log 
                (data_source, collection_type, season_year, week, records_collected, 
                 success, error_message, started_at, completed_at, duration_seconds)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (data_source, collection_type, season_year, week, records_collected,
                  success, error_message, started_at, completed_at, duration))
            
            self.connection.commit()
            return cursor.lastrowid
            
        except sqlite3.Error as e:
            self.logger.error(f"Error logging data collection: {e}")
            return None
    
    def get_latest_data_timestamp(self, data_source, season_year=None):
        """Get the most recent data collection timestamp"""
        cursor = self.connection.cursor()
        
        if season_year:
            cursor.execute("""
                SELECT MAX(completed_at) 
                FROM data_collection_log 
                WHERE data_source = ? AND season_year = ? AND success = 1
            """, (data_source, season_year))
        else:
            cursor.execute("""
                SELECT MAX(completed_at) 
                FROM data_collection_log 
                WHERE data_source = ? AND success = 1
            """, (data_source,))
        
        result = cursor.fetchone()
        return result[0] if result[0] else None
    
    def calculate_fantasy_points(self, stats, scoring_format='ppr'):
        """Calculate fantasy points from raw stats"""
        points = 0
        
        # Passing (1 pt per 25 yards, 4 pts per TD, -2 per INT)
        points += (stats.get('pass_yd', 0) / 25.0)
        points += (stats.get('pass_td', 0) * 4)
        points -= (stats.get('pass_int', 0) * 2)
        
        # Rushing (1 pt per 10 yards, 6 pts per TD)
        points += (stats.get('rush_yd', 0) / 10.0)
        points += (stats.get('rush_td', 0) * 6)
        
        # Receiving (1 pt per 10 yards, 6 pts per TD)
        points += (stats.get('rec_yd', 0) / 10.0)
        points += (stats.get('rec_td', 0) * 6)
        
        # Receptions (depends on scoring format)
        if scoring_format == 'ppr':
            points += stats.get('rec', 0)
        elif scoring_format == 'half_ppr':
            points += (stats.get('rec', 0) * 0.5)
        # Standard scoring adds 0 for receptions
        
        # Fumbles lost (-2 pts each)
        points -= (stats.get('rush_fum', 0) * 2)
        points -= (stats.get('rec_fum', 0) * 2)
        
        return round(points, 2)

def main():
    """Initialize database and create all tables"""
    print("🗄️ Setting up Fantasy Football Database...")
    
    # Create database instance
    db = FantasyDatabase("fantasy_football.db")
    
    # Connect to database
    if not db.connect():
        print("❌ Failed to connect to database")
        return
    
    # Create all tables
    if db.create_tables():
        print("✅ Database schema created successfully")
    else:
        print("❌ Failed to create database schema")
        return
    
    # Insert reference data
    if db.insert_teams():
        print("✅ NFL teams data inserted")
    
    if db.insert_seasons():
        print("✅ Season data inserted")
    
    # Test database structure
    cursor = db.connection.cursor()
    cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' 
        ORDER BY name
    """)
    
    tables = cursor.fetchall()
    print(f"\n📊 Created {len(tables)} tables:")
    for table in tables:
        print(f"  • {table[0]}")
    
    # Check team data
    cursor.execute("SELECT COUNT(*) FROM teams")
    team_count = cursor.fetchone()[0]
    print(f"\n🏈 {team_count} NFL teams loaded")
    
    # Check season data
    cursor.execute("SELECT COUNT(*) FROM seasons")
    season_count = cursor.fetchone()[0]
    print(f"📅 {season_count} seasons configured")
    
    print(f"\n✅ Database setup complete!")
    print(f"Database file: {os.path.abspath(db.db_path)}")
    
    # Close connection
    db.disconnect()

if __name__ == "__main__":
    main()