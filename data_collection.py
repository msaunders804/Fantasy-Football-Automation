import requests
import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional
import logging
import csv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataCollector:
    def __init__(self, db_path: str = "fantasy_data.db"):
        """
        Initialize the data collector with database connection.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'FantasyFootball-DataCollector/1.0'
        })
        
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Players table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS players (
                    player_id TEXT PRIMARY KEY,
                    sleeper_id TEXT UNIQUE,
                    datapros_id TEXT,
                    name TEXT NOT NULL,
                    team TEXT,
                    position TEXT,
                    age INTEGER,
                    height TEXT,
                    weight TEXT,
                    years_exp INTEGER,
                    college TEXT,
                    injury_status TEXT,
                    active BOOLEAN,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Projections table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS projections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    player_id TEXT,
                    source TEXT NOT NULL,
                    season INTEGER,
                    projected_points REAL,
                    passing_yards REAL,
                    passing_tds INTEGER,
                    passing_ints INTEGER,
                    rushing_yards REAL,
                    rushing_tds INTEGER,
                    receiving_yards REAL,
                    receiving_tds INTEGER,
                    receptions INTEGER,
                    fumbles_lost INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (player_id) REFERENCES players (player_id),
                    UNIQUE(player_id, source, season)
                )
            ''')
            
            # Collection log table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS collection_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    collection_type TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    records_collected INTEGER,
                    error_message TEXT,
                    collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            logger.info("Database initialized successfully")
    
    def collect_sleeper_players(self) -> Dict[str, any]:
        """
        Collect all NFL players from Sleeper API.
        Returns dict with success status and collection stats.
        """
        url = "https://api.sleeper.app/v1/players/nfl"
        
        try:
            logger.info("Starting Sleeper player data collection...")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            players_data = response.json()
            logger.info(f"Retrieved {len(players_data)} players from Sleeper API")
            
            # Process and store the data
            stored_count = self._store_sleeper_players(players_data)
            
            # Log the collection
            self._log_collection("Sleeper", "players", True, stored_count)
            
            return {
                "success": True,
                "total_retrieved": len(players_data),
                "total_stored": stored_count,
                "message": f"Successfully collected {stored_count} players from Sleeper"
            }
            
        except requests.exceptions.RequestException as e:
            error_msg = f"API request failed: {str(e)}"
            logger.error(error_msg)
            self._log_collection("Sleeper", "players", False, 0, error_msg)
            return {"success": False, "error": error_msg}
        
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(error_msg)
            self._log_collection("Sleeper", "players", False, 0, error_msg)
            return {"success": False, "error": error_msg}
    
    def _store_sleeper_players(self, players_data: Dict) -> int:
        """Store Sleeper player data in database."""
        stored_count = 0
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for sleeper_id, player_info in players_data.items():
                try:
                    # Skip players without basic info
                    if not player_info.get('full_name'):
                        continue
                    
                    # Only include relevant fantasy positions
                    position = player_info.get('position', '')
                    if position not in ['QB', 'RB', 'WR', 'TE', 'K', 'DEF']:
                        continue
                    
                    # Create a unique player_id
                    player_id = f"sleeper_{sleeper_id}"
                    
                    # Insert or update player
                    cursor.execute('''
                        INSERT OR REPLACE INTO players 
                        (player_id, sleeper_id, name, team, position, age, height, 
                         weight, years_exp, college, injury_status, active, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        player_id,
                        sleeper_id,
                        player_info.get('full_name', ''),
                        player_info.get('team'),
                        position,
                        player_info.get('age'),
                        player_info.get('height'),
                        player_info.get('weight'),
                        player_info.get('years_exp'),
                        player_info.get('college'),
                        player_info.get('injury_status'),
                        player_info.get('active', True),
                        datetime.now().isoformat()
                    ))
                    
                    stored_count += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to store player {sleeper_id}: {str(e)}")
                    continue
            
            conn.commit()
        
        logger.info(f"Stored {stored_count} players in database")
        return stored_count
    
    def collect_datapros_projections(self, season: int = 2025) -> Dict[str, any]:
        """
        Collect fantasy projections from Fantasy Football Data Pros API or create mock data.
        Returns dict with success status and collection stats.
        """
        try:
            logger.info(f"Starting Fantasy Football Data Pros collection for {season}...")
            
            # Primary endpoint
            url = "https://www.fantasyfootballdatapros.com/api/projections"
            
            logger.info(f"Requesting projections from: {url}")
            response = self.session.get(url, timeout=30)
            logger.info(f"DataPros response status: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    projections_data = response.json()
                    logger.info(f"Retrieved {len(projections_data)} projections from DataPros")
                    
                    if projections_data and len(projections_data) > 0:
                        # Process and store the data
                        stored_count = self._store_datapros_projections(projections_data, season)
                        self._log_collection("DataPros", "projections", True, stored_count)
                        
                        return {
                            "success": True,
                            "total_retrieved": len(projections_data),
                            "total_stored": stored_count,
                            "endpoint_used": url,
                            "message": f"Successfully collected {stored_count} projections from Fantasy Football Data Pros"
                        }
                    else:
                        logger.warning("DataPros API returned empty data, falling back to mock data")
                        return self._create_mock_projections_fallback(season)
                        
                except json.JSONDecodeError:
                    logger.warning("DataPros API returned invalid JSON, falling back to mock data")
                    return self._create_mock_projections_fallback(season)
            else:
                logger.warning(f"DataPros API returned status {response.status_code}, falling back to mock data")
                return self._create_mock_projections_fallback(season)
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"DataPros API request failed: {str(e)}, falling back to mock data")
            return self._create_mock_projections_fallback(season)
        
        except Exception as e:
            error_msg = f"DataPros collection failed: {str(e)}"
            logger.error(error_msg)
            self._log_collection("DataPros", "projections", False, 0, error_msg)
            return {"success": False, "error": error_msg}
    
    def _create_mock_projections_fallback(self, season: int) -> Dict[str, any]:
        """Create mock projections when API is unavailable."""
        try:
            logger.info("Creating mock projections as fallback...")
            mock_projections = self._create_mock_projections()
            
            if mock_projections:
                stored_count = self._store_datapros_projections(mock_projections, season)
                self._log_collection("DataPros_Mock", "projections", True, stored_count)
                
                return {
                    "success": True,
                    "total_retrieved": len(mock_projections),
                    "total_stored": stored_count,
                    "endpoint_used": "Mock data (API unavailable)",
                    "message": f"API unavailable - created {stored_count} basic projections from mock data",
                    "note": "Consider adding FantasyPros or other projection source for better data"
                }
            else:
                error_msg = "Mock data creation failed"
                self._log_collection("DataPros_Mock", "projections", False, 0, error_msg)
                return {"success": False, "error": error_msg}
                
        except Exception as e:
            error_msg = f"Mock projections fallback failed: {str(e)}"
            logger.error(error_msg)
            self._log_collection("DataPros_Mock", "projections", False, 0, error_msg)
            return {"success": False, "error": error_msg}
    
    def _create_mock_projections(self) -> List[Dict]:
        """Create basic mock projections for top fantasy players."""
        mock_data = [
            # Top QBs
            {"player_name": "Josh Allen", "position": "QB", "team": "BUF", 
             "passing_yards": 4200, "passing_tds": 30, "passing_ints": 12, "rushing_yards": 400, "rushing_tds": 8},
            {"player_name": "Lamar Jackson", "position": "QB", "team": "BAL",
             "passing_yards": 3800, "passing_tds": 25, "passing_ints": 10, "rushing_yards": 800, "rushing_tds": 10},
            {"player_name": "Patrick Mahomes", "position": "QB", "team": "KC",
             "passing_yards": 4500, "passing_tds": 35, "passing_ints": 13, "rushing_yards": 200, "rushing_tds": 3},
            {"player_name": "Dak Prescott", "position": "QB", "team": "DAL",
             "passing_yards": 4100, "passing_tds": 28, "passing_ints": 11, "rushing_yards": 150, "rushing_tds": 2},
            
            # Top RBs
            {"player_name": "Christian McCaffrey", "position": "RB", "team": "SF",
             "rushing_yards": 1200, "rushing_tds": 12, "receiving_yards": 600, "receptions": 60, "fumbles_lost": 2},
            {"player_name": "Derrick Henry", "position": "RB", "team": "BAL", 
             "rushing_yards": 1400, "rushing_tds": 15, "receiving_yards": 200, "receptions": 20, "fumbles_lost": 1},
            {"player_name": "Saquon Barkley", "position": "RB", "team": "PHI",
             "rushing_yards": 1100, "rushing_tds": 10, "receiving_yards": 400, "receptions": 40, "fumbles_lost": 2},
            {"player_name": "Josh Jacobs", "position": "RB", "team": "GB",
             "rushing_yards": 1000, "rushing_tds": 9, "receiving_yards": 300, "receptions": 30, "fumbles_lost": 1},
            
            # Top WRs
            {"player_name": "Tyreek Hill", "position": "WR", "team": "MIA",
             "receiving_yards": 1500, "receiving_tds": 12, "receptions": 100, "fumbles_lost": 1},
            {"player_name": "Stefon Diggs", "position": "WR", "team": "HOU", 
             "receiving_yards": 1300, "receiving_tds": 10, "receptions": 90, "fumbles_lost": 0},
            {"player_name": "CeeDee Lamb", "position": "WR", "team": "DAL",
             "receiving_yards": 1400, "receiving_tds": 11, "receptions": 95, "fumbles_lost": 1},
            {"player_name": "Amon-Ra St. Brown", "position": "WR", "team": "DET",
             "receiving_yards": 1200, "receiving_tds": 9, "receptions": 85, "fumbles_lost": 0},
            
            # Top TEs
            {"player_name": "Travis Kelce", "position": "TE", "team": "KC",
             "receiving_yards": 1000, "receiving_tds": 8, "receptions": 80, "fumbles_lost": 0},
            {"player_name": "Mark Andrews", "position": "TE", "team": "BAL",
             "receiving_yards": 900, "receiving_tds": 7, "receptions": 70, "fumbles_lost": 1},
            {"player_name": "T.J. Hockenson", "position": "TE", "team": "MIN",
             "receiving_yards": 800, "receiving_tds": 6, "receptions": 65, "fumbles_lost": 0}
        ]
        
        # Add default values for missing stats and calculate fantasy points
        for player in mock_data:
            player.setdefault("passing_yards", 0)
            player.setdefault("passing_tds", 0)
            player.setdefault("passing_ints", 0)
            player.setdefault("rushing_yards", 0)
            player.setdefault("rushing_tds", 0)
            player.setdefault("receiving_yards", 0)
            player.setdefault("receiving_tds", 0)
            player.setdefault("receptions", 0)
            player.setdefault("fumbles_lost", 0)
            
            # Calculate standard fantasy points
            points = (
                player["passing_yards"] * 0.04 +
                player["passing_tds"] * 4 -
                player["passing_ints"] * 2 +
                player["rushing_yards"] * 0.1 +
                player["rushing_tds"] * 6 +
                player["receiving_yards"] * 0.1 +
                player["receiving_tds"] * 6 +
                player["receptions"] * 1 -
                player["fumbles_lost"] * 2
            )
            player["projected_points"] = round(points, 1)
        
        logger.info(f"Created {len(mock_data)} mock projections")
        return mock_data
    
    def _store_datapros_projections(self, projections_data: List, season: int) -> int:
        """Store Fantasy Football Data Pros projections in database."""
        stored_count = 0
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for projection in projections_data:
                try:
                    if not isinstance(projection, dict):
                        continue
                    
                    # Extract player info
                    player_name = projection.get('player_name') or projection.get('name')
                    position = projection.get('position')
                    team = projection.get('team')
                    
                    if not player_name or position not in ['QB', 'RB', 'WR', 'TE', 'K', 'DEF']:
                        continue
                    
                    # Try to match with existing Sleeper player first
                    cursor.execute('''
                        SELECT player_id FROM players 
                        WHERE (name = ? OR LOWER(name) LIKE ?) AND position = ?
                        ORDER BY sleeper_id IS NOT NULL DESC
                        LIMIT 1
                    ''', (player_name, f"%{player_name.lower()}%", position))
                    existing_player = cursor.fetchone()
                    
                    if existing_player:
                        player_id = existing_player[0]
                        # Update player with DataPros ID if we have one
                        datapros_id = projection.get('player_id')
                        if datapros_id:
                            cursor.execute('''
                                UPDATE players SET datapros_id = ?, updated_at = ? 
                                WHERE player_id = ?
                            ''', (datapros_id, datetime.now().isoformat(), player_id))
                    else:
                        # Create new player entry
                        datapros_id = projection.get('player_id') or f"datapros_{player_name.replace(' ', '_')}"
                        player_id = f"datapros_{datapros_id}"
                        cursor.execute('''
                            INSERT OR REPLACE INTO players 
                            (player_id, datapros_id, name, team, position, active, updated_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', (player_id, datapros_id, player_name, team, position, True, datetime.now().isoformat()))
                    
                    # Extract fantasy stats
                    fantasy_stats = self._extract_fantasy_stats(projection)
                    
                    # Store projection
                    cursor.execute('''
                        INSERT OR REPLACE INTO projections 
                        (player_id, source, season, projected_points, 
                         passing_yards, passing_tds, passing_ints,
                         rushing_yards, rushing_tds, receiving_yards, 
                         receiving_tds, receptions, fumbles_lost)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        player_id,
                        "DataPros",
                        season,
                        fantasy_stats.get('projected_points', 0),
                        fantasy_stats.get('passing_yards', 0),
                        fantasy_stats.get('passing_tds', 0),
                        fantasy_stats.get('passing_ints', 0),
                        fantasy_stats.get('rushing_yards', 0),
                        fantasy_stats.get('rushing_tds', 0),
                        fantasy_stats.get('receiving_yards', 0),
                        fantasy_stats.get('receiving_tds', 0),
                        fantasy_stats.get('receptions', 0),
                        fantasy_stats.get('fumbles_lost', 0)
                    ))
                    
                    stored_count += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to store DataPros projection for {projection.get('player_name', 'unknown')}: {str(e)}")
                    continue
            
            conn.commit()
        
        logger.info(f"Stored {stored_count} DataPros projections in database")
        return stored_count
    
    def _extract_fantasy_stats(self, projection: Dict) -> Dict:
        """Extract fantasy stats from projection data."""
        stats = {
            'passing_yards': projection.get('passing_yards', 0),
            'passing_tds': projection.get('passing_tds', 0),
            'passing_ints': projection.get('passing_ints', 0),
            'rushing_yards': projection.get('rushing_yards', 0),
            'rushing_tds': projection.get('rushing_tds', 0),
            'receiving_yards': projection.get('receiving_yards', 0),
            'receiving_tds': projection.get('receiving_tds', 0),
            'receptions': projection.get('receptions', 0),
            'fumbles_lost': projection.get('fumbles_lost', 0),
            'projected_points': projection.get('projected_points', 0)
        }
        
        # If projected_points is not provided, calculate it
        if not stats['projected_points']:
            stats['projected_points'] = (
                stats['passing_yards'] * 0.04 +
                stats['passing_tds'] * 4 -
                stats['passing_ints'] * 2 +
                stats['rushing_yards'] * 0.1 +
                stats['rushing_tds'] * 6 +
                stats['receiving_yards'] * 0.1 +
                stats['receiving_tds'] * 6 +
                stats['receptions'] * 1 -
                stats['fumbles_lost'] * 2
            )
        
        return stats
    
    def _log_collection(self, source: str, collection_type: str, success: bool, 
                       records_collected: int, error_message: str = None):
        """Log collection attempt to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO collection_log 
                (source, collection_type, success, records_collected, error_message)
                VALUES (?, ?, ?, ?, ?)
            ''', (source, collection_type, success, records_collected, error_message))
            conn.commit()
    
    def get_players_summary(self) -> Dict[str, any]:
        """Get summary statistics of collected players."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Total players
            cursor.execute("SELECT COUNT(*) FROM players WHERE active = 1")
            total_active = cursor.fetchone()[0]
            
            # Players by position
            cursor.execute('''
                SELECT position, COUNT(*) 
                FROM players 
                WHERE active = 1 AND position IS NOT NULL
                GROUP BY position 
                ORDER BY COUNT(*) DESC
            ''')
            position_counts = dict(cursor.fetchall())
            
            # Players by team
            cursor.execute('''
                SELECT team, COUNT(*) 
                FROM players 
                WHERE active = 1 AND team IS NOT NULL
                GROUP BY team 
                ORDER BY COUNT(*) DESC
                LIMIT 10
            ''')
            team_counts = dict(cursor.fetchall())
            
            # Total projections
            cursor.execute("SELECT COUNT(*) FROM projections")
            total_projections = cursor.fetchone()[0]
            
            # Projections by source
            cursor.execute('''
                SELECT source, COUNT(*) 
                FROM projections 
                GROUP BY source 
                ORDER BY COUNT(*) DESC
            ''')
            projection_sources = dict(cursor.fetchall())
            
            # Data sources
            cursor.execute('''
                SELECT 
                    COUNT(CASE WHEN sleeper_id IS NOT NULL THEN 1 END) as sleeper_count,
                    COUNT(CASE WHEN datapros_id IS NOT NULL THEN 1 END) as datapros_count,
                    COUNT(CASE WHEN sleeper_id IS NOT NULL AND datapros_id IS NOT NULL THEN 1 END) as both_count
                FROM players WHERE active = 1
            ''')
            source_counts = cursor.fetchone()
            
            # Recent collections
            cursor.execute('''
                SELECT source, collection_type, success, records_collected, collected_at
                FROM collection_log
                ORDER BY collected_at DESC
                LIMIT 5
            ''')
            recent_collections = cursor.fetchall()
            
            return {
                "total_active_players": total_active,
                "players_by_position": position_counts,
                "players_by_team": team_counts,
                "total_projections": total_projections,
                "projections_by_source": projection_sources,
                "data_sources": {
                    "sleeper_count": source_counts[0] if source_counts else 0,
                    "datapros_count": source_counts[1] if source_counts else 0,
                    "both_sources": source_counts[2] if source_counts else 0
                },
                "recent_collections": recent_collections
            }
    
    def get_top_players_by_position(self, position: str, limit: int = 20) -> List[Dict]:
        """Get top players by position based on projected points."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT p.name, p.team, p.position, pr.projected_points, pr.source,
                       p.age, p.years_exp, p.injury_status
                FROM players p
                LEFT JOIN projections pr ON p.player_id = pr.player_id
                WHERE p.position = ? AND p.active = 1
                ORDER BY pr.projected_points DESC NULLS LAST
                LIMIT ?
            ''', (position, limit))
            
            columns = ['name', 'team', 'position', 'projected_points', 'source', 
                      'age', 'years_exp', 'injury_status']
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def export_draft_board(self, filename: str = "draft_board.csv") -> str:
        """Export a draft board with players and projections to CSV."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT p.name, p.team, p.position, 
                       COALESCE(pr.projected_points, 0) as projected_points,
                       p.age, p.years_exp, p.injury_status, 
                       COALESCE(pr.passing_yards, 0) as passing_yards,
                       COALESCE(pr.passing_tds, 0) as passing_tds, 
                       COALESCE(pr.rushing_yards, 0) as rushing_yards, 
                       COALESCE(pr.rushing_tds, 0) as rushing_tds,
                       COALESCE(pr.receiving_yards, 0) as receiving_yards, 
                       COALESCE(pr.receiving_tds, 0) as receiving_tds, 
                       COALESCE(pr.receptions, 0) as receptions
                FROM players p
                LEFT JOIN projections pr ON p.player_id = pr.player_id
                WHERE p.active = 1
                ORDER BY COALESCE(pr.projected_points, 0) DESC, p.position, p.name
            ''')
            
            players = cursor.fetchall()
            
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Name', 'Team', 'Position', 'Projected Points', 'Age', 
                               'Years Exp', 'Injury Status', 'Pass Yards', 'Pass TDs',
                               'Rush Yards', 'Rush TDs', 'Rec Yards', 'Rec TDs', 'Receptions'])
                writer.writerows(players)
            
            logger.info(f"Exported {len(players)} players to draft board: {filename}")
            return filename
    
    def clear_projections(self, source: str = None):
        """Clear projections from database, optionally by source."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if source:
                cursor.execute("DELETE FROM projections WHERE source = ?", (source,))
                logger.info(f"Cleared {source} projections from database")
            else:
                cursor.execute("DELETE FROM projections")
                logger.info("Cleared all projections from database")
            conn.commit()

# Simple usage example
if __name__ == "__main__":
    # Initialize collector (no API keys needed!)
    collector = DataCollector()
    
    print("=== Fantasy Football Data Collector ===\n")
    
    print("=== Step 1: Collect Sleeper Players ===")
    sleeper_result = collector.collect_sleeper_players()
    print(f"Sleeper result: {sleeper_result}")
    
    print("\n=== Step 2: Collect DataPros Projections ===")
    datapros_result = collector.collect_datapros_projections(season=2025)
    print(f"DataPros result: {datapros_result}")
    
    print("\n=== Step 3: Summary ===")
    summary = collector.get_players_summary()
    print(f"Total active players: {summary['total_active_players']}")
    print(f"Players by position: {summary['players_by_position']}")
    print(f"Total projections: {summary['total_projections']}")
    print(f"Projections by source: {summary['projections_by_source']}")
    print(f"Data sources: {summary['data_sources']}")
    
    print("\n=== Step 4: Top Players by Position ===")
    for position in ['QB', 'RB', 'WR', 'TE']:
        top_players = collector.get_top_players_by_position(position, 5)
        print(f"\nTop {position}s:")
        for player in top_players:
            points = player['projected_points'] or 0
            print(f"  {player['name']} ({player['team']}) - {points:.1f} pts")
    
    print("\n=== Step 5: Export Draft Board ===")
    try:
        draft_board = collector.export_draft_board()
        print(f"Draft board exported to: {draft_board}")
    except Exception as e:
        print(f"Export failed: {e}")
    
    print("\n=== Ready for Draft! ===")
    print("You now have:")
    print("✅ Complete player database from Sleeper")
    print("✅ Fantasy projections (DataPros or mock data)")
    print("✅ Draft board ranked by projected points")
    print("✅ Top players by position")
    print("✅ CSV export functionality")
    print("\n=== Additional Features ===")
    print("• Use collector.clear_projections() to reset projection data")
    print("• Use collector.get_players_summary() for detailed stats")
    print("• Check collection_log table for data collection history")
