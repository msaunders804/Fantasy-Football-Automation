# fantasypros_collector.py
"""
FantasyPros API Data Collector
Handles all interactions with FantasyPros API and stores data in SQLite database
"""

import requests
import sqlite3
import json
import time
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
from database_setup import FantasyDatabase

class FantasyProsCollector:
    """Collects and stores data from FantasyPros API"""
    
    def __init__(self, api_key: str, db_path="fantasy_football.db"):
        self.api_key = api_key
        self.base_url = "https://api.fantasypros.com/v2"
        self.db = FantasyDatabase(db_path)
        self.session = requests.Session()
        self.setup_logging()
        
        # Set up headers for API requests - try different auth methods
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        # Rate limiting - FantasyPros has rate limits
        self.rate_limit_delay = 0.5  # 500ms between requests to be safe
        self.last_request_time = 0
    
    def setup_logging(self):
        """Configure logging for FantasyPros collector"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _rate_limit(self):
        """Implement simple rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make API request with error handling"""
        self._rate_limit()
        
        url = f"{self.base_url}/{endpoint}"
        
        # Try different authentication methods
        auth_methods = [
            # Method 1: API key in query params
            {'headers': {'Content-Type': 'application/json'}, 'params': {**(params or {}), 'api_key': self.api_key}},
            # Method 2: Bearer token in headers
            {'headers': {'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'}, 'params': params},
            # Method 3: x-api-key header
            {'headers': {'x-api-key': self.api_key, 'Content-Type': 'application/json'}, 'params': params},
            # Method 4: API key as token query param
            {'headers': {'Content-Type': 'application/json'}, 'params': {**(params or {}), 'token': self.api_key}},
        ]
        
        for i, auth_method in enumerate(auth_methods):
            try:
                self.logger.debug(f"Trying authentication method {i+1} for {endpoint}")
                response = self.session.get(url, headers=auth_method['headers'], 
                                          params=auth_method['params'], timeout=30)
                
                if response.status_code == 200:
                    self.logger.debug(f"API request successful with method {i+1}: {endpoint}")
                    return response.json()
                elif response.status_code == 401:
                    self.logger.debug(f"Auth method {i+1} failed (401), trying next...")
                    continue
                else:
                    response.raise_for_status()
                    
            except requests.exceptions.RequestException as e:
                if i == len(auth_methods) - 1:  # Last method failed
                    self.logger.error(f"All authentication methods failed for {endpoint}: {e}")
                    if hasattr(e, 'response') and e.response is not None:
                        self.logger.error(f"Response status: {e.response.status_code}")
                        self.logger.error(f"Response text: {e.response.text}")
                    return None
                else:
                    continue
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON decode error for {endpoint}: {e}")
                return None
        
        return None
    
    def test_api_connection(self) -> bool:
        """Test API connection and authentication"""
        self.logger.info("Testing FantasyPros API connection...")
        
        # Try a simple endpoint to test authentication
        test_endpoints = [
            "players/nfl",
            "projections/nfl",
            "rankings/nfl"
        ]
        
        for endpoint in test_endpoints:
            self.logger.info(f"Testing endpoint: {endpoint}")
            result = self._make_request(endpoint, {'year': 2024, 'week': 1})
            
            if result:
                self.logger.info(f"✅ API connection successful! Endpoint: {endpoint}")
                self.logger.info(f"Sample response keys: {list(result.keys()) if isinstance(result, dict) else 'List response'}")
                return True
            else:
                self.logger.warning(f"❌ Failed to connect to endpoint: {endpoint}")
        
        self.logger.error("❌ All test endpoints failed")
    def collect_projections(self, season: int, week: int = None, position: str = None, 
                           scoring: str = 'PPR') -> bool:
        """
        Collect player projections from FantasyPros API
        
        Args:
            season: NFL season year
            week: Specific week (None for season-long)
            position: Specific position (QB, RB, WR, TE, K, DST, or None for all)
            scoring: Scoring format (PPR, HALF, STD)
        """
        start_time = datetime.now()
        projection_type = "weekly" if week else "season"
        week_str = f"Week {week}" if week else "Season"
        
        self.logger.info(f"Starting {projection_type} projections collection for {season} {week_str}...")
        
        if not self.db.connect():
            return False
        
        try:
            # Build endpoint and parameters
            endpoint = "projections/nfl"
            params = {
                'year': season,
                'scoring': scoring.upper()
            }
            
            if week:
                params['week'] = week
            
            if position:
                params['position'] = position.upper()
            
            # Get projections data
            projections_data = self._make_request(endpoint, params)
            if not projections_data:
                self.logger.error(f"Failed to fetch projections data for {season} {week_str}")
                return False
            
            cursor = self.db.connection.cursor()
            projections_inserted = 0
            
            # Process players from the response
            players = projections_data.get('players', [])
            if not players:
                self.logger.warning(f"No players found in projections response")
                return False
            
            for player_data in players:
                try:
                    # Extract player information
                    player_name = player_data.get('player_name', '')
                    player_id = self._generate_player_id(player_data)
                    
                    # Get or create player in database
                    player_db_id = self._ensure_player_exists(cursor, player_data, player_id)
                    if not player_db_id:
                        continue
                    
                    # Extract projection stats
                    stats = player_data.get('projected_stats', {})
                    
                    # Calculate fantasy points based on projections
                    fantasy_points = self._calculate_projected_fantasy_points(stats, scoring)
                    
                    # Create projection record
                    cursor.execute("""
                        INSERT OR REPLACE INTO player_projections (
                            player_id, season_year, week, data_source, scoring_format,
                            pass_att, pass_cmp, pass_yd, pass_td, pass_int,
                            rush_att, rush_yd, rush_td, rush_fum,
                            rec_tgt, rec, rec_yd, rec_td, rec_fum,
                            fantasy_points, rank_position, rank_overall,
                            projection_date, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        player_db_id,
                        season,
                        week,
                        'fantasypros',
                        scoring.lower(),
                        stats.get('pass_att', 0),
                        stats.get('pass_cmp', 0),
                        stats.get('pass_yd', 0),
                        stats.get('pass_td', 0),
                        stats.get('pass_int', 0),
                        stats.get('rush_att', 0),
                        stats.get('rush_yd', 0),
                        stats.get('rush_td', 0),
                        stats.get('rush_fum', 0),
                        stats.get('rec_tgt', 0),
                        stats.get('rec', 0),
                        stats.get('rec_yd', 0),
                        stats.get('rec_td', 0),
                        stats.get('rec_fum', 0),
                        fantasy_points,
                        player_data.get('rank_position'),
                        player_data.get('rank_overall'),
                        datetime.now().date(),
                        datetime.now()
                    ))
                    
                    projections_inserted += 1
                    
                    if projections_inserted % 50 == 0:
                        self.logger.info(f"Processed {projections_inserted} projections...")
                        self.db.connection.commit()
                
                except sqlite3.Error as e:
                    self.logger.error(f"Error inserting projection for {player_name}: {e}")
                    continue
                except Exception as e:
                    self.logger.error(f"Unexpected error processing {player_name}: {e}")
                    continue
            
            # Final commit
            self.db.connection.commit()
            
            end_time = datetime.now()
            self.logger.info(f"Successfully collected {projections_inserted} projections for {season} {week_str}")
            
            # Log the collection
            self.db.log_data_collection(
                data_source='fantasypros',
                collection_type=f'{projection_type}_projections',
                season_year=season,
                week=week,
                records_collected=projections_inserted,
                success=True,
                started_at=start_time,
                completed_at=end_time,
                notes=f"Scoring: {scoring}, Position: {position or 'ALL'}"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in projections collection: {e}")
            
            # Log failed collection
            self.db.log_data_collection(
                data_source='fantasypros',
                collection_type=f'{projection_type}_projections',
                season_year=season,
                week=week,
                records_collected=0,
                success=False,
                error_message=str(e),
                started_at=start_time,
                completed_at=datetime.now()
            )
            return False
        finally:
            self.db.disconnect()
    
    def _generate_player_id(self, player_data: Dict) -> str:
        """Generate a consistent player ID from FantasyPros data"""
        # Use FantasyPros player_id if available, otherwise create from name
        fp_player_id = player_data.get('player_id')
        if fp_player_id:
            return f"fp_{fp_player_id}"
        
        # Fallback: create ID from name and team
        name = player_data.get('player_name', '').replace(' ', '_').lower()
        team = player_data.get('team', 'UNK')
        return f"fp_{name}_{team}"
    
    def _ensure_player_exists(self, cursor, player_data: Dict, player_id: str) -> Optional[str]:
        """Ensure player exists in database, create if needed"""
        try:
            # Check if player already exists
            cursor.execute("SELECT player_id FROM players WHERE player_id = ?", (player_id,))
            existing_player = cursor.fetchone()
            
            if existing_player:
                return existing_player[0]
            
            # Player doesn't exist, create new record
            full_name = player_data.get('player_name', '')
            name_parts = full_name.split()
            first_name = name_parts[0] if name_parts else ''
            last_name = ' '.join(name_parts[1:]) if len(name_parts) > 1 else ''
            
            cursor.execute("""
                INSERT INTO players (
                    player_id, fantasypros_id, first_name, last_name, full_name,
                    position, team, status, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                player_id,
                player_data.get('player_id'),
                first_name,
                last_name,
                full_name,
                player_data.get('position'),
                player_data.get('team'),
                'Active',
                datetime.now()
            ))
            
            return player_id
            
        except sqlite3.Error as e:
            self.logger.error(f"Error ensuring player exists: {e}")
            return None
    
    def _calculate_projected_fantasy_points(self, stats: Dict, scoring: str) -> float:
        """Calculate fantasy points from projected stats"""
        points = 0.0
        
        # Passing stats
        points += stats.get('pass_yd', 0) * 0.04  # 1 point per 25 yards
        points += stats.get('pass_td', 0) * 4     # 4 points per TD
        points -= stats.get('pass_int', 0) * 2    # -2 points per INT
        
        # Rushing stats
        points += stats.get('rush_yd', 0) * 0.1   # 1 point per 10 yards
        points += stats.get('rush_td', 0) * 6     # 6 points per TD
        points -= stats.get('rush_fum', 0) * 2    # -2 points per fumble
        
        # Receiving stats
        points += stats.get('rec_yd', 0) * 0.1    # 1 point per 10 yards
        points += stats.get('rec_td', 0) * 6      # 6 points per TD
        points -= stats.get('rec_fum', 0) * 2     # -2 points per fumble
        
        # Reception points based on scoring format
        receptions = stats.get('rec', 0)
        if scoring.upper() == 'PPR':
            points += receptions * 1.0            # 1 point per reception
        elif scoring.upper() == 'HALF':
            points += receptions * 0.5            # 0.5 points per reception
        # Standard scoring: 0 points per reception
        
        return round(points, 2)
    
    def collect_weekly_projections_all_scoring(self, season: int, week: int) -> bool:
        """Collect weekly projections for all scoring formats"""
        self.logger.info(f"Collecting projections for {season} Week {week} - All scoring formats")
        
        scoring_formats = ['PPR', 'HALF', 'STD']
        success_count = 0
        
        for scoring in scoring_formats:
            self.logger.info(f"Collecting {scoring} projections...")
            if self.collect_projections(season, week, scoring=scoring):
                success_count += 1
                self.logger.info(f"✅ {scoring} projections completed")
            else:
                self.logger.error(f"❌ {scoring} projections failed")
            
            # Brief pause between scoring formats
            time.sleep(1)
        
        return success_count == len(scoring_formats)
    
    def collect_season_projections(self, season: int, scoring: str = 'PPR') -> bool:
        """Collect season-long projections"""
        self.logger.info(f"Collecting season-long projections for {season} ({scoring})")
        
        return self.collect_projections(season, week=None, scoring=scoring)
    
    def get_average_projections(self, player_id: str, season: int, weeks: List[int] = None, 
                               scoring: str = 'PPR') -> Optional[Dict]:
        """Get averaged projections for a player over specified weeks"""
        if not self.db.connect():
            return None
        
        try:
            cursor = self.db.connection.cursor()
            
            # Build query based on weeks specified
            if weeks:
                week_placeholders = ','.join(['?'] * len(weeks))
                query = f"""
                    SELECT 
                        AVG(pass_att), AVG(pass_cmp), AVG(pass_yd), AVG(pass_td), AVG(pass_int),
                        AVG(rush_att), AVG(rush_yd), AVG(rush_td), AVG(rush_fum),
                        AVG(rec_tgt), AVG(rec), AVG(rec_yd), AVG(rec_td), AVG(rec_fum),
                        AVG(fantasy_points), COUNT(*) as projection_count
                    FROM player_projections 
                    WHERE player_id = ? AND season_year = ? AND scoring_format = ? 
                    AND week IN ({week_placeholders}) AND data_source = 'fantasypros'
                """
                params = [player_id, season, scoring.lower()] + weeks
            else:
                query = """
                    SELECT 
                        AVG(pass_att), AVG(pass_cmp), AVG(pass_yd), AVG(pass_td), AVG(pass_int),
                        AVG(rush_att), AVG(rush_yd), AVG(rush_td), AVG(rush_fum),
                        AVG(rec_tgt), AVG(rec), AVG(rec_yd), AVG(rec_td), AVG(rec_fum),
                        AVG(fantasy_points), COUNT(*) as projection_count
                    FROM player_projections 
                    WHERE player_id = ? AND season_year = ? AND scoring_format = ? 
                    AND data_source = 'fantasypros'
                """
                params = [player_id, season, scoring.lower()]
            
            cursor.execute(query, params)
            result = cursor.fetchone()
            
            if result and result[15] > 0:  # projection_count > 0
                return {
                    'pass_att': round(result[0] or 0, 2),
                    'pass_cmp': round(result[1] or 0, 2),
                    'pass_yd': round(result[2] or 0, 2),
                    'pass_td': round(result[3] or 0, 2),
                    'pass_int': round(result[4] or 0, 2),
                    'rush_att': round(result[5] or 0, 2),
                    'rush_yd': round(result[6] or 0, 2),
                    'rush_td': round(result[7] or 0, 2),
                    'rush_fum': round(result[8] or 0, 2),
                    'rec_tgt': round(result[9] or 0, 2),
                    'rec': round(result[10] or 0, 2),
                    'rec_yd': round(result[11] or 0, 2),
                    'rec_td': round(result[12] or 0, 2),
                    'rec_fum': round(result[13] or 0, 2),
                    'fantasy_points': round(result[14] or 0, 2),
                    'projection_count': result[15]
                }
            else:
                return None
                
        except sqlite3.Error as e:
            self.logger.error(f"Error getting average projections: {e}")
            return None
        finally:
            self.db.disconnect()
    
    def get_data_freshness(self) -> Dict:
        """Check how fresh our FantasyPros projections data is"""
        if not self.db.connect():
            return {}
        
        try:
            cursor = self.db.connection.cursor()
            
            # Get latest projections by season and type
            cursor.execute("""
                SELECT season_year, week, scoring_format, MAX(updated_at), COUNT(DISTINCT player_id)
                FROM player_projections 
                WHERE data_source = 'fantasypros'
                GROUP BY season_year, week, scoring_format
                ORDER BY season_year DESC, week DESC
            """)
            
            projections_stats = cursor.fetchall()
            
            # Get collection log summary
            cursor.execute("""
                SELECT collection_type, MAX(completed_at), 
                       SUM(records_collected), COUNT(*) as attempts
                FROM data_collection_log 
                WHERE success = 1 AND data_source = 'fantasypros'
                GROUP BY collection_type
                ORDER BY MAX(completed_at) DESC
            """)
            
            collection_summary = cursor.fetchall()
            
            return {
                'projections_stats': [
                    {
                        'season': row[0],
                        'week': row[1] or 'Season',
                        'scoring': row[2],
                        'last_updated': row[3],
                        'player_count': row[4]
                    } for row in projections_stats
                ],
                'collection_summary': [
                    {
                        'type': row[0],
                        'last_run': row[1],
                        'total_records': row[2],
                        'total_attempts': row[3]
                    } for row in collection_summary
                ]
            }
            
        except sqlite3.Error as e:
            self.logger.error(f"Error checking data freshness: {e}")
            return {}
        finally:
            self.db.disconnect()

def main():
    """Main execution for FantasyPros data collection"""
    print("🏈 FANTASYPROS API DATA COLLECTOR")
    print("=" * 40)
    
    # You should pass your API key here
    api_key = "nB1YYSxsb98K5cPxATyZf7rgGzTo7GYEa0mKGIk5"  # Replace with environment variable
    collector = FantasyProsCollector(api_key)
    
    while True:
        print(f"\n" + "=" * 40)
        print("COLLECTION OPTIONS:")
        print("1. Test API connection")
        print("2. Collect weekly projections (single scoring)")
        print("3. Collect weekly projections (all scoring formats)")
        print("4. Collect season projections")
        print("5. Get player average projections")
        print("6. Check data freshness")
        print("7. Exit")
        print("=" * 40)
        
        choice = input("\nSelect option (1-7): ").strip()
        
        if choice == '1':
            print("\n🔌 Testing API connection...")
            if collector.test_api_connection():
                print("✅ API connection successful!")
            else:
                print("❌ API connection failed. Check your API key and internet connection.")
        
        elif choice == '2':
            try:
                season = int(input("Enter season year (e.g., 2024): "))
                week = int(input("Enter week number (1-18): "))
                scoring = input("Enter scoring format (PPR/HALF/STD, default PPR): ").strip() or "PPR"
                
                print(f"\n📊 Collecting {season} Week {week} projections ({scoring})...")
                if collector.collect_projections(season, week, scoring=scoring):
                    print("✅ Projections collected successfully")
                else:
                    print("❌ Projections collection failed")
            except ValueError:
                print("❌ Invalid input. Please enter valid numbers.")
        
        elif choice == '3':
            try:
                season = int(input("Enter season year (e.g., 2024): "))
                week = int(input("Enter week number (1-18): "))
                
                print(f"\n📊 Collecting {season} Week {week} projections (ALL scoring formats)...")
                if collector.collect_weekly_projections_all_scoring(season, week):
                    print("✅ All scoring format projections collected successfully")
                else:
                    print("❌ Some projections collection failed")
            except ValueError:
                print("❌ Invalid input. Please enter valid numbers.")
        
        elif choice == '4':
            try:
                season = int(input("Enter season year (e.g., 2024): "))
                scoring = input("Enter scoring format (PPR/HALF/STD, default PPR): ").strip() or "PPR"
                
                print(f"\n📅 Collecting {season} season projections ({scoring})...")
                if collector.collect_season_projections(season, scoring):
                    print("✅ Season projections collected successfully")
                else:
                    print("❌ Season projections collection failed")
            except ValueError:
                print("❌ Invalid input. Please enter valid numbers.")
        
        elif choice == '5':
            try:
                player_id = input("Enter player ID: ").strip()
                season = int(input("Enter season year (e.g., 2024): "))
                scoring = input("Enter scoring format (PPR/HALF/STD, default PPR): ").strip() or "PPR"
                
                print(f"\n📈 Getting average projections for {player_id}...")
                avg_projections = collector.get_average_projections(player_id, season, scoring=scoring)
                
                if avg_projections:
                    print("✅ Average Projections:")
                    for key, value in avg_projections.items():
                        print(f"  {key}: {value}")
                else:
                    print("❌ No projections found for this player")
            except ValueError:
                print("❌ Invalid input. Please enter valid numbers.")
        
        elif choice == '6':
            print("\n📈 FantasyPros Data Freshness Report:")
            print("-" * 40)
            
            freshness = collector.get_data_freshness()
            
            print(f"\nProjections Summary:")
            for proj_info in freshness.get('projections_stats', [])[:10]:  # Show top 10
                print(f"  {proj_info['season']} {proj_info['week']} ({proj_info['scoring']}): "
                      f"{proj_info['player_count']} players - {proj_info['last_updated']}")
            
            print(f"\nCollection History:")
            for summary in freshness.get('collection_summary', []):
                print(f"  {summary['type']}: "
                      f"{summary['total_records']} records "
                      f"(last: {summary['last_run']})")
        
        elif choice == '7':
            print("👋 Exiting FantasyPros collector")
            break
        
        else:
            print("❌ Invalid option, please try again")

if __name__ == "__main__":
    main()