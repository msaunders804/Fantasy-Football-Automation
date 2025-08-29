# sleeper_collector.py
"""
Sleeper API Data Collector
Handles all interactions with Sleeper API and stores data in SQLite database
"""

import requests
import sqlite3
import json
import time
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
from database_setup import FantasyDatabase

class SleeperCollector:
    """Collects and stores data from Sleeper API"""
    
    def __init__(self, db_path="fantasy_football.db"):
        self.base_url = "https://api.sleeper.app/v1"
        self.db = FantasyDatabase(db_path)
        self.session = requests.Session()
        self.setup_logging()
        
        # Rate limiting - Sleeper is generous but be respectful
        self.rate_limit_delay = 0.1  # 100ms between requests
        self.last_request_time = 0
    
    def setup_logging(self):
        """Configure logging for Sleeper collector"""
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
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            self.logger.debug(f"API request successful: {endpoint}")
            return response.json()
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed for {endpoint}: {e}")
            return None
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error for {endpoint}: {e}")
            return None
    
    def collect_players(self) -> bool:
        """Collect all NFL players from Sleeper API"""
        start_time = datetime.now()
        self.logger.info("Starting player data collection from Sleeper...")
        
        if not self.db.connect():
            return False
        
        try:
            # Get all players
            players_data = self._make_request("players/nfl")
            if not players_data:
                self.logger.error("Failed to fetch players data")
                return False
            
            cursor = self.db.connection.cursor()
            players_inserted = 0
            
            for sleeper_id, player_info in players_data.items():
                try:
                    # Parse player data
                    first_name = player_info.get('first_name', '')
                    last_name = player_info.get('last_name', '')
                    full_name = f"{first_name} {last_name}".strip()
                    
                    # Create consistent player_id (using sleeper_id as primary)
                    player_id = sleeper_id
                    
                    # Insert or update player
                    cursor.execute("""
                        INSERT OR REPLACE INTO players (
                            player_id, sleeper_id, first_name, last_name, full_name,
                            position, team, age, height, weight, years_exp, college, status,
                            updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        player_id,
                        sleeper_id,
                        first_name,
                        last_name,
                        full_name,
                        player_info.get('position'),
                        player_info.get('team'),
                        player_info.get('age'),
                        player_info.get('height'),
                        player_info.get('weight'),
                        player_info.get('years_exp'),
                        player_info.get('college'),
                        player_info.get('status', 'Active'),
                        datetime.now()
                    ))
                    
                    players_inserted += 1
                    
                    if players_inserted % 100 == 0:
                        self.logger.info(f"Processed {players_inserted} players...")
                        self.db.connection.commit()
                
                except sqlite3.Error as e:
                    self.logger.error(f"Error inserting player {sleeper_id}: {e}")
                    continue
            
            # Final commit
            self.db.connection.commit()
            
            end_time = datetime.now()
            self.logger.info(f"Successfully collected {players_inserted} players")
            
            # Log the collection
            self.db.log_data_collection(
                data_source='sleeper',
                collection_type='players',
                records_collected=players_inserted,
                success=True,
                started_at=start_time,
                completed_at=end_time
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in player collection: {e}")
            return False
        finally:
            self.db.disconnect()
    
    def collect_weekly_stats(self, season: int, week: int) -> bool:
        """Collect weekly stats for a specific season/week"""
        start_time = datetime.now()
        self.logger.info(f"Collecting stats for {season} Week {week}...")
        
        if not self.db.connect():
            return False
        
        try:
            # Get weekly stats
            stats_data = self._make_request(f"stats/nfl/regular/{season}/{week}")
            if not stats_data:
                self.logger.warning(f"No stats data for {season} Week {week}")
                return False
            
            cursor = self.db.connection.cursor()
            stats_inserted = 0
            
            for player_id, stats in stats_data.items():
                try:
                    # Verify player exists in our database
                    cursor.execute("SELECT player_id FROM players WHERE player_id = ?", (player_id,))
                    if not cursor.fetchone():
                        self.logger.warning(f"Player {player_id} not found in database, skipping stats")
                        continue
                    
                    # Get player's team for this week (might need separate call for historical accuracy)
                    cursor.execute("SELECT team FROM players WHERE player_id = ?", (player_id,))
                    player_team = cursor.fetchone()
                    team = player_team[0] if player_team else None
                    
                    # Calculate fantasy points for different scoring formats
                    fantasy_ppr = self.db.calculate_fantasy_points(stats, 'ppr')
                    fantasy_half = self.db.calculate_fantasy_points(stats, 'half_ppr')
                    fantasy_std = self.db.calculate_fantasy_points(stats, 'standard')
                    
                    # Create game_id (simplified - you might want to enhance this)
                    game_id = f"{season}_week_{week}_{team}" if team else f"{season}_week_{week}_unknown"
                    
                    # Insert stats
                    cursor.execute("""
                        INSERT OR REPLACE INTO player_stats (
                            player_id, game_id, season_year, week, team,
                            pass_att, pass_cmp, pass_yd, pass_td, pass_int, pass_sack, pass_sack_yd,
                            rush_att, rush_yd, rush_td, rush_fum,
                            rec_tgt, rec, rec_yd, rec_td, rec_fum,
                            fantasy_points_ppr, fantasy_points_half_ppr, fantasy_points_std,
                            data_source, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        player_id, game_id, season, week, team,
                        stats.get('pass_att', 0), stats.get('pass_cmp', 0), stats.get('pass_yd', 0),
                        stats.get('pass_td', 0), stats.get('pass_int', 0), stats.get('pass_sack', 0),
                        stats.get('pass_sack_yd', 0),
                        stats.get('rush_att', 0), stats.get('rush_yd', 0), stats.get('rush_td', 0),
                        stats.get('rush_fum', 0),
                        stats.get('rec_tgt', 0), stats.get('rec', 0), stats.get('rec_yd', 0),
                        stats.get('rec_td', 0), stats.get('rec_fum', 0),
                        fantasy_ppr, fantasy_half, fantasy_std,
                        'sleeper', datetime.now()
                    ))
                    
                    stats_inserted += 1
                    
                except sqlite3.Error as e:
                    self.logger.error(f"Error inserting stats for player {player_id}: {e}")
                    continue
            
            self.db.connection.commit()
            
            end_time = datetime.now()
            self.logger.info(f"Successfully collected {stats_inserted} player stats for {season} Week {week}")
            
            # Log the collection
            self.db.log_data_collection(
                data_source='sleeper',
                collection_type='weekly_stats',
                season_year=season,
                week=week,
                records_collected=stats_inserted,
                success=True,
                started_at=start_time,
                completed_at=end_time
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error collecting weekly stats: {e}")
            
            # Log failed collection
            self.db.log_data_collection(
                data_source='sleeper',
                collection_type='weekly_stats',
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
    
    def collect_season_stats(self, season: int, weeks: List[int] = None) -> bool:
        """Collect stats for entire season or specific weeks"""
        if weeks is None:
            weeks = list(range(1, 19))  # Regular season weeks 1-18
        
        self.logger.info(f"Starting collection for {season} season, weeks: {weeks}")
        
        success_count = 0
        total_weeks = len(weeks)
        
        for week in weeks:
            self.logger.info(f"Collecting {season} Week {week} ({success_count + 1}/{total_weeks})")
            
            if self.collect_weekly_stats(season, week):
                success_count += 1
                self.logger.info(f"✅ Week {week} completed")
            else:
                self.logger.error(f"❌ Week {week} failed")
            
            # Brief pause between weeks
            time.sleep(0.5)
        
        self.logger.info(f"Season collection complete: {success_count}/{total_weeks} weeks successful")
        return success_count == total_weeks
    
    def collect_historical_data(self, seasons: List[int] = None) -> bool:
        """Collect historical data for multiple seasons"""
        if seasons is None:
            seasons = [2021, 2022, 2023]
        
        self.logger.info(f"Starting historical data collection for seasons: {seasons}")
        
        # First, ensure we have current players
        if not self.collect_players():
            self.logger.error("Failed to collect players data")
            return False
        
        success_count = 0
        for season in seasons:
            self.logger.info(f"Collecting data for {season} season...")
            
            if self.collect_season_stats(season):
                success_count += 1
                self.logger.info(f"✅ {season} season completed")
            else:
                self.logger.error(f"❌ {season} season had issues")
        
        self.logger.info(f"Historical collection complete: {success_count}/{len(seasons)} seasons successful")
        return success_count > 0
    
    def get_data_freshness(self) -> Dict:
        """Check how fresh our data is"""
        if not self.db.connect():
            return {}
        
        try:
            cursor = self.db.connection.cursor()
            
            # Get latest player update
            cursor.execute("""
                SELECT MAX(updated_at) FROM players
            """)
            latest_player_update = cursor.fetchone()[0]
            
            # Get latest stats by season
            cursor.execute("""
                SELECT season_year, MAX(week), MAX(updated_at), COUNT(DISTINCT player_id)
                FROM player_stats 
                WHERE data_source = 'sleeper'
                GROUP BY season_year
                ORDER BY season_year DESC
            """)
            
            season_stats = cursor.fetchall()
            
            # Get collection log summary
            cursor.execute("""
                SELECT data_source, collection_type, MAX(completed_at), 
                       SUM(records_collected), COUNT(*) as attempts
                FROM data_collection_log 
                WHERE success = 1
                GROUP BY data_source, collection_type
                ORDER BY MAX(completed_at) DESC
            """)
            
            collection_summary = cursor.fetchall()
            
            return {
                'latest_player_update': latest_player_update,
                'season_stats': [
                    {
                        'season': row[0],
                        'latest_week': row[1],
                        'last_updated': row[2],
                        'player_count': row[3]
                    } for row in season_stats
                ],
                'collection_summary': [
                    {
                        'source': row[0],
                        'type': row[1],
                        'last_run': row[2],
                        'total_records': row[3],
                        'total_attempts': row[4]
                    } for row in collection_summary
                ]
            }
            
        except sqlite3.Error as e:
            self.logger.error(f"Error checking data freshness: {e}")
            return {}
        finally:
            self.db.disconnect()
    
    def update_current_week(self) -> bool:
        """Update stats for the current NFL week"""
        # This is a simplified version - you'd want to determine current week dynamically
        current_season = 2024
        
        # Get the latest week we have data for
        if not self.db.connect():
            return False
        
        try:
            cursor = self.db.connection.cursor()
            cursor.execute("""
                SELECT MAX(week) 
                FROM player_stats 
                WHERE season_year = ? AND data_source = 'sleeper'
            """, (current_season,))
            
            result = cursor.fetchone()
            latest_week = result[0] if result[0] else 0
            
            # Try to collect the next week
            next_week = latest_week + 1
            
            if next_week <= 18:  # Regular season only
                self.logger.info(f"Attempting to collect current week: {next_week}")
                return self.collect_weekly_stats(current_season, next_week)
            else:
                self.logger.info("Season appears complete")
                return True
                
        except sqlite3.Error as e:
            self.logger.error(f"Error updating current week: {e}")
            return False
        finally:
            self.db.disconnect()

def main():
    """Main execution for Sleeper data collection"""
    print("🏈 SLEEPER API DATA COLLECTOR")
    print("=" * 40)
    
    collector = SleeperCollector()
    
    while True:
        print(f"\n" + "=" * 40)
        print("COLLECTION OPTIONS:")
        print("1. Collect current players")
        print("2. Collect specific week stats")
        print("3. Collect full season")
        print("4. Collect historical data (2021-2023)")
        print("5. Update current week")
        print("6. Check data freshness")
        print("7. Exit")
        print("=" * 40)
        
        choice = input("\nSelect option (1-7): ").strip()
        
        if choice == '1':
            print("\n📋 Collecting current players...")
            if collector.collect_players():
                print("✅ Players collected successfully")
            else:
                print("❌ Player collection failed")
        
        elif choice == '2':
            try:
                season = int(input("Enter season year (e.g., 2024): "))
                week = int(input("Enter week number (1-18): "))
                
                print(f"\n📊 Collecting {season} Week {week} stats...")
                if collector.collect_weekly_stats(season, week):
                    print("✅ Weekly stats collected successfully")
                else:
                    print("❌ Weekly stats collection failed")
            except ValueError:
                print("❌ Invalid input. Please enter valid numbers.")
        
        elif choice == '3':
            try:
                season = int(input("Enter season year (e.g., 2024): "))
                start_week = int(input("Start week (default 1): ") or "1")
                end_week = int(input("End week (default 18): ") or "18")
                
                weeks = list(range(start_week, end_week + 1))
                print(f"\n📅 Collecting {season} season weeks {start_week}-{end_week}...")
                
                if collector.collect_season_stats(season, weeks):
                    print("✅ Season stats collected successfully")
                else:
                    print("❌ Season stats collection had issues")
            except ValueError:
                print("❌ Invalid input. Please enter valid numbers.")
        
        elif choice == '4':
            print("\n📚 Collecting historical data (2021-2023)...")
            print("This may take several minutes...")
            
            if collector.collect_historical_data():
                print("✅ Historical data collection completed")
            else:
                print("❌ Historical data collection had issues")
        
        elif choice == '5':
            print("\n🔄 Updating current week...")
            if collector.update_current_week():
                print("✅ Current week updated successfully")
            else:
                print("❌ Current week update failed")
        
        elif choice == '6':
            print("\n📈 Data Freshness Report:")
            print("-" * 30)
            
            freshness = collector.get_data_freshness()
            
            if freshness.get('latest_player_update'):
                print(f"Players last updated: {freshness['latest_player_update']}")
            
            print(f"\nSeason Stats Summary:")
            for season_info in freshness.get('season_stats', []):
                print(f"  {season_info['season']}: Week {season_info['latest_week']} "
                      f"({season_info['player_count']} players)")
            
            print(f"\nCollection History:")
            for summary in freshness.get('collection_summary', []):
                print(f"  {summary['source']} {summary['type']}: "
                      f"{summary['total_records']} records "
                      f"(last: {summary['last_run']})")
        
        elif choice == '7':
            print("👋 Exiting Sleeper collector")
            break
        
        else:
            print("❌ Invalid option, please try again")

if __name__ == "__main__":
    main()