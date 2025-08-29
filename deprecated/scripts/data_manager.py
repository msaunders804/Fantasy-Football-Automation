# data_manager.py
"""
Fantasy Football Data Manager
Unified interface for managing all data collection and database operations
"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
import json
import os

from database_setup import FantasyDatabase
from sleeper_collector import SleeperCollector
from pfr_collector import PFRCollector

class FantasyDataManager:
    """Unified data management for fantasy football analytics"""
    
    def __init__(self, db_path="fantasy_football.db"):
        self.db_path = db_path
        self.db = FantasyDatabase(db_path)
        self.sleeper = SleeperCollector(db_path)
        self.pfr = PFRCollector(db_path)
        self.setup_logging()
    
    def setup_logging(self):
        """Configure logging for data manager"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('fantasy_data_manager.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def initialize_database(self) -> bool:
        """Initialize database with schema and reference data"""
        self.logger.info("Initializing fantasy football database...")
        
        if not self.db.connect():
            self.logger.error("Failed to connect to database")
            return False
        
        try:
            # Create all tables
            if not self.db.create_tables():
                self.logger.error("Failed to create database schema")
                return False
            
            # Insert reference data
            if not self.db.insert_teams():
                self.logger.error("Failed to insert team data")
                return False
            
            if not self.db.insert_seasons():
                self.logger.error("Failed to insert season data")
                return False
            
            self.logger.info("✅ Database initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
            return False
        finally:
            self.db.disconnect()
    
    def collect_foundation_data(self, seasons: List[int] = None) -> Dict[str, bool]:
        """Collect foundational data from all sources"""
        if seasons is None:
            seasons = [2021, 2022, 2023]
        
        self.logger.info(f"Starting foundation data collection for seasons: {seasons}")
        
        results = {
            'sleeper_players': False,
            'sleeper_historical': False,
            'pfr_historical': False,
            'overall_success': False
        }
        
        # Step 1: Collect current players from Sleeper
        self.logger.info("Step 1: Collecting player database...")
        results['sleeper_players'] = self.sleeper.collect_players()
        
        if not results['sleeper_players']:
            self.logger.error("Failed to collect players - cannot proceed")
            return results
        
        # Step 2: Collect historical Sleeper data
        self.logger.info("Step 2: Collecting Sleeper historical data...")
        results['sleeper_historical'] = self.sleeper.collect_historical_data(seasons)
        
        # Step 3: Collect historical PFR data (optional - takes longer)
        collect_pfr = input("\nCollect Pro Football Reference data? (slower but adds advanced stats) (y/N): ").strip().lower()
        
        if collect_pfr == 'y':
            self.logger.info("Step 3: Collecting PFR historical data...")
            print("⚠️  This will take 30-45 minutes due to respectful rate limiting")
            results['pfr_historical'] = self.pfr.collect_historical_data(seasons)
        else:
            self.logger.info("Step 3: Skipping PFR data collection")
            results['pfr_historical'] = True  # Don't count as failure
        
        # Overall success if we have Sleeper data (minimum requirement)
        results['overall_success'] = results['sleeper_players'] and results['sleeper_historical']
        
        if results['overall_success']:
            self.logger.info("✅ Foundation data collection completed successfully")
        else:
            self.logger.error("❌ Foundation data collection failed")
        
        return results
    
    def update_current_data(self) -> Dict[str, bool]:
        """Update with current week/season data"""
        self.logger.info("Updating current data...")
        
        results = {
            'sleeper_players': False,
            'sleeper_current': False,
            'overall_success': False
        }
        
        # Update player database
        results['sleeper_players'] = self.sleeper.collect_players()
        
        # Update current week stats
        results['sleeper_current'] = self.sleeper.update_current_week()
        
        results['overall_success'] = results['sleeper_players'] and results['sleeper_current']
        
        return results
    
    def get_data_summary(self) -> Dict:
        """Get comprehensive summary of available data"""
        if not self.db.connect():
            return {}
        
        try:
            cursor = self.db.connection.cursor()
            
            summary = {
                'database_file': os.path.abspath(self.db_path),
                'database_size_mb': round(os.path.getsize(self.db_path) / (1024*1024), 2),
                'last_updated': datetime.now().isoformat(),
                'tables': {},
                'data_freshness': {},
                'collection_history': {}
            }
            
            # Get table counts
            tables = ['players', 'player_stats', 'advanced_stats', 'team_stats', 'data_collection_log']
            
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                summary['tables'][table] = count
            
            # Get players by position
            cursor.execute("""
                SELECT position, COUNT(*) 
                FROM players 
                WHERE position IS NOT NULL 
                GROUP BY position 
                ORDER BY COUNT(*) DESC
            """)
            
            summary['players_by_position'] = dict(cursor.fetchall())
            
            # Get stats by season
            cursor.execute("""
                SELECT season_year, COUNT(DISTINCT player_id), COUNT(*), MAX(week)
                FROM player_stats 
                GROUP BY season_year 
                ORDER BY season_year DESC
            """)
            
            summary['stats_by_season'] = [
                {
                    'season': row[0],
                    'unique_players': row[1],
                    'total_records': row[2],
                    'latest_week': row[3]
                } for row in cursor.fetchall()
            ]
            
            # Get data freshness
            cursor.execute("""
                SELECT data_source, collection_type, MAX(completed_at), 
                       SUM(records_collected), COUNT(*) as attempts,
                       SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successes
                FROM data_collection_log 
                GROUP BY data_source, collection_type
                ORDER BY MAX(completed_at) DESC
            """)
            
            summary['collection_history'] = [
                {
                    'source': row[0],
                    'type': row[1],
                    'last_run': row[2],
                    'total_records': row[3],
                    'attempts': row[4],
                    'successes': row[5],
                    'success_rate': f"{(row[5]/row[4]*100):.1f}%" if row[4] > 0 else "N/A"
                } for row in cursor.fetchall()
            ]
            
            # Get recent top performers
            cursor.execute("""
                SELECT p.full_name, p.position, p.team, ps.fantasy_points_ppr, ps.week, ps.season_year
                FROM player_stats ps
                JOIN players p ON ps.player_id = p.player_id
                WHERE ps.fantasy_points_ppr > 20
                ORDER BY ps.season_year DESC, ps.week DESC, ps.fantasy_points_ppr DESC
                LIMIT 10
            """)
            
            summary['recent_top_performers'] = [
                {
                    'name': row[0],
                    'position': row[1],
                    'team': row[2],
                    'points': row[3],
                    'week': row[4],
                    'season': row[5]
                } for row in cursor.fetchall()
            ]
            
            return summary
            
        except sqlite3.Error as e:
            self.logger.error(f"Error getting data summary: {e}")
            return {}
        finally:
            self.db.disconnect()
    
    def export_data(self, export_format='csv', output_dir='exports') -> bool:
        """Export data for external analysis"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        if not self.db.connect():
            return False
        
        try:
            # Export main tables to CSV
            tables_to_export = {
                'players': 'SELECT * FROM players',
                'player_stats': 'SELECT * FROM player_stats ORDER BY season_year, week, player_id',
                'advanced_stats': 'SELECT * FROM advanced_stats ORDER BY season_year, week, player_id',
                'team_stats': 'SELECT * FROM team_stats ORDER BY season_year, week, team_abbr'
            }
            
            for table_name, query in tables_to_export.items():
                df = pd.read_sql_query(query, self.db.connection)
                
                if export_format == 'csv':
                    output_file = os.path.join(output_dir, f"{table_name}.csv")
                    df.to_csv(output_file, index=False)
                elif export_format == 'json':
                    output_file = os.path.join(output_dir, f"{table_name}.json")
                    df.to_json(output_file, orient='records', indent=2)
                
                self.logger.info(f"Exported {len(df)} records from {table_name} to {output_file}")
            
            # Export summary report
            summary = self.get_data_summary()
            summary_file = os.path.join(output_dir, "data_summary.json")
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            self.logger.info(f"✅ Data export completed to {output_dir}/")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting data: {e}")
            return False
        finally:
            self.db.disconnect()
    
    def validate_data_quality(self) -> Dict:
        """Run data quality checks"""
        if not self.db.connect():
            return {}
        
        try:
            cursor = self.db.connection.cursor()
            
            quality_report = {
                'checks_run': datetime.now().isoformat(),
                'issues_found': [],
                'data_completeness': {},
                'data_consistency': {},
                'recommendations': []
            }
            
            # Check 1: Players without stats
            cursor.execute("""
                SELECT COUNT(*) FROM players p
                LEFT JOIN player_stats ps ON p.player_id = ps.player_id
                WHERE ps.player_id IS NULL AND p.position IN ('QB', 'RB', 'WR', 'TE')
            """)
            
            players_without_stats = cursor.fetchone()[0]
            if players_without_stats > 50:
                quality_report['issues_found'].append(f"{players_without_stats} relevant players have no stats")
            
            # Check 2: Fantasy points calculation validation
            cursor.execute("""
                SELECT player_id, season_year, week, 
                       fantasy_points_ppr,
                       (pass_yd * 0.04 + pass_td * 4 - pass_int * 2 + 
                        rush_yd * 0.1 + rush_td * 6 + 
                        rec_yd * 0.1 + rec_td * 6 + rec) as calculated_ppr
                FROM player_stats 
                WHERE ABS(fantasy_points_ppr - calculated_ppr) > 1
                LIMIT 5
            """)
            
            calculation_errors = cursor.fetchall()
            if calculation_errors:
                quality_report['issues_found'].append(f"{len(calculation_errors)} records with fantasy point calculation errors")
            
            # Check 3: Data completeness by season
            cursor.execute("""
                SELECT season_year, COUNT(DISTINCT player_id), 
                       COUNT(*), AVG(week)
                FROM player_stats 
                GROUP BY season_year
                ORDER BY season_year DESC
            """)
            
            for row in cursor.fetchall():
                season, unique_players, total_records, avg_week = row
                quality_report['data_completeness'][str(season)] = {
                    'unique_players': unique_players,
                    'total_records': total_records,
                    'avg_week': round(avg_week, 1)
                }
                
                if unique_players < 200:
                    quality_report['issues_found'].append(f"Season {season} has only {unique_players} players")
            
            # Check 4: Missing team assignments
            cursor.execute("""
                SELECT COUNT(*) FROM players 
                WHERE team IS NULL AND position IN ('QB', 'RB', 'WR', 'TE')
            """)
            
            missing_teams = cursor.fetchone()[0]
            if missing_teams > 0:
                quality_report['issues_found'].append(f"{missing_teams} players missing team assignments")
            
            # Check 5: Unrealistic stat values
            cursor.execute("""
                SELECT COUNT(*) FROM player_stats 
                WHERE pass_yd > 600 OR rush_yd > 300 OR rec_yd > 300 
                   OR pass_td > 8 OR rush_td > 4 OR rec_td > 4
            """)
            
            unrealistic_stats = cursor.fetchone()[0]
            if unrealistic_stats > 0:
                quality_report['issues_found'].append(f"{unrealistic_stats} records with potentially unrealistic stats")
            
            # Generate recommendations
            if not quality_report['issues_found']:
                quality_report['recommendations'].append("Data quality looks good! No major issues detected.")
            else:
                if players_without_stats > 50:
                    quality_report['recommendations'].append("Consider running additional data collection to fill missing player stats")
                
                if missing_teams > 0:
                    quality_report['recommendations'].append("Update player database to fill missing team assignments")
                
                if unrealistic_stats > 0:
                    quality_report['recommendations'].append("Review and potentially clean records with extreme statistical values")
            
            quality_report['overall_score'] = max(0, 100 - len(quality_report['issues_found']) * 10)
            
            return quality_report
            
        except sqlite3.Error as e:
            self.logger.error(f"Error running data quality checks: {e}")
            return {}
        finally:
            self.db.disconnect()
    
    def backup_database(self, backup_dir='backups') -> bool:
        """Create database backup"""
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = os.path.join(backup_dir, f"fantasy_football_backup_{timestamp}.db")
        
        try:
            # Simple file copy for SQLite
            import shutil
            shutil.copy2(self.db_path, backup_file)
            
            self.logger.info(f"✅ Database backed up to {backup_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating backup: {e}")
            return False
    
    def quick_query(self, query: str) -> pd.DataFrame:
        """Execute quick SQL query and return as DataFrame"""
        if not self.db.connect():
            return pd.DataFrame()
        
        try:
            df = pd.read_sql_query(query, self.db.connection)
            return df
        except Exception as e:
            self.logger.error(f"Error executing query: {e}")
            return pd.DataFrame()
        finally:
            self.db.disconnect()

def main():
    """Main interface for fantasy data management"""
    print("🏈 FANTASY FOOTBALL DATA MANAGER")
    print("=" * 45)
    
    manager = FantasyDataManager()
    
    # Check if database exists
    if not os.path.exists(manager.db_path):
        print(f"Database not found. Initializing new database at {manager.db_path}")
        if not manager.initialize_database():
            print("❌ Failed to initialize database")
            return
    
    while True:
        print(f"\n" + "=" * 45)
        print("DATA MANAGEMENT OPTIONS:")
        print("1. Initialize database (first time setup)")
        print("2. Collect foundation data (historical)")
        print("3. Update current data")
        print("4. View data summary")
        print("5. Run data quality checks")
        print("6. Export data")
        print("7. Backup database")
        print("8. Run custom query")
        print("9. Access individual collectors")
        print("10. Exit")
        print("=" * 45)
        
        choice = input("\nSelect option (1-10): ").strip()
        
        if choice == '1':
            print("\n🗄️ Initializing database...")
            if manager.initialize_database():
                print("✅ Database initialized successfully")
            else:
                print("❌ Database initialization failed")
        
        elif choice == '2':
            print("\n📚 Foundation Data Collection")
            print("This will collect:")
            print("  • All NFL players")
            print("  • 3 seasons of statistical data")
            print("  • Optional advanced stats from PFR")
            print("\nThis may take 15-60 minutes depending on options selected")
            
            confirm = input("\nProceed with foundation data collection? (y/N): ").strip().lower()
            if confirm == 'y':
                seasons_input = input("Enter seasons (e.g., 2021,2022,2023) or press Enter for default: ").strip()
                
                if seasons_input:
                    seasons = [int(s.strip()) for s in seasons_input.split(',')]
                else:
                    seasons = [2021, 2022, 2023]
                
                results = manager.collect_foundation_data(seasons)
                
                print(f"\n📊 Collection Results:")
                print(f"  Players: {'✅' if results['sleeper_players'] else '❌'}")
                print(f"  Sleeper Historical: {'✅' if results['sleeper_historical'] else '❌'}")
                print(f"  PFR Historical: {'✅' if results['pfr_historical'] else '❌'}")
                print(f"  Overall: {'✅ SUCCESS' if results['overall_success'] else '❌ FAILED'}")
            else:
                print("Foundation collection cancelled")
        
        elif choice == '3':
            print("\n🔄 Updating current data...")
            results = manager.update_current_data()
            
            print(f"\n📊 Update Results:")
            print(f"  Players: {'✅' if results['sleeper_players'] else '❌'}")
            print(f"  Current Stats: {'✅' if results['sleeper_current'] else '❌'}")
            print(f"  Overall: {'✅ SUCCESS' if results['overall_success'] else '❌ FAILED'}")
        
        elif choice == '4':
            print("\n📊 Data Summary")
            print("-" * 30)
            
            summary = manager.get_data_summary()
            
            if summary:
                print(f"Database: {summary['database_file']}")
                print(f"Size: {summary['database_size_mb']} MB")
                
                print(f"\nTable Counts:")
                for table, count in summary['tables'].items():
                    print(f"  {table}: {count:,}")
                
                print(f"\nPlayers by Position:")
                for pos, count in summary['players_by_position'].items():
                    print(f"  {pos}: {count}")
                
                print(f"\nStats by Season:")
                for season_info in summary['stats_by_season']:
                    print(f"  {season_info['season']}: {season_info['unique_players']} players, "
                          f"{season_info['total_records']:,} records, Week {season_info['latest_week']}")
                
                print(f"\nRecent Top Performers:")
                for performer in summary['recent_top_performers'][:5]:
                    print(f"  {performer['name']} ({performer['position']}, {performer['team']}): "
                          f"{performer['points']} pts ({performer['season']} Week {performer['week']})")
            else:
                print("❌ Could not retrieve data summary")
        
        elif choice == '5':
            print("\n🔍 Running data quality checks...")
            quality_report = manager.validate_data_quality()
            
            if quality_report:
                print(f"Overall Quality Score: {quality_report['overall_score']}/100")
                
                if quality_report['issues_found']:
                    print(f"\n⚠️  Issues Found:")
                    for issue in quality_report['issues_found']:
                        print(f"  • {issue}")
                else:
                    print(f"\n✅ No major issues detected")
                
                print(f"\n💡 Recommendations:")
                for rec in quality_report['recommendations']:
                    print(f"  • {rec}")
            else:
                print("❌ Could not run quality checks")
        
        elif choice == '6':
            print("\n📤 Export Data")
            format_choice = input("Export format (csv/json, default csv): ").strip().lower() or 'csv'
            output_dir = input("Output directory (default 'exports'): ").strip() or 'exports'
            
            if manager.export_data(format_choice, output_dir):
                print(f"✅ Data exported to {output_dir}/")
            else:
                print("❌ Export failed")
        
        elif choice == '7':
            print("\n💾 Creating database backup...")
            if manager.backup_database():
                print("✅ Backup created successfully")
            else:
                print("❌ Backup failed")
        
        elif choice == '8':
            print("\n🔍 Custom Query")
            print("Enter SQL query (or 'examples' to see sample queries):")
            query = input("SQL> ").strip()
            
            if query.lower() == 'examples':
                print("\nSample Queries:")
                print("1. SELECT * FROM players WHERE position = 'QB' LIMIT 10")
                print("2. SELECT p.full_name, SUM(ps.fantasy_points_ppr) as total_points")
                print("   FROM players p JOIN player_stats ps ON p.player_id = ps.player_id")
                print("   WHERE ps.season_year = 2023 GROUP BY p.player_id ORDER BY total_points DESC LIMIT 10")
                print("3. SELECT team, AVG(fantasy_points_ppr) as avg_points FROM player_stats")
                print("   WHERE season_year = 2023 GROUP BY team ORDER BY avg_points DESC")
                continue
            
            if query:
                try:
                    df = manager.quick_query(query)
                    if not df.empty:
                        print(f"\nResults ({len(df)} rows):")
                        if len(df) > 20:
                            print(df.head(10))
                            print("...")
                            print(df.tail(10))
                        else:
                            print(df)
                    else:
                        print("No results returned")
                except Exception as e:
                    print(f"❌ Query error: {e}")
        
        elif choice == '9':
            print("\n🔧 Individual Collectors")
            print("1. Sleeper API Collector")
            print("2. Pro Football Reference Collector")
            
            collector_choice = input("Select collector (1-2): ").strip()
            
            if collector_choice == '1':
                print("Launching Sleeper collector...")
                from sleeper_collector import main as sleeper_main
                sleeper_main()
            elif collector_choice == '2':
                print("Launching PFR collector...")
                from pfr_collector import main as pfr_main
                pfr_main()
        
        elif choice == '10':
            print("👋 Exiting Data Manager")
            break
        
        else:
            print("❌ Invalid option, please try again")

if __name__ == "__main__":
    main()