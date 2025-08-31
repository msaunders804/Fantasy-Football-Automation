import pandas as pd
import json
import csv
from typing import List, Dict, Any, Optional
from core.database import DatabaseManager
from projections.aggregator import ProjectionAggregator
from utils.data_utils import standardize_player_name, validate_projection_data

class ProjectionManager:
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.aggregator = ProjectionAggregator(db_manager)
    
    def import_from_file(self, source_name: str, file_path: str, file_format: str = 'csv') -> int:
        """Import projections from CSV or JSON file"""
        try:
            if file_format.lower() == 'csv':
                return self._import_csv(source_name, file_path)
            elif file_format.lower() == 'json':
                return self._import_json(source_name, file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
        except Exception as e:
            raise Exception(f"Failed to import projections: {e}")
    
    def _import_csv(self, source_name: str, file_path: str) -> int:
        """Import projections from CSV file"""
        df = pd.read_csv(file_path)
        return self._process_dataframe(source_name, df)
    
    def _import_json(self, source_name: str, file_path: str) -> int:
        """Import projections from JSON file"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = pd.json_normalize(data)
            
        return self._process_dataframe(source_name, df)
    
    def _process_dataframe(self, source_name: str, df: pd.DataFrame) -> int:
        """Process DataFrame and insert into database"""
        # Get or create source ID
        source_id = self.db.add_projection_source(source_name)
        
        # Standardize column names
        df = self._standardize_columns(df)
        
        # Validate required columns
        required_cols = ['name', 'position']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns: {required_cols}")
        
        projections = []
        players_to_upsert = []
        
        for _, row in df.iterrows():
            # Standardize player name
            player_name = standardize_player_name(row['name'])
            player_id = self._generate_player_id(player_name, row['position'])
            
            # Prepare player data
            players_to_upsert.append({
                'player_id': player_id,
                'name': player_name,
                'position': row['position'],
                'team': row.get('team'),
                'bye_week': row.get('bye_week')
            })
            
            # Prepare projection data
            projection = {
                'player_id': player_id,
                'source_id': source_id,
                'week': row.get('week', 0),
                'points': row.get('fantasy_points') or row.get('points'),
                'pass_yds': row.get('pass_yds', 0),
                'pass_tds': row.get('pass_tds', 0),
                'interceptions': row.get('interceptions', 0),
                'rush_yds': row.get('rush_yds', 0),
                'rush_tds': row.get('rush_tds', 0),
                'receptions': row.get('receptions', 0),
                'rec_yds': row.get('rec_yds', 0),
                'rec_tds': row.get('rec_tds', 0)
            }
            
            # Validate projection data
            if validate_projection_data(projection):
                projections.append(projection)
        
        # Bulk upsert players
        for player in players_to_upsert:
            self.db.upsert_player(**player)
        
        # Bulk insert projections
        count = self.db.bulk_insert_projections(projections)
        return count
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to match expected format"""
        column_mapping = {
            'player_name': 'name',
            'player': 'name',
            'pos': 'position',
            'tm': 'team',
            'fantasy_points': 'points',
            'fpts': 'points',
            'passing_yards': 'pass_yds',
            'passing_tds': 'pass_tds',
            'rushing_yards': 'rush_yds',
            'rushing_tds': 'rush_tds',
            'receiving_yards': 'rec_yds',
            'receiving_tds': 'rec_tds',
            'targets': 'targets',
            'int': 'interceptions'
        }
        
        # Apply column mapping
        df = df.rename(columns=column_mapping)
        
        # Convert column names to lowercase
        df.columns = df.columns.str.lower()
        
        return df
    
    def _generate_player_id(self, name: str, position: str) -> str:
        """Generate consistent player ID"""
        # Remove spaces and special characters, convert to lowercase
        clean_name = ''.join(c for c in name.lower() if c.isalnum())
        return f"{clean_name}_{position.lower()}"
    
    def aggregate_projections(self) -> bool:
        """Run ML aggregation on all projections"""
        try:
            return self.aggregator.aggregate_all_projections()
        except Exception as e:
            raise Exception(f"Failed to aggregate projections: {e}")
    
    def get_rankings(self, position: str = None, limit: int = 50) -> List[Dict]:
        """Get current player rankings"""
        query = """
        SELECT p.name, p.position, p.team, ap.final_points as points,
               ap.vbd_score, ap.tier, ap.confidence_score
        FROM aggregated_projections ap
        JOIN players p ON ap.player_id = p.player_id
        WHERE 1=1
        """
        params = []
        
        if position:
            query += " AND p.position = ?"
            params.append(position)
            
        query += " ORDER BY ap.vbd_score DESC"
        
        if limit:
            query += f" LIMIT {limit}"
            
        return self.db.execute_query(query, tuple(params))
    
    def get_player_projections(self, player_id: str) -> Dict:
        """Get all projections for a specific player"""
        projections = self.db.get_projections(player_id=player_id)
        
        if not projections:
            return None
            
        # Get aggregated projection
        agg_query = """
        SELECT * FROM aggregated_projections 
        WHERE player_id = ? AND week = 0
        """
        agg_result = self.db.execute_query(agg_query, (player_id,))
        
        return {
            'player_id': player_id,
            'raw_projections': projections,
            'aggregated': agg_result[0] if agg_result else None
        }
    
    # Add these methods to the existing ProjectionManager class

    def import_from_fantasypros_api(self, api_key: str, week: int = 0, 
                                scoring: str = 'PPR') -> int:
        """Import projections from FantasyPros API"""
        try:
            from integrations.fantasypros import FantasyProAPI
            
            fp_api = FantasyProAPI(api_key)
            projections_data = fp_api.get_all_positions_projections(week, scoring)
            
            if not projections_data:
                raise ValueError("No projections returned from FantasyPros API")
            
            # Convert to DataFrame and process
            df = pd.DataFrame(projections_data)
            source_name = f"FantasyPros_API_{scoring}"
            
            return self._process_dataframe(source_name, df)
            
        except Exception as e:
            raise Exception(f"Failed to import from FantasyPros API: {e}")

    def import_from_espn_csv(self, file_path: str) -> int:
        """Import projections from ESPN CSV format"""
        try:
            df = pd.read_csv(file_path)
            
            # ESPN-specific column standardization
            df = self._standardize_espn_columns(df)
            
            source_name = "ESPN_CSV"
            return self._process_dataframe(source_name, df)
            
        except Exception as e:
            raise Exception(f"Failed to import ESPN CSV: {e}")

    def _standardize_espn_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize ESPN CSV column names"""
        espn_column_mapping = {
            'Player': 'name',
            'PLAYER': 'name',
            'Name': 'name',
            'POS': 'position',
            'Position': 'position',
            'TEAM': 'team',
            'Team': 'team',
            'FPTS': 'points',
            'Fantasy Points': 'points',
            'Projected Points': 'points',
            'PROJ': 'points',
            'Pass Yds': 'pass_yds',
            'Passing Yards': 'pass_yds',
            'PassYds': 'pass_yds',
            'Pass TD': 'pass_tds',
            'Passing TDs': 'pass_tds',
            'PassTD': 'pass_tds',
            'INT': 'interceptions',
            'Interceptions': 'interceptions',
            'Rush Yds': 'rush_yds',
            'Rushing Yards': 'rush_yds',
            'RushYds': 'rush_yds',
            'Rush TD': 'rush_tds',
            'Rushing TDs': 'rush_tds',
            'RushTD': 'rush_tds',
            'Rec': 'receptions',
            'Receptions': 'receptions',
            'Rec Yds': 'rec_yds',
            'Receiving Yards': 'rec_yds',
            'RecYds': 'rec_yds',
            'Rec TD': 'rec_tds',
            'Receiving TDs': 'rec_tds',
            'RecTD': 'rec_tds'
        }
        
        # Apply ESPN-specific mapping first
        df = df.rename(columns=espn_column_mapping)
        
        # Then apply standard column mapping
        df = self._standardize_columns(df)
        
        # ESPN specific cleaning
        if 'name' in df.columns:
            # ESPN often includes position in player name like "Josh Allen QB"
            df['name'] = df['name'].apply(self._clean_espn_player_name)
        
        return df

    def _clean_espn_player_name(self, name: str) -> str:
        """Clean ESPN player names that include position"""
        if not isinstance(name, str):
            return str(name)
        
        # Remove common ESPN suffixes
        espn_suffixes = [' QB', ' RB', ' WR', ' TE', ' K', ' D/ST', ' DEF']
        
        for suffix in espn_suffixes:
            if name.endswith(suffix):
                name = name[:-len(suffix)].strip()
        
        # Remove team abbreviations in parentheses like "(BUF)"
        import re
        name = re.sub(r'\s*\([A-Z]{2,4}\)\s*', ' ', name).strip()
        
        return name