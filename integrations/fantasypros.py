import requests
import time
import pandas as pd
from typing import Dict, List, Optional, Any
from config.settings import FANTASYPROS_DELAY

class FantasyProAPI:
    """FantasyPros API client for fetching projections"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.base_url = "https://api.fantasypros.com/public/v2/json/nfl"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Fantasy-Draft-Assistant/1.0'
        })
        
        if api_key:
            self.session.headers.update({
                'x-api-key': api_key
            })
    
    def get_projections(self, position: str = None, week: int = 0, 
                       scoring: str = 'PPR', year: int = 2024) -> List[Dict]:
        """
        Get projections from FantasyPros API
        
        Args:
            position: Player position (QB, RB, WR, TE, K, DST)
            week: Week number (0 for season-long)
            scoring: Scoring format (PPR, HALF, STD)
            year: NFL season year
        """
        try:
            # Build URL based on your working curl command
            url = f"{self.base_url}/{year}/projections"
            
            params = {}
            
            # Add position filter if specified
            if position:
                params['position'] = position
            
            # Add scoring format if not PPR (default)
            if scoring.upper() != 'PPR':
                params['scoring'] = scoring.lower()
            
            print(f"Making API request to: {url}")
            print(f"Parameters: {params}")
            print(f"Headers: {dict(self.session.headers)}")
            
            response = self.session.get(url, params=params)
            
            print(f"Response status: {response.status_code}")
            
            # Rate limiting
            time.sleep(FANTASYPROS_DELAY)
            
            if response.status_code == 200:
                try:
                    json_data = response.json()
                    print(f"Response keys: {list(json_data.keys()) if isinstance(json_data, dict) else 'List response'}")
                    return self._parse_fantasypros_response(json_data, position)
                except Exception as e:
                    print(f"Error parsing JSON response: {e}")
                    print(f"Response content: {response.text[:500]}...")
                    return []
            else:
                print(f"FantasyPros API error: {response.status_code}")
                print(f"Response: {response.text}")
                return []
                
        except Exception as e:
            print(f"Error fetching FantasyPros projections: {e}")
            return []
    
    def _parse_fantasypros_response(self, data: Dict, position: str = None) -> List[Dict]:
        """Parse FantasyPros API response into standardized format"""
        projections = []
        
        print(f"Parsing response data type: {type(data)}")
        
        # Handle different response structures
        players_data = []
        
        if isinstance(data, dict):
            # Common FantasyPros response structures
            if 'players' in data:
                players_data = data['players']
            elif 'data' in data:
                players_data = data['data']
            elif 'results' in data:
                players_data = data['results']
            else:
                # Check if the dict itself contains player data
                if 'player_name' in data or 'name' in data:
                    players_data = [data]
                else:
                    # Try to find the main data array
                    for key, value in data.items():
                        if isinstance(value, list) and len(value) > 0:
                            if isinstance(value[0], dict) and ('name' in value[0] or 'player_name' in value[0]):
                                players_data = value
                                break
        elif isinstance(data, list):
            players_data = data
        
        print(f"Found {len(players_data)} players in response")
        
        for i, player_data in enumerate(players_data):
            try:
                if i < 3:  # Debug first few players
                    print(f"Sample player data {i+1}: {player_data}")
                
                projection = {
                    'name': self._get_player_name(player_data),
                    'position': self._get_position(player_data, position),
                    'team': self._get_team(player_data),
                    'bye_week': self._get_bye_week(player_data),
                    'points': self._get_fantasy_points(player_data),
                    'pass_yds': self._get_stat(player_data, ['passing_yards', 'pass_yds', 'py']),
                    'pass_tds': self._get_stat(player_data, ['passing_tds', 'pass_tds', 'ptd']),
                    'interceptions': self._get_stat(player_data, ['interceptions', 'int', 'ints']),
                    'rush_yds': self._get_stat(player_data, ['rushing_yards', 'rush_yds', 'ry']),
                    'rush_tds': self._get_stat(player_data, ['rushing_tds', 'rush_tds', 'rtd']),
                    'receptions': self._get_stat(player_data, ['receptions', 'rec', 'catches']),
                    'rec_yds': self._get_stat(player_data, ['receiving_yards', 'rec_yds', 'rey']),
                    'rec_tds': self._get_stat(player_data, ['receiving_tds', 'rec_tds', 'retd'])
                }
                
                # Only add if we have a valid name
                if projection['name']:
                    projections.append(projection)
                    if i < 3:  # Debug first few projections
                        print(f"Created projection: {projection}")
                    
            except Exception as e:
                print(f"Error parsing player data: {player_data} - {e}")
                continue
        
        print(f"Successfully parsed {len(projections)} projections")
        return projections
    
    def _get_player_name(self, player_data: Dict) -> str:
        """Extract player name from various possible fields"""
        name_fields = ['player_name', 'name', 'full_name', 'display_name', 'player']
        
        for field in name_fields:
            if field in player_data and player_data[field]:
                return str(player_data[field]).strip()
        
        # Try to construct from first/last name
        first_name = player_data.get('first_name', '') or player_data.get('fname', '')
        last_name = player_data.get('last_name', '') or player_data.get('lname', '')
        
        if first_name and last_name:
            return f"{first_name} {last_name}".strip()
        
        return ''
    
    def _get_position(self, player_data: Dict, fallback_position: str = None) -> str:
        """Extract position from player data"""
        pos_fields = ['position', 'pos']
        
        for field in pos_fields:
            if field in player_data and player_data[field]:
                return str(player_data[field]).upper()
        
        return fallback_position or ''
    
    def _get_team(self, player_data: Dict) -> str:
        """Extract team from player data"""
        team_fields = ['team', 'tm', 'team_abbr']
        
        for field in team_fields:
            if field in player_data and player_data[field]:
                return str(player_data[field]).upper()
        
        return ''
    
    def _get_bye_week(self, player_data: Dict) -> int:
        """Extract bye week from player data"""
        bye_fields = ['bye_week', 'bye']
        
        for field in bye_fields:
            if field in player_data and player_data[field]:
                try:
                    return int(player_data[field])
                except (ValueError, TypeError):
                    continue
        
        return None
    
    def _get_fantasy_points(self, player_data: Dict) -> float:
        """Extract fantasy points from player data"""
        points_fields = ['fantasy_points', 'fpts', 'fantasy_pts', 'points', 'projected_points']
        
        for field in points_fields:
            if field in player_data and player_data[field] is not None:
                try:
                    return float(player_data[field])
                except (ValueError, TypeError):
                    continue
        
        return 0.0
    
    def _get_stat(self, player_data: Dict, possible_fields: List[str]) -> float:
        """Extract a stat value from player data using multiple possible field names"""
        for field in possible_fields:
            if field in player_data and player_data[field] is not None:
                try:
                    return float(player_data[field])
                except (ValueError, TypeError):
                    continue
        
        return 0.0
    
    def get_all_positions_projections(self, week: int = 0, 
                                    scoring: str = 'PPR', year: int = 2024) -> List[Dict]:
        """Get projections for all positions"""
        print(f"Fetching all projections for {year} season, {scoring} scoring")
        
        # Try to get all projections in one call first
        all_projections = self.get_projections(None, week, scoring, year)
        
        if all_projections:
            print(f"Successfully fetched {len(all_projections)} projections in one call")
            return all_projections
        
        # If that fails, try position by position
        print("Trying position-by-position approach...")
        all_projections = []
        positions = ['QB', 'RB', 'WR', 'TE', 'K', 'DST']
        
        for position in positions:
            print(f"Fetching {position} projections...")
            position_projections = self.get_projections(position, week, scoring, year)
            all_projections.extend(position_projections)
            
            print(f"{position}: {len(position_projections)} projections")
            
            # Rate limiting between position requests
            time.sleep(FANTASYPROS_DELAY)
        
        print(f"Total projections fetched: {len(all_projections)}")
        return all_projections

    def test_connection(self) -> bool:
        """Test API connection"""
        try:
            url = f"{self.base_url}/2024/projections"
            response = self.session.get(url, params={'limit': 1})
            
            print(f"Test connection - Status: {response.status_code}")
            if response.status_code == 200:
                print("✅ API connection successful!")
                return True
            else:
                print(f"❌ API connection failed: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ API connection error: {e}")
            return False