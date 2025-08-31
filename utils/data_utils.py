import re
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any

def standardize_player_name(name: str) -> str:
    """Standardize player name format"""
    if not name:
        return ""
    
    # Remove extra whitespace
    name = re.sub(r'\s+', ' ', name.strip())
    
    # Handle common name variations
    name_replacements = {
        'Jr.': 'Jr',
        'Sr.': 'Sr',
        'III': 'III',
        'II': 'II'
    }
    
    for old, new in name_replacements.items():
        name = name.replace(old, new)
    
    # Convert to title case
    name = name.title()
    
    # Handle specific cases
    name = re.sub(r"Mccaffrey", "McCaffrey", name)
    name = re.sub(r"Mcdonald", "McDonald", name)
    
    return name

def validate_projection_data(projection: Dict) -> bool:
    """Validate projection data quality"""
    required_fields = ['player_id', 'source_id']
    
    # Check required fields
    for field in required_fields:
        if field not in projection or projection[field] is None:
            return False
    
    # Validate numerical fields
    numerical_fields = ['points', 'pass_yds', 'pass_tds', 'rush_yds', 'rush_tds', 
                       'receptions', 'rec_yds', 'rec_tds']
    
    for field in numerical_fields:
        if field in projection:
            value = projection[field]
            if value is not None and (not isinstance(value, (int, float)) or value < 0):
                return False
    
    # Basic sanity checks
    if projection.get('points', 0) > 500:  # Unrealistic season projection
        return False
    
    if projection.get('pass_yds', 0) > 6000:  # Unrealistic passing yards
        return False
    
    return True

def clean_team_name(team: str) -> str:
    """Standardize team abbreviations"""
    if not team:
        return ""
    
    team_mapping = {
        'JAX': 'JAC',
        'NE': 'NWE',
        'NO': 'NOR',
        'SF': 'SFO',
        'TB': 'TAM',
        'KC': 'KAN',
        'LV': 'LVR',
        'LA': 'LAR'  # Default LA to Rams, handle Chargers separately
    }
    
    team = team.upper().strip()
    return team_mapping.get(team, team)

def normalize_position(position: str) -> str:
    """Normalize position names"""
    if not position:
        return ""
    
    position = position.upper().strip()
    
    # Handle position variations
    position_mapping = {
        'FLEX': 'FLEX',
        'D/ST': 'DEF',
        'DST': 'DEF',
        'PK': 'K'
    }
    
    return position_mapping.get(position, position)

def calculate_scoring_points(stats: Dict, scoring_system: str) -> float:
    """Calculate fantasy points based on scoring system"""
    from config.settings import SCORING_SYSTEMS
    
    if scoring_system not in SCORING_SYSTEMS:
        scoring_system = 'ppr'
    
    scoring = SCORING_SYSTEMS[scoring_system]
    points = 0
    
    # Passing stats
    points += stats.get('pass_yds', 0) * scoring.get('pass_yd', 0.04)  # 1 pt per 25 yds
    points += stats.get('pass_tds', 0) * scoring.get('pass_td', 4)
    points -= stats.get('interceptions', 0) * scoring.get('interception', 2)
    
    # Rushing stats
    points += stats.get('rush_yds', 0) * scoring.get('rush_yd', 0.1)  # 1 pt per 10 yds
    points += stats.get('rush_tds', 0) * scoring.get('rush_td', 6)
    
    # Receiving stats
    points += stats.get('receptions', 0) * scoring.get('rec', 1.0)  # PPR value
    points += stats.get('rec_yds', 0) * scoring.get('rec_yd', 0.1)  # 1 pt per 10 yds
    points += stats.get('rec_tds', 0) * scoring.get('rec_td', 6)
    
    # Fumbles
    points -= stats.get('fumbles_lost', 0) * scoring.get('fumble_lost', 2)
    
    return round(points, 2)

def detect_outliers(data: List[float], method: str = 'iqr') -> List[bool]:
    """Detect outliers in projection data"""
    if not data or len(data) < 3:
        return [False] * len(data)
    
    data_array = np.array(data)
    
    if method == 'iqr':
        Q1 = np.percentile(data_array, 25)
        Q3 = np.percentile(data_array, 75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        return [(x < lower_bound or x > upper_bound) for x in data]
    
    elif method == 'zscore':
        mean_val = np.mean(data_array)
        std_val = np.std(data_array)
        
        if std_val == 0:
            return [False] * len(data)
        
        z_scores = np.abs((data_array - mean_val) / std_val)
        return [z > 3 for z in z_scores]  # 3 standard deviations
    
    return [False] * len(data)

def handle_missing_data(df: pd.DataFrame, method: str = 'median') -> pd.DataFrame:
    """Handle missing data in projections"""
    df_cleaned = df.copy()
    
    numerical_columns = df_cleaned.select_dtypes(include=[np.number]).columns
    
    for col in numerical_columns:
        if method == 'median':
            df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
        elif method == 'mean':
            df_cleaned[col].fillna(df_cleaned[col].mean(), inplace=True)
        elif method == 'zero':
            df_cleaned[col].fillna(0, inplace=True)
    
    # Handle categorical columns
    categorical_columns = df_cleaned.select_dtypes(include=['object']).columns
    
    for col in categorical_columns:
        df_cleaned[col].fillna('Unknown', inplace=True)
    
    return df_cleaned

def format_player_display_name(name: str, position: str, team: str = None) -> str:
    """Format player name for display"""
    display_name = name
    
    if position:
        display_name += f" ({position})"
    
    if team:
        display_name += f" - {team}"
    
    return display_name

def parse_projection_file_headers(headers: List[str]) -> Dict[str, str]:
    """Parse and map projection file headers to standard format"""
    header_mapping = {}
    
    # Common header variations
    mapping_rules = {
        'name': ['name', 'player', 'player_name', 'full_name'],
        'position': ['pos', 'position'],
        'team': ['tm', 'team'],
        'points': ['fpts', 'fantasy_points', 'projected_points', 'points'],
        'pass_yds': ['pass_yds', 'passing_yards', 'py'],
        'pass_tds': ['pass_tds', 'passing_tds', 'ptd'],
        'rush_yds': ['rush_yds', 'rushing_yards', 'ry'],
        'rush_tds': ['rush_tds', 'rushing_tds', 'rtd'],
        'receptions': ['rec', 'receptions', 'catches'],
        'rec_yds': ['rec_yds', 'receiving_yards', 'rec_yards'],
        'rec_tds': ['rec_tds', 'receiving_tds', 'rec_td']
    }
    
    for standard_name, variations in mapping_rules.items():
        for header in headers:
            header_lower = header.lower().strip()
            if header_lower in variations:
                header_mapping[header] = standard_name
                break
    
    return header_mapping