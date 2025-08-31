import os

# Database
DATABASE_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'draft_assistant.db')

# Scoring Systems
SCORING_SYSTEMS = {
    'ppr': {
        'rec': 1.0, 
        'rush_td': 6, 
        'rec_td': 6, 
        'pass_td': 4,
        'pass_yd': 0.04,  # 1 pt per 25 yards
        'rush_yd': 0.1,   # 1 pt per 10 yards
        'rec_yd': 0.1,    # 1 pt per 10 yards
        'interception': -2,
        'fumble_lost': -2
    },
    'half_ppr': {
        'rec': 0.5, 
        'rush_td': 6, 
        'rec_td': 6, 
        'pass_td': 4,
        'pass_yd': 0.04,
        'rush_yd': 0.1,
        'rec_yd': 0.1,
        'interception': -2,
        'fumble_lost': -2
    },
    'standard': {
        'rec': 0.0, 
        'rush_td': 6, 
        'rec_td': 6, 
        'pass_td': 4,
        'pass_yd': 0.04,
        'rush_yd': 0.1,
        'rec_yd': 0.1,
        'interception': -2,
        'fumble_lost': -2
    }
}

# Default Roster Settings
DEFAULT_ROSTER = {
    'QB': 1, 
    'RB': 2, 
    'WR': 2, 
    'TE': 1, 
    'FLEX': 1, 
    'K': 1, 
    'DEF': 1, 
    'BENCH': 6
}

# ML Model Settings
MODEL_SETTINGS = {
    'projection_weights': {'n_estimators': 100, 'random_state': 42},
    'value_predictor': {'learning_rate': 0.1, 'max_depth': 6},
    'draft_strategy': {'test_size': 0.2, 'cv_folds': 5}
}

# API Settings
SLEEPER_BASE_URL = 'https://api.sleeper.app/v1'
FANTASYPROS_API_BASE_URL = 'https://api.fantasypros.com/public/v2/json/'
FANTASYPROS_DELAY = 1  # Seconds between requests
ESPN_API_DELAY = 0.5  # Seconds between ESPN requests

# Projection Source Priorities (higher = more reliable)
SOURCE_PRIORITIES = {
    'FantasyPros_API_PPR': 10,
    'FantasyPros_API_HALF': 10,
    'FantasyPros_API_STD': 10,
    'ESPN_CSV': 8,
    'FantasyPros_CSV': 9,
    'Additional': 5
}

# ESPN specific settings
ESPN_COLUMN_VARIATIONS = {
    'player_variations': ['Player', 'PLAYER', 'Name', 'Full Name'],
    'position_variations': ['POS', 'Position', 'Pos'],
    'team_variations': ['TEAM', 'Team', 'Tm'],
    'points_variations': ['FPTS', 'Fantasy Points', 'Projected Points', 'PROJ', 'Points']
}

# Draft Settings
DRAFT_SETTINGS = {
    'default_rounds': 16,
    'default_teams': 12,
    'cpu_pick_delay': 1.5,  # Seconds delay for CPU picks
    'recommendation_limit': 10
}

# Position Settings
POSITION_SETTINGS = {
    'valid_positions': ['QB', 'RB', 'WR', 'TE', 'K', 'DEF'],
    'skill_positions': ['QB', 'RB', 'WR', 'TE'],
    'flex_eligible': ['RB', 'WR', 'TE']
}

# VBD (Value Based Drafting) Settings
VBD_SETTINGS = {
    'replacement_levels': {
        'QB': 12,  # 12th QB is replacement level
        'RB': 24,  # 24th RB is replacement level  
        'WR': 30,  # 30th WR is replacement level
        'TE': 12,  # 12th TE is replacement level
        'K': 12,   # 12th K is replacement level
        'DEF': 12  # 12th DEF is replacement level
    }
}

# Tier Settings
TIER_SETTINGS = {
    'max_tiers': 6,
    'tier_gap_threshold': 5.0  # VBD gap to create new tier
}

# File Paths
PATHS = {
    'models': 'models/',
    'data': 'data/',
    'exports': 'exports/'
}

# Ensure directories exist
for path in PATHS.values():
    os.makedirs(path, exist_ok=True)