-- Players table
CREATE TABLE IF NOT EXISTS players (
    player_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    position TEXT NOT NULL,
    team TEXT,
    bye_week INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Projection sources
CREATE TABLE IF NOT EXISTS projection_sources (
    source_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    weight REAL DEFAULT 1.0,
    accuracy_score REAL DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Raw projections
CREATE TABLE IF NOT EXISTS projections (
    projection_id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id TEXT NOT NULL,
    source_id INTEGER NOT NULL,
    week INTEGER DEFAULT 0, -- 0 for season-long
    points REAL,
    pass_yds REAL DEFAULT 0,
    pass_tds INTEGER DEFAULT 0,
    interceptions INTEGER DEFAULT 0,
    rush_yds REAL DEFAULT 0,
    rush_tds INTEGER DEFAULT 0,
    receptions INTEGER DEFAULT 0,
    rec_yds REAL DEFAULT 0,
    rec_tds INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (player_id) REFERENCES players (player_id),
    FOREIGN KEY (source_id) REFERENCES projection_sources (source_id)
);

-- Aggregated projections (ML-weighted)
CREATE TABLE IF NOT EXISTS aggregated_projections (
    agg_id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id TEXT NOT NULL,
    week INTEGER DEFAULT 0,
    final_points REAL,
    confidence_score REAL,
    vbd_score REAL,
    tier INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (player_id) REFERENCES players (player_id)
);

-- Leagues
CREATE TABLE IF NOT EXISTS leagues (
    league_id TEXT PRIMARY KEY,
    name TEXT,
    platform TEXT, -- 'sleeper', 'manual'
    num_teams INTEGER DEFAULT 12,
    scoring_system TEXT DEFAULT 'ppr',
    roster_settings TEXT, -- JSON string
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Drafts
CREATE TABLE IF NOT EXISTS drafts (
    draft_id INTEGER PRIMARY KEY AUTOINCREMENT,
    league_id TEXT NOT NULL,
    status TEXT DEFAULT 'active', -- 'active', 'completed', 'paused'
    current_pick INTEGER DEFAULT 1,
    total_rounds INTEGER DEFAULT 16,
    draft_order TEXT, -- JSON array of team names/IDs
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (league_id) REFERENCES leagues (league_id)
);

-- Draft picks
CREATE TABLE IF NOT EXISTS draft_picks (
    pick_id INTEGER PRIMARY KEY AUTOINCREMENT,
    draft_id INTEGER NOT NULL,
    pick_number INTEGER NOT NULL,
    round_number INTEGER NOT NULL,
    team_position INTEGER NOT NULL,
    player_id TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (draft_id) REFERENCES drafts (draft_id),
    FOREIGN KEY (player_id) REFERENCES players (player_id)
);

-- Team rosters (current draft state)
CREATE TABLE IF NOT EXISTS team_rosters (
    roster_id INTEGER PRIMARY KEY AUTOINCREMENT,
    draft_id INTEGER NOT NULL,
    team_position INTEGER NOT NULL,
    player_id TEXT NOT NULL,
    position_slot TEXT, -- 'QB', 'RB1', 'FLEX', etc.
    FOREIGN KEY (draft_id) REFERENCES drafts (draft_id),
    FOREIGN KEY (player_id) REFERENCES players (player_id)
);

-- Model performance tracking
CREATE TABLE IF NOT EXISTS model_performance (
    model_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_type TEXT NOT NULL,
    accuracy_score REAL,
    mae REAL,
    r2_score REAL,
    training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_projections_player_source ON projections(player_id, source_id);
CREATE INDEX IF NOT EXISTS idx_draft_picks_draft_round ON draft_picks(draft_id, round_number);
CREATE INDEX IF NOT EXISTS idx_aggregated_projections_vbd ON aggregated_projections(vbd_score DESC);