# advanced_ml_draft_system.py
"""
Advanced ML-Powered Fantasy Draft System
Complete rebuild with proper data handling, feature engineering, and realistic projections
NOW WITH INTEGRATED DRAFT SIMULATION
"""
# Add these imports if missing
import warnings
warnings.filterwarnings('ignore')

# Ensure the class can be imported properly
__all__ = ['AdvancedMLDraftSystem']

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
import logging
from typing import Dict, List, Optional, Tuple
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb

from database_setup import FantasyDatabase

class AdvancedMLDraftSystem:
    """Professional-grade ML draft system with proper data science practices"""
    
    def __init__(self, db_path="fantasy_football.db", league_settings=None):
        self.db = FantasyDatabase(db_path)
        self.models = {}
        self.scalers = {}
        self.feature_names = {}
        self.setup_logging()
        
        # League settings for VBD calculations
        self.league_settings = league_settings or {
            'teams': 12,
            'roster_spots': {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'FLEX': 1},
            'scoring': 'ppr',
            'playoff_weeks': [14, 15, 16, 17]
        }
        
        # Elite player database for validation
        self.elite_players = {
            'QB': ['Josh Allen', 'Lamar Jackson', 'Patrick Mahomes', 'Joe Burrow', 'Dak Prescott'],
            'RB': ['Christian McCaffrey', 'Austin Ekeler', 'Saquon Barkley', 'Derrick Henry', 'Alvin Kamara'],
            'WR': ['Cooper Kupp', 'Davante Adams', 'Stefon Diggs', 'Tyreek Hill', 'DeAndre Hopkins'],
            'TE': ['Travis Kelce', 'Mark Andrews', 'T.J. Hockenson', 'Kyle Pitts', 'George Kittle']
        }
        
        # Expected point ranges by position for validation
        self.position_ranges = {
            'QB': {'elite': (350, 450), 'good': (280, 350), 'average': (220, 280), 'replacement': (180, 220)},
            'RB': {'elite': (280, 380), 'good': (220, 280), 'average': (160, 220), 'replacement': (100, 160)},
            'WR': {'elite': (260, 360), 'good': (200, 260), 'average': (140, 200), 'replacement': (80, 140)},
            'TE': {'elite': (180, 250), 'good': (140, 180), 'average': (100, 140), 'replacement': (60, 100)}
        }
        
        # Draft simulation state variables
        self.draft_order = []
        self.my_pick_position = 1
        self.current_round = 1
        self.current_pick = 1
        self.drafted_players = set()
        self.my_roster = {
            'QB': [], 'RB': [], 'WR': [], 'TE': [], 'K': [], 'DEF': [], 'BENCH': []
        }
        self.all_team_rosters = {}
        self.available_players = []
        self.draft_strategy = 'BALANCED'
    
    def setup_logging(self):
        """Configure logging"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_comprehensive_training_data(self) -> pd.DataFrame:
        """Load comprehensive training data with proper feature engineering"""
        if not self.db.connect():
            return pd.DataFrame()
        
        try:
            # Get comprehensive player data across multiple seasons
            query = """
            SELECT 
                p.player_id,
                p.full_name,
                p.position,
                p.team,
                p.age,
                p.years_exp,
                ps.season_year,
                ps.week,
                ps.fantasy_points_ppr,
                ps.fantasy_points_half_ppr,
                ps.fantasy_points_std,
                ps.pass_att, ps.pass_cmp, ps.pass_yd, ps.pass_td, ps.pass_int,
                ps.rush_att, ps.rush_yd, ps.rush_td, ps.rush_fum,
                ps.rec_tgt, ps.rec, ps.rec_yd, ps.rec_td, ps.rec_fum,
                ps.snap_count, ps.snap_percentage,
                -- Team context
                ps.team as game_team,
                ps.opponent,
                ps.is_home
            FROM players p
            JOIN player_stats ps ON p.player_id = ps.player_id
            WHERE p.position IN ('QB', 'RB', 'WR', 'TE')
                AND ps.season_year >= 2021
                AND ps.week <= 17  -- Regular season only
                AND ps.fantasy_points_ppr >= 0
            ORDER BY p.player_id, ps.season_year, ps.week
            """
            
            df = pd.read_sql_query(query, self.db.connection)
            self.logger.info(f"Loaded {len(df)} training records from database")
            
            if df.empty:
                self.logger.error("No training data found!")
                return df
            
            # Basic data cleaning
            df = df.fillna(0)
            
            # Remove obvious outliers (extreme performances that skew training)
            for pos in ['QB', 'RB', 'WR', 'TE']:
                pos_data = df[df['position'] == pos]
                if len(pos_data) > 100:
                    # Remove top/bottom 1% to eliminate outliers
                    q99 = pos_data['fantasy_points_ppr'].quantile(0.99)
                    q01 = pos_data['fantasy_points_ppr'].quantile(0.01)
                    df = df[~((df['position'] == pos) & 
                            ((df['fantasy_points_ppr'] > q99) | (df['fantasy_points_ppr'] < q01)))]
            
            self.logger.info(f"After cleaning: {len(df)} records")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading training data: {e}")
            return pd.DataFrame()
        finally:
            self.db.disconnect()
    
    def engineer_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create sophisticated features for ML models"""
        if df.empty:
            return df
        
        self.logger.info("Engineering advanced features...")
        
        # Sort by player and time
        df = df.sort_values(['player_id', 'season_year', 'week'])
        
        # Get target column based on scoring
        target_col = f"fantasy_points_{self.league_settings['scoring']}"
        if target_col not in df.columns:
            target_col = 'fantasy_points_ppr'
        
        # 1. Rolling statistics (multiple windows)
        for window in [3, 6, 12]:
            # Points
            df[f'points_roll_{window}'] = df.groupby('player_id')[target_col].rolling(
                window=window, min_periods=1).mean().reset_index(0, drop=True)
            
            df[f'points_std_{window}'] = df.groupby('player_id')[target_col].rolling(
                window=window, min_periods=2).std().fillna(0).reset_index(0, drop=True)
            
            # Volume stats
            df[f'targets_roll_{window}'] = df.groupby('player_id')['rec_tgt'].rolling(
                window=window, min_periods=1).mean().reset_index(0, drop=True)
            
            df[f'carries_roll_{window}'] = df.groupby('player_id')['rush_att'].rolling(
                window=window, min_periods=1).mean().reset_index(0, drop=True)
        
        # 2. Seasonal context features
        df['games_played_season'] = df.groupby(['player_id', 'season_year']).cumcount() + 1
        df['season_points_so_far'] = df.groupby(['player_id', 'season_year'])[target_col].cumsum()
        df['season_avg_so_far'] = df['season_points_so_far'] / df['games_played_season']
        
        # 3. Efficiency metrics
        df['yards_per_target'] = np.where(df['rec_tgt'] > 0, df['rec_yd'] / df['rec_tgt'], 0)
        df['yards_per_carry'] = np.where(df['rush_att'] > 0, df['rush_yd'] / df['rush_att'], 0)
        df['yards_per_attempt'] = np.where(df['pass_att'] > 0, df['pass_yd'] / df['pass_att'], 0)
        df['catch_rate'] = np.where(df['rec_tgt'] > 0, df['rec'] / df['rec_tgt'], 0)
        df['td_rate'] = (df['pass_td'] + df['rush_td'] + df['rec_td']) / np.maximum(
            df['pass_att'] + df['rush_att'] + df['rec_tgt'], 1)
        
        # 4. Usage and opportunity metrics
        # Calculate team-level totals for share metrics
        team_totals = df.groupby(['game_team', 'season_year', 'week']).agg({
            'rec_tgt': 'sum',
            'rush_att': 'sum',
            'pass_att': 'sum'
        }).reset_index()
        team_totals.columns = ['game_team', 'season_year', 'week', 'team_targets', 'team_carries', 'team_passes']
        
        df = df.merge(team_totals, on=['game_team', 'season_year', 'week'], how='left')
        
        df['target_share'] = np.where(df['team_targets'] > 0, df['rec_tgt'] / df['team_targets'], 0)
        df['carry_share'] = np.where(df['team_carries'] > 0, df['rush_att'] / df['team_carries'], 0)
        
        # 5. Player age and experience features
        df['age_at_season'] = df['age'] - (2024 - df['season_year'])  # Adjust age for season
        df['exp_at_season'] = df['years_exp'] - (2024 - df['season_year'])
        
        # Age curves by position
        df['age_factor'] = df.apply(lambda row: self._calculate_age_factor(
            row['age_at_season'], row['position']), axis=1)
        
        # 6. Recent trends
        df['points_trend_3'] = df.groupby('player_id')[target_col].rolling(
            window=3, min_periods=2).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0
        ).reset_index(0, drop=True)
        
        # 7. Position-specific features
        # QB-specific
        df['qb_efficiency'] = np.where(
            (df['position'] == 'QB') & (df['pass_att'] > 0),
            (df['pass_td'] * 20 - df['pass_int'] * 10 + df['pass_yd'] * 0.25) / df['pass_att'],
            0
        )
        
        # RB-specific
        df['rb_efficiency'] = np.where(
            (df['position'] == 'RB') & (df['rush_att'] + df['rec_tgt'] > 0),
            (df['rush_yd'] + df['rec_yd'] + df['rush_td'] * 10 + df['rec_td'] * 10) / (df['rush_att'] + df['rec_tgt']),
            0
        )
        
        # WR/TE-specific
        df['receiver_efficiency'] = np.where(
            (df['position'].isin(['WR', 'TE'])) & (df['rec_tgt'] > 0),
            (df['rec_yd'] + df['rec_td'] * 15) / df['rec_tgt'],
            0
        )
        
        # 8. Game context features
        df['is_home'] = df['is_home'].fillna(0).astype(int)
        
        # 9. Consistency metrics
        df['consistency_score'] = 1 / (1 + df['points_std_6'])  # Higher = more consistent
        
        # 10. Create position dummy variables
        position_dummies = pd.get_dummies(df['position'], prefix='pos')
        df = pd.concat([df, position_dummies], axis=1)
        
        # Clean data
        df = df.replace([np.inf, -np.inf], 0)
        df = df.fillna(0)
        
        self.logger.info(f"Feature engineering complete. Final shape: {df.shape}")
        return df
    
    def _calculate_age_factor(self, age: float, position: str) -> float:
        """Calculate position-specific age adjustment factor"""
        if pd.isna(age) or age < 20 or age > 40:
            return 1.0
        
        if position == 'QB':
            # QBs peak later, decline slower
            if age <= 25:
                return 0.95  # Still developing
            elif age <= 32:
                return 1.0   # Prime
            elif age <= 36:
                return 0.98  # Slight decline
            else:
                return 0.9   # Notable decline
        
        elif position == 'RB':
            # RBs peak early, decline fast
            if age <= 23:
                return 0.9   # Young, unproven
            elif age <= 27:
                return 1.0   # Prime
            elif age <= 30:
                return 0.85  # Decline
            else:
                return 0.7   # Sharp decline
        
        else:  # WR, TE
            # Receivers have longer primes
            if age <= 24:
                return 0.95  # Developing
            elif age <= 30:
                return 1.0   # Prime
            elif age <= 33:
                return 0.95  # Slight decline
            else:
                return 0.85  # Notable decline
    
    def train_advanced_models(self, df: pd.DataFrame) -> Dict:
        """Train sophisticated ML models with proper validation"""
        self.logger.info("Training advanced ML models...")
        
        # Define feature columns
        feature_columns = [
            'age_at_season', 'exp_at_season', 'games_played_season',
            'points_roll_3', 'points_roll_6', 'points_roll_12',
            'points_std_3', 'points_std_6', 'points_std_12',
            'targets_roll_3', 'targets_roll_6', 'carries_roll_3', 'carries_roll_6',
            'season_avg_so_far', 'yards_per_target', 'yards_per_carry', 'yards_per_attempt',
            'catch_rate', 'td_rate', 'target_share', 'carry_share',
            'age_factor', 'points_trend_3', 'qb_efficiency', 'rb_efficiency', 
            'receiver_efficiency', 'is_home', 'consistency_score'
        ]
        
        # Add position dummies
        position_cols = [col for col in df.columns if col.startswith('pos_')]
        feature_columns.extend(position_cols)
        
        # Filter to available features
        available_features = [col for col in feature_columns if col in df.columns]
        
        # Get target column
        target_col = f"fantasy_points_{self.league_settings['scoring']}"
        if target_col not in df.columns:
            target_col = 'fantasy_points_ppr'
        
        model_results = {}
        
        for position in ['QB', 'RB', 'WR', 'TE']:
            self.logger.info(f"Training {position} model...")
            
            # Filter data for position
            pos_data = df[df['position'] == position].copy()
            
            if len(pos_data) < 500:  # Need substantial data
                self.logger.warning(f"Insufficient data for {position} model ({len(pos_data)} records)")
                continue
            
            # Prepare features and target
            X = pos_data[available_features].fillna(0)
            y = pos_data[target_col].fillna(0)
            
            # Remove extreme outliers more conservatively
            y_q95 = y.quantile(0.95)
            y_q05 = y.quantile(0.05)
            mask = (y >= y_q05) & (y <= y_q95)
            X, y = X[mask], y[mask]
            
            # Chronological split to avoid data leakage
            split_point = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
            y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
            
            # Scale features (important for some algorithms)
            scaler = RobustScaler()  # More robust to outliers than StandardScaler
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Test multiple advanced algorithms
            models = {
                'xgb': xgb.XGBRegressor(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=-1
                ),
                'rf': RandomForestRegressor(
                    n_estimators=200,
                    max_depth=12,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1
                ),
                'gb': GradientBoostingRegressor(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                )
            }
            
            best_model = None
            best_score = float('inf')
            best_model_name = None
            
            for name, model in models.items():
                try:
                    # Train model (tree-based don't need scaling)
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)
                    
                    mae = mean_absolute_error(y_test, predictions)
                    r2 = r2_score(y_test, predictions)
                    
                    self.logger.info(f"{position} {name}: MAE={mae:.2f}, R²={r2:.3f}")
                    
                    if mae < best_score:
                        best_score = mae
                        best_model = model
                        best_model_name = name
                
                except Exception as e:
                    self.logger.error(f"Error training {position} {name} model: {e}")
                    continue
            
            if best_model is not None:
                # Store model and metadata
                self.models[position] = best_model
                self.scalers[position] = scaler
                self.feature_names[position] = available_features
                
                # Calculate feature importance
                if hasattr(best_model, 'feature_importances_'):
                    feature_importance = pd.DataFrame({
                        'feature': available_features,
                        'importance': best_model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    self.logger.info(f"{position} top features: {feature_importance.head(5)['feature'].tolist()}")
                
                model_results[position] = {
                    'model_type': best_model_name,
                    'mae': best_score,
                    'r2': r2_score(y_test, predictions) if 'predictions' in locals() else 0,
                    'feature_count': len(available_features),
                    'training_samples': len(X_train)
                }
        
        self.logger.info(f"Model training complete for {len(self.models)} positions")
        return model_results
    
    def get_2025_player_projections(self) -> List[Dict]:
        """Generate 2025 projections for all relevant players using 2024 as primary data source"""
        if not self.db.connect():
            return []
        
        try:
            # Get current active players with 2024 as primary data source
            query = """
            SELECT DISTINCT
                p.player_id,
                p.full_name,
                p.position,
                p.team,
                p.age,
                p.years_exp,
                p.status,
                -- 2024 performance (PRIMARY for 2025 projections)
                AVG(CASE WHEN ps.season_year = 2024 THEN ps.fantasy_points_ppr END) as avg_2024,
                COUNT(CASE WHEN ps.season_year = 2024 THEN ps.week END) as games_2024,
                SUM(CASE WHEN ps.season_year = 2024 THEN ps.fantasy_points_ppr END) as total_2024,
                AVG(CASE WHEN ps.season_year = 2024 THEN ps.rec_tgt END) as tgt_2024,
                AVG(CASE WHEN ps.season_year = 2024 THEN ps.rush_att END) as rush_2024,
                AVG(CASE WHEN ps.season_year = 2024 THEN ps.pass_att END) as pass_2024,
                -- 2023 performance (SECONDARY for context)
                AVG(CASE WHEN ps.season_year = 2023 THEN ps.fantasy_points_ppr END) as avg_2023,
                COUNT(CASE WHEN ps.season_year = 2023 THEN ps.week END) as games_2023,
                SUM(CASE WHEN ps.season_year = 2023 THEN ps.fantasy_points_ppr END) as total_2023,
                -- Career context
                AVG(ps.fantasy_points_ppr) as career_avg,
                MAX(ps.season_year) as last_active_season
            FROM players p
            LEFT JOIN player_stats ps ON p.player_id = ps.player_id 
                AND ps.season_year >= 2022
            WHERE p.position IN ('QB', 'RB', 'WR', 'TE')
                AND (p.status = 'Active' OR p.status IS NULL)
                AND p.team IS NOT NULL
                AND p.team != ''
                AND p.team != 'FA'
                -- Filter known retired players
                AND p.full_name NOT IN ('Tom Brady', 'Ben Roethlisberger', 'Josh McCown', 
                                      'Adrian Peterson', 'Frank Gore', 'Antonio Brown')
                AND (p.age IS NULL OR p.age < 37)
            GROUP BY p.player_id, p.full_name, p.position, p.team, p.age, p.years_exp, p.status
            HAVING (
                games_2024 > 0 OR  -- Has 2024 data (priority)
                (games_2023 > 8 AND last_active_season >= 2023) OR  -- Good 2023 data and recently active
                (p.age < 25 AND last_active_season >= 2023)  -- Young player recently active
            )
            ORDER BY COALESCE(total_2024, total_2023, 0) DESC
            """
            
            df = pd.read_sql_query(query, self.db.connection)
            self.logger.info(f"Found {len(df)} potential 2025 players")
            
            projections = []
            
            for _, row in df.iterrows():
                # Create 2025 projection prioritizing 2024 data
                projection = self._create_2025_projection_updated(row)
                
                if projection and projection['projected_points'] >= 80:  # Meaningful projections only
                    projections.append(projection)
            
            self.logger.info(f"Generated {len(projections)} meaningful 2025 projections")
            return projections
        except:
            self.logger.warning("Failure")
    
    def calculate_proper_vbd(self, projections: List[Dict]) -> List[Dict]:
        """Calculate proper Value-Based Drafting with correct replacement levels"""
        self.logger.info("Calculating VBD with proper replacement levels...")
        
        # Calculate realistic replacement levels
        teams = self.league_settings['teams']
        roster_spots = self.league_settings['roster_spots']
        
        replacement_levels = {}
        
        for position in ['QB', 'RB', 'WR', 'TE']:
            pos_projections = [p for p in projections if p['position'] == position]
            pos_projections.sort(key=lambda x: x['projected_points'], reverse=True)
            
            if not pos_projections:
                # No players at this position - use fallback
                replacement_levels[position] = self.position_ranges[position]['replacement'][0]
                continue
            
            # Calculate replacement level based on realistic draft patterns
            if position == 'QB':
                # 12th-15th QB drafted (most teams draft 1 QB early)
                replacement_idx = min(teams + 3, len(pos_projections) - 1)
            elif position == 'RB':
                # ~30th RB (teams draft 2-3 RBs, plus handcuffs)
                replacement_idx = min(int(teams * 2.5), len(pos_projections) - 1)
            elif position == 'WR':
                # ~36th WR (teams draft 3-4 WRs)
                replacement_idx = min(teams * 3, len(pos_projections) - 1)
            elif position == 'TE':
                # ~18th TE (most teams draft 1 TE)
                replacement_idx = min(int(teams * 1.5), len(pos_projections) - 1)
            
            replacement_idx = max(0, min(replacement_idx, len(pos_projections) - 1))
            
            if len(pos_projections) > replacement_idx:
                replacement_levels[position] = pos_projections[replacement_idx]['projected_points']
            else:
                # Fallback to position ranges if not enough players
                replacement_levels[position] = self.position_ranges[position]['replacement'][0]
            
            self.logger.info(f"{position}: {len(pos_projections)} players, replacement at #{replacement_idx+1} = {replacement_levels[position]:.1f} pts")
        
        # Calculate VBD scores
        vbd_projections = []
        for projection in projections:
            position = projection['position']
            projected_points = projection['projected_points']
            replacement_level = replacement_levels.get(position, 100)
            
            vbd_score = max(0, projected_points - replacement_level)
            
            # Only include players with meaningful projections
            if projected_points >= 50:  # Minimum threshold
                projection_copy = projection.copy()
                projection_copy['vbd_score'] = round(vbd_score, 1)
                projection_copy['replacement_level'] = round(replacement_level, 1)
                vbd_projections.append(projection_copy)
        
        # Sort by VBD score
        vbd_projections.sort(key=lambda x: x['vbd_score'], reverse=True)
        
        self.logger.info(f"VBD calculation complete. {len(vbd_projections)} players with meaningful projections.")
        return vbd_projections
    
    def get_draft_rankings(self) -> List[Dict]:
        """Generate complete 2025 draft rankings"""
        self.logger.info("Generating 2025 draft rankings...")
        
        # Get player projections
        projections = self.get_2025_player_projections()
        
        if not projections:
            self.logger.error("No projections generated!")
            return []
        
        # Calculate VBD
        rankings = self.calculate_proper_vbd(projections)
        
        # Add draft grades and tiers
        for i, player in enumerate(rankings):
            player['overall_rank'] = i + 1
            player['tier'] = self._assign_tier(player['vbd_score'])
            player['draft_grade'] = self._assign_draft_grade(player['vbd_score'], player['position'])
        
        return rankings
    
    def _assign_tier(self, vbd_score: float) -> str:
        """Assign tier based on VBD score"""
        if vbd_score >= 100:
            return "ELITE"
        elif vbd_score >= 60:
            return "TIER 1"
        elif vbd_score >= 35:
            return "TIER 2"
        elif vbd_score >= 20:
            return "TIER 3"
        elif vbd_score >= 10:
            return "TIER 4"
        else:
            return "TIER 5+"
    
    def _assign_draft_grade(self, vbd_score: float, position: str) -> str:
        """Assign draft grade based on VBD and position"""
        if vbd_score >= 100:
            return "A+"
        elif vbd_score >= 70:
            return "A"
        elif vbd_score >= 50:
            return "A-"
        elif vbd_score >= 35:
            return "B+"
        elif vbd_score >= 25:
            return "B"
        elif vbd_score >= 15:
            return "B-"
        elif vbd_score >= 10:
            return "C+"
        elif vbd_score >= 5:
            return "C"
        else:
            return "C-"
    
    # =========================
    # DRAFT SIMULATION METHODS
    # =========================
    
    def setup_draft_simulation(self):
        """Configure draft settings and initialize simulation"""
        print("\n⚙️ DRAFT SIMULATION SETUP")
        print("-" * 25)
        
        try:
            # Get league settings
            teams = int(input(f"Number of teams (default {self.league_settings['teams']}): ") or str(self.league_settings['teams']))
            my_position = int(input(f"Your draft position (1-{teams}): "))
            rounds = int(input("Number of rounds (default 16): ") or "16")
            
            # Update settings
            self.league_settings.update({
                'teams': teams,
                'rounds': rounds,
                'roster_spots': {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'FLEX': 1, 'K': 1, 'DEF': 1},
                'bench_spots': rounds - 7  # Total spots minus starters
            })
            self.my_pick_position = my_position
            
            # Initialize team rosters for all teams
            self.all_team_rosters = {}
            for team_num in range(1, teams + 1):
                self.all_team_rosters[team_num] = {
                    'QB': [], 'RB': [], 'WR': [], 'TE': [], 'K': [], 'DEF': [], 'BENCH': []
                }
            
            # Generate snake draft order
            self._generate_draft_order()
            
            # Reset draft state
            self.current_round = 1
            self.current_pick = 1
            self.drafted_players = set()
            self.my_roster = {
                'QB': [], 'RB': [], 'WR': [], 'TE': [], 'K': [], 'DEF': [], 'BENCH': []
            }
            
            # Load available players from rankings
            if not self.available_players:
                rankings = self.get_draft_rankings()
                self.available_players = rankings.copy()
            
            print(f"\n✅ Draft simulation configured:")
            print(f"  Teams: {teams}")
            print(f"  Rounds: {rounds}")
            print(f"  Your position: {my_position}")
            print(f"  Available players: {len(self.available_players)}")
            print(f"  Your picks in first 3 rounds: {self._get_my_upcoming_picks()[:3]}")
            
            return True
            
        except ValueError:
            print("❌ Invalid input. Please enter valid numbers.")
            return False
    
    def _generate_draft_order(self):
        """Generate snake draft order"""
        teams = self.league_settings['teams']
        rounds = self.league_settings.get('rounds', 16)
        
        self.draft_order = []
        
        for round_num in range(1, rounds + 1):
            if round_num % 2 == 1:  # Odd rounds (1, 3, 5...)
                round_picks = list(range(1, teams + 1))
            else:  # Even rounds (2, 4, 6...)
                round_picks = list(range(teams, 0, -1))
            
            for pick_in_round, team_position in enumerate(round_picks, 1):
                overall_pick = (round_num - 1) * teams + pick_in_round
                self.draft_order.append({
                    'overall_pick': overall_pick,
                    'round': round_num,
                    'pick_in_round': pick_in_round,
                    'team_position': team_position,
                    'is_my_pick': team_position == self.my_pick_position
                })
    
    def _get_my_upcoming_picks(self) -> List[int]:
        """Get list of my upcoming pick numbers"""
        return [pick['overall_pick'] for pick in self.draft_order if pick['is_my_pick']]
    
    def _draft_player(self, player: Dict, team_position: int = None, is_my_pick: bool = False):
        """Process a player being drafted"""
        # Remove from available players
        self.available_players = [p for p in self.available_players if p['player_id'] != player['player_id']]
        self.drafted_players.add(player['player_id'])
        
        # Determine which team drafted the player
        if team_position is None:
            current_pick_info = self.draft_order[self.current_pick - 1]
            team_position = current_pick_info['team_position']
        
        # Add to appropriate roster
        position = player['position']
        if is_my_pick or team_position == self.my_pick_position:
            # Add to my roster
            if position in self.my_roster:
                self.my_roster[position].append(player)
            else:
                self.my_roster['BENCH'].append(player)
        
        # Add to all teams tracking
        if team_position in self.all_team_rosters:
            if position in self.all_team_rosters[team_position]:
                self.all_team_rosters[team_position][position].append(player)
            else:
                self.all_team_rosters[team_position]['BENCH'].append(player)
        
        # Advance pick counter
        self.current_pick += 1
        
        # Update round if needed
        picks_per_round = self.league_settings['teams']
        self.current_round = ((self.current_pick - 1) // picks_per_round) + 1
    
    def simulate_cpu_pick(self, team_position: int):
        """Simulate a CPU team's draft pick"""
        if not self.available_players:
            return None
        
        # Simple CPU logic - take best available with some positional need consideration
        team_roster = self.all_team_rosters.get(team_position, {})
        
        # Calculate team needs
        qb_count = len(team_roster.get('QB', []))
        rb_count = len(team_roster.get('RB', []))
        wr_count = len(team_roster.get('WR', []))
        te_count = len(team_roster.get('TE', []))
        
        # Find best player considering needs
        best_player = None
        best_score = -float('inf')
        
        for player in self.available_players[:30]:  # Consider top 30 available
            score = player['vbd_score']
            
            # Apply need multipliers
            if player['position'] == 'QB' and qb_count == 0:
                score *= 1.3
            elif player['position'] == 'RB' and rb_count < 2:
                score *= 1.2
            elif player['position'] == 'WR' and wr_count < 2:
                score *= 1.2
            elif player['position'] == 'TE' and te_count == 0:
                score *= 1.1
            
            if score > best_score:
                best_score = score
                best_player = player
        
        if best_player:
            self._draft_player(best_player, team_position)
            return best_player
        
        return Nones
    
    def _create_2025_projection_updated(self, player_row) -> Optional[Dict]:
        """Create 2025 projection for a single player using 2024-prioritized data"""
        try:
            position = player_row['position']
            if position not in self.models:
                return None
            
            # Prioritize 2024 data, fallback to 2023
            primary_avg = player_row['avg_2024'] if player_row['games_2024'] > 0 else player_row['avg_2023']
            primary_games = player_row['games_2024'] if player_row['games_2024'] > 0 else player_row['games_2023']
            
            # Use season totals when available, not per-game averages
            if player_row['total_2024'] and player_row['games_2024'] > 8:
                # Good 2024 season data - use as baseline
                base_projection = player_row['total_2024']
                data_source = '2024'
            elif player_row['total_2023'] and player_row['games_2023'] > 12:
                # Full 2023 season - project forward with age adjustment
                base_projection = player_row['total_2023'] * 0.98  # Slight decline assumption
                data_source = '2023'
            elif primary_avg and primary_games > 4:
                # Partial season - extrapolate to full season
                base_projection = primary_avg * 17  # Project to 17-game season
                data_source = 'extrapolated'
            else:
                # Use position baseline for unknowns
                base_projection = self._get_position_baseline_2025(position)
                data_source = 'baseline'
            
            # Apply 2025 adjustments
            final_projection = self._apply_2025_adjustments(base_projection, player_row, data_source)
            
            # Validate against realistic ranges
            final_projection = self._validate_projection_realistic(final_projection, position, player_row['full_name'])
            
            return {
                'player_id': player_row['player_id'],
                'name': player_row['full_name'],
                'position': position,
                'team': player_row['team'],
                'age': player_row['age'],
                'projected_points': round(final_projection, 1),
                'data_source': data_source,
                'games_2024': player_row['games_2024'] or 0,
                'total_2024': player_row['total_2024'] or 0,
                'games_2023': player_row['games_2023'] or 0,
                'total_2023': player_row['total_2023'] or 0,
                'confidence': self._calculate_projection_confidence(player_row, data_source)
            }
            
        except Exception as e:
            self.logger.error(f"Error projecting {player_row['full_name']}: {e}")
            return None
    
    def _calculate_projection_confidence(self, player_row, data_source: str) -> str:
        """Calculate confidence level in projection"""
        if data_source == '2024' and player_row['games_2024'] > 12:
            return 'HIGH'
        elif data_source == '2024' and player_row['games_2024'] > 6:
            return 'MEDIUM'
        elif data_source == '2023' and player_row['games_2023'] > 14:
            return 'MEDIUM'
        elif self._is_elite_player(player_row['full_name'], player_row['position']):
            return 'MEDIUM'  # Elite players more predictable
        else:
            return 'LOW'
    
    def _get_position_baseline_2025(self, position: str) -> float:
        """Get realistic 2025 baseline by position"""
        baselines = {
            'QB': 250,  # Low-end starter
            'RB': 150,  # Flex-worthy 
            'WR': 120,  # WR3 level
            'TE': 80    # Streaming option
        }
        return baselines.get(position, 100)
    
    def _apply_2025_adjustments(self, base_projection: float, player_row, data_source: str) -> float:
        """Apply realistic 2025 adjustments"""
        adjusted = base_projection
        position = player_row['position']
        age = player_row['age'] or 25
        
        # Age-based adjustments
        if position == 'RB':
            if age <= 24:
                adjusted *= 1.08  # Young RB upside
            elif age <= 27:
                adjusted *= 1.02  # Prime years
            elif age >= 30:
                adjusted *= 0.88  # Decline phase
        elif position == 'QB':
            if age <= 26:
                adjusted *= 1.05  # Still developing
            elif age >= 35:
                adjusted *= 0.95  # Decline risk
        elif position in ['WR', 'TE']:
            if age <= 25:
                adjusted *= 1.03  # Young receiver development
            elif age >= 32:
                adjusted *= 0.94  # Decline risk
        
        # Data recency adjustment
        if data_source == '2024':
            adjusted *= 1.0  # Most reliable
        elif data_source == '2023':
            adjusted *= 0.95  # One year old
        elif data_source == 'extrapolated':
            adjusted *= 0.90  # Less reliable
        
        # Elite player recognition
        if self._is_elite_player(player_row['full_name'], position):
            adjusted *= 1.12  # Elite players maintain performance better
        
        return adjusted
    
    def _validate_projection_realistic(self, projection: float, position: str, player_name: str) -> float:
        """Apply realistic validation with elite player recognition"""
        # Set realistic ranges by position
        realistic_ranges = {
            'QB': {'min': 180, 'max': 420, 'elite_min': 320},
            'RB': {'min': 80, 'max': 380, 'elite_min': 250},
            'WR': {'min': 60, 'max': 350, 'elite_min': 220},
            'TE': {'min': 40, 'max': 220, 'elite_min': 140}
        }
        
        ranges = realistic_ranges[position]
        
        # Apply elite player floors
        if self._is_elite_player(player_name, position):
            projection = max(projection, ranges['elite_min'])
        
        # Apply overall bounds
        projection = max(ranges['min'], min(ranges['max'], projection))
        
        return projection
    
    def _is_elite_player(self, player_name: str, position: str) -> bool:
        """Check if player is in elite tier"""
        elite_list = self.elite_players.get(position, [])
        return any(elite_name.lower() in player_name.lower() for elite_name in elite_list)
    
    def analyze_team_composition(self) -> Dict:
        """Analyze current team composition and needs"""
        analysis = {
            'roster_strength': 0,
            'positional_needs': {},
            'depth_analysis': {},
            'risk_assessment': {},
            'strategy_recommendation': ''
        }
        
        # Calculate roster strength
        total_projected = 0
        for position, players in self.my_roster.items():
            if position != 'BENCH':
                for player in players:
                    total_projected += player['projected_points']
        
        analysis['roster_strength'] = round(total_projected, 1)
        
        # Analyze positional needs
        roster_spots = self.league_settings['roster_spots']
        for position, required in roster_spots.items():
            if position == 'FLEX':
                continue  # Handle flex separately
            
            current_count = len(self.my_roster.get(position, []))
            analysis['positional_needs'][position] = max(0, required - current_count)
        
        # Analyze depth by position
        for position in ['QB', 'RB', 'WR', 'TE']:
            players = self.my_roster.get(position, [])
            if players:
                avg_points = sum(p['projected_points'] for p in players) / len(players)
                best_player = max(players, key=lambda p: p['projected_points'])
                analysis['depth_analysis'][position] = {
                    'count': len(players),
                    'avg_projection': round(avg_points, 1),
                    'best_player': best_player['name'],
                    'depth_quality': self._assess_depth_quality(players)
                }
            else:
                analysis['depth_analysis'][position] = {
                    'count': 0,
                    'avg_projection': 0,
                    'best_player': None,
                    'depth_quality': 'NONE'
                }
        
        # Risk assessment
        analysis['risk_assessment'] = self._assess_roster_risk()
        
        # Strategy recommendation
        analysis['strategy_recommendation'] = self._recommend_strategy()
        
        return analysis
    
    def _assess_depth_quality(self, players: List[Dict]) -> str:
        """Assess quality of positional depth"""
        if not players:
            return 'NONE'
        
        avg_vbd = sum(p['vbd_score'] for p in players) / len(players)
        
        if avg_vbd > 60:
            return 'ELITE'
        elif avg_vbd > 35:
            return 'GOOD'
        elif avg_vbd > 15:
            return 'ADEQUATE'
        else:
            return 'WEAK'
    
    def _assess_roster_risk(self) -> Dict:
        """Assess various roster risks"""
        risks = {
            'injury_risk': 'LOW',
            'bye_week_issues': [],
            'age_risk': 'LOW',
            'correlation_risk': []
        }
        
        # Check for players from same team (correlation risk)
        team_counts = {}
        all_players = []
        for players in self.my_roster.values():
            all_players.extend(players)
        
        for player in all_players:
            team = player.get('team', 'UNK')
            team_counts[team] = team_counts.get(team, 0) + 1
        
        for team, count in team_counts.items():
            if count > 2:
                risks['correlation_risk'].append(f"{count} players from {team}")
        
        # Check age risk (simplified)
        old_players = [p for p in all_players if p.get('age', 25) > 30]
        if len(old_players) > 3:
            risks['age_risk'] = 'HIGH'
        elif len(old_players) > 1:
            risks['age_risk'] = 'MEDIUM'
        
        return risks
    
    def _recommend_strategy(self) -> str:
        """Recommend draft strategy based on current roster"""
        qb_count = len(self.my_roster.get('QB', []))
        rb_count = len(self.my_roster.get('RB', []))
        wr_count = len(self.my_roster.get('WR', []))
        te_count = len(self.my_roster.get('TE', []))
        
        needs = []
        
        # Check critical needs
        if rb_count < 2:
            needs.append(f"Need {2-rb_count} more RB")
        if wr_count < 2:
            needs.append(f"Need {2-wr_count} more WR")
        if qb_count == 0:
            needs.append("Need QB")
        if te_count == 0:
            needs.append("Need TE")
        
        if needs:
            return f"Priority: {', '.join(needs)}"
        else:
            return "Build depth and target upside"
    
    def get_draft_recommendations(self, top_n: int = 8) -> List[Dict]:
        """Get draft recommendations considering team composition"""
        if not self.available_players:
            return []
        
        # Get current team analysis
        team_analysis = self.analyze_team_composition()
        needs = team_analysis['positional_needs']
        
        # Score available players
        recommendations = []
        
        for player in self.available_players[:50]:  # Consider top 50 available
            base_score = player['vbd_score']
            
            # Positional need multiplier
            position = player['position']
            need_multiplier = 1.0
            
            if needs.get(position, 0) > 0:
                need_multiplier = 1.4  # 40% boost for needed positions
            elif position in ['RB', 'WR', 'TE'] and needs.get('FLEX', 0) > 0:
                need_multiplier = 1.1  # 10% boost for flex eligible
            
            # Strategy adjustments
            if self.current_round <= 6:
                # Early rounds - prioritize elite talent
                if player['tier'] in ['ELITE', 'TIER 1']:
                    need_multiplier *= 1.2
            else:
                # Later rounds - prioritize need and upside
                if needs.get(position, 0) > 0:
                    need_multiplier *= 1.3
            
            final_score = base_score * need_multiplier
            
            # Determine recommendation type
            if needs.get(position, 0) > 0:
                rec_type = "FILL NEED"
            elif player['tier'] == 'ELITE':
                rec_type = "ELITE TALENT"
            elif final_score > base_score * 1.1:
                rec_type = "GOOD VALUE"
            else:
                rec_type = "BPA"
            
            recommendations.append({
                **player,
                'draft_score': round(final_score, 1),
                'recommendation_type': rec_type,
                'need_multiplier': round(need_multiplier, 2)
            })
        
        # Sort by draft score
        recommendations.sort(key=lambda x: x['draft_score'], reverse=True)
        return recommendations[:top_n]
    
    def show_draft_board(self):
        """Display comprehensive draft status"""
        print(f"\n🏈 DRAFT BOARD - Round {self.current_round}, Pick #{self.current_pick}")
        print("=" * 80)
        
        # Show my roster
        print(f"\n📋 YOUR ROSTER:")
        total_projected = 0
        
        for position in ['QB', 'RB', 'WR', 'TE', 'K', 'DEF']:
            players = self.my_roster.get(position, [])
            required = self.league_settings['roster_spots'].get(position, 0)
            
            if players:
                print(f"\n{position} ({len(players)}/{required}):")
                for i, player in enumerate(players, 1):
                    print(f"  {i}. {player['name']} ({player['team']}) - {player['projected_points']} pts")
                    total_projected += player['projected_points']
            else:
                print(f"\n{position} (0/{required}): EMPTY")
        
        # Show bench
        bench_players = self.my_roster.get('BENCH', [])
        if bench_players:
            print(f"\nBENCH ({len(bench_players)}):")
            for player in bench_players:
                print(f"  • {player['name']} ({player['position']}, {player['team']}) - {player['projected_points']} pts")
                total_projected += player['projected_points']
        
        print(f"\nTotal Projected Points: {total_projected:.1f}")
        
        # Show team analysis
        analysis = self.analyze_team_composition()
        print(f"\n📊 TEAM ANALYSIS:")
        print(f"Strategy Recommendation: {analysis['strategy_recommendation']}")
        
        if analysis['risk_assessment']['correlation_risk']:
            print(f"⚠️ Correlation Risk: {', '.join(analysis['risk_assessment']['correlation_risk'])}")
        
        # Show upcoming picks
        my_picks = self._get_my_upcoming_picks()
        current_pick_idx = next((i for i, pick in enumerate(my_picks) if pick >= self.current_pick), None)
        
        if current_pick_idx is not None:
            upcoming = my_picks[current_pick_idx:current_pick_idx+3]
            print(f"\n🎯 YOUR NEXT PICKS: {upcoming}")
    
    def draft_player_interactive(self):
        """Interactive player drafting"""
        print(f"\n🎯 YOUR PICK #{self.current_pick}")
        print("-" * 30)
        
        # Show recommendations
        recommendations = self.get_draft_recommendations()
        
        print(f"\n⭐ TOP RECOMMENDATIONS:")
        print("-" * 70)
        print(f"{'#':<2} {'Player':<20} {'Pos':<3} {'Proj':<6} {'VBD':<6} {'Type':<12}")
        print("-" * 70)
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i:<2} {rec['name'][:19]:<20} {rec['position']:<3} "
                  f"{rec['projected_points']:<6.1f} {rec['vbd_score']:<6.1f} {rec['recommendation_type']:<12}")
        
        # Get user choice
        while True:
            choice = input(f"\nDraft player (enter name or number 1-{len(recommendations)}): ").strip()
            
            if choice.isdigit():
                choice_num = int(choice)
                if 1 <= choice_num <= len(recommendations):
                    selected_player = recommendations[choice_num - 1]
                    break
            else:
                # Search by name
                matches = [p for p in self.available_players if choice.lower() in p['name'].lower()]
                if matches:
                    selected_player = matches[0]
                    break
            
            print("❌ Invalid choice. Try again.")
        
        # Confirm and draft
        print(f"\nDrafting: {selected_player['name']} ({selected_player['position']}, {selected_player['team']})")
        print(f"Projection: {selected_player['projected_points']} pts, VBD: {selected_player['vbd_score']}")
        
        confirm = input("Confirm this pick? (Y/n): ").strip().lower()
        if confirm != 'n':
            self._draft_player(selected_player, is_my_pick=True)
            print(f"✅ Drafted {selected_player['name']}!")
            return True
        else:
            print("❌ Pick cancelled")
            return False


# ==================
# MAIN FUNCTION
# ==================

def main():
    """Main entry point for the ML Draft System"""
    print("=" * 80)
    print("🏈 ADVANCED ML FANTASY DRAFT SYSTEM 2025")
    print("=" * 80)
    
    # Initialize the system
    print("\nInitializing system...")
    draft_system = AdvancedMLDraftSystem()
    
    # Main menu loop
    while True:
        print("\n" + "=" * 50)
        print("MAIN MENU")
        print("-" * 50)
        print("1. Train ML Models")
        print("2. Generate 2025 Player Projections")
        print("3. View Draft Rankings")
        print("4. Start Mock Draft Simulation")
        print("5. Configure League Settings")
        print("6. Exit")
        print("-" * 50)
        
        choice = input("Select option (1-6): ").strip()
        
        if choice == '1':
            # Train ML models
            print("\n📊 TRAINING ML MODELS")
            print("-" * 30)
            print("Loading training data...")
            training_data = draft_system.load_comprehensive_training_data()
            
            if not training_data.empty:
                print(f"Loaded {len(training_data)} records")
                print("Engineering features...")
                training_data = draft_system.engineer_advanced_features(training_data)
                
                print("Training models...")
                results = draft_system.train_advanced_models(training_data)
                
                print("\n✅ Training complete!")
                for position, metrics in results.items():
                    print(f"{position}: {metrics['model_type']} - MAE: {metrics['mae']:.2f}, R²: {metrics['r2']:.3f}")
            else:
                print("❌ Failed to load training data")
        
        elif choice == '2':
            # Generate projections
            print("\n🔮 GENERATING 2025 PROJECTIONS")
            print("-" * 30)
            
            if not draft_system.models:
                print("⚠️ Models not trained. Training now...")
                training_data = draft_system.load_comprehensive_training_data()
                if not training_data.empty:
                    training_data = draft_system.engineer_advanced_features(training_data)
                    draft_system.train_advanced_models(training_data)
            
            projections = draft_system.get_2025_player_projections()
            
            if projections:
                print(f"\n✅ Generated projections for {len(projections)} players")
                
                # Show top players by position
                for position in ['QB', 'RB', 'WR', 'TE']:
                    pos_players = [p for p in projections if p['position'] == position]
                    pos_players.sort(key=lambda x: x['projected_points'], reverse=True)
                    
                    print(f"\nTop 5 {position}s:")
                    for i, player in enumerate(pos_players[:5], 1):
                        print(f"  {i}. {player['name'][:20]:<20} - {player['projected_points']:.1f} pts")
            else:
                print("❌ Failed to generate projections")
        
        elif choice == '3':
            # View rankings
            print("\n📋 2025 DRAFT RANKINGS")
            print("-" * 30)
            
            rankings = draft_system.get_draft_rankings()
            
            if rankings:
                # Display top 50 with formatting
                print(f"\n{'Rank':<5} {'Player':<25} {'Pos':<4} {'Team':<4} {'Proj':<7} {'VBD':<7} {'Tier':<10}")
                print("-" * 70)
                
                for player in rankings[:50]:
                    print(f"{player['overall_rank']:<5} {player['name'][:24]:<25} {player['position']:<4} "
                          f"{player['team'][:3]:<4} {player['projected_points']:<7.1f} "
                          f"{player['vbd_score']:<7.1f} {player['tier']:<10}")
                
                print(f"\n... showing top 50 of {len(rankings)} total players")
            else:
                print("❌ No rankings available. Generate projections first.")
        
        elif choice == '4':
            # Mock draft simulation
            print("\n🎮 MOCK DRAFT SIMULATION")
            print("-" * 30)
            
            if not draft_system.available_players:
                print("Loading player rankings...")
                rankings = draft_system.get_draft_rankings()
                if not rankings:
                    print("❌ No rankings available. Generate projections first.")
                    continue
                draft_system.available_players = rankings.copy()
            
            # Setup draft
            if draft_system.setup_draft_simulation():
                print("\n🚀 Starting draft simulation...")
                
                # Draft loop
                total_picks = draft_system.league_settings['teams'] * draft_system.league_settings['rounds']
                
                while draft_system.current_pick <= total_picks:
                    current_pick_info = draft_system.draft_order[draft_system.current_pick - 1]
                    
                    if current_pick_info['is_my_pick']:
                        # User's pick
                        draft_system.show_draft_board()
                        
                        # Options during draft
                        print("\nOptions:")
                        print("1. Draft a player")
                        print("2. View available players by position")
                        print("3. Analyze team needs")
                        print("4. Auto-draft remainder")
                        print("5. Exit draft")
                        
                        action = input("Select action (1-5): ").strip()
                        
                        if action == '1':
                            draft_system.draft_player_interactive()
                        elif action == '2':
                            pos = input("Enter position (QB/RB/WR/TE): ").upper()
                            if pos in ['QB', 'RB', 'WR', 'TE']:
                                pos_players = [p for p in draft_system.available_players if p['position'] == pos][:10]
                                print(f"\nTop available {pos}s:")
                                for i, p in enumerate(pos_players, 1):
                                    print(f"{i}. {p['name'][:20]:<20} - {p['projected_points']:.1f} pts")
                            continue  # Don't advance pick
                        elif action == '3':
                            analysis = draft_system.analyze_team_composition()
                            print(f"\nTeam Analysis:")
                            print(f"Roster Strength: {analysis['roster_strength']:.1f} pts")
                            print(f"Strategy: {analysis['strategy_recommendation']}")
                            for pos, need in analysis['positional_needs'].items():
                                if need > 0:
                                    print(f"  Need {need} {pos}")
                            continue  # Don't advance pick
                        elif action == '4':
                            print("Auto-drafting remainder...")
                            while draft_system.current_pick <= total_picks:
                                current = draft_system.draft_order[draft_system.current_pick - 1]
                                if current['is_my_pick']:
                                    # Auto-pick best available
                                    if draft_system.available_players:
                                        draft_system._draft_player(draft_system.available_players[0], is_my_pick=True)
                                else:
                                    draft_system.simulate_cpu_pick(current['team_position'])
                            break
                        elif action == '5':
                            print("Exiting draft...")
                            break
                    else:
                        # CPU pick
                        cpu_player = draft_system.simulate_cpu_pick(current_pick_info['team_position'])
                        if cpu_player:
                            print(f"Team {current_pick_info['team_position']} drafts: "
                                  f"{cpu_player['name']} ({cpu_player['position']})")
                
                # Draft complete
                if draft_system.current_pick > total_picks:
                    print("\n" + "=" * 50)
                    print("🏆 DRAFT COMPLETE!")
                    print("=" * 50)
                    draft_system.show_draft_board()
                    
                    # Final analysis
                    print("\n📊 FINAL ROSTER ANALYSIS:")
                    analysis = draft_system.analyze_team_composition()
                    print(f"Total Projected Points: {analysis['roster_strength']:.1f}")
                    
                    # Position breakdown
                    for pos in ['QB', 'RB', 'WR', 'TE']:
                        if pos in analysis['depth_analysis']:
                            depth = analysis['depth_analysis'][pos]
                            print(f"{pos}: {depth['count']} players, "
                                  f"Avg: {depth['avg_projection']:.1f} pts, "
                                  f"Quality: {depth['depth_quality']}")
        
        elif choice == '5':
            # Configure league settings
            print("\n⚙️ LEAGUE SETTINGS")
            print("-" * 30)
            print("Current settings:")
            print(f"  Teams: {draft_system.league_settings['teams']}")
            print(f"  Scoring: {draft_system.league_settings['scoring']}")
            print(f"  Roster: {draft_system.league_settings['roster_spots']}")
            
            modify = input("\nModify settings? (y/n): ").strip().lower()
            if modify == 'y':
                teams = input(f"Number of teams [{draft_system.league_settings['teams']}]: ").strip()
                if teams:
                    draft_system.league_settings['teams'] = int(teams)
                
                scoring = input(f"Scoring type (ppr/half_ppr/std) [{draft_system.league_settings['scoring']}]: ").strip()
                if scoring in ['ppr', 'half_ppr', 'std']:
                    draft_system.league_settings['scoring'] = scoring
                
                print("✅ Settings updated!")
        
        elif choice == '6':
            print("\nThank you for using the ML Draft System!")
            print("Good luck in your draft! 🏈")
            break
        
        else:
            print("❌ Invalid selection. Please choose 1-6.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Program interrupted by user")
        print("Exiting...")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()
        print("\nPlease report this error for debugging.")