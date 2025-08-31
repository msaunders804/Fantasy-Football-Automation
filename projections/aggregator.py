import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from core.database import DatabaseManager
from ml.weight_optimizer import ProjectionWeightOptimizer

class ProjectionAggregator:
    """Aggregates multiple projection sources using ML-optimized weights"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.weight_optimizer = ProjectionWeightOptimizer(db_manager)
        self._load_or_train_weights()
    
    def _load_or_train_weights(self):
        """Load existing weights or train new ones"""
        if not self.weight_optimizer.load_model():
            print("Training projection weight optimizer...")
            try:
                self.weight_optimizer.train_source_weights()
                self.weight_optimizer.save_model()
            except Exception as e:
                print(f"Warning: Could not train weight optimizer: {e}")
                print("Using equal weights for all sources")
    
    def aggregate_all_projections(self) -> bool:
        """Aggregate projections for all players"""
        try:
            # Get all players with projections
            players_query = """
            SELECT DISTINCT p.player_id, p.name, p.position, p.team
            FROM players p
            JOIN projections pr ON p.player_id = pr.player_id
            """
            
            players = self.db.execute_query(players_query)
            
            if not players:
                print("No players with projections found")
                return False
            
            # Clear existing aggregated projections
            self.db.execute_update("DELETE FROM aggregated_projections WHERE week = 0")
            
            aggregated_projections = []
            
            for player in players:
                aggregated = self._aggregate_player_projections(player['player_id'])
                
                if aggregated:
                    aggregated_projections.append(aggregated)
            
            # Calculate VBD scores
            self._calculate_vbd_scores(aggregated_projections)
            
            # Calculate tiers
            self._calculate_tiers(aggregated_projections)
            
            # Insert aggregated projections
            self._insert_aggregated_projections(aggregated_projections)
            
            print(f"Aggregated projections for {len(aggregated_projections)} players")
            return True
            
        except Exception as e:
            print(f"Error aggregating projections: {e}")
            return False
    
    def _aggregate_player_projections(self, player_id: str) -> Optional[Dict]:
        """Aggregate projections for a single player"""
        # Get all projections for player
        projections_query = """
        SELECT p.*, ps.name as source_name, ps.weight
        FROM projections p
        JOIN projection_sources ps ON p.source_id = ps.source_id
        WHERE p.player_id = ? AND p.week = 0
        """
        
        projections = self.db.execute_query(projections_query, (player_id,))
        
        if not projections:
            return None
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(projections)
        
        # Get source weights
        source_weights = self.weight_optimizer.get_current_weights()
        
        # Apply weights
        df['effective_weight'] = df['source_name'].map(lambda x: source_weights.get(x, 1.0))
        
        # Weighted average of projections
        numerical_cols = ['points', 'pass_yds', 'pass_tds', 'interceptions', 
                         'rush_yds', 'rush_tds', 'receptions', 'rec_yds', 'rec_tds']
        
        aggregated = {}
        total_weight = df['effective_weight'].sum()
        
        for col in numerical_cols:
            if col in df.columns:
                weighted_sum = (df[col] * df['effective_weight']).sum()
                aggregated[col] = weighted_sum / total_weight if total_weight > 0 else df[col].mean()
            else:
                aggregated[col] = 0
        
        # Calculate confidence score based on consensus
        confidence = self._calculate_confidence_score(df, 'points')
        
        return {
            'player_id': player_id,
            'week': 0,
            'final_points': aggregated['points'],
            'confidence_score': confidence,
            'pass_yds': aggregated['pass_yds'],
            'pass_tds': aggregated['pass_tds'],
            'interceptions': aggregated['interceptions'],
            'rush_yds': aggregated['rush_yds'],
            'rush_tds': aggregated['rush_tds'],
            'receptions': aggregated['receptions'],
            'rec_yds': aggregated['rec_yds'],
            'rec_tds': aggregated['rec_tds']
        }
    
    def _calculate_confidence_score(self, projections_df: pd.DataFrame, column: str) -> float:
        """Calculate confidence score based on projection consensus"""
        if len(projections_df) < 2:
            return 0.5  # Medium confidence for single source
        
        # Lower standard deviation = higher confidence
        std_dev = projections_df[column].std()
        mean_val = projections_df[column].mean()
        
        # Coefficient of variation (normalized standard deviation)
        cv = std_dev / mean_val if mean_val > 0 else 1
        
        # Convert to confidence (0-1 scale)
        confidence = 1 / (1 + cv)  # Higher CV = lower confidence
        
        return min(1.0, max(0.0, confidence))
    
    def _calculate_vbd_scores(self, projections: List[Dict]):
        """Calculate Value-Based Drafting scores"""
        # Group by position
        by_position = {}
        for proj in projections:
            # Get player position
            player_query = "SELECT position FROM players WHERE player_id = ?"
            player_result = self.db.execute_query(player_query, (proj['player_id'],))
            
            if player_result:
                position = player_result[0]['position']
                if position not in by_position:
                    by_position[position] = []
                by_position[position].append(proj)
        
        # Calculate replacement level for each position
        replacement_levels = {
            'QB': 12,  # 12th QB
            'RB': 24,  # 24th RB
            'WR': 30,  # 30th WR
            'TE': 12,  # 12th TE
            'K': 12,   # 12th K
            'DEF': 12  # 12th DEF
        }
        
        for position, position_projections in by_position.items():
            # Sort by projected points
            position_projections.sort(key=lambda x: x['final_points'], reverse=True)
            
            # Get replacement level
            replacement_index = replacement_levels.get(position, 12) - 1
            
            if len(position_projections) > replacement_index:
                replacement_points = position_projections[replacement_index]['final_points']
            else:
                # If not enough players, use minimum points
                replacement_points = min(p['final_points'] for p in position_projections)
            
            # Calculate VBD for each player
            for proj in position_projections:
                proj['vbd_score'] = proj['final_points'] - replacement_points
    
    def _calculate_tiers(self, projections: List[Dict]):
        """Calculate player tiers within positions"""
        by_position = {}
        
        # Group by position
        for proj in projections:
            player_query = "SELECT position FROM players WHERE player_id = ?"
            player_result = self.db.execute_query(player_query, (proj['player_id'],))
            
            if player_result:
                position = player_result[0]['position']
                if position not in by_position:
                    by_position[position] = []
                by_position[position].append(proj)
        
        # Calculate tiers for each position
        for position, position_projections in by_position.items():
            position_projections.sort(key=lambda x: x['vbd_score'], reverse=True)
            
            # Simple tier calculation based on VBD gaps
            if not position_projections:
                continue
                
            # Calculate tier breaks based on VBD score gaps
            vbd_scores = [p['vbd_score'] for p in position_projections]
            
            # Find natural breakpoints using gaps
            gaps = []
            for i in range(1, len(vbd_scores)):
                gap = vbd_scores[i-1] - vbd_scores[i]
                gaps.append((gap, i))
            
            # Sort gaps by size
            gaps.sort(reverse=True)
            
            # Use top gaps as tier breaks (max 6 tiers)
            tier_breaks = sorted([gap[1] for gap in gaps[:5]])  # Top 5 gaps = 6 tiers
            
            # Assign tiers
            current_tier = 1
            break_index = 0
            
            for i, proj in enumerate(position_projections):
                if break_index < len(tier_breaks) and i >= tier_breaks[break_index]:
                    current_tier += 1
                    break_index += 1
                
                proj['tier'] = min(current_tier, 6)  # Cap at tier 6
    
    def _insert_aggregated_projections(self, projections: List[Dict]):
        """Insert aggregated projections into database"""
        query = """
        INSERT INTO aggregated_projections 
        (player_id, week, final_points, confidence_score, vbd_score, tier)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        
        params_list = [
            (p['player_id'], p['week'], p['final_points'], 
             p['confidence_score'], p['vbd_score'], p['tier'])
            for p in projections
        ]
        
        self.db.execute_many(query, params_list)