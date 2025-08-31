import pandas as pd
import numpy as np
from typing import Dict, List
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from ml.base_model import BaseMLModel

class ProjectionWeightOptimizer(BaseMLModel):
    """ML model to optimize weights for different projection sources"""
    
    def __init__(self, db_manager):
        super().__init__("projection_weights", db_manager)
        self.source_weights = {}
    
    def create_model(self):
        """Create Random Forest model for weight optimization"""
        return RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for weight optimization"""
        features = pd.DataFrame()
        
        # Source accuracy features
        if 'source_accuracy' in data.columns:
            features['source_accuracy'] = data['source_accuracy']
        
        # Consensus variance (higher variance = less reliable)
        if 'projection_std' in data.columns:
            features['projection_variance'] = data['projection_std']
            features['inverse_variance'] = 1 / (data['projection_std'] + 0.1)
        
        # Position-specific accuracy
        if 'position' in data.columns:
            position_dummies = pd.get_dummies(data['position'], prefix='pos')
            features = pd.concat([features, position_dummies], axis=1)
        
        # Historical performance features
        if 'historical_mae' in data.columns:
            features['inverse_mae'] = 1 / (data['historical_mae'] + 0.1)
        
        # Recency weight (more recent = higher weight)
        if 'days_since_update' in data.columns:
            features['recency_weight'] = np.exp(-data['days_since_update'] / 7)
        
        return features.fillna(0)
    
    def train_source_weights(self) -> Dict[str, float]:
        """Train model to determine optimal source weights"""
        # Get historical projection vs actual data
        training_data = self._get_historical_projection_data()
        
        if training_data.empty:
            # Default equal weights if no historical data
            sources = self._get_projection_sources()
            return {source: 1.0 for source in sources}
        
        # Train model
        metrics = self.train(training_data, 'actual_points')
        
        # Calculate source weights based on model
        source_weights = self._calculate_source_weights(training_data)
        
        # Update database with new weights
        self._update_source_weights(source_weights)
        
        return source_weights
    
    def _get_historical_projection_data(self) -> pd.DataFrame:
        """Get historical projection vs actual performance data"""
        query = """
        SELECT 
            ps.name as source_name,
            ps.source_id,
            p.points as projected_points,
            -- Would need actual fantasy points table for training
            0 as actual_points,  -- Placeholder
            pl.position,
            julianday('now') - julianday(p.created_at) as days_since_update
        FROM projections p
        JOIN projection_sources ps ON p.source_id = ps.source_id
        JOIN players pl ON p.player_id = pl.player_id
        WHERE p.week = 0  -- Season-long projections
        """
        
        data = pd.DataFrame(self.db.execute_query(query))
        
        if data.empty:
            return data
        
        # Calculate source accuracy metrics
        source_accuracy = data.groupby('source_name').apply(
            lambda x: 1 / (np.mean(np.abs(x['projected_points'] - x['actual_points'])) + 1)
        ).to_dict()
        
        data['source_accuracy'] = data['source_name'].map(source_accuracy)
        
        # Calculate projection variance by player
        projection_stats = data.groupby(['source_name'])['projected_points'].agg(['std', 'mean']).reset_index()
        projection_stats.columns = ['source_name', 'projection_std', 'projection_mean']
        
        data = data.merge(projection_stats, on='source_name')
        
        return data
    
    def _calculate_source_weights(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate optimal weights for each projection source"""
        if not self.is_trained:
            sources = data['source_name'].unique()
            return {source: 1.0 for source in sources}
        
        # Use model to predict optimal weights
        source_weights = {}
        
        for source in data['source_name'].unique():
            source_data = data[data['source_name'] == source]
            
            if not source_data.empty:
                features = self.prepare_features(source_data)
                if not features.empty:
                    # Predict reliability score (higher = better weight)
                    reliability = np.mean(self.predict(source_data))
                    source_weights[source] = max(0.1, reliability)  # Minimum weight of 0.1
                else:
                    source_weights[source] = 1.0
            else:
                source_weights[source] = 1.0
        
        # Normalize weights to sum to number of sources
        total_weight = sum(source_weights.values())
        if total_weight > 0:
            num_sources = len(source_weights)
            source_weights = {k: (v / total_weight) * num_sources for k, v in source_weights.items()}
        
        return source_weights
    
    def _get_projection_sources(self) -> List[str]:
        """Get list of all projection sources"""
        query = "SELECT name FROM projection_sources"
        result = self.db.execute_query(query)
        return [row['name'] for row in result]
    
    def _update_source_weights(self, weights: Dict[str, float]):
        """Update source weights in database"""
        for source_name, weight in weights.items():
            query = """
            UPDATE projection_sources 
            SET weight = ? 
            WHERE name = ?
            """
            self.db.execute_update(query, (weight, source_name))
        
        self.source_weights = weights
    
    def get_current_weights(self) -> Dict[str, float]:
        """Get current source weights"""
        if not self.source_weights:
            query = "SELECT name, weight FROM projection_sources"
            result = self.db.execute_query(query)
            self.source_weights = {row['name']: row['weight'] for row in result}
        
        return self.source_weights