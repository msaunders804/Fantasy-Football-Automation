from abc import ABC, abstractmethod
import pickle
import os
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score

class BaseMLModel(ABC):
    """Base class for all ML models in the fantasy football system"""
    
    def __init__(self, model_name: str, db_manager):
        self.model_name = model_name
        self.db = db_manager
        self.model = None
        self.is_trained = False
        self.feature_columns = []
        self.model_path = f"models/{model_name}.pkl"
        
        # Ensure models directory exists
        os.makedirs("models", exist_ok=True)
    
    @abstractmethod
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for training/prediction"""
        pass
    
    @abstractmethod
    def create_model(self) -> Any:
        """Create the underlying ML model"""
        pass
    
    def train(self, data: pd.DataFrame, target_column: str, test_size: float = 0.2) -> Dict:
        """Train the model"""
        try:
            # Prepare features
            features_df = self.prepare_features(data)
            
            if features_df.empty:
                raise ValueError("No features prepared for training")
            
            # Prepare target
            y = data[target_column].values
            X = features_df.values
            self.feature_columns = features_df.columns.tolist()
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            # Create and train model
            self.model = self.create_model()
            self.model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test)
            
            metrics = {
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred),
                'train_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
            # Cross validation
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
            metrics['cv_mae'] = -cv_scores.mean()
            metrics['cv_std'] = cv_scores.std()
            
            self.is_trained = True
            
            # Save performance to database
            self._save_performance_metrics(metrics)
            
            return metrics
            
        except Exception as e:
            raise Exception(f"Training failed for {self.model_name}: {e}")
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError(f"Model {self.model_name} is not trained")
        
        features_df = self.prepare_features(data)
        
        if features_df.empty:
            return np.array([])
        
        # Ensure feature order matches training
        features_df = features_df.reindex(columns=self.feature_columns, fill_value=0)
        
        return self.model.predict(features_df.values)
    
    def save_model(self):
        """Save model to disk"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'model_name': self.model_name
        }
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self) -> bool:
        """Load model from disk"""
        if not os.path.exists(self.model_path):
            return False
        
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.feature_columns = model_data['feature_columns']
            self.is_trained = True
            
            return True
            
        except Exception as e:
            print(f"Failed to load model {self.model_name}: {e}")
            return False
    
    def _save_performance_metrics(self, metrics: Dict):
        """Save model performance to database"""
        query = """
        INSERT INTO model_performance (model_type, accuracy_score, mae, r2_score)
        VALUES (?, ?, ?, ?)
        """
        
        self.db.execute_update(query, (
            self.model_name,
            metrics.get('cv_mae', metrics['mae']),
            metrics['mae'],
            metrics['r2']
        ))
    
    def get_feature_importance(self) -> Dict:
        """Get feature importance if available"""
        if not self.is_trained:
            return {}
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            return dict(zip(self.feature_columns, importance))
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_)
            return dict(zip(self.feature_columns, importance))
        else:
            return {}
    
    def validate_features(self, data: pd.DataFrame) -> bool:
        """Validate that required features are present"""
        if not self.feature_columns:
            return True  # No validation needed for untrained model
        
        features_df = self.prepare_features(data)
        missing_features = set(self.feature_columns) - set(features_df.columns)
        
        if missing_features:
            print(f"Missing features: {missing_features}")
            return False
        
        return True