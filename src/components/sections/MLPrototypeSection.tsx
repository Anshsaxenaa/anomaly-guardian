import { SectionCard } from "@/components/ui/SectionCard";
import { CodeBlock } from "@/components/ui/CodeBlock";
import { Brain, FlaskConical, TestTube2, Zap } from "lucide-react";

const projectStructure = `ml-anomaly-detector/
├── pyproject.toml          # Poetry dependencies
├── Dockerfile
├── docker-compose.yml
├── README.md
├── src/
│   ├── __init__.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── features.py     # Feature definitions
│   │   ├── trainer.py      # Model training
│   │   └── predictor.py    # Inference
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py         # FastAPI app
│   │   └── schemas.py      # Pydantic models
│   └── utils/
│       ├── __init__.py
│       └── explainer.py    # SHAP explanations
├── tests/
│   ├── __init__.py
│   ├── test_model.py
│   └── test_api.py
├── data/
│   └── sample_builds.csv   # Training data
└── models/
    └── .gitkeep`;

const featuresCode = `# src/model/features.py
"""Feature definitions and preprocessing"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class FeatureConfig:
    """Configuration for model features"""
    
    # Core features for Isolation Forest
    FEATURE_COLUMNS: List[str] = (
        'build_time_seconds',
        'test_fail_count',
        'test_fail_rate',
        'avg_cpu_percent',
        'max_memory_mb',
        'deploy_delta_hours',
        'commit_churn',
    )
    
    # Derived features
    DERIVED_FEATURES: List[str] = (
        'build_time_zscore',
        'resource_pressure',  # cpu * memory
        'churn_velocity',     # churn / build_time
    )


class FeatureProcessor:
    """Preprocess raw data into model features"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_stats = {}
    
    def fit(self, df: pd.DataFrame) -> 'FeatureProcessor':
        """Fit scaler on training data"""
        features = self._extract_features(df)
        self.scaler.fit(features)
        
        # Store stats for z-score calculation
        for col in FeatureConfig.FEATURE_COLUMNS:
            self.feature_stats[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'p95': df[col].quantile(0.95),
                'p99': df[col].quantile(0.99),
            }
        
        return self
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform data to feature matrix"""
        features = self._extract_features(df)
        return self.scaler.transform(features)
    
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one step"""
        self.fit(df)
        return self.transform(df)
    
    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract and engineer features"""
        features = df[list(FeatureConfig.FEATURE_COLUMNS)].copy()
        
        # Add derived features
        features['build_time_zscore'] = (
            (df['build_time_seconds'] - self.feature_stats.get('build_time_seconds', {}).get('mean', df['build_time_seconds'].mean())) /
            max(self.feature_stats.get('build_time_seconds', {}).get('std', df['build_time_seconds'].std()), 1)
        )
        
        features['resource_pressure'] = (
            df['avg_cpu_percent'] * df['max_memory_mb'] / 10000
        )
        
        features['churn_velocity'] = (
            df['commit_churn'] / np.maximum(df['build_time_seconds'] / 60, 1)
        )
        
        return features.fillna(0)
    
    def get_feature_names(self) -> List[str]:
        """Get ordered list of feature names"""
        return list(FeatureConfig.FEATURE_COLUMNS) + list(FeatureConfig.DERIVED_FEATURES)`;

const trainerCode = `# src/model/trainer.py
"""Model training with Isolation Forest"""

import json
import joblib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

from .features import FeatureProcessor


class AnomalyModelTrainer:
    """Train Isolation Forest for CI/CD anomaly detection"""
    
    def __init__(
        self,
        contamination: float = 0.05,  # Expected anomaly rate
        n_estimators: int = 200,
        max_samples: str = 'auto',
        random_state: int = 42
    ):
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=-1,
            warm_start=False
        )
        self.processor = FeatureProcessor()
        self.metadata: Dict[str, Any] = {}
    
    def train(
        self,
        data: pd.DataFrame,
        validate: bool = True
    ) -> Dict[str, Any]:
        """
        Train the model on historical build data.
        
        Returns:
            Dictionary with training metrics
        """
        print(f"Training on {len(data)} samples...")
        
        if validate:
            train_df, val_df = train_test_split(
                data, test_size=0.2, random_state=42
            )
        else:
            train_df = data
            val_df = None
        
        # Fit processor and transform
        X_train = self.processor.fit_transform(train_df)
        
        # Train Isolation Forest
        self.model.fit(X_train)
        
        # Calculate metrics
        train_scores = self.model.decision_function(X_train)
        train_preds = self.model.predict(X_train)
        
        metrics = {
            'train_samples': len(train_df),
            'train_anomaly_rate': (train_preds == -1).mean(),
            'train_score_mean': float(train_scores.mean()),
            'train_score_std': float(train_scores.std()),
        }
        
        if val_df is not None:
            X_val = self.processor.transform(val_df)
            val_scores = self.model.decision_function(X_val)
            val_preds = self.model.predict(X_val)
            
            metrics.update({
                'val_samples': len(val_df),
                'val_anomaly_rate': float((val_preds == -1).mean()),
                'val_score_mean': float(val_scores.mean()),
                'val_score_std': float(val_scores.std()),
            })
        
        self.metadata = {
            'trained_at': datetime.now().isoformat(),
            'feature_names': self.processor.get_feature_names(),
            'feature_stats': self.processor.feature_stats,
            'model_params': self.model.get_params(),
            'metrics': metrics,
        }
        
        print(f"Training complete. Anomaly rate: {metrics['train_anomaly_rate']:.2%}")
        return metrics
    
    def save(self, path: Path) -> None:
        """Save model and metadata"""
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        joblib.dump(self.model, path / 'model.joblib')
        joblib.dump(self.processor, path / 'processor.joblib')
        
        # Save metadata
        with open(path / 'metadata.json', 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
        
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: Path) -> 'AnomalyModelTrainer':
        """Load trained model"""
        trainer = cls()
        trainer.model = joblib.load(path / 'model.joblib')
        trainer.processor = joblib.load(path / 'processor.joblib')
        
        with open(path / 'metadata.json') as f:
            trainer.metadata = json.load(f)
        
        return trainer


if __name__ == "__main__":
    # Example training script
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/sample_builds.csv')
    parser.add_argument('--output', type=str, default='models/v1')
    args = parser.parse_args()
    
    # Load data
    df = pd.read_csv(args.data)
    
    # Train
    trainer = AnomalyModelTrainer(contamination=0.05)
    metrics = trainer.train(df)
    
    # Save
    trainer.save(Path(args.output))
    
    print("\\nMetrics:", json.dumps(metrics, indent=2))`;

const predictorCode = `# src/model/predictor.py
"""Real-time anomaly scoring with feature importance"""

from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import joblib


class AnomalyPredictor:
    """Score builds and explain anomalies"""
    
    def __init__(self, model_path: Path):
        self.model = joblib.load(model_path / 'model.joblib')
        self.processor = joblib.load(model_path / 'processor.joblib')
        self.feature_names = self.processor.get_feature_names()
    
    def score(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Score a single build and return anomaly details.
        
        Args:
            features: Dictionary with feature values
            
        Returns:
            Dictionary with anomaly_score, is_anomaly, and contributing_features
        """
        # Convert to DataFrame
        df = pd.DataFrame([features])
        
        # Transform features
        X = self.processor.transform(df)
        
        # Get raw score (more negative = more anomalous)
        raw_score = self.model.decision_function(X)[0]
        
        # Convert to 0-1 score (higher = more anomalous)
        # Isolation Forest: score < 0 is anomaly
        anomaly_score = self._normalize_score(raw_score)
        
        # Get prediction
        prediction = self.model.predict(X)[0]
        is_anomaly = prediction == -1
        
        # Calculate feature contributions
        contributions = self._calculate_contributions(X[0], features)
        
        return {
            'anomaly_score': float(anomaly_score),
            'is_anomaly': bool(is_anomaly),
            'raw_score': float(raw_score),
            'threshold': 0.8,
            'contributing_features': contributions,
        }
    
    def _normalize_score(self, raw_score: float) -> float:
        """
        Normalize Isolation Forest score to 0-1 range.
        IF scores: negative = anomaly, positive = normal
        We want: 1 = anomaly, 0 = normal
        """
        # Typical IF scores range from -0.5 to 0.5
        # Clip and invert
        normalized = np.clip(-raw_score + 0.5, 0, 1)
        return float(normalized)
    
    def _calculate_contributions(
        self,
        X_scaled: np.ndarray,
        raw_features: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        Calculate which features contribute most to anomaly.
        Uses z-score magnitude as proxy for importance.
        """
        contributions = []
        
        for i, (name, value) in enumerate(zip(self.feature_names, X_scaled)):
            # Higher absolute scaled value = more unusual
            importance = abs(float(value))
            
            # Determine direction
            direction = "high" if value > 0 else "low"
            
            # Get raw value if available
            raw_value = raw_features.get(name.split('_zscore')[0], None)
            
            contributions.append({
                'feature': name,
                'importance': round(importance, 3),
                'direction': direction,
                'value': raw_value,
                'scaled_value': round(float(value), 3),
            })
        
        # Sort by importance
        contributions.sort(key=lambda x: x['importance'], reverse=True)
        
        return contributions[:5]  # Top 5 contributors
    
    def batch_score(self, features_list: List[Dict]) -> List[Dict]:
        """Score multiple builds at once"""
        return [self.score(f) for f in features_list]`;

const fastapiCode = `# src/api/main.py
"""FastAPI endpoint for anomaly scoring"""

import os
from pathlib import Path
from typing import Dict, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from src.model.predictor import AnomalyPredictor


# Pydantic schemas
class BuildFeatures(BaseModel):
    """Input features for scoring"""
    build_time_seconds: float = Field(..., ge=0, description="Build duration")
    test_fail_count: int = Field(0, ge=0, description="Failed tests")
    test_fail_rate: float = Field(0.0, ge=0, le=1, description="Failure rate")
    avg_cpu_percent: float = Field(0.0, ge=0, le=100, description="Avg CPU %")
    max_memory_mb: float = Field(0.0, ge=0, description="Max memory MB")
    deploy_delta_hours: float = Field(24.0, ge=0, description="Hours since last deploy")
    commit_churn: int = Field(0, ge=0, description="Files changed")
    
    # Optional context
    workflow_run_id: Optional[str] = None
    branch: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "build_time_seconds": 450.5,
                "test_fail_count": 3,
                "test_fail_rate": 0.02,
                "avg_cpu_percent": 78.5,
                "max_memory_mb": 4096,
                "deploy_delta_hours": 2.5,
                "commit_churn": 15,
                "workflow_run_id": "run_abc123",
                "branch": "main"
            }
        }


class FeatureContribution(BaseModel):
    """Feature contribution to anomaly"""
    feature: str
    importance: float
    direction: str
    value: Optional[float]
    scaled_value: float


class ScoringResponse(BaseModel):
    """Response from /score endpoint"""
    anomaly_score: float = Field(..., ge=0, le=1, description="0-1 anomaly score")
    is_anomaly: bool = Field(..., description="Score > threshold")
    threshold: float = Field(0.8, description="Anomaly threshold")
    contributing_features: List[FeatureContribution]
    raw_score: float = Field(..., description="Raw IF score")
    
    class Config:
        json_schema_extra = {
            "example": {
                "anomaly_score": 0.87,
                "is_anomaly": True,
                "threshold": 0.8,
                "contributing_features": [
                    {"feature": "build_time_seconds", "importance": 2.3, "direction": "high", "value": 450.5, "scaled_value": 2.3},
                    {"feature": "test_fail_count", "importance": 1.8, "direction": "high", "value": 3, "scaled_value": 1.8}
                ],
                "raw_score": -0.32
            }
        }


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


# Global predictor
predictor: Optional[AnomalyPredictor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup"""
    global predictor
    model_path = Path(os.getenv("MODEL_PATH", "models/v1"))
    
    if model_path.exists():
        predictor = AnomalyPredictor(model_path)
        print(f"Model loaded from {model_path}")
    else:
        print(f"WARNING: Model not found at {model_path}")
    
    yield
    
    predictor = None


# FastAPI app
app = FastAPI(
    title="CI/CD Anomaly Detection API",
    description="ML-powered anomaly scoring for CI/CD pipelines",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": predictor is not None,
        "version": "1.0.0"
    }


@app.post("/score", response_model=ScoringResponse)
async def score_build(features: BuildFeatures):
    """
    Score a build for anomalies.
    
    Returns anomaly_score (0-1) and top contributing features.
    If anomaly_score > 0.8, consider blocking deployment.
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check MODEL_PATH."
        )
    
    # Convert to dict for predictor
    feature_dict = features.model_dump(exclude={'workflow_run_id', 'branch'})
    
    result = predictor.score(feature_dict)
    
    return ScoringResponse(
        anomaly_score=result['anomaly_score'],
        is_anomaly=result['is_anomaly'],
        threshold=result['threshold'],
        contributing_features=[
            FeatureContribution(**c) for c in result['contributing_features']
        ],
        raw_score=result['raw_score']
    )


@app.post("/score/batch", response_model=List[ScoringResponse])
async def score_batch(features_list: List[BuildFeatures]):
    """Score multiple builds at once"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    for features in features_list:
        feature_dict = features.model_dump(exclude={'workflow_run_id', 'branch'})
        result = predictor.score(feature_dict)
        results.append(ScoringResponse(
            anomaly_score=result['anomaly_score'],
            is_anomaly=result['is_anomaly'],
            threshold=result['threshold'],
            contributing_features=[
                FeatureContribution(**c) for c in result['contributing_features']
            ],
            raw_score=result['raw_score']
        ))
    
    return results


if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )`;

const testCode = `# tests/test_model.py
"""Unit tests for ML model"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile

from src.model.features import FeatureProcessor, FeatureConfig
from src.model.trainer import AnomalyModelTrainer
from src.model.predictor import AnomalyPredictor


@pytest.fixture
def sample_data():
    """Generate sample training data"""
    np.random.seed(42)
    n_samples = 1000
    
    # Normal builds
    normal = pd.DataFrame({
        'build_time_seconds': np.random.normal(300, 50, n_samples),
        'test_fail_count': np.random.poisson(1, n_samples),
        'test_fail_rate': np.random.beta(1, 50, n_samples),
        'avg_cpu_percent': np.random.normal(60, 15, n_samples),
        'max_memory_mb': np.random.normal(2048, 512, n_samples),
        'deploy_delta_hours': np.random.exponential(12, n_samples),
        'commit_churn': np.random.poisson(5, n_samples),
    })
    
    # Add some anomalies (5%)
    n_anomalies = 50
    anomalies = pd.DataFrame({
        'build_time_seconds': np.random.normal(800, 100, n_anomalies),  # Slow
        'test_fail_count': np.random.poisson(15, n_anomalies),           # Many failures
        'test_fail_rate': np.random.beta(5, 10, n_anomalies),            # High failure rate
        'avg_cpu_percent': np.random.normal(95, 5, n_anomalies),         # High CPU
        'max_memory_mb': np.random.normal(6000, 500, n_anomalies),       # High memory
        'deploy_delta_hours': np.random.exponential(1, n_anomalies),     # Rapid deploys
        'commit_churn': np.random.poisson(50, n_anomalies),              # Large changes
    })
    
    return pd.concat([normal, anomalies], ignore_index=True)


@pytest.fixture
def trained_model(sample_data):
    """Train model on sample data"""
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = AnomalyModelTrainer(contamination=0.05)
        trainer.train(sample_data, validate=False)
        
        model_path = Path(tmpdir) / 'model'
        trainer.save(model_path)
        
        yield AnomalyPredictor(model_path)


class TestFeatureProcessor:
    """Test feature preprocessing"""
    
    def test_fit_transform(self, sample_data):
        processor = FeatureProcessor()
        X = processor.fit_transform(sample_data)
        
        assert X.shape[0] == len(sample_data)
        assert X.shape[1] == len(processor.get_feature_names())
    
    def test_feature_stats(self, sample_data):
        processor = FeatureProcessor()
        processor.fit(sample_data)
        
        assert 'build_time_seconds' in processor.feature_stats
        assert 'mean' in processor.feature_stats['build_time_seconds']
        assert 'std' in processor.feature_stats['build_time_seconds']
    
    def test_handles_missing_values(self):
        df = pd.DataFrame({
            'build_time_seconds': [300, np.nan, 400],
            'test_fail_count': [1, 2, np.nan],
            'test_fail_rate': [0.01, 0.02, 0.03],
            'avg_cpu_percent': [60, 70, 80],
            'max_memory_mb': [2048, 2048, 2048],
            'deploy_delta_hours': [12, 24, 6],
            'commit_churn': [5, 10, 3],
        })
        
        processor = FeatureProcessor()
        X = processor.fit_transform(df)
        
        assert not np.isnan(X).any()


class TestTrainer:
    """Test model training"""
    
    def test_train_returns_metrics(self, sample_data):
        trainer = AnomalyModelTrainer()
        metrics = trainer.train(sample_data, validate=True)
        
        assert 'train_samples' in metrics
        assert 'val_samples' in metrics
        assert 'train_anomaly_rate' in metrics
        assert 0 < metrics['train_anomaly_rate'] < 0.2
    
    def test_save_and_load(self, sample_data):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / 'model'
            
            # Train and save
            trainer = AnomalyModelTrainer()
            trainer.train(sample_data)
            trainer.save(model_path)
            
            # Check files exist
            assert (model_path / 'model.joblib').exists()
            assert (model_path / 'processor.joblib').exists()
            assert (model_path / 'metadata.json').exists()
            
            # Load
            loaded = AnomalyModelTrainer.load(model_path)
            assert loaded.metadata is not None


class TestPredictor:
    """Test anomaly scoring"""
    
    def test_score_normal_build(self, trained_model):
        normal_features = {
            'build_time_seconds': 300,
            'test_fail_count': 1,
            'test_fail_rate': 0.01,
            'avg_cpu_percent': 60,
            'max_memory_mb': 2048,
            'deploy_delta_hours': 12,
            'commit_churn': 5,
        }
        
        result = trained_model.score(normal_features)
        
        assert 0 <= result['anomaly_score'] <= 1
        assert result['anomaly_score'] < 0.8  # Should not be anomaly
        assert not result['is_anomaly']
    
    def test_score_anomalous_build(self, trained_model):
        anomaly_features = {
            'build_time_seconds': 1000,  # Very slow
            'test_fail_count': 25,       # Many failures
            'test_fail_rate': 0.5,       # 50% failure rate
            'avg_cpu_percent': 98,       # Maxed CPU
            'max_memory_mb': 8000,       # High memory
            'deploy_delta_hours': 0.5,   # Very frequent deploys
            'commit_churn': 100,         # Huge change
        }
        
        result = trained_model.score(anomaly_features)
        
        assert result['anomaly_score'] > 0.5  # Should be flagged
        assert len(result['contributing_features']) > 0
    
    def test_contributing_features_sorted(self, trained_model):
        features = {
            'build_time_seconds': 800,
            'test_fail_count': 20,
            'test_fail_rate': 0.3,
            'avg_cpu_percent': 90,
            'max_memory_mb': 5000,
            'deploy_delta_hours': 1,
            'commit_churn': 50,
        }
        
        result = trained_model.score(features)
        contributions = result['contributing_features']
        
        # Should be sorted by importance descending
        importances = [c['importance'] for c in contributions]
        assert importances == sorted(importances, reverse=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])`;

const runScript = `#!/bin/bash
# run_local.sh - Run ML prototype locally

set -e

echo "=== CI/CD Anomaly Detection - Local Run ==="

# Check Python
python3 --version || { echo "Python 3 required"; exit 1; }

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Generate sample data if not exists
if [ ! -f "data/sample_builds.csv" ]; then
    echo "Generating sample training data..."
    python3 -c "
import numpy as np
import pandas as pd

np.random.seed(42)
n = 1000

# Normal builds
data = pd.DataFrame({
    'build_time_seconds': np.random.normal(300, 50, n),
    'test_fail_count': np.random.poisson(1, n),
    'test_fail_rate': np.random.beta(1, 50, n),
    'avg_cpu_percent': np.random.normal(60, 15, n),
    'max_memory_mb': np.random.normal(2048, 512, n),
    'deploy_delta_hours': np.random.exponential(12, n),
    'commit_churn': np.random.poisson(5, n),
})

# Add anomalies
anomalies = pd.DataFrame({
    'build_time_seconds': np.random.normal(800, 100, 50),
    'test_fail_count': np.random.poisson(15, 50),
    'test_fail_rate': np.random.beta(5, 10, 50),
    'avg_cpu_percent': np.random.normal(95, 5, 50),
    'max_memory_mb': np.random.normal(6000, 500, 50),
    'deploy_delta_hours': np.random.exponential(1, 50),
    'commit_churn': np.random.poisson(50, 50),
})

pd.concat([data, anomalies]).to_csv('data/sample_builds.csv', index=False)
print('Generated data/sample_builds.csv')
"
fi

# Train model
echo "Training model..."
python3 -m src.model.trainer --data data/sample_builds.csv --output models/v1

# Run tests
echo "Running tests..."
python3 -m pytest tests/ -v

# Start API server
echo "Starting API server on http://localhost:8000"
echo "Try: curl -X POST http://localhost:8000/score -H 'Content-Type: application/json' -d '{\"build_time_seconds\": 500, \"test_fail_count\": 5, \"test_fail_rate\": 0.1, \"avg_cpu_percent\": 80, \"max_memory_mb\": 4096, \"deploy_delta_hours\": 2, \"commit_churn\": 20}'"
python3 -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000`;

export function MLPrototypeSection() {
  return (
    <div className="space-y-6">
      <div className="mb-8">
        <h1 className="text-3xl font-bold gradient-text mb-2">ML Prototype</h1>
        <p className="text-muted-foreground">
          Isolation Forest anomaly detection with FastAPI scoring endpoint
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-4">
        <div className="section-card text-center">
          <Brain className="w-8 h-8 text-primary mx-auto mb-2" />
          <div className="font-semibold">Isolation Forest</div>
          <div className="text-xs text-muted-foreground">Unsupervised</div>
        </div>
        <div className="section-card text-center">
          <Zap className="w-8 h-8 text-accent mx-auto mb-2" />
          <div className="font-semibold">FastAPI</div>
          <div className="text-xs text-muted-foreground">/score endpoint</div>
        </div>
        <div className="section-card text-center">
          <FlaskConical className="w-8 h-8 text-success mx-auto mb-2" />
          <div className="font-semibold">7 Features</div>
          <div className="text-xs text-muted-foreground">+ 3 derived</div>
        </div>
        <div className="section-card text-center">
          <TestTube2 className="w-8 h-8 text-warning mx-auto mb-2" />
          <div className="font-semibold">pytest</div>
          <div className="text-xs text-muted-foreground">Unit tests</div>
        </div>
      </div>

      <SectionCard 
        title="Project Structure" 
        icon={<Brain className="w-5 h-5" />}
      >
        <CodeBlock code={projectStructure} language="text" filename="ml-anomaly-detector/" />
      </SectionCard>

      <SectionCard 
        title="Feature Engineering" 
        subtitle="Feature definitions and preprocessing"
        icon={<FlaskConical className="w-5 h-5" />}
        badge="Python"
      >
        <CodeBlock code={featuresCode} language="python" filename="src/model/features.py" />
      </SectionCard>

      <SectionCard 
        title="Model Trainer" 
        subtitle="Isolation Forest training with validation"
        icon={<Brain className="w-5 h-5" />}
        badge="Python"
      >
        <CodeBlock code={trainerCode} language="python" filename="src/model/trainer.py" />
      </SectionCard>

      <SectionCard 
        title="Predictor" 
        subtitle="Scoring with feature importance"
        icon={<Zap className="w-5 h-5" />}
        badge="Python"
      >
        <CodeBlock code={predictorCode} language="python" filename="src/model/predictor.py" />
      </SectionCard>

      <SectionCard 
        title="FastAPI Endpoint" 
        subtitle="/score endpoint with OpenAPI docs"
        icon={<Zap className="w-5 h-5" />}
        badge="Python"
      >
        <CodeBlock code={fastapiCode} language="python" filename="src/api/main.py" />
      </SectionCard>

      <SectionCard 
        title="Unit Tests" 
        subtitle="pytest test suite"
        icon={<TestTube2 className="w-5 h-5" />}
        badge="Python"
      >
        <CodeBlock code={testCode} language="python" filename="tests/test_model.py" />
      </SectionCard>

      <SectionCard 
        title="Run Locally" 
        subtitle="Bash script to train and start API"
        icon={<Zap className="w-5 h-5" />}
        badge="bash"
      >
        <CodeBlock code={runScript} language="bash" filename="run_local.sh" />
      </SectionCard>
    </div>
  );
}
