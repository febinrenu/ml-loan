"""
Unit Tests for Model Training Module
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.train import LoanPredictionModel


@pytest.fixture
def sample_train_data():
    """Create sample training data"""
    np.random.seed(42)
    X_train = pd.DataFrame(np.random.rand(100, 17))
    y_train = np.random.randint(0, 2, 100)
    return X_train, y_train


@pytest.fixture
def sample_test_data():
    """Create sample test data"""
    np.random.seed(42)
    X_test = pd.DataFrame(np.random.rand(30, 17))
    y_test = np.random.randint(0, 2, 30)
    return X_test, y_test


def test_logistic_model_initialization():
    """Test logistic regression model initialization"""
    model = LoanPredictionModel(model_type='logistic')
    assert model is not None
    assert model.model_type == 'logistic'
    assert model.model is not None


def test_xgboost_model_initialization():
    """Test XGBoost model initialization"""
    model = LoanPredictionModel(model_type='xgboost')
    assert model is not None
    assert model.model_type == 'xgboost'
    assert model.model is not None


def test_invalid_model_type():
    """Test that invalid model type raises error"""
    with pytest.raises(ValueError):
        LoanPredictionModel(model_type='invalid')


def test_model_training(sample_train_data):
    """Test model training"""
    X_train, y_train = sample_train_data
    model = LoanPredictionModel(model_type='logistic')
    
    # Train model
    model.train(X_train, y_train)
    
    # Check that model is fitted
    assert hasattr(model.model, 'coef_')


def test_model_prediction(sample_train_data, sample_test_data):
    """Test model prediction"""
    X_train, y_train = sample_train_data
    X_test, y_test = sample_test_data
    
    model = LoanPredictionModel(model_type='logistic')
    model.train(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Check predictions
    assert len(predictions) == len(X_test)
    assert all(pred in [0, 1] for pred in predictions)


def test_model_predict_proba(sample_train_data, sample_test_data):
    """Test model probability prediction"""
    X_train, y_train = sample_train_data
    X_test, y_test = sample_test_data
    
    model = LoanPredictionModel(model_type='logistic')
    model.train(X_train, y_train)
    
    # Get probabilities
    probabilities = model.predict_proba(X_test)
    
    # Check probabilities
    assert probabilities.shape[0] == len(X_test)
    assert probabilities.shape[1] == 2
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_model_evaluation(sample_train_data, sample_test_data):
    """Test model evaluation"""
    X_train, y_train = sample_train_data
    X_test, y_test = sample_test_data
    
    model = LoanPredictionModel(model_type='logistic')
    model.train(X_train, y_train)
    
    # Evaluate model
    metrics = model.evaluate(X_test, y_test, save_plots=False)
    
    # Check metrics
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1_score' in metrics
    assert 'roc_auc' in metrics
    
    # Check metric values are between 0 and 1
    for metric_value in metrics.values():
        assert 0 <= metric_value <= 1


def test_xgboost_training(sample_train_data):
    """Test XGBoost model training"""
    X_train, y_train = sample_train_data
    model = LoanPredictionModel(model_type='xgboost')
    
    # Train model
    model.train(X_train, y_train)
    
    # Check that model is fitted
    assert hasattr(model.model, 'feature_importances_')


def test_xgboost_prediction(sample_train_data, sample_test_data):
    """Test XGBoost prediction"""
    X_train, y_train = sample_train_data
    X_test, y_test = sample_test_data
    
    model = LoanPredictionModel(model_type='xgboost')
    model.train(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Check predictions
    assert len(predictions) == len(X_test)
    assert all(pred in [0, 1] for pred in predictions)


def test_metrics_stored(sample_train_data, sample_test_data):
    """Test that metrics are stored in model"""
    X_train, y_train = sample_train_data
    X_test, y_test = sample_test_data
    
    model = LoanPredictionModel(model_type='logistic')
    model.train(X_train, y_train)
    model.evaluate(X_test, y_test, save_plots=False)
    
    # Check that metrics are stored
    assert model.metrics != {}
    assert len(model.metrics) == 5


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
