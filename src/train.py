"""
Model Training Module for Loan Eligibility Prediction
Trains Logistic Regression and XGBoost models with comprehensive evaluation
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, classification_report,
    roc_curve
)
import xgboost as xgb
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


class LoanPredictionModel:
    """
    Model training and evaluation for loan prediction
    """
    
    def __init__(self, model_type='logistic'):
        """
        Initialize model
        model_type: 'logistic' or 'xgboost'
        """
        self.model_type = model_type
        self.model = None
        self.metrics = {}
        
        if model_type == 'logistic':
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced'
            )
        elif model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False
            )
        else:
            raise ValueError("model_type must be 'logistic' or 'xgboost'")
    
    def train(self, X_train, y_train):
        """Train the model"""
        print(f"\nTraining {self.model_type.upper()} model...")
        self.model.fit(X_train, y_train)
        print("Training completed!")
        return self
    
    def evaluate(self, X_test, y_test, save_plots=True):
        """Evaluate the model"""
        print(f"\nEvaluating {self.model_type.upper()} model...")
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Print metrics
        print("\n" + "="*60)
        print(f"MODEL: {self.model_type.upper()}")
        print("="*60)
        print(f"Accuracy:  {self.metrics['accuracy']:.4f}")
        print(f"Precision: {self.metrics['precision']:.4f}")
        print(f"Recall:    {self.metrics['recall']:.4f}")
        print(f"F1-Score:  {self.metrics['f1_score']:.4f}")
        print(f"ROC-AUC:   {self.metrics['roc_auc']:.4f}")
        print("="*60)
        
        # Classification Report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Rejected', 'Approved']))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Save plots
        if save_plots:
            self.save_evaluation_plots(y_test, y_pred, y_pred_proba, cm)
        
        return self.metrics
    
    def save_evaluation_plots(self, y_test, y_pred, y_pred_proba, cm):
        """Save evaluation plots"""
        os.makedirs('reports', exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Confusion Matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                    xticklabels=['Rejected', 'Approved'],
                    yticklabels=['Rejected', 'Approved'])
        axes[0].set_title(f'Confusion Matrix - {self.model_type.upper()}')
        axes[0].set_ylabel('True Label')
        axes[0].set_xlabel('Predicted Label')
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        axes[1].plot(fpr, tpr, label=f'ROC curve (AUC = {self.metrics["roc_auc"]:.4f})')
        axes[1].plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title(f'ROC Curve - {self.model_type.upper()}')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plot_path = f'reports/{self.model_type}_evaluation.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nEvaluation plots saved to {plot_path}")
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.model.predict_proba(X)
    
    def save_model(self, filepath=None):
        """Save the trained model"""
        if filepath is None:
            filepath = f'models/{self.model_type}_model.pkl'
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        print(f"\nModel saved to {filepath}")
        
        # Save metrics
        metrics_path = f'models/{self.model_type}_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        print(f"Metrics saved to {metrics_path}")
    
    @staticmethod
    def load_model(filepath):
        """Load a saved model"""
        return joblib.load(filepath)


def compare_models(models_metrics):
    """
    Compare multiple models and select the best one
    """
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    df_metrics = pd.DataFrame(models_metrics).T
    print(df_metrics)
    
    # Find best model based on F1-score (balanced metric)
    best_model_name = df_metrics['f1_score'].idxmax()
    best_f1_score = df_metrics['f1_score'].max()
    
    print("\n" + "="*80)
    print(f"BEST MODEL: {best_model_name.upper()} (F1-Score: {best_f1_score:.4f})")
    print("="*80)
    
    return best_model_name, df_metrics


def train_and_evaluate_all_models():
    """
    Train and evaluate all models
    """
    print("\n" + "="*80)
    print("LOAN ELIGIBILITY PREDICTION - MODEL TRAINING PIPELINE")
    print("="*80)
    
    # Load processed data
    print("\nLoading processed data...")
    X_train = pd.read_csv('data/processed/X_train.csv')
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
    y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Dictionary to store all models and metrics
    models = {}
    models_metrics = {}
    
    # Train Logistic Regression
    print("\n" + "-"*80)
    print("1. LOGISTIC REGRESSION")
    print("-"*80)
    lr_model = LoanPredictionModel(model_type='logistic')
    lr_model.train(X_train, y_train)
    lr_metrics = lr_model.evaluate(X_test, y_test)
    lr_model.save_model()
    models['logistic'] = lr_model
    models_metrics['logistic'] = lr_metrics
    
    # Train XGBoost
    print("\n" + "-"*80)
    print("2. XGBOOST")
    print("-"*80)
    xgb_model = LoanPredictionModel(model_type='xgboost')
    xgb_model.train(X_train, y_train)
    xgb_metrics = xgb_model.evaluate(X_test, y_test)
    xgb_model.save_model()
    models['xgboost'] = xgb_model
    models_metrics['xgboost'] = xgb_metrics
    
    # Compare models
    best_model_name, comparison_df = compare_models(models_metrics)
    
    # Save comparison
    os.makedirs('reports', exist_ok=True)
    comparison_df.to_csv('reports/model_comparison.csv')
    print(f"\nModel comparison saved to reports/model_comparison.csv")
    
    # Create a copy of the best model as the production model
    best_model = models[best_model_name].model
    joblib.dump(best_model, 'models/best_model.pkl')
    print(f"\nBest model saved as models/best_model.pkl")
    
    # Save best model info
    best_model_info = {
        'model_type': best_model_name,
        'metrics': models_metrics[best_model_name],
        'timestamp': datetime.now().isoformat()
    }
    with open('models/best_model_info.json', 'w') as f:
        json.dump(best_model_info, f, indent=4)
    
    print("\n" + "="*80)
    print("MODEL TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    return models, models_metrics, best_model_name


if __name__ == "__main__":
    # Run the complete training pipeline
    models, metrics, best_model = train_and_evaluate_all_models()
