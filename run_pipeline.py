"""
Script to run the complete ML pipeline
"""

import sys
import os

print("="*80)
print("LOAN ELIGIBILITY PREDICTOR - COMPLETE PIPELINE")
print("="*80)

# Step 1: Preprocessing
print("\n[1/2] Running Data Preprocessing...")
print("-"*80)
try:
    from src.preprocessing import LoanDataPreprocessor
    preprocessor = LoanDataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.preprocess('loan.csv')
    print("✓ Preprocessing completed successfully!")
except Exception as e:
    print(f"✗ Preprocessing failed: {e}")
    sys.exit(1)

# Step 2: Model Training
print("\n[2/2] Running Model Training...")
print("-"*80)
try:
    from src.train import train_and_evaluate_all_models
    models, metrics, best_model = train_and_evaluate_all_models()
    print("✓ Model training completed successfully!")
except Exception as e:
    print(f"✗ Model training failed: {e}")
    sys.exit(1)

print("\n" + "="*80)
print("PIPELINE COMPLETED SUCCESSFULLY!")
print("="*80)
print("\nYou can now run the Flask app:")
print("  python app.py")
print("\nOr build the Docker image:")
print("  docker build -t loan-predictor .")
print("  docker run -p 5000:5000 loan-predictor")
