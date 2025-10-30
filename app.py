"""
Flask API for Loan Eligibility Prediction
Provides REST endpoints for making predictions
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.preprocessing import LoanDataPreprocessor
except ImportError:
    from preprocessing import LoanDataPreprocessor

app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')
CORS(app)

# Global variables for model and preprocessor
model = None
preprocessor = None

def load_artifacts():
    """Load model and preprocessor"""
    global model, preprocessor
    
    try:
        # Load the best model
        model_path = 'models/best_model.pkl'
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print(f"✓ Model loaded from {model_path}")
        else:
            print(f"✗ Model not found at {model_path}")
            return False
        
        # Load preprocessor
        preprocessor_path = 'models/preprocessor.pkl'
        if os.path.exists(preprocessor_path):
            preprocessor = joblib.load(preprocessor_path)
            print(f"✓ Preprocessor loaded from {preprocessor_path}")
        else:
            print(f"✗ Preprocessor not found at {preprocessor_path}")
            return False
        
        return True
    except Exception as e:
        print(f"✗ Error loading artifacts: {e}")
        return False

@app.route('/')
def home():
    """Serve the main web page"""
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'preprocessor_loaded': preprocessor is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint
    Expects JSON with loan applicant details
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No data provided',
                'success': False
            }), 400
        
        # Validate required fields
        required_fields = [
            'Gender', 'Married', 'Dependents', 'Education', 
            'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome',
            'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area'
        ]
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {", ".join(missing_fields)}',
                'success': False
            }), 400
        
        # Convert data to appropriate types
        input_data = {
            'Gender': str(data['Gender']),
            'Married': str(data['Married']),
            'Dependents': str(data['Dependents']),
            'Education': str(data['Education']),
            'Self_Employed': str(data['Self_Employed']),
            'ApplicantIncome': float(data['ApplicantIncome']),
            'CoapplicantIncome': float(data['CoapplicantIncome']),
            'LoanAmount': float(data['LoanAmount']),
            'Loan_Amount_Term': float(data['Loan_Amount_Term']),
            'Credit_History': float(data['Credit_History']),
            'Property_Area': str(data['Property_Area'])
        }
        
        # Preprocess input
        processed_data = preprocessor.preprocess_input(input_data)
        
        # Make prediction
        prediction = model.predict(processed_data)[0]
        prediction_proba = model.predict_proba(processed_data)[0]
        
        # Prepare response
        result = {
            'success': True,
            'prediction': 'Approved' if prediction == 1 else 'Rejected',
            'prediction_value': int(prediction),
            'confidence': float(max(prediction_proba)),
            'probability_approved': float(prediction_proba[1]),
            'probability_rejected': float(prediction_proba[0]),
            'applicant_name': data.get('Name', 'Applicant')
        }
        
        return jsonify(result)
    
    except ValueError as e:
        return jsonify({
            'error': f'Invalid data type: {str(e)}',
            'success': False
        }), 400
    except Exception as e:
        return jsonify({
            'error': f'Prediction error: {str(e)}',
            'success': False
        }), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    try:
        import json
        
        info_path = 'models/best_model_info.json'
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                info = json.load(f)
            return jsonify(info)
        else:
            return jsonify({
                'message': 'Model info not available'
            })
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'success': False
    }), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    return jsonify({
        'error': 'Internal server error',
        'success': False
    }), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("LOAN ELIGIBILITY PREDICTION API")
    print("="*60)
    
    # Load artifacts
    print("\nLoading model and preprocessor...")
    if load_artifacts():
        print("\n✓ All artifacts loaded successfully!")
        print("\nStarting Flask server...")
        print("="*60)
        
        # Run the app
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        print("\n✗ Failed to load artifacts. Please run preprocessing and training first.")
        print("Run: python src/preprocessing.py")
        print("Then: python src/train.py")
        sys.exit(1)
