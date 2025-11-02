
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Try to import preprocessing module
try:
    from src.preprocessing import LoanDataPreprocessor
    print("✓ Imported LoanDataPreprocessor from src.preprocessing")
except ImportError as e:
    print(f"⚠ Could not import from src.preprocessing: {e}")
    try:
        from preprocessing import LoanDataPreprocessor
        print("✓ Imported LoanDataPreprocessor from preprocessing")
    except ImportError as e2:
        print(f"✗ Could not import LoanDataPreprocessor: {e2}")
        LoanDataPreprocessor = None

app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')

# Disable template caching for development
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Configure CORS to allow all origins
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Global variables for model and preprocessor
model = None
preprocessor = None

# Fallback preprocessing function in case the pickled preprocessor fails
def fallback_preprocess_input(input_data):
    """Fallback preprocessing function with feature engineering"""
    try:
        # Basic feature encoding
        gender = 1 if input_data['Gender'] == 'Male' else 0
        married = 1 if input_data['Married'] == 'Yes' else 0
        dependents = float(input_data['Dependents'].replace('+', ''))
        education = 1 if input_data['Education'] == 'Graduate' else 0
        self_employed = 1 if input_data['Self_Employed'] == 'Yes' else 0
        applicant_income = float(input_data['ApplicantIncome'])
        coapplicant_income = float(input_data['CoapplicantIncome'])
        loan_amount = float(input_data['LoanAmount'])
        loan_term = float(input_data['Loan_Amount_Term'])
        credit_history = float(input_data['Credit_History'])
        property_area = {'Urban': 2, 'Semiurban': 1, 'Rural': 0}[input_data['Property_Area']]
        
        # Feature engineering (same as preprocessing.py)
        total_income = applicant_income + coapplicant_income
        income_loan_ratio = total_income / (loan_amount + 1)
        loan_per_term = loan_amount / (loan_term + 1)
        log_applicant_income = np.log1p(applicant_income)
        log_loan_amount = np.log1p(loan_amount)
        log_total_income = np.log1p(total_income)
        
        # Create feature array in the correct order (17 features total)
        # Order: Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, 
        #        CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area,
        #        TotalIncome, Income_Loan_Ratio, Loan_Amount_Per_Term, Log_ApplicantIncome,
        #        Log_LoanAmount, Log_TotalIncome
        features = [
            gender, married, dependents, education, self_employed,
            applicant_income, coapplicant_income, loan_amount, loan_term,
            credit_history, property_area, total_income, income_loan_ratio,
            loan_per_term, log_applicant_income, log_loan_amount, log_total_income
        ]
        
        # Convert to numpy array
        return np.array([features])
        
    except Exception as e:
        print(f"❌ Fallback preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        raise

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
            print(f"✓ Preprocessor type: {type(preprocessor)}")
            print(f"✓ Preprocessor has preprocess_input method: {hasattr(preprocessor, 'preprocess_input')}")
        else:
            print(f"✗ Preprocessor not found at {preprocessor_path}")
            return False

        return True
    except Exception as e:
        print(f"✗ Error loading artifacts: {e}")
        import traceback
        traceback.print_exc()
        return False

@app.route('/')
def home():
    """Serve the main web page"""
    response = app.make_response(render_template('index.html'))
    # Disable caching to ensure fresh content
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'preprocessor_loaded': preprocessor is not None
    })

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    """
    Prediction endpoint
    Expects JSON with loan applicant details
    """
    # Handle preflight request
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        # Get JSON data from request
        data = request.get_json()
        
        print(f"\n📥 Received prediction request: {data}")
        
        if not data:
            print("❌ No data provided")
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
        
        # Check if model and preprocessor are loaded
        if model is None:
            return jsonify({
                'error': 'Model not loaded. Please check server logs.',
                'success': False
            }), 500

        # Preprocess input
        try:
            if preprocessor is not None and hasattr(preprocessor, 'preprocess_input'):
                processed_data = preprocessor.preprocess_input(input_data)
            else:
                print("⚠ Using fallback preprocessing")
                processed_data = fallback_preprocess_input(input_data)
        except Exception as e:
            print(f"❌ Preprocessing failed: {e}, trying fallback")
            try:
                processed_data = fallback_preprocess_input(input_data)
            except Exception as e2:
                return jsonify({
                    'error': f'Preprocessing failed: {str(e2)}',
                    'success': False
                }), 500
        
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
        
        print(f"✅ Prediction result: {result['prediction']} (confidence: {result['confidence']:.2%})")
        
        return jsonify(result)
    
    except ValueError as e:
        print(f"❌ ValueError: {str(e)}")
        return jsonify({
            'error': f'Invalid data type: {str(e)}',
            'success': False
        }), 400
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        import traceback
        traceback.print_exc()
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

# Load artifacts when the module is imported (for Gunicorn)
print("\n" + "="*60)
print("LOAN ELIGIBILITY PREDICTION API")
print("="*60)
print("\nLoading model and preprocessor...")

if load_artifacts():
    print("\n✓ All artifacts loaded successfully!")
else:
    print("\n⚠ Warning: Failed to load some artifacts. Predictions may fail.")

if __name__ == '__main__':
    # Run the app in development mode
    print("\nStarting Flask development server...")
    print("="*60)
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
