"""
Test the Loan Predictor API
Run this script to test all API endpoints
Make sure the Flask server is running before executing this script!
"""

import requests
import json
import sys
import time

# API Base URL
BASE_URL = "http://localhost:5000"

# Wait a moment for server to be ready
time.sleep(1)

def test_health():
    """Test health check endpoint"""
    print("\n" + "="*60)
    print("Testing Health Check Endpoint")
    print("="*60)
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_prediction(applicant_data):
    """Test prediction endpoint"""
    print("\n" + "="*60)
    print("Testing Prediction Endpoint")
    print("="*60)
    print(f"Applicant: {applicant_data['Name']}")
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=applicant_data,
            headers={'Content-Type': 'application/json'}
        )
        
        print(f"\nStatus Code: {response.status_code}")
        result = response.json()
        print(f"\nResponse:")
        print(json.dumps(result, indent=2))
        
        if result.get('success'):
            print(f"\n{'✅' if result['prediction'] == 'Approved' else '❌'} Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"Approval Probability: {result['probability_approved']:.2%}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_model_info():
    """Test model info endpoint"""
    print("\n" + "="*60)
    print("Testing Model Info Endpoint")
    print("="*60)
    
    try:
        response = requests.get(f"{BASE_URL}/model-info")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def run_all_tests():
    """Run all API tests"""
    print("\n" + "="*70)
    print("🧪 LOAN PREDICTOR API TEST SUITE")
    print("="*70)
    
    results = []
    
    # Test 1: Health Check
    results.append(("Health Check", test_health()))
    
    # Test 2: Model Info
    results.append(("Model Info", test_model_info()))
    
    # Test 3: High-probability approval case
    high_approval_case = {
        "Name": "John Doe",
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": "0",
        "Education": "Graduate",
        "Self_Employed": "No",
        "ApplicantIncome": 8000,
        "CoapplicantIncome": 3000,
        "LoanAmount": 150,
        "Loan_Amount_Term": 360,
        "Credit_History": 1.0,
        "Property_Area": "Urban"
    }
    results.append(("Prediction (High Approval)", test_prediction(high_approval_case)))
    
    # Test 4: Low-probability approval case
    low_approval_case = {
        "Name": "Jane Smith",
        "Gender": "Female",
        "Married": "No",
        "Dependents": "3+",
        "Education": "Not Graduate",
        "Self_Employed": "Yes",
        "ApplicantIncome": 2000,
        "CoapplicantIncome": 0,
        "LoanAmount": 300,
        "Loan_Amount_Term": 360,
        "Credit_History": 0.0,
        "Property_Area": "Rural"
    }
    results.append(("Prediction (Low Approval)", test_prediction(low_approval_case)))
    
    # Test 5: Medium case
    medium_case = {
        "Name": "Alex Johnson",
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": "1",
        "Education": "Graduate",
        "Self_Employed": "No",
        "ApplicantIncome": 5000,
        "CoapplicantIncome": 1500,
        "LoanAmount": 180,
        "Loan_Amount_Term": 360,
        "Credit_History": 1.0,
        "Property_Area": "Semiurban"
    }
    results.append(("Prediction (Medium)", test_prediction(medium_case)))
    
    # Print Summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:<30} {status}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    print("\n" + "="*70)
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    print("="*70)
    
    return passed == total

if __name__ == "__main__":
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        sys.exit(1)
