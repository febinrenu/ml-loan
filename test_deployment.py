#!/usr/bin/env python3
"""
Quick test script to verify the deployment is working
"""

import requests
import json

# Test data
test_data = {
    'Name': 'John Doe',
    'Gender': 'Male',
    'Married': 'Yes',
    'Dependents': '2',
    'Education': 'Graduate',
    'Self_Employed': 'No',
    'ApplicantIncome': 5000,
    'CoapplicantIncome': 1500,
    'LoanAmount': 150,
    'Loan_Amount_Term': 360,
    'Credit_History': 1,
    'Property_Area': 'Urban'
}

def test_local():
    """Test local deployment"""
    print("🧪 Testing Local Deployment...")
    try:
        # Test health endpoint
        health = requests.get('http://localhost:5000/health', timeout=5)
        print(f"✅ Health Check: {health.json()}")
        
        # Test prediction endpoint
        response = requests.post('http://localhost:5000/predict', 
                               json=test_data,
                               headers={'Content-Type': 'application/json'},
                               timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Prediction: {result['prediction']}")
            print(f"✅ Confidence: {result['confidence']:.2%}")
            print(f"✅ Applicant: {result['applicant_name']}")
        else:
            print(f"❌ Error: Status {response.status_code}")
            print(f"   Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to local server. Is it running?")
        print("   Run: python app.py")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_render(render_url):
    """Test Render deployment"""
    print(f"\n🧪 Testing Render Deployment: {render_url}")
    try:
        # Test health endpoint
        health = requests.get(f'{render_url}/health', timeout=10)
        print(f"✅ Health Check: {health.json()}")
        
        # Test prediction endpoint
        response = requests.post(f'{render_url}/predict',
                               json=test_data,
                               headers={'Content-Type': 'application/json'},
                               timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Prediction: {result['prediction']}")
            print(f"✅ Confidence: {result['confidence']:.2%}")
            print(f"✅ Applicant: {result['applicant_name']}")
        else:
            print(f"❌ Error: Status {response.status_code}")
            print(f"   Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Could not connect to {render_url}")
        print("   Make sure your Render deployment is complete")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == '__main__':
    print("="*60)
    print("LOAN ELIGIBILITY PREDICTOR - DEPLOYMENT TEST")
    print("="*60)
    
    # Test local
    test_local()
    
    # Test Render (update with your URL)
    render_url = input("\n📝 Enter your Render URL (or press Enter to skip): ").strip()
    if render_url:
        if not render_url.startswith('http'):
            render_url = f'https://{render_url}'
        test_render(render_url)
    
    print("\n" + "="*60)
    print("Testing Complete!")
    print("="*60)
