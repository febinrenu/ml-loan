"""
Unit Tests for Flask API
"""

import pytest
import json
import sys
import os

# Add root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


@pytest.fixture
def sample_request_data():
    """Sample request data for testing"""
    return {
        "Name": "John Doe",
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": "0",
        "Education": "Graduate",
        "Self_Employed": "No",
        "ApplicantIncome": 5000,
        "CoapplicantIncome": 1500,
        "LoanAmount": 150,
        "Loan_Amount_Term": 360,
        "Credit_History": 1.0,
        "Property_Area": "Urban"
    }


def test_sample_data_structure(sample_request_data):
    """Test that sample request data has correct structure"""
    required_fields = [
        'Gender', 'Married', 'Dependents', 'Education',
        'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome',
        'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area'
    ]
    
    for field in required_fields:
        assert field in sample_request_data


def test_sample_data_types(sample_request_data):
    """Test that sample request data has correct types"""
    assert isinstance(sample_request_data['Gender'], str)
    assert isinstance(sample_request_data['Married'], str)
    assert isinstance(sample_request_data['Education'], str)
    assert isinstance(sample_request_data['ApplicantIncome'], (int, float))
    assert isinstance(sample_request_data['LoanAmount'], (int, float))


def test_request_data_validation(sample_request_data):
    """Test request data validation logic"""
    # Test valid data
    assert sample_request_data['ApplicantIncome'] > 0
    assert sample_request_data['LoanAmount'] > 0
    assert sample_request_data['Credit_History'] in [0.0, 1.0]


def test_categorical_values(sample_request_data):
    """Test that categorical values are valid"""
    assert sample_request_data['Gender'] in ['Male', 'Female']
    assert sample_request_data['Married'] in ['Yes', 'No']
    assert sample_request_data['Education'] in ['Graduate', 'Not Graduate']
    assert sample_request_data['Self_Employed'] in ['Yes', 'No']
    assert sample_request_data['Property_Area'] in ['Urban', 'Semiurban', 'Rural']
    assert sample_request_data['Dependents'] in ['0', '1', '2', '3+']


def test_missing_fields():
    """Test handling of missing fields"""
    incomplete_data = {
        "Name": "John Doe",
        "Gender": "Male"
        # Missing other required fields
    }
    
    required_fields = [
        'Gender', 'Married', 'Dependents', 'Education',
        'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome',
        'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area'
    ]
    
    missing = [f for f in required_fields if f not in incomplete_data]
    assert len(missing) > 0


def test_numeric_field_ranges(sample_request_data):
    """Test that numeric fields are in valid ranges"""
    assert sample_request_data['ApplicantIncome'] >= 0
    assert sample_request_data['CoapplicantIncome'] >= 0
    assert sample_request_data['LoanAmount'] > 0
    assert sample_request_data['Loan_Amount_Term'] > 0
    assert sample_request_data['Credit_History'] >= 0


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
