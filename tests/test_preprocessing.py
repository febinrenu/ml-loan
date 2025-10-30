"""
Unit Tests for Preprocessing Module
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.preprocessing import LoanDataPreprocessor


@pytest.fixture
def sample_data():
    """Create sample loan data for testing"""
    data = {
        'Loan_ID': ['LP001', 'LP002', 'LP003'],
        'Gender': ['Male', 'Female', 'Male'],
        'Married': ['Yes', 'No', 'Yes'],
        'Dependents': ['0', '1', '2'],
        'Education': ['Graduate', 'Graduate', 'Not Graduate'],
        'Self_Employed': ['No', 'Yes', 'No'],
        'ApplicantIncome': [5000, 6000, 4500],
        'CoapplicantIncome': [1500, 0, 2000],
        'LoanAmount': [150, 120, 180],
        'Loan_Amount_Term': [360, 360, 360],
        'Credit_History': [1.0, 1.0, 0.0],
        'Property_Area': ['Urban', 'Rural', 'Semiurban'],
        'Loan_Status': ['Y', 'Y', 'N']
    }
    return pd.DataFrame(data)


@pytest.fixture
def preprocessor():
    """Create a preprocessor instance"""
    return LoanDataPreprocessor()


def test_preprocessor_initialization(preprocessor):
    """Test preprocessor initialization"""
    assert preprocessor is not None
    assert preprocessor.label_encoders == {}
    assert preprocessor.scaler is not None
    assert preprocessor.feature_columns is None


def test_handle_missing_values(preprocessor, sample_data):
    """Test missing value handling"""
    # Add missing values
    df = sample_data.copy()
    df.loc[0, 'Gender'] = np.nan
    df.loc[1, 'LoanAmount'] = np.nan
    
    # Handle missing values
    df_clean = preprocessor.handle_missing_values(df)
    
    # Check no missing values
    assert df_clean.isnull().sum().sum() == 0


def test_feature_engineering(preprocessor, sample_data):
    """Test feature engineering"""
    df = sample_data.copy()
    df_engineered = preprocessor.feature_engineering(df)
    
    # Check new features are created
    assert 'TotalIncome' in df_engineered.columns
    assert 'Income_Loan_Ratio' in df_engineered.columns
    assert 'Loan_Amount_Per_Term' in df_engineered.columns
    assert 'Log_ApplicantIncome' in df_engineered.columns
    assert 'Log_LoanAmount' in df_engineered.columns
    assert 'Log_TotalIncome' in df_engineered.columns
    
    # Check calculations
    assert df_engineered['TotalIncome'].iloc[0] == 6500


def test_encode_categorical_variables(preprocessor, sample_data):
    """Test categorical encoding"""
    df = sample_data.copy()
    df_encoded = preprocessor.encode_categorical_variables(df, is_training=True)
    
    # Check encoding
    assert df_encoded['Gender'].iloc[0] == 1  # Male
    assert df_encoded['Gender'].iloc[1] == 0  # Female
    assert df_encoded['Married'].iloc[0] == 1  # Yes
    assert df_encoded['Married'].iloc[1] == 0  # No
    assert df_encoded['Education'].iloc[0] == 1  # Graduate
    assert df_encoded['Loan_Status'].iloc[0] == 1  # Y


def test_preprocess_input(preprocessor):
    """Test preprocessing of new input data"""
    # Setup preprocessor with feature columns
    preprocessor.feature_columns = [
        'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
        'Loan_Amount_Term', 'Credit_History', 'Property_Area',
        'TotalIncome', 'Income_Loan_Ratio', 'Loan_Amount_Per_Term',
        'Log_ApplicantIncome', 'Log_LoanAmount', 'Log_TotalIncome'
    ]
    
    # Create scaler with proper shape
    from sklearn.preprocessing import StandardScaler
    preprocessor.scaler = StandardScaler()
    dummy_data = np.random.rand(10, len(preprocessor.feature_columns))
    preprocessor.scaler.fit(dummy_data)
    
    # Test input
    input_data = {
        'Gender': 'Male',
        'Married': 'Yes',
        'Dependents': '0',
        'Education': 'Graduate',
        'Self_Employed': 'No',
        'ApplicantIncome': 5000,
        'CoapplicantIncome': 1500,
        'LoanAmount': 150,
        'Loan_Amount_Term': 360,
        'Credit_History': 1.0,
        'Property_Area': 'Urban'
    }
    
    processed = preprocessor.preprocess_input(input_data)
    
    # Check output shape
    assert processed.shape[0] == 1
    assert processed.shape[1] == len(preprocessor.feature_columns)


def test_data_types(preprocessor, sample_data):
    """Test that data types are correct after preprocessing"""
    df = sample_data.copy()
    df = preprocessor.handle_missing_values(df)
    df = preprocessor.feature_engineering(df)
    df = preprocessor.encode_categorical_variables(df)
    
    # Check numeric types
    numeric_cols = ['ApplicantIncome', 'LoanAmount', 'TotalIncome']
    for col in numeric_cols:
        assert pd.api.types.is_numeric_dtype(df[col])


def test_loan_status_encoding(preprocessor, sample_data):
    """Test that loan status is correctly encoded"""
    df = sample_data.copy()
    df_encoded = preprocessor.encode_categorical_variables(df)
    
    # Y should be 1, N should be 0
    assert df_encoded[df_encoded['Loan_Status'] == 1].shape[0] == 2
    assert df_encoded[df_encoded['Loan_Status'] == 0].shape[0] == 1


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
