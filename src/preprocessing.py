"""
Data Preprocessing Module for Loan Eligibility Prediction
Handles missing values, encoding, scaling, and feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os


class LoanDataPreprocessor:
    """
    Preprocessor for loan prediction data
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def load_data(self, file_path):
        """Load the loan dataset"""
        print(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path)
        print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        print("\nHandling missing values...")
        
        # Create a copy to avoid modifying original
        df = df.copy()
        
        # Fill missing values for categorical columns with mode
        categorical_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Loan_Amount_Term', 'Credit_History']
        for col in categorical_cols:
            if col in df.columns:
                mode_value = df[col].mode()[0] if not df[col].mode().empty else df[col].value_counts().index[0]
                df[col].fillna(mode_value, inplace=True)
        
        # Fill missing values for numerical columns with median
        numerical_cols = ['LoanAmount', 'ApplicantIncome', 'CoapplicantIncome']
        for col in numerical_cols:
            if col in df.columns:
                df[col].fillna(df[col].median(), inplace=True)
        
        print(f"Missing values after handling: {df.isnull().sum().sum()}")
        return df
    
    def feature_engineering(self, df):
        """Create new features"""
        print("\nPerforming feature engineering...")
        
        df = df.copy()
        
        # Total Income
        df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
        
        # Income to Loan Ratio
        df['Income_Loan_Ratio'] = df['TotalIncome'] / (df['LoanAmount'] + 1)
        
        # Loan Amount per Term
        df['Loan_Amount_Per_Term'] = df['LoanAmount'] / (df['Loan_Amount_Term'] + 1)
        
        # Log transformations for skewed features
        df['Log_ApplicantIncome'] = np.log1p(df['ApplicantIncome'])
        df['Log_LoanAmount'] = np.log1p(df['LoanAmount'])
        df['Log_TotalIncome'] = np.log1p(df['TotalIncome'])
        
        return df
    
    def encode_categorical_variables(self, df, is_training=True):
        """Encode categorical variables"""
        print("\nEncoding categorical variables...")
        
        df = df.copy()
        
        # Binary encoding for straightforward binary variables
        binary_mappings = {
            'Gender': {'Male': 1, 'Female': 0},
            'Married': {'Yes': 1, 'No': 0},
            'Education': {'Graduate': 1, 'Not Graduate': 0},
            'Self_Employed': {'Yes': 1, 'No': 0},
            'Property_Area': {'Urban': 2, 'Semiurban': 1, 'Rural': 0},
            'Loan_Status': {'Y': 1, 'N': 0}
        }
        
        for col, mapping in binary_mappings.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)
        
        # Handle Dependents (ordinal)
        if 'Dependents' in df.columns:
            df['Dependents'] = df['Dependents'].replace('3+', '3')
            df['Dependents'] = df['Dependents'].astype(float)
        
        return df
    
    def scale_features(self, X_train, X_test=None, is_training=True):
        """Scale numerical features"""
        print("\nScaling features...")
        
        if is_training:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
            
            if X_test is not None:
                X_test_scaled = self.scaler.transform(X_test)
                X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
                return X_train_scaled, X_test_scaled
            return X_train_scaled
        else:
            X_scaled = self.scaler.transform(X_train)
            X_scaled = pd.DataFrame(X_scaled, columns=X_train.columns, index=X_train.index)
            return X_scaled
    
    def preprocess(self, file_path, test_size=0.2, random_state=42, save_artifacts=True):
        """
        Complete preprocessing pipeline
        """
        # Load data
        df = self.load_data(file_path)
        
        # Drop Loan_ID as it's not a feature
        if 'Loan_ID' in df.columns:
            df = df.drop('Loan_ID', axis=1)
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Feature engineering
        df = self.feature_engineering(df)
        
        # Encode categorical variables
        df = self.encode_categorical_variables(df, is_training=True)
        
        # Separate features and target
        X = df.drop('Loan_Status', axis=1)
        y = df['Loan_Status']
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"\nTrain set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Class distribution in training set:")
        print(y_train.value_counts(normalize=True))
        
        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test, is_training=True)
        
        # Save artifacts
        if save_artifacts:
            self.save_preprocessor()
            
            # Save processed data
            os.makedirs('data/processed', exist_ok=True)
            pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv('data/processed/X_train.csv', index=False)
            pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv('data/processed/X_test.csv', index=False)
            pd.DataFrame(y_train).to_csv('data/processed/y_train.csv', index=False)
            pd.DataFrame(y_test).to_csv('data/processed/y_test.csv', index=False)
            print("\nProcessed data saved to data/processed/")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def preprocess_input(self, input_data):
        """
        Preprocess new input data for prediction
        input_data: dict or DataFrame
        """
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()
        
        # Feature engineering
        df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
        df['Income_Loan_Ratio'] = df['TotalIncome'] / (df['LoanAmount'] + 1)
        df['Loan_Amount_Per_Term'] = df['LoanAmount'] / (df['Loan_Amount_Term'] + 1)
        df['Log_ApplicantIncome'] = np.log1p(df['ApplicantIncome'])
        df['Log_LoanAmount'] = np.log1p(df['LoanAmount'])
        df['Log_TotalIncome'] = np.log1p(df['TotalIncome'])
        
        # Encode categorical variables
        binary_mappings = {
            'Gender': {'Male': 1, 'Female': 0},
            'Married': {'Yes': 1, 'No': 0},
            'Education': {'Graduate': 1, 'Not Graduate': 0},
            'Self_Employed': {'Yes': 1, 'No': 0},
            'Property_Area': {'Urban': 2, 'Semiurban': 1, 'Rural': 0}
        }
        
        for col, mapping in binary_mappings.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)
        
        # Handle Dependents
        if 'Dependents' in df.columns:
            df['Dependents'] = df['Dependents'].replace('3+', '3')
            df['Dependents'] = df['Dependents'].astype(float)
        
        # Ensure all required columns are present
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        # Select only the columns used during training
        df = df[self.feature_columns]
        
        # Scale features
        df_scaled = self.scale_features(df, is_training=False)
        
        return df_scaled
    
    def save_preprocessor(self, filepath='models/preprocessor.pkl'):
        """Save the preprocessor"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        print(f"\nPreprocessor saved to {filepath}")
    
    @staticmethod
    def load_preprocessor(filepath='models/preprocessor.pkl'):
        """Load the preprocessor"""
        return joblib.load(filepath)


if __name__ == "__main__":
    # Example usage
    preprocessor = LoanDataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.preprocess('loan.csv')
    
    print("\n" + "="*50)
    print("Preprocessing completed successfully!")
    print("="*50)
    print(f"\nFeatures shape: {X_train.shape}")
    print(f"Features: {preprocessor.feature_columns}")
