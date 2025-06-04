# language: py
# AI-Generated Code Header
# **Intent:** [Simple test script to verify the functionality of all NIR spectroscopic models without logging overhead. Provides quick validation of model implementation and basic performance checking.]
# **Optimization:** [Lightweight testing approach with minimal data processing and reduced cross-validation for fast feedback during development.]
# **Safety:** [Includes basic error handling for each model test and graceful failure reporting without stopping the entire test suite.]

import numpy as np
import pandas as pd
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import data loader
from kiwi_data_loader import load_kiwi_data

# Import all model classes using importlib for numbered filenames
import importlib

# Dynamic imports for numbered model files
pls_module = importlib.import_module('model.M1_pls_model')
PLSModel = pls_module.PLSModel

rf_module = importlib.import_module('model.M2_random_forest_model')
RandomForestModel = rf_module.RandomForestModel

svr_module = importlib.import_module('model.M3_svr_model')
SVRModel = svr_module.SVRModel

xgb_module = importlib.import_module('model.M4_xgboost_model')
XGBoostModel = xgb_module.XGBoostModel

cnn_module = importlib.import_module('model.M5_cnn_model')
CNNModel = cnn_module.CNNModel

def load_test_data():
    """
    Load test data from test.csv file in src/data folder for quick testing.
    
    Returns:
        tuple: (X, y, feature_names) for testing
    """
    print("📥 Loading test data from src/data/test.csv...")
    
    try:
        # Load the test data using kiwi_data_loader with specific_file parameter
        df = load_kiwi_data(data_directory="src/data", specific_file="test.csv")
        
        if df.empty:
            raise ValueError("test.csv file is empty or could not be loaded.")
        
        print(f"Test data loaded: {df.shape[0]} samples, {df.shape[1]} features")
        
        # Identify target column (assuming first column is target like in main data)
        target_col = df.columns[0]  # Assuming first column is target
        
        # Extract features and target
        y = df[target_col].values
        X = df.drop(columns=[target_col]).values
        feature_names = df.drop(columns=[target_col]).columns.tolist()
        
        print(f"Target column: {target_col}")
        print(f"Target range: [{np.min(y):.3f}, {np.max(y):.3f}]")
        print(f"Features: {X.shape[1]} wavelengths")
        
        return X, y, feature_names
        
    except Exception as e:
        print("❌ Failed to load test data!")
        print("💡 Please create a test.csv file in the src/data/ directory.")
        print("💡 Format: First column = target values, remaining columns = wavelength features")
        print(f"💡 Error details: {e}")
        raise ValueError(f"test.csv file not found or could not be loaded: {e}")

def test_model(model_class, model_name, model_params, X, y):
    """
    Test a single model's basic functionality with minimal output.
    
    Args:
        model_class: Model class to test
        model_name (str): Name of the model for display
        model_params (dict): Parameters for model initialization
        X (array): Feature matrix
        y (array): Target values
        
    Returns:
        dict: Test results
    """
    try:
        # Initialize, fit, predict, and evaluate silently
        model = model_class(**model_params)
        model.fit(X, y)
        predictions = model.predict(X)
        cv_results = model.evaluate_with_cv(X, y, cv_folds=2)
        
        # Extract basic metrics for return
        from sklearn.metrics import mean_squared_error, r2_score
        rmse = np.sqrt(mean_squared_error(y, predictions))
        r2 = r2_score(y, predictions)
        
        cv_key = f'{model_name}_cv_results'
        cv_r2 = cv_results[cv_key]['r2_mean'] if cv_key in cv_results else None
        
        return {
            'status': 'SUCCESS',
            'training_r2': r2,
            'cv_r2': cv_r2
        }
        
    except Exception as e:
        return {
            'status': 'FAILED',
            'error': str(e)
        }

def run_all_tests():
    """
    Run tests for all available models with minimal output.
    """
    print("🧪 KIWI NIR MODEL TESTING")
    print("=" * 40)
    
    # Load test data silently
    try:
        X, y, feature_names = load_test_data()
    except Exception as e:
        print("❌ Failed to load test data")
        return
    
    # Define models to test
    models_to_test = [
        {
            'class': PLSModel,
            'name': 'PLS_Regression',
            'params': {'n_components': 5, 'max_components': 10}
        },
        {
            'class': RandomForestModel,
            'name': 'Random_Forest',
            'params': {'n_estimators': 50, 'tune_hyperparameters': False}
        },
        {
            'class': SVRModel,
            'name': 'SVR',
            'params': {'kernel': 'rbf', 'tune_hyperparameters': False}
        },
        {
            'class': XGBoostModel,
            'name': 'XGBoost',
            'params': {'n_estimators': 50, 'tune_hyperparameters': False}
        },
        {
            'class': CNNModel,
            'name': '1D_CNN',
            'params': {'epochs': 10, 'batch_size': 16, 'early_stopping_patience': 3}
        }
    ]
    # Run tests
    test_results = {}
    
    for model_config in models_to_test:
        result = test_model(
            model_config['class'],
            model_config['name'],
            model_config['params'],
            X, y
        )
        test_results[model_config['name']] = result
    
    # Show only success/failure status
    successful_tests = 0
    failed_tests = 0
    
    for model_name, result in test_results.items():
        status = result['status']
        if status == 'SUCCESS':
            successful_tests += 1
            print(f"✅ {model_name}")
        else:
            failed_tests += 1
            print(f"❌ {model_name}")
    
    print("=" * 40)
    print(f"✅ Success: {successful_tests} | ❌ Failed: {failed_tests}")
    
    if failed_tests == 0:
        print("🎉 ALL MODELS READY")
    else:
        print("⚠️ SOME MODELS FAILED")

if __name__ == "__main__":
    run_all_tests() 