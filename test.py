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

# Import all model classes
from model.pls_model import PLSModel
from model.random_forest_model import RandomForestModel
from model.xgboost_model import XGBoostModel
from model.svr_model import SVRModel
from model.cnn_model import CNNModel

def load_test_data():
    """
    Load test data from test.csv file for quick testing.
    
    Returns:
        tuple: (X, y, feature_names) for testing
    """
    print("📥 Loading test data from test.csv...")
    
    try:
        # Load the test data from test.csv
        df = pd.read_csv("test.csv")
        
        if df.empty:
            raise ValueError("test.csv file is empty.")
        
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
        
    except FileNotFoundError:
        print("❌ test.csv file not found!")
        print("💡 Please create a test.csv file in the project root directory.")
        print("💡 Format: First column = target values, remaining columns = wavelength features")
        raise ValueError("test.csv file not found. Please create this file for testing.")
        
    except Exception as e:
        print(f"❌ Error loading test.csv: {e}")
        raise

def test_model(model_class, model_name, model_params, X, y):
    """
    Test a single model's basic functionality.
    
    Args:
        model_class: Model class to test
        model_name (str): Name of the model for display
        model_params (dict): Parameters for model initialization
        X (array): Feature matrix
        y (array): Target values
        
    Returns:
        dict: Test results
    """
    print(f"\n{'='*50}")
    print(f"🧪 TESTING: {model_name}")
    print(f"{'='*50}")
    
    start_time = time.time()
    
    try:
        # 1. Initialize model
        print("1️⃣ Initializing model...")
        model = model_class(**model_params)
        print(f"   ✅ Model initialized: {model.__class__.__name__}")
        
        # 2. Test fitting
        print("2️⃣ Testing model fitting...")
        fit_start = time.time()
        model.fit(X, y)
        fit_time = time.time() - fit_start
        print(f"   ✅ Model fitted in {fit_time:.2f} seconds")
        
        # 3. Test prediction
        print("3️⃣ Testing predictions...")
        predictions = model.predict(X)
        print(f"   ✅ Predictions generated: {len(predictions)} values")
        print(f"   📊 Prediction range: [{np.min(predictions):.3f}, {np.max(predictions):.3f}]")
        
        # 4. Basic performance metrics
        print("4️⃣ Calculating basic metrics...")
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        rmse = np.sqrt(mean_squared_error(y, predictions))
        mae = mean_absolute_error(y, predictions)
        r2 = r2_score(y, predictions)
        
        print(f"   📈 Training Performance:")
        print(f"      RMSE: {rmse:.4f}")
        print(f"      MAE:  {mae:.4f}")
        print(f"      R²:   {r2:.4f}")
        
        # 5. Test cross-validation (simplified)
        print("5️⃣ Testing cross-validation...")
        cv_start = time.time()
        cv_results = model.evaluate_with_cv(X, y, cv_folds=3)  # Quick 3-fold CV
        cv_time = time.time() - cv_start
        print(f"   ✅ Cross-validation completed in {cv_time:.2f} seconds")
        
        # Extract CV results
        cv_key = f'{model_name}_cv_results'
        if cv_key in cv_results:
            cv_data = cv_results[cv_key]
            print(f"   📊 CV Performance:")
            print(f"      RMSE: {cv_data['rmse_mean']:.4f} ± {cv_data['rmse_std']:.4f}")
            print(f"      MAE:  {cv_data['mae_mean']:.4f} ± {cv_data['mae_std']:.4f}")
            print(f"      R²:   {cv_data['r2_mean']:.4f} ± {cv_data['r2_std']:.4f}")
        
        # 6. Test model info (if available)
        print("6️⃣ Testing model info...")
        if hasattr(model, 'get_model_info'):
            model_info = model.get_model_info()
            print(f"   ✅ Model info retrieved: {len(model_info)} parameters")
        else:
            print("   ⚠️ No model info method available")
        
        total_time = time.time() - start_time
        
        result = {
            'status': 'SUCCESS',
            'training_rmse': rmse,
            'training_mae': mae,
            'training_r2': r2,
            'cv_rmse': cv_data['rmse_mean'] if cv_key in cv_results else None,
            'cv_r2': cv_data['r2_mean'] if cv_key in cv_results else None,
            'fit_time': fit_time,
            'total_time': total_time,
            'predictions_count': len(predictions)
        }
        
        print(f"\n✅ {model_name} TEST PASSED")
        print(f"   Total test time: {total_time:.2f} seconds")
        
        return result
        
    except Exception as e:
        total_time = time.time() - start_time
        print(f"\n❌ {model_name} TEST FAILED")
        print(f"   Error: {str(e)}")
        print(f"   Test time: {total_time:.2f} seconds")
        
        return {
            'status': 'FAILED',
            'error': str(e),
            'total_time': total_time
        }

def run_all_tests():
    """
    Run tests for all available models.
    """
    print("🧪 KIWI NIR MODEL TESTING SUITE")
    print("=" * 60)
    print(f"Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load test data
    try:
        X, y, feature_names = load_test_data()
    except Exception as e:
        print(f"❌ Failed to load test data: {e}")
        return
    
    # Define models to test with simplified parameters
    models_to_test = [
        {
            'class': PLSModel,
            'name': 'PLS_Regression',
            'params': {'n_components': 5, 'max_components': 10}  # Reduced for faster testing
        },
        {
            'class': RandomForestModel,
            'name': 'Random_Forest',
            'params': {'n_estimators': 50, 'tune_hyperparameters': False}  # Faster config
        },
        {
            'class': SVRModel,
            'name': 'SVR',
            'params': {'kernel': 'rbf', 'tune_hyperparameters': False}
        }
    ]
    
    # Add optional models with error handling
    try:
        models_to_test.append({
            'class': XGBoostModel,
            'name': 'XGBoost',
            'params': {'n_estimators': 50, 'tune_hyperparameters': False}
        })
    except ImportError:
        print("⚠️ XGBoost not available - skipping XGBoost test")
    
    try:
        models_to_test.append({
            'class': CNNModel,
            'name': '1D_CNN',
            'params': {'epochs': 10, 'batch_size': 16, 'early_stopping_patience': 3}  # Quick training
        })
    except ImportError:
        print("⚠️ TensorFlow not available - skipping CNN test")
    
    # Run tests
    test_results = {}
    total_start_time = time.time()
    
    for model_config in models_to_test:
        result = test_model(
            model_config['class'],
            model_config['name'],
            model_config['params'],
            X, y
        )
        test_results[model_config['name']] = result
    
    # Summary
    total_time = time.time() - total_start_time
    
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print(f"{'='*60}")
    
    successful_tests = 0
    failed_tests = 0
    
    for model_name, result in test_results.items():
        status = result['status']
        if status == 'SUCCESS':
            successful_tests += 1
            print(f"✅ {model_name:15} | R²: {result['training_r2']:.4f} | Time: {result['total_time']:.1f}s")
        else:
            failed_tests += 1
            print(f"❌ {model_name:15} | FAILED: {result['error']}")
    
    print(f"\n📈 RESULTS:")
    print(f"   Successful tests: {successful_tests}")
    print(f"   Failed tests: {failed_tests}")
    print(f"   Total testing time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    print(f"   Test completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if failed_tests == 0:
        print("\n🎉 ALL TESTS PASSED! Models are ready for full evaluation.")
    else:
        print(f"\n⚠️ {failed_tests} test(s) failed. Check the errors above.")

if __name__ == "__main__":
    run_all_tests() 