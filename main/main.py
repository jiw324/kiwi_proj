# language: py
# AI-Generated Code Header
# **Intent:** [Simplified main coordinator script that loads NIR spectroscopic data and runs ML models with graceful dependency handling. Focuses on core models (PLS, Random Forest, SVR) that are most reliable for spectroscopic data.]
# **Optimization:** [Efficiently handles missing dependencies by only importing and running models that are available. Provides comprehensive evaluation while being robust to installation issues.]
# **Safety:** [Includes comprehensive error handling for missing dependencies, graceful degradation when libraries are unavailable, and robust data validation.]

import pandas as pd
import numpy as np
import json
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import data loader
from kiwi_data_loader import load_kiwi_data

# Try to import models individually with graceful error handling
available_models = {}

# Always available models (using sklearn)
try:
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, KFold
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    SKLEARN_AVAILABLE = True
    print("✅ Scikit-learn available")
except ImportError as e:
    SKLEARN_AVAILABLE = False
    print(f"❌ Scikit-learn not available: {e}")

# Try XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    print("✅ XGBoost available")
except ImportError as e:
    XGBOOST_AVAILABLE = False
    print(f"⚠️ XGBoost not available: {e}")

# Try TensorFlow
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
    print("✅ TensorFlow available")
except ImportError as e:
    TENSORFLOW_AVAILABLE = False
    print(f"⚠️ TensorFlow not available: {e}")

class SimpleModel:
    """Simple wrapper for sklearn models with consistent interface."""
    
    def __init__(self, model, model_name, random_state=42):
        self.model = model
        self.model_name = model_name
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, X, y):
        """Fit the model."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Make predictions."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def evaluate_with_cv(self, X, y, cv_folds=5):
        """Evaluate with cross-validation."""
        try:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            
            fold_scores = {'rmse': [], 'mae': [], 'r2': []}
            all_predictions = []
            all_true_values = []
            
            for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Create fresh model and scaler for this fold
                temp_model = type(self.model)(**self.model.get_params())
                temp_scaler = StandardScaler()
                
                # Fit and predict
                X_train_scaled = temp_scaler.fit_transform(X_train)
                X_val_scaled = temp_scaler.transform(X_val)
                
                temp_model.fit(X_train_scaled, y_train)
                y_pred = temp_model.predict(X_val_scaled)
                
                # Calculate metrics
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                mae = mean_absolute_error(y_val, y_pred)
                r2 = r2_score(y_val, y_pred)
                
                fold_scores['rmse'].append(rmse)
                fold_scores['mae'].append(mae)
                fold_scores['r2'].append(r2)
                
                all_predictions.extend(y_pred)
                all_true_values.extend(y_val)
            
            # Calculate results
            results = {
                f'{self.model_name}_cv_results': {
                    'rmse_mean': np.mean(fold_scores['rmse']),
                    'rmse_std': np.std(fold_scores['rmse']),
                    'mae_mean': np.mean(fold_scores['mae']),
                    'mae_std': np.std(fold_scores['mae']),
                    'r2_mean': np.mean(fold_scores['r2']),
                    'r2_std': np.std(fold_scores['r2']),
                    'cv_folds': cv_folds,
                    'n_samples': len(y)
                },
                f'{self.model_name}_overall': {
                    'rmse': np.sqrt(mean_squared_error(all_true_values, all_predictions)),
                    'mae': mean_absolute_error(all_true_values, all_predictions),
                    'r2': r2_score(all_true_values, all_predictions)
                }
            }
            
            return results
            
        except Exception as e:
            return {f'{self.model_name}_error': str(e)}

def prepare_data():
    """Load and prepare the NIR spectroscopic data."""
    print("=== Loading Kiwi NIR Data ===")
    
    df = load_kiwi_data(data_directory="../src/data")
    
    if df.empty:
        raise ValueError("No data loaded. Please check your data files.")
    
    print(f"Data loaded successfully: {df.shape[0]} samples, {df.shape[1]} features")
    
    # Find target column
    target_candidates = ['Unnamed: 0', 'sweetness', 'target', 'y']
    target_col = None
    
    for col in target_candidates:
        if col in df.columns:
            target_col = col
            break
    
    if target_col is None:
        target_col = df.columns[0]
        print(f"Warning: No clear target column found. Using '{target_col}' as target.")
    
    # Extract features and target
    y = df[target_col].values
    X = df.drop(columns=[target_col]).values
    feature_names = df.drop(columns=[target_col]).columns.tolist()
    
    print(f"Target variable: {target_col}")
    print(f"Number of wavelength features: {X.shape[1]}")
    print(f"Target statistics - Mean: {np.mean(y):.3f}, Std: {np.std(y):.3f}")
    
    return X, y, feature_names

def create_models():
    """Create available models based on installed dependencies."""
    models = {}
    
    if SKLEARN_AVAILABLE:
        # PLS Regression - Best for NIR spectroscopic data
        models['PLS_Regression'] = SimpleModel(
            PLSRegression(n_components=10, scale=False),
            'PLS_Regression'
        )
        
        # Random Forest - Good for non-linear relationships
        models['Random_Forest'] = SimpleModel(
            RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'Random_Forest'
        )
        
        # Support Vector Regression
        models['SVR'] = SimpleModel(
            SVR(kernel='rbf', C=1.0, gamma='scale'),
            'SVR'
        )
    
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = SimpleModel(
            xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'XGBoost'
        )
    
    return models

def run_model_evaluation(model, model_name, X, y, cv_folds=5):
    """Run evaluation for a single model."""
    print(f"\n=== Evaluating {model_name} ===")
    start_time = time.time()
    
    try:
        results = model.evaluate_with_cv(X, y, cv_folds=cv_folds)
        end_time = time.time()
        results['training_time_seconds'] = end_time - start_time
        
        print(f"{model_name} evaluation completed in {end_time - start_time:.2f} seconds")
        return results
        
    except Exception as e:
        print(f"Error evaluating {model_name}: {e}")
        return {f'{model_name}_error': str(e), 'training_time_seconds': 0}

def compare_models(all_results):
    """Compare all model results."""
    print("\n=== Model Comparison Summary ===")
    
    comparison_data = []
    
    for model_name, results in all_results.items():
        if 'error' in str(results):
            print(f"{model_name}: FAILED - {results}")
            continue
        
        cv_key = f'{model_name}_cv_results'
        overall_key = f'{model_name}_overall'
        
        if cv_key in results and overall_key in results:
            cv_results = results[cv_key]
            overall_results = results[overall_key]
            
            model_summary = {
                'model': model_name,
                'rmse_mean': cv_results['rmse_mean'],
                'rmse_std': cv_results['rmse_std'],
                'mae_mean': cv_results['mae_mean'],
                'mae_std': cv_results['mae_std'],
                'r2_mean': cv_results['r2_mean'],
                'r2_std': cv_results['r2_std'],
                'overall_rmse': overall_results['rmse'],
                'overall_mae': overall_results['mae'],
                'overall_r2': overall_results['r2'],
                'training_time': results.get('training_time_seconds', 0)
            }
            
            comparison_data.append(model_summary)
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values(['r2_mean', 'rmse_mean'], ascending=[False, True])
        
        print("\nModel Performance Ranking (by R² score):")
        print("=" * 80)
        
        for idx, row in comparison_df.iterrows():
            print(f"{row['model']:15} | R²: {row['r2_mean']:.4f}±{row['r2_std']:.4f} | "
                  f"RMSE: {row['rmse_mean']:.4f}±{row['rmse_std']:.4f} | "
                  f"MAE: {row['mae_mean']:.4f}±{row['mae_std']:.4f} | "
                  f"Time: {row['training_time']:.1f}s")
        
        best_model = comparison_df.iloc[0]
        print(f"\n🏆 Best Model: {best_model['model']} (R² = {best_model['r2_mean']:.4f})")
        
        return comparison_df.to_dict('records')
    else:
        print("No successful model evaluations to compare.")
        return []

def save_results(all_results, comparison_summary, output_file='kiwi_nir_results.json'):
    """Save results to JSON file."""
    final_results = {
        'timestamp': datetime.now().isoformat(),
        'data_info': {
            'description': 'NIR spectroscopic data for kiwi sweetness prediction',
            'features': 'Near-infrared reflectance values at various wavelengths',
            'target': 'Kiwi sweetness values'
        },
        'available_libraries': {
            'sklearn': SKLEARN_AVAILABLE,
            'xgboost': XGBOOST_AVAILABLE,
            'tensorflow': TENSORFLOW_AVAILABLE
        },
        'individual_model_results': all_results,
        'model_comparison': comparison_summary,
        'methodology': {
            'cross_validation_folds': 5,
            'metrics': ['RMSE', 'MAE', 'R²'],
            'preprocessing': 'StandardScaler for feature scaling'
        }
    }
    
    try:
        with open(output_file, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        print(f"\n📊 Results saved to: {output_file}")
    except Exception as e:
        print(f"Error saving results: {e}")

def main():
    """Main function to run all model evaluations."""
    print("🥝 Kiwi NIR Spectroscopic Data - ML Model Evaluation")
    print("=" * 60)
    
    try:
        # 1. Check dependencies
        if not SKLEARN_AVAILABLE:
            print("❌ Scikit-learn is required but not available. Please install it.")
            return
        
        # 2. Load and prepare data
        X, y, feature_names = prepare_data()
        
        # 3. Create available models
        models = create_models()
        
        if not models:
            print("❌ No models available. Please check your dependencies.")
            return
        
        print(f"\n📋 Models to evaluate: {list(models.keys())}")
        
        # 4. Run evaluations
        all_results = {}
        cv_folds = 5
        
        for model_name, model in models.items():
            results = run_model_evaluation(model, model_name, X, y, cv_folds)
            all_results[model_name] = results
        
        # 5. Compare models
        comparison_summary = compare_models(all_results)
        
        # 6. Save results
        save_results(all_results, comparison_summary)
        
        print("\n✅ Evaluation completed successfully!")
        print("📈 Check the generated JSON file for detailed results.")
        
    except Exception as e:
        print(f"\n❌ Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 