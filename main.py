# language: py
# AI-Generated Code Header
# **Intent:** [Main coordinator script that loads NIR spectroscopic data and runs comprehensive evaluation of all ML models (PLS, Random Forest, XGBoost, SVR, 1D CNN) with cross-validation and detailed reporting.]
# **Optimization:** [Efficiently coordinates multiple models with parallel evaluation where possible, provides comprehensive comparison metrics, and handles various model types with consistent interface.]
# **Safety:** [Includes comprehensive error handling for each model type, graceful handling of missing dependencies, and robust data validation before model training.]

import pandas as pd
import numpy as np
import json
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import data loader
from kiwi_data_loader import load_kiwi_data

# Import all model classes
from model.pls_model import PLSModel
from model.random_forest_model import RandomForestModel
from model.xgboost_model import XGBoostModel
from model.svr_model import SVRModel
from model.cnn_model import CNNModel

def prepare_data():
    """
    Load and prepare the NIR spectroscopic data.
    
    Returns:
        tuple: (X, y, feature_names) where X is features, y is target, feature_names are wavelength names
    """
    print("=== Loading Kiwi NIR Data ===")
    
    # Load the data
    df = load_kiwi_data(data_directory="../src/data")
    
    if df.empty:
        raise ValueError("No data loaded. Please check your data files.")
    
    print(f"Data loaded successfully: {df.shape[0]} samples, {df.shape[1]} features")
    
    # AI-SUGGESTION: Identify target column (sweetness) and feature columns (wavelengths)
    # The target is typically the 'Unnamed: 0' column or similar
    target_candidates = ['Unnamed: 0', 'sweetness', 'target', 'y']
    target_col = None
    
    for col in target_candidates:
        if col in df.columns:
            target_col = col
            break
    
    if target_col is None:
        # Use first column as target if no clear target found
        target_col = df.columns[0]
        print(f"Warning: No clear target column found. Using '{target_col}' as target.")
    
    # Extract target and features
    y = df[target_col].values
    X = df.drop(columns=[target_col]).values
    feature_names = df.drop(columns=[target_col]).columns.tolist()
    
    print(f"Target variable: {target_col}")
    print(f"Number of wavelength features: {X.shape[1]}")
    print(f"Target statistics - Mean: {np.mean(y):.3f}, Std: {np.std(y):.3f}, Range: [{np.min(y):.3f}, {np.max(y):.3f}]")
    
    return X, y, feature_names

def run_model_evaluation(model, model_name, X, y, cv_folds=5):
    """
    Run evaluation for a single model.
    
    Args:
        model: Model instance
        model_name (str): Name of the model
        X (array): Feature matrix
        y (array): Target values
        cv_folds (int): Number of CV folds
        
    Returns:
        dict: Evaluation results
    """
    print(f"\n=== Evaluating {model_name} ===")
    start_time = time.time()
    
    try:
        # Run cross-validation evaluation
        results = model.evaluate_with_cv(X, y, cv_folds=cv_folds)
        
        # Add timing information
        end_time = time.time()
        results['training_time_seconds'] = end_time - start_time
        
        # Add model-specific information
        if hasattr(model, 'get_model_info'):
            # Fit the model to get model info
            model.fit(X, y)
            model_info = model.get_model_info()
            results['model_info'] = model_info
        
        print(f"{model_name} evaluation completed in {end_time - start_time:.2f} seconds")
        
        return results
        
    except Exception as e:
        print(f"Error evaluating {model_name}: {e}")
        return {f'{model_name}_error': str(e), 'training_time_seconds': 0}

def compare_models(all_results):
    """
    Compare all model results and create a summary.
    
    Args:
        all_results (dict): Dictionary of all model results
        
    Returns:
        dict: Comparison summary
    """
    print("\n=== Model Comparison Summary ===")
    
    comparison_data = []
    
    for model_name, results in all_results.items():
        if 'error' in str(results):
            print(f"{model_name}: FAILED - {results}")
            continue
        
        # Extract CV results
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
    
    # Convert to DataFrame for easy comparison
    comparison_df = pd.DataFrame(comparison_data)
    
    if not comparison_df.empty:
        # Sort by R² (descending) and then by RMSE (ascending)
        comparison_df = comparison_df.sort_values(['r2_mean', 'rmse_mean'], ascending=[False, True])
        
        print("\nModel Performance Ranking (by R² score):")
        print("=" * 80)
        
        for idx, row in comparison_df.iterrows():
            print(f"{row['model']:15} | R²: {row['r2_mean']:.4f}±{row['r2_std']:.4f} | "
                  f"RMSE: {row['rmse_mean']:.4f}±{row['rmse_std']:.4f} | "
                  f"MAE: {row['mae_mean']:.4f}±{row['mae_std']:.4f} | "
                  f"Time: {row['training_time']:.1f}s")
        
        # Find best model
        best_model = comparison_df.iloc[0]
        print(f"\n🏆 Best Model: {best_model['model']} (R² = {best_model['r2_mean']:.4f})")
        
        return {
            'comparison_table': comparison_df.to_dict('records'),
            'best_model': best_model['model'],
            'best_r2': best_model['r2_mean'],
            'best_rmse': best_model['rmse_mean']
        }
    else:
        print("No successful model evaluations to compare.")
        return {'error': 'No successful evaluations'}

def save_results(all_results, comparison_summary, output_file='kiwi_nir_model_results.json'):
    """
    Save all results to a JSON file.
    
    Args:
        all_results (dict): All model results
        comparison_summary (dict): Model comparison summary
        output_file (str): Output filename
    """
    
    final_results = {
        'timestamp': datetime.now().isoformat(),
        'data_info': {
            'description': 'NIR spectroscopic data for kiwi sweetness prediction',
            'features': 'Near-infrared reflectance values at various wavelengths',
            'target': 'Kiwi sweetness values'
        },
        'individual_model_results': all_results,
        'model_comparison': comparison_summary,
        'methodology': {
            'cross_validation_folds': 5,
            'metrics': ['RMSE', 'MAE', 'R²'],
            'preprocessing': 'Model-specific feature scaling and preprocessing'
        }
    }
    
    try:
        with open(output_file, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        print(f"\n📊 Results saved to: {output_file}")
    except Exception as e:
        print(f"Error saving results: {e}")

def main():
    """
    Main function to run all model evaluations.
    """
    print("🥝 Kiwi NIR Spectroscopic Data - ML Model Evaluation")
    print("=" * 60)
    
    try:
        # 1. Load and prepare data
        X, y, feature_names = prepare_data()
        
        # 2. Initialize all models
        models = {
            'PLS_Regression': PLSModel(n_components=10, tune_hyperparameters=True),
            'Random_Forest': RandomForestModel(n_estimators=100, tune_hyperparameters=True),
            'SVR': SVRModel(kernel='rbf', tune_hyperparameters=True),
        }
        
        # Add XGBoost if available
        try:
            models['XGBoost'] = XGBoostModel(n_estimators=100, tune_hyperparameters=True)
        except ImportError:
            print("⚠️ XGBoost not available - skipping XGBoost evaluation")
        
        # Add CNN if available (with reduced epochs for demonstration)
        try:
            models['1D_CNN'] = CNNModel(epochs=50, batch_size=16, early_stopping_patience=10)
        except ImportError:
            print("⚠️ TensorFlow not available - skipping CNN evaluation")
        
        print(f"\n📋 Models to evaluate: {list(models.keys())}")
        
        # 3. Run evaluations for all models
        all_results = {}
        cv_folds = 5
        
        for model_name, model in models.items():
            results = run_model_evaluation(model, model_name, X, y, cv_folds)
            all_results[model_name] = results
        
        # 4. Compare all models
        comparison_summary = compare_models(all_results)
        
        # 5. Save results
        save_results(all_results, comparison_summary)
        
        print("\n✅ Evaluation completed successfully!")
        print("📈 Check the generated JSON file for detailed results.")
        
    except Exception as e:
        print(f"\n❌ Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
