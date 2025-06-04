# language: py
# AI-Generated Code Header
# **Intent:** [Main coordinator script that loads NIR spectroscopic data and runs comprehensive evaluation of all ML models (PLS, Random Forest, XGBoost, SVR, 1D CNN) with cross-validation and detailed reporting.]
# **Optimization:** [Efficiently coordinates multiple models with parallel evaluation where possible, provides comprehensive comparison metrics, and handles various model types with consistent interface.]
# **Safety:** [Includes comprehensive error handling for each model type, graceful handling of missing dependencies, and robust data validation before model training.]

import pandas as pd
import numpy as np
import json
import time
import logging
import os
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

def setup_logging():
    """
    Set up logging configuration for the kiwi NIR model evaluation.
    Creates both file and console logging with timestamps.
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/kiwi_nir_evaluation_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, mode='w'),
            logging.StreamHandler()  # Also log to console
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("="*80)
    logger.info("🥝 KIWI NIR SPECTROSCOPIC DATA - ML MODEL EVALUATION LOG")
    logger.info("="*80)
    logger.info(f"Log file: {log_filename}")
    logger.info(f"Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return logger

def log_model_results(logger, model_name, results, training_time):
    """
    Log detailed results for a specific model.
    
    Args:
        logger: Logger instance
        model_name (str): Name of the model
        results (dict): Model evaluation results
        training_time (float): Training time in seconds
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"MODEL EVALUATION RESULTS: {model_name}")
    logger.info(f"{'='*60}")
    logger.info(f"Training Time: {training_time:.2f} seconds")
    
    if 'error' in str(results).lower():
        logger.error(f"❌ {model_name} FAILED: {results}")
        return
    
    # Extract CV results
    cv_key = f'{model_name}_cv_results'
    overall_key = f'{model_name}_overall'
    
    if cv_key in results and overall_key in results:
        cv_results = results[cv_key]
        overall_results = results[overall_key]
        
        logger.info("📊 CROSS-VALIDATION RESULTS:")
        logger.info(f"  • RMSE: {cv_results['rmse_mean']:.4f} ± {cv_results['rmse_std']:.4f}")
        logger.info(f"  • MAE:  {cv_results['mae_mean']:.4f} ± {cv_results['mae_std']:.4f}")
        logger.info(f"  • R²:   {cv_results['r2_mean']:.4f} ± {cv_results['r2_std']:.4f}")
        logger.info(f"  • CV Folds: {cv_results['cv_folds']}")
        logger.info(f"  • Samples: {cv_results['n_samples']}")
        
        logger.info("📈 OVERALL PERFORMANCE:")
        logger.info(f"  • Overall RMSE: {overall_results['rmse']:.4f}")
        logger.info(f"  • Overall MAE:  {overall_results['mae']:.4f}")
        logger.info(f"  • Overall R²:   {overall_results['r2']:.4f}")
        
        # Add model-specific information if available
        if 'model_info' in results:
            logger.info("🔧 MODEL CONFIGURATION:")
            model_info = results['model_info']
            for key, value in model_info.items():
                if not isinstance(value, (list, dict)):
                    logger.info(f"  • {key}: {value}")
        
        logger.info(f"✅ {model_name} evaluation completed successfully")
    else:
        logger.warning(f"⚠️ Incomplete results for {model_name}")

def log_final_comparison(logger, comparison_summary, total_time):
    """
    Log the final model comparison and session summary.
    
    Args:
        logger: Logger instance
        comparison_summary (dict): Model comparison results
        total_time (float): Total execution time
    """
    logger.info(f"\n{'='*80}")
    logger.info("🏆 FINAL MODEL COMPARISON & RANKING")
    logger.info(f"{'='*80}")
    
    if 'error' in comparison_summary:
        logger.error(f"❌ No successful model evaluations: {comparison_summary}")
        return
    
    if 'comparison_table' in comparison_summary:
        logger.info("📋 MODEL PERFORMANCE RANKING (by R² score):")
        logger.info("-" * 80)
        
        for i, model_data in enumerate(comparison_summary['comparison_table'], 1):
            logger.info(f"{i}. {model_data['model']:15}")
            logger.info(f"   R²:   {model_data['r2_mean']:.4f} ± {model_data['r2_std']:.4f}")
            logger.info(f"   RMSE: {model_data['rmse_mean']:.4f} ± {model_data['rmse_std']:.4f}")
            logger.info(f"   MAE:  {model_data['mae_mean']:.4f} ± {model_data['mae_std']:.4f}")
            logger.info(f"   Time: {model_data['training_time']:.1f}s")
            logger.info("")
        
        # Log best model
        best_model = comparison_summary.get('best_model', 'Unknown')
        best_r2 = comparison_summary.get('best_r2', 0)
        best_rmse = comparison_summary.get('best_rmse', 0)
        
        logger.info(f"🏆 BEST PERFORMING MODEL: {best_model}")
        logger.info(f"   Best R² Score: {best_r2:.4f}")
        logger.info(f"   Best RMSE: {best_rmse:.4f}")
    
    # Session summary
    logger.info(f"\n{'='*80}")
    logger.info("📊 SESSION SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Total Execution Time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    logger.info(f"Session Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)

def prepare_data():
    """
    Load and prepare the NIR spectroscopic data.
    
    Returns:
        tuple: (X, y, feature_names) where X is features, y is target, feature_names are wavelength names
    """
    print("=== Loading Kiwi NIR Data ===")
    
    # Load the data
    df = load_kiwi_data(data_directory="src/data")
    
    if df.empty:
        raise ValueError("No data loaded. Please check your data files.")
    
    print(f"Data loaded successfully: {df.shape[0]} samples, {df.shape[1]} features")
    
    # AI-SUGGESTION: Display DataFrame head to inspect the data structure
    # print("\n=== DataFrame Head (First 5 rows) ===")
    # print(df.head())
    # print(f"\nColumns: {list(df.columns)}")
    
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
    Main function to run all model evaluations with comprehensive logging.
    """
    # Initialize logging
    logger = setup_logging()
    session_start_time = time.time()
    
    logger.info("🥝 Kiwi NIR Spectroscopic Data - ML Model Evaluation")
    logger.info("=" * 60)
    
    try:
        # 1. Load and prepare data
        logger.info("📥 LOADING AND PREPARING DATA")
        data_start_time = time.time()
        
        X, y, feature_names = prepare_data()
        logger.info(f"Data shape - X: {X.shape}, y: {y.shape}")
        
        data_load_time = time.time() - data_start_time
        logger.info(f"✅ Data loading completed in {data_load_time:.2f} seconds")
        logger.info(f"📊 Dataset Summary:")
        logger.info(f"  • Samples: {X.shape[0]}")
        logger.info(f"  • Features (wavelengths): {X.shape[1]}")
        logger.info(f"  • Target range: [{np.min(y):.3f}, {np.max(y):.3f}]")
        logger.info(f"  • Target mean ± std: {np.mean(y):.3f} ± {np.std(y):.3f}")
        
        # 2. Initialize all models
        logger.info(f"\n🔧 INITIALIZING MODELS")
        models = {
            'PLS_Regression': PLSModel(n_components=10, max_components=20),
            'Random_Forest': RandomForestModel(n_estimators=100, tune_hyperparameters=True),
            'SVR': SVRModel(kernel='rbf', tune_hyperparameters=True),
            'XGBoost': XGBoostModel(n_estimators=100, tune_hyperparameters=True),
            '1D_CNN': CNNModel(epochs=50, batch_size=32, early_stopping_patience=10)
        }
        
        logger.info(f"📋 Models to evaluate: {list(models.keys())}")
        
        # 3. Run evaluations for all models
        logger.info(f"\n🚀 STARTING MODEL EVALUATIONS")
        all_results = {}
        cv_folds = 10
        logger.info(f"Cross-validation setup: {cv_folds}-fold CV")
        
        for model_name, model in models.items():
            logger.info(f"\n🔄 Starting evaluation for: {model_name}")
            model_start_time = time.time()
            
            results = run_model_evaluation(model, model_name, X, y, cv_folds)
            model_end_time = time.time()
            model_training_time = model_end_time - model_start_time
            
            all_results[model_name] = results
            
            # Log detailed results for this model
            log_model_results(logger, model_name, results, model_training_time)
        
        # 4. Compare all models
        logger.info(f"\n📈 COMPARING ALL MODELS")
        comparison_summary = compare_models(all_results)
        
        # 5. Save results
        logger.info(f"\n💾 SAVING RESULTS")
        save_results(all_results, comparison_summary)
        
        # Calculate total time and log final summary
        total_time = time.time() - session_start_time
        log_final_comparison(logger, comparison_summary, total_time)
        
        logger.info("✅ Evaluation completed successfully!")
        logger.info("📈 Check the generated JSON file for detailed results.")
        
    except Exception as e:
        logger.error(f"\n❌ Error in main execution: {e}")
        logger.error("Full traceback:")
        import traceback
        logger.error(traceback.format_exc())
        
        # Log session end even if there was an error
        total_time = time.time() - session_start_time
        logger.info(f"\n⏱️ Session ended after {total_time:.2f} seconds due to error")

if __name__ == "__main__":
    main()
