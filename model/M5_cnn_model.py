# language: py
# AI-Generated Code Header
# **Intent:** [1D Convolutional Neural Network model for NIR spectroscopic data to predict kiwi sweetness. 1D CNN can automatically learn relevant spectral patterns and features directly from raw spectral data.]
# **Optimization:** [Uses TensorFlow/Keras with optimized architecture for 1D spectral data, including dropout for regularization and early stopping for optimal training. Designed to handle varying spectral lengths.]
# **Safety:** [Includes comprehensive error handling for TensorFlow operations, handles potential installation issues gracefully, and provides fallback behavior if deep learning libraries are unavailable.]

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from .base_model import BaseNIRModel

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available. Please install with: pip install tensorflow")

class CNNModel(BaseNIRModel):
    """
    1D Convolutional Neural Network model for NIR spectroscopic data.
    
    1D CNN is excellent for spectral data because:
    - Automatically learns relevant local patterns in spectra
    - Can detect characteristic peaks and absorption bands
    - Handles sequential nature of wavelength data
    - Can capture complex non-linear relationships
    - Requires minimal manual feature engineering
    """
    
    def __init__(self, epochs: int = 100, batch_size: int = 32, 
                 learning_rate: float = 0.001, random_state: int = 42,
                 validation_split: float = 0.2, early_stopping_patience: int = 15):
        """
        Initialize 1D CNN model.
        
        Args:
            epochs (int): Maximum number of training epochs
            batch_size (int): Training batch size
            learning_rate (float): Learning rate for Adam optimizer
            random_state (int): Random state for reproducibility
            validation_split (float): Fraction of data for validation
            early_stopping_patience (int): Patience for early stopping
        """
        super().__init__("1D_CNN", random_state)
        
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required but not installed. Please install with: pip install tensorflow")
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.validation_split = validation_split
        self.early_stopping_patience = early_stopping_patience
        self.history = None
        self.input_shape = None
        
        # AI-SUGGESTION: Initialize scaler as None, create when needed
        self.scaler = None
        
        # Set TensorFlow random seeds
        tf.random.set_seed(random_state)
    
    def _create_model(self, input_shape):
        """
        Create 1D CNN model architecture.
        
        Args:
            input_shape (tuple): Shape of input data (sequence_length, 1)
            
        Returns:
            keras.Model: Compiled CNN model
        """
        # AI-SUGGESTION: Architecture designed for 1D spectroscopic data
        # Multiple conv layers with different filter sizes to capture various spectral features
        model = keras.Sequential([
            # First convolutional block
            layers.Conv1D(filters=32, kernel_size=7, activation='relu', 
                         input_shape=input_shape, padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.2),
            
            # Second convolutional block
            layers.Conv1D(filters=64, kernel_size=5, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.3),
            
            # Third convolutional block
            layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.4),
            
            # Global average pooling to reduce overfitting
            layers.GlobalAveragePooling1D(),
            
            # Dense layers for final prediction
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1)  # Single output for regression
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def _preprocess_features(self, X, fit_scaler=False):
        """
        Apply CNN-specific preprocessing to NIR spectral features.
        
        Args:
            X (array-like): Raw spectral data
            fit_scaler (bool): Whether to fit the scaler (True for training, False for prediction)
            
        Returns:
            array: Preprocessed spectral data
        """
        X_processed = np.array(X, dtype=np.float32)
        
        # AI-SUGGESTION: Standardization helps CNN training stability
        if fit_scaler or self.scaler is None:
            self.scaler = StandardScaler()
            X_processed = self.scaler.fit_transform(X_processed)
        else:
            X_processed = self.scaler.transform(X_processed)
        
        # Reshape for 1D CNN: (samples, sequence_length, features)
        # For spectral data, features = 1 (intensity at each wavelength)
        X_processed = X_processed.reshape(X_processed.shape[0], X_processed.shape[1], 1)
        
        return X_processed
    
    def fit(self, X, y, verbose: int = 1):
        """
        Fit the 1D CNN model to training data.
        
        Args:
            X (array-like): Feature matrix (NIR spectra)
            y (array-like): Target values (sweetness)
            verbose (int): Verbosity level for training
            
        Returns:
            self: Fitted model instance
        """
        try:
            X = np.array(X, dtype=np.float32)
            y = np.array(y, dtype=np.float32)
            
            # Preprocess features
            X_processed = self._preprocess_features(X, fit_scaler=True)
            
            # Store input shape for model creation
            self.input_shape = (X_processed.shape[1], 1)
            
            # Create model
            self.model = self._create_model(self.input_shape)
            
            if verbose > 0:
                print("CNN Model Architecture:")
                self.model.summary()
            
            # Setup callbacks
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=self.early_stopping_patience,
                    restore_best_weights=True,
                    verbose=1 if verbose > 0 else 0
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=10,
                    min_lr=1e-7,
                    verbose=1 if verbose > 0 else 0
                )
            ]
            
            # Train model
            self.history = self.model.fit(
                X_processed, y,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=self.validation_split,
                callbacks=callbacks,
                verbose=verbose
            )
            
            self.is_fitted = True
            print("1D CNN model fitted successfully")
            
            return self
            
        except Exception as e:
            print(f"Error during CNN fitting: {e}")
            raise
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X (array-like): Feature matrix
            
        Returns:
            array: Predicted values
        """
        if not self.is_fitted:
            raise ValueError("CNN must be fitted before making predictions")
        
        try:
            X = np.array(X, dtype=np.float32)
            X_processed = self._preprocess_features(X)
            predictions = self.model.predict(X_processed, verbose=0)
            return predictions.flatten()
        except Exception as e:
            print(f"Error during CNN prediction: {e}")
            raise
    
    def evaluate_with_cv(self, X, y, cv_folds: int = 5, scoring_metrics: list = None):
        """
        Evaluate model using manual cross-validation for neural networks.
        
        Args:
            X (array-like): Feature matrix
            y (array-like): Target values
            cv_folds (int): Number of cross-validation folds
            scoring_metrics (list): Not used for CNN (computed automatically)
            
        Returns:
            dict: Dictionary containing evaluation results
        """
        try:
            X = np.array(X, dtype=np.float32)
            y = np.array(y, dtype=np.float32)
            
            from sklearn.model_selection import KFold
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            
            fold_scores = {metric: [] for metric in ['rmse', 'mae', 'r2']}
            all_predictions = []
            all_true_values = []
            
            print(f"Performing {cv_folds}-fold cross-validation for CNN...")
            
            for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
                print(f"Training fold {fold + 1}/{cv_folds}...")
                
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Create and train model for this fold
                temp_cnn = CNNModel(
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    learning_rate=self.learning_rate,
                    random_state=self.random_state + fold,  # Different seed per fold
                    validation_split=0.15,  # Smaller validation split in CV
                    early_stopping_patience=self.early_stopping_patience
                )
                
                temp_cnn.fit(X_train, y_train, verbose=0)
                
                # Predict on validation fold
                y_pred = temp_cnn.predict(X_val)
                
                # Calculate metrics for this fold
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                mae = mean_absolute_error(y_val, y_pred)
                r2 = r2_score(y_val, y_pred)
                
                fold_scores['rmse'].append(rmse)
                fold_scores['mae'].append(mae)
                fold_scores['r2'].append(r2)
                
                all_predictions.extend(y_pred)
                all_true_values.extend(y_val)
                
                print(f"Fold {fold + 1} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
            
            # Calculate overall statistics
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
                }
            }
            
            # Overall cross-validation predictions
            overall_rmse = np.sqrt(mean_squared_error(all_true_values, all_predictions))
            overall_mae = mean_absolute_error(all_true_values, all_predictions)
            overall_r2 = r2_score(all_true_values, all_predictions)
            
            results[f'{self.model_name}_overall'] = {
                'rmse': overall_rmse,
                'mae': overall_mae,
                'r2': overall_r2
            }
            
            return results
            
        except Exception as e:
            print(f"Error during CNN cross-validation: {e}")
            return {f'{self.model_name}_error': str(e)}
    
    def get_model_info(self):
        """
        Get detailed information about the fitted CNN model.
        
        Returns:
            dict: Model information
        """
        if not self.is_fitted:
            return {"error": "Model not fitted"}
        
        info = {
            "model_type": "1D Convolutional Neural Network",
            "input_shape": self.input_shape,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "total_parameters": self.model.count_params(),
            "trainable_parameters": sum([np.prod(v.get_shape()) for v in self.model.trainable_variables])
        }
        
        # Add training history information
        if self.history is not None:
            final_epoch = len(self.history.history['loss'])
            info["epochs_trained"] = final_epoch
            info["final_training_loss"] = float(self.history.history['loss'][-1])
            info["final_validation_loss"] = float(self.history.history['val_loss'][-1])
            info["best_validation_loss"] = float(min(self.history.history['val_loss']))
        
        return info
    
    def plot_training_history(self):
        """
        Plot training history if matplotlib is available.
        
        Returns:
            matplotlib.figure.Figure or None: Training history plot
        """
        if not self.is_fitted or self.history is None:
            print("Model not fitted or no training history available")
            return None
        
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Plot loss
            ax1.plot(self.history.history['loss'], label='Training Loss')
            ax1.plot(self.history.history['val_loss'], label='Validation Loss')
            ax1.set_title('Model Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            
            # Plot MAE
            ax2.plot(self.history.history['mae'], label='Training MAE')
            ax2.plot(self.history.history['val_mae'], label='Validation MAE')
            ax2.set_title('Model MAE')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Mean Absolute Error')
            ax2.legend()
            
            plt.tight_layout()
            return fig
            
        except ImportError:
            print("Matplotlib not available for plotting")
            return None 