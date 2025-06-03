# AI-Generated Code Header
# **Intent:** Data loading utilities for kiwi CSV files
# **Optimization:** Memory-efficient pandas operations with error handling
# **Safety:** Input validation, file existence checks, and graceful error handling

import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple

class KiwiDataLoader:
    """
    A class to handle loading and processing of kiwi dataset CSV files.
    """
    
    def __init__(self, data_path: str = "../src/data"):
        """
        Initialize the data loader with the specified data directory.
        
        Args:
            data_path (str): Path to the data directory containing CSV files
        """
        self.data_path = Path(data_path)
        self._validate_data_path()
    
    def _validate_data_path(self) -> None:
        """Validate that the data path exists and contains CSV files."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_path}")
        
        csv_files = list(self.data_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in: {self.data_path}")
        
        print(f"Found {len(csv_files)} CSV files in {self.data_path}")
    
    def get_available_files(self) -> List[str]:
        """
        Get a list of all available CSV files in the data directory.
        
        Returns:
            List[str]: List of CSV filenames
        """
        csv_files = list(self.data_path.glob("*.csv"))
        return sorted([f.name for f in csv_files])
    
    def load_single_file(self, filename: str) -> pd.DataFrame:
        """
        Load a single CSV file into a pandas DataFrame.
        
        Args:
            filename (str): Name of the CSV file to load
            
        Returns:
            pd.DataFrame: Loaded data with proper column names
        """
        file_path = self.data_path / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            # Load CSV data
            df = pd.read_csv(file_path)
            
            # Set proper column names
            if df.shape[1] > 1:
                columns = ['frequency'] + [f'measurement_{i}' for i in range(1, df.shape[1])]
                df.columns = columns
            
            print(f"Loaded {filename}: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
            
        except Exception as e:
            raise RuntimeError(f"Error loading {filename}: {str(e)}")
    
    def load_all_files(self) -> Dict[str, pd.DataFrame]:
        """
        Load all CSV files in the data directory.
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping filenames to DataFrames
        """
        all_data = {}
        csv_files = self.get_available_files()
        
        for filename in csv_files:
            try:
                all_data[filename] = self.load_single_file(filename)
            except Exception as e:
                print(f"Warning: Failed to load {filename}: {str(e)}")
                continue
        
        print(f"Successfully loaded {len(all_data)} files")
        return all_data
    
    def load_kiwi_files_by_number(self, numbers: Optional[List[int]] = None) -> Dict[int, pd.DataFrame]:
        """
        Load kiwi files by their number (e.g., kiwi-1.csv, kiwi-2.csv).
        
        Args:
            numbers (Optional[List[int]]): Specific file numbers to load. If None, loads all.
            
        Returns:
            Dict[int, pd.DataFrame]: Dictionary mapping file numbers to DataFrames
        """
        kiwi_data = {}
        
        if numbers is None:
            # Find all kiwi-*.csv files and extract numbers
            kiwi_files = list(self.data_path.glob("kiwi-*.csv"))
            numbers = []
            for file in kiwi_files:
                try:
                    num = int(file.stem.split('-')[1])
                    numbers.append(num)
                except (IndexError, ValueError):
                    continue
            numbers = sorted(numbers)
        
        for num in numbers:
            filename = f"kiwi-{num}.csv"
            try:
                kiwi_data[num] = self.load_single_file(filename)
            except Exception as e:
                print(f"Warning: Failed to load {filename}: {str(e)}")
                continue
        
        return kiwi_data
    
    def get_data_summary(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> Dict:
        """
        Get a summary of the loaded data.
        
        Args:
            data: Single DataFrame or dictionary of DataFrames
            
        Returns:
            Dict: Summary statistics and information
        """
        if isinstance(data, pd.DataFrame):
            return {
                'shape': data.shape,
                'columns': list(data.columns),
                'data_types': data.dtypes.to_dict(),
                'missing_values': data.isnull().sum().to_dict(),
                'summary_stats': data.describe().to_dict()
            }
        
        elif isinstance(data, dict):
            summary = {}
            for name, df in data.items():
                summary[name] = {
                    'shape': df.shape,
                    'columns': list(df.columns),
                    'missing_values': df.isnull().sum().sum()
                }
            return summary
        
        else:
            raise ValueError("Data must be a DataFrame or dictionary of DataFrames")
    
    def combine_kiwi_data(self, kiwi_data: Dict[int, pd.DataFrame], 
                         method: str = 'concat') -> pd.DataFrame:
        """
        Combine multiple kiwi datasets.
        
        Args:
            kiwi_data (Dict[int, pd.DataFrame]): Dictionary of kiwi datasets
            method (str): Method to combine ('concat' or 'merge')
            
        Returns:
            pd.DataFrame: Combined dataset
        """
        if not kiwi_data:
            raise ValueError("No data provided to combine")
        
        if method == 'concat':
            # Add source identifier
            for num, df in kiwi_data.items():
                df = df.copy()
                df['source_file'] = f'kiwi-{num}'
                kiwi_data[num] = df
            
            combined = pd.concat(kiwi_data.values(), ignore_index=True)
            
        elif method == 'merge':
            # Merge on frequency column (assuming first column is frequency)
            combined = None
            for num, df in kiwi_data.items():
                df_copy = df.copy()
                # Rename measurement columns to include source
                measurement_cols = [col for col in df_copy.columns if col.startswith('measurement_')]
                rename_dict = {col: f"{col}_kiwi{num}" for col in measurement_cols}
                df_copy = df_copy.rename(columns=rename_dict)
                
                if combined is None:
                    combined = df_copy
                else:
                    combined = pd.merge(combined, df_copy, on='frequency', how='outer')
        
        else:
            raise ValueError("Method must be 'concat' or 'merge'")
        
        print(f"Combined data shape: {combined.shape}")
        return combined


def load_kiwi_data(data_path: str = "../src/data", 
                  file_numbers: Optional[List[int]] = None) -> Dict[int, pd.DataFrame]:
    """
    Convenience function to quickly load kiwi data files.
    
    Args:
        data_path (str): Path to data directory
        file_numbers (Optional[List[int]]): Specific file numbers to load
        
    Returns:
        Dict[int, pd.DataFrame]: Dictionary mapping file numbers to DataFrames
    """
    loader = KiwiDataLoader(data_path)
    return loader.load_kiwi_files_by_number(file_numbers)


def get_data_info(data_path: str = "../src/data") -> None:
    """
    Print information about available data files.
    
    Args:
        data_path (str): Path to data directory
    """
    try:
        loader = KiwiDataLoader(data_path)
        files = loader.get_available_files()
        
        print("Available data files:")
        print("-" * 40)
        for file in files:
            try:
                df = loader.load_single_file(file)
                print(f"{file}: {df.shape[0]} rows × {df.shape[1]} columns")
            except Exception as e:
                print(f"{file}: Error loading - {str(e)}")
    
    except Exception as e:
        print(f"Error accessing data directory: {str(e)}")


if __name__ == "__main__":
    # Example usage
    print("Kiwi Data Loader")
    print("=" * 50)
    
    try:
        # Show available files
        get_data_info()
        
        # Load specific kiwi files
        print("\nLoading kiwi-1.csv through kiwi-3.csv...")
        kiwi_data = load_kiwi_data(file_numbers=[1, 2, 3])
        
        # Show summary
        loader = KiwiDataLoader()
        summary = loader.get_data_summary(kiwi_data)
        print("\nData Summary:")
        for num, info in summary.items():
            print(f"  kiwi-{num}.csv: {info['shape']} - {info['missing_values']} missing values")
    
    except Exception as e:
        print(f"Error: {str(e)}") 