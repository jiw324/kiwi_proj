# language: py
# AI-Generated Code Header
# **Intent:** [To provide a robust function for loading and consolidating kiwi fruit sensor data from multiple specified CSV files (kiwi-1.csv, kiwi-2.csv, kiwi-3.csv) located in a 'data' subdirectory into a single pandas DataFrame. A main function demonstrates this capability.]
# **Optimization:** [Prioritizes clarity and standard pandas practices for data loading and concatenation. File paths are constructed dynamically. Error handling for common file issues is included.]
# **Safety:** [Incorporates try-except blocks to gracefully handle FileNotFoundError and empty data errors during CSV loading, preventing script crashes and providing informative messages. Assumes consistent CSV structure across files for concatenation.]

import pandas as pd
import os

def load_specific_file(file_path):
    """
    Load a specific CSV file.
    
    Args:
        file_path (str): Path to the specific CSV file to load
        
    Returns:
        pandas.DataFrame: DataFrame containing the loaded data.
                          Returns an empty DataFrame if file loading fails.
    """
    try:
        print(f"Attempting to load file: {os.path.abspath(file_path)}")
        df = pd.read_csv(file_path)
        print(f"Successfully loaded: {file_path}")
        print(f"Data shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'.")
        return pd.DataFrame()
    except pd.errors.EmptyDataError:
        print(f"Error: File '{file_path}' is empty.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading file '{file_path}': {e}")
        return pd.DataFrame()

def load_kiwi_data(data_directory="data", num_files=3, specific_file=None):
    """
    Loads kiwi data from multiple kiwi-X.csv files or a specific file
    located in the specified data_directory and concatenates them.

    Args:
        data_directory (str): The subdirectory where files are stored.
        num_files (int): The number of kiwi files to load (e.g., 3 for kiwi-1 to kiwi-3).
        specific_file (str): If provided, loads only this specific file from data_directory.

    Returns:
        pandas.DataFrame: A single DataFrame containing all data from the loaded files.
                          Returns an empty DataFrame if no files are successfully loaded or an error occurs.
    """
    # If specific file is requested, load only that file
    if specific_file:
        file_path = os.path.join(data_directory, specific_file)
        return load_specific_file(file_path)
    
    # Otherwise, load multiple kiwi files as before
    all_dataframes = []
    # AI-SUGGESTION: For more dynamic file discovery (e.g., any kiwi-*.csv file), 
    # consider using: import glob; file_paths = glob.glob(os.path.join(data_directory, 'kiwi-*.csv'))
    # instead of explicitly listing file numbers.
    # AI-SUGGESTION: Corrected file name generation to use data_directory effectively.
    file_names = [f"kiwi-{i+1}.csv" for i in range(num_files)]

    print(f"Attempting to load data from directory: {os.path.abspath(data_directory)}")

    for file_name in file_names:
        file_path = os.path.join(data_directory, file_name)
        try:
            # AI-SUGGESTION: Based on typical spectroscopic data, it's assumed the first row is a header 
            # (e.g., wavelengths) and the first column is an identifier or a target variable (e.g., sweetness).
            # pd.read_csv default behavior handles this well if the CSV is structured this way.
            df = pd.read_csv(file_path)
            all_dataframes.append(df)
            print(f"Successfully loaded: {file_path}")
        except FileNotFoundError:
            print(f"Warning: File not found at '{file_path}'. Skipping this file.")
        except pd.errors.EmptyDataError:
            print(f"Warning: File '{file_path}' is empty. Skipping this file.")
        except Exception as e:
            print(f"Error loading file '{file_path}': {e}. Skipping this file.")

    if not all_dataframes:
        print("No data was loaded. Returning an empty DataFrame.")
        return pd.DataFrame()

    try:
        # AI-SUGGESTION: If files might have slightly different but compatible columns, 
        # pd.concat can handle this, but an explicit check for column consistency 
        # might be needed for robust error handling depending on requirements.
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        print("Successfully concatenated all loaded DataFrames.")
        return combined_df
    except Exception as e:
        print(f"Error during concatenation of DataFrames: {e}")
        return pd.DataFrame()

def main():
    """
    Main execution function to demonstrate loading kiwi data.
    """
    print("--- Starting Kiwi Data Loading Demonstration ---")
    
    # AI-SUGGESTION: Updated to reflect the actual location of data files relative to the script's execution path.
    # Load data using the default parameters (data/kiwi-1.csv to data/kiwi-3.csv)
    kiwi_data_df = load_kiwi_data(data_directory="../src/data")

    if not kiwi_data_df.empty:
        print("\n--- Combined Kiwi Data Summary ---")
        print("Head of the DataFrame:")
        print(kiwi_data_df.head())
        print(f"\nShape of DataFrame: {kiwi_data_df.shape} (rows, columns)")
        # AI-SUGGESTION: For a more detailed overview of the DataFrame, uncomment the following:
        # print("\nDataFrame Info:")
        # kiwi_data_df.info()
        # print("\nDescriptive Statistics:")
        # print(kiwi_data_df.describe())
    else:
        print("\n--- No Data Loaded ---")
        print("The combined DataFrame is empty. Please check file paths and contents.")

if __name__ == "__main__":
    main() 