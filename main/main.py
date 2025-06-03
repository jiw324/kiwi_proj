# AI-Generated Code Header
# **Intent:** Main application to demonstrate kiwi data loading and analysis
# **Optimization:** Efficient data processing with visualization capabilities
# **Safety:** Comprehensive error handling and data validation

import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Import our custom data loader
from loaddata import KiwiDataLoader, load_kiwi_data, get_data_info

# Configure matplotlib for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class KiwiDataAnalyzer:
    """
    Main class for analyzing kiwi dataset with visualization and statistical analysis.
    """
    
    def __init__(self, data_path: str = "../src/data"):
        """
        Initialize the analyzer with data path.
        
        Args:
            data_path (str): Path to the data directory
        """
        self.data_path = data_path
        self.loader = KiwiDataLoader(data_path)
        self.data: Dict[int, pd.DataFrame] = {}
        self.combined_data: Optional[pd.DataFrame] = None
    
    def load_data(self, file_numbers: Optional[List[int]] = None) -> None:
        """
        Load kiwi data files.
        
        Args:
            file_numbers (Optional[List[int]]): Specific file numbers to load
        """
        print("Loading kiwi data files...")
        print("=" * 50)
        
        try:
            self.data = self.loader.load_kiwi_files_by_number(file_numbers)
            
            if not self.data:
                print("No data files were loaded successfully.")
                return
            
            print(f"\nSuccessfully loaded {len(self.data)} files:")
            for num, df in self.data.items():
                print(f"  kiwi-{num}.csv: {df.shape[0]} rows × {df.shape[1]} columns")
                
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            sys.exit(1)
    
    def analyze_data_structure(self) -> None:
        """Analyze and display the structure of loaded data."""
        if not self.data:
            print("No data loaded. Please load data first.")
            return
        
        print("\nData Structure Analysis")
        print("=" * 50)
        
        for num, df in self.data.items():
            print(f"\nkiwi-{num}.csv Analysis:")
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")
            print(f"  Data types: {df.dtypes.to_dict()}")
            print(f"  Missing values: {df.isnull().sum().sum()}")
            
            if 'frequency' in df.columns:
                freq_col = df['frequency']
                print(f"  Frequency range: {freq_col.min():.2f} - {freq_col.max():.2f}")
                print(f"  Frequency points: {len(freq_col)}")
            
            # Show basic statistics for numerical columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:  # More than just frequency
                measurement_cols = [col for col in numeric_cols if col != 'frequency']
                if measurement_cols:
                    print(f"  Measurement columns: {len(measurement_cols)}")
                    print(f"  Value range: {df[measurement_cols].min().min():.4f} - {df[measurement_cols].max().max():.4f}")
    
    def plot_frequency_response(self, file_numbers: Optional[List[int]] = None, 
                              max_measurements: int = 5) -> None:
        """
        Plot frequency response for selected files.
        
        Args:
            file_numbers (Optional[List[int]]): Specific files to plot
            max_measurements (int): Maximum number of measurement columns to plot per file
        """
        if not self.data:
            print("No data loaded for plotting.")
            return
        
        files_to_plot = file_numbers if file_numbers else list(self.data.keys())
        files_to_plot = [f for f in files_to_plot if f in self.data]
        
        if not files_to_plot:
            print("No valid files specified for plotting.")
            return
        
        print(f"\nPlotting frequency response for files: {files_to_plot}")
        
        # Create subplots
        n_files = len(files_to_plot)
        fig, axes = plt.subplots(n_files, 1, figsize=(12, 4 * n_files))
        if n_files == 1:
            axes = [axes]
        
        for idx, file_num in enumerate(files_to_plot):
            df = self.data[file_num]
            ax = axes[idx]
            
            if 'frequency' not in df.columns:
                ax.text(0.5, 0.5, f"No frequency column in kiwi-{file_num}.csv", 
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            # Get measurement columns
            measurement_cols = [col for col in df.columns if col.startswith('measurement_')]
            measurement_cols = measurement_cols[:max_measurements]  # Limit number of lines
            
            # Plot each measurement
            for col in measurement_cols:
                ax.plot(df['frequency'], df[col], label=col, alpha=0.7, linewidth=1)
            
            ax.set_xlabel('Frequency')
            ax.set_ylabel('Measurement Value')
            ax.set_title(f'Frequency Response - kiwi-{file_num}.csv')
            ax.grid(True, alpha=0.3)
            
            if len(measurement_cols) <= 10:  # Only show legend if not too many lines
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show()
    
    def plot_data_overview(self) -> None:
        """Create an overview plot showing all loaded data."""
        if not self.data:
            print("No data loaded for overview plot.")
            return
        
        print("\nCreating data overview plot...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Number of data points per file
        file_nums = list(self.data.keys())
        data_points = [df.shape[0] for df in self.data.values()]
        
        axes[0, 0].bar(file_nums, data_points, color='skyblue', alpha=0.7)
        axes[0, 0].set_xlabel('File Number')
        axes[0, 0].set_ylabel('Number of Data Points')
        axes[0, 0].set_title('Data Points per File')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Number of measurement columns per file
        measurement_counts = []
        for df in self.data.values():
            measurement_cols = [col for col in df.columns if col.startswith('measurement_')]
            measurement_counts.append(len(measurement_cols))
        
        axes[0, 1].bar(file_nums, measurement_counts, color='lightcoral', alpha=0.7)
        axes[0, 1].set_xlabel('File Number')
        axes[0, 1].set_ylabel('Number of Measurements')
        axes[0, 1].set_title('Measurement Columns per File')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Frequency range comparison
        if all('frequency' in df.columns for df in self.data.values()):
            freq_ranges = []
            for num, df in self.data.items():
                freq_min = df['frequency'].min()
                freq_max = df['frequency'].max()
                axes[1, 0].barh(f'kiwi-{num}', freq_max - freq_min, 
                               left=freq_min, alpha=0.7)
                freq_ranges.append((freq_min, freq_max))
            
            axes[1, 0].set_xlabel('Frequency')
            axes[1, 0].set_ylabel('File')
            axes[1, 0].set_title('Frequency Range per File')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'Frequency data not available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
        
        # Plot 4: Value distribution across all files
        all_values = []
        for df in self.data.values():
            measurement_cols = [col for col in df.columns if col.startswith('measurement_')]
            if measurement_cols:
                values = df[measurement_cols].values.flatten()
                all_values.extend(values[~np.isnan(values)])  # Remove NaN values
        
        if all_values:
            axes[1, 1].hist(all_values, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
            axes[1, 1].set_xlabel('Measurement Value')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Distribution of All Measurement Values')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No measurement data available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        plt.show()
    
    def generate_summary_report(self) -> None:
        """Generate a comprehensive summary report of the loaded data."""
        if not self.data:
            print("No data loaded for summary report.")
            return
        
        print("\nKiwi Data Summary Report")
        print("=" * 60)
        
        total_files = len(self.data)
        total_rows = sum(df.shape[0] for df in self.data.values())
        
        print(f"Dataset Overview:")
        print(f"  Total files loaded: {total_files}")
        print(f"  Total data points: {total_rows:,}")
        
        # File-by-file analysis
        print(f"\nDetailed File Analysis:")
        print("-" * 60)
        
        for num in sorted(self.data.keys()):
            df = self.data[num]
            print(f"\nkiwi-{num}.csv:")
            print(f"  Dimensions: {df.shape[0]:,} rows × {df.shape[1]} columns")
            
            if 'frequency' in df.columns:
                freq_col = df['frequency']
                print(f"  Frequency range: {freq_col.min():.3f} to {freq_col.max():.3f}")
                print(f"  Frequency step: ~{(freq_col.max() - freq_col.min()) / len(freq_col):.6f}")
            
            measurement_cols = [col for col in df.columns if col.startswith('measurement_')]
            if measurement_cols:
                print(f"  Measurement columns: {len(measurement_cols)}")
                
                # Calculate statistics for measurements
                measurement_data = df[measurement_cols]
                print(f"  Value statistics:")
                print(f"    Min: {measurement_data.min().min():.6f}")
                print(f"    Max: {measurement_data.max().max():.6f}")
                print(f"    Mean: {measurement_data.mean().mean():.6f}")
                print(f"    Std: {measurement_data.std().mean():.6f}")
                
                # Check for missing values
                missing_count = measurement_data.isnull().sum().sum()
                if missing_count > 0:
                    print(f"    Missing values: {missing_count}")
        
        # Overall statistics
        all_measurement_data = []
        for df in self.data.values():
            measurement_cols = [col for col in df.columns if col.startswith('measurement_')]
            if measurement_cols:
                all_measurement_data.append(df[measurement_cols])
        
        if all_measurement_data:
            combined_measurements = pd.concat(all_measurement_data, ignore_index=True)
            print(f"\nOverall Dataset Statistics:")
            print("-" * 30)
            print(f"  Total measurement points: {combined_measurements.size:,}")
            print(f"  Global min value: {combined_measurements.min().min():.6f}")
            print(f"  Global max value: {combined_measurements.max().max():.6f}")
            print(f"  Global mean: {combined_measurements.mean().mean():.6f}")
            print(f"  Global std: {combined_measurements.std().mean():.6f}")


def main():
    """Main function to demonstrate the kiwi data analysis capabilities."""
    print("Kiwi Data Analysis Tool")
    print("=" * 60)
    
    # Initialize analyzer
    try:
        analyzer = KiwiDataAnalyzer()
    except Exception as e:
        print(f"Error initializing analyzer: {str(e)}")
        return
    
    # Show available files
    print("Available data files:")
    get_data_info()
    
    # Load data (first 3 files as example)
    print("\nLoading first 3 kiwi files...")
    analyzer.load_data(file_numbers=[1, 2, 3])
    
    # Analyze data structure
    analyzer.analyze_data_structure()
    
    # Generate summary report
    analyzer.generate_summary_report()
    
    # Create visualizations
    print("\nGenerating visualizations...")
    try:
        analyzer.plot_data_overview()
        analyzer.plot_frequency_response(file_numbers=[1, 2], max_measurements=3)
    except Exception as e:
        print(f"Error creating plots: {str(e)}")
        print("Note: Plotting requires matplotlib. Install with: pip install matplotlib seaborn")
    
    print("\nAnalysis complete!")
    print("\nTo use this tool programmatically:")
    print("  from main import KiwiDataAnalyzer")
    print("  analyzer = KiwiDataAnalyzer()")
    print("  analyzer.load_data([1, 2, 3, 4, 5, 6])")
    print("  analyzer.generate_summary_report()")


if __name__ == "__main__":
    main() 