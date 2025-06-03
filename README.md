# Kiwi Data Analysis Project

A Python tool for loading and analyzing kiwi dataset CSV files with automatic data combination capabilities.

## Project Structure

```
kiwi_proj/
├── src/
│   └── data/           # Data directory (ignored by git)
│       ├── kiwi-1.csv
│       ├── kiwi-2.csv
│       ├── ...
│       └── kiwi-6.csv
├── main/
│   ├── loaddata.py     # Data loading utilities with auto-combination
│   └── main.py         # Main analysis application
├── requirements.txt    # Python dependencies
├── .gitignore         # Git ignore file (includes src/data)
└── README.md          # This file
```

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure your data files are in the `src/data` directory.

## Quick Start - Combine All Data

The easiest way to combine all your kiwi data into a single DataFrame:

```python
from main.loaddata import load_and_combine_all_kiwi_data

# Load and combine all data files
df = load_and_combine_all_kiwi_data()

# Save to CSV
df = load_and_combine_all_kiwi_data(save_to_csv='combined_data.csv')
```

## Usage Examples

### 1. Basic Data Combination

```python
from main.loaddata import load_and_combine_all_kiwi_data

# Stack all files vertically (default method)
combined_df = load_and_combine_all_kiwi_data(method='concat')

# Merge files on frequency column (join horizontally)
merged_df = load_and_combine_all_kiwi_data(method='merge')

# Load only specific files
specific_df = load_and_combine_all_kiwi_data(file_numbers=[1, 2, 3])
```

### 2. Using the KiwiDataLoader Class

```python
from main.loaddata import KiwiDataLoader

# Initialize loader
loader = KiwiDataLoader("src/data")

# Load and combine all data
combined_df = loader.load_and_combine_all_data()

# Load specific files and merge on frequency
merged_df = loader.load_and_combine_all_data(
    method='merge', 
    file_numbers=[1, 2, 3]
)
```

### 3. Advanced Usage with Analysis

```python
from main.main import KiwiDataAnalyzer

# Initialize analyzer
analyzer = KiwiDataAnalyzer()

# Load specific files
analyzer.load_data(file_numbers=[1, 2, 3])

# Analyze data structure
analyzer.analyze_data_structure()

# Generate summary report
analyzer.generate_summary_report()

# Create visualizations
analyzer.plot_data_overview()
analyzer.plot_frequency_response()
```

## Data Combination Methods

### 1. Concatenation (`method='concat'`)
- **Purpose**: Stack all files vertically
- **Result**: Single DataFrame with all rows from all files
- **Use case**: When you want to analyze all data points together
- **Columns added**: `source_file`, `file_number`

### 2. Merge (`method='merge'`)
- **Purpose**: Join files horizontally on frequency column
- **Result**: Single DataFrame with all measurements for each frequency
- **Use case**: When you want to compare measurements across files at same frequencies
- **Columns**: `frequency`, `measurement_1_file1`, `measurement_1_file2`, etc.

### 3. Aligned Concatenation (`method='aligned_concat'`)
- **Purpose**: Interpolate all data to common frequency grid, then stack
- **Result**: Aligned data on uniform frequency scale
- **Use case**: When files have different frequency ranges/resolutions

## File Structure

### Data Files
The tool expects CSV files in the format:
- First column: frequency values
- Subsequent columns: measurement data
- Files named: `kiwi-1.csv`, `kiwi-2.csv`, etc.

### Output Files
When using `save_to_csv`, the tool creates:
- `all_kiwi_data_combined.csv`: All data concatenated
- `kiwi_123_merged.csv`: Specific files merged on frequency
- Or custom filename you specify

## Key Features

- **Automatic file discovery**: Finds all kiwi-*.csv files
- **Flexible combination methods**: Stack, merge, or align data
- **Source tracking**: Keeps track of which file each data point came from
- **Error handling**: Graceful handling of missing files or corrupt data
- **Memory efficient**: Uses pandas for optimal memory usage
- **Visualization**: Built-in plotting capabilities
- **Export options**: Save combined data to CSV

## Command Line Usage

Run the main analysis:
```bash
cd main
python main.py
```

Run data loading examples:
```bash
cd main
python loaddata.py
```

## API Reference

### Main Functions

#### `load_and_combine_all_kiwi_data()`
Main function for quick data combination.

**Parameters:**
- `data_path` (str): Path to data directory
- `method` (str): 'concat', 'merge', or 'aligned_concat'
- `add_source_info` (bool): Add source file columns
- `file_numbers` (List[int], optional): Specific files to load
- `save_to_csv` (str, optional): Output filename

**Returns:** `pd.DataFrame` - Combined dataset

#### `KiwiDataLoader.load_and_combine_all_data()`
Class method for data combination with more control.

### Data Analysis

#### `KiwiDataAnalyzer`
Full analysis class with visualization capabilities.

**Methods:**
- `load_data()`: Load specific files
- `analyze_data_structure()`: Analyze loaded data
- `generate_summary_report()`: Create detailed report
- `plot_data_overview()`: Overview visualizations
- `plot_frequency_response()`: Frequency response plots

## Examples Output

After running the tool, you'll see output like:
```
Found 6 CSV files in ../src/data
Loading and combining all kiwi data files...
============================================================
Found 6 kiwi files to combine
Loaded kiwi-1.csv: 142 rows, 150 columns
Loaded kiwi-2.csv: 142 rows, 150 columns
...
Combined data shape: (852, 152)

Combined Data Summary:
  Shape: (852, 152)
  Columns: ['frequency', 'measurement_1', ..., 'source_file', 'file_number']
  Memory usage: 2.45 MB
  Saved to: combined_data.csv
```

## Troubleshooting

1. **FileNotFoundError**: Ensure data files are in `src/data` directory
2. **Import errors**: Install requirements with `pip install -r requirements.txt`
3. **Memory issues**: Use specific `file_numbers` to load subset of data
4. **Visualization errors**: Install matplotlib/seaborn: `pip install matplotlib seaborn`

## Contributing

The project follows the coding standards defined in the user-specific protocol with:
- Proactive validation
- Collaboration-centric output
- Context-aware development
- Consistent formatting
- Continuous improvement focus