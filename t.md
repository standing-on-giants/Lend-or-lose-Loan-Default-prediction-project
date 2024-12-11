# Tuberculosis Metrics Analysis

This project analyzes the relationships between tuberculosis metrics, financial aid, and BCG vaccination coverage using data visualization and clustering techniques. It processes and visualizes data from multiple sources to understand temporal patterns and correlations.

## Features
- Correlation analysis using heatmaps
- Time series visualization of key metrics
- K-means clustering of temporal data
- Scatter Plot Matrix (SPLOM) visualization
- Automated output generation for analysis results

## Prerequisites

### Required Python Version
- Python 3.7 or higher

### Required Python Packages
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- openpyxl (for reading Excel files)

### Installation
Install all required packages using pip:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
```

## Required Data Files
The program expects the following input files in the same directory:
1. `Financial aid dataset.xlsx`
2. `Bacillus Calmette–Guérin (BCG) vaccination coverage 2024-26-11 15-57 UTC.xlsx`
3. `TB_Burden_Country.csv`

## Output
The program creates an `output` directory containing:
- `initial_correlation_matrix.png`: Overall correlation heatmap
- `correlation_heatmaps.png`: Cluster-wise correlation analysis
- `splom.png`: Scatter Plot Matrix visualization
- `time_series.png`: Temporal analysis plots
- `cluster_analysis.txt`: Detailed clustering results

## Usage
Run the script using Python:
```bash
python IMT2022107.py
```
