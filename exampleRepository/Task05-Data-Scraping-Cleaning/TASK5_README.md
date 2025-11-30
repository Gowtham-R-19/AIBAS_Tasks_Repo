# Task 5: Data Scraping, Cleaning & Model Comparison

## Overview
Complete end-to-end pipeline for web scraping, data cleaning, model training, and quantitative comparison.

## Project Structure
```
output/
├── stage1_scraping/        # Scraped and cleaned data
├── stage2_model_prep/      # Train/test splits
├── stage3_ols_model/       # Linear regression baseline
├── stage4_ann_model/       # Neural network model
├── stage5_comparison/      # Quantitative metrics
└── stage6_visualizations/  # Comparison plots
```

## Key Files
- `App06_task5_scraping.py` - Main application
- `output/stage1_scraping/UE_06_dataset04_joint_scraped_data.csv` - Cleaned data
- `output/stage5_comparison/model_comparison_metrics.csv` - Performance metrics
- `output/stage6_visualizations/*.png` - Analysis plots

## Data Processing Pipeline
1. **Scraping**: Extract data from GitHub README
2. **Cleaning**: NaN removal, outlier detection (IQR), normalization
3. **Splitting**: 80/20 train/test split
4. **Training**: OLS and ANN models on same data
5. **Evaluation**: Quantitative metrics (MSE, RMSE, MAE, R²)
6. **Visualization**: Comparative analysis plots

## Model Comparison Results
- **OLS Model**: Linear baseline (scikit-learn LinearRegression)
- **ANN Model**: Non-linear (PyBrain 1-8-1 feedforward network)

See `output/stage5_comparison/model_comparison_metrics.csv` for detailed results.

## Visualizations
- **model_comparison_plots.png**: 4-subplot comprehensive comparison
- **model_fit_comparison.png**: Fitted curves overlay

## Installation Requirements
```bash
python3 -m pip install pandas numpy scikit-learn requests beautifulsoup4 matplotlib pybrain
```

## Running the Application
```bash
python3 App06_task5_scraping.py
```

## Checkpoint System
Each stage checks for existing outputs and skips re-computation if files exist.
To force re-run of a stage, delete the corresponding `output/stageX/` directory.

## Author
Created as part of AIBAS Exercise 06 - Task 5

## References
- Data Source: https://github.com/MarcusGrum/AIBAS/blob/main/README.md
- Previous Tasks: Task 2 (OLS), Task 3 (OLS Stats), Task 4 (ANN)
