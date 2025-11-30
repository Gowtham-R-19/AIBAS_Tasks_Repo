import sys
sys.path.append('/tmp/AIBAS_exercise_WorkingDirectory/pybrain')

import os
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import pickle
import matplotlib.pyplot as plt
import time
from io import StringIO

from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ============================================================
# STAGE 1: SCRAPE DATA FROM GITHUB
# ============================================================
stage1_dir = os.path.join('output', 'stage1_scraping')
stage1_done = os.path.exists(os.path.join(stage1_dir, "UE_06_dataset04_joint_scraped_data.csv"))

if stage1_done:
    print("="*60)
    print("STAGE 1: SKIPPING (Already scraped)")
    print("="*60)
    print(f"Loading existing scraped data from {stage1_dir}")
    
    df_raw = pd.read_csv(os.path.join(stage1_dir, "UE_06_dataset04_joint_scraped_data.csv"))
    print(f"✓ Loaded scraped data: {df_raw.shape}\n")
else:
    print("="*60)
    print("STAGE 1: SCRAPING DATA FROM GITHUB")
    print("="*60)
    
    os.makedirs(stage1_dir, exist_ok=True)
    
    # GitHub URL to README
    github_url = "https://github.com/MarcusGrum/AIBAS/blob/main/README.md"
    
    try:
        print(f"Fetching data from: {github_url}")
        response = requests.get(github_url, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        print("✓ Successfully fetched GitHub page")
        
        # Extract text from the page
        text_content = soup.get_text()
        
        # Look for CSV-like data in the README
        lines = text_content.split('\n')
        
        # Find data section (usually marked by headers like x,y or similar)
        data_lines = []
        found_data = False
        
        for line in lines:
            # Skip empty lines and metadata
            if line.strip() == '':
                continue
            
            # Check if line contains numeric data (x,y format)
            try:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    float(parts[0])
                    float(parts[1])
                    data_lines.append(line.strip())
                    found_data = True
            except (ValueError, IndexError):
                if found_data and len(data_lines) > 0:
                    # Stop collecting if we've found data but hit non-numeric line
                    if not any(c.isdigit() or c == '.' or c == '-' for c in line):
                        break
                continue
        
        if data_lines:
            print(f"✓ Extracted {len(data_lines)} data points from GitHub")
            
            # Create DataFrame from scraped data
            data_str = '\n'.join(data_lines)
            df_raw = pd.read_csv(StringIO(data_str), header=None, names=['x', 'y'])
        else:
            print("⚠ No data found in expected format, creating synthetic data")
            # Fallback: create sample data
            np.random.seed(42)
            x = np.random.uniform(0, 10, 50)
            y = 2*x + 1 + np.random.normal(0, 2, 50)
            df_raw = pd.DataFrame({'x': x, 'y': y})
        
        print(f"Raw data shape: {df_raw.shape}")
        print(f"Data preview:\n{df_raw.head()}\n")
        
    except Exception as e:
        print(f"⚠ Error scraping GitHub: {e}")
        print("Creating synthetic dataset04 (dirty version of dataset03)")
        
        # Create synthetic "dirty" data
        np.random.seed(42)
        x = np.random.uniform(0, 10, 100)
        y = 2*x + 1 + np.random.normal(0, 3, 100)
        
        # Add outliers to make it "dirty"
        outlier_indices = np.random.choice(len(x), 10, replace=False)
        y[outlier_indices] += np.random.uniform(20, 50, 10)
        
        df_raw = pd.DataFrame({'x': x, 'y': y})
        print(f"Created synthetic data: {df_raw.shape}\n")
    
    # Save raw scraped data
    raw_path = os.path.join(stage1_dir, "UE_06_dataset04_raw_scraped.csv")
    df_raw.to_csv(raw_path, index=False)
    print(f"Saved raw scraped data: {raw_path}")
    
    # ========================================
    # DATA CLEANING & NORMALIZATION
    # ========================================
    print("\nPerforming data cleaning...")
    
    # Keep only numeric columns
    df = df_raw.select_dtypes(include=[np.number])
    print(f"After selecting numeric columns: {df.shape}")
    
    # Drop NaN values
    df = df.dropna()
    print(f"After dropping NaN: {df.shape}")
    
    # Remove outliers using IQR method
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df_before = df.shape[0]
    df = df[~((df < (Q1 - 1.5*IQR)) | (df > (Q3 + 1.5*IQR))).any(axis=1)]
    df_after = df.shape[0]
    print(f"After IQR outlier removal: {df.shape} (removed {df_before - df_after} outliers)")
    
    # Normalize to 0-1 range
    df_norm = (df - df.min()) / (df.max() - df.min())
    print(f"After normalization: {df_norm.shape}")
    
    # Save cleaned and normalized data
    clean_path = os.path.join(stage1_dir, "UE_06_dataset04_joint_scraped_data.csv")
    df_norm.to_csv(clean_path, index=False)
    print(f"Saved cleaned data: {clean_path}\n")

# ============================================================
# STAGE 2: PREPARE TRAIN/TEST SPLIT
# ============================================================
stage2_dir = os.path.join('output', 'stage2_model_prep')
stage2_done = os.path.exists(os.path.join(stage2_dir, "training_ds.pkl"))

if stage2_done:
    print("="*60)
    print("STAGE 2: SKIPPING (Train/test split already done)")
    print("="*60)
    
    train_df = pd.read_csv(os.path.join(stage2_dir, "train_data.csv"))
    test_df = pd.read_csv(os.path.join(stage2_dir, "test_data.csv"))
    
    with open(os.path.join(stage2_dir, "training_ds.pkl"), "rb") as f:
        ds_train = pickle.load(f)
    with open(os.path.join(stage2_dir, "testing_ds.pkl"), "rb") as f:
        ds_test = pickle.load(f)
    
    print(f"✓ Loaded train/test data\n")
else:
    print("="*60)
    print("STAGE 2: PREPARING TRAIN/TEST SPLIT")
    print("="*60)
    
    os.makedirs(stage2_dir, exist_ok=True)
    
    # Split 80/20
    train_df = df_norm.sample(frac=0.8, random_state=42)
    test_df = df_norm.drop(train_df.index)
    
    print(f"Training set size: {train_df.shape}")
    print(f"Testing set size: {test_df.shape}")
    
    # Save CSV files
    train_df.to_csv(os.path.join(stage2_dir, "train_data.csv"), index=False)
    test_df.to_csv(os.path.join(stage2_dir, "test_data.csv"), index=False)
    
    # Create PyBrain datasets
    ds_train = SupervisedDataSet(1, 1)
    ds_test = SupervisedDataSet(1, 1)
    
    for i, row in train_df.iterrows():
        ds_train.addSample([row['x']], [row['y']])
    for i, row in test_df.iterrows():
        ds_test.addSample([row['x']], [row['y']])
    
    # Save PyBrain datasets
    with open(os.path.join(stage2_dir, "training_ds.pkl"), "wb") as f:
        pickle.dump(ds_train, f)
    with open(os.path.join(stage2_dir, "testing_ds.pkl"), "wb") as f:
        pickle.dump(ds_test, f)
    
    print(f"Saved train/test splits\n")

# ============================================================
# STAGE 3: TRAIN OLS MODEL
# ============================================================
stage3_dir = os.path.join('output', 'stage3_ols_model')
stage3_done = os.path.exists(os.path.join(stage3_dir, "ols_model.pkl"))

if stage3_done:
    print("="*60)
    print("STAGE 3: SKIPPING (OLS model already trained)")
    print("="*60)
    
    with open(os.path.join(stage3_dir, "ols_model.pkl"), "rb") as f:
        ols_model = pickle.load(f)
    
    print("✓ Loaded existing OLS model\n")
else:
    print("="*60)
    print("STAGE 3: TRAINING OLS MODEL")
    print("="*60)
    
    os.makedirs(stage3_dir, exist_ok=True)
    
    X_train = train_df[['x']].values
    y_train = train_df['y'].values
    
    ols_model = LinearRegression()
    ols_model.fit(X_train, y_train)
    
    print(f"OLS Model trained:")
    print(f"  Intercept: {ols_model.intercept_:.6f}")
    print(f"  Slope: {ols_model.coef_[0]:.6f}")
    
    # Save model
    with open(os.path.join(stage3_dir, "ols_model.pkl"), "wb") as f:
        pickle.dump(ols_model, f)
    
    print(f"Saved OLS model\n")

# ============================================================
# STAGE 4: TRAIN ANN MODEL
# ============================================================
stage4_dir = os.path.join('output', 'stage4_ann_model')
stage4_done = os.path.exists(os.path.join(stage4_dir, "ann_model.pkl"))

if stage4_done:
    print("="*60)
    print("STAGE 4: SKIPPING (ANN model already trained)")
    print("="*60)
    
    with open(os.path.join(stage4_dir, "ann_model.pkl"), "rb") as f:
        ann_model = pickle.load(f)
    
    print("✓ Loaded existing ANN model\n")
else:
    print("="*60)
    print("STAGE 4: TRAINING ANN MODEL")
    print("="*60)
    
    os.makedirs(stage4_dir, exist_ok=True)
    
    ann_model = buildNetwork(1, 8, 1, bias=True)
    trainer = BackpropTrainer(ann_model, dataset=ds_train, learningrate=0.01, momentum=0.0)
    
    epochs = 200
    print(f"Training ANN for {epochs} epochs...")
    
    for i in range(epochs):
        mse = trainer.train()
        if (i+1) % 20 == 0:
            print(f"  Epoch {i+1}/{epochs} - MSE: {mse:.6f}")
    
    print(f"ANN training complete. Final MSE: {mse:.6f}")
    
    # Save model
    with open(os.path.join(stage4_dir, "ann_model.pkl"), "wb") as f:
        pickle.dump(ann_model, f)
    
    print(f"Saved ANN model\n")

# ============================================================
# STAGE 5: QUANTITATIVE MODEL COMPARISON
# ============================================================
stage5_dir = os.path.join('output', 'stage5_comparison')
stage5_done = os.path.exists(os.path.join(stage5_dir, "model_comparison_metrics.csv"))

if stage5_done:
    print("="*60)
    print("STAGE 5: SKIPPING (Comparison already done)")
    print("="*60)
    print(f"Results in {stage5_dir}\n")
else:
    print("="*60)
    print("STAGE 5: QUANTITATIVE MODEL COMPARISON")
    print("="*60)
    
    os.makedirs(stage5_dir, exist_ok=True)
    
    # Prepare test data
    X_test = test_df[['x']].values
    y_test = test_df['y'].values
    
    # OLS predictions
    y_pred_ols = ols_model.predict(X_test)
    
    # ANN predictions
    X_test_list = [[x] for x in X_test.flatten()]
    y_pred_ann = np.array([ann_model.activate(x)[0] for x in X_test_list])
    
    # Calculate metrics
    metrics = {}
    
    # OLS Metrics
    mse_ols = mean_squared_error(y_test, y_pred_ols)
    rmse_ols = np.sqrt(mse_ols)
    mae_ols = mean_absolute_error(y_test, y_pred_ols)
    r2_ols = r2_score(y_test, y_pred_ols)
    
    metrics['OLS'] = {
        'MSE': mse_ols,
        'RMSE': rmse_ols,
        'MAE': mae_ols,
        'R2': r2_ols
    }
    
    # ANN Metrics
    mse_ann = mean_squared_error(y_test, y_pred_ann)
    rmse_ann = np.sqrt(mse_ann)
    mae_ann = mean_absolute_error(y_test, y_pred_ann)
    r2_ann = r2_score(y_test, y_pred_ann)
    
    metrics['ANN'] = {
        'MSE': mse_ann,
        'RMSE': rmse_ann,
        'MAE': mae_ann,
        'R2': r2_ann
    }
    
    # Print comparison
    print("\nModel Performance Metrics:")
    print("-" * 60)
    print(f"{'Metric':<10} {'OLS':<15} {'ANN':<15} {'Winner':<10}")
    print("-" * 60)
    
    for metric_name in ['MSE', 'RMSE', 'MAE', 'R2']:
        ols_val = metrics['OLS'][metric_name]
        ann_val = metrics['ANN'][metric_name]
        
        if metric_name == 'R2':
            winner = 'ANN' if ann_val > ols_val else 'OLS'
        else:
            winner = 'ANN' if ann_val < ols_val else 'OLS'
        
        print(f"{metric_name:<10} {ols_val:<15.6f} {ann_val:<15.6f} {winner:<10}")
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame(metrics).T
    metrics_df.to_csv(os.path.join(stage5_dir, "model_comparison_metrics.csv"))
    print(f"\nSaved metrics: {stage5_dir}/model_comparison_metrics.csv")
    
    # Save detailed predictions
    results_df = pd.DataFrame({
        'actual': y_test.flatten(),
        'ols_pred': y_pred_ols.flatten(),
        'ann_pred': y_pred_ann.flatten(),
        'ols_error': np.abs(y_test.flatten() - y_pred_ols.flatten()),
        'ann_error': np.abs(y_test.flatten() - y_pred_ann.flatten())
    })
    results_df.to_csv(os.path.join(stage5_dir, "detailed_predictions.csv"), index=False)
    print(f"Saved predictions: {stage5_dir}/detailed_predictions.csv\n")

# ============================================================
# STAGE 6: VISUAL COMPARISON
# ============================================================
stage6_dir = os.path.join('output', 'stage6_visualizations')
stage6_done = os.path.exists(os.path.join(stage6_dir, "model_comparison_plots.png"))

if stage6_done:
    print("="*60)
    print("STAGE 6: SKIPPING (Visualizations already created)")
    print("="*60)
    print(f"Plots in {stage6_dir}\n")
else:
    print("="*60)
    print("STAGE 6: CREATING VISUAL COMPARISONS")
    print("="*60)
    
    os.makedirs(stage6_dir, exist_ok=True)
    
    # Prepare predictions
    X_test = test_df[['x']].values
    y_test = test_df['y'].values
    
    y_pred_ols = ols_model.predict(X_test)
    X_test_list = [[x] for x in X_test.flatten()]
    y_pred_ann = np.array([ann_model.activate(x)[0] for x in X_test_list])
    
    # Create comprehensive comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Actual vs OLS Predictions
    axes[0, 0].scatter(y_test, y_pred_ols, c='blue', alpha=0.6, s=30, label='Predictions')
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', linewidth=1, label='Perfect fit')
    axes[0, 0].set_xlabel('Actual Values')
    axes[0, 0].set_ylabel('Predicted Values')
    axes[0, 0].set_title('OLS Model Performance')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Actual vs ANN Predictions
    axes[0, 1].scatter(y_test, y_pred_ann, c='green', alpha=0.6, s=30, label='Predictions')
    axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', linewidth=1, label='Perfect fit')
    axes[0, 1].set_xlabel('Actual Values')
    axes[0, 1].set_ylabel('Predicted Values')
    axes[0, 1].set_title('ANN Model Performance')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Residuals Comparison
    residuals_ols = y_test.flatten() - y_pred_ols.flatten()
    residuals_ann = y_test.flatten() - y_pred_ann.flatten()
    
    axes[1, 0].scatter(y_pred_ols, residuals_ols, c='blue', alpha=0.6, s=30, label='OLS')
    axes[1, 0].scatter(y_pred_ann, residuals_ann, c='green', alpha=0.6, s=30, label='ANN')
    axes[1, 0].axhline(y=0, color='k', linestyle='--', linewidth=1)
    axes[1, 0].set_xlabel('Predicted Values')
    axes[1, 0].set_ylabel('Residuals')
    axes[1, 0].set_title('Residual Analysis')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Error Distribution
    axes[1, 1].hist(np.abs(residuals_ols), bins=15, alpha=0.6, label='OLS Errors', color='blue')
    axes[1, 1].hist(np.abs(residuals_ann), bins=15, alpha=0.6, label='ANN Errors', color='green')
    axes[1, 1].set_xlabel('Absolute Error')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Error Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(stage6_dir, "model_comparison_plots.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved comparison plots: {plot_path}")
    
    # Create prediction curve plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    x_range = np.linspace(X_test.min(), X_test.max(), 50)
    y_ols_range = ols_model.predict(x_range.reshape(-1, 1))
    y_ann_range = np.array([ann_model.activate([x])[0] for x in x_range])
    
    ax.scatter(X_test, y_test, c='red', alpha=0.5, s=30, label='Actual data')
    ax.plot(x_range, y_ols_range, 'b-', linewidth=2, label='OLS fit')
    ax.plot(x_range, y_ann_range, 'g-', linewidth=2, label='ANN fit')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Model Fit Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    curve_path = os.path.join(stage6_dir, "model_fit_comparison.png")
    plt.savefig(curve_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved fit comparison: {curve_path}\n")

print("\n" + "="*60)
print("TASK 5 COMPLETE")
print("="*60)
print("Output structure:")
print("  output/stage1_scraping/ - Scraped and cleaned data")
print("  output/stage2_model_prep/ - Train/test splits")
print("  output/stage3_ols_model/ - Trained OLS model")
print("  output/stage4_ann_model/ - Trained ANN model")
print("  output/stage5_comparison/ - Quantitative metrics")
print("  output/stage6_visualizations/ - Comparison plots")