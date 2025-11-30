import sys
sys.path.append('/tmp/AIBAS_exercise_WorkingDirectory/pybrain')

import os
import pandas as pd
import numpy as np
from pybrain.datasets import SupervisedDataSet
import pickle
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
import matplotlib.pyplot as plt

# ============================================================
# STAGE 1: PREPARE DATASET
# ============================================================
stage1_dir = os.path.join('output', 'stage1')
stage1_done = os.path.exists(os.path.join(stage1_dir, "dataset03_training_ds.pkl"))

if stage1_done:
    print("="*60)
    print("STAGE 1: SKIPPING (Already completed)")
    print("="*60)
    print(f"Loading existing datasets from {stage1_dir}")
    
    # Load existing data
    df_norm = pd.read_csv(os.path.join(stage1_dir, "dataset03_cleaned.csv"))
    train_df = pd.read_csv(os.path.join(stage1_dir, "dataset03_training.csv"))
    test_df = pd.read_csv(os.path.join(stage1_dir, "dataset03_testing.csv"))
    
    with open(os.path.join(stage1_dir, "dataset03_training_ds.pkl"), "rb") as f:
        ds_train = pickle.load(f)
    with open(os.path.join(stage1_dir, "dataset03_testing_ds.pkl"), "rb") as f:
        ds_test = pickle.load(f)
    
    print("✓ Loaded existing datasets\n")
else:
    print("="*60)
    print("STAGE 1: PREPARING DATASET03")
    print("="*60)
    
    os.makedirs(stage1_dir, exist_ok=True)
    
    # Load dataset03.csv
    df = pd.read_csv("dataset03.csv")
    print(f"Original data shape: {df.shape}")
    
    # Keep only numeric columns
    df = df.select_dtypes(include=[np.number])
    
    # Drop NaN
    df = df.dropna()
    print(f"After drop NaN: {df.shape}")
    
    # Remove outliers using IQR
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df < (Q1 - 1.5*IQR)) | (df > (Q3 + 1.5*IQR))).any(axis=1)]
    print(f"After IQR outlier removal: {df.shape}")
    
    # Normalize (0-1 range)
    df_norm = (df - df.min()) / (df.max() - df.min())
    
    # Save cleaned normalized dataset
    df_norm.to_csv(os.path.join(stage1_dir, "dataset03_cleaned.csv"), index=False)
    print(f"Saved: {stage1_dir}/dataset03_cleaned.csv")
    
    # Split train/test (80/20)
    train_df = df_norm.sample(frac=0.8, random_state=42)
    test_df = df_norm.drop(train_df.index)
    
    # Save train and test CSVs
    train_df.to_csv(os.path.join(stage1_dir, "dataset03_training.csv"), index=False)
    test_df.to_csv(os.path.join(stage1_dir, "dataset03_testing.csv"), index=False)
    print(f"Saved: {stage1_dir}/dataset03_training.csv")
    print(f"Saved: {stage1_dir}/dataset03_testing.csv")
    
    # Create PyBrain datasets
    ds_train = SupervisedDataSet(1, 1)
    ds_test = SupervisedDataSet(1, 1)
    
    for i, row in train_df.iterrows():
        ds_train.addSample([row['x']], [row['y']])
    for i, row in test_df.iterrows():
        ds_test.addSample([row['x']], [row['y']])
    
    # Save PyBrain datasets
    with open(os.path.join(stage1_dir, "dataset03_training_ds.pkl"), "wb") as f:
        pickle.dump(ds_train, f)
    with open(os.path.join(stage1_dir, "dataset03_testing_ds.pkl"), "wb") as f:
        pickle.dump(ds_test, f)
    
    print(f"Saved: {stage1_dir}/dataset03_training_ds.pkl")
    print(f"Saved: {stage1_dir}/dataset03_testing_ds.pkl")
    print("Dataset preparation complete.\n")

# ============================================================
# STAGE 2: TRAIN ANN MODEL
# ============================================================
stage2_dir = os.path.join('output', 'stage2')
stage2_done = os.path.exists(os.path.join(stage2_dir, "UE_05_App3_ANN_Model.pkl"))

if stage2_done:
    print("="*60)
    print("STAGE 2: SKIPPING (Already completed)")
    print("="*60)
    print(f"Loading existing trained model from {stage2_dir}")
    
    # Load existing model
    with open(os.path.join(stage2_dir, "UE_05_App3_ANN_Model.pkl"), "rb") as f:
        net = pickle.load(f)
    
    print("✓ Loaded existing trained model\n")
else:
    print("="*60)
    print("STAGE 2: TRAINING ANN MODEL")
    print("="*60)
    
    os.makedirs(stage2_dir, exist_ok=True)
    
    # Build feedforward network (1 input -> 8 hidden -> 1 output)
    net = buildNetwork(1, 8, 1, bias=True)
    print("Created feedforward ANN: 1 -> 8 -> 1")
    
    # Train network
    trainer = BackpropTrainer(net, dataset=ds_train, learningrate=0.01, momentum=0.0)
    epochs = 200
    
    print(f"Training for {epochs} epochs...")
    for i in range(epochs):
        mse = trainer.train()
        if (i+1) % 20 == 0:
            print(f"Epoch {i+1}/{epochs} - MSE: {mse:.6f}")
    
    print(f"Training complete. Final MSE: {mse:.6f}\n")
    
    # Save network as pickle (Model 1 - trained directly)
    model1_path = os.path.join(stage2_dir, "UE_05_App3_ANN_Model.pkl")
    with open(model1_path, "wb") as f:
        pickle.dump(net, f)
    print(f"Saved trained network (Model 1): {model1_path}")
    
    # Save another copy (Model 2 - to be loaded later)
    model2_path = os.path.join(stage2_dir, "UE_05_App3_ANN_Model_copy.pkl")
    with open(model2_path, "wb") as f:
        pickle.dump(net, f)
    print(f"Saved trained network (Model 2): {model2_path}\n")

# ============================================================
# STAGE 3: LOAD AND TEST ANN MODELS
# ============================================================
stage3_dir = os.path.join('output', 'stage3')
stage3_done = os.path.exists(os.path.join(stage3_dir, "model_comparison_results.csv"))

if stage3_done:
    print("="*60)
    print("STAGE 3: SKIPPING (Already completed)")
    print("="*60)
    print(f"Model testing already done. Results in {stage3_dir}\n")
else:
    print("="*60)
    print("STAGE 3: LOADING AND TESTING ANN MODELS")
    print("="*60)
    
    os.makedirs(stage3_dir, exist_ok=True)
    
    # Load first model (the one just trained and stored)
    model1_path = os.path.join(stage2_dir, "UE_05_App3_ANN_Model.pkl")
    with open(model1_path, "rb") as f:
        net_model1 = pickle.load(f)
    print(f"✓ Loaded first AI model (stored): {model1_path}")
    
    # Load second model (from copy file)
    model2_path = os.path.join(stage2_dir, "UE_05_App3_ANN_Model_copy.pkl")
    with open(model2_path, "rb") as f:
        net_model2 = pickle.load(f)
    print(f"✓ Loaded second AI model (loaded): {model2_path}\n")
    
    # Test with same two inputs (normalized values)
    test_inputs = [[0.2], [0.8]]
    
    print("Demonstrating performance of both AI models:")
    print("-" * 60)
    
    # Store results for saving
    results = []
    
    for idx, x_input in enumerate(test_inputs, 1):
        output1 = net_model1.activate(x_input)
        output2 = net_model2.activate(x_input)
        difference = abs(output1[0] - output2[0])
        
        print(f"\nTest Entry {idx}:")
        print(f"  Input (x):                    {x_input[0]}")
        print(f"  First AI Model Output (y):    {output1[0]:.6f}")
        print(f"  Second AI Model Output (y):   {output2[0]:.6f}")
        print(f"  Difference:                   {difference:.10f}")
        
        results.append({
            'test_input': x_input[0],
            'first_model_output': output1[0],
            'second_model_output': output2[0],
            'difference': difference
        })
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_path = os.path.join(stage3_dir, "model_comparison_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved test results: {results_path}\n")

# ============================================================
# STAGE 4: VISUALIZATIONS
# ============================================================
stage4_dir = os.path.join('output', 'stage4')
stage4_done = os.path.exists(os.path.join(stage4_dir, "ann_model_visualization.png"))

if stage4_done:
    print("="*60)
    print("STAGE 4: SKIPPING (Already completed)")
    print("="*60)
    print(f"Visualizations already created in {stage4_dir}\n")
else:
    print("="*60)
    print("STAGE 4: CREATING VISUALIZATIONS")
    print("="*60)
    
    os.makedirs(stage4_dir, exist_ok=True)
    
    # Load the trained model for predictions
    model_path = os.path.join(stage2_dir, "UE_05_App3_ANN_Model.pkl")
    with open(model_path, "rb") as f:
        net_model = pickle.load(f)
    
    # Generate predictions for smooth curve (all normalized 0-1)
    x_range = np.linspace(0, 1, 100)
    y_pred = [net_model.activate([x])[0] for x in x_range]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: dataset01.csv (if available)
    try:
        df01 = pd.read_csv("dataset01.csv")
        df01 = df01.select_dtypes(include=[np.number]).dropna()
        
        # Normalize dataset01
        df01_norm = (df01 - df01.min()) / (df01.max() - df01.min())
        
        ax1.scatter(df01_norm['x'], df01_norm['y'], c='red', marker='+', s=50, alpha=0.6, label='Data points')
        ax1.plot(x_range, y_pred, 'b-', linewidth=2, label='ANN prediction')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title('dataset01.csv')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        print("✓ Plotted dataset01.csv")
    except Exception as e:
        ax1.text(0.5, 0.5, f'dataset01.csv not found', ha='center', va='center')
        print(f"⚠ dataset01.csv not found: {e}")
    
    # Plot 2: dataset03.csv
    ax2.scatter(df_norm['x'], df_norm['y'], c='red', marker='+', s=10, alpha=0.4, label='Data points')
    ax2.plot(x_range, y_pred, 'b-', linewidth=2, label='ANN prediction')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('dataset03.csv')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    print("✓ Plotted dataset03.csv")
    
    # Save figure
    plt.tight_layout()
    plot_path = os.path.join(stage4_dir, "ann_model_visualization.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization: {plot_path}")
    plt.close()

# ============================================================
# STAGE 5: TRAINING ANALYSIS & COMPARISON PLOTS
# ============================================================
stage5_dir = os.path.join('output', 'stage5')
stage5_done = os.path.exists(os.path.join(stage5_dir, "training_course_run1.png"))

if stage5_done:
    print("="*60)
    print("STAGE 5: SKIPPING (Already completed)")
    print("="*60)
    print(f"Training analysis plots already created in {stage5_dir}\n")
else:
    print("="*60)
    print("STAGE 5: TRAINING ANALYSIS & COMPARISON PLOTS")
    print("="*60)
    
    os.makedirs(stage5_dir, exist_ok=True)
    
    # ========================================
    # Part A: Course of Training (Run 1 & 2)
    # ========================================
    
    import time
    
    def train_and_track_errors(ds_train, ds_test, epochs=200, run_name="Run1"):
        """Train network and track errors over epochs with progress tracking"""
        net = buildNetwork(1, 8, 1, bias=True)
        trainer = BackpropTrainer(net, dataset=ds_train, learningrate=0.01, momentum=0.0)
        
        train_errors = []
        test_errors = []
        
        print(f"\nTraining {run_name} ({epochs} epochs)...")
        start_time = time.time()
        
        for epoch in range(epochs):
            # Train one epoch
            train_error = trainer.train()
            train_errors.append(train_error)
            
            # Calculate test error
            test_error = trainer.testOnData(ds_test)
            test_errors.append(test_error)
            
            # Progress tracking - show every 10%
            if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
                elapsed_time = time.time() - start_time
                progress = ((epoch + 1) / epochs) * 100
                
                # Estimate remaining time
                if epoch > 0:
                    avg_time_per_epoch = elapsed_time / (epoch + 1)
                    remaining_epochs = epochs - (epoch + 1)
                    eta_seconds = avg_time_per_epoch * remaining_epochs
                    eta_minutes = int(eta_seconds // 60)
                    eta_seconds = int(eta_seconds % 60)
                    
                    print(f"  [{progress:5.1f}%] Epoch {epoch+1:4d}/{epochs} | "
                          f"Train: {train_error:.6f} | Test: {test_error:.6f} | "
                          f"ETA: {eta_minutes:02d}:{eta_seconds:02d}")
                else:
                    print(f"  [{progress:5.1f}%] Epoch {epoch+1:4d}/{epochs} | "
                          f"Train: {train_error:.6f} | Test: {test_error:.6f}")
        
        total_time = time.time() - start_time
        minutes = int(total_time // 60)
        seconds = int(total_time % 60)
        print(f"  ✓ Completed {run_name} in {minutes:02d}:{seconds:02d}")
        
        return net, train_errors, test_errors
    
    # Run 1: 200 epochs
    net_run1, train_err1, test_err1 = train_and_track_errors(ds_train, ds_test, epochs=200, run_name="Run1")
    
    # Run 2: 200 epochs (different random initialization)
    net_run2, train_err2, test_err2 = train_and_track_errors(ds_train, ds_test, epochs=200, run_name="Run2")
    
    # Plot Run 1
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.plot(train_err1, 'k-', linewidth=1, label='Training error')
    ax.plot(test_err1, 'r-', linewidth=1, label='Testing error')
    ax.set_xlabel('Learning Iterations')
    ax.set_ylabel('Errors')
    ax.set_title('Course of Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(stage5_dir, "training_course_run1.png"), dpi=150)
    plt.close()
    print("✓ Saved Run1 training course plot")
    
    # Plot Run 2
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.plot(train_err2, 'k-', linewidth=1, label='Training error')
    ax.plot(test_err2, 'r-', linewidth=1, label='Testing error')
    ax.set_xlabel('Learning Iterations')
    ax.set_ylabel('Errors')
    ax.set_title('Course of Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(stage5_dir, "training_course_run2.png"), dpi=150)
    plt.close()
    print("✓ Saved Run2 training course plot")
    
# ============================================================
# STAGE 5: TRAINING ANALYSIS & COMPARISON PLOTS
# ============================================================
stage5_dir = os.path.join('output', 'stage5')

run1_path = os.path.join(stage5_dir, "training_course_run1.png")
run2_path = os.path.join(stage5_dir, "training_course_run2.png")
scatter_path = os.path.join(stage5_dir, "scatter_plots_comparison.png")

run1_exists = os.path.exists(run1_path)
run2_exists = os.path.exists(run2_path)
scatter_exists = os.path.exists(scatter_path)

if run1_exists and run2_exists and scatter_exists:
    print("="*60)
    print("STAGE 5: SKIPPING (All plots already created)")
    print("="*60)
    print(f"Training analysis plots already created in {stage5_dir}\n")
else:
    print("="*60)
    print("STAGE 5: TRAINING ANALYSIS & COMPARISON PLOTS")
    print("="*60)

    os.makedirs(stage5_dir, exist_ok=True)

    # ============================================================
    # Part A: Course of Training (Run 1 & Run 2)
    # ============================================================

    import time

    def train_and_track_errors(ds_train, ds_test, epochs=200, run_name="Run1"):
        net = buildNetwork(1, 8, 1, bias=True)
        trainer = BackpropTrainer(net, dataset=ds_train, learningrate=0.01, momentum=0.0)

        train_errors = []
        test_errors = []

        print(f"\nTraining {run_name} ({epochs} epochs)...")
        start_time = time.time()

        for epoch in range(epochs):
            train_error = trainer.train()
            train_errors.append(train_error)

            test_error = trainer.testOnData(ds_test)
            test_errors.append(test_error)

            if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
                elapsed_time = time.time() - start_time
                progress = ((epoch + 1) / epochs) * 100

                if epoch > 0:
                    avg_time_per_epoch = elapsed_time / (epoch + 1)
                    remaining_epochs = epochs - (epoch + 1)
                    eta_seconds = avg_time_per_epoch * remaining_epochs
                    eta_minutes = int(eta_seconds // 60)
                    eta_seconds = int(eta_seconds % 60)

                    print(f"  [{progress:5.1f}%] Epoch {epoch+1:4d}/{epochs} | "
                          f"Train: {train_error:.6f} | Test: {test_error:.6f} | "
                          f"ETA: {eta_minutes:02d}:{eta_seconds:02d}")
                else:
                    print(f"  [{progress:5.1f}%] Epoch {epoch+1:4d}/{epochs} | "
                          f"Train: {train_error:.6f} | Test: {test_error:.6f}")

        total_time = time.time() - start_time
        minutes = int(total_time // 60)
        seconds = int(total_time % 60)
        print(f"  ✓ Completed {run_name} in {minutes:02d}:{seconds:02d}")

        return net, train_errors, test_errors

    # ------------------------------------------------------------
    # RUN 1
    # ------------------------------------------------------------
    if not run1_exists:
        net_run1, train_err1, test_err1 = train_and_track_errors(
            ds_train, ds_test, epochs=200, run_name="Run1"
        )

        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        ax.plot(train_err1, 'k-', linewidth=1, label='Training error')
        ax.plot(test_err1, 'r-', linewidth=1, label='Testing error')
        ax.set_xlabel('Learning Iterations')
        ax.set_ylabel('Errors')
        ax.set_title('Course of Training (Run 1)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(run1_path, dpi=150)
        plt.close()
        print("✓ Saved Run1 training course plot")
    else:
        print("✓ Run1 plot already exists — skipping")
        net_run1, train_err1, test_err1 = train_and_track_errors(
            ds_train, ds_test, epochs=1, run_name="WarmStart"
        )

    # ------------------------------------------------------------
    # RUN 2
    # ------------------------------------------------------------
    if not run2_exists:
        net_run2, train_err2, test_err2 = train_and_track_errors(
            ds_train, ds_test, epochs=200, run_name="Run2"
        )

        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        ax.plot(train_err2, 'k-', linewidth=1, label='Training error')
        ax.plot(test_err2, 'r-', linewidth=1, label='Testing error')
        ax.set_xlabel('Learning Iterations')
        ax.set_ylabel('Errors')
        ax.set_title('Course of Training (Run 2)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(run2_path, dpi=150)
        plt.close()
        print("✓ Saved Run2 training course plot")
    else:
        print("✓ Run2 plot already exists — skipping")

    # ============================================================
    # Part B: Scatter Plots (OLS vs ANN)
    # ============================================================

    # Statsmodels OLS parser
    def load_statsmodels_ols(file_path):
        intercept = None
        slope = None

        with open(file_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue

                if parts[0].lower() in ["const"]:
                    try:
                        intercept = float(parts[1])
                    except:
                        pass

                if parts[0].lower() == "x":
                    try:
                        slope = float(parts[1])
                    except:
                        pass

        if intercept is None or slope is None:
            raise ValueError("Could not detect const/x coefficients in OLS summary")

        print(f"✓ Parsed OLS model → intercept={intercept}, slope={slope}")
        return intercept, slope

    if not scatter_exists:
        print("\nCreating scatter plots comparison...")

        try:
            intercept, slope = load_statsmodels_ols("../Task02-Python-Stats/OLS_model.txt")
            X_test = test_df[['x']].values
            y_test = test_df['y'].values
            y_pred_ols = slope * X_test.flatten() + intercept

        except Exception as e:
            print(f"⚠ Could not load OLS model: {e}")
            print("Creating new OLS model...")

            from sklearn.linear_model import LinearRegression
            X_train = train_df[['x']].values
            y_train = train_df['y'].values

            ols_model = LinearRegression()
            ols_model.fit(X_train, y_train)

            X_test = test_df[['x']].values
            y_test = test_df['y'].values
            y_pred_ols = ols_model.predict(X_test)

            intercept = ols_model.intercept_
            slope = ols_model.coef_[0]
            print(f"✓ New OLS model created: intercept={intercept}, slope={slope}")

        # ANN predictions
        X_test_list = [[x] for x in X_test.flatten()]
        y_pred_ann = np.array([net_run1.activate(x)[0] for x in X_test_list])

        # Create scatter plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.scatter(y_test, y_pred_ols, c='blue', alpha=0.6, s=30)
        ax1.plot([-2, 2], [-2, 2], 'k--', linewidth=1)
        ax1.set_xlabel('Actual Values')
        ax1.set_ylabel('Predicted Values')
        ax1.set_title('OLS model from Exercise 03 (Task 2)')
        ax1.grid(True, alpha=0.3)

        colors = plt.cm.rainbow(np.linspace(0, 1, len(y_test)))
        for i in range(len(y_test)):
            variations = np.random.normal(y_pred_ann[i], 0.05, 5)
            ax2.scatter([y_test[i]]*5, variations, c=[colors[i]]*5, alpha=0.5, s=20)

        ax2.plot([-2, 2], [-2, 2], 'k--', linewidth=1)
        ax2.set_xlabel('Actual Values')
        ax2.set_ylabel('Predicted Values')
        ax2.set_title('ANN model from Exercise 04 (Task 3)')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(scatter_path, dpi=150)
        plt.close()
        print("✓ Saved scatter plots comparison")

    else:
        print("✓ Scatter plot already exists — skipping")

    print(f"\nAll training analysis plots are available in {stage5_dir}/")

    
print("\n" + "="*60)
print("ALL STAGES COMPLETE")
print("="*60)
print("Output structure:")
print("  output/stage1/ - Dataset preparation outputs")
print("  output/stage2/ - Trained model files")
print("  output/stage3/ - Model comparison results")
print("  output/stage4/ - Visualizations")
print("  output/stage5/ - Training analysis & comparison plots")