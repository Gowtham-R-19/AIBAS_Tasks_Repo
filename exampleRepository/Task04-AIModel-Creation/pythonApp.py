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
    
    # Normalize
    df_norm = (df - df.min()) / (df.max() - df.min())
    
    # Save cleaned dataset
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
    
    # Test with same two inputs
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
    
    # Generate predictions for smooth curve
    x_range = np.linspace(0, 1, 100)
    y_pred = [net_model.activate([x])[0] for x in x_range]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: dataset01.csv (if available)
    try:
        df01 = pd.read_csv("../Task02-Python-Stats/dataset01.csv")
        df01 = df01.select_dtypes(include=[np.number]).dropna()
        
        # Normalize dataset01
        df01_norm = (df01 - df01.min()) / (df01.max() - df01.min())
        
        ax1.scatter(df01_norm['x'], df01_norm['y'], c='red', marker='+', s=50, label='Data points')
        ax1.plot(x_range, y_pred, 'b-', linewidth=2, label='ANN prediction')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title('dataset01.csv')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        print("✓ Plotted dataset01.csv")
    except:
        ax1.text(0.5, 0.5, 'dataset01.csv not found', ha='center', va='center')
        print("⚠ dataset01.csv not found")
    
    # Plot 2: dataset03.csv
    ax2.scatter(df_norm['x'], df_norm['y'], c='red', marker='+', s=30, alpha=0.6, label='Data points')
    ax2.plot(x_range, y_pred, 'b-', linewidth=2, label='ANN prediction')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('dataset03.csv')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    print("✓ Plotted dataset03.csv")
    
    # Save figure
    plt.tight_layout()
    plot_path = os.path.join(stage4_dir, "ann_model_visualization.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization: {plot_path}")
    plt.close()

print("\n" + "="*60)
print("ALL STAGES COMPLETE")
print("="*60)
print("Output structure:")
print("  output/stage1/ - Dataset preparation outputs")
print("  output/stage2/ - Trained model files")
print("  output/stage3/ - Model comparison results")
print("  output/stage4/ - Visualizations")