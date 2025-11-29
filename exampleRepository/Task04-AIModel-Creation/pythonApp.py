import sys
sys.path.append('/tmp/AIBAS_exercise_WorkingDirectory/pybrain')

import os
import pandas as pd
import numpy as np
from pybrain.datasets import SupervisedDataSet
import pickle
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

# ============================================================
# STAGE 1: PREPARE DATASET
# ============================================================
print("="*60)
print("STAGE 1: PREPARING DATASET03")
print("="*60)

# Create stage1 directory
stage1_dir = os.path.join('output', 'stage1')
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
print("="*60)
print("STAGE 2: TRAINING ANN MODEL")
print("="*60)

# Create stage2 directory
stage2_dir = os.path.join('output', 'stage2')
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
print("="*60)
print("STAGE 3: LOADING AND TESTING ANN MODELS")
print("="*60)

# Create stage3 directory
stage3_dir = os.path.join('output', 'stage3')
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
print(f"\nSaved test results: {results_path}")

print("\n" + "="*60)
print("ALL STAGES COMPLETE")
print("="*60)
print("Output structure:")
print("  output/stage1/ - Dataset preparation outputs")
print("  output/stage2/ - Trained model files")
print("  output/stage3/ - Model comparison results")