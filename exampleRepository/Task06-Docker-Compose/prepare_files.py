import sys
sys.path.append('/tmp/AIBAS_exercise_WorkingDirectory/pybrain')  # Add PyBrain to path

import os
import pandas as pd
import pickle
import shutil

print("="*60)
print("STAGE 1: PREPARING FILES FOR DOCKER CONTAINERS")
print("="*60)

output_dir = os.path.join('output', 'stage1_prepare_files')
os.makedirs(output_dir, exist_ok=True)

# ============================================================
# 1. GET SINGLE DATA ENTRY FROM TASK 05
# ============================================================
print("\n1. Extracting single data entry from Task 05...")

task05_data_path = "../Task05-Data-Scraping-Cleaning/output/stage1_scraping/UE_06_dataset04_joint_scraped_data.csv"

try:
    # Load Task 05 data
    df = pd.read_csv(task05_data_path)
    print(f"   ✓ Loaded Task 05 data: {df.shape}")
    
    # Get first row as single entry
    single_entry = df.iloc[[0]]  # Keep as DataFrame with 1 row
    
    # Save to currentActivation.csv
    activation_file = os.path.join(output_dir, "currentActivation.csv")
    single_entry.to_csv(activation_file, index=False)
    print(f"   ✓ Saved: {activation_file}")
    print(f"   Data: x={single_entry['x'].values[0]:.6f}, y={single_entry['y'].values[0]:.6f}")
    
except FileNotFoundError:
    print(f"   ⚠ Task 05 data not found, creating sample data...")
    # Create sample data
    single_entry = pd.DataFrame({'x': [0.5], 'y': [0.7]})
    activation_file = os.path.join(output_dir, "currentActivation.csv")
    single_entry.to_csv(activation_file, index=False)
    print(f"   ✓ Created sample: {activation_file}")

# ============================================================
# 2. GET AI MODEL FROM TASK 04
# ============================================================
print("\n2. Copying AI model from Task 04...")

task04_model_path = "../Task04-AIModel-Creation/output/stage2/UE_05_App3_ANN_Model.pkl"

try:
    # Copy model file
    model_file = os.path.join(output_dir, "currentSolution.pkl")
    shutil.copy(task04_model_path, model_file)
    print(f"   ✓ Copied model: {model_file}")
    
    # Verify model can be loaded
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    print(f"   ✓ Model verified (type: {type(model).__name__})")
    
except FileNotFoundError:
    print(f"   ⚠ Task 04 model not found at: {task04_model_path}")
    print(f"   Please ensure Task 04 is completed first!")
    exit(1)

# ============================================================
# 3. CREATE ACTIVATION SCRIPT (UE_07_App5.py)
# ============================================================
print("\n3. Creating activation script (UE_07_App5.py)...")

activation_script = """import sys
sys.path.append('/tmp/AIBAS_exercise_WorkingDirectory/pybrain')

import pickle
import pandas as pd

print("="*60)
print("UE_07_App5: MODEL ACTIVATION")
print("="*60)

# Load model
print("\\nLoading model...")
with open('currentSolution.pkl', 'rb') as f:
    model = pickle.load(f)
print(f"✓ Model loaded: {type(model).__name__}")

# Load activation data
print("\\nLoading activation data...")
df = pd.read_csv('activation_data.csv')
print(f"✓ Data loaded: {df.shape}")
print(f"  Input: x={df['x'].values[0]:.6f}, y={df['y'].values[0]:.6f}")

# Activate model
print("\\nActivating model...")
x_input = [[df['x'].values[0]]]
prediction = model.activate(x_input)
print(f"✓ Prediction: {prediction[0]:.6f}")

print("\\n" + "="*60)
print("ACTIVATION COMPLETE")
print("="*60)
"""

script_file = os.path.join(output_dir, "UE_07_App5.py")
with open(script_file, 'w') as f:
    f.write(activation_script.strip())

print(f"   ✓ Created: {script_file}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*60)
print("STAGE 1 COMPLETE - FILES PREPARED")
print("="*60)
print(f"Output directory: {output_dir}/")
print(f"  - currentActivation.csv  (data entry)")
print(f"  - currentSolution.pkl    (AI model)")
print(f"  - UE_07_App5.py          (activation script)")
print("\nReady for Docker container builds!")