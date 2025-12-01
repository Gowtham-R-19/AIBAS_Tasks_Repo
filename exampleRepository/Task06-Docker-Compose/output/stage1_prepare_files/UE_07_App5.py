import sys
sys.path.append('/tmp/AIBAS_exercise_WorkingDirectory/pybrain')

import pickle
import pandas as pd

print("="*60)
print("UE_07_App5: MODEL ACTIVATION")
print("="*60)

# Load model
print("\nLoading model...")
with open('currentSolution.pkl', 'rb') as f:
    model = pickle.load(f)
print(f"✓ Model loaded: {type(model).__name__}")

# Load activation data
print("\nLoading activation data...")
df = pd.read_csv('activation_data.csv')
print(f"✓ Data loaded: {df.shape}")
print(f"  Input: x={df['x'].values[0]:.6f}, y={df['y'].values[0]:.6f}")

# Activate model
print("\nActivating model...")
x_input = [[df['x'].values[0]]]
prediction = model.activate(x_input)
print(f"✓ Prediction: {prediction[0]:.6f}")

print("\n" + "="*60)
print("ACTIVATION COMPLETE")
print("="*60)