# Stage 1: Dataset Preparation
import sys
import pandas as pd
import numpy as np
from pybrain.datasets import SupervisedDataSet
import pickle

# Include PyBrain path
sys.path.append('/tmp/AIBAS_exercise_WorkingDirectory/pybrain')

def stage1_prepare_dataset():
    df = pd.read_csv("dataset03.csv")
    print(f"Original shape: {df.shape}")
    
    # Clean NaNs and non-numerical
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    
    # IQR outlier removal
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
    
    # Normalize columns
    df = (df - df.min()) / (df.max() - df.min())
    
    # Split train/test
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)
    
    # Save CSV
    train_df.to_csv("dataset03_training.csv", index=False)
    test_df.to_csv("dataset03_testing.csv", index=False)
    
    # Create PyBrain datasets
    train_ds = SupervisedDataSet(1,1)
    for _, row in train_df.iterrows():
        train_ds.addSample(row["x"], row["y"])
    test_ds = SupervisedDataSet(1,1)
    for _, row in test_df.iterrows():
        test_ds.addSample(row["x"], row["y"])
    
    # Save PyBrain datasets
    with open("dataset03_training_ds.pkl","wb") as f:
        pickle.dump(train_ds,f)
    with open("dataset03_testing_ds.pkl","wb") as f:
        pickle.dump(test_ds,f)
    
    print("Stage 1 completed: Dataset prepared.")

# To run stage 1
# stage1_prepare_dataset()
