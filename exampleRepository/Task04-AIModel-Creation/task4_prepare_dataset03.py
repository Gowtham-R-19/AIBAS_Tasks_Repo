import sys
sys.path.append('/tmp/AIBAS_exercise_WorkingDirectory/pybrain')  # essential for PyBrain

import pandas as pd
from pybrain.datasets import SupervisedDataSet
import pickle
from sklearn.preprocessing import MinMaxScaler

# Load dataset03.csv
df = pd.read_csv('dataset03.csv')

# Keep only numerical and drop NaN
df = df.select_dtypes(include='number').dropna()

# Remove outliers using IQR
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

# Normalize
scaler = MinMaxScaler()
df[['x','y']] = scaler.fit_transform(df[['x','y']])

# Split train/test (80%/20%)
train_size = int(len(df)*0.8)
df_train = df.iloc[:train_size]
df_test = df.iloc[train_size:]

# Save CSV files
df.to_csv('dataset03_cleaned.csv', index=False)
df_train.to_csv('dataset03_training.csv', index=False)
df_test.to_csv('dataset03_testing.csv', index=False)

# Prepare PyBrain datasets
ds_train = SupervisedDataSet(1,1)
ds_test  = SupervisedDataSet(1,1)
for i,row in df_train.iterrows():
    ds_train.addSample([row['x']], [row['y']])
for i,row in df_test.iterrows():
    ds_test.addSample([row['x']], [row['y']])

# Save datasets
with open('dataset03_training_ds.pkl', 'wb') as f:
    pickle.dump(ds_train, f)
with open('dataset03_testing_ds.pkl', 'wb') as f:
    pickle.dump(ds_test, f)

print("Dataset preparation complete.")
