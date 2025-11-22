# pythonApp.py
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load CSV
data = pd.read_csv("dataset01.csv")

y = data['y']
x = data[['x']]

# Statistics
print("Number of entries in y:", y.count())
print("Mean of y:", y.mean())
print("Std deviation of y:", y.std())
print("Variance of y:", y.var())
print("Min of y:", y.min())
print("Max of y:", y.max())

# OLS
X_const = sm.add_constant(x)
model = sm.OLS(y, X_const).fit()

with open("OLS_model.txt", "w") as f:
    f.write(model.summary().as_text())

print("OLS model saved.")
