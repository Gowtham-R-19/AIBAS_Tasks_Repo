import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from UE_04_LinearRegDiagnostic import LinearRegDiagnostic

# -----------------------------
# 1. Load CSV
# -----------------------------
data = pd.read_csv("dataset02.csv")

# Ensure x and y columns exist
if not {'x','y'}.issubset(data.columns):
    raise ValueError("CSV must have 'x' and 'y' columns")

# Keep numeric and drop NaNs
data = data[['x','y']].apply(pd.to_numeric, errors='coerce').dropna()

# -----------------------------
# 2. Outlier Removal (IQR)
# -----------------------------
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
data = data[~((data < (Q1 - 1.5*IQR)) | (data > (Q3 + 1.5*IQR))).any(axis=1)]

# -----------------------------
# 3. Normalization
# -----------------------------
scaler = StandardScaler()
data[['x','y']] = scaler.fit_transform(data[['x','y']])

# -----------------------------
# 4. Train/Test Split
# -----------------------------
train, test = train_test_split(data, test_size=0.2, random_state=42)
train.to_csv("dataset02_training.csv", index=False)
test.to_csv("dataset02_testing.csv", index=False)

# -----------------------------
# 5. OLS Regression (Training)
# -----------------------------
X_train = sm.add_constant(train[['x']])
y_train = train['y']
model = sm.OLS(y_train, X_train).fit()

# Save OLS model summary
with open("OLS_model.txt", "w") as f:
    f.write(model.summary().as_text())

# -----------------------------
# 6. Scatter Plot
# -----------------------------
plt.figure(figsize=(8,6))
plt.scatter(train['x'], train['y'], color='orange', label='Training')
plt.scatter(test['x'], test['y'], color='blue', label='Testing')
plt.plot(train['x'], model.predict(X_train), color='red', label='OLS Fit')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Scatter Plot with OLS Fit")
plt.legend()
plt.savefig("UE_04_App2_ScatterVisualizationAndOlsModel.pdf")
plt.close()

# -----------------------------
# 7. Box Plot
# -----------------------------
plt.figure(figsize=(8,6))
plt.boxplot(data[['x','y']].values, labels=['x','y'])
plt.title("Box Plot of x and y")
plt.savefig("UE_04_App2_BoxPlot.pdf")
plt.close()

# -----------------------------
# 8. Diagnostic Plots
# -----------------------------
diagnostic = LinearRegDiagnostic(model)
diagnostic.plot_all("UE_04_App2_DiagnosticPlots.pdf")

print("Task03 completed: CSVs, OLS model, scatter, box, and diagnostic plots generated.")
