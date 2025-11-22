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

# -----------------------------
# 2. Data Cleaning
# -----------------------------
data = data.select_dtypes(include=[np.number]).dropna()

# -----------------------------
# 3. Outlier Removal (IQR method)
# -----------------------------
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
data = data[~((data < (Q1 - 1.5*IQR)) | (data > (Q3 + 1.5*IQR))).any(axis=1)]

# -----------------------------
# 4. Normalization
# -----------------------------
scaler = StandardScaler()
data[data.columns] = scaler.fit_transform(data)

# -----------------------------
# 5. Train/Test Split (80/20)
# -----------------------------
train, test = train_test_split(data, test_size=0.2, random_state=42)
train.to_csv("dataset02_training.csv", index=False)
test.to_csv("dataset02_testing.csv", index=False)

# -----------------------------
# 6. OLS Regression (Training data)
# -----------------------------
y_train = train['y']
X_train = train[['x']]
X_train_const = sm.add_constant(X_train)
model = sm.OLS(y_train, X_train_const).fit()

with open("OLS_model.txt", "w") as f:
    f.write(model.summary().as_text())

# -----------------------------
# 7. Scatter Plot (Training vs Testing)
# -----------------------------
plt.figure(figsize=(8,6))
plt.scatter(train['x'], train['y'], color='orange', label='Training')
plt.scatter(test['x'], test['y'], color='blue', label='Testing')
plt.plot(train['x'], model.predict(X_train_const), color='red', label='OLS Fit')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Scatter Plot with OLS Fit")
plt.legend()
plt.savefig("UE_04_App2_ScatterVisualizationAndOlsModel.pdf")
plt.close()

# -----------------------------
# 8. Box Plot
# -----------------------------
plt.figure(figsize=(8,6))
plt.boxplot(data.values, labels=data.columns)
plt.title("Box Plot of Data")
plt.savefig("UE_04_App2_BoxPlot.pdf")
plt.close()

# -----------------------------
# 9. Diagnostic Plots
# -----------------------------
diagnostic = LinearRegDiagnostic(model)
diagnostic.create_plots("UE_04_App2_DiagnosticPlots.pdf")
