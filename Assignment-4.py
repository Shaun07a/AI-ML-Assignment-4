# ==========================================
# AI-ML Assignment 4
# Breast Cancer Classification using KNN
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ----------------------------------------
# Task 1 : Data Understanding
# ----------------------------------------

df = pd.read_csv("data.csv")

print("First Five Records\n")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
df.info()

print("\nSummary Statistics:")
print(df.describe())

print("\nNumerical Features:")
print(df.select_dtypes(include=["number"]).columns.tolist())

print("\nTarget Variable:")
print("diagnosis")

# ----------------------------------------
# Task 2 : Data Preprocessing
# ----------------------------------------

print("\nMissing Values:")
print(df.isnull().sum())

# Remove unnecessary columns
df.drop(["id", "Unnamed: 32"], axis=1, inplace=True)

# Encode target variable
encoder = LabelEncoder()
df["diagnosis"] = encoder.fit_transform(df["diagnosis"])
# B -> 0
# M -> 1

X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ----------------------------------------
# Task 3 : Model Development
# ----------------------------------------

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

# ----------------------------------------
# Task 4 : Model Evaluation
# ----------------------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n========== Model Evaluation ==========\n")
print(f"Accuracy Score : {accuracy:.4f}")
print(f"Precision      : {precision:.4f}")
print(f"Recall         : {recall:.4f}")
print(f"F1-Score       : {f1:.4f}")

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot(cmap="Blues")

plt.title("Confusion Matrix")

plt.show()