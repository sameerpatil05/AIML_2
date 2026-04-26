# model.py
# Customer Purchase Prediction - Model Training
# DSBDA Mini Project

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Step 1: Load Dataset
df = pd.read_csv("dataset.csv")
print("Dataset loaded. Shape:", df.shape)
print(df.head())

# Step 2: Define Features and Target
X = df[["Age", "Annual_Income", "Spending_Score"]]
y = df["Purchase_Amount"]

# Step 3: Split into Train and Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))

# Step 4: Train the Model
model = LinearRegression()
model.fit(X_train, y_train)
print("\nModel trained successfully.")

# Step 5: Evaluate the Model
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)

print("\n--- Model Evaluation ---")
print(f"MAE      : {mae:.2f}")
print(f"R² Score : {r2:.2f}")

# Step 6: Save the Model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel saved as model.pkl")
