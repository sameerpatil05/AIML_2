"""
model.py - Customer Purchase Prediction ML Pipeline
DSBDA Mini Project
"""

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("=" * 55)
print("   Customer Purchase Prediction - ML Pipeline")
print("=" * 55)

df = pd.read_csv("dataset.csv")
print("\n[1] Dataset loaded successfully.")
print(f"    Shape : {df.shape}")
print(f"\n{df.head()}\n")

# ─────────────────────────────────────────────
# 2. DATA PREPROCESSING
# ─────────────────────────────────────────────
print("[2] Preprocessing...")
print(f"    Missing values:\n{df.isnull().sum()}\n")
print(f"    Basic statistics:\n{df.describe().round(2)}\n")

# ─────────────────────────────────────────────
# 3. EDA – VISUALISATIONS
# ─────────────────────────────────────────────
print("[3] Generating EDA plots...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Customer Purchase Prediction – EDA", fontsize=16, fontweight="bold")

features = ["Age", "Annual_Income", "Spending_Score",
            "Years_as_Customer", "Number_of_Purchases"]
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

for i, (feat, color) in enumerate(zip(features, colors)):
    ax = axes[i // 3][i % 3]
    ax.scatter(df[feat], df["Purchase_Amount"], alpha=0.5, color=color, edgecolors="none")
    ax.set_xlabel(feat, fontsize=10)
    ax.set_ylabel("Purchase Amount (₹)", fontsize=10)
    ax.set_title(f"{feat} vs Purchase Amount", fontsize=11)

# Correlation heatmap in the last subplot
ax_heat = axes[1][2]
corr = df.corr()
im = ax_heat.imshow(corr, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
ax_heat.set_xticks(range(len(corr.columns)))
ax_heat.set_yticks(range(len(corr.columns)))
ax_heat.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=7)
ax_heat.set_yticklabels(corr.columns, fontsize=7)
ax_heat.set_title("Correlation Heatmap", fontsize=11)
plt.colorbar(im, ax=ax_heat)

plt.tight_layout()
plt.savefig("eda_plots.png", dpi=150, bbox_inches="tight")
plt.close()
print("    EDA plots saved → eda_plots.png\n")

# ─────────────────────────────────────────────
# 4. FEATURE SELECTION
# ─────────────────────────────────────────────
print("[4] Feature selection...")
X = df[["Age", "Annual_Income", "Spending_Score",
        "Years_as_Customer", "Number_of_Purchases"]]
y = df["Purchase_Amount"]
print(f"    Features : {list(X.columns)}")
print(f"    Target   : Purchase_Amount\n")

# ─────────────────────────────────────────────
# 5. TRAIN-TEST SPLIT
# ─────────────────────────────────────────────
print("[5] Train-Test split (80 / 20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"    Train size : {X_train.shape[0]} samples")
print(f"    Test  size : {X_test.shape[0]} samples\n")

# Scale features
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# 6. MODEL TRAINING
# ─────────────────────────────────────────────
print("[6] Training Linear Regression model...")
model = LinearRegression()
model.fit(X_train_sc, y_train)
print("    Model trained successfully.\n")

# ─────────────────────────────────────────────
# 7. MODEL EVALUATION
# ─────────────────────────────────────────────
print("[7] Evaluating model...")
y_pred = model.predict(X_test_sc)

mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print(f"    MAE  : {mae:.4f}")
print(f"    MSE  : {mse:.4f}")
print(f"    RMSE : {rmse:.4f}")
print(f"    R²   : {r2:.4f}\n")

# Actual vs Predicted plot
fig2, ax2 = plt.subplots(figsize=(7, 5))
ax2.scatter(y_test, y_pred, alpha=0.6, color="#4C72B0", edgecolors="none")
mn = min(y_test.min(), y_pred.min())
mx = max(y_test.max(), y_pred.max())
ax2.plot([mn, mx], [mn, mx], "r--", linewidth=2, label="Perfect fit")
ax2.set_xlabel("Actual Purchase Amount (₹)", fontsize=12)
ax2.set_ylabel("Predicted Purchase Amount (₹)", fontsize=12)
ax2.set_title("Actual vs Predicted Purchase Amount", fontsize=13, fontweight="bold")
ax2.legend()
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Plot saved → actual_vs_predicted.png\n")

# ─────────────────────────────────────────────
# 8. SAVE MODEL
# ─────────────────────────────────────────────
print("[8] Saving model and scaler...")
with open("model.pkl", "wb") as f:
    pickle.dump({"model": model, "scaler": scaler}, f)
print("    Saved → model.pkl\n")

print("=" * 55)
print("   Pipeline complete! All artefacts generated.")
print("=" * 55)
