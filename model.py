# model.py — Customer Purchase Amount Predictor
# Run: python model.py

import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model    import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score

# 1. Load Data
df = pd.read_csv("dataset.csv")
print(f"[1] Dataset loaded  →  Shape: {df.shape}")

# 2. Clean Data
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
print(f"[2] After cleaning  →  Shape: {df.shape}")

# 3. EDA
print("\n[3] Statistical Summary:")
print(df.describe().round(2).to_string())
print("\n    Correlation with Purchase_Amount:")
corr = df.corr()["Purchase_Amount"].drop("Purchase_Amount").sort_values(ascending=False)
print(corr.round(4).to_string())

# 4. Features & Target
features = ["Age","Annual_Income_LPA","Time_On_Website",
            "Products_Browsed","Discount_Availed"]
target   = "Purchase_Amount"
X = df[features]
y = df[target]
print(f"\n[4] Features: {features}")

# 5. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print(f"[5] Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

# 6. Train
model = LinearRegression()
model.fit(X_train, y_train)
print("[6] LinearRegression trained ✔")

# 7. Evaluate
y_pred = model.predict(X_test)
mae    = mean_absolute_error(y_test, y_pred)
mse    = mean_squared_error(y_test, y_pred)
rmse   = np.sqrt(mse)
r2     = r2_score(y_test, y_pred)
mape   = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print(f"\n[7] Evaluation:")
print(f"    MAE   : ₹{mae:,.0f}")
print(f"    MSE   : ₹{mse:,.0f}")
print(f"    RMSE  : ₹{rmse:,.0f}")
print(f"    R²    : {r2*100:.2f}%")
print(f"    MAPE  : {mape:.2f}%")

print("\n    Feature Coefficients:")
for f, c in zip(features, model.coef_):
    print(f"      {f:<22}: ₹{c:+,.2f}")
print(f"      Intercept             : ₹{model.intercept_:,.2f}")

# 8. Save
with open("model.pkl","wb") as f:
    pickle.dump(model, f)
print("\n[8] model.pkl saved ✔")
