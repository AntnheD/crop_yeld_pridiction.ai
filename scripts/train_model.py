import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os
import sys

# ================================================================
# 1) SAFE CSV LOADER (fixed)
# ================================================================
def load_safe_csv(path):
    print(f"\n🔍 Loading dataset: {path}")

    if not os.path.exists(path):
        print("❌ ERROR: File not found.")
        sys.exit(1)

    try:
        df = pd.read_csv(
            path,
            encoding="utf-8",
            on_bad_lines="skip",        # skip corrupted rows
            skip_blank_lines=True       # ignore empty lines
        )
    except pd.errors.EmptyDataError:
        print("❌ ERROR: CSV file is empty.")
        sys.exit(1)

    if df.shape[1] == 1:
        print("⚠ CSV formatting issue detected. Retrying with different separator...")
        try:
            df = pd.read_csv(path, sep=",", engine="python")
        except:
            print("❌ CSV still unreadable. Fix formatting.")
            sys.exit(1)

    if df.empty:
        print("❌ Dataset contains no data rows.")
        sys.exit(1)

    print("✅ Dataset loaded successfully.")
    print("📌 Shape:", df.shape)
    print("📌 Columns:", list(df.columns))

    expected_cols = [
        "Country","Year","Crop","Area","Production",
        "Yield","Rainfall_mm","Temperature_C","Pesticide_Use"
    ]

    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        print("❌ Missing columns:", missing)
        sys.exit(1)

    print("✔ All required columns exist.\n")
    return df


# DATA PATH
DATA_CSV = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data/raw_dataset.csv"
)

df = load_safe_csv(DATA_CSV)

# ================================================================
# 2) FEATURE SELECTION
# ================================================================
X = df[["Rainfall_mm", "Temperature_C", "Pesticide_Use", "Area", "Production"]].values
y = df["Yield"].values.reshape(-1, 1)

print("📌 X Shape:", X.shape)
print("📌 y Shape:", y.shape)

# ================================================================
# 3) FEATURE ENGINEERING
# ================================================================
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

print("✨ Polynomial features created:", X_poly.shape[1])

# ================================================================
# 4) TRAIN–TEST SPLIT
# ================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_poly, y, test_size=0.3, random_state=42
)

print("\n📌 Train size:", X_train.shape)
print("📌 Test size:", X_test.shape)

# ================================================================
# 5) SCALING
# ================================================================
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Add bias column
X_train_b = np.hstack([np.ones((X_train_s.shape[0], 1)), X_train_s])
X_test_b = np.hstack([np.ones((X_test_s.shape[0], 1)), X_test_s])

print("\n🔧 Bias added → final training features:", X_train_b.shape[1])

# ================================================================
# 6) TRAINING FUNCTION (unchanged)
# ================================================================
def train_advanced(X, y, lr=0.01, iters=5000, lambda_reg=0.001, beta=0.9):
    m, n = X.shape
    theta = np.zeros((n, 1))
    v = np.zeros((n, 1))
    cost_hist = []
    best_cost = float("inf")
    patience = 200
    wait = 0

    for i in range(iters):
        preds = X.dot(theta)
        error = preds - y

        grad = (1/m) * X.T.dot(error) + (lambda_reg/m) * theta
        grad[0] -= (lambda_reg/m) * theta[0]

        v = beta * v + (1 - beta) * grad
        theta -= lr * v

        cost = (1/(2*m)) * np.sum(error**2)
        cost_hist.append(cost)

        if cost < best_cost:
            best_cost = cost
            best_theta = theta.copy()
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            print(f"⏳ Early Stop at iteration {i}")
            break

    return best_theta, cost_hist

# ================================================================
# 7) TRAIN MODEL
# ================================================================
theta, costs = train_advanced(X_train_b, y_train)

print("\n✅ Training complete.")
print("📉 Final cost:", costs[-1])

# ================================================================
# 8) EVALUATE MODEL
# ================================================================
y_pred = X_test_b.dot(theta)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n=== MODEL PERFORMANCE ===")
print(f"MSE: {mse}")
print(f"R² Score: {r2}")

# ================================================================
# 9) SAVE MODEL
# ================================================================
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models/theta_model.npy"
)
np.save(MODEL_PATH, theta)

print(f"\n💾 Model saved → {MODEL_PATH}")
