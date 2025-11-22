import os
import json

# Make sure notebooks folder exists
notebooks_path = "../notebooks"
os.makedirs(notebooks_path, exist_ok=True)  # creates folder if missing

# Save notebook
notebook_file = os.path.join(notebooks_path, "analysis.ipynb")
with open(notebook_file, "w") as f:
    json.dump(notebook_content, f, indent=2)

print(f"✅ Notebook '{notebook_file}' created successfully.")


# JSON content for the notebook
notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Crop Yield Prediction Analysis\n",
    "\n",
    "**Group Members:**\n",
    "- Alice Smith (ID: \"001\")\n",
    "- Bob Johnson (ID: \"002\")\n",
    "- Carol Lee (ID: \"003\")\n",
    "- David Kim (ID: \"004\")\n",
    "\n",
    "This notebook demonstrates:\n",
    "- Loading and exploring a large agricultural dataset (~24k rows)\n",
    "- Polynomial feature engineering and interaction terms\n",
    "- Using a pre-trained model or training an advanced model from scratch\n",
    "- Calculating all key ML metrics and training dynamics\n",
    "- Visualizing performance (charts inline)\n",
    "- Generating charts for README.md usage\n",
    "- Concepts of Big Data and Machine Learning explained inline"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.preprocessing import StandardScaler, PolynomialFeatures\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import mean_squared_error, r2_score\n",
    "import os\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1️⃣ Load Dataset"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "DATA_CSV = '../data/raw_dataset.csv'\n",
    "df = pd.read_csv(DATA_CSV)\n",
    "print('✅ Dataset loaded successfully.')\n",
    "print('Shape:', df.shape)\n",
    "print('Columns:', df.columns.tolist())\n",
    "print('Missing values per column:\\n', df.isna().sum())\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2️⃣ Feature Selection & Target"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "X = df[['Rainfall_mm', 'Temperature_C', 'Pesticide_Use', 'Area', 'Production']].values\n",
    "y = df['Yield'].values.reshape(-1,1)\n",
    "print('X Shape:', X.shape)\n",
    "print('y Shape:', y.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3️⃣ Polynomial Feature Engineering"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "poly = PolynomialFeatures(degree=2, include_bias=False)\n",
    "X_poly = poly.fit_transform(X)\n",
    "print('Original features:', X.shape[1])\n",
    "print('Polynomial features:', X_poly.shape[1])\n",
    "print('Feature names:', poly.get_feature_names_out(['Rain', 'Temp', 'Pest', 'Area', 'Prod']))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4️⃣ Train-Test Split & Scaling"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.3, random_state=42)\n",
    "scaler = StandardScaler()\n",
    "X_train_s = scaler.fit_transform(X_train)\n",
    "X_test_s = scaler.transform(X_test)\n",
    "X_train_b = np.hstack([np.ones((X_train_s.shape[0],1)), X_train_s])\n",
    "X_test_b = np.hstack([np.ones((X_test_s.shape[0],1)), X_test_s])\n",
    "print('Train X:', X_train_b.shape, 'Train y:', y_train.shape)\n",
    "print('Test X:', X_test_b.shape, 'Test y:', y_test.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5️⃣ Advanced Training Function (Momentum + L2 + Early Stopping)"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "def train_advanced(X, y, lr=0.01, iters=8000, lambda_reg=0.001, beta=0.9, eps=1e-8):\n",
    "    m, n = X.shape\n",
    "    theta = np.zeros((n,1))\n",
    "    v = np.zeros((n,1))\n",
    "    cost_hist = []\n",
    "    best_cost = float('inf')\n",
    "    patience = 200\n",
    "    wait = 0\n",
    "    for i in range(iters):\n",
    "        preds = X.dot(theta)\n",
    "        error = preds - y\n",
    "        grad = (1/m) * X.T.dot(error) + (lambda_reg/m)*theta\n",
    "        grad[0] -= (lambda_reg/m)*theta[0]\n",
    "        v = beta*v + (1-beta)*grad\n",
    "        theta -= lr*v\n",
    "        cost = (1/(2*m))*np.sum(error**2) + (lambda_reg/(2*m))*np.sum(theta[1:]**2)\n",
    "        cost_hist.append(cost)\n",
    "        if cost < best_cost:\n",
    "            best_cost = cost\n",
    "            best_theta = theta.copy()\n",
    "            wait = 0\n",
    "        else:\n",
    "            wait += 1\n",
    "        if wait >= patience:\n",
    "            print(f'Early stop at iteration {i}')\n",
    "            break\n",
    "    return best_theta, cost_hist"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6️⃣ Train Model or Load Pre-trained Weights"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "MODEL_FILE = '../models/theta_model.npy'\n",
    "if os.path.exists(MODEL_FILE):\n",
    "    theta = np.load(MODEL_FILE)\n",
    "    print('Loaded pre-trained theta_model.npy')\n",
    "else:\n",
    "    theta, costs = train_advanced(X_train_b, y_train)\n",
    "    np.save(MODEL_FILE, theta)\n",
    "    print('Training complete, model saved')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7️⃣ Model Performance"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "y_pred = X_test_b.dot(theta)\n",
    "mse = mean_squared_error(y_test, y_pred)\n",
    "r2 = r2_score(y_test, y_pred)\n",
    "print('Test MSE:', mse)\n",
    "print('R² Score:', r2)\n",
    "\n",
    "plt.figure(figsize=(6,6))\n",
    "plt.scatter(y_test, y_pred, alpha=0.7)\n",
    "plt.plot([y_test.min(), y_test.max()],[y_test.min(), y_test.max()],'r--')\n",
    "plt.xlabel('Actual Yield')\n",
    "plt.ylabel('Predicted Yield')\n",
    "plt.title('Predicted vs Actual')\n",
    "plt.grid(True)\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.11"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

# Save notebook
with open("../notebooks/analysis.ipynb", "w") as f:
    json.dump(notebook_content, f, indent=2)

print("✅ Notebook 'analysis.ipynb' created successfully in /notebooks/")
