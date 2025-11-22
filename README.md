# Crop Yield Prediction Project

**Group Members:**  
- Antnhe Debebe (ID: ATE/3036/14)  
- Yabsra Abera (ID: ATE/1291/14)  
- Kalkidan Tadese (ID: ATE/1084/14)  
- Eysha Gider(ID: ATE/0652/14)  

---

## Project Overview

This project predicts crop yield using **historical agricultural data** (~24,683 rows).  
We implement **machine learning techniques** with **polynomial features** and **advanced optimization** to model crop yield based on factors like:

- Rainfall (mm)  
- Temperature (°C)  
- Pesticide Use  
- Cultivated Area  
- Production Volume  

We also demonstrate **Big Data concepts** by processing a large dataset and scaling features efficiently.

---

## Dataset

**Source:** `data/raw_dataset.csv`  
**Columns:**

| Column           | Description                           |
|-----------------|---------------------------------------|
| Country          | Country name                           |
| Year             | Year of record                         |
| Crop             | Crop type                              |
| Area             | Cultivated area (hectares)            |
| Production       | Production volume (tons)               |
| Yield            | Crop yield (tons/hectare)             |
| Rainfall_mm      | Rainfall in mm                         |
| Temperature_C    | Average temperature in °C              |
| Pesticide_Use    | Pesticide usage (kg/ha)                |

Shape: `(24683, 9)`  

---

## Methodology

1. **Load and Explore Data**  
   We first load the CSV into pandas and check for missing values.

2. **Feature Engineering**  
   - Original features: `Rainfall_mm, Temperature_C, Pesticide_Use, Area, Production`  
   - Expanded with **PolynomialFeatures(degree=2)** → 20 features including interactions  
   - Bias column added for linear regression intercept  

3. **Train-Test Split & Scaling**  
   - Train set: 70%  
   - Test set: 30%  
   - Features standardized using `StandardScaler`  

4. **Model Training**  
   Advanced optimization with:
   - **Momentum Gradient Descent** (`beta=0.9`)  
   - **L2 Regularization** (`lambda=0.001`)  
   - **Early Stopping** to prevent overfitting  

5. **Evaluation Metrics**  
   - **Mean Squared Error (MSE)**  
   - **R² Score**  
   - Visualizations: Predicted vs Actual Yield  

---

## Model Performance

```text
Test MSE : 5.8797
R² Score : 0.807
