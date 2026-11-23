# ⚡ Household Electricity Consumption Prediction Using Regression Models

An end-to-end Machine Learning project to predict monthly household electricity consumption (in kWh) using advanced regression techniques, exploratory data analysis (EDA), feature engineering, and cross-validation.

---

## 📌 Project Overview

Accurate prediction of household electricity consumption helps utility providers balance power grids, forecast demand, and empowers consumers to optimize energy efficiency. 

This project explores a dataset of **20,000 households** to analyze key consumption drivers (appliance counts, daily usage hours, house size, temperature, seasons) and benchmarks multiple regression algorithms to identify the best-performing model.

---

## 📊 Dataset Description

The dataset (`electricity_consumption_dataset_20000.csv`) consists of 20,000 observations and 18 features:

| Feature Name | Type | Description |
| :--- | :--- | :--- |
| `House_ID` | Identifier | Unique ID for each household |
| `House_Type` | Categorical | Apartment, Villa, Independent House, etc. |
| `Family_Members` | Numerical | Number of members in the household |
| `House_Area_sqft` | Numerical | Total area of the house in square feet |
| `Air_Conditioners` | Numerical | Number of AC units installed |
| `Refrigerators` | Numerical | Number of refrigerators |
| `Washing_Machines`| Numerical | Number of washing machines |
| `Televisions` | Numerical | Number of TV sets |
| `Computers` | Numerical | Number of desktops/laptops |
| `Water_Heaters` | Numerical | Number of geysers/water heaters |
| `Ceiling_Fans` | Numerical | Number of ceiling fans |
| `LED_Lights` | Numerical | Number of LED light fixtures |
| `EV_Charger` | Categorical | Presence of an EV charger (`Yes` / `No`) |
| `Solar_Panels` | Categorical | Presence of solar panels (`Yes` / `No`) |
| `Daily_Usage_Hours`| Numerical | Average appliance operational hours per day |
| `Average_Temperature`| Numerical | Ambient regional temperature (°C) |
| `Season` | Categorical | Season (Summer, Winter, Monsoon, Spring) |
| `Occupancy_Rate` | Numerical | Percentage of time the home is occupied |
| `Electricity_Tariff`| Numerical | Electricity rate per kWh |
| **`Monthly_Electricity_Consumption`** | **Target** | **Total consumption in kWh (Target variable)** |

---

## 🛠️ Key Pipeline Stages

1. **Exploratory Data Analysis (EDA)**:
   - Correlation heatmaps to identify strong linear and non-linear relationships.
   - Outlier detection and distribution analysis using boxplots and histograms.
   - Seasonality and appliance impact analysis.

2. **Feature Engineering & Preprocessing**:
   - One-Hot Encoding for categorical features (`House_Type`, `Season`, `EV_Charger`, `Solar_Panels`).
   - Feature Scaling using `StandardScaler`.
   - Feature selection and alignment with trained feature columns.

3. **Models Evaluated**:
   - **Linear Regression** (Baseline)
   - **Ridge Regression** (L2 Regularization)
   - **Random Forest Regressor** (Ensemble Bagging)
   - **Gradient Boosting Regressor** (Ensemble Boosting)

4. **Model Evaluation & Validation**:
   - Metrics: $R^2$ Score, MAE (Mean Absolute Error), MSE, and RMSE.
   - **5-Fold Cross-Validation** to ensure generalizability and prevent data leakage/overfitting.
   - Residual analysis (normality check & homoscedasticity).

5. **Model Serialization & Inference**:
   - Trained model (`electricity_best_model.pkl`), feature column list, and scaler saved using `joblib`.
   - Includes custom inference function `predict_household_consumption(house_details)` for real-time predictions.

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib
```

### Running the Script

```bash
python electricity_consumption_prediction_using_regression_models.py
```

### Sample Inference

```python
sample_household = {
    'House_Type': 'Villa',
    'Family_Members': 4,
    'House_Area_sqft': 2500,
    'Air_Conditioners': 2,
    'Refrigerators': 2,
    'Washing_Machines': 1,
    'Televisions': 2,
    'Computers': 2,
    'Water_Heaters': 1,
    'Ceiling_Fans': 5,
    'LED_Lights': 25,
    'EV_Charger': 'Yes',
    'Solar_Panels': 'No',
    'Daily_Usage_Hours': 12.0,
    'Average_Temperature': 32.5,
    'Season': 'Summer',
    'Occupancy_Rate': 75,
    'Electricity_Tariff': 7.50
}

predicted_kwh = predict_household_consumption(sample_household)
print(f"Predicted Monthly Electricity Consumption: {predicted_kwh} kWh")
```

---

## 👤 Author
- **Abhinav Raj** ([@rajabhinav0620](mailto:rajabhinav0620@gmail.com))
