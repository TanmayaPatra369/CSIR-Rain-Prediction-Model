"""
Rainfall Prediction Model (June Rainfall Analysis)
--------------------------------------------------
This script analyzes historical June rainfall data and predicts rainfall
using multiple machine learning models:
1. Linear Regression
2. Support Vector Regression (SVR)
3. Polynomial Regression
4. Random Forest Regression

The model uses historical yearly June rainfall totals as the predictor.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# ---------------------------------------------------------------------
# DATA PREPARATION
# ---------------------------------------------------------------------

def prepare_june_rainfall(csv_path):
    df = pd.read_csv(csv_path)

    df = df.drop(df.index[:19])

    rainfall_df = df[['-BEGIN HEADER-', 'Unnamed: 1', 'Unnamed: 5']].copy()
    rainfall_df.columns = ['year', 'dayno', 'rainfall']

    rainfall_df['year'] = rainfall_df['year'].astype(int)
    rainfall_df['dayno'] = rainfall_df['dayno'].astype(int)
    rainfall_df['rainfall'] = rainfall_df['rainfall'].astype(float)

    june_days = rainfall_df[
        (rainfall_df['dayno'] >= 150) &
        (rainfall_df['dayno'] <= 180)
    ]

    june_rain = (
        june_days.groupby('year')['rainfall']
        .sum()
        .reset_index()
    )

    return june_rain


# ---------------------------------------------------------------------
# MODEL EVALUATION
# ---------------------------------------------------------------------

def evaluate_model(model, X, y, model_name):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, test_pred)
    r2 = r2_score(y_test, test_pred)

    print(f"\\n{model_name}")
    print("-" * 40)
    print(f"MSE : {mse:.2f}")
    print(f"R²  : {r2:.4f}")

    return X_train, X_test, y_train, y_test, train_pred, test_pred


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

if __name__ == "__main__":

    CSV_PATH = "rainfall_data.csv"  # Update path

    june_rain = prepare_june_rainfall(CSV_PATH)

    X = june_rain[['year']]
    y = june_rain['rainfall']

    # Linear Regression
    evaluate_model(
        LinearRegression(),
        X,
        y,
        "Linear Regression"
    )

    # SVR
    evaluate_model(
        SVR(kernel="rbf"),
        X,
        y,
        "Support Vector Regression"
    )

    # Polynomial Regression
    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(X)

    evaluate_model(
        LinearRegression(),
        X_poly,
        y,
        "Polynomial Regression"
    )

    # Random Forest
    evaluate_model(
        RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            oob_score=True
        ),
        X,
        y,
        "Random Forest Regression"
    )

    # Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(
        june_rain["year"],
        june_rain["rainfall"],
        marker="o"
    )
    plt.title("June Rainfall Trend")
    plt.xlabel("Year")
    plt.ylabel("Rainfall (mm)")
    plt.grid(True)
    plt.show()
