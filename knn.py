
"""
Advanced KNN Rainfall Prediction System
======================================

Features
--------
1. Annual rainfall analysis
2. JJAS rainfall analysis
3. June rainfall analysis
4. July rainfall analysis
5. August rainfall analysis
6. September rainfall analysis
7. K-value optimization
8. Cross-validation
9. Model evaluation metrics
10. Automated visualizations
11. CSV report export
12. Summary report generation

Author: ChatGPT
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    KFold
)

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


class RainfallDataProcessor:

    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.raw_df = None
        self.cleaned_df = None

    def load_data(self):
        self.raw_df = pd.read_csv(self.csv_path)
        return self.raw_df

    def clean_data(self):

        df = self.raw_df.copy()

        df = df.drop(df.index[:19])

        cols = ['-BEGIN HEADER-', 'Unnamed: 1', 'Unnamed: 5']

        ndf = df[cols].copy()

        ndf.columns = [
            'year',
            'dayno',
            'rainfall'
        ]

        ndf['year'] = ndf['year'].astype(int)
        ndf['dayno'] = ndf['dayno'].astype(int)
        ndf['rainfall'] = ndf['rainfall'].astype(float)

        self.cleaned_df = ndf

        return ndf

    def annual_rainfall(self):

        annual = (
            self.cleaned_df
            .groupby('year')['rainfall']
            .sum()
            .reset_index()
        )

        return annual

    def seasonal_rainfall(self, start_day, end_day):

        subset = self.cleaned_df[
            (self.cleaned_df['dayno'] >= start_day) &
            (self.cleaned_df['dayno'] <= end_day)
        ]

        rainfall = (
            subset
            .groupby('year')['rainfall']
            .sum()
            .reset_index()
        )

        return rainfall


class KNNRainfallModel:

    def __init__(self, dataframe, label):

        self.df = dataframe
        self.label = label

        self.scaler = StandardScaler()

        self.model = None

        self.best_k = None

        self.results = None

    def optimize_k(self,
                   min_k=1,
                   max_k=20):

        X = self.df[['year']]
        y = self.df['rainfall']

        X_scaled = self.scaler.fit_transform(X)

        best_score = -999999

        best_k = None

        for k in range(min_k, max_k + 1):

            model = KNeighborsRegressor(
                n_neighbors=k
            )

            cv = KFold(
                n_splits=5,
                shuffle=True,
                random_state=42
            )

            scores = cross_val_score(
                model,
                X_scaled,
                y,
                cv=cv,
                scoring='r2'
            )

            mean_score = np.mean(scores)

            if mean_score > best_score:

                best_score = mean_score
                best_k = k

        self.best_k = best_k

        return best_k, best_score

    def train(self):

        X = self.df[['year']]
        y = self.df['rainfall']

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42
        )

        X_train_scaled = self.scaler.fit_transform(
            X_train
        )

        X_test_scaled = self.scaler.transform(
            X_test
        )

        self.model = KNeighborsRegressor(
            n_neighbors=self.best_k
        )

        self.model.fit(
            X_train_scaled,
            y_train
        )

        train_pred = self.model.predict(
            X_train_scaled
        )

        test_pred = self.model.predict(
            X_test_scaled
        )

        train_df = pd.DataFrame({
            "Year": X_train.values.flatten(),
            "Actual Rainfall": y_train,
            "Predicted Rainfall": train_pred
        })

        test_df = pd.DataFrame({
            "Year": X_test.values.flatten(),
            "Actual Rainfall": y_test,
            "Predicted Rainfall": test_pred
        })

        combined = pd.concat(
            [train_df, test_df],
            ignore_index=True
        )

        combined["Absolute Error"] = (
            combined["Actual Rainfall"] -
            combined["Predicted Rainfall"]
        ).abs()

        combined["Percentage Error"] = (
            combined["Absolute Error"] /
            combined["Actual Rainfall"]
        ) * 100

        metrics = {
            "MSE": mean_squared_error(
                y_test,
                test_pred
            ),
            "MAE": mean_absolute_error(
                y_test,
                test_pred
            ),
            "R2": r2_score(
                y_test,
                test_pred
            ),
            "Mean Error":
                combined["Absolute Error"].mean(),
            "Mean % Error":
                combined["Percentage Error"].mean()
        }

        self.results = {
            "combined": combined,
            "metrics": metrics
        }

        return self.results

    def plot_actual_rainfall(self):

        df = self.results["combined"]

        df = df.sort_values("Year")

        plt.figure(figsize=(12, 6))

        plt.bar(
            df["Year"],
            df["Actual Rainfall"]
        )

        plt.title(
            f"{self.label} Actual Rainfall"
        )

        plt.xlabel("Year")
        plt.ylabel("Rainfall (mm)")

        plt.grid(True)

        plt.show()

    def plot_predicted_rainfall(self):

        df = self.results["combined"]

        df = df.sort_values("Year")

        plt.figure(figsize=(12, 6))

        plt.plot(
            df["Year"],
            df["Predicted Rainfall"],
            marker="o"
        )

        plt.title(
            f"{self.label} Predicted Rainfall"
        )

        plt.xlabel("Year")
        plt.ylabel("Rainfall (mm)")

        plt.grid(True)

        plt.show()

    def plot_error(self):

        df = self.results["combined"]

        df = df.sort_values("Year")

        plt.figure(figsize=(12, 6))

        plt.bar(
            df["Year"],
            df["Percentage Error"]
        )

        plt.title(
            f"{self.label} Percentage Error"
        )

        plt.xlabel("Year")
        plt.ylabel("% Error")

        plt.grid(True)

        plt.show()

    def export_csv(self):

        filename = (
            self.label.lower()
            .replace(" ", "_")
            + "_results.csv"
        )

        self.results["combined"].to_csv(
            filename,
            index=False
        )

        return filename


def run_analysis(name,
                 data):

    print("=" * 60)
    print(name)
    print("=" * 60)

    model = KNNRainfallModel(
        data,
        name
    )

    best_k, score = model.optimize_k()

    print("Best K:", best_k)
    print("CV R2 :", round(score, 4))

    model.train()

    metrics = model.results["metrics"]

    for key, value in metrics.items():
        print(
            f"{key:<15}: {value:.4f}"
        )

    model.export_csv()

    model.plot_actual_rainfall()
    model.plot_predicted_rainfall()
    model.plot_error()

    return model


def main():

    csv_path = "rainfall_data.csv"

    processor = RainfallDataProcessor(
        csv_path
    )

    processor.load_data()

    processor.clean_data()

    annual = (
        processor.annual_rainfall()
    )

    june = (
        processor.seasonal_rainfall(
            150,
            180
        )
    )

    july = (
        processor.seasonal_rainfall(
            180,
            210
        )
    )

    august = (
        processor.seasonal_rainfall(
            210,
            240
        )
    )

    september = (
        processor.seasonal_rainfall(
            240,
            270
        )
    )

    jjas = (
        processor.seasonal_rainfall(
            150,
            250
        )
    )

    analyses = [
        ("Annual Rainfall", annual),
        ("JJAS Rainfall", jjas),
        ("June Rainfall", june),
        ("July Rainfall", july),
        ("August Rainfall", august),
        ("September Rainfall", september)
    ]

    summary = []

    for title, dataset in analyses:

        model = run_analysis(
            title,
            dataset
        )

        row = {
            "Analysis": title,
            **model.results["metrics"]
        }

        summary.append(row)

    summary_df = pd.DataFrame(
        summary
    )

    print("\nSummary Report")
    print(summary_df)

    summary_df.to_csv(
        "summary_report.csv",
        index=False
    )


if __name__ == "__main__":
    main()
