# =============================================================================
# September Rainfall Prediction — Year-wise Analysis (1981–2024)
# Season  : September (day 240 – 270)
# Dataset : NASA POWER Daily | Location: 12.97°N, 77.59°E
# Models  : Linear Regression | SVR (RBF) | Polynomial Regression | Random Forest
# =============================================================================

# =============================================================================
# SECTION 1 — IMPORTS & GLOBAL CONFIG
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# ── Plot aesthetics ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.figsize":    (12, 5),
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.4,
    "font.size":         11,
})

# ── Constants ──────────────────────────────────────────────────────────────────
FILE_PATH    = "/content/drive/MyDrive/POWER_Point_Daily_19810101_20240430_012d97N_077d59E_LST (1).csv"
RANDOM_STATE = 42
TEST_SIZE    = 0.2

# September window (approximate day-of-year)
SEPT_START = 240    # ~September 1
SEPT_END   = 270    # ~September 27

POLY_DEGREE   = 10
RF_ESTIMATORS = 90

PALETTE = {
    "actual":    "#2196F3",   # blue
    "predicted": "#F44336",   # red
    "error":     "#FF9800",   # orange
    "train":     "#4CAF50",   # green
}


# =============================================================================
# SECTION 2 — DATA LOADING & CLEANING
# =============================================================================

def load_and_clean(filepath: str) -> pd.DataFrame:
    """
    Load the NASA POWER CSV, strip the 19-row metadata header,
    select and rename relevant columns, and cast types.

    Returns
    -------
    df : daily DataFrame with columns [year, day_no, rainfall]
    """
    raw = pd.read_csv(filepath)
    raw = raw.drop(raw.index[:19])

    df = raw[["-BEGIN HEADER-", "Unnamed: 1", "Unnamed: 5"]].copy()
    df.columns = ["year", "day_no", "rainfall"]

    df["year"]     = df["year"].astype(int)
    df["day_no"]   = df["day_no"].astype(int)
    df["rainfall"] = df["rainfall"].astype(float)

    print(f"[Data] Shape       : {df.shape}")
    print(f"[Data] Null values :\n{df.isnull().sum()}\n")
    return df


# =============================================================================
# SECTION 3 — SEPTEMBER EXTRACTION & EDA
# =============================================================================

def extract_september(df: pd.DataFrame,
                      start_day: int = SEPT_START,
                      end_day:   int = SEPT_END) -> pd.DataFrame:
    """
    Filter daily records to the September window (days 240–270) and
    aggregate to annual September totals.

    Returns
    -------
    september : DataFrame with columns [year, rainfall]
    """
    season    = df[(df["day_no"] >= start_day) & (df["day_no"] <= end_day)]
    september = season.groupby("year")["rainfall"].sum().reset_index()
    september.columns = ["year", "rainfall"]

    print(f"[September] Years covered  : {september['year'].min()} – {september['year'].max()}")
    print(f"[September] Mean rainfall  : {september['rainfall'].mean():.2f} mm")
    print(f"[September] Std deviation  : {september['rainfall'].std():.2f} mm")
    print(f"[September] Wettest year   : "
          f"{september.loc[september['rainfall'].idxmax(), 'year']}  "
          f"({september['rainfall'].max():.1f} mm)")
    print(f"[September] Driest year    : "
          f"{september.loc[september['rainfall'].idxmin(), 'year']}  "
          f"({september['rainfall'].min():.1f} mm)\n")
    return september


def plot_september_timeseries(september: pd.DataFrame) -> None:
    """Year-wise September rainfall as a line chart with mean reference."""
    fig, ax = plt.subplots()
    ax.plot(september["year"], september["rainfall"],
            color=PALETTE["actual"], linewidth=1.8, marker="o",
            markersize=4, label="September Rainfall")
    ax.axhline(september["rainfall"].mean(), color="black", linestyle="--",
               linewidth=1.2, label="Long-term mean")
    ax.set_title("Annual September Rainfall (1981–2024)", fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Rainfall (mm)")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_september_bar(september: pd.DataFrame) -> None:
    """Year-wise September rainfall as a bar chart, coloured by above/below mean."""
    mean_val = september["rainfall"].mean()
    colors   = [
        PALETTE["actual"] if v >= mean_val else PALETTE["error"]
        for v in september["rainfall"]
    ]

    fig, ax = plt.subplots()
    ax.bar(september["year"], september["rainfall"],
           color=colors, edgecolor="white", linewidth=0.4, width=0.8)
    ax.axhline(mean_val, color="black", linestyle="--",
               linewidth=1.2, label="Long-term mean")
    ax.legend(handles=[
        mpatches.Patch(color=PALETTE["actual"], label="Above / at mean"),
        mpatches.Patch(color=PALETTE["error"],  label="Below mean"),
        mpatches.Patch(color="black",            label="Mean line"),
    ])
    ax.set_title("Yearly September Rainfall", fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Rainfall (mm)")
    plt.tight_layout()
    plt.show()


# =============================================================================
# SECTION 4 — SHARED UTILITIES
# =============================================================================

def build_result_df(years_train, y_train, y_pred_train,
                    years_test,  y_test,  y_pred_test) -> pd.DataFrame:
    """
    Combine train + test predictions into a single tidy DataFrame.
    Adds Absolute Error, Percentage Error, and a Split label column.

    Returns
    -------
    combined : DataFrame sorted by Year
    """
    train_df = pd.DataFrame({
        "Year":               np.array(years_train).flatten(),
        "Actual Rainfall":    np.array(y_train),
        "Predicted Rainfall": np.array(y_pred_train),
        "Split":              "train",
    })
    test_df = pd.DataFrame({
        "Year":               np.array(years_test).flatten(),
        "Actual Rainfall":    np.array(y_test),
        "Predicted Rainfall": np.array(y_pred_test),
        "Split":              "test",
    })

    combined = pd.concat([train_df, test_df], ignore_index=True)
    combined["Absolute Error"]   = np.abs(
        combined["Actual Rainfall"] - combined["Predicted Rainfall"]
    )
    combined["Percentage Error"] = (
        combined["Absolute Error"] / combined["Actual Rainfall"] * 100
    )
    return combined.sort_values("Year").reset_index(drop=True)


def evaluate(model_name: str, combined: pd.DataFrame) -> dict:
    """
    Print and return MAE, MAPE, and R² for train and test splits.

    Returns
    -------
    metrics : dict
    """
    train = combined[combined["Split"] == "train"]
    test  = combined[combined["Split"] == "test"]

    metrics = {
        "model":      model_name,
        "mae_train":  train["Absolute Error"].mean(),
        "mape_train": train["Percentage Error"].mean(),
        "mae_test":   test["Absolute Error"].mean(),
        "mape_test":  test["Percentage Error"].mean(),
        "r2_test":    r2_score(test["Actual Rainfall"], test["Predicted Rainfall"]),
    }

    print(f"\n{'─' * 58}")
    print(f"  {model_name}")
    print(f"  Train  MAE  : {metrics['mae_train']:.2f} mm  |  "
          f"Train MAPE : {metrics['mape_train']:.2f}%")
    print(f"  Test   MAE  : {metrics['mae_test']:.2f} mm  |  "
          f"Test  MAPE : {metrics['mape_test']:.2f}%")
    print(f"  Test   R²   : {metrics['r2_test']:.4f}")
    print(f"{'─' * 58}")
    return metrics


def plot_actual_vs_predicted(september: pd.DataFrame,
                             combined: pd.DataFrame,
                             model_name: str) -> None:
    """
    Full actual September series (blue line) overlaid with
    test-set predictions (red scatter + dashed line).
    """
    test = combined[combined["Split"] == "test"].sort_values("Year")

    fig, ax = plt.subplots()
    ax.plot(september["year"], september["rainfall"],
            color=PALETTE["actual"], linewidth=1.8, marker="o",
            markersize=4, label="Actual")
    ax.plot(test["Year"], test["Predicted Rainfall"],
            color=PALETTE["predicted"], linestyle="--",
            linewidth=1.4, alpha=0.7)
    ax.scatter(test["Year"], test["Predicted Rainfall"],
               color=PALETTE["predicted"], s=80, zorder=5,
               marker="x", linewidths=2, label="Predicted (test)")
    ax.set_title(f"Actual vs Predicted — {model_name}", fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("September Rainfall (mm)")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_train_test_predictions(combined: pd.DataFrame,
                                model_name: str) -> None:
    """
    Line chart showing predicted rainfall for both train and test splits —
    useful for diagnosing overfitting visually.
    """
    train = combined[combined["Split"] == "train"].sort_values("Year")
    test  = combined[combined["Split"] == "test"].sort_values("Year")

    fig, ax = plt.subplots()
    ax.plot(train["Year"], train["Predicted Rainfall"],
            color=PALETTE["train"], linewidth=1.6, label="Train predictions")
    ax.plot(test["Year"], test["Predicted Rainfall"],
            color=PALETTE["predicted"], linewidth=1.6,
            linestyle="--", label="Test predictions")
    ax.set_title(f"Train vs Test Predictions — {model_name}", fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Predicted September Rainfall (mm)")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_error_bar(combined: pd.DataFrame, model_name: str) -> None:
    """Bar chart of percentage error per year (test set only)."""
    test = combined[combined["Split"] == "test"].sort_values("Year")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(test["Year"], test["Percentage Error"],
           color=PALETTE["error"], edgecolor="white", width=0.7)
    ax.set_title(f"Percentage Error per Year (Test Set) — {model_name}",
                 fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("% Error")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# =============================================================================
# SECTION 5 — MODEL 1: LINEAR REGRESSION
# =============================================================================

def run_linear_regression(september: pd.DataFrame) -> dict:
    """
    Ordinary Least Squares linear regression — simplest baseline.
    Assumes a monotonic linear trend in September rainfall over years.
    """
    X = september[["year"]]
    y = september["rainfall"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    combined = build_result_df(
        X_train["year"].values, y_train, model.predict(X_train),
        X_test["year"].values,  y_test,  model.predict(X_test),
    )

    metrics = evaluate("Linear Regression", combined)
    plot_actual_vs_predicted(september, combined, "Linear Regression")
    plot_train_test_predictions(combined, "Linear Regression")
    plot_error_bar(combined, "Linear Regression")
    return metrics


# =============================================================================
# SECTION 6 — MODEL 2: SVR (RBF KERNEL)
# =============================================================================

def run_svr(september: pd.DataFrame) -> dict:
    """
    Support Vector Regression with RBF kernel.
    Robust to outlier rainfall years; no assumption of linearity.
    """
    X = september["year"].values.reshape(-1, 1)
    y = september["rainfall"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model = SVR(kernel="rbf")
    model.fit(X_train, y_train)

    combined = build_result_df(
        X_train, y_train, model.predict(X_train),
        X_test,  y_test,  model.predict(X_test),
    )

    metrics = evaluate("SVR (RBF kernel)", combined)
    plot_actual_vs_predicted(september, combined, "SVR (RBF kernel)")
    plot_train_test_predictions(combined, "SVR (RBF kernel)")
    plot_error_bar(combined, "SVR (RBF kernel)")
    return metrics


# =============================================================================
# SECTION 7 — MODEL 3: POLYNOMIAL REGRESSION (degree = 10)
# =============================================================================

def run_polynomial_regression(september: pd.DataFrame,
                               degree: int = POLY_DEGREE) -> dict:
    """
    Polynomial feature expansion followed by OLS.
    Degree 10 provides high flexibility but risks overfitting on ~43 rows.
    Inspect the train vs test gap in the comparison table.
    """
    X_raw = september["year"].values.reshape(-1, 1)
    y     = september["rainfall"].values

    poly   = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X_poly, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Column index 1 of the polynomial matrix is the raw year value
    combined = build_result_df(
        X_train[:, 1], y_train, model.predict(X_train),
        X_test[:, 1],  y_test,  model.predict(X_test),
    )

    metrics = evaluate(f"Polynomial Regression (deg={degree})", combined)
    plot_actual_vs_predicted(september, combined,
                             f"Polynomial Regression (deg={degree})")
    plot_train_test_predictions(combined,
                                f"Polynomial Regression (deg={degree})")
    plot_error_bar(combined, f"Polynomial Regression (deg={degree})")
    return metrics


# =============================================================================
# SECTION 8 — MODEL 4: RANDOM FOREST REGRESSOR
# =============================================================================

def run_random_forest(september: pd.DataFrame,
                      n_estimators: int = RF_ESTIMATORS) -> dict:
    """
    Random Forest ensemble with OOB scoring.
    n_estimators=90 preserved from original for stable estimates.
    Typically overfits on small datasets — OOB R² gives a fairer estimate.
    """
    X = september["year"].values.reshape(-1, 1)
    y = september["rainfall"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model = RandomForestRegressor(n_estimators=n_estimators,
                                  random_state=RANDOM_STATE,
                                  oob_score=True)
    model.fit(X_train, y_train)
    print(f"\n[Random Forest] OOB R² : {model.oob_score_:.4f}")

    combined = build_result_df(
        X_train, y_train, model.predict(X_train),
        X_test,  y_test,  model.predict(X_test),
    )

    metrics = evaluate("Random Forest", combined)
    plot_actual_vs_predicted(september, combined, "Random Forest")
    plot_train_test_predictions(combined, "Random Forest")
    plot_error_bar(combined, "Random Forest")
    return metrics


# =============================================================================
# SECTION 9 — MODEL COMPARISON & CONCLUSION
# =============================================================================

def plot_model_comparison(all_metrics: list) -> None:
    """
    Side-by-side horizontal bar charts for Test MAE and Test MAPE,
    followed by a printed summary table with the best model highlighted.
    """
    comparison = (
        pd.DataFrame(all_metrics)
          .sort_values("mae_test")
          .reset_index(drop=True)
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # ── Test MAE ──────────────────────────────────────────────────────────────
    bars = axes[0].barh(comparison["model"], comparison["mae_test"],
                        color=PALETTE["actual"], edgecolor="white")
    axes[0].bar_label(bars, fmt="%.2f mm", padding=4)
    axes[0].set_title("Test MAE  (lower is better)", fontweight="bold")
    axes[0].set_xlabel("Mean Absolute Error (mm)")
    axes[0].invert_yaxis()

    # ── Test MAPE ─────────────────────────────────────────────────────────────
    bars2 = axes[1].barh(comparison["model"], comparison["mape_test"],
                         color=PALETTE["error"], edgecolor="white")
    axes[1].bar_label(bars2, fmt="%.1f%%", padding=4)
    axes[1].set_title("Test MAPE  (lower is better)", fontweight="bold")
    axes[1].set_xlabel("Mean Absolute Percentage Error (%)")
    axes[1].invert_yaxis()

    plt.suptitle("Model Comparison — September Rainfall Prediction",
                 fontweight="bold", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()

    print("\n📊 Final Comparison Table")
    print("=" * 72)
    print(comparison[["model", "mae_train", "mae_test",
                       "mape_train", "mape_test", "r2_test"]]
          .rename(columns={
              "mae_train":  "Train MAE",
              "mae_test":   "Test MAE",
              "mape_train": "Train MAPE%",
              "mape_test":  "Test MAPE%",
              "r2_test":    "Test R²",
          })
          .to_string(index=False))
    print("=" * 72)

    best = comparison.iloc[0]
    print(f"\n✅ Best model by Test MAE : {best['model']}  "
          f"(MAE = {best['mae_test']:.2f} mm, "
          f"MAPE = {best['mape_test']:.1f}%, "
          f"R² = {best['r2_test']:.4f})")


# =============================================================================
# MAIN — ORCHESTRATES ALL SECTIONS
# =============================================================================

def main():
    # ── Mount Drive (Colab only) ───────────────────────────────────────────────
    try:
        from google.colab import drive
        drive.mount("/content/drive")
    except ImportError:
        print("[Info] Not in Colab — skipping Drive mount.")

    # ── Section 2: Load & clean ───────────────────────────────────────────────
    df = load_and_clean(FILE_PATH)

    # ── Section 3: September extraction & EDA ────────────────────────────────
    september = extract_september(df)
    plot_september_timeseries(september)
    plot_september_bar(september)

    # ── Sections 5–8: Train models & collect metrics ──────────────────────────
    print("\n" + "=" * 58)
    print("  MODEL TRAINING & EVALUATION")
    print("=" * 58)

    all_metrics = []
    all_metrics.append(run_linear_regression(september))
    all_metrics.append(run_svr(september))
    all_metrics.append(run_polynomial_regression(september, degree=POLY_DEGREE))
    all_metrics.append(run_random_forest(september, n_estimators=RF_ESTIMATORS))

    # ── Section 9: Comparison ─────────────────────────────────────────────────
    print("\n" + "=" * 58)
    print("  MODEL COMPARISON")
    print("=" * 58)
    plot_model_comparison(all_metrics)


if __name__ == "__main__":
    main()
