# =============================================================================
# August Rainfall Prediction — Year-wise Analysis (1981–2024)
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
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ── Plot aesthetics ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.figsize": (12, 5),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.4,
    "font.size": 11,
})

# ── Constants ──────────────────────────────────────────────────────────────────
FILE_PATH    = "/content/drive/MyDrive/POWER_Point_Daily_19810101_20240430_012d97N_077d59E_LST (1).csv"
RANDOM_STATE = 42
TEST_SIZE    = 0.2
AUGUST_START = 210          # approximate day-of-year for Aug 1
AUGUST_END   = 240          # approximate day-of-year for Aug 28

PALETTE = {
    "actual":    "#2196F3",   # blue
    "predicted": "#F44336",   # red
    "error":     "#FF9800",   # orange
    "neutral":   "#4CAF50",   # green
}


# =============================================================================
# SECTION 2 — DATA LOADING & CLEANING
# =============================================================================

def load_and_clean(filepath: str) -> pd.DataFrame:
    """
    Load the NASA POWER CSV, drop metadata rows, rename columns,
    and cast types.

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

    print(f"[Data] Shape      : {df.shape}")
    print(f"[Data] Null values:\n{df.isnull().sum()}\n")
    return df


# =============================================================================
# SECTION 3 — AUGUST EXTRACTION & EDA
# =============================================================================

def extract_august(df: pd.DataFrame,
                   start_day: int = AUGUST_START,
                   end_day:   int = AUGUST_END) -> pd.DataFrame:
    """
    Filter daily records to the approximate August window (days 210–240)
    and aggregate to annual August totals.

    Returns
    -------
    august_rain : DataFrame with columns [year, rainfall]
    """
    aug = df[(df["day_no"] >= start_day) & (df["day_no"] <= end_day)]
    august_rain = aug.groupby("year")["rainfall"].sum().reset_index()
    august_rain.columns = ["year", "rainfall"]

    print(f"[August] Years covered : {august_rain['year'].min()} – {august_rain['year'].max()}")
    print(f"[August] Mean rainfall : {august_rain['rainfall'].mean():.2f} mm")
    print(f"[August] Std deviation : {august_rain['rainfall'].std():.2f} mm\n")
    return august_rain


def plot_august_distribution(august_rain: pd.DataFrame) -> None:
    """Histogram of August rainfall distribution across all years."""
    fig, ax = plt.subplots()
    ax.hist(august_rain["rainfall"], bins=15,
            color=PALETTE["actual"], edgecolor="white", linewidth=0.6)
    ax.axvline(august_rain["rainfall"].mean(), color="black",
               linestyle="--", linewidth=1.2, label="Mean")
    ax.set_title("Distribution of August Rainfall (1981–2024)", fontweight="bold")
    ax.set_xlabel("Rainfall (mm)")
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_august_timeseries(august_rain: pd.DataFrame) -> None:
    """Year-wise August rainfall as a bar + line chart."""
    fig, ax = plt.subplots()
    ax.bar(august_rain["year"], august_rain["rainfall"],
           color=PALETTE["actual"], alpha=0.6, width=0.8, label="August Rainfall")
    ax.plot(august_rain["year"], august_rain["rainfall"],
            color=PALETTE["actual"], linewidth=1.4, marker="o", markersize=3)
    ax.axhline(august_rain["rainfall"].mean(), color="black",
               linestyle="--", linewidth=1.2, label="Mean")
    ax.set_title("Annual August Rainfall (mm)", fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Rainfall (mm)")
    ax.legend()
    plt.tight_layout()
    plt.show()


# =============================================================================
# SECTION 4 — SHARED UTILITIES
# =============================================================================

def build_result_df(years_train, y_train, y_pred_train,
                    years_test,  y_test,  y_pred_test) -> pd.DataFrame:
    """
    Combine train + test predictions into a single DataFrame with split label,
    absolute error, and percentage error columns.
    """
    train_df = pd.DataFrame({
        "Year":               years_train,
        "Actual Rainfall":    np.array(y_train),
        "Predicted Rainfall": y_pred_train,
        "Split":              "train",
    })
    test_df = pd.DataFrame({
        "Year":               years_test,
        "Actual Rainfall":    np.array(y_test),
        "Predicted Rainfall": y_pred_test,
        "Split":              "test",
    })

    combined = pd.concat([train_df, test_df], ignore_index=True)
    combined["Absolute Error"] = np.abs(
        combined["Actual Rainfall"] - combined["Predicted Rainfall"]
    )
    combined["Percentage Error"] = (
        combined["Absolute Error"] / combined["Actual Rainfall"] * 100
    )
    return combined


def evaluate(model_name: str, combined: pd.DataFrame) -> dict:
    """
    Print and return MAE + MAPE for train and test splits.
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

    print(f"\n{'─'*55}")
    print(f"  {model_name}")
    print(f"  Train MAE  : {metrics['mae_train']:.2f} mm   |  Train MAPE : {metrics['mape_train']:.2f}%")
    print(f"  Test  MAE  : {metrics['mae_test']:.2f} mm   |  Test  MAPE : {metrics['mape_test']:.2f}%")
    print(f"  Test  R²   : {metrics['r2_test']:.4f}")
    print(f"{'─'*55}")
    return metrics


def plot_actual_vs_predicted(august_rain: pd.DataFrame,
                             combined:    pd.DataFrame,
                             model_name:  str) -> None:
    """
    Full actual series (blue line) overlaid with test-set predictions (red scatter).
    """
    test = combined[combined["Split"] == "test"].sort_values("Year")

    fig, ax = plt.subplots()
    ax.plot(august_rain["year"], august_rain["rainfall"],
            color=PALETTE["actual"], linewidth=1.8, marker="o",
            markersize=4, label="Actual")
    ax.scatter(test["Year"], test["Predicted Rainfall"],
               color=PALETTE["predicted"], s=80, zorder=5,
               marker="x", linewidths=2, label="Predicted (test)")
    ax.set_title(f"Actual vs Predicted — {model_name}", fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("August Rainfall (mm)")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_error_bar(combined: pd.DataFrame, model_name: str) -> None:
    """Bar chart of percentage error per year, test set only."""
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

def run_linear_regression(august_rain: pd.DataFrame) -> dict:
    """
    Fit a simple OLS linear regression on year → August rainfall.
    """
    X = august_rain[["year"]]
    y = august_rain["rainfall"]
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
    plot_actual_vs_predicted(august_rain, combined, "Linear Regression")
    plot_error_bar(combined, "Linear Regression")

    return metrics


# =============================================================================
# SECTION 6 — MODEL 2: SUPPORT VECTOR REGRESSION (RBF)
# =============================================================================

def run_svr(august_rain: pd.DataFrame) -> dict:
    """
    Fit an SVR with RBF kernel. Robust to outlier years.
    """
    X = august_rain["year"].values.reshape(-1, 1)
    y = august_rain["rainfall"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model = SVR(kernel="rbf")
    model.fit(X_train, y_train)

    combined = build_result_df(
        X_train.flatten(), y_train, model.predict(X_train),
        X_test.flatten(),  y_test,  model.predict(X_test),
    )

    metrics = evaluate("SVR (RBF kernel)", combined)
    plot_actual_vs_predicted(august_rain, combined, "SVR (RBF kernel)")
    plot_error_bar(combined, "SVR (RBF kernel)")

    return metrics


# =============================================================================
# SECTION 7 — MODEL 3: POLYNOMIAL REGRESSION (degree = 10)
# =============================================================================

def run_polynomial_regression(august_rain: pd.DataFrame,
                               degree: int = 10) -> dict:
    """
    Polynomial feature expansion followed by OLS.
    High degree (10) can overfit on ~43 observations — watch train/test gap.
    """
    X_raw = august_rain["year"].values.reshape(-1, 1)
    y     = august_rain["rainfall"].values

    poly   = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X_poly, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Recover original year values from the polynomial matrix (column index 1)
    combined = build_result_df(
        X_train[:, 1], y_train, model.predict(X_train),
        X_test[:, 1],  y_test,  model.predict(X_test),
    )

    metrics = evaluate(f"Polynomial Regression (deg={degree})", combined)
    plot_actual_vs_predicted(august_rain, combined,
                             f"Polynomial Regression (deg={degree})")
    plot_error_bar(combined, f"Polynomial Regression (deg={degree})")

    return metrics


# =============================================================================
# SECTION 8 — MODEL 4: RANDOM FOREST REGRESSOR
# =============================================================================

def run_random_forest(august_rain: pd.DataFrame,
                      n_estimators: int = 30) -> dict:
    """
    Random Forest ensemble. oob_score=True gives a free out-of-bag estimate.
    Likely to overfit (train MAE ≈ 0) on this small dataset.
    """
    X = august_rain["year"].values.reshape(-1, 1)
    y = august_rain["rainfall"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model = RandomForestRegressor(n_estimators=n_estimators,
                                  random_state=RANDOM_STATE,
                                  oob_score=True)
    model.fit(X_train, y_train)

    print(f"[Random Forest] OOB R² score : {model.oob_score_:.4f}")

    combined = build_result_df(
        X_train.flatten(), y_train, model.predict(X_train),
        X_test.flatten(),  y_test,  model.predict(X_test),
    )

    metrics = evaluate("Random Forest", combined)
    plot_actual_vs_predicted(august_rain, combined, "Random Forest")
    plot_error_bar(combined, "Random Forest")

    return metrics


# =============================================================================
# SECTION 9 — MODEL COMPARISON & CONCLUSION
# =============================================================================

def plot_model_comparison(all_metrics: list[dict]) -> None:
    """
    Side-by-side horizontal bar chart comparing Test MAE and Test MAPE
    for all models.
    """
    comparison = pd.DataFrame(all_metrics).sort_values("mae_test")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # ── Test MAE ──────────────────────────────────────────────────────────────
    bars_mae = axes[0].barh(comparison["model"], comparison["mae_test"],
                            color=PALETTE["actual"], edgecolor="white")
    axes[0].bar_label(bars_mae, fmt="%.2f mm", padding=4)
    axes[0].set_title("Test MAE (lower is better)", fontweight="bold")
    axes[0].set_xlabel("Mean Absolute Error (mm)")
    axes[0].invert_yaxis()

    # ── Test MAPE ─────────────────────────────────────────────────────────────
    bars_mape = axes[1].barh(comparison["model"], comparison["mape_test"],
                             color=PALETTE["error"], edgecolor="white")
    axes[1].bar_label(bars_mape, fmt="%.1f%%", padding=4)
    axes[1].set_title("Test MAPE (lower is better)", fontweight="bold")
    axes[1].set_xlabel("Mean Absolute Percentage Error (%)")
    axes[1].invert_yaxis()

    plt.suptitle("Model Comparison — August Rainfall Prediction",
                 fontweight="bold", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()

    print("\n📊 Final Comparison Table")
    print(comparison[["model", "mae_train", "mae_test",
                       "mape_train", "mape_test", "r2_test"]].to_string(index=False))


# =============================================================================
# MAIN — ORCHESTRATES ALL SECTIONS
# =============================================================================

def main():
    # ── Mount Drive (Colab only) ───────────────────────────────────────────────
    try:
        from google.colab import drive
        drive.mount("/content/drive")
    except ImportError:
        print("[Info] Not running in Colab — skipping drive mount.")

    # ── Section 2: Load & clean ───────────────────────────────────────────────
    df = load_and_clean(FILE_PATH)

    # ── Section 3: August extraction & EDA ───────────────────────────────────
    august_rain = extract_august(df)
    plot_august_distribution(august_rain)
    plot_august_timeseries(august_rain)

    # ── Sections 5–8: Train models & collect metrics ──────────────────────────
    all_metrics = []

    print("\n" + "=" * 55)
    print("  MODEL TRAINING & EVALUATION")
    print("=" * 55)

    all_metrics.append(run_linear_regression(august_rain))
    all_metrics.append(run_svr(august_rain))
    all_metrics.append(run_polynomial_regression(august_rain, degree=10))
    all_metrics.append(run_random_forest(august_rain, n_estimators=30))

    # ── Section 9: Comparison ─────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  MODEL COMPARISON")
    print("=" * 55)
    plot_model_comparison(all_metrics)


if __name__ == "__main__":
    main()
