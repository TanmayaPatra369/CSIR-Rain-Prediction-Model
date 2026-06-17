# =============================================================================
# XGBoost Rainfall Prediction — Multi-Season Analysis (1981–2024)
# Seasons : Annual | JJAS | June | July | August | September
# Dataset : NASA POWER Daily | Location: 12.97°N, 77.59°E
# Model   : XGBoost Regressor (with StandardScaler preprocessing)
# =============================================================================

# =============================================================================
# SECTION 1 — IMPORTS & GLOBAL CONFIG
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

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

# XGBoost hyperparameters (shared across all seasons)
XGB_PARAMS = dict(
    n_estimators  = 100,
    learning_rate = 0.1,
    max_depth     = 3,
    random_state  = RANDOM_STATE,
)

# Season day-of-year windows
SEASONS = {
    "Annual":    (1,   366),
    "JJAS":      (150, 250),
    "June":      (150, 180),
    "July":      (180, 210),
    "August":    (210, 240),
    "September": (240, 270),
}

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
# SECTION 3 — SEASON EXTRACTION
# =============================================================================

def extract_season(df: pd.DataFrame,
                   season_name: str,
                   start_day: int,
                   end_day: int) -> pd.DataFrame:
    """
    Filter daily records to a day-of-year window and aggregate
    to annual totals for that season.

    Parameters
    ----------
    df          : cleaned daily DataFrame
    season_name : label used in print output
    start_day   : first day-of-year (inclusive)
    end_day     : last  day-of-year (inclusive)

    Returns
    -------
    season_df : DataFrame with columns [year, rainfall]
    """
    if start_day == 1 and end_day == 366:
        # Annual — aggregate all days
        season_df = df.groupby("year")["rainfall"].sum().reset_index()
    else:
        mask      = (df["day_no"] >= start_day) & (df["day_no"] <= end_day)
        season_df = (df[mask]
                     .groupby("year")["rainfall"]
                     .sum()
                     .reset_index())

    season_df.columns = ["year", "rainfall"]

    print(f"[{season_name}] Years  : "
          f"{season_df['year'].min()} – {season_df['year'].max()}")
    print(f"[{season_name}] Mean   : {season_df['rainfall'].mean():.2f} mm")
    print(f"[{season_name}] Std    : {season_df['rainfall'].std():.2f} mm")
    print(f"[{season_name}] Wettest: "
          f"{season_df.loc[season_df['rainfall'].idxmax(), 'year']}  "
          f"({season_df['rainfall'].max():.1f} mm)")
    print(f"[{season_name}] Driest : "
          f"{season_df.loc[season_df['rainfall'].idxmin(), 'year']}  "
          f"({season_df['rainfall'].min():.1f} mm)\n")
    return season_df


# =============================================================================
# SECTION 4 — EDA PLOTS
# =============================================================================

def plot_season_timeseries(season_df: pd.DataFrame, season_name: str) -> None:
    """Line chart of annual rainfall with long-term mean reference."""
    fig, ax = plt.subplots()
    ax.plot(season_df["year"], season_df["rainfall"],
            color=PALETTE["actual"], linewidth=1.8,
            marker="o", markersize=4, label=f"{season_name} Rainfall")
    ax.axhline(season_df["rainfall"].mean(), color="black", linestyle="--",
               linewidth=1.2, label="Long-term mean")
    ax.set_title(f"Annual {season_name} Rainfall (1981–2024)",
                 fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Rainfall (mm)")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_season_bar(season_df: pd.DataFrame, season_name: str) -> None:
    """Bar chart coloured by above / below long-term mean."""
    mean_val = season_df["rainfall"].mean()
    colors   = [
        PALETTE["actual"] if v >= mean_val else PALETTE["error"]
        for v in season_df["rainfall"]
    ]
    fig, ax = plt.subplots()
    ax.bar(season_df["year"], season_df["rainfall"],
           color=colors, edgecolor="white", linewidth=0.4, width=0.8)
    ax.axhline(mean_val, color="black", linestyle="--",
               linewidth=1.2, label="Long-term mean")
    ax.legend(handles=[
        mpatches.Patch(color=PALETTE["actual"], label="Above / at mean"),
        mpatches.Patch(color=PALETTE["error"],  label="Below mean"),
        mpatches.Patch(color="black",            label="Mean line"),
    ])
    ax.set_title(f"Yearly {season_name} Rainfall", fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Rainfall (mm)")
    plt.tight_layout()
    plt.show()


# =============================================================================
# SECTION 5 — SHARED MODEL UTILITIES
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


def evaluate(season_name: str, combined: pd.DataFrame) -> dict:
    """
    Print and return MAE, MAPE, and R² for train and test splits.

    Returns
    -------
    metrics : dict
    """
    train = combined[combined["Split"] == "train"]
    test  = combined[combined["Split"] == "test"]

    metrics = {
        "season":     season_name,
        "mae_train":  train["Absolute Error"].mean(),
        "mape_train": train["Percentage Error"].mean(),
        "mae_test":   test["Absolute Error"].mean(),
        "mape_test":  test["Percentage Error"].mean(),
        "r2_test":    r2_score(test["Actual Rainfall"],
                               test["Predicted Rainfall"]),
    }

    print(f"\n{'─' * 58}")
    print(f"  XGBoost — {season_name}")
    print(f"  Train  MAE  : {metrics['mae_train']:.2f} mm  |  "
          f"Train MAPE : {metrics['mape_train']:.2f}%")
    print(f"  Test   MAE  : {metrics['mae_test']:.2f} mm  |  "
          f"Test  MAPE : {metrics['mape_test']:.2f}%")
    print(f"  Test   R²   : {metrics['r2_test']:.4f}")
    print(f"{'─' * 58}")
    return metrics


def plot_actual_vs_predicted(season_df: pd.DataFrame,
                             combined: pd.DataFrame,
                             season_name: str) -> None:
    """
    Full actual series (blue line) overlaid with
    test-set predictions (red scatter + dashed line).
    """
    test = combined[combined["Split"] == "test"].sort_values("Year")

    fig, ax = plt.subplots()
    ax.plot(season_df["year"], season_df["rainfall"],
            color=PALETTE["actual"], linewidth=1.8,
            marker="o", markersize=4, label="Actual")
    ax.plot(test["Year"], test["Predicted Rainfall"],
            color=PALETTE["predicted"], linestyle="--",
            linewidth=1.4, alpha=0.7)
    ax.scatter(test["Year"], test["Predicted Rainfall"],
               color=PALETTE["predicted"], s=80, zorder=5,
               marker="x", linewidths=2, label="Predicted (test)")
    ax.set_title(f"Actual vs Predicted — XGBoost | {season_name}",
                 fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel(f"{season_name} Rainfall (mm)")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_train_test_predictions(combined: pd.DataFrame,
                                season_name: str) -> None:
    """
    Line chart of predicted rainfall for train and test splits —
    used to diagnose overfitting visually.
    """
    train = combined[combined["Split"] == "train"].sort_values("Year")
    test  = combined[combined["Split"] == "test"].sort_values("Year")

    fig, ax = plt.subplots()
    ax.plot(train["Year"], train["Predicted Rainfall"],
            color=PALETTE["train"], linewidth=1.6, label="Train predictions")
    ax.plot(test["Year"], test["Predicted Rainfall"],
            color=PALETTE["predicted"], linewidth=1.6,
            linestyle="--", label="Test predictions")
    ax.set_title(f"Train vs Test Predictions — XGBoost | {season_name}",
                 fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel(f"Predicted {season_name} Rainfall (mm)")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_error_bar(combined: pd.DataFrame, season_name: str) -> None:
    """Bar chart of percentage error per year (test set only)."""
    test = combined[combined["Split"] == "test"].sort_values("Year")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(test["Year"], test["Percentage Error"],
           color=PALETTE["error"], edgecolor="white", width=0.7)
    ax.set_title(
        f"Percentage Error per Year (Test Set) — XGBoost | {season_name}",
        fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("% Error")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# =============================================================================
# SECTION 6 — XGBOOST RUNNER (single season)
# =============================================================================

def run_xgboost(season_df: pd.DataFrame, season_name: str) -> dict:
    """
    Fit an XGBoost regressor on year → rainfall for one season.

    Steps
    -----
    1. Train / test split (80 / 20, stratified by RANDOM_STATE)
    2. StandardScaler normalisation (fit on train, applied to both)
    3. XGBRegressor fit + predict
    4. Build tidy result DataFrame
    5. Evaluate & plot

    Returns
    -------
    metrics : dict  (season, mae_train, mape_train, mae_test, mape_test, r2_test)
    """
    X = season_df[["year"]]
    y = season_df["rainfall"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # ── Scale features ────────────────────────────────────────────────────────
    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # ── Train ─────────────────────────────────────────────────────────────────
    model = XGBRegressor(**XGB_PARAMS)
    model.fit(X_train_scaled, y_train)

    y_pred_train = model.predict(X_train_scaled)
    y_pred_test  = model.predict(X_test_scaled)

    # ── Results ───────────────────────────────────────────────────────────────
    combined = build_result_df(
        X_train["year"].values, y_train, y_pred_train,
        X_test["year"].values,  y_test,  y_pred_test,
    )

    metrics = evaluate(season_name, combined)
    plot_actual_vs_predicted(season_df, combined, season_name)
    plot_train_test_predictions(combined, season_name)
    plot_error_bar(combined, season_name)
    return metrics


# =============================================================================
# SECTION 7 — CROSS-SEASON COMPARISON
# =============================================================================

def plot_cross_season_comparison(all_metrics: list) -> None:
    """
    Compare XGBoost performance across all seasons using side-by-side
    horizontal bar charts for Test MAE and Test MAPE, plus a summary table.
    """
    comparison = (
        pd.DataFrame(all_metrics)
          .sort_values("mae_test")
          .reset_index(drop=True)
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Test MAE ──────────────────────────────────────────────────────────────
    bars = axes[0].barh(comparison["season"], comparison["mae_test"],
                        color=PALETTE["actual"], edgecolor="white")
    axes[0].bar_label(bars, fmt="%.2f mm", padding=4)
    axes[0].set_title("Test MAE  (lower is better)", fontweight="bold")
    axes[0].set_xlabel("Mean Absolute Error (mm)")
    axes[0].invert_yaxis()

    # ── Test MAPE ─────────────────────────────────────────────────────────────
    bars2 = axes[1].barh(comparison["season"], comparison["mape_test"],
                         color=PALETTE["error"], edgecolor="white")
    axes[1].bar_label(bars2, fmt="%.1f%%", padding=4)
    axes[1].set_title("Test MAPE  (lower is better)", fontweight="bold")
    axes[1].set_xlabel("Mean Absolute Percentage Error (%)")
    axes[1].invert_yaxis()

    plt.suptitle("XGBoost — Cross-Season Performance Comparison",
                 fontweight="bold", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()

    # ── R² bar chart ──────────────────────────────────────────────────────────
    r2_sorted = comparison.sort_values("r2_test", ascending=True)
    fig, ax   = plt.subplots(figsize=(10, 4))
    r2_colors = [
        PALETTE["actual"] if v >= 0 else PALETTE["error"]
        for v in r2_sorted["r2_test"]
    ]
    bars3 = ax.barh(r2_sorted["season"], r2_sorted["r2_test"],
                    color=r2_colors, edgecolor="white")
    ax.bar_label(bars3, fmt="%.4f", padding=4)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Test R²  (higher is better)", fontweight="bold")
    ax.set_xlabel("R² Score")
    plt.tight_layout()
    plt.show()

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n📊 XGBoost — Cross-Season Summary Table")
    print("=" * 76)
    print(comparison[["season", "mae_train", "mae_test",
                       "mape_train", "mape_test", "r2_test"]]
          .rename(columns={
              "season":     "Season",
              "mae_train":  "Train MAE",
              "mae_test":   "Test MAE",
              "mape_train": "Train MAPE%",
              "mape_test":  "Test MAPE%",
              "r2_test":    "Test R²",
          })
          .to_string(index=False))
    print("=" * 76)

    best = comparison.iloc[0]
    print(f"\n✅ Best season by Test MAE : {best['season']}  "
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

    # ── Section 2: Load & clean raw daily data ────────────────────────────────
    df = load_and_clean(FILE_PATH)

    # ── Sections 3–6: Loop over all seasons ──────────────────────────────────
    print("\n" + "=" * 58)
    print("  XGBOOST — MULTI-SEASON TRAINING & EVALUATION")
    print("=" * 58)

    all_metrics = []

    for season_name, (start_day, end_day) in SEASONS.items():
        print(f"\n{'━' * 58}")
        print(f"  SEASON : {season_name}  (days {start_day}–{end_day})")
        print(f"{'━' * 58}")

        # Extract season totals
        season_df = extract_season(df, season_name, start_day, end_day)

        # EDA
        plot_season_timeseries(season_df, season_name)
        plot_season_bar(season_df, season_name)

        # Train XGBoost and collect metrics
        metrics = run_xgboost(season_df, season_name)
        all_metrics.append(metrics)

    # ── Section 7: Cross-season comparison ───────────────────────────────────
    print("\n" + "=" * 58)
    print("  CROSS-SEASON COMPARISON")
    print("=" * 58)
    plot_cross_season_comparison(all_metrics)


if __name__ == "__main__":
    main()
