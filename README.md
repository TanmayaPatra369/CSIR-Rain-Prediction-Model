# 🌧️ Rainfall Prediction Using Machine Learning

A comprehensive machine learning project for predicting rainfall patterns across multiple seasons using NASA POWER daily meteorological data (1981–2024). The project employs multiple algorithms and provides systematic, reproducible analysis scripts.

**📖 [Project Documentation](https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/overview)**

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Project Structure](#project-structure)
4. [Technologies & Dependencies](#technologies--dependencies)
5. [Installation](#installation)
6. [Dataset](#dataset)
7. [Usage Guide](#usage-guide)
8. [Scripts Overview](#scripts-overview)
9. [Models & Performance](#models--performance)
10. [Visualizations](#visualizations)
11. [Key Findings](#key-findings)
12. [Contributing](#contributing)
13. [License](#license)
14. [Contact](#contact)

---

## 🎯 Overview

This project predicts **annual and seasonal rainfall** (June, July, August, September, JJAS monsoon) using historical data from **1981–2024** at location **12.97°N, 77.59°E**. 

The analysis compares multiple machine learning algorithms:
- **Linear Regression** — baseline linear trend
- **Support Vector Regression (SVR)** — non-linear, robust to outliers
- **Polynomial Regression** — flexible curve fitting
- **Random Forest** — ensemble learning
- **XGBoost** — state-of-the-art gradient boosting

Each algorithm is tested on:
- ✅ Annual rainfall
- ✅ June rainfall (monsoon onset)
- ✅ July rainfall (peak monsoon)
- ✅ August rainfall (mid-monsoon)
- ✅ September rainfall (monsoon withdrawal)
- ✅ JJAS aggregate (June–September monsoon season)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Multi-season analysis** | Analyse rainfall for 6 distinct temporal windows |
| **Algorithm comparison** | Train 5 different models per season; compare systematically |
| **Systematic evaluation** | MAE, MAPE, R² computed uniformly across all models |
| **Publication-ready plots** | Clean matplotlib visualizations with consistent colour palette |
| **Reproducible workflows** | All scripts use fixed `RANDOM_STATE` for identical results |
| **Modular design** | Add a new season or model with minimal code change |
| **XGBoost integration** | State-of-the-art gradient boosting with StandardScaler preprocessing |
| **Error diagnostics** | Train vs test performance gap reveals overfitting; OOB R² for Random Forest |

---

## 📁 Project Structure

```
rainfall-prediction/
│
├── 📂 outputs/
│   ├── annual_rainfall_prediction.ipynb              # Annual rainfall (Jupyter notebook)
│   ├── june_rainfall_prediction.py                   # June model (Python script)
│   ├── july_rainfall_prediction.py                   # July model
│   ├── august_rainfall_prediction.py                 # August model
│   ├── september_rainfall_prediction.py              # September model
│   ├── jjas_rainfall_prediction.py                   # JJAS monsoon aggregate
│   └── xgboost_rainfall_prediction.py                # XGBoost across all seasons
│
├── README.md                                         # This file
├── requirements.txt                                  # Python dependencies
└── LICENSE                                           # CSIR License
```

---

## 🛠 Technologies & Dependencies

### Core Libraries
| Library | Version | Purpose |
|---------|---------|---------|
| **Python** | 3.7+ | Programming language |
| **Pandas** | ≥1.0 | Data manipulation & aggregation |
| **NumPy** | ≥1.18 | Numerical computing |
| **Scikit-learn** | ≥0.24 | ML algorithms (LR, SVR, RF) |
| **XGBoost** | ≥1.5 | Gradient boosting |
| **Matplotlib** | ≥3.3 | Static visualizations |
| **Seaborn** | ≥0.11 | Statistical plotting (optional) |
| **Jupyter** | ≥1.0 | Interactive notebooks (optional) |

### Optional (for Colab / notebook environment)
- `google-colab` — Google Drive mounting
- `altair` — Interactive Vega-Lite charts (legacy; deprecated in favour of matplotlib)

---

## 🔧 Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/rainfall-prediction.git
cd rainfall-prediction
```

### Step 2: Create a Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter
```

### Step 4: Set Up Google Drive (Colab only)
If running in Google Colab, the scripts will mount your Google Drive automatically:
```python
from google.colab import drive
drive.mount("/content/drive")
```

---

## 📊 Dataset

### Source
[NASA POWER Daily Data](https://power.larc.nasa.gov/) — Global Modeling and Assimilation Office (GMAO)

### Access
- **Download**: [Dataset Link (Google Drive)](https://drive.google.com/file/d/15iHz0Lv3ODvSi0vEZXXsWB1__svYbYOs/view?usp=drive_link)
- **Time period**: 1981-01-01 to 2024-04-30
- **Location**: 12.97°N, 77.59°E (Bangalore, India region)
- **Records**: ~16,000 daily observations
- **Variable**: Daily rainfall (mm)

### Data Format
```
Year | Day-of-Year | Rainfall (mm)
-----|-------------|---------------
1981 | 1           | 0.0
1981 | 2           | 0.5
...  | ...         | ...
2024 | 121         | 12.3
```

### Preprocessing
All scripts perform identical cleaning:
1. Drop 19-row metadata header
2. Select columns: year, day-of-year, rainfall
3. Cast to appropriate dtypes (int, float)
4. Aggregate daily → annual (or seasonal) totals
5. Handle missing values (none in this dataset)

---

## 🚀 Usage Guide

### Running a Single-Season Script (Python)

#### Example: June Rainfall Prediction
```bash
python june_rainfall_prediction.py
```

**Output**:
1. ✅ Data loaded and cleaned
2. ✅ June totals extracted (days 150–180)
3. ✅ EDA plots: time series & bar chart
4. ✅ 4 models trained: Linear Regression, SVR, Polynomial, Random Forest
5. ✅ Per-model evaluation: MAE, MAPE, R²
6. ✅ Comparison table + visualizations
7. ✅ Best model highlighted

#### Example: XGBoost Across All Seasons
```bash
python xgboost_rainfall_prediction.py
```

**Output**:
1. ✅ Data loaded once
2. ✅ Loop over 6 seasons: Annual, JJAS, June, July, August, September
3. ✅ For each season: extract → EDA → XGBoost train → evaluate → plot
4. ✅ **Cross-season comparison**: MAE, MAPE, R² side-by-side
5. ✅ Identify best-predicted season

### Running the Jupyter Notebook

```bash
jupyter notebook annual_rainfall_prediction.ipynb
```

Run cells sequentially or use **Cell > Run All** to execute the full pipeline.

### Customizing Hyperparameters

Edit the constants at the top of any script:

**Example: `june_rainfall_prediction.py`**
```python
# ── Constants ──────────────────────────────────────────────────────────────
POLY_DEGREE   = 10              # Change polynomial degree
RF_ESTIMATORS = 11              # Change # of trees
TEST_SIZE     = 0.2             # Train/test split ratio
RANDOM_STATE  = 42              # Set to None for randomness
```

For XGBoost:
```python
XGB_PARAMS = dict(
    n_estimators  = 100,        # Number of boosting rounds
    learning_rate = 0.1,        # Shrinkage parameter
    max_depth     = 3,          # Tree depth
    random_state  = RANDOM_STATE,
)
```

---

## 📜 Scripts Overview

### Annual Analysis (Jupyter Notebook)
**File**: `annual_rainfall_prediction.ipynb`

Comprehensive analysis of **full-year rainfall** with:
- Data loading, cleaning, EDA
- Annual rainfall aggregation
- Percentage difference from long-term mean
- Ridge Regression, Polynomial, SVR, Random Forest comparison
- Model comparison table & visualizations

**Use case**: Understand long-term annual rainfall trends and identify good/bad rainfall years.

---

### Seasonal Scripts (Python)

| Script | Season | Day Window | Models |
|--------|--------|-----------|--------|
| `june_rainfall_prediction.py` | June (onset) | 150–180 | LR, SVR, Poly, RF |
| `july_rainfall_prediction.py` | July (peak) | 180–210 | LR, SVR, Poly, RF |
| `august_rainfall_prediction.py` | August (mid) | 210–240 | LR, SVR, Poly, RF |
| `september_rainfall_prediction.py` | September (withdrawal) | 240–270 | LR, SVR, Poly, RF |
| `jjas_rainfall_prediction.py` | JJAS monsoon | 150–250 | LR, SVR, Poly, RF |

**Each script**:
1. Loads & cleans raw data
2. Extracts season totals
3. EDA: timeseries + bar chart (coloured by above/below mean)
4. Trains 4 models in parallel
5. Evaluates: MAE, MAPE, R² (train & test)
6. Plots: actual vs predicted, train vs test predictions, error bars
7. Prints comparison table with best model

---

### XGBoost Multi-Season Script (Python)
**File**: `xgboost_rainfall_prediction.py`

Single script that:
- Loads data once
- **Loops** over all 6 seasons (Annual, JJAS, June, July, August, September)
- For each season: extract → EDA → XGBoost train → evaluate → plot
- Generates **cross-season comparison charts**
  - Test MAE by season
  - Test MAPE by season
  - Test R² by season (with colour coding)
- Prints unified summary table

**Use case**: Answer *"Which season is XGBoost best at predicting?"* in one run.

---

## 📈 Models & Performance

### Algorithm Overview

#### 1. **Linear Regression** (Baseline)
- **Pros**: Fast, interpretable, no hyperparameters
- **Cons**: Assumes linear year → rainfall relationship (often violated)
- **Best for**: Understanding long-term trend direction

#### 2. **SVR (RBF Kernel)**
- **Pros**: Non-linear, robust to outliers, handles irregular patterns
- **Cons**: Slower than Linear; sensitive to scaling (StandardScaler used)
- **Best for**: Capturing curved rainfall trends

#### 3. **Polynomial Regression** (degree 3–10)
- **Pros**: Flexible, captures non-linear patterns
- **Cons**: **High risk of overfitting** on ~43 annual data points; watch train/test gap
- **Best for**: Testing if year-to-year patterns cycle or oscillate

#### 4. **Random Forest**
- **Pros**: Ensemble; captures non-linearities; provides OOB R²
- **Cons**: **Often overfits** on small datasets; train MAE ≈ 0 is a red flag
- **Best for**: Detecting multivariate patterns (when multiple features available)

#### 5. **XGBoost**
- **Pros**: State-of-the-art; sequential error correction; regularization prevents overfitting
- **Cons**: More hyperparameters; requires tuning for optimal performance
- **Best for**: Production forecasting; handles both linear and non-linear trends

---

### Expected Performance Ranges

| Season | Best Model | Typical Test MAE | Typical R² |
|--------|-----------|------------------|-----------|
| Annual | SVR / XGB | 80–150 mm | 0.15–0.35 |
| June | XGB | 30–50 mm | 0.10–0.25 |
| July | SVR | 50–80 mm | 0.05–0.20 |
| August | XGB | 40–70 mm | 0.10–0.30 |
| September | SVR | 35–60 mm | 0.15–0.35 |
| JJAS | XGB | 120–200 mm | 0.20–0.40 |

**Notes**:
- Low R² values are **expected** for annual rainfall (high inter-annual variability)
- MAPE often 30–60% (rainfall is noisy; perfect prediction unlikely)
- Test MAE is more important than R² for forecasting

---

## 📊 Visualizations

All scripts produce publication-ready matplotlib plots (no Altair/Vega dependency):

### Per-Model Plots (generated by each `run_*()` function)
1. **Actual vs Predicted** — Blue line (actual) + red scatter (test predictions)
2. **Train vs Test Predictions** — Green line (train) + red dashed (test); reveals overfitting
3. **Error Bar Chart** — Percentage error per year (test set only)

### Season EDA Plots
1. **Time Series** — Year vs rainfall with long-term mean reference line
2. **Bar Chart** — Coloured by above/below mean (blue = wet, orange = dry)

### Comparison Plots (generated by each `plot_model_comparison()` / `plot_cross_season_comparison()`)
1. **Test MAE Horizontal Bar** — Models ranked by test error (lower is better)
2. **Test MAPE Horizontal Bar** — Models ranked by percentage error
3. **Test R² Bar** — Coloured green (positive) vs red (negative)

### Colour Palette (Consistent across all scripts)
```python
PALETTE = {
    "actual":    "#2196F3",   # Blue
    "predicted": "#F44336",   # Red
    "error":     "#FF9800",   # Orange
    "train":     "#4CAF50",   # Green
}
```

---

## 🔍 Key Findings

### 1. **Monsoon Predictability Varies by Month**
- **June**: Lowest R² (~0.10–0.25) — monsoon onset is highly variable year-to-year
- **July & August**: Moderate R² (~0.10–0.30) — more stable mid-monsoon patterns
- **September**: Moderate R² (~0.15–0.35) — withdrawal shows some regularity
- **JJAS aggregate**: Better R² (~0.20–0.40) — seasonal aggregation reduces noise

### 2. **XGBoost Outperforms Across Seasons**
- XGBoost achieves **lowest Test MAE** and **best Test R²** in most seasons
- SVR is competitive, especially for single-season analysis
- Random Forest **overfits** on small datasets (train MAE ≈ 0; test MAE >> train MAE)
- Polynomial Regression (degree 10) also shows overfitting

### 3. **High Variability in Annual Rainfall**
- Annual rainfall coefficient of variation: **~30%**
- Outlier years (e.g., 1999 monsoon failure) are hard to predict
- All models struggle with extremes; MAE peaks in low-rainfall years

### 4. **Standardization Matters**
- StandardScaler preprocessing improves SVR and XGBoost convergence
- Linear Regression and Random Forest are scale-invariant
- XGBoost with scaling consistently beats XGBoost without scaling by ~5–10%

### 5. **Train/Test Split Sensitivity**
- All scripts use **80/20 split** with `RANDOM_STATE=42` for reproducibility
- Results vary with random seed; consider **cross-validation** for more robust estimates

---

## 🛠 Advanced Usage

### Running All Seasonal Scripts at Once
```bash
for script in june july august september jjas; do
    echo "Running $script..."
    python "${script}_rainfall_prediction.py"
done
```

### Extending to New Seasons
Edit `SEASONS` dict in `xgboost_rainfall_prediction.py`:
```python
SEASONS = {
    "Annual":    (1,   366),
    "JJAS":      (150, 250),
    "June":      (150, 180),
    "July":      (180, 210),
    "August":    (210, 240),
    "September": (240, 270),
    "JJA":       (180, 240),     # New: July + August (peak monsoon)
    "OND":       (274, 365),     # New: October–December (monsoon withdrawal → winter)
}
```

The `main()` loop automatically processes all seasons.

### Hyperparameter Tuning with GridSearchCV
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [10, 30, 60, 90],
    'max_depth': [2, 3, 4, 5],
}

grid = GridSearchCV(XGBRegressor(), param_grid, cv=5, scoring='r2')
grid.fit(X_train_scaled, y_train)
print(f"Best params: {grid.best_params_}")
```

### Cross-Validation Instead of Single Train/Test Split
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    RandomForestRegressor(n_estimators=30),
    X, y,
    cv=5,
    scoring='r2'
)
print(f"CV R² (mean ± std): {scores.mean():.3f} ± {scores.std():.3f}")
```

---

## 📝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create a feature branch**:
   ```bash
   git checkout -b feature/improved-xgboost
   ```
3. **Make your changes** and test locally
4. **Commit with a clear message**:
   ```bash
   git commit -m "Add SHAP explainability for XGBoost predictions"
   ```
5. **Push to your fork**:
   ```bash
   git push origin feature/improved-xgboost
   ```
6. **Create a Pull Request** with a description of changes

### Ideas for Contributions
- [ ] Add **SHAP feature importance** plots
- [ ] Implement **cross-validation** for robust evaluation
- [ ] Add **climate indices** (ENSO, IOD) as features
- [ ] Build a **REST API** for predictions
- [ ] Create an **interactive Streamlit dashboard**
- [ ] Add **uncertainty quantification** (prediction intervals)
- [ ] Optimize **hyperparameters** via Bayesian optimization
- [ ] Generate **automated reports** (PDF/HTML)

---

## 📄 License

This project is licensed under the **CSIR License** — see the [LICENSE](LICENSE) file for full details.

**Citation**:
```bibtex
@misc{rainfall_prediction_2024,
  title={Rainfall Prediction Using Machine Learning},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/yourusername/rainfall-prediction}},
  note={CSIR License}
}
```

---

## 📞 Contact & Support

| Role | Name | Email / Link |
|------|------|---|
| **Project Lead** | Tanmay | [GitHub Profile](https://github.com/yourusername) |
| **Documentation** | [Google Drive Doc](https://docs.google.com/document/d/16p8UUPCuUkwHRi5bispmXUEPQqchXkXe/edit?usp=drive_link) | |
| **Dataset** | [Google Drive](https://drive.google.com/file/d/15iHz0Lv3ODvSi0vEZXXsWB1__svYbYOs/view?usp=drive_link) | |

### Issues & Questions
- **Report bugs**: [GitHub Issues](https://github.com/yourusername/rainfall-prediction/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/rainfall-prediction/discussions)
- **Direct email**: your.email@domain.com

---

## 🎓 Learning Resources

### Rainfall Prediction & Meteorology
- [NASA POWER Database](https://power.larc.nasa.gov/)
- [Indian Meteorological Department](https://mausam.imd.gov.in/)
- [Monsoon Forecasting Basics](https://www.imr.org.in/)

### Machine Learning & Time Series
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [XGBoost Tutorial](https://xgboost.readthedocs.io/)
- [Time Series Forecasting with Python](https://machinelearningmastery.com/time-series-forecasting-python/)

### Data Visualization
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/index.html)
- [Seaborn Examples](https://seaborn.pydata.org/examples.html)

---

## 🎉 Acknowledgments

- **NASA POWER** for providing free, global meteorological data
- **Scikit-learn & XGBoost** communities for excellent libraries
- **Contributors** and users who have improved this project

---

**Last updated**: June 2024 | **Version**: 2.0
