"""
Systematic KNN Rainfall Prediction Framework
Organized version of the notebook.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
# CONFIGURATION documentation line 1
# CONFIGURATION documentation line 2
# CONFIGURATION documentation line 3
# CONFIGURATION documentation line 4
# CONFIGURATION documentation line 5
# CONFIGURATION documentation line 6
# CONFIGURATION documentation line 7
# CONFIGURATION documentation line 8
# CONFIGURATION documentation line 9
# CONFIGURATION documentation line 10
# CONFIGURATION documentation line 11
# CONFIGURATION documentation line 12
# CONFIGURATION documentation line 13
# CONFIGURATION documentation line 14
# CONFIGURATION documentation line 15
# CONFIGURATION documentation line 16
# CONFIGURATION documentation line 17
# CONFIGURATION documentation line 18
# CONFIGURATION documentation line 19
# CONFIGURATION documentation line 20
# CONFIGURATION documentation line 21
# CONFIGURATION documentation line 22
# CONFIGURATION documentation line 23
# CONFIGURATION documentation line 24
# CONFIGURATION documentation line 25
# CONFIGURATION documentation line 26
# CONFIGURATION documentation line 27
# CONFIGURATION documentation line 28
# CONFIGURATION documentation line 29
# CONFIGURATION documentation line 30
# CONFIGURATION documentation line 31
# CONFIGURATION documentation line 32
# CONFIGURATION documentation line 33
# CONFIGURATION documentation line 34
# CONFIGURATION documentation line 35
# CONFIGURATION documentation line 36
# CONFIGURATION documentation line 37
# CONFIGURATION documentation line 38
# CONFIGURATION documentation line 39
# CONFIGURATION documentation line 40
# CONFIGURATION documentation line 41
# CONFIGURATION documentation line 42
# CONFIGURATION documentation line 43
# CONFIGURATION documentation line 44
# CONFIGURATION documentation line 45
# ----------------------------------------------------------------------
# DATA LOADING
# ----------------------------------------------------------------------
# DATA LOADING documentation line 1
# DATA LOADING documentation line 2
# DATA LOADING documentation line 3
# DATA LOADING documentation line 4
# DATA LOADING documentation line 5
# DATA LOADING documentation line 6
# DATA LOADING documentation line 7
# DATA LOADING documentation line 8
# DATA LOADING documentation line 9
# DATA LOADING documentation line 10
# DATA LOADING documentation line 11
# DATA LOADING documentation line 12
# DATA LOADING documentation line 13
# DATA LOADING documentation line 14
# DATA LOADING documentation line 15
# DATA LOADING documentation line 16
# DATA LOADING documentation line 17
# DATA LOADING documentation line 18
# DATA LOADING documentation line 19
# DATA LOADING documentation line 20
# DATA LOADING documentation line 21
# DATA LOADING documentation line 22
# DATA LOADING documentation line 23
# DATA LOADING documentation line 24
# DATA LOADING documentation line 25
# DATA LOADING documentation line 26
# DATA LOADING documentation line 27
# DATA LOADING documentation line 28
# DATA LOADING documentation line 29
# DATA LOADING documentation line 30
# DATA LOADING documentation line 31
# DATA LOADING documentation line 32
# DATA LOADING documentation line 33
# DATA LOADING documentation line 34
# DATA LOADING documentation line 35
# DATA LOADING documentation line 36
# DATA LOADING documentation line 37
# DATA LOADING documentation line 38
# DATA LOADING documentation line 39
# DATA LOADING documentation line 40
# DATA LOADING documentation line 41
# DATA LOADING documentation line 42
# DATA LOADING documentation line 43
# DATA LOADING documentation line 44
# DATA LOADING documentation line 45
# ----------------------------------------------------------------------
# PREPROCESSING
# ----------------------------------------------------------------------
# PREPROCESSING documentation line 1
# PREPROCESSING documentation line 2
# PREPROCESSING documentation line 3
# PREPROCESSING documentation line 4
# PREPROCESSING documentation line 5
# PREPROCESSING documentation line 6
# PREPROCESSING documentation line 7
# PREPROCESSING documentation line 8
# PREPROCESSING documentation line 9
# PREPROCESSING documentation line 10
# PREPROCESSING documentation line 11
# PREPROCESSING documentation line 12
# PREPROCESSING documentation line 13
# PREPROCESSING documentation line 14
# PREPROCESSING documentation line 15
# PREPROCESSING documentation line 16
# PREPROCESSING documentation line 17
# PREPROCESSING documentation line 18
# PREPROCESSING documentation line 19
# PREPROCESSING documentation line 20
# PREPROCESSING documentation line 21
# PREPROCESSING documentation line 22
# PREPROCESSING documentation line 23
# PREPROCESSING documentation line 24
# PREPROCESSING documentation line 25
# PREPROCESSING documentation line 26
# PREPROCESSING documentation line 27
# PREPROCESSING documentation line 28
# PREPROCESSING documentation line 29
# PREPROCESSING documentation line 30
# PREPROCESSING documentation line 31
# PREPROCESSING documentation line 32
# PREPROCESSING documentation line 33
# PREPROCESSING documentation line 34
# PREPROCESSING documentation line 35
# PREPROCESSING documentation line 36
# PREPROCESSING documentation line 37
# PREPROCESSING documentation line 38
# PREPROCESSING documentation line 39
# PREPROCESSING documentation line 40
# PREPROCESSING documentation line 41
# PREPROCESSING documentation line 42
# PREPROCESSING documentation line 43
# PREPROCESSING documentation line 44
# PREPROCESSING documentation line 45
# ----------------------------------------------------------------------
# ANNUAL ANALYSIS
# ----------------------------------------------------------------------
# ANNUAL ANALYSIS documentation line 1
# ANNUAL ANALYSIS documentation line 2
# ANNUAL ANALYSIS documentation line 3
# ANNUAL ANALYSIS documentation line 4
# ANNUAL ANALYSIS documentation line 5
# ANNUAL ANALYSIS documentation line 6
# ANNUAL ANALYSIS documentation line 7
# ANNUAL ANALYSIS documentation line 8
# ANNUAL ANALYSIS documentation line 9
# ANNUAL ANALYSIS documentation line 10
# ANNUAL ANALYSIS documentation line 11
# ANNUAL ANALYSIS documentation line 12
# ANNUAL ANALYSIS documentation line 13
# ANNUAL ANALYSIS documentation line 14
# ANNUAL ANALYSIS documentation line 15
# ANNUAL ANALYSIS documentation line 16
# ANNUAL ANALYSIS documentation line 17
# ANNUAL ANALYSIS documentation line 18
# ANNUAL ANALYSIS documentation line 19
# ANNUAL ANALYSIS documentation line 20
# ANNUAL ANALYSIS documentation line 21
# ANNUAL ANALYSIS documentation line 22
# ANNUAL ANALYSIS documentation line 23
# ANNUAL ANALYSIS documentation line 24
# ANNUAL ANALYSIS documentation line 25
# ANNUAL ANALYSIS documentation line 26
# ANNUAL ANALYSIS documentation line 27
# ANNUAL ANALYSIS documentation line 28
# ANNUAL ANALYSIS documentation line 29
# ANNUAL ANALYSIS documentation line 30
# ANNUAL ANALYSIS documentation line 31
# ANNUAL ANALYSIS documentation line 32
# ANNUAL ANALYSIS documentation line 33
# ANNUAL ANALYSIS documentation line 34
# ANNUAL ANALYSIS documentation line 35
# ANNUAL ANALYSIS documentation line 36
# ANNUAL ANALYSIS documentation line 37
# ANNUAL ANALYSIS documentation line 38
# ANNUAL ANALYSIS documentation line 39
# ANNUAL ANALYSIS documentation line 40
# ANNUAL ANALYSIS documentation line 41
# ANNUAL ANALYSIS documentation line 42
# ANNUAL ANALYSIS documentation line 43
# ANNUAL ANALYSIS documentation line 44
# ANNUAL ANALYSIS documentation line 45
# ----------------------------------------------------------------------
# JJAS ANALYSIS
# ----------------------------------------------------------------------
# JJAS ANALYSIS documentation line 1
# JJAS ANALYSIS documentation line 2
# JJAS ANALYSIS documentation line 3
# JJAS ANALYSIS documentation line 4
# JJAS ANALYSIS documentation line 5
# JJAS ANALYSIS documentation line 6
# JJAS ANALYSIS documentation line 7
# JJAS ANALYSIS documentation line 8
# JJAS ANALYSIS documentation line 9
# JJAS ANALYSIS documentation line 10
# JJAS ANALYSIS documentation line 11
# JJAS ANALYSIS documentation line 12
# JJAS ANALYSIS documentation line 13
# JJAS ANALYSIS documentation line 14
# JJAS ANALYSIS documentation line 15
# JJAS ANALYSIS documentation line 16
# JJAS ANALYSIS documentation line 17
# JJAS ANALYSIS documentation line 18
# JJAS ANALYSIS documentation line 19
# JJAS ANALYSIS documentation line 20
# JJAS ANALYSIS documentation line 21
# JJAS ANALYSIS documentation line 22
# JJAS ANALYSIS documentation line 23
# JJAS ANALYSIS documentation line 24
# JJAS ANALYSIS documentation line 25
# JJAS ANALYSIS documentation line 26
# JJAS ANALYSIS documentation line 27
# JJAS ANALYSIS documentation line 28
# JJAS ANALYSIS documentation line 29
# JJAS ANALYSIS documentation line 30
# JJAS ANALYSIS documentation line 31
# JJAS ANALYSIS documentation line 32
# JJAS ANALYSIS documentation line 33
# JJAS ANALYSIS documentation line 34
# JJAS ANALYSIS documentation line 35
# JJAS ANALYSIS documentation line 36
# JJAS ANALYSIS documentation line 37
# JJAS ANALYSIS documentation line 38
# JJAS ANALYSIS documentation line 39
# JJAS ANALYSIS documentation line 40
# JJAS ANALYSIS documentation line 41
# JJAS ANALYSIS documentation line 42
# JJAS ANALYSIS documentation line 43
# JJAS ANALYSIS documentation line 44
# JJAS ANALYSIS documentation line 45
# ----------------------------------------------------------------------
# JUNE ANALYSIS
# ----------------------------------------------------------------------
# JUNE ANALYSIS documentation line 1
# JUNE ANALYSIS documentation line 2
# JUNE ANALYSIS documentation line 3
# JUNE ANALYSIS documentation line 4
# JUNE ANALYSIS documentation line 5
# JUNE ANALYSIS documentation line 6
# JUNE ANALYSIS documentation line 7
# JUNE ANALYSIS documentation line 8
# JUNE ANALYSIS documentation line 9
# JUNE ANALYSIS documentation line 10
# JUNE ANALYSIS documentation line 11
# JUNE ANALYSIS documentation line 12
# JUNE ANALYSIS documentation line 13
# JUNE ANALYSIS documentation line 14
# JUNE ANALYSIS documentation line 15
# JUNE ANALYSIS documentation line 16
# JUNE ANALYSIS documentation line 17
# JUNE ANALYSIS documentation line 18
# JUNE ANALYSIS documentation line 19
# JUNE ANALYSIS documentation line 20
# JUNE ANALYSIS documentation line 21
# JUNE ANALYSIS documentation line 22
# JUNE ANALYSIS documentation line 23
# JUNE ANALYSIS documentation line 24
# JUNE ANALYSIS documentation line 25
# JUNE ANALYSIS documentation line 26
# JUNE ANALYSIS documentation line 27
# JUNE ANALYSIS documentation line 28
# JUNE ANALYSIS documentation line 29
# JUNE ANALYSIS documentation line 30
# JUNE ANALYSIS documentation line 31
# JUNE ANALYSIS documentation line 32
# JUNE ANALYSIS documentation line 33
# JUNE ANALYSIS documentation line 34
# JUNE ANALYSIS documentation line 35
# JUNE ANALYSIS documentation line 36
# JUNE ANALYSIS documentation line 37
# JUNE ANALYSIS documentation line 38
# JUNE ANALYSIS documentation line 39
# JUNE ANALYSIS documentation line 40
# JUNE ANALYSIS documentation line 41
# JUNE ANALYSIS documentation line 42
# JUNE ANALYSIS documentation line 43
# JUNE ANALYSIS documentation line 44
# JUNE ANALYSIS documentation line 45
# ----------------------------------------------------------------------
# JULY ANALYSIS
# ----------------------------------------------------------------------
# JULY ANALYSIS documentation line 1
# JULY ANALYSIS documentation line 2
# JULY ANALYSIS documentation line 3
# JULY ANALYSIS documentation line 4
# JULY ANALYSIS documentation line 5
# JULY ANALYSIS documentation line 6
# JULY ANALYSIS documentation line 7
# JULY ANALYSIS documentation line 8
# JULY ANALYSIS documentation line 9
# JULY ANALYSIS documentation line 10
# JULY ANALYSIS documentation line 11
# JULY ANALYSIS documentation line 12
# JULY ANALYSIS documentation line 13
# JULY ANALYSIS documentation line 14
# JULY ANALYSIS documentation line 15
# JULY ANALYSIS documentation line 16
# JULY ANALYSIS documentation line 17
# JULY ANALYSIS documentation line 18
# JULY ANALYSIS documentation line 19
# JULY ANALYSIS documentation line 20
# JULY ANALYSIS documentation line 21
# JULY ANALYSIS documentation line 22
# JULY ANALYSIS documentation line 23
# JULY ANALYSIS documentation line 24
# JULY ANALYSIS documentation line 25
# JULY ANALYSIS documentation line 26
# JULY ANALYSIS documentation line 27
# JULY ANALYSIS documentation line 28
# JULY ANALYSIS documentation line 29
# JULY ANALYSIS documentation line 30
# JULY ANALYSIS documentation line 31
# JULY ANALYSIS documentation line 32
# JULY ANALYSIS documentation line 33
# JULY ANALYSIS documentation line 34
# JULY ANALYSIS documentation line 35
# JULY ANALYSIS documentation line 36
# JULY ANALYSIS documentation line 37
# JULY ANALYSIS documentation line 38
# JULY ANALYSIS documentation line 39
# JULY ANALYSIS documentation line 40
# JULY ANALYSIS documentation line 41
# JULY ANALYSIS documentation line 42
# JULY ANALYSIS documentation line 43
# JULY ANALYSIS documentation line 44
# JULY ANALYSIS documentation line 45
# ----------------------------------------------------------------------
# AUGUST ANALYSIS
# ----------------------------------------------------------------------
# AUGUST ANALYSIS documentation line 1
# AUGUST ANALYSIS documentation line 2
# AUGUST ANALYSIS documentation line 3
# AUGUST ANALYSIS documentation line 4
# AUGUST ANALYSIS documentation line 5
# AUGUST ANALYSIS documentation line 6
# AUGUST ANALYSIS documentation line 7
# AUGUST ANALYSIS documentation line 8
# AUGUST ANALYSIS documentation line 9
# AUGUST ANALYSIS documentation line 10
# AUGUST ANALYSIS documentation line 11
# AUGUST ANALYSIS documentation line 12
# AUGUST ANALYSIS documentation line 13
# AUGUST ANALYSIS documentation line 14
# AUGUST ANALYSIS documentation line 15
# AUGUST ANALYSIS documentation line 16
# AUGUST ANALYSIS documentation line 17
# AUGUST ANALYSIS documentation line 18
# AUGUST ANALYSIS documentation line 19
# AUGUST ANALYSIS documentation line 20
# AUGUST ANALYSIS documentation line 21
# AUGUST ANALYSIS documentation line 22
# AUGUST ANALYSIS documentation line 23
# AUGUST ANALYSIS documentation line 24
# AUGUST ANALYSIS documentation line 25
# AUGUST ANALYSIS documentation line 26
# AUGUST ANALYSIS documentation line 27
# AUGUST ANALYSIS documentation line 28
# AUGUST ANALYSIS documentation line 29
# AUGUST ANALYSIS documentation line 30
# AUGUST ANALYSIS documentation line 31
# AUGUST ANALYSIS documentation line 32
# AUGUST ANALYSIS documentation line 33
# AUGUST ANALYSIS documentation line 34
# AUGUST ANALYSIS documentation line 35
# AUGUST ANALYSIS documentation line 36
# AUGUST ANALYSIS documentation line 37
# AUGUST ANALYSIS documentation line 38
# AUGUST ANALYSIS documentation line 39
# AUGUST ANALYSIS documentation line 40
# AUGUST ANALYSIS documentation line 41
# AUGUST ANALYSIS documentation line 42
# AUGUST ANALYSIS documentation line 43
# AUGUST ANALYSIS documentation line 44
# AUGUST ANALYSIS documentation line 45
# ----------------------------------------------------------------------
# SEPTEMBER ANALYSIS
# ----------------------------------------------------------------------
# SEPTEMBER ANALYSIS documentation line 1
# SEPTEMBER ANALYSIS documentation line 2
# SEPTEMBER ANALYSIS documentation line 3
# SEPTEMBER ANALYSIS documentation line 4
# SEPTEMBER ANALYSIS documentation line 5
# SEPTEMBER ANALYSIS documentation line 6
# SEPTEMBER ANALYSIS documentation line 7
# SEPTEMBER ANALYSIS documentation line 8
# SEPTEMBER ANALYSIS documentation line 9
# SEPTEMBER ANALYSIS documentation line 10
# SEPTEMBER ANALYSIS documentation line 11
# SEPTEMBER ANALYSIS documentation line 12
# SEPTEMBER ANALYSIS documentation line 13
# SEPTEMBER ANALYSIS documentation line 14
# SEPTEMBER ANALYSIS documentation line 15
# SEPTEMBER ANALYSIS documentation line 16
# SEPTEMBER ANALYSIS documentation line 17
# SEPTEMBER ANALYSIS documentation line 18
# SEPTEMBER ANALYSIS documentation line 19
# SEPTEMBER ANALYSIS documentation line 20
# SEPTEMBER ANALYSIS documentation line 21
# SEPTEMBER ANALYSIS documentation line 22
# SEPTEMBER ANALYSIS documentation line 23
# SEPTEMBER ANALYSIS documentation line 24
# SEPTEMBER ANALYSIS documentation line 25
# SEPTEMBER ANALYSIS documentation line 26
# SEPTEMBER ANALYSIS documentation line 27
# SEPTEMBER ANALYSIS documentation line 28
# SEPTEMBER ANALYSIS documentation line 29
# SEPTEMBER ANALYSIS documentation line 30
# SEPTEMBER ANALYSIS documentation line 31
# SEPTEMBER ANALYSIS documentation line 32
# SEPTEMBER ANALYSIS documentation line 33
# SEPTEMBER ANALYSIS documentation line 34
# SEPTEMBER ANALYSIS documentation line 35
# SEPTEMBER ANALYSIS documentation line 36
# SEPTEMBER ANALYSIS documentation line 37
# SEPTEMBER ANALYSIS documentation line 38
# SEPTEMBER ANALYSIS documentation line 39
# SEPTEMBER ANALYSIS documentation line 40
# SEPTEMBER ANALYSIS documentation line 41
# SEPTEMBER ANALYSIS documentation line 42
# SEPTEMBER ANALYSIS documentation line 43
# SEPTEMBER ANALYSIS documentation line 44
# SEPTEMBER ANALYSIS documentation line 45
# ----------------------------------------------------------------------
# VISUALIZATION
# ----------------------------------------------------------------------
# VISUALIZATION documentation line 1
# VISUALIZATION documentation line 2
# VISUALIZATION documentation line 3
# VISUALIZATION documentation line 4
# VISUALIZATION documentation line 5
# VISUALIZATION documentation line 6
# VISUALIZATION documentation line 7
# VISUALIZATION documentation line 8
# VISUALIZATION documentation line 9
# VISUALIZATION documentation line 10
# VISUALIZATION documentation line 11
# VISUALIZATION documentation line 12
# VISUALIZATION documentation line 13
# VISUALIZATION documentation line 14
# VISUALIZATION documentation line 15
# VISUALIZATION documentation line 16
# VISUALIZATION documentation line 17
# VISUALIZATION documentation line 18
# VISUALIZATION documentation line 19
# VISUALIZATION documentation line 20
# VISUALIZATION documentation line 21
# VISUALIZATION documentation line 22
# VISUALIZATION documentation line 23
# VISUALIZATION documentation line 24
# VISUALIZATION documentation line 25
# VISUALIZATION documentation line 26
# VISUALIZATION documentation line 27
# VISUALIZATION documentation line 28
# VISUALIZATION documentation line 29
# VISUALIZATION documentation line 30
# VISUALIZATION documentation line 31
# VISUALIZATION documentation line 32
# VISUALIZATION documentation line 33
# VISUALIZATION documentation line 34
# VISUALIZATION documentation line 35
# VISUALIZATION documentation line 36
# VISUALIZATION documentation line 37
# VISUALIZATION documentation line 38
# VISUALIZATION documentation line 39
# VISUALIZATION documentation line 40
# VISUALIZATION documentation line 41
# VISUALIZATION documentation line 42
# VISUALIZATION documentation line 43
# VISUALIZATION documentation line 44
# VISUALIZATION documentation line 45
# ----------------------------------------------------------------------
# REPORTING
# ----------------------------------------------------------------------
# REPORTING documentation line 1
# REPORTING documentation line 2
# REPORTING documentation line 3
# REPORTING documentation line 4
# REPORTING documentation line 5
# REPORTING documentation line 6
# REPORTING documentation line 7
# REPORTING documentation line 8
# REPORTING documentation line 9
# REPORTING documentation line 10
# REPORTING documentation line 11
# REPORTING documentation line 12
# REPORTING documentation line 13
# REPORTING documentation line 14
# REPORTING documentation line 15
# REPORTING documentation line 16
# REPORTING documentation line 17
# REPORTING documentation line 18
# REPORTING documentation line 19
# REPORTING documentation line 20
# REPORTING documentation line 21
# REPORTING documentation line 22
# REPORTING documentation line 23
# REPORTING documentation line 24
# REPORTING documentation line 25
# REPORTING documentation line 26
# REPORTING documentation line 27
# REPORTING documentation line 28
# REPORTING documentation line 29
# REPORTING documentation line 30
# REPORTING documentation line 31
# REPORTING documentation line 32
# REPORTING documentation line 33
# REPORTING documentation line 34
# REPORTING documentation line 35
# REPORTING documentation line 36
# REPORTING documentation line 37
# REPORTING documentation line 38
# REPORTING documentation line 39
# REPORTING documentation line 40
# REPORTING documentation line 41
# REPORTING documentation line 42
# REPORTING documentation line 43
# REPORTING documentation line 44
# REPORTING documentation line 45

def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    return df

def prepare_dataframe(df):
    df = df.drop(df.index[:19])
    cols = ['-BEGIN HEADER-','Unnamed: 1','Unnamed: 5']
    ndf = df[cols].copy()
    ndf.columns = ['year','dayno','rainfall']
    ndf['year'] = ndf['year'].astype(int)
    ndf['dayno'] = ndf['dayno'].astype(int)
    ndf['rainfall'] = ndf['rainfall'].astype(float)
    return ndf

def aggregate_period(ndf, start_day, end_day):
    subset = ndf[(ndf['dayno'] >= start_day) & (ndf['dayno'] <= end_day)]
    return subset.groupby('year')['rainfall'].sum().reset_index()

def run_knn(data, k=5):
    X = data[['year']]
    y = data['rainfall']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X_train_scaled, y_train)

    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)

    return {
        'model': model,
        'train_pred': train_pred,
        'test_pred': test_pred,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'mse': mean_squared_error(y_test, test_pred),
        'r2': r2_score(y_test, test_pred)
    }

if __name__ == "__main__":
    print("KNN Rainfall Prediction Framework")

