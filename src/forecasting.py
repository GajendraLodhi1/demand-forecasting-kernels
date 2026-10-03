import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

def load_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    train["date"] = pd.to_datetime(train["date"])
    test["date"] = pd.to_datetime(test["date"])
    return train, test

def add_date_features(df):
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek
    df["week"] = df["date"].dt.isocalendar().week
    df["quarter"] = df["date"].dt.quarter
    return df

def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return np.mean(diff)

def create_lag_features(df):
    df = df.sort_values(by=['store', 'item', 'date'])
    for lag in [1, 7, 14, 28]:
        df[f'lag_{lag}'] = df.groupby(['store', 'item'])['sales'].transform(lambda x: x.shift(lag))
    return df

def create_rolling_features(df):
    df = df.sort_values(by=['store', 'item', 'date'])
    for window in [7, 14, 28]:
        df[f'rolling_mean_{window}'] = df.groupby(['store', 'item'])['sales'].transform(lambda x: x.shift(1).rolling(window).mean())
    return df

def prepare_data(df):
    df = add_date_features(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = df.dropna()
    return df
