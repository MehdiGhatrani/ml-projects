import pandas as pd
from sklearn.model_selection import train_test_split
from .preprocessing import full_pipeline
from pathlib import Path


def load_data(data_path):
    return pd.read_csv(data_path)


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def prepare_data(X_train, X_test):
    X_train_prep = full_pipeline.fit_transform(X_train)
    X_test_prep = full_pipeline.transform(X_test)
    return X_train_prep, X_test_prep


def split_and_prepare_data(df):
    X = df.drop("median_house_value", axis=1)
    y = df["median_house_value"]

    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_prep, X_test_prep = prepare_data(X_train, X_test)

    return X_train_prep, X_test_prep, y_train, y_test



PROJECT_ROOT = Path(__file__).resolve().parent.parent
data_path = PROJECT_ROOT / "data" / "raw" / "housing.csv"

