from typing import List, Tuple

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def get_feature_types(df: pd.DataFrame,) -> Tuple[List[str], List[str]]:
    """
    Identify numerical and categorical feature names.
    """
    numeric_features = (
        df.select_dtypes(include="number")
        .columns
        .tolist()
    )

    categorical_features = (
        df.select_dtypes(
            include=["object", "category", "bool"]
        )
        .columns
        .tolist()
    )

    return numeric_features, categorical_features

def build_preprocessor(numeric_features, categorical_features, scale_numeric=True,)-> ColumnTransformer:
    """
    Build a preprocessor for numerical and categorical features.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor