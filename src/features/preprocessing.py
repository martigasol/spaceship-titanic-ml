from typing import List, Tuple

import pandas as pd


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