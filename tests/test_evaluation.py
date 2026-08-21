import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.evaluate import compare_models


def test_compare_models_returns_model_names_and_numeric_metrics():
    X = pd.DataFrame(
        {
            "feature_a": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "feature_b": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            "feature_c": [0, 0, 1, 1, 0, 0, 1, 1, 0, 1],
        }
    )
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    preprocessor = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    models = [
        ("Logistic Regression", LogisticRegression(max_iter=1000)),
        ("Random Forest", RandomForestClassifier(n_estimators=10, random_state=42)),
    ]

    df = compare_models(models=models, preprocessor=preprocessor, X=X, y=y)

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "Model" in df.columns
    assert all(isinstance(name, str) for name in df["Model"])
    assert all(not isinstance(name, tuple) for name in df["Model"])
    assert {"Mean Accuracy", "Std Accuracy", "Training Time (s)"}.issubset(df.columns)
