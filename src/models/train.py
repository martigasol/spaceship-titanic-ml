from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import RANDOM_STATE

def build_pipeline(preprocessor, model) -> Pipeline:
    """
    Build the baseline machine learning pipeline.
    """
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline