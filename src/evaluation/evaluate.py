import numpy as np
import time
import pandas as pd
from src.models.train import build_pipeline

from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
)

from src.config import (
    N_SPLITS,
    RANDOM_STATE,
)


def evaluate_model(
    pipeline,
    X,
    y,
):
    """
    Evaluate a pipeline using cross validation.
    """

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    start_time = time.perf_counter()
    
    scores = cross_val_score(
        estimator=pipeline,
        X=X,
        y=y,
        cv=cv,
        scoring="accuracy",
    )

    training_time = time.perf_counter() - start_time

    return {
        "scores": scores,
        "mean_accuracy": scores.mean(),
        "std_accuracy": scores.std(),
        "training_time": training_time
    }

def compare_models(
    models,
    preprocessor,
    X,
    y,
) -> pd.DataFrame:
    """
    Compare multiple models using cross validation.
    """

    results = []

    for model_name, model in models:
        pipeline = build_pipeline(preprocessor, model)
        evaluation_results = evaluate_model(pipeline, X, y)

        results.append({
            "Model": model_name,
            "Mean Accuracy": evaluation_results["mean_accuracy"],
            "Std Accuracy": evaluation_results["std_accuracy"],
            "Training Time (s)": evaluation_results["training_time"],
        })

    return pd.DataFrame(results)
