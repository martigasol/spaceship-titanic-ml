import numpy as np

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

    scores = cross_val_score(
        estimator=pipeline,
        X=X,
        y=y,
        cv=cv,
        scoring="accuracy",
    )

    return {
        "scores": scores,
        "mean_accuracy": scores.mean(),
        "std_accuracy": scores.std(),
    }