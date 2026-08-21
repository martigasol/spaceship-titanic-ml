from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
)

from src.config import (
    N_SPLITS,
    RANDOM_STATE,
)

def random_search(
    pipeline,
    param_distributions,
    X,
    y,
    n_iter=15,
):
    """
    Perform hyperparameter tuning using RandomizedSearchCV.
    """

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring="accuracy",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )

    search.fit(X, y)

    return search