from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier

from src.config import RANDOM_STATE


def logistic_regression_model() -> LogisticRegression:
    """
    Create the baseline Logistic Regression model.
    """

    return LogisticRegression(
        random_state=RANDOM_STATE,
        max_iter=1000,
    )

def random_forest_model() -> RandomForestClassifier:
    """
    Create the baseline Random Forest model.
    """
    return RandomForestClassifier(
        n_estimators=100, #trees
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

def catboost_model() -> CatBoostClassifier:
    """
    Create the baseline CatBoost model.
    """

    return CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        random_state=RANDOM_STATE,
        verbose=False,
    )

