from pathlib import Path

import joblib


def save_model(model, path: Path) -> None:
    """
    Save a trained model to disk.
    """
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(model, path)


def load_model(path: Path):
    """
    Load a trained model from disk.
    """
    return joblib.load(path)