import numpy as np
from sklearn.datasets import fetch_openml


def load_diabetes_data() -> tuple[np.ndarray, np.ndarray]:
    data = fetch_openml(name="diabetes", version=1, as_frame=False, parser="auto")
    X = data["data"].astype(float)
    y = data["target"].astype(float)
    return X, y


def no_drift_split(
    X: np.ndarray,
    ref_ratio: float = 0.5,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    indices = rng.permutation(n)
    split = int(n * ref_ratio)
    return X[indices[:split]], X[indices[split:]]


def synthetic_covariate_shift(
    X_ref: np.ndarray,
    X_prod: np.ndarray,
    feature_idx: int = 0,
    shift_magnitude: float = 2.0,
) -> np.ndarray:
    X_shifted = X_prod.copy()
    X_shifted[:, feature_idx] = X_shifted[:, feature_idx] + shift_magnitude
    return X_shifted


def synthetic_concept_shift(
    X_ref: np.ndarray,
    y_ref: np.ndarray,
    X_prod: np.ndarray,
    y_prod: np.ndarray,
    flip_ratio: float = 0.3,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    n = len(y_prod)
    flip_n = int(n * flip_ratio)
    flip_idx = rng.choice(n, flip_n, replace=False)
    y_shifted = y_prod.copy()
    y_shifted[flip_idx] = 1.0 - y_shifted[flip_idx]
    return X_prod, y_shifted
