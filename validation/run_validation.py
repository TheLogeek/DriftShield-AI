import json
from pathlib import Path

import numpy as np

from validation.scenarios import (
    load_diabetes_data,
    no_drift_split,
    synthetic_covariate_shift,
    synthetic_concept_shift,
)
from drift_engine import detect_covariate_drift


RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _feature_types_from_X(X: np.ndarray, feature_names: list[str]) -> dict[str, str]:
    return {name: "numerical" for name in feature_names}


def _X_to_dict(X: np.ndarray, feature_names: list[str]) -> dict[str, list]:
    return {name: X[:, i].tolist() for i, name in enumerate(feature_names)}


def run_no_drift_scenario(X: np.ndarray, feature_names: list[str]) -> dict:
    X_ref, X_prod = no_drift_split(X, ref_ratio=0.5)
    ref_data = _X_to_dict(X_ref, feature_names)
    prod_data = _X_to_dict(X_prod, feature_names)
    ftypes = _feature_types_from_X(X, feature_names)
    result = detect_covariate_drift(ref_data, prod_data, ftypes, fdr=0.05)
    false_positives = sum(1 for r in result["results"] if r["significant"])
    total = len(result["results"])
    return {
        "scenario": "no_drift",
        "total_features": total,
        "false_positives": false_positives,
        "false_positive_rate": false_positives / total if total > 0 else 0.0,
        "details": result,
    }


def run_covariate_shift_scenario(
    X: np.ndarray,
    feature_names: list[str],
    magnitudes: list[float] = None,
) -> list[dict]:
    if magnitudes is None:
        magnitudes = [0.5, 1.0, 2.0, 3.0]
    X_ref, X_prod = no_drift_split(X, ref_ratio=0.5)
    results = []
    for mag in magnitudes:
        X_shifted = synthetic_covariate_shift(X_ref, X_prod, feature_idx=0, shift_magnitude=mag)
        ref_data = _X_to_dict(X_ref, feature_names)
        prod_data = _X_to_dict(X_shifted, feature_names)
        ftypes = _feature_types_from_X(X, feature_names)
        det = detect_covariate_drift(ref_data, prod_data, ftypes, fdr=0.05)
        true_positives = sum(
            1 for r in det["results"] if r["significant"] and r["feature"] == feature_names[0]
        )
        results.append({
            "scenario": "covariate_shift",
            "magnitude": mag,
            "target_feature": feature_names[0],
            "detected": true_positives > 0,
            "total_features_tested": len(det["results"]),
            "details": det,
        })
    return results


def run_all() -> dict:
    print("Loading diabetes dataset...")
    X, y = load_diabetes_data()
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    print("Running no-drift scenario...")
    no_drift = run_no_drift_scenario(X, feature_names)

    print("Running covariate shift scenarios...")
    cov_shifts = run_covariate_shift_scenario(X, feature_names)

    report = {
        "dataset": "diabetes (OpenML)",
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "scenarios": {
            "no_drift": no_drift,
            "covariate_shift": cov_shifts,
        },
    }
    report_path = RESULTS_DIR / "validation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Validation report written to {report_path}")
    return report


if __name__ == "__main__":
    run_all()
