import numpy as np
from scipy.stats import ks_2samp, chi2_contingency
from typing import Optional


def benjamini_hochberg(p_values: list[float], fdr: float = 0.05) -> list[float]:
    m = len(p_values)
    if m == 0:
        return []
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]
    ranks = np.arange(1, m + 1)
    thresholds = (ranks / m) * fdr
    max_sig = -1
    for i in range(m):
        if sorted_p[i] <= thresholds[i]:
            max_sig = i
    corrected = np.full(m, 1.0)
    if max_sig >= 0:
        corrected[: max_sig + 1] = np.minimum(
            sorted_p[: max_sig + 1] * m / (ranks[: max_sig + 1]), 1.0
        )
    unsorted = np.empty(m)
    unsorted[sorted_indices] = corrected
    return unsorted.tolist()


def ks_test(
    reference: list[float],
    production: list[float],
) -> tuple[float, float]:
    if len(reference) < 2 or len(production) < 2:
        return 0.0, 1.0
    stat, p = ks_2samp(reference, production, method="auto")
    return stat, p


def chi_square_test(
    reference_cats: list[str],
    production_cats: list[str],
) -> tuple[float, float]:
    all_cats = sorted(set(reference_cats) | set(production_cats))
    ref_counts = np.array([reference_cats.count(c) for c in all_cats])
    prod_counts = np.array([production_cats.count(c) for c in all_cats])
    mask = (ref_counts + prod_counts) > 0
    ref_counts = ref_counts[mask]
    prod_counts = prod_counts[mask]
    if len(ref_counts) < 2:
        return 0.0, 1.0
    table = np.array([ref_counts, prod_counts])
    stat, p, _, _ = chi2_contingency(table, correction=False)
    return stat, p


def detect_covariate_drift(
    reference_data: dict[str, list],
    production_data: dict[str, list],
    feature_types: dict[str, str],
    fdr: float = 0.05,
) -> dict:
    features = sorted(set(reference_data.keys()) & set(production_data.keys()))
    results = []
    p_values = []

    for feature in features:
        ref_vals = reference_data[feature]
        prod_vals = production_data[feature]
        ftype = feature_types.get(feature, "numerical")

        if ftype == "numerical":
            ref_num = [v for v in ref_vals if isinstance(v, (int, float))]
            prod_num = [v for v in prod_vals if isinstance(v, (int, float))]
            if len(ref_num) < 2 or len(prod_num) < 2:
                continue
            stat, p = ks_test(ref_num, prod_num)
            test_name = "Kolmogorov-Smirnov"
        else:
            ref_str = [str(v) for v in ref_vals]
            prod_str = [str(v) for v in prod_vals]
            if len(set(ref_str)) < 2 or len(set(prod_str)) < 2:
                continue
            stat, p = chi_square_test(ref_str, prod_str)
            test_name = "Chi-Square"

        results.append({
            "feature": feature,
            "test_name": test_name,
            "statistic": stat,
            "p_value": p,
            "drift_type": "covariate",
        })
        p_values.append(p)

    corrected = benjamini_hochberg(p_values, fdr) if p_values else []

    for i, corr_p in enumerate(corrected):
        results[i]["corrected_p_value"] = corr_p
        results[i]["significant"] = bool(corr_p < fdr)

    return {"results": results, "fdr": fdr, "method": "Benjamini-Hochberg"}


def compute_performance_drop(
    training_metric: float,
    recent_labels: list[float],
    recent_predictions: list[float],
    metric: str = "accuracy",
) -> Optional[float]:
    if len(recent_labels) < 10:
        return None
    if metric == "accuracy":
        correct = sum(
            1 for p, l in zip(recent_predictions, recent_labels) if p == l
        )
        live_metric = correct / len(recent_labels)
    elif metric == "mae":
        live_metric = float(
            np.mean([abs(p - l) for p, l in zip(recent_predictions, recent_labels)])
        )
    else:
        return None
    return live_metric


def detect_concept_drift(
    training_metric: float,
    live_metric: float,
    metric: str = "accuracy",
    threshold: float = 0.05,
) -> dict:
    if live_metric is None:
        return {"drift_detected": False, "reason": "insufficient_labeled_data"}
    if metric == "accuracy":
        drift = live_metric < training_metric - threshold
    elif metric == "mae":
        drift = live_metric > training_metric + threshold
    else:
        return {"drift_detected": False, "reason": "unknown_metric"}
    return {
        "drift_detected": drift,
        "training_metric": training_metric,
        "live_metric": live_metric,
        "metric": metric,
        "threshold": threshold,
    }
