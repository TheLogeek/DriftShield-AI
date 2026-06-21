import numpy as np
import pytest

from drift_engine import (
    benjamini_hochberg,
    ks_test,
    chi_square_test,
    detect_covariate_drift,
    compute_performance_drop,
    detect_concept_drift,
)


class TestBenjaminiHochberg:
    def test_all_below_threshold(self):
        p = [0.001, 0.002, 0.003]
        corrected = benjamini_hochberg(p, fdr=0.05)
        assert all(c < 0.05 for c in corrected)
        assert all(c <= 1.0 for c in corrected)

    def test_all_above_threshold(self):
        p = [0.6, 0.7, 0.8]
        corrected = benjamini_hochberg(p, fdr=0.05)
        assert all(c >= 0.05 for c in corrected)

    def test_mixed_significance(self):
        p = [0.001, 0.04, 0.5, 0.7]
        corrected = benjamini_hochberg(p, fdr=0.05)
        assert corrected[0] < 0.05
        assert corrected[1] >= 0.05  # rank 2 threshold = 0.025; 0.04 doesn't survive
        assert corrected[2] >= 0.05

    def test_empty_list(self):
        assert benjamini_hochberg([]) == []

    def test_single_value(self):
        c = benjamini_hochberg([0.03], fdr=0.05)
        assert c[0] < 0.05


class TestKSTest:
    def test_identical_distributions(self):
        ref = [1.0, 2.0, 3.0, 4.0, 5.0]
        prod = [1.0, 2.0, 3.0, 4.0, 5.0]
        stat, p = ks_test(ref, prod)
        assert p > 0.05

    def test_different_distributions(self):
        ref = [1.0, 2.0, 3.0, 4.0, 5.0]
        prod = [100.0, 200.0, 300.0, 400.0, 500.0]
        stat, p = ks_test(ref, prod)
        assert p < 0.05

    def test_insufficient_data(self):
        stat, p = ks_test([1.0], [2.0])
        assert p == 1.0


class TestChiSquareTest:
    def test_identical_distributions(self):
        ref = ["a", "a", "b", "b"]
        prod = ["a", "a", "b", "b"]
        stat, p = chi_square_test(ref, prod)
        assert p > 0.05

    def test_different_distributions(self):
        ref = ["a", "a", "a", "a"]
        prod = ["b", "b", "b", "b"]
        stat, p = chi_square_test(ref, prod)
        assert p < 0.05

    def test_too_few_categories(self):
        stat, p = chi_square_test(["a", "a"], ["a", "a"])
        assert p == 1.0


class TestDetectCovariateDrift:
    def test_no_drift_detected(self):
        ref_data = {"x": [1.0, 2.0, 3.0, 4.0, 5.0]}
        prod_data = {"x": [1.1, 2.1, 2.9, 4.1, 5.0]}
        ftypes = {"x": "numerical"}
        result = detect_covariate_drift(ref_data, prod_data, ftypes, fdr=0.05)
        assert not result["results"][0]["significant"]

    def test_drift_detected(self):
        ref_data = {"x": [1.0, 2.0, 3.0, 4.0, 5.0]}
        prod_data = {"x": [100.0, 200.0, 300.0, 400.0, 500.0]}
        ftypes = {"x": "numerical"}
        result = detect_covariate_drift(ref_data, prod_data, ftypes, fdr=0.05)
        assert result["results"][0]["significant"]


class TestConceptDrift:
    def test_no_drift_when_performance_stable(self):
        result = detect_concept_drift(
            training_metric=0.85,
            live_metric=0.84,
            metric="accuracy",
            threshold=0.05,
        )
        assert not result["drift_detected"]

    def test_drift_when_performance_drops(self):
        result = detect_concept_drift(
            training_metric=0.85,
            live_metric=0.70,
            metric="accuracy",
            threshold=0.05,
        )
        assert result["drift_detected"]

    def test_insufficient_data(self):
        result = detect_concept_drift(
            training_metric=0.85,
            live_metric=None,
            metric="accuracy",
            threshold=0.05,
        )
        assert not result["drift_detected"]


class TestComputePerformanceDrop:
    def test_accuracy_with_sufficient_data(self):
        drop = compute_performance_drop(
            training_metric=0.9,
            recent_labels=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            recent_predictions=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            metric="accuracy",
        )
        assert drop == 1.0

    def test_insufficient_data_returns_none(self):
        drop = compute_performance_drop(
            training_metric=0.9,
            recent_labels=[0, 1],
            recent_predictions=[0, 1],
            metric="accuracy",
        )
        assert drop is None
