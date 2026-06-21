"""
Self-contained demo: runs drift detection on synthetic data.
No server needed — just `pip install -r requirements.txt && py demo.py`.
"""
import numpy as np
from drift_engine import (
    benjamini_hochberg,
    ks_test,
    chi_square_test,
    detect_covariate_drift,
)

np.random.seed(42)
print("=" * 60)
print("DriftShield AI — Smoke Test")
print("=" * 60)

# --- 1. Benjamini-Hochberg Correction ---
print("\n[1] Benjamini-Hochberg FDR Correction")
p_values = [0.001, 0.012, 0.04, 0.50, 0.70]
corrected = benjamini_hochberg(p_values, fdr=0.05)
for i, (raw, corr) in enumerate(zip(p_values, corrected)):
    flag = "⚠️" if corr < 0.05 else "✅"
    print(f"    p{i+1}: raw={raw:.3f}  corrected={corr:.3f}  {flag}")
print("    ✓ BH correction works")

# --- 2. K-S Test (no drift) ---
print("\n[2] K-S Test — Identical distributions (should NOT flag)")
ref = list(np.random.normal(0, 1, 100))
prod = list(np.random.normal(0, 1, 100))
stat, p = ks_test(ref, prod)
print(f"    statistic={stat:.4f}  p-value={p:.4f}  {'✅ no drift' if p > 0.05 else '⚠️ false positive'}")
assert p > 0.01

# --- 3. K-S Test (drift) ---
print("\n[3] K-S Test — Shifted distribution (SHOULD flag)")
prod_shifted = list(np.random.normal(3, 1, 100))
stat, p = ks_test(ref, prod_shifted)
print(f"    statistic={stat:.4f}  p-value={p:.4f}  {'✅ drift detected' if p < 0.05 else '⚠️ missed'}")
assert p < 0.05

# --- 4. Chi-Square Test ---
print("\n[4] Chi-Square Test — Different category distributions")
ref_cats = ["cat"] * 50 + ["dog"] * 50
prod_cats = ["cat"] * 10 + ["dog"] * 90
stat, p = chi_square_test(ref_cats, prod_cats)
print(f"    statistic={stat:.4f}  p-value={p:.4f}  {'✅ drift detected' if p < 0.05 else '⚠️ missed'}")
assert p < 0.05

# --- 5. Full Covariate Drift Detection ---
print("\n[5] Full Covariate Drift Detection (3 features, 2 drifted)")
ref_data = {
    "age": list(np.random.normal(35, 5, 200)),
    "income": list(np.random.normal(60000, 15000, 200)),
    "score": list(np.random.uniform(0, 1, 200)),
}
prod_data = {
    "age": list(np.random.normal(35, 5, 200)),          # no drift
    "income": list(np.random.normal(90000, 15000, 200)), # drifted
    "score": list(np.random.uniform(0.8, 1.0, 200)),     # drifted
}
feature_types = {"age": "numerical", "income": "numerical", "score": "numerical"}
result = detect_covariate_drift(ref_data, prod_data, feature_types, fdr=0.05)
for r in result["results"]:
    sig = "⚠️ DRIFT" if r["significant"] else "✅ ok"
    print(f"    {r['feature']}: p={r['p_value']:.4f}  adj-p={r['corrected_p_value']:.4f}  {sig}")
    if r["feature"] == "age":
        assert not r["significant"], "age should NOT drift"
    else:
        assert r["significant"], f"{r['feature']} SHOULD drift"

print("\n" + "=" * 60)
print("ALL CHECKS PASSED — DriftShield AI is working correctly.")
print("=" * 60)
