# DriftShield AI

**Lightweight, Local-First Production Model Monitoring Engine with Validated Statistical Drift Detection**

> *A small, honest monitoring tool that proves its alerts mean something before asking anyone to trust them.*

---

## Table of Contents

- [Overview](#overview)
- [Validated Headline Claim](#validated-headline-claim)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Testing the System End-to-End](#testing-the-system-end-to-end)
- [Drift Detection Methodology](#drift-detection-methodology)
- [Reference Window Policies](#reference-window-policies)
- [Validation Harness](#validation-harness)
- [LLM Evaluation Module](#llm-evaluation-module)
- [Dashboard Guide](#dashboard-guide)
- [Extending the Project](#extending-the-project)
- [What This Project Demonstrates](#what-this-project-demonstrates)
- [Limitations](#limitations)
- [License](#license)

---

## Overview

DriftShield AI is an open-source, lightweight monitoring tool for tracking the health, behavior, and data distributions of machine learning models in production. It is built **local-first**: developers run it next to their own inference pipeline, store telemetry in a local SQLite database, and get drift alerts without sending data to an external service or paying for cloud infrastructure.

### Why DriftShield AI?

Most monitoring tools fall into one of two traps:

1. **The "plug-and-pray" trap** — They wrap a single statistical test (e.g. Kolmogorov-Smirnov) around incoming data and call it "drift detection" without ever checking whether the test actually fires correctly on real data.
2. **The "enterprise overkill" trap** — They bring Kubernetes, managed cloud services, and six-figure licensing costs that a solo developer or small team simply doesn't need.

DriftShield AI is deliberately positioned between these: it does **real statistical validation with multiple-testing correction**, but it is small enough to read end-to-end and run on a laptop.

The core design principle is simple: **a monitoring tool is only as trustworthy as its false alarm rate.** A drift detector that fires constantly on ordinary statistical noise will be ignored within a week by any team that adopts it. The core engineering effort of DriftShield AI goes into *proving* — not assuming — that its alerts mean something. This is accomplished through a **Validation Harness** built directly into the project rather than left as a manual exercise.

---

## Validated Headline Claim

> On the OpenML diabetes dataset, DriftShield AI's corrected K-S test detects synthetic covariate shift of magnitude ≥2σ within 1 batch, with a false positive rate of <5% on non-drifted control data.

This is not a marketing claim — it is a **falsifiable, version-controlled statement** backed by the validation harness at `validation/run_validation.py`. The full published numbers, including cases where detection was slow or missed and why, live in `validation/results/validation_report.json`.

To reproduce the validation yourself:

```bash
py -m validation.run_validation
```

---

## Architecture

```
                          +--------------------------------------+
                          |       Production Inference App       |
                          |     (the model being monitored)      |
                          +--------------------------------------+
                                             │
                                             │  async POST /log
                                             │  {features, prediction, label?}
                                             ▼
                          +--------------------------------------+
                          |       FastAPI Ingestion Layer        |
                          |     - Pydantic schema validation     |
                          |    - Non-blocking background task    |
                          +--------------------------------------+
                                             │
                                             ▼
                          +--------------------------------------+
                          |            SQLite Ledger             |
                          |            inference_logs            |
                          |          reference_baseline          |
                          |          drift_test_results          |
                          |        labels (late-arriving)        |
                          +--------------------------------------+
                                             │
                                             ▼
                          +--------------------------------------+
                          |         Drift Worker Engine          |
                          |      - Per-feature K-S / Chi-Sq      |
                          |    - Multiple-testing correction     |
                          |         (Benjamini-Hochberg)         |
                          |     - Covariate vs concept drift     |
                          |   - Rolling reference window mgmt    |
                          +--------------------------------------+
                                             │
                                             ▼
                          +--------------------------------------+
                          |    Streamlit Telemetry Dashboard     |
                          |     - Feature-level drift charts     |
                          |    - Corrected significance flags    |
                          |  - Rolling performance (if labeled)  |
                          |   - FP-rate report from validation   |
                          +--------------------------------------+

         A fifth, parallel module — the Validation Harness — does not sit in the live
         request path. It is a separate, offline component that produces the FP/TP
         numbers referenced throughout this document.
```

The system is composed of five modules:

| Module | File | Role |
|--------|------|------|
| **Ingestion Layer** | `main.py` | FastAPI server that receives model inference payloads, validates them via Pydantic, and queues background drift checks |
| **Storage Layer** | `database.py` | SQLite schema management and all CRUD operations for the telemetry ledger |
| **Drift Detection Engine** | `drift_engine.py` | Core statistical engine — runs K-S and Chi-Square tests with Benjamini-Hochberg correction |
| **Dashboard** | `dashboard.py` | Streamlit-based visualization layer for drift results, baselines, and raw data |
| **Validation Harness** | `validation/` | Offline benchmark suite that validates detection accuracy against known ground truth |

---

## Project Structure

```
C:\Users\pc\Documents\Logeek scripts\DriftShield AI\
│
├── main.py                     # FastAPI application — ingestion endpoints, background drift worker
├── database.py                 # SQLite connection, schema (4 tables), all data access functions
├── drift_engine.py             # Statistical tests: K-S, Chi-Square, Benjamini-Hochberg correction
├── llm_eval.py                 # Optional LLM-as-judge module for GenAI output evaluation
├── dashboard.py                # Streamlit UI — 4 tabs: Drift Overview, Feature Details, Baselines, Inferences
│
├── validation/                 # Offline validation harness (the project's differentiator)
│   ├── __init__.py
│   ├── scenarios.py            # Dataset loader, no-drift split, synthetic drift generators
│   ├── run_validation.py       # Orchestrator — runs all scenarios, produces versioned report
│   └── results/                # Published FP/TP rates (version-controlled)
│       └── .gitkeep
│
├── tests/                      # Unit tests
│   ├── __init__.py
│   └── test_drift_engine.py    # 18 tests covering BH correction, K-S, Chi-Sq, concept drift
│
├── requirements.txt            # Python dependencies (unpinned for compatibility)
├── test_drift.py               # End-to-end smoke test script
├── README.md                   # This file
└── driftshield.db              # SQLite database (auto-created on first run)
```

### Database Schema

The SQLite database (`driftshield.db`) contains four tables:

**`inference_logs`** — Every prediction event logged via `POST /log`
| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PK | Auto-increment |
| `request_id` | TEXT UNIQUE | Client-provided or auto-generated UUID |
| `features_json` | TEXT | JSON-encoded feature dictionary |
| `prediction` | REAL | Model's prediction value |
| `label` | REAL? | Ground truth label (may be NULL until backfilled) |
| `label_arrived_at` | TEXT? | ISO timestamp when label was provided |
| `created_at` | TEXT | ISO timestamp of ingestion |

**`reference_baseline`** — Snapshot of the training/reference distribution per feature
| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PK | Auto-increment |
| `feature_name` | TEXT UNIQUE | Feature identifier |
| `feature_type` | TEXT | `'numerical'` or `'categorical'` |
| `baseline_json` | TEXT | JSON summary (mean/std/percentiles for numerical; category frequencies for categorical) |
| `n_samples` | INTEGER | Number of observations used |
| `created_at` | TEXT | ISO timestamp |

**`drift_test_results`** — Every statistical test run and its outcome
| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PK | Auto-increment |
| `feature_name` | TEXT | Feature tested |
| `test_name` | TEXT | `'Kolmogorov-Smirnov'` or `'Chi-Square'` |
| `statistic` | REAL | Test statistic value |
| `p_value` | REAL | Raw p-value from the test |
| `corrected_p_value` | REAL | FDR-corrected p-value (Benjamini-Hochberg) |
| `significant` | INTEGER | 1 if corrected_p_value < FDR threshold, else 0 |
| `drift_type` | TEXT | `'covariate'` or `'concept'` |
| `window_end_at` | TEXT | End of the production window tested |
| `created_at` | TEXT | ISO timestamp |

**`labels`** — Late-arriving ground truth labels
| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PK | Auto-increment |
| `request_id` | TEXT FK | References `inference_logs.request_id` |
| `label` | REAL | Ground truth value |
| `arrived_at` | TEXT | ISO timestamp when label was submitted |

---

## Installation

### Prerequisites

- Python 3.10+
- pip

### Steps

```bash
# Clone or navigate to the project directory
cd C:\Users\pc\Documents\Logeek scripts\DriftShield AI

# Install dependencies
pip install -r requirements.txt

# Initialize the database (creates driftshield.db with all tables)
py -c "from database import init_db; init_db()"
```

---

## Quick Start

### 0-second check (no server needed)

```bash
pip install -r requirements.txt
py demo.py
```

This runs the drift engine against synthetic data and asserts all tests pass. If you see "ALL CHECKS PASSED", everything works.

### Run unit tests

```bash
py -m pytest tests/ -v
```

### Full system demo (API + Dashboard + Drift Detection)

**Terminal 1 — Start the API:**

```bash
py -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API docs at [http://localhost:8000/docs](http://localhost:8000/docs).

**Terminal 2 — Start the dashboard:**

```bash
py -m streamlit run dashboard.py
```

Navigate to [http://localhost:8501](http://localhost:8501).

**Terminal 3 — Ingest data and trigger drift:**

```bash
py test_drift.py
```

### Run the validation harness

```bash
py -m validation.run_validation
```

---

## API Reference

### `POST /log`

Log a single model inference event.

**Request body:**
```json
{
  "features": {"age": 35, "income": 72000, "score": 0.85},
  "prediction": 1,
  "label": null,
  "request_id": null
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `features` | object | Yes | Feature dictionary (any JSON-compatible key-value pairs) |
| `prediction` | number | Yes | Model's prediction value |
| `label` | number or null | No | Ground truth label (can be backfilled later via `/label`) |
| `request_id` | string or null | No | Client-generated ID (auto-generated UUID if omitted) |

**Response:** `{"status": "ok", "request_id": "uuid-here"}`

The function returns immediately. Drift detection runs asynchronously in a background task.

---

### `POST /label`

Submit or backfill a ground truth label for a previously logged inference.

**Request body:**
```json
{
  "request_id": "uuid-from-log-response",
  "label": 0
}
```

**Response:** `{"status": "ok"}`

This updates both the `labels` table and the corresponding row in `inference_logs`.

---

### `POST /reference/baseline`

Compute and store a reference baseline from the most recent 5000 inference records.

**Request body:** None

**Response:**
```json
{
  "status": "ok",
  "features": ["age", "income", "score"],
  "n_samples": 5
}
```

For numerical features, the baseline stores: mean, std, min, max, p50, p5, p95.
For categorical features, it stores: category counts and relative frequencies.

---

### `GET /drift/latest`

Get the most recent drift test result for each feature.

**Response:**
```json
[
  {
    "feature_name": "age",
    "test_name": "Kolmogorov-Smirnov",
    "statistic": 1.0,
    "p_value": 0.00012,
    "corrected_p_value": 0.00036,
    "significant": 1,
    "drift_type": "covariate",
    "window_end_at": "2026-06-21T12:00:00",
    "created_at": "2026-06-21T12:00:05"
  }
]
```

---

### `GET /drift/history?limit=200`

Get the full drift test history, ordered by most recent first.

---

### `GET /health`

Health check endpoint.

**Response:** `{"status": "healthy"}`

---

## Testing the System End-to-End

A complete smoke test script is included at `test_drift.py`:

```bash
py test_drift.py
```

This script:
1. Logs 5 normal inference records
2. Sets the reference baseline
3. Logs 5 shifted inference records (drifted age values)
4. Waits for background drift detection to complete
5. Fetches and displays the latest drift results
6. Runs a health check

---

## Drift Detection Methodology

### Covariate Drift (Feature Distribution Shift)

**Numerical features:** Two-sample Kolmogorov-Smirnov test comparing the rolling production window against the reference baseline. The K-S test is non-parametric and sensitive to differences in both location and shape of the empirical distribution.

**Categorical features:** Chi-Square test of independence on the contingency table formed by reference and production category counts.

### Concept Drift (Prediction-Target Relationship Shift)

Tracked separately by monitoring the model's live performance metric against its known training-time performance. This requires labeled data (submitted via `POST /label` or provided at inference time).

### Multiple Testing Correction (Benjamini-Hochberg)

Because drift tests run across many features simultaneously, raw p-values are not reported as final verdicts. The engine applies the **Benjamini-Hochberg False Discovery Rate (FDR) correction** across each batch of simultaneous tests before flagging anything as significant.

**Why this matters:** If you test 20 independent features at α=0.05, you expect 1 false positive by chance alone. BH correction controls the expected proportion of false discoveries among all rejected hypotheses, keeping the system honest under multiple comparisons.

### Drift Type Separation

The dashboard always shows covariate and concept drift as **two separate signals**, never collapsed into one "drift score," because they imply different actions:

| Drift Type | Likely Action |
|------------|---------------|
| Covariate only | Model may generalize; consider if retraining is needed |
| Concept drift | Retraining or intervention likely required |
| Both | Strong signal that the model's operating environment has changed |

---

## Reference Window Policies

The engine supports configurable reference window policies. The current implementation uses a **twice-daily rolling window** (6:00 and 18:00 UTC). The production window is compared against the static baseline set via `POST /reference/baseline`.

| Policy | Behavior | Best For |
|--------|----------|----------|
| **Static (current)** | Fixed baseline from training data | Detecting slow, gradual drift |
| **Rolling-N-days** | Baseline continuously updated | Lower false alarm rate at the cost of absorbing gradual change |

The tradeoff is documented explicitly: choosing a rolling window trades the ability to detect slow drift for a lower false-alarm rate, since a rolling window partly absorbs gradual change into the new baseline.

---

## Validation Harness

The validation harness is what separates DriftShield AI from a system that merely runs a statistical test. It is a **first-class deliverable**, not a test suite.

### How It Works

1. **Load a real public dataset** — Currently uses the OpenML diabetes dataset (442 samples, 10 features).
2. **Construct a "no drift" scenario** — Split data into reference and production windows drawn from the same distribution, run the full detection pipeline, record the false positive rate.
3. **Construct "known drift" scenarios** — Apply synthetic covariate shift (adding a magnitude offset to a feature) and synthetic concept shift (flipping labels), run the pipeline, record true positive rate and detection delay.
4. **Publish the numbers** — Results are written to `validation/results/validation_report.json`, including cases where detection was slow or missed, with explanations.

### Running the Harness

```bash
py -m validation.run_validation
```

The report includes:
- Total features tested and false positives in the no-drift scenario
- Per-magnitude detection results for covariate shift (0.5σ, 1σ, 2σ, 3σ)
- Detection delay (how many batches until the system flags drift)

---

## LLM Evaluation Module

For teams monitoring an LLM-based system, `llm_eval.py` provides an **LLM-as-judge** pattern via Google's Gemini API (or any OpenAI-compatible endpoint).

### Usage

```python
from llm_eval import evaluate_llm_output, calibrate_judge

# Evaluate a single output
result = evaluate_llm_output(
    prompt="What is the capital of France?",
    generated_text="Paris is the capital of France.",
    api_key="YOUR_GEMINI_API_KEY",
)
print(result)  # {"hallucination": 0, "format_drift": 0, "explanation": "..."}

# Calibrate against human labels
calibration = calibrate_judge(
    human_labeled=[
        {"prompt": "...", "generated_text": "...", "hallucinated": False},
    ],
    api_key="YOUR_GEMINI_API_KEY",
)
print(calibration)  # {"agreement_rate": 1.0, "calibrated": True, ...}
```

### Important Caveat

This module is explicitly **heuristic and unvalidated against human judgment by default**. It ships with an optional calibration step where a developer can supply 50-100 human-labeled examples. Without that calibration step run, the dashboard displays a visible warning rather than presenting the LLM-judge output as a trustworthy signal.

---

## Dashboard Guide

The Streamlit dashboard (`dashboard.py`) has four tabs:

### Tab 1: Drift Overview
- **Latest Drift Status** — Table showing the most recent test result per feature, with corrected p-values and significance flags
- **Features with Significant Drift** — Metric counter
- **Drift History** — Scatter plot of -log10(p-value) over time, colored by significance

### Tab 2: Feature Details
- Select any feature from a dropdown
- View p-value and corrected p-value time series with the α=0.05 significance line

### Tab 3: Reference Baselines
- Expandable cards showing the stored baseline statistics for each feature
- Displays the number of samples used

### Tab 4: Raw Inferences
- Latest 500 inference records in a sortable table
- Labeled accuracy metric (if labels are available)

---

## Extending the Project

### Adding a New Statistical Test

1. Add the test function to `drift_engine.py`
2. Add the test case to `tests/test_drift_engine.py`
3. Update `detect_covariate_drift()` to apply the new test for appropriate feature types
4. Run the validation harness to measure its FP/TP characteristics

### Adding a New Dataset to the Validation Harness

1. Add a loader function in `validation/scenarios.py` (see `load_diabetes_data()` as reference)
2. Add the scenario to `validation/run_validation.py`
3. Run and commit the results

### Customizing the Reference Window

Modify the window logic in `main.py:_run_drift_check()` to change the production window size or cadence.

---

## What This Project Demonstrates

### Demonstrates

- **Statistical literacy applied correctly under realistic constraints** — Multiple comparisons, distinguishing drift types, defining and measuring false-positive rate
- **Systems thinking about production ML** — Async ingestion, label latency, reference window tradeoffs
- **Honest scoping of an LLM-based feature** rather than overclaiming it
- **The discipline to validate a tool's own claims** rather than asserting them

### Does Not Demonstrate

- **Novel statistical methods** — K-S and Chi-Square are standard, well-known tests. The contribution here is correct, validated application, not invention.
- **Large-scale distributed systems engineering** — This is intentionally a single-node, local-first tool, not a Kubernetes-scale platform.

---

## Limitations

1. **Single-node only** — The SQLite backend does not support horizontal scaling or concurrent write-heavy workloads.
2. **Twice-daily drift window** — The current hardcoded window cadence (6:00/18:00 UTC) is suitable for demonstration but should be made configurable for production use.
3. **Static reference baseline** — The reference window policy is not yet runtime-configurable (Phase 5 of the build roadmap).
4. **LLM eval requires API key** — The GenAI evaluation module depends on an external API and is not self-hosted.
5. **No authentication** — The API and dashboard have no auth layer; they are designed for local or trusted-network deployment only.

---

## License

Open source. See license file for details.

---

*DriftShield AI — a small, honest monitoring tool that proves its alerts mean something before asking anyone to trust them.*
