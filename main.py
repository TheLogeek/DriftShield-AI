import asyncio
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from database import init_db, insert_inference, get_inference_window, get_reference_baselines
from database import insert_drift_result, get_recent_inferences, insert_label

from drift_engine import detect_covariate_drift, detect_concept_drift, compute_performance_drop


class LogRequest(BaseModel):
    features: dict = Field(..., description="Feature dictionary")
    prediction: float = Field(..., description="Model prediction")
    label: Optional[float] = Field(None, description="Ground truth label if available")
    request_id: Optional[str] = Field(None, description="Client-side request ID")


class LabelRequest(BaseModel):
    request_id: str = Field(..., description="Request ID to label")
    label: float = Field(..., description="Ground truth label")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="DriftShield AI",
    description="Lightweight, Local-First Production Model Monitoring Engine",
    version="1.0.0",
    lifespan=lifespan,
)

_drift_lock = asyncio.Lock()


@app.post("/log")
async def log_inference(req: LogRequest):
    rid = req.request_id or str(uuid.uuid4())
    insert_inference(
        request_id=rid,
        features=req.features,
        prediction=req.prediction,
        label=req.label,
    )
    asyncio.create_task(_run_drift_check())
    return {"status": "ok", "request_id": rid}


@app.post("/label")
async def log_label(req: LabelRequest):
    insert_label(request_id=req.request_id, label=req.label)
    return {"status": "ok"}


@app.post("/reference/baseline")
async def set_reference_baseline():
    from database import upsert_reference_baseline
    rows = get_recent_inferences(limit=5000)
    if not rows:
        raise HTTPException(status_code=400, detail="No inference data available")
    features_by_name: dict[str, list] = {}
    feature_types: dict[str, str] = {}
    for row in rows:
        import json
        feats = json.loads(row["features_json"])
        for key, val in feats.items():
            features_by_name.setdefault(key, []).append(val)
            if key not in feature_types:
                feature_types[key] = "numerical" if isinstance(val, (int, float)) else "categorical"
    for fname, vals in features_by_name.items():
        import numpy as np
        if feature_types[fname] == "numerical":
            numeric = [v for v in vals if isinstance(v, (int, float))]
            baseline = {
                "mean": float(np.mean(numeric)),
                "std": float(np.std(numeric)) if len(numeric) > 1 else 0.0,
                "min": float(np.min(numeric)),
                "max": float(np.max(numeric)),
                "p50": float(np.median(numeric)),
                "p5": float(np.percentile(numeric, 5)),
                "p95": float(np.percentile(numeric, 95)),
            }
        else:
            from collections import Counter
            counts = Counter(str(v) for v in vals)
            total = sum(counts.values())
            baseline = {
                "categories": {k: v for k, v in counts.most_common()},
                "frequencies": {k: round(v / total, 4) for k, v in counts.most_common()},
            }
        upsert_reference_baseline(fname, feature_types[fname], baseline, len(vals))
    return {"status": "ok", "features": list(features_by_name.keys()), "n_samples": len(rows)}


@app.get("/drift/latest")
async def latest_drift():
    from database import get_latest_drift_results
    return [dict(r) for r in get_latest_drift_results()]


@app.get("/drift/history")
async def drift_history(limit: int = 200):
    from database import get_recent_drift_results
    return [dict(r) for r in get_recent_drift_results(limit)]


@app.get("/health")
async def health():
    return {"status": "healthy"}


async def _run_drift_check():
    async with _drift_lock:
        try:
            refs = get_reference_baselines()
            if not refs:
                return
            import json
            ref_data: dict[str, list] = {}
            feature_types: dict[str, str] = {}
            for r in refs:
                ref_data[r["feature_name"]] = list(json.loads(r["baseline_json"]).values())
                feature_types[r["feature_name"]] = r["feature_type"]

            now = datetime.now(timezone.utc)
            window_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            if now.hour < 6:
                window_start = (now - __import__("datetime").timedelta(days=1)).replace(
                    hour=6, minute=0, second=0, microsecond=0
                )
            elif now.hour < 18:
                window_start = window_start.replace(hour=6)
            else:
                window_start = window_start.replace(hour=18)

            prod_rows = get_inference_window(
                since=window_start.isoformat(),
                until=now.isoformat(),
            )
            if not prod_rows:
                return

            prod_data: dict[str, list] = {}
            for row in prod_rows:
                feats = json.loads(row["features_json"])
                for key, val in feats.items():
                    prod_data.setdefault(key, []).append(val)

            result = detect_covariate_drift(ref_data, prod_data, feature_types, fdr=0.05)

            for r in result["results"]:
                insert_drift_result(
                    feature_name=r["feature"],
                    test_name=r["test_name"],
                    statistic=r["statistic"],
                    p_value=r["p_value"],
                    corrected_p_value=r["corrected_p_value"],
                    significant=r["significant"],
                    drift_type=r["drift_type"],
                    window_end_at=now.isoformat(),
                )
        except Exception:
            import traceback
            traceback.print_exc()
