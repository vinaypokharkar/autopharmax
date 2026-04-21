"""AutoPharmaX FastAPI service.

Boots a singleton InferenceService at startup (loads model + feature
caches + scalers), exposes prediction endpoints, and logs each
prediction to a SQLite audit trail via monitoring.logger.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.inference import InferenceService
from src.api.schemas import (
    CellLineInfo,
    CombinationPredictRequest, CombinationPredictResponse,
    CombinationScanRequest, CombinationScanResponse, CombinationScanRow,
    FeatureContribution,
    HealthResponse,
    SinglePredictRequest, SinglePredictResponse,
)
from src.monitoring.logger import PredictionLogger

log = logging.getLogger("autopharmax.api")
ROOT = Path(__file__).resolve().parents[2]


_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log.info("boot: loading inference service")
    _state["inf"]    = InferenceService()
    _state["logger"] = PredictionLogger(ROOT / "predictions.db")
    log.info("boot: ready (cells=%d drugs=%d device=%s version=%s)",
             len(_state["inf"].cell_feats), len(_state["inf"].drug_feats),
             _state["inf"].device, _state["inf"].model_version)
    yield
    # no teardown needed


app = FastAPI(title="AutoPharmaX", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://frontend:8501", "http://127.0.0.1:8501"],
    allow_methods=["GET", "POST"], allow_headers=["*"],
)


def _inf() -> InferenceService:
    return _state["inf"]


def _log() -> PredictionLogger:
    return _state["logger"]


@app.get("/health", response_model=HealthResponse)
def health():
    inf = _inf()
    ldo_r2 = inf.metrics.get("ldo", {}).get("r2") if inf.metrics else None
    return HealthResponse(
        status="ok",
        model_version=inf.model_version,
        single_drug_ldo_r2=ldo_r2,
        cell_features=len(inf.cell_feats),
        drug_features=len(inf.drug_feats),
        device=str(inf.device),
    )


@app.get("/api/v1/cell-lines", response_model=list[CellLineInfo])
def cell_lines():
    inf = _inf()
    out = []
    for cid in sorted(inf.cell_feats.keys()):
        info = inf.cell_line_info.get(int(cid), {})
        out.append(CellLineInfo(cosmic_id=int(cid),
                                cell_line_name=info.get("cell_line_name"),
                                tissue=info.get("tissue")))
    return out


@app.get("/api/v1/drugs", response_model=list[str])
def drugs():
    inf = _inf()
    return sorted(inf.drug_feats.keys())


@app.get("/api/v1/metrics")
def metrics():
    return _inf().metrics


@app.post("/api/v1/predict/single", response_model=SinglePredictResponse)
def predict_single(req: SinglePredictRequest):
    inf = _inf()
    try:
        r = inf.predict_single(int(req.cosmic_id), req.drug_name)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    _log().log_single(
        cosmic_id=int(req.cosmic_id), drug_name=req.drug_name,
        predicted_ln_ic50=r["predicted_ln_ic50"],
        is_novel=bool(r["is_novel_prediction"]),
        latency_ms=float(r["latency_ms"]),
    )
    return SinglePredictResponse(
        predicted_ln_ic50=r["predicted_ln_ic50"],
        predicted_ic50_uM=r["predicted_ic50_uM"],
        sensitivity_class=r["sensitivity_class"],
        is_novel_prediction=r["is_novel_prediction"],
        actual_ln_ic50=r["actual_ln_ic50"],
        absolute_error=r["absolute_error"],
        top_cell_features=[FeatureContribution(**x) for x in r["top_cell_features"]],
        top_drug_features=[FeatureContribution(**x) for x in r["top_drug_features"]],
    )


@app.post("/api/v1/predict/combination", response_model=CombinationPredictResponse)
def predict_combination(req: CombinationPredictRequest):
    inf = _inf()
    try:
        r = inf.predict_combination(int(req.cosmic_id), req.drug_a, req.drug_b)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    _log().log_combination(
        cosmic_id=int(req.cosmic_id), drug_a=req.drug_a, drug_b=req.drug_b,
        synergy_score=float(r["synergy_score"]),
        latency_ms=float(r["latency_ms"]),
    )
    return CombinationPredictResponse(
        synergy_score=r["synergy_score"],
        synergy_class=r["synergy_class"],
        drug_a_ln_ic50=r["drug_a_ln_ic50"],
        drug_b_ln_ic50=r["drug_b_ln_ic50"],
        is_novel_combination=r["is_novel_combination"],
    )


@app.post("/api/v1/predict/combination/scan", response_model=CombinationScanResponse)
def predict_combination_scan(req: CombinationScanRequest):
    inf = _inf()
    try:
        r = inf.combination_scan(int(req.cosmic_id), req.drug_a, int(req.top_k))
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    _log().log_scan(
        cosmic_id=int(req.cosmic_id), drug_a=req.drug_a, top_k=int(req.top_k),
        latency_ms=float(r["latency_ms"]),
    )
    return CombinationScanResponse(
        cosmic_id=r["cosmic_id"],
        drug_a=r["drug_a"],
        top_k=r["top_k"],
        results=[CombinationScanRow(**row) for row in r["results"]],
    )
