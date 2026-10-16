"""FastAPI service: uvicorn app:app --host 127.0.0.1 --port 8000"""
from functools import lru_cache

from fastapi import FastAPI, HTTPException

from src import config as C
from src.inference import FILES, Predictor
from src.schemas import PredictionInput

app = FastAPI(title="Musanze Harvest Intelligence Service", version="1.0.0",
              description=C.DISCLAIMER)


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    return Predictor()


def _predictor() -> Predictor:
    try:
        return get_predictor()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/health")
def health():
    """Service is up; reports whether trained models exist."""
    ready = all((C.MODELS_DIR / f).exists() for f in FILES)
    return {"status": "ok", "models_ready": ready}


@app.get("/model-info")
def model_info():
    """Verification code, selected models and cluster profiles."""
    m = _predictor().meta
    return {k: m[k] for k in ("verification_code", "student_id", "seed", "sha256", "selected_regression",
                              "selected_classifier", "selected_k", "cluster_profiles",
                              "n_train", "n_test", "disclaimer")}


@app.post("/predict")
def predict(item: PredictionInput):
    """Validated prediction for one farm record."""
    return _predictor().predict(item.model_dump())
