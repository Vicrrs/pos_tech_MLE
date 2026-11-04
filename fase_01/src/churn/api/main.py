"""API FastAPI. Uso: uvicorn churn.api.main:app --port 8000"""

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from churn.api.schemas import CustomerFeatures, HealthResponse, PredictionResponse
from churn.config import MODEL_PATH
from churn.predict import load_artifact, predict_churn

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Carrega o modelo na subida. Sem ele, a API sobe degradada e /predict responde 503."""
    path = Path(os.environ.get("CHURN_MODEL_PATH", MODEL_PATH))
    try:
        app.state.artifact = load_artifact(path)
    except FileNotFoundError:
        logger.error("Modelo não encontrado em %s. Rode `python -m churn.train`.", path)
        app.state.artifact = None
    yield


app = FastAPI(title="Churn API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health(request: Request) -> HealthResponse:
    """Informa se a API está no ar e se o modelo foi carregado."""
    loaded = request.app.state.artifact is not None
    return HealthResponse(status="ok" if loaded else "degraded", model_loaded=loaded)


@app.post("/predict")
def predict(customer: CustomerFeatures, request: Request) -> PredictionResponse:
    """Recebe os dados de um cliente e devolve a propensão de churn."""
    artifact = request.app.state.artifact
    if artifact is None:
        raise HTTPException(status_code=503, detail="Modelo indisponível.")
    result = predict_churn(artifact, pd.DataFrame([customer.model_dump()])).iloc[0]
    metadata = artifact["metadata"]
    return PredictionResponse(
        churn_probability=float(result["churn_probability"]),
        churn_prediction=bool(result["churn_prediction"]),
        threshold=metadata["threshold"],
        model_name=metadata["model_name"],
    )
