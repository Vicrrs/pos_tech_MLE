"""Carga do modelo campeão e predição de churn."""

from pathlib import Path
from typing import TypedDict

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from churn.data import clean_features


class ModelArtifact(TypedDict):
    pipeline: Pipeline
    metadata: dict


def load_artifact(path: Path) -> ModelArtifact:
    """Carrega o artefato salvo por `churn.train`. Abra apenas arquivos de origem confiável."""
    return joblib.load(path)


def predict_churn(artifact: ModelArtifact, customers: pd.DataFrame) -> pd.DataFrame:
    """Devolve a propensão e a decisão de churn de cada cliente."""
    proba = artifact["pipeline"].predict_proba(clean_features(customers))[:, 1]
    return pd.DataFrame(
        {
            "churn_probability": proba,
            "churn_prediction": proba >= artifact["metadata"]["threshold"],
        },
        index=customers.index,
    )
