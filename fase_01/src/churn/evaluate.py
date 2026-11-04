"""Métricas técnicas e de negócio, escolha do limiar e seleção do modelo campeão."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from churn.config import OFFER_COST, RETAINED_MONTHS, RETENTION_SUCCESS_RATE, SEED

THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)


def net_value(
    y_true: np.ndarray,
    flagged: np.ndarray,
    monthly_charges: np.ndarray,
    success_rate: float = RETENTION_SUCCESS_RATE,
    offer_cost: float = OFFER_COST,
) -> float:
    """Valor líquido de ofertar retenção aos clientes sinalizados.

    Cada oferta custa `offer_cost`. Um cliente que cancelaria é retido com probabilidade
    `success_rate`, preservando `RETAINED_MONTHS` meses da sua mensalidade.
    """
    y_true, flagged = np.asarray(y_true), np.asarray(flagged, dtype=bool)
    retained = y_true * success_rate * RETAINED_MONTHS * np.asarray(monthly_charges)
    return float(np.sum(flagged * (retained - offer_cost)))


def best_threshold(y_true: np.ndarray, proba: np.ndarray, monthly_charges: np.ndarray) -> float:
    """Limiar que maximiza o valor líquido. Usar só com predições fora da amostra de treino."""
    values = [net_value(y_true, proba >= t, monthly_charges) for t in THRESHOLD_GRID]
    return float(THRESHOLD_GRID[int(np.argmax(values))])


def ranking_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    """Métricas que independem de limiar."""
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "brier": brier_score_loss(y_true, proba),
    }


def decision_metrics(
    y_true: np.ndarray, proba: np.ndarray, monthly_charges: np.ndarray, threshold: float
) -> dict[str, float]:
    """Métricas no limiar de decisão, com valor líquido normalizado por 1.000 clientes."""
    flagged = proba >= threshold
    return {
        "threshold": threshold,
        "f1": f1_score(y_true, flagged, zero_division=0),
        "precision": precision_score(y_true, flagged, zero_division=0),
        "recall": recall_score(y_true, flagged, zero_division=0),
        "flagged_rate": float(flagged.mean()),
        "net_value_per_1000": net_value(y_true, flagged, monthly_charges) * 1000 / len(y_true),
    }


def bootstrap_roc_auc(
    y_true: np.ndarray, proba: np.ndarray, n_iterations: int = 1000, seed: int = SEED
) -> tuple[float, float]:
    """Intervalo de confiança de 95% da AUC-ROC por bootstrap."""
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(y_true), size=(n_iterations, len(y_true)))
    scores = [roc_auc_score(y_true[idx], proba[idx]) for idx in samples]
    low, high = np.percentile(scores, [2.5, 97.5])
    return float(low), float(high)


def subgroup_metrics(
    features: pd.DataFrame,
    y_true: np.ndarray,
    proba: np.ndarray,
    threshold: float,
    groups: list[str],
) -> pd.DataFrame:
    """Desempenho por subgrupo, para a análise de vieses do Model Card."""
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    rows = []
    for column in groups:
        for value in sorted(features[column].unique()):
            mask = (features[column] == value).to_numpy()
            flagged = proba[mask] >= threshold
            rows.append(
                {
                    "group": column,
                    "value": value,
                    "n": int(mask.sum()),
                    "churn_rate": y_true[mask].mean(),
                    "roc_auc": roc_auc_score(y_true[mask], proba[mask]),
                    "recall": recall_score(y_true[mask], flagged, zero_division=0),
                    "precision": precision_score(y_true[mask], flagged, zero_division=0),
                    "flagged_rate": flagged.mean(),
                }
            )
    return pd.DataFrame(rows)


def select_champion(comparison: pd.DataFrame, complexity: dict[str, int]) -> str:
    """Escolhe o modelo mais simples dentro de um desvio-padrão do melhor em AUC-ROC.

    `comparison` é indexado pelo nome do modelo. Só concorrem os modelos em `complexity`.
    """
    eligible = comparison.loc[list(complexity)]
    best = eligible.loc[eligible["roc_auc_mean"].idxmax()]
    cutoff = best["roc_auc_mean"] - best["roc_auc_std"]
    tied = eligible.index[eligible["roc_auc_mean"] >= cutoff]
    return min(tied, key=complexity.__getitem__)
