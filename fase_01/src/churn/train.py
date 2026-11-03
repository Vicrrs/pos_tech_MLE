"""Compara os modelos, treina o campeão e salva artefato e resultados.

Uso: python -m churn.train
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from churn.config import (
    EXPERIMENTS_DIR,
    FEATURES,
    INNER_FOLDS,
    MODEL_PATH,
    OUTER_FOLDS,
    SEED,
    TEST_SIZE,
)
from churn.data import load_dataset
from churn.evaluate import (
    best_threshold,
    bootstrap_roc_auc,
    decision_metrics,
    net_value,
    ranking_metrics,
    select_champion,
    subgroup_metrics,
)
from churn.models import Candidate, build_candidates, build_pipeline

logger = logging.getLogger(__name__)

SUBGROUPS = ["gender", "SeniorCitizen", "Dependents", "Contract"]
SUCCESS_RATE_SCENARIOS = [0.1, 0.2, 0.3, 0.5]


def build_search(candidate: Candidate) -> GridSearchCV:
    """Busca em grade interna, otimizando AUC-ROC."""
    return GridSearchCV(
        build_pipeline(candidate.estimator),
        candidate.param_grid,
        scoring="roc_auc",
        cv=StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=SEED),
        n_jobs=-1,
    )


def nested_cv(
    candidate: Candidate, X: pd.DataFrame, y: pd.Series
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Validação cruzada aninhada: devolve predições fora do fold e métricas por fold."""
    outer = StratifiedKFold(OUTER_FOLDS, shuffle=True, random_state=SEED)
    oof_proba = np.zeros(len(y))
    fold_metrics = []
    for fit_idx, val_idx in outer.split(X, y):
        search = build_search(candidate).fit(X.iloc[fit_idx], y.iloc[fit_idx])
        oof_proba[val_idx] = search.predict_proba(X.iloc[val_idx])[:, 1]
        fold_metrics.append(ranking_metrics(y.iloc[val_idx], oof_proba[val_idx]))
    return oof_proba, fold_metrics


def summarize(
    fold_metrics: list[dict[str, float]], y: pd.Series, oof_proba: np.ndarray, monthly: pd.Series
) -> dict[str, float]:
    """Linha da tabela comparativa de um modelo."""
    folds = pd.DataFrame(fold_metrics)
    threshold = best_threshold(y, oof_proba, monthly)
    return {
        "roc_auc_mean": folds["roc_auc"].mean(),
        "roc_auc_std": folds["roc_auc"].std(),
        "pr_auc_mean": folds["pr_auc"].mean(),
        "pr_auc_std": folds["pr_auc"].std(),
        "brier_mean": folds["brier"].mean(),
        "f1_at_0.5": decision_metrics(y, oof_proba, monthly, 0.5)["f1"],
        **decision_metrics(y, oof_proba, monthly, threshold),
    }


def evaluate_on_test(
    pipeline, threshold: float, X_test: pd.DataFrame, y_test: pd.Series
) -> tuple[dict, pd.DataFrame]:
    """Avaliação final no conjunto de teste, usado uma única vez, só com o campeão."""
    proba = pipeline.predict_proba(X_test)[:, 1]
    monthly = X_test["MonthlyCharges"]
    low, high = bootstrap_roc_auc(y_test, proba)
    per_1000 = 1000 / len(y_test)
    metrics = {
        **ranking_metrics(y_test, proba),
        "roc_auc_ci95": [low, high],
        **decision_metrics(y_test, proba, monthly, threshold),
        "n_test": len(y_test),
        "churn_rate_test": float(y_test.mean()),
        "net_value_flag_everyone_per_1000": net_value(y_test, np.ones(len(y_test)), monthly)
        * per_1000,
        "net_value_by_success_rate_per_1000": {
            str(rate): net_value(y_test, proba >= threshold, monthly, success_rate=rate) * per_1000
            for rate in SUCCESS_RATE_SCENARIOS
        },
    }
    return metrics, subgroup_metrics(X_test, y_test, proba, threshold, SUBGROUPS)


def run(model_path: Path = MODEL_PATH, experiments_dir: Path = EXPERIMENTS_DIR) -> dict:
    """Executa o fluxo completo de treino e grava os artefatos."""
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    candidates = build_candidates()

    rows, oof_by_model = {}, {}
    for name, candidate in candidates.items():
        logger.info("Validação cruzada aninhada: %s", name)
        oof_by_model[name], fold_metrics = nested_cv(candidate, X_train, y_train)
        rows[name] = summarize(fold_metrics, y_train, oof_by_model[name], X_train["MonthlyCharges"])
    comparison = pd.DataFrame.from_dict(rows, orient="index").rename_axis("model")

    complexity = {n: c.complexity for n, c in candidates.items() if c.complexity is not None}
    champion = select_champion(comparison, complexity)
    threshold = float(comparison.loc[champion, "threshold"])
    logger.info("Campeão: %s (limiar %.2f)", champion, threshold)

    search = build_search(candidates[champion]).fit(X_train, y_train)
    test_metrics, subgroups = evaluate_on_test(search.best_estimator_, threshold, X_test, y_test)

    metadata = {
        "model_name": champion,
        "best_params": {k: repr(v) for k, v in search.best_params_.items()},
        "threshold": threshold,
        "seed": SEED,
        "features": FEATURES,
        "n_train": len(y_train),
        "sklearn_version": sklearn.__version__,
        "test_metrics": test_metrics,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": search.best_estimator_, "metadata": metadata}, model_path)

    experiments_dir.mkdir(parents=True, exist_ok=True)
    comparison.round(4).to_csv(experiments_dir / "model_comparison.csv")
    subgroups.round(4).to_csv(experiments_dir / "subgroup_metrics.csv", index=False)
    (experiments_dir / "champion_metrics.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
    )
    return metadata


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run()
    logger.info("Teste: %s", json.dumps(result["test_metrics"], indent=2))
