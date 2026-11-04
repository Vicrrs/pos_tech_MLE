import numpy as np
import pandas as pd
import pytest

from churn.data import load_dataset
from churn.evaluate import best_threshold, net_value, select_champion
from churn.models import build_candidates, build_pipeline

COMPLEXITY = {"linear": 1, "tree": 2, "mlp": 3}


def comparison(linear: float, tree: float, mlp: float, std: float = 0.01) -> pd.DataFrame:
    return pd.DataFrame(
        {"roc_auc_mean": [linear, tree, mlp, 0.5], "roc_auc_std": [std] * 4},
        index=["linear", "tree", "mlp", "dummy"],
    )


def test_net_value_counts_only_flagged_customers():
    y_true = np.array([1, 0, 1])
    flagged = np.array([True, True, False])
    monthly = np.array([100.0, 100.0, 100.0])
    # Retido: 0.5 * 12 * 100 - 50 = 550. Falso positivo: -50. Não sinalizado: 0.
    value = net_value(y_true, flagged, monthly, success_rate=0.5, offer_cost=50)
    assert value == pytest.approx(500)


def test_best_threshold_separates_perfect_scores():
    y_true = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.8, 0.9])
    threshold = best_threshold(y_true, proba, np.full(4, 100.0))
    assert 0.2 < threshold <= 0.8


def test_select_champion_prefers_simpler_model_when_tied():
    assert select_champion(comparison(0.845, 0.850, 0.848), COMPLEXITY) == "linear"


def test_select_champion_picks_better_model_when_gap_is_real():
    assert select_champion(comparison(0.80, 0.85, 0.82), COMPLEXITY) == "tree"


def test_select_champion_ignores_models_outside_complexity():
    table = comparison(0.70, 0.71, 0.75)
    table.loc["dummy", "roc_auc_mean"] = 0.99
    assert select_champion(table, COMPLEXITY) == "mlp"


@pytest.mark.parametrize("name", ["logistic_regression", "random_forest", "mlp"])
def test_training_is_reproducible_with_fixed_seed(name):
    X, y = load_dataset()
    X, y = X.head(300), y.head(300)
    first = build_pipeline(build_candidates()[name].estimator).fit(X, y).predict_proba(X)
    second = build_pipeline(build_candidates()[name].estimator).fit(X, y).predict_proba(X)
    np.testing.assert_array_equal(first, second)
