"""Pré-processamento e modelos candidatos com suas grades de hiperparâmetros.

Nenhum candidato usa `class_weight`: pesos distorcem as probabilidades, e a API devolve
a propensão de churn. O desbalanceamento é tratado na escolha do limiar de decisão.
"""

from dataclasses import dataclass, field

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, SEED


@dataclass(frozen=True)
class Candidate:
    """Modelo a comparar. `complexity` desempata candidatos estatisticamente equivalentes."""

    estimator: BaseEstimator
    param_grid: dict = field(default_factory=dict)
    complexity: int | None = None


def build_preprocessor() -> ColumnTransformer:
    """Imputa e padroniza numéricas; aplica one-hot nas categóricas."""
    numeric = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer(
        [("numeric", numeric, NUMERIC_FEATURES), ("categorical", categorical, CATEGORICAL_FEATURES)]
    )


def build_pipeline(estimator: BaseEstimator) -> Pipeline:
    """Encadeia pré-processamento e modelo, evitando vazamento nos folds da validação."""
    return Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])


def build_candidates(seed: int = SEED) -> dict[str, Candidate]:
    """Referência ingênua, baseline linear, ensemble de árvores e rede neural."""
    return {
        "dummy": Candidate(DummyClassifier(strategy="prior")),
        "logistic_regression": Candidate(
            LogisticRegression(max_iter=2000, random_state=seed),
            {"model__C": [0.01, 0.1, 1.0, 10.0]},
            complexity=1,
        ),
        "random_forest": Candidate(
            RandomForestClassifier(n_estimators=300, random_state=seed),
            {"model__max_depth": [8, 12, None], "model__min_samples_leaf": [5, 20]},
            complexity=2,
        ),
        "mlp": Candidate(
            MLPClassifier(max_iter=500, early_stopping=True, random_state=seed),
            {"model__hidden_layer_sizes": [(32,), (64, 32)], "model__alpha": [1e-4, 1e-2]},
            complexity=3,
        ),
    }
