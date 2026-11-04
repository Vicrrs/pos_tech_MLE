import joblib
import pytest
from sklearn.linear_model import LogisticRegression

from churn.api.schemas import CUSTOMER_EXAMPLE
from churn.data import load_dataset
from churn.models import build_pipeline


@pytest.fixture
def customer() -> dict:
    return dict(CUSTOMER_EXAMPLE)


@pytest.fixture(scope="session")
def artifact() -> dict:
    """Modelo pequeno, para os testes não dependerem do artefato versionado."""
    X, y = load_dataset()
    pipeline = build_pipeline(LogisticRegression(max_iter=1000)).fit(X.head(1000), y.head(1000))
    return {"pipeline": pipeline, "metadata": {"threshold": 0.3, "model_name": "test_model"}}


@pytest.fixture(scope="session")
def artifact_path(artifact, tmp_path_factory):
    path = tmp_path_factory.mktemp("models") / "model.joblib"
    joblib.dump(artifact, path)
    return path
