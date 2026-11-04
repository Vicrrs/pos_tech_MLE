import pandas as pd
import pytest
from fastapi.testclient import TestClient

from churn.api.main import app
from churn.api.schemas import CustomerFeatures
from churn.config import FEATURES, RAW_DATA_PATH
from churn.predict import predict_churn


@pytest.fixture
def client(artifact_path, monkeypatch):
    monkeypatch.setenv("CHURN_MODEL_PATH", str(artifact_path))
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def client_without_model(tmp_path, monkeypatch):
    monkeypatch.setenv("CHURN_MODEL_PATH", str(tmp_path / "missing.joblib"))
    with TestClient(app) as test_client:
        yield test_client


def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_health_reports_missing_model(client_without_model):
    response = client_without_model.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "degraded", "model_loaded": False}


def test_predict_returns_503_without_model(client_without_model, customer):
    assert client_without_model.post("/predict", json=customer).status_code == 503


def test_predict_matches_model_output(client, artifact, customer):
    response = client.post("/predict", json=customer)
    assert response.status_code == 200
    body = response.json()
    expected = predict_churn(artifact, pd.DataFrame([customer])).iloc[0]
    assert body["churn_probability"] == pytest.approx(expected["churn_probability"])
    assert body["churn_prediction"] == (body["churn_probability"] >= body["threshold"])
    assert body["model_name"] == "test_model"


@pytest.mark.parametrize(
    "field, value",
    [
        ("tenure", -1),
        ("Contract", "Weekly"),
        ("SeniorCitizen", 2),
        ("TotalCharges", "abc"),
        ("unexpected_field", 1),
        ("MultipleLines", "No phone service"),
        ("OnlineSecurity", "No internet service"),
    ],
)
def test_predict_rejects_invalid_payload(client, customer, field, value):
    assert client.post("/predict", json=customer | {field: value}).status_code == 422


def test_predict_rejects_missing_field(client, customer):
    del customer["tenure"]
    assert client.post("/predict", json=customer).status_code == 422


def test_schema_accepts_every_customer_in_the_dataset():
    raw = pd.read_csv(RAW_DATA_PATH)
    raw["TotalCharges"] = pd.to_numeric(raw["TotalCharges"], errors="coerce").fillna(0.0)
    for record in raw[FEATURES].to_dict(orient="records"):
        CustomerFeatures(**record)
