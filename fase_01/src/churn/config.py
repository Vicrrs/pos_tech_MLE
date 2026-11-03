"""Constantes do projeto: caminhos, seed, colunas e premissas de negócio."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "champion.joblib"
EXPERIMENTS_DIR = PROJECT_ROOT / "docs" / "experiments"

SEED = 42
TEST_SIZE = 0.2
OUTER_FOLDS = 5
INNER_FOLDS = 3

TARGET = "Churn"
ID_COLUMN = "customerID"
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Premissas ilustrativas: o dataset não traz esses valores. Devem ser substituídas
# pelos números reais da operadora (ver docs/ml_canvas.md).
OFFER_COST = 50.0
RETENTION_SUCCESS_RATE = 0.30
RETAINED_MONTHS = 12
