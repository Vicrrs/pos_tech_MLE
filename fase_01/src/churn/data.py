"""Carga e limpeza dos dados, compartilhada entre treino e inferência."""

from pathlib import Path

import pandas as pd

from churn.config import FEATURES, RAW_DATA_PATH, TARGET


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Devolve só as features, com tipos corrigidos e na ordem esperada pelo modelo.

    `TotalCharges` vem como texto e fica em branco para clientes com `tenure == 0`,
    que ainda não receberam cobrança: nesse caso o valor correto é 0, não um dado
    ausente. Qualquer outro valor inválido vira NaN e é imputado no pipeline.
    """
    clean = df[FEATURES].copy()
    clean["TotalCharges"] = pd.to_numeric(clean["TotalCharges"], errors="coerce").astype(float)
    clean.loc[clean["TotalCharges"].isna() & (clean["tenure"] == 0), "TotalCharges"] = 0.0
    clean["SeniorCitizen"] = clean["SeniorCitizen"].map({0: "No", 1: "Yes"})
    return clean


def encode_target(churn: pd.Series) -> pd.Series:
    """Converte a coluna alvo de Yes/No para 1/0, rejeitando valores inesperados."""
    encoded = churn.map({"Yes": 1, "No": 0})
    if encoded.isna().any():
        raise ValueError("A coluna alvo deve conter apenas 'Yes' ou 'No'.")
    return encoded.astype(int)


def load_dataset(path: Path = RAW_DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Lê o CSV bruto e devolve features limpas e alvo binário."""
    raw = pd.read_csv(path)
    return clean_features(raw), encode_target(raw[TARGET])
