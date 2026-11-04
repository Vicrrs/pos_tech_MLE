import numpy as np
import pandas as pd
import pytest

from churn.config import FEATURES
from churn.data import clean_features, encode_target, load_dataset
from churn.models import build_preprocessor


@pytest.fixture
def raw_rows(customer) -> pd.DataFrame:
    rows = pd.DataFrame([customer, customer, customer])
    rows["customerID"] = ["a", "b", "c"]
    rows["tenure"] = [0, 5, 7]
    rows["SeniorCitizen"] = [0, 1, 0]
    rows["TotalCharges"] = [" ", "100.5", "not a number"]
    return rows


def test_blank_total_charges_is_zero_for_new_customers(raw_rows):
    assert clean_features(raw_rows)["TotalCharges"].iloc[0] == 0.0


def test_total_charges_is_parsed_and_invalid_values_become_nan(raw_rows):
    total_charges = clean_features(raw_rows)["TotalCharges"]
    assert total_charges.iloc[1] == 100.5
    assert np.isnan(total_charges.iloc[2])


def test_clean_features_keeps_only_features_without_changing_input(raw_rows):
    original = raw_rows.copy()
    clean = clean_features(raw_rows)
    assert list(clean.columns) == FEATURES
    assert clean["SeniorCitizen"].tolist() == ["No", "Yes", "No"]
    pd.testing.assert_frame_equal(raw_rows, original)


def test_encode_target():
    assert encode_target(pd.Series(["Yes", "No"])).tolist() == [1, 0]


def test_encode_target_rejects_unknown_labels():
    with pytest.raises(ValueError):
        encode_target(pd.Series(["Yes", "Maybe"]))


def test_load_dataset_has_no_missing_values():
    X, y = load_dataset()
    assert len(X) == len(y) == 7043
    assert X.isna().sum().sum() == 0
    assert set(y.unique()) == {0, 1}


def test_preprocessor_ignores_unseen_categories():
    X, _ = load_dataset()
    preprocessor = build_preprocessor().fit(X)
    unseen = X.head(1).copy()
    unseen["PaymentMethod"] = "Pix"
    assert not np.isnan(preprocessor.transform(unseen)).any()
