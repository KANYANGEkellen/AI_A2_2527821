import json

import pandas as pd
import pytest

from src.data_generation import generate_dataframe, sha256_file, write_dataset
from src.validation import assert_no_leakage, validate_record, validate_seed


def test_valid_inference(trained, valid):
    from src.inference import Predictor
    out = Predictor().predict(valid)
    assert out["verification_code"] == "AI-A2-1234"
    assert 0 <= out["high_risk_probability"] <= 1
    assert out["yield_estimate_tons"] > 0 and isinstance(out["cluster"], int)


def test_missing_field(valid):
    del valid["season"]
    assert any("Missing field: season" in e for e in validate_record(valid))


def test_wrong_type(valid):
    valid["rainfall_mm"] = "lots"
    assert any("must be a number" in e for e in validate_record(valid))


def test_range_error(valid):
    valid["fertilizer_kg"] = 999
    assert any("out of range" in e for e in validate_record(valid))


def test_reproducibility(tmp_path):
    a = write_dataset(1234, tmp_path / "a.csv")
    b = write_dataset(1234, tmp_path / "b.csv")
    assert sha256_file(a) == sha256_file(b)
    assert len(generate_dataframe(1234)) == 240 + (1234 % 41) + 3


def test_data_problems_present():
    df = generate_dataframe(1234)
    m = round(0.05 * (240 + 1234 % 41))
    assert df["soil_moisture_pct"].isna().sum() >= m
    assert df.duplicated().sum() >= 3


def test_seed_must_match_registration():
    assert validate_seed("ug0001234@ines.ac.rw", 1234) == 1234
    with pytest.raises(ValueError):
        validate_seed("ug0001234@ines.ac.rw", 4671)


def test_leakage_guard():
    with pytest.raises(ValueError):
        assert_no_leakage(["rainfall_mm", "yield_tons"])


def test_artifacts_exist(trained):
    for f in ["regression_metrics.json", "classification_metrics.json", "clustering_report.json",
              "test_predictions.csv", "data_fingerprint.txt", "plots/residuals.png"]:
        assert (trained / "artifacts" / f).exists()
    assert json.loads((trained / "artifacts" / "clustering_report.json").read_text())["selected_k"] in range(2, 7)
    assert "actual_yield_tons" in pd.read_csv(trained / "artifacts" / "test_predictions.csv").columns


def test_api_responses(trained, valid):
    from fastapi.testclient import TestClient
    from app import app
    c = TestClient(app)
    assert c.get("/health").json()["status"] == "ok"
    assert c.get("/model-info").json()["verification_code"] == "AI-A2-1234"
    ok = c.post("/predict", json=valid)
    assert ok.status_code == 200 and "yield_estimate_tons" in ok.json()
    bad = dict(valid, rainfall_mm=5000)
    assert c.post("/predict", json=bad).status_code == 422
    assert c.post("/predict", json={"season": "A"}).status_code == 422
