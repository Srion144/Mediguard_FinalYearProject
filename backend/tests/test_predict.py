"""
Run with (from inside backend/):
    pytest

Three checks: health check works, a valid patient gets a real prediction,
and an invalid patient gets the documented error shape instead of a crash.
"""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check():
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_valid_patient():
    payload = {
        "age": 55,
        "bmi": 32.0,
        "bp_diastolic": 95,
        "glucose_mgdl": 190,
        "cholesterol_cat": 3,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["error"] is False
    assert 0.0 <= body["diabetes_risk_probability"] <= 1.0
    assert 0.0 <= body["cvd_risk_probability"] <= 1.0
    assert len(body["top_diabetes_drivers"]) == 3
    assert len(body["top_cvd_drivers"]) == 3


def test_predict_invalid_patient():
    payload = {
        "age": 999,  # outside the plausible [1, 120] range
        "bmi": 32.0,
        "bp_diastolic": 95,
        "glucose_mgdl": 190,
        "cholesterol_cat": 3,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200  # not a 4xx -- predict() never raises
    body = resp.json()
    assert body["error"] is True
    assert any("age" in m for m in body["messages"])
