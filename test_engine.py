from datetime import date
import pandas as pd
from fastapi.testclient import TestClient
from app import app
from engine.race_prediction import build_race_predictions

client = TestClient(app)

def test_health():
    assert client.get("/api/health").json()["status"] == "ok"

def test_wfa_matches_original():
    response = client.get("/api/wfa", params={
        "race_date": "2026-10-16", "distance_m": 1600, "age": 3
    })
    assert response.status_code == 200
    assert response.json()["wfa_lb"] == 16

def test_no_history_does_not_invent_predictions():
    payload = {
        "race_date": "2026-10-16",
        "distance_m": 1600,
        "runners": [{"Horse": "TEST HORSE", "Age": 4, "Weight": 58,
                     "Official MR": 80, "No.": 1}],
        "histories": {}
    }
    response = client.post("/api/predictions", json=payload)
    assert response.status_code == 200
    assert response.json()["rows"] == []
    assert response.json()["consensus"] == []
    assert response.json()["apply_apprentice_claims"] is False
