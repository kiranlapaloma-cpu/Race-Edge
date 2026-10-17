# Supabase is not required for mocked endpoint tests.
import sys
import types
if "supabase" not in sys.modules:
    try:
        import supabase
    except ImportError:
        sys.modules["supabase"] = types.SimpleNamespace(create_client=lambda *a, **k: None)
from datetime import date
from unittest.mock import patch
from fastapi.testclient import TestClient
from app import app
from race_card_service import prediction_runners

client = TestClient(app)

CARD = {"distance": 1400, "runners": [
    {"saddleNo": 7, "horseName": "Test Horse", "age": 4, "weight": 58,
     "MR": 0, "apprenticeClaim": 2.5, "status": "F", "scratched": False},
    {"saddleNo": 8, "horseName": "Reserve Horse", "age": 3, "weight": 56,
     "status": "R", "scratched": False},
    {"saddleNo": 9, "horseName": "Scratched Horse", "age": 3, "weight": 56,
     "status": "S", "scratched": True},
]}

def test_mapping_preserves_claim_without_applying_it():
    rows = prediction_runners(CARD)
    assert rows[0]["Claim"] == 2.5
    assert rows[0]["Weight"] == 58
    assert rows[0]["Official MR"] is None
    assert {r["Status"] for r in rows} == {"Runner", "Reserve", "Scratched"}

def test_cards_require_key(monkeypatch):
    monkeypatch.setenv("RACE_EDGE_API_KEY", "private-test-key")
    assert client.get("/api/cards/meetings?race_date=2026-10-18").status_code == 401

@patch("app.get_meetings_for_date")
def test_meeting_discovery(mock_meetings, monkeypatch):
    monkeypatch.setenv("RACE_EDGE_API_KEY", "private-test-key")
    mock_meetings.return_value = [{"date": "20261018", "club": 3, "name": "Fairview", "raw": {}}]
    result = client.get("/api/cards/meetings?race_date=2026-10-18", headers={"x-api-key": "private-test-key"})
    assert result.status_code == 200
    assert result.json()["meetings"][0]["name"] == "Fairview"

@patch("app.build_race_predictions")
@patch("app.horse_history")
@patch("app._load_card")
def test_predictions_hide_peak_by_default(mock_card, mock_history, mock_predict, monkeypatch):
    monkeypatch.setenv("RACE_EDGE_API_KEY", "private-test-key")
    mock_card.return_value = CARD
    mock_history.return_value = []
    mock_predict.return_value = {"rows": [], "scenarios": {"Recent Best": {}, "Latest Form": {}, "Peak Ability": {}}, "consensus": []}
    result = client.get("/api/cards/predictions?race_date=2026-10-18&club=3&race=1", headers={"x-api-key": "private-test-key"})
    assert result.status_code == 200
    assert "Peak Ability" not in result.json()["prediction"]["scenarios"]
    assert mock_predict.call_args.kwargs["apply_apprentice_claims"] is False
    result = client.get("/api/cards/predictions?race_date=2026-10-18&club=3&race=1&include_peak=true", headers={"x-api-key": "private-test-key"})
    assert "Peak Ability" in result.json()["prediction"]["scenarios"]
