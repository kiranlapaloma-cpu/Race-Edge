# Race Edge 2.0 — Phase 1 backend

This is the **first executable migration milestone**, not the complete React application.

## What is included
- A FastAPI wrapper for the **current uploaded** `race_prediction(10).py`
- The original `performance_profile(8).py` and `wfa(8).py` used by predictions
- JSON endpoints: `/api/health`, `/api/wfa`, `/api/predictions`
- Minimal tests; no database credentials or live data connections

The API accepts runner records in the same field names as the current
prediction engine (`Horse`, `Age`, `Weight`, `Official MR`, `No.`, etc.).
`histories` maps horse names to the historical run records from the existing
Race Edge database. No historical runs means no invented predictions.

## Run locally
```bash
cd backend
python -m pip install -r requirements.txt
uvicorn app:app --reload
```
Open `http://127.0.0.1:8000/docs` for interactive API documentation.

## Tests
```bash
cd backend
python -m pip install pytest httpx
python -m pytest -q
```

## Scope and next work
This intentionally does **not** replace the Streamlit app yet.
`performance_profile.py` imports Streamlit lazily only inside its legacy
UI rendering function; the API calculation path does not require Streamlit.
Next: isolate calculation-only functions from presentation, connect a
**read-only** Supabase history service, establish real-race golden tests,
then build the React form-study workspace.

Apprentice claims default to **not applied**, matching the prediction
worksheet policy. This backend does not alter any existing Race Edge
calculations or PDF reports.

Do not commit Supabase secrets or service-role keys to GitHub.
