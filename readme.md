# HRV-Lab (LAN) — FastAPI + SQLite (MVP)

ingest Elite HRV CSVs, track interventions/compliance, and view basic charts.

## Run
```bash
python -m venv .venv
source .venv/Scripts/activate    # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000


##Updating gmail token
#delete token.json
#activate venv with source .venv\Scripts\Activate from hrv-lab folder
#Launch the OAuth flow with python -m pipeline.gmail_fetch