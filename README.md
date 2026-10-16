# Musanze Harvest Intelligence Service (AI-A2-7821)

- **Name:** KANYANGE Kellen
- **Registration number:** <25/27821>   **Username:** ug2527821@ines.ac.rw
- **Verification code:** AI-A2-7821
- **Repository URL:** <URL>   **Final commit hash:** <HASH>
- **Python:** 3.12   **Tested OS:** <Windows/Linux/macOS>

> The dataset is **synthetic**. This is a classroom decision-support prototype and
> does not replace an agronomist or decide finance, insurance or public services.

## Problem
Estimate harvest (regression), flag high-risk farms (classification), group similar farms (clustering).

## Setup and run
```
python -m venv .venv
.venv\Scripts\activate          # Windows   (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
python run_all.py --student-id ug2527821@ines.ac.rw --seed 7821 --reg-digits 2527821
python predict.py --json samples/valid.json
python predict.py --json samples/invalid.json
pytest -q
uvicorn app:app --host 127.0.0.1 --port 8000     # docs at http://127.0.0.1:8000/docs
```

## Artifacts
`artifacts/` regression_metrics.json, classification_metrics.json, clustering_report.json,
test_predictions.csv, data_fingerprint.txt, data_quality.json, plots/*.png; `models/` saved pipelines.

## API example
`POST /predict` with `samples/valid.json` -> yield, risk, probability, cluster, verification code.
Invalid input -> HTTP 422 with field-level messages.

## Known limitations
Synthetic data, linear-ish generating formula, small test set (~56 rows), no real agronomy validation.
No extra packages beyond requirements.txt (httpx is needed only by FastAPI's TestClient).
