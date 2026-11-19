# Cincinnati Neighborhood Safety Score

Rebuild and relaunch of the Streamlit app from Gifty Ama Dua Arthur's 2021
University of Cincinnati master's thesis, *A Machine Learning Web Application
for Predicting Neighborhood Safety in The City of Cincinnati*
(OhioLINK ETD, accession ucin1627662505704827).

Enter a Cincinnati street address (or pick one of the 51 SNA neighborhoods),
a date and time, and a property-crime type, and the app returns the thesis
safety score: 1 minus the Random Forest model's estimated likelihood that a
reported crime there and then was cleared by arrest, benchmarked at 74.6%
exactly as in thesis section 4.3.2. A one-tap example address is provided
for first-time visitors. Each result includes the neighborhood's rank among
the 51 neighborhoods, its most common property-crime type, and a chart of
recorded crimes by hour of day, and a separate section compares any two
neighborhoods side by side.

A GitHub Actions workflow (`.github/workflows/retrain.yml`) retrains the
model on the latest city data on the 1st of each month and commits the
refreshed artifacts, which Streamlit Community Cloud then redeploys.

## Run locally

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt imbalanced-learn
.venv/bin/streamlit run app.py
```

## Retrain

```bash
.venv/bin/python train_model.py
```

Downloads the City of Cincinnati Reported Crime dataset (t9kw-3jfx, the
successor to the retired PDI Crime Incidents feed), filters property crimes
from 2020 onward, encodes the target exactly as the thesis did (cleared by
arrest = 1, else 0), applies SMOTE to the training split only, trains a
Random Forest, and writes honest held-out metrics plus the model artifacts
to `model/`. See the `train_model.py` header for the documented deviations
from the thesis (no victim demographics in the current feed; SMOTE on the
training split only).

## Honest performance

Held-out test (16,474 incidents): accuracy 0.717, ROC AUC 0.762,
recall 0.641, precision 0.269. The thesis reported 88% accuracy with SMOTE
applied before splitting; the two numbers are not directly comparable, and
the app says so on its own page. The forest is depth-capped at 16 so the
model file stays at 14 MB for free hosting, at a cost of about 0.002 AUC.
