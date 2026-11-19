# Cincinnati Neighborhood Safety Score

## What this app is trying to say

You do not have to guess whether a place feels safe. The city's own crime
history can give you a number for a specific place, at a specific time,
right on your phone. Most crime research studies offenders, who might
offend again. This work flipped the question to victims, and asked what
their patterns in the data could teach the rest of us about avoiding
becoming one. Safety is not a fixed property of a neighborhood. It
changes with the hour and the day, and it can be estimated, imperfectly,
from patterns instead of impressions.

## About this rebuild

Rebuild and relaunch of the Streamlit app from Gifty Ama Dua Arthur's 2021
University of Cincinnati master's thesis, *A Machine Learning Web Application
for Predicting Neighborhood Safety in The City of Cincinnati*
(OhioLINK ETD, accession ucin1627662505704827).

Enter a Cincinnati street address (or pick one of the 51 SNA neighborhoods),
a date and time, and a property-crime type, and the app returns the thesis
safety score: 1 minus the Random Forest model's estimated likelihood that a
reported crime there and then was cleared by arrest, benchmarked at 74.6%
exactly as in thesis section 4.3.2.

## What is original, and what was added

Kept deliberately separate, in the app and here, so the thesis can be
evaluated against the original work:

**From the original thesis (2021):** the safety score and its definition,
Random Forest with SMOTE oversampling, address/date/time/crime-type
inputs, the 74.6% "very safe" benchmark, and feature importance as the
explanation view.

**Added in the 2026 rebuild** (all grouped in the app under one heading,
"Beyond the thesis: improvements added in 2026"): a one-tap example
address; neighborhood context for each result (rank among the 51
neighborhoods, most common crime type, crimes by hour of day); a
year-by-year trend chart; a plain count of property crimes recorded
within about half a mile of the point in the most recent full year;
shareable result links; tap-the-map location scoring; and a
compare-two-neighborhoods tool. Also changed by necessity: the dataset
is the successor feed (2020-2026), victim demographic inputs are gone
because the new feed does not publish them, and the reported metrics are
honest held-out figures with SMOTE kept out of the test data.

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
