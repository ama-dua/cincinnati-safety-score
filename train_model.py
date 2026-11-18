"""
Train the Neighborhood Safety Score model, rebuilt from Gifty Arthur's 2021
University of Cincinnati thesis:
"A Machine Learning Web Application for Predicting Neighborhood Safety
in The City of Cincinnati."

Faithful to the thesis where the current data allows:
  - Source: City of Cincinnati open data, Cincinnati Police Department
    reported crime (the successor feed to the retired PDI Crime Incidents
    dataset k59e-2pvf, which now requires authentication and stopped
    2024-06-02; the STARS-era feed t9kw-3jfx covers 2020 onward).
  - Property crimes only (StarsCategoryType == "Property Crime").
  - Target: CLSD/ReportStatus grouped exactly as in the thesis target_dict:
    CLEARED BY ARREST (adult or juvenile) = 1, everything else = 0.
  - Features (thesis Table 1): year, month, day, hour, minute, dayofweek,
    lat, lon, ucr_group (here: StarsCategory, integer encoded).
  - SMOTE oversampling + Random Forest, 80/20 train/test split.

Documented deviations from the thesis:
  - Victim age / race / gender are NOT used. The current CPD feed no longer
    publishes victim demographics, so those three thesis features cannot be
    rebuilt from public data. This also removes a source of bias.
  - SMOTE is applied to the TRAINING split only, and all reported metrics
    are measured on the untouched test split. (The thesis applied SMOTE to
    the whole dataset before splitting, which inflates test scores.)
  - Data window is 2020-2026 (the STARS feed) instead of 2010-2021.

Outputs (in model/):
  - safety_model.joblib   trained RandomForest on SMOTE-resampled training data
  - metrics.json          honest test-set metrics + dataset facts
  - neighborhoods.json    per-neighborhood centroid + property-crime counts
  - encodings.json        category encodings used by the app
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

DATA_URL = (
    "https://data.cincinnati-oh.gov/api/v3/views/t9kw-3jfx/"
    "export.csv?accessType=DOWNLOAD"
)
ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "cincinnati_reported_crime.csv"
MODEL_DIR = ROOT / "model"

FEATURES = [
    "year", "month", "day", "hour", "minute", "dayofweek",
    "lat", "lon", "ucr_group",
]


def download_data() -> Path:
    RAW.parent.mkdir(parents=True, exist_ok=True)
    if RAW.exists() and RAW.stat().st_size > 1_000_000:
        print(f"Using cached {RAW} ({RAW.stat().st_size/1e6:.1f} MB)")
        return RAW
    print(f"Downloading {DATA_URL}")
    with requests.get(DATA_URL, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(RAW, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    print(f"Saved {RAW} ({RAW.stat().st_size/1e6:.1f} MB)")
    return RAW


def build_dataset() -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(RAW, low_memory=False)
    df["EventDate"] = pd.to_datetime(df["EventDate"], errors="coerce", format="mixed")
    prop = df[df["StarsCategoryType"] == "Property Crime"].copy()
    # STARS-era feed is effectively 2020 onward; drop the handful of earlier rows.
    prop = prop[prop["EventDate"].dt.year >= 2020]
    prop = prop.dropna(subset=["EventDate", "Lat_X", "Lon_X", "SNANeighborhood"])

    # Target, exactly per thesis target_dict: arrest clearances = 1, else 0.
    status = prop["ReportStatus"].astype(str).str.upper()
    prop["clsd"] = status.str.contains("CLEARED BY ARREST|CLEARED_BY_ARREST").astype(int)

    cats = sorted(prop["StarsCategory"].dropna().unique().tolist())
    cat_map = {c: i for i, c in enumerate(cats)}
    prop["ucr_group"] = prop["StarsCategory"].map(cat_map)

    out = pd.DataFrame(
        {
            "year": prop["EventDate"].dt.year,
            "month": prop["EventDate"].dt.month,
            "day": prop["EventDate"].dt.day,
            "hour": prop["EventDate"].dt.hour,
            "minute": prop["EventDate"].dt.minute,
            "dayofweek": prop["EventDate"].dt.day_of_week,  # Monday=0, as in thesis week_dict
            "lat": prop["Lat_X"].astype(float),
            "lon": prop["Lon_X"].astype(float),
            "ucr_group": prop["ucr_group"].astype(int),
            "clsd": prop["clsd"].astype(int),
            "neighborhood": prop["SNANeighborhood"].astype(str),
        }
    )
    # Cincinnati bounding sanity check (matches lat/lon skew note in the source data)
    out = out[out["lat"].between(38.9, 39.4) & out["lon"].between(-84.9, -84.2)]
    return out.reset_index(drop=True), {"ucr_group": cat_map}


def neighborhood_stats(df: pd.DataFrame) -> list[dict]:
    g = df.groupby("neighborhood")
    stats = []
    for name, sub in g:
        stats.append(
            {
                "name": name.title() if name.isupper() else name,
                "raw_name": name,
                "lat": round(float(sub["lat"].mean()), 6),
                "lon": round(float(sub["lon"].mean()), 6),
                "property_crimes": int(len(sub)),
                "arrest_rate": round(float(sub["clsd"].mean()), 4),
            }
        )
    return sorted(stats, key=lambda s: s["name"])


def main() -> None:
    download_data()
    df, encodings = build_dataset()
    print(f"Property-crime rows: {len(df)}  arrest rate: {df['clsd'].mean():.3f}")

    X, y = df[FEATURES], df["clsd"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"Train {len(X_train)} -> SMOTE {len(X_res)} (positive {int(y_res.sum())})")

    # Depth-capped so the artifact stays small enough to ship to free
    # hosting (unlimited-depth was 623 MB; depth 16 is ~14 MB compressed
    # and costs only ~0.002 ROC AUC in testing).
    model = RandomForestClassifier(
        n_estimators=100, max_depth=16, random_state=42, n_jobs=-1,
        min_samples_leaf=2,
    )
    model.fit(X_res, y_res)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    baseline = float(max(y_test.mean(), 1 - y_test.mean()))
    metrics = {
        "rows": int(len(df)),
        "date_min": str(pd.to_datetime(df[["year", "month", "day"]]).min().date()),
        "date_max": str(pd.to_datetime(df[["year", "month", "day"]]).max().date()),
        "arrest_rate": round(float(y.mean()), 4),
        "test_accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "test_precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
        "test_recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
        "test_f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
        "test_roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "test_log_loss": round(float(log_loss(y_test, proba)), 4),
        "baseline_accuracy_all_no_arrest": round(baseline, 4),
        "n_test": int(len(y_test)),
        "feature_importance": {
            f: round(float(v), 4)
            for f, v in zip(FEATURES, model.feature_importances_)
        },
        "thesis_reference": {
            "thesis_test_accuracy_smote_full_dataset": 0.8847,
            "thesis_rows_2010_2021": 133246,
            "thesis_arrests": 13208,
            "note": "Thesis applied SMOTE before splitting, so its 88% is not "
            "directly comparable to the honest held-out metrics above.",
        },
    }
    print(json.dumps(metrics, indent=2))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "safety_model.joblib", compress=3)
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (MODEL_DIR / "encodings.json").write_text(json.dumps(encodings, indent=2))
    (MODEL_DIR / "neighborhoods.json").write_text(
        json.dumps(neighborhood_stats(df), indent=2)
    )
    print(f"Artifacts written to {MODEL_DIR}")


if __name__ == "__main__":
    main()
