"""
Cincinnati Neighborhood Safety Score — rebuilt and relaunched 2026.

A faithful rebuild of the Streamlit app from Gifty Ama Dua Arthur's 2021
University of Cincinnati master's thesis, "A Machine Learning Web
Application for Predicting Neighborhood Safety in The City of Cincinnati"
(OhioLINK accession ucin1627662505704827).

Method (thesis): Random Forest trained with SMOTE oversampling on City of
Cincinnati Police Department property-crime incidents. The model predicts
the likelihood that a reported property crime at a location and time was
cleared by arrest; the safety score is 1 minus that probability, exactly
as defined in thesis section 4.3.1.

Rebuild notes:
  - The thesis dataset (PDI Crime Incidents, k59e-2pvf) was retired by the
    city on 2024-06-02 and now requires authentication. This rebuild trains
    on its successor feed (Reported Crime, t9kw-3jfx), 2020-2026.
  - Victim age, race and gender were thesis features, but the current CPD
    feed no longer publishes victim demographics, so they are not inputs
    here. The model uses time, location and crime type only.
  - Performance below is measured on a held-out test split that SMOTE
    never touched, so it is lower, and more honest, than the thesis's 88%.
"""

import json
import math
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"

FEATURES = ["year", "month", "day", "hour", "minute", "dayofweek", "lat", "lon", "ucr_group"]
CRIME_LABELS = {
    "Auto Theft": "Auto Theft",
    "Burglary/BE": "Burglary / Breaking & Entering",
    "Personal/Other Theft": "Theft (personal / other)",
    "Theft from Auto": "Theft from Auto",
}
BENCHMARK = 0.746  # Thesis 4.3.2: Cincinnati property-crime rate benchmark (BestPlaces, 2021)


def _data_version() -> str:
    """Hash of the small data files, used as the cache key below.

    Without this, Streamlit's cache can keep serving the previous
    neighborhoods/metrics after a redeploy that only changed the data
    files (which is exactly what the monthly retrain does), pairing new
    code with stale data. Hashing the files makes any data change load
    fresh artifacts.
    """
    import hashlib

    h = hashlib.sha256()
    for name in ("metrics.json", "neighborhoods.json", "encodings.json"):
        h.update((MODEL_DIR / name).read_bytes())
    return h.hexdigest()


@st.cache_resource
def load_artifacts(version: str):
    model = joblib.load(MODEL_DIR / "safety_model.joblib")
    metrics = json.loads((MODEL_DIR / "metrics.json").read_text())
    encodings = json.loads((MODEL_DIR / "encodings.json").read_text())
    neighborhoods = json.loads((MODEL_DIR / "neighborhoods.json").read_text())
    return model, metrics, encodings, neighborhoods


def geocode(address: str):
    """Free OpenStreetMap Nominatim geocoding; returns (lat, lon) or None."""
    q = address if "cincinnati" in address.lower() else f"{address}, Cincinnati, OH"
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": q, "format": "json", "limit": 1},
            headers={"User-Agent": "cincinnati-safety-score-thesis-rebuild/1.0"},
            timeout=15,
        )
        r.raise_for_status()
        hits = r.json()
        if hits:
            return float(hits[0]["lat"]), float(hits[0]["lon"])
    except Exception:
        return None
    return None


def nearest_neighborhood(lat: float, lon: float, neighborhoods: list[dict]) -> dict:
    def dist(n):
        return math.hypot(lat - n["lat"], lon - n["lon"])

    return min(neighborhoods, key=dist)


def main():
    st.set_page_config(page_title="Cincinnati Neighborhood Safety Score", page_icon="🛡️", layout="centered")
    model, metrics, encodings, neighborhoods = load_artifacts(_data_version())
    cat_map = encodings["ucr_group"]

    st.title("Cincinnati Neighborhood Safety Score")
    st.caption(
        "Rebuilt from the 2021 University of Cincinnati master's thesis by "
        "Gifty Ama Dua Arthur: *A Machine Learning Web Application for "
        "Predicting Neighborhood Safety in The City of Cincinnati.*"
    )

    st.subheader("Check a location")
    mode = st.radio("Location", ["Street address", "Neighborhood"], horizontal=True)

    def _try_example():
        st.session_state["address"] = "2600 Vine St"
        st.session_state["auto_predict"] = True

    st.button("Try an example: 2600 Vine St, near the University of Cincinnati",
              on_click=_try_example)

    lat = lon = None
    chosen_name = None
    if mode == "Street address":
        address = st.text_input("Street address in Cincinnati", placeholder="e.g. 2600 Vine St",
                                 key="address")
        if address:
            found = geocode(address)
            if found:
                lat, lon = found
                st.success(f"Found: {lat:.5f}, {lon:.5f}")
            else:
                st.warning("That address could not be located. Check the spelling, or use the Neighborhood option instead.")
    else:
        names = [n["name"] for n in neighborhoods]
        chosen_name = st.selectbox("Neighborhood (SNA)", names)
        n = next(x for x in neighborhoods if x["name"] == chosen_name)
        lat, lon = n["lat"], n["lon"]
        st.caption(f"Uses the neighborhood center: {lat:.5f}, {lon:.5f}")

    col1, col2 = st.columns(2)
    with col1:
        date = st.date_input("Date", value=datetime.now().date())
    with col2:
        time = st.time_input("Time", value=datetime.now().time().replace(second=0, microsecond=0))

    crime_label = st.selectbox(
        "Property crime type the score is modelled on",
        [CRIME_LABELS[k] for k in sorted(CRIME_LABELS)],
    )
    crime_key = next(k for k, v in CRIME_LABELS.items() if v == crime_label)

    st.caption(
        "The original thesis also asked for the visitor's age group, race and gender. "
        "The Cincinnati Police Department's current open-data feed no longer publishes "
        "victim demographics, so this rebuild does not ask for them."
    )

    if st.button("Get my safety score", type="primary") or st.session_state.pop("auto_predict", False):
        if lat is None or lon is None:
            st.error("Add a Cincinnati street address or pick a neighborhood first.")
            return
        dt = datetime.combine(date, time)
        sample = pd.DataFrame(
            [{
                "year": dt.year, "month": dt.month, "day": dt.day,
                "hour": dt.hour, "minute": dt.minute,
                "dayofweek": dt.weekday(), "lat": lat, "lon": lon,
                "ucr_group": cat_map[crime_key],
            }],
            columns=FEATURES,
        )
        p_arrest = float(model.predict_proba(sample)[0, 1])
        safety = 1.0 - p_arrest

        if safety > BENCHMARK:
            st.success(f"Your safety score is {safety * 100:.1f}%")
        else:
            st.warning(f"Your safety score is {safety * 100:.1f}%")
        st.write(
            f"That is 1 minus the model's estimated likelihood ({p_arrest * 100:.1f}%) that a "
            f"reported {crime_label.lower()} at this place and time ended in an arrest, the "
            f"definition used in the thesis. The thesis benchmark is {BENCHMARK * 100:.1f}%: "
            "above it the location was classed as very safe, below it the app advised caution."
        )

        near = nearest_neighborhood(lat, lon, neighborhoods)
        st.subheader(f"Context: {near['name']}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Property crimes recorded", f"{near['property_crimes']:,}")
        c2.metric("Rank of 51 neighborhoods", f"#{near['rank_by_crimes']}")
        c3.metric("Cleared by arrest there", f"{near['arrest_rate'] * 100:.1f}%")
        c4.metric("Citywide arrest rate", f"{metrics['arrest_rate'] * 100:.1f}%")
        st.caption(
            "Rank 1 means the most recorded property crimes in the city over the data "
            "period. The most common type recorded in this neighborhood is "
            f"{CRIME_LABELS.get(near['top_category'], near['top_category'])} "
            f"({near['top_category_share'] * 100:.0f}% of its property crimes)."
        )
        st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}), zoom=13)
        hourly = pd.Series(near["hourly_counts"], index=[f"{h:02d}:00" for h in range(24)])
        peak_hour = int(pd.Series(near["hourly_counts"]).idxmax())
        st.write(f"**When property crime is recorded in {near['name']}** (by hour of day, "
                 f"peaking around {peak_hour:02d}:00):")
        st.bar_chart(hourly)

        with st.expander("What the model weighed most (feature importance)"):
            imp = pd.Series(metrics["feature_importance"]).sort_values(ascending=True)
            st.bar_chart(imp)
        with st.expander("Encoded input sent to the model"):
            st.dataframe(sample)

    st.divider()
    st.subheader("Compare two neighborhoods")
    names_sorted = [n["name"] for n in neighborhoods]
    by_name = {n["name"]: n for n in neighborhoods}
    default_b = "Oakley" if "Oakley" in names_sorted else names_sorted[1]
    default_a = "Westwood" if "Westwood" in names_sorted else names_sorted[0]
    col_a, col_b = st.columns(2)
    with col_a:
        pick_a = st.selectbox("Neighborhood A", names_sorted, index=names_sorted.index(default_a), key="cmp_a")
    with col_b:
        pick_b = st.selectbox("Neighborhood B", names_sorted, index=names_sorted.index(default_b), key="cmp_b")
    a, b = by_name[pick_a], by_name[pick_b]
    comp = pd.DataFrame(
        {
            pick_a: [
                f"{a['property_crimes']:,}", f"#{a['rank_by_crimes']} of 51",
                f"{a['arrest_rate'] * 100:.1f}%",
                f"{CRIME_LABELS.get(a['top_category'], a['top_category'])} ({a['top_category_share'] * 100:.0f}%)",
            ],
            pick_b: [
                f"{b['property_crimes']:,}", f"#{b['rank_by_crimes']} of 51",
                f"{b['arrest_rate'] * 100:.1f}%",
                f"{CRIME_LABELS.get(b['top_category'], b['top_category'])} ({b['top_category_share'] * 100:.0f}%)",
            ],
        },
        index=["Property crimes recorded", "Rank by recorded crimes",
               "Cleared by arrest", "Most common type"],
    )
    st.table(comp)
    st.write("**Recorded property crimes by hour of day:**")
    st.bar_chart(pd.DataFrame(
        {pick_a: a["hourly_counts"], pick_b: b["hourly_counts"]},
        index=[f"{h:02d}:00" for h in range(24)],
    ))

    st.divider()
    st.subheader("How this works, honestly")
    st.write(
        f"The model is a Random Forest trained on **{metrics['rows']:,} property-crime "
        f"incidents** reported to the Cincinnati Police Department from "
        f"{metrics['date_min']} to {metrics['date_max']} (theft, theft from auto, auto theft, "
        f"and burglary). {metrics['arrest_rate'] * 100:.1f}% of them were cleared by arrest. "
        "Following the thesis, the training data was balanced with SMOTE oversampling."
    )
    m = pd.DataFrame(
        {
            "This rebuild (held-out test)": [
                f"{metrics['test_accuracy']:.2f}", f"{metrics['test_roc_auc']:.2f}",
                f"{metrics['test_recall']:.2f}", f"{metrics['test_precision']:.2f}",
            ],
            "Thesis (2021)": ["0.88", "0.95", "0.88", "—"],
        },
        index=["Accuracy", "ROC AUC", "Recall", "Precision"],
    )
    st.table(m)
    st.write(
        "Why lower than the thesis's 88%? Two reasons. The thesis applied SMOTE to the whole "
        "dataset before splitting, which leaks resampled data into the test set and inflates "
        "the score. And this rebuild has no victim or suspect information, only time, place "
        "and crime type. Predicting 'no arrest' for every incident alone scores "
        f"{metrics['baseline_accuracy_all_no_arrest'] * 100:.1f}% accuracy, which is why the "
        "ROC AUC and recall figures matter more than raw accuracy."
    )
    st.warning(
        "Please read the score for what it is: a research model of historical arrest "
        "outcomes, not a guarantee about your personal safety. Police data reflects what was "
        "reported and how it was enforced, which varies by neighborhood. Use it as one input "
        "alongside your own judgment, never as the only one."
    )
    st.caption(
        "Data: City of Cincinnati Open Data, Cincinnati Police Department Reported Crime "
        "(dataset t9kw-3jfx, successor to PDI Crime Incidents k59e-2pvf retired 2024-06-02). "
        "Thesis: OhioLINK ETD, accession ucin1627662505704827. Geocoding: OpenStreetMap Nominatim."
    )


if __name__ == "__main__":
    main()
