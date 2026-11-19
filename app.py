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

STRUCTURE — original vs added, kept visibly separate at the author's
request so the thesis can be evaluated against the original work:
  - The core of this page (the score form, the score itself, the thesis
    benchmark, the method and metrics section) is the thesis app.
  - Everything else lives in one fenced-off section, "Beyond the thesis:
    improvements added in 2026", and each item there is labelled as an
    improvement. Nothing in that section was part of the 2021 thesis.

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
import urllib.parse
from datetime import datetime, time as dtime
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
SHARE_BASE = "https://cincinnati-safety-score.streamlit.app/"


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
    for name in ("metrics.json", "neighborhoods.json", "encodings.json", "grid_recent.json"):
        h.update((MODEL_DIR / name).read_bytes())
    return h.hexdigest()


@st.cache_resource
def load_artifacts(version: str):
    model = joblib.load(MODEL_DIR / "safety_model.joblib")
    metrics = json.loads((MODEL_DIR / "metrics.json").read_text())
    encodings = json.loads((MODEL_DIR / "encodings.json").read_text())
    neighborhoods = json.loads((MODEL_DIR / "neighborhoods.json").read_text())
    grid = json.loads((MODEL_DIR / "grid_recent.json").read_text())
    return model, metrics, encodings, neighborhoods, grid


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


def compute_score(model, cat_map, lat, lon, dt, crime_key):
    """The thesis model itself: safety score = 1 - P(cleared by arrest)."""
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
    return 1.0 - p_arrest, p_arrest, sample


def nearby_count(grid: dict, lat: float, lon: float, radius_km: float = 0.805) -> int:
    """Recorded property crimes within about half a mile, from grid counts.

    Uses cell centres, so it is an approximation, and the app says so.
    """
    cell = grid["cell_deg"]
    total = 0
    for key, n in grid["counts"].items():
        ilat_s, ilon_s = key.split(",")
        clat = (int(ilat_s) + 0.5) * cell
        clon = (int(ilon_s) + 0.5) * cell
        dlat_km = (clat - lat) * 110.99
        dlon_km = (clon - lon) * 86.0  # at Cincinnati's latitude
        if math.hypot(dlat_km, dlon_km) <= radius_km:
            total += n
    return total


def score_banner(safety: float):
    if safety > BENCHMARK:
        st.success(f"Your safety score is {safety * 100:.1f}%")
    else:
        st.warning(f"Your safety score is {safety * 100:.1f}%")


def main():
    st.set_page_config(page_title="Cincinnati Neighborhood Safety Score", page_icon="🛡️", layout="centered")
    model, metrics, encodings, neighborhoods, grid = load_artifacts(_data_version())
    cat_map = encodings["ucr_group"]

    # Shared-result links (a 2026 improvement) carry lat/lon/date/time/crime
    # in the URL. Apply them once per session, before the widgets are drawn.
    qp = st.query_params
    if "lat" in qp and "shared_applied" not in st.session_state:
        try:
            st.session_state["shared_point"] = (float(qp["lat"]), float(qp["lon"]))
            if "d" in qp:
                st.session_state["date_in"] = datetime.fromisoformat(qp["d"]).date()
            if "t" in qp:
                hh, mm = qp["t"].split(":")[:2]
                st.session_state["time_in"] = dtime(int(hh), int(mm))
            if "c" in qp and qp["c"] in CRIME_LABELS:
                st.session_state["crime_in"] = CRIME_LABELS[qp["c"]]
            st.session_state["auto_predict"] = True
        except Exception:
            pass
        st.session_state["shared_applied"] = True

    st.title("Cincinnati Neighborhood Safety Score")
    st.caption(
        "Rebuilt from the 2021 University of Cincinnati master's thesis by "
        "Gifty Ama Dua Arthur: *A Machine Learning Web Application for "
        "Predicting Neighborhood Safety in The City of Cincinnati.*"
    )
    st.caption(
        "How to read this page: the score checker below is the thesis app. "
        "Everything added since is kept in one clearly marked section, "
        "\"Beyond the thesis\", so the original work can be evaluated on its own."
    )

    st.subheader("What this app is trying to say")
    st.write(
        "You do not have to guess whether a place feels safe. The city's own crime "
        "history can give you a number for a specific place, at a specific time, "
        "right on your phone. Most crime research studies offenders, who might "
        "offend again. This work flipped the question to victims, and asked what "
        "their patterns in the data could teach the rest of us about avoiding "
        "becoming one. Safety is not a fixed property of a neighborhood. It "
        "changes with the hour and the day, and it can be estimated, imperfectly, "
        "from patterns instead of impressions."
    )
    st.caption("Plain-language summary of the thesis, written for the 2026 rebuild.")

    # ------------------------------------------------------------------
    # THE THESIS APP (original work)
    # ------------------------------------------------------------------
    st.subheader("Check a location")
    mode = st.radio("Location", ["Street address", "Neighborhood"], horizontal=True)

    def _try_example():
        st.session_state["address"] = "2600 Vine St"
        st.session_state["auto_predict"] = True

    st.button("Try an example: 2600 Vine St, near the University of Cincinnati",
              on_click=_try_example)
    st.caption("The example button is a 2026 convenience. The original app asked you to type the address.")

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
        elif "shared_point" in st.session_state:
            lat, lon = st.session_state["shared_point"]
            st.info(f"Location loaded from a shared link: {lat:.5f}, {lon:.5f}. "
                    "Type an address above to change it.")
    else:
        names = [n["name"] for n in neighborhoods]
        chosen_name = st.selectbox("Neighborhood (SNA)", names)
        n = next(x for x in neighborhoods if x["name"] == chosen_name)
        lat, lon = n["lat"], n["lon"]
        st.caption(f"Uses the neighborhood center: {lat:.5f}, {lon:.5f}")

    col1, col2 = st.columns(2)
    with col1:
        date = st.date_input("Date", value=datetime.now().date(), key="date_in")
    with col2:
        time = st.time_input("Time", value=datetime.now().time().replace(second=0, microsecond=0), key="time_in")

    crime_label = st.selectbox(
        "Property crime type the score is modelled on",
        [CRIME_LABELS[k] for k in sorted(CRIME_LABELS)],
        key="crime_in",
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
        else:
            dt = datetime.combine(date, time)
            safety, p_arrest, sample = compute_score(model, cat_map, lat, lon, dt, crime_key)
            st.session_state["result"] = {
                "lat": lat, "lon": lon, "dt": dt.isoformat(),
                "crime_key": crime_key, "crime_label": crime_label,
                "safety": safety, "p_arrest": p_arrest,
            }
            st.session_state["last_sample"] = sample

    res = st.session_state.get("result")
    if res:
        score_banner(res["safety"])
        st.write(
            f"That is 1 minus the model's estimated likelihood ({res['p_arrest'] * 100:.1f}%) that a "
            f"reported {res['crime_label'].lower()} at this place and time ended in an arrest, the "
            f"definition used in the thesis. The thesis benchmark is {BENCHMARK * 100:.1f}%: "
            "above it the location was classed as very safe, below it the app advised caution."
        )
        st.map(pd.DataFrame({"lat": [res["lat"]], "lon": [res["lon"]]}), zoom=13)
        with st.expander("What the model weighed most (feature importance)"):
            imp = pd.Series(metrics["feature_importance"]).sort_values(ascending=True)
            st.bar_chart(imp)
        with st.expander("Encoded input sent to the model"):
            st.dataframe(st.session_state["last_sample"])

    # ------------------------------------------------------------------
    # BEYOND THE THESIS: IMPROVEMENTS ADDED IN 2026 (not original work)
    # ------------------------------------------------------------------
    st.divider()
    st.header("Beyond the thesis: improvements added in 2026")
    st.info(
        "For anyone evaluating the thesis: nothing in this section was part of "
        "the original 2021 thesis or its app. These are additions made during "
        "the November 2026 rebuild. The thesis app above, its score, and its "
        "definition are unchanged. Each addition is labelled so the boundary "
        "stays visible."
    )

    near = None
    if res:
        near = nearest_neighborhood(res["lat"], res["lon"], neighborhoods)

    st.subheader("Improvement: your result in context")
    if not res:
        st.caption("Get a safety score above first, and this section will put it in context.")
    else:
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
        hourly = pd.Series(near["hourly_counts"], index=[f"{h:02d}:00" for h in range(24)])
        peak_hour = int(pd.Series(near["hourly_counts"]).idxmax())
        st.write(f"**When property crime is recorded in {near['name']}** (by hour of day, "
                 f"peaking around {peak_hour:02d}:00):")
        st.bar_chart(hourly)

        st.write(f"**Is {near['name']} getting better or worse?** Recorded property crimes by year:")
        yearly = pd.Series({y: c for y, c in near["yearly_counts"].items()})
        st.line_chart(yearly)
        city = metrics["citywide_yearly_counts"]
        years_sorted = sorted(city)
        first_y, last_full_y = years_sorted[0], "2025"
        direction = "fell" if city[last_full_y] < city[first_y] else "rose"
        st.caption(
            f"2026 is a partial year (data through {metrics['date_max']}). Citywide, recorded "
            f"property crimes {direction} from {city[first_y]:,} in {first_y} to "
            f"{city[last_full_y]:,} in {last_full_y}, the last full year. A neighborhood trend "
            "is context, not a verdict: recording and reporting practices change too."
        )

        n_near = nearby_count(grid, res["lat"], res["lon"])
        st.metric(f"Property crimes recorded within about half a mile of this point in {grid['year']}",
                  f"{n_near:,}")
        st.caption(
            "A plain count from the city's records, no model involved. It is approximate: "
            "incidents are grouped into small map cells before counting, and only crimes "
            "that were reported and recorded appear at all."
        )

    st.subheader("Improvement: share this result")
    if not res:
        st.caption("Once you have a score, a link will appear here that reopens the app with the same place, date, time and crime type filled in.")
    else:
        dt_res = datetime.fromisoformat(res["dt"])
        params = urllib.parse.urlencode({
            "lat": f"{res['lat']:.5f}", "lon": f"{res['lon']:.5f}",
            "d": dt_res.date().isoformat(), "t": dt_res.strftime("%H:%M"),
            "c": res["crime_key"],
        })
        st.code(f"{SHARE_BASE}?{params}", language=None)
        st.caption("Anyone opening this link sees the same score. The link carries the location and time only, nothing about you.")

    st.subheader("Improvement: tap the map to score a point")
    st.caption(
        "The original app took a typed address. Here you can tap any point in "
        "Cincinnati instead. It uses the same thesis model, scored at the "
        "current date and time."
    )
    try:
        import folium
        from streamlit_folium import st_folium

        fmap = folium.Map(location=[39.1031, -84.5120], zoom_start=12)
        map_out = st_folium(fmap, width=700, height=420, returned_objects=["last_clicked"])
        clicked = (map_out or {}).get("last_clicked")
        if clicked:
            st.session_state["map_point"] = (float(clicked["lat"]), float(clicked["lng"]))
    except Exception:
        st.caption("The interactive map could not be loaded here. The address and neighborhood options above still work.")
    if "map_point" in st.session_state:
        mlat, mlon = st.session_state["map_point"]
        st.write(f"Picked point: {mlat:.5f}, {mlon:.5f} (nearest neighborhood: "
                 f"{nearest_neighborhood(mlat, mlon, neighborhoods)['name']})")
        map_crime_label = st.selectbox(
            "Crime type for the map point",
            [CRIME_LABELS[k] for k in sorted(CRIME_LABELS)],
            key="map_crime",
        )
        map_crime_key = next(k for k, v in CRIME_LABELS.items() if v == map_crime_label)
        if st.button("Score this point"):
            now = datetime.now().replace(second=0, microsecond=0)
            m_safety, m_p, _ = compute_score(model, cat_map, mlat, mlon, now, map_crime_key)
            score_banner(m_safety)
            st.caption(
                f"Estimated likelihood this {map_crime_label.lower()} at the picked point, "
                f"scored at {now.strftime('%Y-%m-%d %H:%M')}, ended in an arrest: {m_p * 100:.1f}%."
            )

    st.subheader("Improvement: compare two neighborhoods")
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
    st.write("**Recorded property crimes by year:**")
    st.line_chart(pd.DataFrame(
        {pick_a: a["yearly_counts"], pick_b: b["yearly_counts"]}
    ))
    st.caption("2026 is a partial year in both charts (data through the date range shown below).")

    # ------------------------------------------------------------------
    # METHOD AND HONESTY (thesis core, unchanged)
    # ------------------------------------------------------------------
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

    st.write("**What is original, and what was added**")
    oc1, oc2 = st.columns(2)
    with oc1:
        st.write(
            "*From the original thesis (2021)*\n"
            "- The safety score itself: 1 minus the modelled likelihood of arrest\n"
            "- Random Forest with SMOTE oversampling\n"
            "- Address, date, time and crime type as inputs\n"
            "- The 74.6% benchmark for \"very safe\"\n"
            "- Feature importance as the explanation view"
        )
    with oc2:
        st.write(
            "*Added or changed in the 2026 rebuild*\n"
            "- A newer dataset (2020-2026), because the thesis dataset was retired\n"
            "- No victim demographic inputs, because the new feed does not publish them\n"
            "- Honest held-out metrics, with SMOTE kept out of the test data\n"
            "- Everything in the \"Beyond the thesis\" section above\n"
            "- A monthly automatic retrain on the latest city data"
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
