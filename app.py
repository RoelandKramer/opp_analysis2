# ============================================================
# file: app.py
# ============================================================
import streamlit as st
import json
import os
from datetime import datetime
import opponent_analysis as oa

import streamlit as st

# --- 1. PASSWORD CHECK FUNCTION ---
def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == "4444":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    st.text_input(
        "Please enter the password to access the app",
        type="password",
        on_change=password_entered,
        key="password",
    )

    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")

    return False

@st.cache_data
def get_latest_match_info(json_data):
    """
    Returns (latest_date: datetime|None, latest_match_name: str|None).
    Uses match_name prefix 'dd-mm-yyyy ...' when possible.
    """
    latest_dt = None
    latest_name = None

    for m in json_data.get("matches", []):
        name = m.get("match_name") or ""
        if len(name) >= 10:
            prefix = name[:10]
            try:
                dt = datetime.strptime(prefix, "%d-%m-%Y")
                if latest_dt is None or dt > latest_dt:
                    latest_dt = dt
                    latest_name = name
            except Exception:
                continue

    return latest_dt, latest_name

@st.cache_data
def get_league_attacking_corner_shot_rates(json_data):
    return oa.compute_league_attacking_corner_shot_rates(json_data)

def _percentile_rank(values, value):
    if not values:
        return None
    # Higher is better. Percentile = % of teams <= team_value.
    n = len(values)
    leq = sum(1 for v in values if v <= value)
    return int(round((leq / n) * 100))

def build_percentiles_for_team(league_stats, team_name, side, min_zone_corners: int = 4):
    team_side = league_stats.get(team_name, {}).get(side, {})
    percentiles = {}

    for zone, d in team_side.items():
        team_total = int(d.get("total", 0))
        if team_total < min_zone_corners:
            # Do not show percentile if this team has too few corners in this zone
            continue

        team_pct = float(d.get("pct", 0.0))

        dist = []
        for _, side_map in league_stats.items():
            zd = side_map.get(side, {}).get(zone)
            if not zd:
                continue
            if int(zd.get("total", 0)) < min_zone_corners:
                continue
            dist.append(float(zd.get("pct", 0.0)))

        percentiles[zone] = _percentile_rank(dist, team_pct)

    return percentiles


# --- 2. MAIN APP LOGIC ---
if check_password():
    # --- CONFIGURATION ---
    st.set_page_config(page_title="Opponent Analysis - Set Pieces", layout="wide")

    DATA_PATH = "data/corner_events_all_matches.json"

    IMG_PATHS = {
        "def_L": "images/no_names_left.png",
        "def_R": "images/no_names_right.png",
        "att_L": "images/left_side_corner.png",
        "att_R": "images/right_side_corner.png",
    }

    # --- CACHED LOADING ---
    @st.cache_data
    def load_local_data(path):
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @st.cache_data
    def get_team_list(json_data):
        return oa.extract_all_teams(json_data)

    @st.cache_data
    def get_analysis_results(json_data, team_name):
        return oa.process_corner_data(json_data, team_name)

    def get_img_path(key):
        path = IMG_PATHS.get(key)
        return path if path and os.path.exists(path) else None

    # --- SIDEBAR ---
    st.sidebar.header("Configuration")
    json_data = load_local_data(DATA_PATH)

    if not json_data:
        st.error(f"❌ Data file not found at: `{DATA_PATH}`")
        st.stop()

    all_teams = get_team_list(json_data)
    selected_team = st.sidebar.selectbox("Select team", all_teams)

    # --- sidebar latest match (bottom) ---
    latest_dt, latest_name = get_latest_match_info(json_data)
    st.sidebar.markdown("---")
    if latest_dt and latest_name:
        st.sidebar.caption(
            f"Latest match in dataset: {latest_dt.strftime('%d-%m-%Y')} — {latest_name}"
        )
    else:
        st.sidebar.caption("Latest match in dataset: -")

    # --- MAIN APP ---
    if json_data and selected_team:
        with st.spinner(f"Analyzing {selected_team}..."):
            results = get_analysis_results(json_data, selected_team)
            viz_config = oa.get_visualization_coords()
            league_stats = get_league_attacking_corner_shot_rates(json_data)

        st.title("Opponent analysis - Set Pieces")
        st.markdown(f"**Matches Analyzed:** {results['used_matches']} | **Team:** {selected_team}")

        # --- ROW 1: ATTACKING PLOTS ---
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"Att. Corners Left ({results['own_left_count']} corners)")
            st.pyplot(
                oa.plot_percent_attacking(
                    get_img_path("att_L"),
                    viz_config["att_L"],
                    viz_config["att_centers_L"],
                    results["attacking"]["left_pct"],
                    "",
                )
            )
        with col2:
            st.subheader(f"Att. Corners Right ({results['own_right_count']} corners)")
            st.pyplot(
                oa.plot_percent_attacking(
                    get_img_path("att_R"),
                    viz_config["att_R"],
                    viz_config["att_centers_R"],
                    results["attacking"]["right_pct"],
                    "",
                )
            )

        # --- NEW ROW: ATTACKING -> SHOT PER ZONE (WITH KKD PERCENTILES) ---
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("How many attacking corners let to a shot? (Left Side)")
            tot_z, shot_z, pct_z = results["attacking_shots"]["left"]
            pctiles = build_percentiles_for_team(league_stats, selected_team, "left")
            st.pyplot(
                oa.plot_shots_attacking_with_percentile(
                    get_img_path("att_L"),
                    viz_config["att_L"]
                    viz_config["att_shot_centers_L"]
                    pct_z,
                    tot_z,
                    shot_z,
                    pctiles,
                )
            )

        with col2:
            st.subheader("How many attacking corners let to a shot? (Right Side)")
            tot_z, shot_z, pct_z = results["attacking_shots"]["right"]
            pctiles = build_percentiles_for_team(league_stats, selected_team, "right")
            st.pyplot(
                oa.plot_shots_attacking_with_percentile(
                    get_img_path("att_R"),
                    viz_config["att_R"],
                    viz_config["att_shot_centers_R"],
                    pct_z,
                    tot_z,
                    shot_z,
                    pctiles,
                )
            )

        # --- ROW 2: TABLES ---
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### 📋 Corner Takers (Left)")
            st.dataframe(results["tables"]["left"], use_container_width=True, hide_index=True)
        with col2:
            st.markdown("##### 📋 Corner Takers (Right)")
            st.dataframe(results["tables"]["right"], use_container_width=True, hide_index=True)

        # --- ROW 3: DEFENDING PLOTS ---
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Def. Corners Left: how many crosses turned into a shot?")
            tot, ids, pcts = results["defensive"]["left"]
            st.pyplot(
                oa.plot_shots_defensive(
                    get_img_path("def_L"),
                    viz_config["def_L"],
                    pcts,
                    tot,
                    ids,
                    "",
                )
            )

        with col2:
            st.subheader("Def. Corners Right: how many crosses turned into a shot?")
            tot, ids, pcts = results["defensive"]["right"]
            st.pyplot(
                oa.plot_shots_defensive(
                    get_img_path("def_R"),
                    viz_config["def_R"],
                    pcts,
                    tot,
                    ids,
                    "",
                )
            )
