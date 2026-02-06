# ============================================================
# file: app.py
# ============================================================
import os
from datetime import datetime

import pandas as pd
import streamlit as st

import opp_analysis_new as oa


# --- 1. PASSWORD CHECK FUNCTION ---
def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
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


# --- 2. MAIN APP LOGIC ---
if check_password():
    st.set_page_config(page_title="Opponent Analysis - Set Pieces", layout="wide")

    # --- CSVs (repo paths) ---
    CORNER_EVENTS_CSV = "data/corner_events_all_matches.csv"
    EVENTS_SEQ_CSV = "data/corner_events_full_sequences.csv"
    POS_SAMPLES_CSV = "data/corner_positions_samples_from_start_to_end.csv"

    IMG_PATHS = {
        "def_L": "images/no_names_left.png",
        "def_R": "images/no_names_right.png",
        "att_L": "images/left_side_corner.png",
        "att_R": "images/right_side_corner.png",
    }

    def get_img_path(key: str):
        path = IMG_PATHS.get(key)
        return path if path and os.path.exists(path) else None

    # --- CACHED LOADING ---
    @st.cache_data
    def load_corner_jsonlike(csv_path: str):
        return oa.load_corner_events_csv_as_jsonlike(csv_path)

    @st.cache_data
    def load_events_sequences(csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path, low_memory=False).where(pd.notnull, None)
        return df

    @st.cache_data
    def load_positions_index(csv_path: str):
        # returns (pos_df_norm, pos_times, pos_frames)
        return oa.load_positions_samples_for_tables(csv_path)

    @st.cache_data
    def get_team_list(json_data):
        return oa.extract_all_teams(json_data)

    @st.cache_data
    def get_analysis_results(json_data, team_name):
        return oa.process_corner_data(json_data, team_name)

    @st.cache_data
    def get_league_stats(json_data):
        return oa.compute_league_attacking_corner_shot_rates(json_data)

    @st.cache_data
    def build_player_tables(team: str, events_df: pd.DataFrame, pos_times, pos_frames):
        att_tbl = oa.build_attacking_corners_player_table(
            team=team,
            events_df=events_df,
            pos_times=pos_times,
            pos_frames=pos_frames,
        )
        def_tbl = oa.build_defending_corners_player_table(
            team=team,
            events_df=events_df,
            pos_times=pos_times,
            pos_frames=pos_frames,
        )
        return att_tbl, def_tbl

    # --- SIDEBAR ---
    st.sidebar.header("Configuration")

    if not os.path.exists(CORNER_EVENTS_CSV):
        st.error(f"❌ Data file not found at: `{CORNER_EVENTS_CSV}`")
        st.stop()

    json_data = load_corner_jsonlike(CORNER_EVENTS_CSV)
    all_teams = get_team_list(json_data)

    if not all_teams:
        st.error("❌ No teams found in dataset.")
        st.stop()

    selected_team = st.sidebar.selectbox("Select team", all_teams)

    latest_dt, latest_name = oa.get_latest_match_info(json_data)
    st.sidebar.markdown("---")
    if latest_dt and latest_name:
        st.sidebar.caption(f"Latest match in dataset: {latest_dt.strftime('%d-%m-%Y')} — {latest_name}")
    else:
        st.sidebar.caption("Latest match in dataset: -")

    # --- MAIN APP ---
    if json_data and selected_team:
        with st.spinner(f"Analyzing {selected_team}..."):
            results = get_analysis_results(json_data, selected_team)
            viz_config = oa.get_visualization_coords()
            league_stats = get_league_stats(json_data)

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
                )
            )

        # --- ROW 1B: ATTACKING -> SHOT PER ZONE (WITH KKD PERCENTILES) ---
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("How many attacking corners led to a shot? (Left Side)")
            tot_z, shot_z, pct_z = results["attacking_shots"]["left"]
            pctiles = oa.build_percentiles_for_team(league_stats, selected_team, "left", min_zone_corners=4)
            st.pyplot(
                oa.plot_shots_attacking_with_percentile(
                    get_img_path("def_L"),
                    viz_config["def_L"],
                    pct_z,
                    tot_z,
                    shot_z,
                    pctiles,
                    min_zone_corners=4,
                    font_size=16,
                )
            )

        with col2:
            st.subheader("How many attacking corners led to a shot? (Right Side)")
            tot_z, shot_z, pct_z = results["attacking_shots"]["right"]
            pctiles = oa.build_percentiles_for_team(league_stats, selected_team, "right", min_zone_corners=4)
            st.pyplot(
                oa.plot_shots_attacking_with_percentile(
                    get_img_path("def_R"),
                    viz_config["def_R"],
                    pct_z,
                    tot_z,
                    shot_z,
                    pctiles,
                    min_zone_corners=4,
                    font_size=16,
                )
            )

        # --- ROW 2: CORNER TAKER TABLES ---
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
                )
            )

        # --- ROW 4: ATTACKING + DEFENDING PLAYER TABLES (BOTTOM, SIDE-BY-SIDE) ---
        st.divider()
        st.markdown("### Corner Player Tables: Who is dangerous, and who is weak?")

        missing = [p for p in (EVENTS_SEQ_CSV, POS_SAMPLES_CSV) if not os.path.exists(p)]
        if missing:
            st.warning(
                "Player tables skipped because these files are missing:\n"
                + "\n".join([f"- `{p}`" for p in missing])
            )
        else:
            with st.spinner("Building player tables..."):
                events_all_df = load_events_sequences(EVENTS_SEQ_CSV)
                _, pos_times, pos_frames = load_positions_index(POS_SAMPLES_CSV)
                att_tbl, def_tbl = build_player_tables(selected_team, events_all_df, pos_times, pos_frames)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### 🟦 Attacking corner players")
                st.dataframe(att_tbl, use_container_width=True, hide_index=True)
            with col2:
                st.markdown("##### 🟥 Defending corner players")
                st.dataframe(def_tbl, use_container_width=True, hide_index=True)
