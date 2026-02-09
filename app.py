# ============================================================
# file: app.py
# ============================================================
from __future__ import annotations

import base64
import os
import re
from dataclasses import dataclass
from typing import Dict, Optional

import matplotlib as mpl
import pandas as pd
import streamlit as st

import opp_analysis_new as oa

# IMPORTANT: Must be the first Streamlit call.
st.set_page_config(page_title="Opponent Analysis - Set Pieces", layout="wide")

APP_BG = "#FFFFFF"


def check_password() -> bool:
    def password_entered() -> None:
        st.session_state["password_attempted"] = True
        pw = st.session_state.get("password", "")
        if pw == "4444":
            st.session_state["password_correct"] = True
            st.session_state.pop("password", None)  # optional: clear input
        else:
            st.session_state["password_correct"] = False

    # Already authenticated this session
    if st.session_state.get("password_correct", False):
        return True

    st.text_input(
        "Please enter the password to access the app",
        type="password",
        on_change=password_entered,
        key="password",
    )

    # Show error only after a real attempt
    if st.session_state.get("password_attempted", False) and not st.session_state.get("password_correct", False):
        st.error("😕 Password incorrect")

    return False

# --- 2) THEME / HEADER HELPERS ---
@dataclass(frozen=True)
class TeamTheme:
    top_hex: str
    rest_hex: str
    logo_relpath: str


def slugify(name: str) -> str:
    s = name.lower().strip()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_-]+", "_", s)
    return s


def build_team_themes() -> Dict[str, TeamTheme]:
    return {
        # ADO
        "ADO Den Haag": TeamTheme("#00802C", "#FFE200", "logos/ado_den_haag.png"),
        # Red/White clubs
        "Almere City FC": TeamTheme("#E3001B", "#FFFFFF", "logos/almere_city_fc.png"),
        "FC Dordrecht": TeamTheme("#D2232A", "#FFFFFF", "logos/fc_dordrecht.png"),
        "Jong Ajax": TeamTheme("#C31F3D", "#FFFFFF", "logos/jong_ajax.png"),
        "Jong AZ": TeamTheme("#DB0021", "#FFFFFF", "logos/jong_az.png"),
        "Jong FC Utrecht": TeamTheme("#ED1A2F", "#FFFFFF", "logos/jong_fc_utrecht.png"),
        "Jong PSV": TeamTheme("#E62528", "#FFFFFF", "logos/jong_psv.png"),
        "TOP Oss": TeamTheme("#D9031F", "#FFFFFF", "logos/top_oss.png"),
        "FC Emmen": TeamTheme("#E43B3B", "#FFFFFF", "logos/fc_emmen.png"),
        "MVV Maastricht": TeamTheme("#FA292F", "#FEFDFB", "logos/mvv_maastricht.png"),
        # Blue/White clubs
        "De Graafschap": TeamTheme("#0C8CCC", "#FFFFFF", "logos/de_graafschap.png"),
        "Eindhoven": TeamTheme("#0474BC", "#FFFFFF", "logos/eindhoven.png"),
        "FC Den Bosch": TeamTheme("#048CD4", "#FFFFFF", "logos/fc_den_bosch.png"),
        # Helmond
        "Helmond Sport": TeamTheme("#000000", "#E2001A", "logos/helmond_sport.png"),
        # RKC
        "RKC Waalwijk": TeamTheme("#2B63B7", "#FEE816", "logos/rkc_waalwijk.png"),
        # Yellow/Black clubs (black on top)
        "Roda JC Kerkrade": TeamTheme("#070E0C", "#FAC300", "logos/roda_jc_kerkrade.png"),
        "SC Cambuur": TeamTheme("#000000", "#FFD800", "logos/sc_cambuur.png"),
        "Vitesse": TeamTheme("#000000", "#FFD500", "logos/vitesse.png"),
        "VVV-Venlo": TeamTheme("#12100B", "#FEE000", "logos/vvv_venlo.png"),
        # Willem II
        "Willem II": TeamTheme("#242C84", "#FFFFFF", "logos/willem_ii.png"),
    }


def read_logo_as_base64(path: str) -> Optional[str]:
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def set_matplotlib_bg(bg_hex: str) -> None:
    """Keep matplotlib figures/axes background consistent with the app."""
    mpl.rcParams["figure.facecolor"] = bg_hex
    mpl.rcParams["savefig.facecolor"] = bg_hex
    mpl.rcParams["axes.facecolor"] = bg_hex


def style_fig_bg(fig, bg_hex: str):
    """Force an already-created fig/axes to match the app background color."""
    try:
        fig.patch.set_facecolor(bg_hex)
        for ax in fig.axes:
            ax.set_facecolor(bg_hex)
    except Exception:
        pass
    return fig


def apply_header(team: str, matches_analyzed: Optional[int], themes: Dict[str, TeamTheme]) -> TeamTheme:
    """
    App background stays white. Only the header banner is themed.
    Also renders a fixed footer bar using team top color.
    Places "Matches Analyzed: X | Team: Y" inside the bottom strip (black text).
    """
    theme = themes.get(team) or TeamTheme("#111827", "#FFFFFF", f"logos/{slugify(team)}.png")

    set_matplotlib_bg(APP_BG)

    logo_b64 = read_logo_as_base64(theme.logo_relpath)
    logo_html = (
        f'<img class="team-logo" src="data:image/png;base64,{logo_b64}" alt="{team} logo" />'
        if logo_b64
        else ""
    )

    title_text_color = "#FFFFFF" if theme.top_hex.upper() != "#FFFFFF" else "#111827"
    subtitle_color = "rgba(255,255,255,0.90)" if title_text_color == "#FFFFFF" else "rgba(17,24,39,0.85)"

    matches_val = matches_analyzed if matches_analyzed is not None else "-"
    meta_line = f"Matches Analyzed: {matches_val} | Team: {team}"

    st.markdown(
        f"""
        <style>
          /* App background ALWAYS white */
          [data-testid="stAppViewContainer"] {{
            background: {APP_BG};
          }}
          [data-testid="stHeader"] {{
            background: transparent;
          }}
          header {{
            background: transparent !important;
          }}
          [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"] {{
            background: transparent;
          }}

          /* Prevent content from being hidden behind the fixed footer bar */
          div.block-container {{
            padding-bottom: 28px;
          }}

          .team-banner {{
            position: relative;
            width: 100%;
            border-radius: 18px;
            overflow: hidden;
            margin: 0.15rem 0 1.25rem 0;
            box-shadow: 0 10px 28px rgba(0,0,0,0.14);
          }}
          .team-banner-top {{
            background: {theme.top_hex};
            padding: 2.4rem 1.4rem 2.1rem 1.4rem;
          }}

          /* Bottom strip with meta at the lower part */
          .team-banner-bottom {{
            background: {theme.rest_hex};
            padding: 0.6rem 1.4rem 0.35rem 1.4rem;
            min-height: 54px;
            display: flex;
            align-items: flex-end;
          }}

          .team-title {{
            margin: 0;
            color: {title_text_color};
            font-size: 1.75rem;
            font-weight: 900;
            letter-spacing: 0.2px;
          }}
          .team-subtitle {{
            margin: 0.35rem 0 0 0;
            color: {subtitle_color};
            font-size: 1.05rem;
            font-weight: 700;
          }}

          /* Meta line: black */
          .team-meta {{
            margin: 0;
            color: #000000;
            font-size: 0.98rem;
            font-weight: 800;
          }}

          .team-logo {{
            position: absolute;
            top: 18px;
            right: 18px;
            height: 78px;
            width: 78px;
            object-fit: contain;
            background: rgba(255,255,255,0.92);
            border-radius: 16px;
            padding: 9px;
          }}

          /* Fixed footer bar in main color */
          .team-footer-bar {{
            position: fixed;
            left: 0;
            right: 0;
            bottom: 0;
            height: 14px;
            background: {theme.top_hex};
            z-index: 9999;
          }}
        </style>

        <div class="team-banner">
          <div class="team-banner-top">
            <p class="team-title">Opponent analysis - Set Pieces</p>
            <p class="team-subtitle">{team}</p>
            {logo_html}
          </div>
          <div class="team-banner-bottom">
            <p class="team-meta">{meta_line}</p>
          </div>
        </div>

        <div class="team-footer-bar"></div>
        """,
        unsafe_allow_html=True,
    )

    return theme


# --- 3) MAIN APP LOGIC ---
if check_password():
    CORNER_EVENTS_CSV = "data/corner_events_all_matches.csv"
    EVENTS_SEQ_CSV = "data/corner_events_full_sequences.csv"
    POS_SAMPLES_CSV = "data/corner_positions_samples_from_start_to_end.csv"

    IMG_PATHS = {
        "def_L": "images/no_names_left.png",
        "def_R": "images/no_names_right.png",
        "att_L": "images/left_side_corner.png",
        "att_R": "images/right_side_corner.png",
    }

    def get_img_path(key: str) -> Optional[str]:
        path = IMG_PATHS.get(key)
        return path if path and os.path.exists(path) else None

    # --- CACHED LOADING ---
    @st.cache_data
    def load_corner_jsonlike(csv_path: str):
        return oa.load_corner_events_csv_as_jsonlike(csv_path)

    @st.cache_data
    def load_events_sequences(csv_path: str) -> pd.DataFrame:
        return pd.read_csv(csv_path, low_memory=False).where(pd.notnull, None)

    @st.cache_data
    def load_positions_index(csv_path: str):
        return oa.load_positions_samples_for_tables(csv_path)

    @st.cache_data
    def get_team_list(json_data):
        return oa.extract_all_teams(json_data)

    @st.cache_data
    def get_analysis_results(json_data, team_name: str):
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

        themes = build_team_themes()
        theme = apply_header(selected_team, results.get("used_matches"), themes)

        # --- ROW 1: ATTACKING PLOTS ---
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"Att. Corners Left ({results['own_left_count']} corners)")
            fig = oa.plot_percent_attacking(
                get_img_path("att_L"),
                viz_config["att_L"],
                viz_config["att_centers_L"],
                results["attacking"]["left_pct"],
            )
            st.pyplot(style_fig_bg(fig, APP_BG))
        with col2:
            st.subheader(f"Att. Corners Right ({results['own_right_count']} corners)")
            fig = oa.plot_percent_attacking(
                get_img_path("att_R"),
                viz_config["att_R"],
                viz_config["att_centers_R"],
                results["attacking"]["right_pct"],
            )
            st.pyplot(style_fig_bg(fig, APP_BG))

        # --- ROW 1B: ATTACKING -> SHOT PER ZONE (WITH KKD PERCENTILES) ---
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("How many attacking corners led to a shot? (Left Side)")
            tot_z, shot_z, pct_z = results["attacking_shots"]["left"]
            pctiles = oa.build_percentiles_for_team(league_stats, selected_team, "left", min_zone_corners=4)
            fig = oa.plot_shots_attacking_with_percentile(
                get_img_path("def_L"),
                viz_config["def_L"],
                pct_z,
                tot_z,
                shot_z,
                pctiles,
                min_zone_corners=4,
                font_size=16,
            )
            st.pyplot(style_fig_bg(fig, APP_BG))

        with col2:
            st.subheader("How many attacking corners led to a shot? (Right Side)")
            tot_z, shot_z, pct_z = results["attacking_shots"]["right"]
            pctiles = oa.build_percentiles_for_team(league_stats, selected_team, "right", min_zone_corners=4)
            fig = oa.plot_shots_attacking_with_percentile(
                get_img_path("def_R"),
                viz_config["def_R"],
                pct_z,
                tot_z,
                shot_z,
                pctiles,
                min_zone_corners=4,
                font_size=16,
            )
            st.pyplot(style_fig_bg(fig, APP_BG))

        # --- ROW 2: CORNER TAKER TABLES ---
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### 📋 Corner Takers (Left)")
            st.dataframe(results["tables"]["left"], width="stretch", hide_index=True)
        with col2:
            st.markdown("##### 📋 Corner Takers (Right)")
            st.dataframe(results["tables"]["right"], width="stretch", hide_index=True)

        # --- ROW 3: DEFENDING PLOTS ---
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Def. Corners Left: how many crosses turned into a shot?")
            tot, ids, pcts = results["defensive"]["left"]
            fig = oa.plot_shots_defensive(
                get_img_path("def_L"),
                viz_config["def_L"],
                pcts,
                tot,
                ids,
            )
            st.pyplot(style_fig_bg(fig, APP_BG))

        with col2:
            st.subheader("Def. Corners Right: how many crosses turned into a shot?")
            tot, ids, pcts = results["defensive"]["right"]
            fig = oa.plot_shots_defensive(
                get_img_path("def_R"),
                viz_config["def_R"],
                pcts,
                tot,
                ids,
            )
            st.pyplot(style_fig_bg(fig, APP_BG))

        # --- ROW 4: PLAYER CHARTS ---
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
                st.markdown("##### 🟦 Attacking corner players (chart)")
                fig_att = oa.plot_attacking_corner_players_headers(att_tbl, max_players=15)
                st.pyplot(style_fig_bg(fig_att, APP_BG), clear_figure=True)

            with col2:
                st.markdown("##### 🟥 Defending corner players (chart)")
                fig_def = oa.plot_defending_corner_players_diverging(def_tbl, max_players=15)
                st.pyplot(style_fig_bg(fig_def, APP_BG), clear_figure=True)
