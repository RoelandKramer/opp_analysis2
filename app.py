# ============================================================
# file: app.py
# ============================================================
from __future__ import annotations

import base64
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import matplotlib as mpl
import pandas as pd
import streamlit as st

import opp_analysis_new as oa
import update_database as upd

# --- Secrets -> env for git push (safe) ---
if "GITHUB_TOKEN" in st.secrets:
    os.environ["GITHUB_TOKEN"] = st.secrets["GITHUB_TOKEN"]
if "GITHUB_REPO" in st.secrets:
    os.environ["GITHUB_REPO"] = st.secrets["GITHUB_REPO"]
os.environ["GITHUB_BRANCH"] = st.secrets.get("GITHUB_BRANCH", "main")

st.set_page_config(page_title="Opponent Analysis - Set Pieces", layout="wide")

APP_BG = "#FFFFFF"


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
        "ADO Den Haag": TeamTheme("#00802C", "#FFE200", "logos/ado_den_haag.png"),
        "Almere City FC": TeamTheme("#E3001B", "#FFFFFF", "logos/almere_city_fc.png"),
        "FC Dordrecht": TeamTheme("#D2232A", "#FFFFFF", "logos/fc_dordrecht.png"),
        "Jong Ajax": TeamTheme("#C31F3D", "#FFFFFF", "logos/jong_ajax.png"),
        "Jong AZ": TeamTheme("#DB0021", "#FFFFFF", "logos/jong_az.png"),
        "Jong FC Utrecht": TeamTheme("#ED1A2F", "#FFFFFF", "logos/jong_fc_utrecht.png"),
        "Jong PSV": TeamTheme("#E62528", "#FFFFFF", "logos/jong_psv.png"),
        "TOP Oss": TeamTheme("#D9031F", "#FFFFFF", "logos/top_oss.png"),
        "FC Emmen": TeamTheme("#E43B3B", "#FFFFFF", "logos/fc_emmen.png"),
        "MVV Maastricht": TeamTheme("#FA292F", "#FEFDFB", "logos/mvv_maastricht.png"),
        "De Graafschap": TeamTheme("#0C8CCC", "#FFFFFF", "logos/de_graafschap.png"),
        "Eindhoven": TeamTheme("#0474BC", "#FFFFFF", "logos/eindhoven.png"),
        "FC Den Bosch": TeamTheme("#048CD4", "#FFFFFF", "logos/fc_den_bosch.png"),
        "Helmond Sport": TeamTheme("#000000", "#E2001A", "logos/helmond_sport.png"),
        "RKC Waalwijk": TeamTheme("#2B63B7", "#FEE816", "logos/rkc_waalwijk.png"),
        "Roda JC Kerkrade": TeamTheme("#070E0C", "#FAC300", "logos/roda_jc_kerkrade.png"),
        "SC Cambuur": TeamTheme("#000000", "#FFD800", "logos/sc_cambuur.png"),
        "Vitesse": TeamTheme("#000000", "#FFD500", "logos/vitesse.png"),
        "VVV-Venlo": TeamTheme("#12100B", "#FEE000", "logos/vvv_venlo.png"),
        "Willem II": TeamTheme("#242C84", "#FFFFFF", "logos/willem_ii.png"),
    }


def read_logo_as_base64(path: str) -> Optional[str]:
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def set_matplotlib_bg(bg_hex: str) -> None:
    mpl.rcParams["figure.facecolor"] = bg_hex
    mpl.rcParams["savefig.facecolor"] = bg_hex
    mpl.rcParams["axes.facecolor"] = bg_hex


def style_fig_bg(fig, bg_hex: str):
    try:
        fig.patch.set_facecolor(bg_hex)
        for ax in fig.axes:
            ax.set_facecolor(bg_hex)
    except Exception:
        pass
    return fig


def apply_header(team: str, matches_analyzed: Optional[int], themes: Dict[str, TeamTheme]) -> TeamTheme:
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
          [data-testid="stAppViewContainer"] {{ background: {APP_BG}; }}
          [data-testid="stHeader"] {{ background: transparent; }}
          header {{ background: transparent !important; }}
          [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"] {{ background: transparent; }}
          div.block-container {{ padding-bottom: 28px; }}

          .team-banner {{
            position: relative; width: 100%; border-radius: 18px; overflow: hidden;
            margin: 0.15rem 0 1.25rem 0; box-shadow: 0 10px 28px rgba(0,0,0,0.14);
          }}
          .team-banner-top {{ background: {theme.top_hex}; padding: 2.4rem 1.4rem 2.1rem 1.4rem; }}
          .team-banner-bottom {{
            background: {theme.rest_hex}; padding: 0.6rem 1.4rem 0.35rem 1.4rem;
            min-height: 54px; display: flex; align-items: flex-end;
          }}
          .team-title {{
            margin: 0; color: {title_text_color}; font-size: 1.75rem; font-weight: 900; letter-spacing: 0.2px;
          }}
          .team-subtitle {{ margin: 0.35rem 0 0 0; color: {subtitle_color}; font-size: 1.05rem; font-weight: 700; }}
          .team-meta {{ margin: 0; color: #000000; font-size: 0.98rem; font-weight: 800; }}
          .team-logo {{
            position: absolute; top: 18px; right: 18px; height: 78px; width: 78px; object-fit: contain;
            background: rgba(255,255,255,0.92); border-radius: 16px; padding: 9px;
          }}
          .team-footer-bar {{
            position: fixed; left: 0; right: 0; bottom: 0; height: 14px; background: {theme.top_hex}; z-index: 9999;
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


CORNER_EVENTS_CSV = "data/corner_events_all_matches.csv"
EVENTS_SEQ_CSV = "data/corner_events_full_sequences.csv"
HEADERS_CSV = "data/corner_positions_headers.csv"

IMG_PATHS = {
    "def_L": "images/no_names_left.png",
    "def_R": "images/no_names_right.png",
    "att_L": "images/left_side_corner.png",
    "att_R": "images/right_side_corner.png",
}


def get_img_path(key: str) -> Optional[str]:
    path = IMG_PATHS.get(key)
    return path if path and os.path.exists(path) else None


@st.cache_data
def load_corner_jsonlike(csv_path: str):
    return oa.load_corner_events_csv_as_jsonlike(csv_path)


@st.cache_data
def load_events_sequences(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path, low_memory=False).where(pd.notnull, None)


@st.cache_data
def load_headers(csv_path: str) -> pd.DataFrame:
    return oa.load_corner_positions_headers(csv_path)


@st.cache_data
def get_team_list(json_data):
    # IMPORTANT: team list should come from full dataset (not match-window filtered)
    return oa.extract_all_teams(json_data)


@st.cache_data
def get_analysis_results(json_data, team_name: str):
    return oa.process_corner_data(json_data, team_name)


@st.cache_data
def get_league_stats(json_data):
    return oa.compute_league_attacking_corner_shot_rates(json_data)


# ---------------- Sidebar ----------------
st.sidebar.header("Configuration")

if not os.path.exists(CORNER_EVENTS_CSV):
    st.error(f"❌ Data file not found at: `{CORNER_EVENTS_CSV}`")
    st.stop()

# Load FULL dataset once (unfiltered)
json_data_full = load_corner_jsonlike(CORNER_EVENTS_CSV)

all_teams = get_team_list(json_data_full)
if not all_teams:
    st.error("❌ No teams found in dataset.")
    st.stop()

selected_team = st.sidebar.selectbox("Select team", all_teams)

# Slider = last X matches, BUT only for selected team
team_matches_all = oa.filter_last_n_matches_for_team(json_data_full, selected_team, None).get("matches", [])
team_total = len(team_matches_all)

n_last = st.sidebar.slider(
    "Analyze last X matches (selected team only)",
    min_value=1,
    max_value=max(team_total, 1),
    value=max(team_total, 1),  # default = all matches for that team
    step=1,
    disabled=(team_total == 0),
)

json_data_view = oa.filter_last_n_matches_for_team(json_data_full, selected_team, n_last)

# Latest match caption should reflect full dataset
latest_dt, latest_name = oa.get_latest_match_info(json_data_full)
st.sidebar.markdown("---")
st.sidebar.caption(
    f"Latest match in dataset: {latest_dt.strftime('%d-%m-%Y')} — {latest_name}"
    if latest_dt and latest_name
    else "Latest match in dataset: -"
)

# --- ADD DATA (single uploader + single button) ---
st.sidebar.markdown("---")
st.sidebar.subheader("Add data")

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

uploaded_files = st.sidebar.file_uploader(
    "Upload SciSports JSON files (Events + Positions)",
    type=["json"],
    accept_multiple_files=True,
    key=f"uploader_{st.session_state.uploader_key}",
)

run_update = st.sidebar.button(
    "Update database",
    type="primary",
    disabled=not uploaded_files,
)



if run_update:
    uploads_root = Path("data/_uploads")
    uploads_root.mkdir(parents=True, exist_ok=True)

    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    batch_dir = uploads_root / f"batch_{stamp}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    for uf in uploaded_files or []:
        (batch_dir / Path(uf.name).name).write_bytes(uf.getbuffer())

    with st.spinner("Updating database CSVs..."):
        result = upd.update_database(
            uploads_dir=batch_dir,
            data_dir=Path("data"),
        )

    if result.get("ok"):
        st.sidebar.success("✅ Database updated. ")
        st.sidebar.write(result)  # <-- add this temporarily so you see github_push_ok/msg
       

        # remove batch files
        try:
            shutil.rmtree(batch_dir, ignore_errors=True)
        except Exception:
            pass

        # clear uploader visually
        st.session_state.uploader_key += 1

        # reload CSVs from disk
        st.cache_data.clear()
        st.rerun()
    else:
        st.sidebar.error(f"❌ Update failed: {result.get('error', 'Unknown error')}")
        st.sidebar.write(result)

# ---------------- Main analysis ----------------
if json_data_view and selected_team:
    st.sidebar.caption(f"Using {len(json_data_view.get('matches', []))} match(es) for {selected_team}")

    with st.spinner(f"Analyzing {selected_team}..."):
        results = get_analysis_results(json_data_view, selected_team)
        viz_config = oa.get_visualization_coords()
        league_stats = get_league_stats(json_data_view)

    themes = build_team_themes()
    _ = apply_header(selected_team, results.get("used_matches"), themes)

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

    st.divider()
    st.markdown("##### 📋 Corner Takers (Left)")
    st.dataframe(results["tables"]["left"], width="stretch", hide_index=True)

    st.divider()
    st.markdown("##### 📋 Corner Takers (Right)")
    st.dataframe(results["tables"]["right"], width="stretch", hide_index=True)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Def. Corners Left: how many crosses turned into a shot?")
        tot, ids, pcts = results["defensive"]["left"]
        fig = oa.plot_shots_defensive(get_img_path("def_L"), viz_config["def_L"], pcts, tot, ids)
        st.pyplot(style_fig_bg(fig, APP_BG))
    with col2:
        st.subheader("Def. Corners Right: how many crosses turned into a shot?")
        tot, ids, pcts = results["defensive"]["right"]
        fig = oa.plot_shots_defensive(get_img_path("def_R"), viz_config["def_R"], pcts, tot, ids)
        st.pyplot(style_fig_bg(fig, APP_BG))

    st.divider()
    st.markdown("### Attacking & Defending corner headers: Who is dangerous, and who is weak?")

    if not os.path.exists(HEADERS_CSV):
        st.warning(f"Player charts skipped because `{HEADERS_CSV}` is missing.")
    elif not os.path.exists(EVENTS_SEQ_CSV):
        st.warning(f"Player charts skipped because `{EVENTS_SEQ_CSV}` is missing (needed to map HOME/AWAY to teams).")
    else:
        with st.spinner("Loading headers + mapping HOME/AWAY to actual teams..."):
            headers_df = load_headers(HEADERS_CSV)
            seq_df = load_events_sequences(EVENTS_SEQ_CSV)

            headers_df = oa.attach_actual_club_from_events(headers_df, seq_df)

            team_c = oa._canon_team(selected_team) or selected_team
            df_team = headers_df[headers_df["club_actual_canon"] == team_c].copy()

        if df_team.empty:
            st.warning("No player header rows found for this team (after HOME/AWAY mapping).")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### 🟦 Attacking corner players (chart)")
                fig_att = oa.plot_attacking_corner_players_headers(df_team, max_players=15)
                st.pyplot(style_fig_bg(fig_att, APP_BG), clear_figure=True)

            with col2:
                st.markdown("##### 🟥 Defending corner players (chart)")
                fig_def = oa.plot_defending_corner_players_diverging(df_team, max_players=15)
                st.pyplot(style_fig_bg(fig_def, APP_BG), clear_figure=True)
