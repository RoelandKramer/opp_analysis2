# ============================================================
# file: app.py
# ============================================================
from __future__ import annotations

import base64
import io
import os
import re
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib as mpl
import pandas as pd
import streamlit as st

import opp_analysis_new as oa
import update_database as upd
from ppt_template_filler import fill_corner_template_pptx, fig_to_png_bytes, fig_to_png_bytes_labels


# --- Secrets -> env for git push (safe) ---
if "GITHUB_TOKEN" in st.secrets:
    os.environ["GITHUB_TOKEN"] = st.secrets["GITHUB_TOKEN"]
if "GITHUB_REPO" in st.secrets:
    os.environ["GITHUB_REPO"] = st.secrets["GITHUB_REPO"]
os.environ["GITHUB_BRANCH"] = st.secrets.get("GITHUB_BRANCH", "main")


# --- Canonicalization fallback for datasets unknown to OA (women teams, etc.) ---
_OA_GET_CANON = oa.get_canonical_team


def _get_canonical_team_safe(name: Any) -> Optional[str]:
    raw = str(name).strip() if name is not None else ""
    if raw.upper() == "NOT_APPLICABLE":
        return None
    canon = _OA_GET_CANON(raw)
    if canon:
        return canon
    return raw or None


oa.get_canonical_team = _get_canonical_team_safe

st.set_page_config(page_title="Opponent Analysis - Set Pieces", layout="wide")

APP_BG = "#FFFFFF"
TEMPLATE_PPTX = "template_opp_analysis.pptx"

DATA_ROOT = Path(os.getenv("APP_DATA_ROOT", "data"))
CORNER_EVENTS_CSV = str(DATA_ROOT / "corner_events_all_matches.csv")
EVENTS_SEQ_CSV = str(DATA_ROOT / "corner_events_full_sequences.csv")
HEADERS_CSV = str(DATA_ROOT / "corner_positions_headers.csv")

CANON_PATCH_VERSION = "v4_fullseq_shots"
DATASET_ID = f"{DATA_ROOT.resolve()}::{CANON_PATCH_VERSION}"

if st.session_state.get("dataset_id") != DATASET_ID:
    st.session_state.dataset_id = DATASET_ID
    st.session_state.dataset_version = 0
    st.session_state.analysis_cache = {}
    st.cache_data.clear()

if "analysis_cache" not in st.session_state:
    st.session_state.analysis_cache = {}


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
        "Ajax W": TeamTheme("#C31F3D", "#FFFFFF", "logos/jong_ajax.png"),
        "ADO Den Haag W": TeamTheme("#00802C", "#FFE200", "logos/ado_den_haag.png"),
        "Ado Den Haag W": TeamTheme("#00802C", "#FFE200", "logos/ado_den_haag.png"),
        "PSV W": TeamTheme("#E62528", "#FFFFFF", "logos/jong_psv.png"),
        "AZ W": TeamTheme("#DB0021", "#FFFFFF", "logos/jong_az.png"),
        "Utrecht W": TeamTheme("#ED1A2F", "#FFFFFF", "logos/jong_fc_utrecht.png"),
        "Excelsior Rotterdam W": TeamTheme("#E2001A", "#000000", "logos/excelsior_rotterdam.png"),
        "SC Heerenveen W": TeamTheme("#004F9F", "#FFFFFF", "logos/sc_heerenveen.png"),
        "FC Twente W": TeamTheme("#E6001A", "#FFFFFF", "logos/fc_twente.png"),
        "Hera United W": TeamTheme("#191970", "#FFFFFF", "logos/hera_united.png"),
        "NAC Breda W": TeamTheme("#282828", "#FFDD25", "logos/nac_breda.png"),
        "Feyenoord W": TeamTheme("#FF0000", "#000000", "logos/feyenoord.png"),
        "PEC Zwolle W": TeamTheme("#1E59AE", "#6AC2EE", "logos/pec_zwolle.png"),
        "PEC ZWOLLE W": TeamTheme("#1E59AE", "#6AC2EE", "logos/pec_zwolle.png"),
    }


def _theme_for_team(themes: Dict[str, TeamTheme], team: str) -> TeamTheme:
    if not team:
        return TeamTheme("#111827", "#FFFFFF", "logos/default.png")
    if team in themes:
        return themes[team]
    key = team.strip()
    if key in themes:
        return themes[key]
    upper_map = {k.upper(): v for k, v in themes.items()}
    v = upper_map.get(key.upper())
    if v:
        return v
    compact = re.sub(r"\s+", " ", key)
    v = upper_map.get(compact.upper())
    if v:
        return v
    return TeamTheme("#111827", "#FFFFFF", f"logos/{slugify(team)}.png")


def set_matplotlib_bg(bg_hex: str) -> None:
    mpl.rcParams["figure.facecolor"] = bg_hex
    mpl.rcParams["savefig.facecolor"] = bg_hex
    mpl.rcParams["axes.facecolor"] = bg_hex


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
def load_corner_jsonlike(csv_path: str, cache_buster: str) -> dict:
    _ = cache_buster
    return oa.load_corner_events_csv_as_jsonlike(csv_path)


@st.cache_data
def load_events_sequences(csv_path: str, cache_buster: str) -> pd.DataFrame:
    _ = cache_buster
    return pd.read_csv(csv_path, low_memory=False).where(pd.notnull, None)


@st.cache_data
def load_headers(csv_path: str, cache_buster: str) -> pd.DataFrame:
    _ = cache_buster
    return oa.load_corner_positions_headers(csv_path)


@st.cache_data
def get_canonical_team_options(json_data_full: dict, cache_buster: str) -> List[str]:
    _ = cache_buster
    canon: set[str] = set()
    for match in (json_data_full.get("matches", []) or []):
        for ev in (match.get("corner_events", []) or []):
            raw_s = str(ev.get("teamName") or "").strip()
            if not raw_s or raw_s.upper() == "NOT_APPLICABLE":
                continue
            c = oa.get_canonical_team(raw_s)
            if c:
                canon.add(c)
    return sorted(canon)


@st.cache_data
def load_shot_map_from_full_sequences(seq_csv_path: str, cache_buster: str) -> Dict[Tuple[str, str], bool]:
    """
    (match_id, sequenceId) -> True if that sequence contains a shot event.
    Source of truth for women shot detection.
    """
    _ = cache_buster
    if not os.path.exists(seq_csv_path):
        return {}

    df = pd.read_csv(seq_csv_path, low_memory=False).where(pd.notnull, None)

    if "match_id" not in df.columns:
        return {}

    seq_col = "sequenceId" if "sequenceId" in df.columns else ("corner_sequence_id" if "corner_sequence_id" in df.columns else None)
    if seq_col is None:
        return {}

    bt_col = "baseTypeName" if "baseTypeName" in df.columns else None
    bid_col = "baseTypeId" if "baseTypeId" in df.columns else None
    if bt_col is None and bid_col is None:
        return {}

    df["match_id"] = df["match_id"].astype(str)
    df[seq_col] = df[seq_col].astype(str)

    is_shot = pd.Series(False, index=df.index)
    if bt_col is not None:
        is_shot |= df[bt_col].astype(str).str.strip().str.upper().eq("SHOT")
    if bid_col is not None:
        bid = pd.to_numeric(df[bid_col], errors="coerce")
        is_shot |= bid.eq(6)

    shot_rows = df.loc[is_shot, ["match_id", seq_col]].dropna()
    out: Dict[Tuple[str, str], bool] = {}
    for mid, sid in shot_rows.itertuples(index=False):
        out[(str(mid), str(sid))] = True
    return out


def _match_dt(match: dict) -> datetime:
    dt = match.get("match_date")
    return dt if isinstance(dt, datetime) else datetime.min


def _match_has_team(match: dict, team_name: str) -> bool:
    team_raw = (team_name or "").strip()
    team_canon = oa.get_canonical_team(team_raw)

    events = match.get("corner_events", []) or []
    if not events:
        return False

    if team_canon:
        teams_in_match_canon = {
            oa.get_canonical_team(ev.get("teamName"))
            for ev in events
            if oa.get_canonical_team(ev.get("teamName"))
        }
        if team_canon in teams_in_match_canon:
            return True

    teams_in_match_raw = {
        str(ev.get("teamName") or "").strip()
        for ev in events
        if str(ev.get("teamName") or "").strip()
    }
    return team_raw in teams_in_match_raw


def _save_uploads_to_batch(uploaded_files, batch_dir: Path) -> int:
    json_count = 0
    for uf in uploaded_files or []:
        name = Path(uf.name).name
        suffix = Path(name).suffix.lower()

        if suffix == ".json":
            out = batch_dir / name
            out.write_bytes(uf.getbuffer())
            json_count += 1
        elif suffix == ".zip":
            zbytes = io.BytesIO(uf.getbuffer())
            with zipfile.ZipFile(zbytes) as z:
                for member in z.infolist():
                    if member.is_dir():
                        continue
                    mname = Path(member.filename).name
                    if not mname.lower().endswith(".json"):
                        continue
                    out = batch_dir / mname
                    out.write_bytes(z.read(member))
                    json_count += 1
    return json_count


def _center_container_css() -> None:
    st.markdown(
        """
        <style>
          [data-testid="stAppViewContainer"] { background: #FFFFFF; }
          [data-testid="stHeader"] { background: transparent; }
          header { background: transparent !important; }
          section.main > div { padding-top: 0.75rem; padding-bottom: 1.5rem; }
          .block-container { max-width: 1200px; }
          div[data-testid="stVerticalBlock"] > div { gap: 0.75rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_header(
    team: str,
    primary_hex: str,
    secondary_hex: str,
    logo_path: str,
    window_label: str,
    matches_analyzed: Optional[int],
) -> None:
    logo_b64 = None
    if logo_path and os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            logo_b64 = base64.b64encode(f.read()).decode("utf-8")

    title_text_color = "#FFFFFF" if (primary_hex or "").upper() != "#FFFFFF" else "#111827"
    subtitle_color = "rgba(255,255,255,0.90)" if title_text_color == "#FFFFFF" else "rgba(17,24,39,0.85)"
    matches_val = matches_analyzed if matches_analyzed is not None else "-"

    st.markdown(
        f"""
        <style>
          .team-banner {{
            position: relative; width: 100%; border-radius: 18px; overflow: hidden;
            margin: 0 0 1.0rem 0; box-shadow: 0 10px 28px rgba(0,0,0,0.14);
          }}
          .team-banner-top {{ background: {primary_hex}; padding: 1.2rem 1.4rem 0.95rem 1.4rem; }}
          .team-banner-bottom {{
            background: {secondary_hex}; padding: 0.55rem 1.4rem 0.4rem 1.4rem;
            min-height: 52px; display: flex; align-items: flex-end;
          }}
          .team-title {{ margin: 0; color: {title_text_color}; font-size: 1.65rem; font-weight: 900; }}
          .team-subtitle {{ margin: 0.35rem 0 0 0; color: {subtitle_color}; font-size: 1.05rem; font-weight: 750; }}
          .team-meta {{ margin: 0; color: #000000; font-size: 0.98rem; font-weight: 800; }}
          .team-logo {{
            position: absolute; top: 12px; right: 16px; height: 74px; width: 74px; object-fit: contain;
            background: rgba(255,255,255,0.92); border-radius: 16px; padding: 9px;
          }}
        </style>
        <div class="team-banner">
          <div class="team-banner-top">
            <p class="team-title">Opponent analysis - Set Pieces</p>
            <p class="team-subtitle">{team}</p>
            {f'<img class="team-logo" src="data:image/png;base64,{logo_b64}" alt="{team} logo" />' if logo_b64 else ""}
          </div>
          <div class="team-banner-bottom">
            <p class="team-meta">Matches Analyzed: {matches_val} | Window: {window_label} | Team: {team}</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _generate_filled_pptx(
    *,
    json_data_full: dict,
    selected_team: str,
    n_last: int,
    themes: Dict[str, TeamTheme],
    matches_analyzed_total: int,
    shot_map: Dict[Tuple[str, str], bool],
) -> Tuple[bytes, str]:
    matches_full = json_data_full.get("matches", []) or []
    team_matches_sorted = sorted(
        [m for m in matches_full if _match_has_team(m, selected_team)],
        key=_match_dt,
        reverse=True,
    )
    team_total = len(team_matches_sorted)
    if team_total == 0:
        raise ValueError("No matches found for this team in the dataset.")

    team_matches_window = team_matches_sorted[:n_last]
    json_data_view = {"matches": team_matches_window}

    cache_key = (st.session_state.dataset_id, st.session_state.dataset_version, selected_team, n_last)
    if cache_key not in st.session_state.analysis_cache:
        results = oa.process_corner_data(json_data_view, selected_team, shot_map=shot_map)
        league_stats = oa.compute_league_attacking_corner_shot_rates(json_data_full, shot_map=shot_map)
        viz_config = oa.get_visualization_coords()
        st.session_state.analysis_cache[cache_key] = {
            "results": results,
            "league_stats": league_stats,
            "viz_config": viz_config,
        }

    cached = st.session_state.analysis_cache[cache_key]
    results = cached["results"]
    league_stats = cached["league_stats"]
    viz_config = cached["viz_config"]

    theme = _theme_for_team(themes, selected_team)
    set_matplotlib_bg("#FFFFFF")

    fig_att_L = oa.plot_percent_attacking(
        get_img_path("att_L"),
        viz_config["att_L"],
        viz_config["att_centers_L"],
        results["attacking"]["left_pct"],
    )
    fig_att_R = oa.plot_percent_attacking(
        get_img_path("att_R"),
        viz_config["att_R"],
        viz_config["att_centers_R"],
        results["attacking"]["right_pct"],
    )

    (tot_L, shot_L, pct_L) = results["attacking_shots"]["left"]
    pctiles_L = oa.build_percentiles_for_team(league_stats, selected_team, "left", min_zone_corners=0)
    fig_att_shots_L = oa.plot_shots_attacking_with_percentile(
        get_img_path("def_L"),
        viz_config["def_L"],
        pct_L,
        tot_L,
        shot_L,
        pctiles_L,
        min_zone_corners=0,
        font_size=16,
    )

    (tot_R, shot_R, pct_R) = results["attacking_shots"]["right"]
    pctiles_R = oa.build_percentiles_for_team(league_stats, selected_team, "right", min_zone_corners=0)
    fig_att_shots_R = oa.plot_shots_attacking_with_percentile(
        get_img_path("def_R"),
        viz_config["def_R"],
        pct_R,
        tot_R,
        shot_R,
        pctiles_R,
        min_zone_corners=0,
        font_size=16,
    )

    (tot_dL, ids_dL, pcts_dL) = results["defensive"]["left"]
    fig_def_L = oa.plot_shots_defensive(get_img_path("def_L"), viz_config["def_L"], pcts_dL, tot_dL, ids_dL)

    (tot_dR, ids_dR, pcts_dR) = results["defensive"]["right"]
    fig_def_R = oa.plot_shots_defensive(get_img_path("def_R"), viz_config["def_R"], pcts_dR, tot_dR, ids_dR)

    fig_att_headers = None
    fig_def_headers = None
    try:
        if os.path.exists(HEADERS_CSV) and os.path.exists(EVENTS_SEQ_CSV):
            headers_df = load_headers(HEADERS_CSV, st.session_state.dataset_id)
            seq_df = load_events_sequences(EVENTS_SEQ_CSV, st.session_state.dataset_id)
            headers_df = oa.attach_actual_club_from_events(headers_df, seq_df)
            team_c = oa._canon_team(selected_team) or selected_team
            df_team = headers_df[headers_df["club_actual_canon"] == team_c].copy()
            if not df_team.empty:
                fig_att_headers = oa.plot_attacking_corner_players_headers(df_team, max_players=15)
                fig_def_headers = oa.plot_defending_corner_players_diverging(df_team, max_players=15)
    except Exception:
        pass

    images_by_shape_name = {
        0: {
            "PH_Corners_left_positions_vis": fig_to_png_bytes(fig_att_L, dpi=360),
            "PH_Corners_right_positions_vis": fig_to_png_bytes(fig_att_R, dpi=360),
            "PH_Corners_left_shots_vis": fig_to_png_bytes(fig_att_shots_L, dpi=360),
            "PH_Corners_right_shots_vis": fig_to_png_bytes(fig_att_shots_R, dpi=360),
        },
        1: {
            "PH_def_left": fig_to_png_bytes(fig_def_L, dpi=360),
            "PH_def_right": fig_to_png_bytes(fig_def_R, dpi=360),
        },
    }

    images_by_token = {
        "{att_corners_headers}": [fig_to_png_bytes_labels(fig_att_headers, dpi=360)] if fig_att_headers is not None else [],
        "{def_corners_headers}": [fig_to_png_bytes_labels(fig_def_headers, dpi=360)] if fig_def_headers is not None else [],
    }

    meta = {
        "{TEAM_NAME}": selected_team,
        "{nlc}": str(results.get("own_left_count", 0)),
        "{nrc}": str(results.get("own_right_count", 0)),
        "{MATCHES_ANALYZED}": str(matches_analyzed_total),
        "{bottom_bar}": theme.rest_hex,
        "{middle_bar}": theme.rest_hex,
    }

    payload = fill_corner_template_pptx(
        template_pptx_path=TEMPLATE_PPTX,
        team_name=selected_team,
        team_primary_hex=theme.top_hex,
        team_secondary_hex=theme.rest_hex,
        logo_path=theme.logo_relpath if theme.logo_relpath and os.path.exists(theme.logo_relpath) else None,
        meta_replacements=meta,
        images_by_token=images_by_token,
        images_by_shape_name=images_by_shape_name,
        left_takers_df=results["tables"]["left"],
        right_takers_df=results["tables"]["right"],
    )
    return payload.pptx_bytes, payload.filename


# ---------------- UI ----------------
_center_container_css()

if not os.path.exists(CORNER_EVENTS_CSV):
    st.error(f"❌ Data file not found at: `{CORNER_EVENTS_CSV}`")
    st.stop()

json_data_full = load_corner_jsonlike(CORNER_EVENTS_CSV, DATASET_ID)

all_teams = get_canonical_team_options(json_data_full, DATASET_ID)
if not all_teams:
    st.error("❌ No teams found in dataset.")
    st.stop()

themes = build_team_themes()

shot_map = load_shot_map_from_full_sequences(EVENTS_SEQ_CSV, DATASET_ID)

if "selected_team" not in st.session_state or st.session_state.selected_team not in all_teams:
    st.session_state.selected_team = all_teams[0]

matches_full = json_data_full.get("matches", []) or []
team_matches_sorted = sorted(
    [m for m in matches_full if _match_has_team(m, st.session_state.selected_team)],
    key=_match_dt,
    reverse=True,
)
team_total_for_header = len(team_matches_sorted) or 1
n_last_default = min(team_total_for_header, st.session_state.get("n_last", team_total_for_header))
window_label = "All" if n_last_default >= team_total_for_header else f"Last {n_last_default}"

theme = _theme_for_team(themes, st.session_state.selected_team)

_render_header(
    team=st.session_state.selected_team,
    primary_hex=theme.top_hex,
    secondary_hex=theme.rest_hex,
    logo_path=theme.logo_relpath,
    window_label=window_label,
    matches_analyzed=team_total_for_header,
)

st.subheader("Configuration")

st.selectbox(
    "Select team",
    all_teams,
    index=all_teams.index(st.session_state.selected_team),
    key="selected_team",
)

team_matches_sorted = sorted(
    [m for m in matches_full if _match_has_team(m, st.session_state.selected_team)],
    key=_match_dt,
    reverse=True,
)
team_total = len(team_matches_sorted)
if team_total == 0:
    st.warning("No matches found for this team in the dataset.")
    st.stop()

n_last = st.slider(
    "Analyze last X matches (selected team only)",
    min_value=1,
    max_value=team_total,
    value=min(st.session_state.get("n_last", team_total), team_total),
    step=1,
    key="n_last",
)

generate = st.button("Generate corner analysis", type="primary", width="stretch")
if generate:
    if not os.path.exists(TEMPLATE_PPTX):
        st.error(f"❌ Template PPTX not found at `{TEMPLATE_PPTX}`. Put it in the repo root.")
        st.stop()

    with st.spinner("Generating Corner Analysis"):
        pptx_bytes, fname = _generate_filled_pptx(
            json_data_full=json_data_full,
            selected_team=st.session_state.selected_team,
            n_last=n_last,
            themes=themes,
            matches_analyzed_total=team_total,
            shot_map=shot_map,
        )

    st.success("✅ PowerPoint generated.")
    st.download_button(
        "Download filled template (.pptx)",
        data=pptx_bytes,
        file_name=fname,
        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        width="stretch",
    )

with st.sidebar:
    st.header("Add data")

    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    uploaded_files = st.file_uploader(
        "Upload SciSports files (JSON or ZIP)",
        type=["json", "zip"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
    )

    run_update = st.button("Update database", type="primary", disabled=not uploaded_files, width="stretch")

    latest_dt, latest_name = oa.get_latest_match_info(json_data_full)
    st.caption(
        f"Latest match in dataset: {latest_dt.strftime('%d-%m-%Y')} — {latest_name}"
        if latest_dt and latest_name
        else "Latest match in dataset: -"
    )

    if run_update:
        uploads_root = Path("data/_uploads")
        uploads_root.mkdir(parents=True, exist_ok=True)

        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        batch_dir = uploads_root / f"batch_{stamp}"
        batch_dir.mkdir(parents=True, exist_ok=True)

        n_json = _save_uploads_to_batch(uploaded_files, batch_dir)

        if n_json == 0:
            st.error("❌ No JSON files found in upload/zip.")
        else:
            with st.spinner("Updating database CSVs..."):
                result = upd.update_database(
                    uploads_dir=batch_dir,
                    data_dir=Path("data"),
                )

            if result.get("ok"):
                st.success(
                    "✅ Database updated. "
                    f"events_all Δ{result.get('added_events_all', 0)}, "
                    f"events_full Δ{result.get('added_events_full', 0)}, "
                    f"headers Δ{result.get('headers_net_new_rows', 0)} | "
                    f"GitHub: {result.get('github_push_msg', '')}"
                )

                try:
                    shutil.rmtree(batch_dir, ignore_errors=True)
                except Exception:
                    pass

                st.session_state.uploader_key += 1
                st.session_state.dataset_version += 1
                st.session_state.analysis_cache = {}
                st.cache_data.clear()
                st.rerun()
            else:
                st.error(f"❌ Update failed: {result.get('error', 'Unknown error')}")
                st.write(result)
