# ============================================================
# file: opp_analysis_new.py
# ============================================================
from __future__ import annotations

import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.cm as cm
import matplotlib.image as mpimg
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon

_MATCH_ID_FROM_NAME = re.compile(r"-\s*(\d+)\s*(?:\(\d+\))?\s*$", re.IGNORECASE)
_DATE_YYYYMMDD = re.compile(r"^(?P<d>\d{8})\s+")
_DATE_DDMMYYYY = re.compile(r"^(?P<d>\d{2}-\d{2}-\d{4})")

HEADER_BODY_PART_NAMES = {"HEAD"}

ShotMap = Dict[Tuple[str, str], bool]  # (match_id, sequenceId) -> has_shot


def truthy(x: Any) -> bool:
    if x is True:
        return True
    if isinstance(x, (int, float)) and x == 1:
        return True
    if isinstance(x, str) and x.strip().lower() in {"true", "1", "yes"}:
        return True
    return False


def _is_true_corner_start(ev: Dict[str, Any]) -> bool:
    return ev.get("possessionTypeName") == "CORNER" and truthy(ev.get("sequenceStart"))


def _safe_int(x: Any, default: int = -1) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _norm_seq_id(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    return s if s else None


def _safe_abs_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        return abs(v) if math.isfinite(v) else None
    except Exception:
        return None


def _match_id_from_path(p: Any) -> Optional[str]:
    if not isinstance(p, str) or not p.strip():
        return None
    name = p.replace("\\", "/").split("/")[-1]
    stem = re.sub(r"\.json\s*$", "", name, flags=re.IGNORECASE)
    m = _MATCH_ID_FROM_NAME.search(stem)
    return m.group(1) if m else None


def _match_date_from_filename(p: Any) -> Optional[datetime]:
    if not isinstance(p, str) or not p.strip():
        return None
    name = p.replace("\\", "/").split("/")[-1]
    stem = re.sub(r"\.json\s*$", "", name, flags=re.IGNORECASE)

    m = _DATE_YYYYMMDD.match(stem)
    if m:
        try:
            return datetime.strptime(m.group("d"), "%Y%m%d")
        except Exception:
            return None

    m = _DATE_DDMMYYYY.match(stem)
    if m:
        try:
            return datetime.strptime(m.group("d"), "%d-%m-%Y")
        except Exception:
            return None

    return None


def _parse_frame_json(x: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(x, str):
        return None
    s = x.strip()
    if not s or s.lower() in {"nan", "none"} or not s.startswith("{"):
        return None
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else None
    except Exception:
        return None


def _load_bg(file_obj_or_path: Optional[str]):
    if file_obj_or_path and os.path.exists(file_obj_or_path):
        return mpimg.imread(file_obj_or_path)
    return np.ones((800, 1400, 3))


TEAM_NAME_MAPPING: Dict[str, Optional[str]] = {
    "ADO Den Haag": "ADO Den Haag",
    "Almere City FC": "Almere City FC",
    "De Graafschap": "De Graafschap",
    "Eindhoven": "Eindhoven",
    "FC Den Bosch": "FC Den Bosch",
    "FC Dordrecht": "FC Dordrecht",
    "FC Emmen": "FC Emmen",
    "Helmond Sport": "Helmond Sport",
    "Jong AZ": "Jong AZ",
    "Jong Ajax": "Jong Ajax",
    "Jong FC Utrecht": "Jong FC Utrecht",
    "Jong PSV": "Jong PSV",
    "MVV Maastricht": "MVV Maastricht",
    "RKC Waalwijk": "RKC Waalwijk",
    "Roda JC Kerkrade": "Roda JC Kerkrade",
    "SC Cambuur": "SC Cambuur",
    "TOP Oss": "TOP Oss",
    "VVV-Venlo": "VVV-Venlo",
    "Vitesse": "Vitesse",
    "Willem II": "Willem II",
    "FC Twente W": "FC Twente W",
    "ADO Den Haag W": "ADO Den Haag W",
    "AZ W": "AZ W",
    "Ajax W": "Ajax W",
    "Excelsior Rotterdam W": "Excelsior Rotterdam W",
    "Feyenoord W": "Feyenoord W",
    "Hera United W": "Hera United W",
    "NAC Breda W": "NAC Breda W",
    "PEC ZWOLLE W": "PEC Zwolle W",
    "PEC Zwolle W": "PEC Zwolle W",
    "PSV W": "PSV W",
    "SC Heerenveen W": "SC Heerenveen W",
    "Utrecht W": "Utrecht W",
    "Almere City": "Almere City FC",
    "Den Bosch": "FC Den Bosch",
    "Dordrecht": "FC Dordrecht",
    "AZ Alkmaar U23": "Jong AZ",
    "Ajax Amsterdam U21": "Jong Ajax",
    "Jong Utrecht": "Jong FC Utrecht",
    "Jong PSV Eindhoven": "Jong PSV",
    "MVV": "MVV Maastricht",
    "VVV Venlo": "VVV-Venlo",
    "VVV-Venlo VVV-Venlo": "VVV-Venlo",
    "NOT_APPLICABLE": None,
}


def get_canonical_team(raw_name: Any) -> Optional[str]:
    if not isinstance(raw_name, str):
        return None
    return TEAM_NAME_MAPPING.get(raw_name.strip())


def _canon_team(name: Any) -> Optional[str]:
    if not isinstance(name, str):
        return None
    s = name.strip()
    if not s:
        return None
    return TEAM_NAME_MAPPING.get(s, s)


def load_corner_events_csv_as_jsonlike(corner_events_csv_path: str) -> Dict[str, Any]:
    if not os.path.exists(corner_events_csv_path):
        raise FileNotFoundError(f"Corner events CSV not found: {corner_events_csv_path}")

    df = pd.read_csv(corner_events_csv_path, low_memory=False)
    if df.empty:
        return {"matches": []}
    df = df.where(pd.notnull(df), None)

    if "match_id" not in df.columns:
        raise ValueError("corner_events_all_matches.csv must contain 'match_id'")

    if "match_name" not in df.columns:
        df["match_name"] = None
    if "source_event_file" not in df.columns:
        df["source_event_file"] = None

    matches: List[Dict[str, Any]] = []
    for mid, g in df.groupby("match_id", dropna=False):
        if mid is None:
            continue

        names = [n for n in g["match_name"].dropna().astype(str).tolist() if n.strip()]
        mname = max(names, key=len) if names else str(mid)

        dt = None
        for v in g["source_event_file"].dropna().astype(str).tolist():
            dt = _match_date_from_filename(v)
            if dt:
                break

        pitch_top_x = (
            float(g["pitch_top_x"].dropna().iloc[0])
            if "pitch_top_x" in g.columns and g["pitch_top_x"].notna().any()
            else None
        )
        pitch_left_y = (
            float(g["pitch_left_y"].dropna().iloc[0])
            if "pitch_left_y" in g.columns and g["pitch_left_y"].notna().any()
            else None
        )

        matches.append(
            {
                "match_id": str(mid),
                "match_name": mname,
                "match_date": dt,
                "pitch_top_x": pitch_top_x,
                "pitch_left_y": pitch_left_y,
                "corner_events": g.to_dict(orient="records"),
            }
        )

    return {"matches": matches}


def get_latest_match_info(json_data: Dict[str, Any]) -> Tuple[Optional[datetime], Optional[str]]:
    latest_dt: Optional[datetime] = None
    latest_name: Optional[str] = None
    for m in json_data.get("matches", []):
        dt = m.get("match_date")
        if isinstance(dt, datetime) and (latest_dt is None or dt > latest_dt):
            latest_dt = dt
            latest_name = m.get("match_name")
    return latest_dt, latest_name


def _iter_numbers(x: Any) -> Iterable[float]:
    if isinstance(x, (int, float)) and math.isfinite(float(x)):
        yield float(x)


def _round_up_to_step(value: float, step: float) -> float:
    return math.ceil(value / step) * step

def build_header_tables_from_full_sequences(
    seq_df: pd.DataFrame,
    *,
    team: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      att_tbl: columns compatible with plot_attacking_corner_players_headers()
      def_tbl: columns compatible with plot_defending_corner_players_diverging()

    Uses ONLY full sequences (corner_events_full_sequences.csv).
    Logic:
      - Attacking: header shots by `team` in CORNER sequences started by `team`
      - Defending: header shots by opponents in CORNER sequences started by opponent (i.e., against `team`)
    """
    if seq_df is None or seq_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = seq_df.copy().where(pd.notnull(seq_df), None)

    # Required columns (best-effort)
    if "match_id" not in df.columns or "sequenceId" not in df.columns:
        return pd.DataFrame(), pd.DataFrame()

    # Identify corner sequence starts
    if "possessionTypeName" not in df.columns or "sequenceStart" not in df.columns:
        return pd.DataFrame(), pd.DataFrame()

    # shot rows
    bt = "baseTypeName" if "baseTypeName" in df.columns else None
    bid = "baseTypeId" if "baseTypeId" in df.columns else None
    if bt is None and bid is None:
        return pd.DataFrame(), pd.DataFrame()

    df["team_canon"] = df["teamName"].apply(_canon_team)
    team_c = _canon_team(team) or team

    df["seqId_norm"] = df["sequenceId"].astype(str)

    # mark corners (start event)
    df["is_corner_start"] = (df["possessionTypeName"] == "CORNER") & df["sequenceStart"].apply(truthy)

    # Determine which sequences are corner sequences and which team started them
    corner_starts = df.loc[df["is_corner_start"], ["match_id", "seqId_norm", "team_canon"]].dropna()
    if corner_starts.empty:
        return pd.DataFrame(), pd.DataFrame()

    # keep first starter per (match, seq)
    corner_starts = (
        corner_starts.groupby(["match_id", "seqId_norm"], as_index=False)
        .agg({"team_canon": "first"})
        .rename(columns={"team_canon": "corner_start_team"})
    )

    df = df.merge(corner_starts, on=["match_id", "seqId_norm"], how="left")

    # Shot detection
    is_shot = pd.Series(False, index=df.index)
    if bt is not None:
        is_shot |= df[bt].astype(str).str.strip().str.upper().eq("SHOT")
    if bid is not None:
        bid_s = pd.to_numeric(df[bid], errors="coerce")
        is_shot |= bid_s.eq(6)

    # header shot detection
    df["bodyPartName_norm"] = df.get("bodyPartName", None)
    df["is_header_shot"] = is_shot & df["bodyPartName_norm"].astype(str).str.strip().str.upper().isin(HEADER_BODY_PART_NAMES)

    # goal detection
    df["is_goal"] = df.get("resultName", "").astype(str).str.strip().str.upper().eq("SUCCESSFUL")

    # Filter to only header shots inside identified corner sequences
    hs = df.loc[df["is_header_shot"] & df["corner_start_team"].notna()].copy()
    if hs.empty:
        return pd.DataFrame(), pd.DataFrame()

    # --- ATTACKING: corner_start_team == team, header shooter is same team (usually true)
    att = hs.loc[hs["corner_start_team"] == team_c].copy()
    if not att.empty:
        att["player_name"] = att.get("playerName", "Unknown").fillna("Unknown").astype(str)
        att_tbl = (
            att.groupby("player_name", as_index=False)
            .agg(
                Attacking_corners_headed=("is_header_shot", "sum"),
                Attacking_corners_headed_and_scored=("is_goal", "sum"),
            )
        )
    else:
        att_tbl = pd.DataFrame(columns=["player_name", "Attacking_corners_headed", "Attacking_corners_headed_and_scored"])

    # --- DEFENDING: corner_start_team != team, i.e. opponent corner sequences against us
    dff = hs.loc[hs["corner_start_team"] != team_c].copy()
    if not dff.empty:
        dff["player_name"] = dff.get("playerName", "Unknown").fillna("Unknown").astype(str)

        # Map to columns that your diverging plot already understands:
        # Faults = Defending_corners_errors + fatal_errors
        # GoalsAllowed = Defending_corners_fatal_errors
        # Clearances = Defending_corners_defended (we don't have it from sequences => 0)
        def_tbl = (
            dff.groupby("player_name", as_index=False)
            .agg(
                Defending_corners_errors=("is_header_shot", "sum"),
                Defending_corners_fatal_errors=("is_goal", "sum"),
            )
        )
        def_tbl["Defending_corners_defended"] = 0
    else:
        def_tbl = pd.DataFrame(columns=["player_name", "Defending_corners_errors", "Defending_corners_fatal_errors", "Defending_corners_defended"])

    return att_tbl, def_tbl
    
def _collect_event_xy(events: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for ev in events:
        for k in ("startPosXM", "endPosXM"):
            xs.extend(_iter_numbers(ev.get(k)))
        for k in ("startPosYM", "endPosYM"):
            ys.extend(_iter_numbers(ev.get(k)))
    return np.array(xs, dtype=float), np.array(ys, dtype=float)


def detect_field_bounds_from_events(
    events: List[Dict[str, Any]],
    q: float = 0.999,
    margin_m: float = 1.0,
    round_step_m: float = 0.5,
) -> Tuple[float, float, float, float]:
    xs, ys = _collect_event_xy(events)
    if xs.size == 0 or ys.size == 0:
        return 52.5, -52.5, 34.0, -34.0
    abs_x, abs_y = np.abs(xs), np.abs(ys)
    half_len = _round_up_to_step(float(np.nanquantile(abs_x, q)) + margin_m, round_step_m)
    half_wid = _round_up_to_step(float(np.nanquantile(abs_y, q)) + margin_m, round_step_m)
    return max(half_len, 45.0), -max(half_len, 45.0), max(half_wid, 28.0), -max(half_wid, 28.0)


def _resolve_pitch_bounds(match: Dict[str, Any], events: List[Dict[str, Any]]) -> Tuple[float, float, float, float]:
    top_x = _safe_abs_float(match.get("pitch_top_x"))
    left_y = _safe_abs_float(match.get("pitch_left_y"))
    if top_x and left_y:
        return max(top_x, 45.0), -max(top_x, 45.0), max(left_y, 28.0), -max(left_y, 28.0)
    return detect_field_bounds_from_events(events)


def get_corner(start_x: float, start_y: float, TOP_X: float, LEFT_Y: float, thresh: float = 10.0) -> Optional[str]:
    corners = {
        "top_left": (TOP_X, LEFT_Y),
        "top_right": (TOP_X, -LEFT_Y),
        "bottom_left": (-TOP_X, LEFT_Y),
        "bottom_right": (-TOP_X, -LEFT_Y),
    }
    best_name, best_dist = None, float("inf")
    for name, (cx, cy) in corners.items():
        d = math.hypot(start_x - cx, start_y - cy)
        if d < best_dist:
            best_dist, best_name = d, name
    return best_name if best_dist <= thresh else None


def build_zones(top_x: float, bottom_x: float, left_y: float, right_y: float):
    length_PA = 16.5
    width_PA = 40.32
    length_PB = 5.49
    width_PB = 18.29

    PA_x = top_x - length_PA
    PB_x = top_x - length_PB
    half_PA = width_PA / 2
    half_PB = width_PB / 2

    PS_x = top_x - 11
    edge_x = top_x - (length_PA + 4)
    GA_band = width_PB / 3

    zones_TL = {
        "Short_Corner_Zone": [(top_x, left_y), (top_x, half_PA), (PA_x, left_y), (PA_x, half_PA)],
        "Front_Zone": [(top_x, half_PA), (top_x, half_PB), (PA_x, half_PB), (PA_x, half_PA)],
        "Back_Zone": [(top_x, -half_PB), (top_x, -half_PA), (PA_x, -half_PA), (PA_x, -half_PB)],
        "GA1": [(top_x, half_PB), (top_x, half_PB - GA_band), (PB_x, half_PB - GA_band), (PB_x, half_PB)],
        "GA2": [(top_x, half_PB - GA_band), (top_x, half_PB - 2 * GA_band), (PB_x, half_PB - 2 * GA_band), (PB_x, half_PB - GA_band)],
        "GA3": [(top_x, half_PB - 2 * GA_band), (top_x, -half_PB), (PB_x, -half_PB), (PB_x, half_PB - 2 * GA_band)],
        "CA1": [(PB_x, half_PB), (PB_x, half_PB - GA_band), (PS_x, half_PB - GA_band), (PS_x, half_PB)],
        "CA2": [(PB_x, half_PB - GA_band), (PB_x, half_PB - 2 * GA_band), (PS_x, half_PB - 2 * GA_band), (PS_x, half_PB - GA_band)],
        "CA3": [(PB_x, half_PB - 2 * GA_band), (PB_x, -half_PB), (PS_x, -half_PB), (PS_x, half_PB - 2 * GA_band)],
        "Edge_Zone": [(PS_x, half_PB), (PS_x, -half_PB), (edge_x, -half_PB), (edge_x, half_PB)],
    }
    zones_BR = {n: [(-x, -y) for x, y in r] for n, r in zones_TL.items()}
    zones_TR = {n: [(x, -y) for x, y in r] for n, r in zones_TL.items()}
    zones_BL = {n: [(-x, y) for x, y in r] for n, r in zones_TL.items()}
    return zones_TL, zones_BR, zones_TR, zones_BL


def point_in_rect(px: float, py: float, rect: List[Tuple[float, float]]) -> bool:
    xs = [p[0] for p in rect]
    ys = [p[1] for p in rect]
    return min(xs) <= px <= max(xs) and min(ys) <= py <= max(ys)


def _assign_zone(ex: float, ey: float, zones: Dict[str, List[Tuple[float, float]]]) -> Optional[str]:
    for n, r in zones.items():
        if point_in_rect(ex, ey, r):
            return n
    return None


def _sequence_has_shot(seq: List[Dict[str, Any]]) -> bool:
    for ev in seq or []:
        bt = str(ev.get("baseTypeName") or "").strip().upper()
        if bt == "SHOT":
            return True
        if _safe_int(ev.get("baseTypeId"), -1) == 6:
            return True
        if ev.get("shotTypeId") is not None or str(ev.get("shotTypeName") or "").strip():
            return True
        rn = str(ev.get("resultName") or "").strip().upper()
        if "SHOT" in rn or "GOAL" in rn:
            return True
        labels = ev.get("labels")
        if isinstance(labels, list):
            s = " ".join(str(x) for x in labels).upper()
            if "SHOT" in s or "GOAL" in s:
                return True
    return False


def _valid_zone_for_shot_lists(zone_val: Any) -> bool:
    return bool(zone_val and str(zone_val).strip() and zone_val != "Short_Corner_Zone")


def _valid_zone_for_attacking_shots(zone_val: Any) -> bool:
    return bool(zone_val and str(zone_val).strip() and zone_val != "Unassigned")


def _extract_jersey_number(ev: Dict[str, Any]) -> Optional[int]:
    candidates = (
        "jerseyNumber", "playerJerseyNumber", "playerJersey", "shirtNumber",
        "playerShirtNumber", "jersey_number", "player_jersey_number",
        "shirt_number",
    )
    for k in candidates:
        if k in ev and ev.get(k) not in (None, "", "NOT_APPLICABLE"):
            n = _safe_int(ev.get(k), default=-1)
            if n > 0:
                return n
    return None


def _display_player_with_jersey(ev: Dict[str, Any], fallback_name: str) -> str:
    name = (ev.get("playerName") or fallback_name or "").strip()
    jersey = _extract_jersey_number(ev)
    return name if jersey is None else f"{jersey}. {name}"


def _build_corner_taker_tables(
    left_corners: List[Dict[str, Any]],
    right_corners: List[Dict[str, Any]],
    min_corners: int = 5,
) -> Dict[str, pd.DataFrame]:
    def _one_side_table(corners: List[Dict[str, Any]]) -> pd.DataFrame:
        total_by_player = Counter()
        zone_counts = defaultdict(Counter)
        cross_attempts = defaultdict(int)
        cross_successes = defaultdict(int)
        display_by_key: Dict[Any, str] = {}

        for e in corners:
            raw_name = (e.get("playerName") or "").strip()
            if not raw_name:
                continue

            pid = _safe_int(e.get("playerId"), default=-1)
            key = pid if pid > 0 else raw_name

            display = _display_player_with_jersey(e, raw_name)
            if key not in display_by_key or (display_by_key[key] == raw_name and display != raw_name):
                display_by_key[key] = display

            z = e.get("zone", "Unassigned") or "Unassigned"
            total_by_player[key] += 1
            zone_counts[key][z] += 1

            if z not in ("Short_Corner_Zone", "Unassigned"):
                cross_attempts[key] += 1
                if e.get("resultName") == "SUCCESSFUL":
                    cross_successes[key] += 1

        rows = []
        for key, total in total_by_player.items():
            if total < min_corners:
                continue

            attempts = cross_attempts[key]
            succ = cross_successes[key]
            rate = round((succ / attempts) * 100.0, 1) if attempts > 0 else np.nan

            counts = zone_counts[key]
            valid_zones = [(z, c) for z, c in counts.items() if z != "Unassigned"]
            valid_zones.sort(key=lambda x: (-x[1], x[0]))

            def fmt_zone(idx: int) -> str:
                if idx >= len(valid_zones):
                    return "-"
                zn, cnt = valid_zones[idx]
                pct = round((cnt / total) * 100.0, 0)
                display_zn = "Short" if zn == "Short_Corner_Zone" else zn
                return f"{display_zn} ({int(pct)}%)"

            rows.append(
                {
                    "Player": display_by_key.get(key, str(key)),
                    "Corners": int(total),
                    "Cross Succ. %": f"{rate}%" if not pd.isna(rate) else "-",
                    "1st Choice": fmt_zone(0),
                    "2nd Choice": fmt_zone(1),
                    "3rd Choice": fmt_zone(2),
                }
            )

        df = pd.DataFrame(rows)
        return df if df.empty else df.sort_values(["Corners"], ascending=[False]).reset_index(drop=True)

    return {"left": _one_side_table(left_corners), "right": _one_side_table(right_corners)}


def _calc_attacking_shot_stats(corners: List[Dict[str, Any]]):
    total_by_zone = Counter()
    shot_by_zone = Counter()

    for c in corners:
        zone = c.get("zone")
        if not _valid_zone_for_attacking_shots(zone):
            continue
        total_by_zone[zone] += 1
        if c.get("seq_has_shot") is True:
            shot_by_zone[zone] += 1

    pct_by_zone: Dict[str, float] = {}
    for z, tot in total_by_zone.items():
        shots = shot_by_zone.get(z, 0)
        pct_by_zone[z] = (shots / tot) * 100.0 if tot > 0 else 0.0

    return total_by_zone, shot_by_zone, pct_by_zone

# ============================================================
# file: opp_analysis_new.py
# replace ENTIRE process_corner_data with this version
# ============================================================
def process_corner_data(
    json_data: Dict[str, Any],
    selected_team_name: str,
    *,
    shot_map: Optional[Dict[Tuple[str, str], bool]] = None,
) -> Dict[str, Any]:
    matches = json_data.get("matches", [])
    opponent_left_side, opponent_right_side = [], []
    own_left_side, own_right_side = [], []
    opponent_seq_with_shot: List[List[Dict[str, Any]]] = []
    used_match_rows: List[Dict[str, Any]] = []

    used_matches = 0
    _seen_seq_keys = set()

    # normalize shot_map keys to (match_id:str, seqId_norm:str)
    shot_map = shot_map or {}

    for match in matches:
        events = match.get("corner_events", [])
        if not events:
            continue

        match_teams = set()
        for ev in events:
            c = get_canonical_team(ev.get("teamName"))
            if c:
                match_teams.add(c)

        if selected_team_name not in match_teams:
            continue

        used_match_rows.append(
            {
                "match_id": match.get("match_id"),
                "match_name": match.get("match_name"),
                "match_date": match.get("match_date"),
            }
        )
        used_matches += 1

        TOP_X, BOTTOM_X, LEFT_Y, RIGHT_Y = _resolve_pitch_bounds(match, events)
        zones_TL, zones_BR, zones_TR, zones_BL = build_zones(TOP_X, BOTTOM_X, LEFT_Y, RIGHT_Y)

        sequences_by_id = defaultdict(list)
        for ev in events:
            if ev.get("sequenceId") is not None:
                sequences_by_id[ev["sequenceId"]].append(ev)

        match_id = str(match.get("match_id") or "")

        for e in events:
            if not _is_true_corner_start(e):
                continue

            raw_team = e.get("teamName", "")
            is_own = get_canonical_team(raw_team) == selected_team_name

            sx, sy, ex, ey = e.get("startPosXM"), e.get("startPosYM"), e.get("endPosXM"), e.get("endPosYM")
            if None in (sx, sy, ex, ey):
                continue

            corner_type = get_corner(float(sx), float(sy), TOP_X, LEFT_Y)
            if not corner_type:
                continue

            if corner_type in ("top_left", "bottom_right"):
                local_zones, e_side = (zones_TL if corner_type == "top_left" else zones_BR), "left"
            else:
                local_zones, e_side = (zones_TR if corner_type == "top_right" else zones_BL), "right"

            zone_end = _assign_zone(float(ex), float(ey), local_zones) or _assign_zone(
                -float(ex), -float(ey), local_zones
            )
            e["zone"], e["corner_side"] = zone_end, e_side

            match_id = str(match.get("match_id") or "").strip()
            
            # --- always define seq_id for downstream grouping ---
            seq_id = e.get("sequenceId")
            
            # ✅ Prefer corner_sequence_id for "corner -> shot"
            corner_seq = e.get("corner_sequence_id")
            if corner_seq is None:
                corner_seq = e.get("corner_sequenceId")
            if corner_seq is None:
                corner_seq = seq_id
            
            corner_seq_s = str(corner_seq).strip() if corner_seq is not None else ""
            
            # Source of truth: full_sequences shot_map (match_id, corner_sequence_id)
            if shot_map and match_id and corner_seq_s:
                seq_has_shot = bool(shot_map.get((match_id, corner_seq_s), False))
            else:
                # Fallback: detect shot inside same sequenceId in corner_events
                seq_has_shot = _sequence_has_shot(sequences_by_id.get(seq_id, [])) if (seq_id is not None) else False
            
            e["seq_has_shot"] = bool(seq_has_shot)
            
            # append corners to left/right lists (your existing code)
            if e_side == "left":
                (own_left_side if is_own else opponent_left_side).append(e)
            else:
                (own_right_side if is_own else opponent_right_side).append(e)
            
            # --- use seq_id safely now ---
            if seq_id is not None:
                seq_evs = sequences_by_id.get(seq_id, [])
                if not seq_evs:
                    continue
            
                key = (match_id, corner_seq_s, is_own)
                if key not in _seen_seq_keys:
                    _seen_seq_keys.add(key)
            
                    for sev in seq_evs:
                        sev["zone"], sev["corner_side"] = zone_end, e_side
            
                    if (not is_own) and seq_has_shot and _valid_zone_for_shot_lists(zone_end):
                        opponent_seq_with_shot.append(seq_evs)     
                        
    def _calc_defensive_stats(opp_corners: List[Dict[str, Any]], opp_shot_seqs: List[List[Dict[str, Any]]], side_filter: str):
        total_zone = Counter([c["zone"] for c in opp_corners if _valid_zone_for_shot_lists(c.get("zone"))])
        shot_seq_ids = defaultdict(set)
        for seq in opp_shot_seqs:
            start = next((x for x in seq if x.get("sequenceStart")), seq[0])
            if start.get("corner_side") == side_filter:
                shot_seq_ids[start.get("zone")].add(start.get("sequenceId"))
        pct = {z: (len(shot_seq_ids[z]) / t) * 100 for z, t in total_zone.items() if t > 0}
        return total_zone, shot_seq_ids, pct

    def_left_tot, def_left_ids, def_left_pct = _calc_defensive_stats(opponent_left_side, opponent_seq_with_shot, "left")
    def_right_tot, def_right_ids, def_right_pct = _calc_defensive_stats(opponent_right_side, opponent_seq_with_shot, "right")

    def _zone_pcts(corners: List[Dict[str, Any]]):
        c = Counter(e["zone"] for e in corners if e.get("zone"))
        tot = sum(c.values())
        return {k: (v / tot) * 100 for k, v in c.items()} if tot else {}

    taker_tables = _build_corner_taker_tables(own_left_side, own_right_side, min_corners=5)
    att_shots_left = _calc_attacking_shot_stats(own_left_side)
    att_shots_right = _calc_attacking_shot_stats(own_right_side)

    return {
        "used_matches": used_matches,
        "own_left_count": len(own_left_side),
        "own_right_count": len(own_right_side),
        "def_left_count": len(opponent_left_side),
        "def_right_count": len(opponent_right_side),
        "defensive": {"left": (def_left_tot, def_left_ids, def_left_pct), "right": (def_right_tot, def_right_ids, def_right_pct)},
        "attacking": {"left_pct": _zone_pcts(own_left_side), "right_pct": _zone_pcts(own_right_side)},
        "attacking_shots": {"left": att_shots_left, "right": att_shots_right},
        "tables": taker_tables,
        "used_matches_table": pd.DataFrame(used_match_rows),
    }

def compute_league_attacking_corner_shot_rates(
    json_data: Dict[str, Any],
    *,
    shot_map: Optional[Dict[Tuple[str, str], bool]] = None,
) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    matches = json_data.get("matches", [])
    stats: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    seen_corner_keys = set()

    def _ensure_bucket(team: str, side: str, zone: str) -> Dict[str, float]:
        stats.setdefault(team, {})
        stats[team].setdefault(side, {})
        stats[team][side].setdefault(zone, {"shots": 0, "total": 0, "pct": 0.0})
        return stats[team][side][zone]

    for match in matches:
        events = match.get("corner_events", []) or []
        if not events:
            continue

        match_id = str(match.get("match_id", "") or "")

        TOP_X, BOTTOM_X, LEFT_Y, RIGHT_Y = _resolve_pitch_bounds(match, events)
        zones_TL, zones_BR, zones_TR, zones_BL = build_zones(TOP_X, BOTTOM_X, LEFT_Y, RIGHT_Y)

        sequences_by_id = defaultdict(list)
        for ev in events:
            sid = _norm_seq_id(ev.get("sequenceId"))
            if sid is not None:
                sequences_by_id[sid].append(ev)

        for e in events:
            if not _is_true_corner_start(e):
                continue

            team = get_canonical_team(e.get("teamName"))
            if not team:
                continue

            sx, sy, ex, ey = e.get("startPosXM"), e.get("startPosYM"), e.get("endPosXM"), e.get("endPosYM")
            if None in (sx, sy, ex, ey):
                continue

            corner_type = get_corner(float(sx), float(sy), TOP_X, LEFT_Y)
            if not corner_type:
                continue

            if corner_type in ("top_left", "bottom_right"):
                local_zones, side = (zones_TL if corner_type == "top_left" else zones_BR), "left"
            else:
                local_zones, side = (zones_TR if corner_type == "top_right" else zones_BL), "right"

            zone_end = _assign_zone(float(ex), float(ey), local_zones) or _assign_zone(-float(ex), -float(ey), local_zones)
            if not _valid_zone_for_attacking_shots(zone_end):
                continue

            seq_id = _norm_seq_id(e.get("sequenceId"))
            corner_key = (match_id, team, seq_id)
            if corner_key in seen_corner_keys:
                continue
            seen_corner_keys.add(corner_key)

            if shot_map is not None and match_id and seq_id:
                seq_has_shot = bool(shot_map.get((match_id, seq_id), False))
            else:
                seq_events = sequences_by_id.get(seq_id, []) if seq_id is not None else []
                seq_has_shot = _sequence_has_shot(seq_events) if seq_events else _sequence_has_shot([e])

            bucket = _ensure_bucket(team, side, zone_end)
            bucket["total"] = int(bucket["total"]) + 1
            if seq_has_shot:
                bucket["shots"] = int(bucket["shots"]) + 1

    for team, side_map in stats.items():
        for side, zone_map in side_map.items():
            for zone, d in zone_map.items():
                total = int(d.get("total", 0))
                shots = int(d.get("shots", 0))
                d["total"] = total
                d["shots"] = shots
                d["pct"] = (shots / total) * 100.0 if total > 0 else 0.0

    return stats


def _percentile_rank(values: List[float], value: float) -> Optional[int]:
    if not values:
        return None
    n = len(values)
    leq = sum(1 for v in values if v <= value)
    return int(round((leq / n) * 100))


def build_percentiles_for_team(
    league_stats: Dict[str, Any],
    team_name: str,
    side: str,
    min_zone_corners: int = 4,
) -> Dict[str, int]:
    team_side = league_stats.get(team_name, {}).get(side, {})
    percentiles: Dict[str, int] = {}

    for zone, d in team_side.items():
        team_total = int(d.get("total", 0))
        if team_total < min_zone_corners:
            continue

        team_pct = float(d.get("pct", 0.0))

        dist: List[float] = []
        for _, side_map in league_stats.items():
            zd = side_map.get(side, {}).get(zone)
            if not zd:
                continue
            if int(zd.get("total", 0)) < min_zone_corners:
                continue
            dist.append(float(zd.get("pct", 0.0)))

        p = _percentile_rank(dist, team_pct)
        if isinstance(p, int):
            percentiles[zone] = p

    return percentiles


def plot_shots_defensive(
    img_file: Optional[str],
    polygons: Dict[str, List[Tuple[float, float]]],
    shot_pct: Dict[str, float],
    total_by_zone: Counter,
    shot_seqids_by_zone: Dict[str, set],
):
    fig, ax = plt.subplots(figsize=(14, 10))
    bg = _load_bg(img_file)
    ax.imshow(bg, aspect="auto")
    ax.set_aspect("auto")
    ax.set_anchor("C")
    vals = list(shot_pct.values())
    norm = plt.Normalize(min(vals) if vals else 0, max(vals) if vals else 1)
    cmap = cm.get_cmap("Reds")

    for zone, poly in polygons.items():
        if zone in shot_pct:
            ax.add_patch(Polygon(poly, closed=True, facecolor=cmap(norm(shot_pct[zone])), edgecolor="none", alpha=0.55))
            xs, ys = [p[0] for p in poly], [p[1] for p in poly]
            cx, cy = (min(xs) + max(xs)) / 2 - 5, (min(ys) + max(ys)) / 2
            ax.text(
                cx,
                cy,
                f"{len(shot_seqids_by_zone.get(zone, set()))}/{total_by_zone.get(zone, 0)}",
                fontsize=22,
                color="white",
                weight="bold",
                ha="center",
                va="center",
                path_effects=[PathEffects.withStroke(linewidth=3, foreground="black")],
            )
    return fig


def plot_percent_attacking(
    img_file: Optional[str],
    polygons: Dict[str, List[Tuple[float, float]]],
    centers: Dict[str, Tuple[float, float]],
    pct_by_zone: Dict[str, float],
):
    fig, ax = plt.subplots(figsize=(14, 10))
    bg = _load_bg(img_file)
    ax.imshow(bg, aspect="auto")
    ax.set_aspect("auto")
    ax.set_anchor("C")
    vals = list(pct_by_zone.values())
    norm = plt.Normalize(min(vals) if vals else 0, max(vals) if vals else 1)
    cmap = cm.get_cmap("Reds")

    for zone, poly in polygons.items():
        if zone in pct_by_zone:
            ax.add_patch(Polygon(poly, closed=True, facecolor=cmap(norm(pct_by_zone[zone])), edgecolor="none", alpha=0.55))

    for zone, (x, y) in centers.items():
        if zone in pct_by_zone:
            ax.text(
                x,
                y,
                f"{pct_by_zone[zone]:.1f}%",
                fontsize=22,
                color="white",
                weight="bold",
                ha="center",
                va="center",
                path_effects=[PathEffects.withStroke(linewidth=3, foreground="black")],
            )
    return fig


def plot_shots_attacking_with_percentile(
    img_file: Optional[str],
    polygons: Dict[str, List[Tuple[float, float]]],
    shot_pct_by_zone: Dict[str, float],
    total_by_zone: Counter,
    shot_by_zone: Counter,
    percentile_by_zone: Dict[str, Optional[int]],
    *,
    min_zone_corners: int = 4,
    font_size: int = 18,
):
    fig, ax = plt.subplots(figsize=(14, 10))
    bg = _load_bg(img_file)
    ax.imshow(bg, aspect="auto")
    ax.set_aspect("auto")
    ax.set_anchor("C")

    eligible_vals = [
        float(shot_pct_by_zone[z])
        for z in shot_pct_by_zone
        if int(total_by_zone.get(z, 0)) >= min_zone_corners
    ]
    norm = plt.Normalize(min(eligible_vals) if eligible_vals else 0, max(eligible_vals) if eligible_vals else 1)
    cmap = cm.get_cmap("Reds")

    for zone, poly in polygons.items():
        tot = int(total_by_zone.get(zone, 0))
        if tot < min_zone_corners or zone not in shot_pct_by_zone:
            continue

        ax.add_patch(Polygon(poly, closed=True, facecolor=cmap(norm(float(shot_pct_by_zone[zone]))), edgecolor="none", alpha=0.55))

        xs, ys = [p[0] for p in poly], [p[1] for p in poly]
        cx, cy = (min(xs) + max(xs)) / 2 - 5, (min(ys) + max(ys)) / 2

        shots = int(shot_by_zone.get(zone, 0))
        p = percentile_by_zone.get(zone)
        p_line = f"Top {int(p)}%" if isinstance(p, int) else "Top -%"
        txt = f"{shots} / {tot}\n{p_line}\nKKD"

        ax.text(
            cx,
            cy,
            txt,
            fontsize=font_size,
            color="white",
            weight="bold",
            ha="center",
            va="center",
            path_effects=[PathEffects.withStroke(linewidth=3, foreground="black")],
        )
    return fig


def get_visualization_coords():
    def_L = {
        "Front_Zone": [(265, 15), (644, 15), (644, 600), (265, 600)],
        "Back_Zone": [(1287, 15), (1667, 15), (1667, 600), (1287, 600)],
        "Short_Corner_Zone": [(10, 15), (260, 15), (260, 605), (10, 605)],
        "GA1": [(655, 14), (865, 14), (865, 207), (655, 207)],
        "GA2": [(867, 14), (1080, 14), (1080, 207), (867, 207)],
        "GA3": [(1080, 14), (1275, 14), (1275, 207), (1080, 207)],
        "CA1": [(650, 212), (865, 212), (865, 397), (650, 397)],
        "CA2": [(867, 212), (1080, 212), (1080, 397), (867, 397)],
        "CA3": [(1080, 212), (1284, 212), (1284, 397), (1080, 397)],
        "Edge_Zone": [(650, 401), (1284, 401), (1284, 776), (650, 776)],
    }
    def_R = {
        "Front_Zone": def_L["Back_Zone"],
        "Back_Zone": def_L["Front_Zone"],
        "Short_Corner_Zone": [(1672, 15), (1922, 15), (1922, 605), (1672, 605)],
        "GA1": def_L["GA3"],
        "GA2": def_L["GA2"],
        "GA3": def_L["GA1"],
        "CA1": def_L["CA3"],
        "CA2": def_L["CA2"],
        "CA3": def_L["CA1"],
        "Edge_Zone": def_L["Edge_Zone"],
    }
    att_L = {
        "Front_Zone": [(218, 135), (508, 135), (508, 575), (218, 575)],
        "Back_Zone": [(970, 135), (1260, 135), (1260, 575), (970, 575)],
        "Short_Corner_Zone": [(30, 575), (30, 135), (218, 135), (218, 575)],
        "GA1": [(505, 135), (660, 135), (660, 280), (505, 280)],
        "GA2": [(660, 135), (822, 135), (822, 280), (660, 280)],
        "GA3": [(822, 135), (970, 135), (970, 280), (822, 280)],
        "CA1": [(505, 285), (660, 285), (660, 425), (505, 425)],
        "CA2": [(664, 285), (822, 285), (822, 425), (664, 425)],
        "CA3": [(972, 285), (822, 285), (822, 425), (972, 425)],
        "Edge_Zone": [(505, 425), (970, 425), (970, 700), (505, 700)],
    }
    att_centers_L = {
        "GA1": (590, 245),
        "GA2": (745, 250),
        "GA3": (900, 250),
        "CA1": (590, 400),
        "CA2": (745, 400),
        "CA3": (900, 400),
        "Edge_Zone": (745, 520),
        "Front_Zone": (350, 380),
        "Back_Zone": (1127, 380),
        "Short_Corner_Zone": (130, 380),
    }
    att_R = {
        "Back_Zone": [(218, 130), (508, 130), (508, 560), (218, 560)],
        "Front_Zone": [(965, 130), (1260, 130), (1260, 560), (965, 560)],
        "Short_Corner_Zone": [(1260, 130), (1430, 130), (1430, 560), (1260, 560)],
        "GA3": [(510, 130), (660, 130), (660, 280), (510, 280)],
        "GA2": [(660, 130), (812, 130), (812, 280), (660, 280)],
        "GA1": [(812, 130), (960, 130), (960, 280), (812, 280)],
        "CA3": [(505, 284), (660, 284), (660, 425), (505, 425)],
        "CA2": [(660, 285), (812, 285), (812, 425), (660, 425)],
        "CA1": [(960, 285), (812, 285), (812, 425), (960, 425)],
        "Edge_Zone": [(505, 425), (960, 425), (960, 690), (505, 690)],
    }
    att_centers_R = {
        "GA3": (590, 250),
        "GA2": (745, 250),
        "GA1": (900, 250),
        "CA3": (590, 400),
        "CA2": (745, 400),
        "CA1": (900, 400),
        "Edge_Zone": (745, 520),
        "Back_Zone": (350, 380),
        "Front_Zone": (1120, 380),
        "Short_Corner_Zone": (1355, 400),
    }
    return {
        "def_L": def_L,
        "def_R": def_R,
        "att_L": att_L,
        "att_centers_L": att_centers_L,
        "att_R": att_R,
        "att_centers_R": att_centers_R,
    }


def load_corner_positions_headers(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"corner_positions_headers.csv not found: {csv_path}")
    return pd.read_csv(csv_path, low_memory=False).where(pd.notnull, None)


def attach_actual_club_from_events(headers_df: pd.DataFrame, events_seq_df: pd.DataFrame) -> pd.DataFrame:
    if headers_df is None or headers_df.empty:
        return headers_df
    if events_seq_df is None or events_seq_df.empty:
        return headers_df

    ev = events_seq_df.copy()

    if "match_id" not in ev.columns:
        if "source_event_file" in ev.columns:
            ev["match_id"] = ev["source_event_file"].apply(_match_id_from_path)
        else:
            return headers_df

    if "groupName" not in ev.columns or "teamName" not in ev.columns:
        return headers_df

    def _norm_group(x: Any) -> Optional[str]:
        if not isinstance(x, str):
            return None
        s = x.strip().upper()
        if s in {"HOME", "H"}:
            return "HOME"
        if s in {"AWAY", "A"}:
            return "AWAY"
        return None

    ev["__group"] = ev["groupName"].apply(_norm_group)
    ev = ev.dropna(subset=["match_id", "__group", "teamName"]).copy()

    map_rows = (
        ev.groupby(["match_id", "__group"])["teamName"]
        .agg(lambda s: s.value_counts().index[0])
        .reset_index()
        .rename(columns={"__group": "club", "teamName": "club_actual"})
    )

    out = headers_df.copy()
    if "match_id" not in out.columns or "club" not in out.columns:
        return out

    out["match_id"] = out["match_id"].astype(str)
    map_rows["match_id"] = map_rows["match_id"].astype(str)

    out = out.merge(map_rows, on=["match_id", "club"], how="left")
    out["club_actual_canon"] = out["club_actual"].apply(_canon_team)
    return out


@dataclass(frozen=True)
class VizPaths:
    data_path: str
    def_L: Optional[str] = None
    def_R: Optional[str] = None
    att_L: Optional[str] = None
    att_R: Optional[str] = None

    def img(self, key: str) -> Optional[str]:
        p = getattr(self, key, None)
        return p if p and os.path.exists(p) else None
