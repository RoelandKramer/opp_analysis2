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


# ============================================================
# 0) UTILITIES
# ============================================================

_MATCH_ID_FROM_NAME = re.compile(r"-\s*(\d+)\s*(?:\(\d+\))?\s*$", re.IGNORECASE)
_DATE_YYYYMMDD = re.compile(r"^(?P<d>\d{8})\s+")
_DATE_DDMMYYYY = re.compile(r"^(?P<d>\d{2}-\d{2}-\d{4})")

HEADER_BODY_PART_NAMES = {"HEAD"}

def truthy(x: Any) -> bool:
    if x is True:
        return True
    if isinstance(x, (int, float)) and x == 1:
        return True
    if isinstance(x, str) and x.strip().lower() in {"true", "1", "yes"}:
        return True
    return False

def _is_true_corner_start(ev: Dict[str, Any]) -> bool:
    # IMPORTANT: sequenceStart can be 'TRUE'/'FALSE' strings in CSV-derived dicts
    return ev.get("possessionTypeName") == "CORNER" and truthy(ev.get("sequenceStart"))

def _safe_int(x: Any, default: int = -1) -> int:
    try:
        return int(x)
    except Exception:
        return default

def filter_last_n_matches(json_data: Dict[str, Any], n_last: Optional[int]) -> Dict[str, Any]:
    """
    Return json_data with only the last n_last matches (by match_date).
    If n_last is None or >= number of dated matches -> returns original json_data.
    Matches without a date are kept at the end (older).
    """
    matches = list(json_data.get("matches", []) or [])
    if not matches:
        return {"matches": []}

    def _dt(m):
        dt = m.get("match_date")
        return dt if isinstance(dt, datetime) else None

    dated = [m for m in matches if _dt(m) is not None]
    undated = [m for m in matches if _dt(m) is None]

    dated.sort(key=lambda m: _dt(m))  # oldest -> newest

    ordered = dated + undated  # undated treated as oldest-ish
    if not n_last or n_last >= len(ordered):
        return {"matches": ordered}

    return {"matches": ordered[-int(n_last):]}

def filter_last_n_matches_for_team(
    json_data: Dict[str, Any],
    selected_team_name: str,
    n_last: Optional[int],
) -> Dict[str, Any]:
    """
    Keep only matches where selected_team_name appears (based on teamName in corner_events),
    then take the last n_last matches by match_date (undated go last/older).
    If n_last is None or >= available -> keep all team matches.
    """
    matches = list(json_data.get("matches", []) or [])
    if not matches:
        return {"matches": []}

    # 1) keep only matches where team appears
    team_matches: List[Dict[str, Any]] = []
    for m in matches:
        events = m.get("corner_events", []) or []
        match_teams = set()
        for ev in events:
            c = get_canonical_team(ev.get("teamName"))
            if c:
                match_teams.add(c)
        if selected_team_name in match_teams:
            team_matches.append(m)

    if not team_matches:
        return {"matches": []}

    # 2) order by match_date (oldest -> newest), undated treated as oldest
    def _dt(mm):
        dt = mm.get("match_date")
        return dt if isinstance(dt, datetime) else None

    dated = [m for m in team_matches if _dt(m) is not None]
    undated = [m for m in team_matches if _dt(m) is None]
    dated.sort(key=lambda m: _dt(m))
    ordered = dated + undated

    # 3) slice last N
    if not n_last or n_last >= len(ordered):
        return {"matches": ordered}
    return {"matches": ordered[-int(n_last):]}
    


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if v != v:
            return None
        return v
    except Exception:
        return None


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


# ============================================================
# 1) TEAM NORMALIZATION
# ============================================================

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
    # aliases
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

def debug_used_matches(json_data: Dict[str, Any], selected_team_name: str) -> pd.DataFrame:
    """
    Debug why matches are (not) counted in process_corner_data for a team.
    Returns one row per match with counts and the first "reason" it would be skipped.
    """
    rows: List[Dict[str, Any]] = []

    for match in json_data.get("matches", []) or []:
        mid = str(match.get("match_id", ""))
        mname = match.get("match_name")
        events = match.get("corner_events", []) or []

        if not events:
            rows.append({
                "match_id": mid,
                "match_name": mname,
                "events": 0,
                "teams_in_match": 0,
                "team_present": False,
                "corner_starts_found": 0,
                "reason": "no_events",
            })
            continue

        # teams found in match
        match_teams = set()
        for ev in events:
            c = get_canonical_team(ev.get("teamName"))
            if c:
                match_teams.add(c)

        team_present = selected_team_name in match_teams
        if not team_present:
            rows.append({
                "match_id": mid,
                "match_name": mname,
                "events": len(events),
                "teams_in_match": len(match_teams),
                "team_present": False,
                "corner_starts_found": 0,
                "reason": "team_not_in_match_teams",
            })
            continue

        # count "true corner starts" as your process_corner_data would
        corner_starts = 0
        corner_starts_truthy = 0

        for e in events:
            # useful to see if sequenceStart formatting is the problem
            if e.get("possessionTypeName") == "CORNER":
                corner_starts += 1
                if _is_true_corner_start(e):
                    corner_starts_truthy += 1

        reason = "ok"
        if corner_starts_truthy == 0:
            reason = "no_true_corner_starts"

        rows.append({
            "match_id": mid,
            "match_name": mname,
            "events": len(events),
            "teams_in_match": len(match_teams),
            "team_present": True,
            "corner_events_found": corner_starts,
            "corner_starts_found": corner_starts_truthy,
            "example_sequenceStart_values": ",".join(
                sorted({str(e.get("sequenceStart")) for e in events if e.get("possessionTypeName") == "CORNER"}))[:200]
            ,
            "reason": reason,
        })

    return pd.DataFrame(rows)


# ============================================================
# 2) CSV LOADERS (json-like wrapper)
# ============================================================

def load_corner_events_csv_as_jsonlike(corner_events_csv_path: str) -> Dict[str, Any]:
    """
    Loads corner_events_all_matches.csv and returns:
      {"matches":[{"match_id","match_name","match_date","pitch_top_x","pitch_left_y","corner_events":[...]},...]}
    Groups by match_id only.
    """
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


def extract_all_teams(json_data: Dict[str, Any]) -> List[str]:
    teams: set[str] = set()
    for m in json_data.get("matches", []):
        for ev in m.get("corner_events", []) or []:
            t = get_canonical_team(ev.get("teamName"))
            if t:
                teams.add(t)
    return sorted(teams)
def attach_actual_club_from_events(
    headers_df: pd.DataFrame,
    events_seq_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    headers_df has club = HOME/AWAY. This function attaches:
      - club_actual (real team name)
      - club_actual_canon (canonical team name)
    so you can filter by selected team.
    """
    if headers_df is None or headers_df.empty:
        return headers_df

    if events_seq_df is None or events_seq_df.empty:
        return headers_df

    ev = events_seq_df.copy()

    # Ensure we have match_id to join on
    if "match_id" not in ev.columns:
        if "source_event_file" in ev.columns:
            ev["match_id"] = ev["source_event_file"].apply(_match_id_from_path)
        else:
            return headers_df

    if "groupName" not in ev.columns or "teamName" not in ev.columns:
        return headers_df

    # Normalize groupName -> HOME/AWAY
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

    # Build match_id -> {HOME: teamName, AWAY: teamName}
    # If multiple rows exist, we take the most common teamName per group in that match
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


# ============================================================
# 3) FIELD + ZONES + SEQUENCE HELPERS (visualizations)
# ============================================================

def _iter_numbers(x: Any) -> Iterable[float]:
    if isinstance(x, (int, float)) and math.isfinite(float(x)):
        yield float(x)


def _round_up_to_step(value: float, step: float) -> float:
    return math.ceil(value / step) * step


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
    return any(ev.get("baseTypeName") == "SHOT" or _safe_int(ev.get("baseTypeId"), -1) == 6 for ev in seq)


def _valid_zone_for_shot_lists(zone_val: Any) -> bool:
    return bool(zone_val and str(zone_val).strip() and zone_val != "Short_Corner_Zone")


def _valid_zone_for_attacking_shots(zone_val: Any) -> bool:
    return bool(zone_val and str(zone_val).strip() and zone_val != "Unassigned")


# ============================================================
# 4) ATT/DEF PROCESSING + TABLES (corner takers)
# ============================================================

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


def process_corner_data(json_data: Dict[str, Any], selected_team_name: str) -> Dict[str, Any]:
    matches = json_data.get("matches", [])
    opponent_left_side, opponent_right_side = [], []
    own_left_side, own_right_side = [], []
    opponent_seq_with_shot: List[List[Dict[str, Any]]] = []
    used_match_rows: List[Dict[str, Any]] = []

    used_matches = 0
    _seen_seq_keys = set()

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
        used_match_rows.append({
                "match_id": match.get("match_id"),
                "match_name": match.get("match_name"),
                "match_date": match.get("match_date"),
            })
        used_matches += 1
        TOP_X, BOTTOM_X, LEFT_Y, RIGHT_Y = _resolve_pitch_bounds(match, events)
        zones_TL, zones_BR, zones_TR, zones_BL = build_zones(TOP_X, BOTTOM_X, LEFT_Y, RIGHT_Y)
        sequences_by_id = defaultdict(list)

        for ev in events:
            if ev.get("sequenceId") is not None:
                sequences_by_id[ev["sequenceId"]].append(ev)

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

            zone_end = _assign_zone(float(ex), float(ey), local_zones) or _assign_zone(-float(ex), -float(ey), local_zones)
            e["zone"], e["corner_side"] = zone_end, e_side
        
            seq_id = e.get("sequenceId")
            seq_has_shot = _sequence_has_shot(sequences_by_id.get(seq_id, [])) if (seq_id is not None) else False
            e["seq_has_shot"] = bool(seq_has_shot)

            if e_side == "left":
                (own_left_side if is_own else opponent_left_side).append(e)
            else:
                (own_right_side if is_own else opponent_right_side).append(e)

            if seq_id is not None:
                seq_evs = sequences_by_id.get(seq_id, [])
                if not seq_evs:
                    continue
                key = (match.get("match_id"), seq_id, is_own)
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


# ============================================================
# 5) LEAGUE PERCENTILES (visualization)
# ============================================================

def compute_league_attacking_corner_shot_rates(json_data: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    matches = json_data.get("matches", [])
    stats: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    seen_corner_keys = set()

    def _ensure_bucket(team: str, side: str, zone: str) -> Dict[str, float]:
        stats.setdefault(team, {})
        stats[team].setdefault(side, {})
        stats[team][side].setdefault(zone, {"shots": 0, "total": 0, "pct": 0.0})
        return stats[team][side][zone]

    for match in matches:
        events = match.get("corner_events", [])
        if not events:
            continue

        TOP_X, BOTTOM_X, LEFT_Y, RIGHT_Y = _resolve_pitch_bounds(match, events)
        zones_TL, zones_BR, zones_TR, zones_BL = build_zones(TOP_X, BOTTOM_X, LEFT_Y, RIGHT_Y)

        sequences_by_id = defaultdict(list)
        for ev in events:
            if ev.get("sequenceId") is not None:
                sequences_by_id[ev["sequenceId"]].append(ev)

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

            seq_id = e.get("sequenceId")
            match_id = match.get("match_id")
            corner_key = (match_id, team, seq_id)
            if corner_key in seen_corner_keys:
                continue
            seen_corner_keys.add(corner_key)

            seq_has_shot = _sequence_has_shot(sequences_by_id.get(seq_id, [])) if (seq_id is not None) else False

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


# ============================================================
# 6) PLOTTING (Streamlit-ready)
# ============================================================

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


# ============================================================
# 7) STATIC COORDS FOR PNG OVERLAYS
# ============================================================

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


# ============================================================
# 8) PLAYER TABLES (events + positions)
# ============================================================

def _frame_players(frame: dict, side: str) -> List[dict]:
    arr = frame.get(side)
    if not isinstance(arr, list):
        return []
    return [p for p in arr if isinstance(p, dict) and "p" in p and "x" in p and "y" in p]


def _player_in_side(frame: dict, side: str, pid: int) -> bool:
    for p in _frame_players(frame, side):
        if _safe_int(p.get("p"), -1) == pid:
            return True
    return False


def _nearest_time(target: int, times: np.ndarray) -> Optional[int]:
    if times.size == 0:
        return None
    i = int(np.searchsorted(times, target))
    if i <= 0:
        return int(times[0])
    if i >= len(times):
        return int(times[-1])
    prev_t = int(times[i - 1])
    next_t = int(times[i])
    return prev_t if abs(prev_t - target) <= abs(next_t - target) else next_t


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _closest_defender(
    frame: dict,
    attacker_side: str,
    attacker_pid: int,
    defender_side: str,
    allowed_defender_ids: set[int],
) -> Optional[Tuple[int, Optional[int], float]]:
    attacker_pl = None
    for p in _frame_players(frame, attacker_side):
        if _safe_int(p.get("p"), -1) == attacker_pid:
            attacker_pl = p
            break
    if not attacker_pl:
        return None

    ax, ay = float(attacker_pl["x"]), float(attacker_pl["y"])
    best_pid, best_jersey, best_d = None, None, 1e18

    for p in _frame_players(frame, defender_side):
        pid = _safe_int(p.get("p"), -1)
        if pid < 0 or pid not in allowed_defender_ids:
            continue
        px, py = float(p["x"]), float(p["y"])
        d = _dist((px, py), (ax, ay))
        if d < best_d:
            best_d = d
            best_pid = pid
            s = _safe_int(p.get("s"), -1)
            best_jersey = None if s < 0 else s

    if best_pid is None:
        return None
    return int(best_pid), best_jersey, float(best_d)


def _infer_home_away_from_seq_df(seq_df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    home, away = None, None
    if seq_df.empty or "groupName" not in seq_df.columns:
        return None, None

    for _, r in seq_df.iterrows():
        tn = r.get("teamName")
        gn = r.get("groupName")
        if not isinstance(tn, str) or not tn.strip():
            continue
        if gn is None:
            continue
        gns = str(gn).strip().upper()
        if gns in {"HOME", "H"} and home is None:
            home = tn.strip()
        elif gns in {"AWAY", "A"} and away is None:
            away = tn.strip()
        if home and away:
            break
    return home, away


def _side_for_team(team_name: str, home: Optional[str], away: Optional[str]) -> Optional[str]:
    if isinstance(home, str) and _canon_team(home) == _canon_team(team_name):
        return "h"
    if isinstance(away, str) and _canon_team(away) == _canon_team(team_name):
        return "a"
    return None


def _infer_sides_from_frame(
    *,
    frame: dict,
    corner_taker_id: Optional[int],
    attacking_team: str,
    home: Optional[str],
    away: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    if isinstance(corner_taker_id, int) and corner_taker_id >= 0:
        if _player_in_side(frame, "h", corner_taker_id):
            return "h", "a"
        if _player_in_side(frame, "a", corner_taker_id):
            return "a", "h"

    att_side = _side_for_team(attacking_team, home, away)
    if att_side == "h":
        return "h", "a"
    if att_side == "a":
        return "a", "h"
    return None, None


def _is_shot_row(r: pd.Series) -> bool:
    return (r.get("baseTypeName") == "SHOT") or (_safe_int(r.get("baseTypeId"), -1) == 6)


def _is_goal_row(r: pd.Series) -> bool:
    return r.get("resultName") == "SUCCESSFUL"


def _is_header_shot_row(r: pd.Series) -> bool:
    if not _is_shot_row(r):
        return False
    bp = r.get("bodyPartName")
    return isinstance(bp, str) and bp.strip().upper() in HEADER_BODY_PART_NAMES


def load_positions_samples_for_tables(
    pos_csv: str,
) -> Tuple[pd.DataFrame, Dict[Tuple[str, str], np.ndarray], Dict[Tuple[str, str, int], dict]]:
    """
    Returns:
      pos_df_norm: normalized DF with match_id_from_file, corner_sequence_id, t, frame(dict)
      times_by_ms: (mid, seq) -> sorted times (np.ndarray)
      frame_by_key: (mid, seq, t) -> frame(dict)
    """
    pos = pd.read_csv(pos_csv, low_memory=False).where(pd.notnull, None)

    if "source_positions_file" in pos.columns:
        pos["match_id_from_file"] = pos["source_positions_file"].apply(_match_id_from_path)
    elif "source_event_file" in pos.columns:
        pos["match_id_from_file"] = pos["source_event_file"].apply(_match_id_from_path)
    else:
        raise ValueError("positions CSV needs source_positions_file or source_event_file")

    if "sequence_id" in pos.columns and "corner_sequence_id" not in pos.columns:
        pos = pos.rename(columns={"sequence_id": "corner_sequence_id"})

    required = ["match_id_from_file", "corner_sequence_id", "t", "frame_json"]
    missing = [c for c in required if c not in pos.columns]
    if missing:
        raise ValueError(f"positions CSV missing required columns: {missing}")

    pos["match_id_from_file"] = pos["match_id_from_file"].astype(str)
    pos["corner_sequence_id"] = pos["corner_sequence_id"].astype(str)
    pos["t"] = pd.to_numeric(pos["t"], errors="coerce").astype("Int64")
    pos = pos.dropna(subset=["t"]).copy()
    pos["t"] = pos["t"].astype(int)

    pos["frame"] = pos["frame_json"].apply(_parse_frame_json)
    pos = pos.dropna(subset=["frame"]).copy()

    times_by_ms: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    frame_by_key: Dict[Tuple[str, str, int], dict] = {}

    for _, r in pos.iterrows():
        mid = str(r["match_id_from_file"])
        seq = str(r["corner_sequence_id"])
        t = int(r["t"])
        fr = r["frame"]
        if not isinstance(fr, dict):
            continue
        times_by_ms[(mid, seq)].append(t)
        frame_by_key[(mid, seq, t)] = fr

    times_np: Dict[Tuple[str, str], np.ndarray] = {}
    for k, ts in times_by_ms.items():
        times_np[k] = np.array(sorted(set(ts)), dtype=int)

    return pos, times_np, frame_by_key


def _build_player_name_map(events_df: pd.DataFrame) -> Dict[int, str]:
    m: Dict[int, str] = {}
    if events_df.empty:
        return m
    for _, r in events_df.iterrows():
        pid = _safe_int(r.get("playerId"), -1)
        pn = r.get("playerName")
        if pid >= 0 and isinstance(pn, str) and pn.strip() and pn != "NOT_APPLICABLE":
            if pid not in m:
                m[pid] = pn.strip()
    return m


def _team_player_ids(events_df: pd.DataFrame, team_canon: str) -> set[int]:
    ids: set[int] = set()
    for _, r in events_df.iterrows():
        t = _canon_team(r.get("teamName"))
        if t != team_canon:
            continue
        pid = _safe_int(r.get("playerId"), -1)
        if pid >= 0:
            ids.add(pid)
    return ids


def _gk_player_ids(events_df: pd.DataFrame, team_canon: str) -> set[int]:
    if events_df is None or events_df.empty:
        return set()
    if "positionTypeName" not in events_df.columns:
        return set()
    gk = events_df[
        (events_df["teamName"].apply(_canon_team) == team_canon)
        & (events_df["positionTypeName"] == "GK")
    ]
    return {int(pid) for pid in pd.to_numeric(gk["playerId"], errors="coerce").dropna().astype(int).unique()}


def filter_headers_for_team(headers_df: pd.DataFrame, team: str) -> pd.DataFrame:
    if headers_df is None or headers_df.empty:
        return pd.DataFrame()
    if "club" not in headers_df.columns:
        return pd.DataFrame()
    team_c = _canon_team(team) or team
    df = headers_df.copy()
    df["__club_c"] = df["club"].apply(_canon_team)
    return df[df["__club_c"] == team_c].copy()


# --- PATCH: add jersey number ("s") before player name in both charts ---
# File: wherever these plotting funcs live (e.g. oa.py)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _build_player_label(df: pd.DataFrame, *, number_col: str = "s") -> pd.Series:
    """
    Returns label like '7. Genrich Sillé' when jersey number exists.
    Falls back to 'Genrich Sillé' if number is missing/invalid.
    """
    name = df.get("player_name", pd.Series(["Unknown"] * len(df), index=df.index)).astype(str)

    if number_col not in df.columns:
        return name

    num = pd.to_numeric(df[number_col], errors="coerce")
    num_str = num.apply(lambda x: "" if pd.isna(x) else str(int(x)))
    return np.where(num_str != "", num_str + ". " + name, name)


def plot_defending_corner_players_diverging(
    def_tbl: pd.DataFrame,
    *,
    max_players: int = 15,
    title: str = "Defending corner players",
):
    """
    Works with your dataset columns by mapping:
      - Clearances  <- Defending_corners_defended
      - Faults      <- Defending_corners_errors + Defending_corners_fatal_errors
      - GoalsAllowed<- Defending_corners_fatal_errors

    Also auto-aggregates (sums) over multiple rows per player_name (+ jersey number if present).
    """
    if def_tbl is None or def_tbl.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return fig

    df = def_tbl.copy()

    if "player_name" not in df.columns:
        df["player_name"] = "Unknown"

    if "Clearances" not in df.columns:
        df["Clearances"] = df.get("Defending_corners_defended", 0)

    if "GoalsAllowed" not in df.columns:
        df["GoalsAllowed"] = df.get("Defending_corners_fatal_errors", 0)

    if "Faults" not in df.columns:
        df["Faults"] = df.get("Defending_corners_errors", 0) + df.get(
            "Defending_corners_fatal_errors", 0
        )

    for c in ("Faults", "GoalsAllowed", "Clearances"):
        df[c] = pd.to_numeric(df.get(c, 0), errors="coerce").fillna(0).astype(int)

    group_cols = ["player_name"] + (["s"] if "s" in df.columns else [])
    df = df.groupby(group_cols, as_index=False)[["Faults", "GoalsAllowed", "Clearances"]].sum()

    df["GoalsAllowed"] = np.minimum(df["GoalsAllowed"], df["Faults"])
    df["Faults_non_goal"] = np.maximum(df["Faults"] - df["GoalsAllowed"], 0)

    df["Total"] = df["Faults"] + df["Clearances"]
    df = df.sort_values(["Total", "Faults", "Clearances"], ascending=[False, False, False]).head(
        max_players
    )

    df = df.iloc[::-1].reset_index(drop=True)

    y = np.arange(len(df))
    names = _build_player_label(df, number_col="s").tolist()

    faults_goal = df["GoalsAllowed"].to_numpy()
    faults_non_goal = df["Faults_non_goal"].to_numpy()
    clearances = df["Clearances"].to_numpy()

    fig_h = max(3.5, 0.45 * len(df) + 1.2)
    fig, ax = plt.subplots(figsize=(10, fig_h))

    ax.barh(y, -faults_goal, color="black", label="Player they marked scored from corner")
    ax.barh(y, -faults_non_goal, left=-faults_goal, color="red", label="Player they marked shot from corner")
    ax.barh(y, clearances, color="green", label="Clearances")

    ax.axvline(0, linewidth=1)

    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_title(title)

    left_max = (faults_goal + faults_non_goal).max() if len(df) else 1
    right_max = clearances.max() if len(df) else 1
    lim = max(left_max, right_max, 1)
    ax.set_xlim(-lim * 1.15, lim * 1.15)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for i in range(len(df)):
        f = int(df.loc[i, "Faults"])
        g = int(df.loc[i, "GoalsAllowed"])
        c = int(df.loc[i, "Clearances"])
        if f > 0:
            ax.text(-f - 0.05, y[i], f"{f}", va="center", ha="right", fontsize=9)
        if c > 0:
            ax.text(c + 0.05, y[i], f"{c}", va="center", ha="left", fontsize=9)
        if g > 0:
            ax.text(-g / 2, y[i], f"{g}", va="center", ha="center", fontsize=9, color="white")

    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_attacking_corner_players_headers(
    att_tbl: pd.DataFrame,
    *,
    max_players: int = 15,
    title: str = "Attacking corner players (Headers & Header goals)",
):
    """
    Works with your dataset columns by mapping:
      - Headshots <- Attacking_corners_headed
      - HeadGoals <- Attacking_corners_headed_and_scored

    Also auto-aggregates (sums) over multiple rows per player_name (+ jersey number if present).
    """
    if att_tbl is None or att_tbl.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return fig

    df = att_tbl.copy()

    if "player_name" not in df.columns:
        df["player_name"] = "Unknown"

    if "Headshots" not in df.columns:
        df["Headshots"] = df.get("Attacking_corners_headed", 0)

    if "HeadGoals" not in df.columns:
        df["HeadGoals"] = df.get("Attacking_corners_headed_and_scored", 0)

    for c in ("Headshots", "HeadGoals"):
        df[c] = pd.to_numeric(df.get(c, 0), errors="coerce").fillna(0).astype(int)

    group_cols = ["player_name"] + (["s"] if "s" in df.columns else [])
    df = df.groupby(group_cols, as_index=False)[["Headshots", "HeadGoals"]].sum()

    df["HeadGoals"] = np.minimum(df["HeadGoals"], df["Headshots"])
    df["Headers_non_goal"] = np.maximum(df["Headshots"] - df["HeadGoals"], 0)

    df = df.sort_values(["Headshots", "HeadGoals"], ascending=[False, False]).head(max_players)
    df = df.iloc[::-1].reset_index(drop=True)

    y = np.arange(len(df))
    names = _build_player_label(df, number_col="s").tolist()

    goals = df["HeadGoals"].to_numpy()
    rest = df["Headers_non_goal"].to_numpy()
    total_headers = df["Headshots"].to_numpy()

    fig_h = max(3.5, 0.45 * len(df) + 1.2)
    fig, ax = plt.subplots(figsize=(10, fig_h))

    ax.barh(y, goals, color="darkgreen", label="Header goals")
    ax.barh(y, rest, left=goals, color="lightgreen", label="Headers")

    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_title(title)

    max_x = max(int(total_headers.max()) if len(df) else 1, 1)
    ax.set_xlim(0, max_x * 1.15)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for i in range(len(df)):
        th = int(total_headers[i])
        g = int(goals[i])
        if th > 0:
            ax.text(th + 0.05, y[i], f"{th}", va="center", ha="left", fontsize=9)
        if g > 0:
            ax.text(g / 2, y[i], f"{g}", va="center", ha="center", fontsize=9, color="white")

    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig
def load_corner_positions_headers(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"corner_positions_headers.csv not found: {csv_path}")
    return pd.read_csv(csv_path, low_memory=False).where(pd.notnull, None)

def _to_int_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)

# def build_attacking_headers_player_table(*, team: str, headers_df: pd.DataFrame) -> pd.DataFrame:
#     if headers_df is None or headers_df.empty:
#         return pd.DataFrame(columns=["player_name", "CornersOnPitch", "Headshots", "HeadGoals"])

#     df = headers_df.copy()
#     if "club" not in df.columns:
#         return pd.DataFrame(columns=["player_name", "CornersOnPitch", "Headshots", "HeadGoals"])

#     team_c = _canon_team(team) or team
#     df["__club_c"] = df["club"].apply(_canon_team)
#     df = df[df["__club_c"] == team_c].copy()
#     if df.empty:
#         return pd.DataFrame(columns=["player_name", "CornersOnPitch", "Headshots", "HeadGoals"])

#     for col in ("Attacking_corners_on_pitch", "Attacking_corners_headed", "Attacking_corners_headed_and_scored"):
#         if col not in df.columns:
#             df[col] = 0
#         df[col] = _to_int_series(df[col])

#     if "player_name" not in df.columns:
#         df["player_name"] = df.get("player_id", "Unknown").astype(str)

#     out = (
#         df.groupby("player_name", dropna=False)[
#             ["Attacking_corners_on_pitch", "Attacking_corners_headed", "Attacking_corners_headed_and_scored"]
#         ]
#         .sum()
#         .reset_index()
#         .rename(
#             columns={
#                 "Attacking_corners_on_pitch": "CornersOnPitch",
#                 "Attacking_corners_headed": "Headshots",
#                 "Attacking_corners_headed_and_scored": "HeadGoals",
#             }
#         )
#     )
#     out["HeadGoals"] = np.minimum(out["HeadGoals"], out["Headshots"])
#     return out.sort_values(["CornersOnPitch", "Headshots", "HeadGoals"], ascending=[False, False, False]).reset_index(drop=True)

# def build_defending_headers_player_table(*, team: str, headers_df: pd.DataFrame) -> pd.DataFrame:
#     if headers_df is None or headers_df.empty:
#         return pd.DataFrame(columns=["player_name", "CornersOnPitch", "Faults", "GoalsAllowed", "Clearances"])

#     df = headers_df.copy()
#     if "club" not in df.columns:
#         return pd.DataFrame(columns=["player_name", "CornersOnPitch", "Faults", "GoalsAllowed", "Clearances"])

#     team_c = _canon_team(team) or team
#     df["__club_c"] = df["club"].apply(_canon_team)
#     df = df[df["__club_c"] == team_c].copy()
#     if df.empty:
#         return pd.DataFrame(columns=["player_name", "CornersOnPitch", "Faults", "GoalsAllowed", "Clearances"])

#     for col in ("Defending_corners_on_pitch", "Defending_corners_errors", "Defending_corners_fatal_errors", "Defending_corners_defended"):
#         if col not in df.columns:
#             df[col] = 0
#         df[col] = _to_int_series(df[col])

#     if "player_name" not in df.columns:
#         df["player_name"] = df.get("player_id", "Unknown").astype(str)

#     out = (
#         df.groupby("player_name", dropna=False)[
#             ["Defending_corners_on_pitch", "Defending_corners_errors", "Defending_corners_fatal_errors", "Defending_corners_defended"]
#         ]
#         .sum()
#         .reset_index()
#         .rename(
#             columns={
#                 "Defending_corners_on_pitch": "CornersOnPitch",
#                 "Defending_corners_errors": "Faults",
#                 "Defending_corners_fatal_errors": "GoalsAllowed",
#                 "Defending_corners_defended": "Clearances",
#             }
#         )
#     )
#     out["GoalsAllowed"] = np.minimum(out["GoalsAllowed"], out["Faults"])
#     return out.sort_values(["CornersOnPitch", "Faults", "GoalsAllowed", "Clearances"], ascending=[False, False, False, False]).reset_index(drop=True)

# ============================================================
# 9) OPTIONAL: paths container
# ============================================================

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
