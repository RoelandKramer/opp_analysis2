# ============================================================
# file: opponent_analysis.py
# ============================================================
import json
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patheffects as PathEffects
from matplotlib.patches import Polygon
import matplotlib.image as mpimg
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ==========================================
# 1. CONFIG & TEAM MAPPING
# ==========================================
TEAM_NAME_MAPPING = {
    # --- TARGET NAMES (The clean ones we want) ---
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

    # --- ALIASES (Map these TO the targets above) ---
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
    "NOT_APPLICABLE": None
}

def get_canonical_team(raw_name: Any) -> Optional[str]:
    if not isinstance(raw_name, str):
        return None
    clean_name = raw_name.strip()
    return TEAM_NAME_MAPPING.get(clean_name)

def extract_all_teams(json_data: Dict[str, Any]) -> List[str]:
    """Returns a sorted list of UNIQUE canonical names."""
    teams = set()
    matches = json_data.get("matches", [])
    for match in matches:
        for ev in match.get("corner_events", []):
            canon = get_canonical_team(ev.get("teamName"))
            if canon:
                teams.add(canon)
    return sorted(list(teams))

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def _safe_int(x: Any, default: int = -1) -> int:
    try:
        return int(x)
    except Exception:
        return default

def _iter_numbers(x: Any) -> Iterable[float]:
    if isinstance(x, (int, float)) and math.isfinite(float(x)):
        yield float(x)

def _round_up_to_step(value, step):
    return math.ceil(value / step) * step

def _collect_event_xy(events):
    xs, ys = [], []
    for ev in events:
        for k in ("startPosXM", "endPosXM"):
            xs.extend(_iter_numbers(ev.get(k)))
        for k in ("startPosYM", "endPosYM"):
            ys.extend(_iter_numbers(ev.get(k)))
    return np.array(xs, dtype=float), np.array(ys, dtype=float)

def detect_field_bounds_from_events(events, q=0.999, margin_m=1.0, round_step_m=0.5):
    xs, ys = _collect_event_xy(events)
    if xs.size == 0 or ys.size == 0:
        return 52.5, -52.5, 34.0, -34.0
    abs_x, abs_y = np.abs(xs), np.abs(ys)
    half_len = _round_up_to_step(float(np.nanquantile(abs_x, q)) + margin_m, round_step_m)
    half_wid = _round_up_to_step(float(np.nanquantile(abs_y, q)) + margin_m, round_step_m)
    return max(half_len, 45.0), -max(half_len, 45.0), max(half_wid, 28.0), -max(half_wid, 28.0)

def _safe_abs_float(x):
    try:
        v = float(x)
        return abs(v) if math.isfinite(v) else None
    except Exception:
        return None

def _resolve_pitch_bounds(match, events):
    top_x = _safe_abs_float(match.get("pitch_top_x"))
    left_y = _safe_abs_float(match.get("pitch_left_y"))
    if top_x is None or left_y is None:
        for ev in events:
            if top_x is None:
                top_x = _safe_abs_float(ev.get("pitch_top_x"))
            if left_y is None:
                left_y = _safe_abs_float(ev.get("pitch_left_y"))
            if top_x and left_y:
                break
    if top_x and left_y:
        return max(top_x, 45.0), -max(top_x, 45.0), max(left_y, 28.0), -max(left_y, 28.0)
    return detect_field_bounds_from_events(events)

def get_corner(start_x, start_y, TOP_X, LEFT_Y, thresh=10.0):
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

def build_zones(top_x, bottom_x, left_y, right_y):
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

def point_in_rect(px, py, rect):
    xs = [p[0] for p in rect]
    ys = [p[1] for p in rect]
    return min(xs) <= px <= max(xs) and min(ys) <= py <= max(ys)

def _assign_zone(ex, ey, zones):
    for n, r in zones.items():
        if point_in_rect(ex, ey, r):
            return n
    return None

def _sequence_has_shot(seq):
    return any(ev.get("baseTypeName") == "SHOT" or ev.get("baseTypeId") == 6 for ev in seq)

def _is_true_corner_start(ev):
    if ev.get("possessionTypeName") != "CORNER":
        return False
    if ev.get("sequenceStart") is not True:
        return False
    return True

def _valid_zone_for_shot_lists(zone_val):
    return zone_val and str(zone_val).strip() != "" and zone_val != "Short_Corner_Zone"

def _valid_zone_for_attacking_shots(zone_val):
    return zone_val and str(zone_val).strip() != "" and zone_val != "Unassigned"

def _extract_jersey_number(ev: Dict[str, Any]) -> Optional[int]:
    """
    Tries common jersey/shirt number field names.
    If none found, returns None (still works).
    """
    candidates = (
        "jerseyNumber",
        "playerJerseyNumber",
        "playerJersey",
        "shirtNumber",
        "playerShirtNumber",
        "jersey_number",
        "player_jersey_number",
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
    if jersey is None:
        return name
    return f"{jersey}. {name}"

# ==========================================
# 3. ANALYSIS LOGIC (With New Table Format)
# ==========================================
def _build_corner_taker_tables(left_corners, right_corners, min_corners=5):
    """
    Shows 'Most Popular', '2nd', '3rd' zones instead of percentages.
    Filters out 'Unassigned' from the ranking.
    Renames Short_Corner_Zone to Short.
    Renames header to Cross Succ. %.
    Includes jersey number when present (e.g. '11. Genrich Sillé').
    """
    def _one_side_table(corners, side_name):
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
            # If we earlier stored plain name but now have a jersey, upgrade the display.
            if key not in display_by_key or (
                display_by_key[key] == raw_name and display != raw_name
            ):
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

            def fmt_zone(idx):
                if idx >= len(valid_zones):
                    return "-"
                zn, cnt = valid_zones[idx]
                pct = round((cnt / total) * 100.0, 0)
                display_zn = "Short" if zn == "Short_Corner_Zone" else zn
                return f"{display_zn} ({int(pct)}%)"

            rows.append({
                "Player": display_by_key.get(key, str(key)),
                "Corners": int(total),
                "Cross Succ. %": f"{rate}%" if not pd.isna(rate) else "-",
                "1st Choice": fmt_zone(0),
                "2nd Choice": fmt_zone(1),
                "3rd Choice": fmt_zone(2),
            })

        df = pd.DataFrame(rows)
        return df if df.empty else df.sort_values(["Corners"], ascending=[False])

    return {
        "left": _one_side_table(left_corners, "left"),
        "right": _one_side_table(right_corners, "right"),
    }

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

    pct_by_zone = {}
    for z, tot in total_by_zone.items():
        shots = shot_by_zone.get(z, 0)
        pct_by_zone[z] = (shots / tot) * 100.0 if tot > 0 else 0.0

    return total_by_zone, shot_by_zone, pct_by_zone

def process_corner_data(json_data, selected_team_name):
    matches = json_data.get("matches", [])
    opponent_left_side, opponent_right_side = [], []
    own_left_side, own_right_side = [], []
    opponent_seq_with_shot = []

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

        used_matches += 1
        TOP_X, BOTTOM_X, LEFT_Y, RIGHT_Y = _resolve_pitch_bounds(match, events)
        zones_TL, zones_BR, zones_TR, zones_BL = build_zones(TOP_X, BOTTOM_X, LEFT_Y, RIGHT_Y)
        sequences_by_id = defaultdict(list)

        for ev in events:
            if ev.get("sequenceId"):
                sequences_by_id[ev["sequenceId"]].append(ev)

        for e in events:
            if not _is_true_corner_start(e):
                continue

            raw_team = e.get("teamName", "")
            is_own = (get_canonical_team(raw_team) == selected_team_name)

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

            seq_has_shot = False
            if seq_id := e.get("sequenceId"):
                seq_evs = sequences_by_id[seq_id]
                seq_has_shot = _sequence_has_shot(seq_evs)
            e["seq_has_shot"] = bool(seq_has_shot)

            if e_side == "left":
                (own_left_side if is_own else opponent_left_side).append(e)
            else:
                (own_right_side if is_own else opponent_right_side).append(e)

            if seq_id := e.get("sequenceId"):
                seq_evs = sequences_by_id[seq_id]
                key = (match.get("match_id"), seq_id, is_own)
                if key not in _seen_seq_keys:
                    _seen_seq_keys.add(key)
                    for sev in seq_evs:
                        sev["zone"], sev["corner_side"] = zone_end, e_side

                    if (not is_own) and seq_has_shot and _valid_zone_for_shot_lists(zone_end):
                        opponent_seq_with_shot.append(seq_evs)

    def _calc_defensive_stats(opp_corners, opp_shot_seqs, side_filter):
        total_zone = Counter([c["zone"] for c in opp_corners if _valid_zone_for_shot_lists(c.get("zone"))])
        shot_seq_ids = defaultdict(set)
        for seq in opp_shot_seqs:
            start = next((x for x in seq if x.get("sequenceStart")), seq[0])
            if start.get("corner_side") == side_filter:
                shot_seq_ids[start.get("zone")].add(start.get("sequenceId"))
        return total_zone, shot_seq_ids, {z: (len(shot_seq_ids[z]) / t) * 100 for z, t in total_zone.items()}

    def_left_tot, def_left_ids, def_left_pct = _calc_defensive_stats(opponent_left_side, opponent_seq_with_shot, "left")
    def_right_tot, def_right_ids, def_right_pct = _calc_defensive_stats(opponent_right_side, opponent_seq_with_shot, "right")

    def _zone_pcts(corners):
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
        "defensive": {
            "left": (def_left_tot, def_left_ids, def_left_pct),
            "right": (def_right_tot, def_right_ids, def_right_pct),
        },
        "attacking": {
            "left_pct": _zone_pcts(own_left_side),
            "right_pct": _zone_pcts(own_right_side),
        },
        "attacking_shots": {
            "left": att_shots_left,   # (total_by_zone, shot_by_zone, pct_by_zone)
            "right": att_shots_right, # (total_by_zone, shot_by_zone, pct_by_zone)
        },
        "tables": taker_tables,
    }

def compute_league_attacking_corner_shot_rates(
    json_data: Dict[str, Any],
) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    """
    Computes, for every team in the dataset, attacking corners -> shot per zone and side.
    Includes Short_Corner_Zone (as requested).
    Returns a fully picklable structure:
        stats[team][side][zone] = {"shots": int, "total": int, "pct": float}
    """
    matches = json_data.get("matches", [])

    # Use only normal dicts (picklable)
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
            if ev.get("sequenceId"):
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

            seq_has_shot = False
            if seq_id:
                seq_has_shot = _sequence_has_shot(sequences_by_id.get(seq_id, []))

            bucket = _ensure_bucket(team, side, zone_end)
            bucket["total"] = int(bucket["total"]) + 1
            if seq_has_shot:
                bucket["shots"] = int(bucket["shots"]) + 1

    # finalize pct (ensure built-in types)
    for team, side_map in stats.items():
        for side, zone_map in side_map.items():
            for zone, d in zone_map.items():
                total = int(d.get("total", 0))
                shots = int(d.get("shots", 0))
                d["total"] = total
                d["shots"] = shots
                d["pct"] = (shots / total) * 100.0 if total > 0 else 0.0

    return stats

# ==========================================
# 4. PLOTTING FUNCTIONS
# ==========================================
def _load_bg(file_obj):
    return mpimg.imread(file_obj) if file_obj else np.ones((800, 1400, 3))

def plot_shots_defensive(img_file, polygons, shot_pct, total_by_zone, shot_seqids_by_zone, title):
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(_load_bg(img_file))
    ax.axis("off")
    vals = list(shot_pct.values())
    norm = plt.Normalize(min(vals) if vals else 0, max(vals) if vals else 1)
    cmap = cm.get_cmap("Reds")

    for zone, poly in polygons.items():
        if zone in shot_pct:
            ax.add_patch(
                Polygon(
                    poly,
                    closed=True,
                    facecolor=cmap(norm(shot_pct[zone])),
                    edgecolor="none",
                    alpha=0.55,
                )
            )
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

def plot_percent_attacking(img_file, polygons, centers, pct_by_zone, title):
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(_load_bg(img_file))
    ax.axis("off")
    vals = list(pct_by_zone.values())
    norm = plt.Normalize(min(vals) if vals else 0, max(vals) if vals else 1)
    cmap = cm.get_cmap("Reds")

    for zone, poly in polygons.items():
        if zone in pct_by_zone:
            ax.add_patch(
                Polygon(
                    poly,
                    closed=True,
                    facecolor=cmap(norm(pct_by_zone[zone])),
                    edgecolor="none",
                    alpha=0.55,
                )
            )
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
    img_file,
    polygons,
    shot_pct_by_zone: Dict[str, float],
    total_by_zone: Counter,
    shot_by_zone: Counter,
    percentile_by_zone: Dict[str, Optional[int]],
    *,
    min_zone_corners: int = 4,
    font_size: int = 18,
):
    """
    Attacking corners -> shot per zone with KKD percentile.
    Text position is computed from the polygon (same approach as plot_shots_defensive).
    Zones with < min_zone_corners are not colored and no text is printed.
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(_load_bg(img_file))
    ax.axis("off")

    eligible_vals = [
        float(shot_pct_by_zone[z])
        for z in shot_pct_by_zone
        if int(total_by_zone.get(z, 0)) >= min_zone_corners
    ]
    norm = plt.Normalize(min(eligible_vals) if eligible_vals else 0, max(eligible_vals) if eligible_vals else 1)
    cmap = cm.get_cmap("Reds")

    for zone, poly in polygons.items():
        tot = int(total_by_zone.get(zone, 0))
        if tot < min_zone_corners:
            continue
        if zone not in shot_pct_by_zone:
            continue

        ax.add_patch(
            Polygon(
                poly,
                closed=True,
                facecolor=cmap(norm(float(shot_pct_by_zone[zone]))),
                edgecolor="none",
                alpha=0.55,
            )
        )

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

# ==========================================
# 5. VISUALIZATION CONFIG
# ==========================================
def get_visualization_coords():
    def_L = {
        "Front_Zone": [(265, 15), (644, 15), (644, 600), (265,600)],
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

    # att_shot_centers_L = dict(att_centers_L)
    # att_shot_centers_L["GA1"] = (590, 215)
    # att_shot_centers_L["GA2"] = (745, 215)
    # att_shot_centers_L["GA3"] = (900, 215)
    
    # att_shot_centers_L["GA1"] = (590, 365)
    # att_shot_centers_L["GA2"] = (745, 365)
    # att_shot_centers_L["GA3"] = (900, 365)

    # att_shot_centers_L["Edge_Zone"] = (745, 580)
    # att_shot_centers_L["Front_Zone"] = (350, 410)
    # att_shot_centers_L["Back_Zone"] = (1120, 410)R
    # att_shot_centers_L["Short_Corner_Zone"] = (130, 410)

    
    # att_shot_centers_R = dict(att_centers_R)

    # att_shot_centers_R["GA1"] = (900, 215)
    # att_shot_centers_R["GA2"] = (745, 215)
    # att_shot_centers_R["GA3"] = (590, 215)
    
    # att_shot_centers_R["GA1"] = (900, 365)
    # att_shot_centers_R["GA2"] = (745, 365)
    # att_shot_centers_R["GA3"] = (590, 365)

    # att_shot_centers_R["Edge_Zone"] = (745, 580)
    # att_shot_centers_R["Front_Zone"] = (1120, 410)
    # att_shot_centers_R["Back_Zone"] = (350, 410)
    # att_shot_centers_R["Short_Corner_Zone"] = (1355, 410)

    return {
        "def_L": def_L,
        "def_R": def_R,
        "att_L": att_L,
        "att_centers_L": att_centers_L,
        "att_R": att_R,
        "att_centers_R": att_centers_R,
    }

