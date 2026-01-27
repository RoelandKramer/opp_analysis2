# opponent_analysis.py
import json
import math
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patheffects as PathEffects
from matplotlib.patches import Polygon
import matplotlib.image as mpimg

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================

def _safe_int(x: Any, default: int = -1) -> int:
    try:
        return int(x)
    except Exception:
        return default

def _iter_numbers(x: Any) -> Iterable[float]:
    if isinstance(x, (int, float)) and math.isfinite(float(x)):
        yield float(x)

def _round_up_to_step(value: float, step: float) -> float:
    return math.ceil(value / step) * step

def _collect_event_xy(events: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[float] = []
    ys: List[float] = []
    for ev in events:
        for k in ("startPosXM", "endPosXM"):
            xs.extend(_iter_numbers(ev.get(k)))
        for k in ("startPosYM", "endPosYM"):
            ys.extend(_iter_numbers(ev.get(k)))
    return np.array(xs, dtype=float), np.array(ys, dtype=float)

def detect_field_bounds_from_events(events: List[Dict[str, Any]], *, q: float = 0.999, margin_m: float = 1.0, round_step_m: float = 0.5) -> Tuple[float, float, float, float]:
    xs, ys = _collect_event_xy(events)
    if xs.size == 0 or ys.size == 0:
        return 52.5, -52.5, 34.0, -34.0
    abs_x = np.abs(xs)
    abs_y = np.abs(ys)
    half_len = float(np.nanquantile(abs_x, q))
    half_wid = float(np.nanquantile(abs_y, q))
    half_len = max(half_len, 45.0)
    half_wid = max(half_wid, 28.0)
    half_len = _round_up_to_step(half_len + margin_m, round_step_m)
    half_wid = _round_up_to_step(half_wid + margin_m, round_step_m)
    return half_len, -half_len, half_wid, -half_wid

def _safe_abs_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if not math.isfinite(v): return None
        return abs(v)
    except Exception: return None

def _resolve_pitch_bounds(match: Dict[str, Any], events: List[Dict[str, Any]]) -> Tuple[float, float, float, float]:
    top_x = _safe_abs_float(match.get("pitch_top_x"))
    left_y = _safe_abs_float(match.get("pitch_left_y"))
    
    if top_x is None or left_y is None:
        for ev in events:
            if not isinstance(ev, dict): continue
            if top_x is None: top_x = _safe_abs_float(ev.get("pitch_top_x"))
            if left_y is None: left_y = _safe_abs_float(ev.get("pitch_left_y"))
            if top_x is not None and left_y is not None: break

    if top_x is not None and left_y is not None:
        top_x = max(top_x, 45.0)
        left_y = max(left_y, 28.0)
        return top_x, -top_x, left_y, -left_y
    
    return detect_field_bounds_from_events(events)

def get_corner(start_x: float, start_y: float, TOP_X: float, LEFT_Y: float, thresh: float = 10.0) -> Optional[str]:
    corners = {
        "top_left": (TOP_X, LEFT_Y),
        "top_right": (TOP_X, -LEFT_Y),
        "bottom_left": (-TOP_X, LEFT_Y),
        "bottom_right": (-TOP_X, -LEFT_Y),
    }
    best_name = None
    best_dist = float("inf")
    for name, (cx, cy) in corners.items():
        d = math.hypot(start_x - cx, start_y - cy)
        if d < best_dist:
            best_dist = d
            best_name = name
    return best_name if best_dist <= thresh else None

def build_zones(top_x: float, bottom_x: float, left_y: float, right_y: float):
    length_PA = 16.5; width_PA = 40.32; length_penalty_spot = 11; length_PB = 5.49; width_PB = 18.29
    PA_x = top_x - length_PA; PB_x = top_x - length_PB
    half_PA = width_PA / 2; half_PB = width_PB / 2
    PS_x = top_x - length_penalty_spot; edge_x = top_x - (length_PA + 4)
    GA_band = width_PB / 3

    zones_top_left = {
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
    zones_bottom_right = {name: [(-x, -y) for (x, y) in rect] for name, rect in zones_top_left.items()}
    zones_top_right_side = {name: [(x, -y) for (x, y) in rect] for name, rect in zones_top_left.items()}
    zones_bottom_left_side = {name: [(-x, y) for (x, y) in rect] for name, rect in zones_top_left.items()}
    return zones_top_left, zones_bottom_right, zones_top_right_side, zones_bottom_left_side

def point_in_rect(px: float, py: float, rect: List[Tuple[float, float]]) -> bool:
    xs = [p[0] for p in rect]; ys = [p[1] for p in rect]
    return (min(xs) <= px <= max(xs)) and (min(ys) <= py <= max(ys))

def _assign_zone(ex: float, ey: float, zones: Dict[str, List[Tuple[float, float]]]) -> Optional[str]:
    for zone_name, rect in zones.items():
        if point_in_rect(ex, ey, rect): return zone_name
    return None

def _sequence_has_shot(seq_events_sorted: List[Dict[str, Any]]) -> bool:
    return any(ev.get("baseTypeName") == "SHOT" or ev.get("baseTypeId") == 6 for ev in seq_events_sorted)

def _is_true_corner_start(ev: Dict[str, Any]) -> bool:
    if ev.get("possessionTypeName") != "CORNER": return False
    if ev.get("sequenceStart") is not True: return False
    st = ev.get("subTypeName", "")
    return isinstance(st, str) and st.strip().upper().startswith("CORNER")

def _valid_zone_for_shot_lists(zone_val: Any) -> bool:
    return zone_val is not None and str(zone_val).strip() != "" and zone_val != "Short_Corner_Zone"

ZONE_ORDER = ["Short_Corner_Zone", "Front_Zone", "Back_Zone", "GA1", "GA2", "GA3", "CA1", "CA2", "CA3", "Edge_Zone", "Unassigned"]

# ==========================================
# 2. PLOTTING FUNCTIONS
# ==========================================

def _load_bg(file_obj, fallback_shape=(800, 1400, 3)):
    if file_obj:
        return mpimg.imread(file_obj)
    return np.ones(fallback_shape, dtype=float)

def plot_shots_defensive(img_file, polygons, shot_pct, total_by_zone, shot_seqids_by_zone, title):
    img = _load_bg(img_file)
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(img)
    ax.axis("off")

    vals = list(shot_pct.values())
    if vals:
        vmin, vmax = min(vals), max(vals)
        if vmin == vmax: vmin, vmax = 0.0, max(1.0, vmax)
    else:
        vmin, vmax = 0.0, 1.0

    norm = plt.Normalize(vmin, vmax)
    cmap = cm.get_cmap("Reds")

    zone_centers = {}
    x_shift = 5
    for zone, poly in polygons.items():
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        zone_centers[zone] = ((min(xs) + max(xs)) / 2 - x_shift, (min(ys) + max(ys)) / 2)

    for zone, poly in polygons.items():
        if zone in shot_pct:
            color = cmap(norm(shot_pct[zone]))
            ax.add_patch(Polygon(poly, closed=True, facecolor=color, edgecolor="none", alpha=0.55))

    text_style = dict(fontsize=22, color="white", weight="bold", ha="center", va="center",
                      path_effects=[PathEffects.withStroke(linewidth=3, foreground="black")])

    for zone, (x, y) in zone_centers.items():
        if zone in shot_pct:
            total = int(total_by_zone.get(zone, 0))
            shots = len(shot_seqids_by_zone.get(zone, set()))
            ax.text(x, y, f"{shots}/{total}", **text_style)

    ax.set_title(title, fontsize=24, weight="bold", y=0.9)
    return fig

def plot_percent_attacking(img_file, polygons, centers, pct_by_zone, title):
    img = _load_bg(img_file)
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(img)
    ax.axis("off")

    vals = list(pct_by_zone.values())
    if vals:
        vmin, vmax = min(vals), max(vals)
        if vmin == vmax: vmin, vmax = 0.0, max(1.0, vmax)
    else:
        vmin, vmax = 0.0, 1.0

    norm = plt.Normalize(vmin, vmax)
    cmap = cm.get_cmap("Reds")

    for zone, poly in polygons.items():
        if zone in pct_by_zone:
            color = cmap(norm(pct_by_zone[zone]))
            ax.add_patch(Polygon(poly, closed=True, facecolor=color, edgecolor="none", alpha=0.55))

    text_style = dict(fontsize=22, color="white", weight="bold", ha="center", va="center",
                      path_effects=[PathEffects.withStroke(linewidth=3, foreground="black")])

    for zone, (x, y) in centers.items():
        if zone in pct_by_zone:
            ax.text(x, y, f"{pct_by_zone[zone]:.1f}%", **text_style)

    ax.set_title(title, fontsize=26, weight="bold", y=0.82)
    return fig

# ==========================================
# 3. ANALYSIS LOGIC
# ==========================================

def _build_corner_taker_tables(left_corners, right_corners, min_corners=3, zone_order=None):
    zone_order = zone_order or ZONE_ORDER

    def _one_side_table(corners, side_name):
        total_by_player = Counter()
        zone_counts = defaultdict(Counter)
        cross_attempts = defaultdict(int)
        cross_successes = defaultdict(int)

        for e in corners:
            player = e.get("playerName", "").strip()
            if not player: continue
            
            z = e.get("zone", "")
            zone = z if (isinstance(z, str) and z.strip()) else "Unassigned"
            
            total_by_player[player] += 1
            zone_counts[player][zone] += 1
            
            is_cross = zone not in ("Short_Corner_Zone", "Unassigned")
            if is_cross:
                cross_attempts[player] += 1
                if e.get("resultName") == "SUCCESSFUL":
                    cross_successes[player] += 1
        
        rows = []
        for player, total in total_by_player.items():
            if total < min_corners: continue
            attempts = cross_attempts[player]
            succ = cross_successes[player]
            rate = round((succ / attempts) * 100.0, 2) if attempts > 0 else np.nan
            
            row = {
                "side": side_name,
                "playerName": player,
                "corners": int(total),
                "cross_attempts": attempts,
                "cross_successes": succ,
                "cross_success_rate_%": rate
            }
            for zn in zone_order:
                cnt = zone_counts[player][zn]
                row[zn] = round((cnt / total) * 100.0, 2) if total > 0 else 0.0
            rows.append(row)
        
        df = pd.DataFrame(rows)
        if df.empty: return pd.DataFrame(columns=["side", "playerName"] + zone_order)
        return df.sort_values(["corners", "cross_success_rate_%"], ascending=[False, False])

    df_left = _one_side_table(left_corners, "left")
    df_right = _one_side_table(right_corners, "right")
    df_all = pd.concat([df_left, df_right], ignore_index=True)
    return {"left": df_left, "right": df_right, "all": df_all}

def extract_all_teams(json_data: Dict[str, Any]) -> List[str]:
    """
    Scans the entire JSON to find every unique teamName present in the corner events.
    Used to populate the Streamlit dropdown.
    """
    teams = set()
    matches = json_data.get("matches", [])
    for match in matches:
        events = match.get("corner_events", [])
        for ev in events:
            if tn := ev.get("teamName"):
                teams.add(tn.strip())
    
    return sorted(list(teams))
  
def process_corner_data(json_data, team_aliases):
    """
    Main entry point for processing the JSON. 
    Returns a dictionary of data ready for visualization.
    """
    aliases_norm = {a.strip().lower() for a in team_aliases if a.strip()}
    
    matches = json_data.get("matches", [])
    opponent_left_side = []
    opponent_right_side = []
    own_left_side = []
    own_right_side = []
    opponent_seq_with_shot = []
    
    used_matches = 0
    _seen_seq_keys = set()

    for match in matches:
        events = match.get("corner_events", [])
        if not events: continue

        # Identify home/away
        home_tm, away_tm = None, None
        for ev in events:
            if ev.get("groupName") == "HOME": home_tm = ev.get("teamName")
            elif ev.get("groupName") == "AWAY": away_tm = ev.get("teamName")
            if home_tm and away_tm: break
            
        teams_in_match = {t.strip().lower() for t in [home_tm or "", away_tm or ""] if t}
        if not (teams_in_match & aliases_norm):
            continue
        
        used_matches += 1
        TOP_X, BOTTOM_X, LEFT_Y, RIGHT_Y = _resolve_pitch_bounds(match, events)
        zones_TL, zones_BR, zones_TR, zones_BL = build_zones(TOP_X, BOTTOM_X, LEFT_Y, RIGHT_Y)

        sequences_by_id = defaultdict(list)
        for ev in events:
            if ev.get("sequenceId"): sequences_by_id[ev["sequenceId"]].append(ev)

        for e in events:
            if not _is_true_corner_start(e): continue
            
            tn = e.get("teamName", "")
            is_own = tn.strip().lower() in aliases_norm
            
            sx, sy = e.get("startPosXM"), e.get("startPosYM")
            ex, ey = e.get("endPosXM"), e.get("endPosYM")
            
            if None in (sx, sy, ex, ey): continue
            
            corner_type = get_corner(float(sx), float(sy), TOP_X, LEFT_Y)
            if not corner_type: continue
            
            e_side = None
            local_zones = {}
            if corner_type == "top_left": 
                local_zones = zones_TL; e_side = "left"
            elif corner_type == "bottom_right": 
                local_zones = zones_BR; e_side = "left"
            elif corner_type == "top_right": 
                local_zones = zones_TR; e_side = "right"
            elif corner_type == "bottom_left": 
                local_zones = zones_BL; e_side = "right"
            
            zone_end = _assign_zone(float(ex), float(ey), local_zones)
            if zone_end is None:
                # Fallback flip
                zone_end = _assign_zone(-float(ex), -float(ey), local_zones)
            
            e["zone"] = zone_end
            e["corner_side"] = e_side

            # Add to lists
            if e_side == "left":
                if is_own: own_left_side.append(e)
                else: opponent_left_side.append(e)
            elif e_side == "right":
                if is_own: own_right_side.append(e)
                else: opponent_right_side.append(e)

            # Sequence Analysis (Shots)
            seq_id = e.get("sequenceId")
            if seq_id:
                seq_evs = sequences_by_id[seq_id]
                key = (match.get("match_id"), seq_id, is_own)
                if key in _seen_seq_keys: continue
                _seen_seq_keys.add(key)
                
                # Propagate info to sequence
                for sev in seq_evs:
                    sev["zone"] = zone_end
                    sev["corner_side"] = e_side
                    
                has_shot = _sequence_has_shot(seq_evs)
                valid_z = _valid_zone_for_shot_lists(zone_end)
                
                if not is_own and has_shot and valid_z:
                    opponent_seq_with_shot.append(seq_evs)

    # --- Aggregations ---
    
    # 1. Defensive Shot Pcts
    def _calc_defensive_stats(opp_corners, opp_shot_seqs, side_filter):
        total_zone = Counter()
        for c in opp_corners:
            if _valid_zone_for_shot_lists(c.get("zone")):
                total_zone[c["zone"]] += 1
        
        shot_seq_ids = defaultdict(set)
        for seq in opp_shot_seqs:
            # Find start event
            start = next((x for x in seq if x.get("sequenceStart")), seq[0])
            if start.get("corner_side") == side_filter:
                shot_seq_ids[start.get("zone")].add(start.get("sequenceId"))
        
        pcts = {z: (len(shot_seq_ids[z])/t)*100.0 for z, t in total_zone.items()}
        return total_zone, shot_seq_ids, pcts

    def_left_tot, def_left_ids, def_left_pct = _calc_defensive_stats(opponent_left_side, opponent_seq_with_shot, "left")
    def_right_tot, def_right_ids, def_right_pct = _calc_defensive_stats(opponent_right_side, opponent_seq_with_shot, "right")

    # 2. Attacking Zone Pcts
    def _zone_pcts(corners):
        c = Counter(e["zone"] for e in corners if e.get("zone"))
        tot = sum(c.values())
        return {k: (v/tot)*100.0 for k,v in c.items()} if tot else {}

    att_left_pct = _zone_pcts(own_left_side)
    att_right_pct = _zone_pcts(own_right_side)

    # 3. Tables
    taker_tables = _build_corner_taker_tables(own_left_side, own_right_side)

    return {
        "used_matches": used_matches,
        "own_left_count": len(own_left_side),
        "own_right_count": len(own_right_side),
        "defensive": {
            "left": (def_left_tot, def_left_ids, def_left_pct),
            "right": (def_right_tot, def_right_ids, def_right_pct)
        },
        "attacking": {
            "left_pct": att_left_pct,
            "right_pct": att_right_pct
        },
        "tables": taker_tables
    }

# ==========================================
# 4. VISUALIZATION CONFIG (Static Pixel Coords)
# ==========================================
def get_visualization_coords():
    """
    Returns the hardcoded PIXEL coordinates for drawing zones 
    on the standard pitch background images.
    """
    # Defensive Layout (Pixels)
    def_L = {
        "Front_Zone": [(265, 15), (640, 15), (640, 595), (265, 595)],
        "Back_Zone": [(1292, 15), (1667, 15), (1667, 595), (1292, 595)],
        "Short_Corner_Zone": [(0, 135), (110, 6000), (110, 575), (0, 575)],
        "GA1": [(655, 14), (865, 14), (865, 203), (655, 203)],
        "GA2": [(867, 14), (1080, 14), (1080, 203), (867, 203)],
        "GA3": [(1080, 14), (1275, 14), (1275, 203), (1080, 203)],
        "CA1": [(653, 212), (865, 212), (865, 397), (653, 397)],
        "CA2": [(867, 212), (1080, 212), (1080, 397), (867, 397)],
        "CA3": [(1080, 212), (1285, 212), (1285, 397), (1080, 397)],
        "Edge_Zone": [(653, 405), (1280, 405), (1280, 770), (653, 770)],
    }
    
    # Defensive Right (Mirrored logic)
    def_R = {
        "Front_Zone": def_L["Back_Zone"], "Back_Zone": def_L["Front_Zone"],
        "Short_Corner_Zone": def_L["Short_Corner_Zone"], 
        "GA1": def_L["GA3"], "GA2": def_L["GA2"], "GA3": def_L["GA1"],
        "CA1": def_L["CA3"], "CA2": def_L["CA2"], "CA3": def_L["CA1"],
        "Edge_Zone": def_L["Edge_Zone"],
    }

    # Attacking Layout (Pixels)
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
        "Edge_Zone": [(502, 425), (960, 425), (960, 690), (502, 690)],
    }

    att_centers_L = {
        "GA1": (590, 245), "GA2": (745, 250), "GA3": (900, 250),
        "CA1": (590, 400), "CA2": (745, 400), "CA3": (900, 400),
        "Edge_Zone": (745, 520), "Front_Zone": (350, 380),
        "Back_Zone": (1127, 380), "Short_Corner_Zone": (130, 380),
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
        "GA3": (590, 250), "GA2": (745, 250), "GA1": (900, 250),
        "CA3": (590, 400), "CA2": (745, 400), "CA1": (900, 400),
        "Edge_Zone": (745, 520), "Back_Zone": (350, 380),
        "Front_Zone": (1120, 380), "Short_Corner_Zone": (1355, 400),
    }

    return {
        "def_L": def_L, "def_R": def_R,
        "att_L": att_L, "att_centers_L": att_centers_L,
        "att_R": att_R, "att_centers_R": att_centers_R
    }
