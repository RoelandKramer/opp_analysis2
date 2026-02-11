# ============================================================
# file: update_database.py
# ============================================================
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

_EVENTS_GLOB = "**/*SciSportsEvents*.json"
_POSITIONS_GLOB = "**/*SciSportsPositions*.json"

_MATCH_ID_RE = re.compile(r"-\s*(\d+)\s*$")


def match_id_from_filename(path: Path) -> Optional[str]:
    m = _MATCH_ID_RE.search(path.stem)
    return m.group(1) if m else None


def normalize_path(p: str | Path) -> str:
    try:
        pp = p if isinstance(p, Path) else Path(str(p))
        return pp.resolve().as_posix().lower()
    except Exception:
        return str(p).replace("\\", "/").lower()


def iter_files_recursive(folder: Path, pattern: str) -> List[Path]:
    return sorted([p for p in folder.glob(pattern) if p.is_file()])


def ensure_csv(path: Path, columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        pd.DataFrame(columns=columns).to_csv(path, index=False, encoding="utf-8")
        return
    df = pd.read_csv(path, low_memory=False)
    changed = False
    for c in columns:
        if c not in df.columns:
            df[c] = pd.NA
            changed = True
    if changed:
        df.to_csv(path, index=False, encoding="utf-8")


def load_norm_set(csv_path: Path, col: str) -> Set[str]:
    if not csv_path.exists():
        return set()
    try:
        df = pd.read_csv(csv_path, usecols=[col], low_memory=False)
        return {normalize_path(v) for v in df[col].dropna().astype(str).tolist()}
    except Exception:
        return set()


def load_set(csv_path: Path, col: str) -> Set[str]:
    if not csv_path.exists():
        return set()
    try:
        df = pd.read_csv(csv_path, usecols=[col], low_memory=False)
        return {str(v) for v in df[col].dropna().astype(str).tolist()}
    except Exception:
        return set()


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def get_events_list(container: Dict[str, Any]) -> List[Dict[str, Any]]:
    data = container.get("data")
    if isinstance(data, list):
        return [e for e in data if isinstance(e, dict)]
    events = container.get("events")
    if isinstance(events, list):
        return [e for e in events if isinstance(e, dict)]
    return []


def truthy(x: Any) -> bool:
    if x is True:
        return True
    if isinstance(x, (int, float)) and x == 1:
        return True
    if isinstance(x, str) and x.strip().lower() in {"true", "1", "yes"}:
        return True
    return False


def safe_int(x: Any, default: int = -1) -> int:
    try:
        return int(x)
    except Exception:
        return default


def safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if v != v:
            return None
        return v
    except Exception:
        return None


def match_name(container: Dict[str, Any], fallback: str) -> str:
    md = container.get("metaData")
    if isinstance(md, dict) and isinstance(md.get("name"), str) and md["name"].strip():
        return md["name"].strip()
    if isinstance(container.get("matchName"), str) and container["matchName"].strip():
        return container["matchName"].strip()
    return fallback


def match_id_from_json(container: Dict[str, Any]) -> Optional[str]:
    md = container.get("metaData")
    if isinstance(md, dict) and md.get("id") is not None:
        return str(md["id"])
    return None


def max_abs_startpos_xy(events: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    max_x: Optional[float] = None
    max_y: Optional[float] = None
    for e in events:
        sx = safe_float(e.get("startPosXM"))
        sy = safe_float(e.get("startPosYM"))
        if sx is not None:
            ax = abs(sx)
            max_x = ax if (max_x is None or ax > max_x) else max_x
        if sy is not None:
            ay = abs(sy)
            max_y = ay if (max_y is None or ay > max_y) else max_y
    return {"pitch_top_x": max_x, "pitch_left_y": max_y}


def _event_time_ms(e: Dict[str, Any]) -> int:
    t = safe_int(e.get("startTimeMs"), -1)
    if t >= 0:
        return t
    return safe_int(e.get("endTimeMs"), -1)


def _find_seq_end_time(events: List[Dict[str, Any]], seq_id: Any, t0: int) -> int:
    candidates: List[Tuple[int, Dict[str, Any]]] = []
    for e in events:
        if e.get("sequenceId") != seq_id:
            continue
        if not truthy(e.get("sequenceEnd")):
            continue
        t = _event_time_ms(e)
        if t >= t0 >= 0:
            candidates.append((t, e))
    if not candidates:
        return t0
    candidates.sort(key=lambda x: x[0])
    e = candidates[0][1]
    t1 = safe_int(e.get("endTimeMs"), -1)
    if t1 >= 0:
        return t1
    t1 = safe_int(e.get("startTimeMs"), -1)
    return t1 if t1 >= 0 else t0


def _corner_sequence_keys(events: List[Dict[str, Any]]) -> List[Tuple[Any, int, int, Dict[str, Any]]]:
    """
    Corner starts: possessionTypeName=='CORNER' and sequenceStart==TRUE.

    Dedupe rule:
      - If sequenceId != -1: keep earliest startTimeMs per sequenceId (per match file).
      - If sequenceId == -1: NEVER dedupe; each start is a separate corner.
    """
    starts: List[Dict[str, Any]] = [
        e
        for e in events
        if e.get("possessionTypeName") == "CORNER"
        and truthy(e.get("sequenceStart"))
        and safe_int(e.get("startTimeMs"), -1) >= 0
        and e.get("sequenceId") is not None
    ]
    if not starts:
        return []

    best_by_seq: Dict[str, Dict[str, Any]] = {}
    best_t0_by_seq: Dict[str, int] = {}
    seq_minus_one_starts: List[Dict[str, Any]] = []

    for s in starts:
        seq_raw = s.get("sequenceId")
        t0 = safe_int(s.get("startTimeMs"), -1)
        if t0 < 0:
            continue
        if safe_int(seq_raw, -999999) == -1:
            seq_minus_one_starts.append(s)
            continue
        seq_key = str(seq_raw)
        if seq_key not in best_t0_by_seq or t0 < best_t0_by_seq[seq_key]:
            best_t0_by_seq[seq_key] = t0
            best_by_seq[seq_key] = s

    out: List[Tuple[Any, int, int, Dict[str, Any]]] = []

    for seq_key, start_ev in best_by_seq.items():
        seq_id = start_ev.get("sequenceId")
        t0 = best_t0_by_seq[seq_key]
        t1 = _find_seq_end_time(events, seq_id, t0)
        out.append((seq_id, t0, t1, start_ev))

    # seqId == -1: keep each as separate; conservative t1=t0
    for start_ev in seq_minus_one_starts:
        t0 = safe_int(start_ev.get("startTimeMs"), -1)
        if t0 >= 0:
            out.append((start_ev.get("sequenceId"), t0, t0, start_ev))

    out.sort(key=lambda x: x[1])
    return out


def _dedupe_append_csv(path: Path, new_df: pd.DataFrame, key_cols: List[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    if new_df.empty:
        return 0
    if not path.exists():
        new_df.to_csv(path, index=False, encoding="utf-8")
        return len(new_df)

    old_df = pd.read_csv(path, low_memory=False)
    all_cols = sorted(set(old_df.columns).union(new_df.columns))
    old_df = old_df.reindex(columns=all_cols)
    new_df = new_df.reindex(columns=all_cols)

    before = len(old_df)
    merged = pd.concat([old_df, new_df], ignore_index=True)

    if all(c in merged.columns for c in key_cols):
        merged = merged.drop_duplicates(subset=key_cols, keep="first")
    else:
        merged = merged.drop_duplicates(keep="first")

    merged.to_csv(path, index=False, encoding="utf-8")
    return max(0, len(merged) - before)


def get_positions_frames_scisports(container: Dict[str, Any]) -> List[Dict[str, Any]]:
    for k in ("data", "frames", "positions"):
        v = container.get(k)
        if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
            if "t" in v[0] and ("h" in v[0] or "a" in v[0]):
                return v
    queue: List[Any] = [container]
    seen: Set[int] = set()
    while queue:
        cur = queue.pop(0)
        if id(cur) in seen:
            continue
        seen.add(id(cur))
        if isinstance(cur, list) and cur and all(isinstance(x, dict) for x in cur):
            if "t" in cur[0] and ("h" in cur[0] or "a" in cur[0]):
                return cur
        if isinstance(cur, dict):
            for vv in cur.values():
                if isinstance(vv, (dict, list)):
                    queue.append(vv)
        elif isinstance(cur, list):
            for vv in cur:
                if isinstance(vv, (dict, list)):
                    queue.append(vv)
    return []


def build_frame_index(frames: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for fr in frames:
        t = safe_int(fr.get("t"), -1)
        if t >= 0 and t not in out:
            out[t] = fr
    return out


def players_from_frame(frame: Dict[str, Any], side_key: str) -> List[Dict[str, Any]]:
    v = frame.get(side_key)
    if isinstance(v, list):
        return [p for p in v if isinstance(p, dict) and p.get("p") is not None]
    return []


def euclid(ax: float, ay: float, bx: float, by: float) -> float:
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def is_shot_event(row: pd.Series) -> bool:
    return str(row.get("baseTypeName", "")).upper() == "SHOT"


def is_goal_event(row: pd.Series) -> bool:
    return is_shot_event(row) and str(row.get("resultName", "")).upper() == "SUCCESSFUL"


def shot_bodypart_is_head(row: pd.Series) -> bool:
    return str(row.get("bodyPartName", "")).upper() == "HEAD"


def is_defensive_clear_or_block(row: pd.Series, defending_group: str) -> bool:
    if str(row.get("groupName", "")).upper() != str(defending_group).upper():
        return False
    base = str(row.get("baseTypeName", "")).upper()
    sub = str(row.get("subTypeName", "")).upper()
    return (base in {"BLOCK", "CLEARANCE", "INTERCEPTION", "DUEL", "TACKLE"}) or (
        sub in {"BLOCKED_SHOT", "SHOT_BLOCKED", "CLEARANCE", "CORNER_CLEARED", "CROSS_BLOCKED", "CROSS_CLEARED"}
    )


def update_database(*, uploads_dir: Path, data_dir: Path) -> Dict[str, Any]:
    """
    Updates:
      - data/corner_events_all_matches.csv
      - data/corner_events_full_sequences.csv
      - data/corner_positions_headers.csv
    using JSONs from uploads_dir.
    """
    try:
        uploads_dir = Path(uploads_dir)
        data_dir = Path(data_dir)

        csv_events_all = data_dir / "corner_events_all_matches.csv"
        csv_events_full = data_dir / "corner_events_full_sequences.csv"
        csv_headers = data_dir / "corner_positions_headers.csv"

        # ensure CSVs exist
        ensure_csv(
            csv_events_all,
            columns=[
                "match_name",
                "match_id",
                "source_event_file",
                "pitch_top_x",
                "pitch_left_y",
                "corner_sequence_id",
                "corner_startTimeMs",
                "corner_endTimeMs",
                "corner_taker_id",
                "corner_taker_name",
            ],
        )
        ensure_csv(
            csv_events_full,
            columns=[
                "match_name",
                "match_id",
                "source_event_file",
                "pitch_top_x",
                "pitch_left_y",
                "corner_sequence_id",
                "corner_startTimeMs",
                "corner_endTimeMs",
                "corner_taker_id",
                "corner_taker_name",
            ],
        )
        ensure_csv(
            csv_headers,
            columns=[
                "player_name",
                "player_id",
                "s",
                "club",
                "club_id",
                "match_id",
                "Attacking_corners_headed",
                "Attacking_corners_headed_and_scored",
                "Defending_corners_defended",
                "Defending_corners_errors",
                "Defending_corners_fatal_errors",
                "Attacking_corners_on_pitch",
                "Defending_corners_on_pitch",
                "source_position_file",
                "source_event_file",
            ],
        )

        # detect new files
        event_files = iter_files_recursive(uploads_dir, _EVENTS_GLOB)
        pos_files = iter_files_recursive(uploads_dir, _POSITIONS_GLOB)

        existing_event_sources = load_norm_set(csv_events_all, "source_event_file")
        existing_match_ids_events = load_set(csv_events_all, "match_id")
        existing_pos_sources = load_norm_set(csv_headers, "source_position_file")

        new_event_files: List[Path] = []
        for p in event_files:
            if normalize_path(p) in existing_event_sources:
                continue
            mid = match_id_from_filename(p)
            if mid and mid in existing_match_ids_events:
                continue
            new_event_files.append(p)

        new_pos_files: List[Path] = [p for p in pos_files if normalize_path(p) not in existing_pos_sources]

        # map uploaded positions by match_id (pick first if duplicates)
        pos_by_mid: Dict[str, Path] = {}
        for p in new_pos_files:
            mid = match_id_from_filename(p)
            if mid and mid not in pos_by_mid:
                pos_by_mid[mid] = p

        # 1) Append events
        rows_all: List[Dict[str, Any]] = []
        rows_full: List[Dict[str, Any]] = []

        for ev_path in new_event_files:
            ev_container = read_json(ev_path)
            if not ev_container:
                continue

            mid = match_id_from_filename(ev_path) or match_id_from_json(ev_container)
            if not mid:
                continue

            events = get_events_list(ev_container)
            if not events:
                continue

            mname = match_name(ev_container, fallback=ev_path.stem)
            bounds = max_abs_startpos_xy(events)

            seq_keys = _corner_sequence_keys(events)
            if not seq_keys:
                continue

            for seq_id, t0, t1, start_ev in seq_keys:
                start_row = dict(start_ev)
                start_row.update(
                    {
                        "match_name": mname,
                        "match_id": str(mid),
                        "source_event_file": str(ev_path),
                        "pitch_top_x": bounds["pitch_top_x"],
                        "pitch_left_y": bounds["pitch_left_y"],
                        "corner_sequence_id": seq_id,
                        "corner_startTimeMs": t0,
                        "corner_endTimeMs": t1,
                        "corner_taker_id": start_ev.get("playerId"),
                        "corner_taker_name": start_ev.get("playerName"),
                    }
                )
                rows_all.append(start_row)

                # full seq events:
                # - if seq_id != -1: include by sequenceId and window t0..t1
                # - if seq_id == -1: include only events with startTimeMs == t0 (best-effort)
                seq_int = safe_int(seq_id, -999999)
                for e in events:
                    te = _event_time_ms(e)
                    if te < 0:
                        continue

                    if seq_int != -1:
                        if e.get("sequenceId") != seq_id:
                            continue
                        if te < t0 or te > t1:
                            continue
                    else:
                        if te != t0:
                            continue

                    row = dict(e)
                    row.update(
                        {
                            "match_name": mname,
                            "match_id": str(mid),
                            "source_event_file": str(ev_path),
                            "pitch_top_x": bounds["pitch_top_x"],
                            "pitch_left_y": bounds["pitch_left_y"],
                            "corner_sequence_id": seq_id,
                            "corner_startTimeMs": t0,
                            "corner_endTimeMs": t1,
                            "corner_taker_id": start_ev.get("playerId"),
                            "corner_taker_name": start_ev.get("playerName"),
                        }
                    )
                    rows_full.append(row)

        df_new_all = pd.DataFrame(rows_all)
        df_new_full = pd.DataFrame(rows_full)

        # IMPORTANT: include corner_startTimeMs because seqId==-1 must not collapse
        added_all = _dedupe_append_csv(
            csv_events_all,
            df_new_all,
            key_cols=["match_id", "corner_sequence_id", "corner_startTimeMs"],
        )
        added_full = _dedupe_append_csv(
            csv_events_full,
            df_new_full,
            key_cols=[
                "match_id",
                "corner_sequence_id",
                "corner_startTimeMs",
                "startTimeMs",
                "playerId",
                "baseTypeId",
                "possessionTypeName",
            ],
        )

        # 2) Update headers based on just-added matches (fast)
        if df_new_full.empty:
            return {
                "ok": True,
                "added_events_all": int(added_all),
                "added_events_full": int(added_full),
                "headers_net_new_rows": 0,
            }

        df_full = pd.read_csv(csv_events_full, low_memory=False).where(pd.notnull, None)
        df_headers = pd.read_csv(csv_headers, low_memory=False).where(pd.notnull, None)

        # index existing header rows by (match_id, player_id)
        def k_mid_pid(mid: Any, pid: Any) -> Tuple[str, str]:
            return (str(mid), str(pid))

        header_idx: Dict[Tuple[str, str], int] = {}
        if {"match_id", "player_id"}.issubset(df_headers.columns):
            for i, r in df_headers[["match_id", "player_id"]].dropna().iterrows():
                header_idx[k_mid_pid(r["match_id"], r["player_id"])] = int(i)

        # name lookup for initialization
        name_lookup: Dict[Tuple[str, str], str] = {}
        if {"match_id", "playerId", "playerName"}.issubset(df_full.columns):
            sub = df_full[["match_id", "playerId", "playerName"]].dropna(subset=["match_id", "playerId"])
            for rr in sub.itertuples(index=False):
                mid = str(getattr(rr, "match_id"))
                pid = str(getattr(rr, "playerId"))
                pn = getattr(rr, "playerName", None)
                if pn is not None and str(pn).strip():
                    name_lookup[(mid, pid)] = str(pn).strip()

        count_cols = [
            "Attacking_corners_headed",
            "Attacking_corners_headed_and_scored",
            "Defending_corners_defended",
            "Defending_corners_errors",
            "Defending_corners_fatal_errors",
            "Attacking_corners_on_pitch",
            "Defending_corners_on_pitch",
        ]
        for c in count_cols:
            if c not in df_headers.columns:
                df_headers[c] = 0
            df_headers[c] = pd.to_numeric(df_headers[c], errors="coerce").fillna(0).astype(int)

        def get_or_create_row(
            *,
            match_id: str,
            player_id: str,
            s: Any,
            club: str,
            club_id: str,
            source_event_file: str,
            source_position_file: str,
        ) -> int:
            key = k_mid_pid(match_id, player_id)
            if key in header_idx:
                i = header_idx[key]
                if pd.isna(df_headers.at[i, "player_name"]):
                    df_headers.at[i, "player_name"] = name_lookup.get((match_id, player_id), pd.NA)
                df_headers.at[i, "source_event_file"] = source_event_file
                df_headers.at[i, "source_position_file"] = source_position_file
                if pd.isna(df_headers.at[i, "s"]):
                    df_headers.at[i, "s"] = s
                if pd.isna(df_headers.at[i, "club"]):
                    df_headers.at[i, "club"] = club
                if pd.isna(df_headers.at[i, "club_id"]):
                    df_headers.at[i, "club_id"] = club_id
                return i

            new_row = {
                "player_name": name_lookup.get((match_id, player_id), pd.NA),
                "player_id": player_id,
                "s": s,
                "club": club,
                "club_id": club_id,
                "match_id": match_id,
                "Attacking_corners_headed": 0,
                "Attacking_corners_headed_and_scored": 0,
                "Defending_corners_defended": 0,
                "Defending_corners_errors": 0,
                "Defending_corners_fatal_errors": 0,
                "Attacking_corners_on_pitch": 0,
                "Defending_corners_on_pitch": 0,
                "source_position_file": source_position_file,
                "source_event_file": source_event_file,
            }
            df_headers.loc[len(df_headers)] = new_row
            idx = int(df_headers.index[-1])
            header_idx[key] = idx
            return idx

        def add(idx: int, col: str, inc: int = 1) -> None:
            df_headers.at[idx, col] = int(df_headers.at[idx, col]) + int(inc)

        # only process event files that were newly ingested
        new_event_sources = df_new_full["source_event_file"].dropna().astype(str).unique().tolist()
        corners_processed = 0
        corners_skipped_no_pos = 0

        for ev_file_str in new_event_sources:
            ev_mid = match_id_from_filename(Path(ev_file_str))
            if not ev_mid:
                mids = df_full.loc[df_full["source_event_file"].astype(str) == ev_file_str, "match_id"].dropna().astype(str).unique()
                ev_mid = mids[0] if len(mids) else None
            if not ev_mid:
                continue

            pos_path = pos_by_mid.get(str(ev_mid))
            if not pos_path:
                corners_skipped_no_pos += 1
                continue

            pos_container = read_json(pos_path)
            if not pos_container:
                corners_skipped_no_pos += 1
                continue

            frames = get_positions_frames_scisports(pos_container)
            if not frames:
                corners_skipped_no_pos += 1
                continue
            frame_by_t = build_frame_index(frames)

            df_m = df_full[df_full["source_event_file"].astype(str) == ev_file_str].copy()
            if df_m.empty:
                continue

            df_m["__seq"] = df_m["corner_sequence_id"].apply(lambda x: safe_int(x, -999999))
            df_m["__t0"] = df_m["corner_startTimeMs"].apply(lambda x: safe_int(x, -1))
            df_m = df_m[df_m["__t0"] >= 0].copy()

            # build corner "units":
            # - seq != -1: one per seq_id using earliest t0
            # - seq == -1: one per t0
            units: List[Tuple[int, int]] = []
            non = df_m[df_m["__seq"] != -1]
            for seq_int, g in non.groupby("__seq"):
                units.append((int(seq_int), int(g["__t0"].min())))
            m1 = df_m[df_m["__seq"] == -1]
            for t0, _g in m1.groupby("__t0"):
                units.append((-1, int(t0)))
            units.sort(key=lambda x: x[1])

            for seq_int, t0 in units:
                corners_processed += 1

                if seq_int != -1:
                    df_seq = df_m[df_m["__seq"] == seq_int].copy()
                else:
                    df_seq = df_m[(df_m["__seq"] == -1) & (df_m["__t0"] == t0)].copy()
                if df_seq.empty:
                    continue

                match_id = str(df_seq["match_id"].iloc[0])

                fr0 = frame_by_t.get(int(t0))
                if not isinstance(fr0, dict):
                    continue

                df_corner_start = df_seq[
                    (df_seq.get("possessionTypeName", pd.Series([None] * len(df_seq))).astype(str) == "CORNER")
                    & (df_seq.get("sequenceStart", pd.Series([None] * len(df_seq))).apply(truthy))
                ].copy()

                corner_group = "HOME"
                if not df_corner_start.empty and "groupName" in df_corner_start.columns and df_corner_start["groupName"].dropna().size:
                    corner_group = str(df_corner_start["groupName"].dropna().iloc[0]).upper()
                elif "groupName" in df_seq.columns and df_seq["groupName"].dropna().size:
                    corner_group = str(df_seq["groupName"].dropna().iloc[0]).upper()

                attacking_side_key = "h" if corner_group == "HOME" else "a"
                defending_side_key = "a" if attacking_side_key == "h" else "h"
                defending_group = "AWAY" if corner_group == "HOME" else "HOME"

                atk_players = players_from_frame(fr0, attacking_side_key)
                def_players = players_from_frame(fr0, defending_side_key)

                # on_pitch counts ONCE per unit
                for p in atk_players:
                    pid = str(p.get("p"))
                    idx = get_or_create_row(
                        match_id=match_id,
                        player_id=pid,
                        s=p.get("s"),
                        club=corner_group,
                        club_id=corner_group,
                        source_event_file=ev_file_str,
                        source_position_file=str(pos_path),
                    )
                    add(idx, "Attacking_corners_on_pitch", 1)

                for p in def_players:
                    pid = str(p.get("p"))
                    idx = get_or_create_row(
                        match_id=match_id,
                        player_id=pid,
                        s=p.get("s"),
                        club=defending_group,
                        club_id=defending_group,
                        source_event_file=ev_file_str,
                        source_position_file=str(pos_path),
                    )
                    add(idx, "Defending_corners_on_pitch", 1)

                # defended (first defensive clear/block in unit)
                for _, r in df_seq.iterrows():
                    if is_defensive_clear_or_block(r, defending_group=defending_group):
                        dpid = r.get("playerId")
                        if pd.notna(dpid):
                            idx = get_or_create_row(
                                match_id=match_id,
                                player_id=str(dpid),
                                s=pd.NA,
                                club=defending_group,
                                club_id=defending_group,
                                source_event_file=ev_file_str,
                                source_position_file=str(pos_path),
                            )
                            add(idx, "Defending_corners_defended", 1)
                        break

                # first shot in unit => errors + headers
                df_shots = df_seq[df_seq.apply(is_shot_event, axis=1)].copy()
                if not df_shots.empty:
                    df_shots["__t"] = df_shots["startTimeMs"].apply(lambda x: safe_int(x, -1))
                    df_shots = df_shots[df_shots["__t"] >= 0].sort_values("__t")
                    if not df_shots.empty:
                        shot_row = df_shots.iloc[0]
                        shot_t = int(shot_row["__t"])
                        shooter_id = shot_row.get("playerId")
                        shooter_id = str(shooter_id) if pd.notna(shooter_id) else None

                        fr_shot = frame_by_t.get(shot_t)
                        if isinstance(fr_shot, dict) and shooter_id is not None:
                            atk_now = players_from_frame(fr_shot, attacking_side_key)
                            def_now = players_from_frame(fr_shot, defending_side_key)

                            shooter = next((p for p in atk_now if str(p.get("p")) == shooter_id), None)
                            if shooter and shooter.get("x") is not None and shooter.get("y") is not None:
                                sx, sy = float(shooter["x"]), float(shooter["y"])

                                closest_def = None
                                closest_d = None
                                for dp in def_now:
                                    if dp.get("x") is None or dp.get("y") is None:
                                        continue
                                    d = euclid(sx, sy, float(dp["x"]), float(dp["y"]))
                                    if closest_d is None or d < closest_d:
                                        closest_d = d
                                        closest_def = dp

                                if closest_def is not None:
                                    def_pid = str(closest_def.get("p"))
                                    idx = get_or_create_row(
                                        match_id=match_id,
                                        player_id=def_pid,
                                        s=closest_def.get("s"),
                                        club=defending_group,
                                        club_id=defending_group,
                                        source_event_file=ev_file_str,
                                        source_position_file=str(pos_path),
                                    )
                                    add(idx, "Defending_corners_errors", 1)
                                    if is_goal_event(shot_row):
                                        add(idx, "Defending_corners_fatal_errors", 1)

                            if shot_bodypart_is_head(shot_row):
                                idx = get_or_create_row(
                                    match_id=match_id,
                                    player_id=shooter_id,
                                    s=pd.NA,
                                    club=corner_group,
                                    club_id=corner_group,
                                    source_event_file=ev_file_str,
                                    source_position_file=str(pos_path),
                                )
                                add(idx, "Attacking_corners_headed", 1)
                                if is_goal_event(shot_row):
                                    add(idx, "Attacking_corners_headed_and_scored", 1)

        before_rows = len(pd.read_csv(csv_headers, low_memory=False)) if csv_headers.exists() else 0
        df_headers.to_csv(csv_headers, index=False, encoding="utf-8")
        after_rows = len(df_headers)

        return {
            "ok": True,
            "added_events_all": int(added_all),
            "added_events_full": int(added_full),
            "headers_net_new_rows": int(after_rows - before_rows),
            "corners_processed": int(corners_processed),
            "corners_skipped_no_pos": int(corners_skipped_no_pos),
        }

    except Exception as e:
        return {"ok": False, "error": str(e)}
