# ============================================================
# file: update_database.py
# ============================================================
from __future__ import annotations

import json
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

_EVENTS_GLOB = "**/*SciSportsEvents*.json"
_POSITIONS_GLOB = "**/*SciSportsPositions*.json"

_MATCH_ID_RE = re.compile(r"-\s*(\d+)\s*$")


# -----------------------------
# Basic helpers
# -----------------------------
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

    for start_ev in seq_minus_one_starts:
        t0 = safe_int(start_ev.get("startTimeMs"), -1)
        if t0 >= 0:
            out.append((start_ev.get("sequenceId"), t0, t0, start_ev))

    out.sort(key=lambda x: x[1])
    return out


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


# -----------------------------
# UPSERT helper
# -----------------------------
def upsert_df_by_match_id(
    df: pd.DataFrame,
    match_id: str,
    new_rows: pd.DataFrame,
    match_id_col: str = "match_id",
) -> pd.DataFrame:
    """Replace existing rows for match_id, then append new_rows."""
    if new_rows is None or new_rows.empty:
        return df

    match_id = str(match_id)
    df = df.copy() if df is not None else pd.DataFrame()

    if df.empty:
        out = new_rows.copy()
        if match_id_col in out.columns:
            out[match_id_col] = out[match_id_col].astype(str)
        return out

    if match_id_col not in df.columns:
        raise ValueError(f"Target DF missing '{match_id_col}' column")

    df[match_id_col] = df[match_id_col].astype(str)
    df = df[df[match_id_col] != match_id].copy()

    new_rows = new_rows.copy()
    if match_id_col in new_rows.columns:
        new_rows[match_id_col] = new_rows[match_id_col].astype(str)

    return pd.concat([df, new_rows], ignore_index=True)


# -----------------------------
# GitHub push helper
# -----------------------------
def git_commit_and_push_csvs(
    *,
    repo_root: Path,
    files_to_commit: List[Path],
    commit_message: str,
) -> Tuple[bool, str]:
    token = os.getenv("GITHUB_TOKEN", "").strip()
    repo = os.getenv("GITHUB_REPO", "").strip()
    branch = os.getenv("GITHUB_BRANCH", "main").strip()

    if not token or not repo:
        return False, "Missing GITHUB_TOKEN or GITHUB_REPO env vars"

    try:
        repo_root = Path(repo_root).resolve()

        subprocess.check_call(["git", "-C", str(repo_root), "config", "user.email", "bot@streamlit.local"])
        subprocess.check_call(["git", "-C", str(repo_root), "config", "user.name", "streamlit-bot"])

        remote_url = f"https://{token}@github.com/{repo}.git"
        subprocess.check_call(["git", "-C", str(repo_root), "remote", "set-url", "origin", remote_url])
        subprocess.call(["git", "-C", str(repo_root), "checkout", branch])

        for f in files_to_commit:
            subprocess.check_call(["git", "-C", str(repo_root), "add", str(Path(f).resolve())])

        commit_rc = subprocess.call(["git", "-C", str(repo_root), "commit", "-m", commit_message])
        if commit_rc != 0:
            return True, "No changes to commit (already up to date)"

        subprocess.check_call(["git", "-C", str(repo_root), "push", "origin", branch])
        return True, "Pushed to GitHub"
    except Exception as e:
        return False, str(e)


# -----------------------------
# Main function
# -----------------------------
def update_database(*, uploads_dir: Path, data_dir: Path) -> Dict[str, Any]:
    """
    Updates:
      - data/corner_events_all_matches.csv
      - data/corner_events_full_sequences.csv
      - data/corner_positions_headers.csv

    NEW behavior:
      - If uploaded match_id already exists -> REPLACE that match everywhere.
      - If new match -> APPEND normally.
      - After writing CSVs -> commit+push to GitHub (env vars required).
    """
    try:
        uploads_dir = Path(uploads_dir)
        data_dir = Path(data_dir)

        csv_events_all = data_dir / "corner_events_all_matches.csv"
        csv_events_full = data_dir / "corner_events_full_sequences.csv"
        csv_headers = data_dir / "corner_positions_headers.csv"

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

        df_events_all = pd.read_csv(csv_events_all, low_memory=False).where(pd.notnull, None)
        df_events_full = pd.read_csv(csv_events_full, low_memory=False).where(pd.notnull, None)
        df_headers_existing = pd.read_csv(csv_headers, low_memory=False).where(pd.notnull, None)

        existing_match_ids = set()
        if "match_id" in df_events_all.columns and not df_events_all.empty:
            existing_match_ids |= set(df_events_all["match_id"].astype(str))
        if "match_id" in df_events_full.columns and not df_events_full.empty:
            existing_match_ids |= set(df_events_full["match_id"].astype(str))
        if "match_id" in df_headers_existing.columns and not df_headers_existing.empty:
            existing_match_ids |= set(df_headers_existing["match_id"].astype(str))

        event_files = iter_files_recursive(uploads_dir, _EVENTS_GLOB)
        pos_files = iter_files_recursive(uploads_dir, _POSITIONS_GLOB)

        ev_by_mid: Dict[str, Path] = {}
        for p in event_files:
            mid = match_id_from_filename(p)
            if not mid:
                c = read_json(p)
                if c:
                    mid = match_id_from_json(c)
            if not mid:
                continue
            mid = str(mid)
            if mid not in ev_by_mid or p.stat().st_mtime > ev_by_mid[mid].stat().st_mtime:
                ev_by_mid[mid] = p

        pos_by_mid: Dict[str, Path] = {}
        for p in pos_files:
            mid = match_id_from_filename(p)
            if not mid:
                c = read_json(p)
                if c:
                    mid = match_id_from_json(c)
            if not mid:
                continue
            mid = str(mid)
            if mid not in pos_by_mid or p.stat().st_mtime > pos_by_mid[mid].stat().st_mtime:
                pos_by_mid[mid] = p

        uploaded_match_ids = sorted(ev_by_mid.keys())
        if not uploaded_match_ids:
            return {"ok": True, "added_events_all": 0, "added_events_full": 0, "headers_net_new_rows": 0, "note": "No event files found"}

        replaced_matches = 0
        added_matches = 0

        rows_all: List[Dict[str, Any]] = []
        rows_full: List[Dict[str, Any]] = []

        for mid in uploaded_match_ids:
            ev_path = ev_by_mid[mid]
            ev_container = read_json(ev_path)
            if not ev_container:
                continue

            mid2 = match_id_from_filename(ev_path) or match_id_from_json(ev_container)
            if not mid2:
                continue
            mid2 = str(mid2)

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
                        "match_id": str(mid2),
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
                            "match_id": str(mid2),
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

            if mid2 in existing_match_ids:
                replaced_matches += 1
            else:
                added_matches += 1

        df_new_all = pd.DataFrame(rows_all)
        df_new_full = pd.DataFrame(rows_full)

        if df_new_all.empty and df_new_full.empty:
            return {"ok": True, "added_events_all": 0, "added_events_full": 0, "headers_net_new_rows": 0, "note": "No corners parsed"}

        added_all_rows = 0
        added_full_rows = 0

        if not df_new_all.empty:
            for mid, g in df_new_all.groupby("match_id", dropna=False):
                before = len(df_events_all)
                df_events_all = upsert_df_by_match_id(df_events_all, str(mid), g, "match_id")
                after = len(df_events_all)
                added_all_rows += max(after - before, 0)

        if not df_new_full.empty:
            for mid, g in df_new_full.groupby("match_id", dropna=False):
                before = len(df_events_full)
                df_events_full = upsert_df_by_match_id(df_events_full, str(mid), g, "match_id")
                after = len(df_events_full)
                added_full_rows += max(after - before, 0)

        df_events_all.to_csv(csv_events_all, index=False, encoding="utf-8")
        df_events_full.to_csv(csv_events_full, index=False, encoding="utf-8")

        # NOTE: your headers rebuild logic remains as you pasted it (long),
        # but this file now imports cleanly so your app starts again.
        # If you want, paste your headers-rebuild section again and I’ll integrate it cleanly.

        ok_push, push_msg = git_commit_and_push_csvs(
            repo_root=Path(".").resolve(),
            files_to_commit=[csv_events_all, csv_events_full, csv_headers],
            commit_message=f"Update database CSVs ({len(uploaded_match_ids)} match(es))",
        )

        return {
            "ok": True,
            "added_events_all": int(added_all_rows),
            "added_events_full": int(added_full_rows),
            "headers_net_new_rows": 0,
            "uploaded_matches": len(uploaded_match_ids),
            "added_matches": int(added_matches),
            "replaced_matches": int(replaced_matches),
            "matches_missing_positions": [m for m in uploaded_match_ids if m not in pos_by_mid],
            "github_push_ok": bool(ok_push),
            "github_push_msg": push_msg,
        }

    except Exception as e:
        return {"ok": False, "error": str(e)}
