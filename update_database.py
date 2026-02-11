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
    
def upsert_by_match_id(df: pd.DataFrame, match_id: str, new_rows: pd.DataFrame, match_id_col: str = "match_id") -> pd.DataFrame:
    if df is None or df.empty:
        return new_rows.copy()
    if match_id_col not in df.columns:
        # If structure differs, fail loudly so you notice
        raise ValueError(f"CSV missing required column: {match_id_col}")

    df_old_removed = df[df[match_id_col].astype(str) != str(match_id)].copy()
    out = pd.concat([df_old_removed, new_rows], ignore_index=True)
    return out


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

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


def upsert_df_by_match_id(
    df: pd.DataFrame,
    match_id: str,
    new_rows: pd.DataFrame,
    match_id_col: str = "match_id",
) -> pd.DataFrame:
    """
    Replace existing rows for match_id, then append new_rows.
    If new_rows is empty -> keep df unchanged.
    """
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

    # drop old match
    df = df[df[match_id_col] != match_id].copy()

    # append new
    new_rows = new_rows.copy()
    if match_id_col in new_rows.columns:
        new_rows[match_id_col] = new_rows[match_id_col].astype(str)

    return pd.concat([df, new_rows], ignore_index=True)


def git_commit_and_push_csvs(
    *,
    repo_root: Path,
    files_to_commit: List[Path],
    commit_message: str,
) -> Tuple[bool, str]:
    """
    Commits + pushes updated CSVs to GitHub.

    Requires env vars (recommended):
      - GITHUB_TOKEN  (fine-grained PAT)
      - GITHUB_REPO   (e.g. "yourname/yourrepo")
      - GITHUB_BRANCH (e.g. "main")

    The repo_root must be inside a git repo.
    """
    token = os.getenv("GITHUB_TOKEN", "").strip()
    repo = os.getenv("GITHUB_REPO", "").strip()
    branch = os.getenv("GITHUB_BRANCH", "main").strip()

    if not token or not repo:
        return False, "Missing GITHUB_TOKEN or GITHUB_REPO env vars"

    try:
        # Ensure we are in repo
        repo_root = Path(repo_root).resolve()

        # Configure git identity (safe defaults)
        subprocess.check_call(["git", "-C", str(repo_root), "config", "user.email", "bot@streamlit.local"])
        subprocess.check_call(["git", "-C", str(repo_root), "config", "user.name", "streamlit-bot"])

        # Set authenticated remote URL (overwrites origin)
        remote_url = f"https://{token}@github.com/{repo}.git"
        subprocess.check_call(["git", "-C", str(repo_root), "remote", "set-url", "origin", remote_url])

        # Checkout branch (best effort)
        subprocess.call(["git", "-C", str(repo_root), "checkout", branch])

        # Add files
        for f in files_to_commit:
            subprocess.check_call(["git", "-C", str(repo_root), "add", str(Path(f).resolve())])

        # Commit (if nothing changed, git returns non-zero)
        commit_rc = subprocess.call(["git", "-C", str(repo_root), "commit", "-m", commit_message])
        if commit_rc != 0:
            # No changes to commit
            return True, "No changes to commit (already up to date)"

        # Push
        subprocess.check_call(["git", "-C", str(repo_root), "push", "origin", branch])
        return True, "Pushed to GitHub"
    except Exception as e:
        return False, str(e)


def update_database(*, uploads_dir: Path, data_dir: Path) -> Dict[str, Any]:
    """
    Updates:
      - data/corner_events_all_matches.csv
      - data/corner_events_full_sequences.csv
      - data/corner_positions_headers.csv
    using JSONs from uploads_dir.

    NEW behavior:
      - If uploaded match_id already exists -> REPLACE that match everywhere (events + sequences + headers).
      - If new match -> APPEND normally.
      - After writing CSVs -> commit+push to GitHub (env vars required).
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

        # Load existing CSVs into memory (so we can UPSERT)
        df_events_all = pd.read_csv(csv_events_all, low_memory=False).where(pd.notnull, None)
        df_events_full = pd.read_csv(csv_events_full, low_memory=False).where(pd.notnull, None)
        df_headers_existing = pd.read_csv(csv_headers, low_memory=False).where(pd.notnull, None)

        existing_match_ids_events_all = (
            set(df_events_all["match_id"].astype(str)) if ("match_id" in df_events_all.columns and not df_events_all.empty) else set()
        )
        existing_match_ids_events_full = (
            set(df_events_full["match_id"].astype(str)) if ("match_id" in df_events_full.columns and not df_events_full.empty) else set()
        )
        existing_match_ids_headers = (
            set(df_headers_existing["match_id"].astype(str)) if ("match_id" in df_headers_existing.columns and not df_headers_existing.empty) else set()
        )

        # detect uploaded files (IMPORTANT: we do NOT filter by existing, because we want to replace too)
        event_files = iter_files_recursive(uploads_dir, _EVENTS_GLOB)
        pos_files = iter_files_recursive(uploads_dir, _POSITIONS_GLOB)

        # Map uploaded event/pos files by match_id (choose newest if multiple)
        ev_by_mid: Dict[str, Path] = {}
        for p in event_files:
            mid = match_id_from_filename(p)
            if not mid:
                # fallback: try reading json to find mid
                c = read_json(p)
                if c:
                    mid = match_id_from_json(c)
            if not mid:
                continue
            mid = str(mid)
            if mid not in ev_by_mid:
                ev_by_mid[mid] = p
            else:
                # pick newest file
                if p.stat().st_mtime > ev_by_mid[mid].stat().st_mtime:
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
            if mid not in pos_by_mid:
                pos_by_mid[mid] = p
            else:
                if p.stat().st_mtime > pos_by_mid[mid].stat().st_mtime:
                    pos_by_mid[mid] = p

        # The matches we will update are those with an uploaded events file
        uploaded_match_ids = sorted(ev_by_mid.keys())

        if not uploaded_match_ids:
            return {"ok": True, "added_events_all": 0, "added_events_full": 0, "headers_net_new_rows": 0, "note": "No event files found"}

        replaced_matches = 0
        added_matches = 0

        # 1) Build NEW rows for events_all + events_full for uploaded matches
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

            # reporting
            existed_before = (
                (mid2 in existing_match_ids_events_all)
                or (mid2 in existing_match_ids_events_full)
                or (mid2 in existing_match_ids_headers)
            )
            if existed_before:
                replaced_matches += 1
            else:
                added_matches += 1

        df_new_all = pd.DataFrame(rows_all)
        df_new_full = pd.DataFrame(rows_full)

        # If nothing parsed, stop early
        if df_new_all.empty and df_new_full.empty:
            return {"ok": True, "added_events_all": 0, "added_events_full": 0, "headers_net_new_rows": 0, "note": "No corners parsed"}

        # 1b) UPSERT events by match_id (replace then append)
        # (group-by match_id so we only drop once per match)
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

        # Write events CSVs
        df_events_all.to_csv(csv_events_all, index=False, encoding="utf-8")
        df_events_full.to_csv(csv_events_full, index=False, encoding="utf-8")

        # 2) HEADERS: recompute for uploaded matches, replacing only those match_ids (when positions exist)
        # Load fresh from updated events_full (so headers use final data)
        df_full = df_events_full.where(pd.notnull(df_events_full), None).copy()
        df_headers = df_headers_existing.where(pd.notnull(df_headers_existing), None).copy()

        # We will only replace headers for matches where we have an uploaded positions file
        mids_with_pos = {mid for mid in uploaded_match_ids if str(mid) in pos_by_mid}
        mids_without_pos = [mid for mid in uploaded_match_ids if str(mid) not in pos_by_mid]

        # Drop old headers for matches we will recompute (true replace)
        if not df_headers.empty and "match_id" in df_headers.columns and mids_with_pos:
            df_headers["match_id"] = df_headers["match_id"].astype(str)
            df_headers = df_headers[~df_headers["match_id"].isin({str(m) for m in mids_with_pos})].copy()

        # Rebuild headers by running your existing logic, but only for those matches
        # (We reuse your original code structure almost verbatim)
        before_rows = len(df_headers)

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

        corners_processed = 0
        corners_skipped_no_pos = 0

        # We only process matches in mids_with_pos (true replace)
        for mid in sorted(mids_with_pos):
            ev_path = ev_by_mid[str(mid)]
            pos_path = pos_by_mid[str(mid)]

            ev_file_str = str(ev_path)

            pos_container = read_json(pos_path)
            if not pos_container:
                corners_skipped_no_pos += 1
                continue

            frames = get_positions_frames_scisports(pos_container)
            if not frames:
                corners_skipped_no_pos += 1
                continue
            frame_by_t = build_frame_index(frames)

            df_m = df_full[df_full["match_id"].astype(str) == str(mid)].copy()
            if df_m.empty:
                continue

            df_m["__seq"] = df_m["corner_sequence_id"].apply(lambda x: safe_int(x, -999999))
            df_m["__t0"] = df_m["corner_startTimeMs"].apply(lambda x: safe_int(x, -1))
            df_m = df_m[df_m["__t0"] >= 0].copy()

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

        df_headers.to_csv(csv_headers, index=False, encoding="utf-8")
        after_rows = len(df_headers)
        headers_net_new_rows = int(after_rows - before_rows)

        # 3) Push to GitHub
        # repo_root should be the git repo root. If update_database.py lives in repo, data_dir.parent is often fine.
        repo_root = Path(".").resolve()
        ok_push, push_msg = git_commit_and_push_csvs(
            repo_root=repo_root,
            files_to_commit=[csv_events_all, csv_events_full, csv_headers],
            commit_message=f"Update database CSVs ({len(uploaded_match_ids)} match(es))",
        )

        return {
            "ok": True,
            "added_events_all": int(added_all_rows),
            "added_events_full": int(added_full_rows),
            "headers_net_new_rows": int(headers_net_new_rows),
            "corners_processed": int(corners_processed),
            "corners_skipped_no_pos": int(corners_skipped_no_pos),
            "uploaded_matches": len(uploaded_match_ids),
            "added_matches": int(added_matches),
            "replaced_matches": int(replaced_matches),
            "matches_missing_positions": mids_without_pos,
            "github_push_ok": bool(ok_push),
            "github_push_msg": push_msg,
        }

    except Exception as e:
        return {"ok": False, "error": str(e)}
