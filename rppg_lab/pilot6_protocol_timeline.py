"""Read-only, provenance-aware Pilot-6 workbook and original-clock timelines."""
from __future__ import annotations

from datetime import date, datetime, time
from pathlib import Path
import re

import numpy as np
from openpyxl import load_workbook

from .artifacts import file_sha256

SHEET_ID = "1613BKOf1fJsOweixkMVrLQpImywScl4cd-q-0Xq-ZMQ"
DEFAULT_METADATA = "sheets/rPPG_BP_Data_Collection_2026-10-24.xlsx"
REQUIRED_SHEETS = ("00_Protocol", "01_Subjects", "02_Sessions", "03_Review", "04_Windows")
SESSION_HEADERS = ("Session_ID", "Subject_ID", "Video_File_or_Link", "Video_Duration_Reported_s",
                   "FPS_Reported", "Hand_ROI_Side", "Hand_Supported", "Hand_Height_Relative_Heart_cm",
                   "Cuff_Arm", "Event_Time_Source", "Manual_Exclude", "Exclude_Reason", "Operator_Notes",
                   *(f"{prefix}{n}{suffix}" for n in (1, 2, 3) for prefix, suffix in
                     (("BP", "_Start_s"), ("BP", "_End_s"), ("SBP", "_mmHg"),
                      ("DBP", "_mmHg"), ("CuffHR", "_bpm"))))


def _cell(value):
    return value.isoformat() if isinstance(value, (date, datetime, time)) else value


def workbook_snapshot(path):
    """Preserve every formula/raw cell AND the separately cached value, all tabs."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required explicit metadata export missing: {path}; no older fallback")
    books = [load_workbook(path, read_only=True, data_only=flag) for flag in (False, True)]
    try:
        tabs = {}
        for sheet in books[0]:
            raw = [[_cell(v) for v in row] for row in sheet.values]
            cached = [[_cell(v) for v in row] for row in books[1][sheet.title].values]
            header = 3 if sheet.title == "00_Protocol" else 2 if sheet.title in (*REQUIRED_SHEETS[1:], "05_Field_Guide", "92_Migration_Audit") or sheet.title.startswith(("90_", "91_")) else None
            tabs[sheet.title] = dict(raw_cells=raw, cached_cells=cached, header_row_1based=header + 1 if header is not None else None,
                                    headers=raw[header] if header is not None and len(raw) > header else [],
                                    row_count=len(raw), column_count=max(map(len, raw), default=0))
        return dict(path=str(path.resolve()), filename=path.name, sha256=file_sha256(path),
                    declared_google_sheet_id=SHEET_ID, declared_title="rPPG BP Data Collection final",
                    identity_scope="User-declared export lineage; local bytes/hash verified. No live Google verification.",
                    worksheets=tabs)
    finally:
        for book in books:
            book.close()


def table_records(snapshot, sheet, id_column):
    tab = snapshot["worksheets"][sheet]
    headers = tab["headers"]
    named = [h for h in headers if h]
    if len(named) != len(set(named)):
        raise ValueError(f"Duplicate nonempty headers in {sheet}")
    result = {}
    start = tab["header_row_1based"]
    for index, row in enumerate(tab["raw_cells"][start:], start + 1):
        values = {h: v for h, v in zip(headers, row) if h}
        identity = values.get(id_column)
        if identity is None or str(identity).startswith("="):
            continue
        identity = str(identity).strip()
        if identity in result:
            raise ValueError(f"Ambiguous duplicate {id_column}: {identity}")
        result[identity] = dict(source_row=index, raw=values)
    return result


def load_sessions(snapshot, videos):
    missing = set(REQUIRED_SHEETS) - snapshot["worksheets"].keys()
    if missing:
        raise ValueError(f"Missing updated worksheets: {sorted(missing)}")
    headers = snapshot["worksheets"]["02_Sessions"]["headers"]
    if set(SESSION_HEADERS) - set(headers):
        raise ValueError(f"Missing updated session headers: {sorted(set(SESSION_HEADERS) - set(headers))}")
    sessions = table_records(snapshot, "02_Sessions", "Session_ID")
    subjects = table_records(snapshot, "01_Subjects", "Subject_ID")
    resolved, used = [], set()
    for video in videos:
        video = Path(video)
        exact = [(sid, r) for sid, r in sessions.items()
                 if r["raw"].get("Video_File_or_Link") and
                 str(r["raw"]["Video_File_or_Link"]).replace("\\", "/").split("/")[-1].casefold() == video.name.casefold()]
        stem = [(sid, r) for sid, r in sessions.items() if sid.casefold() == video.stem.casefold()]
        candidates = {sid: r for sid, r in exact + stem}
        if len(candidates) != 1:
            raise ValueError(f"Ambiguous/unmatched video {video.name}: {list(candidates)}")
        sid, record = next(iter(candidates.items()))
        raw = record["raw"]
        sub = raw["Subject_ID"]
        if sid in used or sub not in subjects or not re.fullmatch(r"S\d{3}_V\d{2}", sid):
            raise ValueError(f"Conflicting session/subject identity: {sid}")
        if video.stem.upper().startswith("S") and video.stem.upper().split("_")[0] != sub:
            raise ValueError(f"Filename/subject conflict: {sid}")
        if stem and exact and stem[0][0] != exact[0][0]:
            raise ValueError(f"Filename mapping conflicts with Session_ID: {video}")
        if raw.get("Video_File_or_Link") and not exact:
            raise ValueError(f"Explicit video field conflicts with Session_ID stem: {sid}")
        used.add(sid)
        resolved.append(dict(session_id=sid, subject_id=sub, video=str(video.resolve()),
                             identity_source="exact_session_id_and_filename" if stem else "explicit_video_filename",
                             source_row=record["source_row"], raw=raw, subject_raw=subjects[sub]["raw"]))
    return resolved


def metadata_changes(previous, updated):
    """Header and ID-keyed changes; other complete tables compared by coordinate."""
    changes = []
    old, new = previous["worksheets"], updated["worksheets"]
    for name in sorted(old.keys() | new.keys()):
        if name not in old or name not in new:
            changes.append(dict(sheet=name, identity="", field="worksheet", old=name if name in old else None,
                                new=name if name in new else None, change="sheet_added_or_removed"))
            continue
        for field in sorted(set(filter(None, old[name]["headers"])) ^ set(filter(None, new[name]["headers"]))):
            changes.append(dict(sheet=name, identity="header", field=field,
                                old=field if field in old[name]["headers"] else None,
                                new=field if field in new[name]["headers"] else None, change="header_added_or_removed"))
        # Window/Review identities themselves are formulas: compare those raw
        # matrices by coordinate without treating cached formulas as observations.
        key = {"01_Subjects": "Subject_ID", "02_Sessions": "Session_ID"}.get(name)
        if key:
            before, after = table_records(previous, name, key), table_records(updated, name, key)
            for identity in sorted(before.keys() | after.keys()):
                a, b = before.get(identity, {}).get("raw", {}), after.get(identity, {}).get("raw", {})
                for field in sorted(a.keys() | b.keys()):
                    if a.get(field) != b.get(field):
                        changes.append(dict(sheet=name, identity=identity, field=field, old=a.get(field), new=b.get(field), change="raw_cell_changed"))
        else:
            for kind in ("raw_cells", "cached_cells"):
                a, b = old[name][kind], new[name][kind]
                for i in range(max(len(a), len(b))):
                    ar, br = a[i] if i < len(a) else [], b[i] if i < len(b) else []
                    for j in range(max(len(ar), len(br))):
                        av, bv = ar[j] if j < len(ar) else None, br[j] if j < len(br) else None
                        if av != bv:
                            changes.append(dict(sheet=name, identity=f"row{i+1}", field=f"column{j+1}", old=av, new=bv, change=kind + "_changed"))
    return changes


def _number(value, name):
    if value is None or value == "":
        return None
    if isinstance(value, str) and value.startswith("="):
        raise ValueError(f"Source field {name} must be a recorded value, not a formula")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"Nonfinite {name}")
    return result


def cuff_events(session, end_sec):
    raw, result, previous_end, previous_start = session["raw"], [], None, None
    # Only explicit, actually reviewed provenance can assert verification.
    verified = raw.get("Event_Time_Source") == "Video_review"
    for index in (1, 2, 3):
        a, b = [_number(raw.get(f"BP{index}_{part}_s"), f"BP{index}_{part}_s") for part in ("Start", "End")]
        if a is not None and (a < 0 or a >= end_sec) or b is not None and (b <= 0 or b > end_sec):
            raise ValueError(f"Cuff bounds outside video: {session['session_id']} BP{index}")
        if a is not None and b is not None and a >= b:
            raise ValueError("Invalid cuff temporal ordering")
        if a is not None and previous_start is not None and a <= previous_start:
            raise ValueError("Nonmonotone cuff starts")
        if a is not None and previous_end is not None and a < previous_end:
            raise ValueError("Overlapping cuff events")
        sbp, dbp = [_number(raw.get(f"{part}{index}_mmHg"), part) for part in ("SBP", "DBP")]
        if sbp is not None and dbp is not None and not 0 < dbp < sbp:
            raise ValueError("Invalid source BP ordering")
        result.append(dict(session_id=session["session_id"], subject_id=session["subject_id"],
                           event_id=f"{session['session_id']}_BP{index}", measurement_index=index,
                           start_sec=a, end_sec=b, sbp_mmHg=sbp, dbp_mmHg=dbp,
                           cuff_hr_bpm=_number(raw.get(f"CuffHR{index}_bpm"), "cuff HR"),
                           event_time_source=raw.get("Event_Time_Source"), timing_verified=verified and a is not None and b is not None,
                           missing_start=a is None, missing_end=b is None,
                           timing_flag="missing_endpoint" if a is None or b is None else "video_reviewed" if verified else "unverified_legacy_bounds",
                           source_row=session["source_row"], label_semantics="one intermittent oscillometric cuff measurement"))
        previous_end, previous_start = b, a if a is not None else previous_start
    return result


def timeline_masks(timestamps, events, guard_sec=5.):
    t = np.asarray(timestamps, float)
    if t.ndim != 1 or not len(t) or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Original strictly increasing PTS required; no concatenation or sorting")
    if not np.isfinite(guard_sec) or guard_sec < 0:
        raise ValueError("Invalid guard")
    active, guarded, unknown = [np.zeros(len(t), bool) for _ in range(3)]
    verified = True
    for event in events:
        a, b = event["start_sec"], event["end_sec"]
        verified &= event["timing_verified"]
        if a is None or b is None:
            # Unknown ends are never replaced by a typical duration or next start.
            unknown |= (t >= a - guard_sec) if a is not None else (t <= b + guard_sec) if b is not None else True
        else:
            active |= (t >= a) & (t < b)
            guarded |= (t >= a - guard_sec) & (t < b + guard_sec)
    candidate = ~(guarded | unknown)
    return dict(cuff_active=active, cuff_boundary_guard=guarded & ~active,
                candidate_cuff_free=candidate, verified_cuff_free=candidate & verified,
                timing_uncertain=unknown | (not verified))


def candidate_intervals(events, start_sec, end_sec, guard_sec):
    blocked = []
    for event in events:
        a, b = event["start_sec"], event["end_sec"]
        blocked.append((max(start_sec, (a - guard_sec) if a is not None else start_sec),
                        min(end_sec, (b + guard_sec) if b is not None else end_sec)))
    cursor, intervals = start_sec, []
    for a, b in sorted(blocked):
        if a > cursor:
            intervals.append((cursor, a))
        cursor = max(cursor, b)
    if cursor < end_sec:
        intervals.append((cursor, end_sec))
    return intervals
