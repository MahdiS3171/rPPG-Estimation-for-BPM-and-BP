"""Protocol-aware Pilot-6 orchestration and immutable input identities."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

from .artifacts import file_sha256, write_json, provenance
from .pilot6_protocol_timeline import (workbook_snapshot, load_sessions, metadata_changes,
                                      cuff_events, candidate_intervals, timeline_masks)
from .pilot_video import mp4_presentation_times
from .waveform_visualization import write_csv


def protected_inventory():
    tracked = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    paths = [Path(p) for p in tracked if Path(p).is_file()]
    for root in ("outputs/pilot6_dual_roi", "outputs/pilot6_frozen_native_aa_v1", "checkpoints", "sheets"):
        paths.extend(p for p in Path(root).rglob("*") if p.is_file())
    return {str(p): file_sha256(p) for p in sorted(set(paths))}


def assert_protected(before):
    changed = [p for p, digest in before.items() if not Path(p).is_file() or file_sha256(p) != digest]
    if changed:
        raise RuntimeError(f"Protected artifacts changed: {changed}")
    return dict(status="PASS", files_checked=len(before), changed=[])


def _acquire_job(video, out, config_dict, face_root, resume):
    from .config import PipelineConfig
    from .pilot6_delay_acquisition import acquire
    acquire(video, out, PipelineConfig(**config_dict), face_root, resume)
    return dict(video=video, status="complete")


def run(args):
    from .pilot_frozen import DEFAULT_CHECKPOINT, OFFICIAL_MODEL_SHA256, OFFICIAL_MANIFEST_SHA256
    from .waveform_inference import FrozenWaveformExtractor
    out = Path(args.out_dir)
    namespace = Path("outputs/pilot6_protocol_aware_delay_v1").resolve()
    if namespace not in out.resolve().parents:
        raise ValueError(f"Outputs must be a new run beneath {namespace}")
    if args.workers < 1:
        raise ValueError("workers must be positive")
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if set(config["policies"]) != {"A", "B", "C"} or any(not np.isfinite(v) or v < 0 for v in config["policies"].values()):
        raise ValueError("Need three explicit nonnegative guard policies A/B/C")
    if config["window_sec"] < 6 or config["stride_sec"] <= 0 or config["window_sec"] % config["stride_sec"]:
        raise ValueError("Need >=6s windows and an integral window/stride ratio for fixed independent subsets")
    # Missing current metadata fails before output reservation, never falls back.
    snapshot = workbook_snapshot(args.metadata_path)
    previous = workbook_snapshot(args.previous_metadata_path) if Path(args.previous_metadata_path).is_file() else None
    videos = [Path(args.video_dir) / f"S{i:03d}_V01.mp4" for i in range(1, 7)]
    missing = [str(v) for v in videos if not v.is_file()]
    available = [v for v in videos if v.is_file()]
    sessions = load_sessions(snapshot, available)
    checkpoint = Path(DEFAULT_CHECKPOINT)
    manifest_path = checkpoint.with_name("waveform_v1_freeze_manifest.json")
    frozen = None
    if checkpoint.is_file() and manifest_path.is_file():
        frozen = FrozenWaveformExtractor(checkpoint, manifest_path)
        if frozen.model_sha256 != OFFICIAL_MODEL_SHA256 or file_sha256(manifest_path) != OFFICIAL_MANIFEST_SHA256:
            raise ValueError("Original Frozen V1 identity mismatch")
        extraction_config = frozen.extraction_config
    else:
        from .config import PipelineConfig
        extraction_config = PipelineConfig(regions=("face",), face_roi="forehead", face_rois=("forehead", "left_cheek", "right_cheek"))
    identity = dict(metadata_path=str(Path(args.metadata_path).resolve()), metadata_sha256=snapshot["sha256"],
                    config_sha256=file_sha256(args.config),
                    video_sha256={str(v): file_sha256(v) for v in available},
                    checkpoint_sha256=frozen.model_sha256 if frozen else None,
                    manifest_sha256=file_sha256(manifest_path) if manifest_path.is_file() else None)
    if out.exists():
        if not args.resume:
            raise FileExistsError("Run already exists; select a fresh run or explicitly resume its matching inputs")
        initial = json.loads((out / "experiment_manifest.json").read_text(encoding="utf-8"))
        if identity != initial["input_identity"]:
            raise ValueError("Resume input identities changed; select a new run")
    else:
        out.mkdir(parents=True, exist_ok=False)
        initial = dict(input_identity=identity, runtime=provenance(), python=sys.version, platform=platform.platform(),
                       config=config, output_directory=str(out.resolve()), random_seed=config["seed"],
                       protected_before=protected_inventory(), missing_inputs=missing,
                       model_role="Optional face-only original API + accepted external V2 adapter. Never a hand or timing model.",
                       inclusion="Prespecified label-free engineering gates; all computable and failed windows retained",
                       timebase="Original MP4 display PTS with supported edit lists; bounded linear native-rate regularization per contiguous run",
                       stage="initialized")
        from importlib.metadata import version
        initial["dependency_versions"] = {p: version(p) for p in ("numpy", "scipy", "pandas", "openpyxl", "mediapipe", "opencv-python", "torch", "matplotlib", "scikit-learn")}
        initial["frozen_face_inference_requested"] = args.frozen_face
        initial["external_frozen_adapter"] = dict(path="configs/pilot6_frozen_timebase_v2.json", sha256=file_sha256("configs/pilot6_frozen_timebase_v2.json"))
        write_json(out / "experiment_manifest.json", initial)
    private = out / "private_metadata"
    private.mkdir(exist_ok=True)
    write_json(private / "updated_raw_and_cached.json", snapshot)
    if previous:
        write_json(private / "previous_raw_and_cached.json", previous)
    changes = metadata_changes(previous, snapshot) if previous else []
    write_csv(out / "metadata_changes.csv", changes)
    audit = {k: v for k, v in snapshot.items() if k != "worksheets"}
    audit.update(worksheets={n: {k: v for k, v in t.items() if k not in ("raw_cells", "cached_cells")}
                            for n, t in snapshot["worksheets"].items()},
                 previous_path=previous["path"] if previous else None,
                 previous_sha256=previous["sha256"] if previous else None,
                 change_count=len(changes), matches=[{k: v for k, v in s.items() if k not in ("raw", "subject_raw")} for s in sessions],
                 missing_inputs=missing, formulas_role="03_Review and 04_Windows preserved, never independent evidence")
    audit["geometry_provenance_note"] = ("Updated subject cells contain positive surface distances for all six, while the protocol preamble still says S005 geometry will be completed later. "
                                         "This annotation/cell discrepancy is unresolved; distances are preserved as unverified raw metadata and never used for delay or BP prediction/PWV.")
    write_json(out / "metadata_audit.json", audit)
    all_events, timelines, intervals = [], [], []
    for session in sessions:
        t, clock = mp4_presentation_times(session["video"])
        events = cuff_events(session, float(t[-1]))
        all_events.extend(events)
        session["events"] = events
        session["pts_start_sec"], session["pts_end_sec"] = float(t[0]), float(t[-1])
        session["manual_excluded"] = str(session["raw"].get("Manual_Exclude", "No")).casefold() in ("yes", "true", "1")
        s_out = out / session["session_id"]
        s_out.mkdir(exist_ok=True)
        masks = {"original_pts": t}
        for policy, guard in config["policies"].items():
            m = timeline_masks(t, events, guard)
            masks.update({policy + "_" + k: v for k, v in m.items()})
            for index, (a, b) in enumerate(candidate_intervals(events, float(t[0]), float(t[-1]), guard)):
                intervals.append(dict(session_id=session["session_id"], subject_id=session["subject_id"], policy=policy,
                                      interval_id=f"{session['session_id']}_{policy}_{index}", start_sec=a, end_sec=b, duration_sec=b-a,
                                      timing_verified=all(e["timing_verified"] for e in events),
                                      event_time_source=session["raw"].get("Event_Time_Source"),
                                      manual_excluded=session["manual_excluded"], role="provisional_recomputed_from_source",
                                      signal_window_possible=b-a >= config["window_sec"]+2*config["filter_edge_sec"],
                                      reasons="manual_exclude" if session["manual_excluded"] else
                                      "insufficient_contiguous_duration_including_filter_edges" if b-a < config["window_sec"]+2*config["filter_edge_sec"] else
                                      "unverified_event_timing" if not all(e["timing_verified"] for e in events) else ""))
            dt = np.r_[np.diff(t), 0.]
            timelines.append(dict(session_id=session["session_id"], policy=policy, guard_sec=guard,
                                  source_frames=len(t), pts_start_sec=t[0], pts_end_sec=t[-1],
                                  **{k + "_sec": float(np.sum(dt[v])) for k, v in m.items()},
                                  event_time_source=session["raw"].get("Event_Time_Source")))
        np.savez_compressed(s_out / "protocol_masks.npz", **masks)
    write_csv(out / "cuff_events.csv", all_events)
    write_csv(out / "session_timeline.csv", timelines)
    write_csv(out / "eligible_intervals.csv", intervals)
    write_json(private / "sessions.json", sessions)
    if args.stage in ("all", "acquire"):
        failures = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs = {pool.submit(_acquire_job, s["video"], str(out / s["session_id"]), extraction_config.to_dict(),
                                args.face_cache_root, args.resume): s for s in sessions}
            for job in as_completed(jobs):
                session = jobs[job]
                try:
                    print(json.dumps(job.result()), flush=True)
                except Exception as exc:
                    failure = dict(session_id=session["session_id"], error_type=type(exc).__name__, detail=str(exc))
                    failures.append(failure)
                    write_json(out / session["session_id"] / "failure.json", failure)
                    print(json.dumps(failure), flush=True)
        initial["acquisition_failures"] = failures
        initial["stage"] = "acquired"
        write_json(out / "experiment_manifest.json", initial)
    if args.stage in ("all", "analyze"):
        from .pilot6_delay_feasibility import analyze_cohort
        analyze_cohort(out, sessions, intervals, config, frozen if args.frozen_face else None, args.workers)
        initial["stage"] = "analyzed"
    from importlib.metadata import version
    initial["dependency_versions"] = {p: version(p) for p in ("numpy", "scipy", "pandas", "openpyxl", "mediapipe", "opencv-python", "torch", "matplotlib", "scikit-learn")}
    initial["frozen_face_inference_requested"] = args.frozen_face
    initial["frozen_identity_status"] = "official_pair_verified" if frozen else "unavailable_optional_branch_blocked"
    initial["window_schedule"] = "v1: phase/camera-gap anchors plus supplemental global grid; primary phase anchors advance without overlap before quality filtering"
    initial["external_frozen_adapter"] = dict(path="configs/pilot6_frozen_timebase_v2.json", sha256=file_sha256("configs/pilot6_frozen_timebase_v2.json"))
    initial["protected_after"] = assert_protected(initial["protected_before"])
    initial["analysis_runtime"] = provenance()
    write_json(out / "experiment_manifest.json", initial)
    print(json.dumps(dict(output=str(out.resolve()), stage=initial["stage"], missing_inputs=missing)), flush=True)
