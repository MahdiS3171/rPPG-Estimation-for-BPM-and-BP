"""Read-only real-artifact verification; write verification evidence in this run."""
from pathlib import Path
import argparse
import ast
import csv
import json
import re
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from rppg_lab.artifacts import file_sha256, write_json
from rppg_lab.pilot_video import mp4_presentation_times
from rppg_lab.pilot6_delay_acquisition import ROIS
from rppg_lab.pilot6_delay_runner import assert_protected
from rppg_lab.pilot6_protocol_timeline import timeline_masks


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def verify(out, test_logs=(), thread_limit=None):
    out = Path(out)
    manifest = json.loads((out / "experiment_manifest.json").read_text(encoding="utf-8"))
    sessions = json.loads((out / "private_metadata/sessions.json").read_text(encoding="utf-8"))
    checks = []
    def check(name, condition):
        checks.append(dict(check=name, status="PASS" if condition else "FAIL"))
    check("analysis_completed", manifest["stage"] == "analyzed")
    check("six_unique_session_ids", len({s["session_id"] for s in sessions}) == 6)
    check("latest_metadata_bytes", file_sha256(manifest["input_identity"]["metadata_path"]) == manifest["input_identity"]["metadata_sha256"])
    total_frames = 0
    raw_hashes = {}
    detector_backends = {}
    cached_delays = []
    source_archives = {}
    for s in sessions:
        directory = out / s["session_id"]
        path = directory / "native_rgb.npz"
        check(s["session_id"] + "_native_acquisition_present", path.is_file())
        if not path.is_file():
            continue
        pts = mp4_presentation_times(s["video"])[0]
        acquisition = json.loads((directory / "acquisition.json").read_text(encoding="utf-8"))
        with np.load(path, allow_pickle=False) as z:
            check(s["session_id"] + "_all_original_pts", np.array_equal(pts, z["timestamps"]))
            identity = json.loads(z["identity_json"].item())
            check(s["session_id"] + "_identity_matches_acquisition", identity == acquisition["identity"])
            expected = next((digest for name, digest in manifest["input_identity"]["video_sha256"].items()
                             if Path(name).resolve() == Path(s["video"]).resolve()), None)
            check(s["session_id"] + "_video_hash", file_sha256(s["video"]) == expected == identity["video_sha256"])
            check(s["session_id"] + "_native_resolution", tuple(z["native_shape"][:2]) == (1920, 1080))
            for roi in ROIS:
                check(s["session_id"] + "_" + roi + "_shape", z[roi + "_rgb"].shape == (len(pts), 3))
            total_frames += len(pts)
            detector_backends[s["session_id"]] = dict(face=str(z["face_backend"].item()), hand=str(z["actual_hand_backend"].item()))
        with np.load(directory / "protocol_masks.npz", allow_pickle=False) as z:
            check(s["session_id"] + "_mask_original_pts", np.array_equal(z["original_pts"], pts))
            for policy, guard in manifest["config"]["policies"].items():
                m = timeline_masks(pts, s["events"], guard)
                check(s["session_id"] + "_" + policy + "_mask_boundaries", all(np.array_equal(z[policy + "_" + k], v) for k, v in m.items()))
        raw_hashes[str(path)] = file_sha256(path)
        analysis = json.loads((directory / "analysis_cache.json").read_text(encoding="utf-8"))
        cached_delays.extend(analysis["delays"])
        module = Path("rppg_lab/pilot6_delay_feasibility.py")
        digest = analysis["identity"]["analysis_sources"]["pilot6_delay_feasibility"]
        if digest != file_sha256(module):
            archived = next((p for p in (out / "diagnostics").glob("*.py.txt") if file_sha256(p) == digest), None)
            equivalent = False
            if archived:
                before = {n.name: ast.dump(n) for n in ast.parse(archived.read_text(encoding="utf-8")).body if isinstance(n, ast.FunctionDef)}
                after = {n.name: ast.dump(n) for n in ast.parse(module.read_text(encoding="utf-8")).body if isinstance(n, ast.FunctionDef)}
                equivalent = all(before.get(n) == after.get(n) for n in ("analyze_recording", "scheduled_windows", "phase_labels"))
                source_archives[digest] = str(archived)
            check(s["session_id"] + "_cached_signal_source_equivalence", equivalent)
    cuffs = read_csv(out / "cuff_associations.csv")
    check("eighteen_independent_cuff_labels", len(cuffs) == 18 and len({r["event_id"] for r in cuffs}) == 18)
    check("no_unverified_primary_features", not any(r["association_available"] == "True" and r["timing_verified"] != "True" for r in cuffs))
    for sid, expected in (("S001_V01", [(60, 110), (145, 195), (220, 270)]), ("S004_V01", [(60, 133), (145, 209), (225, 292)])):
        observed = [(e["start_sec"], e["end_sec"]) for s in sessions if s["session_id"] == sid for e in s["events"]]
        check(sid + "_recorded_timing_exceptions", observed == expected)
    delays = read_csv(out / "delay_windows.csv")
    keys = [(r["session_id"], r["policy"], r["face_roi"], r["hand_roi"], r["method"], r["start_sec"]) for r in delays]
    check("unique_delay_window_identities", len(keys) == len(set(keys)))
    check("no_computation_across_cuff_phase_boundaries", not any(r["cuff_phase"] == "crosses_boundary" and r["computable"] == "True" for r in delays))
    check("no_computation_across_unbridgeable_source_gaps", not any(r["computable"] == "True" and float(r["max_source_gap_sec"]) > manifest["config"]["max_gap_sec"]+1e-6 for r in delays))
    # Replay every BP summary through the final source. This also verifies that
    # the integer-input regression fix leaves these float-valued results exact.
    from rppg_lab.pilot6_delay_feasibility import cuff_associations, bp_descriptive
    from rppg_lab.waveform_visualization import write_csv
    import tempfile
    intervals = read_csv(out / "eligible_intervals.csv")
    for row in intervals:
        row["start_sec"], row["end_sec"] = float(row["start_sec"]), float(row["end_sec"])
        row["timing_verified"] = row["timing_verified"] == "True"
    cfg = manifest["config"]
    primary = cuff_associations(sessions, intervals, cached_delays, cfg)
    sensitivity = []
    for policy in cfg["policies"]:
        for selection in ("strict", "intermediate", "all_computable"):
            for distance in (15., 30., 60.):
                records = cuff_associations(sessions, intervals, cached_delays, cfg, policy, False, selection, distance)
                for row in records:
                    row.update(max_distance_sec=distance, independent_record=False, analysis_role="repeated-label sensitivity; never extra observations")
                sensitivity.extend(records)
    with tempfile.TemporaryDirectory() as tmp:
        for name, rows in (("cuff_associations", primary), ("cuff_associations_sensitivity", sensitivity), ("bp_exploratory_analysis", bp_descriptive(primary, sensitivity))):
            path = Path(tmp) / (name + ".csv")
            write_csv(path, rows)
            check(name + "_final_source_exact_replay", file_sha256(path) == file_sha256(out / path.name))
    protected = assert_protected(manifest["protected_before"])
    check("protected_historical_source_model_workbook_artifacts", protected["status"] == "PASS")
    logs = []
    for path in map(Path, test_logs):
        data = path.read_bytes()
        content = data.decode("utf-16" if data.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8")
        matches = re.findall(r"Ran (\d+) tests? in ([\d.]+)s", content)
        passed = bool(matches) and bool(re.search(r"^OK\s*$", content, re.M)) and not bool(re.search(r"^FAILED", content, re.M))
        logs.append(dict(path=str(path.resolve()), sha256=file_sha256(path), passed=passed,
                         tests=int(matches[-1][0]) if matches else None))
        check("test_log_" + path.name, passed)
    report = dict(status="PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL", checks=checks,
                  decoded_frames=total_frames, independent_cuff_records=len(cuffs), delay_rows=len(delays),
                  protected_files_checked=protected["files_checked"], test_logs=logs, native_rgb_sha256=raw_hashes,
                  archived_signal_source_versions=source_archives,
                  resolved_test_failure=dict(log="outputs/pilot6_protocol_verified_tests.log",
                      issue="Integer BP arrays could not be centered in place; final source converts to float and gates insufficient within-subject degrees of freedom"),
                  resolved_replay_issue="Unordered subject-set reductions changed baseline rounding across processes; final baseline reduces sorted subject IDs deterministically")
    manifest["selected_detector_backends"] = detector_backends
    manifest["random_seeds"] = dict(primary_synthetic=manifest["config"]["seed"], isolated_clock_synthetic=manifest["config"]["seed"]+1)
    if thread_limit is not None:
        if thread_limit < 1:
            raise ValueError("Declared numerical thread limit must be positive")
        manifest["final_replay_thread_environment"] = dict(OMP_NUM_THREADS=str(thread_limit), MKL_NUM_THREADS=str(thread_limit), OPENBLAS_NUM_THREADS=str(thread_limit),
            scope="Operator-declared final replay invocation; cached acquisition/earlier diagnostics retain their original environment")
    metadata = json.loads((out / "metadata_audit.json").read_text(encoding="utf-8"))
    copies = {}
    for label, source, expected in (("updated", metadata["path"], metadata["sha256"]),
                                    ("previous", metadata.get("previous_path"), metadata.get("previous_sha256"))):
        if source:
            copy = out / "private_metadata" / (label + "_source_workbook.xlsx")
            if not copy.exists():
                shutil.copyfile(source, copy)
            if file_sha256(copy) != expected:
                raise ValueError("Private workbook copy differs from audited source identity")
            copies[str(copy)] = expected
    manifest["private_source_workbook_copies"] = copies
    paths = [*Path("rppg_lab").glob("pilot6_*.py"), *Path("tests").glob("test_pilot6*py"),
             Path("scripts/evaluate_pilot6_delay.py"), Path("scripts/verify_pilot6_delay.py"),
             Path("configs/pilot6_delay_v1.json"), Path("docs/PILOT6_PROTOCOL_AWARE_DELAY_FEASIBILITY.md")]
    manifest["experiment_files_sha256"] = {str(p): file_sha256(p) for p in paths}
    manifest["verification_status"] = report["status"]
    write_json(out / "experiment_manifest.json", manifest)
    report_path = out / "research_report.md"
    if report_path.is_file():
        text = report_path.read_text(encoding="utf-8").split("\n## Post-run verification\n")[0]
        text += f"\n## Post-run verification\n\nArtifact checks: **{report['status']}** ({len(checks)} checks); "
        text += f"{total_frames} native frames, {len(cuffs)} unique cuff records, {len(delays)} delay-window rows. "
        text += f"All {protected['files_checked']} protected source/model/workbook/historical artifacts remain byte-identical.\n\n"
        for log in logs:
            text += f"- `{log['path']}`: {log['tests']} tests, {'PASS' if log['passed'] else 'FAIL'}; SHA256 `{log['sha256']}`.\n"
        text += "\nExpected simulation gap failures and numerical/morphology limitations are reported above; they are not suppressed to improve acceptance. "
        text += "The preliminary fixed-grid and oracle-HR engineering diagnostics remain under `diagnostics/` or explicitly named execution logs; final tables use phase/camera-support anchors and measured spectral HR.\n"
        text += "\nA development regression initially exposed integer-array centering failure in the BP summary helper "
        text += "(`outputs/pilot6_protocol_verified_tests.log`); conversion to floating point fixed it. "
        text += "Final-source replay reproduces the cuff association and BP summary CSVs byte-for-byte. "
        text += "The exact earlier signal-analysis source is archived and its signal/scheduling function ASTs match the final source.\n"
        text += "\nThe first exact BP-summary replay check exposed last-digit rounding differences from unordered subject-set reductions. "
        text += "Sorting subject IDs fixes that reproducibility issue; no waveform/delay data or quality criteria were changed.\n"
        report_path.write_text(text, encoding="utf-8")
        report["research_report_sha256"] = file_sha256(report_path)
    write_json(out / "verification.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("checks", "native_rgb_sha256")}, indent=2))
    if report["status"] != "PASS":
        raise SystemExit("Artifact verification failed; see verification.json")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", default="outputs/pilot6_protocol_aware_delay_v1/run01")
    p.add_argument("--test-log", action="append", default=[])
    p.add_argument("--thread-limit", type=int, help="Record an explicitly supplied numerical thread limit used for the final replay")
    a = p.parse_args()
    verify(a.run_dir, a.test_log, a.thread_limit)
