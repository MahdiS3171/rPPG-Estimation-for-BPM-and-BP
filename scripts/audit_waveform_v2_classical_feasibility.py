"""Fixed Gate 08 classical study; locked label-blind predictions precede scoring."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import time
import tracemalloc

import numpy as np
import psutil

from rppg_lab.waveform_v2_scamps import (ScampsRecording, discover_scamps, read_official_split,
                                       source_sha256, _check_path)
from rppg_lab.waveform_v2_classical_cache import (cache_manifest, load_verified_baseline,
                                               read_reference_windows)
from rppg_lab.waveform_v2_classical_study import (METHODS, CONDITIONS, STARTS, prediction_rows,
    score_waveform, aggregate, quality_associations, run_controls, timing_controls, array_sha256)
from rppg_lab.waveform_v2_classical_plots import make_plots

PROTOCOL_COMMIT = "a100379c238b15b5e0f2b5d55b56de4e0520cc78"
PARENT_COMMIT = "4b3d01a3a8e7ef0354b9c5b5f73ad23df9f64675"
PROTOCOL = "docs/WAVEFORM_V2_CLASSICAL_FEASIBILITY_GATE08_PROTOCOL.md"
MANIFEST = "docs/waveform_v2/optical_quality_gate07/local_artifact_manifest.json"
CODE = ("rppg_lab/waveform_v2_classical_study.py", "rppg_lab/waveform_v2_classical_cache.py",
        "rppg_lab/waveform_v2_classical_plots.py", "scripts/audit_waveform_v2_classical_feasibility.py")
FROZEN = ("rppg_lab/classical.py", "rppg_lab/window_priors.py", "rppg_lab/signals.py",
          "rppg_lab/processing.py", "rppg_lab/quality.py", "rppg_lab/waveform_baselines.py",
          "rppg_lab/waveform_alignment.py", "rppg_lab/waveform_v2_scamps.py",
          "rppg_lab/waveform_v2_scamps_roi.py", "rppg_lab/waveform_v2_scamps_inputs.py",
          "rppg_lab/waveform_v2_timing_contract.py", "rppg_lab/region_masks.py", "rppg_lab/roi.py")


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def write_json(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(clean(data), stream, indent=2, allow_nan=False)
        stream.write("\n")


def write_csv(path, rows):
    flattened = []
    for row in rows:
        flat = {k: v for k, v in row.items() if not isinstance(v, dict)}
        for field in ("quality", "metrics"):
            flat.update({f"{field}_{k}": v for k, v in row.get(field, {}).items()})
        flattened.append(clean(flat))
    fields = list(dict.fromkeys(k for r in flattened for k in r))
    with Path(path).open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(flattened)


def check_committed_design():
    subprocess.run(["git", "merge-base", "--is-ancestor", PROTOCOL_COMMIT, "HEAD"], check=True, capture_output=True)
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", *CODE, PROTOCOL], check=True, capture_output=True)
    for name in (*CODE, PROTOCOL):
        subprocess.run(["git", "ls-files", "--error-unmatch", name], check=True, capture_output=True)
    for name in FROZEN:
        original = subprocess.check_output(["git", "show", PARENT_COMMIT+":"+name])
        current = Path(name).read_bytes().replace(b"\r\n", b"\n")
        if current != original.replace(b"\r\n", b"\n"):
            raise ValueError("Frozen preprocessing changed: "+name)
    return {p: source_sha256(p) for p in (*CODE, *FROZEN, PROTOCOL)}


def validate_output(output, source, baseline):
    for path in (output, source, baseline):
        _check_path(path)
    out, src, cache = output.resolve(), source.resolve(), baseline.resolve()
    allowed = Path("outputs").resolve()
    if out.exists():
        raise ValueError("Refusing existing output directory")
    if (allowed not in out.parents or src == out or src in out.parents
            or cache == out or cache in out.parents):
        raise ValueError("Use a fresh directory under outputs, outside sources and baseline cache")
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scamps-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        output = validate_output(args.out_dir, args.scamps_root, args.baseline_root)
    except ValueError as error:
        parser.error(str(error))
    paths = discover_scamps(args.scamps_root)
    if [p.name for p in paths] != [f"P{i:06d}.mat" for i in range(1, 11)]:
        parser.error("Only the ten admitted SCAMPS examples are permitted")
    code_hashes = check_committed_design()
    old = json.loads(Path("docs/waveform_v2/scamps_gate06/roi_coverage_and_inputs.json").read_text())
    prior = {r["filename"]: r for r in old["recordings"]}
    cache = cache_manifest(MANIFEST)
    splits = read_official_split("docs/waveform_v2/scamps/official_split.csv")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile("docs/waveform_v2/scamps/LICENSE_SCAMPS.txt", output/"LICENSE_SCAMPS.txt")
    shutil.copyfile("docs/waveform_v2/scamps/ATTRIBUTION.md", output/"ATTRIBUTION.md")
    started = time.perf_counter(); tracemalloc.start()
    rows, records, predictions, readers, provenance = [], {}, {}, {}, []

    # Phase 1: no HDF5 value reads. Labels cannot influence any recording's prediction.
    for path in paths:
        print(f"{path.name}: verifying source/cache, rebuilding label-blind priors", flush=True)
        reader = ScampsRecording(args.scamps_root, path.name, max_window_frames=300,
                                 expected_sha256=prior[path.name]["source_sha256"])
        expected = [w["input_sha256"] for w in prior[path.name]["windows"]]
        key = f"retained_arrays/{path.stem}.npz"
        baseline = load_verified_baseline(args.baseline_root, path.name, cache[key], expected)
        split = splits["assignments"][path.stem]["split"]
        items = prediction_rows(path.name, split, baseline["rgb"], baseline["valid"],
                                baseline["clipping"], baseline["prediction"])
        rows.extend(items)
        predictions[path.name] = baseline["prediction"]["signals"]
        readers[path.name] = reader
        records[path.name] = baseline
        provenance.append(dict(filename=path.name, split=split, source_sha256_before=reader.info["source_sha256"],
                               cache_sha256=cache[key], input_sha256=baseline["input_sha256"],
                               acquisition_valid_frames=baseline["valid"].sum(-1),
                               fusion_weights=baseline["prediction"]["fusion_weights"],
                               decoded_reference_values_during_prediction=False))
    blind_seconds = time.perf_counter()-started
    assert len(rows) == 480
    write_json(output/"label_blind_predictions.json", dict(reference_values_decoded=False, cells=rows))
    archive = {Path(name).stem: values for name, values in predictions.items()}
    archive.update({Path(name).stem+"_x": data["prediction"]["batch"]["x"] for name, data in records.items()})
    np.savez_compressed(output/"locked_predictions.npz", **archive)
    locked_hash = source_sha256(output/"locked_predictions.npz")
    blind_hash = source_sha256(output/"label_blind_predictions.json")
    write_json(output/"prediction_lock.json", dict(stage="BEFORE_ANY_REFERENCE_VALUE_READ",
        predictions_sha256=locked_hash, metadata_sha256=blind_hash, intended_cells=480,
        all_20_inputs_match_gate06=True, protocol_commit=PROTOCOL_COMMIT,
        implementation_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()))

    # Phase 2: score Train first; Val/Test stay separate in aggregation and plots.
    references = {}
    order = sorted(paths, key=lambda p: ({"Train": 0, "Val": 1, "Test": 2}[splits["assignments"][p.stem]["split"]], p.name))
    scoring_start = time.perf_counter()
    for path in order:
        print(f"{path.name}: fixed {splits['assignments'][path.stem]['split']} reference scoring", flush=True)
        references[path.name] = read_reference_windows(readers[path.name])
        for row in (r for r in rows if r["filename"] == path.name):
            w, r, k = STARTS.index(row["start"]), CONDITIONS.index(row["roi"]), METHODS.index(row["method"])
            raw = references[path.name][w]
            row["raw_reference_sha256"] = array_sha256(raw)
            row["metrics"] = score_waveform(predictions[path.name][w, r, k], raw,
                                             prediction_eligible=row["pulse_estimate_eligible"])
            assert row["prediction_sha256"] == array_sha256(predictions[path.name][w, r, k])
    assert source_sha256(output/"locked_predictions.npz") == locked_hash
    assert source_sha256(output/"label_blind_predictions.json") == blind_hash
    scoring_seconds = time.perf_counter()-scoring_start
    for item in provenance:
        after = source_sha256(args.scamps_root/item["filename"])
        assert after == item["source_sha256_before"]
        item["source_sha256_after"] = after
    summaries = aggregate(rows)
    associations = quality_associations(rows)
    control_start = time.perf_counter()
    controls, timing = run_controls(), timing_controls()
    control_seconds = time.perf_counter()-control_start
    np.savez_compressed(output/"reference_windows.npz", **{Path(n).stem: r for n, r in references.items()})
    write_csv(output/"per_window_metrics.csv", rows)
    write_csv(output/"negative_control_metrics.csv", controls)
    write_json(output/"condition_metrics.json", summaries)
    write_json(output/"quality_associations.json", associations)
    write_json(output/"negative_control_summary.json", aggregate(controls))
    write_json(output/"synthetic_timing.json", timing)
    make_plots(output/"plots", rows, summaries, predictions, references, controls)
    _, traced_peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    memory = psutil.Process().memory_info()
    write_json(output/"provenance_and_validation.json", dict(
        protocol_commit=PROTOCOL_COMMIT, parent_commit=PARENT_COMMIT,
        implementation_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        code_sha256=code_hashes, official_split_sha256=splits["source_sha256"], recordings=provenance,
        baseline_manifest_sha256=source_sha256(MANIFEST),
        methods=METHODS, conditions=CONDITIONS, starts=STARTS, nominal_grid_hz=30,
        reference_access_order=[p.name for p in order], locked_predictions_sha256=locked_hash,
        label_blind_metadata_sha256=blind_hash, all_prediction_hashes_unchanged_after_scoring=True,
        intended_cells=480, train_cells=384, val_cells=48, test_cells=48, control_cells=len(controls),
        computational_cells=sum(r["computational_success"] for r in rows),
        eligible_cells=sum(r["pulse_estimate_eligible"] for r in rows),
        valid_hr_cells=sum(np.isfinite(r["metrics"]["hr_error_bpm"]) for r in rows),
        valid_waveform_cells=sum(np.isfinite(r["metrics"]["zero_lag_r"]) for r in rows),
        runtime_seconds=dict(blind_preparation=blind_seconds, reference_scoring=scoring_seconds,
                             controls=control_seconds, total=time.perf_counter()-started),
        memory_bytes=dict(tracemalloc_peak=traced_peak, process_rss_at_end=memory.rss,
                          process_peak_working_set=getattr(memory, "peak_wset", None)),
        versions=dict(python=platform.python_version(), **{k: importlib.metadata.version(k) for k in
                      ("numpy", "scipy", "h5py", "scikit-learn", "matplotlib", "psutil")}),
        semantic_skin_labels="UNKNOWN", segmentation_mapping="UNVERIFIED", candidate_B_used=False,
        subject_adaptation_used=False, neural_execution=False, allow_training=False))
    artifacts = [dict(path=p.relative_to(output).as_posix(), bytes=p.stat().st_size,
                      sha256=source_sha256(p)) for p in sorted(output.rglob("*")) if p.is_file()]
    write_json(output/"local_artifact_manifest.json", dict(local_root=args.out_dir.as_posix(), files=artifacts))
    print(json.dumps(dict(intended_cells=480, eligible_cells=sum(r["pulse_estimate_eligible"] for r in rows),
                          controls=len(controls), allow_training=False)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
