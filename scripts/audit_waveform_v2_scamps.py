"""Gate 05: read-only admission checks on the TEN supplied SCAMPS examples only.

No training, inference, benchmark/model selection, image cache or dataset download.
Images/plots go only into the explicitly requested fresh local output directory.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import threading
import time
import tracemalloc

import h5py
import numpy as np
import psutil

from rppg_lab.waveform_v2_scamps import (SCHEMA, DOC_REVISION, ScampsRecording,
    discover_scamps, inspect_scamps_file, read_official_split)


def write_new_json(path: Path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def rendering_factor_audit(paths):
    """Compare released motion/action traces, never infer avatar identity from faces."""
    groups = {}
    for path in paths:
        with h5py.File(path, "r") as f:
            names = [k for k in f if k in ("d_pitch", "d_roll", "d_yaw") or k.startswith("au")]
            for name in names:
                values = f[name][:, 0]  # one 600-value auxiliary vector, no RGB
                digest = hashlib.sha256(values.astype("<f8").tobytes()).hexdigest()
                groups.setdefault((name, digest), []).append(path.stem)
    return dict(identity_keys_in_local_files=False, appearance_asset_ids_available=False,
                exact_shared_motion_or_action_traces=[dict(variable=k[0], trace_sha256=k[1], recordings=v)
                    for k, v in sorted(groups.items()) if len(v) > 1],
                interpretation="Shared released motion/action traces are rendering-factor overlap, not proof of shared avatar identity")


def profile_windows(recording: ScampsRecording) -> dict:
    """Sample native RSS plus Python/NumPy allocation peak; do not infer total RSS from arrays."""
    reports = []
    process = psutil.Process()
    for length in (1, 8):
        gc.collect()
        baseline = process.memory_info().rss
        samples = [baseline]
        stop = threading.Event()

        def monitor():
            while not stop.wait(.002):
                samples.append(process.memory_info().rss)

        thread = threading.Thread(target=monitor, daemon=True)
        tracemalloc.start()
        thread.start()
        started = time.perf_counter()
        try:
            window = recording.read_window(300, 300 + length)
            samples.append(process.memory_info().rss)
            current, peak = tracemalloc.get_traced_memory()
            array_bytes = window["rgb"].nbytes + window["d_ppg"].nbytes
            reports.append(dict(window_frames=length, frames_variable="Xsub", start_index=300,
                                returned_rgb_ppg_bytes=array_bytes,
                                rss_baseline_bytes=baseline, sampled_rss_peak_bytes=max(samples),
                                sampled_rss_delta_bytes=max(samples) - baseline,
                                python_numpy_traced_peak_bytes=peak,
                                elapsed_seconds=time.perf_counter() - started,
                                decoded_shape=list(window["rgb"].shape)))
            del window
        finally:
            stop.set()
            thread.join()
            tracemalloc.stop()
    return dict(filename=recording.path.name, source_sha256=recording.info["source_sha256"],
                windows=reports, rss_sampling_interval_seconds=.002,
                limitations=["In-process RSS measurements, not a fresh-process or allocator-independent bound",
                             "Sampling may miss brief native peaks; tracemalloc excludes some native HDF5 memory",
                             "HDF5 time chunks span 75 frames; small windows decompress more data than they return"],
                hdf5_read_cache_bytes=1024 * 1024, max_window_frames=recording.max_window_frames,
                max_window_bytes=recording.max_window_bytes)


def previews(recording: ScampsRecording, output: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 3, figsize=(10, 11), constrained_layout=True)
    for row, index in enumerate((0, 299, 598)):
        window = recording.read_window(index, index + 1)
        independent = recording.read_mask_window(index, index + 1)
        axes[row, 0].imshow(window["rgb"][0])
        axes[row, 0].set_title(f"RGB frame {index}, Xsub")
        for col, name in enumerate(("skin_mask", "combined_mask"), 1):
            values = independent["masks"][name][0]
            im = axes[row, col].imshow(values, cmap="gray", vmin=0, vmax=max(1, float(values.max())))
            axes[row, col].set_title(f"{name}, mask index {index}\nFrame mapping UNVERIFIED", fontsize=10)
            fig.colorbar(im, ax=axes[row, col], fraction=.045, pad=.02)
        for ax in axes[row]:
            ax.axis("off")
    fig.suptitle(f"SCAMPS {recording.path.stem}: engineering preview\n"
                 "McDuff et al., 2022 / R-UDA non-commercial research; masks are separate index streams",
                 fontsize=12)
    fig.savefig(output / "synthetic_frames_masks_preview.png", dpi=130)
    plt.close(fig)
    with h5py.File(recording.path, "r") as f:
        ppg = f["d_ppg"][:, 0]  # 600 scalar labels only, never a whole video
    fig, ax = plt.subplots(figsize=(10, 3.8))
    fig.subplots_adjust(left=.08, right=.98, top=.88, bottom=.24)
    ax.plot(np.arange(len(ppg)), ppg, color="#176b87", linewidth=1.3)
    ax.set(xlabel="Synthetic frame / label index (no capture timestamps)",
           ylabel="Stored d_ppg (unscaled)", title=f"SCAMPS {recording.path.stem}: synthetic PPG target")
    ax.grid(alpha=.2)
    fig.text(.01, .025, "McDuff et al., 2022 / SCAMPS / R-UDA non-commercial research. Not physiological timing validation.", fontsize=8)
    fig.savefig(output / "synthetic_ppg_preview.png", dpi=140)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scamps-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--official-split", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--scan-values", action="store_true", help="Stream all stored numeric values in bounded blocks")
    args = parser.parse_args(argv)
    paths = discover_scamps(args.scamps_root)
    if [p.name for p in paths] != [f"P{i:06d}.mat" for i in range(1, 11)]:
        parser.error("Gate 05 audit is restricted to the exact ten example filenames P000001-P000010")
    output = args.out_dir.resolve()
    root = args.scamps_root.resolve()
    if output == root or root in output.parents:
        parser.error("Output must be outside the SCAMPS source tree")
    if output.exists():
        parser.error("Use a new output directory; historical artifacts must not be overwritten")
    split = read_official_split(args.official_split)
    sources = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for path in paths:
        print(f"Auditing {path.name}: source hash, HDF5 structure" +
              (", bounded full-value scan" if args.scan_values else ""), flush=True)
        info = inspect_scamps_file(path, scan_values=args.scan_values)
        if info["structure_valid"]:
            rec = ScampsRecording(path.parent, path.name, expected_sha256=info["source_sha256"])
            window = rec.read_window(300, 302)
            mask = rec.read_mask_window(300, 302)
            info["lazy_window_check"] = dict(start=300, stop=302, rgb_shape=list(window["rgb"].shape),
                label_indices_equal_frame_indices=bool(np.array_equal(window["frame_indices"], window["label_indices"])),
                paired_values_valid=bool(window["validity"]["paired"].all()),
                masks_associated_to_frames=mask["frame_indices"] is not None,
                mask_shape=list(mask["masks"]["skin_mask"].shape),
                status="PASS_ENGINEERING_ONLY")
        info["official_split"] = split["assignments"].get(path.stem)
        info["engineering_inspected"] = True
        info["independent_generalization_test"] = False
        rows.append(info)
    inventory = dict(schema_version=SCHEMA, dataset_id="SCAMPS", scope="ten_examples_engineering_only",
                     root_hint=root.name, documentation_revision=DOC_REVISION,
                     dataset_version="not_embedded_in_local_files; user-provided official example claim",
                     source_artifacts=sources, recording_count=len(rows), recordings=rows,
                     allow_training=False, allow_absolute_timing_scoring=False)
    write_new_json(output / "dataset_inventory.json", inventory)
    timing = dict(scope="synthetic_index_provenance_not_physical_timing",
                  documentation_claim="RGB frame i and d_ppg[i] are aligned synthetic samples",
                  nominal_render_rate_hz=30., nominal_rate_basis="600 documented frames / 20 s in paper; not embedded metadata",
                  capture_timestamps_present=False, physical_synchronization_verified=False,
                  frame_label_index_shift_applied=0, mask_shift_padding_threshold_or_resize_applied=False,
                  recordings=[dict(filename=r["filename"], source_sha256=r["source_sha256"],
                                   structure_valid=r["structure_valid"], frames=r.get("frame_count"),
                                   lazy_window_check=r.get("lazy_window_check"),
                                   ppg_samples=r.get("waveform_count"), mask_alignment=r.get("mask_alignment"),
                                   mask_lengths={k: r.get("datasets", {}).get(k, {}).get("shape", [None])[-1]
                                                 for k in ("skin_mask", "combined_mask", "numerical_mask")},
                                   allow_training=False, allow_physiological_delay_scoring=False)
                              for r in rows],
                  blockers=["599 mask samples vs 600 frames/targets; mask-to-frame mapping unverified",
                            "Fractional mask semantics and values outside [0,1] require explicit interpretation",
                            "No actual capture timestamps or independent synchronization evidence",
                            "20-second nominal clips cannot support the current default 30-second context plus target"])
    write_new_json(output / "alignment_timing_audit.json", timing)
    assignments = {p.stem: split["assignments"].get(p.stem) for p in paths}
    write_new_json(output / "split_audit.json", dict(source_sha256=split["source_sha256"],
        header_anomaly=split["header_anomaly"],
        official_counts=dict(Counter(v["split"] for v in split["assignments"].values())),
        sample_assignments=assignments,
        sample_counts=dict(Counter(v["split"] if v else "unassigned" for v in assignments.values())),
        avatar_disjointness="official_datasheet_claim_not_independently_verified",
        rendering_factors=rendering_factor_audit(paths), all_ten_engineering_exposed=True,
        forbid_random_window_splits=True))
    if all(r["structure_valid"] for r in rows):
        recording = ScampsRecording(root, paths[0].name, expected_sha256=rows[0]["source_sha256"])
        memory = profile_windows(recording)
        memory["full_collection_uncompressed_bytes"] = sum(v["uncompressed_bytes"]
            for r in rows for v in r["datasets"].values())
        memory["largest_audit_read_bytes"] = max((v.get("value_audit", {}).get("largest_read_bytes", 0)
            for r in rows for v in r["datasets"].values()), default=0)
        write_new_json(output / "memory_usage.json", memory)
        previews(recording, output)
    summary = dict(recordings=len(rows), structures_valid=sum(r["structure_valid"] for r in rows),
                   all_values_finite=sum(r["all_values_finite"] is True for r in rows),
                   training_admitted=0, physical_timing_admitted=0)
    write_new_json(output / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    return 0 if all(r["structure_valid"] and (not args.scan_values or r["all_values_finite"])
                    for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
