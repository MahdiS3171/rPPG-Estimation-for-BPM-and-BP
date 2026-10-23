#!/usr/bin/env python
"""Verify actual Pilot-6 artifacts, clock correspondence and replayed frozen windows."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(key, "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def contribution_diagnostics(directory):
    """Algebraic decomposition of saved outputs; not causal feature importance."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from rppg_lab.waveform_visualization import write_csv
    directory = Path(directory)
    with np.load(directory/"waveform_windows.npz", allow_pickle=False) as z:
        x, attention, weights = z["model_inputs"], z["roi_attention"], z["prior_weights"]
        per_prior = (attention[:,:,None,None] * weights[:,:,:,None] * x[:,:,3:,:]).sum(axis=1)
        residual = (attention[:,:,None] * z["scaled_residuals"]).sum(axis=1)
        reconstructed = per_prior.sum(axis=1) + residual
        valid = np.isfinite(z["waveform"]).all(axis=1)
        error = float(abs(reconstructed[valid]-z["waveform"][valid]).max()) if valid.any() else None
        if error is not None:
            np.testing.assert_allclose(reconstructed[valid], z["waveform"][valid], rtol=2e-6, atol=2e-6)
        rows = []
        ratios = []
        for wi in range(len(valid)):
            output_rms = float(np.sqrt(np.mean(z["waveform"][wi]**2))) if valid[wi] else None
            for name, wave in list(zip(z["prior_names"].tolist(), per_prior[wi])) + [("scaled_residual", residual[wi])]:
                rms = float(np.sqrt(np.mean(wave**2))) if valid[wi] else None
                ratio = rms/output_rms if output_rms and output_rms > 1e-8 else None
                rows.append(dict(window_index=wi, start_sec=float(z["timestamps"][wi,0]),
                    valid=bool(valid[wi]), component=name, rms=rms, rms_relative_to_model=ratio,
                    interpretation="algebraic contribution, correlated components; not variance explained or causal importance"))
                if name == "scaled_residual" and ratio is not None:
                    ratios.append(ratio)
        write_csv(directory/"model_contributions.csv", rows)
        if valid.any():
            wi = int(np.flatnonzero(valid)[0])
            fig, axes = plt.subplots(2,1, figsize=(12,7), sharex=True)
            axes[0].plot(z["timestamps"][wi], z["waveform"][wi], label="Frozen output", color="black")
            axes[0].plot(z["timestamps"][wi], per_prior[wi].sum(axis=0), label="Weighted priors", linestyle="--")
            axes[0].plot(z["timestamps"][wi], residual[wi], label="Weighted scaled residual")
            axes[0].legend(ncol=3, fontsize=8)
            axes[0].set(title=f"Exact saved-output decomposition, first valid window {wi}", ylabel="Model amplitude")
            for name, wave in zip(z["prior_names"].tolist(), per_prior[wi]):
                axes[1].plot(z["timestamps"][wi], wave, label=name)
            axes[1].legend(ncol=6, fontsize=8)
            axes[1].set(xlabel="Original video time (s)", ylabel="Weighted prior amplitude",
                        title="Descriptive correlated components; no causal or physiological validation")
            fig.tight_layout(); fig.savefig(directory/"model_decomposition.png", dpi=150); plt.close(fig)
    return dict(max_absolute_reconstruction_error=error,
        median_residual_rms_relative_to_model=float(np.median(ratios)) if ratios else None,
        interpretation="algebraic output identity; RMS ratios are not explained-variance fractions")


def verify(run_dir, dataset_root="Dataset", recording=None):
    import numpy as np
    import torch
    from rppg_lab.artifacts import file_sha256, write_json
    from rppg_lab.frozen_timebase import TimebaseConfig, adapt_native_face
    from rppg_lab.pilot_frozen import (DEFAULT_CHECKPOINT, OFFICIAL_MODEL_SHA256,
        OFFICIAL_MANIFEST_SHA256, native_extraction, protected_hashes)
    from rppg_lab.pilot_video import mp4_presentation_times
    from rppg_lab.waveform_inference import FrozenWaveformExtractor
    torch.set_num_threads(1)
    root = Path(run_dir)
    manifest_path = root/"experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
    audit_directory = Path(manifest["audit"]) if manifest else root/"audit"
    audit = json.loads((audit_directory/"audit.json").read_text(encoding="utf-8"))
    checkpoint = audit["checkpoint_path"]
    extractor = FrozenWaveformExtractor(checkpoint, audit["manifest_path"])
    assert extractor.model_sha256 == OFFICIAL_MODEL_SHA256
    assert file_sha256(audit["manifest_path"]) == OFFICIAL_MANIFEST_SHA256
    assert file_sha256(audit["metadata"]["path"]) == audit["metadata"]["sha256"]
    current = protected_hashes(checkpoint, audit["manifest_path"])
    assert all(current.get(path) == sha for path, sha in audit["protected_before"].items())
    paths = sorted(root.glob("S*/summary.json"))
    if recording:
        paths = [p for p in paths if p.parent.name == recording]
    if not paths:
        raise ValueError("No requested completed recording summaries")
    results = []
    for path in paths:
        directory = path.parent
        rid = directory.name
        summary = json.loads(path.read_text())
        with np.load(directory/"native_face_rgb.npz", allow_pickle=False) as data:
            raw = {key:data[key].copy() for key in data.files}
        pts, _ = mp4_presentation_times(Path(dataset_root)/(rid+".mp4"))
        np.testing.assert_array_equal(raw["camera_pts"], pts)
        assert len(pts) == summary["frames"]
        identity = json.loads(str(raw["identity_json"].item()))
        assert file_sha256(Path(dataset_root)/(rid+".mp4")) == identity["input_sha256"]
        assert file_sha256(Path("rppg_lab/pilot_frozen_extraction.py")) == identity["source_sha256"]
        face_model = identity["historical_identity"]["face_model"]
        assert file_sha256(face_model["path"]) == face_model["sha256"]
        for source, sha in identity["historical_identity"]["source_sha256"].items():
            assert file_sha256(Path("rppg_lab")/(source+".py")) == sha
        backend = raw["decoder_pts"]
        assert backend.shape == pts.shape and np.isfinite(backend).all()
        delta = (backend-backend[0])-(pts-pts[0])
        # The decoder may choose a different origin, but display intervals must agree.
        assert np.max(abs(delta)) < 1e-5, "Decoder/container frame-PTS correspondence failed"
        with (directory/"window_metrics.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        count = summary["complete_windows"]
        assert len(rows) == count*4
        assert all(r["bp_prediction_kind"] == "unavailable_not_attempted" for r in rows)
        with np.load(directory/"adapted_rgb.npz", allow_pickle=False) as z:
            np.testing.assert_array_equal(z["camera_pts"], pts)
            assert z["rgb_30"].shape == (3, len(z["target_pts"]), 3)
            assert np.all(z["target_source_left_index"] <= z["target_source_right_index"])
            expected_times = np.asarray([z["target_pts"][s:s+300] for s in range(0,len(z["target_pts"])-299,150)])
        config = TimebaseConfig(**json.loads((directory/"timebase_adaptation.json").read_text())["config"])
        adapted, _ = adapt_native_face(native_extraction(raw, extractor), extractor.extraction_config, config)
        with np.load(directory/"waveform_windows.npz", allow_pickle=False) as z:
            np.testing.assert_array_equal(z["timestamps"], expected_times)
            assert z["waveform"].shape == (count, 300)
            valid = np.isfinite(z["waveform"]).all(axis=1)
            assert valid.sum() == summary["methods"]["FrozenWaveformV1"]["valid_windows"]
            assert np.isnan(z["waveform"][~valid]).all()
            np.testing.assert_allclose(z["roi_attention"][valid].sum(axis=1), 1, atol=1e-5)
            selected = np.flatnonzero(valid)
            replay = sorted(set([int(selected[0]), int(selected[-1])])) if len(selected) else []
            for wi in replay:
                prediction = extractor.infer_window(adapted, float(z["timestamps"][wi,0]))
                np.testing.assert_array_equal(prediction["waveform"], z["waveform"][wi])
                np.testing.assert_array_equal(prediction["roi_attention"], z["roi_attention"][wi])
                np.testing.assert_array_equal(prediction["prior_weights"], z["prior_weights"][wi])
                batch = extractor.prepare_window(adapted, float(z["timestamps"][wi,0]))
                np.testing.assert_array_equal(batch["x"][0].numpy(), z["model_inputs"][wi])
        validation = json.loads((directory/"adapter_clock_validation.json").read_text())
        assert validation["passed"] and summary["frozen_state_unchanged"]
        for artifact in ("overview.png", "prior_allocations.png", "roi_geometry.png", "allocations.csv",
                         "adjacent_window_consistency.csv", "roi_masks_examples.npz"):
            assert (directory/artifact).is_file()
        results.append(dict(recording_id=rid, frames=len(pts), windows=count,
            replayed_window_indices=replay, replay_bit_identical=True,
            camera_pts_exact=True, decoder_clock_max_relative_discrepancy_sec=float(abs(delta).max()),
            decoder_clock_origin_offset_sec=float(backend[0]-pts[0]),
            raw_sha256=file_sha256(directory/"native_face_rgb.npz"),
            waveform_sha256=file_sha256(directory/"waveform_windows.npz"),
            decomposition=contribution_diagnostics(directory)))
        print(json.dumps(results[-1]), flush=True)
    if not recording:
        expected = {p["recording_id"] for p in audit["metadata"]["matches"]}
        assert {r["recording_id"] for r in results} == expected, "Missing completed audited recordings"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        assert manifest["protected_historical_files_unchanged"] and not manifest["failures"]
        for path, sha in manifest["new_source_sha256"].items():
            assert file_sha256(path) == sha
    result = dict(status="PASS", recording_selection=recording, recordings=results,
        checkpoint_sha256=extractor.model_sha256, manifest_sha256=OFFICIAL_MANIFEST_SHA256,
        workbook_unchanged=True, historical_files_unchanged=True)
    suffix = "_"+recording if recording else ""
    write_json(root/("artifact_verification"+suffix+".json"), result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset-root", default="Dataset")
    parser.add_argument("--recording")
    args = parser.parse_args()
    verify(args.run_dir, args.dataset_root, args.recording)


if __name__ == "__main__":
    main()
