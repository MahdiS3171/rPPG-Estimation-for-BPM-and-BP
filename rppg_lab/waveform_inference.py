"""Window-level inference for the identity-checked frozen facial extractor.

Frozen Waveform v1 is intended for facial morphology extraction. It is NOT
approved as the source for face-hand inter-site timing. The BP timing path
remains conservative matched face/hand GREEN until learned-waveform phase
behavior is separately validated.

Windows retain the model's signed polarity and original video timestamps.
Continuous reconstruction is deferred: independently normalized overlapping
windows must not be naively concatenated. No DTW or timing correction is used.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .artifacts import file_sha256, json_safe
from .classical import DEFAULT_PRIORS
from .datasets import UBFCMultiROIRPPGDataset
from .extraction_cache import DEFAULT_WAVEFORM_ROIS, FaceROIExtraction, _validate
from .processing import resample_trace
from .signals import standardize_channels
from .waveform_metrics import forward_batch
from .waveform_training import checkpoint_dataset_config, reconstruct_waveform_model
from .window_priors import build_window_priors

TIMING_BOUNDARY = ("Facial morphology extraction only. Not approved for face-hand inter-site timing; "
                   "retain the conservative matched face/hand GREEN timing path.")


class FrozenWaveformExtractor:
    """Accept Phase 1 FaceROIExtraction, returning one frozen-length window.

    The adjacent freeze manifest is mandatory; checkpoint/config/order/hash
    mismatches are rejected. Raw video pixels and reference PPG are not inputs.
    """

    def __init__(self, checkpoint_path, manifest_path=None, device="cpu"):
        checkpoint_path = Path(checkpoint_path)
        manifest_path = Path(manifest_path) if manifest_path else checkpoint_path.with_name("waveform_v1_freeze_manifest.json")
        self.model_sha256 = file_sha256(checkpoint_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if (checkpoint.get("waveform_v1_status") != "frozen_v1"
                or checkpoint.get("freeze_decision") != "FREEZE_WAVEFORM_V1"
                or manifest.get("status") != "frozen_v1"
                or manifest.get("model_sha256") != self.model_sha256):
            raise ValueError("Require a frozen_v1 checkpoint with its matching freeze manifest/hash")
        dataset_config = checkpoint_dataset_config(checkpoint)
        self.roi_names, self.prior_names = tuple(checkpoint["roi_names"]), tuple(checkpoint["prior_names"])
        self.channel_names = tuple(checkpoint["channel_ordering"])
        if (self.roi_names != DEFAULT_WAVEFORM_ROIS or self.prior_names != tuple(DEFAULT_PRIORS)
                or tuple(manifest["roi_ordering"]) != self.roi_names
                or tuple(manifest["prior_ordering"]) != self.prior_names
                or tuple(manifest["channel_ordering"]) != self.channel_names):
            raise ValueError("Incompatible frozen ROI/prior/channel ordering")
        if (manifest["source_candidate_sha256"] != checkpoint["frozen_from_checkpoint_sha256"]
                or manifest["selected_epoch"] != checkpoint["selected_epoch"]
                or manifest["window_length_samples"] != checkpoint["window_length_samples"]
                or manifest["fs"] != checkpoint["fs"]
                or manifest["model_config"] != json_safe(checkpoint["model_config"])
                or manifest["extraction_config"] != json_safe(checkpoint["extraction_config"])
                or manifest["min_valid_fraction"] != checkpoint["min_valid_fraction"]
                or manifest["validation_evaluation_sha256"] != checkpoint["validation_evaluation_sha256"]
                or manifest["locked_test_evaluation_sha256"] != checkpoint["locked_test_evaluation_sha256"]):
            raise ValueError("Frozen provenance/config mismatch")
        self.device = torch.device(device)
        self._model = reconstruct_waveform_model(checkpoint, self.device)
        self.fs_target = checkpoint["fs"]
        self.window_length_samples = checkpoint["window_length_samples"]
        self.min_valid_fraction = checkpoint["min_valid_fraction"]
        self.extraction_config = dataset_config["extraction_config"]
        self.metadata = dict(model_class=checkpoint["model_class"], model_version="Waveform v1",
            waveform_v1_status="frozen_v1", model_sha256=self.model_sha256,
            source_candidate_sha256=checkpoint["frozen_from_checkpoint_sha256"],
            selected_epoch=checkpoint["selected_epoch"], roi_ordering=self.roi_names,
            prior_ordering=self.prior_names, channel_ordering=self.channel_names,
            fs=self.fs_target, window_length_samples=self.window_length_samples,
            continuous_stitching="deferred; independent window outputs only", timing_boundary=TIMING_BOUNDARY)

    def prepare_window(self, extraction: FaceROIExtraction, start_sec: float | None = None) -> dict:
        """Reuse Phase 1 bounded RGB and the exact training support/quality rule.

        Priors see only this window's raw intensities. No recording-level prior
        cache, target, polarity flip, reference alignment or extra normalization
        of the model's prediction is introduced.
        """
        if not isinstance(extraction, FaceROIExtraction):
            raise TypeError("Require Phase 1 FaceROIExtraction, not raw video pixels")
        _validate(extraction, self.extraction_config)
        grid = np.asarray(extraction.shared.timestamps, dtype=float)
        if start_sec is None:
            start = 0
        else:
            if not np.isfinite(start_sec):
                raise ValueError("Window start must be finite")
            start = int(np.argmin(abs(grid - start_sec)))
            if abs(grid[start] - start_sec) > 1e-6:
                raise ValueError("Window start must be on the original uniform video grid")
        end = start + self.window_length_samples
        if end > len(grid):
            raise ValueError("Insufficient support for the frozen window length")
        rgb = np.stack([resample_trace(extraction.traces[name], grid, self.extraction_config.max_gap_sec)[0]
                        for name in self.roi_names])
        record = dict(timestamps=grid, rgb=rgb, traces=extraction.traces)
        # This existing method uses only fs_target, roi_names and
        # min_valid_fraction. Reuse it directly rather than duplicating rules
        # or constructing a reference-dependent UBFC dataset at inference.
        valid, quality = UBFCMultiROIRPPGDataset._window_support(self, record, start, end)
        prior_valid = np.zeros((len(self.roi_names), len(self.prior_names)), bool)
        x = np.zeros((len(self.roi_names), len(self.channel_names), self.window_length_samples), np.float32)
        for r in np.flatnonzero(valid):
            window = rgb[r, start:end]
            priors, success = build_window_priors(window, self.fs_target, self.prior_names)
            valid[r] = bool(success.any())
            if valid[r]:
                x[r, :3] = standardize_channels(window).T
                x[r, 3:] = priors
                prior_valid[r] = success
        if not valid.any():
            raise ValueError("No usable ROI in this frozen-model window")
        return dict(x=torch.from_numpy(x[None]), roi_valid=torch.from_numpy(valid[None]),
            roi_quality=torch.from_numpy(quality[None]), prior_valid=torch.from_numpy(prior_valid[None]),
            timestamps=grid[start:end].copy())

    def infer_window(self, extraction: FaceROIExtraction, start_sec: float | None = None) -> dict:
        """Return a signed window and descriptive allocations, with provenance.

        The unsupervised quality head is intentionally omitted. ROI source
        valid fractions describe acquisition support, not calibrated quality.
        """
        batch = self.prepare_window(extraction, start_sec)
        with torch.inference_mode():
            out = forward_batch(self._model, batch, self.device)
        array = lambda value: value[0].detach().cpu().numpy().copy()
        return dict(waveform=array(out["ppg"]), timestamps=batch["timestamps"],
            roi_attention=array(out["roi_attention"]), prior_weights=array(out["prior_weights"]),
            roi_valid=array(batch["roi_valid"]), prior_valid=array(batch["prior_valid"]),
            roi_source_valid_fraction=array(batch["roi_quality"]),
            scaled_residuals=array(self._model.residual_scale * out["residuals"]),
            metadata=dict(self.metadata))
