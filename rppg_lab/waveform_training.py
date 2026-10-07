"""Transparent waveform candidate selection and strict checkpoint reconstruction."""
from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy

from .config import PipelineConfig
from .extraction_cache import EXTRACTION_CACHE_VERSION
from .losses import WaveformV1LossConfig, waveform_spectral_band
from .models import MultiROIPriorResidualWaveformNet
from .splits import validate_waveform_split_manifest

MORPHOLOGY_SCORE_FORMULA = (
    "0.5 * subject_balanced_metrics.wave_corr_aligned + "
    "0.5 * subject_balanced_metrics.d1_corr_same_lag")
HR_ELIGIBILITY_RULE = (
    "subject_balanced_HR_MAE(epoch) <= subject_balanced_HR_MAE(epoch0) + hr_regression_tolerance_bpm")
SELECTION_TIE_TOLERANCE = 1e-8
TIE_BREAKING_RULE = "scores within 1e-8: higher subject-balanced aligned correlation, lower HR MAE, earlier epoch"


def split_manifest_hash(manifest: dict) -> str:
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def morphology_score(metrics: dict) -> float:
    subject = metrics["subject_balanced_metrics"]
    return 0.5 * subject["wave_corr_aligned"] + 0.5 * subject["d1_corr_same_lag"]


def _selection_record(epoch: int, metrics: dict) -> dict:
    subject = metrics["subject_balanced_metrics"]
    return dict(epoch=epoch, morphology_score=morphology_score(metrics),
                aligned_corr=subject["wave_corr_aligned"], hr_mae=subject["hr_mae"])


def better_morphology(candidate: dict, incumbent: dict | None) -> bool:
    if not all(math.isfinite(candidate[k]) for k in ("morphology_score", "aligned_corr")):
        return False
    if incumbent is None:
        return True
    delta = candidate["morphology_score"] - incumbent["morphology_score"]
    if abs(delta) > SELECTION_TIE_TOLERANCE:
        return delta > 0
    candidate_hr = candidate["hr_mae"] if math.isfinite(candidate["hr_mae"]) else math.inf
    incumbent_hr = incumbent["hr_mae"] if math.isfinite(incumbent["hr_mae"]) else math.inf
    return (candidate["aligned_corr"], -candidate_hr, -candidate["epoch"]) > (
        incumbent["aligned_corr"], -incumbent_hr, -incumbent["epoch"])


class WaveformCheckpointSelector:
    """State is based only on validation; lag/spectra/coarse/quality never rank."""

    def __init__(self, hr_regression_tolerance_bpm: float = 1.0):
        if not math.isfinite(hr_regression_tolerance_bpm) or hr_regression_tolerance_bpm < 0:
            raise ValueError("HR tolerance must be finite and nonnegative")
        self.tolerance = hr_regression_tolerance_bpm
        self.epoch0_reference_metrics = None
        self.best_candidate = self.best_unconstrained = self.best_hr = None

    def consider(self, epoch: int, metrics: dict) -> dict:
        record = _selection_record(epoch, metrics)
        if epoch == 0:
            if self.epoch0_reference_metrics is not None:
                raise ValueError("Epoch-0 reference already established")
            if not all(math.isfinite(record[k]) for k in ("hr_mae", "morphology_score", "aligned_corr")):
                raise ValueError("Epoch-0 HR and morphology selection metrics must be finite")
            self.epoch0_reference_metrics = deepcopy(metrics)
        if self.epoch0_reference_metrics is None:
            raise ValueError("Evaluate epoch 0 before selecting trained epochs")
        baseline = self.epoch0_reference_metrics["subject_balanced_metrics"]["hr_mae"]
        eligible = all(math.isfinite(record[k]) for k in ("hr_mae", "morphology_score", "aligned_corr")) and (
            epoch == 0 or record["hr_mae"] <= baseline + self.tolerance)
        updated = []
        if eligible and better_morphology(record, self.best_candidate):
            self.best_candidate = record
            updated.append("best_waveform_candidate.pt")
        if better_morphology(record, self.best_unconstrained):
            self.best_unconstrained = record
            updated.append("best_morphology_unconstrained.pt")
        if math.isfinite(record["hr_mae"]) and (self.best_hr is None or
                (record["hr_mae"], epoch) < (self.best_hr["hr_mae"], self.best_hr["epoch"])):
            self.best_hr = record
            updated.append("best_hr_diagnostic.pt")
        return dict(morphology_score=record["morphology_score"], hr_eligible=bool(eligible),
                    hr_gate_limit_bpm=baseline + self.tolerance, updated_checkpoints=updated,
                    selected_epochs=dict(candidate=self.best_candidate["epoch"] if self.best_candidate else None,
                                         unconstrained=self.best_unconstrained["epoch"] if self.best_unconstrained else None,
                                         hr_diagnostic=self.best_hr["epoch"] if self.best_hr else None))


def checkpoint_dataset_config(checkpoint: dict, manifest: dict | None = None) -> dict:
    """Validate ordering/schema/config before any model or dataset is evaluated."""
    if checkpoint.get("checkpoint_schema_version") != "waveform_multi_roi_part3_v1":
        raise ValueError("Require a Part 3 waveform checkpoint schema")
    if checkpoint.get("model_class") != "MultiROIPriorResidualWaveformNet":
        raise ValueError("Incompatible model class")
    if checkpoint.get("extraction_cache_version") != EXTRACTION_CACHE_VERSION:
        raise ValueError("Incompatible extraction cache version")
    cfg = checkpoint["model_config"]
    rois, priors = tuple(checkpoint["roi_names"]), tuple(checkpoint["prior_names"])
    channels = ("RGB_R", "RGB_G", "RGB_B", *priors)
    if (tuple(cfg["roi_names"]) != rois or tuple(cfg["prior_names"]) != priors
            or cfg["prior_start"] != 3 or cfg["in_channels"] != len(channels)
            or tuple(checkpoint["channel_ordering"]) != channels
            or tuple(checkpoint["channel_names"]) != channels):
        raise ValueError("Incompatible ROI/prior/channel ordering")
    extraction = dict(checkpoint["extraction_config"])
    for key in ("regions", "skin_cr_range", "skin_cb_range", "face_rois"):
        if key in extraction:
            extraction[key] = tuple(extraction[key])
    config = PipelineConfig(**extraction)
    if (config.regions != ("face",) or config.requested_face_rois != rois or config.face_roi != rois[0]
            or config.sample_rate != checkpoint["fs"]):
        raise ValueError("Incompatible extraction ROI order or sampling rate")
    if checkpoint["window_length_samples"] != round(checkpoint["win_sec"] * checkpoint["fs"]):
        raise ValueError("Incompatible checkpoint window length")
    loss_cfg = WaveformV1LossConfig(**checkpoint["loss_config"])
    band = waveform_spectral_band(checkpoint["fs"], loss_cfg.spectral_fmin_hz, loss_cfg.spectral_fmax_hz)
    if tuple(checkpoint["spectral_band_hz"]) != band:
        raise ValueError("Incompatible checkpoint spectral band")
    saved = checkpoint["split_manifest"]
    validate_waveform_split_manifest(saved)
    if split_manifest_hash(saved) != checkpoint["split_manifest_sha256"]:
        raise ValueError("Checkpoint split manifest hash mismatch")
    if manifest is not None:
        validate_waveform_split_manifest(manifest)
        if split_manifest_hash(manifest) != checkpoint["split_manifest_sha256"]:
            raise ValueError("Supplied split manifest does not match checkpoint")
    for key, group in (("train_subjects", "train_ids"), ("val_subjects", "validation_ids"),
                       ("test_subjects", "test_ids")):
        if checkpoint[key] != saved[group]:
            raise ValueError("Checkpoint subject IDs disagree with manifest")
    return dict(fs_target=checkpoint["fs"], win_sec=checkpoint["win_sec"],
                stride_sec=checkpoint["stride_sec"], roi_names=rois, prior_methods=priors,
                extraction_config=config, min_valid_fraction=checkpoint["min_valid_fraction"],
                max_frames=checkpoint["max_frames"])


def reconstruct_waveform_model(checkpoint: dict, device="cpu") -> MultiROIPriorResidualWaveformNet:
    checkpoint_dataset_config(checkpoint)
    model = MultiROIPriorResidualWaveformNet(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.to(device).eval()
