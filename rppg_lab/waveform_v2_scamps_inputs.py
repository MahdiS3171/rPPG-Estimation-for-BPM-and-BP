"""V1 input-shape contract using its unchanged preprocessing, without model calls."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .classical import DEFAULT_PRIORS
from .processing import resample_trace
from .signals import standardize_channels, standardize_1d
from .types import RGBTrace
from .window_priors import build_window_priors
from .waveform_v2_scamps_roi import ROI_NAMES, RGB_CHANNELS, ScampsROITraces

PRIOR_NAMES = tuple(DEFAULT_PRIORS)
CHANNEL_NAMES = (*RGB_CHANNELS, *PRIOR_NAMES)


@dataclass(frozen=True)
class ScampsInputConfig:
    roi_names: tuple[str, ...] = ROI_NAMES
    prior_names: tuple[str, ...] = PRIOR_NAMES
    nominal_rate_hz: float = 30.
    window_samples: int = 300
    min_valid_fraction: float = .9
    max_gap_sec: float = .15
    subject_adaptation_enabled: bool = False

    def __post_init__(self):
        if (self.roi_names != ROI_NAMES or self.prior_names != PRIOR_NAMES
                or self.nominal_rate_hz != 30. or self.window_samples != 300
                or not np.isfinite(self.min_valid_fraction) or not 0 <= self.min_valid_fraction <= 1
                or not np.isfinite(self.max_gap_sec) or self.max_gap_sec <= 0
                or self.subject_adaptation_enabled is not False):
            raise ValueError("Require fixed V1 ROI/prior order, 30-Hz grid, 300 samples and adaptation disabled")


def prepare_scamps_v1_inputs(traces: ScampsROITraces, starts=(0, 300), *, config=ScampsInputConfig()) -> dict:
    """Return NumPy [B,3,9,300] plus explicit masks and source indices.

    Bounded interpolation reuses V1's acquisition helper on the NOMINAL grid;
    only the supplied 300-sample window reaches priors/normalization. Raw traces,
    labels and their missingness remain available and are never modified.
    Unusable intended windows remain in the denominator with status/masks.
    """
    n = len(traces.frame_indices)
    if (traces.roi_names != ROI_NAMES or traces.rgb_channel_names != RGB_CHANNELS
            or traces.rgb.shape != (3, n, 3) or traces.roi_valid.shape != (3, n)
            or traces.roi_valid.dtype != bool or traces.raw_d_ppg.shape != (n,)
            or traces.rgb.dtype.kind != "f"
            or traces.frame_indices.dtype.kind not in "iu" or traces.label_indices.dtype.kind not in "iu"
            or not np.array_equal(traces.frame_indices, np.arange(n))
            or not np.array_equal(traces.label_indices, traces.frame_indices)
            or not np.array_equal(traces.nominal_grid_seconds, traces.frame_indices / 30.)
            or traces.provenance.get("coordinate_kind") != "documented_nominal_synthetic_frame_grid"
            or traces.provenance.get("capture_timestamps") is not None
            or traces.provenance.get("nominal_rate_hz") != 30.
            or traces.provenance.get("physical_clock_verified") is not False
            or traces.provenance.get("segmentation_used") is not False
            or traces.provenance.get("allow_training") is not False
            or traces.provenance.get("subject_adaptation_enabled") is not False):
        raise ValueError("Invalid trace order, shape, source indices or nominal-coordinate provenance")
    measured = traces.rgb[traces.roi_valid]
    if not np.isfinite(measured).all() or (measured < 0).any() or (measured > 255).any():
        raise ValueError("Valid measured RGB must be finite V1 intensity in [0,255]")
    starts = tuple(starts)
    if (not starts or len(set(starts)) != len(starts)
            or any(type(s) is not int or s < 0 or s + 300 > n for s in starts)):
        raise ValueError("Explicit distinct in-support 300-sample window starts required")
    uniform, imputed = [], []
    for r, name in enumerate(ROI_NAMES):
        trace = RGBTrace(traces.nominal_grid_seconds, traces.rgb[r], name, traces.roi_valid[r])
        values, flags = resample_trace(trace, traces.nominal_grid_seconds, config.max_gap_sec)
        uniform.append(values)
        imputed.append(flags)
    uniform, imputed = np.stack(uniform), np.stack(imputed)
    b = len(starts)
    x = np.zeros((b, 3, 9, 300), dtype=np.float32)
    roi_valid, prior_valid = np.zeros((b, 3), dtype=bool), np.zeros((b, 3, 6), dtype=bool)
    roi_quality, target = np.zeros((b, 3), dtype=np.float32), np.zeros((b, 300), dtype=np.float32)
    label_valid, target_valid = np.zeros((b, 300), dtype=bool), np.zeros(b, dtype=bool)
    raw_target, frame_indices, interpolated, source_valid, raw_rgb = [], [], [], [], []
    for batch, start in enumerate(starts):
        stop = start + 300
        for r in range(3):
            fraction = float(traces.roi_valid[r, start:stop].mean())
            roi_quality[batch, r] = fraction
            values = uniform[r, start:stop]
            if fraction >= config.min_valid_fraction and np.isfinite(values).all():
                priors, success = build_window_priors(values, 30., PRIOR_NAMES)
                if success.any():
                    roi_valid[batch, r], prior_valid[batch, r] = True, success
                    x[batch, r, :3], x[batch, r, 3:] = standardize_channels(values).T, priors
        label = traces.raw_d_ppg[start:stop]
        label_valid[batch] = np.isfinite(label)
        with np.errstate(over="ignore", invalid="ignore"):
            deviation = float(np.std(label)) if label_valid[batch].all() else float("nan")
        target_valid[batch] = np.isfinite(deviation) and deviation > 1e-8
        if target_valid[batch]:
            target[batch] = standardize_1d(label)
        raw_target.append(label.copy())
        raw_rgb.append(traces.rgb[:, start:stop].copy())
        frame_indices.append(traces.frame_indices[start:stop].copy())
        interpolated.append(imputed[:, start:stop].copy())
        source_valid.append(traces.roi_valid[:, start:stop].copy())
    input_valid = roi_valid.any(-1)
    statuses = ["READY_ENGINEERING" if iv and tv else "NO_USABLE_ROI" if not iv else "INVALID_TARGET"
                for iv, tv in zip(input_valid, target_valid)]
    return dict(x=x, y_ppg=target, roi_valid=roi_valid, roi_quality=roi_quality, prior_valid=prior_valid,
                input_valid=input_valid, target_valid=target_valid, label_valid=label_valid,
                frame_indices=np.stack(frame_indices), label_indices=np.stack(frame_indices).copy(),
                raw_d_ppg=np.stack(raw_target), raw_rgb=np.stack(raw_rgb),
                source_roi_valid=np.stack(source_valid), interpolated=np.stack(interpolated),
                status=statuses, intended_windows=b, roi_names=ROI_NAMES, prior_names=PRIOR_NAMES,
                channel_names=CHANNEL_NAMES,
                provenance=dict(traces.provenance, input_schema="scamps_v1_inputs_v0.1",
                                prior_scope="current_300_sample_window_only",
                                rgb_standardization="signals.standardize_channels_window_local",
                                target_standardization="signals.standardize_1d_window_local",
                                interpolation="processing.resample_trace_bounded_nominal_grid",
                                max_gap_sec=config.max_gap_sec, min_valid_fraction=config.min_valid_fraction,
                                classical_filters="unchanged_METHOD_FUNCS_defaults_0.7_4Hz_order3_filtfilt",
                                nominal_window_starts=list(starts), calibration_frame_indices=[],
                                no_reference_hr_or_bp_computed=True, no_model_invoked=True))


def as_torch_inputs(batch: dict) -> dict:
    """CPU tensor conversion only; never loads/calls any network or checkpoint."""
    import torch
    return {key: torch.from_numpy(np.ascontiguousarray(batch[key]))
            for key in ("x", "roi_valid", "roi_quality", "prior_valid")}
