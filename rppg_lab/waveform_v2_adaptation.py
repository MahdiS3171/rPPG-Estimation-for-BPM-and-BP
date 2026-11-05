"""Optional, inference-only session-context prototype; frozen V1 stays unchanged.

Load V1 strictly through its existing API, call eval(), then wrap it. The only
new parameter is context_head.weight. No checkpoint conversion, hooks, labels,
data loading, temporal filtering, resampling, or training is performed here.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
import torch
from torch import nn

from .models import MultiROIPriorResidualWaveformNet
from .waveform_v2_timing_contract import estimate_signed_lag

CONTEXT_FEATURES = ("chroma_r_mean", "chroma_g_mean", "chroma_b_mean",
                    "chroma_r_std", "chroma_g_std", "chroma_b_std",
                    "valid_fraction", "adjacent_chroma_motion")


@dataclass(frozen=True)
class CalibrationSpec:
    """Predeclare one half-open interval on the same session clock as targets."""
    start_s: float = 0.0
    duration_s: float = 30.0

    def __post_init__(self):
        if (not math.isfinite(self.start_s) or not math.isfinite(self.duration_s)
                or self.start_s < 0 or self.duration_s <= 0
                or not math.isfinite(self.end_s)):
            raise ValueError("Calibration requires finite nonnegative start and positive duration")

    @property
    def end_s(self) -> float:
        return self.start_s + self.duration_s


@dataclass(frozen=True)
class SessionContext:
    features: torch.Tensor                 # [B,R,8], label-free and in [0,1]
    usable: torch.Tensor                   # bool [B,R]
    roi_names: tuple[str, ...]
    calibration: CalibrationSpec


@torch.no_grad()
def summarize_calibration(rgb: torch.Tensor, timestamps_s: torch.Tensor,
                          valid: torch.Tensor, *, roi_names: Sequence[str],
                          calibration: CalibrationSpec) -> SessionContext:
    """Pool raw derived RGB [B,R,T,3], reading values only inside calibration.

    No reference/identity arguments exist. Nonfinite, negative, zero-intensity,
    or masked observations count as missing. Caller supplies one session per
    batch row and must keep its context with that same session's target windows.
    """
    names = tuple(roi_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("ROI names must be nonempty and unique")
    if (rgb.ndim != 4 or rgb.shape[1] != len(names) or rgb.shape[-1] != 3
            or rgb.shape[0] < 1 or not rgb.is_floating_point()):
        raise ValueError("rgb must be floating [B,R,T,3] with the declared ROI order")
    if valid.shape != rgb.shape[:-1] or valid.dtype != torch.bool:
        raise ValueError("valid must be bool [B,R,T]")
    t = torch.as_tensor(timestamps_s, dtype=torch.float64, device=rgb.device)
    if (t.ndim != 1 or len(t) != rgb.shape[2] or len(t) < 2
            or not torch.isfinite(t).all() or not torch.all(t[1:] > t[:-1])):
        raise ValueError("timestamps must be finite, strictly increasing [T]")
    selection = (t >= calibration.start_s) & (t < calibration.end_s)
    selected_t = t[selection]
    if selected_t.numel() < 2:
        raise ValueError("Calibration has insufficient timestamp support")
    # Refuse truncated or gapped calibration; never silently select another segment.
    dt = (selected_t[1:] - selected_t[:-1]).median()
    if (selected_t[0] - calibration.start_s > dt * 1.02
            or calibration.end_s - selected_t[-1] > dt * 1.02
            or not torch.all(torch.abs(selected_t[1:] - selected_t[:-1] - dt) <= dt * .02)):
        raise ValueError("Calibration must cover the predeclared interval on a regular grid")
    values = rgb[:, :, selection, :].to(torch.float64)
    good = (valid.to(rgb.device)[:, :, selection] & torch.isfinite(values).all(-1)
            & (values >= 0).all(-1) & (values.amax(-1) > 0))
    values = values.masked_fill(~good[..., None], 0)
    # Scale before summing to keep even very large finite RGB numerically safe.
    scaled = values / values.amax(-1, keepdim=True).clamp_min(torch.finfo(values.dtype).tiny)
    chroma = scaled / scaled.sum(-1, keepdim=True).clamp_min(torch.finfo(values.dtype).tiny)
    count = good.sum(-1)
    mean = chroma.sum(-2) / count.clamp_min(1)[..., None]
    variance = ((chroma - mean[..., None, :]).square() * good[..., None]).sum(-2)
    std = (variance / count.clamp_min(1)[..., None]).sqrt()
    pairs = good[..., 1:] & good[..., :-1]
    motion = ((chroma[..., 1:, :] - chroma[..., :-1, :]).abs().mean(-1) * pairs).sum(-1)
    motion = motion / pairs.sum(-1).clamp_min(1)
    usable = count >= 2
    features = torch.cat((mean, std, (count / len(selected_t))[..., None], motion[..., None]), -1)
    features = features.clamp(0, 1).masked_fill(~usable[..., None], 0)
    return SessionContext(features.detach().clone(), usable.detach().clone(), names, calibration)


class SubjectContextPriorAdapter(nn.Module):
    """Opt-in bounded prior reweighting around an unmodified eval-mode V1.

    alpha' = alpha * exp(offset) / sum(alpha * exp(offset)); this is the
    bounded logit correction to the existing window gate. Already-underflowed
    V1 probabilities remain zero. ROI attention, encoder and residual stay V1.
    Nonzero candidates must pass a per-window engineering phase guard or fall
    back to V1. That guard is not physiological timing or BP validation.
    """
    def __init__(self, v1: MultiROIPriorResidualWaveformNet, *,
                 calibration: CalibrationSpec = CalibrationSpec(),
                 enabled: bool = False, max_logit_offset: float = .25,
                 sample_rate_hz: float = 30.):
        super().__init__()
        if not isinstance(v1, MultiROIPriorResidualWaveformNet) or v1.training:
            raise ValueError("Wrap an existing MultiROIPriorResidualWaveformNet in eval mode")
        if not math.isfinite(max_logit_offset) or not 0 < max_logit_offset <= 1:
            raise ValueError("max_logit_offset must be finite and in (0,1]")
        if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be finite and positive")
        self.v1 = v1
        self.enabled = bool(enabled)
        self.calibration = calibration
        self.max_logit_offset = float(max_logit_offset)
        self.sample_rate_hz = float(sample_rate_hz)
        self.context_head = nn.Linear(len(CONTEXT_FEATURES), v1.num_priors, bias=False)
        self.context_head.to(device=v1.static_prior_logits.device, dtype=v1.static_prior_logits.dtype)
        nn.init.zeros_(self.context_head.weight)
        self.eval()

    def context_offsets(self, context: SessionContext, batch_size: int) -> torch.Tensor:
        shape = (batch_size, self.v1.num_rois)
        if (context.roi_names != self.v1.roi_names or context.calibration != self.calibration
                or context.features.shape != (*shape, len(CONTEXT_FEATURES))
                or context.usable.shape != shape or context.usable.dtype != torch.bool
                or not context.features.is_floating_point()
                or not torch.isfinite(context.features).all()
                or (context.features < 0).any() or (context.features > 1).any()):
            raise ValueError("Context shape, ROI order, interval or finite feature bounds invalid")
        features = context.features.to(self.context_head.weight)
        logits = self.context_head(features)
        if not torch.isfinite(logits).all():
            raise ValueError("Nonfinite context parameters/logits")
        return (self.max_logit_offset * logits.tanh()).masked_fill(
            ~context.usable.to(logits.device)[..., None], 0)

    def _phase_guard(self, before, after, valid_roi):
        """Check final and each valid ROI's absolute added shift on the native grid."""
        t = np.arange(before["ppg"].shape[-1]) / self.sample_rate_hz
        accepted, reasons = [], []
        for b in range(len(before["ppg"])):
            pairs = [(before["ppg"][b], after["ppg"][b])]
            pairs += [(before["roi_ppg"][b, r], after["roi_ppg"][b, r])
                      for r in range(self.v1.num_rois) if valid_roi[b, r]]
            reason = "accepted_engineering_guard"
            for old, new in pairs:
                if torch.equal(old, new):
                    continue
                lag = estimate_signed_lag(t, old.double().cpu().numpy(), new.double().cpu().numpy(),
                                          min_peak_corr=.98)
                if lag.status != "diagnostic_only":
                    reason = "abstain_" + lag.reason
                    break
                if abs(lag.delay_ms) > .5 * 1000 / self.sample_rate_hz:
                    reason = "abstain_added_phase_shift"
                    break
            accepted.append(reason == "accepted_engineering_guard")
            reasons.append(reason)
        return torch.tensor(accepted, device=before["ppg"].device), reasons

    @torch.no_grad()
    def forward(self, x, roi_quality=None, roi_valid=None, prior_valid=None,
                return_dict=False, *, context: SessionContext | None = None,
                target_start_s=None, return_audit=False):
        if self.training or self.v1.training:
            raise ValueError("This prototype supports eval mode only; no training")
        offsets = None
        if self.enabled and context is not None:
            offsets = self.context_offsets(context, x.shape[0])
            if target_start_s is None:
                raise ValueError("Context use requires target start on the calibration clock")
            starts = torch.as_tensor(target_start_s, dtype=torch.float64, device=x.device)
            if (starts.shape != (x.shape[0],) or not torch.isfinite(starts).all()
                    or (starts < self.calibration.end_s).any()):
                raise ValueError("Targets must follow calibration with no overlap or future context")
        original = self.v1(x, roi_quality, roi_valid, prior_valid, return_dict=True)
        result = original
        status = "disabled" if not self.enabled else "missing_context"
        adapted = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        if offsets is not None:
            status = "identity_zero_offset"
            if torch.count_nonzero(offsets):
                alpha = original["prior_weights"]
                # Work in float64 for stable bounded reweighting, including tiny probabilities.
                adjusted = alpha.double() * offsets.double().exp()
                denominator = adjusted.sum(-1, keepdim=True)
                adjusted = (adjusted / denominator.clamp_min(torch.finfo(adjusted.dtype).tiny)).to(alpha)
                # Keep every zero-offset ROI bit-exact, including missing context ROIs.
                unchanged = (offsets == 0).all(-1)
                adjusted = torch.where(unchanged[..., None], alpha, adjusted)
                effective_valid = alpha.sum(-1) > 0
                priors = x[:, :, self.v1.prior_start:, :].masked_fill(~(alpha > 0)[..., None], 0)
                candidate = dict(original)
                candidate["prior_weights"] = adjusted
                fused = (adjusted[..., None] * priors).sum(2)
                candidate["fused_priors"] = torch.where(unchanged[..., None], original["fused_priors"], fused)
                candidate["roi_ppg"] = candidate["fused_priors"] + self.v1.residual_scale * original["residuals"]
                candidate["ppg"] = (original["roi_attention"][..., None] * candidate["roi_ppg"]).sum(1)
                adapted, status = self._phase_guard(original, candidate, effective_valid)
                result = {k: torch.where(adapted.reshape((-1,) + (1,) * (v.ndim - 1)), candidate[k], v)
                          for k, v in original.items()}
        statuses = [status] * x.shape[0] if isinstance(status, str) else status
        audit = dict(intended_windows=x.shape[0], adapted_windows=int(adapted.sum()),
                     fallback_or_identity_windows=int((~adapted).sum()), status_counts=dict(Counter(statuses)),
                     timing_scope="engineering_guard_only_not_physiological_validation")
        value = result if return_dict else result["ppg"]
        return (value, audit) if return_audit else value
