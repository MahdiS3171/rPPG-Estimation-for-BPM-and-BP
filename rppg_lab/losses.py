"""Loss functions for waveform-first rPPG training.

Important design note
---------------------
Facial rPPG and contact finger PPG/BVP are related but not identical signals.
They can be shifted by pulse transit delay and may have different morphology.
For HR/rPPG training, a useful objective should therefore combine:

1. a lag-tolerant waveform term,
2. a frequency/HR-distribution term,
3. a mild spectral-shape term,
4. small regularizers that prevent pathological residuals.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from .waveform_alignment import aligned_overlap, lag_candidates

EPS = 1e-8


@dataclass(frozen=True)
class WaveformV1LossConfig:
    """Engineering starting weights; no anatomical fiducial equivalence assumed."""

    w_corr: float = 1.0
    w_d1: float = 0.25
    w_spec: float = 0.10
    w_hr: float = 0.10
    w_res: float = 0.02
    max_lag_sec: float = 0.5
    derivative_internal_weight: float = 0.25
    spectral_fmin_hz: float = 0.7
    spectral_fmax_hz: float = 8.0

    def __post_init__(self):
        values = tuple(vars(self).values())
        if not all(math.isfinite(v) and v >= 0 for v in values):
            raise ValueError("Waveform v1 configuration must be finite and nonnegative")
        if self.spectral_fmax_hz <= self.spectral_fmin_hz:
            raise ValueError("spectral_fmax_hz must exceed spectral_fmin_hz")


def waveform_spectral_band(fs: float, fmin: float = 0.7, fmax: float = 8.0) -> tuple[float, float]:
    """Effective morphology band, conservatively capped below Nyquist."""
    if not all(math.isfinite(v) for v in (fs, fmin, fmax)) or fs <= 0 or fmin < 0 or fmax <= fmin:
        raise ValueError("Invalid sampling rate or spectral band")
    return float(fmin), min(float(fmax), 0.45 * float(fs))


def _pearson_loss_per_sample(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.shape[-1] < 2:
        # Undefined correlation is neutral for optimization, not a perfect fit.
        return pred.sum(dim=-1) * 0 + 1
    p = pred - pred.mean(dim=-1, keepdim=True)
    t = target - target.mean(dim=-1, keepdim=True)
    corr = (p * t).sum(dim=-1) / torch.sqrt(
        (p.square().sum(dim=-1) + EPS) * (t.square().sum(dim=-1) + EPS))
    return 1 - corr.clamp(-1, 1)


def same_lag_waveform_derivative_loss(
    pred: torch.Tensor, target: torch.Tensor, max_lag: int = 15,
    derivative_internal_weight: float = 0.25,
) -> dict[str, torch.Tensor]:
    """Select one lag per sample minimizing Lcorr + internal_weight * Ld1.

    Components and selected_lag are [B]. Gather retains gradients through both
    selected overlaps. Polarity is preserved. No derivative-specific alignment.
    """
    if pred.ndim != 2 or pred.shape != target.shape or pred.shape[0] < 1:
        raise ValueError("prediction and target must have matching nonempty [B,L] shapes")
    if not math.isfinite(derivative_internal_weight) or derivative_internal_weight < 0:
        raise ValueError("derivative_internal_weight must be finite and nonnegative")
    lags = lag_candidates(pred.shape[-1], max_lag)
    corr, d1 = [], []
    for lag in lags:
        p, t = aligned_overlap(pred, target, lag)
        corr.append(_pearson_loss_per_sample(p, t))
        d1.append(_pearson_loss_per_sample(p.diff(dim=-1), t.diff(dim=-1)))
    corr, d1 = torch.stack(corr), torch.stack(d1)
    chosen = (corr + derivative_internal_weight * d1).argmin(dim=0, keepdim=True)
    return dict(waveform_corr=corr.gather(0, chosen).squeeze(0),
                derivative_corr=d1.gather(0, chosen).squeeze(0),
                selected_lag=pred.new_tensor(lags, dtype=torch.long)[chosen.squeeze(0)])


def waveform_spectral_distance(
    pred: torch.Tensor, target: torch.Tensor, fs: float,
    fmin: float = 0.7, fmax: float = 8.0,
) -> torch.Tensor:
    """Per-window Jensen-Shannon distance of normalized tapered band PSDs.

    This is JS divergence / ln(2), bounded by [0,1], zero for equal shapes.
    Reuses the legacy FFT helper without changing it. Site-dependent harmonics
    make this an auxiliary diagnostic/low-weight loss. No FFT bins returns NaN;
    the training objective explicitly skips that unavailable auxiliary term.
    """
    low, high = waveform_spectral_band(fs, fmin, fmax)
    p, _ = _band_psd(pred, fs, low, high)
    t, _ = _band_psd(target, fs, low, high)
    if p.shape[-1] == 0:
        return pred.sum(dim=-1) * 0 + float("nan")
    p = p / p.sum(dim=-1, keepdim=True)
    t = t / t.sum(dim=-1, keepdim=True)
    m = (p + t) / 2
    js = 0.5 * (p * (p.clamp_min(EPS).log() - m.clamp_min(EPS).log())
                + t * (t.clamp_min(EPS).log() - m.clamp_min(EPS).log())).sum(dim=-1)
    return (js / math.log(2)).clamp(0, 1)


def waveform_v1_training_loss(
    pred: torch.Tensor, target: torch.Tensor, hr_bpm: torch.Tensor, fs: float,
    residuals: torch.Tensor | None = None, roi_valid: torch.Tensor | None = None,
    config: WaveformV1LossConfig | None = None, return_components: bool = False,
) -> torch.Tensor | dict[str, torch.Tensor]:
    """Morphology-oriented waveform objective, opt-in for legacy callers.

    L = w_corr*Lcorr + w_d1*Ld1 + w_spec*JS + w_hr*HR_CE + w_res*residual_MSE.
    No generic smoothing, second derivative, notch, RI or quality-head target.
    Residuals are unscaled [B,R,L] with a validity mask, or prepared [Nvalid,L].
    Scalar component dictionary is optional; otherwise returns a scalar tensor.
    """
    cfg = config or WaveformV1LossConfig()
    waveform_spectral_band(fs, cfg.spectral_fmin_hz, cfg.spectral_fmax_hz)
    terms = same_lag_waveform_derivative_loss(pred, target, int(round(cfg.max_lag_sec * fs)),
                                            cfg.derivative_internal_weight)
    spectral = waveform_spectral_distance(pred, target, fs, cfg.spectral_fmin_hz, cfg.spectral_fmax_hz)
    components = dict(waveform_corr=terms["waveform_corr"].mean(),
                      derivative_corr=terms["derivative_corr"].mean(),
                      spectral=torch.nan_to_num(spectral, nan=0.0).mean(),
                      hr=hr_label_distribution_loss(pred, hr_bpm, fs),
                      residual=pred.sum() * 0)
    if residuals is not None:
        if roi_valid is not None:
            if residuals.ndim != 3 or roi_valid.dtype != torch.bool or roi_valid.shape != residuals.shape[:2]:
                raise ValueError("residuals [B,R,L] require bool roi_valid [B,R]")
            residuals = residuals[roi_valid.to(residuals.device)]
        elif residuals.ndim != 2:
            raise ValueError("Pass prepared [Nvalid,L] residuals or [B,R,L] with roi_valid")
        if residuals.numel():
            components["residual"] = residuals.square().mean()
    weights = dict(waveform_corr=cfg.w_corr, derivative_corr=cfg.w_d1,
                   spectral=cfg.w_spec, hr=cfg.w_hr, residual=cfg.w_res)
    components["total"] = sum(weights[k] * value for k, value in components.items())
    return components if return_components else components["total"]


def negative_pearson_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - Pearson correlation averaged over batch.

    pred and target: (B, L)
    """
    pred = pred - pred.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    num = (pred * target).sum(dim=-1)
    den = torch.sqrt((pred.pow(2).sum(dim=-1) + EPS) * (target.pow(2).sum(dim=-1) + EPS))
    corr = num / den
    return (1.0 - corr).mean()


def time_shifted_negative_pearson_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    max_lag: int = 15,
) -> torch.Tensor:
    """Time-shift tolerant Pearson loss.

    This is useful because face-rPPG and finger/contact PPG can have a small
    physiological delay.  The returned value is the best correlation loss over
    lags in ``[-max_lag, +max_lag]``.
    """
    losses = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            p = pred[..., :lag]
            t = target[..., -lag:]
        elif lag > 0:
            p = pred[..., lag:]
            t = target[..., :-lag]
        else:
            p = pred
            t = target

        if p.shape[-1] < 8:
            continue

        p = p - p.mean(dim=-1, keepdim=True)
        t = t - t.mean(dim=-1, keepdim=True)
        num = (p * t).sum(dim=-1)
        den = torch.sqrt((p.pow(2).sum(dim=-1) + EPS) * (t.pow(2).sum(dim=-1) + EPS))
        corr = num / den
        losses.append(1.0 - corr)

    if not losses:
        return negative_pearson_loss(pred, target)
    stacked = torch.stack(losses, dim=0)  # (num_lags, B)
    return stacked.min(dim=0).values.mean()


def _band_psd(x: torch.Tensor, fs: float, fmin: float = 0.7, fmax: float = 3.5):
    """Return HR-band PSD and frequencies in bpm."""
    x = x - x.mean(dim=-1, keepdim=True)
    # Hann taper reduces leakage for short 8-10 s windows.
    win = torch.hann_window(x.shape[-1], device=x.device, dtype=x.dtype)
    xw = x * win.view(1, -1)
    psd = torch.fft.rfft(xw, dim=-1).abs().pow(2) + EPS
    freqs = torch.fft.rfftfreq(x.shape[-1], d=1.0 / float(fs)).to(x.device)
    band = (freqs >= fmin) & (freqs <= fmax)
    return psd[:, band], freqs[band] * 60.0


def spectral_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE between normalized magnitude spectra over all FFT bins."""
    P = torch.fft.rfft(pred - pred.mean(dim=-1, keepdim=True), dim=-1).abs()
    T = torch.fft.rfft(target - target.mean(dim=-1, keepdim=True), dim=-1).abs()
    P = P / (P.sum(dim=-1, keepdim=True) + EPS)
    T = T / (T.sum(dim=-1, keepdim=True) + EPS)
    return F.mse_loss(P, T)


def hr_label_distribution_loss(
    pred: torch.Tensor,
    hr_bpm: torch.Tensor,
    fs: float,
    fmin: float = 0.7,
    fmax: float = 3.5,
    sigma_bpm: float = 3.0,
    temperature: float = 0.5,
) -> torch.Tensor:
    """Stable frequency-domain HR supervision.

    The previous raw-PSD-softmax version can create extremely large losses,
    because raw PSD magnitudes are not calibrated logits.  Here the logits are
    the log-PSD values, which makes the objective scale-stable and comparable
    across windows.
    """
    psd_b, f_bpm = _band_psd(pred, fs=fs, fmin=fmin, fmax=fmax)
    if psd_b.shape[-1] < 2:
        return pred.new_tensor(0.0)

    logits = torch.log(psd_b + EPS) / float(temperature)
    log_probs = torch.log_softmax(logits, dim=-1)

    hr = hr_bpm.to(pred.device).float().view(-1, 1)
    target = torch.exp(-0.5 * ((f_bpm.view(1, -1) - hr) / float(sigma_bpm)) ** 2)
    target = target / (target.sum(dim=-1, keepdim=True) + EPS)
    return -(target * log_probs).sum(dim=-1).mean()


def spectral_shape_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    fs: float,
    fmin: float = 0.7,
    fmax: float = 3.5,
) -> torch.Tensor:
    """KL-like loss between normalized HR-band spectra of pred and target."""
    psd_p, _ = _band_psd(pred, fs=fs, fmin=fmin, fmax=fmax)
    psd_t, _ = _band_psd(target, fs=fs, fmin=fmin, fmax=fmax)
    p = psd_p / (psd_p.sum(dim=-1, keepdim=True) + EPS)
    t = psd_t / (psd_t.sum(dim=-1, keepdim=True) + EPS)
    return -(t.detach() * torch.log(p + EPS)).sum(dim=-1).mean()


def smoothness_loss(pred: torch.Tensor) -> torch.Tensor:
    """Small penalty on sample-to-sample roughness."""
    return (pred[..., 1:] - pred[..., :-1]).pow(2).mean()


def waveform_combo_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    pearson_weight: float = 1.0,
    spectral_weight: float = 0.2,
    mse_weight: float = 0.1,
    smooth_weight: float = 0.01,
) -> torch.Tensor:
    """Legacy waveform loss kept for ablations."""
    return (
        pearson_weight * negative_pearson_loss(pred, target)
        + spectral_weight * spectral_mse_loss(pred, target)
        + mse_weight * F.mse_loss(pred, target)
        + smooth_weight * smoothness_loss(pred)
    )


def rppg_training_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    hr_bpm: torch.Tensor,
    fs: float,
    epoch: int = 1,
    total_epochs: int = 1,
    max_lag_sec: float = 0.5,
    time_weight: float = 0.50,
    hr_weight: float = 0.50,
    shape_weight: float = 0.10,
    smooth_weight: float = 0.001,
    residual: torch.Tensor | None = None,
    residual_weight: float = 0.02,
) -> torch.Tensor:
    """Main stable rPPG training objective.

    The loss is deliberately conservative.  It should not produce huge values
    like 100+, and it should not force exact zero-lag matching between facial
    rPPG and finger PPG.
    """
    max_lag = int(round(max_lag_sec * fs))

    time_loss = time_shifted_negative_pearson_loss(pred, target, max_lag=max_lag)
    hr_loss = hr_label_distribution_loss(pred=pred, hr_bpm=hr_bpm, fs=fs, sigma_bpm=3.0)
    shape_loss = spectral_shape_loss(pred=pred, target=target, fs=fs)
    smooth = smoothness_loss(pred)

    loss = (
        float(time_weight) * time_loss
        + float(hr_weight) * hr_loss
        + float(shape_weight) * shape_loss
        + float(smooth_weight) * smooth
    )

    if residual is not None and residual_weight > 0:
        loss = loss + float(residual_weight) * residual.pow(2).mean()

    return loss


def soft_hr_distribution_from_labels(
    hr_bpm: torch.Tensor,
    hr_bins: torch.Tensor,
    sigma_bpm: float = 3.0,
) -> torch.Tensor:
    hr = hr_bpm.float().view(-1, 1).to(hr_bins.device)
    bins = hr_bins.float().view(1, -1)
    target = torch.exp(-0.5 * ((bins - hr) / float(sigma_bpm)) ** 2)
    target = target / (target.sum(dim=-1, keepdim=True) + EPS)
    return target


def hr_distribution_loss(
    hr_logits: torch.Tensor,
    hr_bpm: torch.Tensor,
    hr_bins: torch.Tensor,
    sigma_bpm: float = 3.0,
) -> torch.Tensor:
    target = soft_hr_distribution_from_labels(hr_bpm, hr_bins, sigma_bpm=sigma_bpm)
    log_probs = torch.log_softmax(hr_logits, dim=-1)
    return -(target * log_probs).sum(dim=-1).mean()


def hr_quality_training_loss(
    hr_logits: torch.Tensor,
    hr_bpm: torch.Tensor,
    hr_bins: torch.Tensor,
    quality_logit: torch.Tensor | None = None,
    quality_target: torch.Tensor | None = None,
    sigma_bpm: float = 3.0,
    quality_weight: float = 0.25,
) -> torch.Tensor:
    loss = hr_distribution_loss(hr_logits, hr_bpm, hr_bins, sigma_bpm=sigma_bpm)
    if quality_logit is not None and quality_target is not None and quality_weight > 0:
        qt = torch.clamp(quality_target.float().to(quality_logit.device), 0.0, 1.0)
        loss = loss + float(quality_weight) * F.binary_cross_entropy_with_logits(quality_logit, qt)
    return loss



def hr_quality_selection_training_loss(
    hr_logits: torch.Tensor,
    hr_bpm: torch.Tensor,
    hr_bins: torch.Tensor,
    final_hr_bpm: torch.Tensor | None = None,
    selection_logits: torch.Tensor | None = None,
    best_candidate_index: torch.Tensor | None = None,
    selection_target_probs: torch.Tensor | None = None,
    quality_logit: torch.Tensor | None = None,
    quality_target: torch.Tensor | None = None,
    sigma_bpm: float = 3.0,
    dist_weight: float = 1.0,
    hr_reg_weight: float = 0.10,
    selection_weight: float = 0.50,
    soft_selection_weight: float = 0.0,
    quality_weight: float = 0.20,
) -> torch.Tensor:
    """HRQualityNetV2 objective.

    Components:
      - HR distribution loss: learned temporal features must localize HR.
      - Smooth-L1 HR regression on the final blended HR.
      - Candidate-selection loss: learn which fixed input/prior HR estimate is best.
      - Optional soft candidate-selection loss: assign nonzero probability to
        candidates with near-optimal HR errors, which is more stable than a
        hard argmin when several priors are effectively tied.
      - Quality loss: learn whether at least one candidate is reliable.
    """
    loss = float(dist_weight) * hr_distribution_loss(hr_logits, hr_bpm, hr_bins, sigma_bpm=sigma_bpm)

    if final_hr_bpm is not None and hr_reg_weight > 0:
        loss = loss + float(hr_reg_weight) * F.smooth_l1_loss(final_hr_bpm.float(), hr_bpm.float(), beta=2.0)

    if selection_logits is not None and best_candidate_index is not None and selection_weight > 0:
        target = best_candidate_index.long().to(selection_logits.device)
        valid = target >= 0
        if valid.any():
            loss = loss + float(selection_weight) * F.cross_entropy(selection_logits[valid], target[valid])

    if selection_logits is not None and selection_target_probs is not None and soft_selection_weight > 0:
        target_probs = selection_target_probs.float().to(selection_logits.device)
        valid_soft = target_probs.sum(dim=1) > 0
        if valid_soft.any():
            log_probs = torch.log_softmax(selection_logits[valid_soft], dim=-1)
            target_probs = target_probs[valid_soft]
            target_probs = target_probs / (target_probs.sum(dim=-1, keepdim=True) + EPS)
            soft_ce = -(target_probs * log_probs).sum(dim=-1).mean()
            loss = loss + float(soft_selection_weight) * soft_ce

    if quality_logit is not None and quality_target is not None and quality_weight > 0:
        qt = torch.clamp(quality_target.float().to(quality_logit.device), 0.0, 1.0)
        loss = loss + float(quality_weight) * F.binary_cross_entropy_with_logits(quality_logit, qt)

    return loss
