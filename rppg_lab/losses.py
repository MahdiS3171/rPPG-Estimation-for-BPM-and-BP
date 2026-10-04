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

import torch
import torch.nn.functional as F

EPS = 1e-8


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
