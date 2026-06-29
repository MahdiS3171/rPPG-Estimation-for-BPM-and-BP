"""PyTorch models for the rPPG/BP project.

The main scientific path is waveform-first: predict an rPPG/PPG waveform and
estimate HR from the waveform.  For hybrid classical+deep experiments, the
recommended model is ``PriorResidualWaveformNet``: it starts from a weighted
classical-prior waveform and learns only a small residual correction.  This is
important because several classical priors are already strong baselines, and a
deep model should not be allowed to destroy a good prior while trying to rebuild
a waveform from scratch.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 7, dilation: int = 1, dropout: float = 0.0):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation),
            nn.BatchNorm1d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.net(x), inplace=True)


class TemporalEncoder1D(nn.Module):
    """Dilated residual temporal encoder for RGB/prior traces."""

    def __init__(self, in_channels: int, base_channels: int = 32, num_blocks: int = 4, dropout: float = 0.15):
        super().__init__()
        self.input = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
        )
        blocks = []
        for i in range(num_blocks):
            blocks.append(ResidualBlock1D(base_channels, kernel_size=7, dilation=2 ** (i % 4), dropout=dropout))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.input(x))


class HR1DCNN(nn.Module):
    """Simple direct HR regression baseline.

    Input:  (B, C, L)
    Output: (B,)
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x).mean(dim=-1)
        return self.head(z).squeeze(-1)


class WaveformNet(nn.Module):
    """Generic waveform model.

    This model is useful as an RGB-only baseline.  When classical prior channels
    are available, prefer ``PriorResidualWaveformNet`` because it is initialized
    to preserve a known-good prior and only learns a residual correction.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 32, num_blocks: int = 4, dropout: float = 0.15):
        super().__init__()
        self.encoder = TemporalEncoder1D(in_channels, base_channels, num_blocks, dropout)
        self.ppg_head = nn.Conv1d(base_channels, 1, kernel_size=7, padding=3)
        self.quality_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(base_channels, 1), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, return_dict: bool = False):
        feat = self.encoder(x)
        ppg = self.ppg_head(feat).squeeze(1)
        quality = self.quality_head(feat).squeeze(-1)
        if return_dict:
            return {"ppg": ppg, "quality": quality, "features": feat}
        return ppg


def _initial_prior_logits(prior_names: Sequence[str], init_prior: str = "auto") -> torch.Tensor:
    """Conservative initialization for prior fusion weights.

    For UBFC-style face data, CHROM/CHROM_WIN are usually strong.  GREEN is also
    important cross-dataset, so it receives non-negligible weight.  This is only
    an initialization; dynamic gating can adjust the weights during training.
    """
    names = list(prior_names)
    logits = torch.full((len(names),), -2.0, dtype=torch.float32)

    if not names:
        return logits

    if init_prior and init_prior != "auto" and init_prior in names:
        logits[names.index(init_prior)] = 5.0
        return logits

    # General robust prior preference, not a hard rule.
    preferences = {
        "CHROM_WIN": 5.0,
        "CHROM": 3.0,
        "GREEN": 1.5,
        "POS_WIN": -0.5,
        "OMIT": -0.5,
        "PBV": -1.0,
        "LGI": -1.5,
    }
    for i, name in enumerate(names):
        logits[i] = preferences.get(name, -2.0)
    return logits


class PriorResidualWaveformNet(nn.Module):
    """Prior-preserving hybrid rPPG waveform model.

    Input channels are expected to be ``[R, G, B, prior_1, ..., prior_K]``.
    The model output is

        output = weighted_sum(classical_priors) + residual_scale * learned_residual

    The residual head is zero-initialized, so before training the network behaves
    like a conservative fusion of the classical priors.  This prevents the model
    from destroying strong priors such as CHROM_WIN during early training.
    """

    def __init__(
        self,
        in_channels: int,
        prior_names: Sequence[str],
        prior_start: int = 3,
        base_channels: int = 32,
        num_blocks: int = 3,
        dropout: float = 0.20,
        residual_scale: float = 0.10,
        init_prior: str = "auto",
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.prior_start = int(prior_start)
        self.prior_names = list(prior_names)
        self.num_priors = len(self.prior_names)
        self.residual_scale = float(residual_scale)

        if self.num_priors <= 0:
            raise ValueError("PriorResidualWaveformNet requires at least one prior channel.")
        if self.in_channels < self.prior_start + self.num_priors:
            raise ValueError(
                f"in_channels={in_channels} is inconsistent with prior_start={prior_start} "
                f"and num_priors={self.num_priors}."
            )

        self.encoder = TemporalEncoder1D(in_channels, base_channels, num_blocks, dropout)

        self.static_prior_logits = nn.Parameter(_initial_prior_logits(self.prior_names, init_prior=init_prior))
        self.gate_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(base_channels, self.num_priors),
        )
        # Start with purely static weights; learn dynamic corrections only if useful.
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.zeros_(self.gate_head[-1].bias)

        self.residual_head = nn.Conv1d(base_channels, 1, kernel_size=7, padding=3)
        # Zero init makes the initial model exactly a prior fusion.
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

        self.quality_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(base_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, return_dict: bool = False):
        priors = x[:, self.prior_start:self.prior_start + self.num_priors, :]
        feat = self.encoder(x)

        dynamic_delta = self.gate_head(feat)
        logits = self.static_prior_logits.view(1, -1) + dynamic_delta
        alpha = torch.softmax(logits, dim=1)

        fused_prior = (alpha[:, :, None] * priors).sum(dim=1)
        residual = self.residual_head(feat).squeeze(1)
        ppg = fused_prior + self.residual_scale * residual
        quality = self.quality_head(feat).squeeze(-1)

        if return_dict:
            return {
                "ppg": ppg,
                "quality": quality,
                "features": feat,
                "prior_weights": alpha,
                "fused_prior": fused_prior,
                "residual": residual,
            }
        return ppg


class MultiROIWaveformNet(nn.Module):
    """Shared temporal encoder + learnable ROI attention.

    Input shape: (B, R, C, L), where R is number of ROIs and C is channels per
    ROI (RGB + optional classical priors). Optional external roi_quality has
    shape (B, R) and biases attention away from bad ROI traces.
    """

    def __init__(self, in_channels: int, base_channels: int = 32, num_blocks: int = 4, dropout: float = 0.15):
        super().__init__()
        self.shared = TemporalEncoder1D(in_channels, base_channels, num_blocks, dropout)
        self.attn_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(base_channels, 1)
        )
        self.ppg_head = nn.Conv1d(base_channels, 1, kernel_size=7, padding=3)
        self.quality_head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(base_channels, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor, roi_quality: Optional[torch.Tensor] = None, return_dict: bool = False):
        b, r, c, l = x.shape
        flat = x.reshape(b * r, c, l)
        feat = self.shared(flat)  # (B*R, F, L)
        logits = self.attn_head(feat).reshape(b, r)
        if roi_quality is not None:
            logits = logits + torch.log(torch.clamp(roi_quality, min=1e-4))
        alpha = torch.softmax(logits, dim=1)  # (B, R)
        feat = feat.reshape(b, r, feat.shape[1], l)
        fused = (feat * alpha[:, :, None, None]).sum(dim=1)
        ppg = self.ppg_head(fused).squeeze(1)
        quality = self.quality_head(fused).squeeze(-1)
        if return_dict:
            return {"ppg": ppg, "quality": quality, "roi_attention": alpha, "features": fused}
        return ppg


class BPFeatureMLP(nn.Module):
    """First BP baseline for engineered features + metadata + cuff calibration.

    This is intentionally a feature-level baseline. Do not use it to claim BP from
    face video alone. It is for pilot data and sanity checks.
    """

    def __init__(self, in_features: int, hidden: int = 128, dropout: float = 0.15, out_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HealthMultiTaskNet(nn.Module):
    """Forward-compatible multi-task scaffold.

    Main supervised output should remain `ppg` until we have a clean BP dataset.
    BP heads are optional and should only be trained with subject-wise splits,
    calibration metadata, and proper reference measurements.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 32, num_blocks: int = 4, dropout: float = 0.15):
        super().__init__()
        self.backbone = WaveformNet(in_channels, base_channels, num_blocks, dropout)
        self.hr_head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(base_channels, 64), nn.ReLU(), nn.Linear(64, 1))
        self.bp_head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(base_channels, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.backbone(x, return_dict=True)
        feat = out["features"]
        return {
            "ppg": out["ppg"],
            "quality": out["quality"],
            "hr": self.hr_head(feat).squeeze(-1),
            "bp": self.bp_head(feat),
        }


class HRQualityNet(nn.Module):
    """Direct HR-distribution + quality model for window-level rPPG.

    This model is intentionally not waveform-supervised.  It predicts a
    probability distribution over HR bins and a confidence/quality score.  This
    matches the practical target better than forcing facial rPPG to reproduce a
    finger PPG waveform point-by-point.

    Input:  x with shape (B, C, L), usually RGB + classical-prior traces.
    Output dict:
        hr_logits:  (B, num_bins)
        hr_probs:   (B, num_bins)
        hr_bpm:     (B,) expected HR under hr_probs
        quality_logit: (B,)
        quality:    (B,) sigmoid confidence
    """

    def __init__(
        self,
        in_channels: int,
        hr_min: float = 40.0,
        hr_max: float = 180.0,
        hr_step: float = 1.0,
        base_channels: int = 48,
        num_blocks: int = 4,
        dropout: float = 0.20,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.hr_min = float(hr_min)
        self.hr_max = float(hr_max)
        self.hr_step = float(hr_step)

        bins = torch.arange(self.hr_min, self.hr_max + 0.5 * self.hr_step, self.hr_step, dtype=torch.float32)
        self.register_buffer("hr_bins", bins)

        self.encoder = TemporalEncoder1D(
            in_channels=self.in_channels,
            base_channels=base_channels,
            num_blocks=num_blocks,
            dropout=dropout,
        )

        pooled_dim = base_channels * 2
        self.pool_dropout = nn.Dropout(dropout)
        self.hr_head = nn.Sequential(
            nn.Linear(pooled_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, bins.numel()),
        )
        self.quality_head = nn.Sequential(
            nn.Linear(pooled_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor, return_dict: bool = True):
        feat = self.encoder(x)
        avg = feat.mean(dim=-1)
        std = feat.std(dim=-1, unbiased=False)
        pooled = self.pool_dropout(torch.cat([avg, std], dim=1))

        logits = self.hr_head(pooled)
        probs = torch.softmax(logits, dim=-1)
        hr_bpm = (probs * self.hr_bins.view(1, -1)).sum(dim=-1)
        quality_logit = self.quality_head(pooled).squeeze(-1)
        quality = torch.sigmoid(quality_logit)

        if return_dict:
            return {
                "hr_logits": logits,
                "hr_probs": probs,
                "hr_bpm": hr_bpm,
                "quality_logit": quality_logit,
                "quality": quality,
                "features": feat,
            }
        return hr_bpm
