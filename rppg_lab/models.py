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


class MultiROIPriorResidualWaveformNet(nn.Module):
    """Conservative, window-level prior and ROI fusion with shared heads.

    Input ``x`` is [B,R,C,L], with channels [R,G,B,prior_1,...,prior_K].
    With ``return_dict=True``: ppg [B,L], quality [B], features [B,D,L],
    roi_attention [B,R], roi_ppg/fused_priors/residuals [B,R,L], and
    prior_weights [B,R,K]. Invalid branches return zero weights/waveforms.
    ROI and prior ordering are explicit constructor/checkpoint configuration.
    """

    def __init__(
        self,
        in_channels: int,
        prior_names: Sequence[str],
        roi_names: Sequence[str] = ("forehead", "left_cheek", "right_cheek"),
        prior_start: int = 3,
        base_channels: int = 32,
        num_blocks: int = 3,
        dropout: float = 0.20,
        residual_scale: float = 0.10,
        init_prior: str = "auto",
    ) -> None:
        super().__init__()
        if isinstance(roi_names, str) or not roi_names or len(set(roi_names)) != len(roi_names):
            raise ValueError("roi_names must be nonempty and unique")
        if isinstance(prior_names, str) or not prior_names or len(set(prior_names)) != len(prior_names):
            raise ValueError("prior_names must be nonempty and unique")
        self.in_channels = int(in_channels)
        self.prior_names = tuple(prior_names)
        self.roi_names = tuple(roi_names)
        self.prior_start = int(prior_start)
        self.num_priors = len(self.prior_names)
        self.num_rois = len(self.roi_names)
        self.residual_scale = float(residual_scale)
        if self.prior_start < 3 or self.in_channels != self.prior_start + self.num_priors:
            raise ValueError("in_channels must equal prior_start + len(prior_names), with prior_start >= 3")
        if base_channels < 1 or num_blocks < 0 or not 0 <= dropout < 1:
            raise ValueError("Invalid encoder width, block count or dropout")
        if not torch.isfinite(torch.tensor(self.residual_scale)) or self.residual_scale < 0:
            raise ValueError("residual_scale must be finite and nonnegative")
        self.model_config = dict(
            in_channels=self.in_channels, prior_names=self.prior_names, roi_names=self.roi_names,
            prior_start=self.prior_start, base_channels=base_channels, num_blocks=num_blocks,
            dropout=dropout, residual_scale=self.residual_scale, init_prior=init_prior,
        )
        self.shared = TemporalEncoder1D(self.in_channels, base_channels, num_blocks, dropout)
        self.static_prior_logits = nn.Parameter(_initial_prior_logits(self.prior_names, init_prior))
        self.roi_prior_bias = nn.Parameter(torch.zeros(self.num_rois, self.num_priors))
        self.gate_head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                       nn.Linear(base_channels, self.num_priors))
        self.residual_head = nn.Conv1d(base_channels, 1, kernel_size=7, padding=3)
        self.static_roi_logits = nn.Parameter(torch.zeros(self.num_rois))
        self.attn_head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                       nn.Linear(base_channels, 1))
        for head in (self.gate_head[-1], self.residual_head, self.attn_head[-1]):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.quality_head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                          nn.Linear(base_channels, 1), nn.Sigmoid())

    def forward(
        self,
        x: torch.Tensor,
        roi_quality: Optional[torch.Tensor] = None,
        roi_valid: Optional[torch.Tensor] = None,
        prior_valid: Optional[torch.Tensor] = None,
        return_dict: bool = False,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        if x.ndim != 4 or x.shape[1:3] != (self.num_rois, self.in_channels):
            raise ValueError(f"x must have shape [B,{self.num_rois},{self.in_channels},L], got {tuple(x.shape)}")
        b, r, c, length = x.shape
        if b < 1 or length < 1 or not x.is_floating_point():
            raise ValueError("x must be floating point with nonempty batch/time dimensions")

        def mask(value: Optional[torch.Tensor], shape: tuple[int, ...], name: str) -> torch.Tensor:
            if value is None:
                return torch.ones(shape, device=x.device, dtype=torch.bool)
            if value.shape != shape or value.dtype != torch.bool:
                raise ValueError(f"{name} must be a bool tensor of shape {shape}")
            return value.to(x.device)

        valid_roi = mask(roi_valid, (b, r), "roi_valid")
        valid_prior = mask(prior_valid, (b, r, self.num_priors), "prior_valid") & valid_roi[..., None]
        if not valid_roi.any(dim=1).all():
            raise ValueError("Every batch sample must have at least one valid ROI (no valid ROI found)")
        if (valid_roi & ~valid_prior.any(dim=-1)).any():
            raise ValueError("Every valid ROI must have at least one valid prior")

        # Remove invalid numeric channels BEFORE encoding; masked NaNs must not
        # contaminate valid branches or batch normalization.
        channel_valid = torch.cat((valid_roi[..., None].expand(b, r, self.prior_start), valid_prior), dim=-1)
        clean = x.masked_fill(~channel_valid[..., None], 0)
        if not torch.isfinite(clean).all():
            raise ValueError("RGB and valid prior channels must be finite")
        # Missing branches must never enter the shared BatchNorm statistics.
        # index_copy preserves the valid features' autograd connection.
        valid_rows = valid_roi.reshape(-1).nonzero(as_tuple=True)[0]
        valid_feat = self.shared(clean.reshape(b * r, c, length).index_select(0, valid_rows))
        feat_flat = valid_feat.new_zeros(b * r, valid_feat.shape[1], length).index_copy(0, valid_rows, valid_feat)
        feat = feat_flat.reshape(b, r, -1, length).masked_fill(~valid_roi[..., None, None], 0)
        logits = (self.static_prior_logits[None, None, :] + self.roi_prior_bias[None, :, :]
                  + self.gate_head(feat_flat).reshape(b, r, self.num_priors))
        logits = logits.masked_fill(~valid_prior, -torch.inf)
        # An invalid ROI has no priors; avoid softmax(-inf,...,-inf) NaNs.
        logits = logits.masked_fill(~valid_roi[..., None], 0)
        alpha = torch.softmax(logits, dim=-1).masked_fill(~valid_prior, 0)
        priors = clean[:, :, self.prior_start:, :]
        fused_priors = (alpha[..., None] * priors).sum(dim=2)
        residuals = self.residual_head(feat_flat).reshape(b, r, length).masked_fill(~valid_roi[..., None], 0)
        roi_ppg = fused_priors + self.residual_scale * residuals
        roi_logits = self.static_roi_logits[None, :] + self.attn_head(feat_flat).reshape(b, r)
        if roi_quality is not None:
            if roi_quality.shape != (b, r) or not roi_quality.is_floating_point() or not torch.isfinite(roi_quality).all():
                raise ValueError(f"roi_quality must be finite floating point of shape {(b, r)}")
            eps = max(1e-6, torch.finfo(x.dtype).tiny)
            roi_logits = roi_logits + roi_quality.to(device=x.device, dtype=x.dtype).clamp(eps, 1).log()
        beta = torch.softmax(roi_logits.masked_fill(~valid_roi, -torch.inf), dim=1)
        ppg = (beta[..., None] * roi_ppg).sum(dim=1)
        features = (beta[..., None, None] * feat).sum(dim=1)
        quality = self.quality_head(features).squeeze(-1)
        if return_dict:
            return dict(ppg=ppg, quality=quality, features=features, roi_attention=beta,
                        roi_ppg=roi_ppg, prior_weights=alpha, fused_priors=fused_priors, residuals=residuals)
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
            nn.Linear(in_features, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout),
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


class HRQualityNetV2(nn.Module):
    """HR + quality model with explicit candidate/prior selection.

    This model is designed for the current project stage.  It receives temporal
    traces (RGB + classical priors) and also the HR estimates obtained from a
    fixed set of candidate input channels (typically RGB_G + prior channels).

    It predicts:
        1) an HR probability distribution from learned temporal features;
        2) a selection distribution over candidate HR estimates;
        3) a final HR as a learnable blend of learned-HR and candidate-fused HR;
        4) a confidence/quality score.

    The auxiliary selection objective lets the network learn *which prior should
    be trusted* in each window, which is exactly the gap between the best single
    prior and the input-channel oracle.
    """

    def __init__(
        self,
        in_channels: int,
        num_candidates: int,
        hr_min: float = 40.0,
        hr_max: float = 180.0,
        hr_step: float = 1.0,
        base_channels: int = 48,
        num_blocks: int = 4,
        dropout: float = 0.20,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_candidates = int(num_candidates)
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
        self.selection_head = nn.Sequential(
            nn.Linear(pooled_dim, 96),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(96, self.num_candidates),
        )
        self.fusion_gate_head = nn.Sequential(
            nn.Linear(pooled_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )
        self.quality_head = nn.Sequential(
            nn.Linear(pooled_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        candidate_hrs: Optional[torch.Tensor] = None,
        candidate_valid: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        feat = self.encoder(x)
        avg = feat.mean(dim=-1)
        std = feat.std(dim=-1, unbiased=False)
        pooled = self.pool_dropout(torch.cat([avg, std], dim=1))

        hr_logits = self.hr_head(pooled)
        hr_probs = torch.softmax(hr_logits, dim=-1)
        learned_hr = (hr_probs * self.hr_bins.view(1, -1)).sum(dim=-1)

        selection_logits = self.selection_head(pooled)
        if candidate_valid is not None:
            selection_logits = selection_logits.masked_fill(~candidate_valid.bool(), -1e4)
        selection_probs = torch.softmax(selection_logits, dim=-1)

        if candidate_hrs is None:
            candidate_hr = learned_hr
            final_hr = learned_hr
            gate = torch.ones_like(learned_hr)
        else:
            cand = candidate_hrs.to(x.device).float()
            if candidate_valid is not None:
                cand = torch.where(candidate_valid.bool().to(x.device), cand, learned_hr.view(-1, 1))
            candidate_hr = (selection_probs * cand).sum(dim=-1)
            gate = torch.sigmoid(self.fusion_gate_head(pooled).squeeze(-1))
            final_hr = gate * learned_hr + (1.0 - gate) * candidate_hr

        quality_logit = self.quality_head(pooled).squeeze(-1)
        quality = torch.sigmoid(quality_logit)

        if return_dict:
            return {
                "hr_logits": hr_logits,
                "hr_probs": hr_probs,
                "learned_hr_bpm": learned_hr,
                "selection_logits": selection_logits,
                "selection_probs": selection_probs,
                "candidate_hr_bpm": candidate_hr,
                "fusion_gate": gate,
                "hr_bpm": final_hr,
                "quality_logit": quality_logit,
                "quality": quality,
                "features": feat,
            }
        return final_hr


class FrameEncoder2D(nn.Module):
    """Small per-frame CNN used by VideoHRNet.

    The encoder is intentionally compact for the first video-stage experiment:
    it should run on a normal laptop/GPU and act as a sanity-check before moving
    to a heavier PhysFormer-like transformer.
    """

    def __init__(self, out_dim: int = 96, dropout: float = 0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.net(frames)


class VideoHRNet(nn.Module):
    """First spatio-temporal video HR + quality model.

    Input: ``video`` with shape ``(B, T, 3, H, W)``.
    Output: HR probability distribution, expected HR, and quality/confidence.

    This model is intentionally lighter than PhysFormer.  It is the bridge
    between the validated 1D HRQualityNet path and the final heavy video model.
    """

    def __init__(
        self,
        hr_min: float = 40.0,
        hr_max: float = 180.0,
        hr_step: float = 1.0,
        frame_feature_dim: int = 96,
        temporal_channels: int = 96,
        num_temporal_blocks: int = 4,
        dropout: float = 0.20,
    ):
        super().__init__()
        self.hr_min = float(hr_min)
        self.hr_max = float(hr_max)
        self.hr_step = float(hr_step)
        bins = torch.arange(self.hr_min, self.hr_max + 0.5 * self.hr_step, self.hr_step, dtype=torch.float32)
        self.register_buffer("hr_bins", bins)

        self.frame_encoder = FrameEncoder2D(out_dim=frame_feature_dim, dropout=dropout * 0.5)
        self.temporal_in = nn.Sequential(
            nn.Conv1d(frame_feature_dim, temporal_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(temporal_channels),
            nn.ReLU(inplace=True),
        )
        blocks = []
        for i in range(num_temporal_blocks):
            blocks.append(ResidualBlock1D(temporal_channels, kernel_size=7, dilation=2 ** (i % 4), dropout=dropout))
        self.temporal_blocks = nn.Sequential(*blocks)
        pooled_dim = temporal_channels * 2
        self.dropout = nn.Dropout(dropout)
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

    def forward(self, video: torch.Tensor, return_dict: bool = True):
        # video: (B,T,3,H,W)
        b, t, c, h, w = video.shape
        frames = video.reshape(b * t, c, h, w)
        frame_feat = self.frame_encoder(frames).reshape(b, t, -1)  # (B,T,F)
        feat = frame_feat.transpose(1, 2).contiguous()  # (B,F,T)
        feat = self.temporal_blocks(self.temporal_in(feat))
        avg = feat.mean(dim=-1)
        std = feat.std(dim=-1, unbiased=False)
        pooled = self.dropout(torch.cat([avg, std], dim=1))
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
                "frame_features": frame_feat,
            }
        return hr_bpm


class VideoPriorHRNet(nn.Module):
    """Prior-guided video HR + quality model.

    This is the recommended second video-stage model.  It receives both:
      * raw face video clips: (B, T, 3, H, W)
      * validated 1D traces/priors: (B, C, L)
      * candidate HR estimates from RGB_G + prior channels: (B, K)

    The model is deliberately initialized to preserve a strong candidate prior
    (CHROM_WIN when present).  Therefore epoch-0 performance should be close to
    the best classical prior instead of the poor random-video baseline.
    The video branch is then learned as a residual/correction and quality cue.
    """

    def __init__(
        self,
        trace_channels: int,
        num_candidates: int,
        candidate_names: Optional[Sequence[str]] = None,
        hr_min: float = 40.0,
        hr_max: float = 180.0,
        hr_step: float = 1.0,
        frame_feature_dim: int = 64,
        video_temporal_channels: int = 64,
        trace_channels_hidden: int = 48,
        num_video_blocks: int = 3,
        num_trace_blocks: int = 3,
        dropout: float = 0.20,
        residual_scale_bpm: float = 5.0,
    ):
        super().__init__()
        self.trace_channels = int(trace_channels)
        self.num_candidates = int(num_candidates)
        self.candidate_names = list(candidate_names or [f"cand_{i}" for i in range(num_candidates)])
        self.hr_min = float(hr_min)
        self.hr_max = float(hr_max)
        self.hr_step = float(hr_step)
        self.residual_scale_bpm = float(residual_scale_bpm)

        bins = torch.arange(self.hr_min, self.hr_max + 0.5 * self.hr_step, self.hr_step, dtype=torch.float32)
        self.register_buffer("hr_bins", bins)

        self.frame_encoder = FrameEncoder2D(out_dim=frame_feature_dim, dropout=dropout * 0.5)
        self.video_temporal_in = nn.Sequential(
            nn.Conv1d(frame_feature_dim, video_temporal_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(video_temporal_channels),
            nn.ReLU(inplace=True),
        )
        video_blocks = []
        for i in range(num_video_blocks):
            video_blocks.append(ResidualBlock1D(video_temporal_channels, kernel_size=7, dilation=2 ** (i % 4), dropout=dropout))
        self.video_temporal_blocks = nn.Sequential(*video_blocks)

        self.trace_encoder = TemporalEncoder1D(
            in_channels=self.trace_channels,
            base_channels=trace_channels_hidden,
            num_blocks=num_trace_blocks,
            dropout=dropout,
        )

        pooled_dim = 2 * video_temporal_channels + 2 * trace_channels_hidden + 4
        self.dropout = nn.Dropout(dropout)
        self.shared = nn.Sequential(
            nn.Linear(pooled_dim, 192),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(192, 128),
            nn.ReLU(inplace=True),
        )
        self.hr_head = nn.Linear(128, bins.numel())
        # Auxiliary head using only video features. This prevents the video branch
        # from becoming a silent passenger when a strong classical prior is present.
        self.video_hr_head = nn.Sequential(
            nn.Linear(2 * video_temporal_channels, 96),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(96, bins.numel()),
        )
        self.selection_head = nn.Linear(128, self.num_candidates)
        self.fusion_gate_head = nn.Linear(128, 1)
        self.residual_head = nn.Linear(128, 1)
        self.quality_head = nn.Linear(128, 1)

        # Conservative initialization: prefer CHROM_WIN/CHROM/GREEN when they
        # are candidate names, and keep final HR mostly candidate-based at start.
        sel_bias = _initial_prior_logits(self.candidate_names, init_prior="auto")
        if len(sel_bias) == self.num_candidates:
            with torch.no_grad():
                self.selection_head.bias.copy_(sel_bias)
        nn.init.zeros_(self.selection_head.weight)

        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        nn.init.zeros_(self.fusion_gate_head.weight)
        nn.init.constant_(self.fusion_gate_head.bias, -4.0)  # sigmoid ~= 0.018: preserve candidate HR.

    def _encode_video(self, video: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = video.shape
        frames = video.reshape(b * t, c, h, w)
        frame_feat = self.frame_encoder(frames).reshape(b, t, -1)
        feat = frame_feat.transpose(1, 2).contiguous()
        feat = self.video_temporal_blocks(self.video_temporal_in(feat))
        avg = feat.mean(dim=-1)
        std = feat.std(dim=-1, unbiased=False)
        return torch.cat([avg, std], dim=1)

    def _encode_trace(self, x_trace: torch.Tensor) -> torch.Tensor:
        feat = self.trace_encoder(x_trace)
        avg = feat.mean(dim=-1)
        std = feat.std(dim=-1, unbiased=False)
        return torch.cat([avg, std], dim=1)

    def forward(
        self,
        video: torch.Tensor,
        x_trace: torch.Tensor,
        candidate_hrs: Optional[torch.Tensor] = None,
        candidate_valid: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        vpool = self._encode_video(video)
        tpool = self._encode_trace(x_trace)

        video_hr_logits = self.video_hr_head(self.dropout(vpool))
        video_hr_probs = torch.softmax(video_hr_logits, dim=-1)
        video_hr_bpm = (video_hr_probs * self.hr_bins.view(1, -1)).sum(dim=-1)

        # Simple candidate statistics give the fusion MLP scale/context without
        # letting it depend on hidden state alone.
        if candidate_hrs is not None:
            cand = candidate_hrs.to(video.device).float()
            valid = candidate_valid.bool().to(video.device) if candidate_valid is not None else torch.isfinite(cand)
            cand_safe = torch.where(valid, cand, torch.zeros_like(cand))
            count = valid.float().sum(dim=1).clamp_min(1.0)
            cand_mean = cand_safe.sum(dim=1) / count
            cand_min = torch.where(valid, cand, torch.full_like(cand, 999.0)).min(dim=1).values
            cand_max = torch.where(valid, cand, torch.full_like(cand, -999.0)).max(dim=1).values
            cand_spread = (cand_max - cand_min).clamp_min(0.0)
            cand_std = torch.sqrt((((cand_safe - cand_mean[:, None]) ** 2) * valid.float()).sum(dim=1) / count)
            cand_feats = torch.stack([
                (cand_mean - 90.0) / 30.0,
                cand_std / 20.0,
                cand_spread / 60.0,
                count / max(1.0, float(self.num_candidates)),
            ], dim=1)
        else:
            cand_feats = torch.zeros(video.shape[0], 4, device=video.device)

        z = self.shared(self.dropout(torch.cat([vpool, tpool, cand_feats], dim=1)))
        hr_logits = self.hr_head(z)
        hr_probs = torch.softmax(hr_logits, dim=-1)
        learned_hr = (hr_probs * self.hr_bins.view(1, -1)).sum(dim=-1)

        selection_logits = self.selection_head(z)
        if candidate_valid is not None:
            selection_logits = selection_logits.masked_fill(~candidate_valid.bool().to(video.device), -1e4)
        selection_probs = torch.softmax(selection_logits, dim=-1)

        if candidate_hrs is None:
            candidate_hr = learned_hr
        else:
            cand = candidate_hrs.to(video.device).float()
            if candidate_valid is not None:
                cand = torch.where(candidate_valid.bool().to(video.device), cand, learned_hr[:, None])
            candidate_hr = (selection_probs * cand).sum(dim=-1)

        residual = self.residual_scale_bpm * torch.tanh(self.residual_head(z).squeeze(-1))
        candidate_corrected = candidate_hr + residual
        gate = torch.sigmoid(self.fusion_gate_head(z).squeeze(-1))
        final_hr = (1.0 - gate) * candidate_corrected + gate * learned_hr

        quality_logit = self.quality_head(z).squeeze(-1)
        quality = torch.sigmoid(quality_logit)
        if return_dict:
            return {
                "hr_logits": hr_logits,
                "hr_probs": hr_probs,
                "learned_hr_bpm": learned_hr,
                "video_hr_logits": video_hr_logits,
                "video_hr_probs": video_hr_probs,
                "video_hr_bpm": video_hr_bpm,
                "selection_logits": selection_logits,
                "selection_probs": selection_probs,
                "candidate_hr_bpm": candidate_hr,
                "residual_bpm": residual,
                "fusion_gate": gate,
                "hr_bpm": final_hr,
                "quality_logit": quality_logit,
                "quality": quality,
            }
        return final_hr
