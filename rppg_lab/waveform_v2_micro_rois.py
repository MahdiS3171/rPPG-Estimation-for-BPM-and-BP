"""Separate Candidate B: fixed-block patch mixing and whole-ROI abstention.

No label/prediction inputs exist. UNKNOWN independent skin evidence never
becomes confirmed skin through a favorable chroma score.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .waveform_v2_optical_quality import OpticalQualityConfig, finite_median


def partition_polygon(mask: np.ndarray) -> list[np.ndarray]:
    """2x3 bbox cells intersected with the unchanged polygon; disjoint exact union."""
    if mask.ndim != 2 or mask.dtype != bool:
        raise ValueError("Polygon must be a 2-D bool mask")
    cells = [np.zeros_like(mask) for _ in range(6)]
    yy, xx = np.nonzero(mask)
    if not len(xx):
        return cells
    yedges = np.linspace(yy.min(), yy.max()+1, 3).astype(int)
    xedges = np.linspace(xx.min(), xx.max()+1, 4).astype(int)
    for y in range(2):
        for x in range(3):
            cells[y*3+x][yedges[y]:yedges[y+1], xedges[x]:xedges[x+1]] = mask[yedges[y]:yedges[y+1], xedges[x]:xedges[x+1]]
    return cells


@dataclass(frozen=True)
class IndependentSkinEvidence:
    states: np.ndarray                   # [R,blocks]: unknown/verified_skin/verified_non_skin_or_occluded
    source_type: str
    evidence_id: str

    def validate(self, shape):
        if (self.states.shape != shape or not set(self.states.ravel()) <=
                {"unknown", "verified_skin", "verified_non_skin_or_occluded"}
                or self.source_type not in {"controlled_synthetic_ground_truth", "independent_human_review", "validated_segmentation"}
                or not self.evidence_id):
            raise ValueError("Independent skin evidence requires explicit states and provenance")


def frozen_patch_mix(patch_rgb: np.ndarray, valid: np.ndarray, weights: np.ndarray,
                     block_samples: int, clipping: np.ndarray | None = None,
                     max_clipped_fraction: float = .05):
    """No in-frame rescue/renormalization: one unavailable weighted patch abstains."""
    if (patch_rgb.ndim != 4 or patch_rgb.shape[-1] != 3 or valid.shape != patch_rgb.shape[:-1]
            or valid.dtype != bool or type(block_samples) is not int or block_samples < 1):
        raise ValueError("Expected patch RGB [R,P,T,3], bool validity and block length")
    r, p, n, _ = patch_rgb.shape
    blocks = (n+block_samples-1)//block_samples
    if (weights.shape != (r, blocks, p) or not np.isfinite(weights).all()
            or (weights < 0).any() or not np.all(np.isclose(weights.sum(-1), 1) | (weights.sum(-1) == 0))):
        raise ValueError("Weights must be finite, nonnegative, normalized or all-zero")
    if clipping is not None and clipping.shape != valid.shape:
        raise ValueError("Clipping shape mismatch")
    if not np.isfinite(max_clipped_fraction) or not 0 <= max_clipped_fraction <= 1:
        raise ValueError("Clipping limit must be a finite fraction")
    if clipping is not None:
        finite_clipping=clipping[np.isfinite(clipping)]
        if (finite_clipping<0).any() or (finite_clipping>1).any():
            raise ValueError("Clipping values must be fractions or missing")
    output = np.full((r, n, 3), np.nan)
    available = np.zeros((r, n), dtype=bool)
    applied = np.zeros((r, p, n), dtype=float)
    for b, start in enumerate(range(0, n, block_samples)):
        stop = min(start+block_samples, n)
        for roi in range(r):
            selected = weights[roi, b] > 0
            if not selected.any():
                continue
            good = valid[roi, selected, start:stop].all(0)
            values = patch_rgb[roi, selected, start:stop]
            good &= np.isfinite(values).all(axis=(0, 2))
            if clipping is not None:
                clipped = clipping[roi, selected, start:stop]
                good &= np.isfinite(clipped).all(0) & (clipped <= max_clipped_fraction).all(0)
            positions = np.flatnonzero(good) + start
            # Masked NaNs never enter arithmetic. No dynamic patch replacement.
            output[roi, positions] = (weights[roi, b, selected, None, None] * values[:, good]).sum(0)
            available[roi, positions] = True
            for patch in np.flatnonzero(selected):
                applied[roi, patch, positions] = weights[roi, b, patch]
    return output, available, applied


def evaluate_candidate(patch_rgb, patch_valid, patch_metrics, roi_metrics, *,
                       config=OpticalQualityConfig(), independent_evidence=None):
    if (patch_rgb.ndim != 4 or patch_rgb.shape[1] != 6 or patch_rgb.shape[-1] != 3
            or patch_valid.shape != patch_rgb.shape[:-1] or patch_valid.dtype != bool):
        raise ValueError("Require six patches per ROI with explicit acquisition validity")
    r, p, n, _ = patch_rgb.shape
    blocks = (n+config.block_samples-1)//config.block_samples
    for name in ("chroma_compatible_fraction", "clipped_fraction"):
        if patch_metrics[name].shape != (r,p,n) or roi_metrics[name].shape != (r,n):
            raise ValueError("Optical metric shape mismatch")
        finite = patch_metrics[name][np.isfinite(patch_metrics[name])]
        roi_finite=roi_metrics[name][np.isfinite(roi_metrics[name])]
        if (finite < 0).any() or (finite > 1).any() or (roi_finite<0).any() or (roi_finite>1).any():
            raise ValueError("Fraction descriptors must be in [0,1]")
    states = np.full((r, blocks), "unknown", dtype="U40")
    if independent_evidence is not None:
        independent_evidence.validate((r, blocks))
        states = independent_evidence.states
    weights = np.zeros((r, blocks, p))
    eligible = np.zeros_like(weights, dtype=bool)
    reasons = np.full((r,blocks), "ABSTAIN_SKIN_UNVERIFIED", dtype="U80")
    roi_proxy = np.full((r,blocks), np.nan)
    for b, start in enumerate(range(0, n, config.block_samples)):
        stop = min(start+config.block_samples,n)
        for roi in range(r):
            proxy = finite_median(roi_metrics['chroma_compatible_fraction'][roi,start:stop])
            roi_proxy[roi,b] = proxy
            if not np.isfinite(proxy):
                reasons[roi,b] = "ABSTAIN_NO_ACQUISITION"
                continue
            if proxy < config.min_chroma_fraction:
                reasons[roi,b] = "ABSTAIN_WHOLE_ROI_LOW_SKIN_PROXY"
                continue
            if states[roi,b] == 'verified_non_skin_or_occluded':
                reasons[roi,b] = "ABSTAIN_WHOLE_ROI_INDEPENDENT_OCCLUSION"
                continue
            support = patch_valid[roi,:,start:stop].mean(-1)
            scores = np.zeros(p)
            for patch in range(p):
                skin = finite_median(patch_metrics['chroma_compatible_fraction'][roi,patch,start:stop])
                clip = finite_median(patch_metrics['clipped_fraction'][roi,patch,start:stop])
                good = (support[patch] >= config.support_fraction and np.isfinite([skin,clip]).all()
                        and skin >= config.min_chroma_fraction and clip <= config.max_clipped_fraction)
                eligible[roi,b,patch] = good
                if good:
                    scores[patch] = np.clip(skin*(1-clip),config.score_floor,1.)
            if eligible[roi,b].sum() < config.min_eligible_patches:
                reasons[roi,b] = 'ABSTAIN_TOO_FEW_ELIGIBLE_PATCHES'
                continue
            weights[roi,b] = scores/scores.sum()
            reasons[roi,b] = 'ENGINEERING_WITH_INDEPENDENT_SKIN_EVIDENCE' if states[roi,b] == 'verified_skin' else 'ABSTAIN_SKIN_UNVERIFIED'
    provisional_rgb, provisional_valid, provisional_weights = frozen_patch_mix(
        patch_rgb,patch_valid,weights,config.block_samples,patch_metrics['clipped_fraction'],config.max_clipped_fraction)
    trusted_weights = weights.copy()
    trusted_weights[states != 'verified_skin'] = 0
    trusted_rgb, trusted_valid, applied_weights = frozen_patch_mix(
        patch_rgb,patch_valid,trusted_weights,config.block_samples,patch_metrics['clipped_fraction'],config.max_clipped_fraction)
    return dict(provisional_rgb=provisional_rgb,provisional_valid=provisional_valid,
                provisional_weights=provisional_weights,planned_weights=weights,eligible_patches=eligible,
                rgb=trusted_rgb,roi_valid=trusted_valid,applied_weights=applied_weights,
                whole_roi_reason=reasons,whole_roi_chroma_proxy=roi_proxy,independent_skin_state=states,
                anatomical_skin_confidence='UNVERIFIED' if independent_evidence is None else 'EXTERNAL_EVIDENCE_SUPPLIED',
                physiological_pulse_quality='UNVERIFIED',allow_training=False)


def weight_change_diagnostics(patch_rgb, applied_weights):
    """Report Δweight × patch intensity separately from source color changes."""
    if applied_weights.shape != patch_rgb.shape[:-1]:
        raise ValueError("Weight/RGB shape mismatch")
    steps = np.abs(np.diff(applied_weights,axis=-1)).sum(1)
    # A changing availability boundary with missing colors is unassessable, not zero.
    values = patch_rgb[:,:,1:]
    delta = np.diff(applied_weights,axis=-1)
    active = delta != 0
    finite = (~active | np.isfinite(values).all(-1)).all(1)
    safe_values = np.where(active[...,None],values,0)
    induced = np.sum(delta[...,None]*safe_values,axis=1)
    induced[~finite] = np.nan
    return dict(weight_l1_step=steps,weight_change_rgb_term=induced,
                interpretation='mixing_artifact_term_not_physiological_waveform')
