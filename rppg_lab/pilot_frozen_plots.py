"""Participant-pixel-free diagnostics for independent frozen waveform windows."""
from __future__ import annotations

import numpy as np

from .waveform_visualization import normalized_welch, signed_zscore, write_csv


def render_recording(out, raw, timestamps, waveform, baseline, attention, priors, rows, extractor):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .region_masks import _polygon
    names = ("FrozenWaveformV1", "GREEN", "CHROM", "POS")
    own = [r for r in rows if r["method"] == names[0]]
    valid = [i for i, r in enumerate(own) if r["inference_valid"]]
    centers = np.asarray([(r["start_sec"] + r["end_sec"]) / 2 for r in own])
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for name in names:
        values = [r["hr_bpm"] for r in rows if r["method"] == name]
        axes[0].plot(centers, values, ".-", label=name, alpha=.8)
    axes[0].set(ylabel="Spectral HR estimate (bpm)", title="10 s windows / 5 s hop; no waveform reference")
    axes[0].legend(ncol=4, fontsize=8)
    for ri, name in enumerate(extractor.roi_names):
        axes[1].plot(centers, attention[:, ri], label=name)
    axes[1].set(ylabel="ROI attention", ylim=(-.05, 1.05))
    axes[1].legend(ncol=3, fontsize=8)
    labels = {"invalid":0, "uncertain":1, "good_engineering_only":2}
    for y, name in enumerate(names):
        m = [r for r in rows if r["method"] == name]
        for label, color in (("invalid", "#be4b48"), ("uncertain", "#e4b546"), ("good_engineering_only", "#3a9274")):
            ids = [i for i, r in enumerate(m) if r["quality_label"] == label]
            axes[2].scatter(centers[ids], np.full(len(ids), y), color=color, marker="s", s=22,
                            label=label if y == 0 else None)
    axes[2].set(yticks=range(4), yticklabels=names, xlabel="Original video time (s)",
                title="Engineering labels: SNR ≥ 0 dB and periodicity ≥ 0.5; not physiological confidence")
    axes[2].legend(loc="upper center", ncol=3, fontsize=8)
    fig.tight_layout(); fig.savefig(out / "overview.png", dpi=150); plt.close(fig)

    # Deterministic examples include both favorable and unfavorable windows.
    ordered = sorted(valid, key=lambda i: (own[i]["snr_db"], i))
    examples = sorted(set(([valid[0], ordered[0], ordered[len(ordered)//2], ordered[-1]] if valid else [])))
    spectral_rows = []
    for wi in examples:
        fig, axes = plt.subplots(2, 1, figsize=(12, 7))
        for mi, signal in enumerate((waveform[wi], *baseline[wi])):
            axes[0].plot(timestamps[wi], signed_zscore(signal), label=names[mi], alpha=.8)
            spectrum = normalized_welch(signal, extractor.fs_target)
            f, p = spectrum["frequency_hz"], spectrum["density"]
            axes[1].plot(f, p, label=names[mi])
            spectral_rows.extend(dict(window_index=wi, method=names[mi], frequency_hz=float(hz),
                                      normalized_density=float(power)) for hz, power in zip(f, p))
        axes[0].set(xlabel="Original video time (s)", ylabel="Signed z-amplitude (display only)",
                    title=f"Window {wi}: native polarity; no alignment, warping or reference")
        axes[0].legend(ncol=4, fontsize=8)
        axes[1].set(xlim=(0, 8), xlabel="Frequency (Hz)", ylabel="Normalized PSD (1/Hz)",
                    title="10 s spectral support; zero padding does not improve resolving power")
        fig.tight_layout(); fig.savefig(out / f"window_{wi:03d}.png", dpi=150); plt.close(fig)
    write_csv(out / "example_spectra.csv", spectral_rows)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ri, name in enumerate(extractor.roi_names):
        axes[ri].imshow(priors[:, ri].T, aspect="auto", origin="lower", vmin=0, vmax=1,
                        extent=(0, len(priors), -.5, len(extractor.prior_names)-.5))
        axes[ri].set(title=name, xlabel="Window index", yticks=range(len(extractor.prior_names)),
                     yticklabels=extractor.prior_names)
    fig.suptitle("Prior allocations: descriptive learned weights, not causal explanations")
    fig.tight_layout(); fig.savefig(out / "prior_allocations.png", dpi=150); plt.close(fig)

    selected = sorted(set([0, len(raw["camera_pts"])//2, len(raw["camera_pts"])-1]))
    h, w = raw["frame_shape"][:2]
    mask_examples = np.zeros((len(selected), len(extractor.roi_names), h, w), np.uint8)
    fig, axes = plt.subplots(1, len(selected), figsize=(9, 6), squeeze=False)
    colors = np.asarray([[.9,.4,.2],[.2,.7,.9],[.5,.9,.3]])
    for si, fi in enumerate(selected):
        canvas = np.zeros((h, w, 3))
        for ri, name in enumerate(extractor.roi_names):
            vertices = raw[name + "_vertices"][fi]
            if np.isfinite(vertices).all():
                _polygon(mask_examples[si, ri], vertices)
            canvas[mask_examples[si, ri].astype(bool)] = colors[ri]
        axes[0,si].imshow(canvas)
        axes[0,si].set(title=f"PTS {raw['camera_pts'][fi]:.3f} s", xlabel="Native x pixels", ylabel="Native y pixels")
    fig.suptitle("Exact contributing geometric masks; participant pixels omitted")
    fig.tight_layout(); fig.savefig(out / "roi_geometry.png", dpi=120); plt.close(fig)
    np.savez_compressed(out / "roi_masks_examples.npz", frame_indices=np.asarray(selected),
                        camera_pts=raw["camera_pts"][selected], masks=mask_examples,
                        roi_names=np.asarray(extractor.roi_names))
