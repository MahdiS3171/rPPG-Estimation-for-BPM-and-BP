"""Unshifted, predetermined Gate 08 scientific diagnostic plots; local artifacts."""
from pathlib import Path

import numpy as np

from .signals import standardize_1d
from .waveform_v2_classical_study import METHODS, CONDITIONS, STARTS, FS


def make_plots(output, rows, aggregate, signals, references, controls):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    out = Path(output)
    out.mkdir(exist_ok=False)
    colors = ("#d08700", "#007f9c", "#a34399", "#255e35")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    train = [r for r in rows if r["split"] == "Train"]

    for name in ("P000001.mat", "P000003.mat"):
        fig, axes = plt.subplots(6, 2, figsize=(15, 16), constrained_layout=True, sharex=True)
        target = standardize_1d(references[name][0])
        t = np.arange(300)/FS
        for k, method in enumerate(METHODS):
            for column in range(2):
                ax = axes[k, column]
                ax.plot(t, target, color="black", lw=1.4, label="same-index reference")
                choices = range(3) if column == 0 else (3,)
                for r in choices:
                    ax.plot(t, signals[name][0, r, k], color=colors[r], lw=.9, alpha=.85, label=CONDITIONS[r])
                ax.set(title=f"{method}: {'fixed ROIs' if column == 0 else 'existing equal-ROI fusion'}",
                       ylabel="Window-standardized amplitude")
                ax.grid(alpha=.15)
            if k == 0:
                axes[k, 0].legend(fontsize=8, ncol=2)
                axes[k, 1].legend(fontsize=8, ncol=2)
        for ax in axes[-1]:
            ax.set_xlabel("Original nominal synthetic time (s); no lag/sign correction")
        fig.suptitle(f"{name}: predeclared first Train window [0,300)\n"
                     "Native prior polarity; raw standardized d_ppg; synthetic exploratory evidence", fontsize=14)
        fig.savefig(out/f"waveforms_{Path(name).stem}.png", dpi=130)
        plt.close(fig)

    legend = [Line2D([], [], color=colors[r], marker="o", linestyle="", label=roi) for r, roi in enumerate(CONDITIONS)]
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
    for ax, method in zip(axes.flat, METHODS):
        for r, roi in enumerate(CONDITIONS):
            group = [v for v in train if v["method"] == method and v["roi"] == roi]
            ax.scatter([v["metrics"]["reference_hr_bpm"] for v in group],
                       [v["metrics"]["pred_hr_bpm"] for v in group], color=colors[r], s=28, alpha=.7)
        ax.plot([42, 210], [42, 210], color="gray", ls="--", lw=.8)
        ax.set(title=method, xlabel="Same-index reference Welch HR (bpm)", ylabel="Prior Welch HR (bpm)",
               xlim=(42, 210), ylim=(42, 210))
        ax.grid(alpha=.15)
    fig.legend(handles=legend, loc="outside upper center", ncol=4)
    fig.suptitle("All 16 Train windows per condition; fixed methods/ROIs; no quality selection", y=1.06)
    fig.savefig(out/"train_hr_pairs.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
    for ax, method in zip(axes.flat, METHODS):
        for r, roi in enumerate(CONDITIONS):
            group = [v for v in train if v["method"] == method and v["roi"] == roi]
            ax.scatter([v["quality"]["spectral_concentration"] for v in group],
                       [v["metrics"]["hr_abs_error_bpm"] for v in group], color=colors[r], alpha=.7, s=28)
        ax.set(title=method, xlabel="Own-peak spectral concentration (label-free)", ylabel="Absolute HR error (bpm)")
        ax.grid(alpha=.15)
    fig.legend(handles=legend, loc="outside upper center", ncol=4)
    fig.suptitle("Train descriptors versus fixed errors; no fitted quality threshold", y=1.06)
    fig.savefig(out/"train_quality_errors.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    matrix = np.array([[next(v for v in aggregate if v["split"] == "Train" and v["method"] == m and v["roi"] == r)
                        ["recording_balanced"]["zero_lag_r"] for m in METHODS] for r in CONDITIONS])
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    im = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
    ax.set(xticks=np.arange(6), xticklabels=METHODS, yticks=np.arange(4), yticklabels=CONDITIONS,
           title="Train recording-balanced native zero-lag r\nNo sign flip, absolute value or lag correction")
    for r in range(4):
        for k in range(6):
            ax.text(k, r, f"{matrix[r,k]:.3f}", ha="center", va="center")
    fig.colorbar(im, ax=ax, label="Signed Pearson r")
    fig.savefig(out/"train_native_correlations.png", dpi=130)
    plt.close(fig)

    names = list(dict.fromkeys(r["filename"] for r in controls))
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), constrained_layout=True)
    for k, method in enumerate(METHODS):
        concentration, coverage = [], []
        for name in names:
            group = [r for r in controls if r["filename"] == name and r["roi"] == "equal_roi_average" and r["method"] == method]
            values = [r["quality"]["spectral_concentration"] for r in group]
            good = np.asarray(values)[np.isfinite(values)]
            concentration.append(float(good.mean()) if len(good) else np.nan)
            coverage.append(sum(r["pulse_estimate_eligible"] for r in group)/2)
        axes[0].plot(np.arange(8), concentration, marker="o", label=method)
        axes[1].plot(np.arange(8), coverage, marker="o", label=method)
    axes[0].set(ylabel="Own-peak concentration", title="All fixed corruption controls: equal-ROI fusion\n"
                "Coherent illumination can look periodic; no physiological-origin classifier")
    axes[1].set(ylabel="Eligible estimates / 2 windows", ylim=(-.05, 1.05),
                xticks=np.arange(8), xticklabels=[n.replace("_", "\n") for n in names])
    axes[0].set_xticks([])
    axes[0].legend(ncol=6)
    for ax in axes:
        ax.grid(alpha=.15)
    fig.savefig(out/"negative_controls.png", dpi=130)
    plt.close(fig)
