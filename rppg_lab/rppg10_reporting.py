"""HR-only external statistics and deterministic, headless rPPG-10 figures."""
from __future__ import annotations

from pathlib import Path
import hashlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch

from .extraction_cache import DEFAULT_WAVEFORM_ROIS
from .signals import estimate_hr_welch

MODEL = "Waveform v1"
CHROM = "CHROM_WIN multi-ROI average"
METRICS = ("mae_bpm", "rmse_bpm", "bias_bpm", "median_ae_bpm", "p90_ae_bpm",
           "within_3_bpm", "within_5_bpm", "finite_prediction_coverage", "prediction_failure_rate",
           "paired_coverage")


def spectral_diagnostics(waveform, fs=30.) -> dict:
    """Guard legacy HR estimator against silent NaN/constant imputation."""
    x = np.asarray(waveform, float)
    empty = dict(pred_hr_bpm=np.nan, snr_db=np.nan, spectral_concentration_heuristic=np.nan,
                 periodicity=np.nan, harmonic_power_ratio=np.nan, subharmonic_power_ratio=np.nan)
    if x.shape != (300,) or not np.isfinite(x).all() or np.std(x) <= 1e-8:
        return empty
    hr = estimate_hr_welch(x, fs)
    f, power = welch(x, fs=fs, nperseg=len(x), nfft=2048)
    p = lambda hz: float(power[np.argmin(abs(f - hz))])
    denom = max(p(hr.hr_hz), 1e-20)
    centered = x - x.mean()
    lag = int(round(fs / hr.hr_hz))
    periodicity = float(np.dot(centered[:-lag], centered[lag:]) /
                        max(np.linalg.norm(centered[:-lag]) * np.linalg.norm(centered[lag:]), 1e-20))
    return dict(pred_hr_bpm=hr.hr_bpm, snr_db=hr.snr_db,
                spectral_concentration_heuristic=hr.confidence, periodicity=periodicity,
                harmonic_power_ratio=p(2 * hr.hr_hz) / denom,
                subharmonic_power_ratio=p(hr.hr_hz / 2) / denom)


def metric_values(frame) -> dict:
    """All denominators are explicit. Accuracy is conditional on valid pairs."""
    total = len(frame)
    finite = np.isfinite(frame.pred_hr_bpm.to_numpy(float))
    label = frame.reference_valid.to_numpy(bool) & np.isfinite(frame.reference_hr_bpm.to_numpy(float))
    valid = finite & label
    errors = frame.pred_hr_bpm.to_numpy(float)[valid] - frame.reference_hr_bpm.to_numpy(float)[valid]
    ae = abs(errors)
    mean = lambda x: float(np.mean(x)) if len(x) else np.nan
    rate = lambda n: float(n / total) if total else np.nan
    return dict(windows=total, valid_pairs=int(valid.sum()), valid_labels=int(label.sum()),
                finite_predictions=int(finite.sum()), missing_labels=int((~label).sum()),
                prediction_failures=int((~finite).sum()),
                extraction_failures=int(frame.extraction_status.ne("ok").sum()),
                invalid_signal_windows=int((frame.extraction_status.eq("ok").to_numpy() & ~finite).sum()),
                mae_bpm=mean(ae), rmse_bpm=float(np.sqrt(mean(errors ** 2))), bias_bpm=mean(errors),
                median_ae_bpm=float(np.median(ae)) if len(ae) else np.nan,
                p90_ae_bpm=float(np.percentile(ae, 90)) if len(ae) else np.nan,
                within_3_bpm=mean(ae <= 3), within_5_bpm=mean(ae <= 5),
                finite_prediction_coverage=rate(finite.sum()), prediction_failure_rate=rate((~finite).sum()),
                paired_coverage=rate(valid.sum()), within_threshold_denominator=int(valid.sum()),
                mean_snr_db=mean(frame.snr_db.to_numpy(float)[np.isfinite(frame.snr_db.to_numpy(float))]))


def aggregate_metrics(rows, subject_ids, draws=2000, seed=42) -> tuple[pd.DataFrame, dict]:
    """Resample participants, retaining all correlated windows per participant.

    Subject-balanced RMSE/median/P90 mean the mean of subject statistics.
    Window-weighted metrics pool valid pairs. Subjects with no pairs remain
    in coverage denominators; conditional accuracy subject counts are exposed.
    """
    frame = pd.DataFrame(rows)
    subject_ids = sorted(subject_ids)
    if draws < 1 or not subject_ids:
        raise ValueError("Need subjects and positive bootstrap draws")
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(subject_ids), size=(draws, len(subject_ids)))
    per_subject, report = [], {}
    for method in sorted(frame.method.unique()):
        data = frame[frame.method == method]
        groups = [data[data.subject_id == sid] for sid in subject_ids]
        metrics = [metric_values(g) for g in groups]
        per_subject.extend(dict(subject_id=sid, method=method, **m) for sid, m in zip(subject_ids, metrics))
        values = np.array([[m[key] for key in METRICS] for m in metrics], float)
        balanced = {k: float(np.mean(values[np.isfinite(values[:, j]), j]))
                    if np.isfinite(values[:, j]).any() else np.nan for j, k in enumerate(METRICS)}
        pooled = metric_values(data)
        errors = []
        for g in groups:
            keep = g.reference_valid.to_numpy(bool) & np.isfinite(g.reference_hr_bpm) & np.isfinite(g.pred_hr_bpm)
            errors.append((g.loc[keep, "pred_hr_bpm"] - g.loc[keep, "reference_hr_bpm"]).to_numpy(float))
        # Cluster bootstrap for both weighting conventions, with identical draws.
        boot_bal = np.full((draws, len(METRICS)), np.nan)
        boot_win = np.full_like(boot_bal, np.nan)
        for b, sample in enumerate(samples):
            chosen = values[sample]
            counts = np.isfinite(chosen).sum(axis=0)
            boot_bal[b] = np.divide(np.nansum(chosen, axis=0), counts,
                                    out=np.full(len(METRICS), np.nan), where=counts > 0)
            e = np.concatenate([errors[i] for i in sample])
            ae = abs(e)
            n = sum(metrics[i]["windows"] for i in sample)
            finite = sum(metrics[i]["finite_predictions"] for i in sample)
            if len(e):
                boot_win[b, :7] = [ae.mean(), np.sqrt(np.mean(e**2)), e.mean(), np.median(ae),
                                   np.percentile(ae, 90), np.mean(ae <= 3), np.mean(ae <= 5)]
            if n:
                boot_win[b, 7:] = [finite / n, 1 - finite / n, len(e) / n]
        def intervals(a):
            return {k: dict(lower_95=float(np.nanpercentile(a[:, j], 2.5)),
                            upper_95=float(np.nanpercentile(a[:, j], 97.5)),
                            finite_draws=int(np.isfinite(a[:, j]).sum())) if np.isfinite(a[:, j]).any() else None
                    for j, k in enumerate(METRICS)}
        report[method] = dict(window_weighted=pooled, subject_balanced=balanced,
                              subjects=len(subject_ids), scorable_subjects=sum(m["valid_pairs"] > 0 for m in metrics),
                              unscorable_subject_ids=[s for s, m in zip(subject_ids, metrics) if not m["valid_pairs"]],
                              subject_balanced_ci95=intervals(boot_bal), window_weighted_ci95=intervals(boot_win))
    return pd.DataFrame(per_subject), dict(methods=report, bootstrap=dict(unit="subject", draws=draws, seed=seed,
        method="PCG64 participant percentile bootstrap; all windows retained in each sampled cluster"),
        definitions=dict(subject_balanced="mean of finite per-subject metrics; disclose unscorable subjects; coverage includes all subjects",
                         window_weighted="pooled valid window pairs; coverage uses all scheduled windows",
                         within_threshold="valid paired denominator only",
                         dependence="overlapping windows are autocorrelated; they are not independent participants"))


def matched_comparisons(frame) -> dict:
    model = frame[frame.method == MODEL]
    answer = {}
    for method in sorted(set(frame.method) - {MODEL}):
        base = frame[frame.method == method]
        joined = model.merge(base, on=["subject_id", "start_sec"], suffixes=("_model", "_baseline"), validate="one_to_one")
        keep = joined.reference_valid_model & joined.reference_valid_baseline & np.isfinite(joined.pred_hr_bpm_model) & np.isfinite(joined.pred_hr_bpm_baseline)
        j = joined[keep]
        em = abs(j.pred_hr_bpm_model - j.reference_hr_bpm_model)
        eb = abs(j.pred_hr_bpm_baseline - j.reference_hr_bpm_baseline)
        delta = pd.DataFrame(dict(subject_id=j.subject_id, delta=em - eb)).groupby("subject_id").delta.mean()
        answer[method] = dict(matched_windows=len(j), matched_subjects=len(delta),
                              model_mae_bpm=float(em.mean()), baseline_mae_bpm=float(eb.mean()),
                              subject_balanced_model_minus_baseline_mae=float(delta.mean()))
    return answer


def temporal_metrics(frame) -> list:
    result = []
    for (method, minute), data in frame.groupby(["method", "minute"]):
        metrics = [metric_values(g) for _, g in data.groupby("subject_id")]
        m = [r["mae_bpm"] for r in metrics if np.isfinite(r["mae_bpm"])]
        result.append(dict(method=method, minute=int(minute), **metric_values(data),
                           subject_balanced_mae_bpm=float(np.mean(m)) if m else np.nan,
                           scorable_subjects=len(m)))
    return result


def subgroup_metrics(subject_frame, metadata, seed=42, draws=2000) -> list:
    """Prespecified descriptive bins; never infer lighting trajectories from two readings."""
    descriptors = {}
    for sid, fields in metadata.items():
        d = dict(fields)
        try:
            endpoints = [float(v.strip()) for v in d["illumination"].split("-")]
            lux = sum(endpoints) / len(endpoints)
            d["illumination_group"] = "<800 lux" if lux < 800 else "800–<1100 lux" if lux < 1100 else "≥1100 lux"
        except (KeyError, ValueError):
            pass
        descriptors[sid] = d
    result = []
    for method in [MODEL, CHROM]:
        data = subject_frame[subject_frame.method == method]
        for field in ["skin_type", "illumination_group", "temperature_c", "humidity_percent"]:
            labels = sorted({d[field] for d in descriptors.values() if field in d})
            for label in labels:
                ids = [sid for sid, d in descriptors.items() if d.get(field) == label]
                selected = data[data.subject_id.isin(ids)]
                v = selected.mae_bpm.to_numpy(float)
                v = v[np.isfinite(v)]
                lo, hi = np.nan, np.nan
                if len(v) >= 3:
                    means = np.random.default_rng(seed).choice(v, size=(draws, len(v)), replace=True).mean(axis=1)
                    lo, hi = np.percentile(means, [2.5, 97.5])
                result.append(dict(method=method, field=field, group=label, subjects=len(selected),
                                   scorable_subjects=len(v), mean_subject_mae_bpm=float(v.mean()) if len(v) else np.nan,
                                   lower_95=lo, upper_95=hi, ci_supported=len(v) >= 3,
                                   interpretation="descriptive only; n<3 has no uncertainty interval; no fairness inference"))
    return result


def _save(fig, path):
    fig.savefig(path, dpi=130, metadata={"Software": "rppg10 external HR evaluation"})
    plt.close(fig)


def plot_ecg_qc(peaks, subject_id, output):
    """ECG-only detector audit, kept separate from optical diagnostic panels."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), constrained_layout=True)
    for ax, start in zip(axes, [10, 295, 580]):
        stop = min(start + 10, len(peaks.raw) / peaks.fs)
        if stop <= start:
            continue
        a, b = int(start * peaks.fs), int(stop * peaks.fs)
        t = np.arange(a, b) / peaks.fs
        ax.plot(t, peaks.raw[a:b], lw=.6, label="Stored ECG amplitude")
        markers = peaks.times[(peaks.times >= start) & (peaks.times < stop)]
        ax.plot(markers, peaks.raw[np.round(markers * peaks.fs).astype(int)], "rx", label="Detected electrical events")
        ax.set(xlabel="Nominal ECG time (s)", ylabel="ECG (stored units)")
        ax.legend(fontsize=8)
    fig.suptitle(f"{subject_id}: ECG-only R-peak QC; not optical waveform ground truth")
    _save(fig, Path(output) / f"{subject_id}_ecg_qc.png")


def plot_subject(frame, subject_id, windows, peaks, output, seed=42):
    output = Path(output)
    data = frame[frame.subject_id == subject_id]
    model = data[data.method == MODEL].sort_values("start_sec")
    chrom = data[data.method == CHROM].sort_values("start_sec")
    t = (model.start_sec.to_numpy() + 5) / 60
    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    ax.plot(t, model.reference_hr_bpm.where(model.reference_valid), label="ECG median-RR HR (manual sync ±0.5 s)", color="black")
    for g, label in [(model, MODEL), (chrom, CHROM)]:
        ax.plot((g.start_sec + 5) / 60, g.pred_hr_bpm, label=label, alpha=.8)
    ax.set(xlabel="Nominal recording time (min)", ylabel="Heart rate (bpm)", title=f"{subject_id}: HR, invalid gaps retained")
    ax.legend(fontsize=8)
    _save(fig, output / f"{subject_id}_hr.png")
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    for g, label in [(model, MODEL), (chrom, CHROM)]:
        error = abs(g.pred_hr_bpm - g.reference_hr_bpm).where(g.reference_valid)
        axes[0].plot((g.start_sec + 5) / 60, error, alpha=.4, label=label + " window")
        # Full rolling support required; a missing label remains a visible gap.
        axes[0].plot((g.start_sec + 5) / 60, error.rolling(5, min_periods=5).mean(), label=label + " 5-window mean")
    axes[0].set_ylabel("Absolute HR error (bpm)")
    axes[1].plot(t, model.snr_db, label="Waveform spectral SNR")
    axes[1].set_ylabel("SNR (dB; self peak)")
    axes[2].plot(t, model.valid_fraction, label="ROI source support")
    axes[2].step(t, model.reference_valid.astype(int), label="Valid ECG label", where="mid")
    axes[2].step(t, np.isfinite(model.pred_hr_bpm).astype(int), label="Finite prediction", where="mid", alpha=.6)
    axes[2].set(xlabel="Nominal recording time (min)", ylabel="Fraction / indicator", ylim=(-.05, 1.05))
    for ax in axes:
        ax.legend(fontsize=7)
    fig.suptitle(f"{subject_id}: HR error and descriptive signal support (uncalibrated)")
    _save(fig, output / f"{subject_id}_error_quality.png")
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True, constrained_layout=True)
    for r, roi in enumerate(DEFAULT_WAVEFORM_ROIS):
        axes[0].plot(t, model[f"roi_weight_{roi}"], label=roi)
        for prior in ("GREEN", "CHROM", "CHROM_WIN", "PBV", "POS_WIN", "OMIT"):
            axes[r+1].plot(t, model[f"prior_weight_{roi}_{prior}"], label=prior)
        axes[r+1].set_ylabel(roi + "\nprior weight")
    axes[0].set_ylabel("ROI attention")
    for ax in axes:
        ax.set_ylim(-.05, 1.05)
        ax.legend(fontsize=7, ncol=3)
    axes[-1].set_xlabel("Nominal recording time (min)")
    fig.suptitle(f"{subject_id}: descriptive allocations; not calibrated confidence or causal importance")
    _save(fig, output / f"{subject_id}_weights.png")
    usable = model[model.reference_valid & np.isfinite(model.pred_hr_bpm)].copy()
    usable["ae"] = abs(usable.pred_hr_bpm - usable.reference_hr_bpm)
    ranked = usable.sort_values(["ae", "start_sec"])
    if ranked.empty:
        return []
    rng_seed = seed + int.from_bytes(hashlib.sha256(subject_id.encode()).digest()[:4], "little")
    indices = dict(median=len(ranked)//2, worst=len(ranked)-1,
                   random=int(np.random.default_rng(rng_seed).integers(len(ranked))))
    selected = []
    for kind, idx in indices.items():
        row = ranked.iloc[idx]
        w = windows.get(float(row.start_sec))
        if w is None:
            continue
        fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True, constrained_layout=True,
                                  gridspec_kw={"height_ratios": [4, 1]})
        axes[0].plot(w["time"], w["model"], label="Frozen face waveform (signed model output)")
        axes[0].plot(w["time"], w["chrom"], label=CHROM, alpha=.8)
        axes[0].set_ylabel("Independent-window amplitude (a.u.)")
        axes[0].legend(fontsize=8)
        markers = peaks.times[(peaks.times >= row.start_sec) & (peaks.times < row.end_sec)]
        axes[1].eventplot(markers, lineoffsets=.5, linelengths=.7)
        axes[1].set(xlabel="Nominal time (s); manual synchronization ±0.5 s", yticks=[], ylabel="ECG events")
        axes[1].set_xlim(row.start_sec, row.end_sec)
        fig.suptitle(f"{subject_id} {kind}: selection by ECG HR error ({row.ae:.2f} bpm); electrical markers only\n"
                     "No ECG waveform-shape target or physiological transit-time interpretation")
        _save(fig, output / f"{subject_id}_{kind}_window.png")
        selected.append(dict(subject_id=subject_id, selection=kind, start_sec=float(row.start_sec),
                             model_hr_absolute_error_bpm=float(row.ae), rule="ECG-error ranked median/worst; subject-seeded random from valid pairs"))
    return selected


def plot_overview(frame, subjects, groups, output):
    output = Path(output)
    model = subjects[subjects.method == MODEL].sort_values("subject_id")
    chrom = subjects[subjects.method == CHROM].set_index("subject_id").reindex(model.subject_id)
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), constrained_layout=True)
    x = np.arange(len(model))
    axes[0].bar(x-.2, model.mae_bpm, width=.4, label=MODEL)
    axes[0].bar(x+.2, chrom.mae_bpm, width=.4, label=CHROM)
    axes[0].set_ylabel("Subject HR MAE (bpm)")
    axes[1].bar(x-.2, model.finite_prediction_coverage, width=.4, label="Finite model HR")
    axes[1].bar(x+.2, model.paired_coverage, width=.4, label="Valid model/ECG pairs")
    axes[1].set_ylabel("Scheduled-window fraction")
    for ax in axes:
        ax.set_xticks(x, model.subject_id, rotation=70, fontsize=8)
        ax.legend(fontsize=8)
    fig.suptitle("External HR evaluation: participants are the sampling unit; non-clinical / no BP validation")
    _save(fig, output / "subject_mae_coverage.png")
    m = frame[frame.method == MODEL]
    c = frame[frame.method == CHROM]
    joined = m.merge(c, on=["subject_id", "start_sec"], suffixes=("_m", "_c"))
    joined = joined[joined.reference_valid_m & np.isfinite(joined.pred_hr_bpm_m) & np.isfinite(joined.pred_hr_bpm_c)]
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    em = abs(joined.pred_hr_bpm_m - joined.reference_hr_bpm_m)
    ec = abs(joined.pred_hr_bpm_c - joined.reference_hr_bpm_m)
    ax.scatter(ec, em, s=7, alpha=.2)
    lim = max(float(ec.max()), float(em.max()), 1) if len(joined) else 1
    ax.plot([0, lim], [0, lim], "k--", lw=1)
    ax.set(xlabel="Matched CHROM_WIN absolute HR error (bpm)", ylabel="Matched frozen-model absolute HR error (bpm)",
           title=f"Same {len(joined)} windows; correlated within subjects\nNon-clinical HR comparison; no BP validity")
    _save(fig, output / "matched_hr_error_scatter.png")
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    valid = m.reference_valid & np.isfinite(m.pred_hr_bpm)
    y, ref = m.loc[valid, "pred_hr_bpm"], m.loc[valid, "reference_hr_bpm"]
    delta = y - ref
    ax.scatter((y+ref)/2, delta, alpha=.2, s=7)
    if len(delta):
        for level, label in [(delta.mean(), "Window bias"), (delta.mean()-1.96*delta.std(), "Descriptive −1.96 SD"),
                             (delta.mean()+1.96*delta.std(), "Descriptive +1.96 SD")]:
            ax.axhline(level, linestyle="--", label=f"{label}: {level:.2f}")
    ax.set(xlabel="Mean of frozen/ECG HR (bpm)", ylabel="Frozen − ECG HR (bpm)",
           title="Bland–Altman-style HR display: autocorrelated windows\nNon-clinical; no clinical agreement or BP claim")
    ax.legend(fontsize=8)
    _save(fig, output / "hr_agreement.png")
    for field in ["skin_type", "illumination_group"]:
        selected = [g for g in groups if g["field"] == field and g["method"] == MODEL]
        if not selected:
            continue
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        for j, group in enumerate(selected):
            ax.plot(j, group["mean_subject_mae_bpm"], "o", color="tab:blue")
            if group["ci_supported"]:
                ax.vlines(j, group["lower_95"], group["upper_95"], color="tab:blue")
        ax.set_xticks(range(len(selected)), [f"{g['group']}\nn={g['scorable_subjects']}/{g['subjects']}" for g in selected])
        ax.set(ylabel="Mean subject HR MAE (bpm)", xlabel=field,
               title="Descriptive groups; subject bootstrap 95% CI when n≥3\nSparse strata; no fairness inference")
        _save(fig, output / f"{field}_descriptive.png")
