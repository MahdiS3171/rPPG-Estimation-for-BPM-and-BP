"""Private aggregate artifacts and a self-contained, failure-inclusive report."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .artifacts import write_json
from .pilot_analysis import finite_stat
from .pilot6_delay_feasibility import primary_rows, union_seconds, finite
from .waveform_visualization import write_csv


def table(headers, rows):
    def fmt(x):
        if x is None or isinstance(x, (float, np.floating)) and not np.isfinite(x):
            return "unavailable"
        return f"{x:.3f}" if isinstance(x, (float, np.floating)) else str(x).replace("|", "/")
    return ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(fmt(v) for v in row) + " |" for row in rows)]


def quality_summary(quality):
    groups = defaultdict(list)
    for r in quality:
        groups[r["session_id"], r["policy"], r["roi"], r["method"], r["cuff_phase"]].append(r)
    rows = []
    for key, group in groups.items():
        for selection in ("all_computable", "good", "uncertain", "invalid"):
            subset = [r for r in group if r["status"] == selection or selection == "all_computable" and r["status"] != "invalid"]
            rows.append(dict(zip(("session_id", "policy", "roi", "method", "cuff_phase"), key),
                             selection=selection, scheduled=len(group), selected=len(subset), coverage_sec=union_seconds(subset),
                             median_hr_bpm=finite_stat([r["hr_bpm"] for r in subset]),
                             median_snr_db=finite_stat([r["snr_db"] for r in subset]),
                             median_periodicity=finite_stat([r["periodicity"] for r in subset]),
                             reliable_beat_windows=sum(r["beat_detection_reliable"] for r in subset)))
    return rows


def figures(out, sessions, quality, delays, simulation, sensitivity):
    ids = [s["session_id"] for s in sessions]
    colors = {"strict": "tab:green", "intermediate": "tab:orange", "uncertain": "tab:red", "invalid": "gray", "excluded": "black"}
    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    for i, s in enumerate(sessions):
        ax.plot([s["pts_start_sec"], s["pts_end_sec"]], [i, i], color="tab:blue", linewidth=9, alpha=.4)
        for e in s["events"]:
            if e["start_sec"] is not None and e["end_sec"] is not None:
                a, b = e["start_sec"], e["end_sec"]
                ax.broken_barh([(a-5, b-a+10)], (i-.24, .48), facecolors="orange", alpha=.3)
                ax.broken_barh([(a, b-a)], (i-.16, .32), facecolors="tab:red", alpha=.7)
                ax.text((a+b)/2, i+.2, f"BP{e['measurement_index']}", ha="center", fontsize=7)
    ax.set(yticks=range(len(ids)), yticklabels=ids, xlabel="Original video PTS (s)", title="Recorded cuff bounds (red), 5s guards (orange); all legacy timing unverified")
    fig.savefig(out / "cohort_timeline.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(len(ids), 1, figsize=(13, max(4, 2.5*len(ids))), sharex=True, constrained_layout=True)
    for ax, sid in zip(np.atleast_1d(axes), ids):
        group = [r for r in primary_rows(delays) if r["session_id"] == sid]
        for status, color in colors.items():
            computed = [r for r in group if r["status"] == status and r["computable"]]
            ax.scatter([r["start_sec"]+5 for r in computed], [1000*r["delay_xcorr_sec"] for r in computed], color=color, label=status, s=12)
        failed = [r for r in group if not r["computable"]]
        ax.scatter([r["start_sec"]+5 for r in failed], [.03]*len(failed), transform=ax.get_xaxis_transform(), marker="x", s=14, color="gray", label="failed (rug, no delay)")
        for e in next(s for s in sessions if s["session_id"] == sid)["events"]:
            if e["start_sec"] is not None and e["end_sec"] is not None:
                ax.axvspan(e["start_sec"], e["end_sec"], color="tab:red", alpha=.08)
        ax.set(title=sid, ylabel="Hand lag (ms)", ylim=(-330, 330))
        ax.axhline(0, color="black", lw=.5)
    if ids:
        np.atleast_1d(axes)[0].legend(fontsize=7, ncol=5)
        np.atleast_1d(axes)[-1].set_xlabel("Original PTS (s); policy B, combined face / central palm GREEN")
    fig.savefig(out / "delay_vs_time.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    q = [r for r in quality if r["policy"] == "B" and r["roi"] in ("combined", "hand_palm") and r["method"] == "GREEN"]
    for i, roi in enumerate(("combined", "hand_palm")):
        for j, phase in enumerate(("candidate_cuff_free", "cuff_active")):
            g = [r for r in q if r["roi"] == roi and r["cuff_phase"] == phase]
            values = [r["snr_db"] for r in g if finite(r["snr_db"])]
            if values:
                axes[0].boxplot(values, positions=[2*i+j], widths=.6)
            axes[1].bar(2*i+j, sum(r["status"] == "good" for r in g), color=("tab:blue", "tab:red")[j])
            axes[1].text(2*i+j, sum(r["status"] == "good" for r in g)+.5, f"/{len(g)}", ha="center", fontsize=8)
    labels = ["face free*", "face cuff*", "hand free*", "hand cuff*"]
    for ax in axes:
        ax.set(xticks=range(4), xticklabels=labels)
    axes[0].set_ylabel("Spectral SNR (dB)")
    axes[1].set_ylabel("Engineering-good windows / attempted")
    fig.suptitle("*Phase from unverified recorded bounds; overlapping windows are dependent")
    fig.savefig(out / "quality_by_cuff_phase.png", dpi=140)
    plt.close(fig)
    primary = [r for r in primary_rows(delays) if r["computable"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for sid in ids:
        g = [r for r in primary if r["session_id"] == sid]
        axes[0].scatter([1000*r["delay_xcorr_sec"] for r in g], [1000*r["delay_phase_sec"] for r in g], s=8, label=sid)
        axes[1].scatter([1000*r["delay_xcorr_sec"] for r in g], [1000*r["uncertainty_floor_sec"] for r in g], s=8, label=sid)
    axes[0].plot([-300, 300], [-300, 300], color="gray", ls="--")
    axes[0].set(xlabel="Signed correlation delay (ms)", ylabel="Fundamental phase delay (ms)")
    axes[1].set(xlabel="Signed correlation delay (ms)", ylabel="Engineering uncertainty floor (ms)", title="Floor includes native steps and disagreements; no clinical CI")
    if primary:
        axes[0].legend(fontsize=7)
    fig.savefig(out / "delay_uncertainty.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    lookup = {(r["session_id"], r["window_index"], r["method"]): r for r in delays if r["policy"] == "B" and r["face_roi"] == "combined" and r["hand_roi"] == "hand_palm" and r["computable"]}
    for ax, method in zip(axes, ("CHROM", "POS")):
        pair = [(r, lookup[r["session_id"], r["window_index"], method]) for r in primary if (r["session_id"], r["window_index"], method) in lookup]
        ax.scatter([1000*a["delay_xcorr_sec"] for a, b in pair], [1000*b["delay_xcorr_sec"] for a, b in pair], s=9)
        ax.plot([-300, 300], [-300, 300], "--", color="gray")
        ax.set(xlabel="GREEN delay (ms)", ylabel=f"{method} delay (ms)", title="All computable policy-B windows")
    fig.savefig(out / "delay_method_agreement.png", dpi=140)
    plt.close(fig)
    candidate = [r for r in sensitivity if r["policy"] == "B" and r["selection"] == "all_computable" and r["max_distance_sec"] == 30 and r["association_available"]]
    for target in ("sbp", "dbp"):
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        for sid in ids:
            g = [r for r in candidate if r["session_id"] == sid]
            ax.scatter([r["median_delay_ms"] for r in g], [r[target+"_mmHg"] for r in g], label=sid)
        ax.set(xlabel="Pre-cuff optical delay (ms)", ylabel=f"Cuff {target.upper()} (mmHg)",
               title="Unverified legacy timing, all-computable sensitivity only")
        if candidate:
            ax.legend(fontsize=7)
        else:
            ax.text(.5, .5, "No eligible associations", ha="center", transform=ax.transAxes)
        fig.savefig(out / ("delay_vs_"+target+".png"), dpi=140)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for method in ("GREEN", "CHROM", "POS"):
        for ax, scenario in zip(axes, ("clean", "morphology_mismatch")):
            group = [r for r in simulation if r["scenario"] == scenario and r["method"] == method and r["filtered"] and finite(r.get("error_ms"))]
            ax.scatter([r["true_delay_ms"] for r in group], [r["error_ms"] for r in group], s=5, alpha=.35, label=method)
            ax.set(title=scenario, xlabel="Known hand lag (ms)", ylabel="Signed recovery error (ms)")
    axes[0].legend(fontsize=7)
    fig.savefig(out / "synthetic_delay_error.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(max(1, len(ids)), 3, figsize=(14, max(3, len(ids)*2.1)), squeeze=False, constrained_layout=True)
    for row_index, sid in enumerate(ids):
        cache = out / sid / "classical_waveforms.npz"
        if not cache.exists():
            continue
        with np.load(cache, allow_pickle=False) as z:
            t = z["uniform_pts"]
            waves = [z["B_combined_GREEN"], z["B_hand_palm_GREEN"]]
        group = [r for r in primary_rows(delays) if r["session_id"] == sid]
        computable = [r for r in group if r["computable"]]
        failed = [r for r in group if not r["computable"]]
        choices = [computable[0] if computable else None,
                   min(computable, key=lambda r: min(r["face_snr_db"], r["hand_snr_db"])) if computable else None,
                   failed[0] if failed else None]
        for ax, choice, label in zip(axes[row_index], choices, ("first computable", "lowest paired SNR", "first failed")):
            if choice:
                keep = (t >= choice["start_sec"]) & (t < choice["end_sec"])
                for w, site in zip(waves, ("face", "hand")):
                    v = w[keep]
                    valid = np.isfinite(v)
                    scale = v[valid].std() if valid.any() else 0.
                    plot = (v-np.nanmean(v))/scale if scale > 1e-10 else np.full(len(v), np.nan)
                    ax.plot(t[keep], plot, lw=.7, label=site)
                ax.set_title(f"{sid} {label} {choice['start_sec']:.1f}s ({choice['status']})", fontsize=8)
            else:
                ax.text(.1, .5, f"{sid}: {label} unavailable", transform=ax.transAxes)
            ax.set_xlabel("Original PTS (s)", fontsize=7)
    if ids:
        axes[0, 0].legend(fontsize=7)
    fig.savefig(out / "face_hand_examples.png", dpi=140)
    plt.close(fig)


def render_report(out, sessions, config, quality, delays, acquisition, camera, simulations, simulation_summary,
                  stability, associations, sensitivity, bp, failures):
    from .pilot6_clock_controls import isolated_clock_controls
    from .pilot6_delay_validation import summarize_simulations
    extra_controls = isolated_clock_controls(config)
    write_csv(out / "synthetic_isolated_clock_controls.csv", extra_controls)
    write_csv(out / "synthetic_isolated_clock_summary.csv", summarize_simulations(extra_controls))
    all_simulations = simulations + extra_controls
    estimator_groups = defaultdict(list)
    for r in all_simulations:
        for estimator, error in (("signed_xcorr", r.get("error_ms")), ("cross_spectral_phase", r.get("phase_error_ms")),
                                 ("reliable_matched_peaks", 1000*r["delay_beat_median_sec"]-r["true_delay_ms"] if finite(r.get("delay_beat_median_sec")) else None)):
            estimator_groups[r["fs"], r["method"], r["scenario"], r["filtered"], estimator].append(error)
    estimator_summary = []
    for key, values in estimator_groups.items():
        e = np.asarray(values, dtype=float)
        e = e[np.isfinite(e)]
        estimator_summary.append(dict(zip(("fs", "method", "scenario", "filtered", "estimator"), key), attempted=len(values), computable=len(e),
                                      signed_bias_ms=float(e.mean()) if len(e) else None,
                                      mae_ms=float(np.abs(e).mean()) if len(e) else None,
                                      variance_ms2=float(e.var()) if len(e) else None,
                                      max_absolute_error_ms=float(np.abs(e).max()) if len(e) else None))
    write_csv(out / "synthetic_estimator_summary.csv", estimator_summary)
    audit = json.loads((out / "metadata_audit.json").read_text(encoding="utf-8"))
    qsummary = quality_summary(quality)
    write_csv(out / "waveform_quality_summary.csv", qsummary)
    raw_synthetic = {tuple(r[k] for k in ("clock", "hr_bpm", "true_delay_ms", "scenario", "replicate", "method")): r
                     for r in simulations if not r["filtered"]}
    filter_changes = []
    for r in simulations:
        key = tuple(r[k] for k in ("clock", "hr_bpm", "true_delay_ms", "scenario", "replicate", "method"))
        if r["filtered"] and finite(r.get("error_ms")) and finite(raw_synthetic.get(key, {}).get("error_ms")):
            filter_changes.append(dict(zip(("clock", "hr_bpm", "true_delay_ms", "scenario", "replicate", "method"), key),
                                       filter_change_ms=r["error_ms"]-raw_synthetic[key]["error_ms"]))
    write_csv(out / "synthetic_filter_sensitivity.csv", filter_changes)
    figures(out, sessions, quality, delays, simulations, sensitivity)
    primary = primary_rows(delays)
    rejection = Counter(reason for r in primary for reason in r["reasons"].split("|") if reason)
    write_csv(out / "rejection_reasons.csv", [dict(reason=k, windows=v, denominator=len(primary)) for k, v in rejection.most_common()])
    computable_rejection = Counter(reason for r in primary if r["computable"] for reason in r["reasons"].split("|") if reason)
    write_csv(out / "computable_rejection_reasons.csv", [dict(reason=k, windows=v, denominator=sum(r["computable"] for r in primary)) for k, v in computable_rejection.most_common()])
    cohort = []
    for s in sessions:
        g = [r for r in primary if r["session_id"] == s["session_id"]]
        computable = [r for r in g if r["computable"]]
        strict = [r for r in g if r["status"] == "strict"]
        free = [r for r in computable if r["cuff_phase"] == "candidate_cuff_free"]
        cohort.append(dict(session_id=s["session_id"], scheduled=len(g), computable=len(computable), strict=len(strict),
                           computable_union_sec=union_seconds(computable), strict_union_sec=union_seconds(strict),
                           candidate_free_computable_union_sec=union_seconds(free),
                           verified_free_strict_union_sec=union_seconds([r for r in strict if r["timing_verified"]]),
                           strict_nonoverlapping_windows=sum(r["primary_nonoverlapping"] for r in strict),
                           strict_median_delay_ms=1000*finite_stat([r["delay_xcorr_sec"] for r in strict]) if strict else None,
                           strict_median_uncertainty_floor_ms=1000*finite_stat([r["uncertainty_floor_sec"] for r in strict]) if strict else None,
                           median_delay_ms=1000*finite_stat([r["delay_xcorr_sec"] for r in computable]) if computable else None,
                           median_uncertainty_floor_ms=1000*finite_stat([r["uncertainty_floor_sec"] for r in computable]) if computable else None,
                           median_split_difference_ms=1000*finite_stat([r["split_difference_sec"] for r in computable]) if computable else None))
    write_csv(out / "cohort_summary.csv", cohort)
    short_synthetic = []
    groups = defaultdict(list)
    for r in all_simulations:
        if r["filtered"]:
            groups[r["fs"], r["method"], r["scenario"]].append(r)
    for key, g in groups.items():
        errors = np.asarray([r.get("error_ms", np.nan) for r in g], dtype=float)
        valid = np.isfinite(errors)
        e = errors[valid]
        short_synthetic.append(dict(fs=key[0], method=key[1], scenario=key[2], n=len(g), computable=int(valid.sum()),
                                    bias_ms=float(e.mean()) if len(e) else None, mae_ms=float(np.abs(e).mean()) if len(e) else None,
                                    max_error_ms=float(np.max(np.abs(e))) if len(e) else None))
    available = sum(r["association_available"] for r in associations)
    recommendation = ("Proceed with a targeted instrumentation and protocol-reliability pilot before a larger BP prediction study. "
        "Video-review every cuff event, measure camera exposure/rolling shutter or use a validated synchronized acquisition system, "
        "record independent contact waveform timing references, fix polarity/landmark conventions, and repeat recordings per subject. "
        "These six recordings cannot establish clinical BP accuracy, physiological PTT, or a deployable BP model.")
    summary = dict(processed_sessions=len(camera), expected_sessions=6, independent_cuff_events=len(associations),
                   verified_strict_associations=available, cohort=cohort, rejections=dict(rejection),
                   synthetic=short_synthetic, failures=failures, recommendation=recommendation,
                   definitions=dict(verified_measurement="Local video/container timestamps, pixel RGB/coverage, recorded source values and file identities; no external physiological validation",
                                    engineering_diagnostic="Spectral HR/SNR, signed correlation/phase delays, ambiguity, repeatability, interpolation and simulation error",
                                    unverified_physiological_interpretation="A possible relation to pulse propagation; not established by these optical signals",
                                    exploratory_statistical_finding="Cuff-event candidate associations nested within six subjects; unverified legacy timing sensitivity",
                                    unsupported_claim="Validated PTT/PWV, clinical BP accuracy, AAMI/ISO compliance, or physiological validity from clean HR alone"))
    write_json(out / "results_summary.json", summary)
    lines = ["# Pilot-6 protocol-aware Face–Hand Optical Delay feasibility", "",
             "This is an exploratory six-subject feasibility experiment. The measured quantity is **Face–Hand Optical Delay**, "
             "positive when the hand lags the face. There is no contact PPG/ECG timing reference, calibrated exposure clock, or continuous BP label.", "",
             "## A. Metadata identity and changes", "",
             f"Source: `{audit['path']}`. SHA256: `{audit['sha256']}`.",
             f"Declared spreadsheet: `rPPG BP Data Collection final`, ID `{audit['declared_google_sheet_id']}`. "
             "The user-declared export lineage and exact local bytes are recorded; no live Google identity or visual event verification is claimed.",
             f"Previous snapshot: `{audit['previous_path']}`, SHA256 `{audit['previous_sha256']}`. "
             f"Structured raw/header/formula changes: {audit['change_count']}; see `metadata_changes.csv`.",
             "The update fills 15 formerly missing cuff ends (five sessions), retains S004's longer recorded cuff periods, "
             "and replaces `Video_Duration_s` / `Actual_FPS` with `Video_Duration_Reported_s` / `FPS_Reported`. "
             "Reported FPS/resolution, hand support, ROI visibility and cuff-at-heart-level cells are newly populated; those are source assertions, not video verification. "
             "SBP/DBP values are unchanged. All cells, cached results and formulas are retained in ignored `private_metadata/`; derived Review/Windows tables supply no independent evidence.",
             audit.get("geometry_provenance_note", ""), ""]
    lines += table(["Sheet", "Rows", "Columns", "Header row"], [(n, t["row_count"], t["column_count"], t["header_row_1based"]) for n, t in audit["worksheets"].items()])
    lines += ["", "Exact audited headers are in `metadata_audit.json`; session matching uses explicit IDs/filenames and fails on conflicts. No row-order matching.",
              "", "## B–C. Six-recording timeline reconstruction", "",
              "Cuff times are recorded bounds, all with legacy unverified provenance. Policy A uses 0s guards, B 5s, C 10s. "
              "Candidate intervals are recomputed from source session fields; **verified cuff-free time is zero**. "
              "Guard width is sensitivity analysis, not a measured bound on legacy timestamp error. Unknown endpoints remain uncertain and block confident inclusion.", ""]
    lines += table(["Session", "BP1 s", "BP2 s", "BP3 s", "Provenance"], [(s["session_id"], *(f"{e['start_sec']}–{e['end_sec']}" for e in s["events"]), s["events"][0]["event_time_source"]) for s in sessions])
    lines += ["", "![Cohort timeline](cohort_timeline.png)", "",
              "Original PTS are retained in `protocol_masks.npz` and `native_rgb.npz` per session. `session_timeline.csv`, `cuff_events.csv`, "
              "and `eligible_intervals.csv` provide all policy masks, recorded event labels and candidate intervals. No discontinuous segments are concatenated.",
              "", "## D. Native timestamp and acquisition audit", ""]
    lines += table(["Session", "Frames", "Seconds", "Median step ms", "Maximum step ms", "Long steps", "Repeated frames"],
                   [(r["session_id"], r["source_frames"], r["source_seconds"], r["median_interval_ms"], r["max_interval_ms"], r["intervals_over_1_5_median"], r["repeated_decoded_pixel_frames"]) for r in camera])
    lines += ["", "Container display timestamps include original simple edit-list leads; decoder comparison removes only a constant origin. "
              "Long intervals are engineering evidence of clock irregularity, not a recoverable count of capture drops. Repeated exact decoded pixels are separate. "
              "Exposure integration, ROI vertical readout, rolling shutter, compression and hardware latency remain unmeasured.",
              "", "## E. Face and hand ROI validity", "",
              "Every hand frame is decoded/measured at native resolution. Historical native face RGB is reused only after checking original extraction identity, "
              "video SHA256, source, assets and dependencies. Combined face is remeasured as the exact pixel union of the three native facial polygons. "
              "Palm and contracted central-palm hulls reuse existing anatomy. Cropped measurement bounds preserve every contributing native pixel. "
              "Detector identity is locked; the protocol states right hand, but the mirroring convention remains unverified. "
              "Face tracking confidence is unavailable; hand classification scores are not calibrated tracking confidence. "
              "Pixel support and geometric coverage do not prove pulse information.", ""]
    lines += table(["Session", "ROI", "Source valid fraction", "Median pixels", "Clipping"],
                   [(r["session_id"], r["roi"], r["source_validity_fraction"], r["median_pixels"], r["median_clipping_fraction"]) for r in acquisition])
    lines += ["", "## F–G. Independent waveform quality and cuff-phase comparisons", "",
              "GREEN, CHROM and single-window POS retain the existing formulas. Each cuff phase and paired valid run is processed separately on a common native-rate "
              "grid with bounded linear interpolation (maximum 50.001ms), 0.7–4Hz order-3 zero-phase SOS filtering, and 1s segment-edge exclusion. "
              "10s windows advance 5s; primary stability/cuff features are anchored one edge margin inside each contiguous protocol-phase/camera-support segment, "
              "advancing 10s without overlap. Supplemental global/overlapping attempts, including boundary failures, remain in the traces. "
              "Phase anchors were added during engineering verification after a global-grid pass missed short intervals due to edge alignment; "
              "this refinement uses protocol/camera support only, with all quality/timing thresholds unchanged and no BP selection. "
              "All computable, good, intermediate, uncertain and failed rows remain available. No quality rule consumes BP labels. "
              "Full per-method/ROI/phase counts and union coverage are in `waveform_quality_summary.csv`; acquisition and waveform quality remain separate. "
              "Good requires original validity ≥0.9, finite nonflat waveform, SNR ≥0dB, periodicity ≥0.5, clipping ≤0.1 and illumination CV ≤0.05. "
              "Clean HR or SNR does not establish correct timing.", "", "![Quality by cuff phase](quality_by_cuff_phase.png)", "",
              "The optional Frozen V1 branch uses only the original face contract and accepted V2 adapter, and is never a hand or delay predictor. "
              "The original checkpoint and manifest are identity checked in every run; this run's explicit optional-inference status is recorded in the manifest. "
              "No weights, architecture, freeze manifest, training splits or historical results are changed.",
              "", "## H. Known-delay synthetic timing validation", "",
              "The simulation includes 0 and ±20/40/60/80/120ms, 54/78/108bpm, 30/60Hz, all six real irregular acquisition clocks, "
              "unequal amplitudes, noise/drift, pulse morphology mismatch, ±1.5ms timestamp jitter, missing samples, projection and filtering. "
              "Primary estimators use spectral HR measured from each extracted synthetic face waveform, exactly as in the real analysis; true HR is a design parameter only. "
              "The fixed 10s analysis window crosses a deliberately unbridgeable 150ms loss in the gap scenario; failure is the correct outcome. "
              "Both filtered and unfiltered results, signed bias, error variance, failure rates and phase error are in `synthetic_timing_validation.csv` and "
              "`synthetic_timing_summary.csv`. Simulations are engineering controls without any physiological BP labels.", ""]
    lines += ["Additional isolated jitter and isolated single-frame-loss controls are in `synthetic_isolated_clock_controls.csv` / "
              "`synthetic_isolated_clock_summary.csv`; these separate numerical jitter effects from the deliberate large-gap rejection. "
              "`synthetic_estimator_summary.csv` compares correlation, phase and reliable matched-extrema errors/availability. "
              "A missing frame at 30Hz exceeds the configured 50.001ms interpolation allowance; explicit failure is expected there.", ""]
    lines += table(["Hz", "Method", "Scenario", "Attempted", "Computable", "Bias ms", "MAE ms", "Max |error| ms"],
                   [(r["fs"], r["method"], r["scenario"], r["n"], r["computable"], r["bias_ms"], r["mae_ms"], r["max_error_ms"]) for r in short_synthetic])
    lines += ["", "![Synthetic timing error](synthetic_delay_error.png)", "",
              "Positive lag uses hand(t)=face(t−delay). The correlation peak has model-based parabolic subframe interpolation; no excessive upsampling is used. "
              "Fundamental phase is modulo a cardiac period. Reversing signal order reverses signed timing. "
              "Morphology mismatch biases can dominate clean numerical error. Successful simulations do not validate physiological PTT or camera timing.",
              "", "## I–K. Actual delays, ambiguity, repeatability and stability", "",
              "The primary pairing is combined face / palm GREEN, declared independently of BP. Other ROIs/methods are sensitivity diagnostics. "
              "Signed normalized cross-correlation, fundamental cross-spectral phase/coherence and reliable matched algorithm extrema are reported separately. "
              "No polarity is reversed or absolute correlation substituted. Competing full-cycle peaks, search-edge optima and polarity ambiguity remain flagged. "
              "Beat landmarks require both reliable-beat screens, ≥4 unique matches, ≥80% matching and local morphology correlation ≥0.8; "
              "these are algorithm extrema, not validated pulse feet.", ""]
    lines += table(["Session", "Scheduled", "Computable", "Strict", "Computable union s", "Strict union s", "Candidate-free computable s", "Median lag ms", "Median floor ms", "Median half disagreement ms"],
                   [(r["session_id"], r["scheduled"], r["computable"], r["strict"], r["computable_union_sec"], r["strict_union_sec"], r["candidate_free_computable_union_sec"], r["median_delay_ms"], r["median_uncertainty_floor_ms"], r["median_split_difference_ms"]) for r in cohort])
    lines += ["", "Strict timing requires both good waveforms, HR agreement ≤5bpm, signed correlation ≥0.5, coherence ≥0.8, no cycle/polarity/boundary ambiguity, "
              "split/phase disagreement ≤33.334ms, no source interval >1.5 median steps, and agreement across at least two methods and face ROIs within 33.334ms. "
              "Intermediate retains good paired waveforms failing timing checks; uncertain retains other computable delays. "
              "No rule is relaxed for acceptance or BP correlation. Engineering strict status still does not establish physiological validity.",
              "", "The uncertainty floor is the maximum of the median source step, largest window step, half-window disagreement/2, phase/correlation disagreement, "
              "and half the cross-method/ROI range. It is a conservative diagnostic lower bound, **not a calibrated confidence interval**; "
              "unknown systematic camera and algorithmic bias can exceed it. `delay_stability.csv` separates within-session/phase/method variation "
              "using nonoverlapping windows. Between-subject differences can reflect acquisition and morphology bias.",
              "Strict-lag medians, uncertainty floors and nonoverlapping counts are retained in `cohort_summary.csv`; "
              "a near-zero strict estimate below its timing floor does not resolve a physiological delay or even its sign.",
              "", "![All delay traces](delay_vs_time.png)", "", "![Timing uncertainty](delay_uncertainty.png)", "",
              "![Algorithm agreement](delay_method_agreement.png)", "", "![First, low-quality and failed examples](face_hand_examples.png)",
              "", "## L. Independent cuff associations and BP information", "",
              f"`cuff_associations.csv` contains exactly {len(associations)} unique intermittent cuff records; {available} have verified strict preceding features. "
              "All primary associations are unavailable because legacy timing has not been video reviewed. No overlapping window is an independent cuff label. "
              "Candidate sensitivity records explicitly reuse the same event IDs (not new independent observations), compare policies A/B/C, strict/intermediate/all-computable "
              "selection and maximum feature-to-cuff distances 15/30/60s. Reused feature intervals and within-subject dependence remain flagged. "
              "No feature is taken from a later or cuff-active interval. BP is a later intermittent oscillometric reading, not an instantaneous pre-cuff waveform label.", ""]
    lines += table(["Analysis", "Subject/scope", "Target", "Feature", "Events", "Value"],
                   [(r["analysis"], r.get("subject_id", r.get("scope", "")), r.get("target", ""), r.get("feature", ""), r.get("n_events"), r.get("value")) for r in bp])
    lines += ["", "The cuff-label-only baseline uses equal training-subject means in leave-one-subject-out folds. It is a descriptive reference only: "
              "no verified delay/HR feature comparison is available, so no optical BP MAE, learned BP model, or clinical accuracy is reported. "
              "Pooled and within-subject-centered candidate correlations are unverified exploratory sensitivity findings, vulnerable to acquisition bias and N=6 sampling. "
              "Within-subject correlations require at least two subjects with repeated eligible events and three within-subject degrees of freedom; "
              "single-event subjects cannot contribute within-subject changes, and two centered observations do not establish a meaningful perfect correlation. "
              "No p-value or window-level significance calculation is justified.", "", "![Exploratory SBP sensitivity](delay_vs_sbp.png)", "", "![Exploratory DBP sensitivity](delay_vs_dbp.png)",
              "", "## M. Negative findings and explicit rejection reasons", ""]
    lines += table(["Reason (policy B primary pairing)", "Window count; reasons can overlap"], rejection.most_common())
    lines += ["", f"Missing/failed recording analyses: {json.dumps(failures)}.", "",
              "## N. Interpretation, limitations and next collection", "", recommendation, "",
              "No true arterial path length is inferred from suprasternal surface distances. Face and hand lie on different branches; the opposite-arm cuff does not validate "
              "right-hand optical timing. Sitting/rest/support metadata reduce some confounds without providing timing ground truth. "
              "Six people and at most 18 repeated cuff events cannot establish population generalization, clinical BP measurement error or AAMI/ISO compliance.", ""]
    lines += table(["Evidence class", "Meaning"], summary["definitions"].items())
    lines += ["", "## Reproduction and artifact index", "",
              "Run from repository root with the existing environment:", "", "```powershell",
              ".\\.venv\\Scripts\\python.exe scripts/evaluate_pilot6_delay.py --workers 3 --out-dir outputs/pilot6_protocol_aware_delay_v1/reproduction01",
              ".\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v", "```", "",
              "`--metadata-path` selects an explicit newer export and records its path/hash; there is no older-workbook fallback. "
              "`--stage acquire` then `--stage analyze --resume` separates acquisition from analysis within a single input-identity-checked run. "
              "`--frozen-face` optionally executes the original facial API and V2 adapter independently. Existing output directories require explicit matching-input resume; "
              "all outputs must remain within the isolated namespace.", "",
              "`experiment_manifest.json` records Git identity/dirty state, all code hashes, config/hash/seed, input/checkpoint/manifest hashes, Python/dependencies, "
              "detector assets/backend, timebase, source workbook and protected pre/post identities. Per-session acquisition JSON records native cache identity. "
              "`metadata_audit.json`, `metadata_changes.csv`, `session_timeline.csv`, `cuff_events.csv`, `eligible_intervals.csv`, `camera_timebase_audit.csv`, "
              "`roi_validity.csv`, `waveform_quality.csv`, `waveform_quality_summary.csv`, `delay_windows.csv`, `delay_stability.csv`, "
              "`synthetic_timing_validation.csv`, `synthetic_timing_summary.csv`, `cuff_associations.csv`, `cuff_associations_sensitivity.csv`, "
              "`bp_exploratory_analysis.csv`, `rejection_reasons.csv`, `cohort_summary.csv` and `results_summary.json` are the machine-readable evidence. "
              "Tests and execution logs are linked in `verification.json` when post-run verification completes. "
              "All physiological source data and results remain ignored/local; exported figures contain no recognizable subject pixels.", ""]
    (out / "research_report.md").write_text("\n".join(lines), encoding="utf-8")
