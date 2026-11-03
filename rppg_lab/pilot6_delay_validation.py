"""Signed timing diagnostics and label-free, actual-clock synthetic controls.

Numerical interpolation precision is never equated with source timing accuracy.
"""
from __future__ import annotations

from dataclasses import replace
import numpy as np
from scipy.signal import correlate, correlation_lags, find_peaks

from .bp import estimate_delay
from .classical import METHOD_FUNCS
from .config import PipelineConfig
from .processing import shared_grid, resample_trace, zero_phase_bandpass, valid_runs
from .types import RGBTrace
from .quality import signal_quality
from .waveform_visualization import reliable_beats

METHOD_VERSION = "signed_native_xcorr_fundamental_phase_reliable_peaks_v1"


def timing_diagnostics(face, hand, fs, hr_hz, max_delay_sec=.3, landmarks=True):
    """Original signed estimator, augmented with explicit ambiguity/robustness.

    Cross spectral phase is evaluated at the face spectral fundamental with
    coherence over multiple Welch segments. Phase alone is modulo one cycle.
    A wide correlation search diagnoses other cycle peaks; it never changes
    the prespecified primary search range or flips polarity.
    """
    x, y = np.asarray(face, float), np.asarray(hand, float)
    result = estimate_delay(x, y, fs, max_delay_sec, hr_hz)
    result.update(cycle_ambiguity=False, polarity_ambiguity=False, search_boundary=False,
                  split_difference_sec=np.nan, phase_xcorr_difference_sec=np.nan,
                  correlation_peak_width_sec=np.nan, competing_cycle_peak_score=np.nan,
                  matched_beat_count=0, beat_landmarks_available=False,
                  beat_reason="disabled_or_unreliable", beat_detection_repeatability=np.nan)
    if len(x) < 3 * fs or min(np.std(x), np.std(y)) < 1e-10 or not np.isfinite(hr_hz) or hr_hz <= 0:
        result["failure_reason"] = "short_flat_or_unknown_hr"
        return result
    period = 1 / hr_hz
    z, w = x - x.mean(), y - y.mean()
    c = correlate(w, z, mode="full", method="fft") / (np.linalg.norm(z) * np.linalg.norm(w))
    lag = correlation_lags(len(w), len(z)) / fs
    radius = min(max_delay_sec, np.floor(.49 * fs / hr_hz) / fs)
    keep = np.abs(lag) <= radius + 1e-10
    ck, lk = c[keep], lag[keep]
    k = int(np.argmax(ck))
    result["search_boundary"] = k in (0, len(ck) - 1)
    result["polarity_ambiguity"] = bool(-np.min(ck) > np.max(ck) + .1)
    top = ck >= ck[k] - .1
    if np.any(top):
        result["correlation_peak_width_sec"] = float(np.ptp(lk[top]))
    peaks, _ = find_peaks(c)
    rivals = [p for p in peaks if .7 * period < abs(lag[p] - lk[k]) < 1.3 * period]
    if rivals:
        competitor = float(max(c[p] for p in rivals))
        result["competing_cycle_peak_score"] = competitor
        result["cycle_ambiguity"] = competitor >= ck[k] - .05
    delay, phase = result["delay_xcorr_sec"], result["delay_phase_sec"]
    result["phase_xcorr_difference_sec"] = abs(delay - phase)
    halves = [estimate_delay(a, b, fs, max_delay_sec, hr_hz)["delay_xcorr_sec"]
              for a, b in zip(np.array_split(x, 2), np.array_split(y, 2))]
    result["first_half_delay_sec"], result["second_half_delay_sec"] = halves
    result["split_difference_sec"] = abs(halves[1] - halves[0])
    # The historical unrestricted peak matches are retained separately. Primary
    # beat results require both independent reliable-beat screens and morphology.
    for key in ("delay_beat_median_sec", "delay_beat_std_sec", "delay_valid_beat_fraction"):
        result["unrestricted_" + key] = result[key]
        result[key] = np.nan
    if landmarks:
        a, b = reliable_beats(x, fs), reliable_beats(y, fs)
        if a["reliable"] and b["reliable"] and ck[k] >= .7 and not result["polarity_ambiguity"]:
            used, differences, matched_templates = set(), [], []
            for peak in a["peaks"]:
                candidates = [q for q in b["peaks"] if q not in used and abs(q - peak) / fs <= radius]
                if candidates:
                    q = min(candidates, key=lambda q: abs(q - peak))
                    used.add(q)
                    differences.append((q - peak) / fs)
                    half = max(2, int(.2 * fs / hr_hz))
                    if peak >= half and q >= half and peak + half < len(x) and q + half < len(y):
                        matched_templates.append(np.corrcoef(x[peak-half:peak+half+1], y[q-half:q+half+1])[0, 1])
            fraction = len(differences) / max(len(a["peaks"]), len(b["peaks"]), 1)
            result["beat_detection_repeatability"] = fraction
            if len(differences) >= 4 and fraction >= .8 and matched_templates and np.median(matched_templates) >= .8:
                result.update(beat_landmarks_available=True, beat_reason="matched_algorithm_extrema_not_validated_pulse_feet",
                              matched_beat_count=len(differences), delay_beat_median_sec=float(np.median(differences)),
                              delay_beat_std_sec=float(np.std(differences)), delay_valid_beat_fraction=fraction)
            else:
                result["beat_reason"] = "unreliable_matching_or_morphology"
    result["failure_reason"] = "search_boundary" if result["search_boundary"] else "" if np.isfinite(delay) else "noncomputable"
    return result


def pulse(t, hr_bpm, mismatch=False):
    """Smooth asymmetric pulse-like harmonic mixture with slow rate modulation."""
    phase = 2 * np.pi * (hr_bpm / 60 * t + .025 * np.sin(2 * np.pi * .12 * t))
    return (np.sin(phase) + (.44 if mismatch else .28) * np.sin(2 * phase + (.85 if mismatch else .2))
            + .1 * np.sin(3 * phase + .4))


def synthetic_validation(config, clocks=None, full=True):
    """Exercise RGB projection, native-clock interpolation, gaps, filter, estimators.

    Fixed scenario grid and seed, no participant BP or pulse labels are consumed.
    """
    rng = np.random.default_rng(config["seed"])
    rows = []
    sources = {"regular30": np.arange(0, 18, 1/30), "regular60": np.arange(0, 18, 1/60)}
    if full and clocks:
        for name, clock in clocks.items():
            # Preserve the actual irregular intervals at a fixed 20s offset.
            cut = np.asarray(clock)[(clock >= 20) & (clock < 38)]
            sources[name] = cut - cut[0]
    scenarios = ("clean", "noise_drift_amplitude", "morphology_mismatch", "jitter_missing_frames") if full else ("clean",)
    for clock_name, original in sources.items():
        fs = 30. if clock_name == "regular30" else 60.
        for hr in config["synthetic_hr_bpm"]:
            for delay_ms in config["synthetic_delays_ms"]:
                for scenario in scenarios:
                    replicates = config["synthetic_replicates"] if scenario != "clean" else 1
                    for replicate in range(replicates):
                        t = original.copy()
                        if scenario == "jitter_missing_frames":
                            t += rng.uniform(-.0015, .0015, len(t))
                            keep = rng.random(len(t)) > .025
                            # Explicit >gap-limit loss inside the analysis window.
                            keep &= ~((t > 8.8) & (t < 8.95))
                            t = t[keep]
                        grid = shared_grid(t, fs)
                        pairs, interpolation = [], []
                        for site, delay in enumerate((0., delay_ms / 1000)):
                            p = pulse(t - delay, hr, mismatch=site == 1 and scenario == "morphology_mismatch")
                            nuisance = .0015 * np.sin(2*np.pi*3.8*(t-delay) + .3)
                            rgb = np.array([140., 120., 90.]) * (1 + p[:, None] * np.array([.006, .012, .003])
                                  + nuisance[:, None] * np.array([1., .2, -1.]))
                            if scenario in ("noise_drift_amplitude", "jitter_missing_frames"):
                                rgb += (3 * np.sin(.45 * t))[:, None]
                                rgb += rng.normal(0, .12 if scenario == "noise_drift_amplitude" else .05, rgb.shape)
                                rgb *= 1.6 if site else .8
                            trace = RGBTrace(t, rgb, "synthetic", np.ones(len(t), bool))
                            uniform, imputed = resample_trace(trace, grid, config["max_gap_sec"])
                            pairs.append(uniform)
                            interpolation.append(float(imputed.mean()))
                        for method in ("GREEN", "CHROM", "POS"):
                            # Use the same common finite runs as the real delay path.
                            joint = np.isfinite(pairs[0]).all(axis=1) & np.isfinite(pairs[1]).all(axis=1)
                            waves = {}
                            for filtering in (False, True):
                                out = [np.full(len(grid), np.nan), np.full(len(grid), np.nan)]
                                for a, b in valid_runs(joint):
                                    if b-a < 4 * fs:
                                        continue
                                    for site in (0, 1):
                                        sig = METHOD_FUNCS[method](pairs[site][a:b], fs, do_filter=False)
                                        if filtering:
                                            sig = zero_phase_bandpass(sig, fs, config["filter_low_hz"], config["filter_high_hz"])
                                        edge = round(config["filter_edge_sec"] * fs)
                                        out[site][a+edge:b-edge] = sig[edge:-edge]
                                # A fixed 10s interval fails if the known gap crosses it.
                                take = (grid >= 4) & (grid < 14)
                                x, y = [v[take] for v in out]
                                row = dict(clock=clock_name, fs=fs, hr_bpm=hr, true_delay_ms=delay_ms,
                                           scenario=scenario, replicate=replicate, method=method, filtered=filtering,
                                           interpolation_fraction=np.mean(interpolation), source_interval_sec=np.median(np.diff(t)),
                                           source_max_gap_sec=np.max(np.diff(t)), timing_convention="hand(t)=face(t-delay)")
                                if np.isfinite(x).all() and np.isfinite(y).all():
                                    measured_hr = signal_quality(x, fs)["hr_bpm"]
                                    row["estimated_face_hr_bpm"] = measured_hr
                                    row["estimator_hr_source"] = "spectral_face_waveform_not_oracle"
                                    d = timing_diagnostics(x, y, fs, measured_hr/60, config["max_delay_sec"], landmarks=True)
                                    row.update({k: v for k, v in d.items() if not k.startswith("unrestricted_")})
                                    row["error_ms"] = 1000*d["delay_xcorr_sec"]-delay_ms
                                    row["phase_error_ms"] = 1000*d["delay_phase_sec"]-delay_ms
                                    row["noncomputable"] = not np.isfinite(d["delay_xcorr_sec"])
                                    row["recovery_error_limit_ms"] = 1000/fs
                                    row["failure"] = row["noncomputable"] or abs(row["error_ms"]) > 1000/fs
                                    row["failure_reason"] = d["failure_reason"] or ("recovery_error_exceeds_one_source_step" if row["failure"] else "")
                                else:
                                    row.update(error_ms=np.nan, phase_error_ms=np.nan, failure=True, noncomputable=True, recovery_error_limit_ms=1000/fs,
                                               failure_reason="known_gap_or_filter_support_not_bridged")
                                rows.append(row)
    return rows


def summarize_simulations(rows):
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in ("clock", "fs", "method", "scenario", "filtered"))
        groups.setdefault(key, []).append(row)
    summary = []
    for key, group in groups.items():
        errors = np.array([r.get("error_ms", np.nan) for r in group])
        valid = np.isfinite(errors)
        e = errors[valid]
        summary.append(dict(zip(("clock", "fs", "method", "scenario", "filtered"), key),
                            attempted=len(group), computable=int(valid.sum()),
                            noncomputable_rate=1-float(valid.mean()), failure_rate=float(np.mean([r["failure"] for r in group])),
                            bias_ms=float(e.mean()) if len(e) else None,
                            mae_ms=float(np.abs(e).mean()) if len(e) else None,
                            max_absolute_error_ms=float(np.abs(e).max()) if len(e) else None,
                            variance_ms2=float(e.var()) if len(e) else None,
                            cycle_ambiguity_count=sum(r.get("cycle_ambiguity", False) for r in group)))
    return summary


def run_synthetic_controls(out, config, clocks):
    """Identity checked simulation cache supports long acquisition/analysis runs."""
    import json
    import hashlib
    from pathlib import Path
    from .artifacts import file_sha256, write_json
    from .waveform_visualization import write_csv
    out = Path(out)
    identity = dict(config=config, validation_source_sha256=file_sha256(__file__),
                    processing_source_sha256=file_sha256(Path(__file__).with_name("processing.py")),
                    classical_source_sha256=file_sha256(Path(__file__).with_name("classical.py")),
                    estimator_source_sha256=file_sha256(Path(__file__).with_name("bp.py")),
                    clocks={n: hashlib.sha256(np.asarray(t, dtype=np.float64).tobytes()).hexdigest() for n, t in clocks.items()})
    cache = out / "synthetic_results.json"
    if cache.exists():
        saved = json.loads(cache.read_text(encoding="utf-8"))
        if saved["identity"] != identity:
            raise ValueError("Synthetic controls identity differs; use a fresh experiment run")
        return saved["rows"], saved["summary"]
    rows = synthetic_validation(config, clocks)
    summary = summarize_simulations(rows)
    write_csv(out / "synthetic_timing_validation.csv", rows)
    write_csv(out / "synthetic_timing_summary.csv", summary)
    write_json(cache, dict(identity=identity, rows=rows, summary=summary))
    return rows, summary
