"""Isolated, label-free signal/delay analysis and independent cuff associations."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from .artifacts import write_json, file_sha256
from .config import PipelineConfig
from .pilot_analysis import window_quality, finite_stat
from .pilot6_delay_acquisition import ROIS
from .pilot6_protocol_timeline import timeline_masks
from .pilot6_delay_validation import timing_diagnostics, run_synthetic_controls, METHOD_VERSION
from .processing import resample_trace, shared_grid, extract_signal, valid_runs
from .types import RGBTrace
from .waveform_visualization import write_csv, reliable_beats

METHODS = ("GREEN", "CHROM", "POS")
FACE_ROIS = ROIS[:4]
HAND_ROIS = ROIS[4:]


def finite(value):
    return value is not None and np.isfinite(value)


def union_seconds(rows):
    total, end = 0., -np.inf
    for a, b in sorted((r["start_sec"], r["end_sec"]) for r in rows):
        total += max(0., b - max(a, end))
        end = max(b, end)
    return total


def phase_labels(masks):
    return np.where(masks["cuff_active"], 1, np.where(masks["cuff_boundary_guard"], 2,
                    np.where(masks["candidate_cuff_free"], 0, 3)))


def scheduled_windows(phase, clock_support, fs, config):
    """Retain global attempts; anchor primary windows inside each native segment.

    Scheduling uses protocol bounds and camera support only, before ROI quality
    or BP values. Primary windows never overlap, including across phase/gap edges.
    """
    n = len(phase)
    duration, stride = round(config["window_sec"]*fs), round(config["stride_sec"]*fs)
    edge = round(config["filter_edge_sec"]*fs)
    starts = set(range(0, n-duration+1, stride))
    primary = set()
    for label in np.unique(phase):
        for a, b in valid_runs((phase == label) & clock_support):
            starts.update(range(a+edge, b-edge-duration+1, stride))
            primary.update(range(a+edge, b-edge-duration+1, duration))
    return sorted(starts), primary


def analyze_recording(arrays, session, config, out):
    t = arrays["timestamps"]
    fs = float(round(1 / np.median(np.diff(t))))
    if not 29 <= fs <= 61:
        raise ValueError("Unsupported native sample rate")
    cfg = PipelineConfig(sample_rate=fs, max_gap_sec=config["max_gap_sec"], edge_guard_sec=config["filter_edge_sec"],
                         filter_low_hz=config["filter_low_hz"], filter_high_hz=config["filter_high_hz"],
                         min_valid_fraction=config["min_valid_fraction"], min_snr_db=config["min_snr_db"])
    grid = shared_grid(t, fs)
    traces, rgb, imputed = {}, {}, {}
    acquisitions = []
    dt = np.diff(t)
    decoder = arrays["backend_timestamps"]
    clock_audit = dict(session_id=session["session_id"], source_frames=len(t), source_seconds=float(t[-1]-t[0]),
                       median_interval_ms=1000*np.median(dt), max_interval_ms=1000*np.max(dt),
                       intervals_over_1_5_median=int(np.sum(dt > 1.5*np.median(dt))),
                       repeated_decoded_pixel_frames=int(np.sum(arrays["repeated_analysis_frame"])),
                       relative_decoder_pts_max_error_sec=float(np.max(np.abs((decoder-decoder[0])-(t-t[0])))),
                       exposure_timing="unmeasured", rolling_shutter="unmeasured",
                       hand_backend=str(arrays["actual_hand_backend"].item()),
                       hand_classifications="|".join(sorted(set(arrays["hand_identity"].tolist()))),
                       timing_scope="MP4 display clock correspondence, not calibrated exposure time")
    for roi in ROIS:
        quality = {k[len(roi)+1:]: v for k, v in arrays.items() if k.startswith(roi+"_") and v.shape == t.shape}
        values = arrays[roi+"_rgb"]
        trace = RGBTrace(t, values, roi, np.isfinite(values).all(axis=1), quality)
        traces[roi] = trace
        rgb[roi], imputed[roi] = resample_trace(trace, grid, config["max_gap_sec"])
        site = "hand" if roi in HAND_ROIS else "face"
        reason = arrays[site+"_tracking_reason"]
        acquisitions.append(dict(session_id=session["session_id"], roi=roi, source_frames=len(t),
                                  detector_coverage_fraction=float(np.mean(reason == "ok")),
                                  source_validity_fraction=float(trace.validity_mask.mean()),
                                  raw_valid_time_sec=float(np.sum(np.r_[dt, 0.] * trace.validity_mask)),
                                  median_pixels=finite_stat(quality.get("skin_pixel_count", [])),
                                  median_clipping_fraction=finite_stat(quality.get("clipping_fraction", [])),
                                  source_interpolated_grid_fraction=float(imputed[roi].mean()),
                                  detector_reason_counts=json.dumps(dict(Counter(reason.tolist())), sort_keys=True),
                                  confidence_scope="Face confidence missing; hand score is classification only"))
    quality_rows, delays, signals_out = [], [], {"uniform_pts": grid, "source_pts": t}
    duration = round(config["window_sec"]*fs)
    stride = round(config["stride_sec"]*fs)
    clock_trace = RGBTrace(t, np.ones((len(t), 3)), "clock_support", np.ones(len(t), bool))
    clock_support = np.isfinite(resample_trace(clock_trace, grid, config["max_gap_sec"])[0]).all(axis=1)
    phase_names = {0: "candidate_cuff_free", 1: "cuff_active", 2: "cuff_boundary_guard", 3: "timing_unknown"}
    for policy, guard in config["policies"].items():
        masks = timeline_masks(grid, session["events"], guard)
        phase = phase_labels(masks)
        starts, primary_starts = scheduled_windows(phase, clock_support, fs, config)
        # Each phase and every acquisition gap has separate filter support.
        # Masks retain the full clock; no compressed/concatenated segments exist.
        waves, qualities = {}, {}
        for roi in ROIS:
            for method in METHODS:
                wave = np.full(len(grid), np.nan)
                for label in np.unique(phase):
                    s = extract_signal(rgb[roi], grid, roi, replace(cfg, method=method), allowed=phase == label)
                    take = np.isfinite(s.values)
                    wave[take] = s.values[take]
                waves[roi, method] = wave
                signals_out[f"{policy}_{roi}_{method}"] = wave
                for index, a in enumerate(starts):
                    b = a+duration
                    left, right = float(grid[a]), float(grid[a]+config["window_sec"])
                    labels = np.unique(phase[a:b])
                    q = window_quality(wave[a:b], traces[roi], left, right, fs, cfg)
                    beat = reliable_beats(wave[a:b], fs)
                    reasons = list(q["reasons"])
                    if len(labels) != 1:
                        reasons.append("crosses_cuff_phase_boundary")
                        q["status"] = "invalid"
                    native = (t >= left) & (t < right)
                    native_dt = np.diff(t[native])
                    row = dict(session_id=session["session_id"], subject_id=session["subject_id"], policy=policy,
                               roi=roi, method=method, window_index=index, start_sec=left, end_sec=right,
                               primary_nonoverlapping=a in primary_starts,
                               schedule="phase_gap_anchored_primary" if a in primary_starts else "supplemental_global_or_overlapping",
                               cuff_phase=phase_names[int(labels[0])] if len(labels) == 1 else "crosses_boundary",
                               timing_verified=bool(masks["verified_cuff_free"][a:b].all()),
                               **{k: v for k, v in q.items() if k != "reasons"}, reasons="|".join(reasons),
                               beat_detection_reliable=beat["reliable"], detected_beats=len(beat["anchors"]),
                               max_source_gap_sec=float(np.max(native_dt)) if len(native_dt) else np.nan,
                               repeated_pixel_frames=int(arrays["repeated_analysis_frame"][native].sum()))
                    quality_rows.append(row)
                    qualities[roi, method, index] = row
        for face_roi in FACE_ROIS:
            for hand_roi in HAND_ROIS:
                joint = np.isfinite(rgb[face_roi]).all(axis=1) & np.isfinite(rgb[hand_roi]).all(axis=1)
                for method in METHODS:
                    paired = [np.full(len(grid), np.nan) for _ in (0, 1)]
                    for label in np.unique(phase):
                        for site, roi in enumerate((face_roi, hand_roi)):
                            s = extract_signal(rgb[roi], grid, roi, replace(cfg, method=method), allowed=joint & (phase == label))
                            valid = np.isfinite(s.values)
                            paired[site][valid] = s.values[valid]
                    for index, a in enumerate(starts):
                        b = a+duration
                        fq, hq = [qualities[r, method, index] for r in (face_roi, hand_roi)]
                        row = dict(session_id=session["session_id"], subject_id=session["subject_id"], policy=policy,
                                   face_roi=face_roi, hand_roi=hand_roi, method=method, window_index=index,
                                   start_sec=fq["start_sec"], end_sec=fq["end_sec"],
                                   cuff_phase=fq["cuff_phase"], timing_verified=fq["timing_verified"],
                                   primary_nonoverlapping=fq["primary_nonoverlapping"],
                                   face_hr_bpm=fq["hr_bpm"], hand_hr_bpm=hq["hr_bpm"],
                                   face_snr_db=fq["snr_db"], hand_snr_db=hq["snr_db"],
                                   face_quality=fq["status"], hand_quality=hq["status"],
                                   max_source_gap_sec=fq["max_source_gap_sec"], source_interval_sec=float(np.median(dt)),
                                   method_version=METHOD_VERSION, computable=False, status="invalid")
                        x, y = [p[a:b] for p in paired]
                        reasons = []
                        if fq["cuff_phase"] == "crosses_boundary":
                            reasons.append("crosses_cuff_phase_boundary")
                        if not np.isfinite(x).all() or not np.isfinite(y).all():
                            reasons.append("missing_paired_signal_gap_or_filter_edge")
                        if not reasons:
                            hr = fq["hr_bpm"]/60
                            d = timing_diagnostics(x, y, fs, hr, config["max_delay_sec"])
                            row.update(d)
                            row["computable"] = finite(d["delay_xcorr_sec"])
                            if not row["computable"]:
                                reasons.append("delay_noncomputable_or_boundary")
                            for site, q in (("face", fq), ("hand", hq)):
                                if q["status"] != "good":
                                    reasons.append(site + "_waveform_" + q["status"])
                            if abs(fq["hr_bpm"] - hq["hr_bpm"]) > config["max_hr_difference_bpm"]:
                                reasons.append("face_hand_hr_disagreement")
                            if d["delay_xcorr_score"] < config["min_correlation"]:
                                reasons.append("low_signed_correlation")
                            if not finite(d["delay_phase_coherence"]) or d["delay_phase_coherence"] < config["min_coherence"]:
                                reasons.append("low_cross_spectral_coherence")
                            for field in ("cycle_ambiguity", "polarity_ambiguity", "search_boundary"):
                                if d[field]:
                                    reasons.append(field)
                            for field, limit in (("split_difference_sec", "max_split_difference_sec"),
                                                 ("phase_xcorr_difference_sec", "max_estimator_difference_sec")):
                                if not finite(d[field]) or d[field] > config[limit]:
                                    reasons.append(field + "_too_large")
                            if fq["max_source_gap_sec"] > 1.5*np.median(dt)+1e-6:
                                reasons.append("irregular_source_gap")
                            row["uncertainty_floor_sec"] = max(float(np.median(dt)), fq["max_source_gap_sec"],
                                d["split_difference_sec"] / 2 if finite(d["split_difference_sec"]) else 0.,
                                d["phase_xcorr_difference_sec"] if finite(d["phase_xcorr_difference_sec"]) else 0.)
                            if row["computable"]:
                                row["status"] = "strict" if not reasons else "intermediate" if fq["status"] == hq["status"] == "good" else "uncertain"
                        if session["manual_excluded"]:
                            reasons.append("manual_exclude")
                            row["status"] = "excluded"
                        row["reasons"] = "|".join(reasons)
                        delays.append(row)
    # Agreement diagnostics use the same locations/time and never use BP labels.
    by_methods, by_rois = defaultdict(list), defaultdict(list)
    for row in delays:
        by_methods[row["policy"], row["face_roi"], row["hand_roi"], row["window_index"]].append(row)
        by_rois[row["policy"], row["method"], row["hand_roi"], row["window_index"]].append(row)
    for groups, field, threshold in ((by_methods, "cross_method_range_sec", config["max_method_difference_sec"]),
                                     (by_rois, "cross_roi_range_sec", config["max_method_difference_sec"])):
        for group in groups.values():
            good = [r["delay_xcorr_sec"] for r in group if r["computable"] and r["face_quality"] == r["hand_quality"] == "good"]
            spread = float(np.ptp(good)) if len(good) >= 2 else np.nan
            for row in group:
                row[field] = spread
                if not finite(spread) or spread > threshold:
                    row["reasons"] += ("|" if row["reasons"] else "") + field + "_unconfirmed"
                    if row["status"] == "strict":
                        row["status"] = "intermediate"
                if finite(spread) and row["computable"]:
                    row["uncertainty_floor_sec"] = max(row["uncertainty_floor_sec"], spread / 2)
    np.savez_compressed(Path(out) / "classical_waveforms.npz", **signals_out)
    write_json(Path(out) / "camera_timebase_audit.json", clock_audit)
    write_csv(Path(out) / "waveform_quality.csv", quality_rows)
    write_csv(Path(out) / "delay_windows.csv", delays)
    return quality_rows, delays, acquisitions, clock_audit


def analyze_cached_recording(path, session, config):
    """Validate native cache, and reuse only exact code/config/metadata analyses."""
    path = Path(path)
    acquisition = json.loads((path.parent / "acquisition.json").read_text(encoding="utf-8"))
    with np.load(path, allow_pickle=False) as z:
        identity = json.loads(str(z["identity_json"].item()))
        if identity != acquisition["identity"] or identity["video_sha256"] != file_sha256(session["video"]):
            raise ValueError("Native acquisition/video identity conflict")
        for name, digest in identity["source_sha256"].items():
            if file_sha256(Path(__file__).with_name(name + ".py")) != digest:
                raise ValueError(f"Native acquisition source changed: {name}")
        arrays = {k: z[k].copy() for k in z.files}
    fingerprint = dict(native_sha256=file_sha256(path), config=config, events=session["events"],
                       manual_excluded=session["manual_excluded"],
                       analysis_sources={n: file_sha256(Path(__file__).with_name(n + ".py")) for n in
                                         ("pilot6_delay_feasibility", "pilot6_delay_validation", "pilot6_protocol_timeline",
                                          "processing", "pilot_analysis", "quality", "classical", "waveform_visualization")})
    cache = path.parent / "analysis_cache.json"
    if cache.is_file():
        saved = json.loads(cache.read_text(encoding="utf-8"))
        if saved["identity"] == fingerprint:
            return saved["quality"], saved["delays"], saved["acquisition"], saved["camera"]
    q, d, a, c = analyze_recording(arrays, session, config, path.parent)
    write_json(cache, dict(identity=fingerprint, quality=q, delays=d, acquisition=a, camera=c))
    return q, d, a, c


def primary_rows(rows, policy="B"):
    return [r for r in rows if r["policy"] == policy and r["face_roi"] == "combined" and
            r["hand_roi"] == "hand_palm" and r["method"] == "GREEN"]


def stability_summary(rows):
    groups = defaultdict(list)
    for r in rows:
        if r["primary_nonoverlapping"]:
            groups[r["session_id"], r["policy"], r["face_roi"], r["hand_roi"], r["method"], r["cuff_phase"]].append(r)
    summary = []
    for key, group in groups.items():
        for selection in ("all_computable", "strict", "intermediate", "uncertain"):
            subset = [r for r in group if r["computable"] and (selection == "all_computable" or r["status"] == selection)]
            values = np.asarray([r["delay_xcorr_sec"] for r in subset])
            ordered = sorted(subset, key=lambda r: r["start_sec"])
            adjacent = [abs(b["delay_xcorr_sec"]-a["delay_xcorr_sec"]) for a, b in zip(ordered, ordered[1:])
                        if abs(b["start_sec"]-a["end_sec"]) <= 1e-6]
            summary.append(dict(zip(("session_id", "policy", "face_roi", "hand_roi", "method", "cuff_phase"), key),
                selection=selection, attempted_nonoverlapping_windows=len(group), computable_windows=len(subset),
                coverage_sec=union_seconds(subset), median_delay_ms=1000*float(np.median(values)) if len(values) else None,
                delay_std_ms=1000*float(np.std(values)) if len(values) >= 2 else None,
                delay_iqr_ms=1000*float(np.ptp(np.percentile(values, [25, 75]))) if len(values) >= 2 else None,
                adjacent_contiguous_pairs=len(adjacent),
                median_adjacent_difference_ms=1000*float(np.median(adjacent)) if adjacent else None,
                median_uncertainty_floor_ms=1000*finite_stat([r["uncertainty_floor_sec"] for r in subset]) if subset else None,
                matched_nonoverlapping_beats=sum(r.get("matched_beat_count", 0) for r in subset)))
    return summary


def cuff_associations(sessions, intervals, delay_rows, config, policy="B", verified_only=True,
                      selection="strict", max_distance=None):
    """One independent row per source cuff event; no window receives a BP label."""
    distance = config["max_feature_distance_sec"] if max_distance is None else max_distance
    unique = {}
    for row in primary_rows(delay_rows, policy):
        key = (row["session_id"], row["start_sec"], row["end_sec"])
        if key in unique and row != unique[key]:
            raise ValueError("Conflicting duplicate feature window")
        unique[key] = row
    primary = list(unique.values())
    result, seen, used_intervals = [], set(), Counter()
    for session in sessions:
        for event in session["events"]:
            if event["event_id"] in seen:
                raise ValueError("Duplicate cuff event label")
            seen.add(event["event_id"])
            row = dict(event, policy=policy, selection=selection, require_verified=verified_only,
                       association_available=False, feature_start_sec=None, feature_end_sec=None,
                       feature_interval_id=None, feature_to_cuff_sec=None, feature_timing_verified=False,
                       eligible_signal_windows=0, independent_matched_beats=0, median_delay_ms=None,
                       delay_dispersion_ms=None, hr_bpm=None, median_snr_db=None,
                       feature_reused=False, independent_record=True, reason="no_eligible_preceding_interval",
                       feature_windows=[], eligible_window_union_sec=0., feature_interval_contains_excluded_gaps=False,
                       surface_suprasternal_face_cm=session.get("subject_raw", {}).get("Suprasternal_to_FaceROI_cm"),
                       surface_suprasternal_fingertip_cm=session.get("subject_raw", {}).get("Suprasternal_to_IndexFingertip_cm"),
                       hand_height_relative_heart_cm=session["raw"].get("Hand_Height_Relative_Heart_cm"),
                       hand_supported=session["raw"].get("Hand_Supported"), hand_roi_side=session["raw"].get("Hand_ROI_Side"),
                       cuff_arm=session["raw"].get("Cuff_Arm"), pre_video_rest_min=session["raw"].get("Pre_Video_Rest_min"),
                       geometry_scope="Surface distances preserved in private source; no arterial length or PWV inferred")
            if session["manual_excluded"]:
                row["reason"] = "manual_exclude"
            elif event["start_sec"] is None:
                row["reason"] = "missing_cuff_start"
            elif verified_only and not event["timing_verified"]:
                row["reason"] = "unverified_cuff_event_timing"
            else:
                candidates = [i for i in intervals if i["session_id"] == session["session_id"] and i["policy"] == policy and
                              i["end_sec"] <= event["start_sec"] and event["start_sec"]-i["end_sec"] <= distance and
                              (i["timing_verified"] or not verified_only)]
                if candidates:
                    interval = max(candidates, key=lambda i: i["end_sec"])
                    eligible = [r for r in primary if r["session_id"] == session["session_id"] and r["computable"] and
                                r["primary_nonoverlapping"] and r["cuff_phase"] == "candidate_cuff_free" and
                                r["start_sec"] >= interval["start_sec"] and r["end_sec"] <= interval["end_sec"] and
                                event["start_sec"]-r["end_sec"] <= distance and
                                (selection == "all_computable" or r["status"] == selection)]
                    if eligible:
                        values = np.array([r["delay_xcorr_sec"] for r in eligible])*1000
                        a, b = min(r["start_sec"] for r in eligible), max(r["end_sec"] for r in eligible)
                        row.update(association_available=True, feature_start_sec=a, feature_end_sec=b,
                                   feature_interval_id=interval["interval_id"], feature_to_cuff_sec=event["start_sec"]-b,
                                   feature_timing_verified=interval["timing_verified"], eligible_signal_windows=len(eligible),
                                   independent_matched_beats=sum(r.get("matched_beat_count", 0) for r in eligible),
                                   median_delay_ms=float(np.median(values)), delay_dispersion_ms=float(np.std(values)) if len(values) >= 2 else None,
                                   hr_bpm=finite_stat([r["face_hr_bpm"] for r in eligible]),
                                   median_snr_db=finite_stat([min(r["face_snr_db"], r["hand_snr_db"]) for r in eligible]),
                                   feature_windows=[dict(start_sec=r["start_sec"], end_sec=r["end_sec"]) for r in eligible],
                                   eligible_window_union_sec=union_seconds(eligible),
                                   feature_interval_contains_excluded_gaps=union_seconds(eligible) < b-a-1e-6,
                                   median_uncertainty_floor_ms=1000*finite_stat([r.get("uncertainty_floor_sec", np.nan) for r in eligible]) if any(finite(r.get("uncertainty_floor_sec")) for r in eligible) else None,
                                   reason="exploratory_pre_cuff_association" if verified_only else "legacy_candidate_timing_sensitivity_only")
                        used_intervals[interval["interval_id"]] += 1
            result.append(row)
    for row in result:
        row["feature_reused"] = bool(row["feature_interval_id"] and used_intervals[row["feature_interval_id"]] > 1)
        row["dependence_scope"] = "events_nested_within_subject; sensitivity tables reuse same labels"
    return result


def subject_folds(records):
    for subject in sorted(set(r["subject_id"] for r in records)):
        train = [r for r in records if r["subject_id"] != subject]
        test = [r for r in records if r["subject_id"] == subject]
        if set(r["subject_id"] for r in train) & set(r["subject_id"] for r in test):
            raise AssertionError("Subject leakage")
        yield train, test


def bp_descriptive(associations, sensitivity):
    """Descriptive associations, never claimed BP accuracy or clinical validation."""
    rows = []
    for subject in sorted(set(r["subject_id"] for r in associations)):
        group = [r for r in associations if r["subject_id"] == subject]
        for target in ("sbp_mmHg", "dbp_mmHg"):
            v = [r[target] for r in group if finite(r[target])]
            rows.append(dict(analysis="within_subject_bp_range", subject_id=subject, target=target, n_events=len(v),
                             value=float(np.ptp(v)) if v else None, unit="mmHg", interpretation="observed intermittent cuff range"))
    available = [r for r in associations if r["association_available"]]
    rows.append(dict(analysis="regression_eligibility", n_events=len(available),
                     value=None, interpretation="No BP regression/MAE: insufficient verified strict event features" if len(available) < 12 else
                     "Regression deferred; this study prioritizes descriptive feasibility"))
    # Same event IDs, alternative timing/quality policy: these never increase N.
    candidate = [r for r in sensitivity if r["association_available"] and r["policy"] == "B" and
                 r["selection"] == "all_computable" and r["max_distance_sec"] == 30.]
    for target in ("sbp_mmHg", "dbp_mmHg", "hr_bpm"):
        for feature in ("median_delay_ms", "delay_dispersion_ms"):
            complete = [r for r in candidate if finite(r.get(target)) and finite(r.get(feature))]
            for scope in ("pooled", "within_subject_centered"):
                records = complete
                counts = Counter(r["subject_id"] for r in records)
                repeated = {s for s, n in counts.items() if n >= 2}
                if scope == "within_subject_centered":
                    records = [r for r in records if r["subject_id"] in repeated]
                x = np.asarray([r[feature] for r in records], dtype=float)
                y = np.asarray([r[target] for r in records], dtype=float)
                if scope == "within_subject_centered":
                    for subject in sorted(set(r["subject_id"] for r in records)):
                        ix = np.array([r["subject_id"] == subject for r in records])
                        x[ix] -= x[ix].mean()
                        y[ix] -= y[ix].mean()
                within_df = len(records)-len(repeated) if scope == "within_subject_centered" else None
                repeat_support = scope == "pooled" or len(repeated) >= 2 and within_df >= 3
                corr = float(np.corrcoef(x, y)[0, 1]) if repeat_support and len(x) >= 3 and min(np.std(x), np.std(y)) > 1e-10 else None
                rows.append(dict(analysis="unverified_candidate_descriptive_correlation", scope=scope, feature=feature, target=target,
                                 n_events=len(records), n_subjects=len(set(r["subject_id"] for r in records)), value=corr,
                                 n_subjects_with_repeats=len(repeated), within_subject_df=within_df,
                                 interpretation="Exploratory sensitivity only; no validated timing, no clinical prediction accuracy" if repeat_support else
                                 "Unavailable: fewer than two repeated subjects or fewer than three within-subject degrees of freedom"))
    # Training-only mean baseline can be computed on all cuff labels without any
    # video feature, but does not establish a matched feature comparison.
    for target in ("sbp_mmHg", "dbp_mmHg"):
        errors = []
        for train, test in subject_folds(associations):
            means = [np.mean([r[target] for r in train if r["subject_id"] == s and finite(r[target])])
                     for s in sorted(set(r["subject_id"] for r in train))]
            pred = np.mean(means)
            errors.extend(abs(r[target]-pred) for r in test if finite(r[target]))
        rows.append(dict(analysis="training_only_subject_mean_baseline", target=target, n_events=len(errors),
                         value=float(np.mean(errors)) if errors else None, unit="mmHg absolute deviation",
                         interpretation="Cuff-label-only LOSO descriptive baseline; no available verified delay/HR comparison"))
    rows.append(dict(analysis="hr_only_vs_hr_plus_delay", n_events=len(available), value=None,
                     interpretation="Unavailable: verified strict feature records insufficient; no test-informed tuning or feature selection"))
    return rows


def analyze_cohort(out, sessions, intervals, config, frozen=None, workers=1):
    out = Path(out)
    clocks = {}
    from .pilot_video import mp4_presentation_times
    for s in sessions:
        clocks[s["session_id"]] = mp4_presentation_times(s["video"])[0]
    print("Running label-free synthetic controls before real delay interpretation", flush=True)
    simulations, simulation_summary = run_synthetic_controls(out, config, clocks)
    quality, delays, acquisition, camera, failures = [], [], [], [], []
    paths = [(out / s["session_id"] / "native_rgb.npz", s) for s in sessions]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=workers) as pool:
            jobs = {pool.submit(analyze_cached_recording, path, s, config): s for path, s in paths if path.is_file()}
            for job in as_completed(jobs):
                # Cache files contain full typed rows; retain exceptions rather
                # than silently treating a failed recording as absent data.
                try:
                    job.result()
                    print(f"Analysis complete: {jobs[job]['session_id']}", flush=True)
                except Exception as exc:
                    failures.append(dict(session_id=jobs[job]["session_id"], reason="analysis_failed", detail=str(exc)))
    for session in sessions:
        path = out / session["session_id"] / "native_rgb.npz"
        if not path.is_file():
            failures.append(dict(session_id=session["session_id"], reason="native_rgb_missing", path=str(path)))
            continue
        if any(f["session_id"] == session["session_id"] for f in failures):
            continue
        print(f"Analyzing {session['session_id']} on original PTS", flush=True)
        q, d, a, c = analyze_cached_recording(path, session, config)
        quality.extend(q)
        delays.extend(d)
        acquisition.extend(a)
        camera.append(c)
        if frozen is not None:
            from .pilot_frozen import evaluate_native
            from .frozen_timebase import TimebaseConfig
            # The original verified native facial cache and accepted external
            # adapter feed the original API; hand data are never passed here.
            source = json.loads((path.parent / "acquisition.json").read_text())["face_cache_path"]
            with np.load(source, allow_pickle=False) as z:
                native = {k: z[k].copy() for k in z.files}
            frozen_out = path.parent / "frozen_face"
            frozen_out.mkdir(exist_ok=True)
            adapter = TimebaseConfig(**json.loads(Path("configs/pilot6_frozen_timebase_v2.json").read_text(encoding="utf-8")))
            evaluate_native(native, frozen, frozen_out, adapter, session["session_id"])
    associations = cuff_associations(sessions, intervals, delays, config)
    sensitivity = []
    for policy in config["policies"]:
        for selection in ("strict", "intermediate", "all_computable"):
            for distance in (15., 30., 60.):
                records = cuff_associations(sessions, intervals, delays, config, policy, False, selection, distance)
                for row in records:
                    row.update(max_distance_sec=distance, independent_record=False, analysis_role="repeated-label sensitivity; never extra observations")
                sensitivity.extend(records)
    stability = stability_summary(delays)
    bp = bp_descriptive(associations, sensitivity)
    for name, rows in (("waveform_quality", quality), ("delay_windows", delays), ("roi_validity", acquisition),
                       ("camera_timebase_audit", camera), ("delay_stability", stability),
                       ("cuff_associations", associations), ("cuff_associations_sensitivity", sensitivity),
                       ("bp_exploratory_analysis", bp), ("analysis_failures", failures)):
        write_csv(out / (name + ".csv"), rows)
    from .pilot6_delay_reporting import render_report
    render_report(out, sessions, config, quality, delays, acquisition, camera, simulations, simulation_summary,
                  stability, associations, sensitivity, bp, failures)
