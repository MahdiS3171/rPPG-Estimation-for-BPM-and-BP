"""Signal-only diagnostics and cautious subject-grouped exploratory BP baselines."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.signal import correlate, correlation_lags, welch
from scipy.integrate import trapezoid

from .artifacts import write_json
from .bp import estimate_delay
from .bp_models import BPModel
from .processing import shared_grid, resample_trace, extract_signal, zero_phase_bandpass, valid_runs
from .quality import signal_quality
from .waveform_visualization import coarse_feature_diagnostics, normalized_welch, reliable_beats, write_csv

METHODS = ("GREEN", "CHROM", "POS")
BP_FEATURES = ("face_hr_bpm", "hand_hr_bpm", "face_snr_db", "hand_snr_db",
               "face_rise_time_sec", "hand_rise_time_sec", "face_width50_sec", "hand_width50_sec",
               "face_ibi_std_sec", "hand_ibi_std_sec", "delay_xcorr_sec")


def finite_stat(values, stat=np.median):
    x = np.asarray(values,dtype=float)
    return float(stat(x[np.isfinite(x)])) if np.isfinite(x).any() else None


def median_pulse_template(values, fs):
    """Trough-anchored real-time beats, no duration warping or per-beat fit."""
    detection = reliable_beats(values,fs)
    if not detection["reliable"]:
        return None
    beats = [detection["filtered"][a:b+1] for a,_,b in detection["anchors"]]
    samples = np.full((len(beats),max(map(len,beats))),np.nan)
    for index,beat in enumerate(beats):
        samples[index,:len(beat)] = beat
    template = np.nanmedian(samples,axis=0)
    template[np.isfinite(samples).sum(axis=0) < 3] = np.nan
    return template


def validate_timing(frame_rates=(30.,60.), seed=42):
    """Known analytical delays, both signs, noise, irregular common clocks.

    Numerical errors describe this simulation only, never physiological accuracy.
    """
    from .types import RGBTrace
    rng = np.random.default_rng(seed)
    rows = []
    for fs in frame_rates:
        for seconds in (5.,10.,30.):
            for irregular in (False,True):
                t = np.arange(round((seconds+4)*fs))/fs
                source = t + (0.001*np.sin(2*np.pi*.31*t) if irregular else 0)
                grid = shared_grid(source,fs)
                for delay in (-.12,-1/fs,0.,1/fs,.12):
                    def pulse(t):
                        return np.sin(2*np.pi*1.2*t)+.25*np.sin(2*np.pi*2.4*t+.2)
                    pair = []
                    for lag in (0.,delay):
                        x = pulse(source-lag)+rng.normal(0,.01,len(source))
                        trace = RGBTrace(source,np.repeat(x[:,None],3,axis=1),"synthetic",np.ones(len(x),bool))
                        pair.append(resample_trace(trace,grid,.15)[0][:,0])
                    cut = slice(round(2*fs),round((2+seconds)*fs))
                    raw = estimate_delay(*[v[cut] for v in pair],fs,.3,1.2)
                    filtered = [zero_phase_bandpass(x,fs,.7,4)[cut] for x in pair]
                    result = estimate_delay(*filtered,fs,.3,1.2)
                    rows.append(dict(fs=fs,window_sec=seconds,irregular=irregular,true_delay_sec=delay,
                                     estimated_delay_sec=result["delay_xcorr_sec"],
                                     phase_delay_sec=result["delay_phase_sec"],
                                     error_sec=result["delay_xcorr_sec"]-delay,
                                     raw_estimate_sec=raw["delay_xcorr_sec"],
                                     filter_delay_change_sec=result["delay_xcorr_sec"]-raw["delay_xcorr_sec"],
                                     acquisition_interval_sec=1/fs))
    return dict(seed=seed,positive_means="hand_lags_face",rows=rows,
                max_absolute_error_sec=max(abs(r["error_sec"]) for r in rows),
                filter="3rd order Butterworth SOS forward/backward 0.7-4 Hz; 2s simulation edge trim",
                pulse_frequency_hz=1.2,
                limit="5/10/30s synthetic windows at 72 bpm only. No validated physiological timing uncertainty; upsampling does not improve acquired resolution.")


def window_quality(values, trace, a, b, fs, config):
    source = (trace.timestamps >= a) & (trace.timestamps < b)
    valid = float(trace.validity_mask[source].mean()) if source.any() else 0.
    q = signal_quality(values,fs,config.hr_min_hz,config.hr_max_hz)
    q.update(valid_frame_fraction=valid,band_power=None,spectral_concentration=None,
             dominant_frequency_hz=q["hr_bpm"]/60, illumination_cv=None, motion_px_per_sec=None,
             pixel_count=None,mask_coverage=None,detector_confidence=None,clipping_fraction=None)
    for key,target in (("skin_pixel_count","pixel_count"),("mask_coverage","mask_coverage"),
                       ("motion_px_per_sec","motion_px_per_sec"),("detector_confidence","detector_confidence"),
                       ("clipping_fraction","clipping_fraction")):
        if key in trace.quality:
            q[target] = finite_stat(trace.quality[key][source])
    bright = np.asarray(trace.quality.get("brightness",np.full(len(trace.timestamps),np.nan)))[source]
    bright = bright[np.isfinite(bright)]
    if len(bright) and bright.mean() > 0:
        q["illumination_cv"] = float(bright.std()/bright.mean())
    invalid = []
    uncertain = []
    if valid < config.min_valid_fraction:
        invalid.append("insufficient_original_valid_frames")
    if not np.isfinite(values).all():
        invalid.append("missing_signal_or_filter_edge")
    elif np.std(values) < 1e-10:
        invalid.append("flat_signal")
    else:
        f,p = welch(values-values.mean(),fs=fs,nperseg=min(len(values),round(8*fs)))
        band = (f >= .7) & (f <= 4.)
        peak = np.abs(f-q["dominant_frequency_hz"]) <= .15
        q["band_power"] = float(trapezoid(p[band],f[band]))
        q["spectral_concentration"] = float(p[peak].sum()/p[band].sum()) if p[band].sum() > 0 else None
    if not np.isfinite(q["snr_db"]) or q["snr_db"] < config.min_snr_db:
        uncertain.append("low_spectral_snr")
    if not np.isfinite(q["periodicity"]) or q["periodicity"] < .5:
        uncertain.append("weak_periodicity")
    if q["clipping_fraction"] is not None and q["clipping_fraction"] > .1:
        uncertain.append("intensity_clipping")
    if q["illumination_cv"] is not None and q["illumination_cv"] > .05:
        uncertain.append("illumination_variation")
    return dict(**q,status="invalid" if invalid else "uncertain" if uncertain else "good",
                reasons=invalid+uncertain)


def analyze_signals(traces, config, output, timing_verified=True):
    output = Path(output)
    fs = config.sample_rate
    grid = shared_grid(next(iter(traces.values())).timestamps,fs)
    uniform,signals,rows = {},{},[]
    for roi,trace in traces.items():
        uniform[roi],imputed = resample_trace(trace,grid,config.max_gap_sec)
        for method in METHODS:
            cfg = replace(config,method=method)
            signal = extract_signal(uniform[roi],grid,roi,cfg)
            signals[roi,method] = signal.values
            for seconds,hop in ((10.,5.),(30.,30.)):
                previous_hr = None
                previous_primary_template = None
                for start in range(0,len(grid)-round(seconds*fs)+1,round(hop*fs)):
                    stop = start+round(seconds*fs)
                    a,b = float(grid[start]),float(grid[start]+seconds)
                    q = window_quality(signal.values[start:stop],trace,a,b,fs,config)
                    feature = coarse_feature_diagnostics(signal.values[start:stop],fs)
                    feature_values = feature["values"]
                    # Quality-rejected shape values remain unavailable to BP baselines.
                    if q["status"] != "good":
                        feature_values = {key:np.nan for key in feature_values}
                    row = dict(roi=roi,method=method,start_sec=a,end_sec=b,window_sec=seconds,
                               primary_nonoverlapping=bool(round(start/fs) % round(seconds) == 0),
                               interpolated_grid_fraction=float(imputed[start:stop].mean()),
                               adjacent_hr_change_bpm=abs(q["hr_bpm"]-previous_hr) if previous_hr is not None else None,
                               **q,**{f"feature_{k}":v for k,v in feature_values.items()},
                               feature_reasons=feature["reasons"],reliable_beats=feature["reliable_beats"])
                    row["adjacent_primary_template_correlation"] = None
                    row["template_repeatability_reason"] = "not_primary_nonoverlapping"
                    if row["primary_nonoverlapping"]:
                        template = median_pulse_template(signal.values[start:stop],fs) if q["status"] == "good" else None
                        row["template_repeatability_reason"] = "no_adjacent_reliable_primary_template"
                        if template is not None and previous_primary_template is not None:
                            n = min(len(template),len(previous_primary_template))
                            valid = np.isfinite(template[:n])&np.isfinite(previous_primary_template[:n])
                            x,y = template[:n][valid],previous_primary_template[:n][valid]
                            if len(x) >= 3 and min(x.std(),y.std()) > 1e-8:
                                row["adjacent_primary_template_correlation"] = float(np.corrcoef(x,y)[0,1])
                                row["template_repeatability_reason"] = "descriptive_trough_anchored_real_time_no_warp"
                        # A failed intervening primary interval breaks adjacency.
                        previous_primary_template = template
                    rows.append(row)
                    previous_hr = q["hr_bpm"] if np.isfinite(q["hr_bpm"]) else None
    # Method agreement is descriptive, independent of physiological references.
    agreement = []
    for roi in traces:
        subset = [r for r in rows if r["roi"] == roi and r["window_sec"] == 10]
        for a in sorted({r["start_sec"] for r in subset}):
            same = [r for r in subset if r["start_sec"] == a]
            h = [r["hr_bpm"] for r in same if r["status"] != "invalid" and np.isfinite(r["hr_bpm"])]
            record = dict(roi=roi,start_sec=a,available_methods=len(h),
                          hr_range_bpm=max(h)-min(h) if len(h) >= 2 else None,
                          interpretation="shared_RGB_algorithms_agreement_not_independent_sensors_or_accuracy")
            start = int(np.argmin(abs(grid-a)))
            for first,second in (("GREEN","CHROM"),("GREEN","POS"),("CHROM","POS")):
                x,y = (signals[roi,m][start:start+round(10*fs)] for m in (first,second))
                record[first+"_"+second+"_signed_waveform_r"] = float(np.corrcoef(x,y)[0,1]) if np.isfinite(x).all() and np.isfinite(y).all() and min(x.std(),y.std()) > 1e-8 else None
            agreement.append(record)
    # Matched paired segment boundaries for every ROI/method timing ablation.
    delay_rows = []
    for face in ("forehead","left_cheek","right_cheek","combined","cheeks"):
        for hand in ("hand_palm","hand_center"):
            joint = np.isfinite(uniform[face]).all(axis=1)&np.isfinite(uniform[hand]).all(axis=1)
            for method in METHODS:
                cfg = replace(config,method=method)
                pair = [extract_signal(uniform[r],grid,r,cfg,allowed=joint).values for r in (face,hand)]
                for start in range(0,len(grid)-round(10*fs)+1,round(5*fs)):
                    stop = start+round(10*fs)
                    a,b = float(grid[start]),float(grid[start]+10)
                    x,y = [v[start:stop] for v in pair]
                    qualities = [window_quality(v,traces[r],a,b,fs,config) for v,r in zip((x,y),(face,hand))]
                    reasons = [r+":"+s for r,q in zip((face,hand),qualities) for s in q["reasons"]]
                    dt = float(np.median(np.diff(traces[face].timestamps)))
                    local_times = traces[face].timestamps[(traces[face].timestamps >= a)&(traces[face].timestamps < b)]
                    local_max_interval = float(np.max(np.diff(local_times))) if len(local_times) > 1 else dt
                    if local_max_interval > 1.5*dt:
                        reasons.append("long_acquisition_interval_in_window")
                    row = dict(face_roi=face,hand_roi=hand,method=method,start_sec=a,end_sec=b,
                               primary_nonoverlapping=bool(round(start/fs) % 10 == 0),
                               status="invalid",reasons=reasons,acquisition_interval_sec=dt,
                               maximum_acquisition_interval_sec=local_max_interval,
                               delay_uncertainty_floor_sec=max(dt,local_max_interval),delay_xcorr_sec=None,
                               interpretation="experimental_optical_inter_site_delay_not_PTT",
                               positive_means="hand_lags_face",physiological_interpretation_validated=False)
                    if all(np.isfinite(v).all() for v in (x,y)):
                        hr = qualities[0]["hr_bpm"]/60
                        d = estimate_delay(x,y,fs,config.max_delay_sec,hr)
                        row.update(d)
                        if not np.isfinite(d["delay_xcorr_sec"]):
                            reasons.append("unresolved_or_boundary_delay")
                        if not np.isfinite(d["delay_xcorr_score"]) or d["delay_xcorr_score"] < config.min_delay_correlation:
                            reasons.append("weak_correlation")
                        if abs(qualities[0]["hr_bpm"]-qualities[1]["hr_bpm"]) > config.max_hr_difference_bpm:
                            reasons.append("inconsistent_site_hr")
                        halves = [estimate_delay(x[s:s+round(5*fs)],y[s:s+round(5*fs)],fs,config.max_delay_sec,hr)["delay_xcorr_sec"]
                                  for s in (0,round(5*fs))]
                        disagreement = abs(halves[0]-halves[1]) if np.isfinite(halves).all() else np.nan
                        row["half_window_delay_disagreement_sec"] = disagreement
                        uncertainty = max(dt,local_max_interval,disagreement/2 if np.isfinite(disagreement) else dt,
                                          abs(d["delay_phase_sec"]-d["delay_xcorr_sec"]) if np.isfinite(d["delay_xcorr_sec"]) else dt)
                        row["delay_uncertainty_floor_sec"] = uncertainty
                        if not np.isfinite(disagreement) or disagreement > 2*dt:
                            reasons.append("unstable_half_window_delay")
                        if np.isfinite(d["delay_xcorr_sec"]) and abs(d["delay_xcorr_sec"]) <= uncertainty:
                            reasons.append("delay_comparable_to_timing_uncertainty")
                        if not timing_verified:
                            reasons.append("unverified_decoded_clock")
                        row["status"] = "candidate" if not reasons else "uncertain" if all(q["status"] != "invalid" for q in qualities) else "invalid"
                    delay_rows.append(row)
    # Cross-method agreement gate on identically located ROIs/windows.
    groups = {}
    for row in delay_rows:
        groups.setdefault((row["face_roi"],row["hand_roi"],row["start_sec"]),[]).append(row)
    for group in groups.values():
        eligible = [r for r in group if r["status"] != "invalid" and r["delay_xcorr_sec"] is not None and np.isfinite(r["delay_xcorr_sec"])]
        spread = np.ptp([r["delay_xcorr_sec"] for r in eligible]) if len(eligible) >= 2 else np.nan
        for r in group:
            r["cross_method_delay_range_sec"] = float(spread)
            if len(eligible) < 2 or spread > r["acquisition_interval_sec"]:
                r["reasons"].append("cross_method_delay_unconfirmed")
                if r["status"] == "candidate":
                    r["status"] = "uncertain"
    arrays = dict(timestamps=grid)
    arrays.update({r+"_rgb_uniform":v for r,v in uniform.items()})
    arrays.update({r+"__"+m:v for (r,m),v in signals.items()})
    np.savez_compressed(output/"classical_waveforms.npz",**arrays)
    write_csv(output/"quality_hr_morphology.csv",rows)
    write_csv(output/"method_agreement.csv",agreement)
    write_csv(output/"delay_feasibility.csv",delay_rows)
    delay_summary = []
    for face,hand,method in sorted({(r["face_roi"],r["hand_roi"],r["method"]) for r in delay_rows}):
        selected = [r for r in delay_rows if (r["face_roi"],r["hand_roi"],r["method"]) == (face,hand,method)]
        primary = [r for r in selected if r["primary_nonoverlapping"]]
        candidates = [r for r in primary if r["status"] == "candidate"]
        numeric = [r for r in primary if r["delay_xcorr_sec"] is not None and np.isfinite(r["delay_xcorr_sec"])]
        delay_summary.append(dict(face_roi=face,hand_roi=hand,method=method,attempted_windows=len(selected),
                                  statuses={status:sum(r["status"] == status for r in selected) for status in ("candidate","uncertain","invalid")},
                                  primary_attempted=len(primary),primary_candidate_count=len(candidates),
                                  primary_candidate_median_sec=finite_stat([r["delay_xcorr_sec"] for r in candidates]),
                                  primary_candidate_std_sec=finite_stat([r["delay_xcorr_sec"] for r in candidates],np.std),
                                  primary_numeric_median_sec=finite_stat([r["delay_xcorr_sec"] for r in numeric]),
                                  primary_numeric_std_sec=finite_stat([r["delay_xcorr_sec"] for r in numeric],np.std),
                                  primary_uncertainty_floor_median_sec=finite_stat([r["delay_uncertainty_floor_sec"] for r in primary]),
                                  physiological_timing_validated=False))
    write_json(output/"delay_summary.json",delay_summary)
    summary = []
    for roi,trace in traces.items():
        for method in METHODS:
            selected = [r for r in rows if r["roi"] == roi and r["method"] == method and r["window_sec"] == 10]
            good = [r for r in selected if r["status"] == "good" and r["primary_nonoverlapping"]]
            summary.append(dict(roi=roi,method=method,valid_frame_fraction=float(trace.validity_mask.mean()),
                                windows=len(selected),good=sum(r["status"] == "good" for r in selected),
                                uncertain=sum(r["status"] == "uncertain" for r in selected),
                                invalid=sum(r["status"] == "invalid" for r in selected),
                                primary_good_windows=len(good),primary_hr_median_bpm=finite_stat([r["hr_bpm"] for r in good]),
                                primary_template_pair_count=sum(r["adjacent_primary_template_correlation"] is not None for r in good),
                                primary_template_repeatability_median_r=finite_stat([r["adjacent_primary_template_correlation"] for r in good]),
                                primary_hr_mad_bpm=finite_stat([abs(r["hr_bpm"]-np.median([g["hr_bpm"] for g in good])) for r in good]),
                                tracking_missing_intervals_sec=[(float(trace.timestamps[a]),float(trace.timestamps[b-1]))
                                                                for a,b in valid_runs(~trace.validity_mask)]))
    write_json(output/"signal_summary.json",summary)
    render_diagnostics(grid,signals,rows,delay_rows,output,fs)
    return rows,delay_rows,summary


def render_diagnostics(grid,signals,rows,delays,output,fs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(4,1,figsize=(12,13),constrained_layout=True)
    for roi in ("combined","hand_palm"):
        x = signals[roi,"GREEN"]
        sd = np.nanstd(x)
        axes[0].plot(grid,x/sd if sd > 0 else x,label=roi,lw=.6)
        runs = valid_runs(np.isfinite(x))
        if runs:
            a,b = max(runs,key=lambda x:x[1]-x[0])
            psd = normalized_welch(x[a:b],fs)
            axes[1].plot(psd["frequency_hz"],psd["density"],label=roi)
        for status,color in (("good","green"),("uncertain","orange"),("invalid","red")):
            selected = [r for r in rows if r["roi"] == roi and r["method"] == "GREEN" and r["window_sec"] == 10 and r["status"] == status]
            axes[2].scatter([r["start_sec"]+5 for r in selected],[0 if roi == "combined" else 1]*len(selected),color=color,s=15,label=roi+" "+status)
    selected = [r for r in delays if r["face_roi"] == "combined" and r["hand_roi"] == "hand_palm"]
    for method in METHODS:
        records = [r for r in selected if r["method"] == method]
        axes[3].plot([r["start_sec"]+5 for r in records],[r["delay_xcorr_sec"] if r["delay_xcorr_sec"] is not None else np.nan for r in records],".-",label=method,lw=.7)
    supported = [r for r in selected if r["method"] == "GREEN" and r["primary_nonoverlapping"] and
                 r["delay_xcorr_sec"] is not None and np.isfinite(r["delay_xcorr_sec"])]
    if supported:
        axes[3].errorbar([r["start_sec"]+5 for r in supported],[r["delay_xcorr_sec"] for r in supported],
                         yerr=[r["delay_uncertainty_floor_sec"] for r in supported],fmt="none",color="grey",alpha=.35,label="GREEN timing floor (not CI)")
    axes[0].set(title="Synchronized classical GREEN (algorithm SD units; no contact reference)",xlabel="Recording time (s)",ylabel="Relative waveform")
    axes[1].set(title="Longest contiguous usable segment PSD",xlabel="Frequency (Hz)",ylabel="Normalized density",xlim=(.5,5))
    axes[2].set(title="Engineering quality flags; overlapping 10s windows",xlabel="Recording time (s)",yticks=[0,1],yticklabels=["face","hand"])
    axes[3].set(title="Optical delay diagnostics; physiological timing unvalidated",xlabel="Recording time (s)",ylabel="Hand lag (s)")
    for ax in axes:
        ax.legend(fontsize=7,ncol=3); ax.grid(alpha=.2)
    fig.savefig(output/"interpretability.png",dpi=130)
    plt.close(fig)
    # An inspectable xcorr curve on the earliest complete paired interval.
    face,hand = signals["combined","GREEN"],signals["hand_palm","GREEN"]
    runs = [(a,b) for a,b in valid_runs(np.isfinite(face)&np.isfinite(hand)) if b-a >= 10*fs]
    fig,ax = plt.subplots(figsize=(9,3),constrained_layout=True)
    if runs:
        a,b = runs[0]; b = a+round(10*fs)
        x,y = face[a:b]-face[a:b].mean(),hand[a:b]-hand[a:b].mean()
        c = correlate(y,x,method="fft")/(np.linalg.norm(x)*np.linalg.norm(y))
        l = correlation_lags(len(y),len(x))/fs
        ax.plot(l,c); ax.set_xlim(-.3,.3)
    else:
        ax.text(.5,.5,"No complete paired 10s interval",ha="center",transform=ax.transAxes)
    ax.set(title="Earliest complete GREEN interval; no quality-based example selection",xlabel="Hand lag (s)",ylabel="Signed normalized correlation")
    fig.savefig(output/"delay_correlation.png",dpi=130); plt.close(fig)


def exploratory_bp(rows, seed=42):
    """Session aggregates only, leave-one-subject-out, train-only transforms.

    Fixed ridge alpha=10; no feature/hyperparameter selection from held-out data.
    At most one row per independent cuff association (caller aggregates windows).
    """
    eligible = [r for r in rows if r.get("association_eligible") and r.get("observed_cuff_sbp_mmHg") is not None
                and r.get("observed_cuff_dbp_mmHg") is not None]
    keys = [(r["subject_id"],r["source_sheet"],r["source_row"]) for r in eligible]
    if len(keys) != len(set(keys)):
        raise ValueError("A cuff reference must not be duplicated across window samples")
    subjects = sorted({r["subject_id"] for r in eligible})
    if len(subjects) < 3:
        return dict(status="not_evaluated",reason="Need at least three subjects with documented cuff associations and usable session features",
                    eligible_subjects=len(subjects),bp_accuracy_validated=False,predictions=[])
    predictions = []
    feature_sets = {"hr_quality":BP_FEATURES[:4],"hr_quality_morphology":BP_FEATURES[:-1],"hr_quality_morphology_delay":BP_FEATURES}
    for heldout in subjects:
        train = [r for r in eligible if r["subject_id"] != heldout]
        test = [r for r in eligible if r["subject_id"] == heldout]
        if {r["subject_id"] for r in train}&{heldout}:
            raise ValueError("Subject leakage")
        y = np.array([[r["observed_cuff_sbp_mmHg"],r["observed_cuff_dbp_mmHg"]] for r in train])
        # Each training subject receives equal total mass through subject averaging.
        train_aggregates = []
        for sid in sorted({r["subject_id"] for r in train}):
            source = [r for r in train if r["subject_id"] == sid]
            train_aggregates.append({key:finite_stat([r.get(key,np.nan) for r in source]) for key in BP_FEATURES} |
                                    dict(subject_id=sid,observed_cuff_sbp_mmHg=np.mean([r["observed_cuff_sbp_mmHg"] for r in source]),
                                         observed_cuff_dbp_mmHg=np.mean([r["observed_cuff_dbp_mmHg"] for r in source])))
        train = train_aggregates
        y = np.array([[r["observed_cuff_sbp_mmHg"],r["observed_cuff_dbp_mmHg"]] for r in train])
        for ablation,features in feature_sets.items():
            def number(r,k):
                return np.nan if r.get(k) is None else r[k]
            x = np.array([[number(r,k) for k in features] for r in train],float)
            keep = np.isfinite(x).any(axis=0)
            if not keep.any():
                continue
            used = [k for k,v in zip(features,keep) if v]
            for kind in ("mean","ridge"):
                model = BPModel(kind,seed).fit(x[:,keep],y,feature_keys=used,subject_ids=[r["subject_id"] for r in train])
                pred = model.predict(np.array([[number(r,k) for k in used] for r in test],float))
                for r,p in zip(test,pred):
                    predictions.append(dict(subject_id=heldout,recording_id=r["recording_id"],model=kind,ablation=ablation,
                                            observed_cuff_sbp_mmHg=r["observed_cuff_sbp_mmHg"],observed_cuff_dbp_mmHg=r["observed_cuff_dbp_mmHg"],
                                            estimated_sbp_mmHg=float(p[0]),estimated_dbp_mmHg=float(p[1]),
                                            feature_keys=used,dropped_train_empty_features=[k for k in features if k not in used],
                                            prediction_kind="exploratory_subject_held_out_estimate",train_subjects=[r["subject_id"] for r in train]))
    metrics = []
    for ablation in feature_sets:
        for kind in ("mean","ridge"):
            selected = [r for r in predictions if r["model"] == kind and r["ablation"] == ablation]
            if not selected:
                continue
            metric = dict(ablation=ablation,model=kind,subjects=len({r["subject_id"] for r in selected}),observations=len(selected))
            for target in ("sbp","dbp"):
                subject_errors = [[r["estimated_"+target+"_mmHg"]-r["observed_cuff_"+target+"_mmHg"] for r in selected if r["subject_id"] == sid]
                                  for sid in sorted({r["subject_id"] for r in selected})]
                metric[target+"_subject_balanced_mae_mmHg"] = float(np.mean([np.mean(np.abs(e)) for e in subject_errors]))
                metric[target+"_subject_balanced_rmse_mmHg"] = float(np.sqrt(np.mean([np.mean(np.square(e)) for e in subject_errors])))
                metric[target+"_subject_balanced_bias_mmHg"] = float(np.mean([np.mean(e) for e in subject_errors]))
            metrics.append(metric)
    return dict(status="exploratory_only",eligible_subjects=len(subjects),bp_accuracy_validated=False,metrics=metrics,
                protocol="Leave-one-subject-out; aggregate training subjects; train-only imputation/scaling; ridge alpha=10; no tuning",
                predictions=predictions)
