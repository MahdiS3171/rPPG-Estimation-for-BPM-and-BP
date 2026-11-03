"""Isolate jitter and single-frame losses from the large-gap failure control."""
from __future__ import annotations

from dataclasses import replace
import numpy as np

from .classical import METHOD_FUNCS
from .config import PipelineConfig
from .pilot6_delay_validation import pulse, timing_diagnostics
from .processing import resample_trace, shared_grid, extract_signal
from .quality import signal_quality
from .types import RGBTrace


def isolated_clock_controls(config):
    rng = np.random.default_rng(config["seed"] + 1)
    rows = []
    for fs in (30., 60.):
        cfg = PipelineConfig(sample_rate=fs, max_gap_sec=config["max_gap_sec"], edge_guard_sec=config["filter_edge_sec"],
                             filter_low_hz=config["filter_low_hz"], filter_high_hz=config["filter_high_hz"])
        for hr in config["synthetic_hr_bpm"]:
            for ms in config["synthetic_delays_ms"]:
                for scenario in ("isolated_jitter", "single_frame_losses"):
                    for replicate in range(config["synthetic_replicates"]):
                        t = np.arange(0., 18., 1/fs)
                        if scenario == "isolated_jitter":
                            t += rng.uniform(-.0015, .0015, len(t))
                        else:
                            # Isolated single losses only, no consecutive losses.
                            missing = rng.random(len(t)) < .01
                            missing[1:] &= ~missing[:-1]
                            t = t[~missing]
                        grid = shared_grid(t, fs)
                        rgb = []
                        for site, delay in enumerate((0., ms/1000)):
                            p = pulse(t-delay, hr)
                            nuisance = .0015*np.sin(2*np.pi*3.8*(t-delay)+.3)
                            colors = np.array([140., 120., 90.])*(1+p[:, None]*np.array([.006, .012, .003])
                                     + nuisance[:, None]*np.array([1., .2, -1.]))
                            trace = RGBTrace(t, colors*(1.6 if site else .8), "clock_control", np.ones(len(t), bool))
                            rgb.append(resample_trace(trace, grid, config["max_gap_sec"])[0])
                        joint = np.isfinite(rgb[0]).all(axis=1) & np.isfinite(rgb[1]).all(axis=1)
                        for method in ("GREEN", "CHROM", "POS"):
                            waves = [extract_signal(c, grid, "clock_control", replace(cfg, method=method), allowed=joint).values for c in rgb]
                            keep = (grid >= 4) & (grid < 14)
                            x, y = [w[keep] for w in waves]
                            row = dict(clock="isolated_controls", scenario=scenario, replicate=replicate, fs=fs, hr_bpm=hr,
                                       true_delay_ms=ms, method=method, filtered=True, source_max_gap_sec=float(np.max(np.diff(t))),
                                       recovery_error_limit_ms=1000/fs, estimator_hr_source="spectral_face_waveform_not_oracle",
                                       error_ms=np.nan, phase_error_ms=np.nan, failure=True, noncomputable=True,
                                       failure_reason="gap_or_filter_support_not_bridged")
                            if np.isfinite(x).all() and np.isfinite(y).all():
                                measured_hr = signal_quality(x, fs)["hr_bpm"]
                                d = timing_diagnostics(x, y, fs, measured_hr/60, config["max_delay_sec"])
                                row.update(d, estimated_face_hr_bpm=measured_hr)
                                row["error_ms"] = 1000*d["delay_xcorr_sec"]-ms
                                row["phase_error_ms"] = 1000*d["delay_phase_sec"]-ms
                                row["noncomputable"] = not np.isfinite(d["delay_xcorr_sec"])
                                row["failure"] = row["noncomputable"] or abs(row["error_ms"]) > 1000/fs
                                row["failure_reason"] = d["failure_reason"] or ("recovery_error_exceeds_one_source_step" if row["failure"] else "")
                            rows.append(row)
    return rows
