"""Bounded, data-free acquisition-confounding experiment; no model training."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from rppg_lab.waveform_v2_timing_contract import estimate_signed_lag, synthetic_pulse

RATES = (30, 60, 120, 240)
CASES = ('physical_8ms', 'equivalent_with_scan', 'delay_cancelled_by_scan',
         'scan_only', 'different_morphology', 'illumination_only')
REPEATS = 32
NOISE_LEVELS = (0.0, 0.15)


def exposed_pulse(t, *, fps, physical_delay_ms=0.0, row_skew_ms=0.0, morphology=0.0):
    """Box exposure about each row's midpoint, without resampling observed traces."""
    offsets = ((np.arange(9) + 0.5) / 9 - 0.5) * (0.25 / fps)
    return synthetic_pulse(np.asarray(t)[:, None] + offsets + row_skew_ms / 1000,
                           delay_ms=physical_delay_ms, morphology=morphology).mean(axis=1)


def scenario(fps, case):
    t = np.arange(12 * fps, dtype=float) / fps
    skew = 0.36 * 1000 / fps
    physical, scan, morphology = {
        'physical_8ms': (8.0, 0.0, 0.0),
        'equivalent_with_scan': (8.0 + skew, skew, 0.0),
        'delay_cancelled_by_scan': (skew, skew, 0.0),
        'scan_only': (0.0, skew, 0.0),
        'different_morphology': (8.0, 0.0, 1.0),
        'illumination_only': (None, 0.0, 0.0),
    }[case]
    face = exposed_pulse(t, fps=fps)
    hand = exposed_pulse(t, fps=fps, physical_delay_ms=physical or 0.0,
                         row_skew_ms=scan, morphology=morphology)
    # The identical periodic intensity traces in this negative control represent
    # modulated illumination, with NO physiological component.
    apparent = physical - scan if physical is not None and morphology == 0 else None
    return t, face, hand, physical, scan, apparent


def controls():
    output = []
    for fps in RATES:
        t, face, hand, *_ = scenario(fps, 'physical_8ms')
        other = scenario(fps, 'equivalent_with_scan')[2]
        difference = float(np.max(np.abs(hand - other)))
        if difference >= 5e-12:
            raise AssertionError('Equivalent acquisition worlds differ')
        for sign in (-1, 1):
            delay = sign * 2 * 1000 / fps
            shifted = exposed_pulse(t, fps=fps, physical_delay_ms=delay)
            lag = estimate_signed_lag(t, face, shifted)
            if lag.delay_ms is None or abs(lag.delay_ms - delay) >= 2:
                raise AssertionError('Clean signed-delay control failed')
        output.append(dict(fps=fps, equivalent_max_abs_difference=difference,
                           signed_grid_delay_controls='PASS'))
    return output


def run():
    checks = controls()
    rows = []
    summaries = []
    for fps in RATES:
        for case in CASES:
            t, face, hand, physical, skew, apparent = scenario(fps, case)
            for noise in NOISE_LEVELS:
                group = []
                for repeat in range(REPEATS):
                    # Same noise draws across cases preserve the paired-world
                    # comparison; across FPS these are not paired real recordings.
                    rng = np.random.default_rng(20261108 + fps * 1000 + repeat)
                    sigma = noise * np.std(face)
                    x = face + rng.normal(0, sigma, len(t))
                    y = hand + rng.normal(0, sigma, len(t))
                    lag = estimate_signed_lag(t, x, y)
                    value = lag.delay_ms
                    row = dict(fps=fps, case=case, noise_sd_relative=noise, seed_repeat=repeat,
                        physical_delay_ms=physical, row_skew_ms=skew,
                        matched_shape_apparent_delay_ms=apparent, estimate_ms=value,
                        physical_error_ms=value-physical if value is not None and physical is not None else None,
                        apparent_error_ms=value-apparent if value is not None and apparent is not None else None,
                        status=lag.status, reason=lag.reason, peak_correlation=lag.peak_correlation,
                        uncertainty_floor_ms=lag.uncertainty_floor_ms)
                    rows.append(row)
                    group.append(row)
                finite = [r for r in group if r['estimate_ms'] is not None]
                summary = dict(fps=fps, case=case, noise_sd_relative=noise, attempts=len(group),
                    finite_diagnostics=len(finite), abstentions=dict(Counter(
                        r['reason'] for r in group if r['estimate_ms'] is None)),
                    physical_delay_ms=physical, row_skew_ms=skew,
                    matched_shape_apparent_delay_ms=apparent,
                    median_estimate_ms=float(np.median([r['estimate_ms'] for r in finite])) if finite else None)
                for name in ('physical', 'apparent'):
                    errors = [r[name+'_error_ms'] for r in finite if r[name+'_error_ms'] is not None]
                    summary[name+'_bias_ms'] = float(np.mean(errors)) if errors else None
                    summary[name+'_p95_absolute_error_ms'] = float(np.percentile(np.abs(errors), 95)) if errors else None
                summaries.append(summary)
    if len(rows) != 1536:
        raise AssertionError('Intended denominator changed')
    return checks, rows, summaries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    # Only create output under the dedicated namespace, never a raw data tree.
    root = Path(__file__).resolve().parents[1]
    out = args.out.absolute()
    allowed = root / 'outputs' / 'camera_timing_identifiability'
    if allowed.resolve() not in out.resolve().parents:
        parser.error('Use a new directory inside outputs/camera_timing_identifiability/')
    if out.exists():
        parser.error('Output already exists; evidence cannot be overwritten')
    checks, rows, summaries = run()
    sources = ('scripts/audit_camera_timing_identifiability.py',
               'rppg_lab/waveform_v2_timing_contract.py',
               'docs/CAMERA_TIMING_IDENTIFIABILITY_PROTOCOL.md')
    report = dict(schema='camera_timing_identifiability_v1', evidence_type='SYNTHETIC_ACQUISITION_MODEL',
        git_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        git_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True)),
        numpy_version=np.__version__, source_sha256={p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sources},
        independent_real_recordings=0, training=False, controls=checks, intended_trials=len(rows),
        no_noise_repeats_are_duplicates=True, summaries=summaries)
    out.mkdir(parents=True, exist_ok=False)
    (out/'results.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    with (out/'trials.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(dict(output=str(out), controls=checks, intended_trials=len(rows)), indent=2))


if __name__ == '__main__':
    main()
