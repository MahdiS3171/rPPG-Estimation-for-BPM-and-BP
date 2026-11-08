"""Gate 09 inference-only adapter; references cannot enter its prediction API."""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
from scipy.signal import welch
import torch

from .artifacts import file_sha256
from .waveform_inference import FrozenWaveformExtractor
from .waveform_metrics import forward_batch
from .waveform_v2_classical_study import (METHODS, CONDITIONS, array_sha256, usable,
    score_waveform, summarize_group, METRIC_KEYS, finite_mean, predict_recording,
    control_fixtures)
from .waveform_v2_timing_contract import (synthetic_pulse, audit_synthetic_timing_transfer,
                                         demonstration_cases)

CHECKPOINT = Path('checkpoints/waveform_v1_official_ubfc/waveform_v1_frozen.pt')
MANIFEST = CHECKPOINT.with_name('waveform_v1_freeze_manifest.json')
CHECKPOINT_SHA256 = 'd6b2d1eeebc0153197a2a12da736b957b32d72ae00b22631e9afd0b2097131ff'
MANIFEST_SHA256 = 'da5ba397b3be389deec8821f8ea16944b5f5b93e2834bb79c3c147eb3d1f89dc'
INPUT_KEYS = ('x', 'roi_valid', 'roi_quality', 'prior_valid')
ORDER = ('frozen_v1', *(f'{r}/{m}' for r in CONDITIONS for m in METHODS))


def load_official(checkpoint=CHECKPOINT, manifest=MANIFEST):
    """Pin bytes BEFORE deserialization; retain the existing strict V1 loader."""
    if file_sha256(checkpoint) != CHECKPOINT_SHA256 or file_sha256(manifest) != MANIFEST_SHA256:
        raise ValueError('Official checkpoint/manifest SHA256 mismatch; stop without replacement')
    extractor = FrozenWaveformExtractor(checkpoint, manifest, device='cpu')
    model = extractor._model
    if (type(model).__name__ != 'MultiROIPriorResidualWaveformNet'
            or extractor.roi_names != CONDITIONS[:3] or extractor.prior_names != METHODS
            or extractor.channel_names != ('RGB_R', 'RGB_G', 'RGB_B', *METHODS)
            or extractor.fs_target != 30 or extractor.window_length_samples != 300
            or model.training or any(p.device.type != 'cpu' for p in model.parameters())):
        raise ValueError('Official architecture/order/timebase/eval/CPU mismatch')
    return model, dict(extractor.metadata, freeze_manifest_sha256=MANIFEST_SHA256,
                       strict_state_dict=True, subject_adaptation_enabled=False)


def state_fingerprint(model):
    return {k: array_sha256(v.detach().cpu().numpy()) for k, v in model.state_dict().items()}


def validate_inputs(batch):
    shapes = dict(x=(1, 3, 9, 300), roi_valid=(1, 3), roi_quality=(1, 3), prior_valid=(1, 3, 6))
    for k, shape in shapes.items():
        value = np.asarray(batch[k])
        dtype = np.dtype(bool) if k in ('roi_valid', 'prior_valid') else np.dtype('float32')
        if value.shape != shape or value.dtype != dtype:
            raise ValueError('Input shape/dtype mismatch: '+k)
    if (not np.isfinite(batch['roi_quality']).all() or (batch['roi_quality'] < 0).any()
            or (batch['roi_quality'] > 1).any()
            or (batch['roi_valid'] & ~batch['prior_valid'].any(-1)).any()
            or (batch['prior_valid'] & ~batch['roi_valid'][..., None]).any()):
        raise ValueError('Input mask/quality inconsistency')
    valid_channels = np.concatenate((np.repeat(batch['roi_valid'][..., None], 3, -1),
                                    batch['prior_valid']), axis=-1)
    if not np.isfinite(batch['x'][valid_channels]).all():
        raise ValueError('Nonfinite valid channels')
    return {k: array_sha256(batch[k]) for k in INPUT_KEYS}


def infer_one(model, batch, *, flat_source=False):
    """One original-length output, no labels, gradients, preprocessing or adaptation."""
    fingerprints = validate_inputs(batch)
    result = dict(waveform=np.full(300, np.nan, np.float32), roi_attention=np.zeros(3),
        prior_weights=np.zeros((3, 6)), roi_ppg=np.full((3, 300), np.nan),
        fused_priors=np.full((3, 300), np.nan), scaled_residuals=np.full((3, 300), np.nan),
        computational_success=False, pulse_estimate_eligible=False,
        prediction_failure_reason='', input_fingerprints=fingerprints, forward_seconds=0.,
        forward_executed=False, physiological_validation='UNVALIDATED_SYNTHETIC_DIAGNOSTIC')
    if model.training or any(p.device.type != 'cpu' for p in model.parameters()):
        raise ValueError('Inference requires CPU eval model')
    if not batch['roi_valid'].any():
        result['prediction_failure_reason'] = 'NO_USABLE_ROI'
        return result
    tensors = {k: torch.from_numpy(np.ascontiguousarray(batch[k])) for k in INPUT_KEYS}
    started = time.perf_counter()
    result['forward_executed'] = True
    try:
        with torch.inference_mode():
            out = forward_batch(model, tensors, torch.device('cpu'))
            if torch.is_grad_enabled() or any(v.requires_grad for v in out.values() if isinstance(v, torch.Tensor)):
                raise RuntimeError('Gradient execution forbidden')
            if tuple(out['ppg'].shape) != (1, 300):
                result['prediction_failure_reason'] = 'INVALID_OUTPUT_SHAPE'
            else:
                for source, dest in (('ppg', 'waveform'), ('roi_attention', 'roi_attention'),
                        ('prior_weights', 'prior_weights'), ('roi_ppg', 'roi_ppg'),
                        ('fused_priors', 'fused_priors')):
                    result[dest] = out[source][0].cpu().numpy().copy()
                result['scaled_residuals'] = (model.residual_scale*out['residuals'][0]).cpu().numpy().copy()
                result['computational_success'] = bool(np.isfinite(result['waveform']).all())
                result['prediction_failure_reason'] = ('NONFINITE_OUTPUT' if not result['computational_success']
                    else 'FLAT_SOURCE_RGB' if flat_source else 'FLAT_OUTPUT' if not usable(result['waveform']) else '')
                result['pulse_estimate_eligible'] = not bool(result['prediction_failure_reason'])
    except (ValueError, RuntimeError) as error:
        result.update(prediction_failure_reason='FORWARD_FAILURE', forward_error=str(error))
    result['forward_seconds'] = time.perf_counter()-started
    if validate_inputs(batch) != fingerprints:
        raise RuntimeError('Inference mutated inputs/masks')
    if any(p.grad is not None for p in model.parameters()):
        raise RuntimeError('Inference created parameter gradients')
    return result


def harmonic_diagnostics(prediction, reference, metrics):
    p, r = metrics['pred_hr_bpm'], metrics['reference_hr_bpm']
    valid = bool(np.isfinite([p, r]).all())
    error = abs(p-r) if valid else np.nan
    twice = bool(valid and error > 6 and abs(p-2*r) <= 6)
    half = bool(valid and error > 6 and abs(p-.5*r) <= 6)
    out = dict(hr_ratio=p/r if valid and r else np.nan, suspected_double_peak=twice if valid else None,
        suspected_half_peak=half if valid else None,
        other_peak_discrepancy=bool(error > 6 and not twice and not half) if valid else None,
        prediction_band_edge=bool(min(abs(p-42), abs(p-210)) <= 6) if np.isfinite(p) else None,
        reference_band_edge=bool(min(abs(r-42), abs(r-210)) <= 6) if np.isfinite(r) else None)
    for name, signal, hr in (('reference', reference, r), ('prediction', prediction, p)):
        if not usable(signal) or not np.isfinite(hr):
            continue
        f, power = welch(np.asarray(signal, float), fs=30, nperseg=300, nfft=2048)
        den = power[(f >= .7) & (f <= 3.5)].sum()
        for label, factor in (('half', .5), ('selected', 1.), ('double', 2.)):
            center = hr/60*factor
            out[f'{name}_{label}_peak_power_fraction'] = (float(power[(abs(f-center) <= .1)
                & (f >= .7) & (f <= 3.5)].sum()/den) if den > 0 and .7 <= center <= 3.5 else np.nan)
    return out


def scored_row(filename, split, start, condition, signal, reference, *, eligible=True,
               computational=True, reason=''):
    metrics = score_waveform(signal, reference, prediction_eligible=eligible)
    metrics['alignment_gain_r'] = metrics['lag_aligned_r']-metrics['zero_lag_r']
    return dict(filename=filename, split=split, start=start, stop=start+300, condition=condition,
        prediction_sha256=array_sha256(signal), raw_reference_sha256=array_sha256(reference),
        computational_success=computational, pulse_estimate_eligible=eligible,
        prediction_failure_reason=reason, metrics=metrics,
        harmonic=harmonic_diagnostics(signal, reference, metrics))


def aggregate_conditions(rows):
    result = []
    for split in dict.fromkeys(r['split'] for r in rows):
        for condition in dict.fromkeys(r['condition'] for r in rows):
            group = [r for r in rows if (r['split'], r['condition']) == (split, condition)]
            if not group:
                continue
            records = {name: summarize_group([r for r in group if r['filename'] == name])
                       for name in dict.fromkeys(r['filename'] for r in group)}
            means = {k: finite_mean([v[k] for v in records.values()]) for k in METRIC_KEYS}
            result.append(dict(split=split, condition=condition, per_recording=records,
                pooled=summarize_group(group), recording_balanced=dict(**means,
                    hr_rmse_bpm=float(np.sqrt(means['hr_squared_error'])), intended_recordings=len(records),
                    finite_recordings={k: sum(np.isfinite(v[k]) for v in records.values()) for k in METRIC_KEYS}),
                harmonic_counts={k: sum(r['harmonic'].get(k) is True for r in group) for k in
                    ('suspected_double_peak', 'suspected_half_peak', 'other_peak_discrepancy', 'prediction_band_edge')},
                harmonic_evaluable_windows=sum(r['harmonic']['suspected_double_peak'] is not None for r in group)))
    return result


def paired_comparisons(rows):
    """Equal recording weights, metric-specific paired support, no winner selection."""
    lookup = {(r['split'], r['filename'], r['start'], r['condition']): r for r in rows}
    result = []
    for split in dict.fromkeys(r['split'] for r in rows):
        v1 = [r for r in rows if r['split'] == split and r['condition'] == 'frozen_v1']
        for condition in ORDER[1:]:
            item = dict(split=split, comparator=condition, intended_windows=len(v1),
                        intended_recordings=len({r['filename'] for r in v1}), metrics={})
            for metric in METRIC_KEYS:
                pairs = []
                for row in v1:
                    other = lookup.get((split, row['filename'], row['start'], condition))
                    a, b = row['metrics'][metric], other['metrics'][metric] if other else np.nan
                    if np.isfinite([a, b]).all():
                        pairs.append((row['filename'], a, b))
                records = {n: dict(v1=finite_mean([a for f, a, b in pairs if f == n]),
                                  classical=finite_mean([b for f, a, b in pairs if f == n]))
                           for n in dict.fromkeys(f for f, _, _ in pairs)}
                a, b = finite_mean([v['v1'] for v in records.values()]), finite_mean([v['classical'] for v in records.values()])
                item['metrics'][metric] = dict(v1=a, classical=b, v1_minus_classical=a-b,
                    paired_windows=len(pairs), paired_recordings=len(records), per_recording=records)
                if metric == 'hr_squared_error':
                    item['metrics']['hr_rmse_bpm'] = dict(v1=float(np.sqrt(a)), classical=float(np.sqrt(b)),
                        v1_minus_classical=float(np.sqrt(a)-np.sqrt(b)), paired_windows=len(pairs),
                        paired_recordings=len(records))
            result.append(item)
    return result


CONTROL_EXPECTATIONS = {
    'sinusoidal_positive': 'Report native recovery; no assumed learned polarity or tuned pass cutoff.',
    'constant_nonpulsatile': 'Pulse eligibility withheld regardless of a finite neural residual.',
    'slow_illumination_drift': 'Finite periodic output is an unvalidated false-pulse risk.',
    'independent_noise': 'No physiological success from periodicity or accidental correlation.',
    'coherent_pulse_band_illumination': 'Manufactured correlation exposes confounding, never physiology.',
    'clipped_impulses': 'Compare clean positive; no post-hoc robustness threshold.',
    'missing_windows': 'Zero allocation to missing ROI; all missing has no forward and retained failure.',
    'time_permuted': 'Pulse temporal structure destroyed; finite output is not physiological evidence.'}


def run_neural_controls(model):
    rows, outputs = [], {}
    for name, rgb, valid, reference in control_fixtures():
        prediction = predict_recording(rgb, valid)
        batch = prediction['batch']
        outputs[name] = []
        for w, start in enumerate((0, 300)):
            flat = bool(prediction['flat_input'][w, 3].all())
            out = infer_one(model, {k: batch[k][w:w+1].copy() for k in INPUT_KEYS}, flat_source=flat)
            row = scored_row(name, 'Control', start, 'frozen_v1', out['waveform'], reference[start:start+300],
                eligible=out['pulse_estimate_eligible'], computational=out['computational_success'],
                reason=out['prediction_failure_reason'])
            row.update(expectation=CONTROL_EXPECTATIONS[name], roi_attention=out['roi_attention'].tolist(),
                prior_weights=out['prior_weights'].tolist(), roi_valid=batch['roi_valid'][w].tolist(),
                forward_executed=out['forward_executed'], physiological_validation=out['physiological_validation'],
                output_std=float(np.std(out['waveform'])), output_finite=bool(np.isfinite(out['waveform']).all()))
            rows.append(row); outputs[name].append(out['waveform'])
    return rows, outputs


def neural_timing(model):
    t = np.arange(600)/30
    rgb = lambda delays: np.stack([np.array([160., 120., 100.])+synthetic_pulse(t, delay_ms=d)[:, None]
                                  *np.array([.8, 1.2, .5]) for d in delays])
    cases = (('baseline', [0, 0, 0], [45, 45, 45]),
             ('common_input_delay_100ms', [100, 100, 100], [145, 145, 145]),
             ('differential_second_stream_85ms', [0, 0, 0], [130, 130, 130]),
             ('heterogeneous_roi_phases', [0, 120, 120], [45, 165, 165]))
    reports = []
    for name, fd, hd in cases:
        batches = [predict_recording(rgb(d), np.ones((3, 600), bool))['batch'] for d in (fd, hd)]
        for w, start in enumerate((0, 300)):
            out = [infer_one(model, {k: b[k][w:w+1] for k in INPUT_KEYS})['waveform'] for b in batches]
            for anchor, source_delays, known in (('nominal_undelayed', (0, 45), 45),
                                                ('actual_first_roi', (fd[0], hd[0]), hd[0]-fd[0])):
                sources = [synthetic_pulse(t, delay_ms=d)[start:start+300] for d in source_delays]
                report = audit_synthetic_timing_transfer(t[start:start+300], source_face=sources[0],
                    source_hand=sources[1], reconstructed_face=out[0], reconstructed_hand=out[1],
                    known_source_delay_ms=known)
                reports.append(dict(case=name, start=start, anchor=anchor, roi_delays_stream1_ms=fd,
                    roi_delays_stream2_ms=hd, **report))
    return dict(neural=reports, analytical=demonstration_cases(30),
                interpretation='Independent synthetic windows only; no continuous reconstruction, PTT or BP.')
