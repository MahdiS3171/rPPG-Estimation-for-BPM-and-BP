"""Bounded CPU-only Gate 09; lock every actual prediction before cached references."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import importlib.metadata
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
import tracemalloc
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MPLBACKEND'] = 'Agg'
import numpy as np
import psutil
import torch

from rppg_lab.artifacts import file_sha256
from rppg_lab.waveform_v2_frozen_comparison import (CHECKPOINT, MANIFEST, INPUT_KEYS, ORDER,
    METHODS, CONDITIONS, load_official, state_fingerprint, infer_one, scored_row,
    aggregate_conditions, paired_comparisons, run_neural_controls, neural_timing)
from rppg_lab.waveform_v2_frozen_cache import (PARENT, PROTOCOL_COMMIT, PROTOCOL, GATE08,
    read_json, write_json, write_csv, load_inputs, load_references, validate_output, check_design,
    assert_hashes_unchanged)
from rppg_lab.waveform_v2_frozen_plots import make_plots
from rppg_lab.waveform_v2_classical_study import array_sha256
from rppg_lab.waveform_v2_scamps import source_sha256, _check_path

CODE = ('rppg_lab/waveform_v2_frozen_comparison.py', 'rppg_lab/waveform_v2_frozen_cache.py',
        'rppg_lab/waveform_v2_frozen_plots.py', 'scripts/audit_waveform_v2_frozen_comparison.py',
        'scripts/verify_waveform_v2_frozen_safe.py', 'tests/test_waveform_v2_frozen_comparison.py')


def install_subject_guard():
    counts = dict(blocked_io_attempts=0)
    def guard(event, args):
        if event not in ('open', 'os.listdir', 'os.scandir', 'os.remove', 'os.rename', 'os.mkdir', 'os.rmdir'):
            return
        for path in args[:2] if event == 'os.rename' else args[:1]:
            if not isinstance(path, (str, bytes, os.PathLike)):
                continue
            parts = Path(os.path.abspath(os.fsdecode(path))).parts
            if any(p.casefold() == 'dataset' or re.match(r'S(?:00[7-9]|01[01])(?:\D|$)', p.upper()) for p in parts):
                counts['blocked_io_attempts'] += 1
                raise PermissionError('Gate 09 forbids private/reserved subject I/O')
    sys.addaudithook(guard)
    return counts


def inference_guards(stack):
    for target in ('torch.Tensor.backward', 'torch.autograd.backward', 'torch.autograd.grad',
                   'torch.optim.Adam.step', 'torch.optim.AdamW.step', 'torch.optim.SGD.step',
                   'torch.Tensor.cuda', 'torch.nn.Module.cuda', 'torch.cuda._lazy_init'):
        stack.enter_context(patch(target, side_effect=AssertionError('Gate 09 inference only: no training/CUDA')))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gate08-root', type=Path, default=Path(read_json(GATE08/'local_artifact_manifest.json')['local_root']))
    parser.add_argument('--scamps-root', type=Path, default=Path('SCAMPSData'))
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    counts = install_subject_guard()
    try:
        output = validate_output(args.out_dir, args.scamps_root, args.gate08_root)
    except ValueError as error:
        parser.error(str(error))
    code_before = check_design(CODE)
    # Identity is checked before any model construction or data inference.
    model, identity = load_official()
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    state_before = state_fingerprint(model)
    artifact_names = ('locked_predictions.npz', 'reference_windows.npz', 'label_blind_predictions.json')
    old_hashes = {n: file_sha256(args.gate08_root/n) for n in artifact_names}
    records, classical_lookup = load_inputs(args.gate08_root)
    sources = {}
    for row in records:
        name = row['filename']
        if name not in sources:
            path = args.scamps_root/name; _check_path(path)
            observed = source_sha256(path)
            if observed != row['source_sha256']:
                raise ValueError('SCAMPS source identity mismatch: '+name)
            sources[name] = observed
    output.mkdir(parents=True, exist_ok=False)
    for name in ('LICENSE_SCAMPS.txt', 'ATTRIBUTION.md'):
        shutil.copyfile(Path('docs/waveform_v2/scamps')/name, output/name)
    started = time.perf_counter(); tracemalloc.start()
    outputs, blind_rows, archive = {}, [], {}
    with ExitStack() as stack:
        inference_guards(stack)
        for row in records:
            key = (row['filename'], row['start'])
            print(f'{key}: one frozen CPU waveform; reference not opened', flush=True)
            out = infer_one(model, row['batch'])
            outputs[key] = out
            stem = Path(row['filename']).stem+'_'+str(row['start'])
            for field in ('waveform', 'roi_attention', 'prior_weights', 'scaled_residuals', 'roi_ppg', 'fused_priors'):
                archive[stem+'_'+field] = out[field]
            archive[stem+'_indices'] = row['indices']
            for field in INPUT_KEYS:
                archive[stem+'_'+field] = row['batch'][field]
            blind_rows.append(dict(filename=row['filename'], split=row['split'], start=row['start'], stop=row['stop'],
                input_fingerprints=out['input_fingerprints'], prediction_sha256=array_sha256(out['waveform']),
                computational_success=out['computational_success'], pulse_estimate_eligible=out['pulse_estimate_eligible'],
                prediction_failure_reason=out['prediction_failure_reason'], forward_seconds=out['forward_seconds'],
                forward_executed=out['forward_executed'], roi_attention=out['roi_attention'], prior_weights=out['prior_weights']))
        np.savez_compressed(output/'locked_neural_predictions.npz', **archive)
        write_json(output/'label_blind_neural_predictions.json', dict(reference_values_read=False, windows=blind_rows))
        locked = {p: file_sha256(output/p) for p in ('locked_neural_predictions.npz', 'label_blind_neural_predictions.json')}
        write_json(output/'prediction_lock.json', dict(stage='BEFORE_REFERENCE_ARCHIVE_OPEN', hashes=locked,
            intended_windows=20, actual_forward_calls=sum(r['forward_executed'] for r in blind_rows),
            protocol_commit=PROTOCOL_COMMIT, implementation_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()))
        print('All 20 outputs locked. Opening the exact Gate 08 reference archive.', flush=True)
        refs, fixed = load_references(args.gate08_root, records)
        rows, components = [], []
        for split in ('Train', 'Val', 'Test'):
            for row in (r for r in records if r['split'] == split):
                key = (row['filename'], row['start']); raw = refs[key]; out = outputs[key]
                neural = scored_row(row['filename'], split, row['start'], 'frozen_v1', out['waveform'], raw,
                    eligible=out['pulse_estimate_eligible'], computational=out['computational_success'],
                    reason=out['prediction_failure_reason'])
                neural.update(roi_attention=out['roi_attention'].tolist(), prior_weights=out['prior_weights'].tolist(),
                    effective_prior_weights=(out['roi_attention'][:, None]*out['prior_weights']).tolist(),
                    scaled_residual_rms_by_roi=np.sqrt(np.mean(out['scaled_residuals']**2, axis=-1)).tolist())
                rows.append(neural)
                for ri, roi in enumerate(CONDITIONS):
                    for mi, method in enumerate(METHODS):
                        old = classical_lookup[(*key, roi, method)]
                        item = scored_row(row['filename'], split, row['start'], f'{roi}/{method}',
                            row['classical'][ri, mi], raw, eligible=old['pulse_estimate_eligible'],
                            computational=old['computational_success'], reason=old['prediction_failure_reason'])
                        frozen_metrics = fixed[(*key, roi, method)]
                        for metric, value in item['metrics'].items():
                            if metric == 'alignment_gain_r':
                                continue
                            original = frozen_metrics['metrics_'+metric]
                            if isinstance(value, (bool, np.bool_)):
                                equal = str(value) == original
                            else:
                                equal = np.isclose(value, float(original), rtol=1e-12, atol=1e-12, equal_nan=True)
                            if not equal:
                                raise ValueError('Gate 08 baseline metric changed: '+metric)
                        rows.append(item)
                        if ri < 3:
                            components.append(dict(filename=row['filename'], split=split, start=row['start'], roi=roi,
                                component=method, roi_attention=out['roi_attention'][ri], prior_weight=out['prior_weights'][ri, mi],
                                effective_weight=out['roi_attention'][ri]*out['prior_weights'][ri, mi],
                                metrics=item['metrics'], interpretation='DESCRIPTIVE_NOT_CAUSAL_ATTRIBUTION'))
                # Fixed internal mixtures are descriptive; none is used as a new selected comparator.
                for ri, roi in enumerate(CONDITIONS[:3]):
                    for name, signal in (('learned_prior_mixture', out['fused_priors'][ri]),
                                         ('prior_plus_residual', out['roi_ppg'][ri]),
                                         ('scaled_residual', out['scaled_residuals'][ri]),
                                         ('equal_six_prior_mixture', row['batch']['x'][0, ri, 3:].mean(0))):
                        component = scored_row(row['filename'], split, row['start'], name, signal, raw)
                        components.append(dict(filename=row['filename'], split=split, start=row['start'], roi=roi,
                            component=name, roi_attention=out['roi_attention'][ri], metrics=component['metrics'],
                            interpretation='DESCRIPTIVE_NOT_CAUSAL_ATTRIBUTION'))
        summaries, pairs = aggregate_conditions(rows), paired_comparisons(rows)
        print('Scoring complete; running all fixed corruption and timing controls.', flush=True)
        controls, control_outputs = run_neural_controls(model)
        timing = neural_timing(model)
    if len(rows) != 500 or state_fingerprint(model) != state_before:
        raise RuntimeError('Denominator/model-state immutability failure')
    for p, expected in locked.items():
        if file_sha256(output/p) != expected:
            raise RuntimeError('Neural prediction lock changed after scoring')
    for name, expected in sources.items():
        if source_sha256(args.scamps_root/name) != expected:
            raise RuntimeError('SCAMPS source changed')
    for p, expected in old_hashes.items():
        if file_sha256(args.gate08_root/p) != expected:
            raise RuntimeError('Gate 08 source artifact changed')
    assert_hashes_unchanged(code_before)
    if file_sha256(CHECKPOINT) != identity['model_sha256'] or file_sha256(MANIFEST) != identity['freeze_manifest_sha256']:
        raise RuntimeError('Frozen checkpoint changed')
    write_csv(output/'per_window_metrics.csv', rows)
    write_json(output/'condition_metrics.json', summaries)
    write_json(output/'paired_comparisons.json', pairs)
    write_csv(output/'paired_comparisons.csv', [dict(split=p['split'], comparator=p['comparator'],
        intended_windows=p['intended_windows'], intended_recordings=p['intended_recordings'], metric=k, **v)
        for p in pairs for k, v in p['metrics'].items()])
    write_csv(output/'per_recording_metrics.csv', [dict(split=s['split'], condition=s['condition'],
        filename=name, **metrics) for s in summaries for name, metrics in s['per_recording'].items()])
    write_csv(output/'allocation_components.csv', components)
    write_csv(output/'negative_control_metrics.csv', controls)
    write_json(output/'negative_control_summary.json', aggregate_conditions(controls))
    write_json(output/'synthetic_timing.json', timing)
    np.savez_compressed(output/'control_predictions.npz', **{k: np.stack(v) for k, v in control_outputs.items()})
    make_plots(output/'plots', records, outputs, refs, rows, summaries, controls, control_outputs)
    _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    memory = psutil.Process().memory_info()
    write_json(output/'model_identity.json', dict(**identity, state_before=state_before,
        state_after=state_fingerprint(model), model_state_unchanged=True, eval_mode=True,
        cpu_only=True, inference_mode=True, all_parameter_gradients_none=all(p.grad is None for p in model.parameters())))
    write_json(output/'provenance_and_validation.json', dict(parent_commit=PARENT, protocol_commit=PROTOCOL_COMMIT,
        protocol_path=PROTOCOL, implementation_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        code_and_parent_file_sha256=code_before, source_sha256_before_and_after=sources,
        checkpoint_sha256=file_sha256(CHECKPOINT), manifest_sha256=file_sha256(MANIFEST),
        gate08_artifacts_sha256_before_and_after=old_hashes, prediction_lock=locked,
        all_20_input_fingerprints_preserved=True, all_480_baseline_metrics_exactly_reproduced=True,
        all_480_reference_hash_checks_pass=True, all_sources_unchanged=True,
        intended_neural_windows=20, actual_neural_forward_calls=sum(r['forward_executed'] for r in blind_rows),
        scored_neural_windows=sum(r['pulse_estimate_eligible'] for r in blind_rows), classical_cells=480,
        runtime_seconds=dict(neural_forward_sum=sum(r['forward_seconds'] for r in blind_rows),
            neural_forward_per_window=[r['forward_seconds'] for r in blind_rows],
            total_after_input_hash_verification=time.perf_counter()-started),
        memory_bytes=dict(tracemalloc_peak=peak, process_rss_at_end=memory.rss,
            process_peak_working_set=getattr(memory, 'peak_wset', None)),
        versions=dict(python=platform.python_version(), **{k: importlib.metadata.version(k) for k in
            ('torch', 'numpy', 'scipy', 'matplotlib', 'psutil')}),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(), cpu_threads=torch.get_num_threads(),
        blocked_io_attempts=counts['blocked_io_attempts'], private_or_reserved_subject_access=False,
        subject_adaptation_enabled=False, allow_training=False, allow_bp=False, generalization_claim=False))
    files = [dict(path=p.relative_to(output).as_posix(), bytes=p.stat().st_size, sha256=file_sha256(p))
             for p in sorted(output.rglob('*')) if p.is_file()]
    write_json(output/'local_artifact_manifest.json', dict(local_root=args.out_dir.as_posix(), files=files))
    print('Gate 09 complete: 20 intended neural windows, 480 unchanged baseline cells; training disabled.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
