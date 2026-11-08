"""Verify saved Gate 09 outputs independently, with neural execution disabled."""
from pathlib import Path
import argparse
import csv
import json
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
from PIL import Image

from rppg_lab.artifacts import file_sha256
from rppg_lab.waveform_v2_classical_study import METHODS, CONDITIONS, array_sha256
from rppg_lab.waveform_v2_frozen_cache import (read_json, clean, load_inputs, load_references,
    assert_hashes_unchanged)
from rppg_lab.waveform_v2_frozen_comparison import (INPUT_KEYS, CHECKPOINT, MANIFEST,
    CHECKPOINT_SHA256, MANIFEST_SHA256, scored_row, aggregate_conditions, paired_comparisons)
from scripts.audit_waveform_v2_frozen_comparison import install_subject_guard


def verify(output):
    counts = install_subject_guard()
    manifest = read_json(output/'local_artifact_manifest.json')
    for row in manifest['files']:
        p = (output/row['path']).resolve()
        if output.resolve() not in p.parents or not p.is_file() or file_sha256(p) != row['sha256'] or p.stat().st_size != row['bytes']:
            raise ValueError('Saved artifact mismatch: '+row['path'])
    provenance = read_json(output/'provenance_and_validation.json')
    assert_hashes_unchanged(provenance['code_and_parent_file_sha256'])
    assert_hashes_unchanged({CHECKPOINT: CHECKPOINT_SHA256, MANIFEST: MANIFEST_SHA256})
    gate08 = Path(read_json('docs/waveform_v2/classical_gate08/local_artifact_manifest.json')['local_root'])
    records, blind_classical = load_inputs(gate08)
    refs, _ = load_references(gate08, records)
    locked = read_json(output/'prediction_lock.json')
    for name, expected in locked['hashes'].items():
        assert_hashes_unchanged({output/name: expected})
    blind = read_json(output/'label_blind_neural_predictions.json')['windows']
    blind_lookup = {(r['filename'], r['start']): r for r in blind}
    with (output/'per_window_metrics.csv').open(newline='', encoding='utf-8') as stream:
        actual = list(csv.DictReader(stream))
    lookup = {(r['filename'], int(r['start']), r['condition']): r for r in actual}
    if len(actual) != 500 or len(lookup) != 500 or len(blind) != 20 or locked['actual_forward_calls'] != 20:
        raise ValueError('Denominator/unique key/forward count mismatch')
    recomputed = []
    with np.load(output/'locked_neural_predictions.npz', allow_pickle=False) as archive:
        for split in ('Train', 'Val', 'Test'):
            for record in (r for r in records if r['split'] == split):
                filename, start = record['filename'], record['start']
                stem = Path(filename).stem+'_'+str(start)
                for field in INPUT_KEYS:
                    np.testing.assert_array_equal(archive[stem+'_'+field], record['batch'][field])
                    if array_sha256(archive[stem+'_'+field]) != blind_lookup[(filename, start)]['input_fingerprints'][field]:
                        raise ValueError('Mask/tensor fingerprint changed')
                np.testing.assert_array_equal(archive[stem+'_indices'], record['indices'])
                wave = archive[stem+'_waveform']
                if array_sha256(wave) != blind_lookup[(filename, start)]['prediction_sha256']:
                    raise ValueError('Blind waveform changed')
                signals = [('frozen_v1', wave, blind_lookup[(filename, start)])]
                signals += [(f'{roi}/{method}', record['classical'][ri, mi], blind_classical[(filename, start, roi, method)])
                            for ri, roi in enumerate(CONDITIONS) for mi, method in enumerate(METHODS)]
                for condition, signal, status in signals:
                    row = scored_row(filename, split, start, condition, signal, refs[(filename, start)],
                        eligible=status['pulse_estimate_eligible'], computational=status['computational_success'],
                        reason=status['prediction_failure_reason'])
                    saved = lookup[(filename, start, condition)]
                    if saved['prediction_sha256'] != row['prediction_sha256'] or saved['raw_reference_sha256'] != row['raw_reference_sha256']:
                        raise ValueError('Scored source identities changed')
                    for key, value in row['metrics'].items():
                        encoded = saved['metrics_'+key]
                        if isinstance(value, (bool, np.bool_)):
                            assert str(value) == encoded
                        else:
                            np.testing.assert_allclose(value, float(encoded) if encoded else np.nan, rtol=1e-12, atol=1e-12, equal_nan=True)
                    recomputed.append(row)
    if clean(aggregate_conditions(recomputed)) != read_json(output/'condition_metrics.json'):
        raise ValueError('Balanced aggregation differs')
    if clean(paired_comparisons(recomputed)) != read_json(output/'paired_comparisons.json'):
        raise ValueError('Matched paired summaries differ')
    plots = list((output/'plots').glob('*.png'))
    for p in plots:
        with Image.open(p) as im: im.verify()
    if len(plots) != 15 or counts['blocked_io_attempts']:
        raise ValueError('Plot count or protected-I/O verification failure')
    return dict(artifacts_verified=len(manifest['files']), pngs_verified=len(plots),
        scored_rows_recomputed=500, neural_windows=20, fixed_baseline_cells=480,
        exact_original_input_mask_and_index_checks=True, all_summary_and_paired_tables_reproduced=True,
        checkpoint_source_and_prediction_locks_unchanged=True, neural_forward_calls=0,
        checkpoint_load_calls=0, blocked_io_attempts=counts['blocked_io_attempts'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    with ExitStack() as stack:
        for target in ('torch.nn.Module.__call__', 'torch.load', 'torch.Tensor.backward'):
            stack.enter_context(patch(target, side_effect=AssertionError('Saved-result verification has no neural execution')))
        result = verify(args.output)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
