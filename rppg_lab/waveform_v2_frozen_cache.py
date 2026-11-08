"""Read pinned Gate 08 artifacts without reconstructing inputs or reference values."""
from pathlib import Path
import csv
import json
import subprocess

import numpy as np

from .artifacts import file_sha256
from .waveform_v2_classical_study import METHODS, CONDITIONS, array_sha256
from .waveform_v2_frozen_comparison import validate_inputs
from .waveform_v2_scamps import _check_path

PARENT = 'bcd3f98dd65088d41c447a2651e145d70cfc8e05'
PROTOCOL_COMMIT = '3c5efa8'
PROTOCOL = 'docs/WAVEFORM_V2_FROZEN_COMPARISON_GATE09_PROTOCOL.md'
GATE08 = Path('docs/waveform_v2/classical_gate08')
GATE06 = Path('docs/waveform_v2/scamps_gate06/roi_coverage_and_inputs.json')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def write_json(path, data):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(clean(data), stream, indent=2, allow_nan=False)
        stream.write('\n')


def write_csv(path, rows):
    flattened = []
    for row in rows:
        flat = {}
        for key, value in row.items():
            if isinstance(value, dict):
                flat.update({key+'_'+k: (json.dumps(clean(v)) if isinstance(v, (dict, list, tuple)) else clean(v))
                             for k, v in value.items()})
            else:
                flat[key] = json.dumps(clean(value)) if isinstance(value, (list, tuple)) else clean(value)
        flattened.append(flat)
    with Path(path).open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in flattened for k in r)))
        writer.writeheader(); writer.writerows(flattened)


def verified_artifact(root, name):
    _check_path(Path(root))
    expected = {r['path']: r['sha256'] for r in read_json(GATE08/'local_artifact_manifest.json')['files']}
    path = Path(root)/name
    _check_path(path)
    if name not in expected or file_sha256(path) != expected[name]:
        raise ValueError('Gate 08 artifact mismatch: '+name)
    return path


def validate_indices(indices, start, stop):
    values = np.asarray(indices)
    if values.dtype.kind not in 'iu' or not np.array_equal(values, np.arange(start, stop)) or stop-start != 300:
        raise ValueError('Exactly original complete reference/input indices required')


def load_inputs(root):
    """Load only X/predictions and original metadata; no reference NPZ is opened."""
    archive_path = verified_artifact(root, 'locked_predictions.npz')
    lock = read_json(GATE08/'prediction_lock.json')
    if file_sha256(archive_path) != lock['predictions_sha256']:
        raise ValueError('Prediction lock mismatch')
    blind = read_json(verified_artifact(root, 'label_blind_predictions.json'))
    if blind['reference_values_decoded'] is not False or len(blind['cells']) != 480:
        raise ValueError('Invalid fixed classical denominator/lock')
    original = read_json(GATE06)['recordings']
    prior = {r['filename']: r for r in original}
    lookup = {(r['filename'], r['start'], r['roi'], r['method']): r for r in blind['cells']}
    records = []
    with np.load(archive_path, allow_pickle=False) as archive:
        for filename in sorted(prior):
            stem = Path(filename).stem
            x, classical = archive[stem+'_x'].copy(), archive[stem].copy()
            if x.shape != (2, 3, 9, 300) or classical.shape != (2, 4, 6, 300):
                raise ValueError('Fixed archive shape mismatch')
            for w, meta in enumerate(prior[filename]['windows']):
                start, stop = meta['start_index'], meta['stop_index']
                if (start, stop) != ((0, 300), (300, 600))[w] or array_sha256(x[w]) != meta['input_sha256']:
                    raise ValueError('Gate 06 input identity mismatch')
                batch = dict(x=x[w:w+1], roi_valid=np.asarray(meta['roi_valid'], bool)[None],
                    roi_quality=np.asarray(meta['roi_quality'], np.float32)[None],
                    prior_valid=np.asarray(meta['prior_valid'], bool)[None])
                fingerprints = validate_inputs(batch)
                split = None
                for ri, roi in enumerate(CONDITIONS):
                    for mi, method in enumerate(METHODS):
                        row = lookup[(filename, start, roi, method)]
                        if array_sha256(classical[w, ri, mi]) != row['prediction_sha256']:
                            raise ValueError('Fixed classical signal identity mismatch')
                        split = row['split'] if split is None else split
                        if row['split'] != split:
                            raise ValueError('Inconsistent fixed split')
                indices = np.arange(start, stop)
                validate_indices(indices, start, stop)
                records.append(dict(filename=filename, split=split, start=start, stop=stop,
                    indices=indices, batch=batch, input_fingerprints=fingerprints,
                    classical=classical[w], source_sha256=prior[filename]['source_sha256']))
    if len(records) != 20 or len(lookup) != 480:
        raise ValueError('Intended denominator changed')
    return records, lookup


def load_references(root, records):
    path = verified_artifact(root, 'reference_windows.npz')
    with (GATE08/'per_window_metrics.csv').open(newline='', encoding='utf-8') as stream:
        fixed = {(r['filename'], int(r['start']), r['roi'], r['method']): r for r in csv.DictReader(stream)}
    refs = {}
    with np.load(path, allow_pickle=False) as archive:
        for row in records:
            raw = archive[Path(row['filename']).stem][row['start']//300].copy()
            validate_indices(row['indices'], row['start'], row['stop'])
            if raw.shape != (300,):
                raise ValueError('Reference support mismatch')
            for roi in CONDITIONS:
                for method in METHODS:
                    prior = fixed[(row['filename'], row['start'], roi, method)]
                    if array_sha256(raw) != prior['raw_reference_sha256']:
                        raise ValueError('Reference differs from Gate 08')
            refs[(row['filename'], row['start'])] = raw
    return refs, fixed


def validate_output(output, source, baseline):
    for path in (output, source, baseline):
        _check_path(Path(path))
    out, src, cache = (Path(p).resolve() for p in (output, source, baseline))
    allowed = Path('outputs').resolve()
    if out.exists() or allowed not in out.parents or any(p == out or p in out.parents for p in (src, cache)):
        raise ValueError('Use a fresh directory strictly under outputs, outside sources and prior artifacts')
    return out


def assert_hashes_unchanged(hashes):
    for path, expected in hashes.items():
        if file_sha256(path) != expected:
            raise RuntimeError('Source/checkpoint/artifact immutability failure: '+str(path))


def check_design(code):
    subprocess.run(['git', 'merge-base', '--is-ancestor', PROTOCOL_COMMIT, 'HEAD'], check=True, capture_output=True)
    for path in (*code, PROTOCOL):
        subprocess.run(['git', 'ls-files', '--error-unmatch', path], check=True, capture_output=True)
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--', *code, PROTOCOL], check=True, capture_output=True)
    names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', PARENT], text=True).splitlines()
    # No raw data, checkpoint blobs, historical output directories or private paths are tracked.
    for name in names:
        _check_path(Path(name))
        expected = subprocess.check_output(['git', 'show', PARENT+':'+name])
        if Path(name).read_bytes().replace(b'\r\n', b'\n') != expected.replace(b'\r\n', b'\n'):
            raise ValueError('Pre-existing tracked file changed: '+name)
    return {p: file_sha256(p) for p in (*names, *code, PROTOCOL)}
