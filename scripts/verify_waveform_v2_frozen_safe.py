"""Guarded Gate 09 synthetic/official-identity tests; no actual SCAMPS forward."""
from contextlib import ExitStack
import os
from pathlib import Path
import sys
import unittest

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MPLBACKEND'] = 'Agg'


def main():
    root = Path(__file__).absolute().parents[1]
    if Path.cwd() != root:
        raise RuntimeError('Run from repository root')
    from scripts.audit_waveform_v2_frozen_comparison import install_subject_guard, inference_guards
    counts = install_subject_guard()
    # Tests may read exactly the two official frozen files, never other raw/artifact directories.
    forbidden = ('Dataset', 'UBFCData', 'rPPG-10', 'SCAMPSData', 'outputs', 'cache_roi_multi_phase1',
                 'cache_roi_oldpoints', 'cache_rppg10_rgb', 'cache_rppg10_synthetic', 'cache_video', 'cache_video_prior')
    def guard(event, args):
        if event not in ('open', 'os.listdir', 'os.scandir') or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        p = Path(os.path.abspath(os.fsdecode(args[0])))
        if any(p == root/n or root/n in p.parents for n in forbidden):
            counts['blocked_io_attempts'] += 1
            raise PermissionError('Gate 09 test profile forbids raw/historical artifact I/O')
        checkpoint_root = root/'checkpoints'
        if checkpoint_root in p.parents and p not in (checkpoint_root/'waveform_v1_official_ubfc/waveform_v1_frozen.pt',
                checkpoint_root/'waveform_v1_official_ubfc/waveform_v1_freeze_manifest.json'):
            counts['blocked_io_attempts'] += 1
            raise PermissionError('Only the official frozen identity may be read')
    sys.addaudithook(guard)
    import torch
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    sys.path.insert(0, str(root/'tests'))
    import test_waveform_v2_frozen_comparison
    suite = unittest.defaultTestLoader.loadTestsFromModule(test_waveform_v2_frozen_comparison)
    with ExitStack() as stack:
        inference_guards(stack)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(f'GATE09_SAFE: run={result.testsRun} failures={len(result.failures)} errors={len(result.errors)} '
          f'skipped={len(result.skipped)} blocked_io_attempts={counts["blocked_io_attempts"]}')
    return 0 if result.wasSuccessful() and counts['blocked_io_attempts'] == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
