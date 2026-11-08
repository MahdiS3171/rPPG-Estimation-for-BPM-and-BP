"""Gate 08 reviewed regression profile; waveform neural/checkpoint/backward denied."""
from contextlib import ExitStack
import importlib
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from scripts.verify_waveform_v2_safe import install_io_guard
from scripts.verify_waveform_v2_scamps_inputs_safe import SELECTION


def main():
    root = Path(__file__).absolute().parents[1]
    if Path.cwd() != root:
        raise RuntimeError('Run from repository root')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['MPLBACKEND'] = 'Agg'
    counts = install_io_guard(root)
    sys.path.insert(0, str(root/'tests'))
    import torch
    torch.set_num_threads(1)
    selection = dict(SELECTION)
    selection.update(test_waveform_v2_optical_quality=None, test_waveform_v2_classical_study=None)
    suite = unittest.TestSuite()
    for name, classes in selection.items():
        module = importlib.import_module(name)
        if classes is None:
            suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
        else:
            for cls in classes:
                suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(getattr(module, cls)))
    with ExitStack() as stack:
        for target in ('torch.nn.Module.__call__', 'torch.load', 'torch.Tensor.backward'):
            stack.enter_context(patch(target, side_effect=AssertionError('Gate 08 forbids neural/checkpoint/backward execution')))
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(f"GATE08_SAFE: run={result.testsRun} failures={len(result.failures)} errors={len(result.errors)} "
          f"skipped={len(result.skipped)} blocked_io_attempts={counts['blocked_io_attempts']}")
    return 0 if result.wasSuccessful() and counts['blocked_io_attempts'] == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
