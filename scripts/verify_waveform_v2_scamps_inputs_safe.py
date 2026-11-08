"""Gate 06 non-neural test profile; no network calls, checkpoints or real data.

The broader prior gate's runner includes random network forwards. This profile
selects input/geometry/clock contracts and actively blocks Torch model execution.
"""
from contextlib import ExitStack
import importlib
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from scripts.verify_waveform_v2_safe import install_io_guard

SELECTION = {
    "test_waveform_v2_scamps_inputs": None,
    "test_waveform_v2_scamps": None,
    "test_waveform_v2_infrastructure": None,
    "test_waveform_v2_timing_contract": None,
    "test_waveform_v2_integration": None,
    "test_waveform_multi_roi": ("WindowPriorTests",),
    "test_multi_face_roi": ("FaceMaskTests", "MultiFaceConfigTests"),
    "test_frozen_timebase": ("TimebaseTests", "NativePixelTests"),
}


def main():
    root = Path(__file__).absolute().parents[1]
    if Path.cwd() != root:
        raise RuntimeError("Run from the repository root")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["MPLBACKEND"] = "Agg"
    counts = install_io_guard(root)
    sys.path.insert(0, str(root / "tests"))
    import torch
    torch.set_num_threads(1)
    suite = unittest.TestSuite()
    for name, classes in SELECTION.items():
        module = importlib.import_module(name)
        if classes is None:
            suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
        else:
            for cls in classes:
                suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(getattr(module, cls)))
    with ExitStack() as stack:
        for target in ("torch.nn.Module.__call__", "torch.load", "torch.Tensor.backward"):
            stack.enter_context(patch(target, side_effect=AssertionError("Gate 06 forbids neural execution/checkpoint loading/backward")))
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(f"GATE06_SAFE: run={result.testsRun} failures={len(result.failures)} errors={len(result.errors)} "
          f"skipped={len(result.skipped)} blocked_io_attempts={counts['blocked_io_attempts']} neural_execution_allowed=False")
    return 0 if result.wasSuccessful() and counts['blocked_io_attempts'] == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
