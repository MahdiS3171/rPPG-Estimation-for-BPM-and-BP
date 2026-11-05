"""Reviewed CPU/synthetic subset, excluding training, BP fitting and official weights.

Run from the repository root: python -m scripts.verify_waveform_v2_safe
Re-review the allowlist/exclusions if the existing test implementations change.
Python's audit hook is defense in depth, not a sandbox for native libraries.
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys
import unittest

EXISTING_MODULES = (
    "test_core", "test_dual_roi", "test_frozen_timebase", "test_multi_face_roi",
    "test_pilot6", "test_pilot6_delay_feasibility", "test_pilot6_protocol_timeline",
    "test_rppg10_external", "test_waveform_finalization", "test_waveform_multi_roi",
    "test_waveform_v1", "test_waveform_visualization",
)
NEW_MODULES = ("test_waveform_v2_infrastructure", "test_waveform_v2_timing_contract",
               "test_waveform_v2_integration", "test_waveform_v2_adaptation", "test_waveform_v2_scamps")
EXCLUDED_CLASSES = {
    "OfficialFrozenTests": "Loads the original frozen checkpoint",
    "WaveformTrainingIntegrationTests": "Runs a synthetic training epoch",
    "ReviewArtifactTests": "setUpClass invokes the training integration fixture",
    "InvalidROIBatchNormTests": "Performs optimizer steps",
}
EXCLUDED_METHODS = {
    "test_bp_train_only_scaling_and_model_roundtrip": "Fits BP regression",
    "test_baseline_selection_train_only_schema": "Fits BP baselines",
    "test_subject_grouped_baselines_and_duplicate_cuff_guard": "Fits exploratory BP baselines",
    "test_official_frozen_identity_when_available": "Reads original checkpoint and manifest",
    "test_synthetic_backward_and_optimizer_step": "Performs an optimizer step",
    "test_training_epoch_zero_optimizer_and_checkpoint_metadata": "Invokes training CLI",
}


def flatten(suite):
    for entry in suite:
        if isinstance(entry, unittest.TestSuite):
            yield from flatten(entry)
        else:
            yield entry


def install_io_guard(root):
    blocked = [root / name for name in ("Dataset", "UBFCData", "rPPG-10", "PUREData",
               "data/pure", "checkpoints", "outputs", "cache_roi_multi_phase1",
               "cache_roi_oldpoints", "cache_rppg10_rgb", "cache_rppg10_synthetic",
               "cache_video", "cache_video_prior", "SCAMPSData", "data/scamps")]
    prefixes = [os.path.normcase(os.path.abspath(p)) for p in blocked]
    scamps_cache_prefix = os.path.normcase(os.path.abspath(root / "cache_scamps"))
    counters = dict(blocked_io_attempts=0)

    def guard(event, args):
        if event not in ("open", "os.listdir", "os.scandir", "os.remove", "os.rename",
                         "os.mkdir", "os.rmdir"):
            return
        paths = args[:2] if event == "os.rename" else args[:1]
        for path in paths:
            if not isinstance(path, (str, bytes, os.PathLike)):
                continue
            name = os.path.normcase(os.path.abspath(os.fsdecode(path)))
            if name.startswith(scamps_cache_prefix) or any(name == p or name.startswith(p + os.sep) for p in prefixes):
                counters["blocked_io_attempts"] += 1
                raise PermissionError("Verification forbids I/O in protected source/artifact directories")
    sys.addaudithook(guard)
    return counters


def main():
    root = Path(__file__).absolute().parents[1]
    if Path.cwd() != root:
        raise RuntimeError("Run from the repository root")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["MPLBACKEND"] = "Agg"
    counts = install_io_guard(root)
    # Historical fixtures use sibling-module imports rather than tests.*.
    sys.path.insert(0, str(root / "tests"))
    import torch
    torch.set_num_threads(1)
    suite = unittest.TestSuite()
    excluded = []
    for name in (*EXISTING_MODULES, *NEW_MODULES):
        module = importlib.import_module(name)
        for case in flatten(unittest.defaultTestLoader.loadTestsFromModule(module)):
            reason = EXCLUDED_CLASSES.get(type(case).__name__) or EXCLUDED_METHODS.get(case._testMethodName)
            # Conservative exclusions also cover loss-only gradient checks. A source
            # inspection test containing forbidden-call strings is excluded as well.
            if name in EXISTING_MODULES and not reason:
                import inspect
                source = inspect.getsource(getattr(type(case), case._testMethodName))
                if any(token in source for token in (".backward(", "optimizer.step(", "training.main(")):
                    reason = "Executes backward or training plumbing"
            if reason:
                excluded.append((case.id(), reason))
            else:
                suite.addTest(case)
    for name, reason in excluded:
        print(f"EXCLUDED {name}: {reason}", flush=True)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(f"SAFE_VERIFICATION: run={result.testsRun} failures={len(result.failures)} "
          f"errors={len(result.errors)} skipped={len(result.skipped)} "
          f"excluded={len(excluded)} blocked_io_attempts={counts['blocked_io_attempts']}", flush=True)
    return 0 if result.wasSuccessful() and counts["blocked_io_attempts"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
