"""Phase 2 model/data regressions without requiring UBFC or a live detector."""
from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

from rppg_lab.classical import DEFAULT_PRIORS, METHOD_FUNCS
from rppg_lab.config import PipelineConfig
from rppg_lab.datasets import UBFCMultiROIRPPGDataset, UBFCSubject
from rppg_lab.extraction_cache import DEFAULT_WAVEFORM_ROIS, load_face_roi_extraction
from rppg_lab.losses import rppg_training_loss
from rppg_lab.models import MultiROIPriorResidualWaveformNet, _initial_prior_logits
from rppg_lab.signals import standardize_1d
from rppg_lab.types import (DualROIResult, PhysiologicalSignal, RegionResult, RGBTrace,
                            SharedTimebase, VideoMetadata)
from rppg_lab.window_priors import build_window_priors
from scripts import train_waveform_multi_roi_ubfc as training


def synthetic_result(count: int = 600, offset: float = 0.0,
                     missing: dict[str, slice] | None = None) -> DualROIResult:
    t = offset + np.arange(count) / 30
    regions = {}
    for r, name in enumerate(DEFAULT_WAVEFORM_ROIS):
        p = np.sin(2 * np.pi * 1.2 * t + r * 0.1)
        rgb = np.column_stack((100 + 20 * r + 2 * p,
                              120 + 15 * r + 4 * p,
                              80 + 10 * r + np.cos(2 * np.pi * 1.2 * t)))
        valid = np.ones(count, bool)
        if missing and name in missing:
            valid[missing[name]] = False
            rgb[~valid] = np.nan
        trace = RGBTrace(t, rgb, name, valid, {"extraction_valid": valid.copy()})
        # Deliberately unusable recording-global waveform: new training must
        # consume raw RGB, not slice this waveform into priors.
        signal = PhysiologicalSignal(np.full(count, np.nan), t, 30, name, "GREEN")
        regions[name] = RegionResult(trace, signal, {})
    first = regions[DEFAULT_WAVEFORM_ROIS[0]]
    return DualROIResult(first, first, SharedTimebase(t, t), VideoMetadata(Path("synthetic.avi"), 64, 64, 30, count),
                         {}, {}, [], [], {}, regions)


class MultiROIPriorModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(21)
        self.model = MultiROIPriorResidualWaveformNet(9, DEFAULT_PRIORS, base_channels=8, num_blocks=1, dropout=0)
        self.model.eval()
        self.x = torch.randn(2, 3, 9, 300)

    def test_exact_shapes_and_tensor_default(self) -> None:
        out = self.model(self.x, return_dict=True)
        expected = dict(ppg=(2, 300), quality=(2,), features=(2, 8, 300), roi_attention=(2, 3),
                        roi_ppg=(2, 3, 300), prior_weights=(2, 3, 6), fused_priors=(2, 3, 300), residuals=(2, 3, 300))
        self.assertEqual(set(out), set(expected))
        for key, shape in expected.items():
            self.assertEqual(tuple(out[key].shape), shape, key)
        torch.testing.assert_close(self.model(self.x), out["ppg"])

    def test_exact_conservative_initialization(self) -> None:
        out = self.model(self.x, roi_valid=torch.ones(2, 3, dtype=torch.bool),
                         prior_valid=torch.ones(2, 3, 6, dtype=torch.bool), return_dict=True)
        alpha = torch.softmax(_initial_prior_logits(DEFAULT_PRIORS), dim=0)
        expected_priors = (self.x[:, :, 3:] * alpha[None, None, :, None]).sum(dim=2)
        for param in (self.model.roi_prior_bias, self.model.static_roi_logits,
                      self.model.gate_head[-1].weight, self.model.gate_head[-1].bias,
                      self.model.attn_head[-1].weight, self.model.attn_head[-1].bias,
                      self.model.residual_head.weight, self.model.residual_head.bias):
            self.assertEqual(torch.count_nonzero(param).item(), 0)
        self.assertEqual(torch.count_nonzero(out["residuals"]).item(), 0)
        torch.testing.assert_close(out["prior_weights"], alpha[None, None, :].expand(2, 3, 6))
        torch.testing.assert_close(out["roi_attention"], torch.full((2, 3), 1 / 3))
        torch.testing.assert_close(out["fused_priors"], expected_priors)
        torch.testing.assert_close(out["roi_ppg"], expected_priors)
        torch.testing.assert_close(out["ppg"], expected_priors.mean(dim=1), rtol=1e-6, atol=1e-6)

    def test_invalid_roi_zero_and_renormalization(self) -> None:
        valid = torch.tensor([[True, False, True], [False, True, False]])
        out = self.model(self.x, roi_valid=valid, return_dict=True)
        self.assertEqual(out["roi_attention"][~valid].count_nonzero().item(), 0)
        torch.testing.assert_close(out["roi_attention"].sum(dim=1), torch.ones(2))
        torch.testing.assert_close(out["roi_attention"], torch.tensor([[0.5, 0, 0.5], [0, 1, 0.]]))
        for key in ("prior_weights", "roi_ppg", "residuals", "fused_priors"):
            self.assertEqual(out[key][~valid].count_nonzero().item(), 0)

    def test_quality_bias_and_zero_quality_stays_finite(self) -> None:
        q = torch.tensor([[1., 0.5, 0.25], [0., 0., 0.]])
        beta = self.model(self.x, roi_quality=q, return_dict=True)["roi_attention"]
        self.assertTrue(beta[0, 0] > beta[0, 1] > beta[0, 2])
        torch.testing.assert_close(beta[0], q[0] / q[0].sum())
        self.assertTrue(torch.isfinite(beta).all())
        torch.testing.assert_close(beta.sum(dim=1), torch.ones(2))

    def test_prior_masking_and_invalid_roi_with_no_priors(self) -> None:
        prior = torch.ones(2, 3, 6, dtype=torch.bool)
        prior[0, 0, 2] = False
        prior[1, 1] = False
        roi = torch.ones(2, 3, dtype=torch.bool)
        roi[1, 1] = False
        out = self.model(self.x, roi_valid=roi, prior_valid=prior, return_dict=True)
        self.assertEqual(out["prior_weights"][~prior].count_nonzero().item(), 0)
        torch.testing.assert_close(out["prior_weights"].sum(dim=-1), roi.float())
        expected = _initial_prior_logits(DEFAULT_PRIORS)
        expected[2] = -torch.inf
        torch.testing.assert_close(out["prior_weights"][0, 0], expected.softmax(0))
        self.assertTrue(torch.isfinite(out["ppg"]).all())

    def test_no_valid_roi_raises_for_one_sample(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one valid ROI"):
            self.model(self.x, roi_valid=torch.tensor([[True, True, True], [False, False, False]]))

    def test_valid_roi_without_priors_raises(self) -> None:
        mask = torch.ones(2, 3, 6, dtype=torch.bool)
        mask[0, 1] = False
        with self.assertRaisesRegex(ValueError, "at least one valid prior"):
            self.model(self.x, prior_valid=mask)

    def test_masked_nans_cannot_contaminate_output(self) -> None:
        roi = torch.tensor([[True, False, True], [True, False, True]])
        prior = torch.ones(2, 3, 6, dtype=torch.bool)
        prior[:, 0, 0] = False
        baseline = self.model(self.x, roi_valid=roi, prior_valid=prior, return_dict=True)
        corrupt = self.x.clone()
        corrupt[:, 1] = torch.nan
        corrupt[:, 0, 3] = torch.nan
        out = self.model(corrupt, roi_valid=roi, prior_valid=prior, return_dict=True)
        for key in out:
            torch.testing.assert_close(out[key], baseline[key])

    def test_fused_features_use_same_beta_after_training_heads(self) -> None:
        with torch.no_grad():
            self.model.static_roi_logits.copy_(torch.tensor([1., 2., -1.]))
            self.model.residual_head.bias.fill_(0.2)
        out = self.model(self.x, return_dict=True)
        feat = self.model.shared(self.x.reshape(6, 9, 300)).reshape(2, 3, 8, 300)
        torch.testing.assert_close(out["features"], (out["roi_attention"][..., None, None] * feat).sum(1))
        torch.testing.assert_close(out["ppg"], (out["roi_attention"][..., None] * out["roi_ppg"]).sum(1))

    def test_shape_errors_and_order_configuration(self) -> None:
        for kwargs in ({"roi_valid": torch.ones(2, 3)}, {"prior_valid": torch.ones(2, 3, dtype=torch.bool)},
                       {"roi_quality": torch.ones(3)}, {"roi_quality": torch.full((2, 3), torch.nan)}):
            with self.subTest(kwargs=list(kwargs)), self.assertRaises(ValueError):
                self.model(self.x, **kwargs)
        with self.assertRaises(ValueError):
            self.model(torch.randn(2, 9, 300))
        with self.assertRaisesRegex(ValueError, "finite"):
            self.model(torch.full_like(self.x, torch.nan))
        for kwargs in ({"roi_names": ()}, {"prior_names": ()}, {"in_channels": 8}, {"prior_start": -1}):
            options = dict(in_channels=9, prior_names=DEFAULT_PRIORS)
            options.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                MultiROIPriorResidualWaveformNet(**options)
        reordered = MultiROIPriorResidualWaveformNet(9, DEFAULT_PRIORS, roi_names=DEFAULT_WAVEFORM_ROIS[::-1])
        self.assertEqual(reordered.model_config["roi_names"], DEFAULT_WAVEFORM_ROIS[::-1])
        chrom = MultiROIPriorResidualWaveformNet(9, DEFAULT_PRIORS, init_prior="CHROM")
        torch.testing.assert_close(chrom.static_prior_logits, _initial_prior_logits(DEFAULT_PRIORS, "CHROM"))

    def test_synthetic_backward_and_optimizer_step(self) -> None:
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        out = self.model(self.x, return_dict=True)
        target = torch.randn(2, 300)
        loss = rppg_training_loss(out["ppg"], target, torch.tensor([72., 80.]), fs=30, residual=out["residuals"])
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for p in self.model.parameters():
            if p.grad is not None:
                self.assertTrue(torch.isfinite(p.grad).all())
        self.assertGreater(self.model.residual_head.weight.grad.abs().sum().item(), 0)
        self.assertGreater(self.model.gate_head[-1].weight.grad.abs().sum().item(), 0)
        self.assertGreater(self.model.attn_head[-1].weight.grad.abs().sum().item(), 0)
        optimizer.step()
        self.assertGreater(self.model(self.x, return_dict=True)["residuals"].abs().sum().item(), 0)


class WindowPriorTests(unittest.TestCase):
    def test_outside_window_changes_do_not_change_priors(self) -> None:
        rgb = synthetic_result().face.rgb.values.copy()
        changed = rgb.copy()
        changed[:150] *= [1, 20, 0.1]
        changed[450:] += [200, -100, 50]
        first = build_window_priors(rgb[150:450], 30)
        second = build_window_priors(changed[150:450], 30)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertTrue(first[1].all())

    def test_failed_nonfinite_and_wrong_shape_priors_are_masked(self) -> None:
        rgb = synthetic_result().face.rgb.values[:300]
        with patch.dict(METHOD_FUNCS, {"CHROM": lambda *_: np.full(300, np.nan),
                                      "PBV": lambda *_: np.zeros(299),
                                      "OMIT": lambda *_: (_ for _ in ()).throw(RuntimeError("failed"))}):
            priors, valid = build_window_priors(rgb, 30, ("GREEN", "CHROM", "PBV", "OMIT"))
        np.testing.assert_array_equal(valid, [True, False, False, False])
        self.assertTrue((priors[1:] == 0).all())
        np.testing.assert_allclose(priors[0].mean(), 0, atol=1e-6)
        np.testing.assert_allclose(priors[0].std(), 1, atol=1e-5)

    def test_window_only_method_arguments_and_independent_copies(self) -> None:
        rgb = synthetic_result().face.rgb.values[150:450].copy()
        before = rgb.copy()
        seen = []

        def mutating(window, fs):
            seen.append(window.copy())
            window[:] = 0
            return np.sin(np.arange(len(window)))

        def inspecting(window, fs):
            seen.append(window.copy())
            return window[:, 1]

        with patch.dict(METHOD_FUNCS, {"GREEN": mutating, "CHROM": inspecting}):
            build_window_priors(rgb, 30, ("GREEN", "CHROM"))
        np.testing.assert_array_equal(rgb, before)
        for window in seen:
            np.testing.assert_array_equal(window, before)


class MultiROIDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.result = synthetic_result()
        self.video = self.root / "vid.avi"
        self.video.write_bytes(b"mock-video-identity")
        self.gt = self.root / "ground_truth.txt"
        self.write_ground_truth(self.result.shared.timestamps)
        self.subject = UBFCSubject("subject1", self.video, self.gt)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write_ground_truth(self, times: np.ndarray) -> None:
        ppg = np.sin(2 * np.pi * 1.2 * times)
        np.savetxt(self.gt, np.vstack((ppg, np.full(len(times), 150.0), times)))

    def dataset(self, result: DualROIResult | None = None, **kwargs) -> UBFCMultiROIRPPGDataset:
        opts = dict(cache_dir=None, stride_sec=10)
        opts.update(kwargs)
        with patch("rppg_lab.extraction_cache.process_recording", return_value=result or self.result) as process:
            ds = UBFCMultiROIRPPGDataset([self.subject], **opts)
            process.assert_called_once()
            config = process.call_args.args[1]
            self.assertEqual(config.regions, ("face",))
            self.assertEqual(config.requested_face_rois, ds.roi_names)
        return ds

    def test_shape_order_dtype_quality_and_loader(self) -> None:
        ds = self.dataset()
        self.assertEqual(ds.roi_names, DEFAULT_WAVEFORM_ROIS)
        self.assertEqual(ds.prior_names, tuple(DEFAULT_PRIORS))
        self.assertEqual(ds.channel_names, ("RGB_R", "RGB_G", "RGB_B", *DEFAULT_PRIORS))
        item = ds[0]
        for key, shape in dict(x=(3, 9, 300), y_ppg=(300,), y_hr=(), roi_valid=(3,),
                               roi_quality=(3,), prior_valid=(3, 6), start_sec=()).items():
            self.assertEqual(tuple(item[key].shape), shape, key)
        self.assertEqual(item["roi_valid"].dtype, torch.bool)
        self.assertEqual(item["prior_valid"].dtype, torch.bool)
        self.assertEqual(item["x"].dtype, torch.float32)
        self.assertTrue(item["roi_valid"].all() and item["prior_valid"].all())
        torch.testing.assert_close(item["roi_quality"], torch.ones(3))
        self.assertAlmostEqual(float(item["y_hr"]), 72.0, delta=0.5)  # not recording HR=150
        batch = next(iter(DataLoader(ds, batch_size=2)))
        self.assertEqual(tuple(batch["x"].shape), (2, 3, 9, 300))
        self.assertEqual(batch["subject_id"], ["subject1", "subject1"])
        self.assertEqual(tuple(batch["prior_valid"].shape), (2, 3, 6))

    def test_one_bad_cheek_retained_and_attention_zero(self) -> None:
        result = synthetic_result(missing={"left_cheek": slice(None)})
        ds = self.dataset(result)
        self.assertEqual(len(ds), 2)
        item = ds[0]
        torch.testing.assert_close(item["roi_valid"], torch.tensor([True, False, True]))
        self.assertEqual(item["x"][1].count_nonzero().item(), 0)
        self.assertFalse(item["prior_valid"][1].any())
        model = MultiROIPriorResidualWaveformNet(9, DEFAULT_PRIORS, base_channels=8, num_blocks=0)
        batch = next(iter(DataLoader(ds, batch_size=2)))
        out = training.forward_batch(model, batch, torch.device("cpu"))
        self.assertEqual(out["roi_attention"][:, 1].count_nonzero().item(), 0)

    def test_all_rois_bad_windows_excluded(self) -> None:
        result = synthetic_result(missing={name: slice(None) for name in DEFAULT_WAVEFORM_ROIS})
        ds = self.dataset(result)
        self.assertEqual(len(ds), 0)
        self.assertEqual({row["reason"] for row in ds.exclusions}, {"no_usable_roi"})

    def test_bounded_gap_quality_is_source_valid_fraction(self) -> None:
        ds = self.dataset(synthetic_result(missing={"left_cheek": slice(100, 102)}))
        item = ds[0]
        self.assertTrue(item["roi_valid"].all())
        self.assertAlmostEqual(float(item["roi_quality"][1]), 298 / 300, places=6)
        self.assertEqual(ds.records[0]["interpolated"][1, :300].sum(), 2)
        # Enough source validity, but a long gap remains unresolved.
        ds = self.dataset(synthetic_result(missing={"left_cheek": slice(100, 110)}))
        self.assertFalse(ds[0]["roi_valid"][1])
        self.assertGreater(float(ds[0]["roi_quality"][1]), 0.9)

    def test_low_source_validity_rejected_even_when_gap_interpolated(self) -> None:
        result = synthetic_result(missing={"left_cheek": slice(10, 290, 3)})
        ds = self.dataset(result)
        self.assertTrue(np.isfinite(ds.records[0]["rgb"][1, :300]).all())
        self.assertFalse(ds[0]["roi_valid"][1])

    def test_reference_common_support_retains_video_origin_no_extrapolation(self) -> None:
        result = synthetic_result(offset=0.017)
        gt_times = result.shared.timestamps[0] + 0.051 + np.arange(550) / 30
        self.write_ground_truth(gt_times)
        ds = self.dataset(result)
        retained = ds.records[0]["timestamps"]
        np.testing.assert_array_equal(retained, result.shared.timestamps[
            (result.shared.timestamps >= gt_times[0]) & (result.shared.timestamps <= gt_times[-1])])
        expected = np.interp(retained[:300], gt_times, np.sin(2 * np.pi * 1.2 * gt_times).astype(np.float32))
        np.testing.assert_allclose(ds[0]["y_ppg"], standardize_1d(expected), atol=1e-6)
        self.assertEqual(float(ds[0]["start_sec"]), float(retained[0]))

    def test_dataset_outside_window_changes_do_not_change_channels(self) -> None:
        first = self.dataset()
        changed = synthetic_result()
        for region in changed.face_rois.values():
            region.rgb.values[300:] *= [3., 0.01, 10.]
        second = self.dataset(changed)
        torch.testing.assert_close(first[0]["x"], second[0]["x"], rtol=0, atol=0)
        torch.testing.assert_close(first[0]["y_ppg"], second[0]["y_ppg"], rtol=0, atol=0)
        # Check actual algorithm calls throughout construction/item retrieval.
        with patch.dict(METHOD_FUNCS, {"GREEN": lambda window, _: np.arange(len(window), dtype=float)}), \
                patch("rppg_lab.window_priors.build_window_priors", wraps=build_window_priors) as build:
            ds = self.dataset(prior_methods=("GREEN",))
            ds[0]
            self.assertTrue(all(call.args[0].shape == (300, 3) for call in build.call_args_list))

    def test_duplicate_contact_times_explicit_mean_without_video_clock_change(self) -> None:
        t = self.result.shared.timestamps
        ppg = np.sin(2 * np.pi * 1.2 * t)
        # One extra contact observation at an existing time; mean restores ppg.
        duplicate_t = np.insert(t, 100, t[100])
        duplicate_ppg = np.insert(ppg, 100, ppg[100] + 2)
        duplicate_ppg[101] -= 2
        np.savetxt(self.gt, np.vstack((duplicate_ppg, np.full(601, 150), duplicate_t)))
        with self.assertWarnsRegex(RuntimeWarning, "duplicate contact-reference"):
            ds = self.dataset()
        np.testing.assert_array_equal(ds.records[0]["timestamps"], t)
        np.testing.assert_allclose(ds.records[0]["ppg"], ppg, atol=1e-7)
        self.assertEqual(ds.reference_diagnostics[0]["duplicate_timestamp_samples_averaged"], 1)
        duplicate_t[101] = duplicate_t[100] - 0.001
        np.savetxt(self.gt, np.vstack((duplicate_ppg, np.full(601, 150), duplicate_t)))
        with self.assertRaisesRegex(ValueError, "nondecreasing"):
            self.dataset()

    def test_no_reference_overlap_is_excluded(self) -> None:
        self.write_ground_truth(self.result.shared.timestamps + 100)
        ds = self.dataset()
        self.assertEqual(len(ds), 0)
        self.assertEqual(ds.exclusions[0]["reason"], "insufficient_common_support")

    def test_prior_failures_masks_and_all_failed_roi(self) -> None:
        green = METHOD_FUNCS["GREEN"]

        def fail_left(rgb, fs):
            if 110 < rgb[0, 0] < 130:
                raise RuntimeError("left failed")
            return green(rgb, fs)

        with patch.dict(METHOD_FUNCS, {"GREEN": fail_left, "CHROM": lambda rgb, _: np.full(len(rgb), np.inf)}):
            ds = self.dataset(prior_methods=("GREEN", "CHROM"))
            item = ds[0]
        torch.testing.assert_close(item["roi_valid"], torch.tensor([True, False, True]))
        torch.testing.assert_close(item["prior_valid"], torch.tensor([[True, False], [False, False], [True, False]]))
        self.assertEqual(item["x"][:, 4].count_nonzero().item(), 0)
        with patch.dict(METHOD_FUNCS, {"GREEN": lambda *_: (_ for _ in ()).throw(ValueError("failed"))}):
            self.assertEqual(len(self.dataset(prior_methods=("GREEN",))), 0)

    def test_quality_independent_of_target_and_window_standardization(self) -> None:
        first = self.dataset()
        quality = first[0]["roi_quality"].clone()
        np.savetxt(self.gt, np.vstack((np.cos(2 * np.pi * 2 * self.result.shared.timestamps),
                                     np.full(600, 200), self.result.shared.timestamps)))
        second = self.dataset()
        torch.testing.assert_close(quality, second[0]["roi_quality"])
        item = first[0]
        torch.testing.assert_close(item["x"].mean(dim=-1), torch.zeros(3, 9), atol=1e-6, rtol=0)
        torch.testing.assert_close(item["x"].std(dim=-1, correction=0), torch.ones(3, 9), atol=1e-5, rtol=0)

    def test_no_legacy_extractor_dependency(self) -> None:
        with patch("rppg_lab.datasets.extract_rgb_trace", side_effect=AssertionError("legacy called")), \
                patch("rppg_lab.roi.extract_rgb_trace", side_effect=AssertionError("legacy called")), \
                patch("rppg_lab.roi.FaceROIExtractor", side_effect=AssertionError("legacy called")):
            ds = self.dataset()
            self.assertTrue(torch.isfinite(ds[0]["x"]).all())

    def test_reordered_rois_and_priors(self) -> None:
        ds = self.dataset(roi_names=DEFAULT_WAVEFORM_ROIS[::-1], prior_methods=("OMIT", "GREEN"))
        self.assertEqual(ds.roi_names, DEFAULT_WAVEFORM_ROIS[::-1])
        self.assertEqual(ds.prior_names, ("OMIT", "GREEN"))
        expected, valid = build_window_priors(self.result.face_rois["right_cheek"].rgb.values[:300], 30, ds.prior_names)
        np.testing.assert_array_equal(ds[0]["x"][0, 3:], expected)
        self.assertTrue(valid.all())

    def test_cache_round_trip_and_invalidation(self) -> None:
        cache = self.root / "phase1_cache"
        config = replace(PipelineConfig(), regions=("face",), face_roi="forehead", face_rois=DEFAULT_WAVEFORM_ROIS)
        with patch("rppg_lab.extraction_cache.process_recording", return_value=self.result) as process:
            first = load_face_roi_extraction(self.video, config, cache, max_frames=600)
            second = load_face_roi_extraction(self.video, config, cache, max_frames=600)
            self.assertEqual(process.call_count, 1)
            for name in DEFAULT_WAVEFORM_ROIS:
                np.testing.assert_array_equal(first.traces[name].values, second.traces[name].values)
                np.testing.assert_array_equal(first.traces[name].quality["extraction_valid"],
                                              second.traces[name].quality["extraction_valid"])
            np.testing.assert_array_equal(first.shared.timestamps, second.shared.timestamps)
            file = next(cache.glob("*.npz"))
            with np.load(file, allow_pickle=False) as arrays:
                self.assertFalse(any("prior" in key for key in arrays.files))
                self.assertEqual(arrays["face_roi_names"].tolist(), list(DEFAULT_WAVEFORM_ROIS))
            # Relevant geometry/detector/sampling settings and source/model identity.
            for changed in (replace(config, skin_mask=True), replace(config, max_gap_sec=0.1),
                            replace(config, detector_backend="legacy"), replace(config, min_pixels=80),
                            replace(config, face_rois=DEFAULT_WAVEFORM_ROIS[::-1])):
                load_face_roi_extraction(self.video, changed, cache, max_frames=600)
            self.assertEqual(process.call_count, 6)
            self.video.write_bytes(b"different-size-source")
            load_face_roi_extraction(self.video, config, cache, max_frames=600)
            self.assertEqual(process.call_count, 7)
            load_face_roi_extraction(self.video, config, cache, max_frames=500)
            self.assertEqual(process.call_count, 8)
            model = self.root / "face.task"
            model.write_bytes(b"model1")
            model_config = replace(config, face_model=str(model))
            load_face_roi_extraction(self.video, model_config, cache)
            model.write_bytes(b"model2")
            load_face_roi_extraction(self.video, model_config, cache)
            self.assertEqual(process.call_count, 10)

    def test_corrupt_cache_rebuilt(self) -> None:
        cache = self.root / "cache"
        with patch("rppg_lab.extraction_cache.process_recording", return_value=self.result) as process:
            UBFCMultiROIRPPGDataset([self.subject], cache_dir=cache)
            next(cache.glob("*.npz")).write_bytes(b"truncated")
            with self.assertWarnsRegex(RuntimeWarning, "Rebuilding"):
                UBFCMultiROIRPPGDataset([self.subject], cache_dir=cache)
            self.assertEqual(process.call_count, 2)

    def test_invalid_reference_and_bad_configuration(self) -> None:
        np.savetxt(self.gt, np.vstack((np.zeros(600), np.ones(600) * 72, self.result.shared.timestamps)))
        self.assertEqual(len(self.dataset()), 0)
        for kwargs in ({"roi_names": ()}, {"prior_methods": ()}, {"min_valid_fraction": 1.1},
                       {"win_sec": 0}, {"max_frames": 0}, {"extraction_config": PipelineConfig(sample_rate=60)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.dataset(**kwargs)

    def test_training_epoch_zero_optimizer_and_checkpoint_metadata(self) -> None:
        subjects = [self.subject, UBFCSubject("subject2", self.video, self.gt)]
        output = self.root / "training"
        args = ["train", "--ubfc-root", str(self.root), "--out-dir", str(output),
                "--cache-dir", str(self.root / "cache"), "--epochs", "1", "--batch-size", "2",
                "--base-channels", "8", "--num-blocks", "0", "--stride-sec", "10"]
        with patch.object(training, "find_ubfc_subjects", return_value=subjects), \
                patch("rppg_lab.extraction_cache.process_recording", return_value=self.result), \
                patch("sys.argv", args), redirect_stdout(io.StringIO()) as stdout:
            training.main()
        checkpoint = torch.load(output / "waveform_multi_roi_last.pt", map_location="cpu", weights_only=True)
        expected = {"model_class", "model_config", "roi_names", "prior_names", "channel_ordering",
                    "extraction_config", "fs", "window_length_samples", "win_sec", "seed", "train_subjects",
                    "val_subjects", "loss_name", "checkpoint_selection_criterion", "epoch", "optimizer_state"}
        self.assertTrue(expected <= checkpoint.keys())
        self.assertEqual(checkpoint["epoch"], 1)
        self.assertIn("PROVISIONAL", checkpoint["loss_name"])
        self.assertIn("LAST", checkpoint["checkpoint_selection_criterion"])
        self.assertFalse(set(checkpoint["train_subjects"]) & set(checkpoint["val_subjects"]))
        history = json.loads((output / "training_log_PROVISIONAL.json").read_text())
        self.assertEqual(history["epochs"][0]["residual_contribution_max_abs"], 0)
        self.assertGreater(history["epochs"][1]["residual_contribution_max_abs"], 0)
        self.assertTrue((output / "waveform_multi_roi_best_hr_PROVISIONAL.pt").exists())
        rebuilt = MultiROIPriorResidualWaveformNet(**checkpoint["model_config"])
        rebuilt.load_state_dict(checkpoint["model_state"])
        self.assertIn("Mean ROI attention", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
