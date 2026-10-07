"""Part 3 scientific constraints and infrastructure regressions."""
from __future__ import annotations

import copy
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

from rppg_lab.config import PipelineConfig
from rppg_lab.datasets import UBFCSubject, UBFCMultiROIRPPGDataset
from rppg_lab.losses import (WaveformV1LossConfig, same_lag_waveform_derivative_loss,
                             waveform_spectral_distance, waveform_spectral_band,
                             waveform_v1_training_loss, rppg_training_loss)
from rppg_lab.models import MultiROIPriorResidualWaveformNet
from rppg_lab.splits import waveform_v1_split_manifest, validate_waveform_split_manifest
from rppg_lab.waveform_alignment import aligned_overlap, lag_candidates
from rppg_lab.waveform_metrics import waveform_window_metrics, aggregate_waveform_metrics
from rppg_lab.waveform_training import (WaveformCheckpointSelector, better_morphology,
                                       reconstruct_waveform_model, checkpoint_dataset_config,
                                       morphology_score, split_manifest_hash)
from scripts import train_waveform_multi_roi_ubfc as training
from scripts import evaluate_waveform_multi_roi_ubfc as evaluation


class InvalidROIBatchNormTests(unittest.TestCase):
    def test_train_statistics_match_valid_only_encoder(self):
        torch.manual_seed(17)
        model = MultiROIPriorResidualWaveformNet(5, ("GREEN", "CHROM"),
                                                base_channels=8, num_blocks=2, dropout=0)
        reference = copy.deepcopy(model)
        model.train()
        reference.train()
        x = torch.randn(3, 3, 5, 60)
        valid = torch.tensor([[True, False, True], [False, True, False], [True, True, False]])
        x[~valid] = torch.nan
        out = model(x, roi_valid=valid, return_dict=True)
        reference.shared(x.reshape(9, 5, 60)[valid.reshape(-1)])
        actual_bn = [m for m in model.shared.modules() if isinstance(m, torch.nn.BatchNorm1d)]
        expected_bn = [m for m in reference.shared.modules() if isinstance(m, torch.nn.BatchNorm1d)]
        for actual, expected in zip(actual_bn, expected_bn):
            torch.testing.assert_close(actual.running_mean, expected.running_mean, rtol=0, atol=0)
            torch.testing.assert_close(actual.running_var, expected.running_var, rtol=0, atol=0)
            torch.testing.assert_close(actual.num_batches_tracked, expected.num_batches_tracked)
        self.assertTrue(torch.isfinite(out["ppg"]).all())
        self.assertEqual(out["residuals"].count_nonzero().item(), 0)
        self.assertEqual(out["roi_attention"][~valid].count_nonzero().item(), 0)

        # Zero-initialized heads intentionally block encoder gradients at epoch 0.
        # After one optimizer update all four waveform paths must receive them.
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        target = torch.randn(3, 60)
        (out["ppg"] - target).square().mean().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        (model(x, roi_valid=valid) - target).square().mean().backward()
        for module in (model.shared, model.gate_head, model.attn_head, model.residual_head):
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(g.abs().sum().item() for g in grads), 0)
        self.assertTrue(all(p.grad is None for p in model.quality_head.parameters()))


def adversarial_signals(amplitude=.1):
    n = np.arange(300)
    target = np.sin(.07 * n) + amplitude * np.sin(1.8 * n)
    pred = np.sin(.07 * (n - 3)) + amplitude * np.sin(1.8 * (n + 2))
    return pred, target


class WaveformV1LossTests(unittest.TestCase):
    def test_shifted_prediction_correct_lag_and_gradient(self):
        rng = np.random.default_rng(42)
        target = torch.tensor(rng.normal(size=(2, 120)), dtype=torch.float64)
        pred = torch.cat((torch.randn(2, 7, dtype=torch.float64), target[:, :-7]), dim=-1).requires_grad_()
        result = same_lag_waveform_derivative_loss(pred, target, max_lag=10)
        torch.testing.assert_close(result["selected_lag"], torch.tensor([7, 7]))
        for key in ("waveform_corr", "derivative_corr"):
            torch.testing.assert_close(result[key], torch.zeros(2, dtype=torch.float64), atol=1e-8, rtol=0)
        # Perturb the selected overlap to confirm a nonzero valid gradient.
        changed = (pred.detach() + .02 * torch.randn_like(pred)).requires_grad_()
        losses = same_lag_waveform_derivative_loss(changed, target, max_lag=10)
        (losses["waveform_corr"] + .25 * losses["derivative_corr"]).mean().backward()
        self.assertTrue(torch.isfinite(changed.grad).all())
        self.assertGreater(changed.grad.abs().sum().item(), 0)
        self.assertEqual(changed.grad[:, :7].count_nonzero().item(), 0)

    def test_joint_selection_cannot_independently_align_derivative(self):
        pred, target = adversarial_signals(amplitude=.01)
        candidates = []
        for lag in lag_candidates(300, 8):
            p, t = aligned_overlap(pred, target, lag)
            candidates.append((lag, np.corrcoef(p, t)[0, 1], np.corrcoef(np.diff(p), np.diff(t))[0, 1]))
        wave_best = max(candidates, key=lambda row: row[1])[0]
        derivative_best = max(candidates, key=lambda row: row[2])[0]
        self.assertNotEqual(wave_best, derivative_best)
        # Moderate internal weight makes a true compromise distinct from the
        # waveform-only choice; both returned terms must use that one choice.
        result = same_lag_waveform_derivative_loss(torch.tensor(pred[None]), torch.tensor(target[None]), 8, .25)
        selected = int(result["selected_lag"][0])
        expected = max(candidates, key=lambda row: row[1] + .25 * row[2])
        self.assertEqual(selected, expected[0])
        self.assertNotEqual(selected, wave_best)
        self.assertNotEqual(selected, derivative_best)
        self.assertAlmostEqual(float(result["waveform_corr"][0]), 1 - expected[1], places=7)
        self.assertAlmostEqual(float(result["derivative_corr"][0]), 1 - expected[2], places=7)
        self.assertNotAlmostEqual(float(result["waveform_corr"][0]), 1 - max(row[1] for row in candidates), places=5)
        self.assertNotAlmostEqual(float(result["derivative_corr"][0]), 1 - max(row[2] for row in candidates), places=5)

    def test_perfect_signal_components_zero(self):
        torch.manual_seed(3)
        target = torch.randn(3, 300, dtype=torch.float64)
        result = same_lag_waveform_derivative_loss(target, target, 15)
        for key in ("waveform_corr", "derivative_corr"):
            torch.testing.assert_close(result[key], torch.zeros(3, dtype=torch.float64), atol=1e-8, rtol=0)
        torch.testing.assert_close(result["selected_lag"], torch.zeros(3, dtype=torch.long))

    def test_inverted_polarity_is_not_perfect(self):
        torch.manual_seed(4)
        target = torch.randn(2, 300)
        result = same_lag_waveform_derivative_loss(-target, target, 0)
        torch.testing.assert_close(result["waveform_corr"], torch.full((2,), 2.0), atol=1e-6, rtol=0)
        result = same_lag_waveform_derivative_loss(-target, target, 15)
        self.assertTrue((result["waveform_corr"] > .5).all())

    def test_short_and_constant_windows_are_finite(self):
        for length in (1, 2, 3, 4, 8):
            pred = torch.ones(2, length, requires_grad=True)
            parts = waveform_v1_training_loss(pred, torch.ones_like(pred), torch.tensor([72., 75.]),
                                             fs=30, return_components=True)
            self.assertTrue(all(torch.isfinite(v) for v in parts.values()))
            self.assertEqual(float(parts["waveform_corr"].detach()), 1)
            self.assertEqual(float(parts["derivative_corr"].detach()), 1)
            parts["total"].backward()
            self.assertTrue(torch.isfinite(pred.grad).all())

    def test_broad_spectral_band_and_nyquist(self):
        self.assertEqual(waveform_spectral_band(30), (.7, 8.0))
        self.assertEqual(waveform_spectral_band(10), (.7, 4.5))
        from rppg_lab import losses
        for fs, expected_high in ((30, 8.), (10, 4.5)):
            x = torch.randn(2, 300)
            with patch.object(losses, "_band_psd", wraps=losses._band_psd) as psd:
                distance = waveform_spectral_distance(x, x, fs, fmax=100.) if fs == 10 else waveform_spectral_distance(x, x, fs)
            for call in psd.call_args_list:
                self.assertEqual(call.args[2:], (.7, expected_high))
                self.assertLess(call.args[3], fs / 2)
            torch.testing.assert_close(distance, torch.zeros(2))
        time = torch.arange(300) / 30
        target = torch.sin(2 * torch.pi * 1.2 * time)[None]
        pred = target + .8 * torch.sin(2 * torch.pi * 6 * time)[None]
        broad = waveform_spectral_distance(pred, target, 30)
        narrow = waveform_spectral_distance(pred, target, 30, fmax=3.5)
        self.assertGreater(float(broad), float(narrow) + .1)

    def test_reported_components_exact_weighted_total(self):
        cfg = replace(WaveformV1LossConfig(), w_corr=.9, w_d1=.3, w_spec=.12, w_hr=.07, w_res=.03)
        pred, target = torch.randn(2, 300), torch.randn(2, 300)
        residuals = torch.randn(2, 3, 300)
        valid = torch.tensor([[True, False, True], [True, True, False]])
        parts = waveform_v1_training_loss(pred, target, torch.tensor([72., 75.]), 30, residuals, valid, cfg, True)
        self.assertEqual(set(parts), {"total", "waveform_corr", "derivative_corr", "spectral", "hr", "residual"})
        expected = (cfg.w_corr * parts["waveform_corr"] + cfg.w_d1 * parts["derivative_corr"]
                    + cfg.w_spec * parts["spectral"] + cfg.w_hr * parts["hr"] + cfg.w_res * parts["residual"])
        torch.testing.assert_close(parts["total"], expected)
        scalar = waveform_v1_training_loss(pred, target, torch.tensor([72., 75.]), 30, residuals, valid, cfg)
        torch.testing.assert_close(scalar, expected)

    def test_invalid_residuals_never_enter_regularization(self):
        pred, target = torch.randn(2, 120), torch.randn(2, 120)
        valid = torch.tensor([[True, False, True], [False, True, False]])
        residuals = torch.ones(2, 3, 120)
        residuals[~valid] = torch.nan
        residuals.requires_grad_()
        parts = waveform_v1_training_loss(pred, target, torch.tensor([72., 75.]), 30,
                                          residuals, valid, return_components=True)
        self.assertEqual(float(parts["residual"].detach()), 1)
        parts["total"].backward()
        self.assertEqual(residuals.grad[~valid].count_nonzero().item(), 0)
        self.assertTrue((residuals.grad[valid] != 0).all())
        prepared = waveform_v1_training_loss(pred, target, torch.tensor([72., 75.]), 30,
                                             residuals[valid], return_components=True)
        torch.testing.assert_close(prepared["total"], parts["total"])

    def test_invalid_configuration_is_rejected(self):
        for changes in (dict(w_corr=-1), dict(w_hr=np.nan), dict(max_lag_sec=-.1),
                        dict(spectral_fmax_hz=.5), dict(derivative_internal_weight=-.2)):
            with self.assertRaises(ValueError):
                replace(WaveformV1LossConfig(), **changes)


class WaveformV1MetricTests(unittest.TestCase):
    def test_derivative_metric_uses_waveform_selected_lag(self):
        pred, target = adversarial_signals()
        metrics = waveform_window_metrics(pred, target, 30, max_lag_sec=8 / 30)
        self.assertEqual(metrics["selected_lag_samples"], 2)
        self.assertAlmostEqual(metrics["selected_lag_ms"], 2000 / 30)
        p, t = aligned_overlap(pred, target, 2)
        self.assertAlmostEqual(metrics["d1_corr_same_lag"], np.corrcoef(np.diff(p), np.diff(t))[0, 1])
        self.assertLess(metrics["d1_corr_same_lag"], .8)
        expected_rmse = np.sqrt(np.mean(((p - p.mean()) / p.std() - (t - t.mean()) / t.std()) ** 2))
        self.assertAlmostEqual(metrics["aligned_nrmse"], expected_rmse)

    def test_perfect_metrics_and_nonzero_physiological_lag(self):
        t = np.arange(300) / 30
        target = np.sin(2 * np.pi * 1.2 * t) + .25 * np.sin(2 * np.pi * 2.4 * t)
        perfect = waveform_window_metrics(target * 3 + 10, target, 30)
        self.assertAlmostEqual(perfect["wave_corr"], 1)
        self.assertAlmostEqual(perfect["aligned_nrmse"], 0)
        self.assertAlmostEqual(perfect["spectral_similarity"], 1, places=7)
        self.assertAlmostEqual(perfect["hr_mae"], 0)
        delayed = np.r_[np.zeros(4), target[:-4]]
        shifted = waveform_window_metrics(delayed, target, 30)
        self.assertEqual(shifted["selected_lag_samples"], 4)
        self.assertAlmostEqual(shifted["d1_corr_same_lag"], 1)
        self.assertLess(shifted["wave_corr"], shifted["wave_corr_aligned"])

    def test_undefined_metrics_are_nan_not_zero(self):
        for p, t in ((np.ones(300), np.ones(300)), (np.array([1., 2.]), np.array([1., 2.])),
                     (np.full(300, np.nan), np.arange(300)), (np.array([]), np.array([]))):
            result = waveform_window_metrics(p, t, 30)
            for key in ("d1_corr_same_lag", "aligned_nrmse", "selected_lag_samples", "spectral_distance", "hr_mae"):
                self.assertTrue(np.isnan(result[key]), key)
        # No deletion of a missing sample to manufacture adjacent derivatives.
        pred = np.sin(np.arange(300))
        pred[150] = np.nan
        result = waveform_window_metrics(pred, np.sin(np.arange(300)), 30)
        self.assertTrue(np.isnan(result["wave_corr_aligned"]))

    def test_subject_balance_finite_counts_and_selection(self):
        rows = [dict(subject_id="A", wave_corr_aligned=.9, d1_corr_same_lag=.8, hr_mae=1.,
                     aligned_nrmse=.2, spectral_distance=.1) for _ in range(9)]
        rows += [dict(subject_id="B", wave_corr_aligned=.1, d1_corr_same_lag=.2, hr_mae=9.,
                      aligned_nrmse=1., spectral_distance=.5)]
        result = aggregate_waveform_metrics(rows)
        self.assertAlmostEqual(result["window_metrics"]["hr_mae"], 1.8)
        self.assertAlmostEqual(result["subject_balanced_metrics"]["hr_mae"], 5.)
        self.assertAlmostEqual(result["subject_balanced_metrics"]["wave_corr_aligned"], .5)
        self.assertAlmostEqual(morphology_score(result), .5)
        self.assertEqual(result["valid_windows_per_subject"], {"A": 9, "B": 1})
        rows.append(dict(subject_id="B", wave_corr_aligned=np.nan, d1_corr_same_lag=np.nan, hr_mae=np.nan))
        result = aggregate_waveform_metrics(rows)
        self.assertEqual(result["per_subject"]["B"]["finite_window_counts"]["hr_mae"], 1)
        self.assertAlmostEqual(result["subject_balanced_metrics"]["hr_mae"], 5.)

    def test_coarse_morphology_is_diagnostic_only(self):
        t = np.arange(300) / 30
        target = np.sin(2 * np.pi * 1.2 * t)
        result = waveform_window_metrics(target, target, 30)
        for name in ("width50_mean_sec", "upstroke_mean_sec", "fall_time_mean_sec", "area_mean", "ibi_mean_sec"):
            self.assertTrue(np.isfinite(result[f"{name}_abs_error"]))
            self.assertEqual(result[f"{name}_abs_error"], 0)


def epoch_metrics(corr, d1, hr, **diagnostics):
    return dict(subject_balanced_metrics=dict(wave_corr_aligned=corr, d1_corr_same_lag=d1, hr_mae=hr), **diagnostics)


class WaveformSelectionTests(unittest.TestCase):
    def test_best_eligible_morphology_not_best_hr(self):
        selector = WaveformCheckpointSelector(1.)
        epochs = [epoch_metrics(.5, .5, 3.), epoch_metrics(.55, .45, 1.),
                  epoch_metrics(.99, .99, 4.01), epoch_metrics(.8, .8, 3.9)]
        decisions = [selector.consider(i, metrics) for i, metrics in enumerate(epochs)]
        self.assertEqual(selector.best_candidate["epoch"], 3)
        self.assertEqual(selector.best_unconstrained["epoch"], 2)
        self.assertEqual(selector.best_hr["epoch"], 1)
        self.assertFalse(decisions[2]["hr_eligible"])
        self.assertTrue(decisions[0]["hr_eligible"])
        self.assertTrue(decisions[3]["hr_eligible"])

    def test_exact_gate_boundary_and_nan_ineligibility(self):
        selector = WaveformCheckpointSelector(1)
        selector.consider(0, epoch_metrics(.5, .5, 3))
        self.assertTrue(selector.consider(1, epoch_metrics(.8, .8, 4))["hr_eligible"])
        self.assertFalse(selector.consider(2, epoch_metrics(.9, .9, np.nan))["hr_eligible"])
        self.assertFalse(selector.consider(3, epoch_metrics(np.nan, .9, 2))["hr_eligible"])
        self.assertEqual(selector.best_candidate["epoch"], 1)

    def test_score_tie_prefers_aligned_then_hr_then_earlier(self):
        selector = WaveformCheckpointSelector(10)
        selector.consider(0, epoch_metrics(.6, .4, 3))
        selector.consider(1, epoch_metrics(.7, .3 + 1e-9, 4))
        self.assertEqual(selector.best_candidate["epoch"], 1)
        selector.consider(2, epoch_metrics(.7, .3, 2))
        self.assertEqual(selector.best_candidate["epoch"], 2)
        selector.consider(3, epoch_metrics(.7, .3, 2))
        self.assertEqual(selector.best_candidate["epoch"], 2)
        earlier = dict(selector.best_candidate, epoch=1)
        self.assertTrue(better_morphology(earlier, selector.best_candidate))

    def test_diagnostics_and_lag_never_enter_score(self):
        metrics = epoch_metrics(.8, .6, 3, selected_lag_samples=15, quality=0,
                                spectral_distance=1, upstroke_abs_error=100)
        self.assertAlmostEqual(morphology_score(metrics), .7)
        metrics.update(selected_lag_samples=0, quality=1, spectral_distance=0, upstroke_abs_error=0)
        self.assertAlmostEqual(morphology_score(metrics), .7)

    def test_epoch_zero_required_and_finite(self):
        with self.assertRaises(ValueError):
            WaveformCheckpointSelector().consider(1, epoch_metrics(.8, .8, 3))
        with self.assertRaises(ValueError):
            WaveformCheckpointSelector().consider(0, epoch_metrics(np.nan, .8, 3))

    def test_unconstrained_diagnostic_can_retain_undefined_hr(self):
        selector = WaveformCheckpointSelector()
        selector.consider(0, epoch_metrics(.5, .5, 3))
        decision = selector.consider(1, epoch_metrics(.9, .9, np.nan))
        self.assertFalse(decision["hr_eligible"])
        self.assertEqual(selector.best_candidate["epoch"], 0)
        self.assertEqual(selector.best_unconstrained["epoch"], 1)

    def test_epoch0_reference_cannot_drift_with_caller_mutations(self):
        selector = WaveformCheckpointSelector()
        reference = epoch_metrics(.5, .5, 3.)
        selector.consider(0, reference)
        reference["subject_balanced_metrics"]["hr_mae"] = 100.
        self.assertFalse(selector.consider(1, epoch_metrics(.9, .9, 5.))["hr_eligible"])
        self.assertEqual(selector.epoch0_reference_metrics["subject_balanced_metrics"]["hr_mae"], 3.)


class WaveformSplitTests(unittest.TestCase):
    def test_deterministic_three_way_subject_split(self):
        ids = [f"subject{i}" for i in range(20)]
        result = waveform_v1_split_manifest(ids)
        self.assertEqual(result, waveform_v1_split_manifest(ids[::-1]))
        self.assertEqual(len(result["test_ids"]), 4)
        self.assertEqual(len(result["validation_ids"]), 3)
        self.assertEqual(len(result["train_ids"]), 13)
        self.assertNotEqual(result["test_ids"], waveform_v1_split_manifest(ids, seed=5)["test_ids"])
        self.assertEqual(set(ids), set(result["train_ids"] + result["validation_ids"] + result["test_ids"]))
        validate_waveform_split_manifest(result)

    def test_minimum_cohort_and_explicit_two_subject_smoke(self):
        manifest = waveform_v1_split_manifest(["A", "B", "C"])
        self.assertEqual([len(manifest[k]) for k in ("train_ids", "validation_ids", "test_ids")], [1, 1, 1])
        manifest = waveform_v1_split_manifest(["A", "B"])
        self.assertTrue(manifest["smoke_only"])
        self.assertEqual(manifest["test_ids"], [])

    def test_overlap_duplicate_and_incomplete_manifest_rejected(self):
        for ids in (["A"], ["A", "A", "B"], ["A", "", "B"]):
            with self.assertRaises(ValueError):
                waveform_v1_split_manifest(ids)
        original = waveform_v1_split_manifest(["A", "B", "C", "D", "E"])
        for mutation in (dict(test_ids=original["train_ids"]), dict(all_subject_ids=["A"]), dict(algorithm_version="unknown")):
            with self.assertRaises(ValueError):
                validate_waveform_split_manifest({**original, **mutation})

    def test_evaluator_requires_explicit_split(self):
        base = ["--checkpoint", "model.pt", "--ubfc-root", "UBFC", "--out", "results.json"]
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            evaluation.build_parser().parse_args(base)
        self.assertEqual(evaluation.build_parser().parse_args(base + ["--split", "val"]).split, "val")
        self.assertEqual(evaluation.build_parser().parse_args(base + ["--split", "test"]).split, "test")


class WaveformTrainingIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Reuse Phase 2 acquisition fixtures without changing their expectations.
        from test_waveform_multi_roi import synthetic_result
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.output = cls.root / "run"
        cls.result = synthetic_result(missing={"left_cheek": slice(None)})
        video = cls.root / "vid.avi"
        video.write_bytes(b"synthetic-video")
        gt = cls.root / "ground_truth.txt"
        t = cls.result.shared.timestamps
        np.savetxt(gt, np.vstack((np.sin(2 * np.pi * 1.2 * t), np.full(len(t), 72.), t)))
        cls.subjects = [UBFCSubject(f"subject{i}", video, gt) for i in range(1, 6)]
        cls.dataset_subject_calls = []
        cls.expected_outputs = {}
        torch.manual_seed(50)
        cls.input = dict(x=torch.randn(2, 3, 9, 300), roi_quality=torch.ones(2, 3),
                         roi_valid=torch.tensor([[True, False, True], [False, True, True]]),
                         prior_valid=torch.ones(2, 3, 6, dtype=torch.bool))
        save = training.save_checkpoint

        def capture_save(path, model, *args):
            with torch.no_grad():
                cls.expected_outputs[path.name] = model(**cls.input).clone()
            save(path, model, *args)

        def make_dataset(subjects, **kwargs):
            cls.dataset_subject_calls.append([s.subject_id for s in subjects])
            return UBFCMultiROIRPPGDataset(subjects, **kwargs)

        args = ["train", "--ubfc-root", str(cls.root), "--out-dir", str(cls.output),
                "--cache-dir", str(cls.root / "cache"), "--epochs", "1", "--batch-size", "2",
                "--base-channels", "8", "--num-blocks", "1", "--dropout", "0", "--stride-sec", "10"]
        with patch.object(training, "find_ubfc_subjects", return_value=cls.subjects), \
             patch.object(training, "UBFCMultiROIRPPGDataset", side_effect=make_dataset), \
             patch.object(training, "save_checkpoint", side_effect=capture_save), \
             patch("rppg_lab.extraction_cache.process_recording", return_value=cls.result), \
             patch("torch.cuda.is_available", return_value=False), patch("sys.argv", args), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            training.main()
        cls.manifest = json.loads((cls.output / "waveform_v1_split.json").read_text())
        cls.log = json.loads((cls.output / "training_log.json").read_text())
        cls.checkpoint = torch.load(cls.output / "best_waveform_candidate.pt", map_location="cpu", weights_only=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_training_never_constructs_test_dataset_or_logs_test_metrics(self):
        self.assertEqual(self.dataset_subject_calls, [self.manifest["train_ids"], self.manifest["validation_ids"]])
        for group in self.dataset_subject_calls:
            self.assertFalse(set(group) & set(self.manifest["test_ids"]))
        for epoch in self.log["epochs"]:
            serialized = json.dumps(epoch)
            self.assertNotIn("test", serialized)
            self.assertIn("subject_balanced_metrics", epoch)
            self.assertIn("hr_eligible", epoch)
        self.assertEqual(self.log["metadata"]["test_subjects"], self.manifest["test_ids"])

    def test_default_objective_optimizer_smoke_and_complete_component_log(self):
        self.assertEqual(training.build_parser().parse_args(["--ubfc-root", "UBFC"]).objective, "waveform_v1")
        first, trained = self.log["epochs"]
        self.assertEqual(first["residual_contribution_max_abs"], 0)
        self.assertGreater(trained["residual_contribution_max_abs"], 0)
        parts = trained["train_components"]
        self.assertTrue(all(np.isfinite(value) for value in parts.values()))
        expected = (parts["waveform_corr"] + .25 * parts["derivative_corr"] + .1 * parts["spectral"]
                    + .1 * parts["hr"] + .02 * parts["residual"])
        self.assertAlmostEqual(parts["total"], expected, places=6)

    def test_all_checkpoint_names_and_required_metadata(self):
        for name in ("best_waveform_candidate.pt", "best_morphology_unconstrained.pt", "best_hr_diagnostic.pt", "last.pt"):
            self.assertTrue((self.output / name).exists(), name)
        expected = {"model_class", "model_config", "model_state", "roi_names", "prior_names", "channel_ordering",
                    "extraction_config", "extraction_cache_version", "fs", "win_sec", "stride_sec", "objective_name",
                    "loss_weights", "max_lag_sec", "spectral_band_hz", "morphology_score_formula",
                    "hr_regression_tolerance_bpm", "split_manifest_path", "split_manifest", "split_manifest_sha256",
                    "train_subjects", "val_subjects", "test_subjects", "epoch", "validation_metrics",
                    "epoch0_reference_metrics", "checkpoint_selection_criterion", "seed", "provenance", "waveform_v1_status"}
        self.assertTrue(expected <= self.checkpoint.keys())
        self.assertEqual(self.checkpoint["waveform_v1_status"], "candidate_not_frozen")
        self.assertEqual(self.checkpoint["loss_weights"], dict(w_corr=1., w_d1=.25, w_spec=.1, w_hr=.1, w_res=.02))
        self.assertEqual(split_manifest_hash(self.manifest), self.checkpoint["split_manifest_sha256"])

    def test_selected_checkpoint_roundtrip_exact_outputs(self):
        rebuilt = reconstruct_waveform_model(self.checkpoint)
        with torch.no_grad():
            output = rebuilt(**self.input)
        torch.testing.assert_close(output, self.expected_outputs["best_waveform_candidate.pt"], atol=0, rtol=0)

    def test_incompatible_roi_prior_channel_cache_or_manifest_refused(self):
        cp = self.checkpoint
        mutations = [dict(roi_names=cp["roi_names"][::-1]), dict(prior_names=cp["prior_names"][::-1]),
                     dict(channel_ordering=cp["channel_ordering"][::-1]), dict(extraction_cache_version="wrong"),
                     dict(fs=60.), dict(spectral_band_hz=[.7, 3.5]), dict(split_manifest_sha256="wrong")]
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                checkpoint_dataset_config({**cp, **mutation})
        mismatch = waveform_v1_split_manifest(self.manifest["all_subject_ids"], seed=100)
        with self.assertRaises(ValueError):
            checkpoint_dataset_config(cp, mismatch)

    def test_validation_evaluator_reconstructs_saved_configuration(self):
        calls = []

        def make_dataset(subjects, **kwargs):
            calls.append(([s.subject_id for s in subjects], kwargs))
            return UBFCMultiROIRPPGDataset(subjects, **kwargs)

        output = self.root / "validation.json"
        argv = ["evaluate", "--checkpoint", str(self.output / "best_waveform_candidate.pt"),
                "--split-manifest", str(self.output / "waveform_v1_split.json"), "--split", "val",
                "--ubfc-root", str(self.root), "--cache-dir", str(self.root / "cache"),
                "--out", str(output), "--device", "cpu"]
        with patch.object(evaluation, "find_ubfc_subjects", return_value=self.subjects), \
             patch.object(evaluation, "UBFCMultiROIRPPGDataset", side_effect=make_dataset), \
             patch("rppg_lab.extraction_cache.process_recording", return_value=self.result), \
             patch("sys.argv", argv), redirect_stdout(io.StringIO()):
            evaluation.main()
        result = json.loads(output.read_text())
        self.assertEqual(result["split"], "val")
        self.assertEqual(result["subject_ids"], self.manifest["validation_ids"])
        self.assertEqual(calls[0][0], self.manifest["validation_ids"])
        self.assertEqual(calls[0][1]["stride_sec"], self.checkpoint["stride_sec"])
        actual = result["metrics"]["subject_balanced_metrics"]
        expected = self.checkpoint["validation_metrics"]["subject_balanced_metrics"]
        for key in ("hr_mae", "wave_corr_aligned", "d1_corr_same_lag", "aligned_nrmse", "spectral_distance"):
            self.assertAlmostEqual(actual[key], expected[key], places=7)
        self.assertEqual(len(result["metrics"]["window_results"]), result["metrics"]["windows"])

    def test_phase2_ablation_uses_unchanged_legacy_loss(self):
        model = reconstruct_waveform_model(self.checkpoint)
        batch = {**self.input, "y_ppg": torch.randn(2, 300), "y_hr": torch.tensor([72., 75.])}
        with torch.no_grad():
            out = training.forward_batch(model, batch, torch.device("cpu"))
        args = training.build_parser().parse_args(["--ubfc-root", "UBFC", "--objective", "phase2_rppg"])
        parts = training.training_components(out, batch, "cpu", args, WaveformV1LossConfig())
        expected = rppg_training_loss(out["ppg"], batch["y_ppg"], batch["y_hr"], 30,
                                     residual=out["residuals"][batch["roi_valid"]])
        torch.testing.assert_close(parts["total"], expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
