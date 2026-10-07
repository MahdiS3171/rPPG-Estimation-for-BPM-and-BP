"""Part 4 artifact review and locked-test isolation regressions."""
from __future__ import annotations

from contextlib import contextmanager, redirect_stdout, redirect_stderr
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from rppg_lab.artifacts import file_sha256, write_json
from rppg_lab.waveform_baselines import classical_window_predictions
from rppg_lab.waveform_freeze import freeze_waveform_candidate
from rppg_lab.waveform_review import (MAIN_METRICS, build_validation_review, compare_metrics,
    build_locked_test_review, metric_summary, reserve_locked_test, subject_bootstrap, validation_gate, write_review)
from rppg_lab.waveform_training import reconstruct_waveform_model
from rppg_lab.waveform_metrics import forward_batch
from rppg_lab.waveform_inference import FrozenWaveformExtractor
from rppg_lab.extraction_cache import FaceROIExtraction
from scripts import evaluate_waveform_multi_roi_ubfc as evaluation
from scripts import review_waveform_v1_candidate as review_script


def gate_metrics(corr=.6, d1=.5, hr=2.):
    s = dict(wave_corr_aligned=corr, d1_corr_same_lag=d1, aligned_nrmse=.4,
             spectral_distance=.2, spectral_similarity=.8, hr_mae=hr, windows=3, valid_windows=3,
             finite_window_counts={k: 3 for k in MAIN_METRICS if k != "morphology_score"})
    return dict(subject_balanced_metrics=deepcopy(s), window_metrics=deepcopy(s),
        per_subject={"A": deepcopy(s)}, valid_windows_per_subject={"A": 3},
        residual_diagnostics_all_finite=True, residual_contribution_overall_mean_abs=.1,
        residual_contribution_max_abs=.5, residual_contribution_mean_abs={"forehead": .1},
        residual_contribution_max_abs_per_roi={"forehead": .5})


class ReviewMathTests(unittest.TestCase):
    def test_epoch0_comparison_both_weightings_and_score(self):
        before, after = gate_metrics(), gate_metrics(.8, .7, 2.5)
        after["window_metrics"]["hr_mae"] = 1.5
        changes = compare_metrics(before, after)
        self.assertAlmostEqual(changes["subject_balanced"]["selected_minus_epoch0"]["morphology_score"], .2)
        self.assertEqual(changes["subject_balanced"]["selected_minus_epoch0"]["hr_mae"], .5)
        self.assertEqual(changes["window_weighted"]["selected_minus_epoch0"]["hr_mae"], -.5)

    def test_bootstrap_subjects_not_window_counts(self):
        subjects = {"A": dict(gate_metrics()["per_subject"]["A"], windows=1000),
                    "B": gate_metrics(.2, .1, 10)["per_subject"]["A"]}
        result = subject_bootstrap(subjects, 2000, 42)
        self.assertEqual(result["unit"], "subject")
        self.assertEqual(result["intervals"]["hr_mae"]["estimate"], 6.)
        subjects["A"]["windows"] = 1
        self.assertEqual(result, subject_bootstrap(subjects, 2000, 42))

    def test_bootstrap_deterministic_order_invariant(self):
        subjects = {str(i): gate_metrics(.1 * i, .12 * i, i)["per_subject"]["A"] for i in range(1, 6)}
        first = subject_bootstrap(subjects, 71, 42)
        self.assertEqual(first, subject_bootstrap(dict(reversed(list(subjects.items()))), 71, 42))
        self.assertNotEqual(first, subject_bootstrap(subjects, 71, 43))

    def test_bootstrap_does_not_drop_failed_subject(self):
        subjects = {"A": gate_metrics()["per_subject"]["A"], "B": {}}
        self.assertTrue(all(v is None for v in subject_bootstrap(subjects)["intervals"].values()))

    def test_validation_gate_accepts_engineering_boundaries(self):
        cp = dict(validation_metrics=dict(hr_eligible=True))
        result = validation_gate(gate_metrics(), gate_metrics(.8, .45, 3), cp, ["A"], True)
        self.assertEqual(result["validation_decision"], "ACCEPT_FOR_LOCKED_TEST")

    def test_gate_rejects_no_gain_hr_regression_and_large_d1_loss(self):
        cp = dict(validation_metrics=dict(hr_eligible=True))
        for metrics in (gate_metrics(), gate_metrics(.8, .7, 3.0001), gate_metrics(.8, .449),
                        gate_metrics(float("nan")), gate_metrics(.8, .7, float("inf"))):
            self.assertEqual(validation_gate(gate_metrics(), metrics, cp, ["A"], True)
                             ["validation_decision"], "DO_NOT_TEST_OR_FREEZE")

    def test_missing_subject_baselines_or_nonfinite_residual_rejected(self):
        cp = dict(validation_metrics=dict(hr_eligible=True))
        candidate = gate_metrics(.8, .7)
        self.assertEqual(validation_gate(gate_metrics(), candidate, cp, ["A", "B"], True)
                         ["validation_decision"], "DO_NOT_TEST_OR_FREEZE")
        self.assertEqual(validation_gate(gate_metrics(), candidate, cp, ["A"], False)
                         ["validation_decision"], "DO_NOT_TEST_OR_FREEZE")
        candidate["residual_contribution_max_abs"] = None
        self.assertEqual(validation_gate(gate_metrics(), candidate, cp, ["A"], True)
                         ["validation_decision"], "DO_NOT_TEST_OR_FREEZE")

    def test_validity_aware_prior_average_excludes_failed_channels(self):
        x = np.zeros((3, 5, 8))
        x[:, 3] = np.array([1, 10, 3])[:, None]
        x[:, 4] = np.array([2, 99, 4])[:, None]
        roi = np.array([True, False, True])
        prior = np.ones((3, 2), bool)
        prior[2, 1] = False
        predictions = dict(classical_window_predictions(x, roi, prior, ("F", "L", "R"), ("A", "B")))
        self.assertEqual(len(predictions), 8)
        np.testing.assert_equal(predictions["A multi-ROI average"], np.full(8, 2.))
        np.testing.assert_equal(predictions["B multi-ROI average"], np.full(8, 2.))
        self.assertTrue(np.isnan(predictions["L/A"]).all())
        self.assertTrue(np.isnan(predictions["R/B"]).all())


class ReviewArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import test_waveform_v1 as fixtures
        cls.fixture = fixtures.WaveformTrainingIntegrationTests
        cls.fixture.setUpClass()
        f = cls.fixture
        cls.output = f.root / "validation_with_baselines.json"
        argv = ["evaluate", "--checkpoint", str(f.output / "best_waveform_candidate.pt"),
                "--split-manifest", str(f.output / "waveform_v1_split.json"), "--split", "val",
                "--ubfc-root", str(f.root), "--cache-dir", str(f.root / "cache"),
                "--include-baselines", "--out", str(cls.output), "--device", "cpu"]
        with patch.object(evaluation, "find_ubfc_subjects", return_value=f.subjects), \
             patch("rppg_lab.extraction_cache.process_recording", return_value=f.result), \
             patch("sys.argv", argv), redirect_stdout(io.StringIO()):
            evaluation.main()
        cls.artifact = json.loads(cls.output.read_text())

    @classmethod
    def tearDownClass(cls):
        cls.fixture.tearDownClass()

    def make_review(self, artifact=None):
        f = self.fixture
        return build_validation_review(f.log, f.checkpoint, artifact or self.artifact, f.manifest,
            f.output / "best_waveform_candidate.pt", self.output, f.output / "training_log.json",
            f.output / "waveform_v1_split.json", 2000, 42)

    def test_baselines_share_candidate_windows_subjects_targets(self):
        metrics = self.artifact["metrics"]
        identity = lambda rows: [(r["subject_id"], r["start_sec"], r["target_sha256"]) for r in rows]
        expected = identity(metrics["window_results"])
        self.assertEqual(len(metrics["classical_baselines"]), 24)
        for baseline in metrics["classical_baselines"].values():
            self.assertEqual(identity(baseline["window_results"]), expected)
            self.assertEqual(list(baseline["per_subject"]), list(metrics["per_subject"]))

    def test_artifact_review_saved_outputs_and_no_model_inference(self):
        with patch("rppg_lab.models.MultiROIPriorResidualWaveformNet.forward", side_effect=AssertionError("inference")):
            review = self.make_review()
            write_review(review, self.fixture.root / "review.json", self.fixture.root / "review.md")
        self.assertFalse(review["locked_test_consumed"])
        self.assertEqual(review["selected_epoch"], self.fixture.checkpoint["epoch"])
        self.assertEqual(review["validation_bootstrap"]["resamples"], 2000)
        self.assertIn("Locked test was NOT evaluated", (self.fixture.root / "review.md").read_text())

    def test_review_rejects_baseline_target_mismatch(self):
        artifact = deepcopy(self.artifact)
        next(iter(artifact["metrics"]["classical_baselines"].values()))["window_results"][0]["target_sha256"] = "wrong"
        with self.assertRaises(ValueError):
            self.make_review(artifact)

    def test_review_rejects_test_artifact_or_changed_checkpoint_hash(self):
        for mutation in (dict(split="test"), dict(checkpoint_sha256="wrong"), dict(split_manifest_file_sha256="wrong")):
            with self.assertRaises(ValueError):
                self.make_review({**self.artifact, **mutation})

    def test_test_refused_before_dataset_discovery_without_accepted_review(self):
        f = self.fixture
        argv = ["evaluate", "--checkpoint", str(f.output / "best_waveform_candidate.pt"),
                "--split-manifest", str(f.output / "waveform_v1_split.json"), "--split", "test",
                "--ubfc-root", str(f.root), "--out", str(f.root / "locked.json"), "--device", "cpu"]
        rejected = self.make_review()
        rejected["validation_decision"] = "DO_NOT_TEST_OR_FREEZE"
        path = f.root / "rejected.json"
        write_json(path, rejected)
        with patch.object(evaluation, "find_ubfc_subjects", side_effect=AssertionError("test data touched")), \
             patch("sys.argv", argv), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            evaluation.main()
        with patch.object(evaluation, "find_ubfc_subjects", side_effect=AssertionError("test data touched")), \
             patch("sys.argv", argv + ["--validation-review", str(path)]), self.assertRaises(ValueError):
            evaluation.main()
        self.assertFalse((f.output / "locked_test_consumption.json").exists())

    def test_locked_receipt_hashes_and_repeat_refusal(self):
        f = self.fixture
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            candidate, manifest = directory / "candidate.pt", directory / "split.json"
            candidate.write_bytes((f.output / "best_waveform_candidate.pt").read_bytes())
            manifest.write_bytes((f.output / "waveform_v1_split.json").read_bytes())
            review = self.make_review()
            # Receipt policy is tested independently of acceptance performance.
            review.update(validation_decision="ACCEPT_FOR_LOCKED_TEST", checks={"fixture": True})
            review_path = directory / "review.json"
            write_json(review_path, review)
            receipt = reserve_locked_test(candidate, manifest, review_path, directory / "test.json")
            self.assertTrue(receipt["locked_test_consumed"])
            self.assertEqual(receipt["checkpoint_sha256"], file_sha256(candidate))
            self.assertEqual(receipt["split_manifest_file_sha256"], file_sha256(manifest))
            self.assertEqual(receipt["checkpoint_epoch"], f.checkpoint["epoch"])
            self.assertIn("consumed_at_utc", receipt)
            with self.assertRaises(FileExistsError):
                reserve_locked_test(candidate, manifest, review_path, directory / "different_test.json")

    def test_explicit_locked_evaluator_records_hashes_before_synthetic_data_access(self):
        f = self.fixture
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            candidate, manifest = directory / "candidate.pt", directory / "split.json"
            candidate.write_bytes((f.output / "best_waveform_candidate.pt").read_bytes())
            manifest.write_bytes((f.output / "waveform_v1_split.json").read_bytes())
            review = self.make_review()
            review.update(validation_decision="ACCEPT_FOR_LOCKED_TEST", checks={"fixture": True})
            review_path, output = directory / "review.json", directory / "test.json"
            write_json(review_path, review)

            def discover(_):
                self.assertTrue((directory / "locked_test_consumption.json").exists())
                return f.subjects

            argv = ["evaluate", "--checkpoint", str(candidate), "--split-manifest", str(manifest),
                "--split", "test", "--validation-review", str(review_path), "--ubfc-root", str(f.root),
                "--cache-dir", str(f.root / "cache"), "--out", str(output), "--device", "cpu"]
            with patch.object(evaluation, "find_ubfc_subjects", side_effect=discover), \
                 patch("rppg_lab.extraction_cache.process_recording", return_value=f.result), \
                 patch("sys.argv", argv), redirect_stdout(io.StringIO()):
                evaluation.main()
            artifact = json.loads(output.read_text())
            self.assertEqual(artifact["split"], "test")
            self.assertEqual(artifact["subject_ids"], f.manifest["test_ids"])
            self.assertTrue(artifact["locked_test_consumed"])
            self.assertEqual(artifact["checkpoint_sha256"], file_sha256(candidate))
            self.assertEqual(artifact["split_manifest_file_sha256"], file_sha256(manifest))
            self.assertEqual(artifact["epoch"], f.checkpoint["epoch"])
            self.assertIn("consumed_at_utc", artifact["locked_test_receipt"])
            with patch.object(evaluation, "find_ubfc_subjects", side_effect=AssertionError("repeat data access")), \
                 patch("sys.argv", argv), self.assertRaises(FileExistsError):
                evaluation.main()

    def test_review_cli_has_no_test_or_inference_arguments(self):
        parser = review_script.build_parser()
        self.assertNotIn("split", {a.dest for a in parser._actions})
        self.assertNotIn("ubfc_root", {a.dest for a in parser._actions})

    def test_freeze_impossible_without_locked_test_artifact(self):
        review = self.make_review()
        review.update(validation_decision="ACCEPT_FOR_LOCKED_TEST", checks={"fixture": True},
                      freeze_decision="FREEZE_WAVEFORM_V1")
        with self.assertRaises(ValueError):
            freeze_waveform_candidate(self.fixture.output / "best_waveform_candidate.pt",
                self.fixture.output / "waveform_v1_split.json", review, self.fixture.root / "no_freeze")
        self.assertFalse((self.fixture.root / "no_freeze" / "waveform_v1_frozen.pt").exists())

    def synthetic_locked_review(self, directory, passed=True):
        f = self.fixture
        candidate, manifest = directory / "candidate.pt", directory / "split.json"
        candidate.write_bytes((f.output / "best_waveform_candidate.pt").read_bytes())
        manifest.write_bytes((f.output / "waveform_v1_split.json").read_bytes())
        review = self.make_review()
        review.update(validation_decision="ACCEPT_FOR_LOCKED_TEST", checks={"fixture": True})
        review_path, test_path = directory / "review.json", directory / "test.json"
        write_json(review_path, review)
        receipt = reserve_locked_test(candidate, manifest, review_path, test_path)
        # Synthetic test fixture only; no held-out production dataset is loaded.
        test = deepcopy(self.artifact)
        old_sid = f.manifest["validation_ids"][0]
        new_sid = f.manifest["test_ids"][0]
        test.update(split="test", subject_ids=f.manifest["test_ids"], locked_test_consumed=True,
                    locked_test_receipt=receipt)
        metrics = test["metrics"]
        metrics["per_subject"] = {new_sid: metrics["per_subject"][old_sid]}
        metrics["valid_windows_per_subject"] = {new_sid: metrics["valid_windows_per_subject"][old_sid]}
        for row in metrics["window_results"]:
            row["subject_id"] = new_sid
        write_json(test_path, test)
        judgments = {k: dict(passed=passed, reason="Synthetic quantitative fixture evidence") for k in (
            "hr_usable_without_catastrophic_regression", "waveform_correlations_do_not_collapse",
            "no_broad_systematic_subject_failure")}
        final = build_locked_test_review(review, f.checkpoint, test, candidate, manifest, test_path, judgments)
        return candidate, manifest, final

    def test_locked_failure_rejects_freeze_without_another_epoch(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate, manifest, review = self.synthetic_locked_review(Path(tmp), passed=False)
            self.assertEqual(review["freeze_decision"], "REJECTED_AFTER_LOCKED_TEST")
            self.assertEqual(review["selected_epoch"], self.fixture.checkpoint["epoch"])
            with self.assertRaises(ValueError):
                freeze_waveform_candidate(candidate, manifest, review, Path(tmp) / "frozen")

    def test_frozen_state_and_reconstructed_outputs_exact_with_complete_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            candidate, manifest, review = self.synthetic_locked_review(directory)
            result = freeze_waveform_candidate(candidate, manifest, review, directory / "frozen")
            frozen = torch.load(result["model_path"], map_location="cpu", weights_only=True)
            cp = self.fixture.checkpoint
            self.assertNotIn("optimizer_state", frozen)
            self.assertEqual(frozen["waveform_v1_status"], "frozen_v1")
            self.assertEqual(set(cp["model_state"]), set(frozen["model_state"]))
            for name, tensor in cp["model_state"].items():
                other = frozen["model_state"][name]
                self.assertTrue(torch.equal(tensor.contiguous().reshape(-1).view(torch.uint8),
                                            other.contiguous().reshape(-1).view(torch.uint8)))
            with torch.no_grad():
                before = reconstruct_waveform_model(cp)(**self.fixture.input, return_dict=True)
                after = reconstruct_waveform_model(frozen)(**self.fixture.input, return_dict=True)
            for key in before:
                torch.testing.assert_close(before[key], after[key], atol=0, rtol=0)
            required = {"model_sha256", "source_candidate_sha256", "git_commit", "model_class", "model_config",
                "roi_ordering", "prior_ordering", "channel_ordering", "fs", "window_length_samples",
                "extraction_config", "loss_config", "selected_epoch", "train_ids", "validation_ids", "test_ids",
                "validation_metrics", "test_metrics", "validation_bootstrap", "test_bootstrap",
                "split_manifest_sha256", "split_manifest_file_sha256", "date", "status"}
            self.assertTrue(required <= result.keys())
            self.assertEqual(result["model_sha256"], file_sha256(result["model_path"]))
            with self.assertRaises(FileExistsError):
                freeze_waveform_candidate(candidate, manifest, review, directory / "frozen")

    @contextmanager
    def inference_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            candidate, manifest, review = self.synthetic_locked_review(directory)
            frozen = freeze_waveform_candidate(candidate, manifest, review, directory / "frozen")
            extractor = FrozenWaveformExtractor(frozen["model_path"])
            extraction = FaceROIExtraction(self.fixture.result.shared,
                {name: region.rgb for name, region in self.fixture.result.face_rois.items()})
            yield directory, extractor, extraction, frozen

    def test_inference_matches_training_window_processing_and_exact_candidate_output(self):
        from rppg_lab.datasets import UBFCMultiROIRPPGDataset
        from rppg_lab.waveform_training import checkpoint_dataset_config
        with self.inference_fixture() as (_, extractor, extraction, frozen):
            f = self.fixture
            with patch("rppg_lab.extraction_cache.process_recording", return_value=f.result):
                dataset = UBFCMultiROIRPPGDataset([f.subjects[0]], cache_dir=None,
                    **checkpoint_dataset_config(f.checkpoint))
            item = dataset[0]
            inputs = extractor.prepare_window(extraction, float(item["start_sec"]))
            for key in ("x", "roi_quality", "roi_valid", "prior_valid"):
                torch.testing.assert_close(inputs[key][0], item[key], atol=0, rtol=0)
            result = extractor.infer_window(extraction, float(item["start_sec"]))
            with torch.no_grad():
                before = forward_batch(
                    reconstruct_waveform_model(f.checkpoint), inputs, torch.device("cpu"))
            np.testing.assert_array_equal(result["waveform"], before["ppg"][0].numpy())
            np.testing.assert_array_equal(result["roi_attention"], before["roi_attention"][0].numpy())
            np.testing.assert_array_equal(result["prior_weights"], before["prior_weights"][0].numpy())
            self.assertEqual(result["metadata"]["model_sha256"], frozen["model_sha256"])
            self.assertEqual(result["metadata"]["waveform_v1_status"], "frozen_v1")
            self.assertNotIn("quality", result)
            self.assertEqual(len(result["timestamps"]), 300)

    def test_inference_one_invalid_roi_is_fully_masked(self):
        with self.inference_fixture() as (_, extractor, extraction, _):
            result = extractor.infer_window(extraction)
            np.testing.assert_array_equal(result["roi_valid"], [True, False, True])
            self.assertEqual(result["roi_attention"][1], 0)
            self.assertFalse(result["prior_valid"][1].any())
            self.assertFalse(result["prior_weights"][1].any())
            self.assertFalse(result["scaled_residuals"][1].any())
            self.assertAlmostEqual(float(result["roi_attention"].sum()), 1, places=6)

    def test_inference_priors_see_only_current_window_and_ignore_outside_rgb_changes(self):
        from rppg_lab.window_priors import build_window_priors
        with self.inference_fixture() as (_, extractor, extraction, _):
            with patch("rppg_lab.waveform_inference.build_window_priors", wraps=build_window_priors) as prior:
                first = extractor.infer_window(extraction, 5.)
            self.assertEqual(prior.call_count, 2)
            for call, name in zip(prior.call_args_list, ("forehead", "right_cheek")):
                self.assertEqual(call.args[0].shape, (300, 3))
                np.testing.assert_array_equal(call.args[0], extraction.traces[name].values[150:450])
            changed = deepcopy(extraction)
            for trace in changed.traces.values():
                trace.values[:60] += 100
                trace.values[500:] -= 5
            second = extractor.infer_window(changed, 5.)
            for key in ("waveform", "roi_attention", "prior_weights", "timestamps"):
                np.testing.assert_array_equal(first[key], second[key])

    def test_inference_refuses_roi_prior_channel_manifest_order_mismatch(self):
        with self.inference_fixture() as (_, _, _, frozen):
            manifest_path = Path(frozen["model_path"]).with_name("waveform_v1_freeze_manifest.json")
            original = json.loads(manifest_path.read_text())
            for key in ("roi_ordering", "prior_ordering", "channel_ordering"):
                write_json(manifest_path, {**original, key: original[key][::-1]})
                with self.subTest(key=key), self.assertRaises(ValueError):
                    FrozenWaveformExtractor(frozen["model_path"])

    def test_inference_refuses_extraction_order_grid_or_unsupported_window(self):
        with self.inference_fixture() as (_, extractor, extraction, _):
            reversed_rois = FaceROIExtraction(extraction.shared, dict(reversed(list(extraction.traces.items()))))
            with self.assertRaises(ValueError):
                extractor.infer_window(reversed_rois)
            for start in (.011, 15., float("nan")):
                with self.subTest(start=start), self.assertRaises(ValueError):
                    extractor.infer_window(extraction, start)
            with self.assertRaises(TypeError):
                extractor.infer_window(np.zeros((300, 64, 64, 3)))
            bad_clock = deepcopy(extraction)
            bad_clock.shared.timestamps *= 2
            with self.assertRaises(ValueError):
                extractor.infer_window(bad_clock)

    def test_inference_rejects_unfrozen_candidate_or_wrong_manifest_identity(self):
        with self.inference_fixture() as (directory, _, _, frozen):
            manifest_path = Path(frozen["model_path"]).with_name("waveform_v1_freeze_manifest.json")
            with self.assertRaises(ValueError):
                FrozenWaveformExtractor(directory / "candidate.pt", manifest_path)
            original = json.loads(manifest_path.read_text())
            for mutation in (dict(model_sha256="wrong"), dict(min_valid_fraction=.1),
                             dict(extraction_config={**original["extraction_config"], "max_gap_sec": 10})):
                write_json(manifest_path, {**original, **mutation})
                with self.assertRaises(ValueError):
                    FrozenWaveformExtractor(frozen["model_path"])


if __name__ == "__main__":
    unittest.main()
