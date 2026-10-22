"""Part 4.5A scientific constraints, validation isolation and headless rendering.

All fixtures are synthetic. No local UBFC, official checkpoint or locked-test
recording is opened by this file. Tests never train or create a frozen model.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import csv
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
from scipy.integrate import trapezoid
import torch

from rppg_lab.artifacts import file_sha256, json_safe, write_json
from rppg_lab.config import PipelineConfig
from rppg_lab.datasets import UBFCSubject
from rppg_lab.signals import pulse_wave_features, standardize_1d
from rppg_lab.splits import waveform_v1_split_manifest
from rppg_lab.waveform_alignment import aligned_overlap, lag_candidates
from rppg_lab.waveform_metrics import waveform_window_metrics
from rppg_lab.waveform_training import split_manifest_hash
from rppg_lab import waveform_visualization as visual
from scripts import visualize_waveform_v1 as cli


def synthetic_window(sid="synthetic_validation", start=0., lag=4, noise=.03):
    fs = 30.
    time = np.arange(300) / fs
    ref = np.sin(2 * np.pi * 1.2 * time) + .25 * np.sin(4 * np.pi * 1.2 * time + .3)
    pred = np.r_[np.zeros(lag), ref[:-lag]] if lag > 0 else ref.copy()
    pred += noise * np.sin(2 * np.pi * 5.7 * time)
    signals = {visual.MODEL: pred, visual.CHROM: ref + .15 * np.sin(2 * np.pi * 3.6 * time),
               "OMIT multi-ROI average": -ref}
    metrics = {name: waveform_window_metrics(x, ref, fs) for name, x in signals.items()}
    # Deliberately non-default ordering verifies the figure follows metadata.
    rois, priors = ("right_cheek", "forehead", "left_cheek"), ("OMIT", "CHROM_WIN", "GREEN")
    diagnostics = dict(x=np.zeros((3, 6, 300)), roi_attention=np.array([.2, .5, .3]),
        prior_weights=np.array([[.1, .6, .3]] * 3), roi_valid=np.array([True, True, False]),
        prior_valid=np.array([[True, True, True], [True, False, True], [False, False, False]]),
        roi_source_valid_fraction=np.array([1., .9, .1]), scaled_residuals=np.full((3, 300), .01),
        raw_contact_ppg=ref * 30 + 200, target_sha256="synthetic_target", reference_hr_bpm=72.)
    features = {name: visual.coarse_feature_diagnostics(x, fs) for name, x in signals.items()}
    features["reference"] = visual.coarse_feature_diagnostics(ref, fs)
    return visual.VisualWindow(sid, time + start, ref, signals, metrics, rois, priors, diagnostics, features)


class VisualIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cp = self.root / "waveform_v1_frozen.pt"
        self.cp.write_bytes(b"synthetic checkpoint, not an official result")
        self.freeze = self.root / "waveform_v1_freeze_manifest.json"
        self.split = self.root / "split.json"
        self.manifest = waveform_v1_split_manifest([f"subject{i}" for i in range(8)])
        write_json(self.split, self.manifest)
        self.checkpoint = dict(waveform_v1_status="frozen_v1", freeze_decision="FREEZE_WAVEFORM_V1",
                              selected_epoch=25, split_manifest_file_sha256=file_sha256(self.split))
        self.freeze_value = dict(status="frozen_v1", freeze_decision="FREEZE_WAVEFORM_V1", selected_epoch=25,
            model_sha256=visual.OFFICIAL_MODEL_SHA256, split_manifest_file_sha256=file_sha256(self.split),
            split_manifest_sha256=split_manifest_hash(self.manifest),
            **{k: self.manifest[k] for k in ("train_ids", "validation_ids", "test_ids")})
        write_json(self.freeze, self.freeze_value)

    def verified_fixture(self):
        # Hash spoofing is test-local and not exposed as a production option.
        def fixture_hash(path):
            return visual.OFFICIAL_MODEL_SHA256 if Path(path) == self.cp else file_sha256(path)
        return patch.multiple(visual, file_sha256=fixture_hash,
            checkpoint_dataset_config=lambda *_: {}, FrozenWaveformExtractor=lambda *_args, **_kwargs: "verified fixture")

    def test_exact_official_sha256_enforced_before_load_or_data_access(self):
        with patch.object(torch, "load", side_effect=AssertionError("must reject before load")):
            with self.assertRaisesRegex(ValueError, "SHA256"):
                visual.verify_frozen_identity(self.cp, self.split)

    def test_frozen_status_decision_and_epoch_required(self):
        for mutation in (dict(waveform_v1_status="candidate_not_frozen"), dict(freeze_decision="NO"), dict(selected_epoch=24)):
            with self.subTest(mutation=mutation), self.verified_fixture(), \
                 patch.object(torch, "load", return_value={**self.checkpoint, **mutation}):
                with self.assertRaisesRegex(ValueError, "frozen_v1"):
                    visual.verify_frozen_identity(self.cp, self.split)
        for mutation in (dict(status="candidate"), dict(freeze_decision="NO"), dict(model_sha256="altered"), dict(selected_epoch=26)):
            write_json(self.freeze, {**self.freeze_value, **mutation})
            with self.subTest(mutation=mutation), self.verified_fixture(), patch.object(torch, "load", return_value=self.checkpoint):
                with self.assertRaises(ValueError):
                    visual.verify_frozen_identity(self.cp, self.split)

    def test_matching_identity_delegates_reconstruction_to_frozen_extractor(self):
        with self.verified_fixture(), patch.object(torch, "load", return_value=self.checkpoint), \
             patch.object(visual, "FrozenWaveformExtractor", return_value="extractor") as factory:
            result = visual.verify_frozen_identity(self.cp, self.split)
        self.assertEqual(result[0], "extractor")
        factory.assert_called_once_with(self.cp, self.freeze, device="cpu")

    def test_adjacent_manifest_mandatory_even_with_explicit_external_copy(self):
        copy = self.root / "copy.json"
        copy.write_bytes(self.freeze.read_bytes())
        self.freeze.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "adjacent"):
            visual.verify_frozen_identity(self.cp, self.split, copy)

    def test_changed_manifest_or_split_identity_rejected(self):
        copy = self.root / "copy.json"
        write_json(copy, {**self.freeze_value, "extra": True})
        with self.verified_fixture(), self.assertRaisesRegex(ValueError, "adjacent"):
            visual.verify_frozen_identity(self.cp, self.split, copy)
        for mutation in (dict(split_manifest_file_sha256="wrong"), dict(validation_ids=["intruder"])):
            write_json(self.freeze, {**self.freeze_value, **mutation})
            with self.verified_fixture(), patch.object(torch, "load", return_value=self.checkpoint), self.assertRaisesRegex(ValueError, "split"):
                visual.verify_frozen_identity(self.cp, self.split)

    def test_missing_assets_have_useful_diagnostic(self):
        with self.assertRaisesRegex(FileNotFoundError, "official frozen checkpoint"):
            visual.verify_frozen_identity(self.root / "missing.pt", self.split)
        subject = UBFCSubject("validation_fixture", self.cp, self.split)
        with self.assertRaisesRegex(FileNotFoundError, "Phase 1 extraction cache"), \
             patch("rppg_lab.extraction_cache.process_recording", side_effect=AssertionError("no extraction")):
            visual.require_phase1_cache(subject, dict(extraction_config=PipelineConfig(), max_frames=None), self.root)

    def test_exact_validation_filter_and_train_rejection(self):
        ids = self.manifest["validation_ids"]
        self.assertEqual(visual.validation_subject_ids(self.manifest), sorted(ids))
        self.assertEqual(visual.validation_subject_ids(self.manifest, ids[0]), [ids[0]])
        for subject in (ids[0].upper(), self.manifest["train_ids"][0], "../" + ids[0]):
            with self.subTest(subject=subject), self.assertRaisesRegex(ValueError, "exact validation"):
                visual.validation_subject_ids(self.manifest, subject)

    def test_locked_test_subject_rejected_before_any_dataset_access(self):
        sid = self.manifest["test_ids"][0]
        with patch.object(visual, "verify_frozen_identity", return_value=(None, {}, self.manifest, {}, self.freeze)), \
             patch.object(visual, "validation_subject_paths", side_effect=AssertionError("data touched")), \
             patch.object(visual, "UBFCMultiROIRPPGDataset", side_effect=AssertionError("data touched")):
            with self.assertRaisesRegex(ValueError, "Locked-test"):
                visual.run_visual_validation(ubfc_root=self.root, subject=sid)

    def test_paths_access_only_validation_children_without_discovery(self):
        for sid in self.manifest["validation_ids"]:
            folder = self.root / sid
            folder.mkdir()
            (folder / "vid.avi").write_bytes(b"synthetic video")
            (folder / "ground_truth.txt").write_text("synthetic reference")
        # There are deliberately no train/test directories to stat or open.
        with patch.object(Path, "iterdir", side_effect=AssertionError("cohort enumeration forbidden")):
            subjects = visual.validation_subject_paths(self.root, self.manifest)
        self.assertEqual([s.subject_id for s in subjects], sorted(self.manifest["validation_ids"]))

    def test_cli_has_no_test_training_or_identity_override_options(self):
        destinations = {action.dest for action in cli.build_parser()._actions}
        self.assertFalse(destinations & {"split", "test", "train", "device", "allow_candidate", "expected_sha256", "fs", "max_lag"})
        self.assertTrue({"subject", "selection", "start_sec", "all_validation_subjects"} <= destinations)


class VisualSignalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.window = synthetic_window()

    def test_metric_alignment_exactly_reuses_existing_behavior(self):
        w = self.window
        actual = w.metrics[visual.MODEL]
        expected = waveform_window_metrics(w.signals[visual.MODEL], w.reference, 30)
        self.assertEqual(json_safe(actual), json_safe(expected))
        overlap = visual.waveform_overlap(w)
        p, t = aligned_overlap(w.signals[visual.MODEL], w.reference, int(expected["selected_lag_samples"]))
        np.testing.assert_array_equal(overlap["prediction"], p)
        np.testing.assert_array_equal(overlap["reference"], t)
        self.assertEqual(visual.diagnostic_score(actual), .5 * actual["wave_corr_aligned"] + .5 * actual["d1_corr_same_lag"])

    def test_polarity_is_preserved_including_inverted_prediction(self):
        w = deepcopy(self.window)
        w.signals[visual.MODEL] = -w.reference.copy()
        raw = visual.raw_plot_data(w)
        np.testing.assert_array_equal(raw["prediction"], standardize_1d(-w.reference))
        self.assertLess(np.corrcoef(raw["prediction"], raw["reference"])[0, 1], -.999)

    def test_raw_plot_clock_remains_unshifted_and_arrays_are_copies(self):
        w = self.window
        raw = visual.raw_plot_data(w)
        self.assertNotEqual(w.metrics[visual.MODEL]["selected_lag_samples"], 0)
        np.testing.assert_array_equal(raw["timestamps"], w.timestamps)
        self.assertEqual(len(raw["prediction"]), 300)
        raw["timestamps"][0] = 1234
        self.assertNotEqual(w.timestamps[0], 1234)

    def test_derivative_uses_waveform_lag_even_when_d1_prefers_another(self):
        w = deepcopy(self.window)
        n = np.arange(300)
        w.reference = np.sin(.07 * n) + .1 * np.sin(1.8 * n)
        w.signals[visual.MODEL] = np.sin(.07 * (n - 3)) + .1 * np.sin(1.8 * (n + 2))
        w.metrics[visual.MODEL] = waveform_window_metrics(w.signals[visual.MODEL], w.reference, 30)
        selected = int(w.metrics[visual.MODEL]["selected_lag_samples"])
        correlations = []
        for lag in lag_candidates(300, 15):
            p, t = aligned_overlap(w.signals[visual.MODEL], w.reference, lag)
            correlations.append((lag, np.corrcoef(np.diff(p), np.diff(t))[0, 1]))
        self.assertNotEqual(selected, max(correlations, key=lambda row: row[1])[0])
        derivative = visual.derivative_plot_data(w, 30)
        p, t = aligned_overlap(w.signals[visual.MODEL], w.reference, selected)
        np.testing.assert_array_equal(derivative["prediction"], np.diff(p) * 30)
        np.testing.assert_array_equal(derivative["reference"], np.diff(t) * 30)
        np.testing.assert_array_equal(derivative["prediction_z_per_sec"], np.diff(standardize_1d(p)) * 30)
        np.testing.assert_array_equal(derivative["reference_z_per_sec"], np.diff(standardize_1d(t)) * 30)
        self.assertAlmostEqual(np.corrcoef(derivative["prediction"], derivative["reference"])[0, 1], derivative["correlation"])

    def test_common_overlap_uses_same_reference_and_each_methods_fixed_lag(self):
        w = self.window
        result = visual.common_comparison(w, [visual.MODEL, visual.CHROM], 30)
        lo, hi = result["reference_index_start"], result["reference_index_stop"]
        np.testing.assert_array_equal(result["reference"], w.reference[lo:hi])
        for method, data in result["methods"].items():
            lag = int(w.metrics[method]["selected_lag_samples"])
            np.testing.assert_array_equal(data["prediction"], w.signals[method][lo + lag:hi + lag])
            expected = waveform_window_metrics(data["prediction"], result["reference"], 30, max_lag_sec=0, coarse_diagnostics=False)
            self.assertEqual(json_safe(expected), json_safe(data["metrics"]))

    def test_unavailable_baseline_does_not_gain_easier_subset(self):
        w = deepcopy(self.window)
        w.signals[visual.CHROM][:] = np.nan
        w.metrics[visual.CHROM] = waveform_window_metrics(w.signals[visual.CHROM], w.reference, 30)
        comparison = visual.common_comparison(w, [visual.MODEL, visual.CHROM], 30)
        self.assertNotIn(visual.CHROM, comparison["methods"])
        self.assertEqual(comparison["valid_samples"], len(visual.waveform_overlap(w)["reference"]))

    def test_beats_retain_exact_global_aligned_samples_without_warping(self):
        w = self.window
        beats = visual.beat_plot_data(w, 30)
        self.assertTrue(beats["reliable"])
        overlap = visual.waveform_overlap(w)
        p, t = visual.signed_zscore(overlap["prediction"]), visual.signed_zscore(overlap["reference"])
        for i, (l, _peak, r) in enumerate(beats["anchors"]):
            np.testing.assert_array_equal(beats["prediction"][i], p[l:r + 1])
            np.testing.assert_array_equal(beats["reference"][i], t[l:r + 1])
            self.assertEqual(len(beats["prediction"][i]), r - l + 1)
        self.assertEqual(beats["lag_samples"], overlap["lag_samples"])
        self.assertAlmostEqual(beats["seconds"][1] - beats["seconds"][0], 1 / 30)

    def test_reference_anchors_do_not_depend_on_prediction(self):
        w = deepcopy(self.window)
        before = visual.beat_plot_data(w, 30)
        w.signals[visual.MODEL] = np.sin(np.arange(300) * 1.7)
        # Keep the existing global lag: no new prediction beat alignment.
        after = visual.beat_plot_data(w, 30)
        self.assertEqual(before["anchors"], after["anchors"])

    def test_empty_short_flat_and_gapped_beats_are_explicitly_insufficient(self):
        for x in (np.array([]), np.ones(300), np.arange(40), np.r_[np.ones(299), np.nan]):
            with self.subTest(size=len(x)):
                self.assertFalse(visual.reliable_beats(x, 30)["reliable"])
        w = deepcopy(self.window)
        w.reference[:] = 1
        self.assertEqual(visual.beat_plot_data(w, 30)["reason"], "insufficient reliable beats")

    def test_welch_frequency_axis_normalization_and_known_frequency(self):
        for fs in (24., 30., 60.):
            x = np.sin(2 * np.pi * 1.25 * np.arange(int(fs * 10)) / fs)
            result = visual.normalized_welch(x, fs)
            f, psd = welch(x - x.mean(), fs=fs, nperseg=result["nperseg"], nfft=result["nfft"])
            np.testing.assert_array_equal(result["frequency_hz"], f)
            self.assertEqual(f[-1], fs / 2)
            self.assertLess(abs(result["hr_peak_hz"] - 1.25), .01)
            self.assertLess(abs(f[np.argmax(result["density"])] - 1.25), .03)
            self.assertAlmostEqual(float(trapezoid(result["density"], f)), 1.)

    def test_invalid_morphology_features_stay_missing_not_zero(self):
        for signal in (np.ones(300), np.r_[np.ones(299), np.nan], np.array([])):
            result = visual.coarse_feature_diagnostics(signal, 30)
            self.assertTrue(all(np.isnan(v) for v in result["values"].values()))
            self.assertTrue(all(v is None for v in json_safe(result)["values"].values()))

    def test_reliable_features_reuse_existing_values_and_slope_units(self):
        signal = self.window.reference
        result = visual.coarse_feature_diagnostics(signal, 30)
        self.assertGreaterEqual(result["reliable_beats"], 3)
        expected = pulse_wave_features(signal, 30)
        for key in visual.FEATURES:
            if key not in ("hr_bpm", "ibi_cv"):
                self.assertEqual(result["values"][key], expected[key])
        self.assertAlmostEqual(result["values"]["ibi_cv"], expected["ibi_std_sec"] / expected["ibi_mean_sec"])


class VisualSelectionTests(unittest.TestCase):
    def rows(self):
        return [dict(subject_id=sid, start_sec=float(i * 2), morphology_score=score,
                     end_sec_exclusive=float(i * 2 + 10))
                for sid in ("A", "B") for i, score in enumerate((.4, .9, .9, .2, .7, .8, np.nan))]

    def test_best_median_worst_deterministic_tie_order(self):
        rows = self.rows()
        ranked = visual.rank_windows(rows)
        self.assertEqual(json_safe(ranked), json_safe(visual.rank_windows(list(reversed(rows)))))
        selections = visual.select_windows(ranked, ["A"], seed=42)
        self.assertEqual([(r["selection"], r["start_sec"]) for r in selections], [("best", 2.), ("median", 8.), ("worst", 6.)])
        self.assertTrue(all(r["selection_notice"] == visual.SELECTION_NOTICE for r in selections))
        self.assertIsNone(next(r for r in ranked if r["subject_id"] == "A" and r["start_sec"] == 12.)["rank"])

    def test_seeded_random_distinct_and_subject_subset_invariant(self):
        ranked = visual.rank_windows(self.rows())
        first = visual.select_windows(ranked, ["A"], seed=42, include_random=True)
        self.assertEqual(first, visual.select_windows(ranked, ["A"], seed=42, include_random=True))
        self.assertEqual(first, [r for r in visual.select_windows(ranked, ["B", "A"], seed=42, include_random=True) if r["subject_id"] == "A"])
        self.assertEqual(len({r["start_sec"] for r in first}), 4)

    def test_specific_window_requires_exact_retained_start(self):
        ranked = visual.rank_windows(self.rows())
        self.assertEqual(visual.select_windows(ranked, ["A"], start_sec=2.)[0]["selection"], "specific")
        for kwargs in (dict(start_sec=2.01), dict(start_sec=np.nan), dict(start_sec=2., selection="best")):
            with self.assertRaises(ValueError):
                visual.select_windows(ranked, ["A"], **kwargs)

    def test_nonoverlap_is_chronological_before_feature_filtering(self):
        rows = self.rows()
        expected = {(sid, start) for sid in ("A", "B") for start in (0., 10.)}
        self.assertEqual(visual.nonoverlapping_keys(list(reversed(rows))), expected)
        # Missing scores do not change temporal eligibility.
        for row in rows:
            row["morphology_score"] = np.nan
        self.assertEqual(visual.nonoverlapping_keys(rows), expected)

    def test_repeatability_ignores_overlapping_windows_and_preserves_missing(self):
        windows = [synthetic_window("A", start) for start in (0., 2., 10., 12., 20.)]
        rows = visual.window_metric_rows(windows, 30)
        primary = visual.nonoverlapping_keys(rows)
        for w in windows:
            w.features[visual.MODEL]["values"]["width50_mean_sec"] = {0.: .2, 2.: 99., 10.: .4, 12.: 99., 20.: np.nan}[w.start_sec]
        agreement, summaries, repeats = visual.feature_tables(windows, primary)
        r = next(r for r in repeats if r["method"] == visual.MODEL and r["feature"] == "width50_mean_sec")
        self.assertEqual(r["eligible_nonoverlap_windows"], 3)
        self.assertEqual(r["n_valid"], 2)
        self.assertAlmostEqual(r["median"], .3)
        self.assertAlmostEqual(r["mad"], .1)
        self.assertEqual(r["valid_feature_fraction"], 2 / 3)
        self.assertEqual(len([r for r in repeats if r["method"] == "reference"]), len(visual.FEATURES))
        summary = next(r for r in summaries if r["method"] == visual.MODEL and r["feature"] == "width50_mean_sec")
        self.assertEqual(summary["prediction_valid_fraction"], 4 / 5)

    def test_cv_requires_positive_non_near_zero_repeat_observations(self):
        for values in ([], [2.], [0., 0.], [-1., 1.], [1e-10, 2e-10]):
            self.assertIsNone(visual.descriptive_stats(values)["cv"])
        self.assertAlmostEqual(visual.descriptive_stats([1., 2., 3.])["cv"], .5)
        self.assertEqual(visual.descriptive_stats([np.nan])["n_valid"], 0)

    def test_csv_and_json_outputs_deterministic_and_null_safe(self):
        rows = visual.rank_windows(self.rows())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("first", "second"):
                visual.write_csv(root / (name + ".csv"), rows)
                write_json(root / (name + ".json"), rows)
            self.assertEqual((root / "first.csv").read_bytes(), (root / "second.csv").read_bytes())
            self.assertEqual((root / "first.json").read_bytes(), (root / "second.json").read_bytes())
            self.assertNotIn("NaN", (root / "first.json").read_text())
            records = list(csv.DictReader(io.StringIO((root / "first.csv").read_text())))
            self.assertEqual(records[-1]["morphology_score"], "")


class VisualRenderingAndPathTests(unittest.TestCase):
    def test_full_report_ranks_every_validation_window_for_one_specific_figure(self):
        manifest = waveform_v1_split_manifest([f"subject{i}" for i in range(10)])
        ids = sorted(manifest["validation_ids"])
        windows = [synthetic_window(sid, start) for sid in ids for start in (0., 2., 10., 20.)]
        w = windows[0]
        extractor = SimpleNamespace(roi_names=w.roi_names, prior_names=w.prior_names,
                                    channel_names=("RGB_R", "RGB_G", "RGB_B", *w.prior_names))
        dataset = SimpleNamespace(roi_names=extractor.roi_names, prior_names=extractor.prior_names,
                                  channel_names=extractor.channel_names, exclusions=[], reference_diagnostics=[])
        checkpoint = dict(fs=30., selected_epoch=25, window_length_samples=300, stride_sec=2., loss_config=dict(max_lag_sec=.5))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            cache.mkdir()
            cp, split, freeze = root / "frozen.pt", root / "split.json", root / "freeze.json"
            cp.write_bytes(b"synthetic checkpoint")
            write_json(split, manifest)
            write_json(freeze, {"synthetic": True})
            for sid in ids:
                folder = root / sid
                folder.mkdir()
                (folder / "vid.avi").write_bytes(b"synthetic video")
                (folder / "ground_truth.txt").write_text("synthetic fixture")

            def cache_access(subject, *_):
                return dict(subject_id=subject.subject_id, reference_path=str(subject.gt_path))

            def fixture_hash(path):
                return visual.OFFICIAL_MODEL_SHA256 if Path(path) == cp else file_sha256(path)

            with patch.object(visual, "verify_frozen_identity", return_value=(extractor, checkpoint, manifest, {}, freeze)), \
                 patch.object(visual, "require_phase1_cache", side_effect=cache_access), \
                 patch.object(visual, "UBFCMultiROIRPPGDataset", return_value=dataset), \
                 patch.object(visual, "infer_validation_windows", return_value=windows), \
                 patch.object(visual, "render_window", return_value=dict(common_plot_valid_samples=296, reliable_reference_beats=10)), \
                 patch.object(visual, "render_overviews"), patch.object(visual, "file_sha256", side_effect=fixture_hash), \
                 patch.object(visual, "provenance", return_value=dict(git_commit="synthetic", git_dirty=False, created_at_utc="fixed-fixture")):
                for name in ("first", "second"):
                    summary = visual.run_visual_validation(ubfc_root=root, checkpoint_path=cp, split_path=split,
                        cache_dir=cache, out_dir=root / name, subject=ids[0], start_sec=0., progress=None)
            self.assertEqual(summary["validation_windows"], len(windows))
            self.assertEqual(len(summary["selected_windows"]), 1)
            self.assertFalse(summary["locked_test_evaluated"])
            self.assertEqual(summary["validation_subject_ids"], ids)
            self.assertEqual(len(json.loads((root / "first/window_rankings.json").read_text())), len(windows))
            for filename in ("window_metrics.csv", "window_rankings.json", "feature_agreement.csv", "repeatability.csv", "summary.json"):
                self.assertEqual((root / "first" / filename).read_bytes(), (root / "second" / filename).read_bytes())

    def test_headless_six_panel_png_and_checkpoint_driven_order(self):
        w = synthetic_window()
        texts = []
        original = plt.Axes.set_xticks

        def capture(ax, ticks, labels=None, **kwargs):
            if labels is not None:
                texts.append(tuple(labels))
            return original(ax, ticks, labels, **kwargs)

        with tempfile.TemporaryDirectory() as tmp, patch.object(plt.Axes, "set_xticks", capture):
            path = Path(tmp) / "synthetic.png"
            result = visual.render_window(w, path, 30, "best", "OMIT multi-ROI average")
            self.assertEqual(matplotlib.get_backend().lower(), "agg")
            self.assertTrue(path.read_bytes().startswith(b"\x89PNG"))
            self.assertGreater(path.stat().st_size, 50000)
            self.assertIn(w.prior_names, texts)
            self.assertGreater(result["common_plot_valid_samples"], 250)
            self.assertTrue(result["beat_summary_available"])
            visual.save_window_arrays(w, Path(tmp) / "arrays.npz")
            with np.load(Path(tmp) / "arrays.npz", allow_pickle=False) as arrays:
                self.assertEqual(tuple(arrays["prior_names"]), w.prior_names)
                np.testing.assert_array_equal(arrays["timestamps"], w.timestamps)

    def test_overview_pngs_and_trajectories_with_invalid_features(self):
        windows = [synthetic_window("A", start) for start in (0., 10.)]
        windows[0].features[visual.MODEL]["values"]["width50_mean_sec"] = np.nan
        rows = visual.rank_windows(visual.window_metric_rows(windows, 30))
        agreement, _summaries, repeats = visual.feature_tables(windows, visual.nonoverlapping_keys(rows))
        with tempfile.TemporaryDirectory() as tmp:
            visual.render_overviews(windows, rows, agreement, repeats, tmp)
            expected = {"subject_metric_distributions.png", "classical_vs_frozen_scatter.png",
                        "morphology_feature_scatter.png", "stability_summary.png", "A_feature_trajectories.png"}
            self.assertEqual({p.name for p in Path(tmp).glob("*.png")}, expected)

    def test_existing_forward_only_input_tensors_and_checkpoint_prior_order(self):
        w = synthetic_window("A", lag=0)
        item = dict(x=torch.tensor(w.diagnostics["x"], dtype=torch.float32),
            y_ppg=torch.tensor(w.reference), y_hr=torch.tensor(72.), subject_id="A", start_sec=torch.tensor(0.),
            roi_quality=torch.ones(3), roi_valid=torch.tensor([True] * 3), prior_valid=torch.ones(3, 3, dtype=torch.bool))
        for k in range(3):
            item["x"][:, 3 + k] = k + 1

        class Dataset:
            subjects = [SimpleNamespace(subject_id="A")]
            samples = [(0, 0, 300)]
            records = [dict(timestamps=w.timestamps, ppg=w.reference, subject_id="A")]

            def __len__(self):
                return 1

            def __getitem__(self, _):
                return item

        extractor = SimpleNamespace(_model=SimpleNamespace(residual_scale=.1), device=torch.device("cpu"),
                                    roi_names=w.roi_names, prior_names=w.prior_names)
        checkpoint = dict(fs=30., split_manifest=dict(validation_ids=["A"]),
                          loss_config=dict(max_lag_sec=.5, spectral_fmin_hz=.7, spectral_fmax_hz=8.))
        outputs = dict(ppg=torch.tensor(w.reference[None]), roi_attention=torch.tensor([[.2, .5, .3]]),
                       prior_weights=torch.ones(1, 3, 3) / 3, residuals=torch.zeros(1, 3, 300))
        with patch.object(visual, "forward_batch", return_value=outputs) as forward:
            result = visual.infer_validation_windows(extractor, Dataset(), checkpoint, progress=None)
        self.assertEqual(set(forward.call_args.args[1]), {"x", "roi_quality", "roi_valid", "prior_valid"})
        np.testing.assert_array_equal(result[0].signals["OMIT multi-ROI average"], np.ones(300))
        np.testing.assert_array_equal(result[0].signals[visual.CHROM], np.full(300, 2.))
        self.assertEqual(result[0].prior_names, w.prior_names)

    def test_nonvalidation_dataset_refused_before_getitem_or_forward(self):
        dataset = SimpleNamespace(subjects=[SimpleNamespace(subject_id="locked_test_fixture")])
        with patch.object(visual, "forward_batch", side_effect=AssertionError("inference forbidden")), \
             self.assertRaisesRegex(ValueError, "non-validation"):
            visual.infer_validation_windows(None, dataset, dict(fs=30, loss_config={}, split_manifest=dict(validation_ids=["A"])))

    def test_no_calls_to_training_bp_or_test_evaluation(self):
        forbidden = {"train", "backward", "step", "fit", "evaluate_waveform_model", "reserve_locked_test",
                     "freeze_waveform_candidate", "two_site_features", "extract_features", "predict_bp"}
        for path in (Path(visual.__file__), Path(cli.__file__)):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            calls = {node.func.id if isinstance(node.func, ast.Name) else node.func.attr
                     for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute))}
            self.assertFalse(calls & forbidden, calls & forbidden)


if __name__ == "__main__":
    unittest.main()
