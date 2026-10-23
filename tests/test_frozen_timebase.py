"""Synthetic/non-identifiable timebase and frozen-contract regressions."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from rppg_lab.config import PipelineConfig
from rppg_lab.extraction_cache import FaceROIExtraction, DEFAULT_WAVEFORM_ROIS, _validate
from rppg_lab.frozen_timebase import (TimebaseConfig, adapt_native_face, antialias_fir,
                                     filter_characteristics, validate_on_camera_clock, LINEAR_VERSION)
from rppg_lab.pilot_frozen import (DEFAULT_CHECKPOINT, OFFICIAL_MODEL_SHA256,
    OFFICIAL_MANIFEST_SHA256, covered_seconds, s004_decision, native_extraction)
from rppg_lab.processing import shared_grid, resample_trace
from rppg_lab.types import RGBTrace, SharedTimebase, RegionObservation
from rppg_lab.pilot_frozen_extraction import measure_native_masks
from rppg_lab.region_masks import extract_frame_rgb_by_roi, get_face_masks, _polygon
from rppg_lab.roi import FACE_ROIS


def synthetic(t, frequencies=(1.2,), valid=None):
    config = PipelineConfig(regions=("face",), face_roi="forehead", face_rois=DEFAULT_WAVEFORM_ROIS, sample_rate=30)
    valid = np.ones(len(t), bool) if valid is None else valid
    # DC intensities and different pulsatile RGB amplitudes avoid degenerate priors.
    pulse = sum(np.sin(2*np.pi*f*t) for f in frequencies)
    rgb = np.asarray([130, 110, 80])[None] + pulse[:, None]*[1., 2., .3]
    traces = {name:RGBTrace(t.copy(), rgb.copy(), name, valid.copy()) for name in DEFAULT_WAVEFORM_ROIS}
    return FaceROIExtraction(SharedTimebase(shared_grid(t, 30), t.copy()), traces), config


def target_green(extraction, config):
    return resample_trace(extraction.traces[DEFAULT_WAVEFORM_ROIS[0]],
                          extraction.shared.timestamps, config.max_gap_sec)[0][:, 1]


def fitted_amplitude_phase(t, x, frequency):
    good = np.isfinite(x)
    fit = np.column_stack([np.ones(good.sum()), np.sin(2*np.pi*frequency*t[good]),
                           np.cos(2*np.pi*frequency*t[good])])
    coef = np.linalg.lstsq(fit, x[good], rcond=None)[0]
    return float(np.hypot(coef[1], coef[2])), float(np.arctan2(coef[2], coef[1]))


class TimebaseTests(unittest.TestCase):
    def test_filter_design_symmetry_dc_stopband(self):
        h = antialias_fir()
        self.assertEqual(len(h) % 2, 1)
        np.testing.assert_allclose(h, h[::-1], atol=1e-15)
        self.assertAlmostEqual(h.sum(), 1, places=12)
        response = filter_characteristics()
        self.assertLess(response["passband_max_absolute_gain_error"], .001)
        self.assertLess(response["stopband_max_gain_db"], -70)
        self.assertEqual(response["retained_samples_time_shift_sec"], 0)

    def test_rejects_unknown_version_rate_and_invalid_clock(self):
        with self.assertRaises(ValueError):
            TimebaseConfig(version="unversioned")
        for rate in (24, 45, 90):
            raw, c = synthetic(np.arange(1000)/rate)
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                adapt_native_face(raw, c)
        raw, c = synthetic(np.arange(1000)/60)
        raw.shared.original_timestamps[8] = raw.shared.original_timestamps[7]
        with self.assertRaises(ValueError):
            adapt_native_face(raw, c)

    def test_camera_offset_grid_phase_and_no_extrapolation(self):
        t = .0058 + np.arange(1800)/60
        raw, c = synthetic(t)
        adapted, report = adapt_native_face(raw, c)
        np.testing.assert_array_equal(adapted.shared.original_timestamps, t)
        self.assertEqual(adapted.shared.timestamps[0], t[0])
        self.assertLessEqual(adapted.shared.timestamps[-1], t[-1])
        np.testing.assert_allclose(np.diff(adapted.shared.timestamps), 1/30, atol=1e-14)
        self.assertTrue(report["camera_pts_preserved"])
        self.assertTrue(np.isnan(target_green(adapted, c)[0]))
        self.assertTrue(np.isnan(target_green(adapted, c)[-1]))
        _validate(adapted, c)

    def test_passband_amplitude_and_zero_phase(self):
        t = .0058 + np.arange(3600)/60
        for frequency in (.7, 1.2, 3.5, 8.):
            raw, c = synthetic(t, (frequency,))
            adapted, _ = adapt_native_face(raw, c)
            amp, phase = fitted_amplitude_phase(adapted.shared.timestamps, target_green(adapted, c), frequency)
            self.assertAlmostEqual(amp, 2., delta=.002)
            self.assertAlmostEqual(phase, 0., delta=1e-6)

    def test_alias_rejection_vs_discarding_alternate_frames(self):
        t = np.arange(3600)/60
        for frequency, alias in ((20., 10.), (28., 2.)):
            raw, c = synthetic(t, (frequency,))
            adapted, _ = adapt_native_face(raw, c)
            y = target_green(adapted, c)
            amplitude, _ = fitted_amplitude_phase(adapted.shared.timestamps, y, alias)
            naive_amplitude, _ = fitted_amplitude_phase(t[::2], raw.traces[DEFAULT_WAVEFORM_ROIS[0]].g[::2], alias)
            self.assertGreater(naive_amplitude, 1.9)
            self.assertLess(amplitude/naive_amplitude, .001)  # ≥60 dB actual alias attenuation.

    def test_irregular_intervals_use_actual_time(self):
        rng = np.random.default_rng(42)
        t = .0058 + np.arange(3600)/60
        t[1:] += rng.uniform(-.0001, .0001, len(t)-1)
        raw, c = synthetic(t)
        adapted, report = adapt_native_face(raw, c)
        amp, phase = fitted_amplitude_phase(adapted.shared.timestamps, target_green(adapted, c), 1.2)
        self.assertAlmostEqual(amp, 2., delta=.003)
        self.assertAlmostEqual(phase, 0, delta=.001)
        self.assertGreater(report["rois"][DEFAULT_WAVEFORM_ROIS[0]]["regularization_interpolated_fraction"], .9)
        np.testing.assert_array_equal(adapted.traces[DEFAULT_WAVEFORM_ROIS[0]].timestamps, t)

    def test_positive_and_negative_synthetic_phase_delay_preserved(self):
        rng = np.random.default_rng(44)
        t = .0058 + np.arange(3600)/60
        t[1:] += rng.uniform(-.0001, .0001, len(t)-1)
        t = np.delete(t, 900)
        for delay in (-.1, .1):
            raw, c = synthetic(t)
            pulse = np.sin(2*np.pi*1.2*(t-delay))
            for trace in raw.traces.values():
                trace.values = [130,110,80] + pulse[:,None]*[1.,2.,.3]
            adapted, _ = adapt_native_face(raw, c)
            _, phase = fitted_amplitude_phase(adapted.shared.timestamps, target_green(adapted, c), 1.2)
            recovered = -phase/(2*np.pi*1.2)
            self.assertAlmostEqual(recovered, delay, delta=.0005)

    def test_acquisition_gap_never_bridged_and_fir_support_excluded(self):
        t = np.arange(3600)/60
        t = t[(t < 20) | (t > 20.3)]
        raw, c = synthetic(t)
        adapted, report = adapt_native_face(raw, c)
        grid = adapted.shared.timestamps
        y = target_green(adapted, c)
        self.assertTrue(np.isnan(y[(grid >= 20) & (grid <= 20.3)]).all())
        self.assertTrue(np.isnan(y[np.argmin(abs(grid-19.9))]))
        self.assertTrue(np.isfinite(y[np.argmin(abs(grid-19.0))]))
        self.assertEqual(report["acquisition_intervals_exceeding_gap_limit"], 1)

    def test_short_missing_interval_explicitly_interpolated(self):
        t = np.arange(3600)/60
        t = np.delete(t, [1200, 1201])  # 50 ms from remaining neighbors.
        raw, c = synthetic(t)
        adapted, report = adapt_native_face(raw, c)
        self.assertEqual(report["acquisition_intervals_exceeding_gap_limit"], 0)
        y = target_green(adapted, c)
        self.assertTrue(np.isfinite(y[600]))
        self.assertEqual(report["acquisition_intervals_over_1_5_native_step"], 1)

    def test_roi_loss_remains_missing_other_rois_preserved(self):
        t = np.arange(3600)/60
        raw, c = synthetic(t)
        name = DEFAULT_WAVEFORM_ROIS[0]
        raw.traces[name].validity_mask[600:900] = False
        snapshot = raw.traces[name].values.copy()
        adapted, _ = adapt_native_face(raw, c)
        g = adapted.shared.timestamps
        first = target_green(adapted, c)
        second = resample_trace(adapted.traces[DEFAULT_WAVEFORM_ROIS[1]], g, c.max_gap_sec)[0]
        self.assertTrue(np.isnan(first[(g >= 10) & (g < 15)]).all())
        self.assertTrue(np.isfinite(second[(g >= 10) & (g < 15)]).all())
        np.testing.assert_array_equal(raw.traces[name].values, snapshot)
        np.testing.assert_array_equal(adapted.traces[name].quality["acquisition_valid"], raw.traces[name].validity_mask)

    def test_short_supported_runs_not_filled(self):
        raw, c = synthetic(np.arange(30)/60)
        adapted, _ = adapt_native_face(raw, c)
        self.assertFalse(np.isfinite(target_green(adapted, c)).any())

    def test_v2_actual_clock_style_drift_and_dropped_frames_transfer(self):
        t = np.arange(6000)/60
        t += .004*np.sin(2*np.pi*.017*t)
        t = np.delete(t, np.arange(400, 6000, 500))
        raw, c = synthetic(t)
        old = validate_on_camera_clock(t, c, TimebaseConfig(version=LINEAR_VERSION))
        revised = validate_on_camera_clock(t, c)
        self.assertFalse(old["passed"])
        self.assertTrue(revised["passed"])
        self.assertGreater(revised["tests"][3]["fitted_gain"], old["tests"][3]["fitted_gain"])
        self.assertIn("output_ac_rms_relative_to_input", revised["tests"][5])

    def test_real_30_hz_identity_path_preserves_all_raw_arrays(self):
        raw, c = synthetic(.25 + np.arange(1800)/30)
        adapted, report = adapt_native_face(raw, c)
        self.assertEqual(report["operation"], "identity_30_hz")
        for name in DEFAULT_WAVEFORM_ROIS:
            np.testing.assert_array_equal(raw.traces[name].values, adapted.traces[name].values)
            np.testing.assert_array_equal(raw.traces[name].validity_mask, adapted.traces[name].validity_mask)
            np.testing.assert_array_equal(target_green(raw, c), target_green(adapted, c))


class NativePixelTests(unittest.TestCase):
    def test_bounded_mask_measurement_equals_historical_full_frame(self):
        rng = np.random.default_rng(43)
        frame = rng.integers(0, 256, (320, 180, 3), dtype=np.uint8)
        points = rng.uniform([20, 30], [160, 290], (478, 2))
        observation = RegionObservation(0, 0., landmarks=points, valid=True)
        masks = get_face_masks(frame, observation, DEFAULT_WAVEFORM_ROIS)
        masks["missing"] = np.zeros(frame.shape[:2], bool)
        old, new = extract_frame_rgb_by_roi(frame, masks, 50), measure_native_masks(frame, masks, 50)
        for name in masks:
            np.testing.assert_array_equal(old[name][0], new[name][0])
            for key in old[name][1]:
                np.testing.assert_equal(old[name][1][key], new[name][1][key])

    def test_saved_vertices_reconstruct_identical_contributing_pixels(self):
        rng = np.random.default_rng(44)
        frame = np.zeros((320, 180, 3), np.uint8)
        points = rng.uniform([-20, -10], [210, 350], (478, 2))
        obs = RegionObservation(0, 0., landmarks=points, valid=True)
        expected = get_face_masks(frame, obs, DEFAULT_WAVEFORM_ROIS)
        for name in DEFAULT_WAVEFORM_ROIS:
            mask = np.zeros(frame.shape[:2], np.uint8)
            _polygon(mask, points[FACE_ROIS[name]])
            np.testing.assert_array_equal(mask.astype(bool), expected[name])


@unittest.skipUnless(Path(DEFAULT_CHECKPOINT).is_file(), "Official local checkpoint unavailable")
class OfficialFrozenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from rppg_lab.waveform_inference import FrozenWaveformExtractor
        torch.set_num_threads(1)
        cls.extractor = FrozenWaveformExtractor(DEFAULT_CHECKPOINT)

    def test_identity_eval_architecture_and_manifest(self):
        from rppg_lab.artifacts import file_sha256
        e = self.extractor
        self.assertEqual(e.model_sha256, OFFICIAL_MODEL_SHA256)
        self.assertEqual(file_sha256(Path(DEFAULT_CHECKPOINT).with_name("waveform_v1_freeze_manifest.json")), OFFICIAL_MANIFEST_SHA256)
        self.assertFalse(e._model.training)
        self.assertEqual(type(e._model).__name__, "MultiROIPriorResidualWaveformNet")
        self.assertEqual(e.roi_names, DEFAULT_WAVEFORM_ROIS)
        self.assertEqual((e.fs_target, e.window_length_samples), (30., 300))

    def test_historical_model_and_preprocessing_source_text_unchanged(self):
        import hashlib
        manifest = json.loads(Path(DEFAULT_CHECKPOINT).with_name("waveform_v1_freeze_manifest.json").read_text())
        for name in ("models", "classical", "signals", "roi", "region_masks", "detection", "datasets",
                     "extraction_cache", "window_priors", "waveform_training"):
            path = Path("rppg_lab")/(name+".py")
            expected = manifest["training_source_provenance"]["source_sha256"][str(path).replace("/", "\\")]
            lf = path.read_bytes().replace(b"\r\n", b"\n")
            candidates = [path.read_bytes(), lf, lf.replace(b"\n", b"\r\n")]
            self.assertIn(expected, [hashlib.sha256(b).hexdigest() for b in candidates], name)

    def test_modified_manifest_is_rejected_by_original_api(self):
        from rppg_lab.waveform_inference import FrozenWaveformExtractor
        manifest = json.loads(Path(DEFAULT_CHECKPOINT).with_name("waveform_v1_freeze_manifest.json").read_text())
        manifest["fs"] = 60
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"manifest.json"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "provenance/config mismatch"):
                FrozenWaveformExtractor(DEFAULT_CHECKPOINT, path)

    def test_incompatible_60_hz_grid_still_rejected(self):
        raw, _ = synthetic(np.arange(1800)/60)
        raw.shared.timestamps = raw.shared.original_timestamps.copy()
        with self.assertRaisesRegex(ValueError, "sample rate"):
            self.extractor.infer_window(raw)

    def test_window_length_clock_consistency_repeatability_state_unchanged(self):
        raw, _ = synthetic(.0058 + np.arange(2400)/60)
        e = self.extractor
        adapted, _ = adapt_native_face(raw, e.extraction_config)
        state = {k:v.clone() for k,v in e._model.state_dict().items()}
        a = adapted.shared.timestamps[150]
        first = e.infer_window(adapted, a)
        e.infer_window(adapted, a+5.)
        again = e.infer_window(adapted, a)
        np.testing.assert_array_equal(first["waveform"], again["waveform"])
        np.testing.assert_array_equal(first["timestamps"], adapted.shared.timestamps[150:450])
        self.assertEqual(first["waveform"].shape, (300,))
        self.assertTrue(np.isfinite(first["waveform"]).all())
        for k,v in e._model.state_dict().items():
            self.assertTrue(torch.equal(v, state[k]))

    def test_historical_30_hz_prediction_bit_identical(self):
        e = self.extractor
        raw, _ = synthetic(np.arange(900)/30)
        before = e.infer_window(raw, 5.)
        adapted, _ = adapt_native_face(raw, e.extraction_config)
        after = e.infer_window(adapted, 5.)
        for key in ("waveform", "timestamps", "roi_attention", "prior_weights", "scaled_residuals"):
            np.testing.assert_array_equal(before[key], after[key])

    def test_gap_window_rejected_by_original_api(self):
        t = np.arange(1800)/60
        raw, _ = synthetic(t, valid=(t < 10) | (t > 12))
        adapted, _ = adapt_native_face(raw, self.extractor.extraction_config)
        with self.assertRaisesRegex(ValueError, "No usable ROI"):
            self.extractor.infer_window(adapted, 5.)

    def test_synthetic_end_to_end_outputs_failures_allocations_and_report(self):
        import csv
        from rppg_lab.pilot_frozen import evaluate_native, write_comparative_report
        t = .0058 + np.arange(1860)/60
        raw, _ = synthetic(t)
        arrays = dict(camera_pts=t, frame_shape=np.asarray([64,48,3]), actual_backend=np.asarray("synthetic"))
        for name, trace in raw.traces.items():
            arrays[name + "_rgb"] = trace.values
            count = len(FACE_ROIS[name])
            vertices = np.column_stack([24+5*np.cos(np.arange(count)*2*np.pi/count),
                                         32+5*np.sin(np.arange(count)*2*np.pi/count)])
            arrays[name + "_vertices"] = np.broadcast_to(vertices, (len(t), count, 2)).copy()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            summary = evaluate_native(arrays, self.extractor, out, TimebaseConfig(), "S999_V01")
            self.assertEqual(summary["complete_windows"], 5)
            self.assertEqual(summary["methods"]["FrozenWaveformV1"]["invalid_windows"], 1)
            with (out/"window_metrics.csv").open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 20)  # Every method/window, including failure.
            self.assertEqual(sum(r["quality_label"] == "invalid" for r in rows), 4)
            with np.load(out/"waveform_windows.npz", allow_pickle=False) as data:
                self.assertEqual(data["waveform"].shape, (5, 300))
                np.testing.assert_array_equal(data["timestamps"][1], .0058+np.arange(150,450)/30)
                self.assertTrue(np.isnan(data["waveform"][0]).all())
                np.testing.assert_allclose(data["roi_attention"][1:].sum(axis=1), 1, atol=1e-5)
            self.assertTrue((out/"overview.png").is_file())
            self.assertTrue((out/"roi_masks_examples.npz").is_file())
            from scripts.verify_pilot6_frozen import contribution_diagnostics
            decomposition = contribution_diagnostics(out)
            self.assertLess(decomposition["max_absolute_reconstruction_error"], 2e-6)
            self.assertTrue((out/"model_decomposition.png").is_file())
            report = dict(model=dict(model_sha256=OFFICIAL_MODEL_SHA256), manifest_sha256=OFFICIAL_MANIFEST_SHA256)
            write_comparative_report(out, [summary], [], report, s004_decision(summary))
            text = (out/"research_report.md").read_text(encoding="utf-8")
            self.assertIn("No hand inference", text)
            self.assertIn("S999_V01", text)
            self.assertIn("not ground-truth validation", text)
            write_comparative_report(out, [summary], [], report, s004_decision(summary),
                                     TimebaseConfig(max_interpolation_gap_sec=.025001))
            self.assertIn("≤25.001 ms", (out/"research_report.md").read_text(encoding="utf-8"))


class EvaluationGateTests(unittest.TestCase):
    def test_coverage_is_union_not_sum_of_overlapping_windows(self):
        self.assertEqual(covered_seconds([(0,10),(5,15),(20,30)]), 25.)
        self.assertEqual(covered_seconds([]), 0.)

    def test_s004_gate_does_not_claim_physiological_accuracy(self):
        summary = dict(model_sha256=OFFICIAL_MODEL_SHA256, frozen_state_unchanged=True,
            camera_pts_preserved=True, adapter_clock_validation_passed=True,
            methods={"FrozenWaveformV1":dict(usable_union_sec=35, valid_windows=6,
                                                                       engineering_good_windows=0)})
        self.assertEqual(s004_decision(summary)["decision"], "PROCESS_REMAINING_FIVE")
        summary["frozen_state_unchanged"] = False
        self.assertEqual(s004_decision(summary)["decision"], "STOP_AND_INVESTIGATE")


if __name__ == "__main__":
    unittest.main()
