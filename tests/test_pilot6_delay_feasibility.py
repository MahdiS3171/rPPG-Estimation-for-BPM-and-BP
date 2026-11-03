from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from rppg_lab.pilot6_delay_acquisition import polygon_rgb
from rppg_lab.pilot6_delay_validation import timing_diagnostics, pulse, synthetic_validation, summarize_simulations
from rppg_lab.pilot6_delay_feasibility import cuff_associations, subject_folds, union_seconds, analyze_recording, scheduled_windows, bp_descriptive
from rppg_lab.pilot6_protocol_timeline import cuff_events
from rppg_lab.processing import resample_trace, extract_signal
from rppg_lab.config import PipelineConfig
from rppg_lab.region_masks import _polygon, extract_frame_rgb
from rppg_lab.types import RGBTrace
from rppg_lab.pilot6_delay_runner import assert_protected
from rppg_lab.artifacts import file_sha256


CONFIG = json.loads(Path("configs/pilot6_delay_v1.json").read_text())


def fixture():
    from test_pilot6_protocol_timeline import session_fixture
    s = session_fixture()
    s["events"] = cuff_events(s, 301)
    intervals = [dict(session_id=s["session_id"], policy="B", interval_id="pre1", start_sec=0., end_sec=55., timing_verified=False)]
    delay = dict(session_id=s["session_id"], policy="B", face_roi="combined", hand_roi="hand_palm", method="GREEN",
                 computable=True, primary_nonoverlapping=True, cuff_phase="candidate_cuff_free", status="strict",
                 start_sec=40., end_sec=50., delay_xcorr_sec=.04, matched_beat_count=8, face_hr_bpm=72., face_snr_db=3., hand_snr_db=2.)
    return s, intervals, delay


class SignedTimingTests(unittest.TestCase):
    def test_isolated_jitter_and_frame_losses_retain_failures(self):
        from rppg_lab.pilot6_clock_controls import isolated_clock_controls
        rows = isolated_clock_controls(dict(CONFIG, synthetic_hr_bpm=[78], synthetic_delays_ms=[-40, 0, 40], synthetic_replicates=1))
        self.assertEqual(len(rows), 36)
        self.assertTrue(any(r["noncomputable"] for r in rows if r["fs"] == 30 and r["scenario"] == "single_frame_losses"))
        for r in rows:
            if r["scenario"] == "isolated_jitter" and r["method"] == "GREEN":
                self.assertLess(abs(r["error_ms"]), 8)

    def test_known_positive_zero_negative_and_order_conventions(self):
        for fs in (30., 60.):
            t = np.arange(0, 12, 1/fs)
            for ms in (-120, -80, -60, -40, -20, 0, 20, 40, 60, 80, 120):
                x, y = pulse(t, 78), pulse(t-ms/1000, 78)
                d = timing_diagnostics(x, y, fs, 78/60)
                self.assertLess(abs(d["delay_xcorr_sec"]*1000-ms), 8)
                reverse = timing_diagnostics(y, x, fs, 78/60)
                self.assertAlmostEqual(d["delay_xcorr_sec"], -reverse["delay_xcorr_sec"], places=10)

    def test_cycle_ambiguity_explicit_for_periodic_signal(self):
        fs = 60
        t = np.arange(0, 30, 1/fs)
        d = timing_diagnostics(np.sin(2*np.pi*t), np.sin(2*np.pi*(t-.04)), fs, 1.)
        self.assertTrue(d["cycle_ambiguity"])

    def test_inverted_polarity_not_hidden_by_absolute_correlation(self):
        t = np.arange(0, 12, 1/60)
        x = pulse(t, 60)
        d = timing_diagnostics(x, -x, 60, 1.)
        self.assertTrue(d["polarity_ambiguity"])
        self.assertFalse(d["beat_landmarks_available"])

    def test_search_boundary_is_unresolved(self):
        t = np.arange(0, 12, 1/60)
        d = timing_diagnostics(pulse(t, 60), pulse(t-.15, 60), 60, 1., max_delay_sec=.06)
        self.assertTrue(d["search_boundary"])
        self.assertFalse(np.isfinite(d["delay_xcorr_sec"]))

    def test_no_interpolation_or_filtering_across_long_gap(self):
        t = np.r_[np.arange(0, 5, 1/60), np.arange(7, 12, 1/60)]
        x = np.repeat(pulse(t, 78)[:, None], 3, axis=1)+120
        trace = RGBTrace(t, x, "test", np.ones(len(t), bool))
        grid = np.arange(0, 12, 1/60)
        rgb, _ = resample_trace(trace, grid, .050001)
        wave = extract_signal(rgb, grid, "test", PipelineConfig(sample_rate=60, max_gap_sec=.050001))
        self.assertTrue(np.isnan(wave.values[(grid >= 4) & (grid < 8)]).all())
        self.assertTrue(np.array_equal(wave.timestamps, grid))

    def test_synthetic_matrix_contains_all_signs_rates_and_filter_controls(self):
        c = dict(CONFIG, synthetic_hr_bpm=[78], synthetic_replicates=1)
        rows = synthetic_validation(c, full=False)
        self.assertEqual(set(r["true_delay_ms"] for r in rows), set(CONFIG["synthetic_delays_ms"]))
        self.assertEqual(set(r["fs"] for r in rows), {30., 60.})
        self.assertEqual(set(r["filtered"] for r in rows), {True, False})
        self.assertTrue(all(r.get("estimator_hr_source") == "spectral_face_waveform_not_oracle" for r in rows if not r.get("noncomputable")))
        self.assertEqual(len(rows), 132)
        self.assertEqual(len(summarize_simulations(rows)), 12)


class IsolationAndAssociationTests(unittest.TestCase):
    def test_native_polygon_measurement_exact_full_frame_equivalence(self):
        rng = np.random.default_rng(7)
        frame = rng.integers(0, 256, (120, 80, 3), dtype=np.uint8)
        polygons = [np.array([[5.5, 6.1], [32.3, 10.7], [20.9, 80.2]]), np.array([[-4, 16], [44, 3], [49, 27]])]
        mask = np.zeros(frame.shape[:2], np.uint8)
        for p in polygons:
            _polygon(mask, p)
        expected, q = extract_frame_rgb(frame, mask.astype(bool), 3)
        actual, aq = polygon_rgb(frame, polygons, 3)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(q, aq)

    def test_primary_unverified_associations_unavailable_and_unique(self):
        s, intervals, delay = fixture()
        rows = cuff_associations([s], intervals, [delay]*20, CONFIG)
        self.assertEqual(len(rows), 3)
        self.assertEqual(len(set(r["event_id"] for r in rows)), 3)
        self.assertFalse(any(r["association_available"] for r in rows))

    def test_sensitivity_keeps_actual_event_label_and_no_future_features(self):
        s, intervals, delay = fixture()
        future = dict(delay, start_sec=70., end_sec=80.)
        rows = cuff_associations([s], intervals, [delay, future], CONFIG, verified_only=False)
        self.assertTrue(rows[0]["association_available"])
        self.assertEqual(rows[0]["eligible_signal_windows"], 1)
        self.assertEqual(rows[0]["feature_to_cuff_sec"], 10)
        self.assertIsNone(rows[0]["delay_dispersion_ms"])
        self.assertEqual(rows[0]["sbp_mmHg"], s["raw"]["SBP1_mmHg"])
        self.assertFalse(rows[1]["association_available"])

    def test_identical_duplicate_features_do_not_multiply_beats(self):
        s, intervals, delay = fixture()
        rows = cuff_associations([s], intervals, [delay]*20, CONFIG, verified_only=False)
        self.assertEqual(rows[0]["eligible_signal_windows"], 1)
        self.assertEqual(rows[0]["independent_matched_beats"], 8)

    def test_temporal_distance_sensitivity(self):
        s, intervals, delay = fixture()
        rows = cuff_associations([s], intervals, [delay], CONFIG, verified_only=False, max_distance=5)
        self.assertFalse(any(r["association_available"] for r in rows))

    def test_duplicate_cuff_event_is_error(self):
        s, intervals, delay = fixture()
        s["events"].append(deepcopy(s["events"][0]))
        with self.assertRaisesRegex(ValueError, "Duplicate cuff"):
            cuff_associations([s], intervals, [delay], CONFIG)

    def test_no_train_test_subject_leakage_or_window_labels(self):
        records = [dict(subject_id=s, event_id=f"{s}_{i}") for s in ("a", "b", "c") for i in range(3)]
        for train, test in subject_folds(records):
            self.assertFalse(set(r["subject_id"] for r in train) & set(r["subject_id"] for r in test))
            self.assertEqual(len(test), 3)

    def test_two_centered_observations_do_not_manufacture_perfect_bp_correlation(self):
        subjects = ("a", "a", "b", "c", "d", "e")
        records = [dict(subject_id=s, sbp_mmHg=110+i, dbp_mmHg=70+i, association_available=False) for i, s in enumerate(subjects)]
        sensitivity = [dict(r, association_available=True, policy="B", selection="all_computable", max_distance_sec=30.,
                            median_delay_ms=float(i), delay_dispersion_ms=float(i), hr_bpm=72+i) for i, r in enumerate(records)]
        rows = bp_descriptive(records, sensitivity)
        within = [r for r in rows if r.get("scope") == "within_subject_centered"]
        self.assertTrue(within)
        self.assertTrue(all(r["value"] is None and r["within_subject_df"] == 1 for r in within))

    def test_coverage_union_not_overlap_sum(self):
        self.assertEqual(union_seconds([dict(start_sec=0, end_sec=10), dict(start_sec=5, end_sec=15)]), 15)

    def test_historical_protection_detects_any_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"historical.txt"
            p.write_text("frozen")
            before = {str(p): file_sha256(p)}
            self.assertEqual(assert_protected(before)["status"], "PASS")
            p.write_text("modified")
            with self.assertRaisesRegex(RuntimeError, "Protected"):
                assert_protected(before)

    def test_official_frozen_identity_when_available(self):
        from rppg_lab.pilot_frozen import DEFAULT_CHECKPOINT, OFFICIAL_MODEL_SHA256, OFFICIAL_MANIFEST_SHA256
        p = Path(DEFAULT_CHECKPOINT)
        if not p.is_file():
            self.skipTest("Official private checkpoint absent")
        self.assertEqual(file_sha256(p), OFFICIAL_MODEL_SHA256)
        self.assertEqual(file_sha256(p.with_name("waveform_v1_freeze_manifest.json")), OFFICIAL_MANIFEST_SHA256)

    def test_complete_signal_path_preserves_clock_and_failed_boundaries(self):
        from rppg_lab.pilot6_delay_acquisition import ROIS
        s, _, _ = fixture()
        s["events"] = [dict(start_sec=15., end_sec=20., timing_verified=False)]
        t = np.arange(0, 40, 1/30)
        p = pulse(t, 78)
        arrays = dict(timestamps=t, backend_timestamps=t, repeated_analysis_frame=np.zeros(len(t), bool),
                      actual_hand_backend=np.asarray("synthetic_fixture"), hand_identity=np.asarray(["fixture"]*len(t)),
                      face_tracking_reason=np.asarray(["ok"]*len(t)), hand_tracking_reason=np.asarray(["ok"]*len(t)))
        for roi in ROIS:
            arrays[roi+"_rgb"] = np.array([140., 120., 90.])*(1+p[:, None]*np.array([.006, .012, .003]))
        with tempfile.TemporaryDirectory() as tmp:
            q, d, a, clock = analyze_recording(arrays, s, CONFIG, tmp)
            self.assertEqual(clock["source_frames"], len(t))
            self.assertGreaterEqual(len(q), 6*3*7*3)
            self.assertGreaterEqual(len(d), 4*2*3*7*3)
            crossing = [r for r in d if r["cuff_phase"] == "crosses_boundary"]
            self.assertTrue(crossing)
            self.assertFalse(any(r["computable"] for r in crossing))
            with np.load(Path(tmp)/"classical_waveforms.npz") as z:
                np.testing.assert_array_equal(z["source_pts"], t)
                self.assertTrue(np.isnan(z["A_combined_GREEN"][(t >= 14) & (t < 16)]).all())

    def test_short_usable_phase_scheduled_without_overlapping_primary_windows(self):
        phase = np.r_[np.ones(200), np.zeros(15), np.ones(30)]
        starts, primary = scheduled_windows(phase, np.ones(len(phase), bool), 1, CONFIG)
        self.assertIn(201, primary)
        self.assertIn(201, starts)
        ordered = sorted(primary)
        self.assertTrue(all(b-a >= 10 for a, b in zip(ordered, ordered[1:])))


if __name__ == "__main__":
    unittest.main()
