"""Manufactured references only: Gate 08 leakage, coverage, sign and timing checks."""
from copy import deepcopy
import inspect
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from rppg_lab.classical import DEFAULT_PRIORS, METHOD_FUNCS
from rppg_lab.signals import standardize_1d, estimate_hr_welch
from rppg_lab.waveform_baselines import classical_window_predictions
from rppg_lab.waveform_v2_classical_study import (
    METHODS, CONDITIONS, STARTS, array_sha256, predict_recording, prediction_rows,
    score_waveform, trace_quality, aggregate, quality_associations, control_fixtures,
    run_controls, timing_controls)
from rppg_lab.waveform_v2_classical_cache import load_verified_baseline, read_reference_windows
from rppg_lab.waveform_v2_scamps import source_sha256
from scripts.audit_waveform_v2_classical_feasibility import validate_output, write_csv, write_json


class ClassicalPredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.rgb, cls.valid, cls.reference = next(control_fixtures())
        cls.prediction = predict_recording(cls.rgb, cls.valid)

    def test_exact_unchanged_priors_and_raw_rgb_order(self):
        self.assertEqual(METHODS, tuple(DEFAULT_PRIORS))
        for k, method in enumerate(METHODS):
            expected = standardize_1d(METHOD_FUNCS[method](self.rgb[0, :300], 30))
            np.testing.assert_array_equal(expected, self.prediction['signals'][0, 0, k])
        self.assertTrue(np.isnan(self.prediction['batch']['raw_d_ppg']).all())

    def test_existing_fusion_is_reused_exactly(self):
        batch = self.prediction['batch']
        original = dict(classical_window_predictions(batch['x'][0], batch['roi_valid'][0],
                        batch['prior_valid'][0], batch['roi_names'], batch['prior_names']))
        for k, method in enumerate(METHODS):
            np.testing.assert_array_equal(self.prediction['signals'][0, 3, k], original[method+' multi-ROI average'])
        np.testing.assert_allclose(self.prediction['fusion_weights'], 1/3)

    def test_prediction_and_quality_apis_have_no_reference_input(self):
        for function in (predict_recording, prediction_rows, trace_quality):
            self.assertNotIn('reference', inspect.signature(function).parameters)
        before = self.prediction['signals'].copy()
        for target in (self.reference[:300], self.reference[:300][::-1], np.ones(300)):
            score_waveform(before[0, 0, 0], target)
        np.testing.assert_array_equal(before, self.prediction['signals'])

    def test_quality_and_prediction_hashes_unchanged_by_scoring(self):
        rows = prediction_rows('fixture', 'Train', self.rgb, self.valid, np.zeros((3, 600)), self.prediction)
        original = deepcopy(rows)
        for row in rows:
            score_waveform(self.prediction['signals'][STARTS.index(row['start']), CONDITIONS.index(row['roi']), METHODS.index(row['method'])],
                           self.reference[row['start']:row['stop']])
        self.assertEqual(str(rows), str(original))
        self.assertEqual(len(rows), 48)

    def test_constant_input_is_not_pulse_eligible_even_if_prior_computes(self):
        _, rgb, valid, _ = list(control_fixtures())[1]
        p = predict_recording(rgb, valid)
        rows = prediction_rows('constant', 'Control', rgb, valid, np.zeros((3, 600)), p)
        self.assertTrue(any(r['computational_success'] for r in rows))
        self.assertFalse(any(r['pulse_estimate_eligible'] for r in rows))
        self.assertTrue(all(r['prediction_failure_reason'] == 'FLAT_SOURCE_RGB' for r in rows))

    def test_missing_roi_and_whole_window_keep_all_cells(self):
        _, rgb, valid, _ = list(control_fixtures())[6]
        p = predict_recording(rgb, valid)
        rows = prediction_rows('missing', 'Control', rgb, valid, np.zeros((3, 600)), p)
        self.assertEqual(len(rows), 48)
        self.assertFalse(p['computational'][1].any())
        self.assertTrue(np.isnan(p['signals'][1]).all())
        np.testing.assert_array_equal(p['fusion_weights'][0, :, 0], 0)
        np.testing.assert_allclose(p['fusion_weights'][0, :, 1:], .5)
        np.testing.assert_array_equal(p['fusion_weights'][1], 0)

    def test_wrong_shapes_and_compressed_indices_refused(self):
        with self.assertRaises(ValueError): predict_recording(self.rgb[:, :-1], self.valid[:, :-1])
        with self.assertRaises(ValueError): predict_recording(self.rgb, self.valid.astype(float))

    def test_window_locality_future_rgb_cannot_change_first_window(self):
        rgb = self.rgb.copy(); rgb[:, 300:] += [10, 4, 7]
        other = predict_recording(rgb, self.valid)
        np.testing.assert_array_equal(self.prediction['signals'][0], other['signals'][0])


class ClassicalMetricTests(unittest.TestCase):
    def setUp(self):
        self.t = np.arange(300)/30
        self.target = np.sin(2*np.pi*1.2*self.t)

    def test_frequency_and_hr_error_convention(self):
        result = score_waveform(np.sin(2*np.pi*1.5*self.t), self.target)
        self.assertAlmostEqual(result['reference_hr_bpm'], 72, delta=.15)
        self.assertAlmostEqual(result['pred_hr_bpm'], 90, delta=.15)
        self.assertAlmostEqual(result['hr_error_bpm'], 18, delta=.3)
        self.assertAlmostEqual(result['hr_squared_error'], result['hr_error_bpm']**2)

    def test_native_negative_polarity_is_not_repaired(self):
        result = score_waveform(-self.target, self.target)
        self.assertAlmostEqual(result['zero_lag_r'], -1, places=6)
        self.assertGreater(result['lag_aligned_r'], .98)
        self.assertNotEqual(result['selected_lag_samples'], 0)

    def test_positive_lag_means_prediction_lags_reference(self):
        prediction = np.sin(2*np.pi*1.2*(self.t-.1))
        before = prediction.copy()
        result = score_waveform(prediction, self.target)
        self.assertEqual(result['selected_lag_samples'], 3)
        self.assertEqual(result['aligned_overlap_samples'], 297)
        np.testing.assert_array_equal(prediction, before)

    def test_zero_lag_tie_preference(self):
        result = score_waveform(self.target, self.target)
        self.assertEqual(result['selected_lag_samples'], 0)
        self.assertEqual(result['aligned_overlap_samples'], 300)

    def test_primary_reference_unfiltered_and_secondary_explicit(self):
        raw = self.target+np.linspace(-2, 2, 300)
        result = score_waveform(self.target, raw)
        self.assertAlmostEqual(result['zero_lag_r'], np.corrcoef(self.target, standardize_1d(raw))[0, 1])
        self.assertGreater(result['matched_band_reference_zero_lag_r'], result['zero_lag_r'])

    def test_nonfinite_prediction_is_not_time_compressed(self):
        prediction = self.target.copy(); prediction[50] = np.nan
        result = score_waveform(prediction, self.target)
        self.assertFalse(result['prediction_usable'])
        self.assertTrue(np.isnan(result['zero_lag_r']))
        self.assertTrue(np.isfinite(result['reference_hr_bpm']))

    def test_invalid_reference_does_not_erase_prediction_hr(self):
        result = score_waveform(self.target, np.ones(300))
        self.assertFalse(result['reference_usable'])
        self.assertTrue(np.isfinite(result['pred_hr_bpm']))
        self.assertTrue(np.isnan(result['hr_error_bpm']))

    def test_strict_complete_window_support(self):
        for size in (299, 301):
            with self.assertRaises(ValueError): score_waveform(np.ones(size), self.target)

    def test_quality_frequency_and_half_consistency_are_label_free(self):
        q = trace_quality(self.target)
        self.assertAlmostEqual(q['dominant_hr_bpm'], 72, delta=.15)
        self.assertLess(q['half_hr_difference_bpm'], .3)
        self.assertGreater(q['spectral_concentration'], .8)

    def test_balanced_rmse_is_root_of_equal_recording_mse(self):
        rows = []
        for name, errors in [('a', [0., 4.]), ('b', [10.])]:
            for error in errors:
                metrics = score_waveform(self.target, self.target)
                metrics.update(hr_error_bpm=error, hr_abs_error_bpm=abs(error), hr_squared_error=error**2)
                rows.append(dict(filename=name, split='Train', roi='forehead', method='GREEN', metrics=metrics,
                                 computational_success=True, pulse_estimate_eligible=True, prediction_failure_reason=''))
        value = aggregate(rows)[0]
        self.assertEqual(value['pooled']['intended_cells'], 3)
        self.assertAlmostEqual(value['recording_balanced']['hr_abs_error_bpm'], 6)
        self.assertAlmostEqual(value['recording_balanced']['hr_rmse_bpm'], np.sqrt((8+100)/2))


class ClassicalIOAndControlTests(unittest.TestCase):
    def test_reference_reader_reads_only_same_index_d_ppg(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'fixture.h5'
            with h5py.File(path, 'w') as f:
                f['d_ppg'] = np.arange(600, dtype=float)[:, None]
                f['Xsub'] = np.ones(3)
            bounds = []
            reader = SimpleNamespace(path=path, layout=SimpleNamespace(frames=600),
                                     _bounds=lambda *args: bounds.append(args))
            original = h5py.Dataset.__getitem__
            def guarded(ds, key):
                self.assertEqual(ds.name, '/d_ppg')
                return original(ds, key)
            with patch.object(h5py.Dataset, '__getitem__', guarded):
                windows = read_reference_windows(reader)
            np.testing.assert_array_equal(windows.reshape(-1), np.arange(600))
            self.assertEqual(bounds, [(0,300,600,8), (300,600,600,8)])

    def test_verified_cache_needs_no_reference_and_detects_hash_mismatch(self):
        _, rgb, valid, _ = next(control_fixtures())
        p = predict_recording(rgb, valid)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'retained_arrays').mkdir()
            path = root/'retained_arrays/P000001.npz'
            np.savez_compressed(path, baseline_rgb=rgb, baseline_acquisition_valid=valid,
                                frame_indices=np.arange(600), roi_metric_clipped_fraction=np.zeros((3,600)))
            hashes = [array_sha256(x) for x in p['batch']['x']]
            with patch.object(h5py.Dataset, '__getitem__', side_effect=AssertionError('No HDF5 values in blind phase')):
                cache = load_verified_baseline(root, 'P000001.mat', source_sha256(path), hashes)
            np.testing.assert_array_equal(cache['prediction']['signals'], p['signals'])
            with self.assertRaises(ValueError): load_verified_baseline(root, 'P000001.mat', 'wrong', hashes)
            with self.assertRaises(ValueError): load_verified_baseline(root, 'P000001.mat', source_sha256(path), ['wrong']*2)

    def test_fresh_output_guard(self):
        with self.assertRaises(ValueError): validate_output(Path('docs'), Path('SCAMPSData'), Path('outputs/prior'))
        with self.assertRaises(ValueError): validate_output(Path('outputs/prior/child'), Path('SCAMPSData'), Path('outputs/prior'))

    def test_metadata_nonfinite_values_are_null_and_all_rows_retained(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            write_json(path/'a.json', dict(value=np.nan, count=np.int64(2)))
            self.assertIsNone(json.loads((path/'a.json').read_text())['value'])
            write_csv(path/'a.csv', [dict(id=1, metrics=dict(x=np.nan)), dict(id=2, metrics=dict(x=1.))])
            self.assertEqual(len((path/'a.csv').read_text().splitlines()), 3)

    def test_corruption_controls_keep_384_cells_and_constant_abstention(self):
        rows = run_controls()
        self.assertEqual(len(rows), 384)
        constant = [r for r in rows if r['filename'] == 'constant_nonpulsatile']
        self.assertEqual(len(constant), 48)
        self.assertFalse(any(r['metrics']['prediction_usable'] for r in constant))
        missing = [r for r in rows if r['filename'] == 'missing_windows' and r['start'] == 300]
        self.assertEqual(len(missing), 24)
        self.assertFalse(any(r['computational_success'] for r in missing))

    def test_coherent_illumination_is_not_identifiable_as_physiology(self):
        _, rgb, valid, reference = list(control_fixtures())[4]
        p = predict_recording(rgb, valid)
        q = trace_quality(p['signals'][0, 0, 0])
        self.assertGreater(q['spectral_concentration'], .8)
        self.assertAlmostEqual(q['dominant_hr_bpm'], 72, delta=.2)

    def test_timing_controls_expose_existing_fusion_phase_risk(self):
        rows = timing_controls()
        self.assertEqual(len(rows), 7)
        self.assertNotEqual(rows[-1]['outcome'], 'PASS_SYNTHETIC_ONLY')
        self.assertEqual(rows[-1]['weights'], [1/3]*3)

    def test_val_test_do_not_enter_quality_association_support(self):
        q = dict(spectral_concentration=1., snr_db=1., periodicity=1., half_hr_difference_bpm=0.,
                 adjacent_window_hr_difference_bpm=0., inter_roi_hr_range_bpm=0., clipping_fraction_mean=0., rgb_step_mean=0.)
        rows = [dict(filename='val', split='Val', roi='forehead', method='GREEN', quality=q,
                     metrics=dict(hr_abs_error_bpm=0., zero_lag_r=1.))]
        result = quality_associations(rows)
        self.assertTrue(all(r['finite_recording_pairs'] == 0 for r in result))


if __name__ == '__main__':
    unittest.main()
