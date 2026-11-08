"""Gate 09 manufactured-window tests; no participant or actual SCAMPS inference."""
from copy import deepcopy
import csv
import inspect
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from rppg_lab.waveform_v2_frozen_comparison import (CHECKPOINT, MANIFEST, CHECKPOINT_SHA256,
    MANIFEST_SHA256, INPUT_KEYS, ORDER, load_official, state_fingerprint, validate_inputs,
    infer_one, harmonic_diagnostics, scored_row, aggregate_conditions, paired_comparisons,
    run_neural_controls, neural_timing)
from rppg_lab.waveform_v2_classical_study import (control_fixtures, predict_recording,
    array_sha256, score_waveform, METHODS, CONDITIONS)
from rppg_lab.waveform_v2_frozen_cache import (validate_indices, validate_output, load_references,
    verified_artifact, clean, assert_hashes_unchanged)
from rppg_lab.artifacts import file_sha256
from rppg_lab.waveform_training import reconstruct_waveform_model
from rppg_lab.waveform_v2_timing_contract import demonstration_cases


class OfficialIdentityTests(unittest.TestCase):
    def test_official_identity_strict_reconstruction(self):
        model, identity = load_official()
        self.assertEqual(identity['model_sha256'], CHECKPOINT_SHA256)
        self.assertEqual(identity['freeze_manifest_sha256'], MANIFEST_SHA256)
        self.assertTrue(identity['strict_state_dict'])
        self.assertFalse(model.training)
        self.assertEqual(identity['roi_ordering'], CONDITIONS[:3])
        self.assertEqual(identity['prior_ordering'], METHODS)
        self.assertEqual((identity['fs'], identity['window_length_samples']), (30, 300))
        self.assertFalse(identity['subject_adaptation_enabled'])

    def test_bad_checkpoint_rejected_before_deserialization(self):
        with patch('rppg_lab.waveform_v2_frozen_comparison.file_sha256', return_value='bad'), patch('torch.load') as load:
            with self.assertRaises(ValueError): load_official()
            load.assert_not_called()

    def test_bad_manifest_rejected_before_deserialization(self):
        with patch('rppg_lab.waveform_v2_frozen_comparison.file_sha256', side_effect=[CHECKPOINT_SHA256, 'bad']), patch('torch.load') as load:
            with self.assertRaises(ValueError): load_official()
            load.assert_not_called()

    def test_missing_official_is_not_substituted(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError): load_official(Path(tmp)/'absent.pt', MANIFEST)

    def test_existing_strict_loader_rejects_missing_state_key(self):
        checkpoint = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
        checkpoint['model_state'].pop(next(iter(checkpoint['model_state'])))
        with self.assertRaises(RuntimeError): reconstruct_waveform_model(checkpoint)


class FrozenInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, _ = load_official()
        _, rgb, valid, cls.reference = next(control_fixtures())
        cls.prepared = predict_recording(rgb, valid)['batch']

    def batch(self):
        return {k: self.prepared[k][:1].copy() for k in INPUT_KEYS}

    def test_synthetic_inference_is_bit_deterministic(self):
        a, b = infer_one(self.model, self.batch()), infer_one(self.model, self.batch())
        for k in ('waveform', 'roi_attention', 'prior_weights', 'scaled_residuals'):
            np.testing.assert_array_equal(a[k], b[k])

    def test_strict_inference_mode_and_no_gradients(self):
        observed = []
        hook = self.model.register_forward_pre_hook(lambda *_: observed.append((torch.is_grad_enabled(), torch.is_inference_mode_enabled())))
        try: infer_one(self.model, self.batch())
        finally: hook.remove()
        self.assertEqual(observed, [(False, True)])
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_input_masks_and_entire_state_are_unchanged(self):
        batch = self.batch(); before = deepcopy(batch); state = state_fingerprint(self.model)
        infer_one(self.model, batch)
        self.assertEqual(state, state_fingerprint(self.model))
        for key in INPUT_KEYS: np.testing.assert_array_equal(batch[key], before[key])

    def test_reference_is_absent_from_inference_api(self):
        self.assertEqual(tuple(inspect.signature(infer_one).parameters), ('model', 'batch', 'flat_source'))
        out = infer_one(self.model, self.batch()); before = out['waveform'].copy()
        for ref in (self.reference[:300], -self.reference[:300], self.reference[:300][::-1]):
            score_waveform(out['waveform'], ref)
        np.testing.assert_array_equal(before, out['waveform'])

    def test_quality_and_validity_dtypes_are_strict(self):
        for key in INPUT_KEYS:
            batch = self.batch(); batch[key] = batch[key].astype('float64')
            with self.assertRaises(ValueError): validate_inputs(batch)

    def test_shape_mismatch_is_integrity_failure(self):
        batch = self.batch(); batch['x'] = batch['x'][..., :-1]
        with self.assertRaises(ValueError): infer_one(self.model, batch)

    def test_nonfinite_active_input_is_rejected(self):
        batch = self.batch(); batch['x'][0, 0, 0, 12] = np.nan
        with self.assertRaises(ValueError): infer_one(self.model, batch)

    def test_inconsistent_mask_and_quality_are_rejected(self):
        for kind in ('prior', 'quality'):
            batch = self.batch()
            if kind == 'prior': batch['prior_valid'][0, 0] = False
            else: batch['roi_quality'][0, 0] = 1.1
            with self.assertRaises(ValueError): validate_inputs(batch)

    def test_missing_roi_has_exact_zero_allocation(self):
        batch = self.batch(); batch['roi_valid'][0, 0] = False; batch['prior_valid'][0, 0] = False
        batch['x'][0, 0] = np.nan
        out = infer_one(self.model, batch)
        self.assertEqual(out['roi_attention'][0], 0)
        np.testing.assert_array_equal(out['prior_weights'][0], 0)
        self.assertTrue(np.isfinite(out['waveform']).all())

    def test_all_missing_does_not_call_network(self):
        batch = self.batch(); batch['roi_valid'][:] = False; batch['prior_valid'][:] = False
        with patch.object(self.model, 'forward', side_effect=AssertionError('Must not call')):
            out = infer_one(self.model, batch)
        self.assertFalse(out['forward_executed'])
        self.assertEqual(out['prediction_failure_reason'], 'NO_USABLE_ROI')
        self.assertEqual(out['waveform'].shape, (300,))

    def test_flat_source_is_never_pulse_eligible(self):
        out = infer_one(self.model, self.batch(), flat_source=True)
        self.assertTrue(out['computational_success'])
        self.assertFalse(out['pulse_estimate_eligible'])
        self.assertEqual(out['prediction_failure_reason'], 'FLAT_SOURCE_RGB')

    def test_forward_failure_is_retained(self):
        with patch.object(self.model, 'forward', side_effect=RuntimeError('manufactured failure')):
            out = infer_one(self.model, self.batch())
        self.assertEqual(out['prediction_failure_reason'], 'FORWARD_FAILURE')
        self.assertEqual(out['waveform'].shape, (300,))

    def test_malformed_output_shape_is_failure(self):
        with patch.object(self.model, 'forward', return_value={'ppg': torch.zeros(1, 299)}):
            out = infer_one(self.model, self.batch())
        self.assertEqual(out['prediction_failure_reason'], 'INVALID_OUTPUT_SHAPE')

    def test_training_mode_is_refused(self):
        with patch.object(self.model, 'training', True):
            with self.assertRaises(ValueError): infer_one(self.model, self.batch())

    def test_control_denominator_and_expectations(self):
        rows, _ = run_neural_controls(self.model)
        self.assertEqual(len(rows), 16)
        constant = [r for r in rows if r['filename'] == 'constant_nonpulsatile']
        self.assertTrue(all(not r['pulse_estimate_eligible'] for r in constant))
        missing = [r for r in rows if r['filename'] == 'missing_windows' and r['start'] == 300][0]
        self.assertEqual(missing['prediction_failure_reason'], 'NO_USABLE_ROI')
        self.assertFalse(missing['forward_executed'])
        self.assertTrue(all(r['physiological_validation'] == 'UNVALIDATED_SYNTHETIC_DIAGNOSTIC' for r in rows))

    def test_neural_timing_retains_all_cases_and_anchors(self):
        timing = neural_timing(self.model)
        self.assertEqual(len(timing['neural']), 16)
        self.assertEqual(len(timing['analytical']), 6)
        self.assertEqual({r['anchor'] for r in timing['neural']}, {'nominal_undelayed', 'actual_first_roi'})
        self.assertTrue(all(r['scope'] == 'synthetic_only' for r in timing['neural']))
        self.assertTrue(all(r['sample_spacing_ms'] >= 33.3 for r in timing['neural']))


class MetricAndMatchingTests(unittest.TestCase):
    def setUp(self):
        self.t = np.arange(300)/30; self.reference = np.sin(2*np.pi*self.t)

    def row(self, name='a', condition='frozen_v1', signal=None, eligible=True):
        return scored_row(name, 'Train', 0, condition, self.reference if signal is None else signal,
                          self.reference, eligible=eligible, computational=eligible, reason='' if eligible else 'NO_USABLE_ROI')

    def test_signed_native_polarity_and_lag_are_separate(self):
        r = self.row(signal=-self.reference)
        self.assertAlmostEqual(r['metrics']['zero_lag_r'], -1, places=6)
        self.assertGreater(r['metrics']['lag_aligned_r'], .99)
        self.assertGreater(r['metrics']['alignment_gain_r'], 1.9)

    def test_double_and_half_peak_flags(self):
        for pred, ref, flag in ((120., 60., 'suspected_double_peak'), (60., 120., 'suspected_half_peak')):
            metrics = dict(pred_hr_bpm=pred, reference_hr_bpm=ref)
            result = harmonic_diagnostics(self.reference, self.reference, metrics)
            self.assertTrue(result[flag]); self.assertFalse(result['other_peak_discrepancy'])

    def test_harmonics_from_manufactured_spectrum(self):
        signal = .1*self.reference+np.sin(4*np.pi*self.t)
        metrics = score_waveform(signal, self.reference)
        result = harmonic_diagnostics(signal, self.reference, metrics)
        self.assertTrue(result['suspected_double_peak'])

    def test_harmonic_threshold_is_fixed_and_other_errors_retained(self):
        result = harmonic_diagnostics(self.reference, self.reference, dict(pred_hr_bpm=90., reference_hr_bpm=60.))
        self.assertTrue(result['other_peak_discrepancy'])
        result = harmonic_diagnostics(self.reference, self.reference, dict(pred_hr_bpm=np.nan, reference_hr_bpm=60.))
        self.assertIsNone(result['suspected_double_peak'])

    def test_full_failure_denominator_preserves_empty_recording(self):
        a = self.row(); b = self.row('b', signal=np.full(300, np.nan), eligible=False)
        summary = aggregate_conditions([a, b])[0]
        self.assertEqual(summary['pooled']['intended_cells'], 2)
        self.assertEqual(summary['recording_balanced']['intended_recordings'], 2)
        self.assertEqual(summary['recording_balanced']['finite_recordings']['zero_lag_r'], 1)
        self.assertEqual(summary['per_recording']['b']['intended_cells'], 1)

    def test_balanced_rmse_uses_mse_before_root(self):
        a, b = self.row(), self.row('b')
        a['metrics']['hr_squared_error'], b['metrics']['hr_squared_error'] = 0., 100.
        summary = aggregate_conditions([a, b])[0]
        self.assertAlmostEqual(summary['recording_balanced']['hr_rmse_bpm'], np.sqrt(50))

    def test_paired_missingness_and_all_24_comparators(self):
        rows = [self.row(), self.row('b'), self.row(condition=ORDER[1]),
                self.row('b', ORDER[1], np.full(300, np.nan), False)]
        pairs = paired_comparisons(rows)
        self.assertEqual(len(pairs), 24)
        self.assertEqual(pairs[0]['intended_recordings'], 2)
        self.assertEqual(pairs[0]['metrics']['zero_lag_r']['paired_windows'], 1)
        self.assertEqual(pairs[1]['metrics']['zero_lag_r']['paired_windows'], 0)

    def test_exact_reference_indices_required(self):
        validate_indices(np.arange(300, 600), 300, 600)
        for values in (np.arange(1, 301), np.arange(299), np.arange(300).astype(float)):
            with self.assertRaises(ValueError): validate_indices(values, 0, 300)

    def test_reference_archive_hash_and_values_match_all_comparators(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); np.savez(root/'reference_windows.npz', P000001=self.reference[None])
            records = [dict(filename='P000001.mat', start=0, stop=300, indices=np.arange(300))]
            fields = ('filename', 'start', 'roi', 'method', 'raw_reference_sha256')
            with (root/'per_window_metrics.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
                for roi in CONDITIONS:
                    for method in METHODS:
                        writer.writerow(dict(filename='P000001.mat', start=0, roi=roi, method=method,
                                             raw_reference_sha256=array_sha256(self.reference)))
            with patch('rppg_lab.waveform_v2_frozen_cache.GATE08', root), patch('rppg_lab.waveform_v2_frozen_cache.verified_artifact', return_value=root/'reference_windows.npz'):
                refs, _ = load_references(root, records)
                np.testing.assert_array_equal(refs[('P000001.mat', 0)], self.reference)
                np.savez(root/'reference_windows.npz', P000001=(-self.reference)[None])
                with self.assertRaises(ValueError): load_references(root, records)

    def test_bad_gate08_artifact_hash_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'locked_predictions.npz').write_bytes(b'wrong')
            with self.assertRaises(ValueError): verified_artifact(root, 'locked_predictions.npz')

    def test_common_delay_preserves_relative_delay_but_fails_camera_anchor(self):
        cases = {r['case']: r for r in demonstration_cases(30)}
        common = cases['common_latency_100ms']
        self.assertEqual(common['outcome'], 'FAIL')
        self.assertEqual(common['reason'], 'camera_time_anchor_shift')
        self.assertLess(abs(common['differential_delay_error_ms']), 16.7)
        self.assertGreater(abs(common['face_anchor_shift_ms']), 80)

    def test_differential_delay_distortion_detected(self):
        cases = {r['case']: r for r in demonstration_cases(30)}
        self.assertEqual(cases['differential_hand_latency_85ms']['reason'], 'differential_delay_distortion')

    def test_safe_output_existing_outside_and_source_children_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, baseline = Path(tmp)/'source', Path(tmp)/'baseline'
            for output in (Path(tmp)/'outside', Path('outputs'), Path('outputs')/'../escape',
                           source/'child', baseline/'child'):
                with self.assertRaises(ValueError): validate_output(output, source, baseline)
            fresh = Path('outputs')/('unit_gate09_'+Path(tmp).name)
            self.assertEqual(validate_output(fresh, source, baseline), fresh.resolve())

    def test_private_reserved_and_symlink_output_paths_refused(self):
        for output in (Path('Dataset')/'out', Path('outputs/S007/result')):
            with self.assertRaises(ValueError): validate_output(output, Path('SCAMPSData'), Path('outputs/prior'))
        with patch.object(Path, 'is_symlink', return_value=True):
            with self.assertRaises(ValueError): validate_output(Path('outputs/fresh'), Path('SCAMPSData'), Path('outputs/prior'))

    def test_json_nonfinite_values_are_explicit_null(self):
        self.assertEqual(clean(dict(x=np.nan, y=np.array([1., np.inf]))), dict(x=None, y=[1., None]))

    def test_source_or_checkpoint_mutation_fails_immutability_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'source.bin'; path.write_bytes(b'original')
            snapshot = {path: file_sha256(path)}
            assert_hashes_unchanged(snapshot)
            path.write_bytes(b'changed')
            with self.assertRaises(RuntimeError): assert_hashes_unchanged(snapshot)


if __name__ == '__main__':
    unittest.main()
