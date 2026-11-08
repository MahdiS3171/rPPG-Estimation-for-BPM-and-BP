"""Tiny HDF5 / analytical traces only; no real recordings, models or training."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import weakref

import h5py
import numpy as np

from rppg_lab.classical import METHOD_FUNCS
from rppg_lab.datasets import UBFCMultiROIRPPGDataset
from rppg_lab.processing import resample_trace
from rppg_lab.region_masks import get_face_masks, extract_frame_rgb_by_roi
from rppg_lab.roi import FACE_ROIS
from rppg_lab.signals import standardize_channels, standardize_1d
from rppg_lab.types import RegionObservation, RGBTrace
from rppg_lab.window_priors import build_window_priors
from rppg_lab.waveform_v2_scamps import ScampsLayout, ScampsRecording, source_sha256
from rppg_lab.waveform_v2_scamps_roi import (ROI_NAMES, RGB_CHANNELS, ScampsROIConfig,
    ScampsROITraces, extract_scamps_roi_traces)
from rppg_lab.waveform_v2_scamps_inputs import (ScampsInputConfig, CHANNEL_NAMES, PRIOR_NAMES,
    prepare_scamps_v1_inputs, as_torch_inputs)
from rppg_lab.waveform_v2_scamps_mask_audit import continuous_edge_agreement, audit_mask_candidates


def points():
    value = np.full((478, 2), 16.)
    for name, (x, y) in zip(ROI_NAMES, ((16, 6), (8, 22), (24, 22))):
        ids = list(FACE_ROIS[name])
        angle = np.arange(len(ids)) * 2 * np.pi / len(ids)
        value[ids] = np.column_stack((x + 3 * np.cos(angle), y + 3 * np.sin(angle)))
    return value


class FakeDetector:
    def __init__(self, missing=(), error=(), bad_points=False, empty_cheek=False, change_identity=False):
        self.calls, self.missing, self.error = [], missing, error
        self.bad_points, self.empty_cheek, self.change_identity = bad_points, empty_cheek, change_identity
        self.closed = False

    def detect(self, frame, index, coordinate):
        self.calls.append((index, coordinate))
        if index in self.error:
            raise RuntimeError("fixture detector failure")
        if index in self.missing:
            return RegionObservation(index, coordinate, reason="face_not_detected")
        value = points()
        if self.bad_points:
            value[0] = np.nan
        if self.empty_cheek:
            value[list(FACE_ROIS['left_cheek'])] = [8, 22]
        return RegionObservation(index + int(self.change_identity), coordinate, landmarks=value, valid=True)

    def close(self):
        self.closed = True


def hdf_fixture(root):
    path = root / "P000001.mat"
    with h5py.File(path, "w") as f:
        for name, width in (("RawFrames", 40), ("Xsub", 32)):
            c, w, h, t = np.indices((3, width, 32, 6))
            values = .3 + c * .1 + w * .0002 + h * .0001 + t * .001
            f.create_dataset(name, data=values, chunks=(1, 8, 8, 2), compression="gzip")
        for name in ("skin_mask", "combined_mask"):
            f[name] = np.zeros((32, 32, 5), dtype=np.float32)
        f["d_ppg"] = np.arange(6, dtype=float)[:, None]
    return path


def analytical_traces():
    t = np.arange(600) / 30.
    rgb = np.stack([np.column_stack((100 + 2*np.sin(2*np.pi*1.2*t + r*.1),
                                    120 + 4*np.sin(2*np.pi*1.2*t + r*.1),
                                    80 + np.cos(2*np.pi*1.2*t + r*.1))) for r in range(3)])
    return ScampsROITraces(rgb, np.ones((3, 600), dtype=bool), np.arange(600), np.arange(600),
        np.sin(2*np.pi*1.2*t), np.full((3, 600), 100), np.full((3, 600), .1),
        np.full((3, 600), "", dtype="U80"), np.ones(600, dtype=bool), t,
        dict(coordinate_kind="documented_nominal_synthetic_frame_grid", capture_timestamps=None,
             nominal_rate_hz=30., physical_clock_verified=False, segmentation_used=False,
             subject_adaptation_enabled=False, allow_training=False, source_sha256="0"*64))


class StreamingROITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = hdf_fixture(self.root)
        self.layout = ScampsLayout(frames=6, raw_width=40, crop_width=32, height=32)
        self.config = ScampsROIConfig(chunk_frames=2, min_pixels=5)

    def reader(self):
        return ScampsRecording(self.root, self.path.name, layout=self.layout, max_window_frames=4)

    def test_exact_v1_polygon_means_and_source_integrity(self):
        digest = source_sha256(self.path)
        detector = FakeDetector()
        reader = self.reader()
        result = extract_scamps_roi_traces(reader, detector, config=self.config)
        frame = reader.read_window(0, 1)['rgb'][0]
        geometry = np.rint(frame * 255).astype(np.uint8)
        obs = RegionObservation(0, 0., landmarks=points(), valid=True)
        measured = extract_frame_rgb_by_roi(frame*255, get_face_masks(geometry, obs, ROI_NAMES, False), 5)
        for r, name in enumerate(ROI_NAMES):
            np.testing.assert_array_equal(result.rgb[r, 0], measured[name][0])
        self.assertTrue(result.roi_valid.all())
        np.testing.assert_array_equal(result.frame_indices, np.arange(6))
        np.testing.assert_array_equal(result.label_indices, np.arange(6))
        self.assertEqual(source_sha256(self.path), digest)
        self.assertEqual(result.provenance['source_sha256'], digest)
        self.assertIn('window_priors', result.provenance['preprocessing_source_sha256'])

    def test_one_detection_per_original_frame_and_no_unverified_mask_reads(self):
        detector = FakeDetector()
        original = h5py.Dataset.__getitem__
        def no_masks(ds, key):
            self.assertNotIn('mask', ds.name)
            return original(ds, key)
        with patch.object(h5py.Dataset, '__getitem__', new=no_masks):
            result = extract_scamps_roi_traces(self.reader(), detector, config=self.config)
        self.assertEqual(detector.calls, [(i, i/30.) for i in range(6)])
        self.assertFalse(detector.closed)  # caller owns/reset/closes the detector
        self.assertFalse(result.provenance['segmentation_used'])
        self.assertFalse(result.provenance['geometric_fallback_used'])

    def test_nan_or_out_of_range_frames_remain_explicit_failures(self):
        with h5py.File(self.path, 'r+') as f:
            f['Xsub'][0, 0, 0, 2] = np.nan
            f['Xsub'][0, 0, 0, 3] = 1.2
        detector = FakeDetector()
        result = extract_scamps_roi_traces(self.reader(), detector, config=self.config)
        self.assertEqual(len(detector.calls), 4)
        self.assertFalse(result.roi_valid[:, 2:4].any())
        self.assertTrue(np.isnan(result.rgb[:, 2:4]).all())
        self.assertTrue((result.failure_reason[:, 2:4] == 'invalid_rgb_frame').all())
        np.testing.assert_array_equal(result.raw_d_ppg, np.arange(6))

    def test_missing_detection_does_not_reuse_or_substitute_geometry(self):
        result = extract_scamps_roi_traces(self.reader(), FakeDetector(missing=(3,)), config=self.config)
        self.assertFalse(result.roi_valid[:, 3].any())
        self.assertTrue(np.isnan(result.rgb[:, 3]).all())
        self.assertTrue((result.failure_reason[:, 3] == 'face_not_detected').all())
        self.assertTrue(result.roi_valid[:, 4].all())

    def test_detector_exceptions_are_counted_with_original_indices(self):
        result = extract_scamps_roi_traces(self.reader(), FakeDetector(error=(1,)), config=self.config)
        self.assertTrue((result.failure_reason[:, 1] == 'detector_error:RuntimeError').all())
        self.assertEqual(result.rgb.shape, (3, 6, 3))

    def test_nonfinite_landmarks_fail_without_roi_fallback(self):
        result = extract_scamps_roi_traces(self.reader(), FakeDetector(bad_points=True), config=self.config)
        self.assertFalse(result.roi_valid.any())
        self.assertTrue((result.failure_reason == 'invalid_landmarks').all())

    def test_detector_identity_change_is_fatal(self):
        with self.assertRaisesRegex(ValueError, 'identity'):
            extract_scamps_roi_traces(self.reader(), FakeDetector(change_identity=True), config=self.config)

    def test_one_empty_roi_is_not_replaced_with_another_roi(self):
        result = extract_scamps_roi_traces(self.reader(), FakeDetector(empty_cheek=True), config=self.config)
        self.assertFalse(result.roi_valid[1].any())
        self.assertTrue(np.isnan(result.rgb[1]).all())
        self.assertTrue(result.roi_valid[[0, 2]].all())
        self.assertTrue((result.failure_reason[1] == 'insufficient_roi_pixels').all())

    def test_malformed_roi_mask_rejected_instead_of_coerced(self):
        for bad in ({name: np.ones((31, 32), dtype=bool) for name in ROI_NAMES},
                    {name: np.ones((32, 32), dtype=float) for name in ROI_NAMES}):
            with patch('rppg_lab.waveform_v2_scamps_roi.get_face_masks', return_value=bad), self.assertRaises(ValueError):
                extract_scamps_roi_traces(self.reader(), FakeDetector(), config=self.config)

    def test_wrong_channel_or_index_contract_is_rejected(self):
        for change in ('channel', 'indices', 'shape'):
            reader = self.reader()
            read = reader.read_window
            def corrupted(*args, **kwargs):
                block = read(*args, **kwargs)
                if change == 'channel': block['provenance']['channel_order'] = 'BGR'
                if change == 'indices': block['label_indices'] += 1
                if change == 'shape': block['rgb'] = block['rgb'].transpose(0, 3, 1, 2)
                return block
            with self.subTest(change=change), patch.object(reader, 'read_window', side_effect=corrupted), self.assertRaises(ValueError):
                extract_scamps_roi_traces(reader, FakeDetector(), config=self.config)

    def test_budget_checked_before_frame_loading(self):
        reader = self.reader()
        with patch.object(reader, 'read_window', side_effect=AssertionError('frame allocation')), self.assertRaisesRegex(ValueError, 'budget'):
            extract_scamps_roi_traces(reader, FakeDetector(), config=replace(self.config, array_budget_bytes=1))
        for kw in (dict(chunk_frames=65), dict(nominal_rate_hz=29.), dict(min_pixels=0), dict(min_detection_confidence=float('nan'))):
            with self.assertRaises(ValueError):
                ScampsROIConfig(**kw)

    def test_chunk_size_invariance_and_determinism(self):
        first = extract_scamps_roi_traces(self.reader(), FakeDetector(), config=self.config)
        second = extract_scamps_roi_traces(self.reader(), FakeDetector(), config=replace(self.config, chunk_frames=3))
        np.testing.assert_array_equal(first.rgb, second.rgb)
        np.testing.assert_array_equal(first.roi_valid, second.roi_valid)
        self.assertEqual(first.provenance['coordinate_kind'], 'documented_nominal_synthetic_frame_grid')
        self.assertIsNone(first.provenance['capture_timestamps'])
        self.assertFalse(first.provenance['subject_adaptation_enabled'])

    def test_previous_chunk_released_before_next_read(self):
        reader = self.reader()
        read, references = reader.read_window, []
        def bounded(*args, **kwargs):
            if references:
                self.assertIsNone(references[-1]())
            block = read(*args, **kwargs)
            references.append(weakref.ref(block['rgb']))
            return block
        with patch.object(reader, 'read_window', side_effect=bounded):
            extract_scamps_roi_traces(reader, FakeDetector(), config=self.config)
        self.assertEqual(len(references), 3)

    def test_ppg_values_cannot_influence_rgb_geometry_or_validity(self):
        one = extract_scamps_roi_traces(self.reader(), FakeDetector(), config=self.config)
        with h5py.File(self.path, 'r+') as f:
            f['d_ppg'][:] = -500
        two = extract_scamps_roi_traces(self.reader(), FakeDetector(), config=self.config)
        np.testing.assert_array_equal(one.rgb, two.rgb)
        np.testing.assert_array_equal(one.roi_valid, two.roi_valid)
        np.testing.assert_array_equal(one.pixel_count, two.pixel_count)


class InputPreparationTests(unittest.TestCase):
    def test_shape_orders_masks_and_exact_existing_helpers(self):
        traces = analytical_traces()
        result = prepare_scamps_v1_inputs(traces)
        self.assertEqual(result['x'].shape, (2, 3, 9, 300))
        self.assertEqual(result['x'].dtype, np.float32)
        self.assertEqual(result['channel_names'], CHANNEL_NAMES)
        self.assertTrue(result['roi_valid'].all() and result['prior_valid'].all())
        for r in range(3):
            expected, success = build_window_priors(traces.rgb[r, :300], 30.)
            np.testing.assert_array_equal(result['x'][0, r, :3], standardize_channels(traces.rgb[r, :300]).T)
            np.testing.assert_array_equal(result['x'][0, r, 3:], expected)
        np.testing.assert_array_equal(result['y_ppg'][0], standardize_1d(traces.raw_d_ppg[:300]))
        np.testing.assert_array_equal(result['raw_d_ppg'][0], traces.raw_d_ppg[:300])
        np.testing.assert_array_equal(result['frame_indices'], result['label_indices'])

    def test_matches_v1_dataset_item_without_model_forward(self):
        traces = analytical_traces()
        old = UBFCMultiROIRPPGDataset.__new__(UBFCMultiROIRPPGDataset)
        old.roi_names, old.prior_names, old.channel_names = ROI_NAMES, PRIOR_NAMES, CHANNEL_NAMES
        old.fs_target, old.min_valid_fraction, old.samples = 30., .9, [(0, 0, 300)]
        old.records = [dict(rgb=traces.rgb, timestamps=traces.nominal_grid_seconds,
            ppg=traces.raw_d_ppg, hr=np.full(600, 72.), subject_id='analytical_fixture',
            traces={name: RGBTrace(traces.nominal_grid_seconds, traces.rgb[r], name, traces.roi_valid[r])
                    for r, name in enumerate(ROI_NAMES)})]
        expected, actual = old[0], prepare_scamps_v1_inputs(traces, (0,))
        for name in ('x', 'roi_valid', 'roi_quality', 'prior_valid', 'y_ppg'):
            np.testing.assert_array_equal(expected[name].numpy(), actual[name][0])

    def test_wrong_roi_prior_channel_order_rate_and_adaptation_refused(self):
        for kw in (dict(roi_names=ROI_NAMES[::-1]), dict(prior_names=PRIOR_NAMES[::-1]),
                   dict(nominal_rate_hz=60.), dict(window_samples=299), dict(subject_adaptation_enabled=True)):
            with self.assertRaises(ValueError): ScampsInputConfig(**kw)
        traces = analytical_traces()
        for modified in (replace(traces, roi_names=ROI_NAMES[::-1]), replace(traces, rgb_channel_names=RGB_CHANNELS[::-1])):
            with self.assertRaises(ValueError): prepare_scamps_v1_inputs(modified)

    def test_nonfinite_or_long_missing_roi_keeps_masks_and_quality(self):
        traces = analytical_traces()
        traces.roi_valid[1, 100:200] = False
        traces.rgb[1, 100:200] = np.nan
        result = prepare_scamps_v1_inputs(traces, (0,))
        self.assertFalse(result['roi_valid'][0, 1])
        self.assertFalse(result['prior_valid'][0, 1].any())
        self.assertEqual(np.count_nonzero(result['x'][0, 1]), 0)
        self.assertAlmostEqual(float(result['roi_quality'][0, 1]), 2/3, places=6)
        self.assertEqual(result['source_roi_valid'][0, 1].sum(), 200)
        self.assertTrue(np.isnan(result['raw_rgb'][0, 1, 100:200]).all())

    def test_bounded_gap_matches_v1_interpolation_and_records_imputation(self):
        traces = analytical_traces()
        traces.roi_valid[0, 100] = False
        traces.rgb[0, 100] = np.nan
        result = prepare_scamps_v1_inputs(traces, (0,))
        trace = RGBTrace(traces.nominal_grid_seconds, traces.rgb[0], ROI_NAMES[0], traces.roi_valid[0])
        filled, flags = resample_trace(trace, traces.nominal_grid_seconds, .15)
        np.testing.assert_array_equal(result['x'][0, 0, :3], standardize_channels(filled[:300]).T)
        expected, _ = build_window_priors(filled[:300], 30.)
        np.testing.assert_array_equal(result['x'][0, 0, 3:], expected)
        self.assertTrue(result['interpolated'][0, 0, 100])
        self.assertFalse(result['source_roi_valid'][0, 0, 100])
        self.assertTrue(np.isnan(traces.rgb[0, 100]).all())
        self.assertAlmostEqual(float(result['roi_quality'][0, 0]), 299/300, places=6)

    def test_endpoint_loss_is_not_extrapolated(self):
        traces = analytical_traces()
        traces.roi_valid[0, 0] = False
        traces.rgb[0, 0] = np.nan
        result = prepare_scamps_v1_inputs(traces, (0,))
        self.assertFalse(result['roi_valid'][0, 0])
        self.assertFalse(result['interpolated'][0, 0, 0])
        self.assertTrue(result['roi_valid'][0, 1:].all())

    def test_quality_threshold_uses_v1_float64_fraction_before_reporting_float32(self):
        traces = analytical_traces()
        missing = np.arange(5, 300, 10)
        traces.roi_valid[0, missing] = False
        traces.rgb[0, missing] = np.nan
        result = prepare_scamps_v1_inputs(traces, (0,), config=ScampsInputConfig(min_valid_fraction=.90000001))
        self.assertFalse(result['roi_valid'][0, 0])
        exact_boundary = prepare_scamps_v1_inputs(traces, (0,))
        self.assertTrue(exact_boundary['roi_valid'][0, 0])

    def test_finite_but_overflowing_label_variance_is_an_explicit_invalid_target(self):
        traces = analytical_traces()
        traces.raw_d_ppg[:] = np.tile([1e308, -1e308], 300)
        result = prepare_scamps_v1_inputs(traces, (0,))
        self.assertFalse(result['target_valid'][0])
        self.assertTrue(np.isfinite(result['y_ppg']).all())
        np.testing.assert_array_equal(result['raw_d_ppg'][0], traces.raw_d_ppg[:300])

    def test_all_invalid_rois_preserve_attempted_windows(self):
        traces = analytical_traces()
        traces.roi_valid[:] = False
        traces.rgb[:] = np.nan
        result = prepare_scamps_v1_inputs(traces)
        self.assertEqual(result['intended_windows'], 2)
        self.assertEqual(result['status'], ['NO_USABLE_ROI'] * 2)
        self.assertEqual(np.count_nonzero(result['x']), 0)
        self.assertFalse(result['roi_valid'].any() or result['prior_valid'].any())

    def test_target_changes_do_not_change_rgb_priors_or_quality(self):
        traces = analytical_traces()
        one = prepare_scamps_v1_inputs(traces, (0,))
        traces.raw_d_ppg[:] = np.nan
        two = prepare_scamps_v1_inputs(traces, (0,))
        for k in ('x', 'roi_valid', 'roi_quality', 'prior_valid'):
            np.testing.assert_array_equal(one[k], two[k])
        self.assertEqual(two['status'], ['INVALID_TARGET'])
        self.assertFalse(two['label_valid'].any() or two['target_valid'].any())
        self.assertTrue(np.isnan(two['raw_d_ppg']).all())
        self.assertEqual(np.count_nonzero(two['y_ppg']), 0)

    def test_failed_prior_stays_zero_and_invalid_without_dropping_other_priors(self):
        with patch.dict(METHOD_FUNCS, {'CHROM': lambda *_: np.full(300, np.inf)}):
            result = prepare_scamps_v1_inputs(analytical_traces(), (0,))
        self.assertFalse(result['prior_valid'][:, :, 1].any())
        self.assertEqual(np.count_nonzero(result['x'][:, :, 4]), 0)
        self.assertTrue(result['roi_valid'].all())

    def test_all_prior_failures_remain_in_denominator(self):
        def failed(*_): raise RuntimeError('synthetic failure')
        with patch.dict(METHOD_FUNCS, {name: failed for name in PRIOR_NAMES}):
            result = prepare_scamps_v1_inputs(analytical_traces())
        self.assertEqual(result['intended_windows'], 2)
        self.assertEqual(np.count_nonzero(result['x']), 0)
        np.testing.assert_array_equal(result['roi_quality'], np.ones((2, 3)))
        self.assertFalse(result['input_valid'].any())

    def test_outside_window_values_do_not_change_window_preprocessing(self):
        traces = analytical_traces()
        first = prepare_scamps_v1_inputs(traces, (0,))
        traces.rgb[:, 300:] *= .1
        traces.raw_d_ppg[300:] = 999
        second = prepare_scamps_v1_inputs(traces, (0,))
        np.testing.assert_array_equal(first['x'], second['x'])
        np.testing.assert_array_equal(first['y_ppg'], second['y_ppg'])

    def test_constant_and_extreme_values_have_explicit_stable_outcomes(self):
        traces = analytical_traces()
        traces.rgb[:] = 100
        traces.raw_d_ppg[:] = 1
        result = prepare_scamps_v1_inputs(traces, (0,))
        self.assertTrue(np.isfinite(result['x']).all())
        self.assertFalse(result['target_valid'].any())
        traces.rgb[0, 0] = 1e308
        with self.assertRaises(ValueError): prepare_scamps_v1_inputs(traces)

    def test_fictitious_timestamp_or_index_claims_rejected(self):
        for key, value in (('capture_timestamps', []), ('physical_clock_verified', True),
                           ('nominal_rate_hz', 29.), ('segmentation_used', True), ('allow_training', True)):
            traces = analytical_traces()
            traces.provenance[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): prepare_scamps_v1_inputs(traces)
        for field in ('frame_indices', 'label_indices', 'nominal_grid_seconds'):
            traces = analytical_traces()
            setattr(traces, field, getattr(traces, field) + 1)
            with self.subTest(field=field), self.assertRaises(ValueError): prepare_scamps_v1_inputs(traces)

    def test_explicit_window_boundaries_and_no_context_requirement(self):
        traces = analytical_traces()
        for starts in ((), (-1,), (301,), (0, 0), (0.,)):
            with self.assertRaises(ValueError): prepare_scamps_v1_inputs(traces, starts)
        result = prepare_scamps_v1_inputs(traces, (0,))
        self.assertEqual(result['provenance']['calibration_frame_indices'], [])
        self.assertFalse(result['provenance']['subject_adaptation_enabled'])

    def test_deterministic_cpu_tensors_without_network_call(self):
        import torch
        batch = prepare_scamps_v1_inputs(analytical_traces(), (0,))
        again = prepare_scamps_v1_inputs(analytical_traces(), (0,))
        np.testing.assert_array_equal(batch['x'], again['x'])
        with patch.object(torch.nn.Module, '__call__', side_effect=AssertionError('neural network invoked')):
            tensors = as_torch_inputs(batch)
        self.assertEqual(tuple(tensors['x'].shape), (1, 3, 9, 300))
        self.assertEqual(tensors['x'].dtype, torch.float32)
        self.assertEqual(tensors['x'].device.type, 'cpu')


class MaskGeometryTests(unittest.TestCase):
    def test_controlled_spatial_edge_can_distinguish_geometry_without_thresholds(self):
        mask = np.zeros((32, 32)); mask[:, 16:] = 1.004
        frames = []
        for boundary in (8, 16, 24):
            frame = np.zeros((32, 32, 3)); frame[:, boundary:] = .8
            frames.append(frame)
        scores = [continuous_edge_agreement(f, mask) for f in frames]
        self.assertGreater(scores[1], .99)
        self.assertEqual(scores[0], 0.)
        self.assertEqual(scores[2], 0.)
        self.assertEqual(mask.max(), 1.004)

    def test_identical_static_candidates_are_indistinguishable(self):
        rgb = np.random.default_rng(8).uniform(size=(32, 32, 3))
        mask = np.random.default_rng(9).uniform(-.004, 1.004, size=(32, 32))
        scores = [continuous_edge_agreement(rgb.copy(), mask) for _ in range(3)]
        self.assertEqual(scores[0], scores[1])
        self.assertEqual(scores[1], scores[2])

    def test_flat_or_malformed_edge_inputs_fail_explicitly(self):
        self.assertIsNone(continuous_edge_agreement(np.zeros((4, 4, 3)), np.zeros((4, 4))))
        with self.assertRaises(ValueError): continuous_edge_agreement(np.zeros((4, 4, 3)), np.zeros((3, 4)))
        with self.assertRaises(ValueError): continuous_edge_agreement(np.full((4, 4, 3), np.nan), np.zeros((4, 4)))

    def test_candidate_audit_never_reads_ppg_or_applies_offset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = hdf_fixture(root)
            reader = ScampsRecording(root, path.name, layout=ScampsLayout(frames=6, raw_width=40, crop_width=32, height=32))
            digest = source_sha256(path)
            original = h5py.Dataset.__getitem__
            def no_labels(ds, key):
                self.assertNotEqual(ds.name, '/d_ppg')
                return original(ds, key)
            with patch.object(h5py.Dataset, '__getitem__', new=no_labels):
                result = audit_mask_candidates(reader, indices=(1, 3))
            self.assertEqual(result['decision'], 'UNVERIFIED')
            self.assertFalse(result['ppg_read_or_scored'] or result['candidate_offsets_applied'])
            self.assertEqual(result['rows'][0]['candidate_frame_indices'], [0, 1, 2])
            self.assertEqual(digest, source_sha256(path))
            with self.assertRaises(ValueError): audit_mask_candidates(reader, (0,))


if __name__ == '__main__':
    unittest.main()
