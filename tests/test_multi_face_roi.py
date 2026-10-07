"""Deterministic multi-ROI geometry, acquisition, missingness and artifact checks."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from rppg_lab.artifacts import reserve_run, save_result
from rppg_lab.classical import DEFAULT_PRIORS
from rppg_lab.config import PipelineConfig
from rppg_lab.detection import BaseRegionDetector
from rppg_lab.pipeline import process_recording
from rppg_lab.processing import build_classical_priors, extract_signal, resample_trace
from rppg_lab.region_masks import (
    DEFAULT_FACE_ROIS, get_face_masks, get_region_mask,
    extract_frame_rgb, extract_frame_rgb_by_roi, skin_mask,
)
from rppg_lab.roi import FACE_ROIS
from rppg_lab.types import RegionObservation
from rppg_lab.video import VideoReader


def face_points() -> np.ndarray:
    points = np.full((478, 2), 24.0)
    for name, (x, y, radius) in zip(FACE_ROIS, ((24, 12, 7), (12, 36, 9), (38, 36, 6))):
        indices = list(FACE_ROIS[name])
        theta = np.arange(len(indices)) * 2 * np.pi / len(indices)
        points[indices] = np.column_stack((x + radius * np.cos(theta), y + radius * np.sin(theta)))
    return points


def previous_combined_mask(frame: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Independent reference to the pre-refactor recording-path formula."""
    mask = np.zeros(frame.shape[:2], np.uint8)
    h, w = mask.shape
    for indices in FACE_ROIS.values():
        pts = np.rint(points[list(indices)]).astype(np.int32)
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        if len(np.unique(pts, axis=0)) >= 3:
            cv2.fillConvexPoly(mask, cv2.convexHull(pts), 1)
    return mask.astype(bool)


class CountingDetector(BaseRegionDetector):
    def __init__(self, site: str = "face", lose_cheek: bool = False, missing: bool = False):
        self.site, self.lose_cheek, self.missing = site, lose_cheek, missing
        self.calls: list[tuple[int, float]] = []
        self.closed = False

    def detect(self, frame_rgb: np.ndarray, frame_index: int, timestamp: float) -> RegionObservation:
        self.calls.append((frame_index, timestamp))
        if self.missing:
            return RegionObservation(frame_index, timestamp, reason="face_not_detected")
        points = face_points()
        if self.site == "hand":
            points = np.tile([54.0, 32.0], (21, 1))
        elif self.lose_cheek and (frame_index < 5 or 150 <= frame_index < 210 or
                                 300 <= frame_index < 302 or frame_index >= 595):
            points[list(FACE_ROIS["left_cheek"])] = [12, 36]
        return RegionObservation(frame_index, timestamp, bbox=(48, 16, 63, 48),
                                 landmarks=points, valid=True, identity=self.site)

    def close(self) -> None:
        self.closed = True


def write_synthetic_video(path: Path, count: int = 600) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (64, 64))
    if not writer.isOpened():
        raise RuntimeError("MJPG video writer unavailable")
    try:
        for index in range(count):
            t = index / 30
            frame = np.full((64, 64, 3), 100, np.uint8)
            frame[:, :48, 1] = int(round(120 + 10 * np.sin(2 * np.pi * 1.2 * t)))
            frame[:, 48:, 1] = int(round(120 + 10 * np.sin(2 * np.pi * 1.2 * (t - 0.12))))
            writer.write(frame)
    finally:
        writer.release()


class FaceMaskTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = np.zeros((64, 64, 3), np.uint8)
        self.obs = RegionObservation(0, 0.0, landmarks=face_points(), valid=True)

    def test_distinct_masks_and_exact_combined_union(self) -> None:
        masks = get_face_masks(self.frame, self.obs)
        self.assertEqual(tuple(masks), DEFAULT_FACE_ROIS)
        for name in FACE_ROIS:
            self.assertTrue(masks[name].any())
        components = [masks[name] for name in FACE_ROIS]
        for i, first in enumerate(components):
            for second in components[i + 1:]:
                self.assertFalse(np.array_equal(first, second))
                self.assertFalse(np.shares_memory(first, second))
        np.testing.assert_array_equal(masks["combined"], np.logical_or.reduce(components))

    def test_combined_matches_previous_hulls_with_overlap_and_clipping(self) -> None:
        rng = np.random.default_rng(14)
        frame = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        for points in (face_points(), rng.uniform(-20, 85, (478, 2))):
            obs = replace(self.obs, landmarks=points)
            expected = previous_combined_mask(frame, points)
            for gated in (False, True):
                reference = expected & skin_mask(frame) if gated else expected
                masks = get_face_masks(frame, obs, use_skin_mask=gated)
                np.testing.assert_array_equal(masks["combined"], reference)
                np.testing.assert_array_equal(masks["combined"], np.logical_or.reduce([masks[n] for n in FACE_ROIS]))
                color, _ = extract_frame_rgb_by_roi(frame, masks, 1)["combined"]
                np.testing.assert_allclose(color, frame[reference].mean(axis=0), rtol=0, atol=0)

    def test_rgb_means_weight_pixels_and_keep_zero_measurements(self) -> None:
        masks = get_face_masks(self.frame, self.obs)
        colors = {"forehead": [0, 0, 0], "left_cheek": [60, 90, 120], "right_cheek": [150, 180, 210]}
        for name, color in colors.items():
            self.frame[masks[name]] = color
        measured = extract_frame_rgb_by_roi(self.frame, masks, 5)
        for name, color in colors.items():
            mean, q = measured[name]
            np.testing.assert_array_equal(mean, color)
            self.assertTrue(q["extraction_valid"])
            self.assertEqual(q["skin_pixel_count"], int(masks[name].sum()))
            self.assertAlmostEqual(q["brightness"], np.mean(color) / 255)
        counts = np.array([masks[name].sum() for name in colors])
        expected = np.average(list(colors.values()), axis=0, weights=counts)
        np.testing.assert_allclose(measured["combined"][0], expected)
        self.assertFalse(np.allclose(expected, np.mean(list(colors.values()), axis=0)))

    def test_failed_roi_does_not_poison_other_measurements(self) -> None:
        masks = get_face_masks(self.frame, self.obs)
        masks["left_cheek"][:] = False
        measured = extract_frame_rgb_by_roi(self.frame, masks, 5)
        self.assertTrue(np.isnan(measured["left_cheek"][0]).all())
        self.assertFalse(measured["left_cheek"][1]["extraction_valid"])
        for name in ("forehead", "right_cheek", "combined"):
            self.assertTrue(np.isfinite(measured[name][0]).all())

    def test_nonfinite_pixels_excluded_from_valid_count(self) -> None:
        frame = np.array([[[0.0, 0, 0], [np.nan, 10, 20], [30, 60, 90]]])
        color, q = extract_frame_rgb(frame, np.ones((1, 3), bool), 2)
        np.testing.assert_array_equal(color, [15, 30, 45])
        self.assertEqual(q["skin_pixel_count"], 2)
        self.assertEqual(q["mask_pixel_count"], 3)
        self.assertEqual(q["mask_coverage"], 1)
        self.assertTrue(np.isnan(extract_frame_rgb(frame, np.ones((1, 3), bool), 3)[0]).all())

    def test_skin_gate_evaluated_once_and_full_skin_preserved(self) -> None:
        frame = np.full((64, 64, 3), [160, 110, 85], np.uint8)
        with patch("rppg_lab.region_masks.skin_mask", wraps=skin_mask) as gate:
            masks = get_face_masks(frame, self.obs, (*DEFAULT_FACE_ROIS, "full_skin"), True)
            self.assertEqual(gate.call_count, 1)
        for name, mask in masks.items():
            np.testing.assert_array_equal(mask, get_region_mask(frame, self.obs, "face", name, True))

    def test_missing_detection_retains_all_empty_masks(self) -> None:
        masks = get_face_masks(self.frame, replace(self.obs, valid=False))
        self.assertEqual(len(masks), 4)
        self.assertTrue(all(not mask.any() for mask in masks.values()))
        with self.assertRaises(ValueError):
            get_face_masks(self.frame, self.obs, ("unknown",))


class MultiFacePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.video = Path(cls.tmp.name) / "multi.avi"
        write_synthetic_video(cls.video)
        cls.t = np.arange(600) / 30
        cls.face_detector = CountingDetector(lose_cheek=True)
        cls.hand_detector = CountingDetector("hand")
        cls.config = PipelineConfig(hand_roi="bounding_box")
        with patch("rppg_lab.pipeline.VideoReader", wraps=VideoReader) as reader:
            cls.result = process_recording(cls.video, cls.config, timestamps=cls.t,
                                          detectors={"face": cls.face_detector, "hand": cls.hand_detector})
            cls.reader_calls = reader.call_count

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_single_decode_and_landmark_pass(self) -> None:
        self.assertEqual(self.reader_calls, 1)
        expected = list(enumerate(self.t))
        self.assertEqual(self.face_detector.calls, expected)
        self.assertEqual(self.hand_detector.calls, expected)
        self.assertFalse(self.face_detector.closed)
        self.assertFalse(self.hand_detector.closed)
        for index in (0, 130, 350):
            geometry = self.result.face.observations[index].landmarks
            self.assertTrue(all(region.observations[index].landmarks is geometry for region in self.result.face_rois.values()))

    def test_schema_and_timestamp_identity(self) -> None:
        result = self.result
        self.assertEqual(tuple(result.face_rois), DEFAULT_FACE_ROIS)
        self.assertIs(result.face, result.face_rois["combined"])
        for region in (*result.face_rois.values(), result.hand):
            self.assertEqual(region.rgb.values.shape, (600, 3))
            self.assertIs(region.rgb.timestamps, result.shared.original_timestamps)
            self.assertIs(region.rppg.timestamps, result.shared.timestamps)
            np.testing.assert_array_equal(region.rgb.timestamps, self.t)
            self.assertEqual([obs.frame_index for obs in region.observations], list(range(600)))
            np.testing.assert_array_equal([obs.timestamp for obs in region.observations], self.t)

    def test_roi_specific_missingness_diagnostics_and_segments(self) -> None:
        left = self.result.face_rois["left_cheek"]
        self.assertTrue(np.isnan(left.rgb.values[150:210]).all())
        self.assertFalse(left.rgb.validity_mask[150:210].any())
        self.assertEqual(left.observations[160].reason, "insufficient_skin_pixels")
        self.assertTrue(left.observations[160].valid)  # Detection still succeeded.
        self.assertEqual(left.rgb.quality["skin_pixel_count"][160], 0)
        self.assertFalse(left.rgb.quality["extraction_valid"][160])
        self.assertLess(left.quality["valid_frame_fraction"], 1)
        self.assertIn("insufficient_valid_frames", left.quality["reasons"])
        self.assertIn("illumination_std", left.quality)
        self.assertEqual(len(left.rppg.preprocessing["segments"]), 2)
        self.assertTrue(np.isnan(left.rppg.values[120:240]).all())
        for name in ("forehead", "right_cheek", "combined"):
            region = self.result.face_rois[name]
            self.assertTrue(region.rgb.validity_mask.all())
            self.assertIsNone(region.observations[160].reason)
            self.assertTrue(np.isfinite(region.rppg.values[150:210]).all())

    def test_short_gap_marked_and_endpoints_not_extrapolated(self) -> None:
        left = self.result.face_rois["left_cheek"].rgb
        uniform, imputed = resample_trace(left, self.result.shared.timestamps, 0.15)
        self.assertTrue(np.isnan(uniform[:5]).all())
        self.assertTrue(np.isnan(uniform[595:]).all())
        self.assertTrue(np.isnan(uniform[150:210]).all())
        self.assertEqual(np.flatnonzero(imputed).tolist(), [300, 301])
        self.assertTrue(np.isfinite(uniform[300:302]).all())
        self.assertFalse(left.validity_mask[300:302].any())

    def test_priors_reuse_classical_formulas_and_gap_rules(self) -> None:
        grid = self.result.shared.timestamps
        for name, region in self.result.face_rois.items():
            priors = build_classical_priors(region.rgb, grid, self.config, ("GREEN", "CHROM"))
            uniform, _ = resample_trace(region.rgb, grid, self.config.max_gap_sec)
            for method, signal in priors.items():
                expected = extract_signal(uniform, grid, region.rgb.roi_name, replace(self.config, method=method))
                self.assertIs(signal.timestamps, grid)
                np.testing.assert_array_equal(signal.values, expected.values)
                self.assertEqual(signal.algorithm, method)
                if name == "left_cheek":
                    self.assertTrue(np.isnan(signal.values[120:240]).all())
                    self.assertEqual(len(signal.preprocessing["segments"]), 2)
        defaults = build_classical_priors(self.result.face.rgb, grid, self.config)
        self.assertEqual(list(defaults), DEFAULT_PRIORS)
        self.assertTrue(all(signal.values.shape == grid.shape for signal in defaults.values()))
        with self.assertRaises(ValueError):
            build_classical_priors(self.result.face.rgb, grid, self.config, ("bogus",))

    def test_legacy_selected_roi_and_combined_only_mode_match(self) -> None:
        single_config = replace(self.config, face_rois=("combined",))
        single = process_recording(self.video, single_config, timestamps=self.t,
                                   detectors={"face": CountingDetector(lose_cheek=True), "hand": CountingDetector("hand")})
        self.assertEqual(tuple(single.face_rois), ("combined",))
        self.assertEqual(single.face.rgb.roi_name, "face")
        self.assertEqual(single.face.rppg.source_region, "face")
        np.testing.assert_array_equal(single.face.rgb.values, self.result.face.rgb.values)
        np.testing.assert_array_equal(single.face.rppg.values, self.result.face.rppg.values)
        np.testing.assert_array_equal(single.hand.rppg.values, self.result.hand.rppg.values)
        self.assertEqual(json.dumps(single.bp_features), json.dumps(self.result.bp_features))
        self.assertEqual(single.delay, self.result.delay)
        self.assertGreater(single.delay["accepted_window_count"], 0)
        self.assertAlmostEqual(single.delay["face_hand_delay_median_sec"], 0.12, delta=1 / 30)
        selected = process_recording(self.video, replace(self.config, face_roi="left_cheek"),
                                     timestamps=self.t, max_frames=30,
                                     detectors={"face": CountingDetector(lose_cheek=True), "hand": CountingDetector("hand")})
        self.assertIs(selected.face, selected.face_rois["left_cheek"])
        self.assertFalse(selected.face.rgb.validity_mask[:5].any())
        self.assertTrue(selected.face_rois["forehead"].rgb.validity_mask.all())

    def test_artifact_roundtrip_preserves_all_rois(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = reserve_run(Path(tmp) / "run")
            save_result(self.result, output, {"fixture": "synthetic"})
            with np.load(output / "signals.npz", allow_pickle=False) as arrays:
                self.assertEqual(arrays["face_roi_names"].tolist(), list(DEFAULT_FACE_ROIS))
                for name, region in self.result.face_rois.items():
                    prefix = "face_roi_" + name + "_"
                    np.testing.assert_array_equal(arrays[prefix + "rgb"], region.rgb.values)
                    np.testing.assert_array_equal(arrays[prefix + "valid"], region.rgb.validity_mask)
                    np.testing.assert_array_equal(arrays[prefix + "rppg"], region.rppg.values)
                    np.testing.assert_array_equal(arrays[prefix + "brightness"], region.rgb.quality["brightness"])
                self.assertEqual(arrays["face_roi_left_cheek_reason"][160], "insufficient_skin_pixels")
                self.assertEqual(np.flatnonzero(arrays["face_roi_left_cheek_interpolated"]).tolist(), [300, 301])
                np.testing.assert_array_equal(arrays["face_rgb"], self.result.face.rgb.values)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(set(summary["face_rois"]), set(DEFAULT_FACE_ROIS))
            self.assertEqual(summary["face_rois"]["left_cheek"]["quality"]["valid_frame_fraction"],
                             self.result.face_rois["left_cheek"].quality["valid_frame_fraction"])

    def test_owned_detectors_constructed_and_closed_once(self) -> None:
        face, hand = CountingDetector(), CountingDetector("hand")
        with patch("rppg_lab.pipeline.FaceRegionDetector", return_value=face) as face_factory, \
                patch("rppg_lab.pipeline.HandRegionDetector", return_value=hand) as hand_factory:
            process_recording(self.video, self.config, timestamps=self.t, max_frames=10)
        face_factory.assert_called_once()
        hand_factory.assert_called_once()
        self.assertEqual(len(face.calls), 10)
        self.assertTrue(face.closed and hand.closed)

    def test_face_only_missing_face_and_hand_only_schemas(self) -> None:
        missing = process_recording(self.video, replace(self.config, regions=("face",)), timestamps=self.t,
                                    max_frames=10, detectors={"face": CountingDetector(missing=True)})
        self.assertEqual(len(missing.face_rois), 4)
        for region in missing.face_rois.values():
            self.assertTrue(np.isnan(region.rgb.values).all())
            self.assertTrue(np.isnan(region.rppg.values).all())
            self.assertTrue(all(obs.reason == "face_not_detected" for obs in region.observations))
        hand_only = process_recording(self.video, replace(self.config, regions=("hand",)), timestamps=self.t,
                                      max_frames=10, detectors={"hand": CountingDetector("hand")})
        self.assertEqual(hand_only.face_rois, {})
        self.assertTrue(np.isnan(hand_only.face.rgb.values).all())
        self.assertTrue(hand_only.hand.rgb.validity_mask.all())


class MultiFaceConfigTests(unittest.TestCase):
    def test_legacy_json_and_explicit_multi_roi_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text('{"face_roi":"combined"}', encoding="utf-8")
            legacy = PipelineConfig.load(path)
            self.assertEqual(legacy.requested_face_rois, DEFAULT_FACE_ROIS)
            path.write_text(json.dumps({"face_roi": "full_skin", "face_rois": list(DEFAULT_FACE_ROIS)}), encoding="utf-8")
            multi = PipelineConfig.load(path)
            self.assertEqual(multi.requested_face_rois, (*DEFAULT_FACE_ROIS, "full_skin"))
            path.write_text(json.dumps(multi.to_dict()), encoding="utf-8")
            self.assertEqual(multi, PipelineConfig.load(path))
        for rois in ((), ("combined", "combined"), ("no_such_roi",), "combined"):
            with self.subTest(rois=rois), self.assertRaises(ValueError):
                replace(PipelineConfig(), face_rois=rois)


if __name__ == "__main__":
    unittest.main()
