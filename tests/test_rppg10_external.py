"""Synthetic-only external HR protocol tests; no real dataset or training."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import matplotlib
import numpy as np
import pandas as pd
import torch

from rppg_lab.artifacts import file_sha256
from rppg_lab.classical import DEFAULT_PRIORS
from rppg_lab.config import PipelineConfig
from rppg_lab.extraction_cache import DEFAULT_WAVEFORM_ROIS, FaceROIExtraction
from rppg_lab.models import MultiROIPriorResidualWaveformNet
from rppg_lab.types import RGBTrace, SharedTimebase
from rppg_lab.waveform_inference import FrozenWaveformExtractor
from rppg_lab.window_priors import build_window_priors
from rppg_lab import rppg10 as data
from rppg_lab import rppg10_external as external
from rppg_lab import rppg10_reporting as report


def ecg_signal(hr=72., duration=40, fs=1000, irregular=False, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * fs)) / fs
    x = .002 * rng.normal(size=len(t)) + .015 * np.sin(2*np.pi*.3*t)
    beats = np.arange(.7, duration, 60/hr)
    if irregular:
        beats = np.cumsum(np.resize([.4, 1.3], int(duration*2)))
        beats = beats[beats < duration]
    for beat in beats:
        x += np.exp(-.5*((t-beat)/.018)**2) - .2*np.exp(-.5*((t-beat-.04)/.018)**2)
    return x


def crop_video(path, count=600, fps=30., bgr=(30, 90, 160), pulse=True):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (64, 64))
    if not writer.isOpened():
        raise RuntimeError("Synthetic MJPG codec unavailable")
    for i in range(count):
        color = np.array(bgr, float)
        if pulse:
            color += np.sin(2*np.pi*1.2*i/fps) * np.array([2., 5., 3.])
        writer.write(np.broadcast_to(np.clip(color, 0, 255).astype(np.uint8), (64, 64, 3)).copy())
    writer.release()


def subject_fixture(root, sid="Subject_1", count=600):
    folder = Path(root) / sid
    folder.mkdir()
    for i, suffix in enumerate(["Forehead", "Cheek1", "Cheek2"]):
        crop_video(folder / f"{sid}_{suffix}_.avi", count=count, bgr=(30+i*10, 90+i*10, 160+i*10))
    np.save(folder / f"{sid}_ECG.npy", ecg_signal(duration=count/30))
    return folder


def synthetic_extractor():
    """Existing frozen inference API with an untrained in-memory test model.

    This fixture bypasses only file provenance and is never publicly callable
    as an official extractor. It does not create or select a checkpoint.
    """
    torch.manual_seed(42)
    f = FrozenWaveformExtractor.__new__(FrozenWaveformExtractor)
    f.roi_names, f.prior_names = DEFAULT_WAVEFORM_ROIS, tuple(DEFAULT_PRIORS)
    f.channel_names = ("RGB_R", "RGB_G", "RGB_B", *DEFAULT_PRIORS)
    f.extraction_config = PipelineConfig(regions=("face",), face_roi="forehead", face_rois=DEFAULT_WAVEFORM_ROIS)
    f.fs_target, f.window_length_samples, f.min_valid_fraction = 30., 300, .9
    f.device, f.metadata = torch.device("cpu"), {"test_fixture": True}
    f._model = MultiROIPriorResidualWaveformNet(in_channels=9, roi_names=f.roi_names,
        prior_names=f.prior_names, base_channels=4, num_blocks=0, dropout=0.).eval()
    return f


def trace_fixture(length=600):
    t = np.arange(length)/30
    traces = {}
    for i, name in enumerate(DEFAULT_WAVEFORM_ROIS):
        x = np.array([150+i*10, 100+i*5, 60+i*5]) + np.sin(2*np.pi*(1.2+i*.03)*t[:,None])*np.array([1.,3.,2.])
        traces[name] = RGBTrace(t, x, name, np.ones(length, bool))
    return FaceROIExtraction(SharedTimebase(t, t), traces)


def metric_fixture():
    rows = []
    for sid, errors in [("Subject_1", [1., 1., 1.]), ("Subject_2", [9.])]:
        for i, error in enumerate(errors):
            rows.append(dict(subject_id=sid, method=report.MODEL, start_sec=i*5, minute=0,
                             pred_hr_bpm=70+error, reference_hr_bpm=70., reference_valid=True,
                             extraction_status="ok", snr_db=2.))
    return rows


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_exact_discovery_subject4_missing_and_foreign_files(self):
        subject_fixture(self.root)
        excluded = self.root / "Subject_4"
        excluded.mkdir()
        (excluded / "Subject_4_Forehead_.avi").touch()
        folder = self.root / "Subject_1"
        (folder / "Subject_2_Forehead.avi").write_bytes(b"unrelated")
        inv = data.scan_dataset(self.root)
        self.assertEqual(len(inv["expected_usable_ids"]), 26)
        self.assertNotIn("Subject_4", inv["expected_usable_ids"])
        self.assertEqual(inv["discovered_usable_ids"], ["Subject_1"])
        self.assertEqual(len(inv["missing_ids"]), 25)
        self.assertTrue(inv["excluded"][0]["present"])
        self.assertTrue(any("unrelated_file" in q for q in inv["subjects"][0]["qc"]))
        (folder / "Subject_1_Forehead_.avi").rename(folder / "Subject_1_Forehead.avi")
        self.assertTrue(data.scan_dataset(self.root)["subjects"][0]["structure_usable"])

    def test_duplicate_files_rejected_not_silently_selected(self):
        subject_fixture(self.root)
        p = self.root/"Subject_1"/"Subject_1_Forehead.avi"
        p.write_bytes(b"duplicate")
        row = data.scan_dataset(self.root)["subjects"][0]
        self.assertFalse(row["structure_usable"])
        self.assertNotIn("forehead", row["files"])

    def test_bgr_to_rgb_and_nominal_clock_cache(self):
        path = self.root / "color.avi"
        crop_video(path, count=30, bgr=(20, 80, 180), pulse=False)
        first = data.decode_crop(path, "forehead", self.root/"cache")
        np.testing.assert_allclose(first.trace.values[0], [180, 80, 20], atol=3)
        self.assertEqual(first.qc["timestamp_source"], "nominal_fps")
        with patch("cv2.VideoCapture", side_effect=AssertionError("decoded twice")):
            second = data.decode_crop(path, "forehead", self.root/"cache")
        np.testing.assert_equal(first.trace.values, second.trace.values)

    def test_three_crops_mapping_order_and_common_tail(self):
        subject_fixture(self.root)
        crop_video(self.root/"Subject_1"/"Subject_1_Cheek2_.avi", count=450, bgr=(50, 110, 180))
        record = data.scan_dataset(self.root)["subjects"][0]
        self.assertIn("Cheek2", record["files"]["left_cheek"]["path"])
        self.assertIn("Cheek1", record["files"]["right_cheek"]["path"])
        with patch("rppg_lab.extraction_cache.process_recording", side_effect=AssertionError("landmark pipeline called")):
            extraction, qc = data.load_subject_crops(self.root, record, synthetic_extractor().extraction_config)
        self.assertEqual(tuple(extraction.traces), DEFAULT_WAVEFORM_ROIS)
        self.assertEqual(len(extraction.shared.timestamps), 450)
        self.assertFalse(qc["frame_counts_match"])
        self.assertAlmostEqual(qc["discarded_tail_sec"]["forehead"], 5.)
        self.assertGreater(extraction.traces["left_cheek"].r.mean(), extraction.traces["right_cheek"].r.mean())
        self.assertLess(extraction.shared.timestamps[-1], 15)

    def test_divergent_fps_short_zero_and_malformed(self):
        subject_fixture(self.root, count=300)
        crop_video(self.root/"Subject_1"/"Subject_1_Cheek1_.avi", count=300, fps=29.)
        record = data.scan_dataset(self.root)["subjects"][0]
        with self.assertRaisesRegex(ValueError, "Divergent"):
            data.load_subject_crops(self.root, record, synthetic_extractor().extraction_config)
        (self.root/"Subject_1"/"Subject_1_Cheek1_.avi").write_bytes(b"")
        self.assertFalse(data.scan_dataset(self.root)["subjects"][0]["structure_usable"])
        bad = self.root/"bad.avi"
        bad.write_bytes(b"not video")
        with self.assertRaisesRegex(ValueError, "Malformed"):
            data.decode_crop(bad, "forehead")

    def test_reported_frame_count_shortfall_recorded_without_padding(self):
        class FakeCapture:
            def __init__(self): self.index = 0
            def isOpened(self): return True
            def get(self, prop):
                return 30 if prop == cv2.CAP_PROP_FPS else 600 if prop == cv2.CAP_PROP_FRAME_COUNT else self.index/30*1000
            def read(self):
                self.index += 1
                return (True, np.full((64,64,3), 100, np.uint8)) if self.index <= 300 else (False, None)
            def release(self): pass
        path = self.root/"short.avi"
        path.write_bytes(b"fixture")
        with patch("cv2.VideoCapture", return_value=FakeCapture()):
            crop = data.decode_crop(path, "forehead")
        self.assertEqual(len(crop.trace.timestamps), 300)
        self.assertEqual(crop.qc["decoded_shortfall"], 300)
        self.assertIn("short_recording", crop.qc["warnings"])

    def test_metadata_exact_column_join_skips_subject4(self):
        path = self.root/"metadata.xlsx"
        pd.DataFrame([["Characteristics", "Subject 5", "Subject 1"],
                      ["Skin Tone (FitzPatrick)", "IV", "II"],
                      ["Date", "private", "private"]]).to_excel(path, index=False, header=False)
        joined, qc = data.load_metadata(path)
        self.assertEqual(joined["Subject_5"], {"skin_type": "IV"})
        self.assertEqual(joined["Subject_1"], {"skin_type": "II"})
        self.assertNotIn("Subject_4", joined)
        self.assertNotIn("private", json.dumps(joined))


class ECGTests(unittest.TestCase):
    def test_empty_or_malformed_ecg_is_actionable_qc_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"Subject_1_ECG.npy"
            for payload in [b"", b"not a numpy array"]:
                path.write_bytes(payload)
                with self.assertRaisesRegex(ValueError, "Unreadable ECG array"):
                    data.load_ecg(path)

    def test_sampling_rate_shape_known_heart_rate(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"ecg.npy"
            np.save(p, ecg_signal())
            for fs in [0, np.nan, 30, 99999]:
                with self.assertRaises(ValueError): data.load_ecg(p, fs)
            np.save(p, np.zeros((2, 20000)))
            with self.assertRaises(ValueError): data.load_ecg(p)
        for fs in [250, 1000]:
            for hr in [48., 72., 120., 180.]:
                peaks = data.detect_r_peaks(ecg_signal(hr, fs=fs), fs)
                label = data.ecg_reference(peaks, 10, 20)
                self.assertTrue(label["valid"], (fs, hr, label))
                # One-sample RR quantization matters at high HR / lower ECG fs.
                self.assertAlmostEqual(label["hr_bpm"], hr, delta=max(.6, hr**2/(60*fs)+.05))

    def test_low_quality_irregular_nonfinite_and_manual_boundaries(self):
        for x in [np.zeros(40000), np.random.default_rng(2).normal(size=40000), ecg_signal(irregular=True)]:
            ref = data.ecg_reference(data.detect_r_peaks(x), 10, 20)
            self.assertFalse(ref["valid"], ref)
        x = ecg_signal()
        x[10000:11000] = np.nan
        self.assertFalse(data.ecg_reference(data.detect_r_peaks(x), 10, 20)["valid"])
        peaks = data.detect_r_peaks(ecg_signal(duration=20))
        self.assertFalse(data.ecg_reference(peaks, 0, 10)["valid"])
        self.assertFalse(data.ecg_reference(peaks, 15, 25)["valid"])
        self.assertEqual(data.ecg_reference(peaks, 15, 25)["reason"], "missing_ecg_support")

    def test_timing_sensitivity_never_selects_better_optical_label(self):
        p = data.detect_r_peaks(ecg_signal())
        fake = [dict(valid=True, hr_bpm=70., reason="valid"), dict(valid=True, hr_bpm=75., reason="valid")]
        with patch.object(data, "_ecg_window", side_effect=[fake[0], fake[0], fake[0], fake[1]]):
            ref = data.ecg_reference(p, 10, 20)
        self.assertFalse(ref["valid"])
        self.assertTrue(np.isnan(ref["hr_bpm"]))
        self.assertEqual(ref["reason"], "timing_uncertain_or_boundary_support")


class FrozenAdapterTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.f = synthetic_extractor()
        self.extraction = trace_fixture()

    def test_exact_window_prior_recomputation_standardization_and_mask(self):
        with patch("rppg_lab.waveform_inference.build_window_priors", wraps=build_window_priors) as priors:
            batch = self.f.prepare_window(self.extraction, 5.)
        self.assertEqual(tuple(batch["x"].shape), (1, 3, 9, 300))
        self.assertEqual(priors.call_count, 3)
        for i, call in enumerate(priors.call_args_list):
            np.testing.assert_equal(call.args[0], self.extraction.traces[DEFAULT_WAVEFORM_ROIS[i]].values[150:450])
        np.testing.assert_allclose(batch["x"][0,:,:3].mean(axis=2), 0, atol=1e-6)
        changed = deepcopy(self.extraction)
        changed.traces["forehead"].validity_mask[150:450] = False
        masked = self.f.prepare_window(changed, 5.)
        self.assertFalse(masked["roi_valid"][0,0])
        self.assertTrue(torch.equal(masked["x"][0,0], torch.zeros(9,300)))
        with self.assertRaises(ValueError): self.f.prepare_window(self.extraction, 15.)

    def test_ecg_changes_cannot_change_tensor_prediction_attention_or_quality(self):
        p1 = data.detect_r_peaks(ecg_signal(72))
        p2 = data.detect_r_peaks(ecg_signal(120))
        before = self.f.prepare_window(self.extraction, 5.)
        one, w1 = external.evaluate_subject(self.f, self.extraction, p1, "Subject_1", [5.])
        two, w2 = external.evaluate_subject(self.f, self.extraction, p2, "Subject_1", [5.])
        after = self.f.prepare_window(self.extraction, 5.)
        for key in ["x", "roi_quality", "roi_valid", "prior_valid"]:
            self.assertTrue(torch.equal(before[key], after[key]))
        np.testing.assert_equal(w1[5.]["model"], w2[5.]["model"])
        m1 = next(r for r in one if r["method"] == report.MODEL)
        m2 = next(r for r in two if r["method"] == report.MODEL)
        self.assertNotEqual(m1["reference_hr_bpm"], m2["reference_hr_bpm"])
        for key in [k for k in m1 if "weight_" in k] + ["valid_fraction", "pred_hr_bpm"]:
            self.assertEqual(m1[key], m2[key])

    def test_baselines_share_windows_and_exact_priors(self):
        rows, windows = external.evaluate_subject(self.f, self.extraction, data.detect_r_peaks(ecg_signal()), "Subject_1", [5., 10.])
        self.assertEqual(len(rows), 50)
        for name in external.METHODS:
            self.assertEqual([r["start_sec"] for r in rows if r["method"] == name], [5., 10.])
        b = self.f.prepare_window(self.extraction, 5.)
        np.testing.assert_equal(windows[5.]["chrom"], b["x"][0,:,5,:].numpy().mean(axis=0))

    def test_hash_manifest_contract_fail_before_constructor_or_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"bad.pt"
            p.write_bytes(b"synthetic")
            with patch.object(external, "FrozenWaveformExtractor", side_effect=AssertionError("reconstructed")):
                with self.assertRaisesRegex(ValueError, "SHA256"): external.load_official_extractor(p)
            mf = p.with_name("waveform_v1_freeze_manifest.json")
            mf.write_text(json.dumps(dict(freeze_decision="wrong")))
            with patch.object(external, "OFFICIAL_SHA256", file_sha256(p)), patch.object(external, "FrozenWaveformExtractor", side_effect=AssertionError("reconstructed")):
                with self.assertRaisesRegex(ValueError, "manifest"): external.load_official_extractor(p)

    def test_existing_api_status_and_order_guard(self):
        # Tiny intentionally incomplete artifacts exercise guards before state reconstruction.
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"model.pt"
            mf = p.with_name("waveform_v1_freeze_manifest.json")
            torch.save(dict(waveform_v1_status="candidate"), p)
            mf.write_text(json.dumps(dict(status="frozen_v1", model_sha256=file_sha256(p))))
            with self.assertRaisesRegex(ValueError, "frozen_v1"): FrozenWaveformExtractor(p)
            torch.save(dict(waveform_v1_status="frozen_v1", freeze_decision="FREEZE_WAVEFORM_V1",
                            roi_names=["right_cheek", "forehead", "left_cheek"], prior_names=list(DEFAULT_PRIORS),
                            channel_ordering=list(self.f.channel_names)), p)
            mf.write_text(json.dumps(dict(status="frozen_v1", model_sha256=file_sha256(p))))
            with patch("rppg_lab.waveform_inference.checkpoint_dataset_config", return_value={}):
                with self.assertRaisesRegex(ValueError, "ordering"): FrozenWaveformExtractor(p)


class ReportingProtocolTests(unittest.TestCase):
    def test_subject_balance_coverage_valid_denominator_and_bootstrap(self):
        rows = metric_fixture()
        subjects, result = report.aggregate_metrics(rows, ["Subject_1", "Subject_2"], draws=101)
        model = result["methods"][report.MODEL]
        self.assertEqual(model["subject_balanced"]["mae_bpm"], 5.)
        self.assertEqual(model["window_weighted"]["mae_bpm"], 3.)
        again = report.aggregate_metrics(list(reversed(rows)), ["Subject_2", "Subject_1"], draws=101)[1]
        self.assertEqual(json.dumps(result, sort_keys=True), json.dumps(again, sort_keys=True))
        rows += [dict(rows[0], pred_hr_bpm=np.nan, extraction_status="missing_video"),
                 dict(rows[0], reference_valid=False, reference_hr_bpm=np.nan)]
        m = report.metric_values(pd.DataFrame(rows))
        self.assertEqual(m["valid_pairs"], 4)
        self.assertEqual(m["within_5_bpm"], .75)
        self.assertEqual(m["finite_prediction_coverage"], 5/6)
        self.assertEqual(m["extraction_failures"], 1)
        self.assertEqual(m["missing_labels"], 1)

    def test_constant_nonfinite_spectral_rejection_and_matched_subset(self):
        for x in [np.zeros(300), np.full(300,np.nan)]:
            self.assertTrue(np.isnan(report.spectral_diagnostics(x)["pred_hr_bpm"]))
        rows = metric_fixture()
        baseline = [dict(r, method=report.CHROM, pred_hr_bpm=70.) for r in rows]
        baseline[0]["pred_hr_bpm"] = np.nan
        match = report.matched_comparisons(pd.DataFrame(rows+baseline))[report.CHROM]
        self.assertEqual(match["matched_windows"], 3)
        self.assertAlmostEqual(match["model_mae_bpm"], 11/3)

    def test_nonoverwrite_receipt_global_consumption_and_smoke_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = external.build_parser().parse_args(["--dataset-root",tmp,"--out-dir",str(root/"out"),"--max-subjects","1"])
            smoke, mapping = external.validate_options(args)
            cfg = external.configuration(args, smoke, mapping)
            self.assertEqual(cfg["result_type"], "SMOKE")
            inv = dict(data_identity_sha256="synthetic")
            (root/"out").mkdir()
            receipt = external.reserve_receipt(root/"out", cfg, inv, "hash", "manifest", root/"ledger.json")
            self.assertFalse(receipt["external_dataset_consumed"])
            self.assertFalse((root/"ledger.json").exists())
            with self.assertRaises(FileExistsError): external.reserve_receipt(root/"out",cfg,inv,"hash","manifest")
            cfg["result_type"] = "REAL EXTERNAL"
            (root/"real").mkdir()
            external.reserve_receipt(root/"real",cfg,inv,"hash","manifest",root/"ledger.json")
            (root/"other").mkdir()
            with self.assertRaises(FileExistsError): external.reserve_receipt(root/"other",cfg,inv,"hash","manifest",root/"ledger.json")

    def test_options_cannot_change_frozen_window_or_official_fs(self):
        for option in [["--window-sec","8"], ["--stride-sec","0"], ["--ecg-fs","250"], ["--bootstrap-draws","1"]]:
            args = external.build_parser().parse_args(["--dataset-root","unused","--out-dir","unused",*option])
            with self.assertRaises(ValueError): external.validate_options(args)

    def test_missing_ecg_invalid_signal_and_extraction_are_distinct(self):
        f = synthetic_extractor()
        extraction = trace_fixture()
        for trace in extraction.traces.values():
            trace.validity_mask[:] = False
        signal_rows, _ = external.evaluate_subject(f, extraction, None, "Subject_1", [5.])
        failed_rows, _ = external.evaluate_subject(f, None, None, "Subject_1", [5.])
        signal = report.metric_values(pd.DataFrame([signal_rows[0]]))
        failed = report.metric_values(pd.DataFrame([failed_rows[0]]))
        self.assertEqual(signal["invalid_signal_windows"], 1)
        self.assertEqual(signal["extraction_failures"], 0)
        self.assertEqual(signal["missing_labels"], 1)
        self.assertEqual(failed["extraction_failures"], 1)

    def test_validate_only_never_loads_model_or_consumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root/"dataset"
            dataset.mkdir()
            subject_fixture(dataset, count=300)
            args = external.build_parser().parse_args(["--dataset-root",str(dataset),"--out-dir",str(root/"out"),
                "--validate-only","--max-subjects","1"])
            with patch.object(external, "load_official_extractor", side_effect=AssertionError("model touched")):
                result = external.run(args)
            self.assertEqual(result["result_type"], "VALIDATION_ONLY")
            self.assertFalse((root/"out"/"external_evaluation_receipt.json").exists())

    def test_no_training_shape_reference_or_physiological_timing_code(self):
        for module in [data, external, report]:
            source = Path(module.__file__).read_text(encoding="utf-8")
            for forbidden in [".backward(", ".fit(", "torch.save(", "waveform_window_metrics(", "pearsonr(",
                              "aligned_overlap(", "pulse_wave_features(", "correlate(", "optimizer.step("]:
                self.assertNotIn(forbidden, source)

    def test_synthetic_end_to_end_headless_deterministic_and_never_official(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root/"dataset"
            dataset.mkdir()
            subject_fixture(dataset, count=600)
            cp = root/"synthetic_identity_only.pt"
            cp.write_bytes(b"test fixture; no checkpoint model")
            cp.with_name("waveform_v1_freeze_manifest.json").write_text("{}")
            hashes = []
            for run in ["one", "two"]:
                args = external.build_parser().parse_args(["--dataset-root",str(dataset),"--out-dir",str(root/run),
                    "--checkpoint",str(cp),"--smoke","--bootstrap-draws","31","--cache-dir",str(root/"cache")])
                with patch.object(external,"load_official_extractor",return_value=(synthetic_extractor(), {})):
                    summary = external.run(args)
                self.assertEqual(summary["result_type"], "SMOKE")
                self.assertFalse(summary["successful_official_evaluation"])
                self.assertEqual(summary["metrics"]["methods"][report.MODEL]["window_weighted"]["windows"], 3)
                self.assertTrue((root/run/"completion_verification.json").exists())
                hashes.append({p.name: file_sha256(p) for p in (root/run/"figures").glob("*.png")})
            self.assertEqual(hashes[0], hashes[1])
            self.assertEqual(matplotlib.get_backend().lower(), "agg")


if __name__ == "__main__":
    unittest.main()
