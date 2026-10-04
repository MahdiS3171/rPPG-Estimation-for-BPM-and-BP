from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from rppg_lab.models import (
    WaveformNet,
    PriorResidualWaveformNet,
    MultiROIWaveformNet,
    BPFeatureMLP,
    HRQualityNet,
    HRQualityNetV2,
    VideoHRNet,
    VideoPriorHRNet,
)
from rppg_lab.signals import estimate_hr_welch, align_by_xcorr, phase_delay_at_hr, pulse_wave_features
from rppg_lab.datasets import SessionBPDataset
from scripts.train_bp_baseline import (
    split_subjects,
    fit_feature_preprocessor,
    transform_features,
    fit_target_preprocessor,
    transform_targets,
    inverse_targets,
)


class SignalTests(unittest.TestCase):
    def test_hr_and_positive_delay_convention(self) -> None:
        fs = 100.0
        t = np.arange(1000) / fs
        freq = 1.2
        delay = 0.12
        x = np.sin(2 * np.pi * freq * t)
        y = np.sin(2 * np.pi * freq * (t - delay))
        hr = estimate_hr_welch(x, fs)
        self.assertAlmostEqual(hr.hr_bpm, 72.0, delta=0.2)
        xlag, score = align_by_xcorr(x, y, fs, max_lag_sec=0.4)
        plag, coh = phase_delay_at_hr(x, y, fs, hr_hz=freq)
        self.assertAlmostEqual(xlag, delay, delta=1 / fs)
        self.assertAlmostEqual(plag, delay, delta=0.01)
        self.assertGreater(score, 0.9)
        self.assertGreater(coh, 0.9)

    def test_pulse_features_include_reported_morphology(self) -> None:
        fs = 100.0
        t = np.arange(1000) / fs
        x = np.sin(2 * np.pi * 1.2 * t)
        feat = pulse_wave_features(x, fs)
        for key in [
            "fall_time_mean_sec",
            "area_mean",
            "duty_mean",
            "upstroke_slope_mean",
        ]:
            self.assertIn(key, feat)
            self.assertTrue(np.isfinite(feat[key]))


class ModelSmokeTests(unittest.TestCase):
    def test_1d_models(self) -> None:
        x = torch.randn(2, 9, 300)
        self.assertEqual(WaveformNet(9)(x).shape, (2, 300))
        self.assertEqual(PriorResidualWaveformNet(9, ["A", "B", "C", "D", "E", "F"])(x).shape, (2, 300))
        self.assertEqual(MultiROIWaveformNet(9)(torch.randn(2, 3, 9, 300)).shape, (2, 300))
        self.assertEqual(BPFeatureMLP(12)(torch.randn(2, 12)).shape, (2, 2))
        self.assertEqual(HRQualityNet(9)(x, return_dict=True)["hr_bpm"].shape, (2,))
        cands = torch.tensor([[70., 72., 74., 76., 78., 80., 82.], [60., 62., 64., 66., 68., 70., 72.]])
        valid = torch.ones_like(cands, dtype=torch.bool)
        self.assertEqual(HRQualityNetV2(9, 7)(x, cands, valid, return_dict=True)["hr_bpm"].shape, (2,))

    def test_video_models(self) -> None:
        video = torch.randn(2, 8, 3, 32, 32)
        self.assertEqual(VideoHRNet()(video, return_dict=True)["hr_bpm"].shape, (2,))
        model = VideoPriorHRNet(trace_channels=9, num_candidates=7)
        trace = torch.randn(2, 9, 300)
        cands = torch.full((2, 7), 72.0)
        valid = torch.ones_like(cands, dtype=torch.bool)
        self.assertEqual(model(video, trace, cands, valid)["hr_bpm"].shape, (2,))


class BPDatasetTests(unittest.TestCase):
    def test_subject_metadata_and_train_only_preprocessing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for sid, value in [("S001", 1.0), ("S002", 2.0), ("S003", 3.0)]:
                sess = root / sid / "rest_01"
                sess.mkdir(parents=True)
                (sess / "features.json").write_text(json.dumps({"a": value, "b": None}), encoding="utf-8")
                (sess / "labels.json").write_text(json.dumps({"subject_id": sid, "cuff": {"sbp": 120 + value, "dbp": 80 + value}}), encoding="utf-8")
            ds = SessionBPDataset(root)
            self.assertEqual(set(ds.subject_ids), {"S001", "S002", "S003"})
            tr, va, te = split_subjects(ds.subject_ids, 1 / 3, 1 / 3, 42)
            self.assertFalse(set(tr) & set(va))
            self.assertFalse(set(tr) & set(te))
            self.assertFalse(set(va) & set(te))
            x = np.stack([ds[i]["x_raw"].numpy() for i in range(len(ds))])
            idx = [i for i, s in enumerate(ds.subject_ids) if s in tr]
            med, mean, std = fit_feature_preprocessor(x[idx])
            z = transform_features(x, med, mean, std)
            self.assertTrue(np.isfinite(z).all())
            y = np.stack([ds[i]["y"].numpy() for i in range(len(ds))])
            y_mean, y_std = fit_target_preprocessor(y[idx])
            y_scaled = transform_targets(y, y_mean, y_std)
            y_back = inverse_targets(y_scaled, y_mean, y_std)
            self.assertTrue(np.allclose(y, y_back, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
