from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path
import csv
import json
import tempfile
import unittest
from unittest.mock import patch
import cv2
import numpy as np

from rppg_lab.types import RGBTrace, PhysiologicalSignal, RegionObservation, VideoMetadata
from rppg_lab.config import PipelineConfig
from rppg_lab.processing import shared_grid, resample_trace, zero_phase_bandpass, extract_signal
from rppg_lab.pipeline import process_recording
from rppg_lab.bp import extract_features, estimate_delay, FEATURE_KEYS
from rppg_lab.bp_models import BPModel, train_baselines, bp_metrics
from rppg_lab.splits import split_subjects, assert_subject_disjoint, validate_sample_partitions
from rppg_lab.study import BPReference, ReferenceAssociation, Recording
from rppg_lab.quality import signal_quality
from rppg_lab.metrics import regression_metrics
from rppg_lab.artifacts import reserve_run, save_result, provenance
from rppg_lab.detection import BaseRegionDetector, HandRegionDetector
from rppg_lab.classical import METHOD_FUNCS
from rppg_lab.signals import bandpass
from rppg_lab.video import VideoReader
from rppg_lab.datasets import UBFCRPPGDataset, UBFCSubject, subject_split, SessionBPDataset
from rppg_lab.roi import ROIExtractionResult
from scripts.run_bp_pilot import associate_windows


def waves(fs=60.0, seconds=20.0, delay=0.12):
    t = np.arange(int(fs*seconds))/fs
    def pulse(t):
        return np.sin(2*np.pi*1.2*t)+0.2*np.sin(2*np.pi*2.4*t)
    return t,pulse(t),pulse(t-delay)


class TimingIntegrityTests(unittest.TestCase):
    def test_legacy_filter_regression(self):
        from scipy.signal import butter, filtfilt
        rng = np.random.default_rng(42)
        for n in (10,50,600):
            x = rng.normal(size=n)
            centered = x-x.mean()
            expected = centered
            if n >= 18:
                b,a = butter(3,[0.7/15,4/15],btype="band")
                try:
                    expected = filtfilt(b,a,centered)
                except ValueError:
                    pass
            np.testing.assert_allclose(bandpass(x,30),expected.astype(np.float32),rtol=1e-6,atol=1e-7)

    def test_monotonic_and_shape_validation(self):
        for t in ([0,0,1],[0,2,1],[0,np.nan,2]):
            with self.assertRaises(ValueError):
                RGBTrace(np.array(t),np.ones((3,3)),"face",np.ones(3,bool))
        with self.assertRaises(ValueError):
            PhysiologicalSignal(np.ones(3),np.arange(3)/30,60,"face","GREEN")

    def test_known_delay_both_signs(self):
        for delay in (-0.12,0.12,0.0):
            _,x,y = waves(delay=delay)
            feat = estimate_delay(x,y,60,0.3,1.2)
            self.assertAlmostEqual(feat["delay_xcorr_sec"],delay,delta=1/60)
            self.assertAlmostEqual(feat["delay_phase_sec"],delay,delta=0.015)
            self.assertGreater(feat["delay_xcorr_score"],0.9)

    def test_resampling_known_delay_on_shared_grid(self):
        t,x,y = waves()
        # Identical irregular acquisition clock for both sites.
        jittered = t+0.001*np.sin(2*np.pi*0.31*t)
        traces = [RGBTrace(jittered,np.repeat(v[:,None],3,axis=1),site,np.ones(len(t),bool)) for v,site in ((x,"face"),(y,"hand"))]
        grid = shared_grid(jittered,90)
        a,b = [resample_trace(trace,grid,0.1)[0][:,1] for trace in traces]
        feat = estimate_delay(a,b,90,0.3,1.2)
        self.assertAlmostEqual(feat["delay_xcorr_sec"],0.12,delta=1/60)

    def test_long_gap_not_filled_and_endpoints_not_extrapolated(self):
        t = np.arange(100)/30
        valid = np.ones(100,bool)
        valid[30:60] = False
        values = np.ones((100,3))
        values[~valid] = np.nan
        trace = RGBTrace(t,values,"face",valid)
        grid = np.arange(-1,101)/30
        out,imputed = resample_trace(trace,grid,0.15)
        self.assertTrue(np.isnan(out[31:61]).all())
        self.assertTrue(np.isnan(out[0]).all())
        self.assertTrue(np.isnan(out[-1]).all())
        self.assertFalse(imputed.any())

    def test_short_missing_gap_marked(self):
        t = np.arange(20)/30
        valid = np.ones(20,bool)
        valid[5:7] = False
        trace = RGBTrace(t,np.repeat(t[:,None],3,axis=1),"hand",valid)
        values,imputed = resample_trace(trace,t,0.15)
        self.assertTrue(np.isfinite(values).all())
        self.assertEqual(np.flatnonzero(imputed).tolist(),[5,6])

    def test_zero_phase_length_and_delay(self):
        _,x,y = waves()
        a,b = [zero_phase_bandpass(v,60,0.7,4) for v in (x,y)]
        self.assertEqual(a.shape,x.shape)
        feat = estimate_delay(a[120:-120],b[120:-120],60,0.3,1.2)
        self.assertAlmostEqual(feat["delay_xcorr_sec"],0.12,delta=1/60)
        with self.assertRaises(ValueError):
            zero_phase_bandpass(np.ones(3),60,0.7,4)

    def test_no_filter_across_gap(self):
        t,x,_ = waves(fs=30)
        rgb = np.repeat((100+x)[:,None],3,axis=1)
        rgb[250:350] = np.nan
        sig = extract_signal(rgb,t,"face",PipelineConfig())
        self.assertEqual(sig.values.shape,t.shape)
        self.assertTrue(np.isnan(sig.values[220:380]).all())
        self.assertEqual(len(sig.preprocessing["segments"]),2)

    def test_common_timebase_required(self):
        t,x,y = waves()
        face = PhysiologicalSignal(x,t,60,"face","GREEN")
        hand = PhysiologicalSignal(y,t+0.01,60,"hand","GREEN")
        with self.assertRaises(ValueError):
            extract_features(face,hand)

    def test_features_are_label_free_and_flat_is_not_hr(self):
        t,x,y = waves()
        features = extract_features(PhysiologicalSignal(x,t,60,"face","GREEN"),PhysiologicalSignal(y,t,60,"hand","GREEN"))
        self.assertFalse(set(features)&{"sbp","dbp","subject_id"})
        self.assertEqual(set(features),set(FEATURE_KEYS))
        self.assertTrue(np.isfinite(features["face_rise_time_sec"]))
        self.assertTrue(np.isnan(signal_quality(np.ones(600),30)["hr_bpm"]))

    def test_classical_methods_preserved_length_and_green_hr(self):
        t,x,_ = waves(fs=30)
        rgb = np.column_stack([130+0.8*x,110+x,90+0.3*x+0.2*np.sin(2*np.pi*0.23*t)])
        for name,method in METHOD_FUNCS.items():
            with self.subTest(method=name):
                self.assertEqual(method(rgb,30).shape,(len(t),))
        self.assertAlmostEqual(signal_quality(METHOD_FUNCS["GREEN"](rgb,30),30)["hr_bpm"],72,delta=0.3)


class FakeDetector(BaseRegionDetector):
    def __init__(self, missing=False):
        self.missing = missing

    def detect(self,frame_rgb,frame_index,timestamp):
        if self.missing or 200 <= frame_index < 235:
            return RegionObservation(frame_index,timestamp,reason="test_missing")
        # Only hand is used with this geometry in tests.
        points = np.tile([[16.,16.]],(21,1))
        points[[0,1,2,5,9,13,17]] = [[4,26],[4,18],[5,10],[10,5],[16,4],[22,5],[27,12]]
        return RegionObservation(frame_index,timestamp,bbox=(4,4,28,28),landmarks=points,valid=True)


def write_video(path,fs=30,seconds=12):
    writer = cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"MJPG"),fs,(32,32))
    if not writer.isOpened():
        raise RuntimeError("MJPG video writer unavailable")
    for i in range(int(fs*seconds)):
        value = int(round(120+8*np.sin(2*np.pi*1.2*i/fs)))
        writer.write(np.full((32,32,3),value,np.uint8))
    writer.release()


class VideoAndPipelineTests(unittest.TestCase):
    def test_legacy_roi_cache_roundtrip_without_pickle(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = np.arange(10)/30
            colors = np.arange(30,dtype=np.float32).reshape(10,3)
            cached = ROIExtractionResult(t,30,{"face":colors},np.ones(10,dtype=np.float32),(32,32))
            path = Path(tmp)/"cache.npz"
            cached.save_npz(path)
            restored = ROIExtractionResult.load_npz(path)
            np.testing.assert_array_equal(restored.timestamps,t)
            np.testing.assert_array_equal(restored.roi_rgb["face"],colors)
            self.assertEqual(restored.frame_shape,(32,32))

    def test_dual_known_video_delay_and_nominal_clock_gate(self):
        class GeometryDetector(BaseRegionDetector):
            def __init__(self, site):
                self.site = site
            def detect(self, frame, index, timestamp):
                theta = np.arange(478)*2*np.pi/17
                points = np.column_stack([24+14*np.cos(theta),32+14*np.sin(theta)])
                if self.site == "hand":
                    points = points[:21].copy()
                    points[:,0] += 48
                return RegionObservation(index,timestamp,landmarks=points,valid=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"dual.avi"
            writer = cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"MJPG"),30,(96,64))
            self.assertTrue(writer.isOpened())
            for i in range(900):
                frame = np.full((64,96,3),120,np.uint8)
                frame[:,:48,1] = int(round(120+10*np.sin(2*np.pi*1.2*i/30)))
                frame[:,48:,1] = int(round(120+10*np.sin(2*np.pi*1.2*(i/30-0.12))))
                writer.write(frame)
            writer.release()
            detectors = {site:GeometryDetector(site) for site in ("face","hand")}
            result = process_recording(path,PipelineConfig(),detectors=detectors,timestamps=np.arange(900)/30)
            self.assertGreater(result.delay["accepted_window_count"],0)
            self.assertAlmostEqual(result.delay["face_hand_delay_median_sec"],0.12,delta=1/30)
            nominal = process_recording(path,replace(PipelineConfig(),timestamp_mode="nominal"),detectors=detectors)
            self.assertEqual(nominal.delay["accepted_window_count"],0)
            self.assertTrue(all("unverified_nominal_timestamps" in row["reasons"] for row in nominal.bp_features))

    def test_missing_face_hand_full_frame_count_and_common_clock(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp)/"test.avi"
            write_video(video)
            t = np.arange(360)/30
            result = process_recording(video,PipelineConfig(),timestamps=t,
                                       detectors={"face":FakeDetector(True),"hand":FakeDetector(True)})
            self.assertEqual(len(result.face.rgb.timestamps),360)
            self.assertTrue(np.array_equal(result.face.rgb.timestamps,result.hand.rgb.timestamps))
            self.assertTrue(np.array_equal(result.face.rppg.timestamps,result.hand.rppg.timestamps))
            self.assertTrue(np.isnan(result.face.rppg.values).all())
            self.assertEqual(result.face.quality["valid_frame_fraction"],0)
            self.assertTrue(result.exclusions)
            self.assertTrue(all(r["status"] == "excluded" for r in result.bp_features))
            out = reserve_run(Path(tmp)/"run")
            save_result(result,out,provenance(video))
            with np.load(out/"signals.npz",allow_pickle=False) as data:
                self.assertEqual(len(data["hand_valid"]),360)
            summary = json.loads((out/"summary.json").read_text())
            self.assertIsNone(summary["hr"]["face"])
            with self.assertRaises(FileExistsError):
                reserve_run(out)

    def test_hand_only_with_temporary_loss_retains_long_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp)/"test.avi"
            write_video(video)
            config = replace(PipelineConfig(),regions=("hand",))
            result = process_recording(video,config,detectors={"hand":FakeDetector()},timestamps=np.arange(360)/30)
            self.assertEqual(len(result.hand.rgb.values),360)
            self.assertAlmostEqual(result.hr["hand"],72,delta=1.5)
            self.assertTrue(np.isnan(result.hand.rppg.values[200:235]).all())
            self.assertLess(result.hand.quality["valid_frame_fraction"],1)

    def test_external_clock_length_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp)/"test.avi"
            write_video(video,seconds=1)
            with self.assertRaises(ValueError):
                list(VideoReader(video,timestamps=np.arange(29)/30).frames())
            with self.assertRaises(ValueError):
                list(VideoReader(video,timestamps=np.arange(31)/30).frames())

    def test_downsampling_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp)/"test.avi"
            write_video(video,seconds=1)
            with self.assertRaisesRegex(ValueError,"antialiasing"):
                process_recording(video,replace(PipelineConfig(),sample_rate=15,regions=("hand",)),
                                  detectors={"hand":FakeDetector()},timestamps=np.arange(30)/30)

    def test_hand_identity_is_locked_after_loss(self):
        # Selection logic without constructing a MediaPipe model.
        detector = HandRegionDetector.__new__(HandRegionDetector)
        detector._center,detector._identity = None,None
        detector.expected_hand,detector.max_jump_fraction = "any",0.15
        frame = np.zeros((100,100,3),np.uint8)
        left = np.array([[10,10],[20,10],[20,20]],dtype=float)
        right = left+40
        obs = detector._select([(left,"Left",0.8),(right,"Right",0.9)],frame,0,0)
        self.assertEqual(obs.identity,"Right")
        self.assertFalse(detector._select([(left,"Left",0.99)],frame,1,0.03).valid)
        self.assertFalse(detector._select([(right+40,"Right",0.99)],frame,2,0.06).valid)
        self.assertTrue(detector._select([(right+1,"Right",0.9)],frame,3,0.09).valid)


class DatasetAndModelTests(unittest.TestCase):
    def test_split_no_overlap_repeated_sessions_and_seed(self):
        ids = [f"S{i}" for i in range(10) for _ in range(3)]
        split = split_subjects(ids,0.2,0.2,42)
        assert_subject_disjoint(*split)
        self.assertEqual(split,split_subjects(ids,0.2,0.2,42))
        self.assertEqual(set(sum(split,[])),set(ids))
        with self.assertRaises(ValueError):
            assert_subject_disjoint(["S1"],["S1"],[])
        subject = UBFCSubject("S1",Path("a"),Path("b"))
        with self.assertRaises(ValueError):
            subject_split([subject,subject])

    def test_duplicate_recordings_and_windows_rejected(self):
        partitions = {"S1":"train","S2":"test"}
        row = {"subject_id":"S1","recording_id":"a","input_sha256":"hash","start_sec":0,"end_sec":10}
        for duplicate in ({**row,"subject_id":"S2","start_sec":5,"end_sec":15},row):
            with self.assertRaises(ValueError):
                validate_sample_partitions([row,duplicate],partitions)

    def test_bp_train_only_scaling_and_model_roundtrip(self):
        x = np.array([[1,np.nan],[3,5]],dtype=float)
        y = np.array([[120,80],[130,85]],dtype=float)
        model = BPModel("ridge").fit(x,y,feature_keys=["face_hr_bpm","hand_hr_bpm"],subject_ids=["S1","S2"])
        before = model.pipeline.named_steps["scaler"].mean_.copy()
        model.predict(np.array([[10000,np.nan]]))
        np.testing.assert_array_equal(model.pipeline.named_steps["scaler"].mean_,before)
        self.assertEqual(before.tolist(),[2,5])
        with self.assertRaises(ValueError):
            BPModel().fit(x,y,feature_keys=["sbp_mmHg","dbp_mmHg"],subject_ids=["S1","S2"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"model.pkl"
            model.save(path)
            restored = BPModel.load(path)
            np.testing.assert_allclose(model.predict(x),restored.predict(x))

    def test_baseline_selection_train_only_schema(self):
        rows = []
        for sid,base in (("S1",1),("S2",2),("S3",1000)):
            for i in range(2):
                rows.append({"subject_id":sid,"recording_id":sid,"input_sha256":sid,"start_sec":i*5,"end_sec":i*5+10,
                             "status":"accepted","face_hr_bpm":base+i,"hand_hr_bpm":None if sid=="S1" else base,
                             "sbp_mmHg":120+base/100,"dbp_mmHg":80,"reference_start_time":"2026-10-06T10:00:00+03:30",
                             "reference_end_time":"2026-10-06T10:01:00+03:30","reference_device":"test",
                             "reference_relation":"before",
                             "association_method":"nearby_cuff_pilot","association_notes":"explicit approximation"})
        model,report = train_baselines(rows,{"S1":"train","S2":"validation","S3":"test"},kinds=("mean","ridge"))
        self.assertEqual(report["feature_keys"],["face_hr_bpm"])
        self.assertEqual(model.pipeline.named_steps["scaler"].mean_.tolist(),[1.5])
        self.assertIn("sbp",report["locked_test"])
        self.assertIn("dbp",report["locked_test"])

    def test_ubfc_target_exact_video_grid_regression(self):
        # Old implementation resampled GT at .01,.0433,... then paired with
        # video at .0333,.0667,..., introducing a 23 ms artificial offset.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gt = root/"ground_truth.txt"
            t_gt = np.arange(400)/100+0.01
            np.savetxt(gt,np.stack([t_gt,60+t_gt,t_gt]))
            t_vid = np.arange(120)/30
            rgb = np.full((120,3),100.)
            with patch("rppg_lab.datasets.extract_rgb_trace",return_value=(rgb,30,t_vid,np.ones(120))):
                ds = UBFCRPPGDataset([UBFCSubject("S1",root/"video.avi",gt)],prior_methods=[],win_sec=1)
            np.testing.assert_allclose(ds.records[0]["ppg"],ds.records[0]["timestamps"],atol=1e-6)

    def test_bp_null_feature_and_metrics_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/"S1"/"rest"
            root.mkdir(parents=True)
            (root/"features.json").write_text(json.dumps({"a":1,"b":None}))
            (root/"labels.json").write_text(json.dumps({"cuff":{"sbp":120,"dbp":80}}))
            ds = SessionBPDataset(Path(tmp),feature_keys=["a","b"])
            self.assertTrue(np.isnan(ds[0]["x_raw"].numpy()[1]))
        with self.assertRaises(ValueError):
            regression_metrics([1,2],[1])
        report = bp_metrics(np.array([[120,80],[122,82]]),np.array([[121,81],[123,83]]),["S1","S2"])
        self.assertEqual(report["sbp"]["me"],1)


class ConfigAndCuffTests(unittest.TestCase):
    def test_configs_load_and_reject_unknown_fields(self):
        for name in ("default","hr_experiment","bp_pilot"):
            PipelineConfig.load(Path(__file__).parents[1]/"configs"/f"{name}.json")
        with self.assertRaises(ValueError):
            replace(PipelineConfig(),filter_high_hz=20)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"bad.json"
            path.write_text('{"unknown_field":1}')
            with self.assertRaises(TypeError):
                PipelineConfig.load(path)

    def test_reference_validation_and_explicit_association(self):
        for kwargs in ({"sbp":80},{"start_time":"2026-10-06T10:00:00"},{"relation_to_video":"continuous"}):
            values = dict(sbp=120,dbp=80,measurement_index=0,start_time="2026-10-06T10:00:00+03:30",
                          end_time="2026-10-06T10:01:00+03:30",relation_to_video="before",device="test")
            with self.assertRaises(ValueError):
                BPReference(**{**values,**kwargs})
        with self.assertRaises(ValueError):
            ReferenceAssociation(0,0,10,"simultaneous","")
        example = Recording.load(Path(__file__).parents[1]/"configs"/"recording.example.json")
        self.assertEqual(example.subject_id,"S001")

    def test_unassociated_window_never_gets_label(self):
        from types import SimpleNamespace
        ref = BPReference(120,80,0,"2026-10-06T09:58:00+03:30","2026-10-06T09:59:00+03:30","before","cuff")
        recording = Recording("S1","rest","r1",Path("test.avi"),"2026-10-06T10:00:00+03:30",
                              references=[ref],associations=[ReferenceAssociation(0,5,15,"nearby_cuff_pilot","not simultaneous")])
        result = SimpleNamespace(shared=SimpleNamespace(original_timestamps=np.arange(600)/30),video=SimpleNamespace(fps_nominal=30),
                                 bp_features=[{"start_sec":0,"end_sec":10,"status":"accepted","reasons":[]},
                                              {"start_sec":5,"end_sec":15,"status":"accepted","reasons":[]}])
        rows = associate_windows(result,recording,"test_hash")
        self.assertIsNone(rows[0]["sbp_mmHg"])
        self.assertEqual(rows[0]["status"],"excluded")
        self.assertEqual(rows[1]["sbp_mmHg"],120)
        self.assertEqual(rows[1]["association_method"],"nearby_cuff_pilot")


if __name__ == "__main__":
    unittest.main()
