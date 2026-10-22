from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import csv
from datetime import time
import io
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from openpyxl import Workbook

from rppg_lab.config import PipelineConfig
from rppg_lab.pilot_metadata import read_tables, normalize_tables, strict_timestamp, match_video, align_references, fetch_google_workbook
from rppg_lab.pilot_video import mp4_presentation_times, inspect_video, traces_from_arrays
from rppg_lab.pilot_analysis import validate_timing, window_quality, exploratory_bp, analyze_signals, median_pulse_template
from rppg_lab.pilot6 import checkpoint_audit, session_features, write_report, run
from rppg_lab.processing import zero_phase_bandpass, resample_trace, shared_grid
from rppg_lab.region_masks import extract_frame_rgb
from rppg_lab.types import RGBTrace
from rppg_lab.types import RegionObservation
from rppg_lab.pilot_video import extract_raw
from rppg_lab.pilot_detection import PilotHandDetector


def metadata_fixture():
    tables = {"Sessions":dict(headers=["Person","Visit","File","Systolic","Diastolic","Start","End","Source"],
               duplicate_headers=[],rows=[dict(source_row=2,values={"Person":"S001","Visit":"S001_V01",
               "File":"S001_V01.mp4","Systolic":"125","Diastolic":"75","Start":"60","End":"110","Source":"Timer"})])}
    mapping = {"sheets":{"Sessions":{"kind":"session","columns":{"subject_id":"Person","recording_id":"Visit", "session_id":"Visit","video_path":"File"},
               "measurements":[{"sbp_mmHg":"Systolic","dbp_mmHg":"Diastolic","bp_relative_start_sec":"Start","bp_relative_end_sec":"End","event_time_source":"Source"}]}}}
    return tables,mapping


class MetadataTests(unittest.TestCase):
    def test_authorized_google_reads_all_tabs_without_persisting_token(self):
        def response(request,timeout):
            self.assertEqual(request.get_header("Authorization"),"Bearer fixture-private-token")
            if "/values/" in request.full_url:
                value={"values":[["Subject_ID","Original Header"],["S001","value"]]}
            else:
                value={"sheets":[{"properties":{"title":"Actual Subjects","sheetId":1}},
                                 {"properties":{"title":"Actual Sessions","sheetId":2}}]}
            return io.BytesIO(json.dumps(value).encode())
        with tempfile.TemporaryDirectory() as tmp:
            with patch("urllib.request.urlopen",side_effect=response) as get:
                p=fetch_google_workbook("fixture_sheet",tmp,"fixture-private-token")
            self.assertEqual(get.call_count,3)
            self.assertEqual(list(read_tables(p)),["Actual Subjects","Actual Sessions"])
            self.assertNotIn("fixture-private-token",(Path(tmp)/"google_raw_tables.json").read_text())

    def test_google_permission_failure_is_not_fabricated_workbook(self):
        from urllib.error import HTTPError
        with tempfile.TemporaryDirectory() as tmp:
            with patch("urllib.request.urlopen",side_effect=HTTPError("fixture",401,"Unauthorized",None,None)):
                with self.assertRaisesRegex(HTTPError,"401"):
                    fetch_google_workbook("fixture_sheet",tmp)
            self.assertFalse((Path(tmp)/"metadata.xlsx").exists())

    def test_csv_header_preamble_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"tab.csv"
            p.write_text("title,,\ncategory,,\nSubject_ID,Distance cm,Note\nS001,55,\n",encoding="utf-8")
            table=read_tables(p,{"Original Tab":2},"Original Tab")["Original Tab"]
            self.assertEqual(table["headers"],["Subject_ID","Distance cm","Note"])
            self.assertEqual(table["raw_matrix"][0][0],"title")
            self.assertEqual(table["rows"][0]["source_row"],4)

    def test_excel_all_tab_names_and_headers(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"book.xlsx";book=Workbook();sheet=book.active;sheet.title="Actual Name"
            sheet.append(["ID","Exact_Header cm","Source clock"]);sheet.append(["S001",5,time(18,30)]);book.create_sheet("Guide").append(["Meaning"]);book.save(p)
            tables=read_tables(p)
            self.assertEqual(list(tables),["Actual Name","Guide"])
            self.assertEqual(tables["Actual Name"]["headers"][1],"Exact_Header cm")
            self.assertEqual(tables["Actual Name"]["raw_matrix"][1][2],"18:30:00")

    def test_duplicate_headers_cannot_silently_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"bad.csv";p.write_text("ID,ID\nS001,S002\n")
            table=read_tables(p)["bad"]
            self.assertEqual(table["duplicate_headers"],["ID"])
            self.assertEqual(table["rows"],[])

    def test_explicit_mapping_and_raw_values(self):
        tables,mapping=metadata_fixture();records,issues=normalize_tables(tables,mapping)
        self.assertFalse(issues);self.assertEqual(len(records),2)
        self.assertEqual(records[1]["normalized"]["sbp_mmHg"],125)
        self.assertEqual(records[1]["raw"]["Systolic"],"125")
        self.assertEqual(records[1]["measurement_index"],1)

    def test_units_missing_bp_and_absent_header(self):
        tables,mapping=metadata_fixture();tables["Sessions"]["rows"][0]["values"]["Systolic"]=""
        mapping["sheets"]["Sessions"]["columns"]["suprasternal_face_cm"]={"column":"Start","scale":.1}
        mapping["sheets"]["Sessions"]["columns"]["fps"]="MissingFPS"
        records,issues=normalize_tables(tables,mapping)
        self.assertIsNone(records[1]["normalized"]["sbp_mmHg"])
        self.assertEqual(records[0]["normalized"]["suprasternal_face_cm"],6)
        self.assertTrue(any(i["reason"]=="mapped_source_column_missing" for i in issues))

    def test_invalid_bp_not_repaired(self):
        tables,mapping=metadata_fixture();tables["Sessions"]["rows"][0]["values"]["Diastolic"]="130"
        records,issues=normalize_tables(tables,mapping)
        self.assertIsNone(records[1]["normalized"]["sbp_mmHg"])
        self.assertTrue(any(i["reason"]=="invalid_bp_order" for i in issues))

    def test_formula_template_rows_not_subjects(self):
        tables,mapping=metadata_fixture();cfg=mapping["sheets"]["Sessions"];cfg["required_id_column"]="Person"
        tables["Sessions"]["rows"].append(dict(source_row=3,values={"Person":None,"Visit":"cached formula"}))
        records,_=normalize_tables(tables,mapping)
        self.assertEqual(len(records),2)

    def test_no_row_order_match_and_ambiguous_exact_filename(self):
        tables,mapping=metadata_fixture();records,_=normalize_tables(tables,mapping)
        self.assertEqual(match_video("S001_V01.mp4",records)["match_status"],"matched_session")
        self.assertEqual(match_video("S002_V01.mp4",records)["match_status"],"unmatched")
        self.assertEqual(match_video("S001_V01.mp4",records+deepcopy(records))["match_status"],"ambiguous")

    def test_subject_only_not_session(self):
        rec=dict(table_kind="subject",normalized={"subject_id":"S001"})
        self.assertEqual(match_video("S001_V01.mp4",[rec])["match_status"],"subject_only")

    def test_timestamp_calendar_never_guessed(self):
        for text,spec in [("1405-07-22",{}),("2026-10-22 18:30",{"format":"iso8601"}),("2026-10-22T18:30:00",{"format":"iso8601"})]:
            with self.assertRaises(ValueError):strict_timestamp(text,spec)
        self.assertEqual(strict_timestamp("2026-10-22T18:30:00+03:30",{"format":"iso8601"}),"2026-10-22T18:30:00+03:30")
        self.assertIn("+03:30",strict_timestamp("22/10/2026 18:30",{"format":"strptime:%d/%m/%Y %H:%M","utc_offset_minutes":210}))

    def test_bp_alignment_requires_end_and_verified_source(self):
        tables,mapping=metadata_fixture();records,_=normalize_tables(tables,mapping)
        identity=match_video("S001_V01.mp4",records)
        refs=align_references(identity,records,300)
        self.assertEqual(refs[0]["alignment"],"timed_cuff_interval")
        self.assertEqual(refs[0]["start_sec"],60)
        records[1]["normalized"]["event_time_source"]="Legacy_needs_video_confirmation"
        refs=align_references(identity,records,300)
        self.assertEqual(refs[0]["alignment"],"session_only")
        self.assertIsNone(refs[0]["start_sec"])
        self.assertEqual(refs[0]["logged_start_sec"],60)
        records[1]["normalized"]["bp_relative_end_sec"]=None
        self.assertEqual(align_references(identity,records,300)[0]["alignment"],"session_only")

    def test_missing_bp_and_ambiguous_session_cannot_align(self):
        tables,mapping=metadata_fixture();records,_=normalize_tables(tables,mapping);identity=match_video("S001_V01.mp4",records)
        identity["match_status"]="ambiguous"
        self.assertEqual(align_references(identity,records,300)[0]["alignment"],"unavailable")
        records[1]["normalized"]["sbp_mmHg"]=None
        identity["match_status"]="matched_session"
        self.assertEqual(align_references(identity,records,300)[0]["reason"],"missing_bp")


def atom(kind,data):
    return struct.pack(">I4s",len(data)+8,kind)+data


def mp4_clock_fixture(durations,offsets=None,edits=None):
    mvhd=bytearray(24);struct.pack_into(">I",mvhd,12,1000)
    mdhd=bytearray(24);struct.pack_into(">I",mdhd,12,1000)
    hdlr=bytearray(20);hdlr[8:12]=b"vide"
    entries=b"".join(struct.pack(">II",1,d) for d in durations)
    stts=atom(b"stts",b"\0"*4+struct.pack(">I",len(durations))+entries)
    if offsets is not None:
        stts+=atom(b"ctts",b"\1\0\0\0"+struct.pack(">I",len(offsets))+b"".join(struct.pack(">Ii",1,d) for d in offsets))
    mdia=atom(b"mdia",atom(b"mdhd",mdhd)+atom(b"hdlr",hdlr)+atom(b"minf",atom(b"stbl",stts)))
    if edits:
        elst=atom(b"elst",b"\0"*4+struct.pack(">I",len(edits))+b"".join(struct.pack(">Iihh",*e) for e in edits))
        mdia+=atom(b"edts",elst)
    return atom(b"moov",atom(b"mvhd",mvhd)+atom(b"trak",mdia))


class ClockSignalTests(unittest.TestCase):
    def test_template_repeatability_without_phase_warp_or_flat_replacement(self):
        fs=30;t=np.arange(300)/fs
        x=median_pulse_template(np.sin(2*np.pi*1.2*t),fs)
        y=median_pulse_template(np.sin(2*np.pi*1.2*(t-.2)),fs)
        self.assertIsNotNone(x);self.assertIsNotNone(y)
        n=min(len(x),len(y));good=np.isfinite(x[:n])&np.isfinite(y[:n])
        self.assertGreater(np.corrcoef(x[:n][good],y[:n][good])[0,1],.98)
        self.assertIsNone(median_pulse_template(np.ones(300),fs))
    def test_one_hand_adapter_keeps_historical_default_separate(self):
        import mediapipe as mp
        with patch.object(mp.tasks.vision.HandLandmarker,"create_from_options") as create, \
                patch.object(PilotHandDetector,"_base_options",return_value=mp.tasks.BaseOptions(model_asset_buffer=b"fixture")):
            detector=PilotHandDetector(PipelineConfig(detector_backend="tasks"))
            self.assertEqual(create.call_args.args[0].num_hands,1)
            detector.close()

    def test_raw_extraction_tracking_failure_and_cache_identity(self):
        class MissingDetector:
            def __init__(self,*args,**kwargs):pass
            def detect(self,frame,index,t):return RegionObservation(index,t,reason="not_detected")
            def close(self):pass
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);video=root/"S001_V01.mp4"
            writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*"mp4v"),30,(64,64))
            for _ in range(12):writer.write(np.full((64,64,3),100,np.uint8))
            writer.release()
            face_model=root/"face.task";face_model.write_bytes(b"test")
            hand_model=root/"hand.task";hand_model.write_bytes(b"test")
            cfg=PipelineConfig(face_model=str(face_model),hand_model=str(hand_model))
            with patch("rppg_lab.pilot_video.FaceRegionDetector",MissingDetector),patch("rppg_lab.pilot_video.PilotHandDetector",MissingDetector):
                data=extract_raw(video,root/"run",cfg)
                self.assertEqual(len(data["timestamps"]),12)
                self.assertTrue(np.isnan(data["combined_rgb"]).all())
                self.assertTrue(np.isnan(data["hand_palm_rgb"]).all())
                self.assertTrue((data["hand_tracking_reason"]=="not_detected").all())
                cached=extract_raw(video,root/"run",cfg,reuse=True)
                np.testing.assert_equal(data["timestamps"],cached["timestamps"])
                with self.assertRaises(ValueError):extract_raw(video,root/"run",cfg,max_dimension=320,reuse=True)

    def test_variable_frame_intervals_and_signed_ctts(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"v.mp4";p.write_bytes(mp4_clock_fixture([17,16,34,17]))
            t,info=mp4_presentation_times(p)
            np.testing.assert_allclose(t,[0,.017,.033,.067]);self.assertEqual(info["sample_count"],4)
            p.write_bytes(mp4_clock_fixture([20,20,20],offsets=[20,-20,0]))
            t,_=mp4_presentation_times(p)
            np.testing.assert_allclose(t,[0,.02,.04])

    def test_empty_movie_lead_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"v.mp4";p.write_bytes(mp4_clock_fixture([20,20,20],edits=[(9,-1,1,0),(60,0,1,0)]))
            t,info=mp4_presentation_times(p)
            np.testing.assert_allclose(t,[.009,.029,.049]);self.assertEqual(info["empty_lead_sec"],.009)

    def test_complex_edit_and_duplicate_pts_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"v.mp4";p.write_bytes(mp4_clock_fixture([20,20],edits=[(40,0,2,0)]))
            with self.assertRaises(ValueError):mp4_presentation_times(p)
            p.write_bytes(mp4_clock_fixture([20,20],offsets=[0,-20]))
            with self.assertRaises(ValueError):mp4_presentation_times(p)

    def test_synthetic_both_signs_filter_and_irregular_clock(self):
        result=validate_timing()
        self.assertEqual(len(result["rows"]),60)
        self.assertLess(result["max_absolute_error_sec"],1/60)
        for row in result["rows"]:
            if abs(row["true_delay_sec"])>.05:
                self.assertEqual(np.sign(row["true_delay_sec"]),np.sign(row["estimated_delay_sec"]))

    def test_rgb_mask_and_tracking_failures(self):
        frame=np.zeros((10,10,3),np.uint8);frame[2:5,3:7]=[120,100,80];mask=np.zeros((10,10),bool);mask[2:5,3:7]=True
        color,q=extract_frame_rgb(frame,mask,10)
        np.testing.assert_equal(color,[120,100,80]);self.assertEqual(q["skin_pixel_count"],12)
        color,q=extract_frame_rgb(frame,mask,13)
        self.assertTrue(np.isnan(color).all());self.assertFalse(q["extraction_valid"])

    def test_long_gaps_and_bad_timestamps(self):
        t=np.arange(300)/30;valid=np.ones(300,bool);valid[100:150]=False
        values=np.repeat((100+np.sin(2*np.pi*1.2*t))[:,None],3,axis=1)
        trace=RGBTrace(t,values,"face",valid)
        rgb,_=resample_trace(trace,t,.15)
        self.assertTrue(np.isnan(rgb[100:150]).all())
        with self.assertRaises(ValueError):RGBTrace(np.zeros(300),values,"face",valid)

    def test_quality_rejection_flat_missing_and_hr_recovery(self):
        fs=30;t=np.arange(300)/fs;x=np.sin(2*np.pi*1.2*t)
        trace=RGBTrace(t,np.repeat((100+x)[:,None],3,axis=1),"face",np.ones(300,bool))
        cfg=PipelineConfig()
        self.assertEqual(window_quality(np.ones(300),trace,0,10,fs,cfg)["status"],"invalid")
        self.assertEqual(window_quality(np.full(300,np.nan),trace,0,10,fs,cfg)["status"],"invalid")
        q=window_quality(x,trace,0,10,fs,cfg)
        self.assertEqual(q["status"],"good");self.assertAlmostEqual(q["hr_bpm"],72,delta=.5)
        self.assertIsNone(q["detector_confidence"])

    def test_analysis_retains_failed_roi_windows_and_generates_figures(self):
        fs=30;t=np.arange(31*fs)/fs;x=np.sin(2*np.pi*1.2*t)
        names=("forehead","left_cheek","right_cheek","combined","cheeks","hand_palm","hand_center")
        traces={n:RGBTrace(t,np.column_stack([130+.8*x,110+x,90+.3*x]),n,np.ones(len(t),bool)) for n in names}
        traces["hand_center"]=RGBTrace(t,np.full((len(t),3),np.nan),"hand_center",np.zeros(len(t),bool))
        with tempfile.TemporaryDirectory() as tmp:
            q,d,s=analyze_signals(traces,PipelineConfig(),Path(tmp))
            self.assertTrue(any(r["roi"]=="hand_center" and r["status"]=="invalid" for r in q))
            self.assertTrue(all(r["status"]=="invalid" for r in d if r["hand_roi"]=="hand_center"))
            self.assertTrue((Path(tmp)/"interpretability.png").is_file())
            self.assertTrue((Path(tmp)/"delay_correlation.png").is_file())
            self.assertTrue((Path(tmp)/"delay_summary.json").is_file())
            self.assertEqual(len(s),21)


class FrozenBPSafetyTests(unittest.TestCase):
    def test_self_contained_checkpoint_never_substitutes_for_original_manifest(self):
        import torch
        checkpoint=dict(waveform_v1_status="frozen_v1",freeze_decision="FREEZE_WAVEFORM_V1",
                        model_class="test",model_config={},roi_names=[],channel_ordering=[],fs=30,
                        window_length_samples=300,selected_epoch=1,frozen_from_checkpoint_sha256="source",extraction_config={})
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"frozen.pt";torch.save(checkpoint,p)
            with patch("rppg_lab.waveform_training.checkpoint_dataset_config"), \
                 patch("rppg_lab.waveform_training.reconstruct_waveform_model",return_value=SimpleNamespace(training=False)), \
                 patch("rppg_lab.waveform_inference.FrozenWaveformExtractor") as extractor:
                info,model=checkpoint_audit(p,enable=True)
                self.assertEqual(info["reason"],"required_original_freeze_manifest_missing")
                self.assertIsNone(model);extractor.assert_not_called()

    def test_dry_run_preserves_failed_video_and_generates_audit_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/"S001_V01.mp4").write_bytes(b"invalid video fixture")
            metadata=root/"subjects.csv";metadata.write_text("Subject_ID\nS001\n")
            mapping=root/"map.json";mapping.write_text(json.dumps(dict(csv_sheet_name="Subjects",sheets={"Subjects":dict(kind="subject",columns={"subject_id":"Subject_ID"})})))
            args=SimpleNamespace(out_dir=str(root/"run"),reuse=False,metadata_mapping=str(mapping),metadata_file=str(metadata),
                                 google_sheet_id=None,google_token_env="UNUSED_TEST_TOKEN",checkpoint=str(root/"missing.pt"),freeze_manifest=None,
                                 frozen=False,seed=42,dataset_root=str(root),recording=None,dry_run=True,metadata_only=False,
                                 workers=1,max_frames=None,max_dimension=640,private_overlays=False)
            with patch("rppg_lab.pilot6.inspect_video",side_effect=RuntimeError("fixture decode failure")):
                self.assertEqual(run(args),0)
            audit=json.loads((root/"run/dataset_audit.json").read_text())
            self.assertEqual(audit[0]["recording_id"],"S001_V01")
            self.assertEqual(audit[0]["detail"],"fixture decode failure")
            self.assertTrue((root/"run/research_report.md").is_file())
            self.assertFalse(json.loads((root/"run/bp_feasibility.json").read_text())["bp_accuracy_validated"])

    def test_missing_checkpoint_is_explicit(self):
        info,extractor=checkpoint_audit("not_a_real_checkpoint.pt")
        self.assertIsNone(extractor);self.assertEqual(info["reason"],"checkpoint_missing")

    def test_incompatible_checkpoint_does_not_infer(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"bad.pt";torch.save({"model_state":{}},p)
            info,extractor=checkpoint_audit(p,enable=True)
            self.assertIsNone(extractor);self.assertEqual(info["status"],"incompatible")
            self.assertEqual(len(info["sha256"]),64)

    def test_partial_recording_never_becomes_bp_training_observation(self):
        identity=dict(subject_id="S001",recording_id="S001_V01",session_id="S001_V01",match_status="matched_session",matched_row=None)
        refs=[dict(observed_cuff_sbp_mmHg=120,observed_cuff_dbp_mmHg=80,measurement_index=i) for i in (1,2,3)]
        features=session_features(identity,[],[],refs,True)
        self.assertFalse(features["association_eligible"])
        self.assertEqual(features["prediction_kind"],"observed_cuff_session_summary")
        self.assertEqual(exploratory_bp([features])["status"],"not_evaluated")

    def test_subject_grouped_baselines_and_duplicate_cuff_guard(self):
        rows=[dict(subject_id=f"S{i:03}",recording_id=f"S{i:03}_V01",association_eligible=True,
                   source_sheet="session",source_row=i,observed_cuff_sbp_mmHg=110+i,
                   observed_cuff_dbp_mmHg=70+i,face_hr_bpm=70+i,hand_hr_bpm=70+i) for i in range(1,5)]
        report=exploratory_bp(rows)
        self.assertFalse(report["bp_accuracy_validated"])
        self.assertTrue(report["predictions"])
        for r in report["predictions"]:
            self.assertNotIn(r["subject_id"],r["train_subjects"])
            self.assertNotEqual(r["prediction_kind"],"observed_cuff_session_summary")
        with self.assertRaises(ValueError):exploratory_bp(rows+[deepcopy(rows[0])])

    def test_report_includes_failures_and_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"report.md"
            write_report(p,dict(output="synthetic",checkpoint=dict(face_inference="blocked",reason="missing manifest")),
                         [dict(recording_id="S001_V01",status="failed",reason="no hand")])
            text=p.read_text(encoding="utf-8")
            self.assertIn("no hand",text);self.assertIn("not validated",text);self.assertIn("missing manifest",text)


if __name__ == "__main__":
    unittest.main()
