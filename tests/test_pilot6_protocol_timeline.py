from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np
from openpyxl import Workbook

from rppg_lab.pilot6_protocol_timeline import (SESSION_HEADERS, workbook_snapshot, load_sessions, metadata_changes,
                                             cuff_events, timeline_masks, candidate_intervals, REQUIRED_SHEETS)


def session_fixture(number=1):
    raw = {h: None for h in SESSION_HEADERS}
    raw.update(Session_ID=f"S{number:03d}_V01", Subject_ID=f"S{number:03d}", Event_Time_Source="Legacy_needs_video_confirmation",
               Manual_Exclude="No", Hand_ROI_Side="Right", Cuff_Arm="Left")
    starts, ends = ([60, 145, 225], [133, 209, 292]) if number == 4 else ([60, 145 if number == 1 else 140, 220], [110, 195 if number == 1 else 190, 270])
    for i, (a, b) in enumerate(zip(starts, ends), 1):
        raw.update({f"BP{i}_Start_s": a, f"BP{i}_End_s": b, f"SBP{i}_mmHg": 120+i, f"DBP{i}_mmHg": 80+i})
    return dict(session_id=raw["Session_ID"], subject_id=raw["Subject_ID"], raw=raw, source_row=4, manual_excluded=False)


def write_fixture(path, missing_header=None):
    b = Workbook()
    b.remove(b.active)
    for name in REQUIRED_SHEETS:
        s = b.create_sheet(name)
        s.append(["title"])
        s.append(["group"])
        if name == "02_Sessions":
            h = [k for k in SESSION_HEADERS if k != missing_header]
            s.append(h)
            for n in (4, 2, 6, 1, 5, 3):
                raw = session_fixture(n)["raw"]
                s.append([raw.get(k) for k in h])
        elif name == "01_Subjects":
            s.append(["Subject_ID"])
            for n in range(1, 7):
                s.append([f"S{n:03d}"])
        else:
            s.append(["Window_ID"] if name == "04_Windows" else ["Source"])
            s.append(["=1+2"])
    b.save(path)


class ProtocolTimelineTests(unittest.TestCase):
    def test_complete_workbook_headers_formula_and_raw_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"updated.xlsx"
            write_fixture(p)
            snap = workbook_snapshot(p)
            self.assertEqual(len(snap["sha256"]), 64)
            self.assertEqual(list(snap["worksheets"]), list(REQUIRED_SHEETS))
            self.assertEqual(snap["worksheets"]["04_Windows"]["raw_cells"][3][0], "=1+2")
            self.assertIsNone(snap["worksheets"]["04_Windows"]["cached_cells"][3][0])
            self.assertEqual(set(snap["worksheets"]["02_Sessions"]["headers"]), set(SESSION_HEADERS))

    def test_no_fallback_missing_latest(self):
        with self.assertRaisesRegex(FileNotFoundError, "no older fallback"):
            workbook_snapshot("__absent_updated_export__.xlsx")

    def test_updated_headers_required_not_old_aliases(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"bad.xlsx"
            write_fixture(p, "FPS_Reported")
            with self.assertRaisesRegex(ValueError, "FPS_Reported"):
                load_sessions(workbook_snapshot(p), [])

    def test_matching_uses_session_ids_not_row_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"updated.xlsx"
            write_fixture(p)
            s = load_sessions(workbook_snapshot(p), [f"S{n:03d}_V01.mp4" for n in range(1, 7)])
            self.assertEqual([r["subject_id"] for r in s], [f"S{n:03d}" for n in range(1, 7)])

    def test_conflicting_explicit_filename_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"updated.xlsx"
            write_fixture(p)
            s = workbook_snapshot(p)
            tab = s["worksheets"]["02_Sessions"]
            tab["raw_cells"][3][tab["headers"].index("Video_File_or_Link")] = "S001_V01.mp4"
            with self.assertRaisesRegex(ValueError, "Ambiguous"):
                load_sessions(s, ["S001_V01.mp4"])

    def test_duplicate_session_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"updated.xlsx"
            write_fixture(p)
            s = workbook_snapshot(p)
            s["worksheets"]["02_Sessions"]["raw_cells"].append(s["worksheets"]["02_Sessions"]["raw_cells"][3])
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_sessions(s, [])

    def test_s001_s004_actual_event_exceptions(self):
        self.assertEqual([(e["start_sec"], e["end_sec"]) for e in cuff_events(session_fixture(1), 301)], [(60, 110), (145, 195), (220, 270)])
        self.assertEqual([(e["start_sec"], e["end_sec"]) for e in cuff_events(session_fixture(4), 301)], [(60, 133), (145, 209), (225, 292)])

    def test_half_open_guard_boundaries(self):
        t = np.array([54.9, 55, 59.9, 60, 109.9, 110, 114.9, 115, 140])
        m = timeline_masks(t, cuff_events(session_fixture(), 301), 5)
        self.assertEqual(m["cuff_active"].tolist(), [False, False, False, True, True, False, False, False, False])
        self.assertEqual(m["candidate_cuff_free"].tolist(), [True, False, False, False, False, False, False, True, False])
        self.assertFalse(m["verified_cuff_free"].any())
        self.assertTrue(m["timing_uncertain"].all())

    def test_missing_endpoint_not_imputed_or_verified(self):
        s = session_fixture()
        s["raw"]["BP1_End_s"] = None
        events = cuff_events(s, 301)
        self.assertTrue(events[0]["missing_end"])
        m = timeline_masks(np.arange(301.), events)
        self.assertFalse(m["candidate_cuff_free"][55:].any())
        self.assertEqual(candidate_intervals(events, 0, 301, 5), [(0, 55)])

    def test_missing_start_is_conservative(self):
        s = session_fixture()
        s["raw"]["BP1_Start_s"] = None
        m = timeline_masks(np.arange(301.), cuff_events(s, 301))
        self.assertFalse(m["candidate_cuff_free"][:115].any())

    def test_timer_is_not_visual_verification(self):
        s = session_fixture()
        s["raw"]["Event_Time_Source"] = "Timer"
        self.assertFalse(any(e["timing_verified"] for e in cuff_events(s, 301)))

    def test_invalid_order_overlap_and_bounds_fail(self):
        for field, value in (("BP1_End_s", 50), ("BP2_Start_s", 100), ("BP3_End_s", 400), ("BP2_Start_s", 40)):
            s = session_fixture()
            s["raw"][field] = value
            with self.assertRaises(ValueError):
                cuff_events(s, 301)

    def test_formula_source_time_rejected(self):
        s = session_fixture()
        s["raw"]["BP1_End_s"] = "=BP1_Start_s+50"
        with self.assertRaisesRegex(ValueError, "formula"):
            cuff_events(s, 301)

    def test_timestamp_order_not_sorted_or_concatenated(self):
        with self.assertRaisesRegex(ValueError, "no concatenation"):
            timeline_masks([0, 1, 0, 1], [], 5)

    def test_change_report_id_keyed_and_no_source_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"updated.xlsx"
            write_fixture(p)
            old = workbook_snapshot(p)
            new = deepcopy(old)
            tab = new["worksheets"]["02_Sessions"]
            tab["raw_cells"][3][tab["headers"].index("BP1_End_s")] = 150
            changes = metadata_changes(old, new)
            self.assertEqual(changes[0]["identity"], "S004_V01")
            self.assertEqual(changes[0]["field"], "BP1_End_s")
            self.assertEqual(workbook_snapshot(p)["sha256"], old["sha256"])

    def test_derived_window_formula_changes_are_audited_not_observations(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"updated.xlsx"
            write_fixture(p)
            old = workbook_snapshot(p)
            new = deepcopy(old)
            new["worksheets"]["04_Windows"]["raw_cells"][3][0] = "=3+4"
            changes = metadata_changes(old, new)
            self.assertTrue(any(r["sheet"] == "04_Windows" and r["new"] == "=3+4" for r in changes))


if __name__ == "__main__":
    unittest.main()
