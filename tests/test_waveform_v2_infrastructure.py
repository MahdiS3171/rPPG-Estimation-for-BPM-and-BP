"""Synthetic fixture tests only: do not require real PURE, UBFC or private videos."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from rppg_lab.waveform_v2_registry import (discover_pure, inventory_pure,
    draft_participant_split, canonical_json_sha256, write_new_json)
from rppg_lab.waveform_v2_benchmark import matched_method_index, coverage_only


def fixture(root: Path, person: int, trial: int = 1, timestamps=(1000, 1017, 1034),
            *, waveform=(.1, .2, .3)) -> None:
    name = f"{person:02d}-{trial:02d}"
    folder = root / name
    images = folder / name
    images.mkdir(parents=True)
    for i in range(len(timestamps)):
        (images / f"{i:05d}.png").write_bytes(b"test-only-image-payload")
    data = {"/FullPackage": [
        {"Timestamp": ts, "Value": {"waveform": w}}
        for ts, w in zip(timestamps, waveform)]}
    (folder / f"{name}.json").write_text(json.dumps(data), encoding="utf-8")


class PUREInventoryTests(unittest.TestCase):
    def test_safe_discovery_and_inventory_is_not_claim_of_sync(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root, 1, 1)
            fixture(root, 1, 2)
            fixture(root, 2, 1)
            (root / "S007_V01.mp4").write_bytes(b"reserved")
            self.assertEqual(len(discover_pure(root)), 3)
            out = inventory_pure(root)
            self.assertEqual(out["participant_count"], 2)
            self.assertEqual(out["recording_count"], 3)
            self.assertEqual({r["participant_id"] for r in out["recordings"]}, {"PURE:01", "PURE:02"})
            self.assertTrue(all(r["raw_video_sha256"] is None for r in out["recordings"]))
            self.assertTrue(all(not r["allow_absolute_timing_scoring"] for r in out["recordings"]))
            self.assertTrue(all(not r["allow_shape_supervision"] for r in out["recordings"]))
            self.assertTrue(all(r["video_clock_source"] == "unknown" for r in out["recordings"]))
            self.assertTrue(all(r["reference_clock_status"] == "strictly_increasing"
                                for r in out["recordings"]))

    def test_timestamp_violations_and_partial_waveform_are_marked(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root, 1, timestamps=(1000, 1000, 1020))
            fixture(root, 2, timestamps=(1000, 1010, 1005))
            fixture(root, 3, waveform=(.1, float('nan'), .3))
            rows = inventory_pure(root, reference_timestamp_unit="milliseconds")["recordings"]
            self.assertEqual([r["reference_clock_status"] for r in rows],
                             ["duplicates", "nonmonotonic", "strictly_increasing"])
            self.assertIn("incomplete_reference_waveform", rows[2]["admission_blockers"])
            self.assertTrue(all(not r["allow_hr_scoring"] for r in rows))

    def test_image_hash_is_opt_in_and_deterministic(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root, 1)
            a = inventory_pure(root, hash_images=True)
            b = inventory_pure(root, hash_images=True)
            self.assertEqual(a["recordings"][0]["raw_video_sha256"],
                             b["recordings"][0]["raw_video_sha256"])

    def test_missing_reference_flags_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root, 1)
            (root / "01-01" / "01-01.json").unlink()
            row = inventory_pure(root)["recordings"][0]
            self.assertIn("no_valid_reference_entries", row["admission_blockers"])

    def test_access_approval_requires_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root, 1)
            with self.assertRaises(ValueError):
                inventory_pure(root, access_status="verified")

    def test_split_is_subject_disjoint_deterministic_and_only_draft(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for sid in range(1, 7):
                fixture(root, sid)
                fixture(root, sid, trial=2)
            manifest = inventory_pure(root)
            a = draft_participant_split(manifest, seed=31)
            b = draft_participant_split(manifest, seed=31)
            self.assertEqual(canonical_json_sha256(a), canonical_json_sha256(b))
            self.assertEqual(len(a["recording_assignments"]), 12)
            for sid in range(1, 7):
                self.assertEqual(a["recording_assignments"][f"PURE:{sid:02d}-01"],
                                 a["recording_assignments"][f"PURE:{sid:02d}-02"])
            self.assertEqual(a["status"], "DRAFT_NOT_FROZEN")

    def test_write_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "manifest.json"
            write_new_json(path, {"ok": True})
            with self.assertRaises(FileExistsError):
                write_new_json(path, {"ok": False})


class MatchedBenchmarkTests(unittest.TestCase):
    def row(self, method: str, **kw):
        row = dict(dataset_id="PURE", participant_id="PURE:01", recording_id="PURE:01-01",
                   start_ns=0, end_ns=5000000000, method=method,
                   source_sha256="filehash", reference_sha256="refhash",
                   input_contract_id="same_roi_clock_source", reference_hr_bpm=73.)
        row.update(kw)
        return row

    def test_missing_methods_count_as_failures(self):
        rows = [self.row("__intent__"), self.row("GREEN", status="valid", hr_bpm=72.)]
        index = matched_method_index(rows)
        out = coverage_only(index)
        self.assertEqual(out["intended_windows"], 1)
        self.assertEqual(out["matched_all_valid"], 0)
        self.assertEqual(out["valid_by_method"]["GREEN"], 1)
        self.assertEqual(out["failure_reasons"]["CHROM"]["missing_prediction"], 1)

    def test_provenance_mismatch_fails_closed(self):
        with self.assertRaises(ValueError):
            matched_method_index([self.row("__intent__"), self.row("GREEN", status="valid",
                                                         hr_bpm=71, input_contract_id="different")])

    def test_predictions_without_registered_intent_fail(self):
        with self.assertRaises(ValueError):
            matched_method_index([self.row("GREEN", status="valid", hr_bpm=75)])

    def test_reserved_subject_rejected(self):
        with self.assertRaises(ValueError):
            matched_method_index([self.row("__intent__", participant_id="S007")])

    def test_duplicate_and_invalid_prediction_rejected(self):
        intent = self.row("__intent__")
        with self.assertRaises(ValueError):
            matched_method_index([intent, intent])
        with self.assertRaises(ValueError):
            matched_method_index([intent, self.row("GREEN", status="invalid")])


if __name__ == "__main__":
    unittest.main()
