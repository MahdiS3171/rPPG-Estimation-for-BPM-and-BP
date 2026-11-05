"""Tiny synthetic HDF5 fixtures only; never opens SCAMPSData or private datasets."""
from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from rppg_lab.waveform_v2_scamps import (ScampsError, ScampsLayout, ScampsRecording,
    discover_scamps, inspect_scamps_file, numeric_audit, read_official_split, source_sha256)

LAYOUT = ScampsLayout(frames=6, raw_width=8, crop_width=4, height=3)


def fixture(root: Path, *, filename="P000001.mat", mask_count=5, compressed=True):
    path = root / filename
    with h5py.File(path, "w") as f:
        for name, width in (("RawFrames", 8), ("Xsub", 4)):
            # Each axis has a distinguishable contribution; no square-only transpose test.
            c, w, h, t = np.indices((3, width, 3, 6))
            values = .1 * c + .01 * w + .001 * h + .0001 * t
            ds = f.create_dataset(name, data=values,
                **(dict(chunks=(1, 2, 3, 2), compression="gzip") if compressed else {}))
            ds.attrs["Python.Shape"] = [6, 3, width, 3]
            ds.attrs["MATLAB_class"] = np.bytes_(b"double")
        w, h, t = np.indices((4, 3, mask_count))
        for name in ("skin_mask", "combined_mask"):
            values = (w * .1 + h * .01 + t * .001).astype(np.float32)
            ds = f.create_dataset(name, data=values, compression="gzip" if compressed else None)
            ds.attrs["Python.Shape"] = [mask_count, 3, 4]
        ds = f.create_dataset("d_ppg", data=np.arange(6, dtype=np.float64)[:, None] + .25)
        ds.attrs["Python.Shape"] = [6]
    return path


class ScampsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = fixture(self.root)

    def recording(self, **kwargs):
        return ScampsRecording(self.root, self.path.name, layout=LAYOUT, **kwargs)

    def test_discovery_is_explicit_sorted_and_nonrecursive(self):
        fixture(self.root, filename="P000002.mat")
        nested = self.root / "nested"
        nested.mkdir()
        fixture(nested, filename="P000003.mat")
        self.assertEqual([p.name for p in discover_scamps(self.root)], ["P000001.mat", "P000002.mat"])

    def test_empty_missing_and_unexpected_filename_roots_fail(self):
        self.path.unlink()
        with self.assertRaises(ScampsError):
            discover_scamps(self.root)
        with self.assertRaises(FileNotFoundError):
            discover_scamps(self.root / "missing")
        (self.root / "other.mat").write_bytes(b"fixture")
        with self.assertRaises(ScampsError):
            discover_scamps(self.root)

    def test_private_reserved_paths_fail_before_filesystem_access(self):
        with patch.object(Path, "is_symlink", side_effect=AssertionError("filesystem touched")):
            for path in ("Dataset", "Dataset/P000001.mat", "other/S007", "other/S011_V01"):
                with self.subTest(path=path), self.assertRaises(ScampsError):
                    discover_scamps(path)

    def test_symlink_paths_and_traversal_filenames_fail(self):
        with patch.object(Path, "is_symlink", return_value=True):
            with self.assertRaises(ScampsError):
                discover_scamps(self.root)
        for name in ("../P000001.mat", "nested/P000001.mat", "P000001.MAT", "S007.mat"):
            with self.subTest(name=name), self.assertRaises(ScampsError):
                ScampsRecording(self.root, name, layout=LAYOUT)

    def test_metadata_hash_layout_and_no_invented_clock(self):
        info = inspect_scamps_file(self.path, layout=LAYOUT)
        self.assertEqual(info["source_sha256"], hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertEqual(info["datasets"]["Xsub"]["shape"], [3, 4, 3, 6])
        self.assertEqual(info["datasets"]["Xsub"]["chunks"], [1, 2, 3, 2])
        self.assertEqual(info["datasets"]["Xsub"]["compression"], "gzip")
        self.assertEqual(info["mask_alignment"], "UNVERIFIED_LENGTH_MISMATCH")
        self.assertIsNone(info["embedded_capture_timestamps"])
        self.assertIsNone(info["embedded_sampling_rate_hz"])
        self.assertFalse(info["allow_training"])
        self.assertFalse(info["allow_absolute_timing_scoring"])
        self.assertIsNone(info["all_values_finite"])

    def test_explicit_axes_rgb_values_and_ppg_indices(self):
        window = self.recording().read_window(2, 5)
        self.assertEqual(window["rgb"].shape, (3, 3, 4, 3))
        self.assertAlmostEqual(window["rgb"][1, 2, 3, 1], .1 + .03 + .002 + .0003)
        np.testing.assert_array_equal(window["frame_indices"], [2, 3, 4])
        np.testing.assert_array_equal(window["label_indices"], [2, 3, 4])
        np.testing.assert_array_equal(window["d_ppg"], [2.25, 3.25, 4.25])
        self.assertTrue(window["validity"]["paired"].all())
        self.assertEqual(window["provenance"]["channel_order"], "RGB")
        self.assertIsNone(window["provenance"]["capture_timestamps"])
        self.assertFalse(window["provenance"]["physical_synchronization_verified"])

    def test_raw_frames_use_own_width_and_no_fabricated_segmentation(self):
        window = self.recording().read_window(0, 1, frames="RawFrames")
        self.assertEqual(window["rgb"].shape, (1, 3, 8, 3))
        self.assertIsNone(window["masks"])
        with self.assertRaises(ScampsError):
            self.recording().read_window(0, 1, frames="RawFrames", require_segmentation=True)

    def test_short_mask_stream_never_shifts_pads_or_truncates_targets(self):
        rec = self.recording()
        window = rec.read_window(4, 6)
        self.assertEqual(len(window["d_ppg"]), 2)
        self.assertIsNone(window["masks"])
        self.assertFalse(window["validity"]["segmentation_finite"].any())
        with self.assertRaisesRegex(ScampsError, "Segmentation"):
            rec.read_window(0, 2, require_segmentation=True)
        mask = rec.read_mask_window(3, 5)
        np.testing.assert_array_equal(mask["mask_indices"], [3, 4])
        self.assertIsNone(mask["frame_indices"])
        with self.assertRaises(ScampsError):
            rec.read_mask_window(4, 6)

    def test_equal_length_masks_decode_without_binarizing_fractional_values(self):
        self.path.unlink()
        fixture(self.root, mask_count=6)
        window = self.recording().read_window(1, 3, require_segmentation=True)
        self.assertEqual(window["masks"]["skin_mask"].shape, (2, 3, 4))
        self.assertAlmostEqual(float(window["masks"]["skin_mask"][0, 2, 3]), .321, places=6)
        self.assertTrue(window["validity"]["segmentation_finite"].all())
        self.assertFalse(window["provenance"]["allow_training"])

    def test_nonfinite_and_out_of_range_values_remain_explicit(self):
        with h5py.File(self.path, "r+") as f:
            f["Xsub"][0, 0, 0, 1] = np.nan
            f["Xsub"][1, 0, 0, 2] = 1.2
            f["d_ppg"][2, 0] = np.inf
        window = self.recording().read_window(0, 3)
        np.testing.assert_array_equal(window["validity"]["paired"], [True, False, False])
        self.assertTrue(np.isnan(window["rgb"][1]).any())
        self.assertTrue(np.isinf(window["d_ppg"][2]))
        self.assertLess(window["quality"]["finite_pixel_fraction"][1], 1)

    def test_every_source_is_opened_read_only(self):
        actual_file = h5py.File
        modes = []
        def checked_file(*args, **kwargs):
            mode = args[1] if len(args) > 1 else kwargs.get("mode", "r")
            modes.append(mode)
            return actual_file(*args, **kwargs)
        with patch("rppg_lab.waveform_v2_scamps.h5py.File", side_effect=checked_file):
            rec = self.recording()
            rec.read_window(0, 2)
            rec.read_mask_window(0, 2)
        self.assertTrue(modes)
        self.assertEqual(set(modes), {"r"})

    def test_lazy_reads_never_fetch_whole_frame_or_ppg_arrays(self):
        actual_getitem = h5py.Dataset.__getitem__
        reads = []
        def checked_getitem(ds, key):
            reads.append((ds.name, key))
            return actual_getitem(ds, key)
        rec = self.recording()
        with patch.object(h5py.Dataset, "__getitem__", new=checked_getitem):
            rec.read_window(2, 4)
        self.assertEqual(len(reads), 2)
        self.assertEqual(reads[0][1][-1], slice(2, 4))
        self.assertEqual(reads[1][1][0], slice(2, 4))

    def test_invalid_bounds_and_configured_memory_limits(self):
        rec = self.recording(max_window_frames=2)
        for start, stop in ((0, 3), (-1, 1), (0, 0), (5, 7), (True, 2), (0., 2)):
            with self.subTest(bounds=(start, stop)), self.assertRaises(ScampsError):
                rec.read_window(start, stop)
        with self.assertRaises(ScampsError):
            self.recording(max_window_bytes=1).read_window(0, 1)
        with self.assertRaises(ScampsError):
            self.recording(max_window_frames=0)
        with self.assertRaises(ScampsError):
            rec.read_window(0, 1, frames="automatic")

    def test_source_checksums_and_changed_sources_refused(self):
        with self.assertRaisesRegex(ScampsError, "checksum"):
            self.recording(expected_sha256="0" * 64)
        rec = self.recording(expected_sha256=source_sha256(self.path))
        with self.path.open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ScampsError, "changed"):
            rec.read_window(0, 1)

    def test_outputs_are_deterministic_independent_copies_and_source_unchanged(self):
        before = source_sha256(self.path)
        rec = self.recording()
        first, second = rec.read_window(1, 3), rec.read_window(1, 3)
        np.testing.assert_array_equal(first["rgb"], second["rgb"])
        first["rgb"][:] = 0
        np.testing.assert_array_equal(second["rgb"], rec.read_window(1, 3)["rgb"])
        self.assertEqual(before, source_sha256(self.path))

    def test_missing_required_variable_is_reported_and_rejected(self):
        with h5py.File(self.path, "r+") as f:
            del f["d_ppg"]
        info = inspect_scamps_file(self.path, layout=LAYOUT)
        self.assertIn("d_ppg", info["missing_variables"])
        self.assertFalse(info["structure_valid"])
        with self.assertRaises(ScampsError):
            self.recording()

    def test_unexpected_axes_shape_is_not_silently_transposed(self):
        with h5py.File(self.path, "r+") as f:
            data = f["RawFrames"][()].transpose(3, 2, 1, 0)
            del f["RawFrames"]
            f["RawFrames"] = data
        with self.assertRaisesRegex(ScampsError, "unexpected_shape"):
            self.recording()

    def test_conflicting_producer_axis_annotation_refused(self):
        with h5py.File(self.path, "r+") as f:
            f["Xsub"].attrs["Python.Shape"] = [3, 4, 3, 6]
        with self.assertRaisesRegex(ScampsError, "conflicting_producer"):
            self.recording()

    def test_wrong_target_length_and_mask_spatial_dimensions_refused(self):
        for key, shape in (("d_ppg", (5, 1)), ("skin_mask", (3, 4, 5)), ("combined_mask", (4, 3, 3))):
            with self.subTest(key=key):
                with h5py.File(self.path, "r+") as f:
                    original = f[key][()]
                    del f[key]
                    f[key] = np.zeros(shape)
                with self.assertRaises(ScampsError):
                    self.recording()
                with h5py.File(self.path, "r+") as f:
                    del f[key]
                    f[key] = original

    def test_non_hdf5_mat_is_reported_not_decoded_as_other_formats(self):
        self.path.write_bytes(b"MATLAB v5 or malformed fixture")
        info = inspect_scamps_file(self.path, layout=LAYOUT)
        self.assertEqual(info["admission_state"], "REJECTED_STRUCTURE")
        with self.assertRaises(ScampsError):
            self.recording()

    def test_soft_external_and_virtual_hdf5_sources_refused(self):
        for kind in ("soft", "external", "virtual"):
            with self.subTest(kind=kind):
                p = self.root / "P000002.mat"
                with h5py.File(p, "w") as f:
                    if kind == "soft":
                        f["RawFrames"] = h5py.SoftLink("/missing")
                    elif kind == "external":
                        f["RawFrames"] = h5py.ExternalLink("never_open.mat", "/RawFrames")
                    else:
                        virtual = h5py.VirtualLayout(shape=(3, 8, 3, 6), dtype="f8")
                        virtual[:] = h5py.VirtualSource("never_open.mat", "RawFrames", shape=(3, 8, 3, 6))
                        f.create_virtual_dataset("RawFrames", virtual)
                info = inspect_scamps_file(p, layout=LAYOUT, scan_values=True)
                self.assertFalse(info["structure_valid"])
                self.assertIsNone(info["all_values_finite"])

    def test_object_and_complex_numeric_types_refused(self):
        with h5py.File(self.path, "r+") as f:
            del f["d_ppg"]
            f["d_ppg"] = np.zeros((6, 1), dtype=np.complex128)
        with self.assertRaisesRegex(ScampsError, "unsupported_numeric"):
            self.recording()

    def test_auxiliary_waveforms_cannot_have_silent_length_mismatch(self):
        with h5py.File(self.path, "r+") as f:
            f["d_br"] = np.zeros((5, 1))
        with self.assertRaisesRegex(ScampsError, "auxiliary_waveform"):
            self.recording()

    def test_bounded_full_scan_chunked_and_contiguous_arrays(self):
        for compressed in (True, False):
            self.path.unlink()
            fixture(self.root, compressed=compressed)
            info = inspect_scamps_file(self.path, layout=LAYOUT, scan_values=True, max_block_bytes=64)
            self.assertTrue(info["all_values_finite"])
            for item in info["datasets"].values():
                audit = item["value_audit"]
                self.assertEqual(audit["count"], int(np.prod(item["shape"])))
                self.assertLessEqual(audit["largest_read_bytes"], 64)

    def test_full_scan_counts_nonfinite_range_and_fractional_masks(self):
        with h5py.File(self.path, "r+") as f:
            f["d_ppg"][5, 0] = np.nan
            f["combined_mask"][0, 0, 0] = 1.004
        info = inspect_scamps_file(self.path, layout=LAYOUT, scan_values=True, max_block_bytes=64)
        self.assertFalse(info["all_values_finite"])
        self.assertEqual(info["datasets"]["d_ppg"]["value_audit"]["nonfinite_count"], 1)
        self.assertEqual(info["datasets"]["combined_mask"]["value_audit"]["outside_unit_interval_count"], 1)
        self.assertGreater(info["datasets"]["skin_mask"]["value_audit"]["fractional_value_count"], 0)
        with h5py.File(self.path, "r") as f, self.assertRaises(ScampsError):
            numeric_audit(f["d_ppg"], max_block_bytes=1)

    def test_default_release_profile_does_not_accept_arbitrary_fixture_shape(self):
        with self.assertRaises(ScampsError):
            ScampsRecording(self.root, self.path.name)
        for kwargs in (dict(frames=0), dict(frames=True), dict(raw_width=2, crop_width=4)):
            with self.assertRaises(ScampsError):
                ScampsLayout(**kwargs)

    def test_official_split_header_anomaly_is_parsed_explicitly(self):
        p = self.root / "official_split.csv"
        p.write_text("nk,Split\nP000001,https://example.test/P000001.mat,Train\n"
                     "P000002,https://example.test/P000002.mat,Val\n")
        split = read_official_split(p, expected_sha256=source_sha256(p))
        self.assertEqual(split["assignments"]["P000002"]["split"], "Val")
        self.assertEqual(split["header_anomaly"], "2_header_fields_3_data_fields")
        self.assertFalse(split["identity_groups_available"])
        with self.assertRaisesRegex(ScampsError, "hash"):
            read_official_split(p)

    def test_split_duplicates_unknown_labels_bad_urls_and_bad_header_refused(self):
        for body in ("nk,Split\nP000001,url,Train\n", "a,b,c\n",
                     "nk,Split\nP000001,https://example/P000001.mat,Unknown\n",
                     "nk,Split\nP000001,https://example/P000002.mat,Test\n",
                     "nk,Split\nP000001,https://example/P000001.mat,Train\nP000001,https://example/P000001.mat,Val\n",
                     "nk,Split\nP000001,Train\n"):
            p = self.root / "split.csv"
            p.write_text(body)
            with self.subTest(body=body), self.assertRaises(ScampsError):
                read_official_split(p, expected_sha256=source_sha256(p))

    def test_pinned_public_split_snapshot_hash_and_sample_assignments(self):
        snapshot = Path(__file__).absolute().parents[1] / "docs/waveform_v2/scamps/official_split.csv"
        split = read_official_split(snapshot)
        self.assertEqual(len(split["assignments"]), 2800)
        sample = [split["assignments"][f"P{i:06d}"]["split"] for i in range(1, 11)]
        self.assertEqual(sample.count("Train"), 8)
        self.assertEqual(sample.count("Val"), 1)
        self.assertEqual(sample.count("Test"), 1)
        self.assertEqual(split["assignments"]["P000004"]["split"], "Val")
        self.assertEqual(split["assignments"]["P000005"]["split"], "Test")


if __name__ == "__main__":
    unittest.main()
