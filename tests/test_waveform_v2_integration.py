"""Regressions found while reviewing the supplied additive patch."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rppg_lab.waveform_v2_registry import discover_pure, inventory_pure
from tests.test_waveform_v2_infrastructure import fixture


class IntegrationTests(unittest.TestCase):
    def test_malformed_reference_rows_remain_explicit_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root, 1)
            ref = root / "01-01" / "01-01.json"
            ref.write_text(json.dumps({"/FullPackage": [None, 42]}))
            row = inventory_pure(root)["recordings"][0]
            self.assertEqual(row["reference_clock_status"], "invalid_reference_json:ValueError")
            self.assertIn("no_valid_reference_entries", row["admission_blockers"])
            self.assertFalse(row["allow_hr_scoring"])

    def test_private_paths_rejected_before_filesystem_access(self):
        with patch.object(Path, "is_dir", side_effect=AssertionError("filesystem touched")), \
             patch.object(Path, "is_symlink", side_effect=AssertionError("filesystem touched")):
            for name in ("Dataset", "Dataset/child", "other/S007", "other/S011"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    discover_pure(name)

    def test_symlink_root_refused_before_enumeration(self):
        with patch.object(Path, "is_symlink", return_value=True), \
             patch.object(Path, "iterdir", side_effect=AssertionError("enumerated")):
            with self.assertRaisesRegex(ValueError, "Symlink"):
                discover_pure("synthetic_pure")
