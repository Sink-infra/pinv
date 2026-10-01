"""Regressions for the manifest, metadata, input, and reference integrity chain."""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from tests.common.tensor_io import generate_suite, sha256_file, store_bytes, store_text
from tests.st.scripts.verify_result import verify_case_output
from tests.common.cases import CASE_GROUPS

CASES = CASE_GROUPS["correctness"]


class ArtifactIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "suite"
        self.case = next(case for case in CASES if case.name == "correctness_documented_full_rank")
        generate_suite(self.root, (self.case,))
        self.folder = self.root / self.case.name
        self.metadata_path = self.folder / "case.json"
        self.manifest_path = self.root / "manifest.json"
        self.metadata_bytes = self.metadata_path.read_bytes()
        self.manifest_bytes = self.manifest_path.read_bytes()
        self.input_bytes = (self.folder / "a.bin").read_bytes()
        self.actual = Path(self.temporary.name) / "actual.bin"
        np.array([[0.5, 0.0], [0.0, 2.0]], dtype="<f4").tofile(self.actual)

    def tearDown(self):
        self.temporary.cleanup()

    def _metadata(self):
        return json.loads(self.metadata_bytes)

    def _manifest(self):
        return json.loads(self.manifest_bytes)

    def _store_metadata(self, metadata, *, update_manifest=False):
        store_text(self.metadata_path, json.dumps(metadata), encoding='utf-8')
        if update_manifest:
            manifest = self._manifest()
            manifest["cases"][0]["case_metadata_sha256"] = sha256_file(self.metadata_path)
            store_text(self.manifest_path, json.dumps(manifest), encoding='utf-8')

    def test_valid_fixture_and_relocated_suite(self):
        self.assertTrue(verify_case_output(self.folder, self.actual).passed)
        relocated = Path(self.temporary.name) / "relocated"
        shutil.copytree(self.root, relocated)
        self.assertTrue(verify_case_output(relocated / self.case.name, self.actual).passed)

    def test_input_bit_flip_and_zero_replacement_are_rejected(self):
        flipped = bytearray(self.input_bytes)
        flipped[0] ^= 1
        for payload in (bytes(flipped), bytes(len(self.input_bytes))):
            with self.subTest(payload=payload.hex()):
                store_bytes(self.folder / 'a.bin', payload)
                with self.assertRaisesRegex(ValueError, "input SHA256 mismatch"):
                    verify_case_output(self.folder, self.actual)

    def test_input_truncation_and_extra_byte_are_rejected(self):
        for payload in (self.input_bytes[:-1], self.input_bytes + b"\x00"):
            with self.subTest(size=len(payload)):
                store_bytes(self.folder / 'a.bin', payload)
                with self.assertRaisesRegex(ValueError, "bytes"):
                    verify_case_output(self.folder, self.actual)

    def test_changed_attributes_or_shapes_fail_metadata_hash(self):
        for change in ("rtol", "atol", "input_shape", "output_shape"):
            with self.subTest(change=change):
                metadata = self._metadata()
                if change in ("rtol", "atol"):
                    metadata["attributes"]["requested"][change] = 16.0
                    metadata["attributes"]["resolved"][change] = 16.0
                elif change == "input_shape":
                    metadata["input"]["a"]["shape"] = [1, 4]
                else:
                    metadata["expected_output"]["shape"] = [1, 4]
                self._store_metadata(metadata)
                with self.assertRaisesRegex(ValueError, "metadata SHA256 mismatch"):
                    verify_case_output(self.folder, self.actual)

    def test_parent_manifest_is_required(self):
        self.manifest_path.unlink()
        with self.assertRaises(FileNotFoundError):
            verify_case_output(self.folder, self.actual)

    def test_missing_duplicate_and_foreign_manifest_entries_are_rejected(self):
        entry = self._manifest()["cases"][0]
        for entries in ([], [entry, entry], [{**entry, "metadata": "foreign/case.json"}]):
            with self.subTest(entries=entries):
                manifest = self._manifest()
                manifest["cases"] = entries
                store_text(self.manifest_path, json.dumps(manifest), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, "exactly one"):
                    verify_case_output(self.folder, self.actual)

    def test_manifest_schema_and_case_identity_are_checked(self):
        for change in ("schema_version", "operator", "category", "name", "expectation"):
            with self.subTest(change=change):
                manifest = self._manifest()
                if change == "schema_version":
                    manifest[change] = 999
                elif change in ("operator", "category"):
                    manifest[change] = "foreign"
                else:
                    manifest["cases"][0][change] = "foreign"
                store_text(self.manifest_path, json.dumps(manifest), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, "manifest"):
                    verify_case_output(self.folder, self.actual)

    def test_output_shape_is_checked_even_with_matching_metadata_hash(self):
        for shape in ([1, 4], [2.0, 2], [True, 4]):
            with self.subTest(shape=shape):
                metadata = self._metadata()
                metadata["expected_output"]["shape"] = shape
                self._store_metadata(metadata, update_manifest=True)
                with self.assertRaisesRegex(ValueError, "output shape"):
                    verify_case_output(self.folder, self.actual)

    def test_resolved_attributes_are_checked_even_with_matching_metadata_hash(self):
        metadata = self._metadata()
        metadata["attributes"]["resolved"]["rtol"] = 0.01
        self._store_metadata(metadata, update_manifest=True)
        with self.assertRaisesRegex(ValueError, "resolved attributes"):
            verify_case_output(self.folder, self.actual)

    def test_file_metadata_requires_correct_byte_count_dtype_and_hash(self):
        for change in ("bytes", "logical_dtype", "sha256"):
            with self.subTest(change=change):
                metadata = self._metadata()
                metadata["files"]["a"][change] = {"bytes": 20, "logical_dtype": "float64", "sha256": None}[change]
                self._store_metadata(metadata, update_manifest=True)
                with self.assertRaisesRegex(ValueError, "input file metadata"):
                    verify_case_output(self.folder, self.actual)

    def test_cli_integrity_errors_return_status_two(self):
        project = Path(__file__).resolve().parents[2]
        command = [sys.executable, "-m", "tests.st.scripts.verify_result", "--case-dir", str(self.folder), "--actual", str(self.actual)]
        store_bytes(self.folder / 'a.bin', bytes(len(self.input_bytes)))
        corrupted_input = subprocess.run(command, cwd=project, capture_output=True, text=True)
        self.assertEqual(corrupted_input.returncode, 2)
        self.assertIn("input SHA256 mismatch", corrupted_input.stderr)
        store_bytes(self.folder / 'a.bin', self.input_bytes)
        metadata = self._metadata()
        metadata["attributes"]["requested"]["rtol"] = 16.0
        self._store_metadata(metadata)
        corrupted_metadata = subprocess.run(command, cwd=project, capture_output=True, text=True)
        self.assertEqual(corrupted_metadata.returncode, 2)
        self.assertIn("metadata SHA256 mismatch", corrupted_metadata.stderr)


if __name__ == "__main__":
    unittest.main()
