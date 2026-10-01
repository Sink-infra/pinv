"""Shared self-checks for one category; these do not execute the device kernel."""

import json
import math
from pathlib import Path
import tempfile

import numpy as np

from tests.common.tensor_io import audit_catalog, generate_suite, sha256_file
from tests.common.reference import audit_spectral_gap, materialize_case, pinv_reference, validate_case
from tests.st.scripts.verify_result import verify_case_output


def check_declarations(test, cases, category):
    test.assertEqual(audit_catalog(cases), category)
    for case in cases:
        with test.subTest(case=case.name):
            test.assertEqual(case.category, category)
            if case.expectation == "success":
                test.assertEqual(validate_case(case).errors, ())
                audit = audit_spectral_gap(pinv_reference(case, materialize_case(case)))
                test.assertTrue(audit["passed"])
            else:
                test.assertIn(case.expected_error, validate_case(case).errors)


def check_generated_data(test, cases, category):
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / category
        manifest = generate_suite(root, cases)
        test.assertEqual(manifest["category"], category)
        test.assertEqual(manifest["counts"]["total"], len(cases))
        test.assertEqual(manifest["catalog_audit_failures"], [])
        for case, entry in zip(cases, manifest["cases"]):
            with test.subTest(case=case.name):
                folder = root / case.name
                metadata_path = folder / "case.json"
                test.assertEqual(sha256_file(metadata_path), entry["case_metadata_sha256"])
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                test.assertEqual(metadata["input"]["a"]["shape"], list(case.a_shape))
                data = materialize_case(case)
                test.assertEqual((folder / "a.bin").read_bytes(), data.a.storage.tobytes(order="C"))
                for file in metadata["files"].values():
                    path = folder / file["path"]
                    test.assertEqual(path.stat().st_size, file["bytes"])
                    test.assertEqual(sha256_file(path), file["sha256"])
                if case.expectation == "reject":
                    test.assertIsNone(metadata["expected_output"]["shape"])
                    test.assertIsNone(metadata["spectral_audit"])
                    test.assertNotIn("reference_a_pinv", metadata["files"])
                    continue
                test.assertTrue(metadata["spectral_audit"]["passed"])
                policy = metadata["expected_output"]["comparison"]
                test.assertEqual(policy["absolute_tolerance"], 1e-4)
                test.assertEqual(policy["relative_tolerance"], 1e-4)
                reference = np.fromfile(folder / "reference_a_pinv_fp64.bin", dtype="<f8")
                result = pinv_reference(case, data)
                test.assertEqual(reference.size, math.prod(case.a_shape))
                np.testing.assert_array_equal(reference, result.values.ravel())
                # Check float32 output conversion against the unrounded reference.
                actual = folder / "tool_selfcheck_output.bin"
                reference.astype("<f4").tofile(actual)
                test.assertTrue(verify_case_output(folder, actual).passed)
        with test.assertRaises(FileExistsError):
            generate_suite(root, cases)
