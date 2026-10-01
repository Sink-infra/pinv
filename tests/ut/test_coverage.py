"""Coverage declaration, binary-generation, and catalog self-checks."""

from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.ut._checks import check_declarations, check_generated_data
from tests.common.tensor_io import _select_cases, generate_suite
from tests.common.reference import (
    audit_spectral_gap, infer_output_shape, materialize_case, pinv_reference, validate_case,
)
from tests.common.cases import CASE_GROUPS

CASES = CASE_GROUPS["coverage"]


class CoverageTests(unittest.TestCase):
    def test_declarations(self):
        check_declarations(self, CASES, "coverage")

    def test_generated_data(self):
        check_generated_data(self, CASES, "coverage")

    def test_shape_inference_swaps_only_matrix_axes(self):
        for case in CASES:
            with self.subTest(case=case.name):
                if case.expectation == "success":
                    self.assertEqual(infer_output_shape(case), case.a_shape[:-2] + case.a_shape[-2:][::-1])
                else:
                    with self.assertRaises(ValueError):
                        infer_output_shape(case)

    def test_batch_limit_applies_to_each_dimension(self):
        case = next(case for case in CASES if "per_dimension_batch_bound" in case.covers)
        self.assertGreater(math.prod(case.a_shape[:-2]), 64)
        self.assertEqual(validate_case(case).errors, ())
        self.assertEqual(validate_case(replace(case, a_shape=(64, 64, 1, 1))).errors, ())

    def test_zero_and_scalar_reject_shapes_are_preserved(self):
        for shape in ((), (4,), (0, 4), (4, 0), (0, 3, 4)):
            case = next(case for case in CASES if case.a_shape == shape)
            with self.subTest(shape=shape):
                self.assertEqual(materialize_case(case).a.values.shape, shape)

    def test_catalog_errors_are_detected_before_creating_output(self):
        first = CASES[0]
        bad_catalogs = (
            (), (first, first), (first, replace(first, category="correctness")),
            (replace(first, name="coverage_../../escape"),),
            (replace(first, expectation="reject", expected_error="A_DTYPE"),),
            (replace(first, expectation="success", a_dtype="float16"),),
            (replace(first, expectation="unknown"),),
        )
        with tempfile.TemporaryDirectory() as temporary:
            for index, cases in enumerate(bad_catalogs):
                with self.subTest(index=index):
                    output = Path(temporary) / str(index)
                    with self.assertRaises(ValueError):
                        generate_suite(output, cases)
                    self.assertFalse(output.exists())

    def test_selection_rejects_typos_and_empty_intersections(self):
        success = CASES[0]
        rejected = next(case for case in CASES if case.expectation == "reject")
        self.assertEqual(_select_cases(CASES, "success", (success.name,)), (success,))
        with self.assertRaisesRegex(ValueError, "unknown"):
            _select_cases(CASES, "all", ("typo",))
        with self.assertRaisesRegex(ValueError, "no cases"):
            _select_cases(CASES, "success", (rejected.name,))

    def test_requested_defaults_and_resolved_attributes_are_separate(self):
        case = CASES[0]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "suite"
            generate_suite(output, (case,))
            metadata = json.loads((output / case.name / "case.json").read_text())
            attributes = metadata["attributes"]
            self.assertIsNone(attributes["requested"]["rtol"])
            self.assertEqual(attributes["host_default_rtol_sentinel"], -1.0)
            self.assertEqual(attributes["resolved"]["rtol"], np.finfo(np.float32).eps)

    def test_materialization_is_repeatable_and_nonzero(self):
        case = next(case for case in CASES if case.input_pattern == "dense")
        first, second = materialize_case(case), materialize_case(case)
        np.testing.assert_array_equal(first.a.storage, second.a.storage)
        self.assertTrue(np.any(first.a.values != 0))

    def test_spectral_audit_rejects_data_outside_the_promised_gap(self):
        # These synthetic spectra exercise catalog checks.
        for spectrum, rtol in (((1.0, 0.005), None), ((1.0, 0.006), 0.01)):
            case = replace(CASES[0], a_shape=(2, 2), spectrum=spectrum, rtol=rtol)
            with self.subTest(spectrum=spectrum):
                result = pinv_reference(case, materialize_case(case))
                with self.assertRaisesRegex(ValueError, "spectral gap"):
                    audit_spectral_gap(result)
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary) / "suite"
                    with self.assertRaises(ValueError):
                        generate_suite(root, (case,))
                    self.assertFalse((root / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
