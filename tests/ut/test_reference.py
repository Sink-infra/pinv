"""Analytical oracle checks and deliberately faulty output probes."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from tests.ut._checks import check_declarations, check_generated_data
from tests.common.tensor_io import generate_suite, store_bytes
from tests.common.reference import (
    CaseData, TensorValue, audit_spectral_gap, materialize_case, pinv_reference,
    reciprocal_singular_values, resolve_tolerances,
)
from tests.st.scripts.verify_result import verify_case_output
from tests.common.cases import CASE_GROUPS

CASES = CASE_GROUPS["correctness"]


def named(name):
    return next(case for case in CASES if case.name == "correctness_" + name)


def input_data(values):
    actual = np.array(values, dtype=np.float32, copy=True)
    return CaseData(TensorValue("float32", actual, actual.astype("<f4")))


def matrix_product(a, b):
    return np.einsum("ik,kj->ij", a, b, optimize=False)


def wrong_diagonal_inverse(a, threshold):
    """Intentionally use a caller's wrong threshold on a diagonal fixture."""
    diagonal = np.diagonal(a, axis1=-2, axis2=-1).astype(np.float64)
    inverse = np.zeros_like(diagonal)
    np.divide(1.0, diagonal, out=inverse, where=np.abs(diagonal) > threshold)
    result = np.zeros(a.shape[:-2] + (a.shape[-1], a.shape[-2]))
    indices = np.arange(diagonal.shape[-1])
    result[..., indices, indices] = inverse
    return result


class CorrectnessTests(unittest.TestCase):
    def test_declarations(self):
        check_declarations(self, CASES, "correctness")

    def test_generated_data(self):
        check_generated_data(self, CASES, "correctness")

    def test_documented_examples(self):
        expected = {
            "documented_rank_deficient": [[0.5, 0.0], [0.0, 0.0]],
            "documented_full_rank": [[0.5, 0.0], [0.0, 2.0]],
        }
        for name, truth in expected.items():
            case = named(name)
            np.testing.assert_array_equal(pinv_reference(case, materialize_case(case)).values, truth)

    def test_zero_and_negative_scalar(self):
        for name in ("zero_scalar", "zero_batched"):
            case = named(name)
            with np.errstate(all="raise"):
                result = pinv_reference(case, materialize_case(case))
            np.testing.assert_array_equal(result.values, np.zeros(result.values.shape))
        case = named("negative_scalar")
        np.testing.assert_array_equal(pinv_reference(case, materialize_case(case)).values, [[-0.5]])

    def test_constant_rank_one_analytic_result(self):
        case = named("constant_rank_one")
        result = pinv_reference(case, materialize_case(case))
        m, n = case.a_shape
        truth = np.full((n, m), 1.0 / (case.scale * m * n))
        np.testing.assert_allclose(result.values, truth, atol=1e-15, rtol=1e-13)

    def test_row_and_column_vectors_analytic_result(self):
        for shape in ((1, 4), (4, 1)):
            case = replace(named("dense_square"), a_shape=shape)
            data = input_data(np.array((1.0, -2.0, 3.0, -4.0)).reshape(shape))
            truth = data.a.values.T.astype(np.float64) / 30.0
            np.testing.assert_allclose(pinv_reference(case, data).values, truth, atol=1e-15, rtol=1e-13)

    def test_dependent_rows_analytic_result(self):
        case = replace(named("dense_square"), a_shape=(2, 2))
        data = input_data([[1.0, 2.0], [2.0, 4.0]])
        result = pinv_reference(case, data)
        self.assertTrue(audit_spectral_gap(result)["passed"])
        np.testing.assert_allclose(result.values, data.a.values / 25.0, atol=1e-9, rtol=1e-7)

    def test_default_tolerance_rules_use_float32_epsilon_and_max_dimension(self):
        case = replace(named("dense_tall"), a_shape=(5, 13))
        epsilon = float(np.finfo(np.float32).eps)
        self.assertEqual(resolve_tolerances(case), (0.0, 13 * epsilon))
        self.assertEqual(resolve_tolerances(replace(case, atol=0.125)), (0.125, 0.0))
        self.assertEqual(resolve_tolerances(replace(case, atol=0.125, rtol=0.25)), (0.125, 0.25))
        self.assertEqual(resolve_tolerances(replace(case, rtol=0.0)), (0.0, 0.0))

    def test_strict_cutoff_and_masked_zero_division(self):
        # Positive equality is outside the device-input gap; probe only the helper.
        singular = np.array((4.0, 0.5, 0.25, 0.0))
        with np.errstate(all="raise"):
            np.testing.assert_array_equal(reciprocal_singular_values(singular, np.array(0.5)), (0.25, 0, 0, 0))
            np.testing.assert_array_equal(reciprocal_singular_values(singular, np.array(0.0)), (0.25, 2, 4, 0))

    def test_max_threshold_and_batch_local_ranks(self):
        expectations = {
            "atol_dominates": [3], "rtol_dominates": [3], "max_not_min": [2],
            "batch_local_relative_threshold": [3, 3, 3], "batch_mixed_atol_threshold": [2, 3, 0],
        }
        for name, ranks in expectations.items():
            case = named(name)
            result = pinv_reference(case, materialize_case(case))
            self.assertEqual(np.count_nonzero(result.retained, axis=-1).reshape(-1).tolist(), ranks)

    def test_all_discarded_nonzero_data(self):
        for name in ("all_discarded_absolute", "all_discarded_relative"):
            case = named(name)
            data = materialize_case(case)
            self.assertTrue(np.any(data.a.values != 0))
            result = pinv_reference(case, data)
            self.assertFalse(result.retained.any())
            np.testing.assert_array_equal(result.values, np.zeros(result.values.shape))

    def test_transpose_covariance(self):
        for name in ("dense_tall", "dense_wide", "dense_rank_deficient"):
            case = named(name)
            data = materialize_case(case)
            transpose_case = replace(case, a_shape=case.a_shape[::-1])
            expected = pinv_reference(case, data).values.T
            actual = pinv_reference(transpose_case, input_data(data.a.values.T)).values
            np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-11)

    def test_projector_symmetry_idempotence_and_truncation_residual(self):
        for name in ("dense_square", "dense_tall", "dense_wide", "dense_rank_deficient",
                     "rtol_truncation", "repeated_singular_values"):
            with self.subTest(case=name):
                case = named(name)
                data = materialize_case(case)
                result = pinv_reference(case, data)
                a, p = data.a.values.astype(np.float64), result.values
                for projector in (matrix_product(a, p), matrix_product(p, a)):
                    np.testing.assert_allclose(projector, projector.T, atol=1e-11, rtol=1e-10)
                    np.testing.assert_allclose(matrix_product(projector, projector), projector, atol=1e-11, rtol=1e-10)
                np.testing.assert_allclose(matrix_product(matrix_product(p, a), p), p, atol=1e-10, rtol=1e-10)
                # A P A equals the retained approximation, not original A when
                # nonzero singular values were discarded by the contract.
                residual = np.linalg.norm(matrix_product(matrix_product(a, p), a) - a)
                discarded_norm = np.linalg.norm(result.singular_values[~result.retained])
                np.testing.assert_allclose(residual, discarded_norm, atol=1e-11, rtol=1e-7)

    def _verify_fault(self, case, wrong_output):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "suite"
            generate_suite(root, (case,))
            actual = root / "wrong.bin"
            np.asarray(wrong_output, dtype="<f4").tofile(actual)
            self.assertFalse(verify_case_output(root / case.name, actual).passed)

    def test_verifier_detects_ignored_nonzero_truncation(self):
        case = named("default_discards_nonzero")
        a = materialize_case(case).a.values.T.astype(np.float64)
        wrong = np.zeros_like(a)
        np.divide(1.0, a, out=wrong, where=a != 0)
        self._verify_fault(case, wrong)

    def test_verifier_detects_batch_global_smax(self):
        case = named("batch_local_relative_threshold")
        a = materialize_case(case).a.values
        threshold = np.max(np.abs(a)) * float(np.float32(case.rtol))
        self._verify_fault(case, wrong_diagonal_inverse(a, threshold))

    def test_default_rtol_catalog_detects_eps_min_and_single_axis_rules(self):
        cases = tuple(case for case in CASES if "default_rtol_max_dimension" in case.covers)
        self.assertEqual({case.a_shape for case in cases}, {(512, 3), (3, 512)})
        epsilon = float(np.finfo(np.float32).eps)
        detected = {rule: False for rule in ("eps", "min", "m", "n")}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "suite"
            generate_suite(root, cases)
            for case in cases:
                data = materialize_case(case)
                target = pinv_reference(case, data)
                self.assertTrue(audit_spectral_gap(target)["passed"])
                self.assertEqual(int(np.count_nonzero(target.retained)), 2)
                m, n = case.a_shape
                for rule, factor in (("eps", 1), ("min", min(m, n)), ("m", m), ("n", n)):
                    wrong = pinv_reference(replace(case, rtol=factor * epsilon), data).values
                    actual = root / "wrong.bin"
                    wrong.astype("<f4").tofile(actual)
                    detected[rule] |= not verify_case_output(root / case.name, actual).passed
        self.assertTrue(all(detected.values()), detected)

    def test_large_dense_catalog_detects_elementwise_reciprocal(self):
        cases = tuple(case for case in CASES if "large_dense" in case.covers)
        self.assertTrue(cases)
        self.assertTrue(any(case.a_shape[0] > case.a_shape[1] for case in cases))
        self.assertTrue(any(case.a_shape[0] < case.a_shape[1] for case in cases))
        self.assertTrue(any(case.rank is not None for case in cases))
        for case in cases:
            with self.subTest(case=case.name):
                self.assertGreaterEqual(min(case.a_shape), 64)
                a = materialize_case(case).a.values.T.astype(np.float64)
                wrong = np.divide(1.0, a, out=np.zeros_like(a), where=a != 0)
                self._verify_fault(case, wrong)

    def test_dense_max_batch_case_detects_diagonal_only_algorithm(self):
        case = named("batch_64_dense_mixed_ranks")
        self.assertEqual(case.a_shape[0], 64)
        a = materialize_case(case).a.values
        diagonal = np.diagonal(a, axis1=-2, axis2=-1).astype(np.float64)
        inverse = np.divide(1.0, diagonal, out=np.zeros_like(diagonal), where=diagonal != 0)
        wrong = np.zeros(case.a_shape[:-2] + case.a_shape[-2:][::-1])
        indices = np.arange(diagonal.shape[-1])
        wrong[..., indices, indices] = inverse
        self._verify_fault(case, wrong)

    def test_verifier_detects_max_replaced_with_sum_or_min(self):
        for name, operation in (("atol_dominates", "sum"), ("rtol_dominates", "sum"), ("max_not_min", "min")):
            case = named(name)
            a = materialize_case(case).a.values
            absolute, relative = resolve_tolerances(case)
            relative *= np.max(np.abs(a))
            wrong_threshold = absolute + relative if operation == "sum" else min(absolute, relative)
            self._verify_fault(case, wrong_diagonal_inverse(a, wrong_threshold))

    def test_verifier_detects_zero_output_on_nonzero_fringe_case(self):
        case = named("condition_100")
        self._verify_fault(case, np.zeros(case.a_shape[::-1]))

    def test_elementwise_budget_detects_one_outlier_and_uses_reference_magnitude(self):
        case = named("documented_full_rank")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "suite"
            generate_suite(root, (case,))
            folder, actual = root / case.name, root / "actual.bin"
            truth = np.array([[0.5, 0.0], [0.0, 2.0]], dtype="<f4")
            close = truth.copy()
            close[0, 0] += np.float32(0.000125)  # >atol alone, <=atol+rtol*0.5.
            close[0, 1] = np.float32(0.00009)
            close.tofile(actual)
            self.assertTrue(verify_case_output(folder, actual).passed)
            close[0, 1] = np.float32(0.000101)
            close.tofile(actual)
            result = verify_case_output(folder, actual)
            self.assertFalse(result.passed)
            self.assertEqual(result.mismatch_count, 1)
            self.assertEqual(result.first_mismatch, (0, 1))

    def test_verifier_rejects_nonfinite_output_and_wrong_byte_count(self):
        case = named("documented_full_rank")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "suite"
            generate_suite(root, (case,))
            folder, actual = root / case.name, root / "actual.bin"
            for value in (np.nan, np.inf, -np.inf):
                np.full((2, 2), value, dtype="<f4").tofile(actual)
                result = verify_case_output(folder, actual)
                self.assertFalse(result.passed)
                self.assertEqual(result.nonfinite_count, 4)
                self.assertEqual(result.mismatch_count, 4)
                json.dumps(result.__dict__, allow_nan=False)
            for payload in (b"", b"\x00" * 15, b"\x00" * 17, b"\x00" * 20):
                store_bytes(actual, payload)
                with self.assertRaisesRegex(ValueError, "bytes"):
                    verify_case_output(folder, actual)

    def test_verifier_refuses_looser_or_invalid_budgets(self):
        for value in (0.001, -1.0, np.inf, np.nan):
            for parameter in ("atol", "rtol"):
                with self.subTest(value=value, parameter=parameter):
                    with self.assertRaisesRegex(ValueError, "comparison"):
                        verify_case_output(Path("unused"), Path("unused"), **{parameter: value})

    def test_verifier_detects_reference_corruption(self):
        case = named("documented_full_rank")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "suite"
            generate_suite(root, (case,))
            folder = root / case.name
            reference = folder / "reference_a_pinv_fp64.bin"
            content = bytearray(reference.read_bytes())
            content[0] ^= 1
            store_bytes(reference, content)
            with self.assertRaisesRegex(ValueError, "SHA256"):
                verify_case_output(folder, root / "actual.bin")

    def test_oracle_uses_materialized_float32_not_recipe_values(self):
        case = replace(named("documented_full_rank"), spectrum=(1.0, 0.1))
        result = pinv_reference(case, materialize_case(case))
        self.assertEqual(result.values[1, 1], 1.0 / float(np.float32(0.1)))
        self.assertNotEqual(result.values[1, 1], 10.0)

    def test_cli_generation_and_verification_exit_codes(self):
        case = named("documented_full_rank")
        project = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as temporary:
            root, actual = Path(temporary) / "suite", Path(temporary) / "actual.bin"
            command = [sys.executable, "-m", "tests.st.scripts.gen_data", "--suite", "correctness", "--output", str(root), "--case", case.name]
            generated = subprocess.run(command, cwd=project, capture_output=True, text=True)
            self.assertEqual(generated.returncode, 0, generated.stderr)
            duplicate = subprocess.run(command, cwd=project, capture_output=True, text=True)
            self.assertEqual(duplicate.returncode, 2)
            verify = [sys.executable, "-m", "tests.st.scripts.verify_result", "--case-dir", str(root / case.name), "--actual", str(actual)]
            np.array([[0.5, 0.0], [0.0, 2.0]], dtype="<f4").tofile(actual)
            passed = subprocess.run(verify, cwd=project, capture_output=True, text=True)
            self.assertEqual(passed.returncode, 0, passed.stderr)
            self.assertTrue(json.loads(passed.stdout)["passed"])
            np.zeros((2, 2), dtype="<f4").tofile(actual)
            failed = subprocess.run(verify, cwd=project, capture_output=True, text=True)
            self.assertEqual(failed.returncode, 1)
            store_bytes(actual, b'invalid')
            invalid = subprocess.run(verify, cwd=project, capture_output=True, text=True)
            self.assertEqual(invalid.returncode, 2)


if __name__ == "__main__":
    unittest.main()
