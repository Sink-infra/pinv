"""Mock suite checks for fixture adjustments and independent references."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.common.mock_cases import generate
from tests.common.mock_fixture import diagonal_similarity, inverse_block_matrix


class MockSuiteTest(unittest.TestCase):
    def data(self):
        return {"number": 1, "shape": [2, 16, 16], "dtype": "float32",
                "fixture": {"first_word_hex": "40d09e4a", "first_batch_positive_diagonal": False,
                            "batch_conditions": [96, 200], "inverse_near_zero_counts": [0, 0]}}

    def test_fixture_adjustment_has_an_independent_reference(self):
        case = self.data()
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first"
            second = Path(temporary) / "second"
            initial = generate(first, [case])
            a = np.load(first / initial[0]["input"], allow_pickle=False)
            value = np.nextafter(a[1, 1, 2], np.float32(np.inf))
            case["fixture"]["batch_adjustments"] = [
                [], [{"row": 1, "column": 2, "value_hex": format(int(value.view(np.uint32)), "08x")}]]
            changed = generate(second, [case])
            b = np.load(second / changed[0]["input"], allow_pickle=False)
            gold = np.load(second / changed[0]["reference"], allow_pickle=False)
            self.assertEqual(np.count_nonzero(a != b), 1)
            self.assertEqual(b[1, 1, 2], value)
            self.assertNotEqual(initial[0]["input_sha256"], changed[0]["input_sha256"])
            self.assertNotEqual(initial[0]["reference_sha256"], changed[0]["reference_sha256"])
            np.testing.assert_allclose(gold, np.linalg.inv(b.astype(np.float64)), rtol=1e-10, atol=1e-10)

    def test_inverse_block_matrix_has_the_declared_spectrum(self):
        recipe = {"sizes": [3], "regulars": [4.0], "inverse_eigenvalues": [2.0], "coupling": 0.05}
        a = inverse_block_matrix(8, recipe)
        expected = np.eye(8)
        expected[:3, :3] = np.eye(3) / 4 + (2 - 1 / 4) / 3
        expected += 0.05
        self.assertEqual(a.dtype, np.float32)
        self.assertTrue(np.all(np.linalg.eigvalsh(a.astype(np.float64)) > 0))
        np.testing.assert_allclose(np.linalg.inv(a.astype(np.float64)), expected, rtol=2e-7, atol=2e-7)

    def test_dense_inverse_block_is_symmetric_and_nonsingular(self):
        recipe = {"sizes": [3], "regulars": [4.0], "inverse_eigenvalues": [2.0], "coupling": 0.02,
                  "dense_tail": {"size": 4, "minimum": 0.8, "maximum": 1.2, "seed": 11},
                  "entry_floor": 0.0005}
        a = inverse_block_matrix(12, recipe)
        np.testing.assert_allclose(a, a.T, rtol=1e-7, atol=1e-7)
        self.assertTrue(np.all(np.linalg.eigvalsh(a.astype(np.float64)) > 0))
        self.assertTrue(np.all(np.abs(np.linalg.inv(a.astype(np.float64))) > 0.0004))
        for change in ({"dense_tail": {"size": 10, "minimum": 0.8, "maximum": 1.2, "seed": 11}},
                       {"entry_floor": -0.1}, {"entry_floor": 1000.0}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                inverse_block_matrix(12, dict(recipe, **change))

    def test_inverse_block_fixture_preserves_other_batches(self):
        case = self.data()
        with tempfile.TemporaryDirectory() as temporary:
            initial_path = Path(temporary) / "initial"
            changed_path = Path(temporary) / "blocks"
            initial = generate(initial_path, [case])[0]
            case["fixture"]["batch_inverse_blocks"] = [
                None, {"sizes": [3], "regulars": [4.0], "inverse_eigenvalues": [2.0], "coupling": 0.05}]
            changed = generate(changed_path, [case])[0]
            before = np.load(initial_path / initial["input"])
            actual = np.load(changed_path / changed["input"])
            gold = np.load(changed_path / changed["reference"])
            np.testing.assert_array_equal(before[0], actual[0])
            self.assertFalse(np.array_equal(before[1], actual[1]))
            np.testing.assert_allclose(gold, np.linalg.inv(actual.astype(np.float64)), rtol=1e-10, atol=1e-10)

    def test_invalid_inverse_block_constraints(self):
        valid = {"sizes": [3], "regulars": [4.0], "inverse_eigenvalues": [2.0], "coupling": 0.05}
        for values in ({"sizes": []}, {"sizes": [9]}, {"regulars": [0.0]},
                       {"inverse_eigenvalues": [float("nan")]}, {"coupling": -0.1}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                inverse_block_matrix(8, dict(valid, **values))

    def test_similarity_preserves_inverse_relation_and_diagonal(self):
        a = np.array([[3, 1, 0.5], [0.25, 2, 0.5], [0.125, 0.75, 4]], dtype=np.float32)
        d = np.array([1, 2, 0.5])
        b = diagonal_similarity(a, d)
        self.assertEqual(b.dtype, np.float32)
        np.testing.assert_array_equal(np.diag(b), np.diag(a))
        expected = np.linalg.inv(a.astype(np.float64)) * d[:, None] / d[None, :]
        np.testing.assert_allclose(np.linalg.inv(b.astype(np.float64)), expected, rtol=1e-12, atol=1e-12)

    def test_similarity_fixture_has_an_independent_reference(self):
        case = self.data()
        with tempfile.TemporaryDirectory() as temporary:
            before_path = Path(temporary) / "base"
            after_path = Path(temporary) / "similarity"
            initial = generate(before_path, [case])[0]
            diagonal = [1.0, 4.0] + [1.0] * 14
            case["fixture"]["batch_similarity_diagonals"] = [None, diagonal]
            changed = generate(after_path, [case])[0]
            a = np.load(before_path / initial["input"])
            b = np.load(after_path / changed["input"])
            gold = np.load(after_path / changed["reference"])
            np.testing.assert_array_equal(a[0], b[0])
            np.testing.assert_array_equal(np.diag(a[1]), np.diag(b[1]))
            self.assertFalse(np.array_equal(a[1], b[1]))
            np.testing.assert_allclose(gold, np.linalg.inv(b.astype(np.float64)), rtol=1e-10, atol=1e-10)

    def test_similarity_rejects_invalid_diagonals(self):
        a = np.eye(3, dtype=np.float32)
        for diagonal in ([1, 0, 1], [1, -1, 1], [1, float("nan"), 1],
                         [1, float("inf"), 1], [1, 2], [[1, 2, 3]]):
            with self.subTest(diagonal=diagonal), self.assertRaises(ValueError):
                diagonal_similarity(a, diagonal)
        case = self.data()
        case["fixture"]["batch_similarity_diagonals"] = [None]
        with tempfile.TemporaryDirectory() as temporary, self.assertRaisesRegex(ValueError, "similarity batch count"):
            generate(Path(temporary) / "data", [case])

    def test_fixed_first_element_cannot_be_changed(self):
        case = self.data()
        case["fixture"]["batch_adjustments"] = [[{"row": 0, "column": 0, "value_hex": "3f800000"}], []]
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "invalid input adjustment"):
                generate(Path(temporary) / "data", [case])


    def test_output_initial_state_does_not_change_input_or_reference(self):
        case = self.data()
        digests = []
        with tempfile.TemporaryDirectory() as temporary:
            for mode in ("runtime", "nan", "nan-once"):
                case["output_init"] = mode
                manifest = generate(Path(temporary) / mode, [case])
                self.assertEqual(manifest[0]["output_init"], mode)
                digests.append((manifest[0]["input_sha256"], manifest[0]["reference_sha256"]))
        self.assertEqual(len(set(digests)), 1)

    def test_invalid_initial_state_does_not_create_a_fixture(self):
        case = self.data()
        case["output_init"] = "unknown"
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "data"
            with self.assertRaises(ValueError):
                generate(destination, [case])
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
