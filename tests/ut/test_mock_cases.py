"""Mock suite checks for fixture adjustments and independent references."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.common.mock_cases import generate


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

    def test_fixed_first_element_cannot_be_changed(self):
        case = self.data()
        case["fixture"]["batch_adjustments"] = [[{"row": 0, "column": 0, "value_hex": "3f800000"}], []]
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "invalid input adjustment"):
                generate(Path(temporary) / "data", [case])


if __name__ == "__main__":
    unittest.main()
