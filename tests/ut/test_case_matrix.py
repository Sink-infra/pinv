"""Case document types, suite boundaries, and CLI selection."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tests.common.cases import CASE_GROUPS, load_cases, load_document
from tests.common.tensor_io import store_text


class CaseMatrixTests(unittest.TestCase):
    def test_case_counts_and_tuple_attributes(self):
        self.assertEqual({suite: len(cases) for suite, cases in CASE_GROUPS.items()},
                         {"coverage": 34, "correctness": 43})
        self.assertEqual(len(load_document("mock")["cases"]), 12)
        nonscalar = []
        for cases in CASE_GROUPS.values():
            for case in cases:
                for field in ("a_shape", "batch_ranks", "batch_scales", "covers"):
                    self.assertIsInstance(getattr(case, field), tuple)
                if case.spectrum is not None:
                    self.assertIsInstance(case.spectrum, tuple)
                for field in ("atol", "rtol"):
                    value = getattr(case, field)
                    if isinstance(value, tuple):
                        nonscalar.append(case)
                        self.assertEqual(case.expectation, "reject")
        self.assertTrue(nonscalar)

    def test_category_and_duplicate_names_are_rejected(self):
        original = load_document("coverage")
        documents = [
            {**original, "category": "correctness"},
            {**original, "cases": [original["cases"][0], original["cases"][0]]},
            {**original, "cases": [{**original["cases"][0], "category": "correctness"}]},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for document in documents:
                store_text(root / "coverage.json", json.dumps(document))
                with self.assertRaises(ValueError):
                    load_cases("coverage", root)

    def test_cli_lists_suites_and_refuses_cross_suite_selection(self):
        project = Path(__file__).resolve().parents[2]
        command = [sys.executable, "-m", "tests.st.scripts.gen_data"]
        for suite, count in (("coverage", 34), ("correctness", 43), ("mock", 12)):
            with self.subTest(suite=suite):
                result = subprocess.run(command + ["--suite", suite, "--list"],
                                        cwd=project, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(len(json.loads(result.stdout)), count)
        for suite, name in (("coverage", "mock_01"), ("mock", CASE_GROUPS["correctness"][0].name)):
            result = subprocess.run(command + ["--suite", suite, "--case", name, "--list"],
                                    cwd=project, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("unknown case names", result.stderr)
        empty = subprocess.run(command + ["--suite", "mock", "--expectation", "reject", "--list"],
                               cwd=project, capture_output=True, text=True)
        self.assertEqual(empty.returncode, 2)


if __name__ == "__main__":
    unittest.main()
