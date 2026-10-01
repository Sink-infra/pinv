"""Device harness classification and profiling sample validation."""

import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tests.common.tensor_io import store_text
from tests.st.scripts.profile_mock_workloads import profile_samples
from tests.st.scripts.run_tests import compare_mock_output, comparison_record, run_case, timing_summary, output_initialization


class DeviceHarnessTests(unittest.TestCase):
    def test_event_samples_require_complete_positive_measurements(self):
        for samples in ([], [1.0], [1.0, 0.0], [1.0, -1.0], [1.0, float("nan")]):
            with self.assertRaises(ValueError):
                timing_summary(samples, 2)
        self.assertEqual(timing_summary([1.0, 3.0], 2)["median_us"], 2.0)

    def test_mean_and_median_describe_distinct_latency_statistics(self):
        summary = timing_summary([1.0, 1.0, 10.0], 3)
        self.assertEqual(summary["mean_us"], 4.0)
        self.assertEqual(summary["median_us"], 1.0)
        self.assertEqual(summary["samples_us"], [1.0, 1.0, 10.0])

    def test_mock_comparison_counts_nonfinite_and_outlier_elements(self):
        gold = np.array([[0.5, 0.0], [0.0, 2.0]], dtype=np.float64)
        with tempfile.TemporaryDirectory() as temporary:
            actual = Path(temporary) / "actual.bin"
            gold.astype("<f4").tofile(actual)
            self.assertTrue(compare_mock_output(actual, gold).passed)
            values = gold.astype("<f4")
            values[0, 1] = np.nan
            values[1, 0] = 0.001
            values.tofile(actual)
            result = compare_mock_output(actual, gold)
            self.assertEqual(result.first_mismatch, (0, 1))
            self.assertEqual(result.nonfinite_count, 1)
            self.assertEqual(result.mismatch_count, 2)
            np.array([1.0], dtype="<f4").tofile(actual)
            with self.assertRaises(ValueError):
                compare_mock_output(actual, gold)

    def test_error_percentage_uses_element_counts_and_half_up_rounding(self):
        from tests.st.scripts.verify_result import VerificationResult
        result = VerificationResult(False, 20000, 3, 0, 1.0, (0,), "mismatch")
        self.assertEqual(comparison_record(result)["output_error_ratio"], "0.02%")

    def test_output_initialization_follows_the_suite_and_explicit_mode(self):
        self.assertEqual(output_initialization("mock"), "runtime")
        for suite in ("coverage", "correctness"):
            self.assertEqual(output_initialization(suite), "nan")
        for suite in ("mock", "coverage", "correctness"):
            for mode in ("runtime", "nan", "nan-once"):
                self.assertEqual(output_initialization(suite, mode), mode)
        with self.assertRaises(ValueError):
            output_initialization("mock", "zero")
        with self.assertRaises(ValueError):
            output_initialization("unknown")

    def capture(self, status="executed", returncode=0, final_bad=False, timeout=False,
                output_init="runtime", reported_init=None):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case = {"name": "mock_01", "shape": (2, 2), "input": root / "input.bin",
                    "atol": 0.0, "rtol": -1.0, "gold": np.eye(2, dtype=np.float64)}
            def process(command, **kwargs):
                if timeout:
                    raise subprocess.TimeoutExpired(command, 1)
                args = dict(zip(command[1::2], command[2::2]))
                native = {"status": status, "api": "aclnnLinalgPinvGetWorkspaceSize",
                          "code": 561002 if status == "Runtime Error" else 0,
                          "event_micros": [1.0, 2.0], "teardown": [],
                          "output_init": args["--output-init"] if reported_init is None else reported_init}
                store_text(Path(args["--result"]), json.dumps(native))
                np.eye(2, dtype="<f4").tofile(args["--output"])
                final = np.full((2, 2), np.nan, dtype="<f4") if final_bad else np.eye(2, dtype="<f4")
                final.tofile(args["--output"] + ".final.bin")
                return subprocess.CompletedProcess(command, returncode)
            with patch("tests.st.scripts.run_tests.subprocess.run", side_effect=process):
                return run_case(case, root, root / "case", repeats=2, profile=False, output_init=output_init)

    def test_device_failure_preserves_api_and_code(self):
        result = self.capture(status="Runtime Error", returncode=1)
        self.assertEqual(result["status"], "Runtime Error")
        self.assertEqual(result["runner"]["code"], 561002)

    def test_success_requires_runner_exit_and_both_results(self):
        self.assertEqual(self.capture()["status"], "Pass")
        self.assertEqual(self.capture(returncode=1)["status"], "Runtime Error")
        self.assertEqual(self.capture(status="Pass")["status"], "Harness Error")
        result = self.capture(final_bad=True)
        self.assertEqual(result["status"], "Wrong Answer")
        self.assertTrue(result["first"]["passed"])
        self.assertFalse(result["final"]["passed"])

    def test_device_mode_is_forwarded_and_acknowledged(self):
        for mode in ("runtime", "nan", "nan-once"):
            result = self.capture(output_init=mode)
            self.assertEqual(result["status"], "Pass")
            self.assertEqual(result["runner"]["output_init"], mode)
            self.assertEqual(result["output_init"], mode)
        self.assertEqual(self.capture(reported_init="nan")["status"], "Harness Error")

    def test_timeout_is_not_a_successful_capture(self):
        self.assertEqual(self.capture(timeout=True)["status"], "TLE")

    def test_profile_requires_the_full_launch_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "op_summary.csv"
            store_text(path, "Op Name,Task Duration(us)\nLinalgPinv_0,50\nLinalgPinv_1,12\nLinalgPinv_2,14\nOther,1\n")
            self.assertEqual(profile_samples(Path(temporary), 1, 2), [12.0, 14.0])
            with self.assertRaises(ValueError):
                profile_samples(Path(temporary), 1, 3)
            store_text(path, "Op Name,Task Duration(us)\nLinalgPinv_0,nan\n")
            with self.assertRaises(ValueError):
                profile_samples(Path(temporary), 0, 1)


if __name__ == "__main__":
    unittest.main()
