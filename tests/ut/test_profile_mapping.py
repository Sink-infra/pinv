"""Validate operator launch identities and device duration samples."""

import json
from pathlib import Path
import shlex
from subprocess import CompletedProcess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tests.common.tensor_io import store_bytes, store_text
from tests.st.scripts.profile_mock_workloads import profile_samples
from tests.st.scripts.run_tests import run_case


class ProfileMappingTests(unittest.TestCase):
    def sample(self, root, index, duration, *, name="LinalgPinv_kernel", filename="OpBasicInfo.csv"):
        folder = root / "OPPROF" / name / str(index)
        folder.mkdir(parents=True, exist_ok=True)
        store_text(folder / filename, "Op Name,Task Duration(us)\n" + name + "," + str(duration) + "\n")

    def test_numeric_launch_order_and_warmup_exclusion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index in range(12):
                self.sample(root, index, index + 1)
            self.assertEqual(profile_samples(root, 3, 9, "operator"), list(range(4, 13)))

    def test_missing_and_duplicate_launches_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.sample(root, 0, 12)
            with self.assertRaises(ValueError):
                profile_samples(root, 0, 2, "operator")
            self.sample(root, 0, 13, filename="OpBasicInfo_extra.csv")
            with self.assertRaises(ValueError):
                profile_samples(root, 0, 1, "operator")

    def test_foreign_or_multiple_kernel_identities_fail(self):
        for names in (("OtherKernel",), ("LinalgPinv_a", "LinalgPinv_b")):
            with self.subTest(names=names), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                for index, name in enumerate(names):
                    self.sample(root, index, 12, name=name)
                with self.assertRaises(ValueError):
                    profile_samples(root, 0, len(names), "operator")

    def test_invalid_duration_is_rejected(self):
        for duration in ("nan", "inf", 0, -1, "unknown"):
            with self.subTest(duration=duration), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                self.sample(root, 0, duration)
                with self.assertRaises(ValueError):
                    profile_samples(root, 0, 1, "operator")

    def test_launch_index_and_row_count_are_checked(self):
        for index in ("unknown", 2):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                self.sample(root, index, 12)
                with self.assertRaises(ValueError):
                    profile_samples(root, 0, 1, "operator")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.sample(root, 0, 12)
            path = next(root.rglob("OpBasicInfo.csv"))
            store_text(path, path.read_text() + "LinalgPinv_kernel,13\n")
            with self.assertRaises(ValueError):
                profile_samples(root, 0, 1, "operator")

    def test_missing_duration_column_and_backend_are_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.sample(root, 0, 12)
            store_text(next(root.rglob("OpBasicInfo.csv")), "Op Name\nLinalgPinv_kernel\n")
            with self.assertRaises(ValueError):
                profile_samples(root, 0, 1, "operator")
            with self.assertRaises(ValueError):
                profile_samples(root, 0, 1, "unknown")

    def test_native_task_table_excludes_events_and_warmup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store_text(root / "task_time.csv", "kernel_name,kernel_type,task_time(us)\n"
                       "N/A,EVENT_RECORD,0.02\nLinalgPinv_kernel,KERNEL_AIVEC,20\n"
                       "LinalgPinv_kernel,KERNEL_AIVEC,5.2\nLinalgPinv_kernel,KERNEL_AIVEC,5.4\n")
            self.assertEqual(profile_samples(root, 1, 2, "task"), [5.2, 5.4])

    def test_operator_summary_takes_precedence_over_native_table(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store_text(root / "op_summary.csv", "Op Name,Task Duration(us)\nLinalgPinv_kernel,5.2\n")
            store_text(root / "task_time.csv", "kernel_name,kernel_type,task_time(us)\n"
                       "LinalgPinv_kernel,KERNEL_AIVEC,5.2\n")
            self.assertEqual(profile_samples(root, 0, 1, "task"), [5.2])

    def capture(self, incomplete=False, profile_samples_count=None, profiler_replay="kernel", profiler_warmup=5):
        with tempfile.TemporaryDirectory(prefix="pinv profile ") as temporary:
            root = Path(temporary)
            case = {"name": "mock_profile", "shape": (1, 1), "input": root / "a.bin",
                    "atol": 0, "rtol": -1, "gold": np.ones((1, 1), dtype=np.float64)}

            def execute(command, **settings):
                self.assertEqual(command[:2], ["msprof", "op"])
                self.assertIn("--replay-mode=" + profiler_replay, command)
                self.assertIn("--warm-up=" + str(profiler_warmup), command)
                application = next(value.split("=", 1)[1] for value in command if value.startswith("--application="))
                native = shlex.split(application)
                args = dict(zip(native[1::2], native[2::2]))
                self.assertEqual(args["--input"], str(case["input"]))
                self.assertNotIn("--profile", args)
                report = {"status": "executed", "event_micros": [20, 22], "teardown": [],
                          "output_init": args["--output-init"]}
                store_text(Path(args["--result"]), json.dumps(report))
                payload = np.ones(1, dtype="<f4").tobytes()
                store_bytes(Path(args["--output"]), payload)
                store_bytes(Path(args["--output"] + ".final.bin"), payload)
                folder = Path(next(value.split("=", 1)[1] for value in command if value.startswith("--output=")))
                count = 2 if profile_samples_count is None else profile_samples_count
                self.assertIn("--launch-skip-before-match=1", command)
                for index in range(count - 1 if incomplete else count):
                    self.sample(folder, index, 10 + index)
                return CompletedProcess(command, 0)

            with patch("tests.st.scripts.run_tests.subprocess.run", side_effect=execute):
                return run_case(case, root, root / "case", warmup=1, repeats=2, profile=True,
                                profile_samples_count=profile_samples_count, profiler_replay=profiler_replay, profiler_warmup=profiler_warmup)

    def test_operator_capture_keeps_device_and_event_metrics_separate(self):
        report = self.capture()
        self.assertEqual(report["status"], "Pass")
        self.assertEqual(report["kernel"]["samples_us"], [10, 11])
        self.assertEqual(report["event"]["samples_us"], [20, 22])
        self.assertEqual(report["profiler_backend"], "operator")

    def test_application_replay_is_explicit_in_the_record(self):
        report = self.capture(profiler_replay="application")
        self.assertEqual(report["status"], "Pass")
        self.assertEqual(report["profiler_replay"], "application")

    def test_operator_replay_warmup_can_be_disabled(self):
        report = self.capture(profiler_warmup=0)
        self.assertEqual(report["status"], "Pass")
        self.assertEqual(report["profiler_warmup"], 0)

    def test_invalid_operator_replay_warmup_is_rejected(self):
        for value in (-1, 501, "unknown"):
            with self.subTest(value=value):
                self.assertEqual(self.capture(profiler_warmup=value)["status"], "Harness Error")

    def test_profile_subset_has_its_own_sample_count(self):
        report = self.capture(profile_samples_count=1)
        self.assertEqual(report["status"], "Pass")
        self.assertEqual(report["kernel"]["samples_us"], [10])
        self.assertEqual(report["profile_samples"], 1)
        self.assertEqual(len(report["runner"]["event_micros"]), 2)

    def test_profile_cannot_capture_more_iterations_than_execute(self):
        self.assertEqual(self.capture(profile_samples_count=3)["status"], "Harness Error")

    def test_missing_profile_cannot_pass_numeric_validation(self):
        report = self.capture(incomplete=True)
        self.assertTrue(report["first"]["passed"])
        self.assertEqual(report["status"], "Harness Error")


if __name__ == "__main__":
    unittest.main()
