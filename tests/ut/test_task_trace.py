import csv
from pathlib import Path
import tempfile
import unittest

from tests.common.task_trace import complete_task_records


class CompleteTaskTraceTests(unittest.TestCase):
    def capture(self, root, *, raw_duration=18, raw_name="N/A", task_id="7", duplicate=False):
        summary = {"Device_id": "0", "Stream ID": "3", "Task ID": "7", "Op Name": "LinalgPinv_kernel",
                   "Task Start Time(us)": "1000", "Task Duration(us)": "20", "Task Type": "AI_VECTOR_CORE"}
        raw = {"Device_id": "0", "stream_id": "3", "task_id": task_id, "kernel_name": raw_name,
               "task_start(us)": "1001", "task_time(us)": str(raw_duration), "kernel_type": "MIX_AIV"}
        for name, rows in (("op_summary.csv", [summary]), ("task_time.csv", [raw, raw] if duplicate else [raw])):
            with (root / name).open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)

    def test_mixed_vector_body_is_not_complete_task(self):
        with tempfile.TemporaryDirectory() as d:
            self.capture(Path(d))
            row, = complete_task_records(d, kernel_prefix="LinalgPinv")
            self.assertEqual((row["duration_us"], row["raw_duration_us"]), (20, 18))
            self.assertEqual((row["prefix_us"], row["suffix_us"]), (1, 1))

    def test_raw_only_capture_cannot_claim_complete_task(self):
        with tempfile.TemporaryDirectory() as d:
            self.capture(Path(d)); (Path(d) / "op_summary.csv").unlink()
            with self.assertRaises(ValueError):
                complete_task_records(d, kernel_prefix="LinalgPinv")

    def test_identity_foreign_kernel_and_duplicate_are_rejected(self):
        for kwargs in ({"task_id": "8"}, {"raw_name": "OtherKernel"}, {"duplicate": True}):
            with self.subTest(kwargs=kwargs), tempfile.TemporaryDirectory() as d:
                self.capture(Path(d), **kwargs)
                with self.assertRaises(ValueError):
                    complete_task_records(d, kernel_prefix="LinalgPinv")

    def test_raw_span_outside_operation_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            self.capture(Path(d), raw_duration=30)
            with self.assertRaises(ValueError):
                complete_task_records(d, kernel_prefix="LinalgPinv")
