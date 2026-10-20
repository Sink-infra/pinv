"""Backend integration contracts; no fake timing is used as NPU evidence."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import io
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from tests.common.cases import CASE_GROUPS
from tests.common.paired import compare_backends
from tests.common.profiling import profile_samples
from tests.st.backends import get_backend
from tests.st.backends import jit
from tests.st.backends.transport import SessionExecutor
from tests.st.scripts import run_tests


class PairedReports(unittest.TestCase):
    def rows(self):
        return [{"name": "mock_01", "process": i, "shape": [16, 16], "status": "Pass", "device": 0,
                 "input_sha256": "input", "reference_sha256": "reference", "output_init": "runtime",
                 "attributes": {"atol": 0, "rtol": -1}, "steady": True,
                 "environment": {"soc": "Ascend910B3", "hostname": "test-host", "cann_path": "/sdk"},
                 "kernel": {"median_us": 5.0, "samples_us": [5.0] * 9}} for i in range(1, 4)]

    def test_three_matching_processes_and_per_case_limit(self):
        a, b = self.rows(), self.rows()
        self.assertTrue(compare_backends(a, b)["performance_accepted"])
        for r in a:
            r["kernel"]["median_us"] = 5.3
        self.assertFalse(compare_backends(a, b)["performance_accepted"])

    def test_inputs_references_attributes_and_hardware_must_match(self):
        for field, value in (("input_sha256", "other"), ("reference_sha256", "other"), ("device", 1),
                             ("attributes", {"atol": 1}), ("output_init", "nan"),
                             ("environment", {"soc": "Ascend910B4", "hostname": "other", "cann_path": "/sdk"})):
            a, b = self.rows(), self.rows()
            a[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                compare_backends(a, b)

    def test_event_timing_and_one_process_cannot_be_performance_acceptance(self):
        a, b = self.rows(), self.rows()
        self.assertFalse(compare_backends(a[:1], b[:1])["performance_accepted"])
        for r in a + b:
            r["event"] = r.pop("kernel")
        self.assertFalse(compare_backends(a, b)["performance_accepted"])

    def test_duplicate_processes_unsteady_or_unsupported_never_pass(self):
        a, b = self.rows(), self.rows()
        a[1]["process"] = 1
        with self.assertRaises(ValueError):
            compare_backends(a, b)
        a = self.rows()
        a[1]["steady"] = False
        self.assertFalse(compare_backends(a, b)["performance_accepted"])
        a[1]["status"] = "Unsupported"
        report = compare_backends(a, b)
        self.assertFalse(report["correctness_passed"])
        self.assertFalse(report["complete"])


class JitBuildIdentity(unittest.TestCase):
    def test_generated_package_cannot_be_labeled_as_original_native_baseline(self):
        from tests.st.scripts.baseline import require_native_build
        from tests.st.scripts.runtime import ROOT, sha256_file, source_files
        files = {p.relative_to(ROOT / 'code').as_posix(): sha256_file(p)
                 for p in source_files()}
        native = self.root / 'native-fixture'
        native.mkdir()
        (native / 'native.cpp').write_text('native source fixture')
        @contextmanager
        def source():
            yield native
        with patch('code_dsl.convert.native_source.native_source', source):
            with self.assertRaisesRegex(ValueError, 'exact pinned Git source'):
                require_native_build({'layout_files': files})
            receipt = {'layout_files': {'native.cpp': sha256_file(native / 'native.cpp')}}
            self.assertEqual(require_native_build(receipt), receipt)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        docs = {}
        for strategy in ("compact", "local", "rectangular", "peer_pipeline"):
            lib, receipt = jit.artifacts(self.root, strategy)
            lib.parent.mkdir(parents=True, exist_ok=True)
            lib.write_bytes(b"a real binary is required on device; this is a unit fixture")
            lib.with_suffix(".cpp").write_text("unit fixture source")
            doc = docs.setdefault(receipt, {"producer": "TileLang JIT", "dsl_sha256": jit.dsl_digest(),
                                            "complete_continuation": True, "artifacts": {}})
            doc["artifacts"][strategy + "_module"] = {
                "library_sha256": hashlib.sha256(lib.read_bytes()).hexdigest(),
                "source_sha256": hashlib.sha256(lib.with_suffix(".cpp").read_bytes()).hexdigest()}
        for path, doc in docs.items():
            path.write_text(json.dumps(doc))
        (self.root / "host-planner.so").write_bytes(b"fixture")
        (self.root / "tests").mkdir()
        (self.root / "tests/build.json").write_text(json.dumps(jit.identity(self.root)))

    def tearDown(self):
        self.temporary.cleanup()

    def test_changed_binary_source_and_receipt_are_rejected(self):
        self.assertEqual(jit.verify(self.root)["backend"], "jit")
        library, receipt = jit.artifacts(self.root, "compact")
        for path in (library, library.with_suffix(".cpp"), receipt):
            before = path.read_bytes()
            path.write_bytes(before + b" ")
            with self.subTest(path=path), self.assertRaises(ValueError):
                jit.verify(self.root)
            path.write_bytes(before)

    def test_diagnostic_continuations_are_rejected(self):
        _, receipt = jit.artifacts(self.root, "peer_pipeline")
        data = json.loads(receipt.read_text())
        data["complete_continuation"] = False
        receipt.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "probes"):
            jit.verify(self.root)

    def test_unported_shapes_never_dispatch_to_ascend(self):
        plan = SimpleNamespace(strategy="window_cube", geometry=SimpleNamespace(m=256, n=256, shape=(256, 256)))
        with self.assertRaises(jit.UnsupportedStrategy):
            jit.select_strategy(plan)
        plan.strategy = "local"
        plan.geometry.n = 96
        with self.assertRaises(jit.UnsupportedStrategy):
            jit.select_strategy(plan)
        self.assertIn("tests.st.backends.jit_runner", get_backend("jit").command(self.root))


class SharedSuiteIntegration(unittest.TestCase):
    def capture(self, root, unsupported=False, allow=False, project_output=False):
        selected = list(CASE_GROUPS["correctness"][:2])
        seen = []
        def implementation(name):
            return SimpleNamespace(command=lambda build: ["fixture-" + name],
                                   verify=lambda build: {"backend": name})
        def execute(command, **settings):
            name = command[0].split("-", 1)[1]
            args = dict(zip(command[1::2], command[2::2]))
            input_path = Path(args["--input"])
            seen.append((name, input_path, args["--output-init"]))
            report = {"status": "executed", "event_micros": [1.0, 1.1],
                      "output_init": args["--output-init"], "teardown": []}
            if unsupported and name == "jit" and input_path.parent.name == selected[0].name:
                report["status"] = "Unsupported"
            else:
                gold = np.fromfile(input_path.parent / "reference_a_pinv_fp64.bin", dtype="<f8")
                for output in (args["--output"], args["--output"] + ".final.bin"):
                    gold.astype("<f4").tofile(output)
            Path(args["--result"]).write_text(json.dumps(report))
            return subprocess.CompletedProcess(command, 0 if report["status"] == "executed" else 1)
        @contextmanager
        def transport(*args):
            yield execute, {"transport": "unit fixture"}
        destination = root / "code_dsl/results/run" if project_output else root / "results"
        argv = ["--backend", "both", "--suite", "correctness", "--no-build", "--warmup", "0",
                "--repeats", "2", "--output", str(destination)]
        for case in selected:
            argv += ["--case", case.name]
        if allow:
            argv.append("--allow-unsupported")
        with patch.object(run_tests, "get_backend", side_effect=implementation), patch.object(run_tests, "execution_transport", transport), patch.object(run_tests, "ROOT", root if project_output else run_tests.ROOT):
            code = run_tests.main(argv)
        return code, seen, json.loads((destination / "comparison.json").read_text())

    def test_both_backends_share_data_validation_and_output_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            code, seen, report = self.capture(Path(temporary))
        self.assertEqual(code, 0)
        self.assertEqual([r[0] for r in seen], ["jit", "jit", "ascend", "ascend"])
        self.assertEqual(seen[0][1:], seen[2][1:])
        self.assertEqual(seen[1][1:], seen[3][1:])
        self.assertTrue(report["correctness_passed"])

    def test_unsupported_is_explicit_and_requires_scoped_opt_in(self):
        for allow in (False, True):
            with self.subTest(allow=allow), tempfile.TemporaryDirectory() as temporary:
                code, _, report = self.capture(Path(temporary), unsupported=True, allow=allow)
                self.assertEqual(code, 0 if allow else 1)
                self.assertFalse(report["complete"])
                self.assertFalse(report["correctness_passed"])

    def test_ignored_results_directory_supports_the_same_backend_flow(self):
        with tempfile.TemporaryDirectory() as temporary:
            code, _, report = self.capture(Path(temporary), project_output=True)
            self.assertEqual(code, 0)
            self.assertTrue(report["correctness_passed"])

    def test_one_profiler_parser_handles_both_kernel_names(self):
        for prefix in ("LinalgPinv", "linalg_pinv_tilelang_kernel"):
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "task_time.csv"
                path.write_text("kernel_name,kernel_type,task_time(us)\n" +
                                f"{prefix},KERNEL_AIVEC,12\n{prefix},KERNEL_AIVEC,5\n")
                self.assertEqual(profile_samples(path.parent, 1, 1, kernel_prefix=prefix), [5.0])

    def test_duplicate_device_tasks_cannot_replace_missing_launches(self):
        for prefix in ("LinalgPinv", "linalg_pinv_tilelang_kernel"):
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "task_time.csv"
                path.write_text("kernel_name,kernel_type,task_time(us),Device_id,stream_id,task_id\n" +
                                f"{prefix},KERNEL_AIVEC,12,0,1,2\n{prefix},KERNEL_AIVEC,5,0,1,2\n")
                with self.assertRaisesRegex(ValueError, "duplicate device task"):
                    profile_samples(path.parent, 1, 1, kernel_prefix=prefix)


class DeviceTransportTests(unittest.TestCase):
    def test_package_cli_exposes_both_backend_build_and_run_commands(self):
        from tests.st.scripts.runtime import ROOT
        for command in ("build", "run"):
            result = subprocess.run([sys.executable, "-m", "tests", command, "--help"],
                                    cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--backend", result.stdout)
            self.assertIn("ascend", result.stdout)
            self.assertIn("jit", result.stdout)

    def test_service_transport_preserves_arguments_logs_and_global_subprocess(self):
        run = subprocess.run
        result = SimpleNamespace(returncode=0, stdout="stdout", stderr="stderr")
        session = SimpleNamespace(run=lambda command, **kwargs: (calls.append((command, kwargs)) or result))
        calls, log = [], io.StringIO()
        output = SessionExecutor(session)(["runner"], env={}, cwd="/project", timeout=17,
                                          stdout=log, stderr=subprocess.STDOUT)
        self.assertIs(output, result)
        self.assertEqual(calls, [(["runner"], {"environment": {}, "cwd": "/project", "timeout": 17})])
        self.assertEqual(log.getvalue(), "stdoutstderr")
        self.assertIs(subprocess.run, run)

    def test_service_timeout_uses_the_common_timeout_classification(self):
        def timeout(*args, **kwargs):
            raise TimeoutError("device child timed out")
        with self.assertRaises(subprocess.TimeoutExpired):
            SessionExecutor(SimpleNamespace(run=timeout))(["runner"], timeout=17)


if __name__ == "__main__":
    unittest.main()
