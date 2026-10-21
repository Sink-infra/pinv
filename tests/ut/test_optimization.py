import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from tests.common.optimization import qualify_jit, qualify_package
from tests.st.scripts.accept_optimization import main
from tests.ut.test_official_gains import measured


def reports(value, revision):
    jit = measured(value, numbers=range(1, 13))
    identity = dict(stage="tilelang_jit", strategy="window_cube", dsl_sha256=revision,
                    source_sha256=revision + "-device", library_sha256=revision + "-jit")
    jit.update(suite="mock", backend="jit", passed=True, warmup=1000, repeats=5,
               output_init="runtime", profiler_backend="task",
               build=dict(dsl_sha256=revision, planner_sha256="same-planner", sdk="/cann",
                          libraries={"window_cube": identity}))
    for row in jit["cases"]:
        row.update(backend="jit", provenance=copy.deepcopy(identity))
    cann = copy.deepcopy(jit)
    cann.update(backend="ascend", build=dict(source_sha256=revision + "-cann", sdk="/cann",
                library_sha256=revision + "-cann-binary", layout_files={"op_kernel/kernel.h": revision}))
    for row in cann["cases"]:
        row.update(backend="ascend")
        del row["provenance"]
    manifest = dict(dsl_sha256=revision, files_sha256=cann["build"]["layout_files"],
                    strategies={"window_cube": dict(jit_library_sha256=identity["library_sha256"],
                                                    jit_source_sha256=identity["source_sha256"])})
    return jit, cann, manifest


def duration(report, name, value):
    for row in report["cases"]:
        if row["name"] == name:
            row["kernel"]["samples_us"] = [value] * 5


class OptimizationTests(unittest.TestCase):
    def setUp(self):
        self.old_jit, self.old_cann, self.old_manifest = reports(100, "retained")
        self.new_jit, self.new_cann, self.new_manifest = reports(80, "candidate")

    def package(self):
        return qualify_package(self.old_jit, self.new_jit, self.old_cann, self.new_cann,
                               self.old_manifest, self.new_manifest)

    def test_both_paths_need_new_gain_in_the_same_case(self):
        self.assertTrue(self.package()["passed"])
        for no in range(1, 13):
            duration(self.new_cann, f"mock_{no:02d}", 100)
        self.assertFalse(self.package()["passed"])
        for no in range(1, 13):
            duration(self.new_jit, f"mock_{no:02d}", 100)
        duration(self.new_jit, "mock_03", 80)
        duration(self.new_cann, "mock_09", 80)
        self.assertFalse(self.package()["passed"])
        duration(self.new_cann, "mock_03", 85)
        self.assertEqual(self.package()["new_gain_cases"], ["mock_03"])

    def test_correct_regression_is_not_new_jit_benefit(self):
        self.new_jit, _, _ = reports(103, "candidate")
        self.assertFalse(qualify_jit(self.old_jit, self.new_jit)["passed"])
        with self.assertRaisesRegex(ValueError, "before CANN"):
            self.package()

    def test_stable_small_gain_is_retained_but_cannot_submit(self):
        self.new_jit, self.new_cann, self.new_manifest = reports(97, "candidate")
        report = self.package()
        self.assertTrue(report["passed"])
        self.assertFalse(report["official_submission_ready"])
        self.assertEqual(report["official_gain_cases"], [])

    def test_submission_requires_more_than_ten_percent_in_both_paths_same_case(self):
        self.new_jit, self.new_cann, self.new_manifest = reports(90, "candidate")
        self.assertFalse(self.package()["official_submission_ready"])
        duration(self.new_jit, "mock_03", 89)
        duration(self.new_cann, "mock_09", 89)
        self.assertFalse(self.package()["official_submission_ready"])
        duration(self.new_cann, "mock_03", 89)
        report = self.package()
        self.assertTrue(report["official_submission_ready"])
        self.assertEqual(report["official_gain_cases"], ["mock_03"])

    def test_one_lucky_process_and_unstable_samples_do_not_qualify(self):
        for row in self.new_jit["cases"]:
            if row["process"] == 3:
                row["kernel"]["samples_us"] = [101] * 5
        self.assertFalse(qualify_jit(self.old_jit, self.new_jit)["passed"])
        self.new_jit, _, _ = reports(80, "candidate")
        self.new_jit["cases"][0]["steady"] = False
        self.assertFalse(qualify_jit(self.old_jit, self.new_jit)["passed"])

    def test_changed_unrelated_strategy_cannot_credit_unchanged_path(self):
        for row in self.new_jit["cases"]:
            row["provenance"]["source_sha256"] = "retained-device"
        self.new_jit["build"]["libraries"]["window_cube"]["source_sha256"] = "retained-device"
        self.assertFalse(qualify_jit(self.old_jit, self.new_jit)["passed"])

    def test_conversion_and_binary_identity_cannot_be_mixed(self):
        for field, value in (("dsl_sha256", "stale"), ("files_sha256", {"wrong.h": "wrong"})):
            with self.subTest(field=field):
                saved = copy.deepcopy(self.new_manifest)
                self.new_manifest[field] = value
                with self.assertRaises(ValueError):
                    self.package()
                self.new_manifest = saved
        self.new_manifest["strategies"]["window_cube"]["jit_library_sha256"] = "another-binary"
        with self.assertRaisesRegex(ValueError, "measured JIT"):
            self.package()

    def test_all_fixtures_and_workload_identity_required(self):
        self.new_cann["cases"] = [r for r in self.new_cann["cases"] if r["name"] != "mock_09"]
        with self.assertRaisesRegex(ValueError, "twelve"):
            self.package()
        _, self.new_cann, _ = reports(80, "candidate")
        self.new_cann["cases"][8]["input_sha256"] = "different-fixture"
        with self.assertRaisesRegex(ValueError, "workload"):
            self.package()

    def test_process_order_is_not_process_identity(self):
        self.new_jit["cases"].reverse()
        self.assertTrue(self.package()["passed"])

    def test_cli_requires_prior_jit_acceptance_and_unchanged_raw_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            documents = dict(baseline_jit=self.old_jit, candidate_jit=self.new_jit,
                             baseline_cann=self.old_cann, candidate_cann=self.new_cann,
                             baseline_manifest=self.old_manifest, candidate_manifest=self.new_manifest)
            for key, document in documents.items():
                for index, row in enumerate(document.get("cases", [])):
                    raw = root / f"{key}-{index}.json"
                    raw.write_text(json.dumps(row))
                    row["measurement"] = str(raw)
                (root / (key + ".json")).write_text(json.dumps(document))
            jit_args = ["--stage", "jit", "--baseline-jit", str(root / "baseline_jit.json"),
                        "--candidate-jit", str(root / "candidate_jit.json"), "--output", str(root / "jit.json")]
            package_args = ["--stage", "package", "--jit-acceptance", str(root / "jit.json"),
                            "--output", str(root / "package.json")]
            for key in documents:
                package_args += ["--" + key.replace("_", "-"), str(root / (key + ".json"))]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(jit_args), 0)
                self.assertEqual(main(package_args), 0)
                official_args = package_args.copy()
                official_args[1] = "official"
                self.assertEqual(main(official_args), 0)
                raw = Path(self.new_cann["cases"][0]["measurement"])
                raw.write_text(raw.read_text().replace('"Pass"', '"Wrong Answer"'))
                self.assertEqual(main(package_args), 1)
            self.assertIn("raw measurement", json.loads((root / "package.json").read_text())["reason"])

    def test_official_cli_rejects_retained_small_gain_with_nonzero_exit(self):
        self.new_jit, self.new_cann, self.new_manifest = reports(97, "candidate")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            documents = dict(baseline_jit=self.old_jit, candidate_jit=self.new_jit,
                             baseline_cann=self.old_cann, candidate_cann=self.new_cann,
                             baseline_manifest=self.old_manifest, candidate_manifest=self.new_manifest)
            for key, document in documents.items():
                for index, row in enumerate(document.get("cases", [])):
                    raw = root / f"{key}-{index}.json"
                    raw.write_text(json.dumps(row))
                    row["measurement"] = str(raw)
                (root / (key + ".json")).write_text(json.dumps(document))
            args = ["--stage", "official", "--jit-acceptance", str(root / "jit.json"),
                    "--output", str(root / "official.json")]
            for key in documents:
                args += ["--" + key.replace("_", "-"), str(root / (key + ".json"))]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--stage", "jit", "--baseline-jit", str(root / "baseline_jit.json"),
                                       "--candidate-jit", str(root / "candidate_jit.json"),
                                       "--output", str(root / "jit.json")]), 0)
                self.assertEqual(main(args), 1)
            result = json.loads((root / "official.json").read_text())
            self.assertTrue(result["local_benefit_passed"])
            self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
