import copy
import unittest

from tests.common.official_gains import compare_gains


def measured(value, numbers=(3, 9), size=256):
    return {"cases": [dict(name=f"mock_{n:02d}", process=p, shape=[4, size, size], status="Pass",
                           input_sha256=str(n), reference_sha256="reference", attributes={"atol": 0, "rtol": -1},
                           output_init="runtime", device=0, profiler_backend="task", steady=True,
                           environment={"soc": "Ascend910B3", "hostname": "test", "cann_path": "/cann"},
                           kernel={"samples_us": [value - .1, value, value + .1, value, value]})
                      for n in numbers for p in (1, 2, 3)]}


def official(value, numbers=(3, 9)):
    return {"record": {"status": "passed", "cases": [dict(index=n, status="Pass", timeMicros=value) for n in numbers]}}


class OfficialGainTests(unittest.TestCase):
    def test_signed_gain_and_dispersion(self):
        result = compare_gains(measured(100), measured(80), official(200), official(160))
        self.assertTrue(result["passed"])
        row = result["cases"][0]
        self.assertAlmostEqual(row["local_gain_pct"], 20)
        self.assertAlmostEqual(row["gain_gap_pp"], 0)
        self.assertEqual(row["local_candidate"]["sample_count"], 15)
        self.assertEqual(row["local_candidate"]["range_us"], [79.9, 80.1])

    def test_gain_discrepancy_cannot_be_hidden_by_another_case(self):
        candidate = measured(80)
        for row in candidate["cases"]:
            if row["name"] == "mock_09":
                row["kernel"]["samples_us"] = [70] * 5
        report = compare_gains(measured(100), candidate, official(100), official(80))
        self.assertFalse(report["passed"])
        self.assertEqual([r["status"] for r in report["cases"]], ["within_tolerance", "needs_reproduction"])

    def test_regression_is_not_new_benefit_or_submission_progress(self):
        report = compare_gains(measured(100), measured(103), official(100), official(103))
        self.assertTrue(report["passed"])
        self.assertFalse(any(r["new_local_benefit"] for r in report["cases"]))

    def test_three_percentage_points_is_boundary_not_three_percent_of_gain(self):
        for candidate, accepted in ((83, True), (83.1, False), (77, True), (76.9, False)):
            with self.subTest(candidate=candidate):
                report = compare_gains(measured(100), measured(candidate), official(100), official(80))
                self.assertEqual(report["passed"], accepted)

    def test_missing_process_or_small_case_samples_stays_open(self):
        for rows in (measured(80)["cases"][:-1], measured(80, size=64)["cases"]):
            candidate = {"cases": copy.deepcopy(rows)}
            for row in candidate["cases"]:
                row["kernel"]["samples_us"] = [80] * 3
            baseline = measured(100, size=64 if len(rows) == 6 else 256)
            report = compare_gains(baseline, candidate, official(100), official(80))
            self.assertFalse(report["passed"])
            self.assertIn("insufficient_evidence", {r["status"] for r in report["cases"]})

    def test_mismatched_inputs_and_unstable_captures_cannot_pass(self):
        for field, value in (("input_sha256", "different"), ("steady", False)):
            candidate = measured(80)
            candidate["cases"][0][field] = value
            report = compare_gains(measured(100), candidate, official(100), official(80))
            self.assertFalse(report["passed"])


if __name__ == "__main__":
    unittest.main()
