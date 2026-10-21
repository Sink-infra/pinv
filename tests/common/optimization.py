"""Separate new JIT gains from matching exported-package gains."""
from tests.common.measurements import require_same_environment
from tests.common.official_gains import local_case

MOCK_NAMES = {f"mock_{number:02d}" for number in range(1, 13)}
OFFICIAL_GAIN_THRESHOLD_PCT = 10.0
WORKLOAD_FIELDS = ("shape", "input_sha256", "reference_sha256", "attributes", "output_init", "device")


def _report(report, backend):
    if report.get("backend") != backend or report.get("suite") != "mock" or report.get("passed") is not True:
        raise ValueError("a passing native Mock report is required: " + backend)
    if {r["name"] for r in report["cases"]} != MOCK_NAMES:
        raise ValueError("all twelve Mock fixtures are required")
    if backend == "jit":
        build = report["build"]
        for row in report["cases"]:
            identity = row["provenance"]
            if (identity != build["libraries"][identity["strategy"]] or
                    identity["dsl_sha256"] != build["dsl_sha256"]):
                raise ValueError("measured JIT source/library differs from its build")


def _pair(before, after):
    for field in ("warmup", "repeats", "profiler_backend", "output_init"):
        if field not in before or before[field] != after.get(field):
            raise ValueError("A/B timing protocol differs: " + field)
    results = []
    for name in sorted(MOCK_NAMES):
        rows = [r for report in (before, after) for r in report["cases"] if r["name"] == name]
        for field in WORKLOAD_FIELDS:
            if rows[0].get(field) is None or any(r.get(field) != rows[0][field] for r in rows):
                raise ValueError("A/B workload differs: " + name + " " + field)
        require_same_environment(rows)
        a, b = local_case(before, name), local_case(after, name)
        if len(a["process_medians_us"]) != len(b["process_medians_us"]):
            raise ValueError("A/B process counts differ")
        gains = [100 * (1 - new / old) for old, new in
                 zip(a["process_medians_us"], b["process_medians_us"])]
        results.append({"name": name, "baseline": a, "candidate": b,
                        "gain_pct": 100 * (1 - b["median_us"] / a["median_us"]),
                        "process_gains_pct": gains,
                        "stable_new_gain": a["steady"] and b["steady"] and min(gains) > 0,
                        "within_limit": b["median_us"] <= a["median_us"] * 1.05,
                        "steady": a["steady"] and b["steady"]})
    return results


def qualify_jit(baseline, candidate):
    """Run before conversion. A historical-reference gain cannot qualify this round."""
    for report in (baseline, candidate):
        _report(report, "jit")
    if baseline["build"]["dsl_sha256"] == candidate["build"]["dsl_sha256"]:
        raise ValueError("unchanged DSL cannot establish a new optimization")
    rows = _pair(baseline, candidate)
    planner_changed = baseline["build"]["planner_sha256"] != candidate["build"]["planner_sha256"]
    for row in rows:
        old = next(r["provenance"] for r in baseline["cases"] if r["name"] == row["name"])
        new = next(r["provenance"] for r in candidate["cases"] if r["name"] == row["name"])
        row["changed_path"] = planner_changed or any(old[k] != new[k] for k in ("strategy", "source_sha256"))
        row["stable_new_gain"] = row["stable_new_gain"] and row["changed_path"]
    gained = [r["name"] for r in rows if r["stable_new_gain"]]
    return {"stage": "new_jit_benefit", "passed": bool(gained) and
            all(r["steady"] and r["within_limit"] for r in rows), "cases": rows,
            "new_gain_cases": gained, "baseline_dsl_sha256": baseline["build"]["dsl_sha256"],
            "candidate_dsl_sha256": candidate["build"]["dsl_sha256"],
            "scope": "current retained baseline; complete Task; no package or official benefit implied"}


def _conversion(jit, cann, manifest):
    if manifest["dsl_sha256"] != jit["build"]["dsl_sha256"]:
        raise ValueError("conversion belongs to a different DSL")
    if not manifest["files_sha256"] or manifest["files_sha256"] != cann["build"]["layout_files"]:
        raise ValueError("CANN compiled sources differ from the conversion manifest")
    if jit["build"]["sdk"] != cann["build"]["sdk"]:
        raise ValueError("JIT and CANN used different SDKs")
    for row in jit["cases"]:
        identity = row["provenance"]
        strategy = manifest["strategies"][identity["strategy"]]
        if (strategy["jit_library_sha256"] != identity["library_sha256"] or
                strategy["jit_source_sha256"] != identity["source_sha256"]):
            raise ValueError("export did not use the measured JIT binary/source")
        peers = [r for r in cann["cases"] if r["name"] == row["name"]]
        for field in WORKLOAD_FIELDS:
            if any(r.get(field) != row[field] for r in peers):
                raise ValueError("JIT/CANN workload differs: " + field)
        require_same_environment([row] + peers)


def qualify_package(baseline_jit, candidate_jit, baseline_cann, candidate_cann,
                    baseline_manifest, candidate_manifest):
    """Only a same-Case new gain in both paths supports an official submission."""
    jit = qualify_jit(baseline_jit, candidate_jit)
    if not jit["passed"]:
        raise ValueError("new JIT benefit must pass before CANN-package qualification")
    for j, c, m in ((baseline_jit, baseline_cann, baseline_manifest),
                    (candidate_jit, candidate_cann, candidate_manifest)):
        _report(c, "ascend")
        _conversion(j, c, m)
    if baseline_cann["build"]["source_sha256"] == candidate_cann["build"]["source_sha256"]:
        raise ValueError("unchanged CANN source cannot establish this round's benefit")
    rows = _pair(baseline_cann, candidate_cann)
    common = [r["name"] for r in rows if r["stable_new_gain"] and r["name"] in jit["new_gain_cases"]]
    passed = bool(common) and all(r["steady"] and r["within_limit"] for r in rows)
    jit_gains = {r["name"]: r["gain_pct"] for r in jit["cases"]}
    official = [r["name"] for r in rows if r["name"] in common and
                min(r["gain_pct"], jit_gains[r["name"]]) > OFFICIAL_GAIN_THRESHOLD_PCT]
    return {"stage": "new_jit_and_cann_benefit", "passed": passed,
            "official_submission_ready": passed and bool(official),
            "official_gain_threshold_pct": OFFICIAL_GAIN_THRESHOLD_PCT,
            "official_gain_cases": official, "jit": jit, "cann_cases": rows,
            "new_gain_cases": common,
            "baseline_source_sha256": baseline_cann["build"]["source_sha256"],
            "candidate_source_sha256": candidate_cann["build"]["source_sha256"],
            "scope": "new paired local benefit; official submission also requires a same-Case gain above 10% in both paths against the current retained baseline, plus CAC qualification"}
