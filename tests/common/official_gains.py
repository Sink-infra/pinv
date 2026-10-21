"""Audit per-case gain transfer; qualification and new benefit are separate."""
import math
import statistics

from tests.common.measurements import require_same_environment


def _positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def local_case(report, name):
    rows = [r for r in report["cases"] if r["name"] == name]
    if len(rows) < 3 or {r.get("process") for r in rows} != set(range(1, len(rows) + 1)):
        raise ValueError("three distinct processes are required: " + name)
    require_same_environment(rows)
    medians, samples = [], []
    for row in rows:
        minimum = 5 if max(row["shape"][-2:]) <= 64 else 3
        values = row.get("kernel", {}).get("samples_us", [])
        if row["status"] != "Pass" or len(values) < minimum or not all(map(_positive, values)):
            raise ValueError("missing correct complete-Task samples: " + name)
        if row.get("profiler_backend") != "task":
            raise ValueError("complete Task profiling is required: " + name)
        medians.append(statistics.median(values))
        samples.extend(values)
    centre = statistics.median(samples)
    return {"median_us": statistics.median(medians), "process_medians_us": medians,
            "range_us": [min(samples), max(samples)],
            "mad_us": statistics.median(abs(v - centre) for v in samples),
            "sample_count": len(samples), "steady": all(r.get("steady") is True for r in rows)}


def _official(document):
    record = document.get("record", document)
    if record.get("status") != "passed":
        raise ValueError("a completed, passing official evaluation is required")
    result = {}
    for row in record["cases"]:
        number = row.get("index", row.get("test_point"))
        duration = row.get("timeMicros", row.get("elapsedMicros"))
        if row.get("status") != "Pass" or not _positive(duration) or number in result:
            raise ValueError("invalid or duplicate official case")
        result[number] = duration
    return result


def compare_gains(baseline_local, candidate_local, baseline_official, candidate_official, *, tolerance_pp=3.0):
    """Compare signed percentage-point gains, without requiring private inputs.

    Local A/B must share inputs and hardware. Official inputs are deliberately
    not asserted identical to local fixtures. Missing evidence remains open.
    A matching ratio is calibration evidence, never a causal diagnosis.
    """
    if not _positive(tolerance_pp):
        raise ValueError("positive discrepancy threshold required")
    old, new = _official(baseline_official), _official(candidate_official)
    if set(old) != set(new) or not old:
        raise ValueError("official case sets differ")
    results = []
    for number in sorted(old):
        name = f"mock_{number:02d}"
        row = {"case": number, "name": name, "official_baseline_us": old[number],
               "official_candidate_us": new[number], "official_gain_pct": 100 * (1 - new[number] / old[number])}
        try:
            before = [r for r in baseline_local["cases"] if r["name"] == name]
            after = [r for r in candidate_local["cases"] if r["name"] == name]
            for field in ("shape", "input_sha256", "reference_sha256", "attributes", "output_init", "device"):
                values = before + after
                if not values or values[0].get(field) is None or any(r.get(field) != values[0][field] for r in values):
                    raise ValueError("local A/B workload differs: " + field)
            require_same_environment(before + after)
            a, b = local_case(baseline_local, name), local_case(candidate_local, name)
            gain = 100 * (1 - b["median_us"] / a["median_us"])
            delta = gain - row["official_gain_pct"]
            discrepant = abs(delta) > tolerance_pp + 1e-10
            row.update(local_baseline=a, local_candidate=b, local_gain_pct=gain,
                       gain_gap_pp=delta, new_local_benefit=gain > 0 and a["steady"] and b["steady"],
                       status="needs_reproduction" if discrepant else "within_tolerance")
            if not a["steady"] or not b["steady"]:
                row["status"] = "needs_reproduction"
                row["reason"] = "unstable local measurements cannot support a gain claim"
        except (KeyError, TypeError, ValueError) as error:
            row.update(status="insufficient_evidence", reason=str(error))
        results.append(row)
    return {"metric": "median of process medians, complete Task microseconds",
            "tolerance_pp": tolerance_pp, "cases": results,
            "passed": all(r["status"] == "within_tolerance" for r in results),
            "scope": "per-case calibration audit; does not establish root cause or official input equivalence"}
