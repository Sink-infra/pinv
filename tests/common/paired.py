"""Compare matching device measurements without mixing timing domains."""
import statistics

from tests.common.measurements import require_same_environment


def compare_backends(jit, ascend, *, max_slowdown=0.05):
    if not 0 <= max_slowdown <= 1:
        raise ValueError("invalid slowdown limit")
    def group(rows):
        result = {}
        for row in rows:
            result.setdefault(row["name"], []).append(row)
        return result
    left, right = group(jit), group(ascend)
    if set(left) != set(right):
        raise ValueError("backends did not select the same workloads")
    results = []
    for name, lrows in left.items():
        rrows = right[name]
        if len(lrows) != len(rrows):
            raise ValueError("backend process counts differ")
        for rows in (lrows, rrows):
            if {r.get("process") for r in rows} != set(range(1, len(rows) + 1)):
                raise ValueError("backend measurements do not identify independent processes")
        row = {"name": name, "processes": len(lrows)}
        samples = lrows + rrows
        if any(r["status"] != "Pass" for r in samples):
            row.update(status="Unsupported" if any(r["status"] == "Unsupported" for r in samples) else "Failed")
        else:
            for field in ("shape", "input_sha256", "reference_sha256", "attributes", "output_init", "device"):
                if field not in samples[0] or samples[0][field] is None or any(r.get(field) != samples[0][field] for r in samples):
                    raise ValueError(f"backend workloads differ: {name} {field}")
            row.update(status="Pass", input_sha256=samples[0]["input_sha256"],
                       reference_sha256=samples[0]["reference_sha256"])
            if all("kernel" in r for r in samples):
                require_same_environment(samples)
                row.update(jit_us=statistics.median(r["kernel"]["median_us"] for r in lrows),
                           ascend_us=statistics.median(r["kernel"]["median_us"] for r in rrows))
                row["jit_over_ascend"] = row["jit_us"] / row["ascend_us"]
                row["steady"] = all(r.get("steady") is True for r in samples)
                row["within_limit"] = row["jit_over_ascend"] <= 1 + max_slowdown
                row["performance_accepted"] = len(lrows) >= 3 and row["steady"] and row["within_limit"]
        results.append(row)
    measured = [r for r in results if "jit_over_ascend" in r]
    return {"metric": "device kernel Task Duration; median of process medians",
            "scope": "same-host backend comparison; no new official evaluation",
            "max_slowdown": max_slowdown, "complete": all(r["status"] == "Pass" for r in results),
            "cases": results, "correctness_passed": bool(results) and all(r["status"] == "Pass" for r in results),
            "scoped_performance_accepted": bool(measured) and all(r["status"] in ("Pass", "Unsupported") for r in results)
                                           and all(r["performance_accepted"] for r in measured),
            "performance_accepted": len(measured) == len(results) and bool(results) and all(r["performance_accepted"] for r in measured)}
