"""Report independent suite counts and shapes without requiring NumPy."""

import argparse
from tests.common.cases import CASE_GROUPS, load_document
import json
import math


def summary():
    result = {}
    for category in ("coverage", "correctness"):
        cases = CASE_GROUPS[category]
        runnable = [case for case in cases if case.expectation == "success"]
        result[category] = {
            "count": len(cases),
            "runnable": len(runnable),
            "reject": sum(case.expectation == "reject" for case in cases),
            "max_m": max((case.a_shape[-2] for case in runnable), default=0),
            "max_n": max((case.a_shape[-1] for case in runnable), default=0),
            "max_batch_count": max((math.prod(case.a_shape[:-2]) for case in runnable), default=0),
            "max_input_elements": max((math.prod(case.a_shape) for case in runnable), default=0),
        }
    mock = load_document("mock")
    result["mock"] = {
        "count": len(mock["cases"]),
        "runnable": len(mock["cases"]),
        "reject": 0,
        "max_m": max(case["shape"][-2] for case in mock["cases"]),
        "max_n": max(case["shape"][-1] for case in mock["cases"]),
        "max_batch_count": max(math.prod(case["shape"][:-2]) for case in mock["cases"]),
        "max_input_elements": max(math.prod(case["shape"]) for case in mock["cases"]),
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--details", action="store_true")
    args = parser.parse_args()
    result = summary()
    print(json.dumps(result if args.details else {key: value["count"] for key, value in result.items()},
                     indent=2, sort_keys=True))
