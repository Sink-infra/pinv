"""Inspect suites, build and run AscendC or TileLang JIT tests."""

import argparse
from tests.common.cases import CASE_GROUPS, load_document
import json
import math
import sys


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


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "run":
        from tests.st.scripts.run_tests import main as run
        return run(argv[1:])
    if argv and argv[0] == "build":
        from tests.st.scripts.build import main as build
        return build(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__,
        epilog="Build: python -m tests build --help; run: python -m tests run --help")
    parser.add_argument("--details", action="store_true")
    args = parser.parse_args(argv)
    result = summary()
    print(json.dumps(result if args.details else {key: value["count"] for key, value in result.items()},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
