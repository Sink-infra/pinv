"""Generate test tensors and FP64 references for one LinalgPinv suite."""

import argparse
import json
from pathlib import Path

from tests.common.cases import CASE_GROUPS
from tests.common.mock_cases import generate as generate_mock, load_cases as load_mock_cases
from tests.common.tensor_io import _select_cases, generate_suite


def select_mock_cases(expectation, names):
    cases = load_mock_cases()
    requested = set(names)
    missing = requested - {case["name"] for case in cases}
    if missing:
        raise ValueError(f"unknown case names: {sorted(missing)}")
    selected = [case for case in cases if not requested or case["name"] in requested]
    if expectation == "reject" or not selected:
        raise ValueError("selection contains no cases")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, choices=("coverage", "correctness", "mock"))
    parser.add_argument("--output", type=Path, help="directory to create")
    parser.add_argument("--case", action="append", default=[], help="exact case name; repeatable")
    parser.add_argument("--expectation", choices=("all", "success", "reject"), default="all")
    parser.add_argument("--list", action="store_true", help="list the selected cases without generating data")
    args = parser.parse_args()
    try:
        if args.suite == "mock":
            selected = select_mock_cases(args.expectation, args.case)
            listing = [{"name": case["name"], "shape": case["shape"], "dtype": case["dtype"],
                        "expectation": "success"} for case in selected]
        else:
            selected = _select_cases(CASE_GROUPS[args.suite], args.expectation, args.case)
            listing = [{"name": case.name, "shape": case.a_shape, "dtype": case.a_dtype,
                        "expectation": case.expectation} for case in selected]
        if args.list:
            print(json.dumps(listing, indent=2, ensure_ascii=False))
            return 0
        if args.output is None:
            parser.error("--output is required for data generation")
        if args.suite == "mock":
            manifest = generate_mock(args.output.resolve(), selected)
            counts = {"total": len(manifest), "success": len(manifest), "reject": 0}
        else:
            manifest = generate_suite(args.output.resolve(), selected)
            counts = manifest["counts"]
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(counts, sort_keys=True))
    print(f"manifest: {args.output.resolve() / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
