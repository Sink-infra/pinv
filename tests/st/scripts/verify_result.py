"""Verify one raw float32 device output against a generated FP64 reference."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from tests.common.cases import CaseSpec, COMPARISON_ATOL, COMPARISON_RTOL
from tests.common.tensor_io import SCHEMA_VERSION
from tests.common.reference import infer_output_shape, resolve_tolerances


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    element_count: int
    mismatch_count: int
    nonfinite_count: int
    max_abs_error: float
    first_mismatch: Optional[Tuple[int, ...]]
    message: str


def _load_values(
    path: Path, dtype: str, element_count: int, *,
    expected_sha256: Optional[str] = None, label: str = "output",
) -> np.ndarray:
    storage_dtype = np.dtype({"float32": "<f4", "float64": "<f8"}[dtype])
    expected_bytes = element_count * storage_dtype.itemsize
    # Hash and decode the same bytes, so a file replaced between two reads cannot
    # make the comparator consume a different reference from the one it checked.
    payload = path.read_bytes()
    actual_bytes = len(payload)
    if actual_bytes != expected_bytes:
        raise ValueError(f"{path} contains {actual_bytes} bytes, expected exactly {expected_bytes}")
    if expected_sha256 is not None and hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError(f"{label} SHA256 mismatch; regenerate the case")
    return np.frombuffer(payload, dtype=storage_dtype).astype(np.float64)


def _load_case_metadata(case_dir: Path) -> dict:
    """Require the completed parent suite and its recorded metadata hash."""
    manifest = json.loads((case_dir.parent / "manifest.json").read_text(encoding="utf-8"))
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("operator") != "LinalgPinv"):
        raise ValueError("regenerate this suite: unsupported manifest schema or operator")
    entries = manifest.get("cases")
    if not isinstance(entries, list) or any(not isinstance(entry, dict) for entry in entries):
        raise ValueError("invalid manifest case entries")
    relative_metadata = f"{case_dir.name}/case.json"
    matches = [entry for entry in entries if entry.get("metadata") == relative_metadata]
    if len(matches) != 1:
        raise ValueError("manifest must contain exactly one entry for this case")
    entry = matches[0]
    payload = (case_dir / "case.json").read_bytes()
    if hashlib.sha256(payload).hexdigest() != entry.get("case_metadata_sha256"):
        raise ValueError("metadata SHA256 mismatch; regenerate the case")
    metadata = json.loads(payload)
    if (not isinstance(metadata, dict) or metadata.get("schema_version") != SCHEMA_VERSION
            or metadata.get("operator") != "LinalgPinv"):
        raise ValueError("regenerate this case: unsupported metadata schema or operator")
    if (metadata.get("name") != case_dir.name or entry.get("name") != metadata.get("name")
            or metadata.get("category") != manifest.get("category")
            or entry.get("expectation") != metadata.get("expectation")):
        raise ValueError("case identity disagrees with the suite manifest")
    return metadata


def _metadata_shape(value: object, label: str) -> Tuple[int, ...]:
    if (not isinstance(value, list) or len(value) < 2
            or any(type(dimension) is not int or dimension <= 0 for dimension in value)):
        raise ValueError(f"invalid {label} shape")
    return tuple(value)


def _validate_call_metadata(metadata: dict) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    input_spec = metadata["input"]["a"]
    input_shape = _metadata_shape(input_spec["shape"], "input")
    requested = metadata["attributes"]["requested"]
    case = CaseSpec(
        name=metadata["name"], category=metadata["category"], expectation="success",
        a_shape=input_shape, a_dtype=input_spec["dtype"],
        atol=requested["atol"], rtol=requested["rtol"], hermitian=requested["hermitian"],
    )
    expected_shape = infer_output_shape(case)
    output = metadata["expected_output"]
    output_shape = _metadata_shape(output["shape"], "output")
    if output["dtype"] != "float32" or output_shape != expected_shape:
        raise ValueError("output shape or dtype disagrees with the input contract")
    atol, rtol = resolve_tolerances(case)
    if metadata["attributes"]["resolved"] != {"atol": atol, "rtol": rtol, "hermitian": False}:
        raise ValueError("resolved attributes disagree with the requested call")
    if metadata.get("validation_errors") != []:
        raise ValueError("success case carries rejection diagnostics")
    return input_shape, output_shape


def _load_fixture(case_dir: Path, entry: dict, dtype: str, count: int, label: str) -> np.ndarray:
    expected_bytes = count * np.dtype({"float32": "<f4", "float64": "<f8"}[dtype]).itemsize
    if entry["logical_dtype"] != dtype or entry["bytes"] != expected_bytes:
        raise ValueError(f"{label} file metadata disagrees with its shape or dtype")
    digest = entry["sha256"]
    if (not isinstance(digest, str) or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)):
        raise ValueError(f"{label} file metadata must contain a SHA256 digest")
    return _load_values(
        case_dir / entry["path"], dtype, count, expected_sha256=digest, label=label,
    )


def verify_case_output(
    case_dir: Path, actual_path: Path, *,
    atol: float = COMPARISON_ATOL, rtol: float = COMPARISON_RTOL,
) -> VerificationResult:
    """Allow tighter budgets, but never loosen pinv.md's elementwise acceptance."""
    for name, value, maximum in (("atol", atol, COMPARISON_ATOL), ("rtol", rtol, COMPARISON_RTOL)):
        if not math.isfinite(value) or not 0 <= value <= maximum:
            raise ValueError(f"comparison {name} must be finite and in [0, {maximum}]")
    case_dir, actual_path = Path(case_dir).resolve(), Path(actual_path)
    metadata = _load_case_metadata(case_dir)
    if metadata.get("expectation") != "success":
        raise ValueError("rejection cases do not have an output to verify")
    input_shape, shape = _validate_call_metadata(metadata)
    _load_fixture(case_dir, metadata["files"]["a"], "float32", math.prod(input_shape), "input")
    element_count = math.prod(shape)
    expected = _load_fixture(case_dir, metadata["files"]["reference_a_pinv"], "float64", element_count, "reference")
    if not np.isfinite(expected).all():
        raise ValueError("generated finite fixtures must have a finite reference")
    actual = _load_values(actual_path, "float32", element_count)
    finite = np.isfinite(actual)
    finite_diff = np.abs(actual[finite] - expected[finite])
    finite_limit = atol + rtol * np.abs(expected[finite])
    mismatches = ~finite
    mismatches[finite] = finite_diff > finite_limit
    mismatch_count = int(np.count_nonzero(mismatches))
    nonfinite_count = int(np.count_nonzero(~finite))
    max_abs_error = float(np.max(finite_diff)) if finite_diff.size else 0.0
    first = None
    if mismatch_count:
        index = int(np.flatnonzero(mismatches)[0])
        first = tuple(int(part) for part in np.unravel_index(index, shape))
    passed = mismatch_count == 0
    message = "pass" if passed else (
        f"failed: mismatches={mismatch_count}/{element_count}, nonfinite={nonfinite_count}, "
        f"max_finite_abs_error={max_abs_error}, first_mismatch={first}"
    )
    return VerificationResult(passed, element_count, mismatch_count, nonfinite_count, max_abs_error, first, message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", required=True, type=Path)
    parser.add_argument("--actual", required=True, type=Path)
    parser.add_argument("--atol", type=float, default=COMPARISON_ATOL, help="comparison budget, at most 1e-4")
    parser.add_argument("--rtol", type=float, default=COMPARISON_RTOL, help="comparison budget, at most 1e-4")
    args = parser.parse_args()
    try:
        result = verify_case_output(args.case_dir.resolve(), args.actual.resolve(), atol=args.atol, rtol=args.rtol)
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(asdict(result), sort_keys=True, allow_nan=False))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
