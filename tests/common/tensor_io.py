"""Generate one suite's raw inputs, FP64 references, and hashed metadata."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import re
from typing import Iterable, Sequence

import numpy as np

from tests.common.cases import CaseSpec, COMPARISON_ATOL, COMPARISON_RTOL
from tests.common.reference import (
    audit_spectral_gap, infer_output_shape, materialize_case, pinv_reference,
    storage_description, validate_case,
)


SCHEMA_VERSION = 2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_catalog(cases: Sequence[CaseSpec]) -> str:
    categories = {case.category for case in cases}
    if len(categories) != 1 or not categories.issubset(("coverage", "correctness")):
        raise ValueError("generate exactly one nonempty named suite at a time")
    category = next(iter(categories))
    names = [case.name for case in cases]
    if len(names) != len(set(names)):
        raise ValueError("case names must be unique within a suite")
    for case in cases:
        if not re.fullmatch(category + r"_[a-z0-9_]+", case.name):
            raise ValueError(f"unsafe or miscategorized case name: {case.name}")
        errors = validate_case(case).errors
        if case.expectation == "success":
            if errors or case.expected_error is not None:
                raise ValueError(f"success case {case.name} has rejection diagnostics")
        elif case.expectation == "reject":
            if case.expected_error is None or case.expected_error not in errors:
                raise ValueError(f"reject case {case.name} does not trigger its expected label")
        else:
            raise ValueError(f"unknown expectation: {case.expectation}")
    return category


def comparison_policy() -> dict[str, object]:
    return {
        "absolute_tolerance": COMPARISON_ATOL,
        "relative_tolerance": COMPARISON_RTOL,
        "finite_values": "abs(actual-reference) <= atol + rtol*abs(reference), every element",
        "nonfinite_values": "reject all nonfinite outputs for these finite fixtures",
        "reference": "FP64 SVD reconstruction of actual float32 input, with threshold truncation",
        "source": "pinv.md",
        "allow_looser_tolerances": False,
    }


def _emit_array(case_dir: Path, name: str, filename: str, values: np.ndarray, dtype: str) -> tuple[str, dict]:
    path = case_dir / filename
    np.asarray(values).ravel(order="C").tofile(path)
    return name, {
        "path": filename, "sha256": sha256_file(path), "bytes": path.stat().st_size,
        **storage_description(dtype),
    }


def _case_metadata(case: CaseSpec, case_dir: Path) -> dict[str, object]:
    errors = validate_case(case).errors
    data = materialize_case(case)
    result = None
    spectral_audit = None
    if not errors:
        result = pinv_reference(case, data)
        spectral_audit = audit_spectral_gap(result)
    # The spectral promise is checked before any valid case's binary is emitted.
    files = dict((_emit_array(case_dir, "a", "a.bin", data.a.storage, case.a_dtype),))
    if result is not None:
        key, entry = _emit_array(
            case_dir, "reference_a_pinv", "reference_a_pinv_fp64.bin", result.values.astype("<f8"), "float64",
        )
        files[key] = entry
    return {
        "schema_version": SCHEMA_VERSION,
        "operator": "LinalgPinv",
        "name": case.name,
        "category": case.category,
        "expectation": case.expectation,
        "expected_error": case.expected_error,
        "validation_errors": list(errors),
        "input": {"a": {
            "shape": list(case.a_shape), "dtype": case.a_dtype,
            "storage_order": "C", "contiguous": True,
            "strides_in_elements": [int(stride // data.a.values.dtype.itemsize) for stride in data.a.values.strides],
        }},
        "attributes": {
            "requested": {"atol": case.atol, "rtol": case.rtol, "hermitian": case.hermitian},
            "resolved": None if result is None else {
                "atol": result.resolved_atol, "rtol": result.resolved_rtol, "hermitian": False,
            },
            "host_default_rtol_sentinel": -1.0 if case.rtol is None else None,
        },
        "patterns": {
            "a": case.input_pattern, "seed": case.seed, "spectrum": case.spectrum,
            "rank": case.rank, "scale": case.scale,
            "batch_ranks": case.batch_ranks, "batch_scales": case.batch_scales,
        },
        "expected_output": {
            "shape": list(infer_output_shape(case)) if result is not None else None,
            "dtype": "float32" if result is not None else None,
            "comparison": comparison_policy() if result is not None else None,
        },
        "spectral_audit": spectral_audit,
        "covers": list(case.covers), "note": case.note, "files": files,
    }


def generate_suite(output_dir: Path, cases: Sequence[CaseSpec]) -> dict[str, object]:
    """Generate a directory; a final manifest marks completed generation."""
    category = audit_catalog(cases)
    output_dir = Path(output_dir)
    # mkdir with exist_ok=False also prevents accidental overwrite on a race.
    output_dir.mkdir(parents=True, exist_ok=False)
    entries = []
    for case in cases:
        case_dir = output_dir / case.name
        case_dir.mkdir()
        metadata = _case_metadata(case, case_dir)
        metadata_path = case_dir / "case.json"
        store_text(metadata_path, json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
        entries.append({
            "name": case.name, "expectation": case.expectation,
            "metadata": f"{case.name}/case.json", "case_metadata_sha256": sha256_file(metadata_path),
        })
    counts = Counter(case.expectation for case in cases)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "operator": "LinalgPinv", "category": category,
        "schema_order": ["a", "atol", "rtol", "hermitian", "a_pinv"],
        "generation": {
            "deterministic_for_recorded_environment": True, "endianness": "little",
            "overwrite_policy": "refuse", "numpy_version": np.__version__,
            "python_version": platform.python_version(), "platform": platform.platform(),
        },
        "catalog_audit_failures": [],
        "counts": {"total": len(cases), "success": counts["success"], "reject": counts["reject"]},
        "cases": entries,
    }
    store_text(output_dir / 'manifest.json', json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    return manifest


def _select_cases(cases: Sequence[CaseSpec], expectation: str, names: Iterable[str]) -> tuple[CaseSpec, ...]:
    requested = set(names)
    missing = requested - {case.name for case in cases}
    if missing:
        raise ValueError(f"unknown case names: {sorted(missing)}")
    selected = tuple(case for case in cases
                     if (expectation == "all" or case.expectation == expectation)
                     and (not requested or case.name in requested))
    if not selected:
        raise ValueError("selection contains no cases")
    return selected


def store_bytes(path: Path, payload: bytes) -> None:
    np.frombuffer(payload, dtype=np.uint8).tofile(path)


def store_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    store_bytes(path, content.encode(encoding))
