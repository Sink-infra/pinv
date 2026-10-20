"""Shared numerical checks, output policy and timing statistics."""
from dataclasses import asdict
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import math, statistics
import numpy as np
from tests.st.scripts.verify_result import VerificationResult


def output_initialization(suite, mode="auto"):
    if suite not in ("coverage", "correctness", "mock"):
        raise ValueError("unknown suite for output initialization")
    if mode == "auto":
        return "runtime" if suite == "mock" else "nan"
    if mode not in ("runtime", "nan", "nan-once"):
        raise ValueError("unknown output initialization mode")
    return mode


def compare_mock_output(path, expected):
    payload = Path(path).read_bytes()
    if len(payload) != expected.size * 4:
        raise ValueError("output byte count differs from reference")
    actual = np.frombuffer(payload, dtype="<f4").astype(np.float64)
    shape = expected.shape
    expected = expected.ravel(order="C")
    finite = np.isfinite(actual)
    diff = np.abs(actual[finite] - expected[finite])
    mismatches = ~finite
    mismatches[finite] = diff > 1e-4 + 1e-4 * np.abs(expected[finite])
    count = int(np.count_nonzero(mismatches))
    return VerificationResult(count == 0, actual.size, count, int(np.count_nonzero(~finite)),
                              float(np.max(diff)) if diff.size else 0.0,
                              tuple(int(value) for value in np.unravel_index(int(np.flatnonzero(mismatches)[0]), shape)) if count else None,
                              "pass" if count == 0 else "elementwise mismatch")


def comparison_record(result):
    record = asdict(result)
    percent = (Decimal(result.mismatch_count) * 100 / Decimal(result.element_count)).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP)
    record["output_error_ratio"] = str(percent) + "%"
    return record


def timing_summary(samples, repeats):
    if len(samples) != repeats or any(not math.isfinite(value) or value <= 0 for value in samples):
        raise ValueError("device event sample count or duration is invalid")
    return {"samples_us": samples, "median_us": statistics.median(samples),
            "mean_us": statistics.mean(samples), "min_us": min(samples), "max_us": max(samples)}
