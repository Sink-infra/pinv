"""Float32 fixtures and an independent FP64, thresholded SVD test oracle.

CPU NumPy is used only by offline test tools, never as an operator fallback.
No complete pseudoinverse implementation is called here.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Tuple

import numpy as np

from tests.common.cases import (
    CaseSpec, MAX_BATCH_DIM, MAX_DISCARDED_RATIO, MAX_MATRIX_DIM,
    MIN_RETAINED_RATIO,
)


@dataclass(frozen=True)
class ValidationResult:
    errors: Tuple[str, ...]


@dataclass(frozen=True)
class TensorValue:
    logical_dtype: str
    values: np.ndarray
    storage: np.ndarray


@dataclass(frozen=True)
class CaseData:
    a: TensorValue


@dataclass(frozen=True)
class ReferenceResult:
    values: np.ndarray
    singular_values: np.ndarray
    thresholds: np.ndarray
    retained: np.ndarray
    resolved_atol: float
    resolved_rtol: float


def _is_scalar(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, (bool, np.bool_))


def validate_case(case: CaseSpec) -> ValidationResult:
    """Label unsupported inputs; labels are not prescribed ACLNN return codes.

    Do not invent a maximum total batch, a layout requirement, or behavior for
    negative/nonfinite scalar tolerances; those are not specified in pinv.md.
    The fixture catalog uses finite, nonnegative attributes throughout.
    """
    errors = []
    if len(case.a_shape) < 2:
        errors.append("A_RANK")
    else:
        m, n = case.a_shape[-2:]
        if not 1 <= m <= MAX_MATRIX_DIM:
            errors.append("M_RANGE")
        if not 1 <= n <= MAX_MATRIX_DIM:
            errors.append("N_RANGE")
        if any(not 1 <= dimension <= MAX_BATCH_DIM for dimension in case.a_shape[:-2]):
            errors.append("BATCH_DIM_RANGE")
    if case.a_dtype != "float32":
        errors.append("A_DTYPE")
    if not _is_scalar(case.atol):
        errors.append("ATOL_SCALAR")
    if case.rtol is not None and not _is_scalar(case.rtol):
        errors.append("RTOL_SCALAR")
    if case.hermitian is not False:
        errors.append("HERMITIAN_UNSUPPORTED")
    return ValidationResult(tuple(errors))


def infer_output_shape(case: CaseSpec) -> Tuple[int, ...]:
    validation = validate_case(case)
    if validation.errors:
        raise ValueError(f"invalid case {case.name}: {validation.errors}")
    return case.a_shape[:-2] + (case.a_shape[-1], case.a_shape[-2])


def resolve_tolerances(case: CaseSpec) -> Tuple[float, float]:
    """Use actual FP32 Float attributes from the existing Host interface."""
    if validate_case(case).errors:
        raise ValueError(f"cannot resolve attributes of rejected case {case.name}")
    atol = float(np.float32(case.atol))
    if case.rtol is not None:
        rtol = float(np.float32(case.rtol))
    elif atol > 0:
        rtol = 0.0
    else:
        rtol = max(case.a_shape[-2:]) * float(np.finfo(np.float32).eps)
    return atol, rtol


def storage_description(dtype: str) -> dict[str, str]:
    descriptions = {
        "float32": "little-endian IEEE float32",
        "float16": "little-endian IEEE float16",
        "bfloat16": "little-endian uint16 BF16 bits (reject fixtures only)",
        "float64": "little-endian IEEE float64",
        "int32": "little-endian signed int32",
    }
    return {"logical_dtype": dtype, "storage": descriptions[dtype]}


def _tensor(values: np.ndarray, dtype: str) -> TensorValue:
    if dtype == "bfloat16":
        # BF16 occurs only in zero-valued reject fixtures, so no rounding is needed.
        logical = np.array(values, dtype=np.float32, copy=True, order="C")
        bits = (logical.view(np.uint32) >> np.uint32(16)).astype("<u2")
        return TensorValue(dtype, logical, bits)
    storage_dtype = {"float32": "<f4", "float16": "<f2", "float64": "<f8", "int32": "<i4"}[dtype]
    # np.array preserves scalar shape (), unlike np.ascontiguousarray.
    logical = np.array(values, dtype=storage_dtype, copy=True, order="C")
    return TensorValue(dtype, logical, logical)


def _spectrum(case: CaseSpec, index: int, k: int) -> np.ndarray:
    rank = case.batch_ranks[index % len(case.batch_ranks)] if case.batch_ranks else case.rank
    rank = k if rank is None else rank
    if not 0 <= rank <= k:
        raise ValueError(f"fixture rank {rank} is outside [0, {k}]")
    result = np.zeros(k, dtype=np.float64)
    if case.spectrum is not None:
        if len(case.spectrum) > k or any(value < 0 for value in case.spectrum):
            raise ValueError("fixture spectrum must be nonnegative and fit min(m, n)")
        result[:len(case.spectrum)] = case.spectrum
        if case.rank is not None or case.batch_ranks:
            result[rank:] = 0.0
    elif rank:
        # A controlled spectrum avoids accidental tiny singular values from randn.
        result[:rank] = np.geomspace(1.0, 1.0 / 32.0, rank)
    factor = case.batch_scales[index % len(case.batch_scales)] if case.batch_scales else 1.0
    return result * case.scale * factor


def materialize_case(case: CaseSpec) -> CaseData:
    if validate_case(case).errors:
        # Rejection data preserves the invalid rank/shape/dtype exactly; no oracle
        # is computed for it. Unsupported attributes are preserved in metadata.
        return CaseData(_tensor(np.zeros(case.a_shape), case.a_dtype))
    rng = np.random.default_rng(case.seed)
    m, n = case.a_shape[-2:]
    batch_count = math.prod(case.a_shape[:-2])
    result = np.zeros((batch_count, m, n), dtype=np.float64)
    k = min(m, n)
    for index in range(batch_count):
        if case.input_pattern == "zeros":
            continue
        if case.input_pattern == "constant":
            factor = case.batch_scales[index % len(case.batch_scales)] if case.batch_scales else 1.0
            result[index].fill(case.scale * factor)
            continue
        singular = _spectrum(case, index, k)
        if case.input_pattern in ("diagonal", "signed_diagonal", "permuted_diagonal"):
            rows, columns = np.arange(k), np.arange(k)
            if case.input_pattern == "signed_diagonal":
                singular = singular * np.where(np.arange(k) % 2, -1.0, 1.0)
            elif case.input_pattern == "permuted_diagonal":
                rows = rng.permutation(m)[:k]
                columns = rng.permutation(n)[:k]
                singular = singular * rng.choice((-1.0, 1.0), size=k)
            result[index, rows, columns] = singular
        elif case.input_pattern == "dense":
            u, _ = np.linalg.qr(rng.normal(size=(m, k)), mode="reduced")
            v, _ = np.linalg.qr(rng.normal(size=(n, k)), mode="reduced")
            result[index] = np.einsum("ik,jk->ij", u * singular, v, optimize=False)
        else:
            raise ValueError(f"unknown input pattern: {case.input_pattern}")
    return CaseData(_tensor(result.reshape(case.a_shape), case.a_dtype))


def reciprocal_singular_values(singular: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Strict >, with masked division so discarded zeros never produce Inf/NaN."""
    inverse = np.zeros_like(singular, dtype=np.float64)
    np.divide(1.0, singular, out=inverse, where=singular > thresholds)
    return inverse


def pinv_reference(case: CaseSpec, data: CaseData) -> ReferenceResult:
    if validate_case(case).errors:
        raise ValueError(f"cannot evaluate rejected case {case.name}")
    if data.a.logical_dtype != "float32" or data.a.values.dtype != np.dtype("float32"):
        raise ValueError("oracle requires actual float32 input values")
    if data.a.values.shape != case.a_shape:
        raise ValueError("materialized input shape differs from the case declaration")
    actual_input = data.a.values.astype(np.float64)
    if not np.isfinite(actual_input).all():
        raise ValueError("offline fixtures must be finite; nonfinite behavior is unspecified")
    u, singular, vh = np.linalg.svd(actual_input, full_matrices=False)
    atol, rtol = resolve_tolerances(case)
    thresholds = np.maximum(atol, singular[..., :1] * rtol)
    inverse = reciprocal_singular_values(singular, thresholds)
    # Explicit contraction gives a portable batched FP64 reconstruction.
    values = np.einsum(
        "...ik,...kj->...ij", np.swapaxes(vh, -1, -2) * inverse[..., None, :],
        np.swapaxes(u, -1, -2), optimize=False,
    )
    return ReferenceResult(values, singular, thresholds, singular > thresholds, atol, rtol)


def audit_spectral_gap(result: ReferenceResult) -> dict[str, object]:
    """Audit the actual FP32 matrix spectrum, not the pre-quantization recipe."""
    singular = result.singular_values
    largest = singular[..., :1]
    keep = result.retained
    retained_ok = (~keep) | (singular >= MIN_RETAINED_RATIO * largest)
    discarded_ok = keep | (singular <= MAX_DISCARDED_RATIO * result.thresholds)
    if not retained_ok.all() or not discarded_ok.all():
        raise ValueError("materialized float32 fixture violates the pinv.md spectral gap")
    if not np.isfinite(result.values).all():
        raise ValueError("fixture reference is not finite")
    ranks = np.count_nonzero(keep, axis=-1)
    retained_ratios = np.divide(singular, largest, out=np.zeros_like(singular), where=largest != 0)
    discarded_ratios = np.divide(
        singular, result.thresholds, out=np.zeros_like(singular), where=result.thresholds != 0,
    )
    return {
        "passed": True,
        "source": "FP64 SVD of stored float32 input",
        "batch_shape": list(singular.shape[:-1]),
        "retained_ranks": ranks.reshape(-1).tolist(),
        "largest_singular_values": largest.reshape(-1).tolist(),
        "thresholds": result.thresholds.reshape(-1).tolist(),
        "min_retained_relative": float(np.min(retained_ratios[keep])) if keep.any() else None,
        "max_discarded_threshold_ratio": float(np.max(discarded_ratios[~keep])) if (~keep).any() else None,
    }
