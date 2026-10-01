"""Shared specification; each suite owns its own case declarations."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional, Tuple


MAX_MATRIX_DIM = 512
MAX_BATCH_DIM = 64
COMPARISON_ATOL = 1.0e-4
COMPARISON_RTOL = 1.0e-4
MIN_RETAINED_RATIO = 1.0e-2
MAX_DISCARDED_RATIO = 0.1


@dataclass(frozen=True)
class CaseSpec:
    """One call, including inputs outside the supported contract.

    Tuple-valued attributes are used only to describe nonscalar reject cases.
    Spectrum entries are relative to ``scale`` and padded with exact zeros.
    Batch ranks/scales cycle in C order over all leading dimensions.
    """

    name: str
    category: str
    expectation: str
    a_shape: Tuple[int, ...]
    a_dtype: str = "float32"
    input_pattern: str = "dense"
    seed: int = 0
    spectrum: Optional[Tuple[float, ...]] = None
    rank: Optional[int] = None
    scale: float = 1.0
    batch_ranks: Tuple[int, ...] = ()
    batch_scales: Tuple[float, ...] = ()
    atol: object = 0.0
    rtol: object = None
    hermitian: bool = False
    expected_error: Optional[str] = None
    covers: Tuple[str, ...] = ()
    note: str = ""

CASE_ROOT = Path(__file__).resolve().parents[1] / "st" / "cases"


def load_document(suite: str, root: Path = CASE_ROOT) -> dict:
    if suite not in ("coverage", "correctness", "mock"):
        raise ValueError(f"unknown suite: {suite}")
    document = json.loads((Path(root) / (suite + ".json")).read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("category") != suite:
        raise ValueError(f"case document category must be {suite}")
    rows = document.get("cases")
    if not isinstance(rows, list) or not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("a nonempty case list is required")
    names = [row.get("name") for row in rows]
    if any(not isinstance(name, str) or not name.startswith(suite + "_") for name in names):
        raise ValueError("case names must belong to the selected suite")
    if len(names) != len(set(names)):
        raise ValueError("case names must be unique")
    return document


def load_cases(suite: str, root: Path = CASE_ROOT) -> Tuple[CaseSpec, ...]:
    if suite not in ("coverage", "correctness"):
        raise ValueError("CaseSpec requires coverage or correctness")
    cases = []
    for source in load_document(suite, root)["cases"]:
        row = dict(source)
        if row.get("category") != suite:
            raise ValueError("case category disagrees with its document")
        for key in ("a_shape", "spectrum", "batch_ranks", "batch_scales", "covers", "atol", "rtol"):
            if isinstance(row.get(key), list):
                row[key] = tuple(row[key])
        cases.append(CaseSpec(**row))
    return tuple(cases)


CASE_GROUPS = {suite: load_cases(suite) for suite in ("coverage", "correctness")}
