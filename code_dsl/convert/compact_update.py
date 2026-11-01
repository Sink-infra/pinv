"""Export a validated compact TileLang body against an exact main baseline."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

from code_dsl.source import dsl_digest
from code_dsl.convert.window_export import callable_source, definition


def checksum(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline-module", "module-build", "mock-result", "output", "receipt"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    current = dsl_digest()
    report = json.loads(args.mock_result.read_text())
    if not (report.get("backend") == "jit" and report.get("suite") == "mock"
            and report.get("complete") and report.get("passed")
            and report["build"]["dsl_sha256"] == current):
        raise ValueError("complete current JIT Mock evidence required")
    cases = report["cases"]
    if {row["name"] for row in cases} != {f"mock_{index:02d}" for index in range(1, 13)}:
        raise ValueError("all twelve Mock fixtures required")
    receipt = json.loads((args.module_build / "build-receipt.json").read_text())
    artifact = receipt["artifacts"]["compact_module"]
    source = args.module_build / "compact/kernel.cpp"
    library = source.with_suffix(".so")
    if not (receipt["producer"] == "TileLang JIT" and receipt["dsl_sha256"] == current
            and checksum(source) == artifact["source_sha256"]
            and checksum(library) == artifact["library_sha256"]):
        raise ValueError("current immutable compact JIT build required")
    for row in cases:
        if not (row["status"] == "Pass" and row["returncode"] == 0
                and row["first"]["passed"] and row["final"]["passed"]
                and row.get("steady") and row.get("profile_samples", 0) >= 9
                and row.get("profiler_backend") == "task"):
            raise ValueError("correct steady complete-Task evidence required")
        if row["name"] in {"mock_01", "mock_04", "mock_07", "mock_10"}:
            if row["provenance"]["library_sha256"] != artifact["library_sha256"]:
                raise ValueError("measured compact library differs")
    target = args.output / "op_kernel/pinv_dsl_compact.h"
    text = target.read_text()
    before = callable_source(
        (args.baseline_module / "compact/kernel.cpp").read_text(),
        "PvStageCompact", noinline=True).strip()
    existing = definition(text, "PvStageCompact")
    if existing != before:
        raise ValueError("native compact body differs from the measured main baseline")
    after = callable_source(source.read_text(), "PvStageCompactAligned", noinline=True).strip()
    cmake = {path: checksum(path) for path in args.output.rglob("*CMakeList*") if path.is_file()}
    original = checksum(target)
    entry = args.output / "op_kernel/linalg_pinv.cpp"
    entry_text = entry.read_text()
    old_call = "            if constexpr (COMPACT) PvStageCompact(A,A_pinv,work,status,GetBlockNum(),data.m,data.n,data.batch,data.atol,data.rtol,0);"
    new_call = """            if constexpr (COMPACT) {
                if constexpr (TAIL) PvStageCompact(A,A_pinv,work,status,GetBlockNum(),data.m,data.n,data.batch,data.atol,data.rtol,0);
                else PvStageCompactAligned(A,A_pinv,work,status,GetBlockNum(),data.m,data.n,data.batch,data.atol,data.rtol,0);
            }"""
    if entry_text.count(old_call) != 1:
        raise ValueError("unexpected compact launch binding")
    entry_before = checksum(entry)
    replacement = text + "\n" + after + "\n"
    if definition(replacement, "PvStageCompact") != before:
        raise ValueError("historical tail body changed")
    if checksum(target) != original:
        raise ValueError("native body changed during export")
    target.write_text(replacement)
    if checksum(entry) != entry_before:
        raise ValueError("native entry changed during export")
    entry.write_text(entry_text.replace(old_call, new_call, 1))
    if any(checksum(path) != digest for path, digest in cmake.items()):
        raise ValueError("CMake changed")
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps({
        "dsl_sha256": current,
        "mock_result_sha256": checksum(args.mock_result),
        "source_sha256": artifact["source_sha256"],
        "library_sha256": artifact["library_sha256"],
        "native_before": original,
        "native_after": checksum(target),
        "callable_before": sha256(before.encode()).hexdigest(),
        "callable_after": sha256(after.encode()).hexdigest(),
        "preserved_tail_body_sha256": sha256(before.encode()).hexdigest(),
        "entry_before": entry_before,
        "entry_after": checksum(entry),
        "cmake_unchanged": True,
    }, indent=2) + "\n")
    print("compact body exported with unchanged CMake files")


if __name__ == "__main__":
    main()
