"""Fail an official/local gain audit on missing evidence or >3pp discrepancy."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

from tests.common.official_gains import compare_gains


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline-local", "candidate-local", "baseline-official", "candidate-official", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args(argv)
    paths = [args.baseline_local, args.candidate_local, args.baseline_official, args.candidate_official]
    report = compare_gains(*(json.loads(p.read_text()) for p in paths))
    report["evidence"] = [{"path": str(p.resolve()), "sha256": sha256(p.read_bytes()).hexdigest()} for p in paths]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for row in report["cases"]:
        print(row["name"], row["status"], "gain_gap_pp=", row.get("gain_gap_pp"))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
