"""Gate a new optimization first in JIT, then in its exported CANN package."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

from tests.common.optimization import qualify_jit, qualify_package


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("jit", "package"), required=True)
    for name in ("baseline-jit", "candidate-jit", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("jit-acceptance", "baseline-cann", "candidate-cann", "baseline-manifest", "candidate-manifest"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args(argv)
    paths = {"baseline_jit": args.baseline_jit, "candidate_jit": args.candidate_jit}
    if args.stage == "package":
        for name in ("baseline_cann", "candidate_cann", "baseline_manifest", "candidate_manifest", "jit_acceptance"):
            path = getattr(args, name)
            if path is None:
                parser.error("package stage requires --" + name.replace("_", "-"))
            paths[name] = path
    evidence = {key: {"path": str(path.resolve()), "sha256": sha256(path.read_bytes()).hexdigest()}
                for key, path in paths.items()}
    documents = {key: json.loads(path.read_text()) for key, path in paths.items()}
    try:
        for key, document in documents.items():
            if not key.endswith(("_jit", "_cann")):
                continue
            raw = []
            for row in document["cases"]:
                path = Path(row["measurement"])
                measured = json.loads(path.read_text())
                for field in ("name", "status", "backend", "shape", "kernel", "steady", "provenance",
                              "environment", "input_sha256", "reference_sha256", "attributes", "output_init"):
                    if measured.get(field) != row.get(field):
                        raise ValueError("summary differs from raw measurement: " + field)
                raw.append({"path": str(path.resolve()), "sha256": sha256(path.read_bytes()).hexdigest()})
            evidence[key]["measurements"] = raw
        if args.stage == "package":
            accepted = documents.pop("jit_acceptance")
            if (accepted.get("passed") is not True or accepted.get("stage") != "new_jit_benefit" or
                    any(accepted["evidence"][key]["sha256"] != evidence[key]["sha256"]
                        for key in ("baseline_jit", "candidate_jit"))):
                raise ValueError("prior JIT acceptance does not match these measurements")
        report = (qualify_jit(documents["baseline_jit"], documents["candidate_jit"]) if args.stage == "jit"
                  else qualify_package(**documents))
    except (KeyError, TypeError, ValueError, OSError) as error:
        report = {"passed": False, "stage": args.stage, "reason": str(error)}
    report["evidence"] = evidence
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("passed", "stage", "reason", "new_gain_cases") if key in report}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
