"""Execute one current-source suite with device timing and optional kernel profiling."""

import argparse
import hashlib
import json
import shlex
from pathlib import Path
import subprocess

from tests.common.cases import CASE_GROUPS
from tests.common.tensor_io import _select_cases, store_text
from tests.st.scripts.gen_data import select_mock_cases
from tests.common.profiling import profile_samples
from tests.common.measurements import hardware_snapshot, steady_samples
from tests.st.backends import get_backend
from tests.st.backends.transport import execution_transport
from tests.st.scripts.runtime import DEFAULT_BUILD, ROOT, environment
from tests.st.scripts.verify_result import verify_case_output


from tests.common.observations import (output_initialization, compare_mock_output, comparison_record, timing_summary)
from tests.common.workloads import prepare_cases


def _capture_case(case, build_dir, folder, *, device=0, warmup=3, repeats=9, timeout=120, profile=False,
                  output_init="runtime", profiler_backend="operator", profile_samples_count=None, profiler_replay="kernel", profiler_warmup=5, backend="ascend", executor=None):
    folder.mkdir()
    if output_init not in ("runtime", "nan", "nan-once"):
        raise ValueError("unknown output initialization mode")
    result_path = folder / "runner.json"
    actual = folder / "actual.bin"
    implementation = get_backend(backend)
    command = implementation.command(build_dir) + [
               "--input", str(case["input"]), "--output", str(actual),
               "--result", str(result_path), "--shape", ",".join(map(str, case["shape"])),
               "--atol", str(case["atol"]), "--rtol", str(case["rtol"]),
               "--device", str(device), "--warmup", str(warmup), "--repeats", str(repeats),
               "--output-init", output_init]
    if profiler_backend not in ("operator", "task"):
        raise ValueError("unknown profiler backend")
    if profiler_replay not in ("kernel", "application"):
        raise ValueError("unknown operator replay mode")
    if not isinstance(profiler_warmup, int) or not 0 <= profiler_warmup <= 500:
        raise ValueError("invalid operator replay warmup count")
    captured = repeats if profile_samples_count is None else profile_samples_count
    if profile and (not 1 <= captured <= repeats or (profiler_backend == "task" and captured != repeats)):
        raise ValueError("invalid profiling sample count for this backend")
    if profile and profiler_backend == "task":
        command += ["--profile", str(folder / "profile")]
    if profile and profiler_backend == "operator":
        store_text(folder / "runner-command.json", json.dumps(command, indent=2) + "\n")
        command = ["msprof", "op", "--application=" + shlex.join(command),
                   "--output=" + str(folder / "profile"), "--aic-metrics=BasicInfo",
                   "--launch-count=" + str(captured), "--launch-skip-before-match=" + str(warmup),
                   "--warm-up=" + str(profiler_warmup),
                   "--replay-mode=" + profiler_replay]
    store_text(folder / "command.json", json.dumps(command, indent=2) + "\n")
    report = {"name": case["name"], "shape": list(case["shape"]), "status": "Harness Error",
              "device": device,
              "backend": backend, "attributes": {"atol": case["atol"], "rtol": case["rtol"]}, "input_sha256": hashlib.sha256(Path(case["input"]).read_bytes()).hexdigest() if Path(case["input"]).is_file() else None,
              "reference_sha256": hashlib.sha256(case["gold"].tobytes()).hexdigest() if "gold" in case else hashlib.sha256((case["folder"] / "reference_a_pinv_fp64.bin").read_bytes()).hexdigest(),
              "output_init": output_init, "profiler_backend": profiler_backend if profile else None,
              "profile_samples": captured if profile else None,
              "profiler_replay": profiler_replay if profile and profiler_backend == "operator" else None,
              "profiler_warmup": profiler_warmup if profile and profiler_backend == "operator" else None}
    with (folder / "runner.log").open("w") as log:
        try:
            process = (executor or subprocess.run)(command, env=environment(build_dir), stdout=log,
                                     stderr=subprocess.STDOUT, timeout=timeout)
        except subprocess.TimeoutExpired:
            report.update(status="TLE", timeout_seconds=timeout)
            return report
        report["returncode"] = process.returncode
        if not result_path.is_file():
            report["detail"] = "runner produced no structured result; see runner.log"
            return report
        native = json.loads(result_path.read_text())
        if not isinstance(native, dict):
            raise ValueError("runner result must be an object")
        report["runner"] = native
        if native.get("provenance"):
            report["provenance"] = native["provenance"]
        if native.get("soc"):
            report["environment"] = hardware_snapshot(native["soc"])
        if native.get("status") != "executed":
            report["status"] = native["status"] if native.get("status") in ("Runtime Error", "Harness Error", "Unsupported") else "Harness Error"
            return report
        if native.get("output_init") != output_init:
            report["detail"] = "runner output initialization mode differs from the request"
            return report
        if process.returncode or native.get("teardown"):
            report.update(status="Runtime Error", detail="runner exit or resource teardown failed")
            return report
        try:
            report["event"] = timing_summary(native["event_micros"], repeats)
            observations = {}
            for key, path in (("first", actual), ("final", Path(str(actual) + ".final.bin"))):
                comparison = (compare_mock_output(path, case["gold"]) if "gold" in case
                              else verify_case_output(case["folder"], path))
                observations[key] = comparison_record(comparison)
            report.update(observations)
            report["status"] = "Pass" if all(value["passed"] for value in observations.values()) else "Wrong Answer"
            if profile:
                if profiler_backend == "task":
                    (executor or subprocess.run)(["msprof", "--export=on", "--output=" + str(folder / "profile")],
                                   env=environment(build_dir), stdout=log, stderr=subprocess.STDOUT,
                                   timeout=60, check=True)
                measured = profile_samples(folder / "profile", 0 if profiler_backend == "operator" else warmup,
                                           captured, profiler_backend, kernel_prefix=implementation.KERNEL_PREFIX)
                report["kernel"] = timing_summary(measured, captured)
                report["steady"] = steady_samples(measured)
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
            report.update(status="Harness Error", detail=str(error))
    return report

def run_case(case, build_dir, folder, **settings):
    try:
        report = _capture_case(case, build_dir, folder, **settings)
    except (OSError, ValueError, KeyError, TypeError) as error:
        report = {"name": case["name"], "shape": list(case["shape"]),
                  "status": "Harness Error", "detail": str(error)}
    store_text(folder / "result.json", json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, choices=("coverage", "correctness", "mock"))
    parser.add_argument("--backend", choices=("ascend", "jit", "both"), default="ascend")
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-build", action="store_true", help="verify the selected backend's current-source build")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--transport", choices=("direct", "npu-session"), default="direct")
    parser.add_argument("--allow-unsupported", action="store_true", help="allow scoped results, while reporting incomplete coverage")
    parser.add_argument("--max-slowdown", type=float, default=0.05)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profiler-backend", choices=("operator", "task"), default="operator")
    parser.add_argument("--profiler-warmup", type=int, default=5)
    parser.add_argument("--profiler-replay", choices=("kernel", "application"), default="kernel")
    parser.add_argument("--profile-samples", type=int)
    parser.add_argument("--output-init", choices=("auto", "runtime", "nan", "nan-once"), default="auto")
    args = parser.parse_args(argv)
    if not (0 <= args.warmup <= 10000 and 1 <= args.repeats <= 10000 and args.timeout > 0 and args.device >= 0
            and 0 <= args.profiler_warmup <= 500 and 1 <= args.processes <= 64 and 0 <= args.max_slowdown <= 1):
        parser.error("invalid device, counts, slowdown limit or timeout")
    if args.profile_samples is not None and (not args.profile or args.profiler_backend != "operator"
                                              or not 1 <= args.profile_samples <= args.repeats):
        parser.error("profile samples require operator profiling and a count in [1, repeats]")
    output = args.output.resolve()
    artifact_roots = (ROOT / "build", ROOT / "code_dsl/results")
    if output == ROOT or (ROOT in output.parents and not any(base == output or base in output.parents for base in artifact_roots)):
        parser.error("result directory must be outside tracked source directories")
    try:
        output_init = output_initialization(args.suite, args.output_init)
        if args.suite == "mock":
            select_mock_cases("success", args.case)
        else:
            _select_cases(CASE_GROUPS[args.suite], "success", args.case)
        if output.exists():
            raise ValueError("result directory already exists")
        # Prepare once: both implementations consume exactly these input/reference files.
        output.mkdir(parents=True)
        cases = prepare_cases(args.suite, args.case, output / "data")
        names = ["jit", "ascend"] if args.backend == "both" else [args.backend]
        reports = {}
        for name in names:
            implementation = get_backend(name)
            if args.backend == "both":
                build_dir = (args.build_dir or ROOT / "build/tests/backends") / name
            else:
                build_dir = args.build_dir or (DEFAULT_BUILD if name == "ascend" else implementation.DEFAULT_BUILD)
            build_dir = build_dir.resolve()
            if not args.no_build:
                implementation.build(build_dir, args.jobs)
            identity = implementation.verify(build_dir)
            backend_output = output / name if args.backend == "both" else output
            backend_output.mkdir(exist_ok=True)
            report = {"suite": args.suite, "backend": name, "build": identity,
                      "selection": [case["name"] for case in cases],
                      "device": args.device, "warmup": args.warmup, "repeats": args.repeats,
                      "processes": args.processes, "profile": args.profile,
                      "output_init": args.output_init, "profiler_backend": args.profiler_backend if args.profile else None,
                      "profiler_replay": args.profiler_replay if args.profile else None,
                      "profiler_warmup": args.profiler_warmup if args.profile else None,
                      "cases": [], "passed": False}
            def save():
                store_text(backend_output / "result.json", json.dumps(report, indent=2, allow_nan=False) + "\n")
            with execution_transport(args.transport, args.device) as (executor, transport):
                report["execution"] = transport
                save()
                for process in range(1, args.processes + 1):
                    folder = backend_output if args.processes == 1 else backend_output / f"process-{process}"
                    folder.mkdir(exist_ok=True)
                    for case in cases:
                        result = run_case(case, build_dir, folder / case["name"], device=args.device,
                                          warmup=args.warmup, repeats=args.repeats, timeout=args.timeout, profile=args.profile,
                                          output_init=case.get("output_init", output_init) if args.output_init == "auto" else output_init,
                                          profiler_backend=args.profiler_backend, profile_samples_count=args.profile_samples,
                                          profiler_replay=args.profiler_replay, profiler_warmup=args.profiler_warmup,
                                          backend=name, executor=executor)
                        result.update(process=process, measurement=str(folder / case["name"] / "result.json"))
                        report["cases"].append(result)
                        save()
                        print(json.dumps({"backend": name, "process": process, "name": result["name"],
                                          "status": result["status"], "kernel_median_us": result.get("kernel", {}).get("median_us")}), flush=True)
            implementation.verify(build_dir)
            rows = report["cases"]
            report["unsupported_cases"] = sorted({r["name"] for r in rows if r["status"] == "Unsupported"})
            report["complete"] = not report["unsupported_cases"]
            report["passed"] = bool(rows) and all(r["status"] == "Pass" for r in rows)
            report["scoped_passed"] = any(r["status"] == "Pass" for r in rows) and all(r["status"] in ("Pass", "Unsupported") for r in rows)
            save()
            reports[name] = report
        success = all(r["passed"] or (args.allow_unsupported and r["scoped_passed"]) for r in reports.values())
        if args.backend == "both":
            from tests.common.paired import compare_backends
            comparison = compare_backends(reports["jit"]["cases"], reports["ascend"]["cases"], max_slowdown=args.max_slowdown)
            comparison["selection"] = [case["name"] for case in cases]
            comparison["reports"] = {name: {"path": str(output / name / "result.json"),
                                               "sha256": hashlib.sha256((output / name / "result.json").read_bytes()).hexdigest()}
                                     for name in names}
            store_text(output / "comparison.json", json.dumps(comparison, indent=2, allow_nan=False) + "\n")
            if args.profile:
                success = success and comparison["scoped_performance_accepted" if args.allow_unsupported else "performance_accepted"]
    except (OSError, ValueError, RuntimeError, ImportError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
