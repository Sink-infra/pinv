"""Execute one current-source suite with device timing and optional kernel profiling."""

import argparse
from dataclasses import asdict
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import math
import shlex
from pathlib import Path
import statistics
import subprocess

import numpy as np

from tests.common.cases import CASE_GROUPS
from tests.common.mock_cases import generate as generate_mock
from tests.common.tensor_io import _select_cases, generate_suite, store_text
from tests.st.scripts.build import build_current, verify_build
from tests.st.scripts.gen_data import select_mock_cases
from tests.st.scripts.profile_mock_workloads import profile_samples
from tests.st.scripts.runtime import DEFAULT_BUILD, ROOT, environment
from tests.st.scripts.verify_result import VerificationResult, verify_case_output


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


def prepare_cases(suite, names, destination):
    if suite == "mock":
        manifest = generate_mock(destination, select_mock_cases("success", names))
        prepared = []
        for case in manifest:
            arrays = {}
            for key in ("input", "reference"):
                arrays[key] = np.load(destination / case[key], allow_pickle=False)
                if hashlib.sha256(arrays[key].tobytes()).hexdigest() != case[key + "_sha256"]:
                    raise ValueError("Mock array digest differs")
            a, gold = arrays["input"], arrays["reference"]
            shape = tuple(case["shape"])
            if a.dtype != np.float32 or a.shape != shape or gold.dtype != np.float64:
                raise ValueError("Mock array declaration differs")
            if gold.shape != shape[:-2] + shape[-2:][::-1] or not np.isfinite(gold).all():
                raise ValueError("Mock reference shape or values differ")
            raw = destination / (case["name"] + ".bin")
            a.astype("<f4").tofile(raw)
            prepared.append({"name": case["name"], "shape": shape, "input": raw,
                             "atol": case["atol"], "rtol": case["rtol"], "gold": gold,
                             "output_init": case["output_init"]})
        return prepared
    cases = _select_cases(CASE_GROUPS[suite], "success", names)
    generate_suite(destination, cases)
    return [{"name": case.name, "shape": case.a_shape, "input": destination / case.name / "a.bin",
             "atol": case.atol, "rtol": -1.0 if case.rtol is None else case.rtol,
             "folder": destination / case.name} for case in cases]


def _capture_case(case, build_dir, folder, *, device=0, warmup=3, repeats=7, timeout=120, profile=False,
                  output_init="runtime", profiler_backend="operator", profile_samples_count=None, profiler_replay="kernel", profiler_warmup=5):
    folder.mkdir()
    if output_init not in ("runtime", "nan", "nan-once"):
        raise ValueError("unknown output initialization mode")
    result_path = folder / "runner.json"
    actual = folder / "actual.bin"
    command = [str(build_dir / "tests/linalg_pinv_runner"),
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
              "output_init": output_init, "profiler_backend": profiler_backend if profile else None,
              "profile_samples": captured if profile else None,
              "profiler_replay": profiler_replay if profile and profiler_backend == "operator" else None,
              "profiler_warmup": profiler_warmup if profile and profiler_backend == "operator" else None}
    with (folder / "runner.log").open("w") as log:
        try:
            process = subprocess.run(command, env=environment(build_dir), stdout=log,
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
        if native.get("status") != "executed":
            report["status"] = native["status"] if native.get("status") in ("Runtime Error", "Harness Error") else "Harness Error"
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
                    subprocess.run(["msprof", "--export=on", "--output=" + str(folder / "profile")],
                                   env=environment(build_dir), stdout=log, stderr=subprocess.STDOUT,
                                   timeout=60, check=True)
                measured = profile_samples(folder / "profile", 0 if profiler_backend == "operator" else warmup,
                                           captured, profiler_backend)
                report["kernel"] = timing_summary(measured, captured)
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
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-build", action="store_true", help="require an existing matching current-source build")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profiler-backend", choices=("operator", "task"), default="operator")
    parser.add_argument("--profiler-warmup", type=int, default=5, help="operator replay warmup count")
    parser.add_argument("--profiler-replay", choices=("kernel", "application"), default="kernel")
    parser.add_argument("--profile-samples", type=int, help="operator samples after warmup; default: all repeats")
    parser.add_argument("--output-init", choices=("auto", "runtime", "nan", "nan-once"), default="auto",
                        help="auto: runtime allocation for Mock, NaN diagnostics for other suites")
    args = parser.parse_args(argv)
    if not (0 <= args.warmup <= 10000 and 1 <= args.repeats <= 10000 and args.timeout > 0 and args.device >= 0
            and 0 <= args.profiler_warmup <= 500):
        parser.error("invalid device, launch counts, or timeout")
    if args.profile_samples is not None and (not args.profile or args.profiler_backend != "operator"
                                              or not 1 <= args.profile_samples <= args.repeats):
        parser.error("profile samples require operator profiling and a count in [1, repeats]")
    output = args.output.resolve()
    build_dir = args.build_dir.resolve()
    if output == ROOT or ROOT / "tests" == output or ROOT / "tests" in output.parents or ROOT / "code" in output.parents:
        parser.error("result directory must be outside tracked source directories")
    try:
        output_init = output_initialization(args.suite, args.output_init)
        if args.suite == "mock":
            select_mock_cases("success", args.case)
        else:
            _select_cases(CASE_GROUPS[args.suite], "success", args.case)
        if output.exists():
            raise ValueError("result directory already exists")
        if not args.no_build:
            build_current(build_dir, args.jobs)
        identity = verify_build(build_dir)
        output.mkdir(parents=True)
        cases = prepare_cases(args.suite, args.case, output / "data")
        records = []
        for case in cases:
            result = run_case(case, build_dir, output / case["name"], device=args.device,
                              warmup=args.warmup, repeats=args.repeats, timeout=args.timeout, profile=args.profile,
                              output_init=(case.get("output_init", output_init) if args.output_init == "auto" else output_init),
                              profiler_backend=args.profiler_backend, profile_samples_count=args.profile_samples,
                              profiler_replay=args.profiler_replay, profiler_warmup=args.profiler_warmup)
            records.append(result)
            print(json.dumps({"name": result["name"], "status": result["status"],
                              "event_mean_us": result.get("event", {}).get("mean_us"),
                              "kernel_mean_us": result.get("kernel", {}).get("mean_us"),
                              "event_median_us": result.get("event", {}).get("median_us"),
                              "kernel_median_us": result.get("kernel", {}).get("median_us")}), flush=True)
        verify_build(build_dir)
        report = {"suite": args.suite, "build": identity, "device": args.device,
                  "warmup": args.warmup, "repeats": args.repeats, "profile": args.profile,
                  "output_init": args.output_init, "profiler_backend": args.profiler_backend if args.profile else None,
                  "profile_samples": (args.profile_samples or args.repeats) if args.profile else None,
                  "profiler_replay": args.profiler_replay if args.profile and args.profiler_backend == "operator" else None,
                  "profiler_warmup": args.profiler_warmup if args.profile and args.profiler_backend == "operator" else None,
                  "cases": records, "passed": bool(records) and all(row["status"] == "Pass" for row in records)}
        store_text(output / "result.json", json.dumps(report, indent=2, allow_nan=False) + "\n")
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
