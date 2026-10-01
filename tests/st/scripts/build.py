"""Build the current LinalgPinv operator and its linked device runner."""

import argparse
import json
import platform
import shutil
from pathlib import Path
import subprocess

from tests.common.tensor_io import store_text
from tests.st.scripts.runtime import DEFAULT_BUILD, ROOT, SDK, environment, sha256_file, source_digest

RUNNER_SOURCE = ROOT / "tests/st/runner/linalg_pinv_runner.cpp"

def tiling_path(build_dir):
    return Path(build_dir) / "tmp/vendors/custom/op_impl/ai_core/tbe/op_tiling/lib/linux" / platform.machine() / "libcust_opmaster_rt2.0.so"


def build_current(build_dir=DEFAULT_BUILD, jobs=8, compute_unit="ascend910b"):
    if not 1 <= jobs <= 64:
        raise ValueError("jobs must be between 1 and 64")
    build_dir = Path(build_dir).resolve()
    if build_dir == ROOT or (ROOT in build_dir.parents and build_dir != ROOT / "build" and ROOT / "build" not in build_dir.parents):
        raise ValueError("use a build directory outside tracked source directories")
    build_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = build_dir / "tests/build.json"
    receipt_path.parent.mkdir(exist_ok=True)
    receipt_path.unlink(missing_ok=True)
    before = source_digest()
    env = environment(build_dir)
    commands = [
        ["cmake", "-S", str(ROOT / "code"), "-B", str(build_dir),
         "-DCMAKE_BUILD_TYPE=Release", "-DASCEND_COMPUTE_UNIT=" + compute_unit,
         "-DASC_DIR=" + str(SDK / "lib64/cmake")],
        ["cmake", "--build", str(build_dir), "-j" + str(jobs)],
    ]
    with (build_dir / "tests/build.log").open("w") as log:
        for command in commands:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        tiling = tiling_path(build_dir)
        tiling.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(build_dir / "op_host/libcustom_ascendc_cust_optiling.so", tiling)
        headers = list(build_dir.rglob("aclnn_linalg_pinv.h"))
        if len(headers) != 1:
            raise ValueError("exactly one generated LinalgPinv API header is required")
        runner = build_dir / "tests/linalg_pinv_runner"
        command = ["c++", "-std=c++17", "-O2", str(RUNNER_SOURCE),
                   "-I" + str(SDK / "include"), "-I" + str(headers[0].parent),
                   "-L" + str(build_dir), "-L" + str(SDK / "lib64"),
                   "-Wl,-rpath," + str(build_dir), "-Wl,-rpath," + str(SDK / "lib64"),
                   "-lcust_opapi", "-lascendcl", "-lnnopbase", "-lmsprofiler", "-o", str(runner)]
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    if source_digest() != before:
        raise ValueError("source tree changed during build")
    library = build_dir / "libcust_opapi.so"
    receipt = {"source_sha256": before, "sdk": str(SDK), "compute_unit": compute_unit,
               "library_sha256": sha256_file(library),
               "tiling_sha256": sha256_file(tiling_path(build_dir)),
               "runner_source_sha256": sha256_file(RUNNER_SOURCE),
               "runner_sha256": sha256_file(runner)}
    store_text(receipt_path, json.dumps(receipt, indent=2) + "\n")
    return receipt


def verify_build(build_dir):
    build_dir = Path(build_dir).resolve()
    expected = json.loads((build_dir / "tests/build.json").read_text())
    actual = {"source_sha256": source_digest(), "sdk": str(SDK),
              "compute_unit": expected["compute_unit"],
              "library_sha256": sha256_file(build_dir / "libcust_opapi.so"),
              "tiling_sha256": sha256_file(tiling_path(build_dir)),
              "runner_source_sha256": sha256_file(RUNNER_SOURCE),
              "runner_sha256": sha256_file(build_dir / "tests/linalg_pinv_runner")}
    if actual != expected:
        raise ValueError("build does not match the current source, SDK, runner, or library")
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--compute-unit", default="ascend910b")
    args = parser.parse_args()
    try:
        receipt = build_current(args.build_dir, args.jobs, args.compute_unit)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.error(str(error) + "; see the build directory's tests/build.log")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
