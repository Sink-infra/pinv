"""Build the current LinalgPinv operator and its linked device runner."""

import argparse
import json
import platform
import shutil
from pathlib import Path
import subprocess

from tests.common.tensor_io import store_text
from tests.st.scripts.runtime import DEFAULT_BUILD, ROOT, SDK, environment, sha256_file, source_digest, source_files

RUNNER_SOURCE = ROOT / "tests/st/runner/linalg_pinv_runner.cpp"
def standalone_files():
    source = ROOT / "code"
    return tuple(path.relative_to(source).as_posix() for path in source_files(source))

def tiling_path(build_dir):
    return Path(build_dir) / "tmp/vendors/custom/op_impl/ai_core/tbe/op_tiling/lib/linux" / platform.machine() / "libcust_opmaster_rt2.0.so"


def build_current(build_dir=DEFAULT_BUILD, jobs=8, compute_unit="ascend910b", source_layout="project"):
    if not 1 <= jobs <= 64:
        raise ValueError("jobs must be between 1 and 64")
    if source_layout not in ("project", "standalone"):
        raise ValueError("unknown source layout")
    build_dir = Path(build_dir).resolve()
    if build_dir == ROOT or (ROOT in build_dir.parents and build_dir != ROOT / "build" and ROOT / "build" not in build_dir.parents):
        raise ValueError("use a build directory outside tracked source directories")
    build_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = build_dir / "tests/build.json"
    receipt_path.parent.mkdir(exist_ok=True)
    receipt_path.unlink(missing_ok=True)
    before = source_digest()
    env = environment(build_dir)
    source = ROOT / "code"
    if source_layout == "standalone":
        source = build_dir / "tests/source/code"
        if source.exists():
            shutil.rmtree(source)
        for name in standalone_files():
            destination = source / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / "code" / name, destination)
    layout_files = {path.relative_to(source).as_posix(): sha256_file(path)
                    for path in source_files(source)}
    store_text(build_dir / "tests/source.json", json.dumps(
        {"layout": source_layout, "files": layout_files}, indent=2) + "\n")
    commands = [
        ["cmake", "-S", str(source), "-B", str(build_dir),
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
               "runner_sha256": sha256_file(runner),
               "source_layout": source_layout, "layout_files": layout_files}
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
    if "source_layout" in expected:
        layout = expected["source_layout"]
        if layout not in ("project", "standalone"):
            raise ValueError("unknown recorded source layout")
        source = ROOT / "code" if layout == "project" else build_dir / "tests/source/code"
        files = {path.relative_to(source).as_posix(): sha256_file(path)
                 for path in source_files(source)}
        if layout == "standalone" and (set(files) != set(standalone_files()) or any(
                sha256_file(ROOT / "code" / name) != files[name] for name in standalone_files())):
            raise ValueError("standalone source differs from the current source tree")
        actual.update(source_layout=layout, layout_files=files)
    if actual != expected:
        raise ValueError("build does not match the current source, SDK, runner, or library")
    return actual


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("ascend", "jit"), default="ascend")
    parser.add_argument("--resume", action="store_true", help="reuse verified JIT modules when retrying an interrupted build")
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--compute-unit", default="ascend910b")
    parser.add_argument("--source-layout", choices=("project", "standalone"), default="project")
    args = parser.parse_args(argv)
    try:
        if args.backend == "jit":
            from tests.st.backends.jit import build, DEFAULT_BUILD as JIT_BUILD
            if args.source_layout != "project":
                parser.error("--source-layout applies to the AscendC package backend")
            receipt = build(args.build_dir or JIT_BUILD, args.jobs, resume=args.resume)
        else:
            if args.resume:
                parser.error("--resume is specific to the JIT builder; CMake resumes Ascend builds automatically")
            receipt = build_current(args.build_dir or DEFAULT_BUILD, args.jobs, args.compute_unit, args.source_layout)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.error(str(error) + "; see the build directory's tests/build.log")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
