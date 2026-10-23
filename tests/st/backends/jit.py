"""Build identity and runtime dispatch for real TileLang compiler outputs."""
import json
from pathlib import Path
import subprocess
import sys

from code_dsl.source import dsl_digest
from tests.st.scripts.runtime import ROOT, SDK, environment, sha256_file

NAME = "jit"
KERNEL_PREFIX = "linalg_pinv_tilelang_kernel"
DEFAULT_BUILD = ROOT / "build/tests/jit"
HARNESS_FILES = ("tests/st/scripts/build_jit.py", "tests/st/scripts/build_peer_jit.py", "tests/st/scripts/build_window_jit.py",
                 "tests/st/backends/jit.py", "tests/st/backends/jit_runner.py",
                 "code_dsl/op_host/planner.cpp", "code/op_kernel/linalg_pinv_tiling.h",
                 "code/op_kernel/pinv_contract.h")


class UnsupportedStrategy(ValueError):
    """An unported strategy must never silently run through the CANN backend."""


def select_strategy(plan):
    strategy = plan.strategy
    if strategy == "rectangular" or (strategy == "compact" and plan.geometry.n <= 32):
        return strategy
    if strategy == "local" and plan.geometry.n <= 64:
        return strategy
    if strategy == "peer_pipeline" and plan.geometry.m == plan.geometry.n == 200:
        batch=plan.geometry.batch
        if batch*13<=plan.aiv and (batch*13+1)//2<=plan.aic and plan.blocks==(batch*13+1)//2 and plan.fields["stride"]==456:
            return strategy
    if strategy == "window_cube":
        from code_dsl.op_host.recovery_layout import supported
        if supported(plan):return strategy
    raise UnsupportedStrategy(f"TileLang JIT has no validated {strategy} implementation for {plan.geometry.shape}")


def artifacts(build_dir, strategy):
    root = Path(build_dir) / {"peer_pipeline":"peer","window_cube":"window"}.get(strategy,"module")
    library = root / "kernel.so" if strategy in {"peer_pipeline","window_cube"} else root / strategy / "kernel.so"
    return library, root / "build-receipt.json"


def module_identity(build_dir):
    build_dir = Path(build_dir)
    libraries = {}
    for strategy in ("compact", "local", "rectangular", "peer_pipeline", "window_cube"):
        library, receipt = artifacts(build_dir, strategy)
        from tests.common.provenance import jit_provenance
        provenance = jit_provenance(library, receipt, strategy)
        document = json.loads(receipt.read_text())
        if provenance["dsl_sha256"] != dsl_digest():
            raise ValueError("JIT build does not match the current DSL sources")
        source = library.with_suffix(".cpp")
        if sha256_file(source) != provenance["source_sha256"]:
            raise ValueError("JIT generated source changed")
        if strategy in {"peer_pipeline","window_cube"} and document.get("complete_continuation") is not True:
            raise ValueError("communication-only probes cannot be tested as operators")
        libraries[strategy] = provenance
    return libraries


def identity(build_dir):
    build_dir = Path(build_dir)
    libraries = module_identity(build_dir)
    return {"backend": NAME, "dsl_sha256": dsl_digest(), "sdk": str(SDK),
            "harness_sha256": {name: sha256_file(ROOT / name) for name in HARNESS_FILES},
            "planner_sha256": sha256_file(build_dir / "host-planner.so"), "libraries": libraries}


def verify(build_dir):
    recorded = json.loads((Path(build_dir) / "tests/build.json").read_text())
    if recorded != identity(build_dir):
        raise ValueError("JIT build does not match the current DSL, harness, SDK or libraries")
    return recorded


def build(build_dir=DEFAULT_BUILD, jobs=8, *, resume=False):
    build_dir = Path(build_dir).resolve()
    if build_dir == ROOT or (ROOT in build_dir.parents and ROOT / "build" not in build_dir.parents):
        raise ValueError("JIT build directory must be outside tracked source directories")
    if build_dir.exists() and any(build_dir.iterdir()) and not resume:
        raise ValueError("JIT build directory must be empty; use --no-build to verify an existing build")
    build_dir.mkdir(parents=True, exist_ok=True)
    (build_dir / "tests").mkdir(exist_ok=True)
    (build_dir / "tests/build.json").unlink(missing_ok=True)
    env = environment(build_dir)
    commands = [[sys.executable, "-m", "tests.st.scripts." + module, "--output", str(build_dir / name)]
                for module, name in (("build_jit", "module"), ("build_peer_jit", "peer"), ("build_window_jit", "window"))
                if not (resume and (build_dir / name).exists())]
    commands.append(["c++", "-std=c++17", "-O2", "-shared", "-fPIC",
                     str(ROOT / "code_dsl/op_host/planner.cpp"), "-I" + str(SDK / "include"),
                     "-I" + str(SDK / "aarch64-linux/asc/include"),
                     "-I" + str(SDK / "include/experiment/runtime"),
                     "-I" + str(SDK / "include/experiment/metadef"),
                     "-I" + str(SDK / "include/experiment/platform"),
                     "-L" + str(SDK / "lib64"), "-Wl,-rpath," + str(SDK / "lib64"),
                     "-Wl,-z,defs", "-ltiling_api", "-lplatform", "-lregister", "-lunified_dlog", "-lc_sec",
                     "-o", str(build_dir / "host-planner.so")])
    with (build_dir / "tests/build.log").open("a" if resume else "w") as log:
        for argv in commands[:-1]:
            subprocess.run(argv, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        # A retry may reuse verified JIT outputs, never a stale or diagnostic library.
        module_identity(build_dir)
        subprocess.run(commands[-1], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    receipt = identity(build_dir)
    (build_dir / "tests/build.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def command(build_dir):
    return [sys.executable, "-m", "tests.st.backends.jit_runner", "--build-dir", str(build_dir)]
