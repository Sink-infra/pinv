"""CANN environment and content identities for the current source tree."""

import hashlib
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SDK = Path(os.environ.get("PINV_CANN_PATH", "/usr/local/Ascend/cann")).resolve()
DEFAULT_BUILD = ROOT / "build" / "tests" / "npu"


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_digest():
    digest = hashlib.sha256()
    for path in sorted((ROOT / "code").rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(ROOT).as_posix().encode() + b"\0")
            digest.update(path.read_bytes())
    return digest.hexdigest()


def environment(build=None):
    env = dict(os.environ)
    env["ASCEND_HOME_PATH"] = str(SDK)
    env["ASCEND_CANN_PACKAGE_PATH"] = str(SDK)
    env["ASCEND_AICPU_PATH"] = str(SDK)
    env["ASCEND_TOOLKIT_HOME"] = str(SDK)
    env["TOOLCHAIN_HOME"] = str(SDK / "toolkit")
    env["ASCEND_OPP_PATH"] = str(SDK / "opp")
    env["PATH"] = ":".join((str(SDK / "bin"), str(SDK / "aarch64-linux/bin"), str(SDK / "tools/ccec_compiler/bin"), str(SDK / "tools/profiler/bin"), env.get("PATH", "")))
    libraries = [str(SDK / "lib64"), "/usr/local/Ascend/driver/lib64", "/usr/local/Ascend/driver/lib64/driver",
                 "/usr/local/Ascend/driver/lib64/common"]
    if build is not None:
        build = Path(build).resolve()
        libraries[:0] = [str(build), str(build / "op_host")]
        vendor = build / "tmp" / "vendors" / "custom"
        if vendor.is_dir():
            env["ASCEND_CUSTOM_OPP_PATH"] = str(vendor)
    env["LD_LIBRARY_PATH"] = ":".join(libraries + [env.get("LD_LIBRARY_PATH", "")])
    env["PYTHONPATH"] = ":".join((str(ROOT), str(SDK / "python/site-packages"), env.get("PYTHONPATH", "")))
    env["OPENBLAS_NUM_THREADS"] = "2"
    env["OMP_NUM_THREADS"] = "2"
    return env
