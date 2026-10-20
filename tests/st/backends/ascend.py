"""CANN package backend for the shared suite runner."""
from pathlib import Path

NAME = "ascend"
KERNEL_PREFIX = "LinalgPinv"


def command(build_dir):
    return [str(Path(build_dir) / "tests/linalg_pinv_runner")]


def build(build_dir, jobs=8):
    from tests.st.scripts.build import build_current
    return build_current(build_dir, jobs)


def verify(build_dir):
    from tests.st.scripts.build import verify_build
    return verify_build(build_dir)
