"""Explicit subprocess transport; shared services never require monkeypatches."""
import os
import subprocess
from contextlib import contextmanager


class SessionExecutor:
    def __init__(self, session):
        self.session = session

    def __call__(self, command, *, env=None, cwd=None, stdout=None, stderr=None, timeout=600, check=False):
        try:
            result = self.session.run(command, environment=env if env is not None else os.environ.copy(), cwd=cwd, timeout=timeout)
        except TimeoutError as error:
            raise subprocess.TimeoutExpired(command, timeout) from error
        for target, content in ((stdout, result.stdout), (stdout if stderr == subprocess.STDOUT else stderr, result.stderr)):
            if hasattr(target, "write"):
                target.write(content)
        if check and result.returncode:
            raise subprocess.CalledProcessError(result.returncode, command, result.stdout, result.stderr)
        return result


@contextmanager
def execution_transport(name="direct", device=0):
    if name == "direct":
        yield None, {"transport": name}
    elif name == "npu-session":
        from npu_runtime import NpuSession
        with NpuSession(device=device, lock_timeout=1800) as session:
            yield SessionExecutor(session), {"transport": name, "service_pid": session.service_pid,
                                             "lock_wait_seconds": session.lock_wait_seconds}
    else:
        raise ValueError("unknown device transport")
