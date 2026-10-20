"""Retrieve the immutable native source from Git, without a checked-in copy."""
from contextlib import contextmanager
import io
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import tempfile

BASELINE_COMMIT = "ec5ec7a2b5c9555a46636e08d25d9d05f46c0df2"


@contextmanager
def native_source(path=None):
    if path is not None:
        source = Path(path).resolve()
        if not (source / "op_kernel/linalg_pinv.cpp").is_file():
            raise ValueError("native source directory is incomplete")
        yield source
        return
    root = Path(__file__).resolve().parents[2]
    archive = subprocess.run(["git", "archive", BASELINE_COMMIT, "code"], cwd=root,
                             check=True, capture_output=True).stdout
    with tempfile.TemporaryDirectory(prefix="pinv-native-") as temporary:
        destination = Path(temporary)
        with tarfile.open(fileobj=io.BytesIO(archive)) as package:
            for member in package.getmembers():
                parts = PurePosixPath(member.name).parts
                if not parts or parts[0] != "code" or ".." in parts or member.issym() or member.islnk():
                    raise ValueError("unexpected native archive entry")
                if member.isfile():
                    target = destination.joinpath(*parts[1:])
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(package.extractfile(member).read())
        yield destination
