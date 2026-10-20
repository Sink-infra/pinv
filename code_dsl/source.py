"""Source identity shared by DSL compilation, export and test validation."""

from hashlib import sha256
from pathlib import Path


def dsl_digest():
    root = Path(__file__).resolve().parent
    paths = sorted(path for folder in ("op_kernel", "op_host")
                   for path in (root / folder).rglob("*")
                   if path.is_file() and path.suffix in (".py", ".h", ".cpp"))
    digest = sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()
