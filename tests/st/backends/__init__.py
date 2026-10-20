"""Device backends for the shared test lifecycle."""
from importlib import import_module


def get_backend(name):
    if name not in ("ascend", "jit"):
        raise ValueError("unknown device backend: " + name)
    return import_module("tests.st.backends." + name)
