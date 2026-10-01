"""Generate the fixed Mock suite and independent references."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .mock_fixture import make_fixture
from .cases import load_document


def load_cases():
    return load_document("mock")["cases"]


def reference(a, atol=0.0, rtol=-1.0):
    m, n = a.shape[-2:]
    relative = np.float32(rtol).item()
    absolute = np.float32(atol).item()
    if relative < 0:
        relative = 0.0 if absolute > 0 else max(m, n) * 2.0 ** -23
    u, singular, vh = np.linalg.svd(a.astype(np.float64), full_matrices=False)
    threshold = np.maximum(absolute, singular[..., :1] * relative)
    inv = np.divide(1.0, singular, out=np.zeros_like(singular), where=singular > threshold)
    return (vh.swapaxes(-2, -1) * inv[..., None, :]) @ u.swapaxes(-2, -1)


def generate(destination, cases):
    if not cases or len({case["number"] for case in cases}) != len(cases):
        raise ValueError("a nonempty suite with unique case numbers is required")
    destination.mkdir(parents=True, exist_ok=False)
    manifest = []
    for case in cases:
        number = case["number"]
        shape = tuple(case["shape"])
        if case["dtype"] != "float32" or len(shape) < 2 or any(size <= 0 for size in shape):
            raise ValueError("positive float32 matrix shapes are required")
        seed = case.get("seed", 42 + number)
        a = make_fixture(np.random.default_rng(seed), shape, case["fixture"])
        atol, rtol = case.get("atol", 0.0), case.get("rtol", -1.0)
        gold = reference(a, atol, rtol)
        name = case.get("name", f"mock_{number:02d}")
        filename = f"case_{number:02d}"
        np.save(destination / (filename + ".npy"), a, allow_pickle=False)
        np.save(destination / (filename + "_reference.npy"), gold, allow_pickle=False)
        manifest.append(dict(number=number, name=name, shape=list(shape), dtype="float32",
                             atol=atol, rtol=rtol, seed=seed,
                             input=filename + ".npy", reference=filename + "_reference.npy",
                             input_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
                             reference_sha256=hashlib.sha256(gold.tobytes()).hexdigest()))
    with (destination / "manifest.json").open("w") as stream:
        json.dump(manifest, stream, indent=2)
    return manifest
