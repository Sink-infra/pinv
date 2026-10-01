"""Construct dense float32 fixtures from fixed scalar constraints."""

import math
import numpy as np


def sample_matrix(rng, n, first, positive, target):
    for attempt in range(100):
        a = rng.normal(size=(n,n)).astype(np.float64)
        u,s,v = np.linalg.svd(a, full_matrices=False)
        s = np.maximum(s, s[0] / max(4.0, target / 4))
        a = (u * s) @ v
        if positive:
            idx = np.arange(n)
            a[idx,idx] = np.abs(a[idx,idx]) + 0.5
        a[0,0] = first
        if np.linalg.cond(a) < target:
            return a
    raise RuntimeError('No suitable dense matrix')

def tune_condition(a, target):
    u,_,_ = np.linalg.svd(a, full_matrices=False)
    vec = u[:,-1].copy()
    vec[-1] = 0
    denom = float(np.linalg.solve(a,vec)[-1])
    if abs(denom) < 1e-12:
        raise RuntimeError('Degenerate condition direction')
    pole = -1 / denom
    lo,hi = 0.0,1.0
    for _ in range(30):
        mid = (lo+hi)/2
        b = a.copy()
        b[:,-1] += mid*pole*vec
        if np.linalg.cond(b) < target:
            lo = mid
        else:
            hi = mid
    b = a.copy()
    b[:,-1] += ((lo+hi)/2)*pole*vec
    return b

def tune_inverse_zeros(a, count):
    g = np.linalg.inv(a)
    idx = np.argsort(np.abs(g).reshape(-1))
    flat = g.reshape(-1).copy()
    flat[idx[:count]] = 0
    if count < flat.size:
        pos = idx[count:]
        floor = max(0.003, float(np.median(np.abs(flat))) * 0.01)
        small = pos[np.abs(flat[pos]) < floor]
        flat[small] = np.where(flat[small] < 0, -floor, floor)
    b = np.linalg.inv(flat.reshape(g.shape))
    factor = a[0,0] / b[0,0]
    if factor <= 0:
        raise RuntimeError('Sign changed during inverse transformation')
    b *= factor
    b = b.astype(np.float32)
    b[0,0] = np.float32(a[0,0])
    return b

def make_fixture(rng, shape, recipe):
    m, n = shape[-2:]
    batch = math.prod(shape[:-2])
    if m != n or not recipe:
        raise ValueError("square shapes and fixture parameters are required")
    conditions = recipe["batch_conditions"]
    counts = recipe["inverse_near_zero_counts"]
    if len(conditions) != batch or len(counts) != batch:
        raise ValueError("recipe batch count differs")
    first = np.array([int(recipe["first_word_hex"], 16)], dtype=np.uint32).view(np.float32)[0].item()
    positive = recipe["first_batch_positive_diagonal"]
    seeds = recipe.get("batch_seeds", [None] * batch)
    scales = recipe.get("batch_scales", [1.0] * batch)
    if len(seeds) != batch or len(scales) != batch:
        raise ValueError("recipe batch count differs")
    changes = recipe.get("batch_adjustments", [[] for _ in range(batch)])
    if len(changes) != batch:
        raise ValueError("recipe adjustment count differs")
    matrices = []
    for index, (target, count) in enumerate(zip(conditions, counts)):
        if not math.isfinite(target) or target < 2 or not isinstance(count, int) or not 0 <= count < n * n:
            raise ValueError("invalid fixture constraint")
        value = first if index == 0 else (5.0 if positive else -1.0)
        current = np.random.default_rng(seeds[index]) if seeds[index] is not None else rng
        a = sample_matrix(current, n, value, positive, target)
        a = tune_condition(a, target)
        scale = scales[index]
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("invalid batch scale")
        stored = (tune_inverse_zeros(a, count) * np.float32(scale)).astype(np.float32)
        for change in changes[index]:
            row, column = change["row"], change["column"]
            if not 0 <= row < n or not 0 <= column < n or (index == 0 and row == 0 and column == 0):
                raise ValueError("invalid input adjustment")
            stored[row, column] = np.array([int(change["value_hex"], 16)], dtype=np.uint32).view(np.float32)[0]
        matrices.append(stored)
    return np.stack(matrices).reshape(shape)
