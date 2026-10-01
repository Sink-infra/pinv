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

def inverse_block_matrix(n, recipe):
    sizes = recipe["sizes"]
    regulars = recipe["regulars"]
    inverse_eigenvalues = recipe["inverse_eigenvalues"]
    coupling = recipe["coupling"]
    if not sizes or len(sizes) != len(regulars) or len(sizes) != len(inverse_eigenvalues):
        raise ValueError("inverse block parameter count differs")
    if any(not isinstance(size, int) or size < 1 for size in sizes) or sum(sizes) > n:
        raise ValueError("invalid inverse block sizes")
    if (any(not math.isfinite(value) or value <= 0 for value in regulars + inverse_eigenvalues)
            or not math.isfinite(coupling) or coupling < 0):
        raise ValueError("invalid inverse block spectrum")
    inverse = np.eye(n, dtype=np.float64)
    start = 0
    for size, regular, eigenvalue in zip(sizes, regulars, inverse_eigenvalues):
        block = np.full((size, size), (eigenvalue - 1 / regular) / size, dtype=np.float64)
        block[np.arange(size), np.arange(size)] += 1 / regular
        inverse[start:start + size, start:start + size] = block
        start += size
    dense = recipe.get("dense_tail")
    if dense is not None:
        size, lower, upper = dense["size"], dense["minimum"], dense["maximum"]
        if (not isinstance(size, int) or size < 1 or start + size > n
                or not math.isfinite(lower) or not math.isfinite(upper) or not 0 < lower <= upper):
            raise ValueError("invalid dense inverse block")
        q, _ = np.linalg.qr(np.random.default_rng(dense["seed"]).normal(size=(size, size)))
        inverse[start:start + size, start:start + size] = (q * np.linspace(lower, upper, size)) @ q.T
    inverse += coupling
    floor = recipe.get("entry_floor", 0.0)
    if not math.isfinite(floor) or floor < 0:
        raise ValueError("invalid inverse entry floor")
    small = np.abs(inverse) < floor
    inverse[small] = np.where(inverse[small] < 0, -floor, floor)
    if np.linalg.eigvalsh(inverse)[0] <= 0:
        raise ValueError("inverse block constraints must be positive definite")
    return np.linalg.inv(inverse).astype(np.float32)


def diagonal_similarity(matrix, diagonal):
    a = np.asarray(matrix)
    d = np.asarray(diagonal, dtype=np.float64)
    if (a.ndim != 2 or a.shape[0] != a.shape[1] or d.shape != (a.shape[0],)
            or not np.isfinite(d).all() or np.any(d <= 0)):
        raise ValueError("invalid similarity diagonal")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            result = (a.astype(np.float64) * d[:, None] / d[None, :]).astype(np.float32)
    except FloatingPointError as error:
        raise ValueError("nonfinite similarity matrix") from error
    if not np.isfinite(result).all():
        raise ValueError("nonfinite similarity matrix")
    return result


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
    blocks = recipe.get("batch_inverse_blocks", [None] * batch)
    if len(blocks) != batch:
        raise ValueError("inverse block batch count differs")
    diagonals = recipe.get("batch_similarity_diagonals", [None] * batch)
    if len(diagonals) != batch:
        raise ValueError("similarity batch count differs")
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
        if blocks[index] is not None:
            stored = (inverse_block_matrix(n, blocks[index]) * np.float32(scale)).astype(np.float32)
        for change in changes[index]:
            row, column = change["row"], change["column"]
            if not 0 <= row < n or not 0 <= column < n or (index == 0 and row == 0 and column == 0):
                raise ValueError("invalid input adjustment")
            stored[row, column] = np.array([int(change["value_hex"], 16)], dtype=np.uint32).view(np.float32)[0]
        if diagonals[index] is not None:
            stored = diagonal_similarity(stored, diagonals[index])
        matrices.append(stored)
    return np.stack(matrices).reshape(shape)
