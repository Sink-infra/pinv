"""Exact IEEE FP32 power-of-two normalization for validated fast paths.

The caller has checked 0.1 <= magnitude < 2**127. Rounding the mantissa
up to the exponent field therefore returns exactly the old loop's power,
including exact powers and their immediate FP32 neighbors.
"""

def ceil_power_bits(magnitude):
    import tilelang.language as T
    bits=T.reinterpret('uint32',magnitude)
    rounded=(bits+T.uint32(0x007fffff)) & T.uint32(0x7f800000)
    return rounded
