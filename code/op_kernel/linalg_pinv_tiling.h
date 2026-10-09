#pragma once
#include <cstdint>
struct LinalgPinvTilingData {
    uint32_t m, n, batch, rows, cols, stride, local, parallel, localInverse, inverseTile, resident;
    float atol, rtol;
};
