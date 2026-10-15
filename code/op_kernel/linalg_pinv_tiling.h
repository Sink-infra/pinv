#pragma once
#include <cstdint>
// WINDOW templates carry the window width in local until the fallback restores zero.
struct LinalgPinvTilingData {
    uint32_t m, n, batch, rows, cols, stride, local, parallel, localInverse, inverseTile, resident, groupCount, singleResident;
    float atol, rtol;
};
