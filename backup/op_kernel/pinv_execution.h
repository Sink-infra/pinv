#pragma once
#include "kernel_operator.h"
#include "pinv_contract.h"
// AIV uses two logical blocks per physical block in paired mixed kernels.
// Cube services always call AscendC::GetBlockNum directly.
__aicore__ inline uint32_t PinvVectorBlockCount() {
#if defined(TILING_KEY_VAR) && (TILING_KEY_VAR & PINV_KEY_CUBE)
    return AscendC::GetBlockNum() * 2;
#else
    return AscendC::GetBlockNum();
#endif
}
