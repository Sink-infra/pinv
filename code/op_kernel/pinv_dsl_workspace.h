#pragma once
#include "pinv_contract.h"
// Shape/ABI planning only. No input values or numerical decisions.
#if defined(__NPU_ARCH__)
#define PV_STAGE_LAYOUT_FN __aicore__ inline
#else
#define PV_STAGE_LAYOUT_FN inline
#endif
namespace pinv_dsl {
PV_STAGE_LAYOUT_FN uint64_t WorkFloats(uint32_t batch,uint32_t rows,uint32_t cols,uint32_t stride) {
    uint64_t rect=pinv::RectWorkspaceFloats(batch,rows,cols);
    uint64_t peer=batch*(2*uint64_t(cols)*stride+pinv::kMetadataFloats);
    uint64_t words=rect>peer?rect:peer;
    return (words+127)/128*128;
}
}
#undef PV_STAGE_LAYOUT_FN
