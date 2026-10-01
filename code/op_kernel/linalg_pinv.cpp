// Kernel侧核函数实现
#include "kernel_operator.h"

#include "linalg_pinv_tiling.h"
#include "tiling_key_linalg_pinv.h"

template <class DT_A>
class KernelLinalgPinv {
public:
    __aicore__ inline KernelLinalgPinv() {}
    __aicore__ inline void Init(GM_ADDR A, GM_ADDR A_pinv, uint32_t length) {

    }
    __aicore__ inline void Process() {

    }
private:

};

template <typename DT_A>
 __global__ __aicore__ void linalg_pinv(GM_ADDR A, GM_ADDR A_pinv, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(LinalgPinvTilingData);
    GET_TILING_DATA_WITH_STRUCT(LinalgPinvTilingData, tiling_data, tiling);
    KernelLinalgPinv<DT_A> op;
    op.Init(A, A_pinv, tiling_data.length);
    op.Process();
}
