#if defined(TILING_KEY_VAR) && (TILING_KEY_VAR & 4096UL)
#define K_MAX_SHAPE_DIM 0
#endif
#include "kernel_operator.h"
#include "linalg_pinv_tiling.h"
using namespace AscendC;
#include "pinv_cube_residual.h"
#if defined(TILING_KEY_VAR) && (TILING_KEY_VAR & 32768UL)
__aicore__ inline uint32_t PinvVectorCount(){return GetBlockNum()*2;}
#define GetBlockNum PinvVectorCount
#endif
#include "pinv_pipeline.h"
#include "pinv_peer_full.h"
#include "pinv_window_pipeline.h"
#include "tiling_key_linalg_pinv.h"
#include "pinv_stable.h"
#include "pinv_local.h"
#include "pinv_compact.h"
#include "pinv_compact_aligned.h"
#include "pinv_compact_quick.h"
template <typename DT_A, bool COMPACT, bool TAIL, bool WAVE, bool LOCAL, bool GROUP, bool PIPELINE, bool WINDOW, bool CUBE>
__global__ __aicore__ void linalg_pinv(GM_ADDR A, GM_ADDR A_pinv, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    REGISTER_TILING_DEFAULT(LinalgPinvCubeTilingData);
    GET_TILING_DATA_WITH_STRUCT(LinalgPinvTilingData, data, tiling);
    if constexpr(CUBE){
#if defined(TILING_KEY_VAR) && (TILING_KEY_VAR & 32768UL)
#if (TILING_KEY_VAR & 16384UL)
        GET_TILING_DATA_WITH_STRUCT(LinalgPinvCubeTilingData,cubeData,tiling);
        if ASCEND_IS_AIC { PinvCubeService(GetUserWorkspace(workspace),A_pinv,cubeData);return; }
#else
        if ASCEND_IS_AIC {return;}
#endif
#endif
    }
    PinvCubeGuard<CUBE && WINDOW> cubeGuard((CUBE && WINDOW) ? GetUserWorkspace(workspace) : nullptr,data);
    uint32_t windowState=0;
    if constexpr (PIPELINE) {
        if constexpr (WINDOW) {
            windowState=TryWindowPipelinePinv(A,A_pinv,GetUserWorkspace(workspace),data);
            if((windowState&15)==(uint32_t(1)<<data.batch)-1)return;
            data.local=0;
        }else{
#if defined(TILING_KEY_VAR) && TILING_KEY_VAR==45056UL
            if(TryPeerFullPinv(A,A_pinv,GetUserWorkspace(workspace),data))return;
#else
            if(TryPipelinePinv(A,A_pinv,GetUserWorkspace(workspace),data))return;
#endif
        }
    }
    if constexpr (COMPACT || LOCAL) if (GetBlockIdx() >= data.batch) return;
    if constexpr (COMPACT && !TAIL) {
        if(data.rows>8 && data.rows<=16 && data.cols%8==0)
            if(TryCompactQuickLight<16>(A,A_pinv,data))return;
    }
    TPipe pipe;
    if constexpr (COMPACT) {
        data.parallel = 0; data.local = 1; data.localInverse = 1;
        if constexpr (TAIL) {
            if (data.rows > 8 && data.rows <= 16) {
                Pinv<TAIL, 16> op; op.Init(A, A_pinv, GetUserWorkspace(workspace), data, &pipe); op.Process();
            } else {
                Pinv<TAIL> op; op.Init(A, A_pinv, GetUserWorkspace(workspace), data, &pipe); op.Process();
            }
        } else {
            if (data.rows > 8 && data.rows <= 16) {
                Pinv<TAIL, 16> op; op.Init(A, A_pinv, GetUserWorkspace(workspace), data, &pipe); op.Process();
            } else {
                Pinv<TAIL> op; op.Init(A, A_pinv, GetUserWorkspace(workspace), data, &pipe); op.Process();
            }
        }
    } else {
        if constexpr (WAVE) { data.parallel = 1; data.local = 0; data.localInverse = 0; }
        if constexpr (LOCAL) { data.parallel = 0; data.groupCount = 0; data.localInverse = 1; }
        if constexpr (LOCAL) {
            LocalPinv<WAVE, LOCAL> op; op.Init(A, A_pinv, GetUserWorkspace(workspace), data, &pipe); op.Process();
        } else {
            StablePinv<WAVE, LOCAL, GROUP> op; op.Init(A, A_pinv, GetUserWorkspace(workspace), data, &pipe);
            if constexpr (WINDOW) op.ProcessPrepared(windowState);
            else op.Process();
        }
    }
}
