#pragma once
#include "kernel_operator.h"
#include "linalg_pinv_tiling.h"
using namespace AscendC;
template <uint32_t PAD>
class CompactQuickLight {
    TBuf<TPosition::VECCALC> mem;
    LocalTensor<float> matrix, lowMatrix, u, prod, tmp, red, gather, norms;
    GlobalTensor<float> input,result;
    float atol,rtol;
    __aicore__ inline void VS(){SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);}
    __aicore__ inline void SV(){SetFlag<HardEvent::S_V>(0);WaitFlag<HardEvent::S_V>(0);}
    __aicore__ inline float Root(float value){red.SetValue(0,value);SV();Sqrt(red,red,1);VS();return red.GetValue(0);}
public:
    __aicore__ inline void Init(GM_ADDR in, GM_ADDR dst, const LinalgPinvTilingData &data, TPipe &pipe){
        constexpr uint32_t pitch=2*PAD+8,count=PAD*pitch;
        pipe.InitBuffer(mem,(2*count+pitch+3*PAD+64)*sizeof(float));
        auto all=mem.Get<float>();uint32_t off=0;
        matrix=all;off+=count;lowMatrix=all[off];off+=count;u=all[off];off+=pitch;
        prod=all[off];off+=PAD;tmp=all[off];off+=32;red=all[off];off+=32;
        gather=all[off];off+=PAD;norms=all[off];
        input.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(in));
        result.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dst));
        atol=data.atol;rtol=data.rtol;
    }
    __aicore__ inline bool RunAligned(uint32_t offset) {
        constexpr uint32_t n = PAD, width = 2 * PAD, pitch = 2 * PAD + 8;
        constexpr uint32_t count = n * pitch;
        Duplicate(matrix, 0.0f, count); PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE2>(0); WaitFlag<HardEvent::V_MTE2>(0);
        DataCopyPad(matrix, input[offset], DataCopyExtParams{static_cast<uint16_t>(n), n * 4, 0, (pitch - n) / 8, 0}, DataCopyPadExtParams<float>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_V>(0); WaitFlag<HardEvent::MTE2_V>(0);
        Abs(lowMatrix, matrix, count); PipeBarrier<PIPE_V>();
        ReduceMax(red, lowMatrix, tmp, count, false); VS();
        float scale = red.GetValue(0);
        if (!(scale >= 0.1f && scale < 1.7014118e38f)) return false;
        float power = 1.0f;
        while (power < scale) power *= 2.0f;
        while (power * 0.5f >= scale) power *= 0.5f;
        Muls(matrix, matrix, 1.0f / power, count); PipeBarrier<PIPE_V>();
        Mul(lowMatrix, matrix, matrix, count); PipeBarrier<PIPE_V>();
        ReduceSum(red, lowMatrix, tmp, count); VS();
        float total = red.GetValue(0);
        auto values = reinterpret_cast<__ubuf__ float *>(matrix.GetPhyAddr());
        for (uint32_t i = 0; i < n; ++i) values[i * pitch + n + i] = 1.0f;
        auto indices = gather.ReinterpretCast<uint32_t>();
        uint32_t order[PAD];
        for (uint32_t i = 0; i < n; ++i) indices.SetValue(i, i * pitch * sizeof(float));
        uint64_t available = (uint64_t(1) << n) - 1;
        SV();
        for (uint32_t column = 0; column < n; ++column) {
            Gather(norms, matrix, indices, column * sizeof(float), n); PipeBarrier<PIPE_V>();
            Abs(prod, norms, n); PipeBarrier<PIPE_V>();
            uint64_t mask[2] = {available, 0};
            WholeReduceMax(red, prod, mask, 1, 1, 1, 8); VS();
            float best, pivotBits;
            GetReduceRepeatMaxMinSpr<float>(best, pivotBits);
            uint32_t pivot = *reinterpret_cast<uint32_t *>(&pivotBits);
            if (!(best > 0.0f)) return false;
            available &= ~(uint64_t(1) << pivot);
            order[pivot] = column;
            Muls(matrix[pivot * pitch], matrix[pivot * pitch], -1.0f / values[pivot * pitch + column], width); PipeBarrier<PIPE_V>();
            norms.SetValue(pivot, 0.0f); SV();
            Brcb(lowMatrix, norms, n / 8, {static_cast<uint8_t>(pitch / 8), static_cast<uint8_t>(pitch)}); PipeBarrier<PIPE_V>();
            MulAddDst(matrix, matrix[pivot * pitch], lowMatrix, static_cast<uint64_t>(width), static_cast<uint8_t>(n), {1, 1, 0, static_cast<uint8_t>(pitch / 8), 0, static_cast<uint8_t>(pitch / 8)});
            PipeBarrier<PIPE_V>();
        }
        Duplicate(lowMatrix, 0.0f, count); PipeBarrier<PIPE_V>();
        Mul(lowMatrix, matrix[n], matrix[n], static_cast<uint64_t>(n), static_cast<uint8_t>(n), {1, 1, 1, static_cast<uint8_t>(pitch / 8), static_cast<uint8_t>(pitch / 8), static_cast<uint8_t>(pitch / 8)});
        PipeBarrier<PIPE_V>(); ReduceSum(red, lowMatrix, tmp, count); VS();
        float inverseTotal = red.GetValue(0), cutoff = Root(total) * rtol;
        float absolute = atol / power; cutoff = cutoff > absolute ? cutoff : absolute;
        if (!(inverseTotal > 0.0f && inverseTotal * cutoff * cutoff < 0.81f)) return false;
        Muls(matrix, matrix, -1.0f / power, count); PipeBarrier<PIPE_V>();
        for (uint32_t row = 0; row < n; ++row) {
            uint32_t target = order[row];
            DataCopy(lowMatrix[target * pitch], matrix[row * pitch], pitch);
        }
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(0); WaitFlag<HardEvent::V_MTE3>(0);
        SetFlag<HardEvent::S_MTE3>(0); WaitFlag<HardEvent::S_MTE3>(0);
        DataCopyPad(result[offset], lowMatrix[n], DataCopyExtParams{static_cast<uint16_t>(n), n * 4, (pitch - n) / 8, 0, 0});
        SetFlag<HardEvent::MTE3_V>(0); WaitFlag<HardEvent::MTE3_V>(0);
        SetFlag<HardEvent::MTE3_MTE2>(0); WaitFlag<HardEvent::MTE3_MTE2>(0);
        return true;
    }
    __aicore__ inline bool Run(uint32_t offset){
        return RunAligned(offset);
    }
};
template <uint32_t PAD>
__aicore__ inline bool TryCompactQuickLight(GM_ADDR in,GM_ADDR dst,const LinalgPinvTilingData &data){
    TPipe pipe;CompactQuickLight<PAD> op;op.Init(in,dst,data,pipe);
    bool valid=op.Run(GetBlockIdx()*data.m*data.n);
    PipeBarrier<PIPE_ALL>();pipe.Destroy();return valid;
}
