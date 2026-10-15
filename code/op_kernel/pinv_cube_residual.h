#pragma once
#include "kernel_operator.h"
#include "linalg_pinv_tiling.h"
using namespace AscendC;
#if defined(TILING_KEY_VAR) && (TILING_KEY_VAR & 32768UL)
#define ASCENDC_CUBE_ONLY
#include "lib/matmul_intf.h"
__aicore__ inline __gm__ float *PinvCubeArea(GM_ADDR ws,const LinalgPinvTilingData &t){
    uint64_t count=(uint64_t(2)*t.batch*t.cols*t.stride+uint64_t(t.batch)*8192+127)/128*128;
    return reinterpret_cast<__gm__ float*>(ws)+count;
}
__aicore__ inline void PinvCubeService(GM_ADDR ws,const LinalgPinvCubeTilingData &data){
    auto base=PinvCubeArea(ws,data.base);uint32_t n=data.base.cols;
    GlobalTensor<int32_t> control;control.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(base));
    GlobalTensor<int8_t> a,b;GlobalTensor<int32_t> c;
    a.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(base+512));
    b.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(base+512)+6*n*n);
    c.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(base+512)+3*n*n);
    TPipe pipe;
    using A=matmul::MatmulType<TPosition::GM,CubeFormat::ND,int8_t>;
    using B=matmul::MatmulType<TPosition::GM,CubeFormat::ND,int8_t>;
    using C=matmul::MatmulType<TPosition::GM,CubeFormat::ND,int32_t>;
    matmul::Matmul<A,B,C,C,CFG_MDL> mm;mm.Init(&data.cube,&pipe);
    for(;;){
        CrossCoreWaitFlag<2>(8);
        DataCacheCleanAndInvalid<int32_t,CacheLine::SINGLE_CACHE_LINE,DcciDst::CACHELINE_OUT>(control);
        if(control.GetValue(0)==0){
            PipeBarrier<PIPE_ALL>();CrossCoreSetFlag<2,PIPE_FIX>(10);return;
        }
        uint32_t extent=6*n,tile=((extent+GetBlockNum()-1)/GetBlockNum()+15)/16*16;
        uint32_t row=GetBlockIdx()*tile;
        if(row<extent){
            mm.SetOrgShape(extent,extent,n);
            mm.SetSingleShape(extent-row<tile?extent-row:tile,extent,n);
            mm.SetTensorA(a[row*n]);mm.SetTensorB(b);
            mm.IterateAll(c[row*extent]);mm.End();
        }
        PipeBarrier<PIPE_ALL>();CrossCoreSetFlag<2,PIPE_FIX>(10);
    }
}
__aicore__ inline void PinvCubeStop(GM_ADDR ws,const LinalgPinvTilingData &data){
    GlobalTensor<int32_t> control;control.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(PinvCubeArea(ws,data)));
    if(GetBlockIdx()==0){
        control.SetValue(0,0);
        DataCacheCleanAndInvalid<int32_t,CacheLine::SINGLE_CACHE_LINE,DcciDst::CACHELINE_OUT>(control);
    }
    SyncAll();PipeBarrier<PIPE_ALL>();CrossCoreSetFlag<2,PIPE_MTE3>(8);CrossCoreWaitFlag<2>(10);
}
template<bool RIGHT>
__aicore__ inline float PinvCubeQuantize(LocalTensor<float> values,LocalTensor<float> scratch[],
    GlobalTensor<int8_t> digits,uint32_t n,uint32_t owned,uint32_t core,uint32_t cores,float scale){
    uint32_t count=owned*n;
    auto integers=scratch[0].ReinterpretCast<int32_t>();auto rounded=scratch[1];
    auto halfValues=scratch[2].ReinterpretCast<half>();auto small=scratch[3].ReinterpretCast<int8_t>();
    Muls(values,values,128.0f/scale,count);PipeBarrier<PIPE_V>();
    for(uint32_t limb=0;limb<6;++limb){
        Cast(integers,values,RoundMode::CAST_RINT,count);PipeBarrier<PIPE_V>();
        Cast(rounded,integers,RoundMode::CAST_RINT,count);PipeBarrier<PIPE_V>();
        Cast(halfValues,rounded,RoundMode::CAST_NONE,count);PipeBarrier<PIPE_V>();
        Cast(small,halfValues,RoundMode::CAST_RINT,count);
        SetFlag<HardEvent::V_MTE3>(0);WaitFlag<HardEvent::V_MTE3>(0);
        uint32_t destination=RIGHT?core*6*n+limb*n:limb*n*n+core*n;
        uint32_t step=RIGHT?(cores*6-1)*n:(cores-1)*n;
        DataCopyPad(digits[destination],small,
            DataCopyExtParams{static_cast<uint16_t>(owned),n,0,step,0});
        SetFlag<HardEvent::MTE3_V>(0);WaitFlag<HardEvent::MTE3_V>(0);
        Sub(values,values,rounded,count);PipeBarrier<PIPE_V>();
        Muls(values,values,128.0f,count);PipeBarrier<PIPE_V>();
    }
    Mul(scratch[0],values,values,count);PipeBarrier<PIPE_V>();
    ReduceSum(scratch[2],scratch[0],scratch[1],count);
    SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);
    return scratch[2].GetValue(0);
}
__aicore__ inline void PinvCubeBuildResidual(GM_ADDR ws,const LinalgPinvTilingData &data,
    GlobalTensor<float> inverse,uint32_t offset,LocalTensor<float> cache,
    LocalTensor<float> high,LocalTensor<float> low,LocalTensor<float> scratch[],
    uint32_t owned,uint32_t core,uint32_t cores,float inverseScale,float &sumErrorA,float &sumErrorX){
    uint32_t n=data.cols,count=owned*n;auto base=PinvCubeArea(ws,data);
    GlobalTensor<int8_t> a,b;GlobalTensor<int32_t> c,control;
    a.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(base+512));
    b.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(base+512)+6*n*n);
    c.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(base+512)+3*n*n);
    control.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(base));
    float localErrorA=PinvCubeQuantize<false>(cache,scratch,a,n,owned,core,cores,4.0f);
    SetFlag<HardEvent::V_MTE2>(0);WaitFlag<HardEvent::V_MTE2>(0);
    DataCopyPad(cache,inverse[offset+core*n],
        DataCopyExtParams{static_cast<uint16_t>(owned),n*4,(cores-1)*n*4,0,0},
        DataCopyPadExtParams<float>{false,0,0,0});
    SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
    float localErrorX=PinvCubeQuantize<true>(cache,scratch,b,n,owned,core,cores,inverseScale);
    auto statistic=scratch[7];statistic.SetValue(0,localErrorA);statistic.SetValue(1,localErrorX);
    GlobalTensor<float> statistics;statistics.SetGlobalBuffer(base);
    SetFlag<HardEvent::S_MTE3>(0);WaitFlag<HardEvent::S_MTE3>(0);
    DataCopyPad(statistics[32+core*8],statistic,DataCopyExtParams{1,8,0,0,0});
    SetFlag<HardEvent::MTE3_V>(0);WaitFlag<HardEvent::MTE3_V>(0);
    if(core==0){
        control.SetValue(0,1);
        DataCacheCleanAndInvalid<int32_t,CacheLine::SINGLE_CACHE_LINE,DcciDst::CACHELINE_OUT>(control);
    }
    SyncAll();PipeBarrier<PIPE_ALL>();CrossCoreSetFlag<2,PIPE_MTE3>(8);CrossCoreWaitFlag<2>(10);SyncAll();
    SetFlag<HardEvent::V_MTE2>(0);WaitFlag<HardEvent::V_MTE2>(0);
    DataCopy(statistic,statistics[32],cores*8);
    SetFlag<HardEvent::MTE2_S>(0);WaitFlag<HardEvent::MTE2_S>(0);
    sumErrorA=0.0f;sumErrorX=0.0f;
    for(uint32_t peer=0;peer<cores;++peer){sumErrorA+=statistic.GetValue(peer*8);sumErrorX+=statistic.GetValue(peer*8+1);}
    auto raw=scratch[0].ReinterpretCast<int32_t>();auto part=scratch[1],sum=scratch[2],bv=scratch[3];
    auto av=scratch[4],err=scratch[5],alt=scratch[6];
    float rowWeight=4.0f*inverseScale/16384.0f;
    for(uint32_t i=0;i<6;++i){
        float weight=rowWeight;
        for(uint32_t j=0;j<6;++j){
            SetFlag<HardEvent::V_MTE2>(0);WaitFlag<HardEvent::V_MTE2>(0);
            DataCopyPad(raw,c[(i*n+core)*6*n+j*n],
                DataCopyExtParams{static_cast<uint16_t>(owned),n*4,(cores*6-1)*n*4,0,0},
                DataCopyPadExtParams<int32_t>{false,0,0,0});
            SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
            Cast(part,raw,RoundMode::CAST_RINT,count);PipeBarrier<PIPE_V>();
            Muls(part,part,-weight,count);PipeBarrier<PIPE_V>();
            Add(sum,high,part,count);PipeBarrier<PIPE_V>();
            Sub(bv,sum,high,count);PipeBarrier<PIPE_V>();
            Sub(av,sum,bv,count);PipeBarrier<PIPE_V>();
            Sub(err,high,av,count);Sub(alt,part,bv,count);PipeBarrier<PIPE_V>();
            Add(err,err,alt,count);PipeBarrier<PIPE_V>();
            Add(low,low,err,count);DataCopy(high,sum,count);PipeBarrier<PIPE_V>();
            weight*=0.0078125f;
        }
        rowWeight*=0.0078125f;
    }
}
#endif
template<bool ENABLE>
class PinvCubeGuard{
    GM_ADDR ws;LinalgPinvTilingData data;
public:
    __aicore__ inline PinvCubeGuard(GM_ADDR workspace,const LinalgPinvTilingData &t){
        if constexpr(ENABLE){ws=workspace;data=t;}
    }
    __aicore__ inline ~PinvCubeGuard(){
#if defined(TILING_KEY_VAR) && (TILING_KEY_VAR & 32768UL)
        if constexpr(ENABLE)PinvCubeStop(ws,data);
#endif
    }
};
