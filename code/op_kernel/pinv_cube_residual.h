#pragma once
#include "kernel_operator.h"
#include "linalg_pinv_tiling.h"
using namespace AscendC;
#if defined(TILING_KEY_VAR) && (TILING_KEY_VAR & 32768UL)
#define ASCENDC_CUBE_ONLY
#include "lib/matmul_intf.h"
#endif
#if defined(TILING_KEY_VAR) && (TILING_KEY_VAR & 32768UL) && (TILING_KEY_VAR & 16384UL)
__aicore__ inline __gm__ float *PinvCubeArea(GM_ADDR ws,const LinalgPinvTilingData &t){
    uint64_t count=(uint64_t(2)*t.batch*t.cols*t.stride+uint64_t(t.batch)*8192+127)/128*128;
    return reinterpret_cast<__gm__ float*>(ws)+count;
}
__aicore__ inline void PinvCubeService(GM_ADDR ws,GM_ADDR inverse,const LinalgPinvCubeTilingData &data){
    auto base=PinvCubeArea(ws,data.base);uint32_t n=data.base.cols;
    GlobalTensor<int32_t> control;control.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(base));
    for(;;){
        CrossCoreWaitFlag<2>(8);
        DataCacheCleanAndInvalid<int32_t,CacheLine::SINGLE_CACHE_LINE,DcciDst::CACHELINE_OUT>(control);
        int32_t command=control.GetValue(0);
        if(command==0){PipeBarrier<PIPE_ALL>();CrossCoreSetFlag<2,PIPE_FIX>(10);return;}
        if(command==1){
            GlobalTensor<int8_t> a,b;GlobalTensor<int32_t> c;
            a.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(base+512));
            b.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(base+512)+5*n*n);
            c.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(base+512)+5*n*n/2);
            TPipe pipe;
            using A=matmul::MatmulType<TPosition::GM,CubeFormat::ND,int8_t>;
            using B=matmul::MatmulType<TPosition::GM,CubeFormat::ND,int8_t>;
            using C=matmul::MatmulType<TPosition::GM,CubeFormat::ND,int32_t>;
            matmul::Matmul<A,B,C,C,CFG_MDL> mm;mm.Init(&data.cube,&pipe);
            uint32_t extent=5*n,tile=((extent+GetBlockNum()-1)/GetBlockNum()+15)/16*16;
            uint32_t row=GetBlockIdx()*tile;
            if(row<extent){
                mm.SetOrgShape(extent,extent,n);mm.SetSingleShape(extent-row<tile?extent-row:tile,extent,n);
                mm.SetTensorA(a[row*n]);mm.SetTensorB(b);
                mm.IterateAll(c[row*extent]);mm.End();
            }
            PipeBarrier<PIPE_ALL>();pipe.Destroy();
        }else{
            uint32_t batch=static_cast<uint32_t>(control.GetValue(1));
            GlobalTensor<float> a,b,c;
            a.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(inverse)+uint64_t(batch)*n*n);
            b.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ws)+uint64_t(data.base.batch+batch)*n*data.base.stride);
            c.SetGlobalBuffer(base+512+5*n*n/2);
            TPipe pipe;
            using A=matmul::MatmulType<TPosition::GM,CubeFormat::ND,float>;
            using B=matmul::MatmulType<TPosition::GM,CubeFormat::ND,float>;
            using C=matmul::MatmulType<TPosition::GM,CubeFormat::ND,float>;
            matmul::Matmul<A,B,C,C,CFG_MDL> mm;mm.Init(&data.correction,&pipe);mm.SetHF32(false);
            uint32_t tile=((n+GetBlockNum()-1)/GetBlockNum()+15)/16*16,row=GetBlockIdx()*tile;
            if(row<n){
                mm.SetOrgShape(n,n,n);mm.SetSingleShape(n-row<tile?n-row:tile,n,n);
                mm.SetTensorA(a[row*n]);mm.SetTensorB(b);mm.IterateAll(c[row*n]);mm.End();
            }
            PipeBarrier<PIPE_ALL>();pipe.Destroy();
        }
        PipeBarrier<PIPE_ALL>();CrossCoreSetFlag<2,PIPE_FIX>(10);
    }
}
__aicore__ inline void PinvCubeCorrect(GM_ADDR ws,const LinalgPinvTilingData &t,uint32_t batch,
    LocalTensor<float> out,uint32_t owned,uint32_t core,uint32_t cores){
    auto base=PinvCubeArea(ws,t);uint32_t n=t.cols;GlobalTensor<int32_t> control;GlobalTensor<float> product;
    control.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(base));product.SetGlobalBuffer(base+512+5*n*n/2);
    if(core==0){
        control.SetValue(0,2);control.SetValue(1,static_cast<int32_t>(batch));
        DataCacheCleanAndInvalid<int32_t,CacheLine::SINGLE_CACHE_LINE,DcciDst::CACHELINE_OUT>(control);
    }
    SyncAll();PipeBarrier<PIPE_ALL>();CrossCoreSetFlag<2,PIPE_MTE3>(8);CrossCoreWaitFlag<2>(10);SyncAll();
    SetFlag<HardEvent::V_MTE2>(0);WaitFlag<HardEvent::V_MTE2>(0);
    DataCopyPad(out,product[core*n],DataCopyExtParams{static_cast<uint16_t>(owned),n*4,(cores-1)*n*4,0,0},DataCopyPadExtParams<float>{false,0,0,0});
    SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
}
__aicore__ inline void PinvCubeStop(GM_ADDR ws,const LinalgPinvTilingData &data){
    GlobalTensor<int32_t> control;control.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(PinvCubeArea(ws,data)));
    if(GetBlockIdx()==0){
        control.SetValue(0,0);
        DataCacheCleanAndInvalid<int32_t,CacheLine::SINGLE_CACHE_LINE,DcciDst::CACHELINE_OUT>(control);
    }
    SyncAll();PipeBarrier<PIPE_ALL>();CrossCoreSetFlag<2,PIPE_MTE3>(8);CrossCoreWaitFlag<2>(10);
}
#include "pinv_cube_digits.h"
__aicore__ inline void PinvCubeBuildResidual(GM_ADDR ws,const LinalgPinvTilingData &data,
    GlobalTensor<float> inverse,uint32_t offset,LocalTensor<float> cache,
    LocalTensor<float> high,LocalTensor<float> low,LocalTensor<float> scratch[],
    uint32_t owned,uint32_t core,uint32_t cores,float &inverseScale,float &sumErrorA,float &sumErrorX){
    uint32_t n=data.cols,count=owned*n;auto base=PinvCubeArea(ws,data);
    GlobalTensor<int8_t> a,b;GlobalTensor<int32_t> c,control;
    a.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(base+512));
    b.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(base+512)+5*n*n);
    c.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(base+512)+5*n*n/2);
    control.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(base));
    float localErrorA=PinvCubeQuantize<false>(cache,scratch,a,n,owned,core,cores,8.0f);
    SetFlag<HardEvent::V_MTE2>(0);WaitFlag<HardEvent::V_MTE2>(0);
    DataCopyPad(cache,inverse[offset+core*n],
        DataCopyExtParams{static_cast<uint16_t>(owned),n*4,(cores-1)*n*4,0,0},
        DataCopyPadExtParams<float>{false,0,0,0});
    SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
    inverseScale=PinvCubePeakScale(base,cache,scratch,count,core,cores,inverseScale);
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
    float rowWeight=8.0f*inverseScale/65536.0f;
    for(uint32_t i=0;i<5;++i){
        float weight=rowWeight;
        for(uint32_t j=0;j<5;++j){
            SetFlag<HardEvent::V_MTE2>(0);WaitFlag<HardEvent::V_MTE2>(0);
            DataCopyPad(raw,c[(i*n+core)*5*n+j*n],
                DataCopyExtParams{static_cast<uint16_t>(owned),n*4,(cores*5-1)*n*4,0,0},
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
            weight*=0.00390625f;
        }
        rowWeight*=0.00390625f;
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
#if defined(TILING_KEY_VAR) && (TILING_KEY_VAR & 32768UL) && (TILING_KEY_VAR & 16384UL)
        if constexpr(ENABLE)PinvCubeStop(ws,data);
#endif
    }
};
