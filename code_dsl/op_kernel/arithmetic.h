#pragma once
#include "kernel_operator.h"
namespace tl::ascend {
// A uint64 register transports two FP32 words; no FP64 arithmetic is used.
struct PvWide { float h,l; };
union PvWideWord { PvWide pair; uint64_t word; };
__aicore__ inline uint64_t pv_dd_pack(float h,float l){PvWideWord x;x.pair={h,l};return x.word;}
__aicore__ inline float pv_dd_h(uint64_t a){PvWideWord x;x.word=a;return x.pair.h;}
__aicore__ inline float pv_dd_l(uint64_t a){PvWideWord x;x.word=a;return x.pair.l;}
__aicore__ inline uint64_t pv_dd_plus(uint64_t left,uint64_t right){
    PvWideWord x,y;x.word=left;y.word=right;PvWide a=x.pair,b=y.pair;
    volatile float sum=a.h+b.h;
    volatile float bv=sum-a.h,av=sum-bv,br=b.h-bv,ar=a.h-av;
    volatile float e=ar+br;e=e+a.l;e=e+b.l;
    volatile float h=sum+e,z=h-sum;
    return pv_dd_pack(h,e-z);
}
__aicore__ inline uint64_t pv_dd_minus(uint64_t a,uint64_t b){return pv_dd_plus(a,pv_dd_pack(-pv_dd_h(b),-pv_dd_l(b)));}
__aicore__ inline uint64_t pv_dd_times(uint64_t left,uint64_t right){
    PvWideWord x,y;x.word=left;y.word=right;PvWide a=x.pair,b=y.pair;
    volatile float ahc=4097.0f*a.h,bhc=4097.0f*b.h;
    volatile float ahv=ahc-a.h,bhv=bhc-b.h;
    volatile float ah=ahc-ahv,bh=bhc-bhv,al=a.h-ah,bl=b.h-bh;
    volatile float p=a.h*b.h,e=ah*bh;e=e-p;
    volatile float x1=ah*bl;e=e+x1;
    volatile float x2=al*bh;e=e+x2;
    volatile float x3=al*bl;e=e+x3;
    volatile float x4=a.h*b.l;e=e+x4;
    volatile float x5=a.l*b.h;e=e+x5;
    return pv_dd_plus(pv_dd_pack(p,0),pv_dd_pack(e,0));
}
__aicore__ inline uint64_t pv_dd_divide(uint64_t a,uint64_t b){
    float q=pv_dd_h(a)/pv_dd_h(b);
    uint64_t residual=pv_dd_minus(a,pv_dd_times(b,pv_dd_pack(q,0)));
    return pv_dd_plus(pv_dd_pack(q,0),pv_dd_pack(pv_dd_h(residual)/pv_dd_h(b),0));
}
}
// Scalar arithmetic results use explicit FP32-pair buffers because the
// pinned backend's custom instruction hook is a statement, not an expression.
namespace tl::ascend {
__aicore__ inline uint64_t PvReadPair(AscendC::LocalTensor<float> a){return pv_dd_pack(a.GetValue(0),a.GetValue(1));}
__aicore__ inline void PvWritePair(AscendC::LocalTensor<float> d,uint64_t a){d.SetValue(0,pv_dd_h(a));d.SetValue(1,pv_dd_l(a));}
__aicore__ inline void pv_pair_pack(AscendC::LocalTensor<float> d,AscendC::LocalTensor<float>,AscendC::LocalTensor<float>,float h,float l){d.SetValue(0,h);d.SetValue(1,l);}
__aicore__ inline void pv_pair_plus(AscendC::LocalTensor<float> d,AscendC::LocalTensor<float> a,AscendC::LocalTensor<float> b){PvWritePair(d,pv_dd_plus(PvReadPair(a),PvReadPair(b)));}
__aicore__ inline void pv_pair_minus(AscendC::LocalTensor<float> d,AscendC::LocalTensor<float> a,AscendC::LocalTensor<float> b){PvWritePair(d,pv_dd_minus(PvReadPair(a),PvReadPair(b)));}
__aicore__ inline void pv_pair_times(AscendC::LocalTensor<float> d,AscendC::LocalTensor<float> a,AscendC::LocalTensor<float> b){PvWritePair(d,pv_dd_times(PvReadPair(a),PvReadPair(b)));}
__aicore__ inline void pv_pair_divide(AscendC::LocalTensor<float> d,AscendC::LocalTensor<float> a,AscendC::LocalTensor<float> b){PvWritePair(d,pv_dd_divide(PvReadPair(a),PvReadPair(b)));}
}
