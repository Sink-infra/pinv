#pragma once
// Elementary compensated dot/scale/add primitives from ec5ec7a. All
// decomposition, truncation, certification and reconstruction control is DSL.
// This type has no inverse, SVD, Jacobi sweep or operator entry point.
#include "kernel_operator.h"
namespace tl::ascend {
class PvArithmetic {
public:
    using Wide=PvWide;
    AscendC::LocalTensor<float> pool,calc[8],red,bh,bl,certH,certL,certT;
    __aicore__ inline void VS(){AscendC::SetFlag<AscendC::HardEvent::V_S>(0);AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);}
    __aicore__ inline void SV(){AscendC::SetFlag<AscendC::HardEvent::S_V>(0);AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);}
    __aicore__ inline Wide Unpack(uint64_t a){PvWideWord x;x.word=a;return x.pair;}
    __aicore__ inline uint64_t Pack(Wide a){return pv_dd_pack(a.h,a.l);}
    __aicore__ inline Wide Plus(Wide a,Wide b){return Unpack(pv_dd_plus(Pack(a),Pack(b)));}
    __aicore__ inline Wide Minus(Wide a,Wide b){return Unpack(pv_dd_minus(Pack(a),Pack(b)));}
    __aicore__ inline Wide Times(Wide a,Wide b){return Unpack(pv_dd_times(Pack(a),Pack(b)));}
    __aicore__ inline Wide Divide(Wide a,Wide b){return Unpack(pv_dd_divide(Pack(a),Pack(b)));}
    __aicore__ inline float Root(float x){red.SetValue(0,x>0?x:0);SV();AscendC::Sqrt(red,red,1);VS();return red.GetValue(0);}
    __aicore__ inline Wide RootPair(Wide x){float h=Root(x.h);if(!(h>0))return {0,0};return Plus({h,0},Divide(Minus(x,Times({h,0},{h,0})),{2*h,0}));}
    __aicore__ inline void Init(AscendC::LocalTensor<float> p,int width){
        pool=p;bh=p[6*width];bl=p[7*width];
        for(int i=0;i<8;++i)calc[i]=p[(8+i)*width];
        certH=p[16*width];certL=p[17*width];certT=p[18*width];red=p[20*width];
    }
    __aicore__ inline void PairAdd(LocalTensor<float> h,LocalTensor<float> l,
        LocalTensor<float> a,LocalTensor<float> b,LocalTensor<float> c,LocalTensor<float> d,uint32_t count){
        auto sum=calc[0],bv=calc[1],av=calc[2],e=calc[3],v=calc[4];
        Add(sum,a,c,count);PipeBarrier<PIPE_V>();Sub(bv,sum,a,count);PipeBarrier<PIPE_V>();
        Sub(av,sum,bv,count);Sub(e,c,bv,count);PipeBarrier<PIPE_V>();
        Sub(v,a,av,count);PipeBarrier<PIPE_V>();Add(e,e,v,count);PipeBarrier<PIPE_V>();
        Add(e,e,b,count);PipeBarrier<PIPE_V>();Add(e,e,d,count);PipeBarrier<PIPE_V>();
        Add(h,sum,e,count);PipeBarrier<PIPE_V>();Sub(v,h,sum,count);PipeBarrier<PIPE_V>();
        Sub(l,e,v,count);PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Scale(LocalTensor<float> h,LocalTensor<float> l,
        LocalTensor<float> a,LocalTensor<float> b,Wide factor,uint32_t count){
        auto hi=calc[0],lo=calc[1],p=calc[2],e=calc[3],v=calc[4],z=calc[5];
        float split=factor.h*4097.0f,fh=split-(split-factor.h),fl=factor.h-fh;
        Muls(hi,a,4097.0f,count);PipeBarrier<PIPE_V>();Sub(lo,hi,a,count);PipeBarrier<PIPE_V>();
        Sub(hi,hi,lo,count);PipeBarrier<PIPE_V>();Sub(lo,a,hi,count);
        Muls(p,a,factor.h,count);Muls(e,hi,fh,count);PipeBarrier<PIPE_V>();Sub(e,e,p,count);PipeBarrier<PIPE_V>();
        Muls(v,hi,fl,count);PipeBarrier<PIPE_V>();Add(e,e,v,count);PipeBarrier<PIPE_V>();
        Muls(v,lo,fh,count);PipeBarrier<PIPE_V>();Add(e,e,v,count);PipeBarrier<PIPE_V>();
        Muls(v,lo,fl,count);PipeBarrier<PIPE_V>();Add(e,e,v,count);PipeBarrier<PIPE_V>();
        Muls(v,a,factor.l,count);Muls(z,b,factor.h,count);PipeBarrier<PIPE_V>();
        Add(e,e,v,count);PipeBarrier<PIPE_V>();Add(e,e,z,count);PipeBarrier<PIPE_V>();
        Muls(v,b,factor.l,count);PipeBarrier<PIPE_V>();Add(e,e,v,count);PipeBarrier<PIPE_V>();
        Add(h,p,e,count);PipeBarrier<PIPE_V>();Sub(v,h,p,count);PipeBarrier<PIPE_V>();Sub(l,e,v,count);PipeBarrier<PIPE_V>();
    }
    __aicore__ inline Wide Dot(LocalTensor<float> a,LocalTensor<float> b,
        LocalTensor<float> c,LocalTensor<float> d,uint32_t count){
        auto p=calc[0],e=calc[1],hi=calc[2],lo=calc[3],ch=calc[4],cl=calc[5],v=calc[6],z=calc[7];
        Muls(hi,a,4097.0f,count);Muls(ch,c,4097.0f,count);PipeBarrier<PIPE_V>();
        Sub(lo,hi,a,count);Sub(cl,ch,c,count);PipeBarrier<PIPE_V>();
        Sub(hi,hi,lo,count);Sub(ch,ch,cl,count);PipeBarrier<PIPE_V>();
        Sub(lo,a,hi,count);Sub(cl,c,ch,count);Mul(p,a,c,count);PipeBarrier<PIPE_V>();
        Mul(e,hi,ch,count);PipeBarrier<PIPE_V>();Sub(e,e,p,count);PipeBarrier<PIPE_V>();
        Mul(v,hi,cl,count);PipeBarrier<PIPE_V>();Add(e,e,v,count);PipeBarrier<PIPE_V>();
        Mul(v,lo,ch,count);PipeBarrier<PIPE_V>();Add(e,e,v,count);PipeBarrier<PIPE_V>();
        Mul(v,lo,cl,count);PipeBarrier<PIPE_V>();Add(e,e,v,count);PipeBarrier<PIPE_V>();
        Mul(v,a,d,count);Mul(z,b,c,count);PipeBarrier<PIPE_V>();
        Add(e,e,v,count);PipeBarrier<PIPE_V>();Add(e,e,z,count);PipeBarrier<PIPE_V>();
        Mul(v,b,d,count);PipeBarrier<PIPE_V>();Add(e,e,v,count);PipeBarrier<PIPE_V>();
        // Pairwise TwoSum keeps the FP32 reduction error in the low component.
        for(uint32_t size=count;size>8;size/=2){
            uint32_t half=size/2;
            Add(hi,p,p[half],half);PipeBarrier<PIPE_V>();Sub(lo,hi,p,half);PipeBarrier<PIPE_V>();
            Sub(ch,hi,lo,half);Sub(cl,p[half],lo,half);PipeBarrier<PIPE_V>();
            Sub(v,p,ch,half);PipeBarrier<PIPE_V>();Add(cl,cl,v,half);PipeBarrier<PIPE_V>();
            Add(cl,cl,e,half);PipeBarrier<PIPE_V>();Add(cl,cl,e[half],half);PipeBarrier<PIPE_V>();
            Add(p,hi,cl,half);PipeBarrier<PIPE_V>();Sub(v,p,hi,half);PipeBarrier<PIPE_V>();
            Sub(e,cl,v,half);PipeBarrier<PIPE_V>();
        }
        VS();Wide sum={0,0};for(uint32_t i=0;i<8;++i)sum=Plus(sum,{p.GetValue(i),e.GetValue(i)});
        return sum;
    }
// Three-component arithmetic is confined to posterior residual measurements.
// Error-free FP32 products and TwoSum leave a third-order accumulation remainder.
    __aicore__ inline void CertTwoSum(LocalTensor<float> sum,LocalTensor<float> error,
        LocalTensor<float> a,LocalTensor<float> b,uint32_t count){
        auto v=calc[6],w=calc[7];
        Add(sum,a,b,count);PipeBarrier<PIPE_V>();Sub(v,sum,a,count);PipeBarrier<PIPE_V>();
        Sub(w,sum,v,count);PipeBarrier<PIPE_V>();Sub(w,a,w,count);Sub(v,b,v,count);PipeBarrier<PIPE_V>();
        Add(error,w,v,count);PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void CertAdd(LocalTensor<float> a,LocalTensor<float> b,LocalTensor<float> c,
        LocalTensor<float> value,uint32_t count){
        CertTwoSum(calc[0],calc[1],a,value,count);
        CertTwoSum(calc[2],calc[3],b,calc[1],count);
        CertTwoSum(calc[4],calc[5],c,calc[3],count);
        CertTwoSum(a,calc[2],calc[0],calc[2],count);
        CertTwoSum(b,c,calc[2],calc[4],count);
    }
    __aicore__ inline void CertProduct(LocalTensor<float> p,LocalTensor<float> error,
        LocalTensor<float> a,LocalTensor<float> b,uint32_t count){
        auto hi=calc[0],lo=calc[1],ch=calc[2],cl=calc[3],v=calc[4];
        Muls(hi,a,4097.0f,count);Muls(ch,b,4097.0f,count);PipeBarrier<PIPE_V>();
        Sub(lo,hi,a,count);Sub(cl,ch,b,count);PipeBarrier<PIPE_V>();
        Sub(hi,hi,lo,count);Sub(ch,ch,cl,count);PipeBarrier<PIPE_V>();
        Sub(lo,a,hi,count);Sub(cl,b,ch,count);Mul(p,a,b,count);PipeBarrier<PIPE_V>();
        Mul(error,hi,ch,count);PipeBarrier<PIPE_V>();Sub(error,error,p,count);PipeBarrier<PIPE_V>();
        Mul(v,hi,cl,count);PipeBarrier<PIPE_V>();Add(error,error,v,count);PipeBarrier<PIPE_V>();
        Mul(v,lo,ch,count);PipeBarrier<PIPE_V>();Add(error,error,v,count);PipeBarrier<PIPE_V>();
        Mul(v,lo,cl,count);PipeBarrier<PIPE_V>();Add(error,error,v,count);PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void CertProductScalar(LocalTensor<float> p,LocalTensor<float> error,
        LocalTensor<float> a,float b,uint32_t count){
        auto hi=calc[0],lo=calc[1],v=calc[2];
        volatile float split=b*4097.0f;
        volatile float temp=split-b;
        volatile float ch=split-temp;
        volatile float cl=b-ch;
        Muls(hi,a,4097.0f,count);PipeBarrier<PIPE_V>();Sub(lo,hi,a,count);PipeBarrier<PIPE_V>();
        Sub(hi,hi,lo,count);PipeBarrier<PIPE_V>();Sub(lo,a,hi,count);Muls(p,a,b,count);PipeBarrier<PIPE_V>();
        Muls(error,hi,static_cast<float>(ch),count);PipeBarrier<PIPE_V>();Sub(error,error,p,count);PipeBarrier<PIPE_V>();
        Muls(v,hi,static_cast<float>(cl),count);PipeBarrier<PIPE_V>();Add(error,error,v,count);PipeBarrier<PIPE_V>();
        Muls(v,lo,static_cast<float>(ch),count);PipeBarrier<PIPE_V>();Add(error,error,v,count);PipeBarrier<PIPE_V>();
        Muls(v,lo,static_cast<float>(cl),count);PipeBarrier<PIPE_V>();Add(error,error,v,count);PipeBarrier<PIPE_V>();
    }
    __aicore__ inline Wide CertScalarSum(float a,float b){
        volatile float sum=a+b;
        volatile float v=sum-a;
        volatile float w=sum-v;
        volatile float ar=a-w;
        volatile float br=b-v;
        return {sum,ar+br};
    }
    __aicore__ inline void CertScalarAdd(float &a,float &b,float &c,float value){
        Wide first=CertScalarSum(a,value),second=CertScalarSum(b,first.l),third=CertScalarSum(c,second.l);
        first=CertScalarSum(first.h,second.h);
        second=CertScalarSum(first.l,third.h);
        a=first.h;b=second.h;c=second.l;
    }
    __aicore__ inline void CertScale(LocalTensor<float> a,Wide factor,uint32_t count){
        Duplicate(certH,0.0f,count);Duplicate(certL,0.0f,count);Duplicate(certT,0.0f,count);PipeBarrier<PIPE_V>();
        CertProductScalar(bh,bl,a,factor.h,count);
        CertAdd(certH,certL,certT,bh,count);CertAdd(certH,certL,certT,bl,count);
        CertProductScalar(bh,bl,a,factor.l,count);
        CertAdd(certH,certL,certT,bh,count);CertAdd(certH,certL,certT,bl,count);
    }
    __aicore__ inline Wide CertDot(LocalTensor<float> a,LocalTensor<float> b,
        LocalTensor<float> c,LocalTensor<float> d,uint32_t count){
        Duplicate(certH,0.0f,count);Duplicate(certL,0.0f,count);Duplicate(certT,0.0f,count);PipeBarrier<PIPE_V>();
        LocalTensor<float> left[2]={a,b},right[2]={c,d};
        for(uint32_t i=0;i<2;++i)for(uint32_t j=0;j<2;++j){
            CertProduct(bh,bl,left[i],right[j],count);
            CertAdd(certH,certL,certT,bh,count);CertAdd(certH,certL,certT,bl,count);
        }
        for(uint32_t size=count;size>8;size/=2){
            uint32_t half=size/2;
            CertAdd(certH,certL,certT,certH[half],half);
            CertAdd(certH,certL,certT,certL[half],half);
            CertAdd(certH,certL,certT,certT[half],half);
        }
        VS();float sh=0,sl=0,st=0;
        for(uint32_t i=0;i<8;++i){
            CertScalarAdd(sh,sl,st,certH.GetValue(i));
            CertScalarAdd(sh,sl,st,certL.GetValue(i));
            CertScalarAdd(sh,sl,st,certT.GetValue(i));
        }
        return Plus({sh,sl},{st,0});
    }

    __aicore__ inline void Eliminate(LocalTensor<float> x,LocalTensor<float> y,LocalTensor<float> u,LocalTensor<float> v,Wide factor,uint32_t s) {
        auto qh = calc[0], ql = calc[1], bh = calc[2], bl = calc[3], e = calc[4], part = calc[5], product = calc[6], aux = calc[7];
        volatile float ac = factor.h * 4097.0f;
        volatile float az = ac - factor.h;
        volatile float ah = ac - az;
        volatile float al = factor.h - ah;
        Muls(bh, u, 4097.0f, s); PipeBarrier<PIPE_V>();
        Sub(bl, bh, u, s); PipeBarrier<PIPE_V>(); Sub(bh, bh, bl, s); PipeBarrier<PIPE_V>();
        Sub(bl, u, bh, s); Muls(product, u, factor.h, s); PipeBarrier<PIPE_V>();
        Muls(e, bh, static_cast<float>(ah), s); PipeBarrier<PIPE_V>(); Sub(e, e, product, s); PipeBarrier<PIPE_V>();
        Muls(part, bl, static_cast<float>(ah), s); PipeBarrier<PIPE_V>(); Add(e, e, part, s); PipeBarrier<PIPE_V>();
        Muls(part, bh, static_cast<float>(al), s); PipeBarrier<PIPE_V>(); Add(e, e, part, s); PipeBarrier<PIPE_V>();
        Muls(part, bl, static_cast<float>(al), s); PipeBarrier<PIPE_V>(); Add(e, e, part, s); PipeBarrier<PIPE_V>();
        Muls(part, v, factor.h, s); PipeBarrier<PIPE_V>(); Add(e, e, part, s); PipeBarrier<PIPE_V>();
        Muls(part, u, factor.l, s); PipeBarrier<PIPE_V>(); Add(e, e, part, s); PipeBarrier<PIPE_V>();
        Add(qh, product, e, s); PipeBarrier<PIPE_V>(); Sub(ql, qh, product, s); PipeBarrier<PIPE_V>(); Sub(ql, e, ql, s); PipeBarrier<PIPE_V>();
        Sub(bh, x, qh, s); PipeBarrier<PIPE_V>(); Sub(bl, bh, x, s); PipeBarrier<PIPE_V>();
        Sub(e, bh, bl, s); PipeBarrier<PIPE_V>(); Sub(e, x, e, s); Muls(part, qh, -1.0f, s); PipeBarrier<PIPE_V>();
        Sub(part, part, bl, s); PipeBarrier<PIPE_V>(); Add(e, e, part, s); PipeBarrier<PIPE_V>();
        Add(e, e, y, s); PipeBarrier<PIPE_V>(); Sub(e, e, ql, s); PipeBarrier<PIPE_V>();
        Add(x, bh, e, s); PipeBarrier<PIPE_V>(); Sub(aux, x, bh, s); PipeBarrier<PIPE_V>(); Sub(y, e, aux, s); PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Normalize(LocalTensor<float> x,LocalTensor<float> y,LocalTensor<float> u,LocalTensor<float> v,Wide divisor,uint32_t s) {
        float reciprocal = 1.0f / divisor.h;
        Muls(u, x, reciprocal, s); Duplicate(v, 0.0f, s); PipeBarrier<PIPE_V>();
        Eliminate(x,y,u,v,divisor,s);
        Muls(calc[0], x, reciprocal, s); PipeBarrier<PIPE_V>();
        Add(x, u, calc[0], s); PipeBarrier<PIPE_V>();
        Sub(y, x, u, s); PipeBarrier<PIPE_V>();
        Sub(y, calc[0], y, s); PipeBarrier<PIPE_V>();
    }

};

__aicore__ inline void PvDotStore(PvArithmetic& p,AscendC::LocalTensor<float> dst,int ah,int al,int bh,int bl,int n){PvWritePair(dst,p.Pack(p.Dot(p.pool[ah],p.pool[al],p.pool[bh],p.pool[bl],n)));}
__aicore__ inline void PvCertDotStore(PvArithmetic& p,AscendC::LocalTensor<float> dst,int ah,int al,int bh,int bl,int n){PvWritePair(dst,p.Pack(p.CertDot(p.pool[ah],p.pool[al],p.pool[bh],p.pool[bl],n)));}
__aicore__ inline void PvRootPairStore(PvArithmetic& p,AscendC::LocalTensor<float> dst,AscendC::LocalTensor<float> a){PvWritePair(dst,p.Pack(p.RootPair(p.Unpack(PvReadPair(a)))));}
__aicore__ inline void PvRootStore(PvArithmetic& p,AscendC::LocalTensor<float> dst,float a){dst.SetValue(0,p.Root(a));}
__aicore__ inline void PvScaleAt(PvArithmetic& p,int h,int l,int a,int b,float fh,float fl,int n){p.Scale(p.pool[h],p.pool[l],p.pool[a],p.pool[b],{fh,fl},n);}
__aicore__ inline void PvAddAt(PvArithmetic& p,int h,int l,int a,int b,int c,int d,int n){p.PairAdd(p.pool[h],p.pool[l],p.pool[a],p.pool[b],p.pool[c],p.pool[d],n);}
__aicore__ inline void PvCertScaleAt(PvArithmetic& p,int a,float fh,float fl,int n){p.CertScale(p.pool[a],{fh,fl},n);}
__aicore__ inline void PvCertAddAt(PvArithmetic& p,int a,int b,int c,int value,int n){p.CertAdd(p.pool[a],p.pool[b],p.pool[c],p.pool[value],n);}
__aicore__ inline void PvNormalizeAt(PvArithmetic& p,int h,int l,int u,int v,float fh,float fl,int n){p.Normalize(p.pool[h],p.pool[l],p.pool[u],p.pool[v],{fh,fl},n);}
__aicore__ inline void PvEliminateAt(PvArithmetic& p,int h,int l,int u,int v,float fh,float fl,int n){p.Eliminate(p.pool[h],p.pool[l],p.pool[u],p.pool[v],{fh,fl},n);}
}
#define pv_arith_init(d,a,b,width) PvArithmetic pvArithmetic; pvArithmetic.Init(d,width)
#define pv_pair_dot(d,a,b,...) PvDotStore(pvArithmetic,d,__VA_ARGS__)
#define pv_pair_cert_dot(d,a,b,...) PvCertDotStore(pvArithmetic,d,__VA_ARGS__)
#define pv_pair_root(d,a,b) PvRootPairStore(pvArithmetic,d,a)
#define pv_root(d,a,b,value) PvRootStore(pvArithmetic,d,value)
#define pv_pair_scale(d,a,b,...) PvScaleAt(pvArithmetic,__VA_ARGS__)
#define pv_pair_add(d,a,b,...) PvAddAt(pvArithmetic,__VA_ARGS__)
#define pv_cert_scale(d,a,b,...) PvCertScaleAt(pvArithmetic,__VA_ARGS__)
#define pv_cert_add(d,a,b,...) PvCertAddAt(pvArithmetic,__VA_ARGS__)

#define pv_pair_normalize(d,a,b,...) PvNormalizeAt(pvArithmetic,__VA_ARGS__)
#define pv_pair_eliminate(d,a,b,...) PvEliminateAt(pvArithmetic,__VA_ARGS__)
