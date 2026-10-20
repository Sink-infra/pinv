#pragma once
namespace tl::ascend {
template<class D,class A,class B> __aicore__ inline void pc_invalidate(D d,A,B){AscendC::DataCacheCleanAndInvalid<float,AscendC::CacheLine::SINGLE_CACHE_LINE,AscendC::DcciDst::CACHELINE_OUT>(d);}
template<class D,class A,class B> __aicore__ inline void pc_done(D,A,B){AscendC::CrossCoreSetFlag<2,PIPE_FIX>(10);}

__aicore__ inline void FrRestorePipe(AscendC::TPipe*p){
#if __NPU_ARCH__ == 2201 || __NPU_ARCH__ == 3510
#ifdef SPLIT_CORE_CUBE
 ::g_cubeTPipePtr=p;
#elif defined(SPLIT_CORE_VEC)
 ::g_vecTPipePtr=p;
#else
 ::g_tPipePtr=p;
#endif
#else
 ::g_tPipePtr=p;
#endif
}
struct FusedBuffers { AscendC::GlobalTensor<float> input,output,work,status; };
}
#define fr_bind(d,a,b) FusedBuffers fr{d,a,b,S}
#define fr_raw(d,a,b,bl,n,ba,st,wi,at,rt,so) FusedRaw((GM_ADDR)fr.input.GetPhyAddr(),(GM_ADDR)fr.output.GetPhyAddr(),(GM_ADDR)fr.work.GetPhyAddr(),(GM_ADDR)(fr.work.GetPhyAddr()+so),bl,n,ba,st,wi,at,rt,fftsAddr),tl::ascend::FrRestorePipe(&pipe)
#define fr_prepared_raw(d,a,b,bl,n,ba,st,wi,at,rt,so) FusedPreparedRaw((GM_ADDR)fr.input.GetPhyAddr(),(GM_ADDR)fr.output.GetPhyAddr(),(GM_ADDR)fr.work.GetPhyAddr(),(GM_ADDR)(fr.work.GetPhyAddr()+so),bl,n,ba,st,wi,at,rt,fftsAddr),tl::ascend::FrRestorePipe(&pipe)
#define fr_quant(d,a,b,off,qa,qy,qn,bl,n,job,sa,sy) FusedQuant((GM_ADDR)fr.input.GetPhyAddr(),(GM_ADDR)fr.output.GetPhyAddr(),(GM_ADDR)(fr.work.GetPhyAddr()+off+qa),(GM_ADDR)(fr.work.GetPhyAddr()+off+qy),(GM_ADDR)(fr.work.GetPhyAddr()+off+qn),bl,n,job,sa,sy,fftsAddr),tl::ascend::FrRestorePipe(&pipe)
#define fr_integer(d,a,b,off,qa,qy,pr,bl,n) FusedInteger((GM_ADDR)(d.GetPhyAddr()+off+qa),(GM_ADDR)(d.GetPhyAddr()+off+qy),(GM_ADDR)(d.GetPhyAddr()+off+pr),bl,(GM_ADDR)d.GetPhyAddr(),n,5,fftsAddr)
#define fr_decode(d,a,b,off,pr,rr,yh,rh,bl,n,job,sa,sy) FusedDecode((GM_ADDR)(fr.work.GetPhyAddr()+off+pr),(GM_ADDR)fr.output.GetPhyAddr(),(GM_ADDR)(fr.work.GetPhyAddr()+off+rr),(GM_ADDR)(fr.work.GetPhyAddr()+off+yh),(GM_ADDR)(fr.work.GetPhyAddr()+off+rh),bl,n,job,sa,sy,fftsAddr),tl::ascend::FrRestorePipe(&pipe)
#define fr_half(d,a,b,off,yh,rh,co,bl,n) FusedHalf((GM_ADDR)(d.GetPhyAddr()+off+yh),(GM_ADDR)(d.GetPhyAddr()+off+rh),(GM_ADDR)(d.GetPhyAddr()+off+co),bl,(GM_ADDR)d.GetPhyAddr(),n,1,fftsAddr)
#define fr_combine(d,a,b,off,co,bl,n,job,sy) FusedCombine((GM_ADDR)fr.output.GetPhyAddr(),(GM_ADDR)(fr.work.GetPhyAddr()+off+co),bl,n,job,sy,fftsAddr),tl::ascend::FrRestorePipe(&pipe)
#define fr_norms(d,a,b,off,rr,pa,bl,n) FusedNorms((GM_ADDR)(d.GetPhyAddr()+off+rr),(GM_ADDR)(d.GetPhyAddr()+off+pa),bl,n,fftsAddr),tl::ascend::FrRestorePipe(&pipe)
#define fr_certify(d,a,b,off,co,pa,qn,st,bl,n,job,sa,sy,af,yf) FusedCertify((GM_ADDR)fr.output.GetPhyAddr(),(GM_ADDR)(fr.work.GetPhyAddr()+off+co),(GM_ADDR)(fr.work.GetPhyAddr()+off+pa),(GM_ADDR)(fr.work.GetPhyAddr()+off+qn),(GM_ADDR)(fr.status.GetPhyAddr()+st),bl,n,job,sa,sy,af,yf,fftsAddr),tl::ascend::FrRestorePipe(&pipe)
#define fr_fallback(d,a,b,so,bl,job,n,ba,at,rt) PvDslRectangular((GM_ADDR)fr.input.GetPhyAddr(),(GM_ADDR)fr.output.GetPhyAddr(),(GM_ADDR)fr.work.GetPhyAddr(),(GM_ADDR)(fr.work.GetPhyAddr()+so),bl,job,n,n,ba,at,rt,fftsAddr),tl::ascend::FrRestorePipe(&pipe)


