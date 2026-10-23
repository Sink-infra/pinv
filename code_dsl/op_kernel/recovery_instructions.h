#pragma once
#include "kernel_operator.h"
#include <type_traits>
namespace Catlass { namespace layout {struct RowMajor{};} }
namespace tl::ascend {
using namespace AscendC;
template<class T,unsigned M,unsigned N>
__aicore__ inline void copy_gm_to_l1(LocalTensor<T>d,GlobalTensor<T>s,unsigned stride,unsigned rows=0,unsigned cols=0){
 if constexpr(sizeof(T)==4 && M>16){
  unsigned live=rows?rows:M;
  for(unsigned part=0;part<M/16;++part)if(live>part*16)
   copy_gm_to_l1<T,16,N>(d[part*16*N],s[part*16*stride],stride,(live-part*16<16?live-part*16:16),cols);
 }else{
  Nd2NzParams p;p.ndNum=1;p.nValue=rows?rows:M;p.dValue=cols?cols:N;p.srcDValue=stride;p.srcNdMatrixStride=0;p.dstNzC0Stride=M;p.dstNzNStride=1;p.dstNzMatrixStride=0;DataCopy(d,s,p);
 }
}
template<class T,unsigned M,unsigned K>
__aicore__ inline void copy_l1_to_l0a(LocalTensor<T>d,LocalTensor<T>s,unsigned rows,unsigned cols){
 constexpr unsigned width=32/sizeof(T);
 LoadData2DParams p;p.startIndex=0;p.repeatTimes=cols/width;p.srcStride=M/16;p.sid=0;p.dstGap=0;p.ifTranspose=false;p.addrMode=0;
 for(unsigned i=0;i<rows/16;++i)LoadData(d[i*cols*16],s[i*16*width],p);
}
template<class T,unsigned K,unsigned N>
__aicore__ inline void copy_l1_to_l0b(LocalTensor<T>d,LocalTensor<T>s,unsigned rows,unsigned cols){
 if constexpr(sizeof(T)==1){
  LoadData2dTransposeParams p;p.startIndex=0;p.repeatTimes=cols/32;p.srcStride=K/32;p.dstGap=1;p.dstFracGap=0;
  for(unsigned i=0;i<rows/32;++i)LoadDataWithTranspose(d[i*cols*32],s[i*1024],p);
 }else if constexpr(sizeof(T)==4){
  static_assert(K%16==0);
  LoadData2dTransposeParams p;p.startIndex=0;p.repeatTimes=N/16;p.srcStride=1;p.dstGap=0;p.dstFracGap=N/16-1;
  for(unsigned part=0;part<K/16;++part)LoadDataWithTranspose(d[part*16*N],s[part*16*N],p);
 }else{
  LoadData2DParams p;p.startIndex=0;p.repeatTimes=cols/16;p.srcStride=K/16;p.sid=0;p.dstGap=0;p.ifTranspose=true;p.addrMode=0;
  for(unsigned i=0;i<rows/16;++i)LoadData(d[i*cols*16],s[i*256],p);
 }
}
template<class T,class U,unsigned M,unsigned N>
__aicore__ inline void mma(LocalTensor<T>a,LocalTensor<T>b,LocalTensor<U>c,bool init,unsigned k,unsigned char=0){
 if constexpr(sizeof(T)==4)SetHF32Mode(HF32Mode::DISABLE);
 MmadParams p;p.m=M;p.n=N;p.k=k;p.cmatrixInitVal=init;Mmad(c,a,b,p);
}
template<class T,class U,unsigned M,unsigned N,unsigned K,bool TA=false,bool TB=false>
__aicore__ inline void gemm_v0(LocalTensor<T> const&a,LocalTensor<T> const&b,LocalTensor<U> const&c,TBuf<TPosition::A2>&ab,TBuf<TPosition::B2>&bb,bool init){
 static_assert((K==128 || (K%16==0 && K<=128 && sizeof(T)==4)) && !TA && !TB);
 auto aa=ab.Get<T>();auto ba=bb.Get<T>();
 SetFlag<HardEvent::MTE2_MTE1>(0);WaitFlag<HardEvent::MTE2_MTE1>(0);
 SetFlag<HardEvent::FIX_M>(0);WaitFlag<HardEvent::FIX_M>(0);
 SetFlag<HardEvent::M_MTE1>(0);WaitFlag<HardEvent::M_MTE1>(0);
 copy_l1_to_l0a<T,M,K>(aa,a,M,K);copy_l1_to_l0b<T,K,N>(ba,b,K,N);
 SetFlag<HardEvent::MTE1_M>(0);WaitFlag<HardEvent::MTE1_M>(0);PipeBarrier<PIPE_M>();
 mma<T,U,M,N>(aa,ba,c,init,K);
 SetFlag<HardEvent::M_MTE1>(0);WaitFlag<HardEvent::M_MTE1>(0);
 SetFlag<HardEvent::MTE1_MTE2>(0);WaitFlag<HardEvent::MTE1_MTE2>(0);
 SetFlag<HardEvent::M_FIX>(0);WaitFlag<HardEvent::M_FIX>(0);
}
template<class T,class U,class Layout,unsigned M,unsigned N,bool Relu=false>
__aicore__ inline void copy_l0c_to_gm(GlobalTensor<U>d,LocalTensor<T>s,unsigned stride,unsigned rows=0,unsigned cols=0){
 FixpipeParamsV220 p;p.nSize=cols?cols:N;p.mSize=rows?rows:M;p.srcStride=M;p.dstStride=stride;p.ndNum=1;p.srcNdStride=0;p.dstNdStride=0;Fixpipe<T,U,CFG_ROW_MAJOR>(d,s,p);
}
__aicore__ inline void br_load_a_shape(LocalTensor<float>d,GlobalTensor<float>s,LocalTensor<float>,int cols,int stride,int rows){Nd2NzParams p;p.ndNum=1;p.nValue=rows;p.dValue=cols;p.srcDValue=stride;p.srcNdMatrixStride=0;p.dstNzC0Stride=rows;p.dstNzNStride=1;p.dstNzMatrixStride=0;DataCopy(d,s,p);}
}
