#pragma once
namespace tl::ascend {
__aicore__ inline void br_sub(AscendC::LocalTensor<float>d,AscendC::LocalTensor<float>a,AscendC::LocalTensor<float>b,int n){AscendC::Sub(d,a,b,n);}
__aicore__ inline void bp_store_half(AscendC::GlobalTensor<half>d,AscendC::LocalTensor<half>a,AscendC::GlobalTensor<half>,int n){AscendC::DataCopy(d,a,n);}
}


namespace tl::ascend {
__aicore__ inline void br_load_a(LocalTensor<float>d,GlobalTensor<float>s,LocalTensor<float>,int cols,int stride){Nd2NzParams p;p.ndNum=1;p.nValue=64;p.dValue=cols;p.srcDValue=stride;p.srcNdMatrixStride=0;p.dstNzC0Stride=64;p.dstNzNStride=1;p.dstNzMatrixStride=0;DataCopy(d,s,p);}
}

#pragma once
#include "kernel_operator.h"
namespace tl::ascend {
template<class D,class A,class B> __aicore__ inline void br_cube_sync(D,A,B){AscendC::CrossCoreSetFlag<0,PIPE_FIX>(6);AscendC::CrossCoreWaitFlag<0,PIPE_MTE2>(6);}
}
