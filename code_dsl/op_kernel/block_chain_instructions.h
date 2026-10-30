#pragma once
namespace tl::ascend {
template<class D,class S,class X> __aicore__ inline void bc_store(D d,S s,X,int m,int n,int stride){AscendC::FixpipeParamsV220 p;p.nSize=n;p.mSize=m;p.srcStride=m;p.dstStride=stride;p.ndNum=1;p.srcNdStride=0;p.dstNdStride=0;AscendC::Fixpipe<float,float,AscendC::CFG_ROW_MAJOR>(d,s,p);}
}
