#pragma once
namespace tl::ascend {
template<class D,class A,class B> __aicore__ inline void pp_mailbox_cache(D d,A,B){AscendC::DataCacheCleanAndInvalid<float,AscendC::CacheLine::SINGLE_CACHE_LINE,AscendC::DcciDst::CACHELINE_OUT>(d);}
}

namespace tl::ascend {
template<class D,class A,class B> __aicore__ inline void pp_mailbox_load(D d,A,B,float &value){value=*reinterpret_cast<const volatile __gm__ float*>(d.GetPhyAddr());}
}
