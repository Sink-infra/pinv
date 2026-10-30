#pragma once
namespace tl::ascend {
template<class D,class A,class B> __aicore__ inline void ba_atomic_on(D,A,B){AscendC::SetAtomicAdd<float>();}
template<class D,class A,class B> __aicore__ inline void ba_atomic_off(D,A,B){AscendC::DisableDmaAtomic();}
}
