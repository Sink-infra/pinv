#pragma once
#include "kernel_operator.h"
namespace tl::ascend {
// Individual transfers, reductions and hardware synchronization only.
template<class D,class A,class B> __aicore__ inline void pv_all_barrier(D,A,B){AscendC::PipeBarrier<PIPE_ALL>();}
template<class D,class A,class B> __aicore__ inline void pv_sync_all(D,A,B){AscendC::SyncAll();}
__aicore__ inline void PvSetSyncBase(uint64_t address){if(address)AscendC::SetSyncBaseAddr(address);}
template<class D,class A,class B> __aicore__ inline void pv_cross_set0(D,A,B,int flag){AscendC::CrossCoreSetFlag<0,PIPE_MTE3>(flag);}
template<class D,class A,class B> __aicore__ inline void pv_cross_set1(D,A,B,int flag){AscendC::CrossCoreSetFlag<1,PIPE_MTE3>(flag);}
template<class D,class A,class B> __aicore__ inline void pv_cross_set2(D,A,B,int flag){AscendC::CrossCoreSetFlag<2,PIPE_MTE3>(flag);}
template<class D,class A,class B> __aicore__ inline void pv_cross_wait(D,A,B,int flag){AscendC::CrossCoreWaitFlag(flag);}
template<class D,class A,class B> __aicore__ inline void pv_cross_wait2(D,A,B,int flag){AscendC::CrossCoreWaitFlag<2>(flag);}
__aicore__ inline void pv_gather_offsets(AscendC::LocalTensor<float> d,AscendC::LocalTensor<float> a,AscendC::LocalTensor<float> offsets,int base,int n){AscendC::Gather(d,a,offsets.ReinterpretCast<uint32_t>(),uint32_t(base),n);}
__aicore__ inline void pv_iota(AscendC::LocalTensor<float> d,AscendC::LocalTensor<float>,AscendC::LocalTensor<float>,int n){AscendC::CreateVecIndex(d.ReinterpretCast<int32_t>(),int32_t(0),n);}
__aicore__ inline void pv_scale_indices(AscendC::LocalTensor<float> d,AscendC::LocalTensor<float> a,AscendC::LocalTensor<float>,int scale,int n){AscendC::Muls(d.ReinterpretCast<int32_t>(),a.ReinterpretCast<int32_t>(),scale,n);}
__aicore__ inline void pv_store_u32(AscendC::LocalTensor<float> d,AscendC::LocalTensor<float>,AscendC::LocalTensor<float>,int index,uint32_t value){d.ReinterpretCast<uint32_t>().SetValue(index,value);}
// Each binding emits one reduction; its chunking and dependency schedule live
// in reduction.py. The mask adapter only encodes the hardware's even lanes.
__aicore__ inline void pv_whole_max(AscendC::LocalTensor<float> d,AscendC::LocalTensor<float> a,AscendC::LocalTensor<float>,int mask,int repeats){AscendC::WholeReduceMax(d,a,uint64_t(mask),uint8_t(repeats),1,1,8);}
__aicore__ inline void pv_even_max(AscendC::LocalTensor<float> d,AscendC::LocalTensor<float> a,AscendC::LocalTensor<float>,int count){uint64_t mask[2]={uint64_t(0x5555)&((uint64_t(1)<<(count*2))-1),0};AscendC::WholeReduceMax(d,a,mask,1,1,1,8);}

}
#define pv_sync_init(d,a,b) PvSetSyncBase(fftsAddr)

namespace tl::ascend {
__aicore__ inline void pp_clear_history(AscendC::LocalTensor<float> d,AscendC::LocalTensor<float>,AscendC::LocalTensor<float>,int count,int pitch){AscendC::Duplicate(d,0.0f,uint64_t(8),uint8_t(count),1,uint8_t(pitch));}
}
