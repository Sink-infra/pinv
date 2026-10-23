#pragma once
#include "kernel_operator.h"
namespace tl::ascend {
using namespace AscendC;
__aicore__ inline void hp_copy(LocalTensor<half>d,LocalTensor<half>a,LocalTensor<half>,int n){DataCopy(d,a,n);}
__aicore__ inline void hp_scale(LocalTensor<half>d,LocalTensor<half>a,LocalTensor<half>,float v,int n){Muls(d,a,static_cast<half>(v),n);}
__aicore__ inline void hp_gather(LocalTensor<half>d,LocalTensor<half>a,LocalTensor<float>i,uint32_t base,int n){Gather(d,a,i.ReinterpretCast<uint32_t>(),base,n);}
__aicore__ inline void hp_brcb(LocalTensor<half>d,LocalTensor<half>a,LocalTensor<half>,int n){Brcb(d,a,uint8_t(n/8),BrcbRepeatParams{1,8});}
__aicore__ inline void hp_mad(LocalTensor<half>d,LocalTensor<half>a,LocalTensor<half>b,int width,int rows,int pitch){MulAddDst(d,a,b,uint64_t(width),uint8_t(rows),BinaryRepeatParams{1,1,0,uint8_t(pitch/16),0,1});}
__aicore__ inline void hp_compact(LocalTensor<half>d,LocalTensor<half>a,LocalTensor<half>,int rows,int width,int sourceGap){DataCopy(d,a,DataCopyParams{uint16_t(rows),uint16_t(width/16),uint16_t(sourceGap/16),0});}
__aicore__ inline void hp_float(LocalTensor<float>d,LocalTensor<half>a,LocalTensor<float>,int n){Cast(d,a,RoundMode::CAST_NONE,n);}
__aicore__ inline void hp_half(LocalTensor<half>d,LocalTensor<float>a,LocalTensor<half>,int n){Cast(d,a,RoundMode::CAST_NONE,n);}
__aicore__ inline void hp_rows(LocalTensor<half>d,LocalTensor<half>a,LocalTensor<half>,int rows,int width,int sourceGap,int destinationGap){DataCopy(d,a,DataCopyParams{uint16_t(rows),uint16_t(width/16),uint16_t(sourceGap/16),uint16_t(destinationGap/16)});}
__aicore__ inline void hp_store_header(GlobalTensor<float>d,LocalTensor<half>a,GlobalTensor<float>,int rows,int pitch){GlobalTensor<half> target;target.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(const_cast<__gm__ float*>(d.GetPhyAddr())));DataCopy(target,a,DataCopyParams{uint16_t(rows),1,uint16_t(pitch/16-1),0});}
__aicore__ inline void hp_load_header(LocalTensor<half>d,GlobalTensor<float>a,LocalTensor<half>,int rows,int pitch){GlobalTensor<half> source;source.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(const_cast<__gm__ float*>(a.GetPhyAddr())));DataCopy(d,source,DataCopyParams{uint16_t(rows),1,0,uint16_t(pitch/16-1)});}
__aicore__ inline void hp_abs(LocalTensor<half>d,LocalTensor<half>a,LocalTensor<half>,int n){Abs(d,a,n);}
__aicore__ inline void hp_whole_max(LocalTensor<half>d,LocalTensor<half>a,LocalTensor<half>,int mask,int repeats){WholeReduceMax(d,a,uint64_t(mask),uint8_t(repeats),1,1,8);}
__aicore__ inline void hp_reduce_scalars(LocalTensor<half>,LocalTensor<half>,LocalTensor<half>,float &best,int32_t &position){half value,bits;GetReduceRepeatMaxMinSpr<half>(value,bits);best=static_cast<float>(value);union{half value;uint16_t word;} index;index.value=bits;position=int32_t(index.word);}
__aicore__ inline void hp_index(LocalTensor<half>d,LocalTensor<half>,LocalTensor<half>,int offset,int32_t &position){position=int32_t(d.ReinterpretCast<uint16_t>().GetValue(offset));}
}
