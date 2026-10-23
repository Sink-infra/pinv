#pragma once
#include "kernel_operator.h"
using namespace AscendC;
namespace tl::ascend {
// These adapters contain individual transfers/instructions, never a solver.
__aicore__ inline void pv_fill(LocalTensor<float> d, LocalTensor<float>, LocalTensor<float>, float v,int n){Duplicate(d,v,n);}
__aicore__ inline void pv_abs(LocalTensor<float> d, LocalTensor<float> a, LocalTensor<float>,int n){Abs(d,a,n);}
__aicore__ inline void pv_scale(LocalTensor<float> d, LocalTensor<float> a, LocalTensor<float>,float v,int n){Muls(d,a,v,n);}
__aicore__ inline void pv_mul(LocalTensor<float> d, LocalTensor<float> a, LocalTensor<float> b,int n){Mul(d,a,b,n);}
__aicore__ inline void pv_add(LocalTensor<float> d, LocalTensor<float> a, LocalTensor<float> b,int n){Add(d,a,b,n);}
__aicore__ inline void pv_div(LocalTensor<float> d, LocalTensor<float> a, LocalTensor<float> b,int n){Div(d,a,b,n);}
__aicore__ inline void pv_copy(LocalTensor<float> d, LocalTensor<float> a, LocalTensor<float>,int n){DataCopy(d,a,n);}
__aicore__ inline void pv_copy_ub_rows(LocalTensor<float> d,LocalTensor<float> a,LocalTensor<float>,int rows,int width,int sourceGap,int destinationGap){DataCopy(d,a,DataCopyParams{uint16_t(rows),uint16_t(width/8),uint16_t(sourceGap/8),uint16_t(destinationGap/8)});}
__aicore__ inline void pv_sqrt(LocalTensor<float> d, LocalTensor<float> a, LocalTensor<float>,int n){Sqrt(d,a,n);}
__aicore__ inline void pv_reduce_max(LocalTensor<float> d, LocalTensor<float> a, LocalTensor<float> w,int n){ReduceMax(d,a,w,n,false);}
__aicore__ inline void pv_reduce_sum(LocalTensor<float> d, LocalTensor<float> a, LocalTensor<float> w,int n){ReduceSum(d,a,w,n);}
__aicore__ inline void pv_load_rows(LocalTensor<float> d, GlobalTensor<float> a, LocalTensor<float>,int rows,int cols,int sourceGap,int destinationGap,bool padding=true){DataCopyPad(d,a,DataCopyExtParams{uint16_t(rows),uint32_t(cols*4),uint32_t(sourceGap*4),uint32_t(destinationGap/8),0},DataCopyPadExtParams<float>{padding,0,uint8_t((8-cols%8)%8),0});}
__aicore__ inline void pv_store_rows(GlobalTensor<float> d,LocalTensor<float> a,GlobalTensor<float>,int rows,int cols,int sourceGap,int destinationGap){DataCopyPad(d,a,DataCopyExtParams{uint16_t(rows),uint32_t(cols*4),uint32_t(sourceGap/8),uint32_t(destinationGap*4),0});}
__aicore__ inline void pv_gather(LocalTensor<float> d,LocalTensor<float> a,LocalTensor<uint32_t> offsets,uint32_t base,int n){Gather(d,a,offsets,base,n);}
__aicore__ inline void pv_max_available(LocalTensor<float> d,LocalTensor<float> a,LocalTensor<float>,uint32_t bits){uint64_t mask[2]={bits,0};WholeReduceMax(d,a,mask,1,1,1,8);}
__aicore__ inline void pv_max_available64(LocalTensor<float> d,LocalTensor<float> a,LocalTensor<float>,uint64_t bits){uint64_t mask[2]={bits,0};WholeReduceMax(d,a,mask,1,1,1,8);}
__aicore__ inline void pv_max_two32(LocalTensor<float> d,LocalTensor<float> a,LocalTensor<float>,uint32_t lo,uint32_t hi){uint64_t mask[2]={uint64_t(lo)|(uint64_t(hi)<<32),0};WholeReduceMax(d,a,mask,1,1,1,8);}
__aicore__ inline void pv_mask_init(LocalTensor<uint32_t> d,LocalTensor<uint32_t>,LocalTensor<uint32_t>,int n){uint64_t bits=n==64?~uint64_t(0):(uint64_t(1)<<n)-1;d.SetValue(0,uint32_t(bits));d.SetValue(1,uint32_t(bits>>32));}
__aicore__ inline void pv_mask_clear(LocalTensor<uint32_t> d,LocalTensor<uint32_t>,LocalTensor<uint32_t>,int bit){uint32_t index=uint32_t(bit)/32;d.SetValue(index,d.GetValue(index)&~(uint32_t(1)<<(uint32_t(bit)%32)));}
__aicore__ inline void pv_max_mask_buffer(LocalTensor<float> d,LocalTensor<float> a,LocalTensor<uint32_t> bits){uint64_t mask[2]={uint64_t(bits.GetValue(0))|(uint64_t(bits.GetValue(1))<<32),0};WholeReduceMax(d,a,mask,1,1,1,8);}
// The DSL stores raw offset words in the shared flat pool at native offsets.
__aicore__ inline void pv_gather_bits(LocalTensor<float> d,LocalTensor<float> a,LocalTensor<float> offsets,uint32_t base,int n){Gather(d,a,offsets.ReinterpretCast<uint32_t>(),base,n);}
struct PvLaneMask {uint64_t bits;__aicore__ inline explicit PvLaneMask(int n):bits(n==64?~uint64_t(0):(uint64_t(1)<<n)-1){}};
__aicore__ inline void PvMaskReduce(LocalTensor<float> d,LocalTensor<float> a,const PvLaneMask &state){uint64_t mask[2]={state.bits,0};WholeReduceMax(d,a,mask,1,1,1,8);}
__aicore__ inline void PvMaskRemove(PvLaneMask &state,int bit){state.bits&=~(uint64_t(1)<<bit);}
__aicore__ inline void pv_reduce_registers(LocalTensor<float> d,LocalTensor<float>,LocalTensor<float>){float best,bits;GetReduceRepeatMaxMinSpr<float>(best,bits);d.SetValue(0,best);d.SetValue(1,bits);}
// Scalar-output ABI for the same SDK instruction. Allocation, lifetime and
// TileLang alloc_var controls synchronization of these scalar values.
__aicore__ inline void pv_reduce_scalars(LocalTensor<float>,LocalTensor<float>,LocalTensor<float>,float &best,int32_t &position){float bits;GetReduceRepeatMaxMinSpr<float>(best,bits);union {float value;uint32_t word;} index;index.value=bits;position=int32_t(index.word);}
__aicore__ inline void pv_brcb_rows(LocalTensor<float> d,LocalTensor<float> a,LocalTensor<float>,int n,int pitch){Brcb(d,a,uint8_t(n/8),BrcbRepeatParams{uint8_t(pitch/8),uint8_t(pitch)});}
__aicore__ inline void pv_gj_update(LocalTensor<float> d,LocalTensor<float> pivot,LocalTensor<float> factors,int width,int n,int pitch){MulAddDst(d,pivot,factors,uint64_t(width),uint8_t(n),BinaryRepeatParams{1,1,0,uint8_t(pitch/8),0,uint8_t(pitch/8)});}
__aicore__ inline void pv_square_rows(LocalTensor<float> d,LocalTensor<float> a,LocalTensor<float>,int n,int pitch){Mul(d,a,a,uint64_t(n),uint8_t(n),BinaryRepeatParams{1,1,1,uint8_t(pitch/8),uint8_t(pitch/8),uint8_t(pitch/8)});}
__aicore__ inline void pv_square_padded_rows(LocalTensor<float> d,LocalTensor<float> a,LocalTensor<float>,int width,int rows,int pitch){Mul(d,a,a,uint64_t(width),uint8_t(rows),BinaryRepeatParams{1,1,1,uint8_t(pitch/8),uint8_t(pitch/8),uint8_t(pitch/8)});}
__aicore__ inline void pv_pack15(LocalTensor<float> d,LocalTensor<float> a,LocalTensor<float>,int rows,int pitch){uint64_t kept=0;GatherMask(d,a,uint8_t(7),true,rows,{1,uint8_t(rows),uint16_t(pitch/8),0},kept);}
__aicore__ inline void pv_mad_strided(LocalTensor<float> d,LocalTensor<float> a,LocalTensor<float> b,int mask,int repeats,int db,int ab,int bb,int dr,int ar,int br){MulAddDst(d,a,b,uint64_t(mask),uint8_t(repeats),BinaryRepeatParams{uint8_t(db),uint8_t(ab),uint8_t(bb),uint8_t(dr),uint8_t(ar),uint8_t(br)});}
}
#define pv_mask_begin(d,a,b,n) PvLaneMask pvLaneMask(n)
#define pv_mask_reduce(d,a,b) PvMaskReduce(d,a,pvLaneMask)
#define pv_mask_remove(d,a,b,pivot) PvMaskRemove(pvLaneMask,pivot)


namespace tl::ascend {
__aicore__ inline void bl_index_brcb(LocalTensor<float>d,LocalTensor<uint32_t>a,LocalTensor<float>,int n){Brcb(d.ReinterpretCast<int32_t>(),a.ReinterpretCast<int32_t>(),uint8_t(n/8),BrcbRepeatParams{1,8});}
__aicore__ inline void bl_index_matrix(LocalTensor<float>d,LocalTensor<float>a,LocalTensor<uint32_t>b,int width,int rows){Add(d.ReinterpretCast<int32_t>(),a.ReinterpretCast<int32_t>(),b.ReinterpretCast<int32_t>(),uint64_t(width),uint8_t(rows),BinaryRepeatParams{1,0,1,uint8_t(width/8),1,0});}
__aicore__ inline void bl_gather(LocalTensor<float>d,LocalTensor<float>a,LocalTensor<float>indices,int n){Gather(d,a,indices.ReinterpretCast<uint32_t>(),uint32_t(0),n);}
}

namespace tl::ascend {
__aicore__ inline void lv_abs_nomask(AscendC::LocalTensor<float>d,AscendC::LocalTensor<float>a,AscendC::LocalTensor<float>){AscendC::Abs<float,false>(d,a,uint64_t(64),1,AscendC::UnaryRepeatParams{1,1,8,8});}
__aicore__ inline void lv_scale_nomask(AscendC::LocalTensor<float>d,AscendC::LocalTensor<float>a,AscendC::LocalTensor<float>,float scale){AscendC::Muls<float,false>(d,a,scale,uint64_t(64),1,AscendC::UnaryRepeatParams{1,1,8,8});}
__aicore__ inline void lv_mad_nomask(AscendC::LocalTensor<float>d,AscendC::LocalTensor<float>a,AscendC::LocalTensor<float>b,int rows,int pitch){AscendC::MulAddDst<float,float,false>(d,a,b,uint64_t(64),uint8_t(rows),AscendC::BinaryRepeatParams{1,1,0,uint8_t(pitch),0,1});}
}
