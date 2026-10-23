#pragma once
#include "pinv_dsl_common.h"
#pragma once
#include "kernel_operator.h"
#include <type_traits>
namespace Catlass { namespace layout {struct RowMajor{};} }
namespace tl::ascend {
using namespace AscendC;
template<class T,unsigned M,unsigned N>
__aicore__ inline void copy_gm_to_l1(LocalTensor<T>d,GlobalTensor<T>s,unsigned stride,unsigned rows=0,unsigned cols=0){
 Nd2NzParams p;p.ndNum=1;p.nValue=rows?rows:M;p.dValue=cols?cols:N;p.srcDValue=stride;p.srcNdMatrixStride=0;p.dstNzC0Stride=M;p.dstNzNStride=1;p.dstNzMatrixStride=0;DataCopy(d,s,p);
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
  static_assert(K==16);
  LoadData2dTransposeParams p;p.startIndex=0;p.repeatTimes=N/16;p.srcStride=1;p.dstGap=0;p.dstFracGap=N/16-1;
  LoadDataWithTranspose(d,s,p);
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
 static_assert((K==128 || (K==16 && sizeof(T)==4)) && !TA && !TB);
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
}

using namespace Catlass;
namespace tl::ascend {
 __aicore__ inline void rq_half(LocalTensor<half> d,LocalTensor<float> a,LocalTensor<half>,int n){Cast(d,a,RoundMode::CAST_NONE,n);}
 __aicore__ inline void rq_i8(LocalTensor<int8_t> d,LocalTensor<half> a,LocalTensor<int8_t>,int n){Cast(d,a,RoundMode::CAST_RINT,n);}
 __aicore__ inline void rq_store(GlobalTensor<int8_t> d,LocalTensor<int8_t> a,GlobalTensor<int8_t>,int rows,int cols,int gap){DataCopyPad(d,a,DataCopyExtParams{uint16_t(rows),uint32_t(cols),0,uint32_t(gap),0});}
 }

namespace tl::ascend {
    __aicore__ inline void rd_load(LocalTensor<int32_t> d,GlobalTensor<int32_t> a,LocalTensor<int32_t>,int rows,int cols,int gap){DataCopyPad(d,a,DataCopyExtParams{uint16_t(rows),uint32_t(cols*4),uint32_t(gap*4),0,0},DataCopyPadExtParams<int32_t>{false,0,0,0});}
    __aicore__ inline void rd_store_half(GlobalTensor<half> d,LocalTensor<half> a,GlobalTensor<half>,int rows,int cols,int gap){DataCopyPad(d,a,DataCopyExtParams{uint16_t(rows),uint32_t(cols*2),0,uint32_t(gap*2),0});}
    }

namespace tl::ascend {
__aicore__ inline void PvDslWindowRows( GM_ADDR A_handle,  GM_ADDR X_handle,  GM_ADDR W_handle, int32_t blocks, uint32_t n, uint32_t stride, uint32_t all_batch, uint32_t job, uint32_t core, uint32_t cores, uint32_t window, uint64_t fftsAddr) {
  AscendC::TPipe pipe;

  AscendC::GlobalTensor<float> A;
  A.SetGlobalBuffer((__gm__ float*)A_handle);
  AscendC::GlobalTensor<float> X;
  X.SetGlobalBuffer((__gm__ float*)X_handle);
  AscendC::GlobalTensor<float> W;
  W.SetGlobalBuffer((__gm__ float*)W_handle);

  AscendC::TBuf<AscendC::TPosition::VECCALC> ascend_ub; pipe.InitBuffer(ascend_ub, 196352);
  auto cid = AscendC::GetBlockIdx();
  if ASCEND_IS_AIV {
    }
  uint32_t k = (uint32_t)0;
  uint32_t wide = (uint32_t)0;
  uint32_t owned = (uint32_t)0;
  uint32_t factor_rows = (uint32_t)0;
  uint32_t packet_stride = (uint32_t)0;
  auto pool = ascend_ub.GetWithOffset<float>(49088, 0);
  int valid = 0;
  float power = 0.000000e+00f;
  uint32_t window_end = (uint32_t)0;
  float inverse = 0.000000e+00f;
  float peak = 0.000000e+00f;
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIV {
    tl::ascend::pv_sync_init(W[0], W[0], W[0]);
    k = ((((n + (uint32_t)8) - (uint32_t)1) / (uint32_t)8) * (uint32_t)8);
    wide = ((((k + (uint32_t)64) - (uint32_t)1) / (uint32_t)64) * (uint32_t)64);
    owned = ((((n + cores) - (uint32_t)1) - core) / cores);
    factor_rows = ((((owned + (uint32_t)8) - (uint32_t)1) / (uint32_t)8) * (uint32_t)8);
    packet_stride = (k + (uint32_t)16);
    uint32_t packets = (owned * stride);
    uint32_t output = (packets + ((uint32_t)16 * packet_stride));
    uint32_t factors = (output + max((owned * k), (uint32_t)512));
    uint32_t meta = (factors + ((uint32_t)8 * max(factor_rows, (uint32_t)16)));
    uint32_t norms = (meta + (uint32_t)64);
    uint32_t vector = (norms + wide);
    uint32_t product = (vector + wide);
    uint32_t accumulator = (product + wide);
    uint32_t scratch = (accumulator + wide);
    uint32_t red = (scratch + wide);
    uint32_t row_indices = (red + (uint32_t)32);
    uint32_t packet_indices = (row_indices + factor_rows);
    uint32_t mapping = (packet_indices + (uint32_t)16);
    uint64_t work = ((((uint64_t)job) * ((uint64_t)n)) * ((uint64_t)stride));
    uint64_t metadata = ((((((uint64_t)all_batch) * (uint64_t)2) * ((uint64_t)n)) * ((uint64_t)stride)) + (((uint64_t)job) * (uint64_t)8192));
    tl::ascend::pv_fill(pool[0], pool[0], pool[0], 0.000000e+00f, (owned * stride));
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
    tl::ascend::pv_load_rows(pool[0], A[(((((uint64_t)job) * ((uint64_t)n)) * ((uint64_t)n)) + ((uint64_t)(core * n)))], pool[0], owned, n, ((cores - (uint32_t)1) * n), (stride - k));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
    tl::ascend::pv_fill(pool[norms], pool[norms], pool[norms], 1.000000e+00f, k);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
    for (uint32_t i = (uint32_t)0; i < factor_rows; ++i) {
      uint32_t condval;
      if ((i < owned)) {
        condval = ((i * stride) * (uint32_t)4);
      } else {
        condval = (uint32_t)0;
      }
      uint32_t index = condval;
      tl::ascend::pv_store_u32(pool[0], pool[0], pool[0], (row_indices + i), index);
    }
    for (int32_t i_1 = 0; i_1 < 16; ++i_1) {
      tl::ascend::pv_store_u32(pool[0], pool[0], pool[0], (packet_indices + ((uint32_t)i_1)), ((((uint32_t)i_1) * packet_stride) * (uint32_t)4));
    }
    AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
    valid = 1;
    power = 1.000000e+00f;
    window_end = n;
    if ((uint32_t)0 < window) {
      window_end = min(n, window);
    }
    for (uint32_t panel = (uint32_t)0; panel < (((n + (uint32_t)16) - (uint32_t)1) / (uint32_t)16); ++panel) {
      uint32_t first = (panel * (uint32_t)16);
      uint32_t rows = min((n - first), (uint32_t)16);
      tl::ascend::pv_all_barrier(pool[0], pool[0], pool[0]);
      tl::ascend::pv_cross_set0(pool[0], pool[0], pool[0], 0);
      tl::ascend::pv_cross_wait(pool[0], pool[0], pool[0], 0);
      tl::ascend::pv_load_rows(pool[meta], W[(metadata + ((uint64_t)(panel * (uint32_t)64)))], pool[meta], 1, 64, 0, 0);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
      if (pool.GetValue((meta + (uint32_t)34)) <= 5.000000e-01f) {
        valid = 0;
      }
      power = pool.GetValue((meta + (uint32_t)32));
      uint32_t condval_1;
      if (((uint32_t)0 < window)) {
        condval_1 = (first * packet_stride);
      } else {
        condval_1 = (uint32_t)0;
      }
      uint32_t packet_begin = condval_1;
      if (valid != 0) {
        if (first == (uint32_t)0) {
          for (uint32_t chunk = (uint32_t)0; chunk < ((((owned * stride) + (uint32_t)16320) - (uint32_t)1) / (uint32_t)16320); ++chunk) {
            uint32_t begin = (chunk * (uint32_t)16320);
            tl::ascend::pv_scale(pool[begin], pool[begin], pool[begin], (1.000000e+00f / power), min(((owned * stride) - begin), (uint32_t)16320));
          }
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
          AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        for (uint32_t column = (uint32_t)0; column < rows; ++column) {
          int32_t selected = ((int32_t)pool.GetValue((meta + column)));
          pool.SetValue((norms + ((uint32_t)selected)), (((float)((int32_t)((first + column) + (uint32_t)1))) * -1.000000e+00f));
          if ((((uint32_t)selected) % cores) == core) {
            pool.SetValue((((((((uint32_t)selected) / cores) * stride) + k) + first) + column), 1.000000e+00f);
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
            tl::ascend::pv_store_rows(W[((work + ((uint64_t)packet_begin)) + ((uint64_t)(column * packet_stride)))], pool[(((((uint32_t)selected) / cores) * stride) + first)], W[((work + ((uint64_t)packet_begin)) + ((uint64_t)(column * packet_stride)))], 1, packet_stride, 0, 0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
          }
        }
      }
      tl::ascend::pv_cross_set0(pool[0], pool[0], pool[0], 1);
      tl::ascend::pv_cross_wait(pool[0], pool[0], pool[0], 1);
      if (valid != 0) {
        tl::ascend::pv_load_rows(pool[packets], W[(work + ((uint64_t)packet_begin))], pool[packets], 1, (rows * packet_stride), 0, 0);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        for (uint32_t column_1 = (uint32_t)0; column_1 < rows; ++column_1) {
          tl::ascend::pv_gather_offsets(pool[product], pool[packets], pool[packet_indices], (column_1 * (uint32_t)4), rows);
          float reciprocal = pool.GetValue(((meta + (uint32_t)16) + column_1));
          tl::ascend::pv_scale(pool[(packets + (column_1 * packet_stride))], pool[(packets + (column_1 * packet_stride))], pool[(packets + (column_1 * packet_stride))], reciprocal, packet_stride);
          if ((column_1 + (uint32_t)1) < rows) {
            tl::ascend::pv_scale(pool[product], pool[product], pool[product], -1.000000e+00f, 16);
            tl::ascend::pv_brcb_rows(pool[factors], pool[product], pool[factors], 16, 8);
            for (uint32_t chunk_1 = (uint32_t)0; chunk_1 < (((packet_stride + (uint32_t)64) - (uint32_t)1) / (uint32_t)64); ++chunk_1) {
              uint32_t col = (chunk_1 * (uint32_t)64);
              tl::ascend::pv_mad_strided(pool[((packets + ((column_1 + (uint32_t)1) * packet_stride)) + col)], pool[((packets + (column_1 * packet_stride)) + col)], pool[(factors + ((column_1 + (uint32_t)1) * (uint32_t)8))], min((packet_stride - col), (uint32_t)64), ((rows - column_1) - (uint32_t)1), 1, 1, 0, (packet_stride / (uint32_t)8), 0, 1);
            }
          }
        }
        for (uint32_t column_2 = (uint32_t)0; column_2 < rows; ++column_2) {
          int32_t selected_1 = ((int32_t)pool.GetValue((meta + column_2)));
          if ((((uint32_t)selected_1) % cores) == core) {
            tl::ascend::pv_copy(pool[(((((uint32_t)selected_1) / cores) * stride) + first)], pool[(packets + (column_2 * packet_stride))], pool[(((((uint32_t)selected_1) / cores) * stride) + first)], packet_stride);
          }
          tl::ascend::pv_gather_offsets(pool[vector], pool[0], pool[row_indices], ((first + column_2) * (uint32_t)4), factor_rows);
          if ((((uint32_t)selected_1) % cores) == core) {
            AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
            pool.SetValue((vector + (((uint32_t)selected_1) / cores)), 0.000000e+00f);
            AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
          }
          tl::ascend::pv_scale(pool[vector], pool[vector], pool[vector], -1.000000e+00f, factor_rows);
          tl::ascend::pv_brcb_rows(pool[factors], pool[vector], pool[factors], factor_rows, 8);
          uint32_t pivot_column = (first + column_2);
          bool consumed = (((pivot_column % (uint32_t)8) == (uint32_t)7) || ((pivot_column + (uint32_t)1) == n));
          int32_t condval_2;
          if (consumed) {
            condval_2 = 8;
          } else {
            condval_2 = 0;
          }
          uint32_t live_begin = (((pivot_column / (uint32_t)8) * (uint32_t)8) + ((uint32_t)condval_2));
          int32_t condval_3;
          if (consumed) {
            condval_3 = 0;
          } else {
            condval_3 = 8;
          }
          uint32_t live_limit = ((live_begin + k) + ((uint32_t)condval_3));
          for (uint32_t chunk_2 = (uint32_t)0; chunk_2 < ((((live_limit - live_begin) + (uint32_t)64) - (uint32_t)1) / (uint32_t)64); ++chunk_2) {
            uint32_t col_1 = (live_begin + (chunk_2 * (uint32_t)64));
            tl::ascend::pv_mad_strided(pool[col_1], pool[(((packets + (column_2 * packet_stride)) + col_1) - first)], pool[factors], min((live_limit - col_1), (uint32_t)64), owned, 1, 1, 0, (stride / (uint32_t)8), 0, 1);
          }
        }
      }
      if ((((uint32_t)0 < window) && ((first + rows) == window_end)) && (window_end < n)) {
        tl::ascend::pv_all_barrier(pool[0], pool[0], pool[0]);
        tl::ascend::pv_cross_set0(pool[0], pool[0], pool[0], 3);
        tl::ascend::pv_cross_wait(pool[0], pool[0], pool[0], 3);
        if (valid != 0) {
          AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
          AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
          uint32_t next_columns = min(window, (n - window_end));
          tl::ascend::pv_store_rows(W[((work + ((uint64_t)(core * k))) + ((uint64_t)window_end))], pool[window_end], W[((work + ((uint64_t)(core * k))) + ((uint64_t)window_end))], owned, next_columns, (stride - next_columns), ((cores * k) - next_columns));
        }
        tl::ascend::pv_all_barrier(pool[0], pool[0], pool[0]);
        tl::ascend::pv_cross_set0(pool[0], pool[0], pool[0], 2);
        tl::ascend::pv_cross_wait(pool[0], pool[0], pool[0], 2);
        tl::ascend::pv_cross_set0(pool[0], pool[0], pool[0], 4);
        tl::ascend::pv_cross_wait(pool[0], pool[0], pool[0], 4);
        window_end = min(n, (window_end + window));
      }
    }
    inverse = 0.000000e+00f;
    peak = 0.000000e+00f;
    if (valid != 0) {
      AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
      AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
      for (uint32_t j = (uint32_t)0; j < n; ++j) {
        int32_t index_1 = ((((int32_t)(pool.GetValue((norms + j)) * -1.000000e+00f)) * 4) - 4);
        tl::ascend::pv_store_u32(pool[0], pool[0], pool[0], (mapping + j), ((uint32_t)index_1));
      }
      AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
      AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
      tl::ascend::pv_fill(pool[accumulator], pool[accumulator], pool[accumulator], 0.000000e+00f, wide);
      AscendC::PipeBarrier<PIPE_V>();
      for (uint32_t i_2 = (uint32_t)0; i_2 < owned; ++i_2) {
        tl::ascend::pv_gather_offsets(pool[(output + (i_2 * k))], pool[((i_2 * stride) + k)], pool[mapping], 0, n);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::pv_mul(pool[product], pool[(output + (i_2 * k))], pool[(output + (i_2 * k))], n);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::pv_add(pool[accumulator], pool[accumulator], pool[product], n);
        AscendC::PipeBarrier<PIPE_V>();
      }
      tl::ascend::pv_reduce_sum(pool[red], pool[accumulator], pool[scratch], k);
      AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
      AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
      inverse = pool.GetValue(red);
      tl::ascend::pv_abs(pool[0], pool[output], pool[0], (owned * k));
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::pv_reduce_max(pool[red], pool[0], pool[(owned * k)], (owned * k));
      AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
      AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
      peak = (pool.GetValue(red) / power);
      for (uint32_t i_3 = (uint32_t)0; i_3 < owned; ++i_3) {
        tl::ascend::pv_scale(pool[(output + (i_3 * k))], pool[(output + (i_3 * k))], pool[(output + (i_3 * k))], (1.000000e+00f / power), k);
        AscendC::PipeBarrier<PIPE_V>();
        int32_t target = (((int32_t)(pool.GetValue(((norms + core) + (i_3 * cores))) * -1.000000e+00f)) - 1);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        tl::ascend::pv_store_rows(X[(((((uint64_t)job) * ((uint64_t)n)) * ((uint64_t)n)) + ((uint64_t)(((uint32_t)target) * n)))], pool[(output + (i_3 * k))], X[(((((uint64_t)job) * ((uint64_t)n)) * ((uint64_t)n)) + ((uint64_t)(((uint32_t)target) * n)))], 1, n, 0, 0);
      }
      tl::ascend::pv_all_barrier(pool[0], pool[0], pool[0]);
    }
    pool.SetValue(0, ((float)valid));
    pool.SetValue(1, inverse);
    pool.SetValue(2, peak);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
    tl::ascend::pv_store_rows(W[((metadata + (uint64_t)2048) + ((uint64_t)(core * (uint32_t)8)))], pool[0], W[((metadata + (uint64_t)2048) + ((uint64_t)(core * (uint32_t)8)))], 1, 3, 0, 0);
    tl::ascend::pv_all_barrier(pool[0], pool[0], pool[0]);
  }
  pipe.Destroy();
}

}
namespace tl::ascend {
__aicore__ inline void PvDslPipelineProducer( GM_ADDR A_handle,  GM_ADDR X_handle,  GM_ADDR W_handle, int32_t blocks, int32_t n, int32_t stride, int32_t all_batch, int32_t job, int32_t core, int32_t cores, int32_t window, uint64_t fftsAddr) {
  AscendC::TPipe pipe;

  AscendC::GlobalTensor<float> A;
  A.SetGlobalBuffer((__gm__ float*)A_handle);
  AscendC::GlobalTensor<float> X;
  X.SetGlobalBuffer((__gm__ float*)X_handle);
  AscendC::GlobalTensor<float> W;
  W.SetGlobalBuffer((__gm__ float*)W_handle);

  AscendC::TBuf<AscendC::TPosition::VECCALC> ascend_ub; pipe.InitBuffer(ascend_ub, 196352);
  auto cid = AscendC::GetBlockIdx();
  if ASCEND_IS_AIV {
    }
  int k = 0;
  int wide = 0;
  int capacity = 0;
  float magnitude = 0.000000e+00f;
  float total = 0.000000e+00f;
  float power = 0.000000e+00f;
  auto pool = ascend_ub.GetWithOffset<float>(49088, 0);
  int extent = 0;
  int valid = 0;
  int size = 0;
  int pitch = 0;
  int window_end = 0;
  int columns = 0;
  int next_pitch = 0;
  float best = 0.000000e+00f;
  int chunk = 0;
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIV {
    tl::ascend::pv_sync_init(W[0], W[0], W[0]);
    k = (((n + 7) / 8) * 8);
    wide = (((k + 63) / 64) * 64);
    capacity = (n * k);
    if (0 < window) {
      capacity = (n * window);
    }
    int32_t scratch = capacity;
    int32_t vector = (((k * 8) + scratch) + 4096);
    int32_t product = (vector + wide);
    int32_t temporary = (product + wide);
    int32_t permutation = ((temporary + k) + 64);
    int32_t records = (permutation + k);
    magnitude = 0.000000e+00f;
    total = 0.000000e+00f;
    power = 1.000000e+00f;
    if (window == 0) {
      tl::ascend::pv_load_rows(pool[0], A[((job * n) * n)], pool[0], n, n, 0, 0);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
    }
    extent = (n * k);
    if (0 < window) {
      extent = (n * n);
    }
    for (int32_t chunk_1 = 0; chunk_1 < ((extent + 4095) / 4096); ++chunk_1) {
      int32_t count = min((extent - (chunk_1 * 4096)), 4096);
      if (0 < window) {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
        tl::ascend::pv_load_rows(pool[scratch], A[((chunk_1 * 4096) + ((job * n) * n))], pool[scratch], 1, count, 0, 0);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        tl::ascend::pv_abs(pool[scratch], pool[scratch], pool[scratch], count);
      } else {
        tl::ascend::pv_abs(pool[scratch], pool[(chunk_1 * 4096)], pool[scratch], count);
      }
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::pv_reduce_max(pool[(temporary + 32)], pool[scratch], pool[product], count);
      AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
      AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
      magnitude = max(magnitude, pool.GetValue((((int64_t)temporary) + (int64_t)32)));
    }
    valid = 1;
    if ((magnitude < 1.000000e-01f) || (1.701412e+38f <= magnitude)) {
      valid = 0;
    }
    if (valid != 0) {
      while (1) {
        if (!((power < magnitude))) { break; }
        power = (power * 2.000000e+00f);
      }
      while (1) {
        if (!((magnitude <= (power * 5.000000e-01f)))) { break; }
        power = (power * 5.000000e-01f);
      }
    }
    if (window == 0) {
      for (int32_t chunk_2 = 0; chunk_2 < ((extent + 16319) / 16320); ++chunk_2) {
        tl::ascend::pv_scale(pool[(chunk_2 * 16320)], pool[(chunk_2 * 16320)], pool[(chunk_2 * 16320)], (1.000000e+00f / power), min((extent - (chunk_2 * 16320)), 16320));
      }
      AscendC::PipeBarrier<PIPE_V>();
    }
    for (int32_t chunk_3 = 0; chunk_3 < ((extent + 4095) / 4096); ++chunk_3) {
      int32_t count_1 = min((extent - (chunk_3 * 4096)), 4096);
      if (0 < window) {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
        tl::ascend::pv_load_rows(pool[scratch], A[((chunk_3 * 4096) + ((job * n) * n))], pool[scratch], 1, count_1, 0, 0);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        tl::ascend::pv_scale(pool[scratch], pool[scratch], pool[scratch], (1.000000e+00f / power), count_1);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::pv_mul(pool[scratch], pool[scratch], pool[scratch], count_1);
      } else {
        tl::ascend::pv_mul(pool[scratch], pool[(chunk_3 * 4096)], pool[(chunk_3 * 4096)], count_1);
      }
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::pv_reduce_sum(pool[(temporary + 32)], pool[scratch], pool[product], count_1);
      AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
      AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
      total = (total + pool.GetValue((((int64_t)temporary) + (int64_t)32)));
    }
    if (0 < window) {
      tl::ascend::pv_load_rows(pool[0], A[((job * n) * n)], pool[0], n, window, (n - window), 0);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
      for (int32_t chunk_4 = 0; chunk_4 < (((n * window) + 16319) / 16320); ++chunk_4) {
        tl::ascend::pv_scale(pool[(chunk_4 * 16320)], pool[(chunk_4 * 16320)], pool[(chunk_4 * 16320)], (1.000000e+00f / power), min(((n * window) - (chunk_4 * 16320)), 16320));
      }
      AscendC::PipeBarrier<PIPE_V>();
    }
    for (int32_t row = 0; row < k; ++row) {
      pool.SetValue((((int64_t)permutation) + ((int64_t)row)), ((float)row));
    }
    AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
    size = n;
    pitch = k;
    if (0 < window) {
      pitch = window;
    }
    window_end = n;
    if (0 < window) {
      window_end = min(n, window);
    }
    for (int32_t panel = 0; panel < ((n + 15) / 16); ++panel) {
      if ((0 < window) && ((panel * 16) == window_end)) {
        window_end = min(n, (window_end + window));
        tl::ascend::pv_all_barrier(pool[0], pool[0], pool[0]);
        tl::ascend::pv_cross_set0(pool[0], pool[0], pool[0], 3);
        tl::ascend::pv_cross_wait(pool[0], pool[0], pool[0], 3);
        tl::ascend::pv_cross_set0(pool[0], pool[0], pool[0], 2);
        tl::ascend::pv_cross_wait(pool[0], pool[0], pool[0], 2);
        if (valid != 0) {
          for (int32_t row_1 = 0; row_1 < size; ++row_1) {
            int32_t original = ((int32_t)pool.GetValue((((int64_t)permutation) + ((int64_t)row_1))));
            tl::ascend::pv_load_rows(pool[(row_1 * pitch)], W[(((panel * 16) + ((job * n) * stride)) + (original * k))], pool[(row_1 * pitch)], 1, (window_end - (panel * 16)), 0, 0);
          }
          AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        }
        tl::ascend::pv_all_barrier(pool[0], pool[0], pool[0]);
        tl::ascend::pv_cross_set0(pool[0], pool[0], pool[0], 4);
        tl::ascend::pv_cross_wait(pool[0], pool[0], pool[0], 4);
      }
      if (valid != 0) {
        tl::ascend::pv_iota(pool[(temporary + 64)], pool[(temporary + 64)], pool[(temporary + 64)], size);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::pv_scale_indices(pool[(temporary + 64)], pool[(temporary + 64)], pool[(temporary + 64)], (pitch * 4), size);
        AscendC::PipeBarrier<PIPE_V>();
        int32_t block = min(size, 16);
        for (int32_t pivot = 0; pivot < block; ++pivot) {
          int32_t remaining = (size - pivot);
          tl::ascend::pv_gather_offsets(pool[vector], pool[0], pool[(temporary + 64)], ((pivot * (pitch + 1)) * 4), remaining);
          AscendC::PipeBarrier<PIPE_V>();
          tl::ascend::pv_abs(pool[product], pool[vector], pool[product], remaining);
          AscendC::PipeBarrier<PIPE_V>();
          if (63 < remaining) {
            tl::ascend::pv_whole_max(pool[(temporary + 32)], pool[product], pool[(temporary + 32)], 64, (remaining / 64));
          }
          if (0 < (remaining % 64)) {
            tl::ascend::pv_whole_max(pool[((((remaining / 64) * 2) + temporary) + 32)], pool[(((remaining / 64) * 64) + product)], pool[((((remaining / 64) * 2) + temporary) + 32)], (remaining % 64), 1);
          }
          AscendC::PipeBarrier<PIPE_V>();
          int32_t condval;
          if ((0 < (remaining % 64))) {
            condval = 1;
          } else {
            condval = 0;
          }
          tl::ascend::pv_even_max(pool[temporary], pool[(temporary + 32)], pool[temporary], ((remaining / 64) + condval));
          AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
          AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
          tl::ascend::pv_reduce_scalars(pool[0], pool[0], pool[0], best, chunk);
            float v_ = pool.GetValue(((((int64_t)temporary) + ((int64_t)chunk)) + (int64_t)33));
          int32_t selected = ((((chunk / 2) * 64) + pivot) + ((int32_t)(*(uint32_t *)(&(v_)))));
          if (best <= 1.000000e-06f) {
            valid = 0;
            break;
          }
          float divisor = pool.GetValue(((((int64_t)vector) + ((int64_t)selected)) - ((int64_t)pivot)));
          if (selected != pivot) {
            float old = pool.GetValue((((int64_t)permutation) + ((int64_t)pivot)));
            pool.SetValue((((int64_t)permutation) + ((int64_t)pivot)), pool.GetValue((((int64_t)permutation) + ((int64_t)selected))));
            pool.SetValue((((int64_t)permutation) + ((int64_t)selected)), old);
          }
          pool.SetValue((((((int64_t)panel) * (int64_t)64) + ((int64_t)records)) + ((int64_t)pivot)), pool.GetValue((((int64_t)permutation) + ((int64_t)pivot))));
          pool.SetValue(((((((int64_t)panel) * (int64_t)64) + ((int64_t)records)) + ((int64_t)pivot)) + (int64_t)16), (1.000000e+00f / divisor));
          if (1 < remaining) {
            pool.SetValue(((((int64_t)vector) + ((int64_t)selected)) - ((int64_t)pivot)), 0.000000e+00f);
            tl::ascend::pv_copy(pool[scratch], pool[(selected * pitch)], pool[scratch], pitch);
            AscendC::PipeBarrier<PIPE_V>();
            tl::ascend::pv_scale(pool[scratch], pool[scratch], pool[scratch], (-1.000000e+00f / divisor), pitch);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
            tl::ascend::pv_brcb_rows(pool[(scratch + 4096)], pool[vector], pool[(scratch + 4096)], (((remaining + 7) / 8) * 8), 8);
            AscendC::PipeBarrier<PIPE_V>();
            for (int32_t chunk_5 = 0; chunk_5 < (((pitch + 63) - ((pivot / 8) * 8)) / 64); ++chunk_5) {
              for (int32_t row_chunk = 0; row_chunk < ((remaining + 254) / 255); ++row_chunk) {
                tl::ascend::pv_mad_strided(pool[(((chunk_5 * 64) + ((pivot / 8) * 8)) + (((row_chunk * 255) + pivot) * pitch))], pool[(((chunk_5 * 64) + ((pivot / 8) * 8)) + scratch)], pool[(((row_chunk * 2040) + scratch) + 4096)], min(((pitch - ((pivot / 8) * 8)) - (chunk_5 * 64)), 64), min((remaining - (row_chunk * 255)), 255), 1, 1, 0, (pitch / 8), 0, 1);
              }
            }
            AscendC::PipeBarrier<PIPE_V>();
            if (selected != pivot) {
              tl::ascend::pv_copy(pool[(selected * pitch)], pool[(pivot * pitch)], pool[(selected * pitch)], pitch);
              AscendC::PipeBarrier<PIPE_V>();
            }
          }
        }
        int32_t remaining_1 = (size - block);
        columns = remaining_1;
        if (0 < window) {
          columns = min(remaining_1, ((window_end - block) - (panel * 16)));
        }
        if ((0 < window) && (((panel * 16) + block) == window_end)) {
          columns = min(window, remaining_1);
        }
        next_pitch = (((columns + 7) / 8) * 8);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
        if ((window <= 0) || (((panel * 16) + block) != window_end)) {
          for (int32_t row_chunk_1 = 0; row_chunk_1 < ((remaining_1 + 15) / 16); ++row_chunk_1) {
            tl::ascend::pv_copy_ub_rows(pool[((row_chunk_1 * next_pitch) * 16)], pool[((((row_chunk_1 * 16) + block) * pitch) + block)], pool[((row_chunk_1 * next_pitch) * 16)], min((remaining_1 - (row_chunk_1 * 16)), 16), next_pitch, (pitch - next_pitch), 0);
            AscendC::PipeBarrier<PIPE_V>();
          }
        }
        if (0 < remaining_1) {
          AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
          AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
          tl::ascend::pv_copy(pool[scratch], pool[(permutation + block)], pool[scratch], (((remaining_1 + 7) / 8) * 8));
          AscendC::PipeBarrier<PIPE_V>();
          tl::ascend::pv_copy(pool[permutation], pool[scratch], pool[permutation], (((remaining_1 + 7) / 8) * 8));
          AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
          AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
        size = remaining_1;
        pitch = next_pitch;
      }
      if (0 < panel) {
        tl::ascend::pv_cross_wait(pool[0], pool[0], pool[0], 0);
        tl::ascend::pv_cross_wait(pool[0], pool[0], pool[0], 1);
      }
      pool.SetValue((((((int64_t)panel) * (int64_t)64) + ((int64_t)records)) + (int64_t)32), power);
      pool.SetValue((((((int64_t)panel) * (int64_t)64) + ((int64_t)records)) + (int64_t)33), total);
      pool.SetValue((((((int64_t)panel) * (int64_t)64) + ((int64_t)records)) + (int64_t)34), ((float)valid));
      AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
      AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
      tl::ascend::pv_store_rows(W[(((job * 8192) + (panel * 64)) + (((all_batch * n) * stride) * 2))], pool[((panel * 64) + records)], W[(((job * 8192) + (panel * 64)) + (((all_batch * n) * stride) * 2))], 1, 64, 0, 0);
      tl::ascend::pv_cross_set0(pool[0], pool[0], pool[0], 0);
      tl::ascend::pv_cross_set0(pool[0], pool[0], pool[0], 1);
    }
    tl::ascend::pv_cross_wait(pool[0], pool[0], pool[0], 0);
    tl::ascend::pv_cross_wait(pool[0], pool[0], pool[0], 1);
    tl::ascend::pv_all_barrier(pool[0], pool[0], pool[0]);
    pool.SetValue(0, ((float)valid));
    pool.SetValue(1, 0.000000e+00f);
    pool.SetValue(2, 0.000000e+00f);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
    tl::ascend::pv_store_rows(W[((((job * 8192) + (core * 8)) + (((all_batch * n) * stride) * 2)) + 2048)], pool[0], W[((((job * 8192) + (core * 8)) + (((all_batch * n) * stride) * 2)) + 2048)], 1, 3, 0, 0);
    tl::ascend::pv_all_barrier(pool[0], pool[0], pool[0]);
  }
  pipe.Destroy();
}

}
namespace tl::ascend {
__aicore__ inline void PvDslPipelinePreparedRows( GM_ADDR A_handle,  GM_ADDR X_handle,  GM_ADDR W_handle, int32_t blocks, uint32_t n, uint32_t stride, uint32_t all_batch, uint32_t job, uint32_t core, uint32_t cores, uint32_t window, uint64_t fftsAddr) {
  AscendC::TPipe pipe;

  AscendC::GlobalTensor<float> A;
  A.SetGlobalBuffer((__gm__ float*)A_handle);
  AscendC::GlobalTensor<float> X;
  X.SetGlobalBuffer((__gm__ float*)X_handle);
  AscendC::GlobalTensor<float> W;
  W.SetGlobalBuffer((__gm__ float*)W_handle);

  AscendC::TBuf<AscendC::TPosition::VECCALC> ascend_ub; pipe.InitBuffer(ascend_ub, 196352);
  auto cid = AscendC::GetBlockIdx();
  if ASCEND_IS_AIV {
    }
  uint32_t k = (uint32_t)0;
  uint32_t wide = (uint32_t)0;
  uint32_t owned = (uint32_t)0;
  uint32_t factor_rows = (uint32_t)0;
  uint32_t packet_stride = (uint32_t)0;
  auto pool = ascend_ub.GetWithOffset<float>(49088, 0);
  int valid = 0;
  float power = 0.000000e+00f;
  uint32_t window_end = (uint32_t)0;
  float inverse = 0.000000e+00f;
  float peak = 0.000000e+00f;
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIV {
    tl::ascend::pv_sync_init(W[0], W[0], W[0]);
    k = ((((n + (uint32_t)8) - (uint32_t)1) / (uint32_t)8) * (uint32_t)8);
    wide = ((((k + (uint32_t)64) - (uint32_t)1) / (uint32_t)64) * (uint32_t)64);
    owned = ((((n + cores) - (uint32_t)1) - core) / cores);
    factor_rows = ((((owned + (uint32_t)8) - (uint32_t)1) / (uint32_t)8) * (uint32_t)8);
    packet_stride = (k + (uint32_t)16);
    uint32_t packets = (owned * stride);
    uint32_t output = (packets + ((uint32_t)16 * packet_stride));
    uint32_t small = (output + (((uint32_t)16 * factor_rows) * (uint32_t)8));
    uint32_t center = (small + (owned * (uint32_t)16));
    uint32_t factors = (output + max(max((owned * k), (uint32_t)512), (((((uint32_t)16 * factor_rows) * (uint32_t)8) + (owned * (uint32_t)16)) + (uint32_t)256)));
    uint32_t meta = (factors + ((uint32_t)8 * max(factor_rows, (uint32_t)16)));
    uint32_t norms = (meta + (uint32_t)64);
    uint32_t vector = (norms + wide);
    uint32_t product = (vector + wide);
    uint32_t accumulator = (product + wide);
    uint32_t scratch = (accumulator + wide);
    uint32_t red = (scratch + wide);
    uint32_t row_indices = (red + (uint32_t)32);
    uint32_t packet_indices = (row_indices + factor_rows);
    uint32_t mapping = (packet_indices + (uint32_t)16);
    uint64_t work = ((((uint64_t)job) * ((uint64_t)n)) * ((uint64_t)stride));
    uint64_t metadata = ((((((uint64_t)all_batch) * (uint64_t)2) * ((uint64_t)n)) * ((uint64_t)stride)) + (((uint64_t)job) * (uint64_t)8192));
    tl::ascend::pv_fill(pool[0], pool[0], pool[0], 0.000000e+00f, (owned * stride));
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
    tl::ascend::pv_load_rows(pool[0], A[(((((uint64_t)job) * ((uint64_t)n)) * ((uint64_t)n)) + ((uint64_t)(core * n)))], pool[0], owned, n, ((cores - (uint32_t)1) * n), (stride - k));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
    tl::ascend::pv_fill(pool[norms], pool[norms], pool[norms], 1.000000e+00f, k);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
    for (uint32_t i = (uint32_t)0; i < factor_rows; ++i) {
      uint32_t condval;
      if ((i < owned)) {
        condval = ((i * (uint32_t)16) * (uint32_t)4);
      } else {
        condval = (uint32_t)0;
      }
      uint32_t index = condval;
      tl::ascend::pv_store_u32(pool[0], pool[0], pool[0], (row_indices + i), index);
    }
    for (int32_t i_1 = 0; i_1 < 16; ++i_1) {
      tl::ascend::pv_store_u32(pool[0], pool[0], pool[0], (packet_indices + ((uint32_t)i_1)), ((((uint32_t)i_1) * packet_stride) * (uint32_t)4));
    }
    AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
    valid = 1;
    power = 1.000000e+00f;
    window_end = n;
    if ((uint32_t)0 < window) {
      window_end = min(n, window);
    }
    for (uint32_t panel = (uint32_t)0; panel < (((n + (uint32_t)16) - (uint32_t)1) / (uint32_t)16); ++panel) {
      uint32_t first = (panel * (uint32_t)16);
      uint32_t rows = min((n - first), (uint32_t)16);
      tl::ascend::pv_all_barrier(pool[0], pool[0], pool[0]);
      tl::ascend::pv_cross_set0(pool[0], pool[0], pool[0], 0);
      tl::ascend::pv_cross_wait(pool[0], pool[0], pool[0], 0);
      tl::ascend::pv_load_rows(pool[meta], W[(metadata + ((uint64_t)(panel * (uint32_t)64)))], pool[meta], 1, 64, 0, 0);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
      if (pool.GetValue((meta + (uint32_t)34)) <= 5.000000e-01f) {
        valid = 0;
      }
      power = pool.GetValue((meta + (uint32_t)32));
      uint32_t condval_1;
      if (((uint32_t)0 < window)) {
        condval_1 = (first * packet_stride);
      } else {
        condval_1 = (uint32_t)0;
      }
      uint32_t packet_begin = condval_1;
      if (valid != 0) {
        if (first == (uint32_t)0) {
          for (uint32_t chunk = (uint32_t)0; chunk < ((((owned * stride) + (uint32_t)16320) - (uint32_t)1) / (uint32_t)16320); ++chunk) {
            uint32_t begin = (chunk * (uint32_t)16320);
            tl::ascend::pv_scale(pool[begin], pool[begin], pool[begin], (1.000000e+00f / power), min(((owned * stride) - begin), (uint32_t)16320));
          }
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
          AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        for (uint32_t column = (uint32_t)0; column < rows; ++column) {
          int32_t selected = ((int32_t)pool.GetValue((meta + column)));
          pool.SetValue((norms + ((uint32_t)selected)), (((float)((int32_t)((first + column) + (uint32_t)1))) * -1.000000e+00f));
          if ((((uint32_t)selected) % cores) == core) {
            pool.SetValue((((((((uint32_t)selected) / cores) * stride) + k) + first) + column), 1.000000e+00f);
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
            tl::ascend::pv_store_rows(W[((work + ((uint64_t)packet_begin)) + ((uint64_t)(column * packet_stride)))], pool[(((((uint32_t)selected) / cores) * stride) + first)], W[((work + ((uint64_t)packet_begin)) + ((uint64_t)(column * packet_stride)))], 1, packet_stride, 0, 0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
          }
        }
      }
      tl::ascend::pv_cross_set0(pool[0], pool[0], pool[0], 1);
      if (valid != 0) {
        tl::ascend::pv_load_rows(pool[center], W[((work + ((uint64_t)(n * packet_stride))) + ((uint64_t)(panel * (uint32_t)256)))], pool[center], 1, 256, 0, 0);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        tl::ascend::pv_copy_ub_rows(pool[small], pool[first], pool[small], owned, 16, (stride - (uint32_t)16), 0);
        tl::ascend::pv_fill(pool[output], pool[output], pool[output], 0.000000e+00f, (((uint32_t)16 * factor_rows) * (uint32_t)8));
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t column_1 = (uint32_t)0; column_1 < rows; ++column_1) {
          int32_t selected_1 = ((int32_t)pool.GetValue((meta + column_1)));
          if ((((uint32_t)selected_1) % cores) == core) {
            tl::ascend::pv_copy(pool[(small + ((((uint32_t)selected_1) / cores) * (uint32_t)16))], pool[(center + (column_1 * (uint32_t)16))], pool[(small + ((((uint32_t)selected_1) / cores) * (uint32_t)16))], 16);
            if ((uint32_t)0 < column_1) {
              tl::ascend::pp_clear_history(pool[(output + ((((uint32_t)selected_1) / cores) * (uint32_t)8))], pool[(output + ((((uint32_t)selected_1) / cores) * (uint32_t)8))], pool[(output + ((((uint32_t)selected_1) / cores) * (uint32_t)8))], column_1, factor_rows);
            }
          }
          tl::ascend::pv_gather_offsets(pool[vector], pool[small], pool[row_indices], (column_1 * (uint32_t)4), factor_rows);
          if ((((uint32_t)selected_1) % cores) == core) {
            AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
            pool.SetValue((vector + (((uint32_t)selected_1) / cores)), 0.000000e+00f);
            AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
          }
          tl::ascend::pv_scale(pool[vector], pool[vector], pool[vector], -1.000000e+00f, factor_rows);
          tl::ascend::pv_brcb_rows(pool[(output + ((column_1 * factor_rows) * (uint32_t)8))], pool[vector], pool[(output + ((column_1 * factor_rows) * (uint32_t)8))], factor_rows, 8);
          tl::ascend::pv_mad_strided(pool[small], pool[(center + (column_1 * (uint32_t)16))], pool[(output + ((column_1 * factor_rows) * (uint32_t)8))], 16, owned, 1, 1, 0, 2, 0, 1);
          AscendC::PipeBarrier<PIPE_V>();
        }
      }
      tl::ascend::pv_cross_wait(pool[0], pool[0], pool[0], 1);
      tl::ascend::pv_all_barrier(pool[0], pool[0], pool[0]);
      tl::ascend::pv_cross_set0(pool[0], pool[0], pool[0], 5);
      tl::ascend::pv_cross_wait(pool[0], pool[0], pool[0], 5);
      if (valid != 0) {
        tl::ascend::pv_load_rows(pool[packets], W[(work + ((uint64_t)packet_begin))], pool[packets], 1, (rows * packet_stride), 0, 0);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        for (uint32_t column_2 = (uint32_t)0; column_2 < rows; ++column_2) {
          int32_t selected_2 = ((int32_t)pool.GetValue((meta + column_2)));
          if ((((uint32_t)selected_2) % cores) == core) {
            tl::ascend::pv_copy(pool[(((((uint32_t)selected_2) / cores) * stride) + first)], pool[(packets + (column_2 * packet_stride))], pool[(((((uint32_t)selected_2) / cores) * stride) + first)], packet_stride);
          }
        }
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t column_3 = (uint32_t)0; column_3 < rows; ++column_3) {
          for (uint32_t chunk_1 = (uint32_t)0; chunk_1 < (((k + (uint32_t)64) - (uint32_t)1) / (uint32_t)64); ++chunk_1) {
            uint32_t col = ((first + (uint32_t)16) + (chunk_1 * (uint32_t)64));
            tl::ascend::pv_mad_strided(pool[col], pool[(((packets + (column_3 * packet_stride)) + (uint32_t)16) + (chunk_1 * (uint32_t)64))], pool[(output + ((column_3 * factor_rows) * (uint32_t)8))], min((k - (chunk_1 * (uint32_t)64)), (uint32_t)64), owned, 1, 1, 0, (stride / (uint32_t)8), 0, 1);
          }
          AscendC::PipeBarrier<PIPE_V>();
        }
      }
      if ((((uint32_t)0 < window) && ((first + rows) == window_end)) && (window_end < n)) {
        tl::ascend::pv_all_barrier(pool[0], pool[0], pool[0]);
        tl::ascend::pv_cross_set0(pool[0], pool[0], pool[0], 3);
        tl::ascend::pv_cross_wait(pool[0], pool[0], pool[0], 3);
        if (valid != 0) {
          AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
          AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
          uint32_t next_columns = min(window, (n - window_end));
          tl::ascend::pv_store_rows(W[((work + ((uint64_t)(core * k))) + ((uint64_t)window_end))], pool[window_end], W[((work + ((uint64_t)(core * k))) + ((uint64_t)window_end))], owned, next_columns, (stride - next_columns), ((cores * k) - next_columns));
        }
        tl::ascend::pv_all_barrier(pool[0], pool[0], pool[0]);
        tl::ascend::pv_cross_set0(pool[0], pool[0], pool[0], 2);
        tl::ascend::pv_cross_wait(pool[0], pool[0], pool[0], 2);
        tl::ascend::pv_cross_set0(pool[0], pool[0], pool[0], 4);
        tl::ascend::pv_cross_wait(pool[0], pool[0], pool[0], 4);
        window_end = min(n, (window_end + window));
      }
    }
    inverse = 0.000000e+00f;
    peak = 0.000000e+00f;
    if (valid != 0) {
      AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
      AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
      for (uint32_t j = (uint32_t)0; j < n; ++j) {
        int32_t index_1 = ((((int32_t)(pool.GetValue((norms + j)) * -1.000000e+00f)) * 4) - 4);
        tl::ascend::pv_store_u32(pool[0], pool[0], pool[0], (mapping + j), ((uint32_t)index_1));
      }
      AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
      AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
      tl::ascend::pv_fill(pool[accumulator], pool[accumulator], pool[accumulator], 0.000000e+00f, wide);
      AscendC::PipeBarrier<PIPE_V>();
      for (uint32_t i_2 = (uint32_t)0; i_2 < owned; ++i_2) {
        tl::ascend::pv_gather_offsets(pool[(output + (i_2 * k))], pool[((i_2 * stride) + k)], pool[mapping], 0, n);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::pv_mul(pool[product], pool[(output + (i_2 * k))], pool[(output + (i_2 * k))], n);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::pv_add(pool[accumulator], pool[accumulator], pool[product], n);
        AscendC::PipeBarrier<PIPE_V>();
      }
      tl::ascend::pv_reduce_sum(pool[red], pool[accumulator], pool[scratch], k);
      AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
      AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
      inverse = pool.GetValue(red);
      tl::ascend::pv_abs(pool[0], pool[output], pool[0], (owned * k));
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::pv_reduce_max(pool[red], pool[0], pool[(owned * k)], (owned * k));
      AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
      AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
      peak = (pool.GetValue(red) / power);
      for (uint32_t i_3 = (uint32_t)0; i_3 < owned; ++i_3) {
        tl::ascend::pv_scale(pool[(output + (i_3 * k))], pool[(output + (i_3 * k))], pool[(output + (i_3 * k))], (1.000000e+00f / power), k);
        AscendC::PipeBarrier<PIPE_V>();
        int32_t target = (((int32_t)(pool.GetValue(((norms + core) + (i_3 * cores))) * -1.000000e+00f)) - 1);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        tl::ascend::pv_store_rows(X[(((((uint64_t)job) * ((uint64_t)n)) * ((uint64_t)n)) + ((uint64_t)(((uint32_t)target) * n)))], pool[(output + (i_3 * k))], X[(((((uint64_t)job) * ((uint64_t)n)) * ((uint64_t)n)) + ((uint64_t)(((uint32_t)target) * n)))], 1, n, 0, 0);
      }
      tl::ascend::pv_all_barrier(pool[0], pool[0], pool[0]);
    }
    pool.SetValue(0, ((float)valid));
    pool.SetValue(1, inverse);
    pool.SetValue(2, peak);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
    tl::ascend::pv_store_rows(W[((metadata + (uint64_t)2048) + ((uint64_t)(core * (uint32_t)8)))], pool[0], W[((metadata + (uint64_t)2048) + ((uint64_t)(core * (uint32_t)8)))], 1, 3, 0, 0);
    tl::ascend::pv_all_barrier(pool[0], pool[0], pool[0]);
  }
  pipe.Destroy();
}

}
namespace tl::ascend {
__aicore__ inline void PvDslPipelinePreparedProducer( GM_ADDR A_handle,  GM_ADDR X_handle,  GM_ADDR W_handle, int32_t blocks, int32_t n, int32_t stride, int32_t all_batch, int32_t job, int32_t core, int32_t cores, int32_t window, uint64_t fftsAddr) {
  AscendC::TPipe pipe;

  AscendC::GlobalTensor<float> A;
  A.SetGlobalBuffer((__gm__ float*)A_handle);
  AscendC::GlobalTensor<float> X;
  X.SetGlobalBuffer((__gm__ float*)X_handle);
  AscendC::GlobalTensor<float> W;
  W.SetGlobalBuffer((__gm__ float*)W_handle);

  AscendC::TBuf<AscendC::TPosition::VECCALC> ascend_ub; pipe.InitBuffer(ascend_ub, 196352);
  auto cid = AscendC::GetBlockIdx();
  if ASCEND_IS_AIV {
    }
  int k = 0;
  int wide = 0;
  int capacity = 0;
  float magnitude = 0.000000e+00f;
  float total = 0.000000e+00f;
  float power = 0.000000e+00f;
  auto pool = ascend_ub.GetWithOffset<float>(49088, 0);
  int extent = 0;
  int valid = 0;
  int size = 0;
  int pitch = 0;
  int window_end = 0;
  int columns = 0;
  int next_pitch = 0;
  float best = 0.000000e+00f;
  int chunk = 0;
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIV {
    tl::ascend::pv_sync_init(W[0], W[0], W[0]);
    k = (((n + 7) / 8) * 8);
    wide = (((k + 63) / 64) * 64);
    capacity = (n * k);
    if (0 < window) {
      capacity = (n * window);
    }
    int32_t scratch = capacity;
    int32_t vector = (((k * 8) + scratch) + 4096);
    int32_t product = (vector + wide);
    int32_t temporary = (product + wide);
    int32_t permutation = ((temporary + k) + 64);
    int32_t records = (permutation + k);
    magnitude = 0.000000e+00f;
    total = 0.000000e+00f;
    power = 1.000000e+00f;
    if (window == 0) {
      tl::ascend::pv_load_rows(pool[0], A[((job * n) * n)], pool[0], n, n, 0, 0);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
    }
    extent = (n * k);
    if (0 < window) {
      extent = (n * n);
    }
    for (int32_t chunk_1 = 0; chunk_1 < ((extent + 4095) / 4096); ++chunk_1) {
      int32_t count = min((extent - (chunk_1 * 4096)), 4096);
      if (0 < window) {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
        tl::ascend::pv_load_rows(pool[scratch], A[((chunk_1 * 4096) + ((job * n) * n))], pool[scratch], 1, count, 0, 0);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        tl::ascend::pv_abs(pool[scratch], pool[scratch], pool[scratch], count);
      } else {
        tl::ascend::pv_abs(pool[scratch], pool[(chunk_1 * 4096)], pool[scratch], count);
      }
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::pv_reduce_max(pool[(temporary + 32)], pool[scratch], pool[product], count);
      AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
      AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
      magnitude = max(magnitude, pool.GetValue((((int64_t)temporary) + (int64_t)32)));
    }
    valid = 1;
    if ((magnitude < 1.000000e-01f) || (1.701412e+38f <= magnitude)) {
      valid = 0;
    }
    if (valid != 0) {
      while (1) {
        if (!((power < magnitude))) { break; }
        power = (power * 2.000000e+00f);
      }
      while (1) {
        if (!((magnitude <= (power * 5.000000e-01f)))) { break; }
        power = (power * 5.000000e-01f);
      }
    }
    if (window == 0) {
      for (int32_t chunk_2 = 0; chunk_2 < ((extent + 16319) / 16320); ++chunk_2) {
        tl::ascend::pv_scale(pool[(chunk_2 * 16320)], pool[(chunk_2 * 16320)], pool[(chunk_2 * 16320)], (1.000000e+00f / power), min((extent - (chunk_2 * 16320)), 16320));
      }
      AscendC::PipeBarrier<PIPE_V>();
    }
    for (int32_t chunk_3 = 0; chunk_3 < ((extent + 4095) / 4096); ++chunk_3) {
      int32_t count_1 = min((extent - (chunk_3 * 4096)), 4096);
      if (0 < window) {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
        tl::ascend::pv_load_rows(pool[scratch], A[((chunk_3 * 4096) + ((job * n) * n))], pool[scratch], 1, count_1, 0, 0);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        tl::ascend::pv_scale(pool[scratch], pool[scratch], pool[scratch], (1.000000e+00f / power), count_1);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::pv_mul(pool[scratch], pool[scratch], pool[scratch], count_1);
      } else {
        tl::ascend::pv_mul(pool[scratch], pool[(chunk_3 * 4096)], pool[(chunk_3 * 4096)], count_1);
      }
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::pv_reduce_sum(pool[(temporary + 32)], pool[scratch], pool[product], count_1);
      AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
      AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
      total = (total + pool.GetValue((((int64_t)temporary) + (int64_t)32)));
    }
    if (0 < window) {
      tl::ascend::pv_load_rows(pool[0], A[((job * n) * n)], pool[0], n, window, (n - window), 0);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
      for (int32_t chunk_4 = 0; chunk_4 < (((n * window) + 16319) / 16320); ++chunk_4) {
        tl::ascend::pv_scale(pool[(chunk_4 * 16320)], pool[(chunk_4 * 16320)], pool[(chunk_4 * 16320)], (1.000000e+00f / power), min(((n * window) - (chunk_4 * 16320)), 16320));
      }
      AscendC::PipeBarrier<PIPE_V>();
    }
    for (int32_t row = 0; row < k; ++row) {
      pool.SetValue((((int64_t)permutation) + ((int64_t)row)), ((float)row));
    }
    AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
    size = n;
    pitch = k;
    if (0 < window) {
      pitch = window;
    }
    window_end = n;
    if (0 < window) {
      window_end = min(n, window);
    }
    for (int32_t panel = 0; panel < ((n + 15) / 16); ++panel) {
      if ((0 < window) && ((panel * 16) == window_end)) {
        window_end = min(n, (window_end + window));
        tl::ascend::pv_all_barrier(pool[0], pool[0], pool[0]);
        tl::ascend::pv_cross_set0(pool[0], pool[0], pool[0], 3);
        tl::ascend::pv_cross_wait(pool[0], pool[0], pool[0], 3);
        tl::ascend::pv_cross_set0(pool[0], pool[0], pool[0], 2);
        tl::ascend::pv_cross_wait(pool[0], pool[0], pool[0], 2);
        if (valid != 0) {
          for (int32_t row_1 = 0; row_1 < size; ++row_1) {
            int32_t original = ((int32_t)pool.GetValue((((int64_t)permutation) + ((int64_t)row_1))));
            tl::ascend::pv_load_rows(pool[(row_1 * pitch)], W[(((panel * 16) + ((job * n) * stride)) + (original * k))], pool[(row_1 * pitch)], 1, (window_end - (panel * 16)), 0, 0);
          }
          AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        }
        tl::ascend::pv_all_barrier(pool[0], pool[0], pool[0]);
        tl::ascend::pv_cross_set0(pool[0], pool[0], pool[0], 4);
        tl::ascend::pv_cross_wait(pool[0], pool[0], pool[0], 4);
      }
      if (valid != 0) {
        tl::ascend::pv_iota(pool[(temporary + 64)], pool[(temporary + 64)], pool[(temporary + 64)], size);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::pv_scale_indices(pool[(temporary + 64)], pool[(temporary + 64)], pool[(temporary + 64)], (pitch * 4), size);
        AscendC::PipeBarrier<PIPE_V>();
        int32_t block = min(size, 16);
        for (int32_t pivot = 0; pivot < block; ++pivot) {
          int32_t remaining = (size - pivot);
          tl::ascend::pv_gather_offsets(pool[vector], pool[0], pool[(temporary + 64)], ((pivot * (pitch + 1)) * 4), remaining);
          AscendC::PipeBarrier<PIPE_V>();
          tl::ascend::pv_abs(pool[product], pool[vector], pool[product], remaining);
          AscendC::PipeBarrier<PIPE_V>();
          if (63 < remaining) {
            tl::ascend::pv_whole_max(pool[(temporary + 32)], pool[product], pool[(temporary + 32)], 64, (remaining / 64));
          }
          if (0 < (remaining % 64)) {
            tl::ascend::pv_whole_max(pool[((((remaining / 64) * 2) + temporary) + 32)], pool[(((remaining / 64) * 64) + product)], pool[((((remaining / 64) * 2) + temporary) + 32)], (remaining % 64), 1);
          }
          AscendC::PipeBarrier<PIPE_V>();
          int32_t condval;
          if ((0 < (remaining % 64))) {
            condval = 1;
          } else {
            condval = 0;
          }
          tl::ascend::pv_even_max(pool[temporary], pool[(temporary + 32)], pool[temporary], ((remaining / 64) + condval));
          AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
          AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
          tl::ascend::pv_reduce_scalars(pool[0], pool[0], pool[0], best, chunk);
            float v_ = pool.GetValue(((((int64_t)temporary) + ((int64_t)chunk)) + (int64_t)33));
          int32_t selected = ((((chunk / 2) * 64) + pivot) + ((int32_t)(*(uint32_t *)(&(v_)))));
          if (best <= 1.000000e-06f) {
            valid = 0;
            break;
          }
          float divisor = pool.GetValue(((((int64_t)vector) + ((int64_t)selected)) - ((int64_t)pivot)));
          if (selected != pivot) {
            float old = pool.GetValue((((int64_t)permutation) + ((int64_t)pivot)));
            pool.SetValue((((int64_t)permutation) + ((int64_t)pivot)), pool.GetValue((((int64_t)permutation) + ((int64_t)selected))));
            pool.SetValue((((int64_t)permutation) + ((int64_t)selected)), old);
          }
          pool.SetValue((((((int64_t)panel) * (int64_t)64) + ((int64_t)records)) + ((int64_t)pivot)), pool.GetValue((((int64_t)permutation) + ((int64_t)pivot))));
          pool.SetValue(((((((int64_t)panel) * (int64_t)64) + ((int64_t)records)) + ((int64_t)pivot)) + (int64_t)16), (1.000000e+00f / divisor));
          tl::ascend::pv_copy(pool[scratch], pool[(selected * pitch)], pool[scratch], pitch);
          AscendC::PipeBarrier<PIPE_V>();
          tl::ascend::pv_scale(pool[scratch], pool[scratch], pool[scratch], (-1.000000e+00f / divisor), pitch);
          AscendC::PipeBarrier<PIPE_V>();
          tl::ascend::pv_scale(pool[(((((panel % 2) * 256) + (((n + 15) / 16) * 64)) + (pivot * 16)) + records)], pool[scratch], pool[(((((panel % 2) * 256) + (((n + 15) / 16) * 64)) + (pivot * 16)) + records)], -1.000000e+00f, 16);
          if (1 < remaining) {
            pool.SetValue(((((int64_t)vector) + ((int64_t)selected)) - ((int64_t)pivot)), 0.000000e+00f);
            AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
            tl::ascend::pv_brcb_rows(pool[(scratch + 4096)], pool[vector], pool[(scratch + 4096)], (((remaining + 7) / 8) * 8), 8);
            AscendC::PipeBarrier<PIPE_V>();
            for (int32_t chunk_5 = 0; chunk_5 < (((pitch + 63) - ((pivot / 8) * 8)) / 64); ++chunk_5) {
              for (int32_t row_chunk = 0; row_chunk < ((remaining + 254) / 255); ++row_chunk) {
                tl::ascend::pv_mad_strided(pool[(((chunk_5 * 64) + ((pivot / 8) * 8)) + (((row_chunk * 255) + pivot) * pitch))], pool[(((chunk_5 * 64) + ((pivot / 8) * 8)) + scratch)], pool[(((row_chunk * 2040) + scratch) + 4096)], min(((pitch - ((pivot / 8) * 8)) - (chunk_5 * 64)), 64), min((remaining - (row_chunk * 255)), 255), 1, 1, 0, (pitch / 8), 0, 1);
              }
            }
            AscendC::PipeBarrier<PIPE_V>();
            if (selected != pivot) {
              tl::ascend::pv_copy(pool[(selected * pitch)], pool[(pivot * pitch)], pool[(selected * pitch)], pitch);
              AscendC::PipeBarrier<PIPE_V>();
            }
          }
        }
        int32_t remaining_1 = (size - block);
        columns = remaining_1;
        if (0 < window) {
          columns = min(remaining_1, ((window_end - block) - (panel * 16)));
        }
        if ((0 < window) && (((panel * 16) + block) == window_end)) {
          columns = min(window, remaining_1);
        }
        next_pitch = (((columns + 7) / 8) * 8);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
        if ((window <= 0) || (((panel * 16) + block) != window_end)) {
          for (int32_t row_chunk_1 = 0; row_chunk_1 < ((remaining_1 + 15) / 16); ++row_chunk_1) {
            tl::ascend::pv_copy_ub_rows(pool[((row_chunk_1 * next_pitch) * 16)], pool[((((row_chunk_1 * 16) + block) * pitch) + block)], pool[((row_chunk_1 * next_pitch) * 16)], min((remaining_1 - (row_chunk_1 * 16)), 16), next_pitch, (pitch - next_pitch), 0);
            AscendC::PipeBarrier<PIPE_V>();
          }
        }
        if (0 < remaining_1) {
          AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
          AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
          tl::ascend::pv_copy(pool[scratch], pool[(permutation + block)], pool[scratch], (((remaining_1 + 7) / 8) * 8));
          AscendC::PipeBarrier<PIPE_V>();
          tl::ascend::pv_copy(pool[permutation], pool[scratch], pool[permutation], (((remaining_1 + 7) / 8) * 8));
          AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
          AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
        size = remaining_1;
        pitch = next_pitch;
      }
      if (0 < panel) {
        tl::ascend::pv_cross_wait(pool[0], pool[0], pool[0], 0);
        tl::ascend::pv_cross_wait(pool[0], pool[0], pool[0], 1);
        tl::ascend::pv_cross_wait(pool[0], pool[0], pool[0], 5);
      }
      if (valid != 0) {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        tl::ascend::pv_store_rows(W[(((panel * 256) + ((job * n) * stride)) + (n * (k + 16)))], pool[((((panel % 2) * 256) + (((n + 15) / 16) * 64)) + records)], W[(((panel * 256) + ((job * n) * stride)) + (n * (k + 16)))], 1, 256, 0, 0);
      }
      pool.SetValue((((((int64_t)panel) * (int64_t)64) + ((int64_t)records)) + (int64_t)32), power);
      pool.SetValue((((((int64_t)panel) * (int64_t)64) + ((int64_t)records)) + (int64_t)33), total);
      pool.SetValue((((((int64_t)panel) * (int64_t)64) + ((int64_t)records)) + (int64_t)34), ((float)valid));
      AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
      AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
      tl::ascend::pv_store_rows(W[(((job * 8192) + (panel * 64)) + (((all_batch * n) * stride) * 2))], pool[((panel * 64) + records)], W[(((job * 8192) + (panel * 64)) + (((all_batch * n) * stride) * 2))], 1, 64, 0, 0);
      tl::ascend::pv_cross_set0(pool[0], pool[0], pool[0], 0);
      tl::ascend::pv_cross_set0(pool[0], pool[0], pool[0], 1);
      tl::ascend::pv_cross_set0(pool[0], pool[0], pool[0], 5);
    }
    tl::ascend::pv_cross_wait(pool[0], pool[0], pool[0], 0);
    tl::ascend::pv_cross_wait(pool[0], pool[0], pool[0], 1);
    tl::ascend::pv_cross_wait(pool[0], pool[0], pool[0], 5);
    tl::ascend::pv_all_barrier(pool[0], pool[0], pool[0]);
    pool.SetValue(0, ((float)valid));
    pool.SetValue(1, 0.000000e+00f);
    pool.SetValue(2, 0.000000e+00f);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
    tl::ascend::pv_store_rows(W[((((job * 8192) + (core * 8)) + (((all_batch * n) * stride) * 2)) + 2048)], pool[0], W[((((job * 8192) + (core * 8)) + (((all_batch * n) * stride) * 2)) + 2048)], 1, 3, 0, 0);
    tl::ascend::pv_all_barrier(pool[0], pool[0], pool[0]);
  }
  pipe.Destroy();
}

}
namespace tl::ascend {
__aicore__ inline void PvDslPipelineTransformer( GM_ADDR A_handle,  GM_ADDR X_handle,  GM_ADDR W_handle, int32_t blocks, uint32_t n, uint32_t stride, uint32_t all_batch, uint32_t job, uint32_t core, uint32_t cores, uint32_t window, uint64_t fftsAddr) {
  AscendC::TPipe pipe;

  AscendC::GlobalTensor<float> A;
  A.SetGlobalBuffer((__gm__ float*)A_handle);
  AscendC::GlobalTensor<float> X;
  X.SetGlobalBuffer((__gm__ float*)X_handle);
  AscendC::GlobalTensor<float> W;
  W.SetGlobalBuffer((__gm__ float*)W_handle);

  AscendC::TBuf<AscendC::TPosition::VECCALC> ascend_ub; pipe.InitBuffer(ascend_ub, 196352);
  auto cid = AscendC::GetBlockIdx();
  if ASCEND_IS_AIV {
    }
  uint32_t k = (uint32_t)0;
  auto pool = ascend_ub.GetWithOffset<float>(49088, 0);
  int valid = 0;
  uint32_t window_end = (uint32_t)0;
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIV {
    tl::ascend::pv_sync_init(W[0], W[0], W[0]);
    k = ((((n + (uint32_t)8) - (uint32_t)1) / (uint32_t)8) * (uint32_t)8);
    k = (k + (uint32_t)16);
    uint32_t factors = ((uint32_t)16 * k);
    uint32_t meta = (factors + (uint32_t)128);
    uint32_t product = (meta + (uint32_t)64);
    uint32_t packet_indices = (product + (uint32_t)256);
    uint64_t work = ((((uint64_t)job) * ((uint64_t)n)) * ((uint64_t)stride));
    uint64_t metadata = ((((((uint64_t)all_batch) * (uint64_t)2) * ((uint64_t)n)) * ((uint64_t)stride)) + (((uint64_t)job) * (uint64_t)8192));
    AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
    for (int32_t i = 0; i < 16; ++i) {
      tl::ascend::pv_store_u32(pool[0], pool[0], pool[0], (packet_indices + ((uint32_t)i)), ((((uint32_t)i) * k) * (uint32_t)4));
    }
    AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
    valid = 1;
    window_end = min(n, window);
    for (uint32_t panel = (uint32_t)0; panel < (((n + (uint32_t)16) - (uint32_t)1) / (uint32_t)16); ++panel) {
      uint32_t first = (panel * (uint32_t)16);
      uint32_t rows = min((n - first), (uint32_t)16);
      uint32_t packet_begin = (first * k);
      tl::ascend::pv_all_barrier(pool[0], pool[0], pool[0]);
      tl::ascend::pv_cross_set0(pool[0], pool[0], pool[0], 0);
      tl::ascend::pv_cross_wait(pool[0], pool[0], pool[0], 0);
      tl::ascend::pv_load_rows(pool[meta], W[(metadata + ((uint64_t)(panel * (uint32_t)64)))], pool[meta], 1, 64, 0, 0);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
      if (pool.GetValue((meta + (uint32_t)34)) <= 5.000000e-01f) {
        valid = 0;
      }
      tl::ascend::pv_cross_set0(pool[0], pool[0], pool[0], 1);
      tl::ascend::pv_cross_wait(pool[0], pool[0], pool[0], 1);
      if (valid != 0) {
        tl::ascend::pv_load_rows(pool[0], W[(work + ((uint64_t)packet_begin))], pool[0], 1, (rows * k), 0, 0);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        for (uint32_t column = (uint32_t)0; column < rows; ++column) {
          tl::ascend::pv_gather_offsets(pool[product], pool[0], pool[packet_indices], (column * (uint32_t)4), rows);
          float reciprocal = pool.GetValue(((meta + (uint32_t)16) + column));
          tl::ascend::pv_scale(pool[(column * k)], pool[(column * k)], pool[(column * k)], reciprocal, k);
          if ((column + (uint32_t)1) < rows) {
            tl::ascend::pv_scale(pool[product], pool[product], pool[product], -1.000000e+00f, 16);
            tl::ascend::pv_brcb_rows(pool[factors], pool[product], pool[factors], 16, 8);
            for (uint32_t chunk = (uint32_t)0; chunk < (((k + (uint32_t)64) - (uint32_t)1) / (uint32_t)64); ++chunk) {
              uint32_t col = (chunk * (uint32_t)64);
              tl::ascend::pv_mad_strided(pool[(((column + (uint32_t)1) * k) + col)], pool[((column * k) + col)], pool[(factors + ((column + (uint32_t)1) * (uint32_t)8))], min((k - col), (uint32_t)64), ((rows - column) - (uint32_t)1), 1, 1, 0, (k / (uint32_t)8), 0, 1);
            }
          }
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        tl::ascend::pv_store_rows(W[(work + ((uint64_t)packet_begin))], pool[0], W[(work + ((uint64_t)packet_begin))], 1, (rows * k), 0, 0);
      }
      tl::ascend::pv_all_barrier(pool[0], pool[0], pool[0]);
      tl::ascend::pv_cross_set0(pool[0], pool[0], pool[0], 5);
      tl::ascend::pv_cross_wait(pool[0], pool[0], pool[0], 5);
      if ((((uint32_t)0 < window) && ((first + rows) == window_end)) && (window_end < n)) {
        tl::ascend::pv_all_barrier(pool[0], pool[0], pool[0]);
        tl::ascend::pv_cross_set0(pool[0], pool[0], pool[0], 3);
        tl::ascend::pv_cross_wait(pool[0], pool[0], pool[0], 3);
        tl::ascend::pv_cross_set0(pool[0], pool[0], pool[0], 2);
        tl::ascend::pv_cross_wait(pool[0], pool[0], pool[0], 2);
        tl::ascend::pv_cross_set0(pool[0], pool[0], pool[0], 4);
        tl::ascend::pv_cross_wait(pool[0], pool[0], pool[0], 4);
        window_end = min(n, (window_end + window));
      }
    }
    tl::ascend::pv_all_barrier(pool[0], pool[0], pool[0]);
    pool.SetValue(0, ((float)valid));
    pool.SetValue(1, 0.000000e+00f);
    pool.SetValue(2, 0.000000e+00f);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
    tl::ascend::pv_store_rows(W[((metadata + (uint64_t)2048) + ((uint64_t)(core * (uint32_t)8)))], pool[0], W[((metadata + (uint64_t)2048) + ((uint64_t)(core * (uint32_t)8)))], 1, 3, 0, 0);
    tl::ascend::pv_all_barrier(pool[0], pool[0], pool[0]);
  }
  pipe.Destroy();
}

}
#undef pv_pipeline_rows
#define pv_pipeline_rows(d,a,b,...) PvDslWindowRows((GM_ADDR)pvPipelineBuffers.inputTensor.GetPhyAddr(),(GM_ADDR)pvPipelineBuffers.outputTensor.GetPhyAddr(),(GM_ADDR)pvPipelineBuffers.workspaceTensor.GetPhyAddr(),__VA_ARGS__,fftsAddr)
#define pv_pipeline_producer(d,a,b,...) PvDslPipelineProducer((GM_ADDR)pvPipelineBuffers.inputTensor.GetPhyAddr(),(GM_ADDR)pvPipelineBuffers.outputTensor.GetPhyAddr(),(GM_ADDR)pvPipelineBuffers.workspaceTensor.GetPhyAddr(),__VA_ARGS__,fftsAddr)
#define pv_pipeline_prepared_rows(d,a,b,...) PvDslPipelinePreparedRows((GM_ADDR)pvPipelineBuffers.inputTensor.GetPhyAddr(),(GM_ADDR)pvPipelineBuffers.outputTensor.GetPhyAddr(),(GM_ADDR)pvPipelineBuffers.workspaceTensor.GetPhyAddr(),__VA_ARGS__,fftsAddr)
#define pv_pipeline_prepared_producer(d,a,b,...) PvDslPipelinePreparedProducer((GM_ADDR)pvPipelineBuffers.inputTensor.GetPhyAddr(),(GM_ADDR)pvPipelineBuffers.outputTensor.GetPhyAddr(),(GM_ADDR)pvPipelineBuffers.workspaceTensor.GetPhyAddr(),__VA_ARGS__,fftsAddr)
#define pv_pipeline_transformer(d,a,b,...) PvDslPipelineTransformer((GM_ADDR)pvPipelineBuffers.inputTensor.GetPhyAddr(),(GM_ADDR)pvPipelineBuffers.outputTensor.GetPhyAddr(),(GM_ADDR)pvPipelineBuffers.workspaceTensor.GetPhyAddr(),__VA_ARGS__,fftsAddr)
namespace tl::ascend {
__aicore__ inline void FusedRaw( GM_ADDR A_handle,  GM_ADDR X_handle,  GM_ADDR W_handle,  GM_ADDR S_handle, int32_t blocks, int32_t n, int32_t batch, int32_t stride, int32_t window, float atol, float rtol, uint64_t fftsAddr) {
  AscendC::TPipe pipe;

  AscendC::GlobalTensor<float> A;
  A.SetGlobalBuffer((__gm__ float*)A_handle);
  AscendC::GlobalTensor<float> X;
  X.SetGlobalBuffer((__gm__ float*)X_handle);
  AscendC::GlobalTensor<float> W;
  W.SetGlobalBuffer((__gm__ float*)W_handle);
  AscendC::GlobalTensor<float> S;
  S.SetGlobalBuffer((__gm__ float*)S_handle);

  AscendC::TBuf<AscendC::TPosition::VECCALC> ascend_ub; pipe.InitBuffer(ascend_ub, 4096);
  auto bid = AscendC::GetBlockIdx();
  if ASCEND_IS_AIV {
    }
  int valid = 0;
  auto pool = ascend_ub.GetWithOffset<float>(1024, 0);
  float inverse_total = 0.000000e+00f;
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIV {
    tl::ascend::pv_pipeline_bind(A[0], X[0], W[0]);
    tl::ascend::pv_sync_init(W[0], W[0], W[0]);
    if (((blocks * 2) / batch) <= ((bid % ((blocks * 2) / batch)) + 1)) {
      tl::ascend::pv_pipeline_producer(W[0], W[0], W[0], blocks, n, stride, batch, (bid / ((blocks * 2) / batch)), (bid % ((blocks * 2) / batch)), (((blocks * 2) / batch) - 1), window);
    } else {
      tl::ascend::pv_pipeline_rows(W[0], W[0], W[0], blocks, n, stride, batch, (bid / ((blocks * 2) / batch)), (bid % ((blocks * 2) / batch)), (((blocks * 2) / batch) - 1), window);
    }
    tl::ascend::pv_sync_all(W[0], W[0], W[0]);
    valid = 1;
    for (int32_t b = 0; b < batch; ++b) {
      tl::ascend::pv_load_rows(pool[0], W[(((b * 8192) + (((batch * n) * stride) * 2)) + 2048)], pool[0], 1, (((blocks * 2) / batch) * 8), 0, 0);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
      inverse_total = 0.000000e+00f;
      for (int32_t c = 0; c < ((blocks * 2) / batch); ++c) {
        if (pool.GetValue((((int64_t)c) * (int64_t)8)) < 5.000000e-01f) {
          valid = 0;
        }
        inverse_total = (inverse_total + pool.GetValue(((((int64_t)c) * (int64_t)8) + (int64_t)1)));
      }
      tl::ascend::pv_load_rows(pool[384], W[((b * 8192) + (((batch * n) * stride) * 2))], pool[384], 1, 64, 0, 0);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
      float power = pool.GetValue(416);
      float total = pool.GetValue(417);
      pool.SetValue(640, total);
      AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
      AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
      tl::ascend::pv_sqrt(pool[640], pool[640], pool[640], 1);
      AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
      AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
      float cutoff = max((pool.GetValue(640) * rtol), (atol / power));
      if (((inverse_total <= 0.000000e+00f) || (8.100000e+01f <= ((inverse_total * cutoff) * cutoff))) || (3.600000e+09f <= (total * inverse_total))) {
        valid = 0;
      }
    }
    if (bid == 0) {
      S.SetValue(0, ((float)valid));
      S.SetValue(31, ((float)valid));
    }
    tl::ascend::pv_all_barrier(W[0], W[0], W[0]);
  }
  pipe.Destroy();
}

}
namespace tl::ascend {
__aicore__ inline void FusedPreparedRaw( GM_ADDR A_handle,  GM_ADDR X_handle,  GM_ADDR W_handle,  GM_ADDR S_handle, int32_t blocks, int32_t n, int32_t batch, int32_t stride, int32_t window, float atol, float rtol, uint64_t fftsAddr) {
  AscendC::TPipe pipe;

  AscendC::GlobalTensor<float> A;
  A.SetGlobalBuffer((__gm__ float*)A_handle);
  AscendC::GlobalTensor<float> X;
  X.SetGlobalBuffer((__gm__ float*)X_handle);
  AscendC::GlobalTensor<float> W;
  W.SetGlobalBuffer((__gm__ float*)W_handle);
  AscendC::GlobalTensor<float> S;
  S.SetGlobalBuffer((__gm__ float*)S_handle);

  AscendC::TBuf<AscendC::TPosition::VECCALC> ascend_ub; pipe.InitBuffer(ascend_ub, 4096);
  auto bid = AscendC::GetBlockIdx();
  if ASCEND_IS_AIV {
    }
  int valid = 0;
  auto pool = ascend_ub.GetWithOffset<float>(1024, 0);
  float inverse_total = 0.000000e+00f;
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIV {
    tl::ascend::pv_pipeline_bind(A[0], X[0], W[0]);
    tl::ascend::pv_sync_init(W[0], W[0], W[0]);
    if ((((bid % ((blocks * 2) / batch)) + 2) < ((blocks * 2) / batch)) || ((bid % ((blocks * 2) / batch)) < 1)) {
      tl::ascend::pv_pipeline_prepared_rows(W[0], W[0], W[0], blocks, n, stride, batch, (bid / ((blocks * 2) / batch)), (bid % ((blocks * 2) / batch)), max((((blocks * 2) / batch) - 2), 1), window);
    } else {
      if ((bid % ((blocks * 2) / batch)) == max((((blocks * 2) / batch) - 2), 1)) {
        tl::ascend::pv_pipeline_transformer(W[0], W[0], W[0], blocks, n, stride, batch, (bid / ((blocks * 2) / batch)), (bid % ((blocks * 2) / batch)), max((((blocks * 2) / batch) - 2), 1), window);
      } else {
        tl::ascend::pv_pipeline_prepared_producer(W[0], W[0], W[0], blocks, n, stride, batch, (bid / ((blocks * 2) / batch)), (bid % ((blocks * 2) / batch)), max((((blocks * 2) / batch) - 2), 1), window);
      }
    }
    tl::ascend::pv_sync_all(W[0], W[0], W[0]);
    valid = 1;
    for (int32_t b = 0; b < batch; ++b) {
      tl::ascend::pv_load_rows(pool[0], W[(((b * 8192) + (((batch * n) * stride) * 2)) + 2048)], pool[0], 1, (((blocks * 2) / batch) * 8), 0, 0);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
      inverse_total = 0.000000e+00f;
      for (int32_t c = 0; c < ((blocks * 2) / batch); ++c) {
        if (pool.GetValue((((int64_t)c) * (int64_t)8)) < 5.000000e-01f) {
          valid = 0;
        }
        inverse_total = (inverse_total + pool.GetValue(((((int64_t)c) * (int64_t)8) + (int64_t)1)));
      }
      tl::ascend::pv_load_rows(pool[384], W[((b * 8192) + (((batch * n) * stride) * 2))], pool[384], 1, 64, 0, 0);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
      float power = pool.GetValue(416);
      float total = pool.GetValue(417);
      pool.SetValue(640, total);
      AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
      AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
      tl::ascend::pv_sqrt(pool[640], pool[640], pool[640], 1);
      AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
      AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
      float cutoff = max((pool.GetValue(640) * rtol), (atol / power));
      if (((inverse_total <= 0.000000e+00f) || (8.100000e+01f <= ((inverse_total * cutoff) * cutoff))) || (3.600000e+09f <= (total * inverse_total))) {
        valid = 0;
      }
    }
    if (bid == 0) {
      S.SetValue(0, ((float)valid));
      S.SetValue(31, ((float)valid));
    }
    tl::ascend::pv_all_barrier(W[0], W[0], W[0]);
  }
  pipe.Destroy();
}

}
namespace tl::ascend {
__aicore__ inline void FusedQuant( GM_ADDR A_handle,  GM_ADDR Y_handle,  GM_ADDR QA_handle,  GM_ADDR QY_handle,  GM_ADDR QN_handle, int32_t blocks, int32_t n, int32_t job, float scale_a, float scale_y, uint64_t fftsAddr) {
  AscendC::TPipe pipe;

  AscendC::GlobalTensor<float> A;
  A.SetGlobalBuffer((__gm__ float*)A_handle);
  AscendC::GlobalTensor<float> Y;
  Y.SetGlobalBuffer((__gm__ float*)Y_handle);
  AscendC::GlobalTensor<int8_t> QA;
  QA.SetGlobalBuffer((__gm__ int8_t*)QA_handle);
  AscendC::GlobalTensor<int8_t> QY;
  QY.SetGlobalBuffer((__gm__ int8_t*)QY_handle);
  AscendC::GlobalTensor<float> QN;
  QN.SetGlobalBuffer((__gm__ float*)QN_handle);

  AscendC::TBuf<AscendC::TPosition::VECCALC> ascend_ub; pipe.InitBuffer(ascend_ub, 63488);
  auto core = AscendC::GetBlockIdx();
  if ASCEND_IS_AIV {
    }
  auto values = ascend_ub.GetWithOffset<float>(2048, 0);
  auto integers = ascend_ub.GetWithOffset<int>(2048, 8192);
  auto high = ascend_ub.GetWithOffset<float>(2048, 16384);
  auto low = ascend_ub.GetWithOffset<float>(2048, 24576);
  auto temp = ascend_ub.GetWithOffset<float>(2048, 32768);
  auto carry = ascend_ub.GetWithOffset<float>(2048, 49152);
  auto rounded = ascend_ub.GetWithOffset<float>(2048, 40960);
  auto half_buf = ascend_ub.GetWithOffset<half>(2048, 57344);
  auto small = ascend_ub.GetWithOffset<int8_t>(2048, 61440);
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIV {
    tl::ascend::pv_fill(values[0], values[0], values[0], 0.000000e+00f, 2048);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
    tl::ascend::pv_load_rows(values[0], A[(n * ((job * n) + core))], values[0], ((((n + blocks) - core) - 1) / blocks), n, ((blocks - 1) * n), 0);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
    {
    AscendC::Muls(values[0], values[0], (2.560000e+02f / scale_a), 2048);
    }
    AscendC::PipeBarrier<PIPE_V>();
    {
    AscendC::Muls(values[0], values[0], 6.553600e+04f, 2048);
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(integers[0], values[0], AscendC::RoundMode::CAST_RINT, 2048);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(high[0], integers[0], AscendC::RoundMode::CAST_RINT, 2048);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub(values[0], values[0], high[0], 2048);
    AscendC::PipeBarrier<PIPE_V>();
    {
    AscendC::Muls(values[0], values[0], 6.553600e+04f, 2048);
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(integers[0], values[0], AscendC::RoundMode::CAST_RINT, 2048);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(low[0], integers[0], AscendC::RoundMode::CAST_RINT, 2048);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub(values[0], values[0], low[0], 2048);
    AscendC::PipeBarrier<PIPE_V>();
    {
    AscendC::Muls(values[0], values[0], 2.560000e+02f, 2048);
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(temp[0], values[0], values[0], 2048);
    AscendC::PipeBarrier<PIPE_V>();
    tl::ascend::pv_reduce_sum(carry[0], temp[0], rounded[0], 2048);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    tl::ascend::pv_store_rows(QN[(core * 16)], carry[0], QN[(core * 16)], 1, 8, 0, 0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    for (int32_t part = 0; part < 2; ++part) {
      for (int32_t digit = 0; digit < 2; ++digit) {
        {
        AscendC::Adds(carry[0], low[0], 1.280000e+02f, 2048);
        }
        AscendC::PipeBarrier<PIPE_V>();
        {
        AscendC::Muls(carry[0], carry[0], 3.906250e-03f, 2048);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(integers[0], carry[0], AscendC::RoundMode::CAST_FLOOR, 2048);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(carry[0], integers[0], AscendC::RoundMode::CAST_RINT, 2048);
        AscendC::PipeBarrier<PIPE_V>();
        {
        AscendC::Muls(temp[0], carry[0], 2.560000e+02f, 2048);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(rounded[0], low[0], temp[0], 2048);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::rq_half(half_buf[0], rounded[0], half_buf[0], 2048);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::rq_i8(small[0], half_buf[0], small[0], 2048);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        tl::ascend::rq_store(QA[(n * ((((4 - digit) - (part * 2)) * n) + core))], small[0], QA[(n * ((((4 - digit) - (part * 2)) * n) + core))], ((((n + blocks) - core) - 1) / blocks), n, ((blocks - 1) * n));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        tl::ascend::pv_copy(low[0], carry[0], low[0], 2048);
        AscendC::PipeBarrier<PIPE_V>();
      }
      if (part == 0) {
        AscendC::Add(low[0], high[0], low[0], 2048);
        AscendC::PipeBarrier<PIPE_V>();
      }
    }
    tl::ascend::rq_half(half_buf[0], low[0], half_buf[0], 2048);
    AscendC::PipeBarrier<PIPE_V>();
    tl::ascend::rq_i8(small[0], half_buf[0], small[0], 2048);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    tl::ascend::rq_store(QA[(core * n)], small[0], QA[(core * n)], ((((n + blocks) - core) - 1) / blocks), n, ((blocks - 1) * n));
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    tl::ascend::pv_fill(values[0], values[0], values[0], 0.000000e+00f, 2048);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
    tl::ascend::pv_load_rows(values[0], Y[(n * ((job * n) + core))], values[0], ((((n + blocks) - core) - 1) / blocks), n, ((blocks - 1) * n), 0);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
    {
    AscendC::Muls(values[0], values[0], (2.560000e+02f / scale_y), 2048);
    }
    AscendC::PipeBarrier<PIPE_V>();
    {
    AscendC::Muls(values[0], values[0], 6.553600e+04f, 2048);
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(integers[0], values[0], AscendC::RoundMode::CAST_RINT, 2048);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(high[0], integers[0], AscendC::RoundMode::CAST_RINT, 2048);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub(values[0], values[0], high[0], 2048);
    AscendC::PipeBarrier<PIPE_V>();
    {
    AscendC::Muls(values[0], values[0], 6.553600e+04f, 2048);
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(integers[0], values[0], AscendC::RoundMode::CAST_RINT, 2048);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(low[0], integers[0], AscendC::RoundMode::CAST_RINT, 2048);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub(values[0], values[0], low[0], 2048);
    AscendC::PipeBarrier<PIPE_V>();
    {
    AscendC::Muls(values[0], values[0], 2.560000e+02f, 2048);
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(temp[0], values[0], values[0], 2048);
    AscendC::PipeBarrier<PIPE_V>();
    tl::ascend::pv_reduce_sum(carry[0], temp[0], rounded[0], 2048);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    tl::ascend::pv_store_rows(QN[((core * 16) + 8)], carry[0], QN[((core * 16) + 8)], 1, 8, 0, 0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    for (int32_t part_1 = 0; part_1 < 2; ++part_1) {
      for (int32_t digit_1 = 0; digit_1 < 2; ++digit_1) {
        {
        AscendC::Adds(carry[0], low[0], 1.280000e+02f, 2048);
        }
        AscendC::PipeBarrier<PIPE_V>();
        {
        AscendC::Muls(carry[0], carry[0], 3.906250e-03f, 2048);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(integers[0], carry[0], AscendC::RoundMode::CAST_FLOOR, 2048);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(carry[0], integers[0], AscendC::RoundMode::CAST_RINT, 2048);
        AscendC::PipeBarrier<PIPE_V>();
        {
        AscendC::Muls(temp[0], carry[0], 2.560000e+02f, 2048);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(rounded[0], low[0], temp[0], 2048);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::rq_half(half_buf[0], rounded[0], half_buf[0], 2048);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::rq_i8(small[0], half_buf[0], small[0], 2048);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        tl::ascend::rq_store(QY[(((core * n) * 5) + (((4 - digit_1) - (part_1 * 2)) * n))], small[0], QY[(((core * n) * 5) + (((4 - digit_1) - (part_1 * 2)) * n))], ((((n + blocks) - core) - 1) / blocks), n, (((blocks * 5) - 1) * n));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        tl::ascend::pv_copy(low[0], carry[0], low[0], 2048);
        AscendC::PipeBarrier<PIPE_V>();
      }
      if (part_1 == 0) {
        AscendC::Add(low[0], high[0], low[0], 2048);
        AscendC::PipeBarrier<PIPE_V>();
      }
    }
    tl::ascend::rq_half(half_buf[0], low[0], half_buf[0], 2048);
    AscendC::PipeBarrier<PIPE_V>();
    tl::ascend::rq_i8(small[0], half_buf[0], small[0], 2048);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    tl::ascend::rq_store(QY[((core * n) * 5)], small[0], QY[((core * n) * 5)], ((((n + blocks) - core) - 1) / blocks), n, (((blocks * 5) - 1) * n));
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
  }
  pipe.Destroy();
}

}
namespace tl::ascend {
__aicore__ inline void FusedDecode( GM_ADDR P_handle,  GM_ADDR Y_handle,  GM_ADDR R_handle,  GM_ADDR YH_handle,  GM_ADDR RH_handle, int32_t blocks, int32_t n, int32_t job, float scale_a, float scale_y, uint64_t fftsAddr) {
  AscendC::TPipe pipe;

  AscendC::GlobalTensor<int> P;
  P.SetGlobalBuffer((__gm__ int*)P_handle);
  AscendC::GlobalTensor<float> Y;
  Y.SetGlobalBuffer((__gm__ float*)Y_handle);
  AscendC::GlobalTensor<float> R;
  R.SetGlobalBuffer((__gm__ float*)R_handle);
  AscendC::GlobalTensor<half> YH;
  YH.SetGlobalBuffer((__gm__ half*)YH_handle);
  AscendC::GlobalTensor<half> RH;
  RH.SetGlobalBuffer((__gm__ half*)RH_handle);

  AscendC::TBuf<AscendC::TPosition::VECCALC> ascend_ub; pipe.InitBuffer(ascend_ub, 106496);
  auto core = AscendC::GetBlockIdx();
  if ASCEND_IS_AIV {
    }
  auto high = ascend_ub.GetWithOffset<float>(2048, 0);
  auto low = ascend_ub.GetWithOffset<float>(2048, 8192);
  auto raw = ascend_ub.GetWithOffset<int>(2048, 16384);
  float weight = 0.000000e+00f;
  auto part = ascend_ub.GetWithOffset<float>(2048, 24576);
  auto roundtrip = ascend_ub.GetWithOffset<int>(2048, 32768);
  auto remainder = ascend_ub.GetWithOffset<float>(2048, 40960);
  auto total = ascend_ub.GetWithOffset<float>(2048, 49152);
  auto bv = ascend_ub.GetWithOffset<float>(2048, 57344);
  auto av = ascend_ub.GetWithOffset<float>(2048, 65536);
  auto error = ascend_ub.GetWithOffset<float>(2048, 73728);
  auto alt = ascend_ub.GetWithOffset<float>(2048, 81920);
  auto h_r = ascend_ub.GetWithOffset<half>(2048, 90112);
  auto cache = ascend_ub.GetWithOffset<float>(2048, 94208);
  auto h_y = ascend_ub.GetWithOffset<half>(2048, 102400);
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIV {
    tl::ascend::pv_fill(high[0], high[0], high[0], 0.000000e+00f, 2048);
    tl::ascend::pv_fill(low[0], low[0], low[0], 0.000000e+00f, 2048);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(raw[0], low[0], AscendC::RoundMode::CAST_RINT, 2048);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
    for (int32_t row = 0; row < ((((n + blocks) - core) - 1) / blocks); ++row) {
      high.SetValue((((((int64_t)row) * ((int64_t)n)) + ((int64_t)core)) + (((int64_t)row) * ((int64_t)blocks))), 1.000000e+00f);
    }
    AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
    weight = (scale_a * 3.906250e-03f);
    weight = (weight * (scale_y * 3.906250e-03f));
    for (int32_t diagonal = 0; diagonal < 6; ++diagonal) {
      AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
      tl::ascend::rd_load(raw[0], P[(((diagonal * n) + core) * n)], raw[0], ((((n + blocks) - core) - 1) / blocks), n, ((blocks - 1) * n));
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
      AscendC::Cast(part[0], raw[0], AscendC::RoundMode::CAST_RINT, 2048);
      AscendC::PipeBarrier<PIPE_V>();
      if (diagonal == 4) {
        AscendC::Cast(roundtrip[0], part[0], AscendC::RoundMode::CAST_RINT, 2048);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(raw[0], raw[0], roundtrip[0], 2048);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(remainder[0], raw[0], AscendC::RoundMode::CAST_RINT, 2048);
        AscendC::PipeBarrier<PIPE_V>();
      }
      {
      AscendC::Muls(part[0], part[0], (weight * -1.000000e+00f), 2048);
      }
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Add(total[0], high[0], part[0], 2048);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Sub(bv[0], total[0], high[0], 2048);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Sub(av[0], total[0], bv[0], 2048);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Sub(error[0], high[0], av[0], 2048);
      AscendC::Sub(alt[0], part[0], bv[0], 2048);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Add(error[0], error[0], alt[0], 2048);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Add(low[0], low[0], error[0], 2048);
      tl::ascend::pv_copy(high[0], total[0], high[0], 2048);
      AscendC::PipeBarrier<PIPE_V>();
      if (diagonal == 4) {
        {
        AscendC::Muls(part[0], remainder[0], (weight * -1.000000e+00f), 2048);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(total[0], high[0], part[0], 2048);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(bv[0], total[0], high[0], 2048);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(av[0], total[0], bv[0], 2048);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(error[0], high[0], av[0], 2048);
        AscendC::Sub(alt[0], part[0], bv[0], 2048);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(error[0], error[0], alt[0], 2048);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(low[0], low[0], error[0], 2048);
        tl::ascend::pv_copy(high[0], total[0], high[0], 2048);
        AscendC::PipeBarrier<PIPE_V>();
      }
      weight = (weight * 3.906250e-03f);
    }
    AscendC::Add(high[0], high[0], low[0], 2048);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    tl::ascend::pv_store_rows(R[(core * n)], high[0], R[(core * n)], ((((n + blocks) - core) - 1) / blocks), n, 0, ((blocks - 1) * n));
    {
    AscendC::Muls(part[0], high[0], 4.096000e+03f, 2048);
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(h_r[0], part[0], AscendC::RoundMode::CAST_NONE, 2048);
    tl::ascend::pv_fill(cache[0], cache[0], cache[0], 0.000000e+00f, 2048);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
    tl::ascend::pv_load_rows(cache[0], Y[(n * ((job * n) + core))], cache[0], ((((n + blocks) - core) - 1) / blocks), n, ((blocks - 1) * n), 0);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
    {
    AscendC::Muls(cache[0], cache[0], (1.000000e+00f / scale_y), 2048);
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(h_y[0], cache[0], AscendC::RoundMode::CAST_NONE, 2048);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    tl::ascend::rd_store_half(YH[(core * n)], h_y[0], YH[(core * n)], ((((n + blocks) - core) - 1) / blocks), n, ((blocks - 1) * n));
    tl::ascend::rd_store_half(RH[(core * n)], h_r[0], RH[(core * n)], ((((n + blocks) - core) - 1) / blocks), n, ((blocks - 1) * n));
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
  }
  pipe.Destroy();
}

}
namespace tl::ascend {
__aicore__ inline void FusedInteger( GM_ADDR A_handle,  GM_ADDR B_handle,  GM_ADDR C_handle, int32_t blocks,  GM_ADDR shape_hint, int64_t n, int64_t planes, uint64_t fftsAddr) {
  AscendC::TPipe pipe;

  AscendC::GlobalTensor<int8_t> A;
  A.SetGlobalBuffer((__gm__ int8_t*)A_handle);
  AscendC::GlobalTensor<int8_t> B;
  B.SetGlobalBuffer((__gm__ int8_t*)B_handle);
  AscendC::GlobalTensor<int> C;
  C.SetGlobalBuffer((__gm__ int*)C_handle);
  AscendC::GlobalTensor<uint8_t> ShapeHint;
  ShapeHint.SetGlobalBuffer((__gm__ uint8_t*)shape_hint);

  AscendC::TBuf<AscendC::TPosition::A2> ascend_l0a;
  pipe.InitBuffer(ascend_l0a, 65536);
  AscendC::TBuf<AscendC::TPosition::B2> ascend_l0b;
  pipe.InitBuffer(ascend_l0b, 65536);
  AscendC::TBuf<AscendC::TPosition::A1> ascend_l1; pipe.InitBuffer(ascend_l1, 524032);
  AscendC::TBuf<AscendC::TPosition::CO1> ascend_l0c; pipe.InitBuffer(ascend_l0c, 131072);
  AscendC::TBuf<AscendC::TPosition::VECCALC> ascend_ub; pipe.InitBuffer(ascend_ub, 196352);
  auto bid = AscendC::GetBlockIdx();
  if ASCEND_IS_AIV {
    }
  int diagonals = 0;
  auto a = ascend_l1.GetWithOffset<int8_t>(40960, 0);
  auto b = ascend_l1.GetWithOffset<int8_t>(163840, 40960);
  int first = 0;
  int last = 0;
  auto c = ascend_l0c.GetWithOffset<int>(4096, 0);
  int clear = 0;
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIC {
    for (int32_t iteration = 0; iteration < ((((((n + 31) / 32) * ((n + 127) / 128)) + blocks) - 1) / blocks); ++iteration) {
      if ((bid + (iteration * blocks)) < (((n + 31) / 32) * ((n + 127) / 128))) {
        for (int32_t i = 0; i < planes; ++i) {
          for (int32_t kk = 0; kk < (n / 128); ++kk) {
            tl::ascend::copy_gm_to_l1<int8_t, 32, 128>(a[((i * 8192) + (kk * 4096))], A[((kk * 128) + (((((bid + (iteration * blocks)) / ((n + 127) / 128)) * 32) + (i * n)) * n))], n, ((1 <= (((n * (planes - i)) / 32) - ((bid + (iteration * blocks)) / ((n + 127) / 128)))) ? 32 : ((0 < ((n * (planes - i)) - (((bid + (iteration * blocks)) / ((n + 127) / 128)) * 32))) ? ((n * (planes - i)) - (((bid + (iteration * blocks)) / ((n + 127) / 128)) * 32)) : 0)), ((1 <= ((n / 128) - kk)) ? 128 : ((0 < (n - (kk * 128))) ? (n - (kk * 128)) : 0)));
            tl::ascend::copy_gm_to_l1<int8_t, 128, 128>(b[((i * 32768) + (kk * 16384))], B[(((((bid + (iteration * blocks)) % ((n + 127) / 128)) * 128) + (((kk * planes) * n) * 128)) + (i * n))], (planes * n), ((1 <= ((n / 128) - kk)) ? 128 : ((0 < (n - (kk * 128))) ? (n - (kk * 128)) : 0)), ((1 <= (((n * (planes - i)) / 128) - ((bid + (iteration * blocks)) % ((n + 127) / 128)))) ? 128 : ((0 < ((n * (planes - i)) - (((bid + (iteration * blocks)) % ((n + 127) / 128)) * 128))) ? ((n * (planes - i)) - (((bid + (iteration * blocks)) % ((n + 127) / 128)) * 128)) : 0)));
          }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        diagonals = ((planes * 2) - 1);
        if (6 < diagonals) {
          diagonals = 6;
        }
        for (int32_t diagonal = 0; diagonal < diagonals; ++diagonal) {
          first = 0;
          last = planes;
          if (planes <= diagonal) {
            first = ((diagonal + 1) - planes);
          }
          if ((diagonal + 1) < planes) {
            last = (diagonal + 1);
          }
          for (int32_t i_1 = first; i_1 < last; ++i_1) {
            for (int32_t kk_1 = 0; kk_1 < (n / 128); ++kk_1) {
              clear = 0;
              if (i_1 == first) {
                if (kk_1 == 0) {
                  clear = 1;
                }
              }
              tl::ascend::gemm_v0<int8_t, int, 32, 128, 128, false, false>(a[((i_1 * 8192) + (kk_1 * 4096))], b[(((diagonal * 32768) + (kk_1 * 16384)) - (i_1 * 32768))], c[0], ascend_l0a, ascend_l0b, (clear != 0));
              AscendC::PipeBarrier<PIPE_ALL>();
            }
          }
          tl::ascend::copy_l0c_to_gm<int, int, layout::RowMajor, 32, 128, 0>(C[((((bid + (iteration * blocks)) % ((n + 127) / 128)) * 128) + (((((bid + (iteration * blocks)) / ((n + 127) / 128)) * 32) + (diagonal * n)) * n))], c[0], n, ((1 <= (((n * (((planes * 2) - diagonal) - 1)) / 32) - ((bid + (iteration * blocks)) / ((n + 127) / 128)))) ? 32 : ((0 < ((n * (((planes * 2) - diagonal) - 1)) - (((bid + (iteration * blocks)) / ((n + 127) / 128)) * 32))) ? ((n * (((planes * 2) - diagonal) - 1)) - (((bid + (iteration * blocks)) / ((n + 127) / 128)) * 32)) : 0)), ((1 <= ((n / 128) - ((bid + (iteration * blocks)) % ((n + 127) / 128)))) ? 128 : ((0 < (n - (((bid + (iteration * blocks)) % ((n + 127) / 128)) * 128))) ? (n - (((bid + (iteration * blocks)) % ((n + 127) / 128)) * 128)) : 0)));
          AscendC::PipeBarrier<PIPE_ALL>();
        }
      }
    }
  }
  pipe.Destroy();
}

}
namespace tl::ascend {
__aicore__ inline void FusedHalf( GM_ADDR A_handle,  GM_ADDR B_handle,  GM_ADDR C_handle, int32_t blocks,  GM_ADDR shape_hint, int64_t n, int64_t planes, uint64_t fftsAddr) {
  AscendC::TPipe pipe;

  AscendC::GlobalTensor<half> A;
  A.SetGlobalBuffer((__gm__ half*)A_handle);
  AscendC::GlobalTensor<half> B;
  B.SetGlobalBuffer((__gm__ half*)B_handle);
  AscendC::GlobalTensor<float> C;
  C.SetGlobalBuffer((__gm__ float*)C_handle);
  AscendC::GlobalTensor<uint8_t> ShapeHint;
  ShapeHint.SetGlobalBuffer((__gm__ uint8_t*)shape_hint);

  AscendC::TBuf<AscendC::TPosition::A2> ascend_l0a;
  pipe.InitBuffer(ascend_l0a, 65536);
  AscendC::TBuf<AscendC::TPosition::B2> ascend_l0b;
  pipe.InitBuffer(ascend_l0b, 65536);
  AscendC::TBuf<AscendC::TPosition::A1> ascend_l1; pipe.InitBuffer(ascend_l1, 524032);
  AscendC::TBuf<AscendC::TPosition::CO1> ascend_l0c; pipe.InitBuffer(ascend_l0c, 131072);
  AscendC::TBuf<AscendC::TPosition::VECCALC> ascend_ub; pipe.InitBuffer(ascend_ub, 196352);
  auto bid = AscendC::GetBlockIdx();
  if ASCEND_IS_AIV {
    }
  auto c = ascend_l0c.GetWithOffset<float>(4096, 0);
  auto a = ascend_l1.GetWithOffset<half>(4096, 0);
  auto b = ascend_l1.GetWithOffset<half>(16384, 8192);
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIC {
    for (int32_t iteration = 0; iteration < (((((((planes * n) + 31) / 32) * (((planes * n) + 127) / 128)) + blocks) - 1) / blocks); ++iteration) {
      if ((bid + (iteration * blocks)) < ((((planes * n) + 31) / 32) * (((planes * n) + 127) / 128))) {
        for (int32_t kk = 0; kk < (n / 128); ++kk) {
          tl::ascend::copy_gm_to_l1<half, 32, 128>(a[0], A[((kk * 128) + ((((bid + (iteration * blocks)) / (((planes * n) + 127) / 128)) * n) * 32))], n, ((1 <= (((planes * n) / 32) - ((bid + (iteration * blocks)) / (((planes * n) + 127) / 128)))) ? 32 : ((0 < ((planes * n) - (((bid + (iteration * blocks)) / (((planes * n) + 127) / 128)) * 32))) ? ((planes * n) - (((bid + (iteration * blocks)) / (((planes * n) + 127) / 128)) * 32)) : 0)), ((1 <= ((n / 128) - kk)) ? 128 : ((0 < (n - (kk * 128))) ? (n - (kk * 128)) : 0)));
          tl::ascend::copy_gm_to_l1<half, 128, 128>(b[0], B[((((bid + (iteration * blocks)) % (((planes * n) + 127) / 128)) * 128) + (((kk * planes) * n) * 128))], (planes * n), ((1 <= ((n / 128) - kk)) ? 128 : ((0 < (n - (kk * 128))) ? (n - (kk * 128)) : 0)), ((1 <= (((planes * n) / 128) - ((bid + (iteration * blocks)) % (((planes * n) + 127) / 128)))) ? 128 : ((0 < ((planes * n) - (((bid + (iteration * blocks)) % (((planes * n) + 127) / 128)) * 128))) ? ((planes * n) - (((bid + (iteration * blocks)) % (((planes * n) + 127) / 128)) * 128)) : 0)));
          AscendC::PipeBarrier<PIPE_ALL>();
          tl::ascend::gemm_v0<half, float, 32, 128, 128, false, false>(a[0], b[0], c[0], ascend_l0a, ascend_l0b, (kk == 0));
          AscendC::PipeBarrier<PIPE_ALL>();
        }
        tl::ascend::copy_l0c_to_gm<float, float, layout::RowMajor, 32, 128, 0>(C[((((bid + (iteration * blocks)) % (((planes * n) + 127) / 128)) * 128) + (((((bid + (iteration * blocks)) / (((planes * n) + 127) / 128)) * planes) * n) * 32))], c[0], (planes * n), ((1 <= (((planes * n) / 32) - ((bid + (iteration * blocks)) / (((planes * n) + 127) / 128)))) ? 32 : ((0 < ((planes * n) - (((bid + (iteration * blocks)) / (((planes * n) + 127) / 128)) * 32))) ? ((planes * n) - (((bid + (iteration * blocks)) / (((planes * n) + 127) / 128)) * 32)) : 0)), ((1 <= (((planes * n) / 128) - ((bid + (iteration * blocks)) % (((planes * n) + 127) / 128)))) ? 128 : ((0 < ((planes * n) - (((bid + (iteration * blocks)) % (((planes * n) + 127) / 128)) * 128))) ? ((planes * n) - (((bid + (iteration * blocks)) % (((planes * n) + 127) / 128)) * 128)) : 0)));
        AscendC::PipeBarrier<PIPE_ALL>();
      }
    }
  }
  pipe.Destroy();
}

}
namespace tl::ascend {
__aicore__ inline void FusedNorms( GM_ADDR R_handle,  GM_ADDR P_handle, int32_t blocks, int32_t n, uint64_t fftsAddr) {
  AscendC::TPipe pipe;

  AscendC::GlobalTensor<float> R;
  R.SetGlobalBuffer((__gm__ float*)R_handle);
  AscendC::GlobalTensor<float> P;
  P.SetGlobalBuffer((__gm__ float*)P_handle);

  AscendC::TBuf<AscendC::TPosition::VECCALC> ascend_ub; pipe.InitBuffer(ascend_ub, 9216);
  auto core = AscendC::GetBlockIdx();
  if ASCEND_IS_AIV {
    }
  auto values = ascend_ub.GetWithOffset<float>(2048, 0);
  auto sums = ascend_ub.GetWithOffset<float>(256, 8192);
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIV {
    tl::ascend::pv_load_rows(values[0], R[(core * n)], values[0], ((((n + blocks) - core) - 1) / blocks), n, ((blocks - 1) * n), 0);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
    tl::ascend::pv_mul(values[0], values[0], values[0], (((((n + blocks) - core) - 1) / blocks) * n));
    tl::ascend::pv_fill(sums[0], sums[0], sums[0], 0.000000e+00f, n);
    AscendC::PipeBarrier<PIPE_V>();
    for (int32_t r = 0; r < ((((n + blocks) - core) - 1) / blocks); ++r) {
      tl::ascend::pv_add(sums[0], sums[0], values[(r * n)], n);
      AscendC::PipeBarrier<PIPE_V>();
    }
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    tl::ascend::pv_store_rows(P[(core * n)], sums[0], P[(core * n)], 1, n, 0, 0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
  }
  pipe.Destroy();
}

}
namespace tl::ascend {
__aicore__ inline void FusedCertify( GM_ADDR Y_handle,  GM_ADDR C_handle,  GM_ADDR P_handle,  GM_ADDR QN_handle,  GM_ADDR S_handle, int32_t blocks, int32_t n, int32_t job, float scale_a, float scale_y, float af, float yf, uint64_t fftsAddr) {
  AscendC::TPipe pipe;

  AscendC::GlobalTensor<float> Y;
  Y.SetGlobalBuffer((__gm__ float*)Y_handle);
  AscendC::GlobalTensor<float> C;
  C.SetGlobalBuffer((__gm__ float*)C_handle);
  AscendC::GlobalTensor<float> P;
  P.SetGlobalBuffer((__gm__ float*)P_handle);
  AscendC::GlobalTensor<float> QN;
  QN.SetGlobalBuffer((__gm__ float*)QN_handle);
  AscendC::GlobalTensor<float> S;
  S.SetGlobalBuffer((__gm__ float*)S_handle);

  AscendC::TBuf<AscendC::TPosition::VECCALC> ascend_ub; pipe.InitBuffer(ascend_ub, 101632);
  auto core = AscendC::GetBlockIdx();
  if ASCEND_IS_AIV {
    }
  auto partials = ascend_ub.GetWithOffset<float>(16384, 0);
  auto qn = ascend_ub.GetWithOffset<float>(1024, 65536);
  auto yy = ascend_ub.GetWithOffset<float>(2048, 69632);
  auto cc = ascend_ub.GetWithOffset<float>(2048, 77824);
  auto columns = ascend_ub.GetWithOffset<float>(256, 86016);
  auto red = ascend_ub.GetWithOffset<float>(32, 87040);
  auto scratch = ascend_ub.GetWithOffset<float>(256, 87168);
  auto colnorm = ascend_ub.GetWithOffset<float>(256, 88192);
  float factor = 0.000000e+00f;
  auto sums = ascend_ub.GetWithOffset<float>(2048, 89216);
  float rf = 0.000000e+00f;
  float qa2 = 0.000000e+00f;
  float qy2 = 0.000000e+00f;
  auto stat = ascend_ub.GetWithOffset<float>(32, 97408);
  float da = 0.000000e+00f;
  float dy = 0.000000e+00f;
  float rr = 0.000000e+00f;
  float tail = 0.000000e+00f;
  float eta_a = 0.000000e+00f;
  float eta_b = 0.000000e+00f;
  float unit = 0.000000e+00f;
  int valid = 0;
  float corrected_total = 0.000000e+00f;
  float worst = 0.000000e+00f;
  auto square = ascend_ub.GetWithOffset<float>(256, 97536);
  auto bound = ascend_ub.GetWithOffset<float>(256, 98560);
  auto budget = ascend_ub.GetWithOffset<float>(256, 99584);
  auto temp = ascend_ub.GetWithOffset<float>(256, 100608);
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIV {
    tl::ascend::pv_load_rows(partials[0], P[0], partials[0], 1, (blocks * n), 0, 0);
    tl::ascend::pv_load_rows(qn[0], QN[0], qn[0], 1, (blocks * 16), 0, 0);
    tl::ascend::pv_load_rows(yy[0], Y[(n * ((job * n) + core))], yy[0], ((((n + blocks) - core) - 1) / blocks), n, ((blocks - 1) * n), 0);
    tl::ascend::pv_load_rows(cc[0], C[(core * n)], cc[0], ((((n + blocks) - core) - 1) / blocks), n, ((blocks - 1) * n), 0);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
    tl::ascend::pv_fill(columns[0], columns[0], columns[0], 0.000000e+00f, n);
    AscendC::PipeBarrier<PIPE_V>();
    for (int32_t c = 0; c < blocks; ++c) {
      tl::ascend::pv_add(columns[0], columns[0], partials[(c * n)], n);
      AscendC::PipeBarrier<PIPE_V>();
    }
    tl::ascend::pv_reduce_sum(red[0], columns[0], scratch[0], n);
    tl::ascend::pv_sqrt(colnorm[0], columns[0], colnorm[0], n);
    AscendC::PipeBarrier<PIPE_V>();
    tl::ascend::pv_sqrt(red[0], red[0], red[0], 1);
    factor = (scale_y * 1.562500e-02f);
    factor = (factor * 1.562500e-02f);
    tl::ascend::pv_scale(cc[0], cc[0], cc[0], factor, (((((n + blocks) - core) - 1) / blocks) * n));
    AscendC::PipeBarrier<PIPE_V>();
    tl::ascend::pv_add(sums[0], yy[0], cc[0], (((((n + blocks) - core) - 1) / blocks) * n));
    AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
    rf = (red.GetValue(0) * 1.001000e+00f);
    qa2 = 0.000000e+00f;
    qy2 = 0.000000e+00f;
    for (int32_t c_1 = 0; c_1 < blocks; ++c_1) {
      qa2 = (qa2 + qn.GetValue((((int64_t)c_1) * (int64_t)16)));
      qy2 = (qy2 + qn.GetValue(((((int64_t)c_1) * (int64_t)16) + (int64_t)8)));
    }
    stat.SetValue(8, qa2);
    stat.SetValue(9, qy2);
    stat.SetValue(10, ((float)n));
    AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
    tl::ascend::pv_sqrt(stat[8], stat[8], stat[8], 3);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
    da = ((stat.GetValue(8) * scale_a) * 1.001000e+00f);
    dy = ((stat.GetValue(9) * scale_y) * 1.001000e+00f);
    da = (da * 3.906250e-03f);
    dy = (dy * 3.906250e-03f);
    da = (da * 3.906250e-03f);
    dy = (dy * 3.906250e-03f);
    da = (da * 3.906250e-03f);
    dy = (dy * 3.906250e-03f);
    da = (da * 3.906250e-03f);
    dy = (dy * 3.906250e-03f);
    da = (da * 3.906250e-03f);
    dy = (dy * 3.906250e-03f);
    da = (da * 3.906250e-03f);
    dy = (dy * 3.906250e-03f);
    rr = ((((af + da) + ((scale_a * ((float)n)) * 3.921569e-03f)) * ((yf + dy) + ((scale_y * ((float)n)) * 3.921569e-03f))) * 1.024000e+03f);
    rr = (rr * 3.906250e-03f);
    rr = (rr * 3.906250e-03f);
    rr = (rr * 3.906250e-03f);
    rr = (rr * 3.906250e-03f);
    rr = (rr * 3.906250e-03f);
    rr = (rr * 3.906250e-03f);
    tail = (scale_a * 1.525879e-05f);
    tail = (tail * (scale_y * 1.525879e-05f));
    tail = (tail * (((float)n) * ((float)n)));
    tail = (tail * 1.638400e+04f);
    tail = (tail * 3.906250e-03f);
    tail = (tail * 3.906250e-03f);
    tail = (tail * 3.906250e-03f);
    tail = (tail * 3.906250e-03f);
    tail = (tail * 3.020000e+00f);
    da = ((((((((da * yf) + (dy * af)) + (da * dy)) + rr) + ((((2.000000e+00f * rf) * 3.906250e-03f) * 3.906250e-03f) * 3.906250e-03f)) + (((float)n) * 1.000000e-33f)) + tail) * 1.250000e+00f);
    eta_a = (stat.GetValue(10) * scale_y);
    eta_b = (stat.GetValue(10) * 1.562500e-02f);
    eta_b = (eta_b * 1.562500e-02f);
    unit = 1.000000e+00f;
    eta_a = (eta_a * 3.906250e-03f);
    eta_b = (eta_b * 3.906250e-03f);
    unit = (unit * 3.906250e-03f);
    eta_a = (eta_a * 3.906250e-03f);
    eta_b = (eta_b * 3.906250e-03f);
    unit = (unit * 3.906250e-03f);
    eta_a = (eta_a * 3.906250e-03f);
    eta_b = (eta_b * 3.906250e-03f);
    unit = (unit * 3.906250e-03f);
    valid = 1;
    if (((rf < 0.000000e+00f) || (da < 0.000000e+00f)) || (5.000000e-01f <= (rf + da))) {
      valid = 0;
    }
    float denominator = max(5.000000e-01f, ((1.000000e+00f - rf) - da));
    corrected_total = 0.000000e+00f;
    worst = -1.000000e+30f;
    for (int32_t row = 0; row < ((((n + blocks) - core) - 1) / blocks); ++row) {
      tl::ascend::pv_mul(square[0], yy[(row * n)], yy[(row * n)], n);
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::pv_reduce_sum(red[0], square[0], scratch[0], n);
      tl::ascend::pv_mul(square[0], cc[(row * n)], cc[(row * n)], n);
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::pv_reduce_sum(red[8], square[0], scratch[0], n);
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::pv_sqrt(red[0], red[0], red[0], 16);
      AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
      AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
      float yn = (red.GetValue(0) * 1.001000e+00f);
      float cn = (red.GetValue(8) * 1.001000e+00f);
      float dcrow = (((((2.100000e-03f * yn) + eta_a) * rf) + ((stat.GetValue(10) * eta_b) * yn)) + ((stat.GetValue(10) * eta_a) * eta_b));
      float ct = (((cn + dcrow) + (yn * da)) / denominator);
      float scale = (((2.100000e-03f * yn) + ct) + eta_a);
      float constant = ((((yn + ct) * da) + (eta_b * yn)) + (eta_a * eta_b));
      tl::ascend::pv_scale(bound[0], colnorm[0], bound[0], scale, n);
      AscendC::PipeBarrier<PIPE_V>();
      {
      AscendC::Adds(bound[0], bound[0], constant, 256);
      }
      tl::ascend::pv_abs(budget[0], yy[(row * n)], budget[0], n);
      tl::ascend::pv_abs(temp[0], cc[(row * n)], temp[0], n);
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::pv_add(budget[0], budget[0], temp[0], n);
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::pv_scale(budget[0], budget[0], budget[0], unit, n);
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::pv_add(bound[0], bound[0], budget[0], n);
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::pv_scale(bound[0], bound[0], bound[0], 1.250000e+00f, n);
      tl::ascend::pv_abs(budget[0], sums[(row * n)], budget[0], n);
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::pv_scale(budget[0], budget[0], budget[0], 8.000000e-05f, n);
      AscendC::PipeBarrier<PIPE_V>();
      {
      AscendC::Adds(budget[0], budget[0], 8.000000e-05f, 256);
      }
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Sub(bound[0], bound[0], budget[0], 256);
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::pv_reduce_max(red[0], bound[0], scratch[0], n);
      tl::ascend::pv_mul(square[0], sums[(row * n)], sums[(row * n)], n);
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::pv_reduce_sum(red[8], square[0], scratch[0], n);
      AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
      AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
      if (0.000000e+00f < red.GetValue(0)) {
        valid = 0;
      }
      worst = max(worst, red.GetValue(0));
      corrected_total = (corrected_total + red.GetValue(8));
    }
    stat.SetValue(0, ((float)valid));
    stat.SetValue(1, corrected_total);
    stat.SetValue(2, worst);
    stat.SetValue(3, da);
    stat.SetValue(4, rf);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
    tl::ascend::pv_store_rows(S[(core * 8)], stat[0], S[(core * 8)], 1, 8, 0, 0);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    tl::ascend::pv_store_rows(Y[(n * ((job * n) + core))], sums[0], Y[(n * ((job * n) + core))], ((((n + blocks) - core) - 1) / blocks), n, 0, ((blocks - 1) * n));
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
  }
  pipe.Destroy();
}

}
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



#include "pinv_dsl_block.h"
__aicore__ inline void PvStageWindow( GM_ADDR A_handle,  GM_ADDR X_handle,  GM_ADDR W_handle,  GM_ADDR S_handle, int32_t blocks, int32_t n, int32_t batch, int32_t stride, int32_t window, float atol, float rtol, int32_t fallback_enabled, uint64_t fftsAddr) {

  AscendC::GlobalTensor<float> A;
  A.SetGlobalBuffer((__gm__ float*)A_handle);
  AscendC::GlobalTensor<float> X;
  X.SetGlobalBuffer((__gm__ float*)X_handle);
  AscendC::GlobalTensor<float> W;
  W.SetGlobalBuffer((__gm__ float*)W_handle);
  AscendC::GlobalTensor<float> S;
  S.SetGlobalBuffer((__gm__ float*)S_handle);

  auto bid = AscendC::GetBlockIdx();
  if ASCEND_IS_AIV {
    }
  int blocked = 0;
  int running = 0;
  float inverse = 0.000000e+00f;
  float peak_cached = 0.000000e+00f;
  int valid = 0;
  int refine = 0;
  float sy = 0.000000e+00f;
  float corrected = 0.000000e+00f;
  auto vid = AscendC::GetSubBlockIdx();
  blocked = 0;
  if (((((((((256 <= n) && (n <= 256)) && ((n % 128) == 0)) && (batch <= 16)) && (((blocks * 2) % batch) == 0)) && (((batch * n) / 64) <= (blocks * 2))) && ((batch * n) <= 2561)) && (((((batch * n) * n) * 11) <= ((((((((((batch * 8256) + (blocks * 32)) + ((n * n) * 27)) + (((n * n) / 2) * 2)) + ((blocks * n) * 2)) + (((n * n) * 5) / 2)) + 127) / 128) * 128) + (((((batch * n) * stride) + 63) / 64) * 128))) || (0 <= ((batch * n) * n)))) && (15 <= ((n - 1) / max((((blocks * 2) / batch) - 1), 1)))) {
    blocked = 1;
  }
  if (blocked != 0) {
    tl::ascend::bf_block(A[0], X[0], W[0], blocks, n, batch);
  }
  if ASCEND_IS_AIC {
    tl::ascend::pv_sync_init(W[0], W[0], W[0]);
    if (blocked != 0) {
      tl::ascend::pv_cross_wait2(W[0], W[0], W[0], 8);
      tl::ascend::br_gemm(W[0], W[0], W[0], (((batch * n) * n) * 2), (((batch * n) * n) * 8), (((batch * n) * n) * 4), blocks, batch, n, n, n);
      AscendC::PipeBarrier<PIPE_ALL>();
      tl::ascend::pc_done(W[0], W[0], W[0]);
      tl::ascend::pv_cross_wait2(W[0], W[0], W[0], 8);
      tl::ascend::br_gemm(W[0], W[0], W[0], (((batch * n) * n) * 9), (((batch * n) * n) * 9), (((batch * n) * n) * 11), blocks, batch, n, n, n);
      AscendC::PipeBarrier<PIPE_ALL>();
      tl::ascend::pc_done(W[0], W[0], W[0]);
      tl::ascend::pv_cross_wait2(W[0], W[0], W[0], 8);
      tl::ascend::br_gemm(W[0], W[0], W[0], (((batch * n) * n) * 8), (((batch * n) * n) * 9), (((batch * n) * n) * 10), blocks, batch, n, n, n);
      AscendC::PipeBarrier<PIPE_ALL>();
      tl::ascend::pc_done(W[0], W[0], W[0]);
    }
    running = 1;
    while (1) {
      if (!((running != 0))) { break; }
      tl::ascend::pv_cross_wait2(W[0], W[0], W[0], 8);
      tl::ascend::pc_invalidate(S[(batch * 512)], S[(batch * 512)], S[(batch * 512)]);
      int32_t next_job = ((int32_t)S.GetValue((((int64_t)batch) * (int64_t)512)));
      if (next_job < 0) {
        running = 0;
      } else {
        tl::ascend::fr_integer(W[0], W[0], W[0], ((batch * 8192) + (((((batch * n) * stride) + 63) / 64) * 128)), 0, (((n * n) * 5) / 4), (((n * n) * 5) / 2), blocks, n);
        AscendC::PipeBarrier<PIPE_ALL>();
        tl::ascend::pc_done(W[0], W[0], W[0]);
        tl::ascend::pv_cross_wait2(W[0], W[0], W[0], 8);
        tl::ascend::fr_half(W[0], W[0], W[0], ((batch * 8192) + (((((batch * n) * stride) + 63) / 64) * 128)), (((n * n) * 26) + (((n * n) * 5) / 2)), ((((n * n) * 26) + ((n * n) / 2)) + (((n * n) * 5) / 2)), ((((n * n) * 26) + (((n * n) / 2) * 2)) + (((n * n) * 5) / 2)), blocks, n);
        AscendC::PipeBarrier<PIPE_ALL>();
      }
      tl::ascend::pc_done(W[0], W[0], W[0]);
    }
  }
  if ASCEND_IS_AIV {
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> ascend_ub; pipe.InitBuffer(ascend_ub, 16384);
    auto pool = ascend_ub.GetWithOffset<float>(4096, 0);
    tl::ascend::fr_bind(A[0], X[0], W[0]);
    if (blocked != 0) {
      tl::ascend::bf_pack(W[0], W[0], W[0], blocks, n, batch);
      tl::ascend::pv_sync_all(W[0], W[0], W[0]);
      tl::ascend::pv_cross_set2(W[0], W[0], W[0], 8);
      tl::ascend::pv_cross_wait2(W[0], W[0], W[0], 10);
      tl::ascend::bf_decode(W[0], W[0], W[0], blocks, n, batch);
      tl::ascend::pv_sync_all(W[0], W[0], W[0]);
      tl::ascend::pv_cross_set2(W[0], W[0], W[0], 8);
      tl::ascend::pv_cross_wait2(W[0], W[0], W[0], 10);
      tl::ascend::bf_accumulate(W[0], W[0], W[0], blocks, n, batch);
      tl::ascend::pv_sync_all(W[0], W[0], W[0]);
      tl::ascend::pv_cross_set2(W[0], W[0], W[0], 8);
      tl::ascend::pv_cross_wait2(W[0], W[0], W[0], 10);
      tl::ascend::bf_finish(W[0], W[0], W[0], blocks, n, batch, stride);
    } else {
      if (((((((2 < ((blocks * 2) / batch)) && ((n % 16) == 0)) && (0 < window)) && ((window % 16) == 0)) && (((((((((n + 63) / 64) * 320) + (((n + 7) / 8) * 136)) + (max(((((n - 1) / max((((blocks * 2) / batch) - 2), 1)) / 8) + 1), 2) * 64)) + ((((n - 1) / max((((blocks * 2) / batch) - 2), 1)) / 8) * 8)) + max((max(((((n - 1) / max((((blocks * 2) / batch) - 2), 1)) + 1) * ((n + 7) / 8)), 64) * 8), ((((((n - 1) / max((((blocks * 2) / batch) - 2), 1)) / 8) * 1024) + (((n - 1) / max((((blocks * 2) / batch) - 2), 1)) * 16)) + 1296))) + ((((n - 1) / max((((blocks * 2) / batch) - 2), 1)) + 1) * stride)) <= 48712)) && (((((((n + 63) / 64) * 192) + (((n + 7) / 8) * 80)) + (((n + 15) / 16) * 64)) + (n * window)) <= 44416)) && (window < ((((n - 1) / max((((blocks * 2) / batch) - 1), 1)) * 4) + 4))) {
        tl::ascend::fr_prepared_raw(W[0], W[0], W[0], blocks, n, batch, stride, window, atol, rtol, (((((((batch * 8192) + (((((batch * n) * stride) + 63) / 64) * 128)) + (blocks * 32)) + ((n * n) * 27)) + (((n * n) / 2) * 2)) + ((blocks * n) * 2)) + (((n * n) * 5) / 2)));
      } else {
        tl::ascend::fr_raw(W[0], W[0], W[0], blocks, n, batch, stride, window, atol, rtol, (((((((batch * 8192) + (((((batch * n) * stride) + 63) / 64) * 128)) + (blocks * 32)) + ((n * n) * 27)) + (((n * n) / 2) * 2)) + ((blocks * n) * 2)) + (((n * n) * 5) / 2)));
      }
    }
    for (int32_t job = 0; job < batch; ++job) {
      tl::ascend::pv_load_rows(pool[0], W[(((job * 8192) + (((batch * n) * stride) * 2)) + 2048)], pool[0], 1, (((blocks * 2) / batch) * 8), 0, 0);
      tl::ascend::pv_load_rows(pool[384], W[((job * 8192) + (((batch * n) * stride) * 2))], pool[384], 1, 64, 0, 0);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
      inverse = 0.000000e+00f;
      peak_cached = 0.000000e+00f;
      valid = 1;
      for (int32_t c = 0; c < ((blocks * 2) / batch); ++c) {
        if (pool.GetValue((((int64_t)c) * (int64_t)8)) < 5.000000e-01f) {
          valid = 0;
        }
        inverse = (inverse + pool.GetValue(((((int64_t)c) * (int64_t)8) + (int64_t)1)));
        peak_cached = max(peak_cached, pool.GetValue(((((int64_t)c) * (int64_t)8) + (int64_t)2)));
      }
      float power = pool.GetValue(416);
      float total = pool.GetValue(417);
      pool.SetValue(3648, total);
      pool.SetValue(3649, inverse);
      AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
      AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
      tl::ascend::pv_sqrt(pool[3648], pool[3648], pool[3648], 2);
      AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
      AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
      float af = ((pool.GetValue(3648) * power) * 1.001000e+00f);
      float yf = ((pool.GetValue(3649) / power) * 1.001000e+00f);
      float cutoff = max((pool.GetValue(3648) * rtol), (atol / power));
      if (((inverse <= 0.000000e+00f) || (8.100000e+01f <= ((inverse * cutoff) * cutoff))) || (1.000000e+16f <= (inverse * total))) {
        valid = 0;
      }
      refine = 0;
      if ((valid != 0) && (1.000000e+15f <= (inverse * total))) {
        refine = 1;
      }
      sy = 1.000000e+00f;
      float sa = (power * 8.000000e+00f);
      if (refine != 0) {
        float peak = peak_cached;
        if ((peak <= 0.000000e+00f) || (1.000000e+20f <= peak)) {
          refine = 0;
          valid = 0;
        }
        if (refine != 0) {
          while (1) {
            if (!((sy < (peak * 4.000000e+00f)))) { break; }
            sy = (sy * 2.000000e+00f);
          }
          while (1) {
            if (!(((0.000000e+00f < peak) && ((peak * 4.000000e+00f) <= (sy * 5.000000e-01f))))) { break; }
            sy = (sy * 5.000000e-01f);
          }
          tl::ascend::fr_quant(W[0], W[0], W[0], ((batch * 8192) + (((((batch * n) * stride) + 63) / 64) * 128)), 0, (((n * n) * 5) / 4), (((((n * n) * 27) + (((n * n) / 2) * 2)) + ((blocks * n) * 2)) + (((n * n) * 5) / 2)), (blocks * 2), n, job, sa, sy);
        }
      }
      if (bid == 0) {
        pool.SetValue(3584, ((float)refine));
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
        tl::ascend::pv_store_rows(S[(job * 512)], pool[3584], S[(job * 512)], 1, 8, 0, 0);
      }
      if (refine != 0) {
        if (bid == 0) {
          pool.SetValue(3600, ((float)job));
          AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
          AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
          tl::ascend::pv_store_rows(S[(batch * 512)], pool[3600], S[(batch * 512)], 1, 16, 0, 0);
        }
        tl::ascend::pv_sync_all(W[0], W[0], W[0]);
        tl::ascend::pv_cross_set2(W[0], W[0], W[0], 8);
        tl::ascend::pv_cross_wait2(W[0], W[0], W[0], 10);
      }
      if (refine != 0) {
        tl::ascend::fr_decode(W[0], W[0], W[0], ((batch * 8192) + (((((batch * n) * stride) + 63) / 64) * 128)), (((n * n) * 5) / 2), (((n * n) * 25) + (((n * n) * 5) / 2)), (((n * n) * 26) + (((n * n) * 5) / 2)), ((((n * n) * 26) + ((n * n) / 2)) + (((n * n) * 5) / 2)), (blocks * 2), n, job, sa, sy);
        tl::ascend::fr_norms(W[0], W[0], W[0], ((batch * 8192) + (((((batch * n) * stride) + 63) / 64) * 128)), (((n * n) * 25) + (((n * n) * 5) / 2)), ((((n * n) * 27) + (((n * n) / 2) * 2)) + (((n * n) * 5) / 2)), (blocks * 2), n);
      }
      if (refine != 0) {
        tl::ascend::pv_sync_all(W[0], W[0], W[0]);
        tl::ascend::pv_cross_set2(W[0], W[0], W[0], 8);
        tl::ascend::pv_cross_wait2(W[0], W[0], W[0], 10);
      }
      if (refine != 0) {
        tl::ascend::fr_certify(W[0], W[0], W[0], ((batch * 8192) + (((((batch * n) * stride) + 63) / 64) * 128)), ((((n * n) * 26) + (((n * n) / 2) * 2)) + (((n * n) * 5) / 2)), ((((n * n) * 27) + (((n * n) / 2) * 2)) + (((n * n) * 5) / 2)), (((((n * n) * 27) + (((n * n) / 2) * 2)) + ((blocks * n) * 2)) + (((n * n) * 5) / 2)), ((job * 512) + 32), (blocks * 2), n, job, sa, sy, af, yf);
        tl::ascend::pv_sync_all(W[0], W[0], W[0]);
        tl::ascend::pv_load_rows(pool[0], S[((job * 512) + 32)], pool[0], 1, (blocks * 16), 0, 0);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
        corrected = 0.000000e+00f;
        for (int32_t c_1 = 0; c_1 < (blocks * 2); ++c_1) {
          if (pool.GetValue((((int64_t)c_1) * (int64_t)8)) < 5.000000e-01f) {
            valid = 0;
          }
          corrected = (corrected + pool.GetValue(((((int64_t)c_1) * (int64_t)8) + (int64_t)1)));
        }
        float absolute_cutoff = max((af * rtol), atol);
        if (8.100000e+01f <= ((corrected * absolute_cutoff) * absolute_cutoff)) {
          valid = 0;
        }
      }
      if ((valid == 0) && (fallback_enabled != 0)) {
        tl::ascend::pv_sync_all(W[0], W[0], W[0]);
        if (bid == job) {
          tl::ascend::fr_fallback(W[0], W[0], W[0], (((((((batch * 8224) + (((((batch * n) * stride) + 63) / 64) * 128)) + (blocks * 32)) + ((n * n) * 27)) + (((n * n) / 2) * 2)) + ((blocks * n) * 2)) + (((n * n) * 5) / 2)), blocks, job, n, batch, atol, rtol);
          tl::ascend::pc_invalidate(W[((((((((batch * 8224) + (((((batch * n) * stride) + 63) / 64) * 128)) + (job * 32)) + (blocks * 32)) + ((n * n) * 27)) + (((n * n) / 2) * 2)) + ((blocks * n) * 2)) + (((n * n) * 5) / 2))], W[((((((((batch * 8224) + (((((batch * n) * stride) + 63) / 64) * 128)) + (job * 32)) + (blocks * 32)) + ((n * n) * 27)) + (((n * n) / 2) * 2)) + ((blocks * n) * 2)) + (((n * n) * 5) / 2))], W[((((((((batch * 8224) + (((((batch * n) * stride) + 63) / 64) * 128)) + (job * 32)) + (blocks * 32)) + ((n * n) * 27)) + (((n * n) / 2) * 2)) + ((blocks * n) * 2)) + (((n * n) * 5) / 2))]);
        }
        tl::ascend::pv_sync_all(W[0], W[0], W[0]);
        tl::ascend::pv_load_rows(pool[3616], W[((((((((batch * 8224) + (((((batch * n) * stride) + 63) / 64) * 128)) + (job * 32)) + (blocks * 32)) + ((n * n) * 27)) + (((n * n) / 2) * 2)) + ((blocks * n) * 2)) + (((n * n) * 5) / 2))], pool[3616], 1, 8, 0, 0);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
        valid = ((int32_t)pool.GetValue(3616));
      }
      if (bid == 0) {
        pool.SetValue(3616, ((float)valid));
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
        tl::ascend::pv_store_rows(S[((job * 512) + 16)], pool[3616], S[((job * 512) + 16)], 1, 8, 0, 0);
      }
    }
    if (bid == 0) {
      pool.SetValue(3600, -1.000000e+00f);
      AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
      AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
      tl::ascend::pv_store_rows(S[(batch * 512)], pool[3600], S[(batch * 512)], 1, 16, 0, 0);
    }
    tl::ascend::pv_sync_all(W[0], W[0], W[0]);
    tl::ascend::pv_cross_set2(W[0], W[0], W[0], 8);
    tl::ascend::pv_cross_wait2(W[0], W[0], W[0], 10);
    pipe.Destroy();
  }
  AscendC::PipeBarrier<PIPE_ALL>();
}

