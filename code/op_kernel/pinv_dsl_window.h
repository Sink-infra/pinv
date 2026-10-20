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
 }else{
  LoadData2DParams p;p.startIndex=0;p.repeatTimes=cols/16;p.srcStride=K/16;p.sid=0;p.dstGap=0;p.ifTranspose=true;p.addrMode=0;
  for(unsigned i=0;i<rows/16;++i)LoadData(d[i*cols*16],s[i*256],p);
 }
}
template<class T,class U,unsigned M,unsigned N>
__aicore__ inline void mma(LocalTensor<T>a,LocalTensor<T>b,LocalTensor<U>c,bool init,unsigned k,unsigned char=0){
 MmadParams p;p.m=M;p.n=N;p.k=k;p.cmatrixInitVal=init;Mmad(c,a,b,p);
}
template<class T,class U,unsigned M,unsigned N,unsigned K,bool TA=false,bool TB=false>
__aicore__ inline void gemm_v0(LocalTensor<T> const&a,LocalTensor<T> const&b,LocalTensor<U> const&c,TBuf<TPosition::A2>&ab,TBuf<TPosition::B2>&bb,bool init){
 static_assert(K==128 && !TA && !TB);
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
    __aicore__ inline void rq_float(LocalTensor<float> d,LocalTensor<half> a,LocalTensor<float>,int n){Cast(d,a,RoundMode::CAST_NONE,n);}
    __aicore__ inline void rq_i8(LocalTensor<int8_t> d,LocalTensor<half> a,LocalTensor<int8_t>,int n){Cast(d,a,RoundMode::CAST_RINT,n);}
    __aicore__ inline void rq_store(GlobalTensor<int8_t> d,LocalTensor<int8_t> a,GlobalTensor<int8_t>,int rows,int cols,int gap){DataCopyPad(d,a,DataCopyExtParams{uint16_t(rows),uint32_t(cols),0,uint32_t(gap),0});}
    }

namespace tl::ascend {
    __aicore__ inline void rd_load(LocalTensor<int32_t> d,GlobalTensor<int32_t> a,LocalTensor<int32_t>,int rows,int cols,int gap){DataCopyPad(d,a,DataCopyExtParams{uint16_t(rows),uint32_t(cols*4),uint32_t(gap*4),0,0},DataCopyPadExtParams<int32_t>{false,0,0,0});}
    __aicore__ inline void rd_store_half(GlobalTensor<half> d,LocalTensor<half> a,GlobalTensor<half>,int rows,int cols,int gap){DataCopyPad(d,a,DataCopyExtParams{uint16_t(rows),uint32_t(cols*2),0,uint32_t(gap*2),0});}
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
      capacity = max((n * window), (((n - window) * (((n + 7) - window) / 8)) * 8));
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
    for (int32_t panel = 0; panel < ((n + 15) / 16); ++panel) {
      if ((0 < window) && ((panel * 16) == window)) {
        tl::ascend::pv_all_barrier(pool[0], pool[0], pool[0]);
        tl::ascend::pv_cross_set0(pool[0], pool[0], pool[0], 3);
        tl::ascend::pv_cross_wait(pool[0], pool[0], pool[0], 3);
        tl::ascend::pv_cross_set0(pool[0], pool[0], pool[0], 2);
        tl::ascend::pv_cross_wait(pool[0], pool[0], pool[0], 2);
        if (valid != 0) {
          for (int32_t row_1 = 0; row_1 < size; ++row_1) {
            int32_t original = ((int32_t)pool.GetValue((((int64_t)permutation) + ((int64_t)row_1))));
            tl::ascend::pv_load_rows(pool[(row_1 * pitch)], W[((((job * n) * stride) + (original * k)) + window)], pool[(row_1 * pitch)], 1, size, 0, 0);
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
        if ((0 < window) && (((panel * 16) + block) < window)) {
          columns = ((window - block) - (panel * 16));
        }
        next_pitch = (((columns + 7) / 8) * 8);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
        if ((window <= 0) || (((panel * 16) + block) != window)) {
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
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
    tl::ascend::pv_store_rows(W[((((job * 8192) + (core * 8)) + (((all_batch * n) * stride) * 2)) + 2048)], pool[0], W[((((job * 8192) + (core * 8)) + (((all_batch * n) * stride) * 2)) + 2048)], 1, 2, 0, 0);
    tl::ascend::pv_all_barrier(pool[0], pool[0], pool[0]);
  }
  pipe.Destroy();
}

}
#define pv_pipeline_producer(d,a,b,...) PvDslPipelineProducer((GM_ADDR)pvPipelineBuffers.inputTensor.GetPhyAddr(),(GM_ADDR)pvPipelineBuffers.outputTensor.GetPhyAddr(),(GM_ADDR)pvPipelineBuffers.workspaceTensor.GetPhyAddr(),__VA_ARGS__,fftsAddr)
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
  auto rounded = ascend_ub.GetWithOffset<float>(2048, 16384);
  auto planes = ascend_ub.GetWithOffset<half>(10240, 24576);
  auto integers = ascend_ub.GetWithOffset<int>(2048, 8192);
  auto temp = ascend_ub.GetWithOffset<float>(2048, 45056);
  auto carry = ascend_ub.GetWithOffset<float>(2048, 53248);
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
    for (int32_t limb = 0; limb < 5; ++limb) {
      AscendC::Cast(integers[0], values[0], AscendC::RoundMode::CAST_RINT, 2048);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Cast(rounded[0], integers[0], AscendC::RoundMode::CAST_RINT, 2048);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Sub(values[0], values[0], rounded[0], 2048);
      AscendC::PipeBarrier<PIPE_V>();
      {
      AscendC::Muls(values[0], values[0], 2.560000e+02f, 2048);
      }
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::rq_half(planes[(limb * 2048)], rounded[0], planes[(limb * 2048)], 2048);
      AscendC::PipeBarrier<PIPE_V>();
    }
    AscendC::Mul(temp[0], values[0], values[0], 2048);
    AscendC::PipeBarrier<PIPE_V>();
    tl::ascend::pv_reduce_sum(carry[0], temp[0], rounded[0], 2048);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    tl::ascend::pv_store_rows(QN[(core * 16)], carry[0], QN[(core * 16)], 1, 8, 0, 0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    for (int32_t rev = 0; rev < 4; ++rev) {
      tl::ascend::rq_float(rounded[0], planes[(8192 - (rev * 2048))], rounded[0], 2048);
      AscendC::PipeBarrier<PIPE_V>();
      {
      AscendC::Adds(carry[0], rounded[0], -1.270000e+02f, 2048);
      }
      AscendC::PipeBarrier<PIPE_V>();
      {
      AscendC::Maxs(carry[0], carry[0], 0.000000e+00f, 2048);
      }
      AscendC::PipeBarrier<PIPE_V>();
      {
      AscendC::Mins(carry[0], carry[0], 1.000000e+00f, 2048);
      }
      AscendC::PipeBarrier<PIPE_V>();
      {
      AscendC::Muls(temp[0], carry[0], 2.560000e+02f, 2048);
      }
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Sub(rounded[0], rounded[0], temp[0], 2048);
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::rq_half(planes[(8192 - (rev * 2048))], rounded[0], planes[(8192 - (rev * 2048))], 2048);
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::rq_float(rounded[0], planes[(6144 - (rev * 2048))], rounded[0], 2048);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Add(rounded[0], rounded[0], carry[0], 2048);
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::rq_half(planes[(6144 - (rev * 2048))], rounded[0], planes[(6144 - (rev * 2048))], 2048);
      AscendC::PipeBarrier<PIPE_V>();
    }
    for (int32_t limb_1 = 0; limb_1 < 5; ++limb_1) {
      tl::ascend::rq_i8(small[0], planes[(limb_1 * 2048)], small[0], 2048);
      AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
      tl::ascend::rq_store(QA[(n * ((limb_1 * n) + core))], small[0], QA[(n * ((limb_1 * n) + core))], ((((n + blocks) - core) - 1) / blocks), n, ((blocks - 1) * n));
      AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    }
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
    for (int32_t limb_2 = 0; limb_2 < 5; ++limb_2) {
      AscendC::Cast(integers[0], values[0], AscendC::RoundMode::CAST_RINT, 2048);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Cast(rounded[0], integers[0], AscendC::RoundMode::CAST_RINT, 2048);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Sub(values[0], values[0], rounded[0], 2048);
      AscendC::PipeBarrier<PIPE_V>();
      {
      AscendC::Muls(values[0], values[0], 2.560000e+02f, 2048);
      }
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::rq_half(planes[(limb_2 * 2048)], rounded[0], planes[(limb_2 * 2048)], 2048);
      AscendC::PipeBarrier<PIPE_V>();
    }
    AscendC::Mul(temp[0], values[0], values[0], 2048);
    AscendC::PipeBarrier<PIPE_V>();
    tl::ascend::pv_reduce_sum(carry[0], temp[0], rounded[0], 2048);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    tl::ascend::pv_store_rows(QN[((core * 16) + 8)], carry[0], QN[((core * 16) + 8)], 1, 8, 0, 0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    for (int32_t rev_1 = 0; rev_1 < 4; ++rev_1) {
      tl::ascend::rq_float(rounded[0], planes[(8192 - (rev_1 * 2048))], rounded[0], 2048);
      AscendC::PipeBarrier<PIPE_V>();
      {
      AscendC::Adds(carry[0], rounded[0], -1.270000e+02f, 2048);
      }
      AscendC::PipeBarrier<PIPE_V>();
      {
      AscendC::Maxs(carry[0], carry[0], 0.000000e+00f, 2048);
      }
      AscendC::PipeBarrier<PIPE_V>();
      {
      AscendC::Mins(carry[0], carry[0], 1.000000e+00f, 2048);
      }
      AscendC::PipeBarrier<PIPE_V>();
      {
      AscendC::Muls(temp[0], carry[0], 2.560000e+02f, 2048);
      }
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Sub(rounded[0], rounded[0], temp[0], 2048);
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::rq_half(planes[(8192 - (rev_1 * 2048))], rounded[0], planes[(8192 - (rev_1 * 2048))], 2048);
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::rq_float(rounded[0], planes[(6144 - (rev_1 * 2048))], rounded[0], 2048);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Add(rounded[0], rounded[0], carry[0], 2048);
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::rq_half(planes[(6144 - (rev_1 * 2048))], rounded[0], planes[(6144 - (rev_1 * 2048))], 2048);
      AscendC::PipeBarrier<PIPE_V>();
    }
    for (int32_t limb_3 = 0; limb_3 < 5; ++limb_3) {
      tl::ascend::rq_i8(small[0], planes[(limb_3 * 2048)], small[0], 2048);
      AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
      tl::ascend::rq_store(QY[(((core * n) * 5) + (limb_3 * n))], small[0], QY[(((core * n) * 5) + (limb_3 * n))], ((((n + blocks) - core) - 1) / blocks), n, (((blocks * 5) - 1) * n));
      AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    }
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

  AscendC::TBuf<AscendC::TPosition::VECCALC> ascend_ub; pipe.InitBuffer(ascend_ub, 90112);
  auto core = AscendC::GetBlockIdx();
  if ASCEND_IS_AIV {
    }
  auto high = ascend_ub.GetWithOffset<float>(2048, 0);
  auto low = ascend_ub.GetWithOffset<float>(2048, 8192);
  auto raw = ascend_ub.GetWithOffset<int>(2048, 16384);
  float row_weight = 0.000000e+00f;
  auto part = ascend_ub.GetWithOffset<float>(2048, 24576);
  float weight = 0.000000e+00f;
  auto total = ascend_ub.GetWithOffset<float>(2048, 32768);
  auto bv = ascend_ub.GetWithOffset<float>(2048, 40960);
  auto av = ascend_ub.GetWithOffset<float>(2048, 49152);
  auto error = ascend_ub.GetWithOffset<float>(2048, 57344);
  auto alt = ascend_ub.GetWithOffset<float>(2048, 65536);
  auto h_r = ascend_ub.GetWithOffset<half>(2048, 73728);
  auto cache = ascend_ub.GetWithOffset<float>(2048, 77824);
  auto h_y = ascend_ub.GetWithOffset<half>(2048, 86016);
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
    row_weight = (scale_a * 3.906250e-03f);
    row_weight = (row_weight * (scale_y * 3.906250e-03f));
    for (int32_t i = 0; i < 5; ++i) {
      weight = row_weight;
      for (int32_t j = 0; j < 5; ++j) {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
        tl::ascend::rd_load(raw[0], P[(n * ((((i * n) * 5) + (core * 5)) + j))], raw[0], ((((n + blocks) - core) - 1) / blocks), n, (((blocks * 5) - 1) * n));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::Cast(part[0], raw[0], AscendC::RoundMode::CAST_RINT, 2048);
        AscendC::PipeBarrier<PIPE_V>();
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
        weight = (weight * 3.906250e-03f);
      }
      row_weight = (row_weight * 3.906250e-03f);
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
  auto a = ascend_l1.GetWithOffset<int8_t>(40960, 0);
  auto b = ascend_l1.GetWithOffset<int8_t>(163840, 40960);
  auto c = ascend_l0c.GetWithOffset<int>(4096, 0);
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
        for (int32_t i_1 = 0; i_1 < planes; ++i_1) {
          for (int32_t j = 0; j < planes; ++j) {
            for (int32_t kk_1 = 0; kk_1 < (n / 128); ++kk_1) {
              tl::ascend::gemm_v0<int8_t, int, 32, 128, 128, false, false>(a[((i_1 * 8192) + (kk_1 * 4096))], b[((j * 32768) + (kk_1 * 16384))], c[0], ascend_l0a, ascend_l0b, (kk_1 == 0));
              AscendC::PipeBarrier<PIPE_ALL>();
            }
            tl::ascend::copy_l0c_to_gm<int, int, layout::RowMajor, 32, 128, 0>(C[(((((bid + (iteration * blocks)) % ((n + 127) / 128)) * 128) + ((((((bid + (iteration * blocks)) / ((n + 127) / 128)) * 32) + (i_1 * n)) * planes) * n)) + (j * n))], c[0], (planes * n), ((1 <= (((n * (planes - i_1)) / 32) - ((bid + (iteration * blocks)) / ((n + 127) / 128)))) ? 32 : ((0 < ((n * (planes - i_1)) - (((bid + (iteration * blocks)) / ((n + 127) / 128)) * 32))) ? ((n * (planes - i_1)) - (((bid + (iteration * blocks)) / ((n + 127) / 128)) * 32)) : 0)), ((1 <= (((n * (planes - j)) / 128) - ((bid + (iteration * blocks)) % ((n + 127) / 128)))) ? 128 : ((0 < ((n * (planes - j)) - (((bid + (iteration * blocks)) % ((n + 127) / 128)) * 128))) ? ((n * (planes - j)) - (((bid + (iteration * blocks)) % ((n + 127) / 128)) * 128)) : 0)));
            AscendC::PipeBarrier<PIPE_ALL>();
          }
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
    da = (((((((da * yf) + (dy * af)) + (da * dy)) + rr) + ((((2.000000e+00f * rf) * 3.906250e-03f) * 3.906250e-03f) * 3.906250e-03f)) + (((float)n) * 1.000000e-33f)) * 1.250000e+00f);
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
#define fr_quant(d,a,b,off,qa,qy,qn,bl,n,job,sa,sy) FusedQuant((GM_ADDR)fr.input.GetPhyAddr(),(GM_ADDR)fr.output.GetPhyAddr(),(GM_ADDR)(fr.work.GetPhyAddr()+off+qa),(GM_ADDR)(fr.work.GetPhyAddr()+off+qy),(GM_ADDR)(fr.work.GetPhyAddr()+off+qn),bl,n,job,sa,sy,fftsAddr),tl::ascend::FrRestorePipe(&pipe)
#define fr_integer(d,a,b,off,qa,qy,pr,bl,n) FusedInteger((GM_ADDR)(d.GetPhyAddr()+off+qa),(GM_ADDR)(d.GetPhyAddr()+off+qy),(GM_ADDR)(d.GetPhyAddr()+off+pr),bl,(GM_ADDR)d.GetPhyAddr(),n,5,fftsAddr)
#define fr_decode(d,a,b,off,pr,rr,yh,rh,bl,n,job,sa,sy) FusedDecode((GM_ADDR)(fr.work.GetPhyAddr()+off+pr),(GM_ADDR)fr.output.GetPhyAddr(),(GM_ADDR)(fr.work.GetPhyAddr()+off+rr),(GM_ADDR)(fr.work.GetPhyAddr()+off+yh),(GM_ADDR)(fr.work.GetPhyAddr()+off+rh),bl,n,job,sa,sy,fftsAddr),tl::ascend::FrRestorePipe(&pipe)
#define fr_half(d,a,b,off,yh,rh,co,bl,n) FusedHalf((GM_ADDR)(d.GetPhyAddr()+off+yh),(GM_ADDR)(d.GetPhyAddr()+off+rh),(GM_ADDR)(d.GetPhyAddr()+off+co),bl,(GM_ADDR)d.GetPhyAddr(),n,1,fftsAddr)
#define fr_combine(d,a,b,off,co,bl,n,job,sy) FusedCombine((GM_ADDR)fr.output.GetPhyAddr(),(GM_ADDR)(fr.work.GetPhyAddr()+off+co),bl,n,job,sy,fftsAddr),tl::ascend::FrRestorePipe(&pipe)
#define fr_norms(d,a,b,off,rr,pa,bl,n) FusedNorms((GM_ADDR)(d.GetPhyAddr()+off+rr),(GM_ADDR)(d.GetPhyAddr()+off+pa),bl,n,fftsAddr),tl::ascend::FrRestorePipe(&pipe)
#define fr_certify(d,a,b,off,co,pa,qn,st,bl,n,job,sa,sy,af,yf) FusedCertify((GM_ADDR)fr.output.GetPhyAddr(),(GM_ADDR)(fr.work.GetPhyAddr()+off+co),(GM_ADDR)(fr.work.GetPhyAddr()+off+pa),(GM_ADDR)(fr.work.GetPhyAddr()+off+qn),(GM_ADDR)(fr.status.GetPhyAddr()+st),bl,n,job,sa,sy,af,yf,fftsAddr),tl::ascend::FrRestorePipe(&pipe)
#define fr_fallback(d,a,b,so,bl,job,n,ba,at,rt) PvDslRectangular((GM_ADDR)fr.input.GetPhyAddr(),(GM_ADDR)fr.output.GetPhyAddr(),(GM_ADDR)fr.work.GetPhyAddr(),(GM_ADDR)(fr.work.GetPhyAddr()+so),bl,job,n,n,ba,at,rt,fftsAddr),tl::ascend::FrRestorePipe(&pipe)



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
  int running = 0;
  float inverse = 0.000000e+00f;
  int valid = 0;
  int refine = 0;
  float sy = 0.000000e+00f;
  float peak = 0.000000e+00f;
  float corrected = 0.000000e+00f;
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIC {
    tl::ascend::pv_sync_init(W[0], W[0], W[0]);
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
    tl::ascend::fr_raw(W[0], W[0], W[0], blocks, n, batch, stride, window, atol, rtol, (((((((batch * 8192) + (((((batch * n) * stride) + 63) / 64) * 128)) + (blocks * 32)) + ((n * n) * 27)) + (((n * n) / 2) * 2)) + ((blocks * n) * 2)) + (((n * n) * 5) / 2)));
    for (int32_t job = 0; job < batch; ++job) {
      tl::ascend::pv_load_rows(pool[0], W[(((job * 8192) + (((batch * n) * stride) * 2)) + 2048)], pool[0], 1, (((blocks * 2) / batch) * 8), 0, 0);
      tl::ascend::pv_load_rows(pool[384], W[((job * 8192) + (((batch * n) * stride) * 2))], pool[384], 1, 64, 0, 0);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
      inverse = 0.000000e+00f;
      valid = 1;
      for (int32_t c = 0; c < ((blocks * 2) / batch); ++c) {
        if (pool.GetValue((((int64_t)c) * (int64_t)8)) < 5.000000e-01f) {
          valid = 0;
        }
        inverse = (inverse + pool.GetValue(((((int64_t)c) * (int64_t)8) + (int64_t)1)));
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
      if ((valid != 0) && (3.600000e+09f <= (inverse * total))) {
        refine = 1;
      }
      sy = 1.000000e+00f;
      float sa = (power * 8.000000e+00f);
      if (refine != 0) {
        tl::ascend::pv_load_rows(pool[0], X[(n * ((job * n) + bid))], pool[0], (((((blocks * 2) + n) - bid) - 1) / (blocks * 2)), n, (((blocks * 2) - 1) * n), 0);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        tl::ascend::pv_abs(pool[0], pool[0], pool[0], ((((((blocks * 2) + n) - bid) - 1) / (blocks * 2)) * n));
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::pv_reduce_max(pool[2560], pool[0], pool[2048], ((((((blocks * 2) + n) - bid) - 1) / (blocks * 2)) * n));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        tl::ascend::pv_store_rows(S[(((job * 512) + (bid * 8)) + 32)], pool[2560], S[(((job * 512) + (bid * 8)) + 32)], 1, 8, 0, 0);
        tl::ascend::pv_sync_all(W[0], W[0], W[0]);
        tl::ascend::pv_load_rows(pool[2688], S[((job * 512) + 32)], pool[2688], 1, (blocks * 16), 0, 0);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
        peak = 0.000000e+00f;
        for (int32_t c_1 = 0; c_1 < (blocks * 2); ++c_1) {
          peak = max(peak, pool.GetValue(((((int64_t)c_1) * (int64_t)8) + (int64_t)2688)));
        }
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
        tl::ascend::pv_sync_all(W[0], W[0], W[0]);
      }
      if (refine != 0) {
        tl::ascend::fr_decode(W[0], W[0], W[0], ((batch * 8192) + (((((batch * n) * stride) + 63) / 64) * 128)), (((n * n) * 5) / 2), (((n * n) * 25) + (((n * n) * 5) / 2)), (((n * n) * 26) + (((n * n) * 5) / 2)), ((((n * n) * 26) + ((n * n) / 2)) + (((n * n) * 5) / 2)), (blocks * 2), n, job, sa, sy);
        tl::ascend::fr_norms(W[0], W[0], W[0], ((batch * 8192) + (((((batch * n) * stride) + 63) / 64) * 128)), (((n * n) * 25) + (((n * n) * 5) / 2)), ((((n * n) * 27) + (((n * n) / 2) * 2)) + (((n * n) * 5) / 2)), (blocks * 2), n);
      }
      if (refine != 0) {
        tl::ascend::pv_sync_all(W[0], W[0], W[0]);
        tl::ascend::pv_cross_set2(W[0], W[0], W[0], 8);
        tl::ascend::pv_cross_wait2(W[0], W[0], W[0], 10);
        tl::ascend::pv_sync_all(W[0], W[0], W[0]);
      }
      if (refine != 0) {
        tl::ascend::fr_certify(W[0], W[0], W[0], ((batch * 8192) + (((((batch * n) * stride) + 63) / 64) * 128)), ((((n * n) * 26) + (((n * n) / 2) * 2)) + (((n * n) * 5) / 2)), ((((n * n) * 27) + (((n * n) / 2) * 2)) + (((n * n) * 5) / 2)), (((((n * n) * 27) + (((n * n) / 2) * 2)) + ((blocks * n) * 2)) + (((n * n) * 5) / 2)), ((job * 512) + 32), (blocks * 2), n, job, sa, sy, af, yf);
        tl::ascend::pv_sync_all(W[0], W[0], W[0]);
        tl::ascend::pv_load_rows(pool[0], S[((job * 512) + 32)], pool[0], 1, (blocks * 16), 0, 0);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
        corrected = 0.000000e+00f;
        for (int32_t c_2 = 0; c_2 < (blocks * 2); ++c_2) {
          if (pool.GetValue((((int64_t)c_2) * (int64_t)8)) < 5.000000e-01f) {
            valid = 0;
          }
          corrected = (corrected + pool.GetValue(((((int64_t)c_2) * (int64_t)8) + (int64_t)1)));
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
      tl::ascend::pv_sync_all(W[0], W[0], W[0]);
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
    tl::ascend::pv_sync_all(W[0], W[0], W[0]);
    pipe.Destroy();
  }
  AscendC::PipeBarrier<PIPE_ALL>();
}

