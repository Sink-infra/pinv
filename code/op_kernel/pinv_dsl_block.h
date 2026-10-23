#pragma once
// BEGIN DSL block strategy
#pragma once
namespace tl::ascend {
__aicore__ inline void br_sub(AscendC::LocalTensor<float>d,AscendC::LocalTensor<float>a,AscendC::LocalTensor<float>b,int n){AscendC::Sub(d,a,b,n);}
__aicore__ inline void bp_store_half(AscendC::GlobalTensor<half>d,AscendC::LocalTensor<half>a,AscendC::GlobalTensor<half>,int n){AscendC::DataCopy(d,a,n);}
}


namespace tl::ascend {
__aicore__ inline void br_load_a(LocalTensor<float>d,GlobalTensor<float>s,LocalTensor<float>,int cols,int stride){Nd2NzParams p;p.ndNum=1;p.nValue=64;p.dValue=cols;p.srcDValue=stride;p.srcNdMatrixStride=0;p.dstNzC0Stride=64;p.dstNzNStride=1;p.dstNzMatrixStride=0;DataCopy(d,s,p);}
}

using uint=unsigned int;using uchar=unsigned char;using ushort=unsigned short;
namespace tl::ascend {
__aicore__ inline void BlockQuick( GM_ADDR A_handle,  GM_ADDR Y_handle,  GM_ADDR W_handle,  GM_ADDR S_handle, int32_t blocks, int32_t job, int32_t m, int32_t n, int32_t batch, float atol, float rtol, uint64_t fftsAddr) {
  AscendC::TPipe pipe;

  AscendC::GlobalTensor<float> A;
  A.SetGlobalBuffer((__gm__ float*)A_handle);
  AscendC::GlobalTensor<float> Y;
  Y.SetGlobalBuffer((__gm__ float*)Y_handle);
  AscendC::GlobalTensor<float> W;
  W.SetGlobalBuffer((__gm__ float*)W_handle);
  AscendC::GlobalTensor<float> S;
  S.SetGlobalBuffer((__gm__ float*)S_handle);

  AscendC::TBuf<AscendC::TPosition::VECCALC> ascend_ub; pipe.InitBuffer(ascend_ub, 77344);
  auto cid = AscendC::GetBlockIdx();
  if ASCEND_IS_AIV {
    }
  auto accepted = ascend_ub.GetWithOffset<int>(8, 0);
  int count = 0;
  int dma_pad = 0;
  float condition = 0.000000e+00f;
  int valid = 0;
  auto pool = ascend_ub.GetWithOffset<float>(19136, 32);
  float power = 0.000000e+00f;
  uint32_t power_word = (uint32_t)0;
  auto indices = ascend_ub.GetWithOffset<uint32_t>(64, 76576);
  int previous = 0;
  float recovered = 0.000000e+00f;
  auto order = ascend_ub.GetWithOffset<uint32_t>(128, 76832);
  int word = 0;
  float metric = 0.000000e+00f;
  float best = 0.000000e+00f;
  int pivot = 0;
  int consumed = 0;
  int base = 0;
  int active = 0;
  float best_1 = 0.000000e+00f;
  int pivot_1 = 0;
  int consumed_1 = 0;
  int base_1 = 0;
  int active_1 = 0;
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIV {
    accepted.SetValue(0, 0);
    count = 4608;
    dma_pad = 64;
    condition = 1.000000e+30f;
    valid = 1;
    tl::ascend::pv_fill(pool[0], pool[0], pool[0], 0.000000e+00f, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
    tl::ascend::pv_load_rows(pool[0], A[(job * 4096)], pool[0], 64, 64, 0, (72 - dma_pad), (bool)0);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
    tl::ascend::pv_abs(pool[8704], pool[0], pool[8704], count);
    AscendC::PipeBarrier<PIPE_V>();
    tl::ascend::pv_reduce_max(pool[19008], pool[8704], pool[18496], count);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
    float magnitude = pool.GetValue(19008);
    if ((magnitude < 1.000000e-30f) || (1.701412e+38f <= magnitude)) {
      valid = 0;
    }
    power = 1.000000e+00f;
    if (valid != 0) {
      power_word = (((*(uint32_t *)(&(magnitude))) + (uint32_t)8388607) & (uint32_t)2139095040);
      power = (*(float *)(&(power_word)));
      tl::ascend::pv_scale(pool[0], pool[0], pool[0], (1.000000e+00f / power), count);
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::pv_mul(pool[8704], pool[0], pool[0], count);
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::pv_reduce_sum(pool[19008], pool[8704], pool[18496], count);
      AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
      AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
      float total = pool.GetValue(19008);
      for (int32_t i = 0; i < 64; ++i) {
        indices.SetValue(i, ((uint32_t)(i * 288)));
      }
      tl::ascend::pv_fill(pool[19040], pool[19040], pool[19040], 0.000000e+00f, 64);
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::pv_mask_begin(pool[0], pool[0], pool[0], 64);
      previous = 19105;
      recovered = 0.000000e+00f;
      AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
      AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
      for (int32_t pivot_tile = 0; pivot_tile < 64; ++pivot_tile) {
        tl::ascend::pv_gather(pool[19040], pool[0], indices[0], (pivot_tile * 4), 64);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::pv_abs(pool[18432], pool[19040], pool[18432], 64);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::pv_mask_reduce(pool[19008], pool[18432], pool[19008]);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
        tl::ascend::pv_reduce_scalars(pool[0], pool[0], pool[0], best, pivot);
        if (best <= 0.000000e+00f) {
          valid = 0;
          break;
        }
        tl::ascend::pv_mask_remove(pool[0], pool[0], pool[0], pivot);
        order.SetValue(pivot, ((uint32_t)(pivot_tile * 4)));
        order.SetValue((pivot_tile + 64), ((uint32_t)(pivot * 288)));
        consumed = 0;
        base = 0;
        float divisor = pool.GetValue((((int64_t)pivot) + (int64_t)19040));
        pool.SetValue(((((int64_t)pivot) * (int64_t)72) + ((int64_t)pivot_tile)), (divisor + 1.000000e+00f));
        pool.SetValue(((((int64_t)previous) + ((int64_t)pivot_tile)) - (int64_t)1), recovered);
        recovered = (-1.000000e+00f / divisor);
        previous = (pivot * 72);
        pool.SetValue((((int64_t)pivot) + (int64_t)19040), 0.000000e+00f);
        AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
        active = 64;
        if (consumed != 0) {
          active = (active - 8);
        }
        tl::ascend::pv_scale(pool[((pivot * 72) + base)], pool[((pivot * 72) + base)], pool[((pivot * 72) + base)], (-1.000000e+00f / divisor), active);
        tl::ascend::pv_brcb_rows(pool[17408], pool[19040], pool[17408], 64, 8);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::pv_mad_strided(pool[base], pool[((pivot * 72) + base)], pool[17408], 64, 64, 1, 1, 0, 9, 0, 1);
        AscendC::PipeBarrier<PIPE_V>();
      }
      if (valid != 0) {
        AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
        pool.SetValue((((int64_t)previous) + (int64_t)63), recovered);
        AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
        tl::ascend::pv_fill(pool[8704], pool[8704], pool[8704], 0.000000e+00f, count);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::pv_square_padded_rows(pool[8704], pool[0], pool[8704], 64, 64, 72);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::pv_reduce_sum(pool[19008], pool[8704], pool[18496], count);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
        float inverse_total = pool.GetValue(19008);
        condition = (total * inverse_total);
        pool.SetValue(19008, total);
        AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
        tl::ascend::pv_sqrt(pool[19008], pool[19008], pool[19008], 1);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
        if (inverse_total <= 0.000000e+00f) {
          valid = 0;
        } else {
          tl::ascend::pv_scale(pool[0], pool[0], pool[0], (-1.000000e+00f / power), count);
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
          AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
          tl::ascend::bl_index_brcb(pool[17408], order[64], pool[17408], 64);
          AscendC::PipeBarrier<PIPE_V>();
          tl::ascend::bl_index_matrix(pool[12800], pool[17408], order[0], 64, 64);
          AscendC::PipeBarrier<PIPE_V>();
          tl::ascend::bl_gather(pool[8704], pool[0], pool[12800], 4096);
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
          AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
          tl::ascend::pv_store_rows(Y[(job * 4096)], pool[8704], Y[(job * 4096)], 64, 64, (64 - dma_pad), 0);
          AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
          AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
          AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
          AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        }
      }
    }
    accepted.SetValue(0, valid);
    accepted.SetValue(1, (*(int32_t *)(&(condition))));
    S.SetValue((((int64_t)job) * (int64_t)32), ((float)accepted.GetValue(0)));
    word = accepted.GetValue(1);
    metric = (*(float *)(&(word)));
    S.SetValue(((((int64_t)job) * (int64_t)32) + (int64_t)1), metric);
  }
  pipe.Destroy();
}

__aicore__ inline void BlockGemm( GM_ADDR A_handle,  GM_ADDR P_handle,  GM_ADDR C_handle, int32_t blocks,  GM_ADDR shape_hint, int64_t inner, int64_t cols, int64_t batch, int64_t rows, uint64_t fftsAddr) {
  AscendC::TPipe pipe;

  AscendC::GlobalTensor<float> A;
  A.SetGlobalBuffer((__gm__ float*)A_handle);
  AscendC::GlobalTensor<float> P;
  P.SetGlobalBuffer((__gm__ float*)P_handle);
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
  int previous_job = 0;
  int previous_column = 0;
  auto b = ascend_l1.GetWithOffset<float>(32768, 0);
  auto a = ascend_l1.GetWithOffset<float>(16384, 131072);
  auto c = ascend_l0c.GetWithOffset<float>(8192, 0);
  int clear = 0;
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIC {
    previous_job = -1;
    previous_column = -1;
    for (int32_t iteration = 0; iteration < (((((batch * (rows / 64)) * (cols / 128)) + blocks) - 1) / blocks); ++iteration) {
      if ((bid + (iteration * blocks)) < ((batch * (rows / 64)) * (cols / 128))) {
        if ((((bid + (iteration * blocks)) / ((rows / 64) * (cols / 128))) != previous_job) || (((bid + (iteration * blocks)) % (cols / 128)) != previous_column)) {
          for (int32_t kk = 0; kk < (inner / 16); ++kk) {
            tl::ascend::copy_gm_to_l1<float, 16, 128>(b[(kk * 2048)], P[((((bid + (iteration * blocks)) % (cols / 128)) * 128) + (((kk * 16) + (((bid + (iteration * blocks)) / ((rows / 64) * (cols / 128))) * inner)) * cols))], cols, ((1 <= (((inner * (batch - ((bid + (iteration * blocks)) / ((rows / 64) * (cols / 128))))) / 16) - kk)) ? 16 : ((0 < ((inner * (batch - ((bid + (iteration * blocks)) / ((rows / 64) * (cols / 128))))) - (kk * 16))) ? ((inner * (batch - ((bid + (iteration * blocks)) / ((rows / 64) * (cols / 128))))) - (kk * 16)) : 0)), ((1 <= ((cols / 128) - ((bid + (iteration * blocks)) % (cols / 128)))) ? 128 : ((0 < (cols - (((bid + (iteration * blocks)) % (cols / 128)) * 128))) ? (cols - (((bid + (iteration * blocks)) % (cols / 128)) * 128)) : 0)));
          }
          previous_job = ((bid + (iteration * blocks)) / ((rows / 64) * (cols / 128)));
          previous_column = ((bid + (iteration * blocks)) % (cols / 128));
        }
        tl::ascend::br_load_a(a[0], A[((((((bid + (iteration * blocks)) / (cols / 128)) % (rows / 64)) * 64) + (((bid + (iteration * blocks)) / ((rows / 64) * (cols / 128))) * rows)) * inner)], a[0], inner, inner);
        AscendC::PipeBarrier<PIPE_ALL>();
        for (int32_t kk_1 = 0; kk_1 < (inner / 16); ++kk_1) {
          clear = 0;
          if (kk_1 == 0) {
            clear = 1;
          }
          tl::ascend::gemm_v0<float, float, 64, 128, 16, false, false>(a[(kk_1 * 1024)], b[(kk_1 * 2048)], c[0], ascend_l0a, ascend_l0b, (clear != 0));
          AscendC::PipeBarrier<PIPE_ALL>();
        }
        tl::ascend::copy_l0c_to_gm<float, float, layout::RowMajor, 64, 128, 0>(C[((((bid + (iteration * blocks)) % (cols / 128)) * 128) + ((((((bid + (iteration * blocks)) / (cols / 128)) % (rows / 64)) * 64) + (((bid + (iteration * blocks)) / ((rows / 64) * (cols / 128))) * rows)) * cols))], c[0], cols, ((1 <= (((rows * (batch - ((bid + (iteration * blocks)) / ((rows / 64) * (cols / 128))))) / 64) - (((bid + (iteration * blocks)) / (cols / 128)) % (rows / 64)))) ? 64 : ((0 < ((rows * (batch - ((bid + (iteration * blocks)) / ((rows / 64) * (cols / 128))))) - ((((bid + (iteration * blocks)) / (cols / 128)) % (rows / 64)) * 64))) ? ((rows * (batch - ((bid + (iteration * blocks)) / ((rows / 64) * (cols / 128))))) - ((((bid + (iteration * blocks)) / (cols / 128)) % (rows / 64)) * 64)) : 0)), ((1 <= ((cols / 128) - ((bid + (iteration * blocks)) % (cols / 128)))) ? 128 : ((0 < (cols - (((bid + (iteration * blocks)) % (cols / 128)) * 128))) ? (cols - (((bid + (iteration * blocks)) % (cols / 128)) * 128)) : 0)));
        AscendC::PipeBarrier<PIPE_ALL>();
      }
    }
  }
  pipe.Destroy();
}

}

#define br_quick(d,a,b,ca,ci,so,bl,job,ba) BlockQuick((GM_ADDR)(d.GetPhyAddr()+ca),(GM_ADDR)(d.GetPhyAddr()+ci),(GM_ADDR)d.GetPhyAddr(),(GM_ADDR)(d.GetPhyAddr()+so),bl,job,64,64,ba,0.0f,0.0f,fftsAddr),tl::ascend::FrRestorePipe(&pipe)
#define br_gemm(d,a,b,oa,ob,oc,bl,ba,m,k,n) BlockGemm((GM_ADDR)(d.GetPhyAddr()+oa),(GM_ADDR)(d.GetPhyAddr()+ob),(GM_ADDR)(d.GetPhyAddr()+oc),bl,(GM_ADDR)d.GetPhyAddr(),k,n,ba,m,fftsAddr)
namespace tl::ascend {
__aicore__ inline void BlockRaw( GM_ADDR A_handle,  GM_ADDR Y_handle,  GM_ADDR W_handle,  GM_ADDR S_handle, int32_t blocks, int32_t n, int32_t batch, uint64_t fftsAddr) {

  AscendC::GlobalTensor<float> A;
  A.SetGlobalBuffer((__gm__ float*)A_handle);
  AscendC::GlobalTensor<float> Y;
  Y.SetGlobalBuffer((__gm__ float*)Y_handle);
  AscendC::GlobalTensor<float> W;
  W.SetGlobalBuffer((__gm__ float*)W_handle);
  AscendC::GlobalTensor<float> S;
  S.SetGlobalBuffer((__gm__ float*)S_handle);

  auto bid = AscendC::GetBlockIdx();
  if ASCEND_IS_AIV {
    }
  int row_tile = 0;
  float magnitude = 0.000000e+00f;
  int begin = 0;
  int count = 0;
  float magnitude_1 = 0.000000e+00f;
  float power = 0.000000e+00f;
  int admitted = 0;
  float cost = 0.000000e+00f;
  int chosen = 0;
  float best = 0.000000e+00f;
  int original = 0;
  auto vid = AscendC::GetSubBlockIdx();
  row_tile = 8;
  if ((blocks * 64) <= (batch * n)) {
    row_tile = 32;
  }
  if ASCEND_IS_AIC {
    tl::ascend::pv_sync_init(W[0], W[0], W[0]);
    for (int32_t stage = 0; stage < (n / 64); ++stage) {
      tl::ascend::pv_cross_wait2(W[0], W[0], W[0], 8);
      tl::ascend::br_gemm(W[0], W[0], W[0], (((batch * n) * 128) + (((batch * n) * n) * 4)), (((batch * 4096) + ((batch * n) * 128)) + (((batch * n) * n) * 4)), (((batch * 4096) + ((batch * n) * 256)) + (((batch * n) * n) * 4)), blocks, batch, 64, 64, n);
      AscendC::PipeBarrier<PIPE_ALL>();
      tl::ascend::pc_done(W[0], W[0], W[0]);
      tl::ascend::pv_cross_wait2(W[0], W[0], W[0], 8);
      tl::ascend::br_gemm(W[0], W[0], W[0], (((batch * 4096) + ((batch * n) * 384)) + (((batch * n) * n) * 4)), (((batch * 4096) + ((batch * n) * 256)) + (((batch * n) * n) * 4)), (((batch * 4096) + ((batch * n) * 448)) + (((batch * n) * n) * 4)), blocks, batch, n, 64, n);
      AscendC::PipeBarrier<PIPE_ALL>();
      tl::ascend::pc_done(W[0], W[0], W[0]);
    }
  }
  if ASCEND_IS_AIV {
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> ascend_ub; pipe.InitBuffer(ascend_ub, 98304);
    auto pool = ascend_ub.GetWithOffset<float>(24576, 0);
    tl::ascend::fr_bind(A[0], Y[0], W[0]);
    tl::ascend::pv_sync_init(W[0], W[0], W[0]);
    if (bid < (batch * ((blocks * 2) / batch))) {
      magnitude = 0.000000e+00f;
      for (int32_t chunk = 0; chunk < ((((((blocks * 2) / batch) * 4096) + (n * n)) - 1) / (((blocks * 2) / batch) * 4096)); ++chunk) {
        begin = (((bid % ((blocks * 2) / batch)) * 4096) + ((chunk * ((blocks * 2) / batch)) * 4096));
        if (begin < (n * n)) {
          count = ((n * n) - begin);
          if (4096 < count) {
            count = 4096;
          }
          tl::ascend::pv_load_rows(pool[0], A[((((bid / ((blocks * 2) / batch)) * n) * n) + begin)], pool[0], 1, count, 0, 0);
          AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
          tl::ascend::pv_abs(pool[0], pool[0], pool[0], count);
          AscendC::PipeBarrier<PIPE_V>();
          tl::ascend::pv_reduce_max(pool[16384], pool[0], pool[4096], count);
          AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
          AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
          magnitude = max(magnitude, pool.GetValue(16384));
        }
      }
      pool.SetValue(0, magnitude);
      AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
      AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
      tl::ascend::pv_store_rows(S[((batch * 32) + (bid * 8))], pool[0], S[((batch * 32) + (bid * 8))], 1, 8, 0, 0);
      tl::ascend::pv_all_barrier(W[0], W[0], W[0]);
    }
    tl::ascend::pv_sync_all(W[0], W[0], W[0]);
    if (bid < batch) {
      tl::ascend::pv_load_rows(pool[0], S[((batch * 32) + ((bid * ((blocks * 2) / batch)) * 8))], pool[0], 1, (((blocks * 2) / batch) * 8), 0, 0);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
      magnitude_1 = 0.000000e+00f;
      for (int32_t lane = 0; lane < ((blocks * 2) / batch); ++lane) {
        magnitude_1 = max(magnitude_1, pool.GetValue((((int64_t)lane) * (int64_t)8)));
      }
      power = 1.000000e+00f;
      if ((0.000000e+00f < magnitude_1) && (magnitude_1 < 1.000000e+30f)) {
        while (1) {
          if (!((power < magnitude_1))) { break; }
          power = (power * 2.000000e+00f);
        }
        while (1) {
          if (!((magnitude_1 <= (power * 5.000000e-01f)))) { break; }
          power = (power * 5.000000e-01f);
        }
      }
      pool.SetValue(0, power);
      AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
      AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
      tl::ascend::pv_store_rows(S[(bid * 32)], pool[0], S[(bid * 32)], 1, 8, 0, 0);
      tl::ascend::pv_all_barrier(W[0], W[0], W[0]);
    }
    tl::ascend::pv_sync_all(W[0], W[0], W[0]);
    tl::ascend::pv_load_rows(pool[20480], S[0], pool[20480], 1, (batch * 32), 0, 0);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
    for (int32_t iteration = 0; iteration < ((((blocks * 2) + ((batch * n) / row_tile)) - 1) / (blocks * 2)); ++iteration) {
      int32_t row = ((((iteration * blocks) * 2) + bid) * row_tile);
      if (row < (batch * n)) {
        float power_1 = pool.GetValue((((((int64_t)row) / ((int64_t)n)) * (int64_t)32) + (int64_t)20480));
        tl::ascend::pv_load_rows(pool[0], A[(row * n)], pool[0], row_tile, n, 0, 0);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        tl::ascend::pv_scale(pool[0], pool[0], pool[0], (1.000000e+00f / power_1), (row_tile * n));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        tl::ascend::pv_store_rows(W[(row * n)], pool[0], W[(row * n)], row_tile, n, 0, 0);
        tl::ascend::pv_all_barrier(W[0], W[0], W[0]);
      }
    }
    tl::ascend::pv_sync_all(W[0], W[0], W[0]);
    for (int32_t stage_1 = 0; stage_1 < (n / 64); ++stage_1) {
      if (bid < (batch * ((n / 64) - stage_1))) {
        tl::ascend::pv_load_rows(pool[0], W[(((((bid % ((n / 64) - stage_1)) + stage_1) * n) * 64) + (((bid / ((n / 64) - stage_1)) * n) * n))], pool[0], 64, 64, (n - 64), 0);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(0);
        tl::ascend::pv_store_rows(W[((bid * 4096) + (((batch * n) * n) * 4))], pool[0], W[((bid * 4096) + (((batch * n) * n) * 4))], 1, 4096, 0, 0);
        tl::ascend::pv_all_barrier(W[0], W[0], W[0]);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        tl::ascend::br_quick(W[0], W[0], W[0], (((batch * n) * n) * 4), (((batch * n) * 64) + (((batch * n) * n) * 4)), (((batch * 4096) + ((batch * n) * 449)) + (((batch * n) * n) * 6)), blocks, bid, (batch * ((n / 64) - stage_1)));
        admitted = ((int32_t)W.GetValue(((((((int64_t)batch) * (int64_t)4096) + ((((int64_t)batch) * ((int64_t)n)) * (int64_t)449)) + (((int64_t)bid) * (int64_t)32)) + (((((int64_t)batch) * ((int64_t)n)) * ((int64_t)n)) * (int64_t)6))));
        cost = 1.000000e+30f;
        if (admitted != 0) {
          cost = W.GetValue((((((((int64_t)batch) * (int64_t)4096) + ((((int64_t)batch) * ((int64_t)n)) * (int64_t)449)) + (((int64_t)bid) * (int64_t)32)) + (((((int64_t)batch) * ((int64_t)n)) * ((int64_t)n)) * (int64_t)6)) + (int64_t)1));
        }
        pool.SetValue(0, cost);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
        tl::ascend::pv_store_rows(W[((((batch * 4096) + ((batch * n) * 448)) + (bid * 8)) + (((batch * n) * n) * 6))], pool[0], W[((((batch * 4096) + ((batch * n) * 448)) + (bid * 8)) + (((batch * n) * n) * 6))], 1, 8, 0, 0);
        tl::ascend::pv_all_barrier(W[0], W[0], W[0]);
      }
      tl::ascend::pv_sync_all(W[0], W[0], W[0]);
      tl::ascend::pv_load_rows(pool[18432], W[(((batch * 4096) + ((batch * n) * 448)) + (((batch * n) * n) * 6))], pool[18432], 1, ((batch * ((n / 64) - stage_1)) * 8), 0, 0);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
      for (int32_t iteration_1 = 0; iteration_1 < ((((blocks * 2) + ((batch * n) / row_tile)) - 1) / (blocks * 2)); ++iteration_1) {
        int32_t row_1 = ((((iteration_1 * blocks) * 2) + bid) * row_tile);
        if (row_1 < (batch * n)) {
          chosen = 0;
          best = 1.000000e+30f;
          for (int32_t candidate = 0; candidate < ((n / 64) - stage_1); ++candidate) {
            float cost_1 = pool.GetValue((((((((int64_t)row_1) / ((int64_t)n)) * ((((int64_t)n) / (int64_t)64) - ((int64_t)stage_1))) * (int64_t)8) + (((int64_t)candidate) * (int64_t)8)) + (int64_t)18432));
            if ((0.000000e+00f <= cost_1) && (cost_1 < best)) {
              chosen = candidate;
              best = cost_1;
            }
          }
          int32_t position = ((chosen * 64) + (stage_1 * 64));
          if ((row_1 % n) == (stage_1 * 64)) {
            pool.SetValue(0, best);
            pool.SetValue(1, ((float)position));
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
            tl::ascend::pv_store_rows(S[((((stage_1 * batch) * 32) + ((row_1 / n) * 32)) + 512)], pool[0], S[((((stage_1 * batch) * 32) + ((row_1 / n) * 32)) + 512)], 1, 8, 0, 0);
            tl::ascend::pv_all_barrier(W[0], W[0], W[0]);
          }
          original = (row_1 % n);
          if (((stage_1 * 64) <= (row_1 % n)) && ((row_1 % n) < ((stage_1 * 64) + 64))) {
            original = ((position + (row_1 % n)) - (stage_1 * 64));
          } else {
            if ((position <= (row_1 % n)) && ((row_1 % n) < (position + 64))) {
              original = (((stage_1 * 64) + (row_1 % n)) - position);
            }
          }
          tl::ascend::pv_load_rows(pool[16384], W[(n * (((row_1 / n) * n) + original))], pool[16384], row_tile, 64, (n - 64), 0);
          AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
          if (((stage_1 * 64) <= (row_1 % n)) && ((row_1 % n) < ((stage_1 * 64) + 64))) {
            tl::ascend::pv_fill(pool[16384], pool[16384], pool[16384], 0.000000e+00f, (row_tile * 64));
          }
          AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
          AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
          tl::ascend::pv_store_rows(W[((((batch * 4096) + ((batch * n) * 384)) + (row_1 * 64)) + (((batch * n) * n) * 4))], pool[16384], W[((((batch * 4096) + ((batch * n) * 384)) + (row_1 * 64)) + (((batch * n) * n) * 4))], row_tile, 64, 0, 0);
          tl::ascend::pv_fill(pool[0], pool[0], pool[0], 0.000000e+00f, (row_tile * n));
          AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
          AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
          tl::ascend::pv_load_rows(pool[0], W[((n * (((row_1 / n) * n) + original)) + 64)], pool[0], row_tile, (n - 64), 64, 64);
          AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
          if (((stage_1 * 64) <= (row_1 % n)) && ((row_1 % n) < ((stage_1 * 64) + 64))) {
            for (int32_t lane_1 = 0; lane_1 < row_tile; ++lane_1) {
              pool.SetValue((((((((int64_t)n) * (((int64_t)lane_1) + (int64_t)1)) + (((int64_t)row_1) % ((int64_t)n))) + ((int64_t)lane_1)) - (((int64_t)stage_1) * (int64_t)64)) - (int64_t)64), 1.000000e+00f);
            }
          }
          AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
          AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
          tl::ascend::pv_store_rows(W[((((batch * n) * n) * 2) + (row_1 * n))], pool[0], W[((((batch * n) * n) * 2) + (row_1 * n))], row_tile, n, 0, 0);
          tl::ascend::pv_all_barrier(W[0], W[0], W[0]);
          if (((stage_1 * 64) <= (row_1 % n)) && ((row_1 % n) < ((stage_1 * 64) + 64))) {
            tl::ascend::pv_store_rows(W[(((((batch * 4096) + ((batch * n) * 128)) + (((row_1 / n) * n) * 64)) + (((batch * n) * n) * 4)) + (((row_1 % n) - (stage_1 * 64)) * n))], pool[0], W[(((((batch * 4096) + ((batch * n) * 128)) + (((row_1 / n) * n) * 64)) + (((batch * n) * n) * 4)) + (((row_1 % n) - (stage_1 * 64)) * n))], row_tile, n, 0, 0);
            tl::ascend::pv_all_barrier(W[0], W[0], W[0]);
            tl::ascend::pv_load_rows(pool[16384], W[((((((((row_1 / n) * ((n / 64) - stage_1)) * 4096) + (chosen * 4096)) + ((row_1 % n) * 64)) + ((batch * n) * 64)) + (((batch * n) * n) * 4)) - (stage_1 * 4096))], pool[16384], row_tile, 64, 0, 0);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(0);
            tl::ascend::pv_store_rows(W[((((((row_1 / n) * 4096) + ((batch * n) * 128)) + ((row_1 % n) * 64)) + (((batch * n) * n) * 4)) - (stage_1 * 4096))], pool[16384], W[((((((row_1 / n) * 4096) + ((batch * n) * 128)) + ((row_1 % n) * 64)) + (((batch * n) * n) * 4)) - (stage_1 * 4096))], row_tile, 64, 0, 0);
            tl::ascend::pv_all_barrier(W[0], W[0], W[0]);
          }
        }
      }
      tl::ascend::pv_sync_all(W[0], W[0], W[0]);
      tl::ascend::pv_cross_set2(W[0], W[0], W[0], 8);
      tl::ascend::pv_cross_wait2(W[0], W[0], W[0], 10);
      tl::ascend::pv_sync_all(W[0], W[0], W[0]);
      tl::ascend::pv_cross_set2(W[0], W[0], W[0], 8);
      tl::ascend::pv_cross_wait2(W[0], W[0], W[0], 10);
      tl::ascend::pv_sync_all(W[0], W[0], W[0]);
      for (int32_t iteration_2 = 0; iteration_2 < ((((blocks * 2) + ((batch * n) / row_tile)) - 1) / (blocks * 2)); ++iteration_2) {
        int32_t row_2 = ((((iteration_2 * blocks) * 2) + bid) * row_tile);
        if (row_2 < (batch * n)) {
          if (((stage_1 * 64) <= (row_2 % n)) && ((row_2 % n) < ((stage_1 * 64) + 64))) {
            tl::ascend::pv_load_rows(pool[0], W[(((((batch * 4096) + ((batch * n) * 256)) + (((row_2 / n) * n) * 64)) + (((batch * n) * n) * 4)) + (((row_2 % n) - (stage_1 * 64)) * n))], pool[0], row_tile, n, 0, 0);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
          } else {
            tl::ascend::pv_load_rows(pool[0], W[((((batch * n) * n) * 2) + (row_2 * n))], pool[0], row_tile, n, 0, 0);
            tl::ascend::pv_load_rows(pool[16384], W[((((batch * 4096) + ((batch * n) * 448)) + (((batch * n) * n) * 4)) + (row_2 * n))], pool[16384], row_tile, n, 0, 0);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
            tl::ascend::br_sub(pool[0], pool[0], pool[16384], (row_tile * n));
          }
          tl::ascend::pv_all_barrier(W[0], W[0], W[0]);
          tl::ascend::pv_store_rows(W[(row_2 * n)], pool[0], W[(row_2 * n)], row_tile, n, 0, 0);
          tl::ascend::pv_all_barrier(W[0], W[0], W[0]);
        }
      }
      tl::ascend::pv_sync_all(W[0], W[0], W[0]);
    }
    tl::ascend::pv_load_rows(pool[20480], S[0], pool[20480], 1, (batch * 32), 0, 0);
    tl::ascend::pv_load_rows(pool[22016], S[512], pool[22016], 1, 512, 0, 0);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
    for (int32_t iteration_3 = 0; iteration_3 < ((((blocks * 2) + ((batch * n) / row_tile)) - 1) / (blocks * 2)); ++iteration_3) {
      int32_t row_3 = ((((iteration_3 * blocks) * 2) + bid) * row_tile);
      if (row_3 < (batch * n)) {
        float power_2 = pool.GetValue((((((int64_t)row_3) / ((int64_t)n)) * (int64_t)32) + (int64_t)20480));
        for (int32_t col = 0; col < (n / 64); ++col) {
          pool.SetValue((col + 21200), ((float)col));
        }
        for (int32_t stage_2 = 0; stage_2 < (n / 64); ++stage_2) {
          int32_t chosen_1 = (((int32_t)pool.GetValue(((((((int64_t)stage_2) * ((int64_t)batch)) * (int64_t)32) + ((((int64_t)row_3) / ((int64_t)n)) * (int64_t)32)) + (int64_t)22017))) / 64);
          float old = pool.GetValue((stage_2 + 21200));
          pool.SetValue((stage_2 + 21200), pool.GetValue((((int64_t)chosen_1) + (int64_t)21200)));
          pool.SetValue((((int64_t)chosen_1) + (int64_t)21200), old);
        }
        for (int32_t col_1 = 0; col_1 < (n / 64); ++col_1) {
          int32_t original_1 = ((int32_t)pool.GetValue((col_1 + 21200)));
          tl::ascend::pv_load_rows(pool[0], W[((col_1 * 64) + (row_3 * n))], pool[0], row_tile, 64, (n - 64), 0);
          AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
          tl::ascend::pv_scale(pool[0], pool[0], pool[0], (1.000000e+00f / power_2), (row_tile * 64));
          AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
          AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
          tl::ascend::pv_store_rows(Y[((original_1 * 64) + (row_3 * n))], pool[0], Y[((original_1 * 64) + (row_3 * n))], row_tile, 64, 0, (n - 64));
          tl::ascend::pv_all_barrier(W[0], W[0], W[0]);
        }
      }
    }
    tl::ascend::pv_sync_all(W[0], W[0], W[0]);
    pipe.Destroy();
  }
  AscendC::PipeBarrier<PIPE_ALL>();
}

}
#define bf_block(d,a,b,bl,n,ba) BlockRaw((GM_ADDR)d.GetPhyAddr(),(GM_ADDR)a.GetPhyAddr(),(GM_ADDR)b.GetPhyAddr(),(GM_ADDR)S.GetPhyAddr(),bl,n,ba,fftsAddr)
namespace tl::ascend {
__aicore__ inline void BlockPack( GM_ADDR A_handle,  GM_ADDR Y_handle,  GM_ADDR AH_handle,  GM_ADDR YH_handle,  GM_ADDR YN_handle,  GM_ADDR S_handle, int32_t blocks, int32_t n, int32_t batch, uint64_t fftsAddr) {
  AscendC::TPipe pipe;

  AscendC::GlobalTensor<float> A;
  A.SetGlobalBuffer((__gm__ float*)A_handle);
  AscendC::GlobalTensor<float> Y;
  Y.SetGlobalBuffer((__gm__ float*)Y_handle);
  AscendC::GlobalTensor<float> AH;
  AH.SetGlobalBuffer((__gm__ float*)AH_handle);
  AscendC::GlobalTensor<half> YH;
  YH.SetGlobalBuffer((__gm__ half*)YH_handle);
  AscendC::GlobalTensor<float> YN;
  YN.SetGlobalBuffer((__gm__ float*)YN_handle);
  AscendC::GlobalTensor<float> S;
  S.SetGlobalBuffer((__gm__ float*)S_handle);

  AscendC::TBuf<AscendC::TPosition::VECCALC> ascend_ub; pipe.InitBuffer(ascend_ub, 16384);
  auto bid = AscendC::GetBlockIdx();
  if ASCEND_IS_AIV {
    }
  auto value = ascend_ub.GetWithOffset<float>(4096, 0);
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIV {
    for (int32_t iteration = 0; iteration < ((((blocks * 2) + (((batch * n) * n) / 4096)) - 1) / (blocks * 2)); ++iteration) {
      if ((((iteration * blocks) * 8192) + (bid * 4096)) < ((batch * n) * n)) {
        float power = S.GetValue((((((((int64_t)iteration) * ((int64_t)blocks)) * (int64_t)8192) + (((int64_t)bid) * (int64_t)4096)) / (((int64_t)n) * ((int64_t)n))) * (int64_t)32));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
        tl::ascend::pv_load_rows(value[0], A[(((iteration * blocks) * 8192) + (bid * 4096))], value[0], 1, 4096, 0, 0);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        tl::ascend::pv_scale(value[0], value[0], value[0], (1.000000e+00f / power), 4096);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        tl::ascend::pv_store_rows(AH[(((iteration * blocks) * 8192) + (bid * 4096))], value[0], AH[(((iteration * blocks) * 8192) + (bid * 4096))], 1, 4096, 0, 0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
        tl::ascend::pv_load_rows(value[0], Y[(((iteration * blocks) * 8192) + (bid * 4096))], value[0], 1, 4096, 0, 0);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        tl::ascend::pv_scale(value[0], value[0], value[0], power, 4096);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        tl::ascend::pv_store_rows(YN[(((iteration * blocks) * 8192) + (bid * 4096))], value[0], YN[(((iteration * blocks) * 8192) + (bid * 4096))], 1, 4096, 0, 0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
      }
    }
  }
  pipe.Destroy();
}

}
#define bf_pack(d,a,b,bl,n,ba) BlockPack((GM_ADDR)fr.input.GetPhyAddr(),(GM_ADDR)fr.output.GetPhyAddr(),(GM_ADDR)(d.GetPhyAddr()+2*(ba*n*n)),(GM_ADDR)(d.GetPhyAddr()+3*(ba*n*n)),(GM_ADDR)(d.GetPhyAddr()+8*(ba*n*n)),(GM_ADDR)fr.status.GetPhyAddr(),bl,n,ba,fftsAddr),tl::ascend::FrRestorePipe(&pipe)
namespace tl::ascend {
__aicore__ inline void BlockProduct( GM_ADDR A_handle,  GM_ADDR Y_handle,  GM_ADDR P_handle, int32_t blocks,  GM_ADDR shape_hint, int64_t n, int64_t batch, uint64_t fftsAddr) {
  AscendC::TPipe pipe;

  AscendC::GlobalTensor<half> A;
  A.SetGlobalBuffer((__gm__ half*)A_handle);
  AscendC::GlobalTensor<half> Y;
  Y.SetGlobalBuffer((__gm__ half*)Y_handle);
  AscendC::GlobalTensor<float> P;
  P.SetGlobalBuffer((__gm__ float*)P_handle);
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
  auto a = ascend_l1.GetWithOffset<half>(16384, 0);
  auto b = ascend_l1.GetWithOffset<half>(65536, 32768);
  auto c = ascend_l0c.GetWithOffset<float>(4096, 0);
  int clear = 0;
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIC {
    for (int32_t iteration = 0; iteration < (((((batch * (n / 32)) * (n / 128)) + blocks) - 1) / blocks); ++iteration) {
      if ((bid + (iteration * blocks)) < ((batch * (n / 32)) * (n / 128))) {
        for (int32_t plane = 0; plane < 2; ++plane) {
          for (int32_t kk = 0; kk < (n / 128); ++kk) {
            tl::ascend::copy_gm_to_l1<half, 32, 128>(a[((plane * 8192) + (kk * 4096))], A[((kk * 128) + (((((((bid + (iteration * blocks)) / (n / 128)) % (n / 32)) * 32) + ((plane * batch) * n)) + (((bid + (iteration * blocks)) / ((n / 32) * (n / 128))) * n)) * n))], n, ((1 <= ((((((batch * n) * 2) - ((plane * batch) * n)) - (((bid + (iteration * blocks)) / ((n / 32) * (n / 128))) * n)) / 32) - (((bid + (iteration * blocks)) / (n / 128)) % (n / 32)))) ? 32 : ((0 < (((((batch * n) * 2) - ((plane * batch) * n)) - (((bid + (iteration * blocks)) / ((n / 32) * (n / 128))) * n)) - ((((bid + (iteration * blocks)) / (n / 128)) % (n / 32)) * 32))) ? (((((batch * n) * 2) - ((plane * batch) * n)) - (((bid + (iteration * blocks)) / ((n / 32) * (n / 128))) * n)) - ((((bid + (iteration * blocks)) / (n / 128)) % (n / 32)) * 32)) : 0)), ((1 <= ((n / 128) - kk)) ? 128 : ((0 < (n - (kk * 128))) ? (n - (kk * 128)) : 0)));
            tl::ascend::copy_gm_to_l1<half, 128, 128>(b[((plane * 32768) + (kk * 16384))], Y[((((bid + (iteration * blocks)) % (n / 128)) * 128) + ((((kk * 128) + ((plane * batch) * n)) + (((bid + (iteration * blocks)) / ((n / 32) * (n / 128))) * n)) * n))], n, ((1 <= ((((((batch * n) * 2) - ((plane * batch) * n)) - (((bid + (iteration * blocks)) / ((n / 32) * (n / 128))) * n)) / 128) - kk)) ? 128 : ((0 < (((((batch * n) * 2) - ((plane * batch) * n)) - (((bid + (iteration * blocks)) / ((n / 32) * (n / 128))) * n)) - (kk * 128))) ? (((((batch * n) * 2) - ((plane * batch) * n)) - (((bid + (iteration * blocks)) / ((n / 32) * (n / 128))) * n)) - (kk * 128)) : 0)), ((1 <= ((n / 128) - ((bid + (iteration * blocks)) % (n / 128)))) ? 128 : ((0 < (n - (((bid + (iteration * blocks)) % (n / 128)) * 128))) ? (n - (((bid + (iteration * blocks)) % (n / 128)) * 128)) : 0)));
          }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        for (int32_t term = 0; term < 4; ++term) {
          for (int32_t kk_1 = 0; kk_1 < (n / 128); ++kk_1) {
            clear = 0;
            if (kk_1 == 0) {
              clear = 1;
            }
            tl::ascend::gemm_v0<half, float, 32, 128, 128, false, false>(a[(((term / 2) * 8192) + (kk_1 * 4096))], b[(((term % 2) * 32768) + (kk_1 * 16384))], c[0], ascend_l0a, ascend_l0b, (clear != 0));
            AscendC::PipeBarrier<PIPE_ALL>();
          }
          tl::ascend::copy_l0c_to_gm<float, float, layout::RowMajor, 32, 128, 0>(P[((((bid + (iteration * blocks)) % (n / 128)) * 128) + (((((((bid + (iteration * blocks)) / (n / 128)) % (n / 32)) * 32) + ((term * batch) * n)) + (((bid + (iteration * blocks)) / ((n / 32) * (n / 128))) * n)) * n))], c[0], n, ((1 <= ((((((batch * n) * 4) - ((term * batch) * n)) - (((bid + (iteration * blocks)) / ((n / 32) * (n / 128))) * n)) / 32) - (((bid + (iteration * blocks)) / (n / 128)) % (n / 32)))) ? 32 : ((0 < (((((batch * n) * 4) - ((term * batch) * n)) - (((bid + (iteration * blocks)) / ((n / 32) * (n / 128))) * n)) - ((((bid + (iteration * blocks)) / (n / 128)) % (n / 32)) * 32))) ? (((((batch * n) * 4) - ((term * batch) * n)) - (((bid + (iteration * blocks)) / ((n / 32) * (n / 128))) * n)) - ((((bid + (iteration * blocks)) / (n / 128)) % (n / 32)) * 32)) : 0)), ((1 <= ((n / 128) - ((bid + (iteration * blocks)) % (n / 128)))) ? 128 : ((0 < (n - (((bid + (iteration * blocks)) % (n / 128)) * 128))) ? (n - (((bid + (iteration * blocks)) % (n / 128)) * 128)) : 0)));
          AscendC::PipeBarrier<PIPE_ALL>();
        }
      }
    }
  }
  pipe.Destroy();
}

}
#define bf_product(d,a,b,bl,n,ba) BlockProduct((GM_ADDR)(d.GetPhyAddr()+2*(ba*n*n)),(GM_ADDR)(d.GetPhyAddr()+3*(ba*n*n)),(GM_ADDR)(d.GetPhyAddr()+4*(ba*n*n)),bl,(GM_ADDR)d.GetPhyAddr(),n,ba,fftsAddr)
namespace tl::ascend {
__aicore__ inline void BlockDecode( GM_ADDR P_handle,  GM_ADDR R_handle, int32_t blocks, int32_t n, int32_t batch, uint64_t fftsAddr) {
  AscendC::TPipe pipe;

  AscendC::GlobalTensor<float> P;
  P.SetGlobalBuffer((__gm__ float*)P_handle);
  AscendC::GlobalTensor<float> R;
  R.SetGlobalBuffer((__gm__ float*)R_handle);

  AscendC::TBuf<AscendC::TPosition::VECCALC> ascend_ub; pipe.InitBuffer(ascend_ub, 32768);
  auto bid = AscendC::GetBlockIdx();
  if ASCEND_IS_AIV {
    }
  auto value = ascend_ub.GetWithOffset<float>(4096, 0);
  auto identity = ascend_ub.GetWithOffset<float>(4096, 16384);
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIV {
    for (int32_t iteration = 0; iteration < ((((blocks * 2) + (((batch * n) * n) / 4096)) - 1) / (blocks * 2)); ++iteration) {
      if ((((iteration * blocks) * 8192) + (bid * 4096)) < ((batch * n) * n)) {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
        tl::ascend::pv_load_rows(value[0], P[(((iteration * blocks) * 8192) + (bid * 4096))], value[0], 1, 4096, 0, 0);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        tl::ascend::pv_fill(identity[0], identity[0], identity[0], 0.000000e+00f, 4096);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
        for (int32_t row = 0; row < (4096 / n); ++row) {
          identity.SetValue(((((int64_t)row) * ((int64_t)n)) + ((((((((int64_t)iteration) * ((int64_t)blocks)) * (int64_t)8192) + (((int64_t)bid) * (int64_t)4096)) / ((int64_t)n)) + ((int64_t)row)) % ((int64_t)n))), 1.000000e+00f);
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
        AscendC::Sub(value[0], identity[0], value[0], 4096);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        tl::ascend::pv_store_rows(R[(((iteration * blocks) * 8192) + (bid * 4096))], value[0], R[(((iteration * blocks) * 8192) + (bid * 4096))], 1, 4096, 0, 0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
      }
    }
  }
  pipe.Destroy();
}

}
#define bf_decode(d,a,b,bl,n,ba) BlockDecode((GM_ADDR)(d.GetPhyAddr()+4*(ba*n*n)),(GM_ADDR)(d.GetPhyAddr()+9*(ba*n*n)),bl,n,ba,fftsAddr),tl::ascend::FrRestorePipe(&pipe)
namespace tl::ascend {
__aicore__ inline void BlockFinish( GM_ADDR A_handle,  GM_ADDR Y_handle,  GM_ADDR W_handle,  GM_ADDR S_handle, int32_t blocks, int32_t n, int32_t batch, int32_t stride, uint64_t fftsAddr) {
  AscendC::TPipe pipe;

  AscendC::GlobalTensor<float> A;
  A.SetGlobalBuffer((__gm__ float*)A_handle);
  AscendC::GlobalTensor<float> Y;
  Y.SetGlobalBuffer((__gm__ float*)Y_handle);
  AscendC::GlobalTensor<float> W;
  W.SetGlobalBuffer((__gm__ float*)W_handle);
  AscendC::GlobalTensor<float> S;
  S.SetGlobalBuffer((__gm__ float*)S_handle);

  AscendC::TBuf<AscendC::TPosition::VECCALC> ascend_ub; pipe.InitBuffer(ascend_ub, 98304);
  auto bid = AscendC::GetBlockIdx();
  if ASCEND_IS_AIV {
    }
  float total = 0.000000e+00f;
  float inverse = 0.000000e+00f;
  float peak = 0.000000e+00f;
  auto pool = ascend_ub.GetWithOffset<float>(24576, 0);
  int valid = 0;
  float sum = 0.000000e+00f;
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIV {
    if ((bid / ((blocks * 2) / batch)) < batch) {
      float power = S.GetValue(((((int64_t)bid) / ((((int64_t)blocks) * (int64_t)2) / ((int64_t)batch))) * (int64_t)32));
      total = 0.000000e+00f;
      inverse = 0.000000e+00f;
      peak = 0.000000e+00f;
      for (int32_t iteration = 0; iteration < (((((n * n) / 4096) + ((blocks * 2) / batch)) - 1) / ((blocks * 2) / batch)); ++iteration) {
        if ((((bid % ((blocks * 2) / batch)) * 4096) + ((iteration * ((blocks * 2) / batch)) * 4096)) < (n * n)) {
          AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
          AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
          tl::ascend::pv_load_rows(pool[0], W[(((((bid % ((blocks * 2) / batch)) * 4096) + ((iteration * ((blocks * 2) / batch)) * 4096)) + (((batch * n) * n) * 8)) + (((bid / ((blocks * 2) / batch)) * n) * n))], pool[0], 1, 4096, 0, 0);
          tl::ascend::pv_load_rows(pool[4096], W[(((((bid % ((blocks * 2) / batch)) * 4096) + ((iteration * ((blocks * 2) / batch)) * 4096)) + (((batch * n) * n) * 10)) + (((bid / ((blocks * 2) / batch)) * n) * n))], pool[4096], 1, 4096, 0, 0);
          tl::ascend::pv_load_rows(pool[8192], A[((((bid % ((blocks * 2) / batch)) * 4096) + ((iteration * ((blocks * 2) / batch)) * 4096)) + (((bid / ((blocks * 2) / batch)) * n) * n))], pool[8192], 1, 4096, 0, 0);
          AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
          tl::ascend::pv_add(pool[0], pool[0], pool[4096], 4096);
          AscendC::PipeBarrier<PIPE_V>();
          tl::ascend::pv_abs(pool[4096], pool[0], pool[4096], 4096);
          AscendC::PipeBarrier<PIPE_V>();
          tl::ascend::pv_reduce_max(pool[12288], pool[4096], pool[16384], 4096);
          AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
          AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
          peak = max(peak, (pool.GetValue(12288) / power));
          tl::ascend::pv_mul(pool[4096], pool[0], pool[0], 4096);
          tl::ascend::pv_scale(pool[8192], pool[8192], pool[8192], (1.000000e+00f / power), 4096);
          AscendC::PipeBarrier<PIPE_V>();
          tl::ascend::pv_mul(pool[8192], pool[8192], pool[8192], 4096);
          AscendC::PipeBarrier<PIPE_V>();
          tl::ascend::pv_reduce_sum(pool[12288], pool[4096], pool[16384], 4096);
          AscendC::PipeBarrier<PIPE_V>();
          tl::ascend::pv_reduce_sum(pool[12296], pool[8192], pool[16384], 4096);
          AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
          AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
          inverse = (inverse + pool.GetValue(12288));
          total = (total + pool.GetValue(12296));
          tl::ascend::pv_scale(pool[0], pool[0], pool[0], (1.000000e+00f / power), 4096);
          AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
          AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
          tl::ascend::pv_store_rows(Y[((((bid % ((blocks * 2) / batch)) * 4096) + ((iteration * ((blocks * 2) / batch)) * 4096)) + (((bid / ((blocks * 2) / batch)) * n) * n))], pool[0], Y[((((bid % ((blocks * 2) / batch)) * 4096) + ((iteration * ((blocks * 2) / batch)) * 4096)) + (((bid / ((blocks * 2) / batch)) * n) * n))], 1, 4096, 0, 0);
          tl::ascend::pv_all_barrier(W[0], W[0], W[0]);
        }
      }
      valid = 1;
      for (int32_t stage = 0; stage < (n / 64); ++stage) {
        if (1.000000e+29f <= S.GetValue(((((((int64_t)stage) * ((int64_t)batch)) * (int64_t)32) + ((((int64_t)bid) / ((((int64_t)blocks) * (int64_t)2) / ((int64_t)batch))) * (int64_t)32)) + (int64_t)512))) {
          valid = 0;
        }
      }
      pool.SetValue(0, ((float)valid));
      pool.SetValue(1, inverse);
      pool.SetValue(2, peak);
      pool.SetValue(8, total);
      AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
      AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
      tl::ascend::pv_store_rows(W[(((((bid / ((blocks * 2) / batch)) * 8192) + ((bid % ((blocks * 2) / batch)) * 8)) + (((batch * n) * stride) * 2)) + 2048)], pool[0], W[(((((bid / ((blocks * 2) / batch)) * 8192) + ((bid % ((blocks * 2) / batch)) * 8)) + (((batch * n) * stride) * 2)) + 2048)], 1, 8, 0, 0);
      tl::ascend::pv_store_rows(W[(((((bid / ((blocks * 2) / batch)) * 8192) + ((bid % ((blocks * 2) / batch)) * 8)) + (((batch * n) * stride) * 2)) + 3072)], pool[8], W[(((((bid / ((blocks * 2) / batch)) * 8192) + ((bid % ((blocks * 2) / batch)) * 8)) + (((batch * n) * stride) * 2)) + 3072)], 1, 8, 0, 0);
    }
    tl::ascend::pv_sync_all(W[0], W[0], W[0]);
    if (((bid / ((blocks * 2) / batch)) < batch) && ((bid % ((blocks * 2) / batch)) == 0)) {
      tl::ascend::pv_load_rows(pool[0], W[((((bid / ((blocks * 2) / batch)) * 8192) + (((batch * n) * stride) * 2)) + 3072)], pool[0], 1, (((blocks * 2) / batch) * 8), 0, 0);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
      sum = 0.000000e+00f;
      for (int32_t c = 0; c < ((blocks * 2) / batch); ++c) {
        sum = (sum + pool.GetValue((((int64_t)c) * (int64_t)8)));
      }
      pool.SetValue(0, S.GetValue(((((int64_t)bid) / ((((int64_t)blocks) * (int64_t)2) / ((int64_t)batch))) * (int64_t)32)));
      pool.SetValue(1, sum);
      AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
      AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
      tl::ascend::pv_store_rows(W[((((bid / ((blocks * 2) / batch)) * 8192) + (((batch * n) * stride) * 2)) + 32)], pool[0], W[((((bid / ((blocks * 2) / batch)) * 8192) + (((batch * n) * stride) * 2)) + 32)], 1, 8, 0, 0);
    }
    tl::ascend::pv_sync_all(W[0], W[0], W[0]);
  }
  pipe.Destroy();
}

}
#define bf_finish(d,a,b,bl,n,ba,st) BlockFinish((GM_ADDR)fr.input.GetPhyAddr(),(GM_ADDR)fr.output.GetPhyAddr(),(GM_ADDR)d.GetPhyAddr(),(GM_ADDR)fr.status.GetPhyAddr(),bl,n,ba,st,fftsAddr),tl::ascend::FrRestorePipe(&pipe)
namespace tl::ascend {
__aicore__ inline void BlockAccumulate( GM_ADDR W_handle, int32_t blocks, int32_t n, int32_t batch, uint64_t fftsAddr) {
  AscendC::TPipe pipe;

  AscendC::GlobalTensor<float> W;
  W.SetGlobalBuffer((__gm__ float*)W_handle);

  AscendC::TBuf<AscendC::TPosition::VECCALC> ascend_ub; pipe.InitBuffer(ascend_ub, 32768);
  auto bid = AscendC::GetBlockIdx();
  if ASCEND_IS_AIV {
    }
  auto pool = ascend_ub.GetWithOffset<float>(8192, 0);
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIV {
    for (int32_t iteration = 0; iteration < ((((blocks * 2) + (((batch * n) * n) / 4096)) - 1) / (blocks * 2)); ++iteration) {
      if ((((iteration * blocks) * 8192) + (bid * 4096)) < ((batch * n) * n)) {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
        tl::ascend::pv_load_rows(pool[0], W[((((iteration * blocks) * 8192) + (bid * 4096)) + (((batch * n) * n) * 9))], pool[0], 1, 4096, 0, 0);
        tl::ascend::pv_load_rows(pool[4096], W[((((iteration * blocks) * 8192) + (bid * 4096)) + (((batch * n) * n) * 11))], pool[4096], 1, 4096, 0, 0);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        tl::ascend::pv_add(pool[0], pool[0], pool[4096], 4096);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        tl::ascend::pv_store_rows(W[((((iteration * blocks) * 8192) + (bid * 4096)) + (((batch * n) * n) * 9))], pool[0], W[((((iteration * blocks) * 8192) + (bid * 4096)) + (((batch * n) * n) * 9))], 1, 4096, 0, 0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
      }
    }
  }
  pipe.Destroy();
}

}
#define bf_accumulate(d,a,b,bl,n,ba) BlockAccumulate((GM_ADDR)d.GetPhyAddr(),bl,n,ba,fftsAddr),tl::ascend::FrRestorePipe(&pipe)
// END DSL block strategy
