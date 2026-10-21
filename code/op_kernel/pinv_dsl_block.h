#pragma once
// BEGIN DSL block strategy
#pragma once
namespace tl::ascend {
__aicore__ inline void br_sub(AscendC::LocalTensor<float>d,AscendC::LocalTensor<float>a,AscendC::LocalTensor<float>b,int n){AscendC::Sub(d,a,b,n);}
__aicore__ inline void bp_store_half(AscendC::GlobalTensor<half>d,AscendC::LocalTensor<half>a,AscendC::GlobalTensor<half>,int n){AscendC::DataCopy(d,a,n);}
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
    count = 8704;
    dma_pad = 64;
    condition = 1.000000e+30f;
    valid = 1;
    tl::ascend::pv_fill(pool[0], pool[0], pool[0], 0.000000e+00f, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
    tl::ascend::pv_load_rows(pool[0], A[(job * 4096)], pool[0], 64, 64, 0, (136 - dma_pad), (bool)0);
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
        indices.SetValue(i, ((uint32_t)(i * 544)));
      }
      tl::ascend::pv_fill(pool[19040], pool[19040], pool[19040], 0.000000e+00f, 64);
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::pv_mask_begin(pool[0], pool[0], pool[0], 64);
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
        order.SetValue((pivot_tile + 64), ((uint32_t)((pivot * 544) + 256)));
        consumed = 0;
        base = ((pivot_tile / 8) * 8);
        if (((pivot_tile % 8) == 7) || (pivot_tile == 63)) {
          consumed = 1;
          base = (base + 8);
        }
        float divisor = pool.GetValue((((int64_t)pivot) + (int64_t)19040));
        pool.SetValue((((int64_t)pivot) + (int64_t)19040), 0.000000e+00f);
        pool.SetValue((((((int64_t)pivot) * (int64_t)136) + ((int64_t)pivot_tile)) + (int64_t)64), 1.000000e+00f);
        AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
        active = 72;
        if (consumed != 0) {
          active = (active - 8);
        }
        tl::ascend::pv_scale(pool[((pivot * 136) + base)], pool[((pivot * 136) + base)], pool[((pivot * 136) + base)], (-1.000000e+00f / divisor), active);
        tl::ascend::pv_brcb_rows(pool[17408], pool[19040], pool[17408], 64, 8);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::pv_mad_strided(pool[base], pool[((pivot * 136) + base)], pool[17408], 64, 64, 1, 1, 0, 17, 0, 1);
        if (consumed == 0) {
          tl::ascend::pv_mad_strided(pool[(base + 64)], pool[(((pivot * 136) + base) + 64)], pool[17408], 64, 8, 17, 0, 1, 136, 0, 8);
        }
        AscendC::PipeBarrier<PIPE_V>();
      }
      if (valid != 0) {
        tl::ascend::pv_fill(pool[8704], pool[8704], pool[8704], 0.000000e+00f, count);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::pv_square_padded_rows(pool[8704], pool[64], pool[8704], 64, 64, 136);
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
        for (int32_t kk_1 = 0; kk_1 < (inner / 16); ++kk_1) {
          tl::ascend::copy_gm_to_l1<float, 64, 16>(a[(kk_1 * 1024)], A[((kk_1 * 16) + ((((((bid + (iteration * blocks)) / (cols / 128)) % (rows / 64)) * 64) + (((bid + (iteration * blocks)) / ((rows / 64) * (cols / 128))) * rows)) * inner))], inner, ((1 <= (((rows * (batch - ((bid + (iteration * blocks)) / ((rows / 64) * (cols / 128))))) / 64) - (((bid + (iteration * blocks)) / (cols / 128)) % (rows / 64)))) ? 64 : ((0 < ((rows * (batch - ((bid + (iteration * blocks)) / ((rows / 64) * (cols / 128))))) - ((((bid + (iteration * blocks)) / (cols / 128)) % (rows / 64)) * 64))) ? ((rows * (batch - ((bid + (iteration * blocks)) / ((rows / 64) * (cols / 128))))) - ((((bid + (iteration * blocks)) / (cols / 128)) % (rows / 64)) * 64)) : 0)), ((1 <= ((inner / 16) - kk_1)) ? 16 : ((0 < (inner - (kk_1 * 16))) ? (inner - (kk_1 * 16)) : 0)));
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        for (int32_t kk_2 = 0; kk_2 < (inner / 16); ++kk_2) {
          clear = 0;
          if (kk_2 == 0) {
            clear = 1;
          }
          tl::ascend::gemm_v0<float, float, 64, 128, 16, false, false>(a[(kk_2 * 1024)], b[(kk_2 * 2048)], c[0], ascend_l0a, ascend_l0b, (clear != 0));
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
  float magnitude = 0.000000e+00f;
  float power = 0.000000e+00f;
  int admitted = 0;
  float cost = 0.000000e+00f;
  int chosen = 0;
  float best = 0.000000e+00f;
  int original = 0;
  auto vid = AscendC::GetSubBlockIdx();
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
    AscendC::TBuf<AscendC::TPosition::VECCALC> ascend_ub; pipe.InitBuffer(ascend_ub, 49152);
    auto pool = ascend_ub.GetWithOffset<float>(12288, 0);
    tl::ascend::fr_bind(A[0], Y[0], W[0]);
    tl::ascend::pv_sync_init(W[0], W[0], W[0]);
    if (bid < batch) {
      magnitude = 0.000000e+00f;
      for (int32_t chunk = 0; chunk < (((n * n) + 4095) / 4096); ++chunk) {
        tl::ascend::pv_load_rows(pool[0], A[((chunk * 4096) + ((bid * n) * n))], pool[0], 1, min(((n * n) - (chunk * 4096)), 4096), 0, 0);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        tl::ascend::pv_abs(pool[0], pool[0], pool[0], min(((n * n) - (chunk * 4096)), 4096));
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::pv_reduce_max(pool[8192], pool[0], pool[4096], min(((n * n) - (chunk * 4096)), 4096));
        AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
        magnitude = max(magnitude, pool.GetValue(8192));
      }
      power = 1.000000e+00f;
      if ((0.000000e+00f < magnitude) && (magnitude < 1.000000e+30f)) {
        while (1) {
          if (!((power < magnitude))) { break; }
          power = (power * 2.000000e+00f);
        }
        while (1) {
          if (!((magnitude <= (power * 5.000000e-01f)))) { break; }
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
    tl::ascend::pv_load_rows(pool[10240], S[0], pool[10240], 1, (batch * 32), 0, 0);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
    for (int32_t iteration = 0; iteration < ((((blocks * 2) + ((batch * n) / 8)) - 1) / (blocks * 2)); ++iteration) {
      if ((((iteration * blocks) * 16) + (bid * 8)) < (batch * n)) {
        float power_1 = pool.GetValue(((((((((int64_t)iteration) * ((int64_t)blocks)) * (int64_t)16) + (((int64_t)bid) * (int64_t)8)) / ((int64_t)n)) * (int64_t)32) + (int64_t)10240));
        tl::ascend::pv_load_rows(pool[0], A[((((iteration * blocks) * 16) + (bid * 8)) * n)], pool[0], 8, n, 0, 0);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        tl::ascend::pv_scale(pool[0], pool[0], pool[0], (1.000000e+00f / power_1), (n * 8));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        tl::ascend::pv_store_rows(W[((((iteration * blocks) * 16) + (bid * 8)) * n)], pool[0], W[((((iteration * blocks) * 16) + (bid * 8)) * n)], 8, n, 0, 0);
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
      tl::ascend::pv_load_rows(pool[9216], W[(((batch * 4096) + ((batch * n) * 448)) + (((batch * n) * n) * 6))], pool[9216], 1, ((batch * ((n / 64) - stage_1)) * 8), 0, 0);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
      for (int32_t iteration_1 = 0; iteration_1 < ((((blocks * 2) + ((batch * n) / 8)) - 1) / (blocks * 2)); ++iteration_1) {
        if ((((iteration_1 * blocks) * 16) + (bid * 8)) < (batch * n)) {
          chosen = 0;
          best = 1.000000e+30f;
          for (int32_t candidate = 0; candidate < ((n / 64) - stage_1); ++candidate) {
            float cost_1 = pool.GetValue(((((((((((int64_t)iteration_1) * ((int64_t)blocks)) * (int64_t)16) + (((int64_t)bid) * (int64_t)8)) / ((int64_t)n)) * ((((int64_t)n) / (int64_t)64) - ((int64_t)stage_1))) * (int64_t)8) + (((int64_t)candidate) * (int64_t)8)) + (int64_t)9216));
            if ((0.000000e+00f <= cost_1) && (cost_1 < best)) {
              chosen = candidate;
              best = cost_1;
            }
          }
          int32_t position = ((chosen * 64) + (stage_1 * 64));
          if (((((iteration_1 * blocks) * 16) + (bid * 8)) % n) == (stage_1 * 64)) {
            pool.SetValue(0, best);
            pool.SetValue(1, ((float)position));
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
            tl::ascend::pv_store_rows(S[((((stage_1 * batch) * 32) + (((((iteration_1 * blocks) * 16) + (bid * 8)) / n) * 32)) + 512)], pool[0], S[((((stage_1 * batch) * 32) + (((((iteration_1 * blocks) * 16) + (bid * 8)) / n) * 32)) + 512)], 1, 8, 0, 0);
            tl::ascend::pv_all_barrier(W[0], W[0], W[0]);
          }
          original = ((((iteration_1 * blocks) * 16) + (bid * 8)) % n);
          if (((stage_1 * 64) <= ((((iteration_1 * blocks) * 16) + (bid * 8)) % n)) && (((((iteration_1 * blocks) * 16) + (bid * 8)) % n) < ((stage_1 * 64) + 64))) {
            original = ((position + ((((iteration_1 * blocks) * 16) + (bid * 8)) % n)) - (stage_1 * 64));
          } else {
            if ((position <= ((((iteration_1 * blocks) * 16) + (bid * 8)) % n)) && (((((iteration_1 * blocks) * 16) + (bid * 8)) % n) < (position + 64))) {
              original = (((stage_1 * 64) + ((((iteration_1 * blocks) * 16) + (bid * 8)) % n)) - position);
            }
          }
          tl::ascend::pv_load_rows(pool[8192], W[(n * ((((((iteration_1 * blocks) * 16) + (bid * 8)) / n) * n) + original))], pool[8192], 8, 64, (n - 64), 0);
          AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
          if (((stage_1 * 64) <= ((((iteration_1 * blocks) * 16) + (bid * 8)) % n)) && (((((iteration_1 * blocks) * 16) + (bid * 8)) % n) < ((stage_1 * 64) + 64))) {
            tl::ascend::pv_fill(pool[8192], pool[8192], pool[8192], 0.000000e+00f, 512);
          }
          AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
          AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
          tl::ascend::pv_store_rows(W[(((((batch * 4096) + ((iteration_1 * blocks) * 1024)) + (bid * 512)) + ((batch * n) * 384)) + (((batch * n) * n) * 4))], pool[8192], W[(((((batch * 4096) + ((iteration_1 * blocks) * 1024)) + (bid * 512)) + ((batch * n) * 384)) + (((batch * n) * n) * 4))], 8, 64, 0, 0);
          tl::ascend::pv_fill(pool[0], pool[0], pool[0], 0.000000e+00f, (n * 8));
          AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
          AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
          tl::ascend::pv_load_rows(pool[0], W[((n * ((((((iteration_1 * blocks) * 16) + (bid * 8)) / n) * n) + original)) + 64)], pool[0], 8, (n - 64), 64, 64);
          AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
          if (((stage_1 * 64) <= ((((iteration_1 * blocks) * 16) + (bid * 8)) % n)) && (((((iteration_1 * blocks) * 16) + (bid * 8)) % n) < ((stage_1 * 64) + 64))) {
            for (int32_t lane = 0; lane < 8; ++lane) {
              pool.SetValue((((((((int64_t)n) * (((int64_t)lane) + (int64_t)1)) + ((((((int64_t)iteration_1) * ((int64_t)blocks)) * (int64_t)16) + (((int64_t)bid) * (int64_t)8)) % ((int64_t)n))) + ((int64_t)lane)) - (((int64_t)stage_1) * (int64_t)64)) - (int64_t)64), 1.000000e+00f);
            }
          }
          AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
          AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
          tl::ascend::pv_store_rows(W[((((batch * n) * n) * 2) + ((((iteration_1 * blocks) * 16) + (bid * 8)) * n))], pool[0], W[((((batch * n) * n) * 2) + ((((iteration_1 * blocks) * 16) + (bid * 8)) * n))], 8, n, 0, 0);
          tl::ascend::pv_all_barrier(W[0], W[0], W[0]);
          if (((stage_1 * 64) <= ((((iteration_1 * blocks) * 16) + (bid * 8)) % n)) && (((((iteration_1 * blocks) * 16) + (bid * 8)) % n) < ((stage_1 * 64) + 64))) {
            tl::ascend::pv_store_rows(W[(((((batch * 4096) + ((batch * n) * 128)) + ((((((iteration_1 * blocks) * 16) + (bid * 8)) / n) * n) * 64)) + (((batch * n) * n) * 4)) + ((((((iteration_1 * blocks) * 16) + (bid * 8)) % n) - (stage_1 * 64)) * n))], pool[0], W[(((((batch * 4096) + ((batch * n) * 128)) + ((((((iteration_1 * blocks) * 16) + (bid * 8)) / n) * n) * 64)) + (((batch * n) * n) * 4)) + ((((((iteration_1 * blocks) * 16) + (bid * 8)) % n) - (stage_1 * 64)) * n))], 8, n, 0, 0);
            tl::ascend::pv_all_barrier(W[0], W[0], W[0]);
            tl::ascend::pv_load_rows(pool[8192], W[(((((((((((iteration_1 * blocks) * 16) + (bid * 8)) / n) * ((n / 64) - stage_1)) * 4096) + (chosen * 4096)) + (((((iteration_1 * blocks) * 16) + (bid * 8)) % n) * 64)) + ((batch * n) * 64)) + (((batch * n) * n) * 4)) - (stage_1 * 4096))], pool[8192], 8, 64, 0, 0);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(0);
            tl::ascend::pv_store_rows(W[(((((((((iteration_1 * blocks) * 16) + (bid * 8)) / n) * 4096) + ((batch * n) * 128)) + (((((iteration_1 * blocks) * 16) + (bid * 8)) % n) * 64)) + (((batch * n) * n) * 4)) - (stage_1 * 4096))], pool[8192], W[(((((((((iteration_1 * blocks) * 16) + (bid * 8)) / n) * 4096) + ((batch * n) * 128)) + (((((iteration_1 * blocks) * 16) + (bid * 8)) % n) * 64)) + (((batch * n) * n) * 4)) - (stage_1 * 4096))], 8, 64, 0, 0);
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
      for (int32_t iteration_2 = 0; iteration_2 < ((((blocks * 2) + ((batch * n) / 8)) - 1) / (blocks * 2)); ++iteration_2) {
        if ((((iteration_2 * blocks) * 16) + (bid * 8)) < (batch * n)) {
          if (((stage_1 * 64) <= ((((iteration_2 * blocks) * 16) + (bid * 8)) % n)) && (((((iteration_2 * blocks) * 16) + (bid * 8)) % n) < ((stage_1 * 64) + 64))) {
            tl::ascend::pv_load_rows(pool[0], W[(((((batch * 4096) + ((batch * n) * 256)) + ((((((iteration_2 * blocks) * 16) + (bid * 8)) / n) * n) * 64)) + (((batch * n) * n) * 4)) + ((((((iteration_2 * blocks) * 16) + (bid * 8)) % n) - (stage_1 * 64)) * n))], pool[0], 8, n, 0, 0);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
          } else {
            tl::ascend::pv_load_rows(pool[0], W[((((batch * n) * n) * 2) + ((((iteration_2 * blocks) * 16) + (bid * 8)) * n))], pool[0], 8, n, 0, 0);
            tl::ascend::pv_load_rows(pool[8192], W[((((batch * 4096) + ((batch * n) * 448)) + (((batch * n) * n) * 4)) + ((((iteration_2 * blocks) * 16) + (bid * 8)) * n))], pool[8192], 8, n, 0, 0);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
            tl::ascend::br_sub(pool[0], pool[0], pool[8192], (n * 8));
          }
          tl::ascend::pv_all_barrier(W[0], W[0], W[0]);
          tl::ascend::pv_store_rows(W[((((iteration_2 * blocks) * 16) + (bid * 8)) * n)], pool[0], W[((((iteration_2 * blocks) * 16) + (bid * 8)) * n)], 8, n, 0, 0);
          tl::ascend::pv_all_barrier(W[0], W[0], W[0]);
        }
      }
      tl::ascend::pv_sync_all(W[0], W[0], W[0]);
    }
    tl::ascend::pv_load_rows(pool[10240], S[0], pool[10240], 1, (batch * 32), 0, 0);
    tl::ascend::pv_load_rows(pool[11008], S[512], pool[11008], 1, 512, 0, 0);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
    for (int32_t iteration_3 = 0; iteration_3 < ((((blocks * 2) + ((batch * n) / 8)) - 1) / (blocks * 2)); ++iteration_3) {
      if ((((iteration_3 * blocks) * 16) + (bid * 8)) < (batch * n)) {
        float power_2 = pool.GetValue(((((((((int64_t)iteration_3) * ((int64_t)blocks)) * (int64_t)16) + (((int64_t)bid) * (int64_t)8)) / ((int64_t)n)) * (int64_t)32) + (int64_t)10240));
        for (int32_t col = 0; col < (n / 64); ++col) {
          pool.SetValue((col + 10600), ((float)col));
        }
        for (int32_t stage_2 = 0; stage_2 < (n / 64); ++stage_2) {
          int32_t chosen_1 = (((int32_t)pool.GetValue(((((((int64_t)stage_2) * ((int64_t)batch)) * (int64_t)32) + (((((((int64_t)iteration_3) * ((int64_t)blocks)) * (int64_t)16) + (((int64_t)bid) * (int64_t)8)) / ((int64_t)n)) * (int64_t)32)) + (int64_t)11009))) / 64);
          float old = pool.GetValue((stage_2 + 10600));
          pool.SetValue((stage_2 + 10600), pool.GetValue((((int64_t)chosen_1) + (int64_t)10600)));
          pool.SetValue((((int64_t)chosen_1) + (int64_t)10600), old);
        }
        for (int32_t col_1 = 0; col_1 < (n / 64); ++col_1) {
          int32_t original_1 = ((int32_t)pool.GetValue((col_1 + 10600)));
          tl::ascend::pv_load_rows(pool[0], W[((col_1 * 64) + ((((iteration_3 * blocks) * 16) + (bid * 8)) * n))], pool[0], 8, 64, (n - 64), 0);
          AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
          tl::ascend::pv_scale(pool[0], pool[0], pool[0], (1.000000e+00f / power_2), 512);
          AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
          AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
          tl::ascend::pv_store_rows(Y[((original_1 * 64) + ((((iteration_3 * blocks) * 16) + (bid * 8)) * n))], pool[0], Y[((original_1 * 64) + ((((iteration_3 * blocks) * 16) + (bid * 8)) * n))], 8, 64, 0, (n - 64));
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
  AscendC::GlobalTensor<half> AH;
  AH.SetGlobalBuffer((__gm__ half*)AH_handle);
  AscendC::GlobalTensor<half> YH;
  YH.SetGlobalBuffer((__gm__ half*)YH_handle);
  AscendC::GlobalTensor<float> YN;
  YN.SetGlobalBuffer((__gm__ float*)YN_handle);
  AscendC::GlobalTensor<float> S;
  S.SetGlobalBuffer((__gm__ float*)S_handle);

  AscendC::TBuf<AscendC::TPosition::VECCALC> ascend_ub; pipe.InitBuffer(ascend_ub, 57344);
  auto bid = AscendC::GetBlockIdx();
  if ASCEND_IS_AIV {
    }
  auto value = ascend_ub.GetWithOffset<float>(4096, 0);
  auto half_buf = ascend_ub.GetWithOffset<half>(4096, 16384);
  auto high = ascend_ub.GetWithOffset<float>(4096, 24576);
  auto low = ascend_ub.GetWithOffset<float>(4096, 40960);
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
        {
        AscendC::Muls(value[0], value[0], (1.000000e+00f / power), 4096);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(half_buf[0], value[0], AscendC::RoundMode::CAST_NONE, 4096);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(high[0], half_buf[0], AscendC::RoundMode::CAST_NONE, 4096);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(low[0], value[0], high[0], 4096);
        AscendC::PipeBarrier<PIPE_V>();
        {
        AscendC::Muls(low[0], low[0], 4.096000e+03f, 4096);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        tl::ascend::bp_store_half(AH[(((iteration * blocks) * 8192) + (bid * 4096))], half_buf[0], AH[(((iteration * blocks) * 8192) + (bid * 4096))], 4096);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::Cast(half_buf[0], low[0], AscendC::RoundMode::CAST_NONE, 4096);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        tl::ascend::bp_store_half(AH[((((iteration * blocks) * 8192) + (bid * 4096)) + ((batch * n) * n))], half_buf[0], AH[((((iteration * blocks) * 8192) + (bid * 4096)) + ((batch * n) * n))], 4096);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
        tl::ascend::pv_load_rows(value[0], Y[(((iteration * blocks) * 8192) + (bid * 4096))], value[0], 1, 4096, 0, 0);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        {
        AscendC::Muls(value[0], value[0], power, 4096);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        tl::ascend::pv_store_rows(YN[(((iteration * blocks) * 8192) + (bid * 4096))], value[0], YN[(((iteration * blocks) * 8192) + (bid * 4096))], 1, 4096, 0, 0);
        AscendC::Cast(half_buf[0], value[0], AscendC::RoundMode::CAST_NONE, 4096);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(high[0], half_buf[0], AscendC::RoundMode::CAST_NONE, 4096);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(low[0], value[0], high[0], 4096);
        AscendC::PipeBarrier<PIPE_V>();
        {
        AscendC::Muls(low[0], low[0], 4.096000e+03f, 4096);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        tl::ascend::bp_store_half(YH[(((iteration * blocks) * 8192) + (bid * 4096))], half_buf[0], YH[(((iteration * blocks) * 8192) + (bid * 4096))], 4096);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::Cast(half_buf[0], low[0], AscendC::RoundMode::CAST_NONE, 4096);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        tl::ascend::bp_store_half(YH[((((iteration * blocks) * 8192) + (bid * 4096)) + ((batch * n) * n))], half_buf[0], YH[((((iteration * blocks) * 8192) + (bid * 4096)) + ((batch * n) * n))], 4096);
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

  AscendC::TBuf<AscendC::TPosition::VECCALC> ascend_ub; pipe.InitBuffer(ascend_ub, 131072);
  auto bid = AscendC::GetBlockIdx();
  if ASCEND_IS_AIV {
    }
  auto high = ascend_ub.GetWithOffset<float>(4096, 0);
  auto low = ascend_ub.GetWithOffset<float>(4096, 16384);
  auto part = ascend_ub.GetWithOffset<float>(4096, 32768);
  auto total = ascend_ub.GetWithOffset<float>(4096, 49152);
  auto bv = ascend_ub.GetWithOffset<float>(4096, 65536);
  auto av = ascend_ub.GetWithOffset<float>(4096, 81920);
  auto error = ascend_ub.GetWithOffset<float>(4096, 98304);
  auto alt = ascend_ub.GetWithOffset<float>(4096, 114688);
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIV {
    for (int32_t iteration = 0; iteration < ((((blocks * 2) + (((batch * n) * n) / 4096)) - 1) / (blocks * 2)); ++iteration) {
      if ((((iteration * blocks) * 8192) + (bid * 4096)) < ((batch * n) * n)) {
        tl::ascend::pv_fill(high[0], high[0], high[0], 0.000000e+00f, 4096);
        tl::ascend::pv_fill(low[0], low[0], low[0], 0.000000e+00f, 4096);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
        for (int32_t row = 0; row < (4096 / n); ++row) {
          high.SetValue(((((int64_t)row) * ((int64_t)n)) + ((((((((int64_t)iteration) * ((int64_t)blocks)) * (int64_t)8192) + (((int64_t)bid) * (int64_t)4096)) / ((int64_t)n)) + ((int64_t)row)) % ((int64_t)n))), 1.000000e+00f);
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
        for (int32_t term = 0; term < 4; ++term) {
          AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
          AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
          tl::ascend::pv_load_rows(part[0], P[((((iteration * blocks) * 8192) + (bid * 4096)) + (((term * batch) * n) * n))], part[0], 1, 4096, 0, 0);
          AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
          if (term == 0) {
            {
            AscendC::Muls(part[0], part[0], -1.000000e+00f, 4096);
            }
          } else {
            if (term == 3) {
              {
              AscendC::Muls(part[0], part[0], -2.441406e-04f, 4096);
              }
              AscendC::PipeBarrier<PIPE_V>();
              {
              AscendC::Muls(part[0], part[0], 2.441406e-04f, 4096);
              }
            } else {
              {
              AscendC::Muls(part[0], part[0], -2.441406e-04f, 4096);
              }
            }
          }
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Add(total[0], high[0], part[0], 4096);
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Sub(bv[0], total[0], high[0], 4096);
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Sub(av[0], total[0], bv[0], 4096);
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Sub(error[0], high[0], av[0], 4096);
          AscendC::Sub(alt[0], part[0], bv[0], 4096);
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Add(error[0], error[0], alt[0], 4096);
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Add(low[0], low[0], error[0], 4096);
          AscendC::PipeBarrier<PIPE_V>();
          tl::ascend::pv_copy(high[0], total[0], high[0], 4096);
          AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::Add(high[0], high[0], low[0], 4096);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        tl::ascend::pv_store_rows(R[(((iteration * blocks) * 8192) + (bid * 4096))], high[0], R[(((iteration * blocks) * 8192) + (bid * 4096))], 1, 4096, 0, 0);
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
// END DSL block strategy
