#pragma once
#include "pinv_dsl_common.h"
__aicore__ inline void PvStageCompact( GM_ADDR A_handle,  GM_ADDR X_handle,  GM_ADDR W_handle,  GM_ADDR S_handle, int32_t blocks, int32_t m, int32_t n, int32_t batch, float atol, float rtol, uint64_t fftsAddr) {
  if (static_cast<int32_t>(AscendC::GetBlockIdx()) >= batch) return;
  AscendC::TPipe pipe;

  AscendC::GlobalTensor<float> A;
  A.SetGlobalBuffer((__gm__ float*)A_handle);
  AscendC::GlobalTensor<float> X;
  X.SetGlobalBuffer((__gm__ float*)X_handle);
  AscendC::GlobalTensor<float> W;
  W.SetGlobalBuffer((__gm__ float*)W_handle);
  AscendC::GlobalTensor<float> S;
  S.SetGlobalBuffer((__gm__ float*)S_handle);

  AscendC::TBuf<AscendC::TPosition::VECCALC> ascend_ub; pipe.InitBuffer(ascend_ub, 6016);
  auto bid = AscendC::GetBlockIdx();
  if ASCEND_IS_AIV {
    }
  int job = 0;
  int offset = 0;
  int dimension = 0;
  int accepted = 0;
  int valid = 0;
  auto pool = ascend_ub.GetWithOffset<float>(1440, 0);
  float power = 0.000000e+00f;
  uint32_t power_word = (uint32_t)0;
  auto indices = ascend_ub.GetWithOffset<uint32_t>(64, 5760);
  uint64_t permutation = (uint64_t)0;
  int valid_1 = 0;
  float power_1 = 0.000000e+00f;
  uint32_t power_word_1 = (uint32_t)0;
  int decision = 0;
  uint32_t index_word = (uint32_t)0;
  float best = 0.000000e+00f;
  int pivot = 0;
  uint32_t shift = (uint32_t)0;
  float best_1 = 0.000000e+00f;
  int pivot_1 = 0;
  uint32_t shift_1 = (uint32_t)0;
  uint32_t shift_2 = (uint32_t)0;
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIV {
    tl::ascend::pv_module_bind(A[0], X[0], W[0]);
    job = bid;
    while (1) {
      if (!((job < batch))) { break; }
      offset = ((job * m) * n);
      dimension = n;
      accepted = 0;
      if ((m == n) && (n <= 32)) {
        if ((15 <= n) && (n <= 16)) {
          if (0 < dimension) {
            int32_t count = (dimension * 40);
            valid = 1;
            tl::ascend::pv_fill(pool[0], pool[0], pool[0], 0.000000e+00f, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
            tl::ascend::pv_load_rows(pool[0], A[offset], pool[0], dimension, dimension, 0, 24, (0 < (dimension % 8)));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
            tl::ascend::pv_abs(pool[640], pool[0], pool[640], count);
            AscendC::PipeBarrier<PIPE_V>();
            tl::ascend::pv_reduce_max(pool[1368], pool[640], pool[1336], count);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
            float magnitude = pool.GetValue(1368);
            power = 1.000000e+00f;
            if ((magnitude < 1.000000e-01f) || (1.701412e+38f <= magnitude)) {
              valid = 0;
            }
            if (valid != 0) {
              power_word = (((*(uint32_t *)(&(magnitude))) + (uint32_t)8388607) & (uint32_t)2139095040);
              power = (*(float *)(&(power_word)));
              tl::ascend::pv_scale(pool[0], pool[0], pool[0], (1.000000e+00f / power), count);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_mul(pool[640], pool[0], pool[0], count);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_reduce_sum(pool[1368], pool[640], pool[1336], count);
              AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
              AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
              float total = pool.GetValue(1368);
              for (int32_t i = 0; i < dimension; ++i) {
                pool.SetValue(((((int64_t)i) * (int64_t)41) + (int64_t)16), 1.000000e+00f);
                indices.SetValue(i, ((uint32_t)(i * 160)));
              }
              tl::ascend::pv_fill(pool[1416], pool[1416], pool[1416], 0.000000e+00f, 16);
              AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
              AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
              for (int32_t column = 0; column < dimension; ++column) {
                tl::ascend::pv_gather(pool[1416], pool[0], indices[0], (column * 4), dimension);
                AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
                float divisor = pool.GetValue((((int64_t)column) + (int64_t)1416));
                if (max(divisor, (divisor * -1.000000e+00f)) <= 3.906250e-03f) {
                  valid = 0;
                  break;
                }
                tl::ascend::pv_scale(pool[(column * 40)], pool[(column * 40)], pool[(column * 40)], (-1.000000e+00f / divisor), 32);
                pool.SetValue((((int64_t)column) + (int64_t)1416), 0.000000e+00f);
                AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
                tl::ascend::pv_brcb_rows(pool[640], pool[1416], pool[640], 16, 40);
                AscendC::PipeBarrier<PIPE_V>();
                tl::ascend::pv_gj_update(pool[0], pool[(column * 40)], pool[640], 32, dimension, 40);
                AscendC::PipeBarrier<PIPE_V>();
              }
              if (valid != 0) {
                tl::ascend::pv_fill(pool[640], pool[640], pool[640], 0.000000e+00f, count);
                AscendC::PipeBarrier<PIPE_V>();
                tl::ascend::pv_square_padded_rows(pool[640], pool[16], pool[640], 16, dimension, 40);
                AscendC::PipeBarrier<PIPE_V>();
                tl::ascend::pv_reduce_sum(pool[1368], pool[640], pool[1336], count);
                AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
                float inverse_total = pool.GetValue(1368);
                pool.SetValue(1368, total);
                AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
                tl::ascend::pv_sqrt(pool[1368], pool[1368], pool[1368], 1);
                AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
                float cutoff = max((pool.GetValue(1368) * rtol), (atol / power));
                if ((inverse_total <= 0.000000e+00f) || (8.100000e-01f <= ((inverse_total * cutoff) * cutoff))) {
                  valid = 0;
                }
                if (valid != 0) {
                  tl::ascend::pv_scale(pool[0], pool[0], pool[0], (-1.000000e+00f / power), count);
                  AscendC::PipeBarrier<PIPE_V>();
                  if (dimension == 15) {
                    tl::ascend::pv_pack15(pool[640], pool[16], pool[640], dimension, 40);
                    AscendC::PipeBarrier<PIPE_V>();
                  }
                  AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
                  AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
                  if (dimension == 15) {
                    tl::ascend::pv_store_rows(X[offset], pool[640], X[offset], 1, (dimension * dimension), 0, 0);
                  } else {
                    tl::ascend::pv_store_rows(X[offset], pool[16], X[offset], dimension, dimension, 24, 0);
                  }
                  AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
                  AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
                }
              }
            }
            accepted = valid;
          }
          if (accepted == 0) {
            int32_t count_1 = (dimension * 40);
            permutation = (uint64_t)0;
            valid_1 = 1;
            tl::ascend::pv_fill(pool[0], pool[0], pool[0], 0.000000e+00f, count_1);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
            tl::ascend::pv_load_rows(pool[0], A[offset], pool[0], dimension, dimension, 0, 24, (0 < (dimension % 8)));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
            tl::ascend::pv_abs(pool[640], pool[0], pool[640], count_1);
            AscendC::PipeBarrier<PIPE_V>();
            tl::ascend::pv_reduce_max(pool[1368], pool[640], pool[1336], count_1);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
            float scale = pool.GetValue(1368);
            power_1 = 1.000000e+00f;
            if ((scale < 1.000000e-01f) || (1.701412e+38f <= scale)) {
              valid_1 = 0;
            }
            if (valid_1 != 0) {
              power_word_1 = (((*(uint32_t *)(&(scale))) + (uint32_t)8388607) & (uint32_t)2139095040);
              power_1 = (*(float *)(&(power_word_1)));
              tl::ascend::pv_scale(pool[0], pool[0], pool[0], (1.000000e+00f / power_1), count_1);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_mul(pool[640], pool[0], pool[0], count_1);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_reduce_sum(pool[1368], pool[640], pool[1336], count_1);
              AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
              AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
              float total_1 = pool.GetValue(1368);
              for (int32_t i_1 = 0; i_1 < dimension; ++i_1) {
                pool.SetValue(((((int64_t)i_1) * (int64_t)41) + (int64_t)16), 1.000000e+00f);
                index_word = ((uint32_t)(i_1 * 160));
                pool.SetValue((((int64_t)i_1) + (int64_t)1400), (*(float *)(&(index_word))));
              }
              tl::ascend::pv_mask_begin(pool[0], pool[0], pool[0], dimension);
              tl::ascend::pv_fill(pool[1416], pool[1416], pool[1416], 0.000000e+00f, 16);
              AscendC::PipeBarrier<PIPE_V>();
              AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
              AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
              for (int32_t pivot_tile = 0; pivot_tile < dimension; ++pivot_tile) {
                tl::ascend::pv_gather_bits(pool[1416], pool[0], pool[1400], (pivot_tile * 4), dimension);
                AscendC::PipeBarrier<PIPE_V>();
                tl::ascend::pv_abs(pool[1320], pool[1416], pool[1320], 16);
                AscendC::PipeBarrier<PIPE_V>();
                tl::ascend::pv_mask_reduce(pool[1368], pool[1320], pool[1368]);
                AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
                tl::ascend::pv_reduce_scalars(pool[0], pool[0], pool[0], best, pivot);
                if (best <= 0.000000e+00f) {
                  valid_1 = 0;
                  break;
                }
                tl::ascend::pv_mask_remove(pool[0], pool[0], pool[0], pivot);
                shift = ((uint32_t)(pivot * 4));
                permutation = (permutation | (((uint64_t)pivot_tile) << ((uint64_t)shift)));
                float factor = (-1.000000e+00f / pool.GetValue(((((int64_t)pivot) * (int64_t)40) + ((int64_t)pivot_tile))));
                tl::ascend::pv_scale(pool[(pivot * 40)], pool[(pivot * 40)], pool[(pivot * 40)], factor, 32);
                AscendC::PipeBarrier<PIPE_V>();
                pool.SetValue((((int64_t)pivot) + (int64_t)1416), 0.000000e+00f);
                AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
                tl::ascend::pv_brcb_rows(pool[640], pool[1416], pool[640], 16, 40);
                AscendC::PipeBarrier<PIPE_V>();
                tl::ascend::pv_gj_update(pool[0], pool[(pivot * 40)], pool[640], 32, dimension, 40);
                AscendC::PipeBarrier<PIPE_V>();
              }
              if (valid_1 != 0) {
                tl::ascend::pv_fill(pool[640], pool[640], pool[640], 0.000000e+00f, count_1);
                AscendC::PipeBarrier<PIPE_V>();
                tl::ascend::pv_square_padded_rows(pool[640], pool[16], pool[640], 16, dimension, 40);
                AscendC::PipeBarrier<PIPE_V>();
                tl::ascend::pv_reduce_sum(pool[1368], pool[640], pool[1336], count_1);
                AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
                float inverse_total_1 = pool.GetValue(1368);
                pool.SetValue(1368, total_1);
                AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
                tl::ascend::pv_sqrt(pool[1368], pool[1368], pool[1368], 1);
                AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
                float cutoff_1 = max((pool.GetValue(1368) * rtol), (atol / power_1));
                if ((inverse_total_1 <= 0.000000e+00f) || (8.100000e-01f <= ((inverse_total_1 * cutoff_1) * cutoff_1))) {
                  valid_1 = 0;
                }
                if (valid_1 != 0) {
                  tl::ascend::pv_scale(pool[0], pool[0], pool[0], (-1.000000e+00f / power_1), count_1);
                  AscendC::PipeBarrier<PIPE_V>();
                  for (int32_t row = 0; row < dimension; ++row) {
                    shift_2 = ((uint32_t)(row * 4));
                    int32_t target = ((int32_t)((permutation >> ((uint64_t)shift_2)) & (uint64_t)15));
                    tl::ascend::pv_copy(pool[((target * 40) + 640)], pool[(row * 40)], pool[((target * 40) + 640)], 40);
                  }
                  AscendC::PipeBarrier<PIPE_V>();
                  if (dimension == 15) {
                    tl::ascend::pv_pack15(pool[0], pool[656], pool[0], dimension, 40);
                    AscendC::PipeBarrier<PIPE_V>();
                  }
                  AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
                  AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
                  if (dimension == 15) {
                    tl::ascend::pv_store_rows(X[offset], pool[0], X[offset], 1, (dimension * dimension), 0, 0);
                  } else {
                    tl::ascend::pv_store_rows(X[offset], pool[656], X[offset], dimension, dimension, 24, 0);
                  }
                  AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
                  AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
                }
              }
            }
            accepted = valid_1;
          }
        }
        if (accepted == 0) {
          decision = 0;
          tl::ascend::pv_module_resident_accurate(pool[0], pool[0], pool[0], decision, blocks, job, m, n, batch, atol, rtol);
          if (decision == 0) {
            tl::ascend::pv_module_resident_rectangular(pool[0], pool[0], pool[0], decision, blocks, job, m, n, batch, atol, rtol);
          }
        }
      } else {
        tl::ascend::pv_module_rectangular(S[0], S[0], S[0], blocks, job, m, n, batch, atol, rtol);
      }
      job = (job + blocks);
    }
  }
  pipe.Destroy();
}
