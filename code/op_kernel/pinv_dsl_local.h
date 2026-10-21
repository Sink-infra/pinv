#pragma once
#include "pinv_dsl_common.h"
__aicore__ inline void PvStageLocal( GM_ADDR A_handle,  GM_ADDR X_handle,  GM_ADDR W_handle,  GM_ADDR S_handle, int32_t blocks, int32_t m, int32_t n, int32_t batch, float atol, float rtol, uint64_t fftsAddr) {
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

  AscendC::TBuf<AscendC::TPosition::VECCALC> ascend_ub; pipe.InitBuffer(ascend_ub, 77344);
  auto bid = AscendC::GetBlockIdx();
  if ASCEND_IS_AIV {
    }
  int job = 0;
  int offset = 0;
  int dimension = 0;
  int accepted = 0;
  auto accepted_slot = ascend_ub.GetWithOffset<int>(8, 0);
  int count = 0;
  int dma_pad = 0;
  int valid = 0;
  auto pool = ascend_ub.GetWithOffset<float>(19136, 32);
  float power = 0.000000e+00f;
  uint32_t power_word = (uint32_t)0;
  auto indices = ascend_ub.GetWithOffset<uint32_t>(64, 76576);
  auto order = ascend_ub.GetWithOffset<uint32_t>(128, 76832);
  int count_1 = 0;
  int dma_pad_1 = 0;
  int valid_1 = 0;
  float power_1 = 0.000000e+00f;
  uint32_t power_word_1 = (uint32_t)0;
  int count_2 = 0;
  int dma_pad_2 = 0;
  int valid_2 = 0;
  float power_2 = 0.000000e+00f;
  uint32_t power_word_2 = (uint32_t)0;
  int decision = 0;
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
  float best_2 = 0.000000e+00f;
  int pivot_2 = 0;
  int consumed_2 = 0;
  int base_2 = 0;
  int active_2 = 0;
  float best_3 = 0.000000e+00f;
  int pivot_3 = 0;
  int consumed_3 = 0;
  int base_3 = 0;
  int active_3 = 0;
  float best_4 = 0.000000e+00f;
  int pivot_4 = 0;
  int consumed_4 = 0;
  int base_4 = 0;
  int active_4 = 0;
  float best_5 = 0.000000e+00f;
  int pivot_5 = 0;
  int consumed_5 = 0;
  int base_5 = 0;
  int active_5 = 0;
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIV {
    tl::ascend::pv_module_bind(A[0], X[0], W[0]);
    job = bid;
    while (1) {
      if (!((job < batch))) { break; }
      offset = ((job * m) * n);
      dimension = n;
      accepted = 0;
      accepted_slot.SetValue(0, 0);
      if ((m == n) && (n <= 64)) {
        if (33 <= n) {
          if (dimension <= 56) {
            int32_t n_1 = dimension;
            count = (n_1 * 120);
            dma_pad = (((n_1 + 7) / 8) * 8);
            valid = 1;
            tl::ascend::pv_fill(pool[0], pool[0], pool[0], 0.000000e+00f, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
            tl::ascend::pv_load_rows(pool[0], A[offset], pool[0], n_1, n_1, 0, (120 - dma_pad), (0 < (n_1 % 8)));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
            tl::ascend::pv_abs(pool[8704], pool[0], pool[8704], count);
            AscendC::PipeBarrier<PIPE_V>();
            tl::ascend::pv_reduce_max(pool[19008], pool[8704], pool[18496], count);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
            float magnitude = pool.GetValue(19008);
            if ((magnitude < 1.000000e-01f) || (1.701412e+38f <= magnitude)) {
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
              for (int32_t i = 0; i < 56; ++i) {
                indices.SetValue(i, ((uint32_t)(i * 480)));
                if (n_1 <= i) {
                  order.SetValue(i, (uint32_t)0);
                  order.SetValue((i + 56), (uint32_t)0);
                }
              }
              tl::ascend::pv_fill(pool[19040], pool[19040], pool[19040], 0.000000e+00f, 56);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_mask_begin(pool[0], pool[0], pool[0], n_1);
              AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
              AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
              for (int32_t pivot_tile = 0; pivot_tile < n_1; ++pivot_tile) {
                tl::ascend::pv_gather(pool[19040], pool[0], indices[0], (pivot_tile * 4), n_1);
                AscendC::PipeBarrier<PIPE_V>();
                tl::ascend::pv_abs(pool[18432], pool[19040], pool[18432], 56);
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
                order.SetValue((((int64_t)pivot_tile) + (int64_t)56), ((uint32_t)((pivot * 480) + 224)));
                consumed = 0;
                base = ((pivot_tile / 8) * 8);
                float divisor = pool.GetValue((((int64_t)pivot) + (int64_t)19040));
                pool.SetValue((((int64_t)pivot) + (int64_t)19040), 0.000000e+00f);
                pool.SetValue((((((int64_t)pivot) * (int64_t)120) + ((int64_t)pivot_tile)) + (int64_t)56), 1.000000e+00f);
                AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
                active = 64;
                if (consumed != 0) {
                  active = (active - 8);
                }
                tl::ascend::pv_scale(pool[((pivot * 120) + base)], pool[((pivot * 120) + base)], pool[((pivot * 120) + base)], (-1.000000e+00f / divisor), active);
                tl::ascend::pv_brcb_rows(pool[17408], pool[19040], pool[17408], 56, 8);
                AscendC::PipeBarrier<PIPE_V>();
                tl::ascend::pv_mad_strided(pool[base], pool[((pivot * 120) + base)], pool[17408], 64, n_1, 1, 1, 0, 15, 0, 1);
                AscendC::PipeBarrier<PIPE_V>();
              }
              if (valid != 0) {
                tl::ascend::pv_fill(pool[8704], pool[8704], pool[8704], 0.000000e+00f, count);
                AscendC::PipeBarrier<PIPE_V>();
                tl::ascend::pv_square_padded_rows(pool[8704], pool[56], pool[8704], 56, n_1, 120);
                AscendC::PipeBarrier<PIPE_V>();
                tl::ascend::pv_reduce_sum(pool[19008], pool[8704], pool[18496], count);
                AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
                float inverse_total = pool.GetValue(19008);
                pool.SetValue(19008, total);
                AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
                tl::ascend::pv_sqrt(pool[19008], pool[19008], pool[19008], 1);
                AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
                float cutoff = max((pool.GetValue(19008) * rtol), (atol / power));
                if ((inverse_total <= 0.000000e+00f) || (8.100000e-01f <= ((inverse_total * cutoff) * cutoff))) {
                  valid = 0;
                } else {
                  tl::ascend::pv_scale(pool[0], pool[0], pool[0], (-1.000000e+00f / power), count);
                  AscendC::PipeBarrier<PIPE_V>();
                  AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
                  tl::ascend::bl_index_brcb(pool[17408], order[56], pool[17408], 56);
                  AscendC::PipeBarrier<PIPE_V>();
                  tl::ascend::bl_index_matrix(pool[11840], pool[17408], order[0], 56, 56);
                  AscendC::PipeBarrier<PIPE_V>();
                  tl::ascend::bl_gather(pool[8704], pool[0], pool[11840], 3136);
                  AscendC::PipeBarrier<PIPE_V>();
                  AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
                  tl::ascend::pv_store_rows(X[offset], pool[8704], X[offset], n_1, n_1, (56 - dma_pad), 0);
                  AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
                  AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
                }
              }
            }
            accepted_slot.SetValue(0, valid);
          } else {
            if (dimension == 64) {
              count_1 = 8704;
              dma_pad_1 = 64;
              valid_1 = 1;
              tl::ascend::pv_fill(pool[0], pool[0], pool[0], 0.000000e+00f, count_1);
              AscendC::PipeBarrier<PIPE_V>();
              AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
              AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
              tl::ascend::pv_load_rows(pool[0], A[offset], pool[0], 64, 64, 0, (136 - dma_pad_1), (bool)0);
              AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
              AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
              tl::ascend::pv_abs(pool[8704], pool[0], pool[8704], count_1);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_reduce_max(pool[19008], pool[8704], pool[18496], count_1);
              AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
              AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
              float magnitude_1 = pool.GetValue(19008);
              if ((magnitude_1 < 1.000000e-01f) || (1.701412e+38f <= magnitude_1)) {
                valid_1 = 0;
              }
              power_1 = 1.000000e+00f;
              if (valid_1 != 0) {
                power_word_1 = (((*(uint32_t *)(&(magnitude_1))) + (uint32_t)8388607) & (uint32_t)2139095040);
                power_1 = (*(float *)(&(power_word_1)));
                tl::ascend::pv_scale(pool[0], pool[0], pool[0], (1.000000e+00f / power_1), count_1);
                AscendC::PipeBarrier<PIPE_V>();
                tl::ascend::pv_mul(pool[8704], pool[0], pool[0], count_1);
                AscendC::PipeBarrier<PIPE_V>();
                tl::ascend::pv_reduce_sum(pool[19008], pool[8704], pool[18496], count_1);
                AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
                float total_1 = pool.GetValue(19008);
                for (int32_t i_1 = 0; i_1 < 64; ++i_1) {
                  indices.SetValue(i_1, ((uint32_t)(i_1 * 544)));
                }
                tl::ascend::pv_fill(pool[19040], pool[19040], pool[19040], 0.000000e+00f, 64);
                AscendC::PipeBarrier<PIPE_V>();
                tl::ascend::pv_mask_begin(pool[0], pool[0], pool[0], 64);
                AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
                for (int32_t pivot_tile_1 = 0; pivot_tile_1 < 64; ++pivot_tile_1) {
                  tl::ascend::pv_gather(pool[19040], pool[0], indices[0], (pivot_tile_1 * 4), 64);
                  AscendC::PipeBarrier<PIPE_V>();
                  tl::ascend::pv_abs(pool[18432], pool[19040], pool[18432], 64);
                  AscendC::PipeBarrier<PIPE_V>();
                  tl::ascend::pv_mask_reduce(pool[19008], pool[18432], pool[19008]);
                  AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
                  tl::ascend::pv_reduce_scalars(pool[0], pool[0], pool[0], best_2, pivot_2);
                  if (best_2 <= 0.000000e+00f) {
                    valid_1 = 0;
                    break;
                  }
                  tl::ascend::pv_mask_remove(pool[0], pool[0], pool[0], pivot_2);
                  order.SetValue(pivot_2, ((uint32_t)(pivot_tile_1 * 4)));
                  order.SetValue((pivot_tile_1 + 64), ((uint32_t)((pivot_2 * 544) + 256)));
                  consumed_2 = 0;
                  base_2 = ((pivot_tile_1 / 8) * 8);
                  if (((pivot_tile_1 % 8) == 7) || (pivot_tile_1 == 63)) {
                    consumed_2 = 1;
                    base_2 = (base_2 + 8);
                  }
                  float divisor_1 = pool.GetValue((((int64_t)pivot_2) + (int64_t)19040));
                  pool.SetValue((((int64_t)pivot_2) + (int64_t)19040), 0.000000e+00f);
                  pool.SetValue((((((int64_t)pivot_2) * (int64_t)136) + ((int64_t)pivot_tile_1)) + (int64_t)64), 1.000000e+00f);
                  AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
                  active_2 = 72;
                  if (consumed_2 != 0) {
                    active_2 = (active_2 - 8);
                  }
                  tl::ascend::pv_scale(pool[((pivot_2 * 136) + base_2)], pool[((pivot_2 * 136) + base_2)], pool[((pivot_2 * 136) + base_2)], (-1.000000e+00f / divisor_1), active_2);
                  tl::ascend::pv_brcb_rows(pool[17408], pool[19040], pool[17408], 64, 8);
                  AscendC::PipeBarrier<PIPE_V>();
                  tl::ascend::pv_mad_strided(pool[base_2], pool[((pivot_2 * 136) + base_2)], pool[17408], 64, 64, 1, 1, 0, 17, 0, 1);
                  if (consumed_2 == 0) {
                    tl::ascend::pv_mad_strided(pool[(base_2 + 64)], pool[(((pivot_2 * 136) + base_2) + 64)], pool[17408], 64, 8, 17, 0, 1, 136, 0, 8);
                  }
                  AscendC::PipeBarrier<PIPE_V>();
                }
                if (valid_1 != 0) {
                  tl::ascend::pv_fill(pool[8704], pool[8704], pool[8704], 0.000000e+00f, count_1);
                  AscendC::PipeBarrier<PIPE_V>();
                  tl::ascend::pv_square_padded_rows(pool[8704], pool[64], pool[8704], 64, 64, 136);
                  AscendC::PipeBarrier<PIPE_V>();
                  tl::ascend::pv_reduce_sum(pool[19008], pool[8704], pool[18496], count_1);
                  AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
                  float inverse_total_1 = pool.GetValue(19008);
                  pool.SetValue(19008, total_1);
                  AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
                  tl::ascend::pv_sqrt(pool[19008], pool[19008], pool[19008], 1);
                  AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
                  float cutoff_1 = max((pool.GetValue(19008) * rtol), (atol / power_1));
                  if ((inverse_total_1 <= 0.000000e+00f) || (8.100000e-01f <= ((inverse_total_1 * cutoff_1) * cutoff_1))) {
                    valid_1 = 0;
                  } else {
                    tl::ascend::pv_scale(pool[0], pool[0], pool[0], (-1.000000e+00f / power_1), count_1);
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
                    tl::ascend::pv_store_rows(X[offset], pool[8704], X[offset], 64, 64, (64 - dma_pad_1), 0);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
                  }
                }
              }
              accepted_slot.SetValue(0, valid_1);
            } else {
              int32_t n_2 = dimension;
              count_2 = (n_2 * 136);
              dma_pad_2 = (((n_2 + 7) / 8) * 8);
              valid_2 = 1;
              tl::ascend::pv_fill(pool[0], pool[0], pool[0], 0.000000e+00f, count_2);
              AscendC::PipeBarrier<PIPE_V>();
              AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
              AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
              tl::ascend::pv_load_rows(pool[0], A[offset], pool[0], n_2, n_2, 0, (136 - dma_pad_2), (0 < (n_2 % 8)));
              AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
              AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
              tl::ascend::pv_abs(pool[8704], pool[0], pool[8704], count_2);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::pv_reduce_max(pool[19008], pool[8704], pool[18496], count_2);
              AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
              AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
              float magnitude_2 = pool.GetValue(19008);
              if ((magnitude_2 < 1.000000e-01f) || (1.701412e+38f <= magnitude_2)) {
                valid_2 = 0;
              }
              power_2 = 1.000000e+00f;
              if (valid_2 != 0) {
                power_word_2 = (((*(uint32_t *)(&(magnitude_2))) + (uint32_t)8388607) & (uint32_t)2139095040);
                power_2 = (*(float *)(&(power_word_2)));
                tl::ascend::pv_scale(pool[0], pool[0], pool[0], (1.000000e+00f / power_2), count_2);
                AscendC::PipeBarrier<PIPE_V>();
                tl::ascend::pv_mul(pool[8704], pool[0], pool[0], count_2);
                AscendC::PipeBarrier<PIPE_V>();
                tl::ascend::pv_reduce_sum(pool[19008], pool[8704], pool[18496], count_2);
                AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
                float total_2 = pool.GetValue(19008);
                for (int32_t i_2 = 0; i_2 < 64; ++i_2) {
                  indices.SetValue(i_2, ((uint32_t)(i_2 * 544)));
                  if (n_2 <= i_2) {
                    order.SetValue(i_2, (uint32_t)0);
                    order.SetValue((i_2 + 64), (uint32_t)0);
                  }
                }
                tl::ascend::pv_fill(pool[19040], pool[19040], pool[19040], 0.000000e+00f, 64);
                AscendC::PipeBarrier<PIPE_V>();
                tl::ascend::pv_mask_begin(pool[0], pool[0], pool[0], n_2);
                AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
                for (int32_t pivot_tile_2 = 0; pivot_tile_2 < n_2; ++pivot_tile_2) {
                  tl::ascend::pv_gather(pool[19040], pool[0], indices[0], (pivot_tile_2 * 4), n_2);
                  AscendC::PipeBarrier<PIPE_V>();
                  tl::ascend::pv_abs(pool[18432], pool[19040], pool[18432], 64);
                  AscendC::PipeBarrier<PIPE_V>();
                  tl::ascend::pv_mask_reduce(pool[19008], pool[18432], pool[19008]);
                  AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
                  tl::ascend::pv_reduce_scalars(pool[0], pool[0], pool[0], best_4, pivot_4);
                  if (best_4 <= 0.000000e+00f) {
                    valid_2 = 0;
                    break;
                  }
                  tl::ascend::pv_mask_remove(pool[0], pool[0], pool[0], pivot_4);
                  order.SetValue(pivot_4, ((uint32_t)(pivot_tile_2 * 4)));
                  order.SetValue((((int64_t)pivot_tile_2) + (int64_t)64), ((uint32_t)((pivot_4 * 544) + 256)));
                  consumed_4 = 0;
                  base_4 = ((pivot_tile_2 / 8) * 8);
                  if (((pivot_tile_2 % 8) == 7) || ((pivot_tile_2 + 1) == n_2)) {
                    consumed_4 = 1;
                    base_4 = (base_4 + 8);
                  }
                  float divisor_2 = pool.GetValue((((int64_t)pivot_4) + (int64_t)19040));
                  pool.SetValue((((int64_t)pivot_4) + (int64_t)19040), 0.000000e+00f);
                  pool.SetValue((((((int64_t)pivot_4) * (int64_t)136) + ((int64_t)pivot_tile_2)) + (int64_t)64), 1.000000e+00f);
                  AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
                  active_4 = 72;
                  if (consumed_4 != 0) {
                    active_4 = (active_4 - 8);
                  }
                  tl::ascend::pv_scale(pool[((pivot_4 * 136) + base_4)], pool[((pivot_4 * 136) + base_4)], pool[((pivot_4 * 136) + base_4)], (-1.000000e+00f / divisor_2), active_4);
                  tl::ascend::pv_brcb_rows(pool[17408], pool[19040], pool[17408], 64, 8);
                  AscendC::PipeBarrier<PIPE_V>();
                  tl::ascend::pv_mad_strided(pool[base_4], pool[((pivot_4 * 136) + base_4)], pool[17408], 64, n_2, 1, 1, 0, 17, 0, 1);
                  if (consumed_4 == 0) {
                    if (7 < n_2) {
                      tl::ascend::pv_mad_strided(pool[(base_4 + 64)], pool[(((pivot_4 * 136) + base_4) + 64)], pool[17408], 64, (n_2 / 8), 17, 0, 1, 136, 0, 8);
                    }
                    if (0 < (n_2 % 8)) {
                      tl::ascend::pv_mad_strided(pool[((((n_2 / 8) * 1088) + base_4) + 64)], pool[(((pivot_4 * 136) + base_4) + 64)], pool[(((n_2 / 8) * 64) + 17408)], 8, (n_2 % 8), 1, 1, 0, 17, 0, 1);
                    }
                  }
                  AscendC::PipeBarrier<PIPE_V>();
                }
                if (valid_2 != 0) {
                  tl::ascend::pv_fill(pool[8704], pool[8704], pool[8704], 0.000000e+00f, count_2);
                  AscendC::PipeBarrier<PIPE_V>();
                  tl::ascend::pv_square_padded_rows(pool[8704], pool[64], pool[8704], 64, n_2, 136);
                  AscendC::PipeBarrier<PIPE_V>();
                  tl::ascend::pv_reduce_sum(pool[19008], pool[8704], pool[18496], count_2);
                  AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
                  float inverse_total_2 = pool.GetValue(19008);
                  pool.SetValue(19008, total_2);
                  AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
                  tl::ascend::pv_sqrt(pool[19008], pool[19008], pool[19008], 1);
                  AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
                  AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
                  float cutoff_2 = max((pool.GetValue(19008) * rtol), (atol / power_2));
                  if ((inverse_total_2 <= 0.000000e+00f) || (8.100000e-01f <= ((inverse_total_2 * cutoff_2) * cutoff_2))) {
                    valid_2 = 0;
                  } else {
                    tl::ascend::pv_scale(pool[0], pool[0], pool[0], (-1.000000e+00f / power_2), count_2);
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
                    tl::ascend::pv_store_rows(X[offset], pool[8704], X[offset], n_2, n_2, (64 - dma_pad_2), 0);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
                  }
                }
              }
              accepted_slot.SetValue(0, valid_2);
            }
          }
          accepted = accepted_slot.GetValue(0);
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
