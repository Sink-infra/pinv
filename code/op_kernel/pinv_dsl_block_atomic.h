#pragma once
#include "pinv_dsl_common.h"
#pragma once
namespace tl::ascend {
template<class D,class A,class B> __aicore__ inline void ba_atomic_on(D,A,B){AscendC::SetAtomicAdd<float>();}
template<class D,class A,class B> __aicore__ inline void ba_atomic_off(D,A,B){AscendC::DisableDmaAtomic();}
}

namespace tl::ascend {
__aicore__ inline void BlockInplacePanel( GM_ADDR D_handle,  GM_ADDR P_handle, int32_t blocks, int32_t start_row,  GM_ADDR shape_hint, int64_t n, int64_t batch, uint64_t fftsAddr) {
  AscendC::InitSocState();

  AscendC::GlobalTensor<float> D;
  D.SetGlobalBuffer((__gm__ float*)D_handle);
  AscendC::GlobalTensor<float> P;
  P.SetGlobalBuffer((__gm__ float*)P_handle);
  AscendC::GlobalTensor<uint8_t> ShapeHint;
  ShapeHint.SetGlobalBuffer((__gm__ uint8_t*)shape_hint);

  AscendC::LocalTensor<float> ascend_l0a(AscendC::TPosition::A2, 0, 16384);
  AscendC::LocalTensor<float> ascend_l0b(AscendC::TPosition::B2, 0, 16384);



  auto bid = AscendC::GetBlockIdx();
  if ASCEND_IS_AIV {
    }
  AscendC::LocalTensor<float> a(AscendC::TPosition::A1, 0, 4096);
  AscendC::LocalTensor<float> b(AscendC::TPosition::A1, 16384, 8192);
  AscendC::LocalTensor<float> c(AscendC::TPosition::CO1, 0, 8192);
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIC {
    for (int32_t iteration = 0; iteration < ((((batch * (n / 128)) + blocks) - 1) / blocks); ++iteration) {
      if ((bid + (iteration * blocks)) < (batch * (n / 128))) {
        tl::ascend::br_load_a_shape(a[0], D[(((bid + (iteration * blocks)) / (n / 128)) * 4096)], a[0], 64, 64, 64);
        tl::ascend::copy_gm_to_l1<float, 64, 128>(b[0], P[((((bid + (iteration * blocks)) % (n / 128)) * 128) + (((((bid + (iteration * blocks)) / (n / 128)) * n) + start_row) * n))], n, ((64 <= ((n * (batch - ((bid + (iteration * blocks)) / (n / 128)))) - start_row)) ? 64 : ((0 < ((n * (batch - ((bid + (iteration * blocks)) / (n / 128)))) - start_row)) ? ((n * (batch - ((bid + (iteration * blocks)) / (n / 128)))) - start_row) : 0)), ((1 <= ((n / 128) - ((bid + (iteration * blocks)) % (n / 128)))) ? 128 : ((0 < (n - (((bid + (iteration * blocks)) % (n / 128)) * 128))) ? (n - (((bid + (iteration * blocks)) % (n / 128)) * 128)) : 0)));
        AscendC::PipeBarrier<PIPE_ALL>();
        tl::ascend::gemm_v0<float, float, 64, 128, 64, false, false>(a[0], b[0], c[0], ascend_l0a, ascend_l0b, (bool)1);
        AscendC::PipeBarrier<PIPE_ALL>();
        tl::ascend::copy_l0c_to_gm<float, float, layout::RowMajor, 64, 128, 0>(P[((((bid + (iteration * blocks)) % (n / 128)) * 128) + (((((bid + (iteration * blocks)) / (n / 128)) * n) + start_row) * n))], c[0], n, ((64 <= ((n * (batch - ((bid + (iteration * blocks)) / (n / 128)))) - start_row)) ? 64 : ((0 < ((n * (batch - ((bid + (iteration * blocks)) / (n / 128)))) - start_row)) ? ((n * (batch - ((bid + (iteration * blocks)) / (n / 128)))) - start_row) : 0)), ((1 <= ((n / 128) - ((bid + (iteration * blocks)) % (n / 128)))) ? 128 : ((0 < (n - (((bid + (iteration * blocks)) % (n / 128)) * 128))) ? (n - (((bid + (iteration * blocks)) % (n / 128)) * 128)) : 0)));
        AscendC::PipeBarrier<PIPE_ALL>();
      }
    }
  }

}

}
namespace tl::ascend {
__aicore__ inline void BlockAtomicRows( GM_ADDR F_handle,  GM_ADDR P_handle, int32_t blocks, int32_t start_row,  GM_ADDR shape_hint, int64_t n, int64_t batch, uint64_t fftsAddr) {
  AscendC::InitSocState();

  AscendC::GlobalTensor<float> F;
  F.SetGlobalBuffer((__gm__ float*)F_handle);
  AscendC::GlobalTensor<float> P;
  P.SetGlobalBuffer((__gm__ float*)P_handle);
  AscendC::GlobalTensor<uint8_t> ShapeHint;
  ShapeHint.SetGlobalBuffer((__gm__ uint8_t*)shape_hint);

  AscendC::LocalTensor<float> ascend_l0a(AscendC::TPosition::A2, 0, 16384);
  AscendC::LocalTensor<float> ascend_l0b(AscendC::TPosition::B2, 0, 16384);



  auto bid = AscendC::GetBlockIdx();
  if ASCEND_IS_AIV {
    }
  AscendC::LocalTensor<float> a(AscendC::TPosition::A1, 0, 4096);
  AscendC::LocalTensor<float> b(AscendC::TPosition::A1, 16384, 16384);
  AscendC::LocalTensor<float> c(AscendC::TPosition::CO1, 0, 16384);
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIC {
    for (int32_t iteration = 0; iteration < (((((batch * (n / 64)) * (n / 256)) + blocks) - 1) / blocks); ++iteration) {
      if ((bid + (iteration * blocks)) < ((batch * (n / 64)) * (n / 256))) {
        if (((((bid + (iteration * blocks)) / (n / 256)) % (n / 64)) * 64) != start_row) {
          tl::ascend::br_load_a_shape(a[0], F[(((((bid + (iteration * blocks)) / (n / 256)) % (n / 64)) * 4096) + ((((bid + (iteration * blocks)) / ((n / 64) * (n / 256))) * n) * 64))], a[0], 64, 64, 64);
          tl::ascend::copy_gm_to_l1<float, 64, 256>(b[0], P[((((bid + (iteration * blocks)) % (n / 256)) * 256) + (((((bid + (iteration * blocks)) / ((n / 64) * (n / 256))) * n) + start_row) * n))], n, ((64 <= ((n * (batch - ((bid + (iteration * blocks)) / ((n / 64) * (n / 256))))) - start_row)) ? 64 : ((0 < ((n * (batch - ((bid + (iteration * blocks)) / ((n / 64) * (n / 256))))) - start_row)) ? ((n * (batch - ((bid + (iteration * blocks)) / ((n / 64) * (n / 256))))) - start_row) : 0)), ((1 <= ((n / 256) - ((bid + (iteration * blocks)) % (n / 256)))) ? 256 : ((0 < (n - (((bid + (iteration * blocks)) % (n / 256)) * 256))) ? (n - (((bid + (iteration * blocks)) % (n / 256)) * 256)) : 0)));
          AscendC::PipeBarrier<PIPE_ALL>();
          tl::ascend::gemm_v0<float, float, 64, 256, 64, false, false>(a[0], b[0], c[0], ascend_l0a, ascend_l0b, (bool)1);
          AscendC::PipeBarrier<PIPE_ALL>();
          tl::ascend::ba_atomic_on(P[0], P[0], P[0]);
          tl::ascend::copy_l0c_to_gm<float, float, layout::RowMajor, 64, 256, 0>(P[((((bid + (iteration * blocks)) % (n / 256)) * 256) + ((((((bid + (iteration * blocks)) / (n / 256)) % (n / 64)) * 64) + (((bid + (iteration * blocks)) / ((n / 64) * (n / 256))) * n)) * n))], c[0], n, ((1 <= (((n * (batch - ((bid + (iteration * blocks)) / ((n / 64) * (n / 256))))) / 64) - (((bid + (iteration * blocks)) / (n / 256)) % (n / 64)))) ? 64 : ((0 < ((n * (batch - ((bid + (iteration * blocks)) / ((n / 64) * (n / 256))))) - ((((bid + (iteration * blocks)) / (n / 256)) % (n / 64)) * 64))) ? ((n * (batch - ((bid + (iteration * blocks)) / ((n / 64) * (n / 256))))) - ((((bid + (iteration * blocks)) / (n / 256)) % (n / 64)) * 64)) : 0)), ((1 <= ((n / 256) - ((bid + (iteration * blocks)) % (n / 256)))) ? 256 : ((0 < (n - (((bid + (iteration * blocks)) % (n / 256)) * 256))) ? (n - (((bid + (iteration * blocks)) % (n / 256)) * 256)) : 0)));
          AscendC::PipeBarrier<PIPE_ALL>();
          tl::ascend::ba_atomic_off(P[0], P[0], P[0]);
        }
      }
    }
  }

}

}
#define br_inplace_panel(d,a,b,oa,op,bl,ba,n,st) BlockInplacePanel((GM_ADDR)(d.GetPhyAddr()+oa),(GM_ADDR)(d.GetPhyAddr()+op),bl,st,(GM_ADDR)d.GetPhyAddr(),n,ba,fftsAddr)
#define br_atomic_rows(d,a,b,oa,op,bl,ba,n,st) BlockAtomicRows((GM_ADDR)(d.GetPhyAddr()+oa),(GM_ADDR)(d.GetPhyAddr()+op),bl,st,(GM_ADDR)d.GetPhyAddr(),n,ba,fftsAddr)
namespace tl::ascend {
template<class D,class S,class X> __aicore__ inline void bc_store(D d,S s,X,int m,int n,int stride){AscendC::FixpipeParamsV220 p;p.nSize=n;p.mSize=m;p.srcStride=m;p.dstStride=stride;p.ndNum=1;p.srcNdStride=0;p.dstNdStride=0;AscendC::Fixpipe<float,float,AscendC::CFG_ROW_MAJOR>(d,s,p);}
}

namespace tl::ascend {
__aicore__ inline void BlockChainRows( GM_ADDR F_handle,  GM_ADDR D_handle,  GM_ADDR P_handle,  GM_ADDR Tmp_handle,  GM_ADDR Y_handle, int32_t blocks, int32_t start_row,  GM_ADDR shape_hint, int64_t n, int64_t batch, uint64_t fftsAddr) {
  AscendC::InitSocState();

  AscendC::GlobalTensor<float> F;
  F.SetGlobalBuffer((__gm__ float*)F_handle);
  AscendC::GlobalTensor<float> D;
  D.SetGlobalBuffer((__gm__ float*)D_handle);
  AscendC::GlobalTensor<float> P;
  P.SetGlobalBuffer((__gm__ float*)P_handle);
  AscendC::GlobalTensor<float> Tmp;
  Tmp.SetGlobalBuffer((__gm__ float*)Tmp_handle);
  AscendC::GlobalTensor<float> Y;
  Y.SetGlobalBuffer((__gm__ float*)Y_handle);
  AscendC::GlobalTensor<uint8_t> ShapeHint;
  ShapeHint.SetGlobalBuffer((__gm__ uint8_t*)shape_hint);

  AscendC::LocalTensor<float> ascend_l0a(AscendC::TPosition::A2, 0, 16384);
  AscendC::LocalTensor<float> ascend_l0b(AscendC::TPosition::B2, 0, 16384);



  auto bid = AscendC::GetBlockIdx();
  if ASCEND_IS_AIV {
    }
  AscendC::LocalTensor<float> f(AscendC::TPosition::A1, 0, 4096);
  AscendC::LocalTensor<float> inv(AscendC::TPosition::A1, 16384, 4096);
  AscendC::LocalTensor<float> c1(AscendC::TPosition::CO1, 0, 4096);
  AscendC::LocalTensor<float> a(AscendC::TPosition::A1, 32768, 4096);
  AscendC::LocalTensor<float> p(AscendC::TPosition::A1, 49152, 16384);
  AscendC::LocalTensor<float> c2(AscendC::TPosition::CO1, 16384, 16384);
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIC {
    for (int32_t iteration = 0; iteration < (((((batch * (n / 64)) * (n / 256)) + blocks) - 1) / blocks); ++iteration) {
      if ((bid + (iteration * blocks)) < ((batch * (n / 64)) * (n / 256))) {
        if (((((bid + (iteration * blocks)) / (n / 256)) % (n / 64)) * 64) != start_row) {
          tl::ascend::br_load_a_shape(f[0], F[(((((bid + (iteration * blocks)) / (n / 256)) % (n / 64)) * 4096) + ((((bid + (iteration * blocks)) / ((n / 64) * (n / 256))) * n) * 64))], f[0], 64, 64, 64);
          tl::ascend::copy_gm_to_l1<float, 64, 64>(inv[0], D[(((bid + (iteration * blocks)) / ((n / 64) * (n / 256))) * 4096)], 64, ((1 <= (batch - ((bid + (iteration * blocks)) / ((n / 64) * (n / 256))))) ? 64 : (((((bid + (iteration * blocks)) / ((n / 64) * (n / 256))) - batch) < 0) ? ((batch * 64) - (((bid + (iteration * blocks)) / ((n / 64) * (n / 256))) * 64)) : 0)), 64);
          AscendC::PipeBarrier<PIPE_ALL>();
          tl::ascend::gemm_v0<float, float, 64, 64, 64, false, false>(f[0], inv[0], c1[0], ascend_l0a, ascend_l0b, (bool)1);
          AscendC::PipeBarrier<PIPE_ALL>();
          tl::ascend::bc_store(Tmp[(((((bid + (iteration * blocks)) / (n / 256)) % (n / 64)) * 4096) + ((((bid + (iteration * blocks)) / ((n / 64) * (n / 256))) * n) * 64))], c1[0], Tmp[(((((bid + (iteration * blocks)) / (n / 256)) % (n / 64)) * 4096) + ((((bid + (iteration * blocks)) / ((n / 64) * (n / 256))) * n) * 64))], 64, 64, 64);
          AscendC::PipeBarrier<PIPE_ALL>();
          tl::ascend::br_load_a_shape(a[0], Tmp[(((((bid + (iteration * blocks)) / (n / 256)) % (n / 64)) * 4096) + ((((bid + (iteration * blocks)) / ((n / 64) * (n / 256))) * n) * 64))], a[0], 64, 64, 64);
        } else {
          tl::ascend::br_load_a_shape(a[0], D[(((bid + (iteration * blocks)) / ((n / 64) * (n / 256))) * 4096)], a[0], 64, 64, 64);
        }
        tl::ascend::copy_gm_to_l1<float, 64, 256>(p[0], P[((((bid + (iteration * blocks)) % (n / 256)) * 256) + ((((bid + (iteration * blocks)) / ((n / 64) * (n / 256))) * n) * 64))], n, ((1 <= (batch - ((bid + (iteration * blocks)) / ((n / 64) * (n / 256))))) ? 64 : (((((bid + (iteration * blocks)) / ((n / 64) * (n / 256))) - batch) < 0) ? ((batch * 64) - (((bid + (iteration * blocks)) / ((n / 64) * (n / 256))) * 64)) : 0)), ((1 <= ((n / 256) - ((bid + (iteration * blocks)) % (n / 256)))) ? 256 : ((0 < (n - (((bid + (iteration * blocks)) % (n / 256)) * 256))) ? (n - (((bid + (iteration * blocks)) % (n / 256)) * 256)) : 0)));
        AscendC::PipeBarrier<PIPE_ALL>();
        tl::ascend::gemm_v0<float, float, 64, 256, 64, false, false>(a[0], p[0], c2[0], ascend_l0a, ascend_l0b, (bool)1);
        AscendC::PipeBarrier<PIPE_ALL>();
        if (((((bid + (iteration * blocks)) / (n / 256)) % (n / 64)) * 64) != start_row) {
          tl::ascend::ba_atomic_on(Y[0], Y[0], Y[0]);
        }
        tl::ascend::bc_store(Y[((((bid + (iteration * blocks)) % (n / 256)) * 256) + ((((((bid + (iteration * blocks)) / (n / 256)) % (n / 64)) * 64) + (((bid + (iteration * blocks)) / ((n / 64) * (n / 256))) * n)) * n))], c2[0], Y[((((bid + (iteration * blocks)) % (n / 256)) * 256) + ((((((bid + (iteration * blocks)) / (n / 256)) % (n / 64)) * 64) + (((bid + (iteration * blocks)) / ((n / 64) * (n / 256))) * n)) * n))], 64, 256, n);
        AscendC::PipeBarrier<PIPE_ALL>();
        if (((((bid + (iteration * blocks)) / (n / 256)) % (n / 64)) * 64) != start_row) {
          tl::ascend::ba_atomic_off(Y[0], Y[0], Y[0]);
        }
      }
    }
  }

}

}
#define br_chain(d,a,b,fo,di,pi,tm,sw,bl,ba,n,st) BlockChainRows((GM_ADDR)(d.GetPhyAddr()+fo),(GM_ADDR)(d.GetPhyAddr()+di),(GM_ADDR)(d.GetPhyAddr()+pi),(GM_ADDR)(d.GetPhyAddr()+tm),(GM_ADDR)(d.GetPhyAddr()+sw),bl,st,(GM_ADDR)d.GetPhyAddr(),n,ba,fftsAddr)
