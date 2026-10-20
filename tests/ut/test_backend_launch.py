"""Guard against returning collective peer workers before their barriers."""
import unittest
from code_dsl.op_kernel.backend import vector_only_source


def source(parameters,program):
    return '''extern "C" __global__ __aicore__ void linalg_pinv_tilelang_kernel('''+parameters+''') {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
  AscendC::TPipe pipe;
  AscendC::TBuf<AscendC::TPosition::VECCALC> ascend_ub; pipe.InitBuffer(ascend_ub, 64);
  auto pool = ascend_ub.GetWithOffset<float>(8, 0);
  auto bid = AscendC::GetBlockIdx();
  pipe.Destroy();
'''+program+'''
}
void linalg_pinv_tilelang_kernel_tiling() {}
'''


class LaunchSafety(unittest.TestCase):
    def test_collective_peer_workers_do_not_exit(self):
        program='''  tl::ascend::pv_module_bind(A[0], X[0], W[0]);
  tl::ascend::pv_sync_all(W[0], W[0], W[0]);'''
        result=vector_only_source(source('int32_t blocks, int32_t n, int32_t batch',program),paired=True)
        self.assertNotIn('>= batch) return;',result)
        self.assertIn('tl::ascend::pv_sync_all(',result)

    def test_independent_jobs_exit_before_allocator(self):
        program='''  tl::ascend::pv_module_bind(A[0], X[0], W[0]);
  for (int32_t iteration = 0; iteration < (batch+blocks-1)/blocks; ++iteration) {
    job = (bid + (iteration * blocks));
    if (job < batch) { work(job); }
  }'''
        result=vector_only_source(source('int32_t blocks, int32_t m, int32_t n, int32_t batch',program))
        self.assertLess(result.index('>= batch) return;'),result.index('AscendC::TPipe pipe;'))
        self.assertIn('job = (bid + (iteration * blocks));',result)

    def test_division_free_job_loop_keeps_inactive_guard(self):
        program='''  tl::ascend::pv_module_bind(A[0], X[0], W[0]);
  job = bid;
  while (1) {
    if (!((job < batch))) { break; }
    work(job);
    job = (job + blocks);
  }'''
        result=vector_only_source(source('int32_t blocks, int32_t m, int32_t n, int32_t batch',program))
        self.assertLess(result.index('>= batch) return;'),result.index('AscendC::TPipe pipe;'))
        self.assertIn('job = (job + blocks);',result)


    def test_flat_metadata_precedes_sdk_include(self):
        program='  work();'
        result=vector_only_source('#include "kernel_operator.h"\n'+source('int32_t blocks',program))
        self.assertLess(result.index('#define K_MAX_SHAPE_DIM 0'),result.index('#include "kernel_operator.h"'))

    def test_consumed_shape_metadata_is_not_discarded(self):
        with self.assertRaisesRegex(ValueError,'consumed shape metadata'):
            vector_only_source(source('int32_t blocks','  work.GetShapeInfo();'))


if __name__=='__main__':unittest.main()
