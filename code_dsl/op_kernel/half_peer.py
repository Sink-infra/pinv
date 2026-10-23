"""Dual approximate pivot selectors with separate FP32 numerical workers."""
from pathlib import Path
from code_dsl.op_kernel.parallel_components import parallel_instruction_source
from code_dsl.op_kernel.half_peer_producer import make_half_peer_macro

def half_peer_kernel():
 import tilelang.language as T
 from code_dsl.op_kernel.intrinsics import emit
 operation=make_half_peer_macro()
 header=parallel_instruction_source()+'\n'+Path(__file__).with_name('half_pivot_instructions.h').read_text()
 @T.prim_func
 def kernel(A:T.Tensor((1,),'float32'),X:T.Tensor((1,),'float32'),W:T.Tensor((1,),'float32'),blocks:T.int32,n:T.int32,stride:T.int32,all_batch:T.int32,job:T.int32,core:T.int32,workers:T.int32,window:T.int32):
  T.func_attr({'global_symbol':'linalg_pinv_tilelang','pinv.component':'half_peer_selector','pinv.max_n':208})
  with T.Kernel(blocks,threads=1,is_npu=True):
   T.import_source(header)
   state=T.alloc_ub((32768,),'float32')
   matrix=T.alloc_ub((24576,),'float16');half_work=T.alloc_ub((4096,),'float16')
   with T.Scope('V'):
    T.reads(A[0],W[0]);T.writes(W[0])
    emit('pv_sync_init',W.access_ptr('rw'))
    operation(A,W,state,matrix,half_work,n,stride,all_batch,job,core,workers)
 return kernel
