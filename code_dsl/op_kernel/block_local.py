"""Private resident64 factor inverse with a cached condition product."""
from code_dsl.op_kernel.local import _make_quick_schedule
from code_dsl.op_kernel.kernels import instruction_source
def quick_kernel():
 import tilelang.language as T
 operation=_make_quick_schedule(64,8,1,full_tile=True,condition_metric=True,minimum_magnitude=1e-30)
 @T.prim_func
 def kernel(A:T.Tensor((1,),'float32'),Y:T.Tensor((1,),'float32'),W:T.Tensor((1,),'float32'),S:T.Tensor((1,),'float32'),
            blocks:T.int32,job:T.int32,m:T.int32,n:T.int32,batch:T.int32,atol:T.float32,rtol:T.float32):
  T.func_attr({'global_symbol':'linalg_pinv_tilelang','pinv.component':'block_factor_inverse'})
  with T.Kernel(blocks,threads=1,is_npu=True):
   T.import_source(instruction_source())
   pool=T.alloc_ub((19136,),'float32');indices=T.alloc_ub((64,),'uint32');order=T.alloc_ub((128,),'uint32');accepted=T.alloc_ub((8,),'int32')
   with T.Scope('V'):
    T.reads(A[0]);T.writes(Y[0],S[0])
    accepted[0]=0
    operation(A,Y,pool,indices,order,accepted,job*4096,64,T.float32(0),T.float32(0))
    S[job*32]=T.Cast('float32',accepted[0])
    word=T.alloc_var('int32');word=accepted[1]
    metric=T.alloc_var('float32');metric=T.reinterpret('float32',word)
    S[job*32+1]=metric
 return kernel
