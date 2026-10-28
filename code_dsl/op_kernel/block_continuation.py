"""Residual decoding and shared continuation components."""
from code_dsl.op_kernel.intrinsics import emit
from code_dsl.op_kernel.block_certificate import pack_kernel,finish_kernel

def decode_kernel(header):
 import tilelang.language as T
 @T.prim_func
 def kernel(P:T.Tensor((1,),'float32'),R:T.Tensor((1,),'float32'),blocks:T.int32,n:T.int32,batch:T.int32):
  T.func_attr({'global_symbol':'linalg_pinv_tilelang'})
  with T.Kernel(blocks,threads=1,is_npu=True) as bid:
   T.import_source(header)
   value=T.alloc_ub((4096,),'float32');identity=T.alloc_ub((4096,),'float32')
   with T.Scope('V'):
    T.reads(P[0]);T.writes(R[0])
    all=batch*n*n;cores=2*blocks
    for iteration in T.serial(T.ceildiv(all//4096,cores)):
     start=(bid+iteration*cores)*4096
     if start<all:
      T.set_flag('V','MTE2',0);T.wait_flag('V','MTE2',0)
      emit('pv_load_rows',value.access_ptr('w'),P.access_ptr('r',offset=start),1,4096,0,0)
      T.set_flag('MTE2','V',0);T.wait_flag('MTE2','V',0)
      emit('pv_fill',identity.access_ptr('w'),T.float32(0),4096)
      T.set_flag('V','S',0);T.wait_flag('V','S',0)
      for row in T.serial(4096//n):identity[row*n+(start//n+row)%n]=T.float32(1)
      T.set_flag('S','V',0);T.wait_flag('S','V',0)
      T.tile.sub(value,identity,value)
      T.set_flag('V','MTE3',0);T.wait_flag('V','MTE3',0)
      emit('pv_store_rows',R.access_ptr('w',offset=start),value.access_ptr('r'),1,4096,0,0)
      T.set_flag('MTE3','V',0);T.wait_flag('MTE3','V',0)
      T.tile.add(value,value,identity);T.pipe_barrier('V')
      T.set_flag('V','MTE3',0);T.wait_flag('V','MTE3',0)
      emit('pv_store_rows',R.access_ptr('w',offset=all+start),value.access_ptr('r'),1,4096,0,0)
      T.set_flag('MTE3','V',0);T.wait_flag('MTE3','V',0)
 return kernel

def accumulate_kernel(header):
 import tilelang.language as T
 @T.prim_func
 def kernel(W:T.Tensor((1,),'float32'),blocks:T.int32,n:T.int32,batch:T.int32):
  T.func_attr({'global_symbol':'linalg_pinv_tilelang'})
  with T.Kernel(blocks,threads=1,is_npu=True) as bid:
   T.import_source(header)
   pool=T.alloc_ub((8192,),'float32')
   with T.Scope('V'):
    T.reads(W[0]);T.writes(W[0])
    extent=n*n;all=batch*extent;cores=blocks*2
    for iteration in T.serial(T.ceildiv(all//4096,cores)):
     start=(bid+iteration*cores)*4096
     if start<all:
      T.set_flag('V','MTE2',0);T.wait_flag('V','MTE2',0)
      emit('pv_load_rows',pool.access_ptr('w'),W.access_ptr('r',offset=9*all+start),1,4096,0,0)
      emit('pv_load_rows',pool.access_ptr('w',offset=4096),W.access_ptr('r',offset=11*all+start),1,4096,0,0)
      T.set_flag('MTE2','V',0);T.wait_flag('MTE2','V',0)
      emit('pv_add',pool.access_ptr('w'),pool.access_ptr('r'),pool.access_ptr('r',offset=4096),4096);T.pipe_barrier('V')
      T.set_flag('V','MTE3',0);T.wait_flag('V','MTE3',0)
      emit('pv_store_rows',W.access_ptr('w',offset=9*all+start),pool.access_ptr('r'),1,4096,0,0)
      T.set_flag('MTE3','V',0);T.wait_flag('MTE3','V',0)
 return kernel


