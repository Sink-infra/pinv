"""Experimental first Newton residual in native FP32; certificate unchanged."""
from code_dsl.op_kernel.intrinsics import emit

def pack_kernel(header):
 import tilelang.language as T
 @T.prim_func
 def kernel(A:T.Tensor((1,),'float32'),Y:T.Tensor((1,),'float32'),AH:T.Tensor((1,),'float32'),YH:T.Tensor((1,),'float16'),YN:T.Tensor((1,),'float32'),S:T.Tensor((1,),'float32'),blocks:T.int32,n:T.int32,batch:T.int32):
  T.func_attr({'global_symbol':'linalg_pinv_tilelang'})
  with T.Kernel(blocks,threads=1,is_npu=True) as bid:
   T.import_source(header)
   value=T.alloc_ub((4096,),'float32')
   with T.Scope('V'):
    T.reads(A[0],Y[0],S[0]);T.writes(AH[0],YN[0])
    extent=n*n;all=batch*extent;cores=2*blocks
    for iteration in T.serial(T.ceildiv(all//4096,cores)):
     start=(bid+iteration*cores)*4096
     if start<all:
      job=start//extent;power=S[job*32]
      for side in T.unroll(2):
       T.set_flag('V','MTE2',0);T.wait_flag('V','MTE2',0)
       if side==0:emit('pv_load_rows',value.access_ptr('w'),A.access_ptr('r',offset=start),1,4096,0,0)
       else:emit('pv_load_rows',value.access_ptr('w'),Y.access_ptr('r',offset=start),1,4096,0,0)
       T.set_flag('MTE2','V',0);T.wait_flag('MTE2','V',0)
       if side==0:emit('pv_scale',value.access_ptr('w'),value.access_ptr('r'),T.float32(1)/power,4096)
       else:emit('pv_scale',value.access_ptr('w'),value.access_ptr('r'),power,4096)
       T.set_flag('V','MTE3',0);T.wait_flag('V','MTE3',0)
       if side==0:emit('pv_store_rows',AH.access_ptr('w',offset=start),value.access_ptr('r'),1,4096,0,0)
       else:emit('pv_store_rows',YN.access_ptr('w',offset=start),value.access_ptr('r'),1,4096,0,0)
       T.set_flag('MTE3','V',0);T.wait_flag('MTE3','V',0)
 return kernel

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
 return kernel

def finish_kernel(header):
 import tilelang.language as T
 @T.prim_func
 def kernel(A:T.Tensor((1,),'float32'),Y:T.Tensor((1,),'float32'),W:T.Tensor((1,),'float32'),S:T.Tensor((1,),'float32'),blocks:T.int32,n:T.int32,batch:T.int32,stride:T.int32):
  T.func_attr({'global_symbol':'linalg_pinv_tilelang'})
  with T.Kernel(blocks,threads=1,is_npu=True) as bid:
   T.import_source(header)
   pool=T.alloc_ub((24576,),'float32')
   with T.Scope('V'):
    T.reads(A[0],W[0],S[0]);T.writes(Y[0],W[0])
    extent=n*n;all=batch*extent;cores=blocks*2;group=cores//batch;job=bid//group;core=bid%group
    if job<batch:
     power=S[job*32];total=T.alloc_var('float32');total=T.float32(0);inverse=T.alloc_var('float32');inverse=T.float32(0);peak=T.alloc_var('float32');peak=T.float32(0)
     for iteration in T.serial(T.ceildiv(extent//4096,group)):
      start=(core+iteration*group)*4096
      if start<extent:
       T.set_flag('V','MTE2',0);T.wait_flag('V','MTE2',0)
       emit('pv_load_rows',pool.access_ptr('w'),W.access_ptr('r',offset=8*all+job*extent+start),1,4096,0,0)
       emit('pv_load_rows',pool.access_ptr('w',offset=4096),W.access_ptr('r',offset=10*all+job*extent+start),1,4096,0,0)
       emit('pv_load_rows',pool.access_ptr('w',offset=8192),A.access_ptr('r',offset=job*extent+start),1,4096,0,0)
       T.set_flag('MTE2','V',0);T.wait_flag('MTE2','V',0)
       emit('pv_add',pool.access_ptr('w'),pool.access_ptr('r'),pool.access_ptr('r',offset=4096),4096);T.pipe_barrier('V')
       emit('pv_abs',pool.access_ptr('w',offset=4096),pool.access_ptr('r'),4096);T.pipe_barrier('V')
       emit('pv_reduce_max',pool.access_ptr('w',offset=12288),pool.access_ptr('r',offset=4096),pool.access_ptr('rw',offset=16384),4096)
       T.set_flag('V','S',0);T.wait_flag('V','S',0);peak=T.max(peak,pool[12288]/power)
       emit('pv_mul',pool.access_ptr('w',offset=4096),pool.access_ptr('r'),pool.access_ptr('r'),4096)
       emit('pv_scale',pool.access_ptr('w',offset=8192),pool.access_ptr('r',offset=8192),T.float32(1)/power,4096);T.pipe_barrier('V')
       emit('pv_mul',pool.access_ptr('w',offset=8192),pool.access_ptr('r',offset=8192),pool.access_ptr('r',offset=8192),4096);T.pipe_barrier('V')
       emit('pv_reduce_sum',pool.access_ptr('w',offset=12288),pool.access_ptr('r',offset=4096),pool.access_ptr('rw',offset=16384),4096);T.pipe_barrier('V')
       emit('pv_reduce_sum',pool.access_ptr('w',offset=12296),pool.access_ptr('r',offset=8192),pool.access_ptr('rw',offset=16384),4096)
       T.set_flag('V','S',0);T.wait_flag('V','S',0);inverse+=pool[12288];total+=pool[12296]
       emit('pv_scale',pool.access_ptr('w'),pool.access_ptr('r'),T.float32(1)/power,4096)
       T.set_flag('V','MTE3',0);T.wait_flag('V','MTE3',0)
       emit('pv_store_rows',Y.access_ptr('w',offset=job*extent+start),pool.access_ptr('r'),1,4096,0,0)
       emit('pv_all_barrier',W.access_ptr('rw'))
     valid=T.alloc_var('int32');valid=1
     for stage in T.serial(n//64):
      if not S[512+(stage*batch+job)*32]<T.float32(1e29):valid=0
     metadata=2*batch*n*stride+job*8192
     pool[0]=T.Cast('float32',valid);pool[1]=inverse;pool[2]=peak;pool[8]=total
     T.set_flag('S','MTE3',0);T.wait_flag('S','MTE3',0)
     emit('pv_store_rows',W.access_ptr('w',offset=metadata+2048+core*8),pool.access_ptr('r'),1,8,0,0)
     emit('pv_store_rows',W.access_ptr('w',offset=metadata+3072+core*8),pool.access_ptr('r',offset=8),1,8,0,0)
    emit('pv_sync_all',W.access_ptr('rw'))
    if job<batch and core==0:
     metadata=2*batch*n*stride+job*8192
     emit('pv_load_rows',pool.access_ptr('w'),W.access_ptr('r',offset=metadata+3072),1,group*8,0,0)
     T.set_flag('MTE2','S',0);T.wait_flag('MTE2','S',0)
     sum=T.alloc_var('float32');sum=T.float32(0)
     for c in T.serial(group):sum+=pool[c*8]
     pool[0]=S[job*32];pool[1]=sum
     T.set_flag('S','MTE3',0);T.wait_flag('S','MTE3',0)
     emit('pv_store_rows',W.access_ptr('w',offset=metadata+32),pool.access_ptr('r'),1,8,0,0)
    emit('pv_sync_all',W.access_ptr('rw'))
 return kernel
