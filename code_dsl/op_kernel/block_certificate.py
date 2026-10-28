"""Overlap a conservative inverse norm certificate with the correction product."""
from code_dsl.op_kernel.intrinsics import emit

def pack_kernel(header):
 import tilelang.language as T
 @T.prim_func
 def kernel(A:T.Tensor((1,),'float32'),Y:T.Tensor((1,),'float32'),AH:T.Tensor((1,),'float32'),YH:T.Tensor((1,),'float16'),YN:T.Tensor((1,),'float32'),S:T.Tensor((1,),'float32'),blocks:T.int32,n:T.int32,batch:T.int32,stride:T.int32,atol:T.float32,rtol:T.float32):
  T.func_attr({'global_symbol':'linalg_pinv_tilelang'})
  with T.Kernel(blocks,threads=1,is_npu=True) as bid:
   T.import_source(header)
   pool=T.alloc_ub((24576,),'float32')
   with T.Scope('V'):
    T.reads(A[0],YN[0],AH[0],S[0]);T.writes(AH[0])
    extent=n*n;all=batch*extent;cores=2*blocks;group=cores//batch;job=bid//group;core=bid%group
    if job<batch:
     power=S[job*32]
     inverse=T.alloc_var('float32');inverse=T.float32(0)
     total=T.alloc_var('float32');total=T.float32(0)
     residual=T.alloc_var('float32');residual=T.float32(0)
     for iteration in T.serial(T.ceildiv(extent//4096,group)):
      start=(core+iteration*group)*4096
      if start<extent:
       T.set_flag('V','MTE2',0);T.wait_flag('V','MTE2',0)
       emit('pv_load_rows',pool.access_ptr('w'),YN.access_ptr('r',offset=job*extent+start),1,4096,0,0)
       emit('pv_load_rows',pool.access_ptr('w',offset=4096),A.access_ptr('r',offset=job*extent+start),1,4096,0,0)
       emit('pv_load_rows',pool.access_ptr('w',offset=8192),AH.access_ptr('r',offset=9*all+job*extent+start),1,4096,0,0)
       T.set_flag('MTE2','V',0);T.wait_flag('MTE2','V',0)
       emit('pv_scale',pool.access_ptr('w',offset=4096),pool.access_ptr('r',offset=4096),T.float32(1)/power,4096);T.pipe_barrier('V')
       emit('pv_mul',pool.access_ptr('w'),pool.access_ptr('r'),pool.access_ptr('r'),4096)
       emit('pv_mul',pool.access_ptr('w',offset=4096),pool.access_ptr('r',offset=4096),pool.access_ptr('r',offset=4096),4096)
       emit('pv_mul',pool.access_ptr('w',offset=8192),pool.access_ptr('r',offset=8192),pool.access_ptr('r',offset=8192),4096);T.pipe_barrier('V')
       emit('pv_reduce_sum',pool.access_ptr('w',offset=12288),pool.access_ptr('r'),pool.access_ptr('rw',offset=16384),4096);T.pipe_barrier('V')
       emit('pv_reduce_sum',pool.access_ptr('w',offset=12296),pool.access_ptr('r',offset=4096),pool.access_ptr('rw',offset=16384),4096);T.pipe_barrier('V')
       emit('pv_reduce_sum',pool.access_ptr('w',offset=12304),pool.access_ptr('r',offset=8192),pool.access_ptr('rw',offset=16384),4096)
       T.set_flag('V','S',0);T.wait_flag('V','S',0)
       inverse+=pool[12288];total+=pool[12296];residual+=pool[12304]
     valid=T.alloc_var('int32');valid=1
     for stage in T.serial(n//64):
      if not S[512+(stage*batch+job)*32]<T.float32(1e29):valid=0
     pool[0]=inverse;pool[1]=total;pool[2]=residual;pool[3]=T.Cast('float32',valid)
     T.set_flag('S','MTE3',0);T.wait_flag('S','MTE3',0)
     metadata=2*batch*extent+job*8192
     emit('pv_store_rows',AH.access_ptr('w',offset=metadata+4096+core*32),pool.access_ptr('r'),1,8,0,0)
    emit('pv_sync_all',AH.access_ptr('rw'))
    if job<batch and core==0:
     metadata=2*batch*n*stride+job*8192
     emit('pv_load_rows',pool.access_ptr('w'),AH.access_ptr('r',offset=2*batch*extent+job*8192+4096),1,group*32,0,0)
     T.set_flag('MTE2','S',0);T.wait_flag('MTE2','S',0)
     inverse=T.alloc_var('float32');inverse=T.float32(0)
     total=T.alloc_var('float32');total=T.float32(0)
     residual=T.alloc_var('float32');residual=T.float32(0)
     fast=T.alloc_var('int32');fast=1
     for c in T.serial(group):
      inverse+=pool[c*32];total+=pool[c*32+1];residual+=pool[c*32+2]
      if pool[c*32+3]<T.float32(.5):fast=0
     wi=T.alloc_var('int32');wi=T.reinterpret('int32',inverse)
     wt=T.alloc_var('int32');wt=T.reinterpret('int32',total)
     wr=T.alloc_var('int32');wr=T.reinterpret('int32',residual)
     if T.bitwise_and(wi,2139095040)==2139095040 or T.bitwise_and(wt,2139095040)==2139095040 or T.bitwise_and(wr,2139095040)==2139095040:fast=0
     if not (inverse>T.float32(0) and total>T.float32(0) and residual>=T.float32(0) and residual<T.float32(1)):fast=0
     pool[512]=total;pool[513]=inverse;pool[514]=residual
     T.set_flag('S','V',0);T.wait_flag('S','V',0)
     emit('pv_sqrt',pool.access_ptr('w',offset=512),pool.access_ptr('r',offset=512),3)
     T.set_flag('V','S',0);T.wait_flag('V','S',0)
     factor=T.float32(1.001)+pool[514]
     inverse_bound=inverse*factor*factor
     peak_bound=pool[513]*factor/S[job*32]
     threshold=T.max(pool[512]*rtol,atol/S[job*32])
     if not (inverse_bound*threshold*threshold<T.float32(81) and inverse_bound*total<T.float32(1e15)):fast=0
     emit('pv_fill',pool.access_ptr('w',offset=1024),T.float32(0),group*8)
     T.set_flag('V','S',0);T.wait_flag('V','S',0)
     for c in T.serial(group):pool[1024+c*8]=T.float32(1)
     pool[1024]=T.Cast('float32',fast);pool[1025]=inverse_bound;pool[1026]=peak_bound
     pool[0]=S[job*32];pool[1]=total
     pool[528]=T.Cast('float32',fast)
     T.set_flag('S','MTE3',0);T.wait_flag('S','MTE3',0)
     emit('pv_store_rows',AH.access_ptr('w',offset=metadata+2048),pool.access_ptr('r',offset=1024),1,group*8,0,0)
     emit('pv_store_rows',AH.access_ptr('w',offset=metadata+32),pool.access_ptr('r'),1,8,0,0)
     emit('pv_store_rows',S.access_ptr('w',offset=job*512+16),pool.access_ptr('r',offset=528),1,8,0,0)
 return kernel

def finish_kernel(header):
 import tilelang.language as T
 @T.prim_func
 def kernel(A:T.Tensor((1,),'float32'),Y:T.Tensor((1,),'float32'),W:T.Tensor((1,),'float32'),S:T.Tensor((1,),'float32'),blocks:T.int32,n:T.int32,batch:T.int32,stride:T.int32,atol:T.float32,rtol:T.float32):
  T.func_attr({'global_symbol':'linalg_pinv_tilelang'})
  with T.Kernel(blocks,threads=1,is_npu=True) as bid:
   T.import_source(header)
   pool=T.alloc_ub((24576,),'float32')
   with T.Scope('V'):
    T.reads(W[0],S[0]);T.writes(Y[0],W[0],S[0])
    extent=n*n;all=batch*extent;cores=blocks*2;group=cores//batch;job=bid//group;core=bid%group
    if job<batch:
     power=S[job*32]
     for iteration in T.serial(T.ceildiv(extent//4096,group)):
      start=(core+iteration*group)*4096
      if start<extent:
       T.set_flag('V','MTE2',0);T.wait_flag('V','MTE2',0)
       emit('pv_load_rows',pool.access_ptr('w'),W.access_ptr('r',offset=8*all+job*extent+start),1,4096,0,0)
       emit('pv_load_rows',pool.access_ptr('w',offset=4096),W.access_ptr('r',offset=10*all+job*extent+start),1,4096,0,0)
       T.set_flag('MTE2','V',0);T.wait_flag('MTE2','V',0)
       emit('pv_add',pool.access_ptr('w'),pool.access_ptr('r'),pool.access_ptr('r',offset=4096),4096);T.pipe_barrier('V')
       emit('pv_scale',pool.access_ptr('w'),pool.access_ptr('r'),T.float32(1)/power,4096)
       T.set_flag('V','MTE3',0);T.wait_flag('V','MTE3',0)
       emit('pv_store_rows',Y.access_ptr('w',offset=job*extent+start),pool.access_ptr('r'),1,4096,0,0)
       emit('pv_all_barrier',W.access_ptr('rw'))
    emit('pv_sync_all',W.access_ptr('rw'))
 return kernel
