"""Runtime FP32 products with alignment-based row tiles and cached K64 data."""
from code_dsl.op_kernel.intrinsics import emit

def block_gemm():
 import tilelang.language as T
 from tvm import tir
 B=tir.Var('batch','int32');M=tir.Var('rows','int32');K=tir.Var('inner','int32');N=tir.Var('cols','int32')
 @T.macro
 def small(A,P,C,blocks,bid):
  a_small=T.alloc_L1((4,64,64),'float32');b_small=T.alloc_L1((4,64,128),'float32');c_small=T.alloc_L0C((64,128),'float32')
  previous_job_small=T.alloc_var('int32');previous_job_small=-1
  previous_column_small=T.alloc_var('int32');previous_column_small=-1
  tiles_r=M//64;tiles_c=N//128
  for iteration in T.serial(T.ceildiv(B*tiles_r*tiles_c,blocks)):
   tile=bid+iteration*blocks
   if tile<B*tiles_r*tiles_c:
    job=tile//(tiles_r*tiles_c);row=tile//tiles_c%tiles_r;col=tile%tiles_c
    if job!=previous_job_small or col!=previous_column_small:
     for kk in T.serial(K//64):T.copy(P[job*K+kk*64,col*128],b_small[kk,:,:])
     previous_job_small=job;previous_column_small=col
    emit('br_load_a_shape',a_small.access_ptr('w'),A.access_ptr('r',offset=(job*M+row*64)*K),K,K,64)
    T.barrier_all()
    for kk in T.serial(K//64):
     clear_small=T.alloc_var('int32');clear_small=0
     if kk==0:clear_small=1
     T.gemm_v0(a_small[kk,:,:],b_small[kk,:,:],c_small,init=(clear_small!=0));T.barrier_all()
    T.copy(c_small,C[job*M+row*64,col*128]);T.barrier_all()
 @T.macro
 def large(A,P,C,blocks,bid):
  a_large=T.alloc_L1((4,128,64),'float32');b_large=T.alloc_L1((4,64,128),'float32');c_large=T.alloc_L0C((128,128),'float32')
  previous_job_large=T.alloc_var('int32');previous_job_large=-1
  previous_column_large=T.alloc_var('int32');previous_column_large=-1
  tiles_r=M//128;tiles_c=N//128
  for iteration in T.serial(T.ceildiv(B*tiles_r*tiles_c,blocks)):
   tile=bid+iteration*blocks
   if tile<B*tiles_r*tiles_c:
    job=tile//(tiles_r*tiles_c);row=tile//tiles_c%tiles_r;col=tile%tiles_c
    if job!=previous_job_large or col!=previous_column_large:
     for kk in T.serial(K//64):T.copy(P[job*K+kk*64,col*128],b_large[kk,:,:])
     previous_job_large=job;previous_column_large=col
    emit('br_load_a_shape',a_large.access_ptr('w'),A.access_ptr('r',offset=(job*M+row*128)*K),K,K,128)
    T.barrier_all()
    for kk in T.serial(K//64):
     clear_large=T.alloc_var('int32');clear_large=0
     if kk==0:clear_large=1
     T.gemm_v0(a_large[kk,:,:],b_large[kk,:,:],c_large,init=(clear_large!=0));T.barrier_all()
    T.copy(c_large,C[job*M+row*128,col*128]);T.barrier_all()
 @T.macro
 def large_k128(A,P,C,blocks,bid):
  a_k128=T.alloc_L1((2,128,128),'float32');b_k128=T.alloc_L1((2,128,128),'float32');c_k128=T.alloc_L0C((128,128),'float32')
  previous_job_k128=T.alloc_var('int32');previous_job_k128=-1
  previous_column_k128=T.alloc_var('int32');previous_column_k128=-1
  tiles_r=M//128;tiles_c=N//128
  for iteration in T.serial(T.ceildiv(B*tiles_r*tiles_c,blocks)):
   tile=bid+iteration*blocks
   if tile<B*tiles_r*tiles_c:
    job=tile//(tiles_r*tiles_c);row=tile//tiles_c%tiles_r;col=tile%tiles_c
    if job!=previous_job_k128 or col!=previous_column_k128:
     for kk in T.serial(K//128):T.copy(P[job*K+kk*128,col*128],b_k128[kk,:,:])
     previous_job_k128=job;previous_column_k128=col
    emit('br_load_a_shape',a_k128.access_ptr('w'),A.access_ptr('r',offset=(job*M+row*128)*K),K,K,128)
    T.barrier_all()
    for kk in T.serial(K//128):
     clear_k128=T.alloc_var('int32');clear_k128=0
     if kk==0:clear_k128=1
     T.gemm_v0(a_k128[kk,:,:],b_k128[kk,:,:],c_k128,init=(clear_k128!=0));T.barrier_all()
    T.copy(c_k128,C[job*M+row*128,col*128]);T.barrier_all()
 @T.prim_func
 def kernel(A:T.Tensor((B*M,K),'float32'),P:T.Tensor((B*K,N),'float32'),C:T.Tensor((B*M,N),'float32'),blocks:T.int32):
  T.func_attr({'global_symbol':'linalg_pinv_tilelang','pinv.scope':'runtime row-tile dispatch with cached FP32 block products'})
  with T.Kernel(blocks,threads=1,is_npu=True) as bid:
   T.import_source('#define PINV_STATIC_CUBE_STORAGE 1')
   with T.Scope('C'):
    if M>=128 and M%128==0:
     if K>=128 and K<=256 and K%128==0:
      large_k128(A,P,C,blocks,bid)
     else:
      large(A,P,C,blocks,bid)
    else:
     small(A,P,C,blocks,bid)
 hint=tir.Var('shape_hint','handle');hint_buffer=tir.decl_buffer((B,M,K,N),dtype='uint8',name='ShapeHint')
 buffers=dict(kernel.buffer_map);buffers[hint]=hint_buffer
 return tir.PrimFunc(list(kernel.params)+[hint],kernel.body,kernel.ret_type,buffers,kernel.attrs)
