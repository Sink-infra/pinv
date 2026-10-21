from code_dsl.op_kernel.intrinsics import emit
"""Runtime batched FP32 products with cached K16 L1 tiles; primitive only."""
def block_gemm():
 import tilelang.language as T
 from tvm import tir
 B=tir.Var('batch','int32');M=tir.Var('rows','int32');K=tir.Var('inner','int32');N=tir.Var('cols','int32')
 @T.prim_func
 def kernel(A:T.Tensor((B*M,K),'float32'),P:T.Tensor((B*K,N),'float32'),C:T.Tensor((B*M,N),'float32'),blocks:T.int32):
  T.func_attr({'global_symbol':'linalg_pinv_tilelang','pinv.scope':'runtime cached FP32 block products'})
  with T.Kernel(blocks,threads=1,is_npu=True) as bid:
   a=T.alloc_L1((16,64,16),'float32');b=T.alloc_L1((16,16,128),'float32');c=T.alloc_L0C((64,128),'float32')
   with T.Scope('C'):
    previous_job=T.alloc_var('int32');previous_job=-1
    previous_column=T.alloc_var('int32');previous_column=-1
    tiles_r=M//64;tiles_c=N//128
    for iteration in T.serial(T.ceildiv(B*tiles_r*tiles_c,blocks)):
     tile=bid+iteration*blocks
     if tile<B*tiles_r*tiles_c:
      job=tile//(tiles_r*tiles_c);row=tile//tiles_c%tiles_r;col=tile%tiles_c
      if job!=previous_job or col!=previous_column:
       for kk in T.serial(K//16):T.copy(P[job*K+kk*16,col*128],b[kk,:,:])
       previous_job=job;previous_column=col
      emit('br_load_a',a.access_ptr('w'),A.access_ptr('r',offset=(job*M+row*64)*K),K,K)
      T.barrier_all()
      for kk in T.serial(K//16):
       clear=T.alloc_var('int32');clear=0
       if kk==0:clear=1
       T.gemm_v0(a[kk,:,:],b[kk,:,:],c,init=(clear!=0));T.barrier_all()
      T.copy(c,C[job*M+row*64,col*128]);T.barrier_all()
 hint=tir.Var('shape_hint','handle');hint_buffer=tir.decl_buffer((B,M,K,N),dtype='uint8',name='ShapeHint')
 buffers=dict(kernel.buffer_map);buffers[hint]=hint_buffer
 return tir.PrimFunc(list(kernel.params)+[hint],kernel.body,kernel.ret_type,buffers,kernel.attrs)
