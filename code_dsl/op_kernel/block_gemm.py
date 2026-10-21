from code_dsl.op_kernel.intrinsics import emit
"""Runtime batched FP32 products with cached K16 L1 tiles; primitive only."""
def block_gemm():
 import tilelang.language as T
 from tvm import tir
 B=tir.Var('batch','int32');M=tir.Var('rows','int32');K=tir.Var('inner','int32');N=tir.Var('cols','int32')
 @T.macro
 def products(A,P,C,blocks,bid,a,b,c,tile_m):
   previous_job=T.alloc_var('int32');previous_job=-1
   previous_column=T.alloc_var('int32');previous_column=-1
   tiles_r=M//tile_m;tiles_c=N//128
   for iteration in T.serial(T.ceildiv(B*tiles_r*tiles_c,blocks)):
    tile=bid+iteration*blocks
    if tile<B*tiles_r*tiles_c:
     job=tile//(tiles_r*tiles_c);row=tile//tiles_c%tiles_r;col=tile%tiles_c
     if job!=previous_job or col!=previous_column:
      for kk in T.serial(K//16):T.copy(P[job*K+kk*16,col*128],b[kk,:,:])
      previous_job=job;previous_column=col
     emit('br_load_am',a.access_ptr('w'),A.access_ptr('r',offset=(job*M+row*tile_m)*K),tile_m,K,K)
     T.barrier_all()
     for kk in T.serial(K//16):
      clear=T.alloc_var('int32');clear=0
      if kk==0:clear=1
      T.gemm_v0(a[kk,:,:],b[kk,:,:],c,init=(clear!=0));T.barrier_all()
     T.copy(c,C[job*M+row*tile_m,col*128]);T.barrier_all()
 @T.prim_func
 def kernel(A:T.Tensor((B*M,K),'float32'),P:T.Tensor((B*K,N),'float32'),C:T.Tensor((B*M,N),'float32'),blocks:T.int32):
  T.func_attr({'global_symbol':'linalg_pinv_tilelang','pinv.scope':'runtime cached FP32 block products'})
  with T.Kernel(blocks,threads=1,is_npu=True) as bid:
   a64=T.alloc_L1((16,64,16),'float32');b64=T.alloc_L1((16,16,128),'float32');c64=T.alloc_L0C((64,128),'float32')
   a128=T.alloc_L1((16,128,16),'float32');b128=T.alloc_L1((16,16,128),'float32');c128=T.alloc_L0C((128,128),'float32')
   with T.Scope('C'):
    T.reads(A[0,0],P[0,0]);T.writes(C[0,0])
    if M>=128 and M%128==0:products(A,P,C,blocks,bid,a128,b128,c128,128)
    else:products(A,P,C,blocks,bid,a64,b64,c64,64)
 hint=tir.Var('shape_hint','handle');hint_buffer=tir.decl_buffer((B,M,K,N),dtype='uint8',name='ShapeHint')
 buffers=dict(kernel.buffer_map);buffers[hint]=hint_buffer
 return tir.PrimFunc(list(kernel.params)+[hint],kernel.body,kernel.ret_type,buffers,kernel.attrs)
