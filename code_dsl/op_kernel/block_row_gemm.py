from code_dsl.op_kernel.intrinsics import emit
def row_gemm():
 import tilelang.language as T
 from tvm import tir
 B=tir.Var('batch','int32');M=tir.Var('rows','int32');K=tir.Var('inner','int32');N=tir.Var('cols','int32')
 @T.prim_func
 def kernel(A:T.Tensor((B*M,K),'float32'),P:T.Tensor((B*K,N),'float32'),C:T.Tensor((B*M,N),'float32'),blocks:T.int32):
  T.func_attr({'global_symbol':'linalg_pinv_tilelang','pinv.scope':'row-owned K64 FP32 update product'})
  with T.Kernel(blocks,threads=1,is_npu=True) as bid:
   T.import_source('#define PINV_STATIC_CUBE_STORAGE 1')
   with T.Scope('C'):
    a=T.alloc_L1((64,64),'float32');b=T.alloc_L1((64,256),'float32');c=T.alloc_L0C((64,256),'float32')
    tiles_r=M//64;tiles_c=N//256
    for iteration in T.serial(T.ceildiv(B*tiles_r*tiles_c,blocks)):
     tile=bid+iteration*blocks
     if tile<B*tiles_r*tiles_c:
      job=tile//(tiles_r*tiles_c);row=tile//tiles_c%tiles_r;col=tile%tiles_c
      emit('br_load_a_shape',a.access_ptr('w'),A.access_ptr('r',offset=(job*M+row*64)*K),K,K,64)
      T.copy(P[job*K,col*256],b)
      T.barrier_all()
      T.gemm_v0(a,b,c,init=True);T.barrier_all()
      T.copy(c,C[job*M+row*64,col*256]);T.barrier_all()
 hint=tir.Var('shape_hint','handle');buf=tir.decl_buffer((B,M,K,N),dtype='uint8',name='ShapeHint');buffers=dict(kernel.buffer_map);buffers[hint]=buf
 return tir.PrimFunc(list(kernel.params)+[hint],kernel.body,kernel.ret_type,buffers,kernel.attrs)
