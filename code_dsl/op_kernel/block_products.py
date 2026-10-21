"""TileLang caches both half planes before all four product terms."""
def product_kernel():
 import tilelang.language as T
 from tvm import tir
 B=tir.Var('batch','int32');N=tir.Var('n','int32')
 @T.prim_func
 def kernel(A:T.Tensor((2*B*N,N),'float16'),Y:T.Tensor((2*B*N,N),'float16'),P:T.Tensor((4*B*N,N),'float32'),blocks:T.int32):
  T.func_attr({'global_symbol':'linalg_pinv_tilelang'})
  with T.Kernel(blocks,threads=1,is_npu=True) as bid:
   a=T.alloc_L1((2,2,32,128),'float16');b=T.alloc_L1((2,2,128,128),'float16');c=T.alloc_L0C((32,128),'float32')
   with T.Scope('C'):
    rt=N//32;ct=N//128;per=B*rt*ct
    for iteration in T.serial(T.ceildiv(per,blocks)):
     tile=bid+iteration*blocks
     if tile<per:
      job=tile//(rt*ct);row=tile//ct%rt;col=tile%ct
      for plane in T.serial(2):
       for kk in T.serial(N//128):
        T.copy(A[plane*B*N+job*N+row*32,kk*128],a[plane,kk,:,:])
        T.copy(Y[plane*B*N+job*N+kk*128,col*128],b[plane,kk,:,:])
      T.barrier_all()
      for term in T.serial(4):
       for kk in T.serial(N//128):
        clear=T.alloc_var('int32');clear=0
        if kk==0:clear=1
        T.gemm_v0(a[term//2,kk,:,:],b[term%2,kk,:,:],c,init=(clear!=0));T.barrier_all()
       T.copy(c,P[term*B*N+job*N+row*32,col*128]);T.barrier_all()
 hint=tir.Var('shape_hint','handle');hint_buffer=tir.decl_buffer((B,N),dtype='uint8',name='ShapeHint')
 buffers=dict(kernel.buffer_map);buffers[hint]=hint_buffer
 return tir.PrimFunc(list(kernel.params)+[hint],kernel.body,kernel.ret_type,buffers,kernel.attrs)
