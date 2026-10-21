"""Cached L1 digit GEMM and FP16 correction GEMM with runtime extents.

The loops, tiling, data reuse and accumulation schedule are TileLang.
The exported runtime supplies only SDK transfer and MMA primitives.
"""

def integer_gemm_kernel():
 import tilelang.language as T
 from tvm import tir
 N=tir.Var('n','int32');P=tir.Var('planes','int32')
 @T.prim_func
 def kernel(A:T.Tensor((P*N,N),'int8'),B:T.Tensor((N,P*N),'int8'),C:T.Tensor(((2*P-1)*N,N),'int32'),blocks:T.int32):
  T.func_attr({'global_symbol':'linalg_pinv_tilelang','pinv.scope':'exact digit convolution by weight'})
  with T.Kernel(blocks,threads=1,is_npu=True) as bid:
   a=T.alloc_L1((5,2,32,128),'int8');b=T.alloc_L1((5,2,128,128),'int8');c=T.alloc_L0C((32,128),'int32')
   with T.Scope('C'):
    for iteration in T.serial(T.ceildiv(T.ceildiv(N,32)*T.ceildiv(N,128),blocks)):
     tile=bid+iteration*blocks
     if tile<T.ceildiv(N,32)*T.ceildiv(N,128):
      row=tile//T.ceildiv(N,128);col=tile%T.ceildiv(N,128)
      for i in T.serial(P):
       for kk in T.serial(N//128):
        T.copy(A[i*N+row*32,kk*128],a[i,kk,:,:])
        T.copy(B[kk*128,i*N+col*128],b[i,kk,:,:])
      T.barrier_all()
      diagonals=T.alloc_var('int32');diagonals=2*P-1
      if diagonals>6:diagonals=6
      for diagonal in T.serial(diagonals):
       first=T.alloc_var('int32');first=0
       last=T.alloc_var('int32');last=P
       if diagonal>=P:first=diagonal-P+1
       if diagonal+1<P:last=diagonal+1
       for i in T.serial(first,last):
        j=diagonal-i
        for kk in T.serial(N//128):
         clear=T.alloc_var('int32');clear=0
         if i==first:
          if kk==0:clear=1
         T.gemm_v0(a[i,kk,:,:],b[j,kk,:,:],c,init=(clear!=0));T.barrier_all()
       T.copy(c,C[diagonal*N+row*32,col*128]);T.barrier_all()
 hint=tir.Var('shape_hint','handle');hint_buffer=tir.decl_buffer((P,),dtype='uint8',name='ShapeHint')
 buffers=dict(kernel.buffer_map);buffers[hint]=hint_buffer
 return tir.PrimFunc(list(kernel.params)+[hint],kernel.body,kernel.ret_type,buffers,kernel.attrs)

def half_gemm_kernel():
    import tilelang.language as T
    from tvm import tir
    N=tir.Var('n','int32');P=tir.Var('planes','int32')
    @T.prim_func
    def kernel(A:T.Tensor((P*N,N),'float16'),B:T.Tensor((N,P*N),'float16'),C:T.Tensor((P*N,P*N),'float32'),blocks:T.int32):
     T.func_attr({'global_symbol':'linalg_pinv_tilelang','pinv.scope':'DSL tiled FP32 GEMM diagnostic'})
     with T.Kernel(blocks,threads=1,is_npu=True) as bid:
      a=T.alloc_L1((32,128),'float16');b=T.alloc_L1((128,128),'float16');c=T.alloc_L0C((32,128),'float32')
      with T.Scope('C'):
       for iteration in T.serial(T.ceildiv(T.ceildiv(P*N,32)*T.ceildiv(P*N,128),blocks)):
        tile=bid+iteration*blocks
        if tile<T.ceildiv(P*N,32)*T.ceildiv(P*N,128):
         row=tile//T.ceildiv(P*N,128);col=tile%T.ceildiv(P*N,128)
         for kk in T.serial(N//128):
          T.copy(A[row*32,kk*128],a);T.copy(B[kk*128,col*128],b);T.barrier_all()
          T.gemm_v0(a,b,c,init=(kk==0));T.barrier_all()
         T.copy(c,C[row*32,col*128]);T.barrier_all()
    hint=tir.Var('shape_hint','handle')
    hint_buffer=tir.decl_buffer((P,),dtype='uint8',name='ShapeHint')
    buffers=dict(kernel.buffer_map);buffers[hint]=hint_buffer
    kernel=tir.PrimFunc(list(kernel.params)+[hint],kernel.body,kernel.ret_type,buffers,kernel.attrs)
    return kernel
