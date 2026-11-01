"""Explicit two-slot K64 Cube schedule; original products remain the fallback."""
from code_dsl.op_kernel.intrinsics import emit
def block_gemm():
 import tilelang.language as T
 from tvm import tir
 B=tir.Var('batch','int32');M=tir.Var('rows','int32');K=tir.Var('inner','int32');N=tir.Var('cols','int32')
 @T.prim_func
 def kernel(A:T.Tensor((B*M,K),'float32'),P:T.Tensor((B*K,N),'float32'),C:T.Tensor((B*M,N),'float32'),blocks:T.int32):
  T.func_attr({'global_symbol':'linalg_pinv_tilelang','pinv.scope':'two-slot K64 Cube transfer and compute overlap'})
  with T.Kernel(blocks,threads=1,is_npu=True) as bid:
   T.import_source('#define PINV_STATIC_CUBE_STORAGE 1')
   with T.Scope('C'):
    a=T.alloc_L1((4,128,64),'float32');b=T.alloc_L1((4,64,128),'float32');c=T.alloc_L0C((128,128),'float32')
    la=T.alloc_L0A((2,128,64),'float32');lb=T.alloc_L0B((2,64,128),'float32')
    if M>=128 and M%128==0 and N%128==0 and K>=64 and K<=256 and K%64==0:
     tiles_r=M//128;tiles_c=N//128
     for iteration in T.serial(T.ceildiv(B*tiles_r*tiles_c,blocks)):
      tile=bid+iteration*blocks
      if tile<B*tiles_r*tiles_c:
       job=tile//(tiles_r*tiles_c);row=tile//tiles_c%tiles_r;col=tile%tiles_c
       for kk in T.serial(K//64):emit('br_load_b_batch',b.access_ptr('w',offset=kk*8192),P.access_ptr('r',offset=(job*K+kk*64)*N+col*128),64,128,N)
       emit('br_load_a_shape',a.access_ptr('w'),A.access_ptr('r',offset=(job*M+row*128)*K),K,K,128)
       T.set_flag('MTE2','MTE1',0);T.wait_flag('MTE2','MTE1',0)
       T.set_flag('FIX','M',0);T.wait_flag('FIX','M',0)
       T.set_flag('M','MTE1',0);T.set_flag('M','MTE1',1)
       for kk in T.serial(K//64):
        slot=kk%2
        T.wait_flag('M','MTE1',slot)
        emit('bg_copy_a',la.access_ptr('w',offset=slot*8192),a.access_ptr('r',offset=kk*8192))
        emit('bg_copy_b',lb.access_ptr('w',offset=slot*8192),b.access_ptr('r',offset=kk*8192))
        T.set_flag('MTE1','M',slot);T.wait_flag('MTE1','M',slot)
        T.pipe_barrier('M')
        clear=T.alloc_var('int32');clear=0
        if kk==0:clear=1
        emit('bg_mma',c.access_ptr('rw'),la.access_ptr('r',offset=slot*8192),lb.access_ptr('r',offset=slot*8192),clear)
        T.set_flag('M','MTE1',slot)
       T.wait_flag('M','MTE1',0);T.wait_flag('M','MTE1',1)
       T.set_flag('MTE1','MTE2',0);T.wait_flag('MTE1','MTE2',0)
       T.set_flag('M','FIX',0);T.wait_flag('M','FIX',0)
       T.copy(c,C[job*M+row*128,col*128]);T.barrier_all()
    else:emit('bg_legacy',A.access_ptr('r'),P.access_ptr('r'),C.access_ptr('w'),blocks,B,M,K,N)
 hint=tir.Var('shape_hint','handle');buf=tir.decl_buffer((B,M,K,N),dtype='uint8',name='ShapeHint');buffers=dict(kernel.buffer_map);buffers[hint]=buf
 return tir.PrimFunc(list(kernel.params)+[hint],kernel.body,kernel.ret_type,buffers,kernel.attrs)
