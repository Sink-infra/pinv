"""Independent Cube row owners for the associative block update."""
from code_dsl.op_kernel.intrinsics import emit
from code_dsl.op_kernel.block_atomic_products import _hint

def chain_kernel():
 import tilelang.language as T
 from tvm import tir
 B=tir.Var('batch','int32');N=tir.Var('n','int32')
 @T.prim_func
 def kernel(F:T.Tensor((B*N,64),'float32'),D:T.Tensor((B*64,64),'float32'),P:T.Tensor((B*64,N),'float32'),Tmp:T.Tensor((B*N,64),'float32'),Y:T.Tensor((B*N,N),'float32'),blocks:T.int32,start_row:T.int32):
  T.func_attr({'global_symbol':'linalg_pinv_tilelang','pinv.scope':'independent Cube coefficient products and row updates'})
  with T.Kernel(blocks,threads=1,is_npu=True) as bid:
   T.import_source('#define PINV_STATIC_CUBE_STORAGE 1')
   with T.Scope('C'):
    T.reads(F[0,0],D[0,0],P[0,0]);T.writes(Tmp[0,0],Y[0,0])
    f=T.alloc_L1((64,64),'float32');inv=T.alloc_L1((64,64),'float32');a=T.alloc_L1((64,64),'float32');p=T.alloc_L1((64,256),'float32')
    c1=T.alloc_L0C((64,64),'float32');c2=T.alloc_L0C((64,256),'float32')
    tiles_r=N//64;tiles_c=N//256
    for iteration in T.serial(T.ceildiv(B*tiles_r*tiles_c,blocks)):
     tile=bid+iteration*blocks
     if tile<B*tiles_r*tiles_c:
      job=tile//(tiles_r*tiles_c);row=tile//tiles_c%tiles_r;col=tile%tiles_c
      if row*64!=start_row:
       emit('br_load_a_shape',f.access_ptr('w'),F.access_ptr('r',offset=(job*N+row*64)*64),64,64,64)
       T.copy(D[job*64,0],inv)
       T.barrier_all();T.gemm_v0(f,inv,c1,init=True);T.barrier_all()
       emit('bc_store',Tmp.access_ptr('w',offset=(job*N+row*64)*64),c1.access_ptr('r'),64,64,64)
       T.barrier_all()
       emit('br_load_a_shape',a.access_ptr('w'),Tmp.access_ptr('r',offset=(job*N+row*64)*64),64,64,64)
      else:
       emit('br_load_a_shape',a.access_ptr('w'),D.access_ptr('r',offset=job*4096),64,64,64)
      T.copy(P[job*64,col*256],p)
      T.barrier_all();T.gemm_v0(a,p,c2,init=True);T.barrier_all()
      if row*64!=start_row:emit('ba_atomic_on',Y.access_ptr('rw'))
      emit('bc_store',Y.access_ptr('w',offset=(job*N+row*64)*N+col*256),c2.access_ptr('r'),64,256,N)
      T.barrier_all()
      if row*64!=start_row:emit('ba_atomic_off',Y.access_ptr('rw'))
 return _hint(kernel,B,N)
