"""FP32 block products with one owner for each updated row tile."""
from code_dsl.op_kernel.intrinsics import emit

def _hint(function,B,N):
 from tvm import tir
 hint=tir.Var('shape_hint','handle')
 buffers=dict(function.buffer_map)
 buffers[hint]=tir.decl_buffer((B,N),dtype='uint8',name='ShapeHint')
 return tir.PrimFunc(list(function.params)+[hint],function.body,function.ret_type,buffers,function.attrs)

def inplace_panel_kernel():
 import tilelang.language as T
 from tvm import tir
 B=tir.Var('batch','int32');N=tir.Var('n','int32')
 @T.prim_func
 def kernel(D:T.Tensor((B*64,64),'float32'),P:T.Tensor((B*N,N),'float32'),blocks:T.int32,start_row:T.int32):
  T.func_attr({'global_symbol':'linalg_pinv_tilelang','pinv.scope':'inplace panel with disjoint 512 byte column slices'})
  with T.Kernel(blocks,threads=1,is_npu=True) as bid:
   T.import_source('#define PINV_STATIC_CUBE_STORAGE 1')
   with T.Scope('C'):
    T.reads(D[0,0],P[0,0]);T.writes(P[0,0])
    a=T.alloc_L1((64,64),'float32')
    b=T.alloc_L1((64,128),'float32')
    c=T.alloc_L0C((64,128),'float32')
    for iteration in T.serial(T.ceildiv(B*(N//128),blocks)):
     tile=bid+iteration*blocks
     if tile<B*(N//128):
      job=tile//(N//128);col=tile%(N//128)
      emit('br_load_a_shape',a.access_ptr('w'),D.access_ptr('r',offset=job*4096),64,64,64)
      T.copy(P[job*N+start_row,col*128],b)
      T.barrier_all()
      T.gemm_v0(a,b,c,init=True)
      T.barrier_all()
      T.copy(c,P[job*N+start_row,col*128])
      T.barrier_all()
 return _hint(kernel,B,N)

def atomic_rows_kernel():
 import tilelang.language as T
 from tvm import tir
 B=tir.Var('batch','int32');N=tir.Var('n','int32')
 @T.prim_func
 def kernel(F:T.Tensor((B*N,64),'float32'),P:T.Tensor((B*N,N),'float32'),blocks:T.int32,start_row:T.int32):
  T.func_attr({'global_symbol':'linalg_pinv_tilelang','pinv.scope':'atomic block update excluding the read-only pivot row tile'})
  with T.Kernel(blocks,threads=1,is_npu=True) as bid:
   T.import_source('#define PINV_STATIC_CUBE_STORAGE 1')
   with T.Scope('C'):
    T.reads(F[0,0],P[0,0]);T.writes(P[0,0])
    a=T.alloc_L1((64,64),'float32')
    b=T.alloc_L1((64,256),'float32')
    c=T.alloc_L0C((64,256),'float32')
    tiles_r=N//64;tiles_c=N//256
    for iteration in T.serial(T.ceildiv(B*tiles_r*tiles_c,blocks)):
     tile=bid+iteration*blocks
     if tile<B*tiles_r*tiles_c:
      job=tile//(tiles_r*tiles_c);row=tile//tiles_c%tiles_r;col=tile%tiles_c
      if row*64!=start_row:
       emit('br_load_a_shape',a.access_ptr('w'),F.access_ptr('r',offset=(job*N+row*64)*64),64,64,64)
       T.copy(P[job*N+start_row,col*256],b)
       T.barrier_all()
       T.gemm_v0(a,b,c,init=True)
       T.barrier_all()
       emit('ba_atomic_on',P.access_ptr('rw'))
       T.copy(c,P[job*N+row*64,col*256])
       T.barrier_all()
       emit('ba_atomic_off',P.access_ptr('rw'))
 return _hint(kernel,B,N)
