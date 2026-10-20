"""Two rounded words, balanced radix-256 extraction and exact discarded tail."""
from code_dsl.op_kernel.kernels import instruction_source
from code_dsl.op_kernel.intrinsics import emit

def quant_kernel():
 import tilelang.language as T
 header=instruction_source()+'''
 namespace tl::ascend {
 __aicore__ inline void rq_half(LocalTensor<half> d,LocalTensor<float> a,LocalTensor<half>,int n){Cast(d,a,RoundMode::CAST_NONE,n);}
 __aicore__ inline void rq_i8(LocalTensor<int8_t> d,LocalTensor<half> a,LocalTensor<int8_t>,int n){Cast(d,a,RoundMode::CAST_RINT,n);}
 __aicore__ inline void rq_store(GlobalTensor<int8_t> d,LocalTensor<int8_t> a,GlobalTensor<int8_t>,int rows,int cols,int gap){DataCopyPad(d,a,DataCopyExtParams{uint16_t(rows),uint32_t(cols),0,uint32_t(gap),0});}
 }
 '''
 @T.prim_func
 def kernel(A:T.Tensor((1,),'float32'),Y:T.Tensor((1,),'float32'),QA:T.Tensor((1,),'int8'),QY:T.Tensor((1,),'int8'),QN:T.Tensor((1,),'float32'),
            blocks:T.int32,n:T.int32,job:T.int32,scale_a:T.float32,scale_y:T.float32):
  T.func_attr({'global_symbol':'linalg_pinv_tilelang','pinv.scope':'two word balanced residual quantization'})
  with T.Kernel(blocks,threads=1,is_npu=True) as core:
   T.import_source(header)
   values=T.alloc_ub((2048,),'float32');integers=T.alloc_ub((2048,),'int32')
   rounded=T.alloc_ub((2048,),'float32');carry=T.alloc_ub((2048,),'float32');temp=T.alloc_ub((2048,),'float32')
   high=T.alloc_ub((2048,),'float32');low=T.alloc_ub((2048,),'float32')
   half_buf=T.alloc_ub((2048,),'float16');small=T.alloc_ub((2048,),'int8')
   with T.Scope('V'):
    T.reads(A[0],Y[0]);T.writes(QA[0],QY[0],QN[0])
    owned=(n+blocks-1-core)//blocks;count=owned*n
    for side in T.unroll(2):
     emit('pv_fill',values.access_ptr('w'),T.float32(0),2048);T.pipe_barrier('V')
     T.set_flag('V','MTE2',0);T.wait_flag('V','MTE2',0)
     if side==0:emit('pv_load_rows',values.access_ptr('w'),A.access_ptr('r',offset=job*n*n+core*n),owned,n,(blocks-1)*n,0)
     else:emit('pv_load_rows',values.access_ptr('w'),Y.access_ptr('r',offset=job*n*n+core*n),owned,n,(blocks-1)*n,0)
     T.set_flag('MTE2','V',0);T.wait_flag('MTE2','V',0)
     if side==0:T.tile.mul(values,values,T.float32(256)/scale_a)
     else:T.tile.mul(values,values,T.float32(256)/scale_y)
     T.pipe_barrier('V')
     T.tile.mul(values,values,T.float32(65536));T.pipe_barrier('V')
     T.tile.cast(integers,values,'CAST_RINT',2048);T.pipe_barrier('V')
     T.tile.cast(high,integers,'CAST_RINT',2048);T.pipe_barrier('V')
     T.tile.sub(values,values,high);T.pipe_barrier('V')
     T.tile.mul(values,values,T.float32(65536));T.pipe_barrier('V')
     T.tile.cast(integers,values,'CAST_RINT',2048);T.pipe_barrier('V')
     T.tile.cast(low,integers,'CAST_RINT',2048);T.pipe_barrier('V')
     T.tile.sub(values,values,low);T.pipe_barrier('V')
     T.tile.mul(values,values,T.float32(256));T.pipe_barrier('V')
     T.tile.mul(temp,values,values);T.pipe_barrier('V')
     emit('pv_reduce_sum',carry.access_ptr('w'),temp.access_ptr('r'),rounded.access_ptr('rw'),2048)
     T.set_flag('V','MTE3',0);T.wait_flag('V','MTE3',0)
     emit('pv_store_rows',QN.access_ptr('w',offset=core*16+side*8),carry.access_ptr('r'),1,8,0,0)
     T.set_flag('MTE3','V',0);T.wait_flag('MTE3','V',0)
     for part in T.serial(2):
      for digit in T.serial(2):
       T.tile.add(carry,low,T.float32(128));T.pipe_barrier('V')
       T.tile.mul(carry,carry,T.float32(0.00390625));T.pipe_barrier('V')
       T.tile.cast(integers,carry,'CAST_FLOOR',2048);T.pipe_barrier('V')
       T.tile.cast(carry,integers,'CAST_RINT',2048);T.pipe_barrier('V')
       T.tile.mul(temp,carry,T.float32(256));T.pipe_barrier('V')
       T.tile.sub(rounded,low,temp);T.pipe_barrier('V')
       emit('rq_half',half_buf.access_ptr('w'),rounded.access_ptr('r'),2048);T.pipe_barrier('V')
       emit('rq_i8',small.access_ptr('w'),half_buf.access_ptr('r'),2048)
       T.set_flag('V','MTE3',0);T.wait_flag('V','MTE3',0)
       limb=4-part*2-digit
       if side==0:emit('rq_store',QA.access_ptr('w',offset=limb*n*n+core*n),small.access_ptr('r'),owned,n,(blocks-1)*n)
       else:emit('rq_store',QY.access_ptr('w',offset=core*5*n+limb*n),small.access_ptr('r'),owned,n,(blocks*5-1)*n)
       T.set_flag('MTE3','V',0);T.wait_flag('MTE3','V',0)
       emit('pv_copy',low.access_ptr('w'),carry.access_ptr('r'),2048);T.pipe_barrier('V')
      if part==0:
       T.tile.add(low,high,low);T.pipe_barrier('V')
     emit('rq_half',half_buf.access_ptr('w'),low.access_ptr('r'),2048);T.pipe_barrier('V')
     emit('rq_i8',small.access_ptr('w'),half_buf.access_ptr('r'),2048)
     T.set_flag('V','MTE3',0);T.wait_flag('V','MTE3',0)
     if side==0:emit('rq_store',QA.access_ptr('w',offset=core*n),small.access_ptr('r'),owned,n,(blocks-1)*n)
     else:emit('rq_store',QY.access_ptr('w',offset=core*5*n),small.access_ptr('r'),owned,n,(blocks*5-1)*n)
     T.set_flag('MTE3','V',0);T.wait_flag('MTE3','V',0)
 return kernel
