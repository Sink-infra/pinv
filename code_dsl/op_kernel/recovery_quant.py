"""Five balanced INT8 digits and exact discarded-digit norm, scheduled in TileLang."""
from code_dsl.op_kernel.kernels import instruction_source
from code_dsl.op_kernel.intrinsics import emit

def quant_kernel():
    import tilelang.language as T
    header=instruction_source()+'''
    namespace tl::ascend {
    __aicore__ inline void rq_half(LocalTensor<half> d,LocalTensor<float> a,LocalTensor<half>,int n){Cast(d,a,RoundMode::CAST_NONE,n);}
    __aicore__ inline void rq_float(LocalTensor<float> d,LocalTensor<half> a,LocalTensor<float>,int n){Cast(d,a,RoundMode::CAST_NONE,n);}
    __aicore__ inline void rq_i8(LocalTensor<int8_t> d,LocalTensor<half> a,LocalTensor<int8_t>,int n){Cast(d,a,RoundMode::CAST_RINT,n);}
    __aicore__ inline void rq_store(GlobalTensor<int8_t> d,LocalTensor<int8_t> a,GlobalTensor<int8_t>,int rows,int cols,int gap){DataCopyPad(d,a,DataCopyExtParams{uint16_t(rows),uint32_t(cols),0,uint32_t(gap),0});}
    }
    '''
    @T.prim_func
    def kernel(A:T.Tensor((1,),'float32'),Y:T.Tensor((1,),'float32'),QA:T.Tensor((1,),'int8'),QY:T.Tensor((1,),'int8'),QN:T.Tensor((1,),'float32'),
               blocks:T.int32,n:T.int32,job:T.int32,scale_a:T.float32,scale_y:T.float32):
     T.func_attr({'global_symbol':'linalg_pinv_tilelang','pinv.scope':'residual digit quantization diagnostic'})
     with T.Kernel(blocks,threads=1,is_npu=True) as core:
      T.import_source(header)
      values=T.alloc_ub((2048,),'float32');integers=T.alloc_ub((2048,),'int32')
      rounded=T.alloc_ub((2048,),'float32');carry=T.alloc_ub((2048,),'float32');temp=T.alloc_ub((2048,),'float32')
      planes=T.alloc_ub((10240,),'float16');small=T.alloc_ub((2048,),'int8')
      with T.Scope('V'):
       T.reads(A[0],Y[0]);T.writes(QA[0],QY[0],QN[0])
       owned=(n+blocks-1-core)//blocks;count=owned*n
       for side in T.unroll(2):
        emit('pv_fill',values.access_ptr('w'),T.float32(0),2048);T.pipe_barrier('V')
        T.set_flag('V','MTE2',0);T.wait_flag('V','MTE2',0)
        if side==0:
         emit('pv_load_rows',values.access_ptr('w'),A.access_ptr('r',offset=job*n*n+core*n),owned,n,(blocks-1)*n,0)
        else:
         emit('pv_load_rows',values.access_ptr('w'),Y.access_ptr('r',offset=job*n*n+core*n),owned,n,(blocks-1)*n,0)
        T.set_flag('MTE2','V',0);T.wait_flag('MTE2','V',0)
        if side==0:T.tile.mul(values,values,T.float32(256)/scale_a)
        else:T.tile.mul(values,values,T.float32(256)/scale_y)
        T.pipe_barrier('V')
        for limb in T.serial(5):
         T.tile.cast(integers,values,'CAST_RINT',2048);T.pipe_barrier('V')
         T.tile.cast(rounded,integers,'CAST_RINT',2048);T.pipe_barrier('V')
         T.tile.sub(values,values,rounded);T.pipe_barrier('V')
         T.tile.mul(values,values,T.float32(256));T.pipe_barrier('V')
         emit('rq_half',planes.access_ptr('w',offset=limb*2048),rounded.access_ptr('r'),2048);T.pipe_barrier('V')
        T.tile.mul(temp,values,values);T.pipe_barrier('V')
        emit('pv_reduce_sum',carry.access_ptr('w'),temp.access_ptr('r'),rounded.access_ptr('rw'),2048)
        T.set_flag('V','MTE3',0);T.wait_flag('V','MTE3',0)
        emit('pv_store_rows',QN.access_ptr('w',offset=core*16+side*8),carry.access_ptr('r'),1,8,0,0)
        T.set_flag('MTE3','V',0);T.wait_flag('MTE3','V',0)
        for rev in T.serial(4):
         limb=4-rev
         emit('rq_float',rounded.access_ptr('w'),planes.access_ptr('r',offset=limb*2048),2048);T.pipe_barrier('V')
         T.tile.add(carry,rounded,T.float32(-127));T.pipe_barrier('V')
         T.tile.max(carry,carry,T.float32(0));T.pipe_barrier('V')
         T.tile.min(carry,carry,T.float32(1));T.pipe_barrier('V')
         T.tile.mul(temp,carry,T.float32(256));T.pipe_barrier('V')
         T.tile.sub(rounded,rounded,temp);T.pipe_barrier('V')
         emit('rq_half',planes.access_ptr('w',offset=limb*2048),rounded.access_ptr('r'),2048);T.pipe_barrier('V')
         emit('rq_float',rounded.access_ptr('w'),planes.access_ptr('r',offset=(limb-1)*2048),2048);T.pipe_barrier('V')
         T.tile.add(rounded,rounded,carry);T.pipe_barrier('V')
         emit('rq_half',planes.access_ptr('w',offset=(limb-1)*2048),rounded.access_ptr('r'),2048);T.pipe_barrier('V')
        for limb in T.serial(5):
         emit('rq_i8',small.access_ptr('w'),planes.access_ptr('r',offset=limb*2048),2048)
         T.set_flag('V','MTE3',0);T.wait_flag('V','MTE3',0)
         if side==0:emit('rq_store',QA.access_ptr('w',offset=limb*n*n+core*n),small.access_ptr('r'),owned,n,(blocks-1)*n)
         else:emit('rq_store',QY.access_ptr('w',offset=core*5*n+limb*n),small.access_ptr('r'),owned,n,(blocks*5-1)*n)
         T.set_flag('MTE3','V',0);T.wait_flag('MTE3','V',0)
    return kernel
