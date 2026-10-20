"""Compensated reconstruction of I - A X from exact digit products."""
from code_dsl.op_kernel.kernels import instruction_source
from code_dsl.op_kernel.intrinsics import emit

def decode_kernel():
    import tilelang.language as T
    header=instruction_source()+'''
    namespace tl::ascend {
    __aicore__ inline void rd_load(LocalTensor<int32_t> d,GlobalTensor<int32_t> a,LocalTensor<int32_t>,int rows,int cols,int gap){DataCopyPad(d,a,DataCopyExtParams{uint16_t(rows),uint32_t(cols*4),uint32_t(gap*4),0,0},DataCopyPadExtParams<int32_t>{false,0,0,0});}
    __aicore__ inline void rd_store_half(GlobalTensor<half> d,LocalTensor<half> a,GlobalTensor<half>,int rows,int cols,int gap){DataCopyPad(d,a,DataCopyExtParams{uint16_t(rows),uint32_t(cols*2),0,uint32_t(gap*2),0});}
    }
    '''
    @T.macro
    def add_term(high,low,part,total,bv,av,error,alt):
     T.tile.add(total,high,part);T.pipe_barrier('V')
     T.tile.sub(bv,total,high);T.pipe_barrier('V')
     T.tile.sub(av,total,bv);T.pipe_barrier('V')
     T.tile.sub(error,high,av);T.tile.sub(alt,part,bv);T.pipe_barrier('V')
     T.tile.add(error,error,alt);T.pipe_barrier('V')
     T.tile.add(low,low,error);emit('pv_copy',high.access_ptr('w'),total.access_ptr('r'),2048);T.pipe_barrier('V')
    @T.prim_func
    def decode(P:T.Tensor((1,),'int32'),Y:T.Tensor((1,),'float32'),R:T.Tensor((1,),'float32'),YH:T.Tensor((1,),'float16'),RH:T.Tensor((1,),'float16'),
               blocks:T.int32,n:T.int32,job:T.int32,scale_a:T.float32,scale_y:T.float32):
     T.func_attr({'global_symbol':'linalg_pinv_tilelang','pinv.scope':'compensated residual decode diagnostic'})
     with T.Kernel(blocks,threads=1,is_npu=True) as core:
      T.import_source(header)
      raw=T.alloc_ub((2048,),'int32');part=T.alloc_ub((2048,),'float32')
      high=T.alloc_ub((2048,),'float32');low=T.alloc_ub((2048,),'float32')
      total=T.alloc_ub((2048,),'float32');bv=T.alloc_ub((2048,),'float32');av=T.alloc_ub((2048,),'float32')
      error=T.alloc_ub((2048,),'float32');alt=T.alloc_ub((2048,),'float32');cache=T.alloc_ub((2048,),'float32')
      roundtrip=T.alloc_ub((2048,),'int32');remainder=T.alloc_ub((2048,),'float32')
      h_y=T.alloc_ub((2048,),'float16');h_r=T.alloc_ub((2048,),'float16')
      with T.Scope('V'):
       T.reads(P[0],Y[0]);T.writes(R[0],YH[0],RH[0])
       owned=(n+blocks-1-core)//blocks
       emit('pv_fill',high.access_ptr('w'),T.float32(0),2048)
       emit('pv_fill',low.access_ptr('w'),T.float32(0),2048)
       T.pipe_barrier('V');T.tile.cast(raw,low,'CAST_RINT',2048)
       T.set_flag('V','S',0);T.wait_flag('V','S',0)
       for row in T.serial(owned):high[row*n+core+row*blocks]=T.float32(1)
       T.set_flag('S','V',0);T.wait_flag('S','V',0)
       weight=T.alloc_var('float32');weight=scale_a/T.float32(256)
       weight*=scale_y/T.float32(256)
       for diagonal in T.serial(9):
        T.set_flag('V','MTE2',0);T.wait_flag('V','MTE2',0)
        emit('rd_load',raw.access_ptr('w'),P.access_ptr('r',offset=(diagonal*n+core)*n),owned,n,(blocks-1)*n)
        T.set_flag('MTE2','V',0);T.wait_flag('MTE2','V',0)
        T.tile.cast(part,raw,'CAST_RINT',2048);T.pipe_barrier('V')
        if diagonal==4:
         T.tile.cast(roundtrip,part,'CAST_RINT',2048);T.pipe_barrier('V')
         T.tile.sub(raw,raw,roundtrip);T.pipe_barrier('V')
         T.tile.cast(remainder,raw,'CAST_RINT',2048);T.pipe_barrier('V')
        T.tile.mul(part,part,-weight);T.pipe_barrier('V')
        add_term(high,low,part,total,bv,av,error,alt)
        if diagonal==4:
         T.tile.mul(part,remainder,-weight);T.pipe_barrier('V')
         add_term(high,low,part,total,bv,av,error,alt)
        weight*=T.float32(1/256)
       T.tile.add(high,high,low);T.pipe_barrier('V')
       T.set_flag('V','MTE3',0);T.wait_flag('V','MTE3',0)
       emit('pv_store_rows',R.access_ptr('w',offset=core*n),high.access_ptr('r'),owned,n,0,(blocks-1)*n)
       T.tile.mul(part,high,T.float32(4096));T.pipe_barrier('V');T.tile.cast(h_r,part,'CAST_NONE',2048)
       emit('pv_fill',cache.access_ptr('w'),T.float32(0),2048);T.pipe_barrier('V')
       T.set_flag('V','MTE2',0);T.wait_flag('V','MTE2',0)
       emit('pv_load_rows',cache.access_ptr('w'),Y.access_ptr('r',offset=job*n*n+core*n),owned,n,(blocks-1)*n,0)
       T.set_flag('MTE2','V',0);T.wait_flag('MTE2','V',0)
       T.tile.mul(cache,cache,T.float32(1)/scale_y);T.pipe_barrier('V');T.tile.cast(h_y,cache,'CAST_NONE',2048)
       T.set_flag('V','MTE3',0);T.wait_flag('V','MTE3',0)
       emit('rd_store_half',YH.access_ptr('w',offset=core*n),h_y.access_ptr('r'),owned,n,(blocks-1)*n)
       emit('rd_store_half',RH.access_ptr('w',offset=core*n),h_r.access_ptr('r'),owned,n,(blocks-1)*n)
       T.set_flag('MTE3','V',0);T.wait_flag('MTE3','V',0)
    return decode
