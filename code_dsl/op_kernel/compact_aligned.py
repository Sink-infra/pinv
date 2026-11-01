"""Bounded raw-input inverse for a full compact Vector tile.

The input Frobenius norm is bounded by n*n*max(abs(A))**2. The exact
inverse norm still certifies the singular-value cutoff. A failed range,
pivot or cutoff check enters the existing pivoted continuation.
"""
from code_dsl.op_kernel.intrinsics import emit
from code_dsl.op_kernel.static_reduce import make_reduction
from code_dsl.op_kernel.normalization import ceil_power_bits


def make_aligned_macro(full_tile=False,unroll=1):
    if not full_tile:
        raise ValueError("aligned compact requires its full tile")
    import tilelang.language as T
    reduce_max=make_reduction('max')
    @T.macro
    def operation(A,X,ub,indices,order,accepted,offset,runtime_n,atol,rtol):
        n=16 if full_tile else runtime_n
        count=n*40
        valid=T.alloc_var('int32');valid=1
        emit('pv_fill',ub.access_ptr('w'),T.float32(0),count)
        T.pipe_barrier('V')
        T.set_flag('V','MTE2',0);T.wait_flag('V','MTE2',0)
        if n+1==16:
            emit('pv_load_rows',ub.access_ptr('w',offset=640),A.access_ptr('r',offset=offset),1,n*n,0,0)
        else:
            emit('pv_load_rows',ub.access_ptr('w'),A.access_ptr('r',offset=offset),n,n,0,24,n%8!=0)
        T.set_flag('MTE2','V',0);T.wait_flag('MTE2','V',0)
        if n+1==16:
            for lane in T.unroll(16):
                indices[lane]=T.Cast('uint32',lane*4)
                indices[16+lane]=T.Cast('uint32',lane*60)
            T.set_flag('S','V',0);T.wait_flag('S','V',0)
            emit('bl_index_brcb',ub.access_ptr('w',offset=896),indices.access_ptr('r',offset=16),16)
            T.pipe_barrier('V')
            emit('bl_index_matrix',ub.access_ptr('w',offset=1152),ub.access_ptr('r',offset=896),indices.access_ptr('r'),16,16)
            T.pipe_barrier('V')
            emit('pv_gather_bits',ub.access_ptr('w',offset=896),ub.access_ptr('r',offset=640),ub.access_ptr('r',offset=1152),0,n*16)
            T.pipe_barrier('V')
            emit('cp_zero_pad',ub.access_ptr('w',offset=896),n)
            T.pipe_barrier('V')
            emit('pv_copy_ub_rows',ub.access_ptr('w'),ub.access_ptr('r',offset=896),n,16,0,24)
            T.pipe_barrier('V')
        emit('pv_abs',ub.access_ptr('w',offset=640),ub.access_ptr('r'),count)
        T.pipe_barrier('V')
        reduce_max(ub,1368,640,1336,count)
        T.set_flag('V','S',0);T.wait_flag('V','S',0)
        magnitude=ub[1368]
        power=T.alloc_var('float32');power=T.float32(1)
        if not (magnitude>=T.float32(0.1) and magnitude<T.float32(65536 if full_tile else 1.7014118e38)):valid=0
        if valid!=0:
            power_word=T.alloc_var('uint32')
            power_word=ceil_power_bits(magnitude)
            pivot_floor=T.alloc_var('float32');pivot_floor=T.float32(0.00390625)
            total=T.alloc_var('float32')
            if full_tile:
                pivot_floor=T.reinterpret('float32',power_word)*T.float32(0.00390625)
                total=T.Cast('float32',n*n)*magnitude*magnitude*T.float32(1.00001)
            else:
                power=T.reinterpret('float32',power_word)
                emit('pv_scale',ub.access_ptr('w'),ub.access_ptr('r'),T.float32(1)/power,count)
                T.pipe_barrier('V')
                emit('pv_mul',ub.access_ptr('w',offset=640),ub.access_ptr('r'),ub.access_ptr('r'),count)
                T.pipe_barrier('V')
                emit('ns_sum',ub.access_ptr('w',offset=1368),ub.access_ptr('r',offset=640),count)
                T.set_flag('V','S',0);T.wait_flag('V','S',0)
                emit('ns_acc',ub.access_ptr('rw'),total.buffer.data)
            for i in T.serial(n):
                ub[i*40+16+i]=T.float32(-1 if full_tile else 1)
                indices[i]=T.Cast('uint32',i*160)
            emit('pv_fill',ub.access_ptr('w',offset=1416),T.float32(0),16)
            T.set_flag('S','V',0);T.wait_flag('S','V',0)
            for column_tile in T.serial(n//unroll):
                for column_lane in T.unroll(unroll):
                    column=column_tile*unroll+column_lane
                    emit('pv_gather',ub.access_ptr('w',offset=1416),ub.access_ptr('r'),indices.access_ptr('r'),column*4,n)
                    T.set_flag('V','S',0);T.wait_flag('V','S',0)
                    divisor=ub[1416+column]
                    if not (T.max(divisor,-divisor)>pivot_floor):
                        valid=0
                        T.loop_break()
                    base=((column+1)//8)*8
                    live=32-base
                    emit('pv_scale',ub.access_ptr('w',offset=column*40+base),ub.access_ptr('r',offset=column*40+base),T.float32(-1)/divisor,live)
                    ub[1416+column]=T.float32(0)
                    T.set_flag('S','V',0);T.wait_flag('S','V',0)
                    emit('pv_brcb_rows',ub.access_ptr('w',offset=672 if full_tile else 640),ub.access_ptr('r',offset=1416),16,40)
                    T.pipe_barrier('V')
                    emit('pv_gj_update',ub.access_ptr('rw',offset=base),ub.access_ptr('r',offset=column*40+base),ub.access_ptr('r',offset=672 if full_tile else 640),live,n,40)
                    T.pipe_barrier('V')
            if valid!=0:
                emit('lv_square_packed_rows',ub.access_ptr('w',offset=640),ub.access_ptr('r',offset=16),16,n,40)
                T.pipe_barrier('V')
                emit('ns_sum',ub.access_ptr('w',offset=1368),ub.access_ptr('r',offset=640),n*16)
                T.set_flag('V','S',0);T.wait_flag('V','S',0)
                inverse_total=T.alloc_var('float32')
                emit('ns_acc',ub.access_ptr('rw'),inverse_total.buffer.data)
                absolute=atol/power
                cutoff=T.max(total*rtol*rtol,absolute*absolute)
                if not (inverse_total>T.float32(0) and inverse_total*cutoff<T.float32(0.81)):valid=0
                if valid!=0:
                    if not full_tile:
                        emit('pv_scale',ub.access_ptr('w'),ub.access_ptr('r'),T.float32(-1)/power,count)
                        T.pipe_barrier('V')
                    if n==15:
                        emit('pv_pack15',ub.access_ptr('w',offset=640),ub.access_ptr('r',offset=16),n,40)
                        T.pipe_barrier('V')
                    T.set_flag('V','MTE3',0);T.wait_flag('V','MTE3',0)
                    T.set_flag('S','MTE3',0);T.wait_flag('S','MTE3',0)
                    if n==15:emit('pv_store_rows',X.access_ptr('w',offset=offset),ub.access_ptr('r',offset=640),1,n*n,0,0)
                    else:emit('pv_store_rows',X.access_ptr('w',offset=offset),ub.access_ptr('r',offset=16),n,n,24,0)
                    T.set_flag('MTE3','V',0);T.wait_flag('MTE3','V',0)
                    T.set_flag('MTE3','MTE2',0);T.wait_flag('MTE3','MTE2',0)
        accepted[0]=valid
    return operation
