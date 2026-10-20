"""Small resident paths. All numerical decisions and elimination loops are DSL.

The aligned light macro mirrors CompactQuickLight; it reports acceptance to
its caller, which must invoke the precise/truncated continuation on rejection.
It is not exported as a complete pseudoinverse until that continuation exists.
"""
from code_dsl.op_kernel.intrinsics import emit
from code_dsl.op_kernel.control import admission_writer
from code_dsl.op_kernel.normalization import ceil_power_bits


def make_light_macro(unroll=1,scalar_admission=False):
    import tilelang.language as T
    store_admission=admission_writer(scalar_admission)

    @T.macro
    def light(A,X,ub,indices,order,accepted,offset,n,atol,rtol):
        # Strategy tile capacity: 16. Shape values remain runtime arguments of
        # the containing shared kernel; no per-input compilation is used.
        pitch=40
        count=n*pitch
        # The 16-entry permutation fits in one scalar register (four bits per
        # row), so it need not make a round trip through vector UB storage.
        permutation=T.alloc_var('uint64');permutation=T.uint64(0)
        valid=T.alloc_var('int32')
        valid=1
        emit('pv_fill',ub.access_ptr('w'),T.float32(0),count)
        T.pipe_barrier('V')
        T.set_flag('V','MTE2',0)
        T.wait_flag('V','MTE2',0)
        emit('pv_load_rows',ub.access_ptr('w'),A.access_ptr('r',offset=offset),n,n,0,24,n%8!=0)
        T.set_flag('MTE2','V',0)
        T.wait_flag('MTE2','V',0)
        emit('pv_abs',ub.access_ptr('w',offset=640),ub.access_ptr('r'),count)
        T.pipe_barrier('V')
        emit('pv_reduce_max',ub.access_ptr('w',offset=1368),ub.access_ptr('r',offset=640),ub.access_ptr('rw',offset=1336),count)
        T.set_flag('V','S',0)
        T.wait_flag('V','S',0)
        scale=ub[1368]
        power=T.alloc_var('float32')
        power=T.float32(1)
        if not (scale>=T.float32(0.1) and scale<T.float32(1.7014118e38)):
            valid=0
        if valid!=0:
            power_word=T.alloc_var('uint32')
            power_word=ceil_power_bits(scale)
            power=T.reinterpret('float32',power_word)
            emit('pv_scale',ub.access_ptr('w'),ub.access_ptr('r'),T.float32(1)/power,count)
            T.pipe_barrier('V')
            emit('pv_mul',ub.access_ptr('w',offset=640),ub.access_ptr('r'),ub.access_ptr('r'),count)
            T.pipe_barrier('V')
            emit('pv_reduce_sum',ub.access_ptr('w',offset=1368),ub.access_ptr('r',offset=640),ub.access_ptr('rw',offset=1336),count)
            T.set_flag('V','S',0)
            T.wait_flag('V','S',0)
            total=ub[1368]
            for i in T.serial(n):
                ub[i*40+16+i]=T.float32(1)
                index_word=T.alloc_var('uint32')
                index_word=T.Cast('uint32',i*160)
                ub[1400+i]=T.reinterpret('float32',index_word)
            emit('pv_mask_begin',ub.access_ptr('rw'),n)
            emit('pv_fill',ub.access_ptr('w',offset=1416),T.float32(0),16)
            T.pipe_barrier('V')
            T.set_flag('S','V',0)
            T.wait_flag('S','V',0)
            for pivot_tile in T.serial(n//unroll):
                for lane in T.unroll(unroll):
                    column=pivot_tile*unroll+lane
                    emit('pv_gather_bits', ub.access_ptr('w', offset=1416), ub.access_ptr('r'), ub.access_ptr('r',offset=1400), column * 4, n)
                    T.pipe_barrier('V')
                    emit('pv_abs', ub.access_ptr('w', offset=1320), ub.access_ptr('r', offset=1416), 16)
                    T.pipe_barrier('V')
                    emit('pv_mask_reduce', ub.access_ptr('w', offset=1368), ub.access_ptr('r', offset=1320))
                    T.set_flag('V', 'S', 0)
                    T.wait_flag('V', 'S', 0)
                    best=T.alloc_var('float32')
                    pivot=T.alloc_var('int32')
                    emit('pv_reduce_scalars',ub.access_ptr('rw'),best.buffer.data,pivot.buffer.data)
                    if not best>T.float32(0):
                        valid = 0
                        T.loop_break()
                    emit('pv_mask_remove', ub.access_ptr('rw'), pivot)
                    shift=T.alloc_var('uint32');shift=T.Cast('uint32',pivot*4)
                    permutation=permutation | (T.Cast('uint64',column)<<shift)
                    factor = T.float32(-1) / ub[pivot * 40 + column]
                    emit('pv_scale', ub.access_ptr('w', offset=pivot * 40), ub.access_ptr('r', offset=pivot * 40), factor, 32)
                    T.pipe_barrier('V')
                    ub[1416 + pivot] = T.float32(0)
                    T.set_flag('S', 'V', 0)
                    T.wait_flag('S', 'V', 0)
                    emit('pv_brcb_rows', ub.access_ptr('w', offset=640), ub.access_ptr('r', offset=1416), 16, 40)
                    T.pipe_barrier('V')
                    emit('pv_gj_update', ub.access_ptr('rw'), ub.access_ptr('r', offset=pivot * 40), ub.access_ptr('r', offset=640), 32, n, 40)
                    T.pipe_barrier('V')
            if valid!=0:
                for tail in T.serial(n%unroll):
                    column=n//unroll*unroll+tail
                    emit('pv_gather_bits', ub.access_ptr('w', offset=1416), ub.access_ptr('r'), ub.access_ptr('r',offset=1400), column * 4, n)
                    T.pipe_barrier('V')
                    emit('pv_abs', ub.access_ptr('w', offset=1320), ub.access_ptr('r', offset=1416), 16)
                    T.pipe_barrier('V')
                    emit('pv_mask_reduce', ub.access_ptr('w', offset=1368), ub.access_ptr('r', offset=1320))
                    T.set_flag('V', 'S', 0)
                    T.wait_flag('V', 'S', 0)
                    best=T.alloc_var('float32')
                    pivot=T.alloc_var('int32')
                    emit('pv_reduce_scalars',ub.access_ptr('rw'),best.buffer.data,pivot.buffer.data)
                    if not best>T.float32(0):
                        valid = 0
                        T.loop_break()
                    emit('pv_mask_remove', ub.access_ptr('rw'), pivot)
                    shift=T.alloc_var('uint32');shift=T.Cast('uint32',pivot*4)
                    permutation=permutation | (T.Cast('uint64',column)<<shift)
                    factor = T.float32(-1) / ub[pivot * 40 + column]
                    emit('pv_scale', ub.access_ptr('w', offset=pivot * 40), ub.access_ptr('r', offset=pivot * 40), factor, 32)
                    T.pipe_barrier('V')
                    ub[1416 + pivot] = T.float32(0)
                    T.set_flag('S', 'V', 0)
                    T.wait_flag('S', 'V', 0)
                    emit('pv_brcb_rows', ub.access_ptr('w', offset=640), ub.access_ptr('r', offset=1416), 16, 40)
                    T.pipe_barrier('V')
                    emit('pv_gj_update', ub.access_ptr('rw'), ub.access_ptr('r', offset=pivot * 40), ub.access_ptr('r', offset=640), 32, n, 40)
                    T.pipe_barrier('V')
            if valid!=0:
                emit('pv_fill',ub.access_ptr('w',offset=640),T.float32(0),count)
                T.pipe_barrier('V')
                emit('pv_square_padded_rows',ub.access_ptr('w',offset=640),ub.access_ptr('r',offset=16),16,n,40)
                T.pipe_barrier('V')
                emit('pv_reduce_sum',ub.access_ptr('w',offset=1368),ub.access_ptr('r',offset=640),ub.access_ptr('rw',offset=1336),count)
                T.set_flag('V','S',0)
                T.wait_flag('V','S',0)
                inverse_total=ub[1368]
                ub[1368]=total
                T.set_flag('S','V',0)
                T.wait_flag('S','V',0)
                emit('pv_sqrt',ub.access_ptr('w',offset=1368),ub.access_ptr('r',offset=1368),1)
                T.set_flag('V','S',0)
                T.wait_flag('V','S',0)
                cutoff=T.max(ub[1368]*rtol,atol/power)
                if not (inverse_total>T.float32(0) and inverse_total*cutoff*cutoff<T.float32(0.81)):
                    valid=0
                if valid!=0:
                    emit('pv_scale',ub.access_ptr('w'),ub.access_ptr('r'),T.float32(-1)/power,count)
                    T.pipe_barrier('V')
                    for row in T.serial(n):
                        shift=T.alloc_var('uint32');shift=T.Cast('uint32',row*4)
                        target=T.Cast('int32',(permutation>>shift)&T.uint64(15))
                        emit('pv_copy',ub.access_ptr('w',offset=640+target*40),ub.access_ptr('r',offset=row*40),40)
                    T.pipe_barrier('V')
                    if n==15:
                        emit('pv_pack15',ub.access_ptr('w'),ub.access_ptr('r',offset=656),n,40)
                        T.pipe_barrier('V')
                    T.set_flag('V','MTE3',0)
                    T.wait_flag('V','MTE3',0)
                    T.set_flag('S','MTE3',0)
                    T.wait_flag('S','MTE3',0)
                    if n==15:
                        emit('pv_store_rows',X.access_ptr('w',offset=offset),ub.access_ptr('r'),1,n*n,0,0)
                    else:
                        emit('pv_store_rows',X.access_ptr('w',offset=offset),ub.access_ptr('r',offset=656),n,n,24,0)
                    T.set_flag('MTE3','V',0)
                    T.wait_flag('MTE3','V',0)
                    T.set_flag('MTE3','MTE2',0)
                    T.wait_flag('MTE3','MTE2',0)
        store_admission(accepted,valid)
    return light

