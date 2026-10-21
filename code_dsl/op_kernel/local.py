"""Resident logical-pivot inverse with the original packed lane schedule.

Dimensions and padding are runtime values. Only elementary instructions are
external; lazy identity initialization, pivot order, lane consumption and the
Frobenius cutoff admission gate are DSL control flow.
"""
from code_dsl.op_kernel.intrinsics import emit
from code_dsl.op_kernel.control import admission_writer
from code_dsl.op_kernel.normalization import ceil_power_bits


def _make_quick_schedule(tile_pad, factor_pitch, unroll, full_tile=False,scalar_admission=False):
    import tilelang.language as T
    store_admission=admission_writer(scalar_admission)

    @T.macro
    def quick(A,X,pool,indices,order,accepted,offset,runtime_n,atol,rtol):
        # Runtime shapes share one of two vector-width tiles. Dispatch happens
        # outside elimination; all lane strides and vector widths are DSL tile
        # constants, rather than descriptors cached by an external C++ helper.
        pad=tile_pad
        pitch=2*tile_pad+8
        n=tile_pad if full_tile else runtime_n
        count=T.alloc_var('int32');count=n*pitch
        # The tile may be wider than ceildiv(n,8)*8. DMA block strides
        # advance from the actual transfer padding, not from tile capacity.
        dma_pad=T.alloc_var('int32');dma_pad=T.ceildiv(n,8)*8
        lo=8704
        factors=17408
        product=18432
        scratch=18496
        red=19008
        norms=19040
        valid=T.alloc_var('int32');valid=1
        emit('pv_fill',pool.access_ptr('w'),T.float32(0),count)
        T.pipe_barrier('V')
        T.set_flag('V','MTE2',0);T.wait_flag('V','MTE2',0)
        emit('pv_load_rows',pool.access_ptr('w'),A.access_ptr('r',offset=offset),n,n,0,pitch-dma_pad,n%8!=0)
        T.set_flag('MTE2','V',0);T.wait_flag('MTE2','V',0)
        emit('pv_abs',pool.access_ptr('w',offset=lo),pool.access_ptr('r'),count)
        T.pipe_barrier('V')
        emit('pv_reduce_max',pool.access_ptr('w',offset=red),pool.access_ptr('r',offset=lo),pool.access_ptr('rw',offset=scratch),count)
        T.set_flag('V','S',0);T.wait_flag('V','S',0)
        magnitude=pool[red]
        if not (magnitude>=T.float32(0.1) and magnitude<T.float32(1.7014118e38)):valid=0
        power=T.alloc_var('float32');power=T.float32(1)
        if valid!=0:
            power_word=T.alloc_var('uint32')
            power_word=ceil_power_bits(magnitude)
            power=T.reinterpret('float32',power_word)
            emit('pv_scale',pool.access_ptr('w'),pool.access_ptr('r'),T.float32(1)/power,count)
            T.pipe_barrier('V')
            emit('pv_mul',pool.access_ptr('w',offset=lo),pool.access_ptr('r'),pool.access_ptr('r'),count)
            T.pipe_barrier('V')
            emit('pv_reduce_sum',pool.access_ptr('w',offset=red),pool.access_ptr('r',offset=lo),pool.access_ptr('rw',offset=scratch),count)
            T.set_flag('V','S',0);T.wait_flag('V','S',0)
            total=pool[red]
            for i in T.serial(pad):
                # Gather consumes exactly n offsets; padded slots are never
                # read. An unconditional ramp also avoids a per-lane branch.
                indices[i]=T.Cast('uint32',i*pitch*4)
                if pad>=56 and not full_tile:
                    if i>=n:
                        order[i]=T.Cast('uint32',0);order[pad+i]=T.Cast('uint32',0)
            emit('pv_fill',pool.access_ptr('w',offset=norms),T.float32(0),pad)
            T.pipe_barrier('V')
            emit('pv_mask_begin',pool.access_ptr('rw'),n)
            live=tile_pad+8
            T.set_flag('S','V',0);T.wait_flag('S','V',0)
            for pivot_tile in T.serial(n//unroll):
                for lane in T.unroll(unroll):
                    p=pivot_tile*unroll+lane
                    emit('pv_gather', pool.access_ptr('w', offset=norms), pool.access_ptr('r'), indices.access_ptr('r'), p * 4, n)
                    T.pipe_barrier('V')
                    emit('pv_abs', pool.access_ptr('w', offset=product), pool.access_ptr('r', offset=norms), pad)
                    T.pipe_barrier('V')
                    emit('pv_mask_reduce', pool.access_ptr('w', offset=red), pool.access_ptr('r', offset=product))
                    T.set_flag('V', 'S', 0)
                    T.wait_flag('V', 'S', 0)
                    best=T.alloc_var('float32')
                    pivot=T.alloc_var('int32')
                    emit('pv_reduce_scalars',pool.access_ptr('rw'),best.buffer.data,pivot.buffer.data)
                    if not best>T.float32(0):
                        valid = 0
                        T.loop_break()
                    emit('pv_mask_remove', pool.access_ptr('rw'), pivot)
                    order[pivot] = T.Cast('uint32',p*4)
                    if pad>=56:order[pad+p]=T.Cast('uint32',(pivot*pitch+pad)*4)
                    consumed = T.alloc_var('int32')
                    consumed = 0
                    base = T.alloc_var('int32')
                    base = p // 8 * 8
                    if live > 64 and (p % 8 == 7 or p + 1 == n):
                        consumed = 1
                        base += 8
                    divisor = pool[norms + pivot]
                    pool[norms + pivot] = T.float32(0)
                    pool[pivot * pitch + pad + p] = T.float32(1)
                    T.set_flag('S', 'V', 0)
                    T.wait_flag('S', 'V', 0)
                    active = T.alloc_var('int32')
                    active = live
                    if consumed != 0:
                        active -= 8
                    emit('pv_scale', pool.access_ptr('w', offset=pivot * pitch + base), pool.access_ptr('r', offset=pivot * pitch + base), T.float32(-1) / divisor, active)
                    emit('pv_brcb_rows', pool.access_ptr('w', offset=factors), pool.access_ptr('r', offset=norms), pad, factor_pitch)
                    T.pipe_barrier('V')
                    emit('pv_mad_strided', pool.access_ptr('rw', offset=base), pool.access_ptr('r', offset=pivot * pitch + base), pool.access_ptr('r', offset=factors), T.min(live, 64), n, 1, 1, 0, pitch//8, 0, factor_pitch//8)
                    if live > 64 and consumed == 0:
                        groups = n // 8
                        rest = n % 8
                        if groups > 0:
                            emit('pv_mad_strided', pool.access_ptr('rw', offset=base + 64), pool.access_ptr('r', offset=pivot * pitch + base + 64), pool.access_ptr('r', offset=factors), 64, groups, pitch//8, 0, factor_pitch//8, pitch, 0, factor_pitch)
                        if rest > 0:
                            emit('pv_mad_strided', pool.access_ptr('rw', offset=groups * 8 * pitch + base + 64), pool.access_ptr('r', offset=pivot * pitch + base + 64), pool.access_ptr('r', offset=factors + groups * 8 * factor_pitch), 8, rest, 1, 1, 0, pitch//8, 0, factor_pitch//8)
                    T.pipe_barrier('V')
            if valid!=0:
                for tail in T.serial(n%unroll):
                    p=n//unroll*unroll+tail
                    emit('pv_gather', pool.access_ptr('w', offset=norms), pool.access_ptr('r'), indices.access_ptr('r'), p * 4, n)
                    T.pipe_barrier('V')
                    emit('pv_abs', pool.access_ptr('w', offset=product), pool.access_ptr('r', offset=norms), pad)
                    T.pipe_barrier('V')
                    emit('pv_mask_reduce', pool.access_ptr('w', offset=red), pool.access_ptr('r', offset=product))
                    T.set_flag('V', 'S', 0)
                    T.wait_flag('V', 'S', 0)
                    best=T.alloc_var('float32')
                    pivot=T.alloc_var('int32')
                    emit('pv_reduce_scalars',pool.access_ptr('rw'),best.buffer.data,pivot.buffer.data)
                    if not best>T.float32(0):
                        valid = 0
                        T.loop_break()
                    emit('pv_mask_remove', pool.access_ptr('rw'), pivot)
                    order[pivot] = T.Cast('uint32',p*4)
                    if pad>=56:order[pad+p]=T.Cast('uint32',(pivot*pitch+pad)*4)
                    consumed = T.alloc_var('int32')
                    consumed = 0
                    base = T.alloc_var('int32')
                    base = p // 8 * 8
                    if live > 64 and (p % 8 == 7 or p + 1 == n):
                        consumed = 1
                        base += 8
                    divisor = pool[norms + pivot]
                    pool[norms + pivot] = T.float32(0)
                    pool[pivot * pitch + pad + p] = T.float32(1)
                    T.set_flag('S', 'V', 0)
                    T.wait_flag('S', 'V', 0)
                    active = T.alloc_var('int32')
                    active = live
                    if consumed != 0:
                        active -= 8
                    emit('pv_scale', pool.access_ptr('w', offset=pivot * pitch + base), pool.access_ptr('r', offset=pivot * pitch + base), T.float32(-1) / divisor, active)
                    emit('pv_brcb_rows', pool.access_ptr('w', offset=factors), pool.access_ptr('r', offset=norms), pad, factor_pitch)
                    T.pipe_barrier('V')
                    emit('pv_mad_strided', pool.access_ptr('rw', offset=base), pool.access_ptr('r', offset=pivot * pitch + base), pool.access_ptr('r', offset=factors), T.min(live, 64), n, 1, 1, 0, pitch//8, 0, factor_pitch//8)
                    if live > 64 and consumed == 0:
                        groups = n // 8
                        rest = n % 8
                        if groups > 0:
                            emit('pv_mad_strided', pool.access_ptr('rw', offset=base + 64), pool.access_ptr('r', offset=pivot * pitch + base + 64), pool.access_ptr('r', offset=factors), 64, groups, pitch//8, 0, factor_pitch//8, pitch, 0, factor_pitch)
                        if rest > 0:
                            emit('pv_mad_strided', pool.access_ptr('rw', offset=groups * 8 * pitch + base + 64), pool.access_ptr('r', offset=pivot * pitch + base + 64), pool.access_ptr('r', offset=factors + groups * 8 * factor_pitch), 8, rest, 1, 1, 0, pitch//8, 0, factor_pitch//8)
                    T.pipe_barrier('V')
            if valid!=0:
                emit('pv_fill',pool.access_ptr('w',offset=lo),T.float32(0),count)
                T.pipe_barrier('V')
                emit('pv_square_padded_rows',pool.access_ptr('w',offset=lo),pool.access_ptr('r',offset=pad),pad,n,pitch)
                T.pipe_barrier('V')
                emit('pv_reduce_sum',pool.access_ptr('w',offset=red),pool.access_ptr('r',offset=lo),pool.access_ptr('rw',offset=scratch),count)
                T.set_flag('V','S',0);T.wait_flag('V','S',0)
                inverse_total=pool[red]
                pool[red]=total
                T.set_flag('S','V',0);T.wait_flag('S','V',0)
                emit('pv_sqrt',pool.access_ptr('w',offset=red),pool.access_ptr('r',offset=red),1)
                T.set_flag('V','S',0);T.wait_flag('V','S',0)
                cutoff=T.max(pool[red]*rtol,atol/power)
                if not (inverse_total>T.float32(0) and inverse_total*cutoff*cutoff<T.float32(0.81)):
                    valid=0
                else:
                    emit('pv_scale',pool.access_ptr('w'),pool.access_ptr('r'),T.float32(-1)/power,count)
                    T.pipe_barrier('V')
                    if pad>=56:
                        # Output and offset tiles fit the dead space before factors.
                        T.set_flag('S','V',0);T.wait_flag('S','V',0)
                        emit('bl_index_brcb',pool.access_ptr('w',offset=factors),order.access_ptr('r',offset=pad),pad)
                        T.pipe_barrier('V')
                        emit('bl_index_matrix',pool.access_ptr('w',offset=lo+pad*pad),pool.access_ptr('r',offset=factors),order.access_ptr('r'),pad,pad)
                        T.pipe_barrier('V')
                        emit('bl_gather',pool.access_ptr('w',offset=lo),pool.access_ptr('r'),pool.access_ptr('r',offset=lo+pad*pad),pad*pad)
                        T.pipe_barrier('V')
                        T.set_flag('V','MTE3',0);T.wait_flag('V','MTE3',0)
                        emit('pv_store_rows',X.access_ptr('w',offset=offset),pool.access_ptr('r',offset=lo),n,n,pad-dma_pad,0)
                        T.set_flag('MTE3','V',0);T.wait_flag('MTE3','V',0)
                        T.set_flag('MTE3','MTE2',0);T.wait_flag('MTE3','MTE2',0)
                    else:
                        # Store byte offsets during pivoting, then transfer the
                        # permutation in bulk on the Vector pipe. This removes
                        # the scalar UB read/write loop from the output schedule.
                        T.set_flag('S','V',0);T.wait_flag('S','V',0)
                        T.copy(order[0:pad],indices[0:pad])
                        T.pipe_barrier('V')
                        # Keep one row->pivot permutation. Write each physical
                        # row into its logical output slot in the UB output tile.
                        for physical_row in T.serial(n):
                            output_row=T.Cast('int32',order[physical_row])//4
                            emit('pv_gather',pool.access_ptr('w',offset=lo+output_row*pitch),pool.access_ptr('r'),indices.access_ptr('r'),(physical_row*pitch+pad)*4,n)
                        T.pipe_barrier('V')
                        T.set_flag('V','MTE3',0);T.wait_flag('V','MTE3',0)
                        T.set_flag('S','MTE3',0);T.wait_flag('S','MTE3',0)
                        emit('pv_store_rows',X.access_ptr('w',offset=offset),pool.access_ptr('r',offset=lo),n,n,pitch-dma_pad,0)
                        T.set_flag('MTE3','V',0);T.wait_flag('MTE3','V',0)
                        T.set_flag('MTE3','MTE2',0);T.wait_flag('MTE3','MTE2',0)
        store_admission(accepted,valid)
    return quick


def make_local_quick_macro(unroll=1,factor_pitch=8,scalar_admission=False):
    import tilelang.language as T
    one_vector=_make_quick_schedule(56,factor_pitch,unroll,scalar_admission=scalar_admission)
    two_vectors=_make_quick_schedule(64,factor_pitch,unroll,scalar_admission=scalar_admission)
    two_vectors_aligned=_make_quick_schedule(64,8,unroll,full_tile=True,scalar_admission=scalar_admission)

    @T.macro
    def quick(A,X,pool,indices,order,accepted,offset,n,atol,rtol):
        if n<=56:
            one_vector(A,X,pool,indices,order,accepted,offset,n,atol,rtol)
        elif n==64:
            two_vectors_aligned(A,X,pool,indices,order,accepted,offset,n,atol,rtol)
        else:
            two_vectors(A,X,pool,indices,order,accepted,offset,n,atol,rtol)
    return quick
