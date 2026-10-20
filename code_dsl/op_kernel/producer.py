"""Runtime full/window pivot producer, expressed and scheduled in TileLang.

The producer overlaps the next Schur panel with the workers consuming the
previous packet. The window transition reloads worker-produced trailing rows.
"""
from code_dsl.op_kernel.intrinsics import emit


def make_producer_macro():
    import tilelang.language as T
    from code_dsl.op_kernel.reduction import make_argmax_macro
    argmax=make_argmax_macro()

    @T.macro
    def producer(A,W,pool,n,stride,all_batch,job,core,window):
        k=T.alloc_var('int32');k=T.ceildiv(n,8)*8
        wide=T.alloc_var('int32');wide=T.ceildiv(k,64)*64
        capacity=T.alloc_var('int32');capacity=n*k
        if window>0:capacity=n*window
        scratch=capacity
        factors=scratch+4096
        vector=factors+8*k
        product=vector+wide
        temporary=product+wide
        red=temporary+32
        indices=red+32
        permutation=indices+k
        records=permutation+k
        metadata=2*all_batch*n*stride+job*8192
        stage=job*n*stride
        offset=job*n*n
        magnitude=T.alloc_var('float32');magnitude=T.float32(0)
        total=T.alloc_var('float32');total=T.float32(0)
        power=T.alloc_var('float32');power=T.float32(1)
        if window==0:
            emit('pv_load_rows',pool.access_ptr('w'),A.access_ptr('r',offset=offset),n,n,0,0)
            T.set_flag('MTE2','V',0);T.wait_flag('MTE2','V',0)
        extent=T.alloc_var('int32');extent=n*k
        if window>0:extent=n*n
        for chunk in T.serial(T.ceildiv(extent,4096)):
            begin=chunk*4096
            count=T.min(extent-begin,4096)
            if window>0:
                T.set_flag('V','MTE2',0);T.wait_flag('V','MTE2',0)
                emit('pv_load_rows',pool.access_ptr('w',offset=scratch),A.access_ptr('r',offset=offset+begin),1,count,0,0)
                T.set_flag('MTE2','V',0);T.wait_flag('MTE2','V',0)
                emit('pv_abs',pool.access_ptr('w',offset=scratch),pool.access_ptr('r',offset=scratch),count)
            else:
                emit('pv_abs',pool.access_ptr('w',offset=scratch),pool.access_ptr('r',offset=begin),count)
            T.pipe_barrier('V')
            emit('pv_reduce_max',pool.access_ptr('w',offset=red),pool.access_ptr('r',offset=scratch),pool.access_ptr('rw',offset=product),count)
            T.set_flag('V','S',0);T.wait_flag('V','S',0)
            magnitude=T.max(magnitude,pool[red])
        valid=T.alloc_var('int32');valid=1
        if not (magnitude>=T.float32(0.1) and magnitude<T.float32(1.7014118e38)):valid=0
        if valid!=0:
            while power<magnitude:power*=T.float32(2)
            while power*T.float32(0.5)>=magnitude:power*=T.float32(0.5)
        if window==0:
            for chunk in T.serial(T.ceildiv(extent,16320)):
                begin=chunk*16320
                emit('pv_scale',pool.access_ptr('w',offset=begin),pool.access_ptr('r',offset=begin),T.float32(1)/power,T.min(extent-begin,16320))
            T.pipe_barrier('V')
        for chunk in T.serial(T.ceildiv(extent,4096)):
            begin=chunk*4096
            count=T.min(extent-begin,4096)
            if window>0:
                T.set_flag('V','MTE2',0);T.wait_flag('V','MTE2',0)
                emit('pv_load_rows',pool.access_ptr('w',offset=scratch),A.access_ptr('r',offset=offset+begin),1,count,0,0)
                T.set_flag('MTE2','V',0);T.wait_flag('MTE2','V',0)
                emit('pv_scale',pool.access_ptr('w',offset=scratch),pool.access_ptr('r',offset=scratch),T.float32(1)/power,count)
                T.pipe_barrier('V')
                emit('pv_mul',pool.access_ptr('w',offset=scratch),pool.access_ptr('r',offset=scratch),pool.access_ptr('r',offset=scratch),count)
            else:
                emit('pv_mul',pool.access_ptr('w',offset=scratch),pool.access_ptr('r',offset=begin),pool.access_ptr('r',offset=begin),count)
            T.pipe_barrier('V')
            emit('pv_reduce_sum',pool.access_ptr('w',offset=red),pool.access_ptr('r',offset=scratch),pool.access_ptr('rw',offset=product),count)
            T.set_flag('V','S',0);T.wait_flag('V','S',0)
            total+=pool[red]
        if window>0:
            emit('pv_load_rows',pool.access_ptr('w'),A.access_ptr('r',offset=offset),n,window,n-window,0)
            T.set_flag('MTE2','V',0);T.wait_flag('MTE2','V',0)
            for chunk in T.serial(T.ceildiv(n*window,16320)):
                begin=chunk*16320
                emit('pv_scale',pool.access_ptr('w',offset=begin),pool.access_ptr('r',offset=begin),T.float32(1)/power,T.min(n*window-begin,16320))
            T.pipe_barrier('V')
        for row in T.serial(k):pool[permutation+row]=T.Cast('float32',row)
        T.set_flag('S','V',0);T.wait_flag('S','V',0)
        size=T.alloc_var('int32');size=n
        pitch=T.alloc_var('int32');pitch=k
        if window>0:pitch=window
        window_end=T.alloc_var('int32');window_end=n
        if window>0:window_end=T.min(n,window)
        for panel in T.serial(T.ceildiv(n,16)):
            first=panel*16
            if window>0 and first==window_end:
                window_end=T.min(n,window_end+window)
                emit('pv_all_barrier',pool.access_ptr('rw'))
                emit('pv_cross_set0',pool.access_ptr('rw'),3)
                emit('pv_cross_wait',pool.access_ptr('rw'),3)
                emit('pv_cross_set0',pool.access_ptr('rw'),2)
                emit('pv_cross_wait',pool.access_ptr('rw'),2)
                if valid!=0:
                    for row in T.serial(size):
                        original=T.Cast('int32',pool[permutation+row])
                        emit('pv_load_rows',pool.access_ptr('w',offset=row*pitch),W.access_ptr('r',offset=stage+original*k+first),1,window_end-first,0,0)
                    T.set_flag('MTE2','V',0);T.wait_flag('MTE2','V',0)
                emit('pv_all_barrier',pool.access_ptr('rw'))
                emit('pv_cross_set0',pool.access_ptr('rw'),4)
                emit('pv_cross_wait',pool.access_ptr('rw'),4)
            if valid!=0:
                emit('pv_iota',pool.access_ptr('w',offset=indices),size)
                T.pipe_barrier('V')
                emit('pv_scale_indices',pool.access_ptr('w',offset=indices),pool.access_ptr('r',offset=indices),pitch*4,size)
                T.pipe_barrier('V')
                block=T.min(size,16)
                for pivot in T.serial(block):
                    remaining=size-pivot
                    emit('pv_gather_offsets',pool.access_ptr('w',offset=vector),pool.access_ptr('r'),pool.access_ptr('r',offset=indices),(pivot*pitch+pivot)*4,remaining)
                    T.pipe_barrier('V')
                    emit('pv_abs',pool.access_ptr('w',offset=product),pool.access_ptr('r',offset=vector),remaining)
                    T.pipe_barrier('V')
                    best=T.alloc_var('float32')
                    chunk=T.alloc_var('int32')
                    argmax(pool,product,red,temporary,remaining,best,chunk)
                    selected=pivot+chunk//2*64+T.Cast('int32',T.reinterpret('uint32',pool[red+chunk+1]))
                    if not best>T.float32(1e-6):
                        valid=0
                        T.loop_break()
                    divisor=pool[vector+selected-pivot]
                    if selected!=pivot:
                        old=pool[permutation+pivot]
                        pool[permutation+pivot]=pool[permutation+selected]
                        pool[permutation+selected]=old
                    pool[records+panel*64+pivot]=pool[permutation+pivot]
                    pool[records+panel*64+16+pivot]=T.float32(1)/divisor
                    if remaining>1:
                        pool[vector+selected-pivot]=T.float32(0)
                        emit('pv_copy',pool.access_ptr('w',offset=scratch),pool.access_ptr('r',offset=selected*pitch),pitch)
                        T.pipe_barrier('V')
                        emit('pv_scale',pool.access_ptr('w',offset=scratch),pool.access_ptr('r',offset=scratch),T.float32(-1)/divisor,pitch)
                        T.pipe_barrier('V')
                        T.set_flag('S','V',0);T.wait_flag('S','V',0)
                        emit('pv_brcb_rows',pool.access_ptr('w',offset=factors),pool.access_ptr('r',offset=vector),T.ceildiv(remaining,8)*8,8)
                        T.pipe_barrier('V')
                        for chunk in T.serial(T.ceildiv(pitch-pivot//8*8,64)):
                            col=pivot//8*8+chunk*64
                            for row_chunk in T.serial(T.ceildiv(remaining,255)):
                                row=row_chunk*255
                                emit('pv_mad_strided',pool.access_ptr('rw',offset=(pivot+row)*pitch+col),pool.access_ptr('r',offset=scratch+col),pool.access_ptr('r',offset=factors+row*8),T.min(pitch-col,64),T.min(remaining-row,255),1,1,0,pitch//8,0,1)
                        T.pipe_barrier('V')
                        if selected!=pivot:
                            emit('pv_copy',pool.access_ptr('w',offset=selected*pitch),pool.access_ptr('r',offset=pivot*pitch),pitch)
                            T.pipe_barrier('V')
                remaining=size-block
                columns=T.alloc_var('int32');columns=remaining
                if window>0:columns=T.min(remaining,window_end-first-block)
                if window>0 and first+block==window_end:columns=T.min(window,remaining)
                next_pitch=T.alloc_var('int32');next_pitch=T.ceildiv(columns,8)*8
                T.set_flag('V','S',0);T.wait_flag('V','S',0)
                if not (window>0 and first+block==window_end):
                    for row_chunk in T.serial(T.ceildiv(remaining,16)):
                        row=row_chunk*16
                        emit('pv_copy_ub_rows',pool.access_ptr('w',offset=row*next_pitch),pool.access_ptr('r',offset=(row+block)*pitch+block),T.min(remaining-row,16),next_pitch,pitch-next_pitch,0)
                        T.pipe_barrier('V')
                if remaining>0:
                    T.set_flag('S','V',0);T.wait_flag('S','V',0)
                    emit('pv_copy',pool.access_ptr('w',offset=scratch),pool.access_ptr('r',offset=permutation+block),T.ceildiv(remaining,8)*8)
                    T.pipe_barrier('V')
                    emit('pv_copy',pool.access_ptr('w',offset=permutation),pool.access_ptr('r',offset=scratch),T.ceildiv(remaining,8)*8)
                    T.set_flag('V','S',0);T.wait_flag('V','S',0)
                T.set_flag('S','V',0);T.wait_flag('S','V',0)
                size=remaining
                pitch=next_pitch
            if panel>0:
                emit('pv_cross_wait',pool.access_ptr('rw'),0)
                emit('pv_cross_wait',pool.access_ptr('rw'),1)
            record=records+panel*64
            pool[record+32]=power
            pool[record+33]=total
            pool[record+34]=T.Cast('float32',valid)
            T.set_flag('S','MTE3',0);T.wait_flag('S','MTE3',0)
            emit('pv_store_rows',W.access_ptr('w',offset=metadata+panel*64),pool.access_ptr('r',offset=record),1,64,0,0)
            emit('pv_cross_set0',pool.access_ptr('rw'),0)
            emit('pv_cross_set0',pool.access_ptr('rw'),1)
        emit('pv_cross_wait',pool.access_ptr('rw'),0)
        emit('pv_cross_wait',pool.access_ptr('rw'),1)
        emit('pv_all_barrier',pool.access_ptr('rw'))
        pool[0]=T.Cast('float32',valid)
        pool[1]=T.float32(0)
        T.set_flag('S','MTE3',0);T.wait_flag('S','MTE3',0)
        emit('pv_store_rows',W.access_ptr('w',offset=metadata+2048+core*8),pool.access_ptr('r'),1,2,0,0)
        emit('pv_all_barrier',pool.access_ptr('rw'))
    return producer
