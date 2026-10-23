"""Two-producer full-column pipeline, with the peer protocol in TileLang.

The two producers share pivot headers but retain separate column partitions.
Shape, batch, worker count and all strides are runtime parameters.
"""
from code_dsl.op_kernel.intrinsics import emit


def make_half_peer_macro():
    import tilelang.language as T
    from code_dsl.op_kernel.half_reduction import make_half_argmax_macro
    argmax=make_half_argmax_macro()

    @T.macro
    def producer(A,W,pool,matrix,half_work,n,stride,all_batch,job,core,workers):
        role=T.alloc_var('int32');role=core-workers
        k=T.alloc_var('int32');k=T.ceildiv(n,8)*8
        wide=T.alloc_var('int32');wide=T.ceildiv(k,64)*64
        split=T.alloc_var('int32');split=T.ceildiv(n,32)*16
        own_width=T.alloc_var('int32');own_width=split
        if role!=0:own_width=T.ceildiv(n-split,16)*16
        pitch=T.alloc_var('int32');pitch=own_width
        if role!=0:pitch+=16
        if pitch<16:pitch=16
        capacity=T.alloc_var('int32');capacity=n*pitch
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
        header=metadata+3072
        offset=job*n*n
        actual_width=T.alloc_var('int32');actual_width=split
        prefix=T.alloc_var('int32');prefix=0
        if role!=0:
            actual_width=n-split
            prefix=16
        source_column=T.alloc_var('int32');source_column=0
        if role!=0:source_column=split
        emit('pv_fill',pool.access_ptr('w'),T.float32(0),capacity)
        T.pipe_barrier('V')
        T.set_flag('V','MTE2',0);T.wait_flag('V','MTE2',0)
        emit('pv_load_rows',pool.access_ptr('w',offset=prefix),A.access_ptr('r',offset=offset+source_column),n,actual_width,n-actual_width,pitch-T.ceildiv(actual_width,8)*8)
        T.set_flag('MTE2','V',0);T.wait_flag('MTE2','V',0)
        local_scale=T.alloc_var('float32');local_scale=T.float32(0)
        local_total=T.alloc_var('float32');local_total=T.float32(0)
        local_power=T.alloc_var('float32');local_power=T.float32(1)
        for chunk in T.serial(T.ceildiv(capacity,4096)):
            begin=chunk*4096
            count=T.min(capacity-begin,4096)
            emit('pv_abs',pool.access_ptr('w',offset=scratch),pool.access_ptr('r',offset=begin),count)
            T.pipe_barrier('V')
            emit('pv_reduce_max',pool.access_ptr('w',offset=red),pool.access_ptr('r',offset=scratch),pool.access_ptr('rw',offset=product),count)
            T.set_flag('V','S',0);T.wait_flag('V','S',0)
            local_scale=T.max(local_scale,pool[red])
        if local_scale>T.float32(0) and local_scale<T.float32(1.7014118e38):
            while local_power<local_scale:local_power*=T.float32(2)
            while local_power*T.float32(0.5)>=local_scale:local_power*=T.float32(0.5)
        for chunk in T.serial(T.ceildiv(capacity,16320)):
            begin=chunk*16320
            emit('pv_scale',pool.access_ptr('w',offset=begin),pool.access_ptr('r',offset=begin),T.float32(1)/local_power,T.min(capacity-begin,16320))
        T.pipe_barrier('V')
        for chunk in T.serial(T.ceildiv(capacity,4096)):
            begin=chunk*4096
            count=T.min(capacity-begin,4096)
            emit('pv_mul',pool.access_ptr('w',offset=scratch),pool.access_ptr('r',offset=begin),pool.access_ptr('r',offset=begin),count)
            T.pipe_barrier('V')
            emit('pv_reduce_sum',pool.access_ptr('w',offset=red),pool.access_ptr('r',offset=scratch),pool.access_ptr('rw',offset=product),count)
            T.set_flag('V','S',0);T.wait_flag('V','S',0)
            local_total+=pool[red]
        pool[scratch]=local_scale
        pool[scratch+1]=local_total
        pool[scratch+2]=local_power
        T.set_flag('S','MTE3',0);T.wait_flag('S','MTE3',0)
        emit('pv_store_rows',W.access_ptr('w',offset=header+role*8),pool.access_ptr('r',offset=scratch),1,8,0,0)
        emit('pv_cross_set1',pool.access_ptr('rw'),4)
        emit('pv_cross_wait',pool.access_ptr('rw'),4)
        T.set_flag('MTE3','MTE2',0);T.wait_flag('MTE3','MTE2',0)
        emit('pv_load_rows',pool.access_ptr('w',offset=scratch),W.access_ptr('r',offset=header),1,16,0,0)
        T.set_flag('MTE2','S',0);T.wait_flag('MTE2','S',0)
        magnitude=T.max(pool[scratch],pool[scratch+8])
        sum0=pool[scratch+1]
        sum1=pool[scratch+9]
        power0=pool[scratch+2]
        power1=pool[scratch+10]
        emit('pv_cross_set1',pool.access_ptr('rw'),5)
        emit('pv_cross_wait',pool.access_ptr('rw'),5)
        valid=T.alloc_var('int32');valid=1
        if not (magnitude>=T.float32(0.1) and magnitude<T.float32(1.7014118e38)):valid=0
        power=T.alloc_var('float32');power=T.float32(1)
        if valid!=0:
            while power<magnitude:power*=T.float32(2)
            while power*T.float32(0.5)>=magnitude:power*=T.float32(0.5)
        ratio0=power0/power
        ratio1=power1/power
        total=(sum0*ratio0*ratio0+sum1*ratio1*ratio1)*T.float32(1.00001)
        for chunk in T.serial(T.ceildiv(capacity,16320)):
            begin=chunk*16320
            emit('pv_scale',pool.access_ptr('w',offset=begin),pool.access_ptr('r',offset=begin),local_power/power,T.min(capacity-begin,16320))
        T.pipe_barrier('V')
        for chunk in T.serial(T.ceildiv(capacity,16320)):
            begin=chunk*16320
            emit('hp_half',matrix.access_ptr('w',offset=begin),pool.access_ptr('r',offset=begin),T.min(capacity-begin,16320))
        T.pipe_barrier('V')
        for row in T.serial(k):pool[permutation+row]=T.Cast('float32',row)
        T.set_flag('S','V',0);T.wait_flag('S','V',0)
        size=T.alloc_var('int32');size=n
        header_issued=T.alloc_var('int32');header_issued=0
        pending_ready=T.alloc_var('int32');pending_ready=0
        for panel in T.serial(T.ceildiv(n,16)):
            first=panel*16
            if valid!=0:
                if header_issued!=0:
                    if pending_ready!=0:emit('pv_cross_wait',pool.access_ptr('rw'),7)
                    emit('pv_cross_wait',pool.access_ptr('rw'),9)
                    header_issued=0
                owner=T.alloc_var('int32');owner=0
                if first>=split:owner=1
                if role==owner:
                    T.set_flag('V','MTE3',0);T.wait_flag('V','MTE3',0)
                    emit('hp_store_header',W.access_ptr('w',offset=header),matrix.access_ptr('r'),size,pitch)
                    emit('pv_cross_set1',pool.access_ptr('rw'),7)
                    T.set_flag('MTE3','V',0);T.wait_flag('MTE3','V',0)
                    pending_ready=1
                else:
                    emit('pv_cross_set1',pool.access_ptr('rw'),7)
                    emit('pv_cross_wait',pool.access_ptr('rw'),7)
                    T.set_flag('V','MTE2',0);T.wait_flag('V','MTE2',0)
                    emit('hp_load_header',matrix.access_ptr('w'),W.access_ptr('r',offset=header),size,pitch)
                    T.set_flag('MTE2','V',0);T.wait_flag('MTE2','V',0)
                    pending_ready=0
                emit('pv_cross_set1',pool.access_ptr('rw'),9)
                header_issued=1
                emit('pv_iota',pool.access_ptr('w',offset=indices),size)
                T.pipe_barrier('V')
                emit('pv_scale_indices',pool.access_ptr('w',offset=indices),pool.access_ptr('r',offset=indices),pitch*2,size)
                T.pipe_barrier('V')
                block=T.min(size,16)
                for pivot in T.serial(block):
                    remaining=size-pivot
                    emit('hp_gather',half_work.access_ptr('w'),matrix.access_ptr('r'),pool.access_ptr('r',offset=indices),(pivot*pitch+pivot)*2,remaining)
                    T.pipe_barrier('V')
                    emit('hp_abs',half_work.access_ptr('w',offset=3840),half_work.access_ptr('r'),remaining)
                    T.pipe_barrier('V')
                    best=T.alloc_var('float32')
                    chunk=T.alloc_var('int32')
                    argmax(pool,half_work,product,temporary,remaining,best,chunk)
                    selected=pivot+chunk
                    if not best>T.float32(1e-6):
                        valid=0
                        T.loop_break()
                    divisor=T.Cast('float32',half_work[selected-pivot])
                    if selected!=pivot:
                        old=pool[permutation+pivot]
                        pool[permutation+pivot]=pool[permutation+selected]
                        pool[permutation+selected]=old
                    pool[records+panel*64+pivot]=pool[permutation+pivot]
                    pool[records+panel*64+16+pivot]=T.float32(1)/divisor
                    if remaining>1:
                        half_work[selected-pivot]=T.float16(0)
                        emit('hp_copy',half_work.access_ptr('w',offset=3584),matrix.access_ptr('r',offset=selected*pitch),pitch)
                        T.pipe_barrier('V')
                        emit('hp_scale',half_work.access_ptr('w',offset=3584),half_work.access_ptr('r',offset=3584),T.float32(-1)/divisor,pitch)
                        T.pipe_barrier('V')
                        T.set_flag('S','V',0);T.wait_flag('S','V',0)
                        emit('hp_brcb',half_work.access_ptr('w',offset=256),half_work.access_ptr('r'),T.ceildiv(remaining,8)*8)
                        T.pipe_barrier('V')
                        for chunk in T.serial(T.ceildiv(pitch-pivot//16*16,128)):
                            col=pivot//16*16+chunk*128
                            for row_chunk in T.serial(T.ceildiv(remaining,255)):
                                row=row_chunk*255
                                emit('hp_mad',matrix.access_ptr('rw',offset=(pivot+row)*pitch+col),half_work.access_ptr('r',offset=3584+col),half_work.access_ptr('r',offset=256+row*16),T.min(pitch-col,128),T.min(remaining-row,255),pitch)
                        T.pipe_barrier('V')
                        if selected!=pivot:
                            emit('hp_copy',matrix.access_ptr('w',offset=selected*pitch),matrix.access_ptr('r',offset=pivot*pitch),pitch)
                            T.pipe_barrier('V')
                remaining=size-block
                drop=T.alloc_var('int32');drop=0
                old_prefix=T.alloc_var('int32');old_prefix=16
                if role==owner:
                    drop=block
                    old_prefix=0
                new_width=T.alloc_var('int32');new_width=own_width-drop
                next_owner=T.alloc_var('int32');next_owner=0
                if first+block>=split:next_owner=1
                new_prefix=T.alloc_var('int32');new_prefix=16
                if role==next_owner:new_prefix=0
                next_pitch=T.alloc_var('int32');next_pitch=T.max(16,new_prefix+new_width)
                T.set_flag('V','S',0);T.wait_flag('V','S',0)
                if remaining>0 and new_width>0:
                    for row_chunk in T.serial(T.ceildiv(remaining,16)):
                        row=row_chunk*16
                        emit('hp_rows',matrix.access_ptr('w',offset=row*next_pitch+new_prefix),matrix.access_ptr('r',offset=(row+block)*pitch+old_prefix+drop),T.min(remaining-row,16),new_width,pitch-new_width,next_pitch-new_width)
                        T.pipe_barrier('V')
                if remaining>0:
                    T.set_flag('S','V',0);T.wait_flag('S','V',0)
                    emit('pv_copy',pool.access_ptr('w',offset=scratch),pool.access_ptr('r',offset=permutation+block),T.ceildiv(remaining,8)*8)
                    T.pipe_barrier('V')
                    emit('pv_copy',pool.access_ptr('w',offset=permutation),pool.access_ptr('r',offset=scratch),T.ceildiv(remaining,8)*8)
                    T.set_flag('V','S',0);T.wait_flag('V','S',0)
                T.set_flag('S','V',0);T.wait_flag('S','V',0)
                size=remaining
                own_width=new_width
                pitch=next_pitch
            if panel>0:
                emit('pv_cross_wait',pool.access_ptr('rw'),0)
                emit('pv_cross_wait',pool.access_ptr('rw'),1)
            record=records+panel*64
            pool[record+32]=power
            pool[record+33]=total
            pool[record+34]=T.Cast('float32',valid)
            if role==1:
                T.set_flag('S','MTE3',0);T.wait_flag('S','MTE3',0)
                emit('pv_store_rows',W.access_ptr('w',offset=metadata+panel*64),pool.access_ptr('r',offset=record),1,64,0,0)
            emit('pv_cross_set0',pool.access_ptr('rw'),0)
            emit('pv_cross_set0',pool.access_ptr('rw'),1)
        if header_issued!=0:
            if pending_ready!=0:emit('pv_cross_wait',pool.access_ptr('rw'),7)
            emit('pv_cross_wait',pool.access_ptr('rw'),9)
        emit('pv_cross_wait',pool.access_ptr('rw'),0)
        emit('pv_cross_wait',pool.access_ptr('rw'),1)
        emit('pv_all_barrier',pool.access_ptr('rw'))
        pool[0]=T.Cast('float32',valid)
        pool[1]=T.float32(0)
        T.set_flag('S','MTE3',0);T.wait_flag('S','MTE3',0)
        emit('pv_store_rows',W.access_ptr('w',offset=metadata+2048+core*8),pool.access_ptr('r'),1,2,0,0)
        emit('pv_all_barrier',pool.access_ptr('rw'))
    return producer
