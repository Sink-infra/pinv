"""Two-producer full-column pipeline, with the peer protocol in TileLang.

The two producers share pivot headers but retain separate column partitions.
Shape, batch, worker count and all strides are runtime parameters.
"""
from code_dsl.op_kernel.intrinsics import emit


def make_peer_tile_producer_macro():
    import tilelang.language as T
    from code_dsl.op_kernel.reduction import make_argmax_macro
    argmax=make_argmax_macro()

    @T.macro
    def producer(A,W,pool,n,stride,all_batch,job,core,workers):
        role=T.alloc_var('int32');role=core-9
        k=T.alloc_var('int32');k=T.ceildiv(200,8)*8
        wide=T.alloc_var('int32');wide=T.ceildiv(k,64)*64
        half=T.alloc_var('int32');half=48
        own_width=T.alloc_var('int32');own_width=64
        if role==1 or role==2:own_width=48
        if role==3:own_width=40
        pitch=T.alloc_var('int32');pitch=own_width
        if role!=0:pitch+=16
        if pitch<16:pitch=16
        pitch=72
        capacity=T.alloc_var('int32');capacity=200*pitch
        scratch=capacity
        factors=scratch+16320
        vector=factors+8*k
        product=vector+wide
        temporary=product+wide
        red=temporary+32
        indices=red+32
        permutation=indices+k
        records=permutation+k
        metadata=2*all_batch*200*456+job*8192
        header=metadata+3072
        norm_header=metadata+6400
        offset=job*200*200
        ready=metadata+6656
        ack=metadata+6848
        W[ready+role*32]=T.float32(0)
        emit('pp_mailbox_cache',W.access_ptr('rw',offset=ready+role*32))
        W[ack+role*32]=T.float32(0)
        emit('pp_mailbox_cache',W.access_ptr('rw',offset=ack+role*32))

        actual_width=T.alloc_var('int32');actual_width=own_width
        prefix=T.alloc_var('int32');prefix=0
        if role!=0:prefix=16
        source_column=T.alloc_var('int32');source_column=0
        if role==1:source_column=64
        if role==2:source_column=112
        if role==3:source_column=160
        emit('pv_fill',pool.access_ptr('w'),T.float32(0),capacity)
        T.pipe_barrier('V')
        T.set_flag('V','MTE2',0);T.wait_flag('V','MTE2',0)
        emit('pv_load_rows',pool.access_ptr('w',offset=prefix),A.access_ptr('r',offset=offset+source_column),200,actual_width,200-actual_width,pitch-own_width)
        T.set_flag('MTE2','V',0);T.wait_flag('MTE2','V',0)
        local_scale=T.alloc_var('float32');local_scale=T.float32(0)
        local_total=T.alloc_var('float32');local_total=T.float32(0)
        local_power=T.alloc_var('float32');local_power=T.float32(1)
        for chunk in T.serial(T.ceildiv(capacity,16320)):
            begin=chunk*16320
            count=T.min(capacity-begin,16320)
            emit('pv_abs',pool.access_ptr('w',offset=scratch),pool.access_ptr('r',offset=begin),count)
            T.pipe_barrier('V')
            emit('pv_reduce_max',pool.access_ptr('w',offset=red),pool.access_ptr('r',offset=scratch),pool.access_ptr('rw',offset=product),count)
            T.set_flag('V','S',0);T.wait_flag('V','S',0)
            local_scale=T.max(local_scale,pool[red])
        if local_scale>T.float32(0) and local_scale<T.float32(1.7014118e38):
            while local_power<local_scale:local_power*=T.float32(2)
            while local_power*T.float32(0.5)>=local_scale:local_power*=T.float32(0.5)
        local_ratio=local_scale/local_power
        local_total=T.Cast('float32',200*own_width)*local_ratio*local_ratio*T.float32(1.00001)
        pool[scratch]=local_scale
        pool[scratch+1]=local_total
        pool[scratch+2]=local_power
        T.set_flag('S','MTE3',0);T.wait_flag('S','MTE3',0)
        emit('pv_store_rows',W.access_ptr('w',offset=norm_header+role*8),pool.access_ptr('r',offset=scratch),1,8,0,0)
        emit('pv_cross_set0',pool.access_ptr('rw'),4)
        emit('pv_cross_wait',pool.access_ptr('rw'),4)
        T.set_flag('MTE3','MTE2',0);T.wait_flag('MTE3','MTE2',0)
        emit('pv_load_rows',pool.access_ptr('w',offset=scratch),W.access_ptr('r',offset=norm_header),1,32,0,0)
        T.set_flag('MTE2','S',0);T.wait_flag('MTE2','S',0)
        magnitude=T.alloc_var('float32');magnitude=T.float32(0)
        for part in T.serial(4):magnitude=T.max(magnitude,pool[scratch+part*8])
        valid=T.alloc_var('int32');valid=1
        if not (magnitude>=T.float32(0.1) and magnitude<T.float32(1.7014118e38)):valid=0
        power=T.alloc_var('float32');power=T.float32(1)
        if valid!=0:
            while power<magnitude:power*=T.float32(2)
            while power*T.float32(0.5)>=magnitude:power*=T.float32(0.5)
        total=T.alloc_var('float32');total=T.float32(0)
        for part in T.serial(4):
            ratio=pool[scratch+part*8+2]/power
            total+=pool[scratch+part*8+1]*ratio*ratio
        total*=T.float32(1.00001)
        for chunk in T.serial(T.ceildiv(capacity,16320)):
            begin=chunk*16320
            emit('pv_scale',pool.access_ptr('w',offset=begin),pool.access_ptr('r',offset=begin),T.float32(1)/power,T.min(capacity-begin,16320))
        T.pipe_barrier('V')
        for row in T.serial(k):pool[permutation+row]=T.Cast('float32',row)
        T.set_flag('S','V',0);T.wait_flag('S','V',0)
        size=T.alloc_var('int32');size=200
        for panel in T.serial(T.ceildiv(200,16)):
            first=panel*16
            if valid!=0:
                owner=T.alloc_var('int32');owner=0
                owner=0
                if first>=64:owner=1
                if first>=112:owner=2
                if first>=160:owner=3
                matrix=T.alloc_var('int32');matrix=first*pitch
                if role==owner:matrix+=prefix+first-source_column
                active_width=T.alloc_var('int32');active_width=own_width
                if role!=owner:active_width+=16
                if active_width<16:active_width=16
                if panel>0:
                    for part in T.serial(4):
                        emit('pp_mailbox_cache',W.access_ptr('rw',offset=ack+part*32))
                        polls=T.alloc_var('int32');polls=0
                        observed=T.alloc_var('float32')
                        emit('pp_mailbox_load',W.access_ptr('r',offset=ack+part*32),observed.buffer.data)
                        while observed<T.Cast('float32',panel) and polls<10000:
                            emit('pp_mailbox_cache',W.access_ptr('rw',offset=ack+part*32))
                            emit('pp_mailbox_load',W.access_ptr('r',offset=ack+part*32),observed.buffer.data);polls+=1
                        if polls>=10000:valid=0
                if role==owner:
                    T.set_flag('V','MTE3',0);T.wait_flag('V','MTE3',0)
                    emit('pv_store_rows',W.access_ptr('w',offset=header),pool.access_ptr('r',offset=matrix),size,16,pitch-16,0)
                    T.set_flag('MTE3','S',0);T.wait_flag('MTE3','S',0)
                    W[ready+role*32]=T.Cast('float32',panel+1)
                    emit('pp_mailbox_cache',W.access_ptr('rw',offset=ready+role*32))
                if role!=owner:
                    emit('pp_mailbox_cache',W.access_ptr('rw',offset=ready+owner*32))
                    polls=T.alloc_var('int32');polls=0
                    observed=T.alloc_var('float32')
                    emit('pp_mailbox_load',W.access_ptr('r',offset=ready+owner*32),observed.buffer.data)
                    while observed<T.Cast('float32',panel+1) and polls<10000:
                        emit('pp_mailbox_cache',W.access_ptr('rw',offset=ready+owner*32))
                        emit('pp_mailbox_load',W.access_ptr('r',offset=ready+owner*32),observed.buffer.data);polls+=1
                    if polls>=10000:valid=0
                    T.set_flag('V','MTE2',0);T.wait_flag('V','MTE2',0)
                    emit('pv_load_rows',pool.access_ptr('w',offset=matrix),W.access_ptr('r',offset=header),size,16,0,pitch-16)
                    T.set_flag('MTE2','V',0);T.wait_flag('MTE2','V',0)
                else:
                    T.set_flag('MTE3','V',0);T.wait_flag('MTE3','V',0)
                W[ack+role*32]=T.Cast('float32',panel+1)
                emit('pp_mailbox_cache',W.access_ptr('rw',offset=ack+role*32))
                emit('pv_iota',pool.access_ptr('w',offset=indices),size)
                T.pipe_barrier('V')
                emit('pv_scale_indices',pool.access_ptr('w',offset=indices),pool.access_ptr('r',offset=indices),pitch*4,size)
                T.pipe_barrier('V')
                block=T.min(size,16)
                for pivot in T.serial(block):
                    remaining=size-pivot
                    emit('pv_gather_offsets',pool.access_ptr('w',offset=vector),pool.access_ptr('r',offset=matrix),pool.access_ptr('r',offset=indices),(pivot*pitch+pivot)*4,remaining)
                    T.pipe_barrier('V')
                    emit('pv_abs',pool.access_ptr('w',offset=product),pool.access_ptr('r',offset=vector),T.min(remaining,32))
                    T.pipe_barrier('V')
                    best=T.alloc_var('float32')
                    chunk=T.alloc_var('int32')
                    emit('pv_whole_max',pool.access_ptr('w',offset=red),pool.access_ptr('r',offset=product),T.min(remaining,32),1)
                    T.set_flag('V','S',0);T.wait_flag('V','S',0)
                    emit('pv_reduce_scalars',pool.access_ptr('rw'),best.buffer.data,chunk.buffer.data)
                    selected=pivot+chunk
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
                        emit('pv_copy',pool.access_ptr('w',offset=scratch),pool.access_ptr('r',offset=matrix+selected*pitch),active_width)
                        T.pipe_barrier('V')
                        emit('pv_scale',pool.access_ptr('w',offset=scratch),pool.access_ptr('r',offset=scratch),T.float32(-1)/divisor,active_width)
                        T.pipe_barrier('V')
                        T.set_flag('S','V',0);T.wait_flag('S','V',0)
                        emit('pv_brcb_rows',pool.access_ptr('w',offset=factors),pool.access_ptr('r',offset=vector),T.ceildiv(remaining,8)*8,8)
                        T.pipe_barrier('V')
                        for chunk in T.serial(T.ceildiv(active_width-pivot//8*8,64)):
                            col=pivot//8*8+chunk*64
                            for row_chunk in T.serial(T.ceildiv(remaining,255)):
                                row=row_chunk*255
                                emit('pv_mad_strided',pool.access_ptr('rw',offset=matrix+(pivot+row)*pitch+col),pool.access_ptr('r',offset=scratch+col),pool.access_ptr('r',offset=factors+row*8),T.min(active_width-col,64),T.min(remaining-row,255),1,1,0,pitch//8,0,1)
                        T.pipe_barrier('V')
                        if selected!=pivot:
                            emit('pv_copy',pool.access_ptr('w',offset=matrix+selected*pitch),pool.access_ptr('r',offset=matrix+pivot*pitch),active_width)
                            T.pipe_barrier('V')
                remaining=size-block
                drop=T.alloc_var('int32');drop=0
                old_prefix=T.alloc_var('int32');old_prefix=16
                if role==owner:
                    drop=block
                    old_prefix=0
                new_width=T.alloc_var('int32');new_width=own_width-drop
                next_owner=T.alloc_var('int32');next_owner=0
                next_owner=0
                if first+block>=64:next_owner=1
                if first+block>=112:next_owner=2
                if first+block>=160:next_owner=3
                new_prefix=T.alloc_var('int32');new_prefix=16
                if role==next_owner:new_prefix=0
                next_pitch=T.alloc_var('int32');next_pitch=T.max(16,new_prefix+new_width)
                T.set_flag('V','S',0);T.wait_flag('V','S',0)
                if remaining>0:
                    T.set_flag('S','V',0);T.wait_flag('S','V',0)
                    emit('pv_copy',pool.access_ptr('w',offset=scratch),pool.access_ptr('r',offset=permutation+block),T.ceildiv(remaining,8)*8)
                    T.pipe_barrier('V')
                    emit('pv_copy',pool.access_ptr('w',offset=permutation),pool.access_ptr('r',offset=scratch),T.ceildiv(remaining,8)*8)
                    T.set_flag('V','S',0);T.wait_flag('V','S',0)
                T.set_flag('S','V',0);T.wait_flag('S','V',0)
                size=remaining
                own_width=new_width
            if panel>0:
                emit('pv_cross_wait',pool.access_ptr('rw'),0)
                emit('pv_cross_wait',pool.access_ptr('rw'),1)
            record=records+panel*64
            pool[record+32]=power
            pool[record+33]=total
            pool[record+34]=T.Cast('float32',valid)
            if role==3:
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
