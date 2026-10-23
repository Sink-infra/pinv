"""Shared inverse workers for full and windowed pivot producers.

The native packet protocol, pivot mapping, lazy identity and shrinking live
column schedule are expressed as DSL. Window width is a runtime parameter.
"""
from code_dsl.op_kernel.intrinsics import emit


def make_peer_tile_rows_macro(*, cache_peak=False):
    import tilelang.language as T

    @T.macro
    def workers(A,X,W,pool,n,stride,all_batch,job,core,cores,window):
        k=T.alloc_var('int32');k=T.ceildiv(T.int32(200),8)*8
        wide=T.alloc_var('int32');wide=T.ceildiv(k,64)*64
        owned=T.alloc_var('int32');owned=(T.int32(200)+T.int32(9)-1-core)//T.int32(9)
        factor_rows=T.alloc_var('int32');factor_rows=T.ceildiv(owned,8)*8
        packet_stride=T.alloc_var('int32');packet_stride=k+16
        packets=owned*T.int32(456)
        output=packets+16*packet_stride
        factors=output+T.max(owned*k,512)
        meta=factors+8*T.max(factor_rows,16)
        norms=meta+64
        vector=norms+wide
        product=vector+wide
        accumulator=product+wide
        scratch=accumulator+wide
        red=scratch+wide
        row_indices=red+32
        packet_indices=row_indices+factor_rows
        mapping=packet_indices+16
        work=T.Cast('int64',job)*T.int32(200)*T.int32(456)
        metadata=T.Cast('int64',all_batch)*2*T.int32(200)*T.int32(456)+T.Cast('int64',job)*8192
        emit('pv_fill',pool.access_ptr('w'),T.float32(0),owned*T.int32(456))
        T.pipe_barrier('V')
        T.set_flag('V','MTE2',0);T.wait_flag('V','MTE2',0)
        emit('pv_load_rows',pool.access_ptr('w'),A.access_ptr('r',offset=T.Cast('int64',job)*T.int32(200)*T.int32(200)+core*T.int32(200)),owned,T.int32(200),(T.int32(9)-1)*T.int32(200),T.int32(456)-k)
        T.set_flag('MTE2','V',0);T.wait_flag('MTE2','V',0)
        emit('pv_fill',pool.access_ptr('w',offset=norms),T.float32(1),k)
        T.set_flag('V','S',0);T.wait_flag('V','S',0)
        for i in T.serial(factor_rows):
            index=T.if_then_else(i<owned,i*T.int32(456)*4,0)
            emit('pv_store_u32',pool.access_ptr('w'),row_indices+i,T.Cast('int32',index))
        for i in T.serial(16):emit('pv_store_u32',pool.access_ptr('w'),packet_indices+i,T.Cast('int32',i*packet_stride*4))
        T.set_flag('S','V',0);T.wait_flag('S','V',0)
        valid=T.alloc_var('int32');valid=1
        power=T.alloc_var('float32');power=T.float32(1)
        window_end=T.alloc_var('int32');window_end=T.int32(200)
        if window>0:window_end=T.min(T.int32(200),window)
        emit('pv_cross_set0',pool.access_ptr('rw'),4)
        emit('pv_cross_wait',pool.access_ptr('rw'),4)
        emit('pv_cross_set0',pool.access_ptr('rw'),5)
        emit('pv_cross_wait',pool.access_ptr('rw'),5)
        for panel in T.serial(T.ceildiv(T.int32(200),16)):
            first=panel*16
            rows=T.min(T.int32(200)-first,16)
            emit('pv_all_barrier',pool.access_ptr('rw'))
            emit('pv_cross_set0',pool.access_ptr('rw'),0)
            emit('pv_cross_wait',pool.access_ptr('rw'),0)
            emit('pv_load_rows',pool.access_ptr('w',offset=meta),W.access_ptr('r',offset=metadata+panel*64),1,64,0,0)
            T.set_flag('MTE2','S',0);T.wait_flag('MTE2','S',0)
            T.set_flag('MTE2','V',0);T.wait_flag('MTE2','V',0)
            if not (pool[meta+34]>T.float32(0.5)):valid=0
            power=pool[meta+32]
            packet_begin=T.if_then_else(window>0,first*packet_stride,0)
            if valid!=0:
                if first==0:
                    for chunk in T.serial(T.ceildiv(owned*T.int32(456),16320)):
                        begin=chunk*16320
                        emit('pv_scale',pool.access_ptr('w',offset=begin),pool.access_ptr('r',offset=begin),T.float32(1)/power,T.min(owned*T.int32(456)-begin,16320))
                    T.pipe_barrier('V')
                    # The identity is inserted by the scalar pipe. A V barrier
                    # only orders Vector instructions: explicitly finish input
                    # normalization before those first-panel scalar stores.
                    T.set_flag('V','S',0);T.wait_flag('V','S',0)
                T.set_flag('V','MTE3',0);T.wait_flag('V','MTE3',0)
                for column in T.serial(rows):
                    selected=T.Cast('int32',pool[meta+column])
                    pool[norms+selected]=-T.Cast('float32',T.Cast('int32',first+column+1))
                    if selected%T.int32(9)==core:
                        pool[(selected//T.int32(9))*T.int32(456)+k+first+column]=T.float32(1)
                        T.set_flag('S','MTE3',0);T.wait_flag('S','MTE3',0)
                        emit('pv_store_rows',W.access_ptr('w',offset=work+packet_begin+column*packet_stride),pool.access_ptr('r',offset=(selected//T.int32(9))*T.int32(456)+first),1,packet_stride,0,0)
                        T.set_flag('MTE3','V',0);T.wait_flag('MTE3','V',0)
            emit('pv_cross_set0',pool.access_ptr('rw'),1)
            emit('pv_cross_wait',pool.access_ptr('rw'),1)
            if valid!=0:
                emit('pv_load_rows',pool.access_ptr('w',offset=packets),W.access_ptr('r',offset=work+packet_begin),1,rows*packet_stride,0,0)
                T.set_flag('MTE2','V',0);T.wait_flag('MTE2','V',0)
                # Match the native packet dependency schedule in TileLang.
                # All Vector operations remain ordered; retain explicit
                # cross-pipe events for scalar writes and DMA, rather than
                # serializing the Vector pipeline after each instruction.
                for column in T.serial(rows):
                    emit('pv_gather_offsets',pool.access_ptr('w',offset=product),pool.access_ptr('r',offset=packets),pool.access_ptr('r',offset=packet_indices),column*4,rows)
                    reciprocal=pool[meta+16+column]
                    emit('pv_scale',pool.access_ptr('w',offset=packets+column*packet_stride),pool.access_ptr('r',offset=packets+column*packet_stride),-reciprocal,packet_stride)
                    if column+1<rows:
                        emit('pv_brcb_rows',pool.access_ptr('w',offset=factors),pool.access_ptr('r',offset=product),16,8)
                        for chunk in T.serial(T.ceildiv(packet_stride,64)):
                            col=chunk*64
                            emit('pv_mad_strided',pool.access_ptr('rw',offset=packets+(column+1)*packet_stride+col),pool.access_ptr('r',offset=packets+column*packet_stride+col),pool.access_ptr('r',offset=factors+(column+1)*8),T.min(packet_stride-col,64),rows-column-1,1,1,0,packet_stride//8,0,1)
                for column in T.serial(rows):
                    selected=T.Cast('int32',pool[meta+column])
                    emit('pv_gather_offsets',pool.access_ptr('w',offset=vector),pool.access_ptr('r'),pool.access_ptr('r',offset=row_indices),(first+column)*4,factor_rows)
                    emit('pv_brcb_rows',pool.access_ptr('w',offset=factors),pool.access_ptr('r',offset=vector),factor_rows,8)
                    pivot_column=first+column
                    consumed=pivot_column%8==7 or pivot_column+1==T.int32(200)
                    live_begin=pivot_column//8*8+T.if_then_else(consumed,8,0)
                    live_limit=live_begin+k+T.if_then_else(consumed,0,8)
                    for chunk in T.serial(T.ceildiv(live_limit-live_begin,64)):
                        col=live_begin+chunk*64
                        emit('pv_mad_strided',pool.access_ptr('rw',offset=col),pool.access_ptr('r',offset=packets+column*packet_stride+col-first),pool.access_ptr('r',offset=factors),T.min(live_limit-col,64),owned,1,1,0,T.int32(456)//8,0,1)
                    if selected%T.int32(9)==core:
                        emit('pv_copy',pool.access_ptr('w',offset=(selected//T.int32(9))*T.int32(456)+first),pool.access_ptr('r',offset=packets+column*packet_stride),packet_stride)
            if window>0 and first+rows==window_end and window_end<T.int32(200):
                emit('pv_all_barrier',pool.access_ptr('rw'))
                emit('pv_cross_set0',pool.access_ptr('rw'),3)
                emit('pv_cross_wait',pool.access_ptr('rw'),3)
                if valid!=0:
                    T.set_flag('V','MTE3',0);T.wait_flag('V','MTE3',0)
                    next_columns=T.min(window,T.int32(200)-window_end)
                    emit('pv_store_rows',W.access_ptr('w',offset=work+core*k+window_end),pool.access_ptr('r',offset=window_end),owned,next_columns,T.int32(456)-next_columns,T.int32(9)*k-next_columns)
                emit('pv_all_barrier',pool.access_ptr('rw'))
                emit('pv_cross_set0',pool.access_ptr('rw'),2)
                emit('pv_cross_wait',pool.access_ptr('rw'),2)
                emit('pv_cross_set0',pool.access_ptr('rw'),4)
                emit('pv_cross_wait',pool.access_ptr('rw'),4)
                window_end=T.min(T.int32(200),window_end+window)
        inverse=T.alloc_var('float32');inverse=T.float32(0)
        peak=T.alloc_var('float32');peak=T.float32(0)
        if valid!=0:
            T.set_flag('V','S',0);T.wait_flag('V','S',0)
            T.set_flag('S','V',0);T.wait_flag('S','V',0)
            emit('pv_scale',pool.access_ptr('w',offset=mapping),pool.access_ptr('r',offset=norms),T.float32(-4),T.int32(200))
            T.pipe_barrier('V')
            emit('pv_add_scalar',pool.access_ptr('w',offset=mapping),pool.access_ptr('r',offset=mapping),T.float32(-4),T.int32(200))
            T.pipe_barrier('V')
            emit('pv_cast_indices',pool.access_ptr('w',offset=mapping),pool.access_ptr('r',offset=mapping),T.int32(200))
            T.pipe_barrier('V')
            emit('pv_fill',pool.access_ptr('w',offset=accumulator),T.float32(0),wide)
            T.pipe_barrier('V')
            for i in T.serial(owned):
                emit('pv_gather_offsets',pool.access_ptr('w',offset=output+i*k),pool.access_ptr('r',offset=i*T.int32(456)+k),pool.access_ptr('r',offset=mapping),0,T.int32(200))
                T.pipe_barrier('V')
                emit('pv_mul',pool.access_ptr('w',offset=product),pool.access_ptr('r',offset=output+i*k),pool.access_ptr('r',offset=output+i*k),T.int32(200))
                T.pipe_barrier('V')
                emit('pv_add',pool.access_ptr('w',offset=accumulator),pool.access_ptr('r',offset=accumulator),pool.access_ptr('r',offset=product),T.int32(200))
                T.pipe_barrier('V')
            emit('pv_reduce_sum',pool.access_ptr('w',offset=red),pool.access_ptr('r',offset=accumulator),pool.access_ptr('rw',offset=scratch),k)
            T.set_flag('V','S',0);T.wait_flag('V','S',0)
            inverse=pool[red]
            if cache_peak:
                # The original augmented rows are dead after the output gather.
                # Their two halves hold absolute values and reduction scratch.
                emit('pv_abs',pool.access_ptr('w'),pool.access_ptr('r',offset=output),owned*k)
                T.pipe_barrier('V')
                emit('pv_reduce_max',pool.access_ptr('w',offset=red),pool.access_ptr('r'),pool.access_ptr('rw',offset=owned*k),owned*k)
                T.set_flag('V','S',0);T.wait_flag('V','S',0)
                peak=pool[red]/power
            for i in T.serial(owned):
                emit('pv_scale',pool.access_ptr('w',offset=output+i*k),pool.access_ptr('r',offset=output+i*k),T.float32(-1)/power,k)
                T.pipe_barrier('V')
                target=T.Cast('int32',-pool[norms+core+i*T.int32(9)])-1
                T.set_flag('V','MTE3',0);T.wait_flag('V','MTE3',0)
                emit('pv_store_rows',X.access_ptr('w',offset=T.Cast('int64',job)*T.int32(200)*T.int32(200)+target*T.int32(200)),pool.access_ptr('r',offset=output+i*k),1,T.int32(200),0,0)
            emit('pv_all_barrier',pool.access_ptr('rw'))
        pool[0]=T.Cast('float32',valid)
        pool[1]=inverse
        if cache_peak:pool[2]=peak
        T.set_flag('S','MTE3',0);T.wait_flag('S','MTE3',0)
        emit('pv_store_rows',W.access_ptr('w',offset=metadata+2048+core*8),pool.access_ptr('r'),1,3 if cache_peak else 2,0,0)
        emit('pv_all_barrier',pool.access_ptr('rw'))
    return workers
