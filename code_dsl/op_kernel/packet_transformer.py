from code_dsl.op_kernel.intrinsics import emit

def make_transformer_macro(*, cache_peak=False):
    import tilelang.language as T
    @T.macro
    def transformer(A,X,W,pool,n,stride,all_batch,job,core,cores,window):
        k=T.alloc_var('uint32');k=T.ceildiv(n,8)*8
        packet_stride=T.alloc_var('uint32');packet_stride=k+16
        packets=0
        factors=16*packet_stride
        meta=factors+128
        product=meta+64
        packet_indices=product+256
        work=T.Cast('uint64',job)*n*stride
        metadata=T.Cast('uint64',all_batch)*2*n*stride+T.Cast('uint64',job)*8192
        T.set_flag('V','S',0);T.wait_flag('V','S',0)
        for i in T.serial(16):emit('pv_store_u32',pool.access_ptr('w'),packet_indices+i,T.Cast('uint32',i*packet_stride*4))
        T.set_flag('S','V',0);T.wait_flag('S','V',0)
        valid=T.alloc_var('int32');valid=1
        window_end=T.alloc_var('uint32');window_end=T.min(n,window)
        for panel in T.serial(T.ceildiv(n,16)):
            first=panel*16
            rows=T.min(n-first,16)
            packet_begin=first*packet_stride
            emit('pv_all_barrier',pool.access_ptr('rw'))
            emit('pv_cross_set0',pool.access_ptr('rw'),0)
            emit('pv_cross_wait',pool.access_ptr('rw'),0)
            emit('pv_load_rows',pool.access_ptr('w',offset=meta),W.access_ptr('r',offset=metadata+panel*64),1,64,0,0)
            T.set_flag('MTE2','S',0);T.wait_flag('MTE2','S',0)
            T.set_flag('MTE2','V',0);T.wait_flag('MTE2','V',0)
            if not (pool[meta+34]>T.float32(.5)):valid=0
            emit('pv_cross_set0',pool.access_ptr('rw'),1)
            emit('pv_cross_wait',pool.access_ptr('rw'),1)
            if valid!=0:
                emit('pv_load_rows',pool.access_ptr('w',offset=packets),W.access_ptr('r',offset=work+packet_begin),1,rows*packet_stride,0,0)
                T.set_flag('MTE2','V',0);T.wait_flag('MTE2','V',0)
                for column in T.serial(rows):
                    emit('pv_gather_offsets',pool.access_ptr('w',offset=product),pool.access_ptr('r',offset=packets),pool.access_ptr('r',offset=packet_indices),column*4,rows)
                    reciprocal=pool[meta+16+column]
                    emit('pv_scale',pool.access_ptr('w',offset=packets+column*packet_stride),pool.access_ptr('r',offset=packets+column*packet_stride),reciprocal,packet_stride)
                    if column+1<rows:
                        emit('pv_scale',pool.access_ptr('w',offset=product),pool.access_ptr('r',offset=product),T.float32(-1),16)
                        emit('pv_brcb_rows',pool.access_ptr('w',offset=factors),pool.access_ptr('r',offset=product),16,8)
                        for chunk in T.serial(T.ceildiv(packet_stride,64)):
                            col=chunk*64
                            emit('pv_mad_strided',pool.access_ptr('rw',offset=packets+(column+1)*packet_stride+col),pool.access_ptr('r',offset=packets+column*packet_stride+col),pool.access_ptr('r',offset=factors+(column+1)*8),T.min(packet_stride-col,64),rows-column-1,1,1,0,packet_stride//8,0,1)
                T.set_flag('V','MTE3',0);T.wait_flag('V','MTE3',0)
                emit('pv_store_rows',W.access_ptr('w',offset=work+packet_begin),pool.access_ptr('r',offset=packets),1,rows*packet_stride,0,0)
            emit('pv_all_barrier',pool.access_ptr('rw'))
            emit('pv_cross_set0',pool.access_ptr('rw'),5)
            emit('pv_cross_wait',pool.access_ptr('rw'),5)
            if window>0 and first+rows==window_end and window_end<n:
                emit('pv_all_barrier',pool.access_ptr('rw'))
                emit('pv_cross_set0',pool.access_ptr('rw'),3)
                emit('pv_cross_wait',pool.access_ptr('rw'),3)
                emit('pv_cross_set0',pool.access_ptr('rw'),2)
                emit('pv_cross_wait',pool.access_ptr('rw'),2)
                emit('pv_cross_set0',pool.access_ptr('rw'),4)
                emit('pv_cross_wait',pool.access_ptr('rw'),4)
                window_end=T.min(n,window_end+window)
        emit('pv_all_barrier',pool.access_ptr('rw'))
        pool[0]=T.Cast('float32',valid)
        pool[1]=T.float32(0)
        if cache_peak:pool[2]=T.float32(0)
        T.set_flag('S','MTE3',0);T.wait_flag('S','MTE3',0)
        emit('pv_store_rows',W.access_ptr('w',offset=metadata+2048+core*8),pool.access_ptr('r'),1,3 if cache_peak else 2,0,0)
        emit('pv_all_barrier',pool.access_ptr('rw'))
    return transformer
