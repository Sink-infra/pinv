from hashlib import sha256

def tile_pipeline_kernel(component_source, fallback_source):
    """Thirteen AIVs per matrix, with immutable statistics and shared admission."""
    import tilelang.language as T
    from code_dsl.op_kernel.intrinsics import emit
    lanes = 2
    producers = 4
    source = component_source + ('\n' + fallback_source if fallback_source else '')

    @T.prim_func
    def kernel(A: T.Tensor((1,), 'float32'), X: T.Tensor((1,), 'float32'), W: T.Tensor((1,), 'float32'), S: T.Tensor((1,), 'float32'), blocks: T.int32, n: T.int32, batch: T.int32, stride: T.int32, window: T.int32, atol: T.float32, rtol: T.float32):
        T.func_attr({'global_symbol': 'linalg_pinv_tilelang', 'pinv.strategy': 'peer_pipeline', 'pinv.complete_continuation': fallback_source is not None, 'pinv.paired': True, 'pinv.component_source.sha256': sha256(source.encode()).hexdigest()})
        with T.Kernel(blocks, threads=1, is_npu=True) as bid:
            T.import_source(source)
            pool = T.alloc_ub((1024,), 'float32')
            with T.Scope('V'):
                T.reads(A[0], W[0], S[0])
                T.writes(X[0], W[0], S[0])
                emit('pv_pipeline_bind', A.access_ptr('r'), X.access_ptr('w'), W.access_ptr('rw'))
                emit('pv_sync_init', W.access_ptr('rw'))
                group_size = 13
                workers = group_size - producers
                job = bid // group_size
                core = bid % group_size
                if job < batch:
                    if core >= workers:
                        emit('pv_pipeline_peer', W.access_ptr('rw'), blocks, n, stride, batch, job, core, workers, window)
                    else:
                        emit('pv_pipeline_rows', W.access_ptr('rw'), blocks, n, stride, batch, job, core, workers, window)
                else:
                    emit('pv_cross_set0', W.access_ptr('rw'), 4)
                    emit('pv_cross_wait', W.access_ptr('rw'), 4)
                    for dummy_panel in T.serial(13):
                        emit('pv_cross_set0', W.access_ptr('rw'), 0)
                        emit('pv_cross_wait', W.access_ptr('rw'), 0)
                        emit('pv_cross_set0', W.access_ptr('rw'), 1)
                        emit('pv_cross_wait', W.access_ptr('rw'), 1)
                emit('pv_sync_all', W.access_ptr('rw'))
                valid = T.alloc_var('int32')
                valid = 1
                if bid < batch:
                    b = bid
                    metadata = 2 * batch * n * stride + b * 8192
                    emit('pv_load_rows', pool.access_ptr('w'), W.access_ptr('r', offset=metadata + 2048), 1, group_size * 8, 0, 0)
                    T.set_flag('MTE2', 'S', 0)
                    T.wait_flag('MTE2', 'S', 0)
                    inverse_total = T.alloc_var('float32')
                    inverse_total = T.float32(0)
                    for c in T.serial(group_size):
                        if pool[c * 8] < T.float32(0.5):
                            valid = 0
                        inverse_total += pool[c * 8 + 1]
                    emit('pv_load_rows', pool.access_ptr('w', offset=384), W.access_ptr('r', offset=metadata), 1, 64, 0, 0)
                    T.set_flag('MTE2', 'S', 0)
                    T.wait_flag('MTE2', 'S', 0)
                    power = pool[416]
                    total = pool[417]
                    pool[640] = total
                    T.set_flag('S', 'V', 0)
                    T.wait_flag('S', 'V', 0)
                    emit('pv_sqrt', pool.access_ptr('w', offset=640), pool.access_ptr('r', offset=640), 1)
                    T.set_flag('V', 'S', 0)
                    T.wait_flag('V', 'S', 0)
                    cutoff = T.max(pool[640] * rtol, atol / power)
                    if not (inverse_total > T.float32(0) and inverse_total * cutoff * cutoff < T.float32(81) and (total * inverse_total < T.float32(3600000000.0))):
                        valid = 0
                    pool[0] = T.Cast('float32', valid)
                    T.set_flag('S', 'MTE3', 0)
                    T.wait_flag('S', 'MTE3', 0)
                    emit('pv_store_rows', W.access_ptr('w', offset=2 * batch * n * stride + 7040 + bid * 8), pool.access_ptr('r'), 1, 8, 0, 0)
                    T.set_flag('MTE3', 'MTE2', 0)
                    T.wait_flag('MTE3', 'MTE2', 0)
                emit('pv_sync_all', W.access_ptr('rw'))
                emit('pv_load_rows', pool.access_ptr('w'), W.access_ptr('r', offset=2 * batch * n * stride + 7040), 1, batch * 8, 0, 0)
                T.set_flag('MTE2', 'S', 0)
                T.wait_flag('MTE2', 'S', 0)
                valid = 1
                for b in T.serial(batch):
                    if pool[b * 8] < T.float32(0.5):
                        valid = 0
                if bid == 0:
                    S[0] = T.Cast('float32', valid)
                    S[31] = T.Cast('float32', valid)
                if fallback_source is not None:
                    if valid == 0:
                        emit('pv_sync_all', W.access_ptr('rw'))
                        if bid < batch:
                            emit('pv_module_bind', A.access_ptr('r'), X.access_ptr('w'), W.access_ptr('rw'))
                            emit('pv_module_rectangular', S.access_ptr('rw'), blocks, bid, n, n, batch, atol, rtol)
                emit('pv_all_barrier', W.access_ptr('rw'))
    return kernel
