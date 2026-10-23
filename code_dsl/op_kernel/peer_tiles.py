"""TileLang peer tile using 13 AIVs per matrix within the platform budget."""

def _tile_component(operation,producer):
    import tilelang.language as T
    from hashlib import sha256
    from code_dsl.op_kernel.parallel_components import parallel_instruction_source
    from code_dsl.op_kernel.intrinsics import emit
    instructions=parallel_instruction_source()
    @T.prim_func
    def kernel(A:T.Tensor((1,),'float32'),X:T.Tensor((1,),'float32'),W:T.Tensor((1,),'float32'),
               blocks:T.int32,n:T.int32,stride:T.int32,all_batch:T.int32,job:T.int32,core:T.int32,cores:T.int32,window:T.int32):
        T.func_attr({'global_symbol':'linalg_pinv_tilelang','pinv.component':'peer_tile',
                     'pinv.instructions.sha256':sha256(instructions.encode()).hexdigest()})
        with T.Kernel(blocks,threads=1,is_npu=True):
            T.import_source(instructions)
            pool=T.alloc_ub((49088,),'float32')
            with T.Scope('V'):
                T.reads(A[0],W[0])
                T.writes(X[0],W[0])
                emit('pv_sync_init',W.access_ptr('rw'))
                if producer:
                    operation(A,W,pool,n,stride,all_batch,job,core,cores)
                else:
                    operation(A,X,W,pool,n,stride,all_batch,job,core,cores,window)
    return kernel

def tile_rows_kernel():
    from code_dsl.op_kernel.peer_tile_rows import make_peer_tile_rows_macro
    return _tile_component(make_peer_tile_rows_macro(),False)

def tile_peer_kernel():
    from code_dsl.op_kernel.peer_tile_producer import make_peer_tile_producer_macro
    return _tile_component(make_peer_tile_producer_macro(),True)


def tile_pipeline_kernel(component_source,fallback_source):
    from code_dsl.op_kernel.peer_tile_pipeline import tile_pipeline_kernel as make_pipeline
    return make_pipeline(component_source,fallback_source)
