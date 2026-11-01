"""DSL entry points for the communicating inverse pipeline components."""
from pathlib import Path
from hashlib import sha256
from code_dsl.op_kernel.kernels import instruction_source


def parallel_instruction_source():
    return instruction_source()+'\n'+Path(__file__).with_name('parallel_instructions.h').read_text()+'\n'+Path(__file__).with_name('peer_mailbox.h').read_text()+'\n'+Path(__file__).with_name('peer_finish_instructions.h').read_text()


def rows_kernel(operation=None):
    import tilelang.language as T
    from code_dsl.op_kernel.pipeline_rows import make_pipeline_rows_macro
    from code_dsl.op_kernel.intrinsics import emit
    operation=make_pipeline_rows_macro() if operation is None else operation;instructions=parallel_instruction_source()
    @T.prim_func
    def kernel(A:T.Tensor((1,),'float32'),X:T.Tensor((1,),'float32'),W:T.Tensor((1,),'float32'),
               blocks:T.int32,n:T.uint32,stride:T.uint32,all_batch:T.uint32,job:T.uint32,core:T.uint32,cores:T.uint32,window:T.uint32):
        T.func_attr({'global_symbol':'linalg_pinv_tilelang','pinv.component':'pipeline_rows',
                     'pinv.instructions.sha256':sha256(instructions.encode()).hexdigest()})
        with T.Kernel(blocks,threads=1,is_npu=True):
            T.import_source(instructions)
            pool=T.alloc_ub((49088,),'float32')
            with T.Scope('V'):
                T.reads(A[0],W[0])
                T.writes(X[0],W[0])
                emit('pv_sync_init',W.access_ptr('rw'))
                operation(A,X,W,pool,n,stride,all_batch,job,core,cores,window)
    return kernel


def producer_kernel(operation=None):
    import tilelang.language as T
    from code_dsl.op_kernel.producer import make_producer_macro
    from code_dsl.op_kernel.intrinsics import emit
    operation=make_producer_macro() if operation is None else operation;instructions=parallel_instruction_source()
    @T.prim_func
    def kernel(A:T.Tensor((1,),'float32'),X:T.Tensor((1,),'float32'),W:T.Tensor((1,),'float32'),
               blocks:T.int32,n:T.int32,stride:T.int32,all_batch:T.int32,job:T.int32,core:T.int32,cores:T.int32,window:T.int32):
        T.func_attr({'global_symbol':'linalg_pinv_tilelang','pinv.component':'pipeline_producer',
                     'pinv.instructions.sha256':sha256(instructions.encode()).hexdigest()})
        with T.Kernel(blocks,threads=1,is_npu=True):
            T.import_source(instructions)
            pool=T.alloc_ub((49088,),'float32')
            with T.Scope('V'):
                T.reads(A[0],W[0])
                T.writes(W[0])
                emit('pv_sync_init',W.access_ptr('rw'))
                operation(A,W,pool,n,stride,all_batch,job,core,window)
    return kernel


def peer_producer_kernel(operation=None):
    import tilelang.language as T
    from code_dsl.op_kernel.peer_producer import make_peer_producer_macro
    from code_dsl.op_kernel.intrinsics import emit
    operation=make_peer_producer_macro() if operation is None else operation;instructions=parallel_instruction_source()
    @T.prim_func
    def kernel(A:T.Tensor((1,),'float32'),X:T.Tensor((1,),'float32'),W:T.Tensor((1,),'float32'),
               blocks:T.int32,n:T.int32,stride:T.int32,all_batch:T.int32,job:T.int32,core:T.int32,cores:T.int32,window:T.int32):
        T.func_attr({'global_symbol':'linalg_pinv_tilelang','pinv.component':'peer_producer',
                     'pinv.instructions.sha256':sha256(instructions.encode()).hexdigest()})
        with T.Kernel(blocks,threads=1,is_npu=True):
            T.import_source(instructions)
            pool=T.alloc_ub((49088,),'float32')
            with T.Scope('V'):
                T.reads(A[0],W[0])
                T.writes(W[0])
                emit('pv_sync_init',W.access_ptr('rw'))
                operation(A,W,pool,n,stride,all_batch,job,core,cores)
    return kernel


def transformer_kernel(*, cache_peak=False):
    import tilelang.language as T
    from code_dsl.op_kernel.packet_transformer import make_transformer_macro as make_peer_producer_macro
    from code_dsl.op_kernel.intrinsics import emit
    operation=make_peer_producer_macro(cache_peak=cache_peak);instructions=parallel_instruction_source()
    @T.prim_func
    def kernel(A:T.Tensor((1,),'float32'),X:T.Tensor((1,),'float32'),W:T.Tensor((1,),'float32'),
               blocks:T.int32,n:T.uint32,stride:T.uint32,all_batch:T.uint32,job:T.uint32,core:T.uint32,cores:T.uint32,window:T.uint32):
        T.func_attr({'global_symbol':'linalg_pinv_tilelang','pinv.component':'peer_producer',
                     'pinv.instructions.sha256':sha256(instructions.encode()).hexdigest()})
        with T.Kernel(blocks,threads=1,is_npu=True):
            T.import_source(instructions)
            pool=T.alloc_ub((49088,),'float32')
            with T.Scope('V'):
                T.reads(A[0],W[0])
                T.writes(W[0])
                emit('pv_sync_init',W.access_ptr('rw'))
                operation(A,X,W,pool,n,stride,all_batch,job,core,cores,window)
    return kernel



def prepared_rows_kernel():
    from code_dsl.op_kernel.prepared_rows import make_prepared_rows_macro
    return rows_kernel(make_prepared_rows_macro())


def prepared_producer_kernel():
    from code_dsl.op_kernel.prepared_producer import make_prepared_producer_macro
    return producer_kernel(make_prepared_producer_macro())

def parallel_device_functions(compiled):
    from code_dsl.op_kernel.components import DECLARATION,BOUNDARY
    parts=[];records={}
    names=tuple(name for name in ('rows','producer','peer','tile_rows','tile_peer','prepared_rows','prepared_producer','transformer') if name in compiled)
    if 'rows' not in names or not {'producer','peer'}.intersection(names):raise ValueError('a producer and its workers are required')
    for name in names:
        source=compiled[name].get_kernel_source()
        if source.count(DECLARATION)!=1 or source.count(BOUNDARY)!=1:raise ValueError('unexpected parallel component source')
        function=source[source.index(DECLARATION):source.index(BOUNDARY)]
        metadata='  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);\n'
        if function.count(metadata)!=1:raise ValueError('unexpected launch metadata')
        separate=name in {'tile_rows','tile_peer'} or (name in {'rows','peer'} and set(names)=={'rows','peer'})
        linkage='static __attribute__((noinline))' if separate else 'inline'
        adapted=function.replace(DECLARATION,'__aicore__ '+linkage+' void PvDslPipeline'+''.join(part.title() for part in name.split('_'))+'(',1).replace(metadata,'',1)
        if function[function.index('{'):].replace(metadata,'',1)!=adapted[adapted.index('{'):]:raise ValueError('component body changed')
        parts.append(adapted)
        records[name]={'source_sha256':sha256(source.encode()).hexdigest(),'callable_body_sha256':sha256(adapted[adapted.index('{'):].encode()).hexdigest()}
    header=parallel_instruction_source()+'\nusing uint=unsigned int;using uchar=unsigned char;using ushort=unsigned short;\n'
    header+='namespace tl::ascend {\n'+'\n'.join(parts)+'\nstruct PvPipelineBuffers { AscendC::GlobalTensor<float> inputTensor,outputTensor,workspaceTensor; };\n}\n'
    header+='#define pv_pipeline_bind(d,a,b) PvPipelineBuffers pvPipelineBuffers{d,a,b}\n'
    for name in names:
        symbol=''.join(part.title() for part in name.split('_'))
        header+=f'#define pv_pipeline_{name}(d,a,b,...) PvDslPipeline{symbol}((GM_ADDR)pvPipelineBuffers.inputTensor.GetPhyAddr(),(GM_ADDR)pvPipelineBuffers.outputTensor.GetPhyAddr(),(GM_ADDR)pvPipelineBuffers.workspaceTensor.GetPhyAddr(),__VA_ARGS__,fftsAddr)\n'
    return header,records


def pipeline_kernel(component_source,*,paired=False,peer=False,prepared=False,prepared_checked=False,fallback_source=None,tiles=False,producer_count=None,tile_group=None):
    """Full/window/peer pipeline; optional DSL Jacobi continuation on rejection.

    Window refinement remains a separate, unfinished strategy. Passing a DSL
    continuation makes the non-window pipeline complete for its runtime shapes.
    """
    import tilelang.language as T
    from code_dsl.op_kernel.intrinsics import emit
    lanes=2 if paired else 1
    producers=producer_count if producer_count is not None else (2 if peer else 1)
    source=component_source+('\n'+fallback_source if fallback_source else '')
    @T.prim_func
    def kernel(A:T.Tensor((1,),'float32'),X:T.Tensor((1,),'float32'),W:T.Tensor((1,),'float32'),S:T.Tensor((1,),'float32'),
               blocks:T.int32,n:T.int32,batch:T.int32,stride:T.int32,window:T.int32,atol:T.float32,rtol:T.float32):
        T.func_attr({'global_symbol':'linalg_pinv_tilelang','pinv.strategy':'peer_pipeline' if peer else 'pipeline',
                     'pinv.complete_continuation':fallback_source is not None,'pinv.paired':paired,
                     'pinv.component_source.sha256':sha256(source.encode()).hexdigest()})
        with T.Kernel(blocks,threads=1,is_npu=True) as bid:
            T.import_source(source)
            pool=T.alloc_ub((1024,),'float32')
            with T.Scope('V'):
                T.reads(A[0],W[0],S[0])
                T.writes(X[0],W[0],S[0])
                emit('pv_pipeline_bind',A.access_ptr('r'),X.access_ptr('w'),W.access_ptr('rw'))
                emit('pv_sync_init',W.access_ptr('rw'))
                group_size=tile_group if tile_group is not None else blocks*lanes//batch
                workers=group_size-producers
                job=bid//group_size
                core=bid%group_size
                if tile_group is not None:
                    if job<batch:
                        if core>=workers:
                            emit('pv_pipeline_peer',W.access_ptr('rw'),blocks,n,stride,batch,job,core,workers,window)
                        else:
                            emit('pv_pipeline_rows',W.access_ptr('rw'),blocks,n,stride,batch,job,core,workers,window)
                    else:
                        # A rounded AIC launch has one padding AIV for odd batches.
                        # It participates in all collectives without matrix accesses.
                        emit('pv_cross_set0',W.access_ptr('rw'),4)
                        emit('pv_cross_wait',W.access_ptr('rw'),4)
                        emit('pv_cross_set0',W.access_ptr('rw'),5)
                        emit('pv_cross_wait',W.access_ptr('rw'),5)
                        for dummy_panel in T.serial(13):
                            emit('pv_cross_set0',W.access_ptr('rw'),0)
                            emit('pv_cross_wait',W.access_ptr('rw'),0)
                            emit('pv_cross_set0',W.access_ptr('rw'),1)
                            emit('pv_cross_wait',W.access_ptr('rw'),1)
                elif prepared:
                    prepared_workers=T.max(group_size-2,1)
                    owned=T.ceildiv(n,prepared_workers)
                    factor_rows=T.ceildiv(owned,8)*8
                    k=T.ceildiv(n,8)*8
                    wide=T.ceildiv(k,64)*64
                    output_size=T.max(T.max(owned*k,512),16*factor_rows*8+owned*16+256)
                    worker_peak=owned*stride+16*(k+16)+output_size+8*T.max(factor_rows,16)+64+5*wide+32+factor_rows+16+k
                    producer_peak=n*window+4096+10*k+3*wide+64+T.ceildiv(n,16)*64+512
                    if prepared_checked or group_size>2 and n%16==0 and window>0 and window%16==0 and worker_peak<=49088 and producer_peak<=49088:
                        if core<prepared_workers:
                            emit('pv_pipeline_prepared_rows',W.access_ptr('rw'),blocks,n,stride,batch,job,core,prepared_workers,window)
                        elif core==prepared_workers:
                            emit('pv_pipeline_transformer',W.access_ptr('rw'),blocks,n,stride,batch,job,core,prepared_workers,window)
                        else:
                            emit('pv_pipeline_prepared_producer',W.access_ptr('rw'),blocks,n,stride,batch,job,core,prepared_workers,window)
                    else:
                        if core>=workers:
                            if peer:
                                emit('pv_pipeline_peer',W.access_ptr('rw'),blocks,n,stride,batch,job,core,workers,window)
                            else:
                                emit('pv_pipeline_producer',W.access_ptr('rw'),blocks,n,stride,batch,job,core,workers,window)
                        else:
                            emit('pv_pipeline_rows',W.access_ptr('rw'),blocks,n,stride,batch,job,core,workers,window)
                else:
                    if tiles and n==200 and stride==520 and group_size==16 and window==0:
                        if core>=workers:
                            emit('pv_pipeline_tile_peer',W.access_ptr('rw'),blocks,n,stride,batch,job,core,workers,window)
                        else:
                            emit('pv_pipeline_tile_rows',W.access_ptr('rw'),blocks,n,stride,batch,job,core,workers,window)
                    else:
                        if core>=workers:
                            if peer:
                                emit('pv_pipeline_peer',W.access_ptr('rw'),blocks,n,stride,batch,job,core,workers,window)
                            else:
                                emit('pv_pipeline_producer',W.access_ptr('rw'),blocks,n,stride,batch,job,core,workers,window)
                        else:
                            emit('pv_pipeline_rows',W.access_ptr('rw'),blocks,n,stride,batch,job,core,workers,window)
                emit('pv_sync_all',W.access_ptr('rw'))
                # Every core observes the same admission decision, matching the
                # native protocol. Numerical fallback is a separate DSL stage.
                valid=T.alloc_var('int32');valid=1
                for b in T.serial(batch):
                    metadata=2*batch*n*stride+b*8192
                    emit('pv_load_rows',pool.access_ptr('w'),W.access_ptr('r',offset=metadata+2048),1,group_size*8,0,0)
                    T.set_flag('MTE2','S',0);T.wait_flag('MTE2','S',0)
                    inverse_total=T.alloc_var('float32');inverse_total=T.float32(0)
                    for c in T.serial(group_size):
                        if pool[c*8]<T.float32(0.5):valid=0
                        inverse_total+=pool[c*8+1]
                    emit('pv_load_rows',pool.access_ptr('w',offset=384),W.access_ptr('r',offset=metadata),1,64,0,0)
                    T.set_flag('MTE2','S',0);T.wait_flag('MTE2','S',0)
                    power=pool[416]
                    total=pool[417]
                    pool[640]=total
                    T.set_flag('S','V',0);T.wait_flag('S','V',0)
                    emit('pv_sqrt',pool.access_ptr('w',offset=640),pool.access_ptr('r',offset=640),1)
                    T.set_flag('V','S',0);T.wait_flag('V','S',0)
                    cutoff=T.max(pool[640]*rtol,atol/power)
                    if not (inverse_total>T.float32(0) and inverse_total*cutoff*cutoff<T.float32(81) and total*inverse_total<T.float32(3.6e9)):valid=0
                if bid==0:
                    S[0]=T.Cast('float32',valid)
                    # The continuation writes S[job*32]. Preserve admission
                    # separately in the first job's reserved cache line, on
                    # the same core that may later update S[0].
                    S[31]=T.Cast('float32',valid)
                if fallback_source is not None:
                    if valid==0:
                        # Finish every peer's metadata reads before Jacobi
                        # reuses the same workspace. Fast acceptance needs no
                        # additional global barrier or kernel launch.
                        emit('pv_sync_all',W.access_ptr('rw'))
                        if bid<batch:
                            emit('pv_module_bind',A.access_ptr('r'),X.access_ptr('w'),W.access_ptr('rw'))
                            emit('pv_module_rectangular',S.access_ptr('rw'),blocks,bid,n,n,batch,atol,rtol)
                emit('pv_all_barrier',W.access_ptr('rw'))
    return kernel
