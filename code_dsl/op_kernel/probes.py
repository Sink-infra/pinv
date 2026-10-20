"""Incomplete-path diagnostics; never included in the project export."""
from pathlib import Path


def compact_light_probe():
    import tilelang.language as T
    from code_dsl.op_kernel.compact import make_light_macro
    light=make_light_macro()
    instructions=Path(__file__).with_name('instructions.h').read_text()
    @T.prim_func
    def kernel(A:T.Tensor((1,),'float32'),X:T.Tensor((1,),'float32'),W:T.Tensor((1,),'float32'),
               blocks:T.int32,m:T.int32,n:T.int32,batch:T.int32,atol:T.float32,rtol:T.float32):
        T.func_attr({'global_symbol':'linalg_pinv_tilelang','pinv.probe':'light-v1'})
        with T.Kernel(blocks,threads=1,is_npu=True) as bid:
            T.import_source(instructions)
            ub=T.alloc_ub((1440,),'float32')
            indices=T.alloc_ub((16,),'uint32')
            order=T.alloc_ub((16,),'int32')
            accepted=T.alloc_ub((8,),'int32')
            with T.Scope('V'):
                for iteration in T.serial(T.ceildiv(batch,blocks)):
                    job=bid+iteration*blocks
                    if job<batch:
                        accepted[0]=0
                        if m==n and n>=15 and n<=16:
                            light(A,X,ub,indices,order,accepted,job*m*n,n,atol,rtol)
                        W[job]=T.Cast('float32',accepted[0])
    return kernel


def local_quick_probe():
    import tilelang.language as T
    from code_dsl.op_kernel.local import make_local_quick_macro
    quick=make_local_quick_macro()
    instructions=Path(__file__).with_name('instructions.h').read_text()
    @T.prim_func
    def kernel(A:T.Tensor((1,),'float32'),X:T.Tensor((1,),'float32'),W:T.Tensor((1,),'float32'),
               blocks:T.int32,m:T.int32,n:T.int32,batch:T.int32,atol:T.float32,rtol:T.float32):
        T.func_attr({'global_symbol':'linalg_pinv_tilelang','pinv.probe':'local-quick-v1'})
        with T.Kernel(blocks,threads=1,is_npu=True) as bid:
            T.import_source(instructions)
            pool=T.alloc_ub((19136,),'float32')
            indices=T.alloc_ub((64,),'uint32')
            order=T.alloc_ub((128,),'uint32')
            accepted=T.alloc_ub((8,),'int32')
            with T.Scope('V'):
                for iteration in T.serial(T.ceildiv(batch,blocks)):
                    job=bid+iteration*blocks
                    if job<batch:
                        accepted[0]=0
                        if m==n and n>32 and n<=64:
                            quick(A,X,pool,indices,order,accepted,job*m*n,n,atol,rtol)
                        W[job]=T.Cast('float32',accepted[0])
    return kernel


def accurate_probe():
    import tilelang.language as T
    from code_dsl.op_kernel.resident import make_accurate_macro
    from code_dsl.op_kernel.kernels import instruction_source
    accurate=make_accurate_macro();instructions=instruction_source()
    @T.prim_func
    def kernel(A:T.Tensor((1,),'float32'),X:T.Tensor((1,),'float32'),W:T.Tensor((1,),'float32'),
               blocks:T.int32,m:T.int32,n:T.int32,batch:T.int32,atol:T.float32,rtol:T.float32):
        T.func_attr({'global_symbol':'linalg_pinv_tilelang','pinv.probe':'accurate-v1'})
        with T.Kernel(blocks,threads=1,is_npu=True) as bid:
            T.import_source(instructions)
            pool=T.alloc_ub((44128,),'float32')
            accepted=T.alloc_ub((8,),'int32')
            with T.Scope('V'):
                for iteration in T.serial(T.ceildiv(batch,blocks)):
                    job=bid+iteration*blocks
                    if job<batch:
                        accepted[0]=0
                        if m==n and n<=96:
                            accurate(A,X,pool,accepted,job*m*n,n,atol,rtol)
                        W[job]=T.Cast('float32',accepted[0])
    return kernel
