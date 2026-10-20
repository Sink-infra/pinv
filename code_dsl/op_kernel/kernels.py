"""Shared strategy kernels; tensor geometry is always a launch argument."""
from hashlib import sha256
from pathlib import Path


FLAT_TENSOR_METADATA='#ifndef K_MAX_SHAPE_DIM\n#define K_MAX_SHAPE_DIM 0\n#endif\n'


def instruction_source():
    root=Path(__file__).parent
    return FLAT_TENSOR_METADATA+'\n'.join((root/name).read_text() for name in ['instructions.h','arithmetic.h','precision.h'])


def rectangular_kernel():
    import tilelang.language as T
    from code_dsl.op_kernel.rectangular import make_rectangular_macro
    operation=make_rectangular_macro()
    instructions=instruction_source()
    @T.prim_func
    def kernel(A:T.Tensor((1,),'float32'),X:T.Tensor((1,),'float32'),W:T.Tensor((1,),'float32'),
               blocks:T.int32,m:T.int32,n:T.int32,batch:T.int32,atol:T.float32,rtol:T.float32):
        T.func_attr({'global_symbol':'linalg_pinv_tilelang','pinv.strategy':'rectangular',
                     'pinv.instructions.sha256':sha256(instructions.encode()).hexdigest()})
        with T.Kernel(blocks,threads=1,is_npu=True) as bid:
            T.import_source(instructions)
            # Maximum of RectLocalFloats(rows,cols) for 1..512. Runtime offsets
            # preserve the original layout; no shape-specialized export exists.
            pool=T.alloc_ub((26144,),'float32')
            accepted=T.alloc_ub((8,),'int32')
            with T.Scope('V'):
                for iteration in T.serial(T.ceildiv(batch,blocks)):
                    job=bid+iteration*blocks
                    if job<batch:
                        operation(A,X,W,pool,accepted,job,m,n,batch,atol,rtol)
    return kernel


def resident_kernel(strategy):
    if strategy not in {'compact','local'}:raise ValueError(strategy)
    import tilelang.language as T
    from code_dsl.op_kernel.compact import make_light_macro
    from code_dsl.op_kernel.local import make_local_quick_macro
    from code_dsl.op_kernel.resident import make_accurate_macro
    from code_dsl.op_kernel.rectangular import make_rectangular_macro
    fast=make_light_macro() if strategy=='compact' else make_local_quick_macro()
    fast_min=15 if strategy=='compact' else 33
    fast_max=16 if strategy=='compact' else 64
    inverse_max=32 if strategy=='compact' else 96
    accurate=make_accurate_macro();rectangular=make_rectangular_macro()
    instructions=instruction_source()
    capacity=26144 if strategy=='compact' else 44128
    @T.prim_func
    def kernel(A:T.Tensor((1,),'float32'),X:T.Tensor((1,),'float32'),W:T.Tensor((1,),'float32'),
               blocks:T.int32,m:T.int32,n:T.int32,batch:T.int32,atol:T.float32,rtol:T.float32):
        T.func_attr({'global_symbol':'linalg_pinv_tilelang','pinv.strategy':strategy,
                     'pinv.instructions.sha256':sha256(instructions.encode()).hexdigest()})
        with T.Kernel(blocks,threads=1,is_npu=True) as bid:
            T.import_source(instructions)
            pool=T.alloc_ub((capacity,),'float32')
            indices=T.alloc_ub((64,),'uint32')
            order=T.alloc_ub((128,),'uint32')
            accepted=T.alloc_ub((8,),'int32')
            with T.Scope('V'):
                for iteration in T.serial(T.ceildiv(batch,blocks)):
                    job=bid+iteration*blocks
                    if job<batch:
                        accepted[0]=0
                        if m==n and n<=inverse_max:
                            if n>=fast_min and n<=fast_max:
                                fast(A,X,pool,indices,order,accepted,job*m*n,n,atol,rtol)
                            if accepted[0]==0:
                                accurate(A,X,pool,accepted,job*m*n,n,atol,rtol)
                        if accepted[0]==0:
                            rectangular(A,X,W,pool,accepted,job,m,n,batch,atol,rtol)
    return kernel
