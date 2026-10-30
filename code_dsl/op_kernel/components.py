"""Multi-function DSL lowering for the pinned single-PrimFunc JIT backend.

Each mathematical subroutine is a real TileLang PrimFunc. Its generated device
body is imported into the final TileLang module. The declaration changes from
a launch entry to a device subroutine and launch-only metadata is
removed; arithmetic, transfers and synchronization remain verbatim. This avoids the
backend's pathological analysis of the combined inline TIR; no original C++
solver is imported. Hashes retain both the component TIR and generated bodies.
"""
from hashlib import sha256
import re
from code_dsl.op_kernel.intrinsics import emit
from code_dsl.op_kernel.kernels import instruction_source

COMPONENTS=('light','quick','accurate','rectangular','resident_accurate','resident_rectangular')
DECLARATION='extern "C" __global__ __aicore__ void linalg_pinv_tilelang_kernel('
BOUNDARY='\nvoid linalg_pinv_tilelang_kernel_tiling()'


def component_kernel(name,unroll=1,factor_pitch=8):
    import tilelang.language as T
    if name not in COMPONENTS:raise ValueError(name)
    resident=name.startswith('resident_')
    algorithm=name.removeprefix('resident_')
    if algorithm=='rectangular':
        from code_dsl.op_kernel.rectangular import make_rectangular_macro
        operation=make_rectangular_macro(workspace_local=resident)
        @T.macro
        def invoke(A,X,W,pool,indices,order,accepted,job,m,n,batch,atol,rtol):
            operation(A,X,W,pool,accepted,job,m,n,batch,atol,rtol)
    elif algorithm=='accurate':
        from code_dsl.op_kernel.resident import make_accurate_macro
        operation=make_accurate_macro()
        @T.macro
        def invoke(A,X,W,pool,indices,order,accepted,job,m,n,batch,atol,rtol):
            operation(A,X,pool,accepted,job*m*n,n,atol,rtol)
    else:
        if name=='light':
            from code_dsl.op_kernel.compact import make_light_macro
            operation=make_light_macro(unroll=unroll)
        else:
            from code_dsl.op_kernel.local import make_local_quick_macro
            operation=make_local_quick_macro(unroll=unroll,factor_pitch=factor_pitch)
        @T.macro
        def invoke(A,X,W,pool,indices,order,accepted,job,m,n,batch,atol,rtol):
            operation(A,X,pool,indices,order,accepted,job*m*n,n,atol,rtol)
    capacity={'light':1440,'quick':19136,'accurate':44128,'rectangular':26144}[algorithm]
    if name=='resident_accurate':capacity=20544  # 20*w+32+2*n*(w+8)+4*(w+8), n<=64
    if name=='resident_rectangular':capacity=3296  # 20*w+32+8*rows+3*basis, rows/cols<=64
    workspace_capacity=16384 if name=='resident_rectangular' else 1
    instructions=instruction_source()
    @T.prim_func
    def kernel(A:T.Tensor((1,),'float32'),X:T.Tensor((1,),'float32'),W:T.Tensor((1,),'float32'),S:T.Tensor((1,),'float32'),
               blocks:T.int32,job:T.int32,m:T.int32,n:T.int32,batch:T.int32,atol:T.float32,rtol:T.float32):
        T.func_attr({'global_symbol':'linalg_pinv_tilelang','pinv.component':name,
                     'pinv.instructions.sha256':sha256(instructions.encode()).hexdigest()})
        with T.Kernel(blocks,threads=1,is_npu=True):
            T.import_source(instructions)
            pool=T.alloc_ub((capacity,),'float32')
            indices=T.alloc_ub((64,),'uint32')
            order=T.alloc_ub((128,),'uint32')
            accepted=T.alloc_ub((8,),'int32')
            resident_work=T.alloc_ub((workspace_capacity,),'float32')
            with T.Scope('V'):
                accepted[0]=0
                if name=='resident_rectangular':
                    invoke(A,X,resident_work,pool,indices,order,accepted,job,m,n,batch,atol,rtol)
                else:
                    invoke(A,X,W,pool,indices,order,accepted,job,m,n,batch,atol,rtol)
                S[0 if resident else job*32]=T.Cast('float32',accepted[0])
    return kernel


def device_functions(compiled,*,include_instructions=True):
    if set(compiled)!=set(COMPONENTS):raise ValueError('all DSL components are required')
    parts=[];records={};framework_includes=set()
    for name in COMPONENTS:
        source=compiled[name].get_kernel_source()
        if source.count(DECLARATION)!=1 or source.count(BOUNDARY)!=1:raise ValueError('unexpected component source')
        function=source[source.index(DECLARATION):source.index(BOUNDARY)]
        # Native TileLang operations can import additional runtime templates
        # (for example T.copy). Preserve their declarations when composing the
        # separately lowered DSL functions, along with the unchanged bodies.
        if name not in {'light','quick'}:
            framework_includes.update(line for line in source[:source.index(DECLARATION)].splitlines()
                                      if line.startswith('#include "tl_templates/')
                                      and line!='#include "tl_templates/ascend/common.h"')
        body=function[function.index('{'):]
        symbol='PvDsl'+''.join(part.title() for part in name.split('_'))
        # Keep the complete precise continuation out of the hot entry's
        # instruction stream. This is a device-function lowering decision;
        # its generated numerical body remains byte-for-byte unchanged.
        adapted=function.replace(DECLARATION,'__aicore__ static __attribute__((noinline)) void '+symbol+'(',1)
        metadata='  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);\n'
        if body.count(metadata)!=1:raise ValueError('expected exactly one component launch metadata statement')
        adapted=adapted.replace(metadata,'',1)
        binding='  AscendC::GlobalTensor<float> S;\n  S.SetGlobalBuffer((__gm__ float*)S_handle);\n'
        expected=body.replace(metadata,'',1)
        abi=['global entry to static noinline device function','remove launch-only task type metadata']
        if name.startswith('resident_'):
            if adapted.count('GM_ADDR S_handle')!=1 or adapted.count(binding)!=1:
                raise ValueError('unexpected status tensor ABI')
            adapted=adapted.replace('GM_ADDR S_handle','int32_t &module_status',1).replace(binding,'',1)
            expected=expected.replace(binding,'',1)
            status_stores=list(re.finditer(r'(?m)^(\s*)S.SetValue\(0, (.*)\);$',expected))
            if len(status_stores)!=1:raise ValueError('unexpected component status result')
            store=status_stores[0]
            replacement=store[1]+'module_status = static_cast<int32_t>('+store[2]+');'
            expected=expected[:store.start()]+replacement+expected[store.end():]
            adapted=re.sub(r'(?m)^(\s*)S.SetValue\(0, (.*)\);$',lambda m:m[1]+'module_status = static_cast<int32_t>('+m[2]+');',adapted)
            abi.append('marshal final boolean status store as int32 scalar output reference; float 0/1 converts exactly; no numerical body changes')
        assert adapted[adapted.index('{'):]==expected
        # Fast routines are fused as TIR in the caller. Import only reachable
        # fallback functions; unused copies of fast functions can depend on
        # runtime helpers generated exclusively for their own PrimFunc.
        if name not in {'light','quick'}:parts.append(adapted)
        records[name]={'symbol':symbol,'emitted_source_sha256':sha256(source.encode()).hexdigest(),
                       'device_body_sha256':sha256(body.encode()).hexdigest(),
                       'callable_body_sha256':sha256(expected.encode()).hexdigest(),
                       'abi_changes':abi,
                       'tir_sha256':sha256(compiled[name].prim_func.script().encode()).hexdigest() if hasattr(compiled[name],'prim_func') else None}
    source='\n'.join(sorted(framework_includes))+'\n'
    if include_instructions:source+=instruction_source()
    source+='\nusing uint=unsigned int;using uchar=unsigned char;using ushort=unsigned short;\n'
    source+='namespace tl::ascend {\n'+'\n'.join(parts)+'\n'
    source+='struct PvModuleBuffers { AscendC::GlobalTensor<float> inputTensor,outputTensor,workspaceTensor; };\n}\n'
    source+='#define pv_module_bind(d,a,b) PvModuleBuffers pvModuleBuffers{d,a,b}\n'
    for name in COMPONENTS:
        if name in {'light','quick'}:continue
        symbol='PvDsl'+''.join(part.title() for part in name.split('_'))
        resident=name.startswith('resident_')
        status='status' if resident else '(GM_ADDR)d.GetPhyAddr()'
        arguments='d,a,b,status,...' if resident else 'd,a,b,...'
        source+=f'#define pv_module_{name}({arguments}) {symbol}((GM_ADDR)pvModuleBuffers.inputTensor.GetPhyAddr(),(GM_ADDR)pvModuleBuffers.outputTensor.GetPhyAddr(),(GM_ADDR)pvModuleBuffers.workspaceTensor.GetPhyAddr(),{status},__VA_ARGS__,fftsAddr)\n'
    return source,records


def composed_kernel(strategy,component_source,unroll=1,factor_pitch=8):
    import tilelang.language as T
    if strategy not in {'compact','local','rectangular'}:raise ValueError(strategy)
    instructions=component_source
    if strategy=='compact':
        from code_dsl.op_kernel.static_storage import MARKER
        instructions+='\n'+MARKER
    lower=15 if strategy=='compact' else 33
    upper=16 if strategy=='compact' else 64
    maximum=32 if strategy=='compact' else 64
    if strategy=='compact':
        from code_dsl.op_kernel.compact import make_light_macro
        pivoted=make_light_macro(unroll=unroll,scalar_admission=True)
        from code_dsl.op_kernel.compact_nopivot import make_nopivot_macro
        trial=make_nopivot_macro()
        aligned_trial=make_nopivot_macro(full_tile=True,unroll=16)
        @T.macro
        def fast(A,X,pool,indices,order,accepted,offset,n,atol,rtol):
            if n==16:
                aligned_trial(A,X,pool,indices,order,accepted,offset,n,atol,rtol)
            elif n>0:
                trial(A,X,pool,indices,order,accepted,offset,n,atol,rtol)
            if accepted[0]==0:
                pivoted(A,X,pool,indices,order,accepted,offset,n,atol,rtol)
    else:
        from code_dsl.op_kernel.local import make_local_quick_macro
        fast=make_local_quick_macro(unroll=unroll,factor_pitch=factor_pitch)
    capacity=1440 if strategy=='compact' else 19136
    @T.prim_func
    def kernel(A:T.Tensor((1,),'float32'),X:T.Tensor((1,),'float32'),W:T.Tensor((1,),'float32'),S:T.Tensor((1,),'float32'),
               blocks:T.int32,m:T.int32,n:T.int32,batch:T.int32,atol:T.float32,rtol:T.float32):
        T.func_attr({'global_symbol':'linalg_pinv_tilelang','pinv.strategy':strategy,
                     'pinv.launch_policy':'compact_job_grid_local_native',
                     'pinv.component_source.sha256':sha256(instructions.encode()).hexdigest()})
        with T.Kernel(blocks,threads=1,is_npu=True) as bid:
            T.import_source(instructions)
            pool=T.alloc_ub((capacity,),'float32')
            indices=T.alloc_ub((64,),'uint32')
            order=T.alloc_ub((128,),'uint32')
            accepted_slot=T.alloc_ub((8,),'int32')
            accepted=T.alloc_var('int32')
            decision=T.alloc_var('int32')
            with T.Scope('V'):
                T.reads(A[0],W[0],S[0])
                T.writes(X[0],W[0],S[0])
                emit('pv_module_bind',A.access_ptr('r'),X.access_ptr('w'),W.access_ptr('rw'))
                job=T.alloc_var('int32');job=bid
                while job<batch:
                    offset=T.alloc_var('int32');offset=job*m*n
                    dimension=T.alloc_var('int32');dimension=n
                    accepted=0
                    if strategy!='compact':
                        accepted_slot[0]=0
                    if strategy!='rectangular':
                        if m==n and n<=maximum:
                            if n>=lower and n<=upper:
                                # Fuse the fast path into the caller's DSL
                                # scope. Compact admission is scalar, while
                                # local admission stays in its measured UB layout.
                                if strategy=='compact':
                                    fast(A,X,pool,indices,order,accepted.buffer,offset,dimension,atol,rtol)
                                else:
                                    # Retain the previously measured local
                                    # strategy's UB status and allocation order.
                                    fast(A,X,pool,indices,order,accepted_slot,offset,dimension,atol,rtol)
                                    accepted=accepted_slot[0]
                            if accepted==0:
                                decision=0
                                emit('pv_module_resident_accurate',pool.access_ptr('rw'),decision.buffer.data,blocks,job,m,n,batch,atol,rtol)
                                if decision==0:
                                    emit('pv_module_resident_rectangular',pool.access_ptr('rw'),decision.buffer.data,blocks,job,m,n,batch,atol,rtol)
                        else:
                            emit('pv_module_rectangular',S.access_ptr('rw'),blocks,job,m,n,batch,atol,rtol)
                    else:
                        emit('pv_module_rectangular',S.access_ptr('rw'),blocks,job,m,n,batch,atol,rtol)
                    job+=blocks
    return kernel
