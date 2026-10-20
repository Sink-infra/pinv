"""Launch metadata and allocator lifetime corrections for the pinned backend.

Vector programs use one AIV lane per block. The blocked factorization retains
MIX AIC 1:1 metadata for the SDK matrix multiplication primitive. Only unused
backend allocations are removed; numerical instructions remain unchanged.
"""
from contextlib import contextmanager
import re


def vector_only_source(source,mixed=False,paired=False):
    # All DSL tensors use explicit flat addresses and transfer descriptors.
    # The SDK ShapeInfo arrays are unused; define their extent before every
    # Ascend header, including peer imports that precede the component source.
    from code_dsl.op_kernel.kernels import FLAT_TENSOR_METADATA
    if 'SetShapeInfo(' in source or 'GetShapeInfo(' in source:
        raise ValueError('flat tensor lowering cannot discard consumed shape metadata')
    if not source.startswith(FLAT_TENSOR_METADATA):source=FLAT_TENSOR_METADATA+source
    declaration='extern "C" __global__ __aicore__ void linalg_pinv_tilelang_kernel('
    if source.count(declaration)!=1:raise ValueError('one outer device entry is required')
    position=source.index(declaration)
    # Imported device subroutines have already been lowered and hashed. Never
    # rewrite their numerical bodies while lowering the containing module.
    return source[:position]+_vector_device_source(source[position:],mixed=mixed,paired=paired)


def _vector_device_source(source, mixed=False, paired=False):
    old = 'KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);'
    new = 'KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);'
    if source.count(old) != 1 or 'KERNEL_TYPE_MIX_AIC_1_2' in source or 'bid = bid / 2' in source:
        raise ValueError('unexpected launch layout; Vector-only lowering requires one lane per block')
    if paired and mixed:raise ValueError('paired Vector launch and SDK server mode are separate layouts')
    if paired:
        source=source.replace(old,'KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);',1)
    elif not mixed:
        source=source.replace(old, new, 1)
    header=source[:source.index('{')]
    if 'tl::ascend::pv_module_bind(' in source and 'int32_t m,' in header:
        # The resident host schedule deliberately preserves the native core
        # grid. A grid-stride job is bid + iteration*blocks, so bid>=batch
        # cannot do any numerical work. Return before creating private UB or
        # event allocators, matching the native inactive-core entry path.
        indexed_loop=('job = (bid + (iteration * blocks));' in source and 'if (job < batch)' in source)
        strided_loop=('job = bid;' in source and 'job = (job + blocks);' in source and 'job < batch' in source)
        if ('int32_t batch' not in header or not (indexed_loop or strided_loop) or 'pv_sync_all(' in source):
            raise ValueError('inactive-core lowering requires independent grid-stride jobs')
        pipe='  AscendC::TPipe pipe;'
        if source.count(pipe)!=1:raise ValueError('unexpected resident allocator entry')
        source=source.replace(pipe,'  if (static_cast<int32_t>(AscendC::GetBlockIdx()) >= batch) return;\n'+pipe,1)
    # The pinned code generator also allocates all Cube memory spaces even
    # when the DSL uses only UB. Remove declarations/initializers only after
    # proving each generated buffer has no numerical consumer.
    for name,position in (('ascend_l0a','A2'),('ascend_l0b','B2'),('ascend_l1','A1'),('ascend_l0c','CO1')):
        count=source.count(name)
        if not count:continue
        if count!=2:
            raise ValueError('Cube buffer has a consumer: '+name)
        declaration=rf'AscendC::TBuf<AscendC::TPosition::{position}>\s+{name}\s*;'
        initialization=rf'pipe\.InitBuffer\({name},\s*[0-9]+\)\s*;'
        for line in source.split('\n'):
            if name in line and not re.fullmatch(rf'\s*(?:{declaration}\s*)?(?:{initialization}\s*)?',line):
                raise ValueError('unexpected generated allocation layout: '+name)
        source='\n'.join(line for line in source.split('\n') if name not in line)
    if 'ascend_ub' in source:
        allocation=re.compile(r'\s*AscendC::TBuf<AscendC::TPosition::VECCALC> ascend_ub; pipe.InitBuffer\(ascend_ub, ([0-9]+)\);\s*')
        view=re.compile(r'\s*auto \w+ = ascend_ub.GetWithOffset<([\w:]+)>\(([0-9]+), ([0-9]+)\);\s*')
        widths={'half':2,'int8_t':1,'uint8_t':1,'int16_t':2,'uint16_t':2,'int':4,'uint':4,'uchar':1,'ushort':2,'float':4,'uint32_t':4,'int32_t':4,'int64_t':8,'uint64_t':8}
        allocated=[];maximum=0
        for line in source.splitlines():
            if 'ascend_ub' not in line:continue
            a,v=allocation.fullmatch(line),view.fullmatch(line)
            if a:allocated.append((line,int(a[1])))
            elif v and v[1] in widths:maximum=max(maximum,int(v[3])+int(v[2])*widths[v[1]])
            else:raise ValueError('unbounded or unknown UB consumer: '+line.strip())
        if len(allocated)==1 and maximum==0:
            source=source.replace(allocated[0][0],'',1)
        elif len(allocated)!=1 or not 0<maximum<=allocated[0][1]:
            raise ValueError('invalid static UB allocation extent')
        else:
            line,size=allocated[0];required=(maximum+31)//32*32
            source=source.replace(line,line.replace(f'ascend_ub, {size}',f'ascend_ub, {required}'),1)
    if mixed:
        # SDK registration enters the AIC server and returns on that core.
        # Its UB must be available to the SDK, rather than consumed by the
        # Vector program's otherwise-hoisted allocations and views.
        lines=source.splitlines()
        registration=[i for i,l in enumerate(lines) if 'tl::ascend::ls_cube_init(' in l]
        if len(registration)!=1:raise ValueError('missing mixed SDK registration')
        position=registration[0]
        resources=[(i,l) for i,l in enumerate(lines) if 'ascend_ub' in l]
        if not resources or any(i>=position for i,_ in resources):
            raise ValueError('unexpected mixed UB declaration placement')
        moved=[l for _,l in resources]
        names=[]
        for _,l in resources:
            m=re.search(r'auto (\w+) = ascend_ub.GetWithOffset',l)
            if m:names.append(m[1])
        before='\n'.join(l for i,l in enumerate(lines[:position]) if i not in {j for j,_ in resources})
        before=before[before.index('extern "C" __global__ __aicore__ void linalg_pinv_tilelang_kernel('):]
        if any(re.search(r'\b'+name+r'\b',before) for name in names):
            raise ValueError('UB view consumed before mixed SDK registration')
        source='\n'.join(l for i,l in enumerate(lines) if i not in {j for j,_ in resources})
        line=lines[position]
        source=source.replace(line,line+'\n'+'\n'.join(moved),1)
    if 'AscendC::TPipe pipe;' in source:
        # The pinned emitter destroys the allocator before the program, while
        # SDK instructions such as FetchEventID still use its event pools.
        # Keep that same pipe alive through the complete device function.
        start=source.index('extern "C" __global__ __aicore__ void linalg_pinv_tilelang_kernel(')
        end=source.index('\nvoid linalg_pinv_tilelang_kernel_tiling()',start)
        early='  pipe.Destroy();\n'
        if source.count(early)!=1 or not start<source.index(early)<end:
            raise ValueError('unexpected TPipe lifetime layout')
        if re.search(r'tl::ascend::pv_\w+\(',source[start:source.index(early)]):
            raise ValueError('numerical instruction precedes the allocator lifetime boundary')
        source=source.replace(early,'',1)
        end=source.index('\nvoid linalg_pinv_tilelang_kernel_tiling()',start)
        close=source.rfind('\n}',start,end)
        if close<0:raise ValueError('missing device function boundary')
        source=source[:close]+'\n  pipe.Destroy();'+source[close:]
    if mixed:
        # End SDK client lifetime before destroying its event allocator.
        source=source.replace('AscendC::TPipe pipe;','AscendC::TPipe pipe;\n  {',1)
        source=source.replace('\n  pipe.Destroy();','\n  }\n  pipe.Destroy();',1)
    if 'tl::ascend::pv_module_bind(' in source and 'ascend_ub' not in source:
        # A composition entry owns no local memory or hardware events. Each
        # generated DSL subroutine owns its own allocator and event lifetime.
        if 'ascend_ub' in source or source.count('AscendC::TPipe pipe;')!=1 or source.count('pipe.Destroy();')!=1:
            raise ValueError('unexpected allocator consumer in the composition entry')
        without=source.replace('AscendC::TPipe pipe;','').replace('pipe.Destroy();','')
        if re.search(r'\bpipe\b',without):raise ValueError('composition entry has another allocator consumer')
        source=without
    return source


@contextmanager
def vector_only_backend(mixed=False,paired=False):
    from tilelang.jit.adapter.wrapper import TLWrapper
    import tilelang,tvm
    previous = TLWrapper.wrap
    previous_vid=tilelang.transform.AscendVidReduction
    previous_host=tilelang.transform.HostProcesser
    @tvm.tir.transform.prim_func_pass(opt_level=0,name='PinvVerifiedSingleVector')
    def single_vector(function,module,context):
        # Upstream e909216/src/transform/ascend_vid_reduction.cc rewrites
        # buffers only for threadIdx.x == 2. For one thread it is an identity
        # traversal through IRMutatorWithAnalyzer, whose range analysis stalls
        # on the dynamic packed-lane indices. Prove that no rewrite applies.
        def check(node):
            if isinstance(node,tvm.tir.AttrStmt) and node.attr_key=='thread_extent':
                if getattr(node.node,'thread_tag','')=='threadIdx.x':
                    if not isinstance(node.value,tvm.tir.IntImm) or int(node.value)!=1:
                        raise ValueError('single-Vector compatibility pass cannot handle multiple Vector threads')
        tvm.tir.stmt_functor.post_order_visit(function.body,check)
        return function
    tilelang.transform.AscendVidReduction=lambda:single_vector
    @tvm.tir.transform.prim_func_pass(opt_level=0,name='PinvRuntimeHostABI')
    def runtime_host(function,module,context):
        # Pinned ascend_host.cc never populates either host-tiling attribute;
        # its visitors only delegate to IRMutatorWithAnalyzer. Runtime shapes
        # are explicit scalar ABI arguments here. Keep that IR and add the
        # same empty attributes without pathological interval propagation.
        single_vector_check=[]
        def check(node):
            if isinstance(node,tvm.tir.AttrStmt) and node.attr_key=='thread_extent' and getattr(node.node,'thread_tag','')=='threadIdx.x':
                single_vector_check.append(isinstance(node.value,tvm.tir.IntImm) and int(node.value)==1)
        tvm.tir.stmt_functor.post_order_visit(function.body,check)
        if not single_vector_check or not all(single_vector_check):raise ValueError('runtime host ABI requires the verified single-Vector layout')
        return function.with_attr('tiling_map',{}).with_attr('var_sequence',[])
    tilelang.transform.HostProcesser=lambda:runtime_host
    def wrap(self, source):
        return vector_only_source(previous(self, source),mixed=mixed,paired=paired)
    TLWrapper.wrap = wrap
    try:
        yield
    finally:
        TLWrapper.wrap = previous
        tilelang.transform.AscendVidReduction=previous_vid
        tilelang.transform.HostProcesser=previous_host


@contextmanager
def jit_compiler_options(level=3):
    """Configure the pinned JIT's compiler, without recompiling exported files.

    The pinned adapter hardcodes -O2 and exposes no compile-options argument.
    Intercept only its own bisheng invocation; keep source bytes unchanged and
    capture the actual compiler input as JIT provenance. The native CANN build
    uses -O3. A matching TIR policy attribute must distinguish the JIT cache.
    """
    if level not in {2,3}:raise ValueError('supported JIT optimization levels are 2 and 3')
    from tilelang.jit.adapter import libgen
    from pathlib import Path
    from hashlib import sha256
    import os,shutil
    original=libgen.subprocess

    class Compiler:
        destination=None
        commands=[]
        def __getattr__(self,name):return getattr(original,name)
        def run(self,command,*args,**kwargs):
            if isinstance(command,(list,tuple)) and command and command[0]=='bisheng' and '-xasc' in command:
                if command.count('-O2')!=1:raise ValueError('pinned JIT compiler options changed')
                command=[('-O'+str(level)) if x=='-O2' else x for x in command]
                sources=[Path(x) for x in command if str(x).endswith('.cpp')]
                if len(sources)!=1:raise ValueError('one JIT compiler input is required')
                data=sources[0].read_bytes()
                record={'argv':command,'source_sha256':sha256(data).hexdigest(),'optimization':'-O'+str(level)}
                if self.destination is not None:
                    target=Path(self.destination)/'jit_compiler_input.cpp';target.write_bytes(data)
                    record['saved_input']=str(target)
                self.commands.append(record)
                result=original.run(command,*args,**kwargs)
                if result.returncode==0 and os.environ.get('PINV_JIT_MEMORY_TMP')=='1':
                    # LLVM can create large temporary objects. Keep them on
                    # the caller-selected memory scratch when persistent
                    # storage is inode-constrained. That mount is noexec:
                    # persist the identical final JIT library, then make its
                    # temporary path point to the executable filesystem.
                    if self.destination is None:raise ValueError('memory-scratch JIT requires an artifact destination')
                    output=Path(command[command.index('-o')+1])
                    if not str(output.resolve()).startswith('/dev/shm/'):
                        raise ValueError('memory-scratch JIT output must be under /dev/shm')
                    saved=Path(self.destination).resolve()/'jit_loadable.so'
                    before=sha256(output.read_bytes()).hexdigest()
                    shutil.copyfile(output,saved)
                    if sha256(saved.read_bytes()).hexdigest()!=before:raise ValueError('JIT library copy changed bytes')
                    output.unlink();output.symlink_to(saved)
                    record['loadable_copy']={'path':str(saved),'sha256':before,'source_bytes_unchanged':True}
                return result
            return original.run(command,*args,**kwargs)
    compiler=Compiler();compiler.commands=[]
    libgen.subprocess=compiler
    try:yield compiler
    finally:libgen.subprocess=original
