"""The baseline host schedule, evaluated using actual CANN platform data.

No input matrix is sent to this host library. It only plans tiling, selects a
computational strategy and produces runtime launch parameters. Its C++ body
is extracted from the immutable native host implementation.
"""
import ctypes as C
from dataclasses import dataclass
from pathlib import Path
import struct
from code_dsl.op_host.geometry import geometry


class NativePlan(C.Structure):
    _fields_=[('tiling',C.c_ubyte*4096),('workspace',C.c_uint64),('lib_workspace',C.c_uint64),('ub_size',C.c_uint64),
              ('tiling_bytes',C.c_uint32),('blocks',C.c_uint32),('schedule',C.c_uint32),('flags',C.c_uint32*9),
              ('aic',C.c_uint32),('aiv',C.c_uint32)]


@dataclass(frozen=True)
class Plan:
    geometry:object
    strategy:str
    blocks:int
    vector_blocks:int
    tiling:bytes
    flags:tuple
    workspace_floats:int
    status_floats:int
    native_workspace_bytes:int
    ub_bytes:int
    aic:int
    aiv:int
    schedule:int

    @property
    def fields(self):
        return dict(zip(('m','n','batch','rows','cols','stride','local','parallel','local_inverse','inverse_tile','resident','group_count','single_resident','atol','rtol'),struct.unpack('<13I2f',self.tiling[:60])))


def make_plan(shape,*,atol=0.0,rtol=None,hermitian=False,soc,library=Path('build/host-planner.so')):
    g=geometry(shape,atol=atol,rtol=rtol,hermitian=hermitian)
    lib=C.CDLL(str(library.resolve()))
    lib.pv_host_plan_size.restype=C.c_uint64
    lib.pv_host_plan_alignment.restype=C.c_uint64
    size,alignment=int(lib.pv_host_plan_size()),int(lib.pv_host_plan_alignment())
    if size<C.sizeof(NativePlan) or alignment<1 or alignment&(alignment-1):raise RuntimeError('invalid host plan ABI')
    storage=C.create_string_buffer(size+alignment-1)
    address=(C.addressof(storage)+alignment-1)&~(alignment-1)
    lib.pv_host_plan.argtypes=[C.c_int64,C.c_int64,C.c_uint64,C.c_float,C.c_float,C.c_char_p,C.c_void_p]
    result=lib.pv_host_plan(g.m,g.n,g.batch,g.atol,-1.0 if rtol is None else rtol,soc.encode(),address)
    if result:raise RuntimeError('CANN host planning failed: '+str(result))
    native=C.cast(address,C.POINTER(NativePlan)).contents
    flags=tuple(native.flags)
    compact,tail,wave,local,group,pipeline,window,cube,rect=flags
    if rect:strategy='rectangular'
    elif compact:strategy='compact'
    elif local:strategy='local'
    elif pipeline:
        strategy='window_cube' if window and cube else 'window' if window else 'peer_pipeline' if cube else 'pipeline'
    else:strategy='distributed'
    if not (native.aic and native.aiv and native.ub_size):raise RuntimeError('missing actual platform capabilities')
    rows=max(g.m,g.n);cols=min(g.m,g.n)
    head=max(8,1<<(rows-1).bit_length());basis=max(8,1<<(cols-1).bit_length())
    rect_floats=2*g.batch*cols*(head+basis)
    native_user=max(0,int(native.workspace)-int(native.lib_workspace))
    words=max(rect_floats,(native_user+3)//4)
    return Plan(g,strategy,int(native.blocks),int(native.blocks)*(2 if cube else 1),bytes(native.tiling[:native.tiling_bytes]),
                flags,words,g.batch*32,int(native.workspace),int(native.ub_size),int(native.aic),int(native.aiv),int(native.schedule))
