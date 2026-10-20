"""Shape/attribute validation only; no numerical work or CPU fallback."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Geometry:
    shape:tuple
    output_shape:tuple
    m:int
    n:int
    batch:int
    atol:float
    rtol:float


def geometry(shape,*,atol=0.0,rtol=None,hermitian=False):
    shape=tuple(shape)
    if len(shape)<2 or any(type(v) is not int for v in shape):raise ValueError('dense matrix shape required')
    m,n=shape[-2:]
    if not (1<=m<=512 and 1<=n<=512):raise ValueError('matrix axes must be in [1,512]')
    if any(not 1<=v<=64 for v in shape[:-2]):raise ValueError('batch axes must be in [1,64]')
    batch=math.prod(shape[:-2])
    if batch>4294967295:raise ValueError('batch index overflows the original ABI')
    if hermitian:raise ValueError('hermitian=True is unsupported')
    atol=float(atol)
    if not math.isfinite(atol) or atol<0:raise ValueError('nonnegative finite scalar atol required')
    resolved=max(m,n)*2**-23 if atol==0 else 0.0
    if rtol is not None:
        resolved=float(rtol)
        if not math.isfinite(resolved) or resolved<0:raise ValueError('nonnegative finite scalar rtol required')
    return Geometry(shape,shape[:-2]+(n,m),m,n,batch,atol,resolved)
