"""Unmodified CANN task-clock collection, never host elapsed-time estimates."""
import ctypes as C
import csv
from pathlib import Path
import subprocess


class TaskProfiler:
    def __init__(self,runtime,path):self.runtime=runtime;self.path=Path(path).resolve();self.config=None
    def __enter__(self):
        lib=self.runtime.lib
        lib.aclprofInit.argtypes=[C.c_char_p,C.c_size_t]
        lib.aclprofCreateConfig.argtypes=[C.POINTER(C.c_uint32),C.c_uint32,C.c_int,C.c_void_p,C.c_uint64]
        lib.aclprofCreateConfig.restype=C.c_void_p
        for name in ['aclprofStart','aclprofStop','aclprofDestroyConfig']:
            getattr(lib,name).argtypes=[C.c_void_p]
        path=str(self.path).encode()
        self.runtime.check(lib.aclprofInit(path,len(path)))
        devices=(C.c_uint32*1)(self.runtime.device)
        self.config=lib.aclprofCreateConfig(devices,1,0xff,None,2)
        if not self.config:raise RuntimeError('aclprofCreateConfig failed')
        self.runtime.check(lib.aclprofStart(self.config));return self
    def __exit__(self,*exception):
        lib=self.runtime.lib
        self.runtime.synchronize()
        self.runtime.check(lib.aclprofStop(self.config))
        self.runtime.check(lib.aclprofDestroyConfig(self.config))
        self.runtime.check(lib.aclprofFinalize())


def export_samples(path,warmup,repeats,kernel='linalg_pinv_tilelang_kernel'):
    path=Path(path)
    if not list(path.rglob('task_time*.csv')):
        with (path.parent/'profile-export.log').open('w') as log:
            subprocess.run(['msprof','--export=on','--output='+str(path)],stdout=log,stderr=subprocess.STDOUT,timeout=120,check=True)
    tables=sorted(path.rglob('task_time*.csv'))
    if len(tables)!=1:raise ValueError('expected exactly one device scheduling table')
    with tables[0].open(newline='') as f:rows=list(csv.DictReader(f))
    rows=[row for row in rows if str(row.get('kernel_name','')).startswith(kernel)]
    if len(rows)!=warmup+repeats:raise ValueError(f'expected {warmup+repeats} device tasks, found {len(rows)}')
    kinds={'KERNEL_AIVEC','KERNEL_AICORE','MIX_AIC','AI_VECTOR_CORE','AI_CORE'}
    if any(row.get('kernel_type','') not in kinds for row in rows):raise ValueError('unexpected task type: '+str({r.get('kernel_type') for r in rows}))
    keys=[(row['Device_id'],row['stream_id'],row['task_id']) for row in rows]
    if len(set(keys))!=len(keys):raise ValueError('duplicate device task identity')
    from tests.common.profiling import profile_samples
    return profile_samples(path,warmup,repeats,'task',kernel_prefix=kernel),tables
