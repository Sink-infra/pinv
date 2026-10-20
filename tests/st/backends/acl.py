"""Small ACL buffer/stream binding shared by both measured DSL variants.

Run exclusively inside a foreground child of npu_runtime.NpuSession. This
avoids the CANN TBE TVM / TileLang TVM namespace collision in torch-npu 2.6.
There is no host numerical solver, operator implementation, or CPU fallback.
"""
import ctypes as C
import numpy as np


class AclRuntime:
    def __init__(self, device=0):
        self.device = device
        self.lib = C.CDLL("libascendcl.so")
        self.lib.aclGetRecentErrMsg.restype = C.c_char_p
        self.lib.aclrtMalloc.argtypes = [C.POINTER(C.c_void_p), C.c_size_t, C.c_int]
        self.lib.aclrtMemcpy.argtypes = [C.c_void_p, C.c_size_t, C.c_void_p, C.c_size_t, C.c_int]
        self.lib.aclrtFree.argtypes = [C.c_void_p]
        self.lib.aclrtSynchronizeStream.argtypes = [C.c_void_p]
        self.buffers = []
        self.check(self.lib.aclInit(None))
        self.check(self.lib.aclrtSetDevice(device))
        self.context, self.stream = C.c_void_p(), C.c_void_p()
        self.check(self.lib.aclrtCreateContext(C.byref(self.context), device))
        self.check(self.lib.aclrtCreateStream(C.byref(self.stream)))
        self.lib.aclrtGetSocName.restype = C.c_char_p
        self.soc = self.lib.aclrtGetSocName().decode()

    def check(self, result):
        if result:
            detail = self.lib.aclGetRecentErrMsg()
            raise RuntimeError(f"ACL status {result}: {detail.decode(errors='replace') if detail else ''}")

    def allocate(self, count, values=None, dtype=np.float32):
        ptr = C.c_void_p()
        size = max(int(count) * np.dtype(dtype).itemsize, 32)
        self.check(self.lib.aclrtMalloc(C.byref(ptr), size, 0))
        self.buffers.append(ptr)
        if values is not None:
            data = np.ascontiguousarray(values, dtype=dtype)
            if data.nbytes > size:
                raise ValueError("initial data exceeds device allocation")
            self.check(self.lib.aclrtMemcpy(ptr, size, C.c_void_p(data.ctypes.data), data.nbytes, 1))
        return ptr

    def read(self, ptr, shape):
        data = np.empty(shape, dtype=np.float32)
        self.synchronize()
        self.check(self.lib.aclrtMemcpy(C.c_void_p(data.ctypes.data), data.nbytes, ptr, data.nbytes, 2))
        return data

    def synchronize(self):
        self.check(self.lib.aclrtSynchronizeStream(self.stream))

    def close(self):
        self.synchronize()
        for ptr in reversed(self.buffers):
            self.check(self.lib.aclrtFree(ptr))
        self.lib.aclrtDestroyStream(self.stream)
        self.lib.aclrtDestroyContext(self.context)
        self.lib.aclFinalize()


def bind_launch(library,*,module=False):
    library.call.argtypes = [C.c_void_p]*(4 if module else 3)+[C.c_int32]*4+[C.c_float]*2+[C.c_void_p]
    library.call.restype = None
    def launch(buffers,blocks,m,n,batch,atol,rtol,stream):
        library.call(*buffers,blocks,m,n,batch,atol,rtol,stream)
    return launch
