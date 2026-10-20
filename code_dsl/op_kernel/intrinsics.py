"""Lower vector instruction adapters through the backend's generic call hook.

The pinned release silently drops unknown ``tir.call_extern`` calls. Its
experimental three-buffer codegen hook emits a named primitive with explicit
buffer offsets and scalar arguments. This adapter uses that hook; it never
rewrites generated C++ or embeds an elimination/solve implementation.
"""


def emit(name, *args):
    from tvm import tir
    buffers = [a for a in args if isinstance(a, tir.Call) and a.op.name == "tir.tvm_access_ptr"]
    scalars = [a for a in args if not (isinstance(a, tir.Call) and a.op.name == "tir.tvm_access_ptr")]
    if not 1 <= len(buffers) <= 3:
        raise ValueError("instruction adapter requires one to three tensor operands")
    buffers += [buffers[0]] * (3-len(buffers))
    return tir.call_intrin("handle", tir.op.Op.get("tl.ascend_gather_mask_experiment"),
                           name, *buffers, *scalars)
