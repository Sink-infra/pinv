"""TileLang scheduling for the native two-level pivot reduction."""
from code_dsl.op_kernel.intrinsics import emit


def make_argmax_macro():
    import tilelang.language as T

    @T.macro
    def argmax(pool,values,chunks,result,length,best,position):
        full=length//64
        tail=length%64
        if full>0:
            emit('pv_whole_max',pool.access_ptr('w',offset=chunks),pool.access_ptr('r',offset=values),64,full)
        if tail>0:
            emit('pv_whole_max',pool.access_ptr('w',offset=chunks+2*full),pool.access_ptr('r',offset=values+64*full),tail,1)
        T.pipe_barrier('V')
        emit('pv_even_max',pool.access_ptr('w',offset=result),pool.access_ptr('r',offset=chunks),full+T.if_then_else(tail>0,1,0))
        T.set_flag('V','S',0);T.wait_flag('V','S',0)
        # Keep the SDK reduction outputs in DSL scalar variables. The caller
        # reconstructs the two-level index; no UB result spill is needed.
        emit('pv_reduce_scalars',pool.access_ptr('rw'),best.buffer.data,position.buffer.data)
    return argmax
