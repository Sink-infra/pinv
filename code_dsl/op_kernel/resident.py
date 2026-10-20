"""Compensated resident elimination and the original inverse-norm gate.

The matrix, pivot selection, row exchange and admission decision are DSL.
Normalize/Eliminate are elementary compensated vector division and axpy
instructions, extracted from pinv_vector_precision.h without solver control.
"""
from code_dsl.op_kernel.intrinsics import emit


def make_accurate_macro():
    import tilelang.language as T

    @T.macro
    def accurate(A,X,pool,accepted,offset,n,atol,rtol):
        pad=T.alloc_var('int32');pad=T.ceildiv(n,8)*8
        width=T.alloc_var('int32');width=2*pad
        pitch=T.alloc_var('int32');pitch=width+8
        matrix=20*width+32
        low=matrix+n*pitch
        bx=low+n*pitch
        by=bx+pitch
        bu=by+pitch
        bv=bu+pitch
        red=20*width
        prod=4*width
        tmp=5*width
        emit('pv_arith_init',pool.access_ptr('rw'),width)
        magnitude=T.alloc_var('float32');magnitude=T.float32(0)
        valid=T.alloc_var('int32');valid=1
        power=T.alloc_var('float32');power=T.float32(1)
        total=T.alloc_var('float32');total=T.float32(0)
        for i in T.serial(n):
            emit('pv_fill',pool.access_ptr('w',offset=matrix+i*pitch),T.float32(0),pitch)
            emit('pv_fill',pool.access_ptr('w',offset=low+i*pitch),T.float32(0),pitch)
            T.pipe_barrier('V')
            T.set_flag('V','MTE2',0);T.wait_flag('V','MTE2',0)
            emit('pv_load_rows',pool.access_ptr('w',offset=matrix+i*pitch),A.access_ptr('r',offset=offset+i*n),1,n,0,0)
            T.set_flag('MTE2','V',0);T.wait_flag('MTE2','V',0)
            T.set_flag('MTE2','S',0);T.wait_flag('MTE2','S',0)
            emit('pv_abs',pool.access_ptr('w',offset=prod),pool.access_ptr('r',offset=matrix+i*pitch),pad)
            T.pipe_barrier('V')
            emit('pv_reduce_max',pool.access_ptr('w',offset=red),pool.access_ptr('r',offset=prod),pool.access_ptr('rw',offset=tmp),pad)
            T.set_flag('V','S',0);T.wait_flag('V','S',0)
            magnitude=T.max(magnitude,pool[red])
            pool[matrix+i*pitch+pad+i]=T.float32(1)
            T.set_flag('S','V',0);T.wait_flag('S','V',0)
        if not (magnitude>T.float32(0)):valid=0
        if valid!=0:
            while power<magnitude and power<T.float32(1.7014118e38):power*=T.float32(2)
            while power*T.float32(0.5)>=magnitude:power*=T.float32(0.5)
            for i in T.serial(n):
                emit('pv_scale',pool.access_ptr('w',offset=matrix+i*pitch),pool.access_ptr('r',offset=matrix+i*pitch),T.float32(1)/power,pad)
                T.pipe_barrier('V')
                emit('pv_mul',pool.access_ptr('w',offset=prod),pool.access_ptr('r',offset=matrix+i*pitch),pool.access_ptr('r',offset=matrix+i*pitch),pad)
                T.pipe_barrier('V')
                emit('pv_reduce_sum',pool.access_ptr('w',offset=red),pool.access_ptr('r',offset=prod),pool.access_ptr('rw',offset=tmp),pad)
                T.set_flag('V','S',0);T.wait_flag('V','S',0)
                total+=pool[red]
            for p in T.serial(n):
                if valid!=0:
                    T.set_flag('V','S',0);T.wait_flag('V','S',0)
                    pivot=T.alloc_var('int32');pivot=p
                    best=T.alloc_var('float32');best=T.float32(0)
                    for i in T.serial(p,n):
                        entry=pool[matrix+i*pitch+p]
                        absolute=T.max(entry,-entry)
                        if absolute>best:
                            best=absolute;pivot=i
                    if not (best>T.float32(0)):
                        valid=0
                    else:
                        if pivot!=p:
                            emit('pv_copy',pool.access_ptr('w',offset=bx),pool.access_ptr('r',offset=matrix+p*pitch),pitch)
                            emit('pv_copy',pool.access_ptr('w',offset=by),pool.access_ptr('r',offset=low+p*pitch),pitch)
                            T.pipe_barrier('V')
                            emit('pv_copy',pool.access_ptr('w',offset=matrix+p*pitch),pool.access_ptr('r',offset=matrix+pivot*pitch),pitch)
                            emit('pv_copy',pool.access_ptr('w',offset=low+p*pitch),pool.access_ptr('r',offset=low+pivot*pitch),pitch)
                            T.pipe_barrier('V')
                            emit('pv_copy',pool.access_ptr('w',offset=matrix+pivot*pitch),pool.access_ptr('r',offset=bx),pitch)
                            emit('pv_copy',pool.access_ptr('w',offset=low+pivot*pitch),pool.access_ptr('r',offset=by),pitch)
                            T.pipe_barrier('V')
                        T.set_flag('V','S',0);T.wait_flag('V','S',0)
                        dh=pool[matrix+p*pitch+p]
                        dl=pool[low+p*pitch+p]
                        T.set_flag('S','V',0);T.wait_flag('S','V',0)
                        emit('pv_pair_normalize',pool.access_ptr('rw'),matrix+p*pitch,low+p*pitch,bu,bv,dh,dl,width)
                        T.set_flag('V','S',0);T.wait_flag('V','S',0)
                        for i in T.serial(n):
                            if i!=p:
                                fh=pool[matrix+i*pitch+p]
                                fl=pool[low+i*pitch+p]
                                emit('pv_pair_eliminate',pool.access_ptr('rw'),matrix+i*pitch,low+i*pitch,matrix+p*pitch,low+p*pitch,fh,fl,width)
                                T.set_flag('V','S',0);T.wait_flag('V','S',0)
            if valid!=0:
                inverse_total=T.alloc_var('float32');inverse_total=T.float32(0)
                for i in T.serial(n):
                    emit('pv_mul',pool.access_ptr('w',offset=prod),pool.access_ptr('r',offset=matrix+i*pitch+pad),pool.access_ptr('r',offset=matrix+i*pitch+pad),pad)
                    T.pipe_barrier('V')
                    emit('pv_reduce_sum',pool.access_ptr('w',offset=red),pool.access_ptr('r',offset=prod),pool.access_ptr('rw',offset=tmp),pad)
                    T.set_flag('V','S',0);T.wait_flag('V','S',0)
                    inverse_total+=pool[red]
                pool[red]=total
                T.set_flag('S','V',0);T.wait_flag('S','V',0)
                emit('pv_sqrt',pool.access_ptr('w',offset=red),pool.access_ptr('r',offset=red),1)
                T.set_flag('V','S',0);T.wait_flag('V','S',0)
                cutoff=T.max(pool[red]*rtol,atol/power)
                if not (inverse_total>T.float32(0) and inverse_total*cutoff*cutoff<T.float32(0.81)):
                    valid=0
                else:
                    for i in T.serial(n):
                        emit('pv_add',pool.access_ptr('w',offset=bx),pool.access_ptr('r',offset=matrix+i*pitch+pad),pool.access_ptr('r',offset=low+i*pitch+pad),pad)
                        T.pipe_barrier('V')
                        emit('pv_scale',pool.access_ptr('w',offset=bx),pool.access_ptr('r',offset=bx),T.float32(1)/power,pad)
                        T.pipe_barrier('V')
                        T.set_flag('V','MTE3',0);T.wait_flag('V','MTE3',0)
                        T.set_flag('S','MTE3',0);T.wait_flag('S','MTE3',0)
                        emit('pv_store_rows',X.access_ptr('w',offset=offset+i*n),pool.access_ptr('r',offset=bx),1,n,0,0)
                        T.set_flag('MTE3','V',0);T.wait_flag('MTE3','V',0)
                        T.set_flag('MTE3','MTE2',0);T.wait_flag('MTE3','MTE2',0)
        accepted[0]=valid
    return accurate
