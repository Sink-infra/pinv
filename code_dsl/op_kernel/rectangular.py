"""Runtime-shaped direct Jacobi with the original posterior certificate.

P()/w_* expressions are expanded by pairdsl into elementary FP32-pair
instructions. Decomposition, rank decisions, the error certificate and
reconstruction are TileLang loops/branches. No host numerical fallback exists.
"""
from code_dsl.op_kernel.intrinsics import emit
from code_dsl.op_kernel.pairdsl import pair_macro


def make_rectangular_macro(workspace_local=False):
    import tilelang.language as T
    workspace_job_stride=0 if workspace_local else 1

    @T.macro
    def store_global(W,pool,dst,src,count):
        T.set_flag('V','MTE3',0);T.wait_flag('V','MTE3',0)
        T.set_flag('S','MTE3',0);T.wait_flag('S','MTE3',0)
        emit('pv_store_rows',W.access_ptr('w',offset=dst),pool.access_ptr('r',offset=src),1,count,0,0)
        T.set_flag('MTE3','V',0);T.wait_flag('MTE3','V',0)
        T.set_flag('MTE3','MTE2',0);T.wait_flag('MTE3','MTE2',0)

    @T.macro
    def store(W,pool,dst,src,count):
        if workspace_local:
            emit('pv_copy',W.access_ptr('w',offset=dst),pool.access_ptr('r',offset=src),count)
            T.pipe_barrier('V')
        else:
            store_global(W,pool,dst,src,count)

    @T.macro
    def load(W,pool,src,dst,count):
        if workspace_local:
            emit('pv_copy',pool.access_ptr('w',offset=dst),W.access_ptr('r',offset=src),count)
            T.pipe_barrier('V')
            T.set_flag('V','S',0);T.wait_flag('V','S',0)
        else:
            T.set_flag('V','MTE2',0);T.wait_flag('V','MTE2',0)
            emit('pv_load_rows',pool.access_ptr('w',offset=dst),W.access_ptr('r',offset=src),1,count,0,0)
            T.set_flag('MTE2','V',0);T.wait_flag('MTE2','V',0)
            T.set_flag('MTE2','S',0);T.wait_flag('MTE2','S',0)

    @T.macro
    def row(W,pool,high,low,width,i,h,l):
        load(W,pool,high+i*width,h,width)
        load(W,pool,low+i*width,l,width)

    @T.macro
    def original(A,pool,offset,column,m,n,rows,head,width,power,h,l):
        emit('pv_fill',pool.access_ptr('w',offset=h),T.float32(0),width)
        emit('pv_fill',pool.access_ptr('w',offset=l),T.float32(0),width)
        T.pipe_barrier('V')
        T.set_flag('V','MTE2',0);T.wait_flag('V','MTE2',0)
        if m>=n:
            gather=20*width+32
            emit('pv_load_rows',pool.access_ptr('w',offset=gather),A.access_ptr('r',offset=offset+column),rows,1,n-1,0)
            T.set_flag('MTE2','S',0);T.wait_flag('MTE2','S',0)
            for j in T.serial(rows):
                pool[h+j]=pool[gather+j*8]
            T.set_flag('S','V',0);T.wait_flag('S','V',0)
        else:
            emit('pv_load_rows',pool.access_ptr('w',offset=h),A.access_ptr('r',offset=offset+column*n),1,n,0,0)
            T.set_flag('MTE2','V',0);T.wait_flag('MTE2','V',0)
        emit('pv_scale',pool.access_ptr('w',offset=h),pool.access_ptr('r',offset=h),T.float32(1)/power,head)
        T.pipe_barrier('V')

    def scale(pool,h,l,a,b,fh,fl,count):
        return emit('pv_pair_scale',pool.access_ptr('rw'),h,l,a,b,fh,fl,count)
    def add(pool,h,l,a,b,c,d,count):
        return emit('pv_pair_add',pool.access_ptr('rw'),h,l,a,b,c,d,count)
    def cert_scale(pool,a,fh,fl,count):
        return emit('pv_cert_scale',pool.access_ptr('rw'),a,fh,fl,count)
    def cert_add(pool,a,b,c,v,count):
        return emit('pv_cert_add',pool.access_ptr('rw'),a,b,c,v,count)

    @pair_macro
    def rectangular(A,X,W,pool,accepted,job,m,n,batch,atol,rtol):
        rows=T.max(m,n)
        cols=T.min(m,n)
        head=T.alloc_var('int32');head=8
        basis=T.alloc_var('int32');basis=8
        while head<rows:head*=2
        while basis<cols:basis*=2
        width=head+basis
        sigma=20*width+32+8*rows
        sigma_low=sigma+basis
        retained=sigma+2*basis
        emit('pv_arith_init',pool.access_ptr('rw'),width)
        accepted[0]=0
        offset=job*m*n
        high=workspace_job_stride*job*2*cols*width
        low=high+cols*width
        power=T.alloc_var('float32');power=T.float32(1)
        peak=T.alloc_var('float32');peak=T.float32(0)
        for i in T.serial(cols):
            original(A,pool,offset,i,m,n,rows,head,width,power,0,width)
            emit('pv_abs',pool.access_ptr('w',offset=4*width),pool.access_ptr('r'),head)
            T.pipe_barrier('V')
            emit('pv_reduce_max',pool.access_ptr('w',offset=20*width),pool.access_ptr('r',offset=4*width),pool.access_ptr('rw',offset=8*width),head)
            T.set_flag('V','S',0);T.wait_flag('V','S',0)
            peak=T.max(peak,pool[20*width])
        if peak==T.float32(0):
            emit('pv_fill',pool.access_ptr('w',offset=6*width),T.float32(0),width)
            T.pipe_barrier('V')
            for output_row in T.serial(n):
                store_global(X,pool,offset+output_row*m,6*width,m)
            accepted[0]=1
        else:
            while power<peak and power<T.float32(1.7014118e38):power*=T.float32(2)
            while power*T.float32(0.5)>=peak:power*=T.float32(0.5)
            for attempt in T.serial(2):
                if accepted[0]==0:
                    for i in T.serial(cols):
                        original(A,pool,offset,i,m,n,rows,head,width,power,0,width)
                        T.set_flag('V','S',0);T.wait_flag('V','S',0)
                        pool[head+i]=T.float32(1)
                        T.set_flag('S','V',0);T.wait_flag('S','V',0)
                        store(W,pool,high+i*width,0,width)
                        store(W,pool,low+i*width,width,width)
                    sweep=T.alloc_var('int32');sweep=0
                    changed=T.alloc_var('int32');changed=1
                    limit=T.if_then_else(attempt==0,40,120)
                    orth=T.if_then_else(attempt==0,T.float32(1e-6),T.float32(2e-13))
                    while sweep<limit and changed!=0:
                        changed=0
                        for p in T.serial(cols):
                            row(W,pool,high,low,width,p,0,width)
                            for q in T.serial(p+1,cols):
                                row(W,pool,high,low,width,q,2*width,3*width)
                                w_a=dot_pair(pool,0,width,0,width,head)
                                w_d=dot_pair(pool,2*width,3*width,2*width,3*width,head)
                                w_h=dot_pair(pool,0,width,2*width,3*width,head)
                                norm_product=root(w_a.h*w_d.h)
                                if w_a.h>0 and w_d.h>0 and T.abs(value(w_h))>orth*norm_product:
                                    w_tau=(w_d-w_a)/(P(2.0,0.0)*w_h)
                                    if T.abs(w_tau.h)>T.float32(1e8):
                                        w_z=P(0.5,0.0)/w_tau
                                    else:
                                        w_root=root_pair(P(1.0,0.0)+w_tau*w_tau)
                                        w_absolute=w_tau
                                        if w_tau.h<T.float32(0):w_absolute=-w_tau
                                        w_z=P(T.if_then_else(w_tau.h<0,T.float32(-1),T.float32(1)),0.0)/(w_absolute+w_root)
                                    w_c=P(1.0,0.0)/root_pair(P(1.0,0.0)+w_z*w_z)
                                    w_sn=w_c*w_z
                                    if attempt!=0:
                                        scale(pool,4*width,5*width,0,width,w_c.h,w_c.l,width)
                                        scale(pool,6*width,7*width,2*width,3*width,-w_sn.h,-w_sn.l,width)
                                        add(pool,4*width,5*width,4*width,5*width,6*width,7*width,width)
                                        store(W,pool,high+p*width,4*width,width)
                                        store(W,pool,low+p*width,5*width,width)
                                        scale(pool,4*width,5*width,0,width,w_sn.h,w_sn.l,width)
                                        scale(pool,6*width,7*width,2*width,3*width,w_c.h,w_c.l,width)
                                        add(pool,4*width,5*width,4*width,5*width,6*width,7*width,width)
                                    else:
                                        emit('pv_scale',pool.access_ptr('w',offset=4*width),pool.access_ptr('r'),w_c.h,width)
                                        emit('pv_scale',pool.access_ptr('w',offset=6*width),pool.access_ptr('r',offset=2*width),-w_sn.h,width)
                                        T.pipe_barrier('V')
                                        emit('pv_add',pool.access_ptr('w',offset=4*width),pool.access_ptr('r',offset=4*width),pool.access_ptr('r',offset=6*width),width)
                                        emit('pv_fill',pool.access_ptr('w',offset=5*width),T.float32(0),width)
                                        T.pipe_barrier('V')
                                        store(W,pool,high+p*width,4*width,width)
                                        store(W,pool,low+p*width,5*width,width)
                                        emit('pv_scale',pool.access_ptr('w',offset=4*width),pool.access_ptr('r'),w_sn.h,width)
                                        emit('pv_scale',pool.access_ptr('w',offset=6*width),pool.access_ptr('r',offset=2*width),w_c.h,width)
                                        T.pipe_barrier('V')
                                        emit('pv_add',pool.access_ptr('w',offset=4*width),pool.access_ptr('r',offset=4*width),pool.access_ptr('r',offset=6*width),width)
                                        T.pipe_barrier('V')
                                    store(W,pool,high+q*width,4*width,width)
                                    store(W,pool,low+q*width,5*width,width)
                                    row(W,pool,high,low,width,p,0,width)
                                    changed+=1
                        sweep+=1
                    # Posterior spectral and componentwise output certificate.
                    maximum=T.alloc_var('float32');maximum=T.float32(0)
                    minimum=T.alloc_var('float32');minimum=T.float32(1.7014118e38)
                    total=T.alloc_var('float32');total=T.float32(0)
                    rank=T.alloc_var('int32');rank=0
                    for i in T.serial(cols):
                        row(W,pool,high,low,width,i,0,width)
                        w_norm=root_pair(cert_dot_pair(pool,0,width,0,width,head))
                        magnitude=value(w_norm)
                        pool[sigma+i]=w_norm.h
                        pool[sigma_low+i]=w_norm.l
                        maximum=T.max(maximum,magnitude)
                        total+=magnitude*magnitude
                    cutoff=T.max(atol/power,rtol*maximum)
                    for i in T.serial(cols):
                        magnitude=pool[sigma+i]+pool[sigma_low+i]
                        if magnitude>cutoff:pool[retained+i]=T.float32(1)
                        else:pool[retained+i]=T.float32(0)
                        if magnitude>cutoff:
                            rank+=1
                            minimum=T.min(minimum,magnitude)
                    T.set_flag('S','V',0);T.wait_flag('S','V',0)
                    w_ov=P(0.0,0.0);w_ou=P(0.0,0.0)
                    w_r1=P(0.0,0.0);w_r2=P(0.0,0.0)
                    discarded_gram=T.alloc_var('float32');discarded_gram=T.float32(0)
                    v_gram_total=T.alloc_var('float32');v_gram_total=T.float32(0)
                    u_gram_total=T.alloc_var('float32');u_gram_total=T.float32(0)
                    for i in T.serial(cols):
                        row(W,pool,high,low,width,i,0,width)
                        w_ni=P(pool[sigma+i],pool[sigma_low+i])
                        row_bound=T.alloc_var('float32');row_bound=T.float32(0)
                        for j in T.serial(cols):
                            row(W,pool,high,low,width,j,2*width,3*width)
                            w_vv=cert_dot_pair(pool,head,width+head,2*width+head,3*width+head,basis)
                            if i==j:v_gram_total+=value(w_vv)
                            w_v=w_vv-P(T.if_then_else(i==j,T.float32(1),T.float32(0)),0.0)
                            w_ov=w_ov+w_v*w_v
                            w_cc=cert_dot_pair(pool,0,width,2*width,3*width,head)
                            if pool[retained+i]!=0 and pool[retained+j]!=0:
                                w_nj=P(pool[sigma+j],pool[sigma_low+j])
                                w_uu=w_cc/(w_ni*w_nj)
                                if i==j:u_gram_total+=value(w_uu)
                                w_u=w_uu-P(T.if_then_else(i==j,T.float32(1),T.float32(0)),0.0)
                                w_ou=w_ou+w_u*w_u
                            elif pool[retained+i]==0 and pool[retained+j]==0:
                                row_bound+=T.abs(value(w_cc))
                        discarded_gram=T.max(discarded_gram,row_bound)
                        emit('pv_fill',pool.access_ptr('w',offset=4*width),T.float32(0),width)
                        emit('pv_fill',pool.access_ptr('w',offset=5*width),T.float32(0),width)
                        emit('pv_fill',pool.access_ptr('w',offset=19*width),T.float32(0),width)
                        T.pipe_barrier('V')
                        for j in T.serial(cols):
                            original(A,pool,offset,j,m,n,rows,head,width,power,2*width,3*width)
                            T.set_flag('V','S',0);T.wait_flag('V','S',0)
                            w_vij=P(pool[head+j],pool[width+head+j])
                            T.set_flag('S','V',0);T.wait_flag('S','V',0)
                            cert_scale(pool,2*width,w_vij.h,w_vij.l,head)
                            cert_add(pool,4*width,5*width,19*width,16*width,head)
                            cert_add(pool,4*width,5*width,19*width,17*width,head)
                            cert_add(pool,4*width,5*width,19*width,18*width,head)
                            if pool[retained+i]!=0:
                                w_rhs=cert_dot_pair(pool,2*width,3*width,0,width,head)/w_ni-w_vij*w_ni
                                w_r2=w_r2+w_rhs*w_rhs
                        emit('pv_scale',pool.access_ptr('w',offset=6*width),pool.access_ptr('r'),T.float32(-1),head)
                        emit('pv_scale',pool.access_ptr('w',offset=7*width),pool.access_ptr('r',offset=width),T.float32(-1),head)
                        T.pipe_barrier('V')
                        cert_add(pool,4*width,5*width,19*width,6*width,head)
                        cert_add(pool,4*width,5*width,19*width,7*width,head)
                        emit('pv_add',pool.access_ptr('w',offset=5*width),pool.access_ptr('r',offset=5*width),pool.access_ptr('r',offset=19*width),head)
                        T.pipe_barrier('V')
                        w_r1=w_r1+cert_dot_pair(pool,4*width,5*width,4*width,5*width,head)
                    norm=T.alloc_var('float32');norm=root(total)*T.float32(1.001)
                    head_depth=T.alloc_var('int32');head_depth=0
                    basis_depth=T.alloc_var('int32');basis_depth=0
                    size=T.alloc_var('int32');size=head
                    while size>1:
                        head_depth+=1;size//=2
                    size=basis
                    while size>1:
                        basis_depth+=1;size//=2
                    dot_gamma0=T.float32(32*2**-48)+T.float32(8*2**-72)*T.Cast('float32',head_depth+4)
                    basis_gamma0=T.float32(8*2**-48)+T.float32(8*2**-72)*T.Cast('float32',basis_depth+4)
                    norm_gamma0=T.float32(32*2**-48)+T.float32(8*2**-72)*T.Cast('float32',head_depth+8)
                    product_gamma0=T.float32(8*2**-72)*T.Cast('float32',cols+4)
                    output_gamma=T.float32(8*2**-48)*T.Cast('float32',cols+4)
                    dot_gamma=dot_gamma0/(1-dot_gamma0)*T.float32(1.001)
                    basis_gamma=basis_gamma0/(1-basis_gamma0)*T.float32(1.001)
                    norm_gamma=norm_gamma0/(1-norm_gamma0)*T.float32(1.001)
                    product_gamma=product_gamma0/(1-product_gamma0)*T.float32(1.001)
                    eta_v=root(value(w_ov))*T.float32(1.001)+basis_gamma*v_gram_total*T.float32(1.001)
                    eta_u=root(value(w_ou))*T.float32(1.001)+norm_gamma*u_gram_total*T.float32(1.001)
                    certified=T.alloc_var('int32');certified=0
                    error=T.alloc_var('float32');error=T.float32(0)
                    if eta_v<T.float32(0.25) and eta_u<T.float32(0.25):
                        dv=eta_v/(1+root(1-eta_v))
                        du=eta_u/(1+root(1-eta_u))
                        first_residual=root(value(w_r1))*T.float32(1.001)
                        root_cols=root(T.Cast('float32',cols))
                        basis_denominator=root(1-eta_v)-product_gamma*root_cols
                        if basis_denominator>T.float32(0):
                            norm=(norm+first_residual)/basis_denominator*T.float32(1.001)
                            e1=first_residual+product_gamma*norm*root_cols
                            discarded=root(discarded_gram)*T.float32(1.001)+e1+norm*dv
                            root_rank=root(T.Cast('float32',rank))
                            delta=e1+root(value(w_r2))*T.float32(1.001)+dot_gamma*norm*root_rank+(norm+maximum)*(du+dv)
                            lower=maximum*T.float32(1-2**-23)-delta
                            upper=maximum*T.float32(1+2**-23)+delta
                            threshold_low=T.max(atol/power,rtol*lower)
                            threshold_high=T.max(atol/power,rtol*upper)
                            if rank==0:
                                if discarded+delta<=threshold_low:certified=1
                            else:
                                smallest=minimum*T.float32(1-2**-23)-delta
                                discarded_upper=T.if_then_else(rank==cols,T.float32(0),discarded+delta)
                                if smallest>threshold_high and (rank==cols or discarded_upper<=threshold_low) and smallest>discarded_upper:
                                    lipschitz=T.float32(1)/(smallest*(smallest-discarded_upper))
                                    error=(lipschitz*delta+(du+dv)/minimum+output_gamma*root_rank/minimum)/power*T.float32(1.01)
                                    if error>=T.float32(0) and error<T.float32(1.7014118e38):certified=1
                    if certified!=0:
                        valid=T.alloc_var('int32');valid=1
                        count=T.if_then_else(m>=n,head,basis)
                        first=T.if_then_else(m>=n,0,head)
                        for output_row in T.serial(n):
                            emit('pv_fill',pool.access_ptr('w',offset=4*width),T.float32(0),width)
                            emit('pv_fill',pool.access_ptr('w',offset=5*width),T.float32(0),width)
                            T.pipe_barrier('V')
                            for i in T.serial(cols):
                                if pool[retained+i]!=0:
                                    row(W,pool,high,low,width,i,0,width)
                                    w_norm=P(pool[sigma+i],pool[sigma_low+i])
                                    T.set_flag('V','S',0);T.wait_flag('V','S',0)
                                    position=T.if_then_else(m>=n,head+output_row,output_row)
                                    w_factor=P(pool[position],pool[width+position])/(w_norm*w_norm)
                                    T.set_flag('S','V',0);T.wait_flag('S','V',0)
                                    scale(pool,6*width,7*width,first,width+first,w_factor.h,w_factor.l,count)
                                    add(pool,4*width,5*width,4*width,5*width,6*width,7*width,count)
                            emit('pv_scale',pool.access_ptr('w',offset=4*width),pool.access_ptr('r',offset=4*width),T.float32(1)/power,count)
                            emit('pv_scale',pool.access_ptr('w',offset=5*width),pool.access_ptr('r',offset=5*width),T.float32(1)/power,count)
                            T.pipe_barrier('V')
                            emit('pv_add',pool.access_ptr('w',offset=6*width),pool.access_ptr('r',offset=4*width),pool.access_ptr('r',offset=5*width),count)
                            T.set_flag('V','S',0);T.wait_flag('V','S',0)
                            for col in T.serial(m):
                                output_value=pool[6*width+col]
                                bound=error+T.float32(2**-24/(1-2**-24))*T.abs(output_value)
                                budget=(T.float32(1e-4)+T.float32(1e-4)*T.abs(output_value))/T.float32(1.0001)
                                if not (bound<=budget):valid=0
                            store_global(X,pool,offset+output_row*m,6*width,m)
                        if valid!=0:accepted[0]=1
            if accepted[0]==0:
                emit('pv_fill',pool.access_ptr('w',offset=6*width),T.float32(0),width)
                T.pipe_barrier('V')
                emit('pv_div',pool.access_ptr('w',offset=6*width),pool.access_ptr('r',offset=6*width),pool.access_ptr('r',offset=6*width),width)
                T.pipe_barrier('V')
                for output_row in T.serial(n):
                    store_global(X,pool,offset+output_row*m,6*width,m)
    return rectangular
