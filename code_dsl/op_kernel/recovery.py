"""Runtime window inversion, certified precision recovery and DSL Jacobi fallback."""
from hashlib import sha256
from code_dsl.op_kernel.intrinsics import emit

def window_recovery_kernel(component_source,*,block_only=False):
    import tilelang.language as T
    from code_dsl.op_kernel.prepared_layout import use_prepared
    from code_dsl.op_kernel.block_layout import use_block
    header=component_source
    @T.prim_func
    def kernel(A:T.Tensor((1,),'float32'),X:T.Tensor((1,),'float32'),W:T.Tensor((1,),'float32'),S:T.Tensor((1,),'float32'),
     blocks:T.int32,n:T.int32,batch:T.int32,stride:T.int32,window:T.int32,atol:T.float32,rtol:T.float32,fallback_enabled:T.int32):
     T.func_attr({'global_symbol':'linalg_pinv_tilelang','pinv.scope':'window inverse with certified precision recovery'})
     with T.Kernel(blocks,threads=1,is_npu=True) as bid:
      T.import_source(header)
      pool=T.alloc_ub((4096,),'float32')
      extent=n*n
      off=T.ceildiv(2*batch*n*stride+batch*8192,128)*128
      qa=0;qy=5*extent//4;pr=5*extent//2;rr=pr+25*extent;yh=rr+extent;rh=yh+extent//2;co=rh+extent//2;pa=co+extent;qn=pa+blocks*2*n
      blocked=T.alloc_var('int32');blocked=1
      if not block_only:
       blocked=0
       if use_block(n,stride,batch,blocks):blocked=1
      if blocked!=0:emit('bf_block',A.access_ptr('r'),X.access_ptr('rw'),W.access_ptr('rw'),blocks,n,batch)
      with T.Scope('C'):
       emit('pv_sync_init',W.access_ptr('rw'))
       if blocked!=0:
        emit('pv_cross_wait2',W.access_ptr('rw'),8)
        emit('br_gemm',W.access_ptr('rw'),2*batch*n*n,8*batch*n*n,4*batch*n*n,blocks,batch,n,n,n)
        T.barrier_all();emit('pc_done',W.access_ptr('rw'))
        emit('pv_cross_wait2',W.access_ptr('rw'),8)
        emit('br_gemm',W.access_ptr('rw'),8*batch*n*n,9*batch*n*n,10*batch*n*n,blocks,batch,n,n,n)
        T.barrier_all();emit('pc_done',W.access_ptr('rw'))
       running=T.alloc_var('int32');running=1
       while running!=0:
        emit('pv_cross_wait2',W.access_ptr('rw'),8)
        emit('pc_invalidate',S.access_ptr('rw',offset=batch*512))
        next_job=T.Cast('int32',S[batch*512])
        if next_job<0:running=0
        else:
         emit('fr_integer',W.access_ptr('rw'),off,qa,qy,pr,blocks,n)
         T.barrier_all();emit('pc_done',W.access_ptr('rw'))
         emit('pv_cross_wait2',W.access_ptr('rw'),8)
         emit('fr_half',W.access_ptr('rw'),off,yh,rh,co,blocks,n)
         T.barrier_all()
        emit('pc_done',W.access_ptr('rw'))
      with T.Scope('V'):
       T.reads(A[0],W[0],S[0]);T.writes(X[0],W[0],S[0])
       emit('fr_bind',A.access_ptr('r'),X.access_ptr('rw'),W.access_ptr('rw'))
       if blocked!=0:
        emit('bf_pack',W.access_ptr('rw'),blocks,n,batch)
        emit('pv_sync_all',W.access_ptr('rw'));emit('pv_cross_set2',W.access_ptr('rw'),8);emit('pv_cross_wait2',W.access_ptr('rw'),10);emit('pv_sync_all',W.access_ptr('rw'))
        emit('bf_decode',W.access_ptr('rw'),blocks,n,batch)
        emit('pv_sync_all',W.access_ptr('rw'));emit('pv_cross_set2',W.access_ptr('rw'),8);emit('pv_cross_wait2',W.access_ptr('rw'),10);emit('pv_sync_all',W.access_ptr('rw'))
        emit('bf_finish',W.access_ptr('rw'),blocks,n,batch,stride)
       else:
        if use_prepared(n,stride,batch,blocks,window):
         emit('fr_prepared_raw',W.access_ptr('rw'),blocks,n,batch,stride,window,atol,rtol,off+qn+blocks*32)
        else:
         emit('fr_raw',W.access_ptr('rw'),blocks,n,batch,stride,window,atol,rtol,off+qn+blocks*32)
       cores=blocks*2
       group_size=cores//batch
       for job in T.serial(batch):
        metadata=2*batch*n*stride+job*8192
        emit('pv_load_rows',pool.access_ptr('w'),W.access_ptr('r',offset=metadata+2048),1,group_size*8,0,0)
        emit('pv_load_rows',pool.access_ptr('w',offset=384),W.access_ptr('r',offset=metadata),1,64,0,0)
        T.set_flag('MTE2','S',0);T.wait_flag('MTE2','S',0)
        inverse=T.alloc_var('float32');inverse=T.float32(0)
        peak_cached=T.alloc_var('float32');peak_cached=T.float32(0)
        valid=T.alloc_var('int32');valid=1
        for c in T.serial(group_size):
         if pool[c*8]<T.float32(.5):valid=0
         inverse+=pool[c*8+1]
         peak_cached=T.max(peak_cached,pool[c*8+2])
        power=pool[416];total=pool[417]
        pool[3648]=total;pool[3649]=inverse
        T.set_flag('S','V',0);T.wait_flag('S','V',0)
        emit('pv_sqrt',pool.access_ptr('w',offset=3648),pool.access_ptr('r',offset=3648),2)
        T.set_flag('V','S',0);T.wait_flag('V','S',0)
        af=pool[3648]*power*T.float32(1.001);yf=pool[3649]/power*T.float32(1.001)
        cutoff=T.max(pool[3648]*rtol,atol/power)
        if not (inverse>T.float32(0) and inverse*cutoff*cutoff<T.float32(81) and inverse*total<T.float32(1e16)):valid=0
        refine=T.alloc_var('int32');refine=0
        if valid!=0 and inverse*total>=T.float32(3.6e9):refine=1
        sy=T.alloc_var('float32');sy=T.float32(1)
        sa=power*T.float32(8)
        if refine!=0:
         peak=peak_cached
         if not (peak>T.float32(0) and peak<T.float32(1e20)):refine=0;valid=0
         if refine!=0:
          while sy<peak*T.float32(4):sy*=T.float32(2)
          while peak>T.float32(0) and sy*T.float32(.5)>=peak*T.float32(4):sy*=T.float32(.5)
          emit('fr_quant',W.access_ptr('rw'),off,qa,qy,qn,cores,n,job,sa,sy)
        if bid==0:
         pool[3584]=T.Cast('float32',refine)
         T.set_flag('S','MTE3',0);T.wait_flag('S','MTE3',0)
         emit('pv_store_rows',S.access_ptr('w',offset=job*512),pool.access_ptr('r',offset=3584),1,8,0,0)
        if refine!=0:
         if bid==0:
          pool[3600]=T.Cast('float32',job)
          T.set_flag('S','MTE3',0);T.wait_flag('S','MTE3',0)
          emit('pv_store_rows',S.access_ptr('w',offset=batch*512),pool.access_ptr('r',offset=3600),1,16,0,0)
         emit('pv_sync_all',W.access_ptr('rw'));emit('pv_cross_set2',W.access_ptr('rw'),8);emit('pv_cross_wait2',W.access_ptr('rw'),10);emit('pv_sync_all',W.access_ptr('rw'))
        if refine!=0:
         emit('fr_decode',W.access_ptr('rw'),off,pr,rr,yh,rh,cores,n,job,sa,sy)
         emit('fr_norms',W.access_ptr('rw'),off,rr,pa,cores,n)
        if refine!=0:
         emit('pv_sync_all',W.access_ptr('rw'));emit('pv_cross_set2',W.access_ptr('rw'),8);emit('pv_cross_wait2',W.access_ptr('rw'),10);emit('pv_sync_all',W.access_ptr('rw'))
        if refine!=0:
         emit('fr_certify',W.access_ptr('rw'),off,co,pa,qn,job*512+32,cores,n,job,sa,sy,af,yf)
         emit('pv_sync_all',W.access_ptr('rw'))
         emit('pv_load_rows',pool.access_ptr('w'),S.access_ptr('r',offset=job*512+32),1,cores*8,0,0)
         T.set_flag('MTE2','S',0);T.wait_flag('MTE2','S',0)
         corrected=T.alloc_var('float32');corrected=T.float32(0)
         for c in T.serial(cores):
          if pool[c*8]<T.float32(.5):valid=0
          corrected+=pool[c*8+1]
         absolute_cutoff=T.max(af*rtol,atol)
         if not corrected*absolute_cutoff*absolute_cutoff<T.float32(81):valid=0
        if valid==0 and fallback_enabled!=0:
         emit('pv_sync_all',W.access_ptr('rw'))
         fallback_status=off+qn+cores*16+batch*32
         if bid==job:
          emit('fr_fallback',W.access_ptr('rw'),fallback_status,blocks,job,n,batch,atol,rtol)
          emit('pc_invalidate',W.access_ptr('rw',offset=fallback_status+job*32))
         emit('pv_sync_all',W.access_ptr('rw'))
         emit('pv_load_rows',pool.access_ptr('w',offset=3616),W.access_ptr('r',offset=fallback_status+job*32),1,8,0,0)
         T.set_flag('MTE2','S',0);T.wait_flag('MTE2','S',0)
         valid=T.Cast('int32',pool[3616])
        if bid==0:
         pool[3616]=T.Cast('float32',valid)
         T.set_flag('S','MTE3',0);T.wait_flag('S','MTE3',0)
         emit('pv_store_rows',S.access_ptr('w',offset=job*512+16),pool.access_ptr('r',offset=3616),1,8,0,0)
        emit('pv_sync_all',W.access_ptr('rw'))
       if bid==0:
        pool[3600]=T.float32(-1)
        T.set_flag('S','MTE3',0);T.wait_flag('S','MTE3',0)
        emit('pv_store_rows',S.access_ptr('w',offset=batch*512),pool.access_ptr('r',offset=3600),1,16,0,0)
       emit('pv_sync_all',W.access_ptr('rw'));emit('pv_cross_set2',W.access_ptr('rw'),8);emit('pv_cross_wait2',W.access_ptr('rw'),10);emit('pv_sync_all',W.access_ptr('rw'))
      T.barrier_all()
    return kernel.with_attr("pinv.strategy","block_cube" if block_only else "window_cube").with_attr("pinv.complete_continuation",True).with_attr("pinv.component_source.sha256",sha256(header.encode()).hexdigest()).with_attr("pinv.jit.optimization_level",3)
