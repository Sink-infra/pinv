"""Distributed residual norms and an a posteriori certificate for the FP16 correction."""
from code_dsl.op_kernel.kernels import instruction_source
from code_dsl.op_kernel.intrinsics import emit

def certificate_kernels():
    import tilelang.language as T
    @T.prim_func
    def norm_columns(R:T.Tensor((1,),'float32'),P:T.Tensor((1,),'float32'),blocks:T.int32,n:T.int32):
     T.func_attr({'global_symbol':'linalg_pinv_tilelang','pinv.component':'residual column norm partials'})
     with T.Kernel(blocks,threads=1,is_npu=True) as core:
      T.import_source(instruction_source())
      values=T.alloc_ub((2048,),'float32');sums=T.alloc_ub((256,),'float32')
      with T.Scope('V'):
       owned=(n+blocks-1-core)//blocks
       emit('pv_load_rows',values.access_ptr('w'),R.access_ptr('r',offset=core*n),owned,n,(blocks-1)*n,0)
       T.set_flag('MTE2','V',0);T.wait_flag('MTE2','V',0)
       emit('pv_mul',values.access_ptr('w'),values.access_ptr('r'),values.access_ptr('r'),owned*n)
       emit('pv_fill',sums.access_ptr('w'),T.float32(0),n);T.pipe_barrier('V')
       for r in T.serial(owned):
        emit('pv_add',sums.access_ptr('w'),sums.access_ptr('r'),values.access_ptr('r',offset=r*n),n);T.pipe_barrier('V')
       T.set_flag('V','MTE3',0);T.wait_flag('V','MTE3',0)
       emit('pv_store_rows',P.access_ptr('w',offset=core*n),sums.access_ptr('r'),1,n,0,0)
       T.set_flag('MTE3','V',0);T.wait_flag('MTE3','V',0)
    @T.prim_func
    def certify(Y:T.Tensor((1,),'float32'),C:T.Tensor((1,),'float32'),P:T.Tensor((1,),'float32'),QN:T.Tensor((1,),'float32'),S:T.Tensor((1,),'float32'),
     blocks:T.int32,n:T.int32,job:T.int32,scale_a:T.float32,scale_y:T.float32,af:T.float32,yf:T.float32):
     T.func_attr({'global_symbol':'linalg_pinv_tilelang','pinv.component':'a posteriori precision certificate and correction'})
     with T.Kernel(blocks,threads=1,is_npu=True) as core:
      T.import_source(instruction_source())
      partials=T.alloc_ub((16384,),'float32');qn=T.alloc_ub((1024,),'float32')
      columns=T.alloc_ub((256,),'float32');colnorm=T.alloc_ub((256,),'float32')
      yy=T.alloc_ub((2048,),'float32');cc=T.alloc_ub((2048,),'float32')
      sums=T.alloc_ub((2048,),'float32');square=T.alloc_ub((256,),'float32')
      bound=T.alloc_ub((256,),'float32');budget=T.alloc_ub((256,),'float32');temp=T.alloc_ub((256,),'float32')
      red=T.alloc_ub((32,),'float32');scratch=T.alloc_ub((256,),'float32');stat=T.alloc_ub((32,),'float32')
      with T.Scope('V'):
       T.reads(Y[0],C[0],P[0],QN[0]);T.writes(Y[0],S[0])
       owned=(n+blocks-1-core)//blocks
       emit('pv_load_rows',partials.access_ptr('w'),P.access_ptr('r'),1,blocks*n,0,0)
       emit('pv_load_rows',qn.access_ptr('w'),QN.access_ptr('r'),1,blocks*16,0,0)
       emit('pv_load_rows',yy.access_ptr('w'),Y.access_ptr('r',offset=job*n*n+core*n),owned,n,(blocks-1)*n,0)
       emit('pv_load_rows',cc.access_ptr('w'),C.access_ptr('r',offset=core*n),owned,n,(blocks-1)*n,0)
       T.set_flag('MTE2','V',0);T.wait_flag('MTE2','V',0)
       emit('pv_fill',columns.access_ptr('w'),T.float32(0),n);T.pipe_barrier('V')
       for c in T.serial(blocks):
        emit('pv_add',columns.access_ptr('w'),columns.access_ptr('r'),partials.access_ptr('r',offset=c*n),n);T.pipe_barrier('V')
       emit('pv_reduce_sum',red.access_ptr('w'),columns.access_ptr('r'),scratch.access_ptr('rw'),n)
       emit('pv_sqrt',colnorm.access_ptr('w'),columns.access_ptr('r'),n);T.pipe_barrier('V')
       emit('pv_sqrt',red.access_ptr('w'),red.access_ptr('r'),1)
       factor=T.alloc_var('float32');factor=scale_y/T.float32(64);factor/=T.float32(64)
       emit('pv_scale',cc.access_ptr('w'),cc.access_ptr('r'),factor,owned*n);T.pipe_barrier('V')
       emit('pv_add',sums.access_ptr('w'),yy.access_ptr('r'),cc.access_ptr('r'),owned*n)
       T.set_flag('V','S',0);T.wait_flag('V','S',0)
       rf=T.alloc_var('float32');rf=red[0]*T.float32(1.001)
       qa2=T.alloc_var('float32');qa2=T.float32(0)
       qy2=T.alloc_var('float32');qy2=T.float32(0)
       for c in T.serial(blocks):
        qa2+=qn[c*16];qy2+=qn[c*16+8]
       stat[8]=qa2;stat[9]=qy2;stat[10]=T.Cast('float32',n)
       T.set_flag('S','V',0);T.wait_flag('S','V',0)
       emit('pv_sqrt',stat.access_ptr('w',offset=8),stat.access_ptr('r',offset=8),3)
       T.set_flag('V','S',0);T.wait_flag('V','S',0)
       da=T.alloc_var('float32');da=stat[8]*scale_a*T.float32(1.001)
       dy=T.alloc_var('float32');dy=stat[9]*scale_y*T.float32(1.001)
       # The exact digit products are accumulated with TwoSum. For 25
       # terms, the low-part accumulation is bounded by 625*u^2*B.
       # Use 1024*u^2 and bound B with the sum of digit-plane norms:
       # ||A||+deltaA+n*scaleA/255, and the corresponding inverse bound.
       # Include the final FP32 rounding of the residual and underflow.
       for step in T.unroll(6):da/=T.float32(256);dy/=T.float32(256)
       rr=T.alloc_var('float32');rr=(af+da+scale_a*T.Cast('float32',n)/T.float32(255))*(yf+dy+scale_y*T.Cast('float32',n)/T.float32(255))*T.float32(1024)
       for step in T.unroll(6):rr/=T.float32(256)
       # Balanced digits have Frobenius norm at most 128*n.
       # Omitted diagonals 6,7,8 contain 3,2,1 products respectively.
       # 3.02 exceeds (3+2/256+1/65536), including rounding slack.
       tail=T.alloc_var('float32');tail=scale_a/T.float32(65536)
       tail*=scale_y/T.float32(65536)
       tail*=T.Cast('float32',n)*T.Cast('float32',n)
       tail*=T.float32(16384)
       for step in T.unroll(4):tail/=T.float32(256)
       tail*=T.float32(3.02)
       er=T.alloc_var('float32');er=(da*yf+dy*af+da*dy+rr+T.float32(2)*rf/T.float32(256)/T.float32(256)/T.float32(256)+T.Cast('float32',n)*T.float32(1e-33)+tail)*T.float32(1.25)
       eta_a=T.alloc_var('float32');eta_a=stat[10]*scale_y
       eta_b=T.alloc_var('float32');eta_b=stat[10]/T.float32(64);eta_b/=T.float32(64)
       unit=T.alloc_var('float32');unit=T.float32(1)
       for step in T.unroll(3):eta_a/=T.float32(256);eta_b/=T.float32(256);unit/=T.float32(256)
       valid=T.alloc_var('int32');valid=1
       if not (rf>=T.float32(0) and er>=T.float32(0) and rf+er<T.float32(.5)):valid=0
       denominator=T.max(T.float32(.5),T.float32(1)-rf-er)
       corrected_total=T.alloc_var('float32');corrected_total=T.float32(0)
       worst=T.alloc_var('float32');worst=T.float32(-1e30)
       for row in T.serial(owned):
        emit('pv_mul',square.access_ptr('w'),yy.access_ptr('r',offset=row*n),yy.access_ptr('r',offset=row*n),n);T.pipe_barrier('V')
        emit('pv_reduce_sum',red.access_ptr('w'),square.access_ptr('r'),scratch.access_ptr('rw'),n)
        emit('pv_mul',square.access_ptr('w'),cc.access_ptr('r',offset=row*n),cc.access_ptr('r',offset=row*n),n);T.pipe_barrier('V')
        emit('pv_reduce_sum',red.access_ptr('w',offset=8),square.access_ptr('r'),scratch.access_ptr('rw'),n)
        T.pipe_barrier('V');emit('pv_sqrt',red.access_ptr('w'),red.access_ptr('r'),16)
        T.set_flag('V','S',0);T.wait_flag('V','S',0)
        yn=red[0]*T.float32(1.001);cn=red[8]*T.float32(1.001)
        gamma=T.float32(.0021)
        dcrow=(gamma*yn+eta_a)*rf+stat[10]*eta_b*yn+stat[10]*eta_a*eta_b
        ct=(cn+dcrow+yn*er)/denominator
        scale=gamma*yn+ct+eta_a
        constant=(yn+ct)*er+eta_b*yn+eta_a*eta_b
        emit('pv_scale',bound.access_ptr('w'),colnorm.access_ptr('r'),scale,n);T.pipe_barrier('V')
        T.tile.add(bound,bound,constant)
        emit('pv_abs',budget.access_ptr('w'),yy.access_ptr('r',offset=row*n),n)
        emit('pv_abs',temp.access_ptr('w'),cc.access_ptr('r',offset=row*n),n);T.pipe_barrier('V')
        emit('pv_add',budget.access_ptr('w'),budget.access_ptr('r'),temp.access_ptr('r'),n);T.pipe_barrier('V')
        emit('pv_scale',budget.access_ptr('w'),budget.access_ptr('r'),unit,n);T.pipe_barrier('V')
        emit('pv_add',bound.access_ptr('w'),bound.access_ptr('r'),budget.access_ptr('r'),n);T.pipe_barrier('V')
        emit('pv_scale',bound.access_ptr('w'),bound.access_ptr('r'),T.float32(1.25),n)
        emit('pv_abs',budget.access_ptr('w'),sums.access_ptr('r',offset=row*n),n);T.pipe_barrier('V')
        emit('pv_scale',budget.access_ptr('w'),budget.access_ptr('r'),T.float32(.00008),n);T.pipe_barrier('V')
        T.tile.add(budget,budget,T.float32(.00008));T.pipe_barrier('V')
        T.tile.sub(bound,bound,budget);T.pipe_barrier('V')
        emit('pv_reduce_max',red.access_ptr('w'),bound.access_ptr('r'),scratch.access_ptr('rw'),n)
        emit('pv_mul',square.access_ptr('w'),sums.access_ptr('r',offset=row*n),sums.access_ptr('r',offset=row*n),n);T.pipe_barrier('V')
        emit('pv_reduce_sum',red.access_ptr('w',offset=8),square.access_ptr('r'),scratch.access_ptr('rw'),n)
        T.set_flag('V','S',0);T.wait_flag('V','S',0)
        if not red[0]<=T.float32(0):valid=0
        worst=T.max(worst,red[0]);corrected_total+=red[8]
       stat[0]=T.Cast('float32',valid);stat[1]=corrected_total;stat[2]=worst;stat[3]=er;stat[4]=rf
       T.set_flag('S','MTE3',0);T.wait_flag('S','MTE3',0)
       emit('pv_store_rows',S.access_ptr('w',offset=core*8),stat.access_ptr('r'),1,8,0,0)
       T.set_flag('V','MTE3',0);T.wait_flag('V','MTE3',0)
       emit('pv_store_rows',Y.access_ptr('w',offset=job*n*n+core*n),sums.access_ptr('r'),owned,n,0,(blocks-1)*n)
       T.set_flag('MTE3','V',0);T.wait_flag('MTE3','V',0)
    return {'norms':norm_columns,'certify':certify}
