"""Four runtime block stages with condition-based block row pivot selection.

Private block factorization. The containing window strategy supplies precision
recovery, acceptance checks and the Jacobi continuation.
"""
from code_dsl.op_kernel.intrinsics import emit

def raw_kernel(header):
 import tilelang.language as T
 @T.prim_func
 def kernel(A:T.Tensor((1,),'float32'),Y:T.Tensor((1,),'float32'),W:T.Tensor((1,),'float32'),S:T.Tensor((1,),'float32'),
            blocks:T.int32,n:T.int32,batch:T.int32):
  T.func_attr({'global_symbol':'linalg_pinv_tilelang','pinv.scope':'private block inverse with lazy identity columns'})
  with T.Kernel(blocks,threads=1,is_npu=True) as bid:
   T.import_source(header)
   pool=T.alloc_ub((24576,),'float32')
   cores=blocks*2
   row_tile=T.alloc_var('int32');row_tile=8
   if batch*n>=cores*32:row_tile=32
   extent=n*n;aug=batch*extent*2
   source=T.alloc_var('int32');source=0;swapped=T.alloc_var('int32');swapped=aug
   ca=2*aug;ci=ca+batch*n*64
   diag=ci+batch*n*64
   panel_input=diag+batch*4096;panel=panel_input+batch*64*2*n
   factors=panel+batch*64*2*n;update=factors+batch*n*64
   scores=update+aug;status=scores+batch*n
   with T.Scope('C'):
    emit('pv_sync_init',W.access_ptr('rw'))
    cube_swap=T.alloc_var('int32');cube_swap=aug
    for stage in T.serial(n//64):
     cube_swap=aug
     if stage%2!=0:cube_swap=0
     emit('pv_cross_wait2',W.access_ptr('rw'),8)
     emit('br_chain',W.access_ptr('rw'),factors,diag,panel_input,panel,cube_swap,blocks,batch,n,stage*64)
     T.barrier_all();emit('pc_done',W.access_ptr('rw'))
   with T.Scope('V'):
    T.reads(A[0],W[0],S[0]);T.writes(Y[0],W[0],S[0])
    emit('fr_bind',A.access_ptr('r'),Y.access_ptr('rw'),W.access_ptr('rw'))
    emit('pv_sync_init',W.access_ptr('rw'))
    # Per-matrix power-of-two normalization, independent of fixture identity.
    group=cores//batch
    if bid<batch*group:
     job=bid//group;lane=bid%group
     magnitude=T.alloc_var('float32');magnitude=T.float32(0)
     for chunk in T.serial(T.ceildiv(extent,4096*group)):
      begin=T.alloc_var('int32');begin=(lane+chunk*group)*4096
      if begin<extent:
       count=T.alloc_var('int32');count=extent-begin
       if count>4096:count=4096
       emit('pv_load_rows',pool.access_ptr('w'),A.access_ptr('r',offset=job*extent+begin),1,count,0,0)
       T.set_flag('MTE2','V',0);T.wait_flag('MTE2','V',0)
       emit('pv_abs',pool.access_ptr('w'),pool.access_ptr('r'),count);T.pipe_barrier('V')
       emit('pv_reduce_max',pool.access_ptr('w',offset=16384),pool.access_ptr('r'),pool.access_ptr('rw',offset=4096),count)
       T.set_flag('V','S',0);T.wait_flag('V','S',0)
       magnitude=T.max(magnitude,pool[16384])
     pool[0]=magnitude
     T.set_flag('S','MTE3',0);T.wait_flag('S','MTE3',0)
     emit('pv_store_rows',S.access_ptr('w',offset=batch*32+bid*8),pool.access_ptr('r'),1,8,0,0)
     emit('pv_all_barrier',W.access_ptr('rw'))
    emit('pv_sync_all',W.access_ptr('rw'))
    if bid<batch:
     emit('pv_load_rows',pool.access_ptr('w'),S.access_ptr('r',offset=batch*32+bid*group*8),1,group*8,0,0)
     T.set_flag('MTE2','S',0);T.wait_flag('MTE2','S',0)
     magnitude=T.alloc_var('float32');magnitude=T.float32(0)
     for lane in T.serial(group):magnitude=T.max(magnitude,pool[lane*8])
     power=T.alloc_var('float32');power=T.float32(1)
     if magnitude>T.float32(0) and magnitude<T.float32(1e30):
      while power<magnitude:power*=T.float32(2)
      while power*T.float32(.5)>=magnitude:power*=T.float32(.5)
     pool[0]=power
     T.set_flag('S','MTE3',0);T.wait_flag('S','MTE3',0)
     emit('pv_store_rows',S.access_ptr('w',offset=bid*32),pool.access_ptr('r'),1,8,0,0)
     emit('pv_all_barrier',W.access_ptr('rw'))
    emit('pv_sync_all',W.access_ptr('rw'))
    emit('pv_load_rows',pool.access_ptr('w',offset=20480),S.access_ptr('r'),1,batch*32,0,0)
    T.set_flag('MTE2','S',0);T.wait_flag('MTE2','S',0)
    for iteration in T.serial(T.ceildiv(batch*n//row_tile,cores)):
     row=(bid+iteration*cores)*row_tile
     if row<batch*n:
      job=row//n;power=pool[20480+job*32]
      emit('pv_load_rows',pool.access_ptr('w'),A.access_ptr('r',offset=row*n),row_tile,n,0,0)
      T.set_flag('MTE2','V',0);T.wait_flag('MTE2','V',0)
      emit('pv_scale',pool.access_ptr('w'),pool.access_ptr('r'),T.float32(1)/power,row_tile*n)
      T.set_flag('V','MTE3',0);T.wait_flag('V','MTE3',0)
      emit('pv_store_rows',W.access_ptr('w',offset=source+row*n),pool.access_ptr('r'),row_tile,n,0,0)
      emit('pv_store_rows',W.access_ptr('w',offset=11*batch*extent+row*n),pool.access_ptr('r'),row_tile,n,0,0)
      emit('pv_all_barrier',W.access_ptr('rw'))
    emit('pv_sync_all',W.access_ptr('rw'))
    for stage in T.serial(n//64):
     st=stage*64;remaining=n//64-stage;candidates=batch*remaining
     source=0;swapped=aug
     if stage%2!=0:
      source=aug;swapped=0
     # All remaining block candidates are independent local64 DSL inverses.
     if bid<candidates:
      job=bid//remaining;position=st+bid%remaining*64
      emit('br_quick_strided',W.access_ptr('rw'),source+job*extent+position*n,ci+bid*4096,status+bid*32,blocks,n)
      admitted=T.alloc_var('int32');admitted=T.Cast('int32',W[status+bid*32])
      cost=T.alloc_var('float32');cost=T.float32(1e30)
      if admitted!=0:cost=W[status+bid*32+1]
      pool[0]=cost
      T.set_flag('S','MTE3',0);T.wait_flag('S','MTE3',0)
      emit('pv_store_rows',W.access_ptr('w',offset=scores+bid*8),pool.access_ptr('r'),1,8,0,0)
      emit('pv_all_barrier',W.access_ptr('rw'))
     emit('pv_sync_all',W.access_ptr('rw'))
     emit('pv_load_rows',pool.access_ptr('w',offset=18432),W.access_ptr('r',offset=scores),1,candidates*8,0,0)
     T.set_flag('MTE2','S',0);T.wait_flag('MTE2','S',0)
     for iteration in T.serial(T.ceildiv(batch*n//row_tile,cores)):
      row=(bid+iteration*cores)*row_tile
      if row<batch*n:
       job=row//n;i=row%n
       chosen=T.alloc_var('int32');chosen=0
       best=T.alloc_var('float32');best=T.float32(1e30)
       for candidate in T.serial(remaining):
        cost=pool[18432+(job*remaining+candidate)*8]
        if cost>=T.float32(0) and cost<best:chosen=candidate;best=cost
       position=st+chosen*64
       if i==st:
        pool[0]=best;pool[1]=T.Cast('float32',position)
        T.set_flag('S','MTE3',0);T.wait_flag('S','MTE3',0)
        emit('pv_store_rows',S.access_ptr('w',offset=512+(stage*batch+job)*32),pool.access_ptr('r'),1,8,0,0)
        emit('pv_all_barrier',W.access_ptr('rw'))
       original=T.alloc_var('int32');original=i
       if i>=st and i<st+64:original=position+i-st
       elif i>=position and i<position+64:original=st+i-position
       emit('pv_load_rows',pool.access_ptr('w',offset=16384),W.access_ptr('r',offset=source+job*extent+original*n),row_tile,64,n-64,0)
       T.set_flag('MTE2','V',0);T.wait_flag('MTE2','V',0)
       if i>=st and i<st+64:emit('pv_fill',pool.access_ptr('w',offset=16384),T.float32(0),row_tile*64)
       else:emit('pv_scale',pool.access_ptr('w',offset=16384),pool.access_ptr('r',offset=16384),T.float32(-1),row_tile*64)
       T.set_flag('V','MTE3',0);T.wait_flag('V','MTE3',0)
       emit('pv_store_rows',W.access_ptr('w',offset=factors+row*64),pool.access_ptr('r',offset=16384),row_tile,64,0,0)
       emit('pv_fill',pool.access_ptr('w'),T.float32(0),row_tile*n)
       T.set_flag('V','MTE2',0);T.wait_flag('V','MTE2',0)
       emit('pv_load_rows',pool.access_ptr('w'),W.access_ptr('r',offset=source+job*extent+original*n+64),row_tile,n-64,64,64)
       T.set_flag('MTE2','S',0);T.wait_flag('MTE2','S',0)
       if i>=st and i<st+64:
        for lane in T.serial(row_tile):pool[lane*n+n-64+i-st+lane]=T.float32(1)
       T.set_flag('S','MTE3',0);T.wait_flag('S','MTE3',0)
       emit('pv_store_rows',W.access_ptr('w',offset=swapped+row*n),pool.access_ptr('r'),row_tile,n,0,0)
       emit('pv_all_barrier',W.access_ptr('rw'))
       if i>=st and i<st+64:
        emit('pv_store_rows',W.access_ptr('w',offset=panel_input+job*64*n+(i-st)*n),pool.access_ptr('r'),row_tile,n,0,0)
        emit('pv_all_barrier',W.access_ptr('rw'))
        emit('pv_load_rows',pool.access_ptr('w',offset=16384),W.access_ptr('r',offset=ci+(job*remaining+chosen)*4096+(i-st)*64),row_tile,64,0,0)
        T.set_flag('MTE2','MTE3',0);T.wait_flag('MTE2','MTE3',0)
        emit('pv_store_rows',W.access_ptr('w',offset=diag+job*4096+(i-st)*64),pool.access_ptr('r',offset=16384),row_tile,64,0,0)
        emit('pv_all_barrier',W.access_ptr('rw'))
     emit('pv_sync_all',W.access_ptr('rw'));emit('pv_cross_set2',W.access_ptr('rw'),8);emit('pv_cross_wait2',W.access_ptr('rw'),10)
     emit('pv_sync_all',W.access_ptr('rw'))
    source=0
    if (n//64)%2!=0:source=aug
    emit('pv_load_rows',pool.access_ptr('w',offset=22016),S.access_ptr('r',offset=512),1,512,0,0)
    T.set_flag('MTE2','S',0);T.wait_flag('MTE2','S',0)
    for iteration in T.serial(T.ceildiv(batch*n//row_tile,cores)):
     row=(bid+iteration*cores)*row_tile
     if row<batch*n:
      job=row//n
      for col in T.serial(n//64):pool[21200+col]=T.Cast('float32',col)
      for stage in T.serial(n//64):
       chosen=T.Cast('int32',pool[22016+(stage*batch+job)*32+1])//64
       old=pool[21200+stage];pool[21200+stage]=pool[21200+chosen];pool[21200+chosen]=old
      for col in T.serial(n//64):
       original=T.Cast('int32',pool[21200+col])
       emit('pv_load_rows',pool.access_ptr('w'),W.access_ptr('r',offset=source+row*n+col*64),row_tile,64,n-64,0)
       T.set_flag('MTE2','MTE3',0);T.wait_flag('MTE2','MTE3',0)
       emit('pv_store_rows',W.access_ptr('w',offset=8*batch*extent+row*n+original*64),pool.access_ptr('r'),row_tile,64,0,n-64)
       emit('pv_all_barrier',W.access_ptr('rw'))
    emit('pv_sync_all',W.access_ptr('rw'))
  T.barrier_all()
 return kernel
