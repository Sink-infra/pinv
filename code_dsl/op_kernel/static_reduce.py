"""Vector-only multistage norm reductions; event ownership remains explicit."""
from code_dsl.op_kernel.intrinsics import emit

def make_reduction(kind,wide=False):
 import tilelang.language as T
 instruction='sr_sum' if kind=='sum' else 'sr_max'
 @T.macro
 def reduction(pool,destination,source,scratch,count):
  full=count//64
  tail=count%64
  groups=T.ceildiv(count,64)
  if full>0:
   emit(instruction,pool.access_ptr('w',offset=scratch),pool.access_ptr('r',offset=source),64,full)
  if tail>0:
   emit(instruction,pool.access_ptr('w',offset=scratch+full),pool.access_ptr('r',offset=source+full*64),tail,1)
  T.pipe_barrier('V')
  if wide:
   if groups>64:
    emit(instruction,pool.access_ptr('w',offset=scratch+128),pool.access_ptr('r',offset=scratch),64,1)
    emit(instruction,pool.access_ptr('w',offset=scratch+129),pool.access_ptr('r',offset=scratch+64),groups-64,1)
    T.pipe_barrier('V')
    emit(instruction,pool.access_ptr('w',offset=destination),pool.access_ptr('r',offset=scratch+128),2,1)
   else:
    emit(instruction,pool.access_ptr('w',offset=destination),pool.access_ptr('r',offset=scratch),groups,1)
  else:
   emit(instruction,pool.access_ptr('w',offset=destination),pool.access_ptr('r',offset=scratch),groups,1)
 return reduction
