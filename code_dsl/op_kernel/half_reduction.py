"""Native half-width pivot reduction and explicit cross-chunk selection."""
from code_dsl.op_kernel.intrinsics import emit

def make_half_argmax_macro():
 import tilelang.language as T
 @T.macro
 def argmax(state,half_work,product,temporary,length,best,position):
  if length<=128:
   emit('hp_whole_max',half_work.access_ptr('w',offset=3728),half_work.access_ptr('r',offset=3840),length,1)
   T.set_flag('V','S',0);T.wait_flag('V','S',0)
   emit('hp_reduce_scalars',half_work.access_ptr('rw'),best.buffer.data,position.buffer.data)
  else:
   emit('hp_whole_max',half_work.access_ptr('w',offset=3728),half_work.access_ptr('r',offset=3840),128,1)
   emit('hp_whole_max',half_work.access_ptr('w',offset=3730),half_work.access_ptr('r',offset=3968),length-128,1)
   T.pipe_barrier('V')
   emit('hp_float',state.access_ptr('w',offset=product),half_work.access_ptr('r',offset=3728),8)
   T.pipe_barrier('V')
   emit('pv_even_max',state.access_ptr('w',offset=temporary),state.access_ptr('r',offset=product),2)
   T.set_flag('V','S',0);T.wait_flag('V','S',0)
   emit('pv_reduce_scalars',state.access_ptr('rw'),best.buffer.data,position.buffer.data)
   local_index=T.alloc_var('int32')
   emit('hp_index',half_work.access_ptr('r',offset=3728),position+1,local_index.buffer.data)
   position=position//2*128+local_index
 return argmax
