"""Lower both block schedules with a caller-owned Vector allocator."""
from hashlib import sha256
import shutil

def lower_raw(root,digest,standalone):
 import tilelang
 from code_dsl.op_kernel.block_inverse import raw_kernel
 from code_dsl.op_kernel.block_components import adapt
 from code_dsl.op_kernel.recovery_backend import composition_backend
 from code_dsl.op_kernel.backend import jit_compiler_options
 from code_dsl.op_kernel.borrowed_storage import lower_function
 name='block_raw'
 symbol='BlockRaw'
 destination=root/'components'/name
 destination.mkdir(parents=True,exist_ok=False)
 f=raw_kernel(standalone).with_attr('pinv.dsl.sha256',digest).with_attr('pinv.jit.optimization_level',3)
 (destination/'kernel.tir').write_text(f.script())
 config={'tl.disable_safe_memory_legalize':True,'tl.ascend_auto_sync':False,'tl.ascend_memory_planning':True}
 with composition_backend(),jit_compiler_options(3) as compiler:
  compiler.destination=destination
  result=tilelang.compile(f,target='ascendc',platform='A2',out_idx=[],pass_configs=config)
  if not compiler.commands:raise ValueError('fresh block lowering required')
  source=result.get_kernel_source()
  (destination/'kernel.cpp').write_text(source)
  shutil.copy2(result.adapter.lib._name,destination/'kernel.so')
 body,storage=lower_function(adapt(source,symbol),98304)
 record={'source_sha256':sha256(source.encode()).hexdigest(),'tir_sha256':sha256(f.script().encode()).hexdigest(),'compiler':compiler.commands,'callable_body_sha256':sha256(body[body.index('{'):].encode()).hexdigest(),'storage_policy':storage}
 return body,record
