"""Bounded physical views for an explicitly scheduled Cube component.

The pinned planner cannot infer L1 roles from custom copy instructions.
Only the two buffers declared by T.alloc_L1 are remapped; arithmetic,
transfers, synchronization and L0 slots remain generated TileLang code.
"""
from pathlib import Path
from hashlib import sha256
import json

def lower_pipeline(root,digest):
 import tilelang,tvm
 from code_dsl.op_kernel.block_gemm_pipeline import block_gemm
 from code_dsl.op_kernel.backend import vector_only_backend
 from code_dsl.op_kernel.recovery_backend import cube_component_source
 destination=Path(root)/'components/block_gemm_pipeline'
 destination.mkdir(parents=True,exist_ok=False)
 function=block_gemm().with_attr('pinv.dsl.sha256',digest).with_attr('pinv.jit.optimization_level',3)
 tir=function.script();(destination/'kernel.tir').write_text(tir)
 if tir.count('scope="shared.dyn"')!=2:raise ValueError('explicit L1 allocation set changed')
 config={'tl.disable_safe_memory_legalize':True,'tl.ascend_auto_sync':False,'tl.ascend_memory_planning':False}
 with vector_only_backend(),tvm.transform.PassContext(opt_level=3,config=config):
  artifact=tilelang.lower(function,target='ascendc',platform='A2')
 raw=artifact.kernel_source;(destination/'lowered.cpp').write_text(raw)
 views=['  auto a = ascend_ub.GetWithOffset<float>(32768, 131072);','  auto b = ascend_ub.GetWithOffset<float>(32768, 0);']
 if 'ascend_l1, 524032' not in raw:raise ValueError('unexpected L1 capacity')
 for view in views:
  if raw.count(view)!=1:raise ValueError('explicit L1 view changed')
  raw=raw.replace(view,view.replace('ascend_ub','ascend_l1'),1)
 source=cube_component_source(raw);(destination/'kernel.cpp').write_text(source)
 record={'tir_sha256':sha256(tir.encode()).hexdigest(),'source_sha256':sha256(source.encode()).hexdigest(),'physical_l1_views':[(0,131072),(131072,131072)],'l0a_bytes':65536,'l0b_bytes':65536,'pipeline_slots':2,'k_tile':64,'arithmetic_and_events_unchanged':True}
 (destination/'storage-receipt.json').write_text(json.dumps(record,indent=2))
 return source,record
