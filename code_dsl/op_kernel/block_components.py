"""Lower and compose the private runtime block strategy for the window JIT."""
from pathlib import Path
from hashlib import sha256
import json,re,shutil
from code_dsl.op_kernel.components import DECLARATION
BOUNDARY='\nvoid linalg_pinv_tilelang_kernel_tiling('
BEGIN='// BEGIN DSL block strategy\n'
END='// END DSL block strategy\n'

def adapt(source,symbol):
 function=source[source.index(DECLARATION):source.index(BOUNDARY)]
 metadata=re.findall(r'^  KERNEL_TASK_TYPE_DEFAULT\([^\n]+\);\n',function,re.M)
 if len(metadata)!=1:raise ValueError('block launch metadata changed')
 result=function.replace(DECLARATION,'__aicore__ inline void '+symbol+'(',1).replace(metadata[0],'',1)
 if function[function.index('{'):].replace(metadata[0],'',1)!=result[result.index('{'):]:raise ValueError('block numerical body changed')
 return result

def append_block(root,lower,digest):
 import tilelang
 from code_dsl.op_kernel.local import _make_quick_schedule
 from code_dsl.op_kernel.block_local import quick_kernel
 from code_dsl.op_kernel.block_gemm import block_gemm
 from code_dsl.op_kernel.block_inverse import raw_kernel
 from code_dsl.op_kernel.block_continuation import pack_kernel,decode_kernel,finish_kernel
 from code_dsl.op_kernel.block_products import product_kernel
 from code_dsl.op_kernel.kernels import instruction_source
 from code_dsl.op_kernel.recovery_backend import composition_backend
 from code_dsl.op_kernel.backend import jit_compiler_options
 op=Path(__file__).parent;records={};parts=[]
 instructions=(op/'block_instructions.h').read_text()
 header=instructions+'\nusing uint=unsigned int;using uchar=unsigned char;using ushort=unsigned short;\n'
 def add(name,function,**kwargs):
  source=lower(name,function,**kwargs)
  symbol={'block_quick':'BlockQuick','block_gemm':'BlockGemm','block_pack':'BlockPack','block_product':'BlockProduct','block_decode':'BlockDecode','block_finish':'BlockFinish'}[name]
  body=adapt(source,symbol)
  records[name]={'source_sha256':sha256(source.encode()).hexdigest(),'callable_body_sha256':sha256(body[body.index('{'):].encode()).hexdigest()}
  return source,body
 quick_source,quick=add('block_quick',quick_kernel())
 gemm_source,gemm=add('block_gemm',block_gemm(),cube=True)
 header+='namespace tl::ascend {\n'+quick+'\n'+gemm+'\n}\n'
 header+="""
#define br_quick(d,a,b,ca,ci,so,bl,job,ba) BlockQuick((GM_ADDR)(d.GetPhyAddr()+ca),(GM_ADDR)(d.GetPhyAddr()+ci),(GM_ADDR)d.GetPhyAddr(),(GM_ADDR)(d.GetPhyAddr()+so),bl,job,64,64,ba,0.0f,0.0f,fftsAddr),tl::ascend::FrRestorePipe(&pipe)
#define br_gemm(d,a,b,oa,ob,oc,bl,ba,m,k,n) BlockGemm((GM_ADDR)(d.GetPhyAddr()+oa),(GM_ADDR)(d.GetPhyAddr()+ob),(GM_ADDR)(d.GetPhyAddr()+oc),bl,(GM_ADDR)d.GetPhyAddr(),k,n,ba,m,fftsAddr)
"""
 standalone=instruction_source()+'\n'+(op/'parallel_instructions.h').read_text()+'\n'+(op/'recovery_bindings.h').read_text()+'\n'+header
 destination=root/'components/block_raw';destination.mkdir(parents=True,exist_ok=False)
 f=raw_kernel(standalone).with_attr('pinv.dsl.sha256',digest).with_attr('pinv.jit.optimization_level',3)
 (destination/'kernel.tir').write_text(f.script())
 config={'tl.disable_safe_memory_legalize':True,'tl.ascend_auto_sync':False,'tl.ascend_memory_planning':True}
 with composition_backend(),jit_compiler_options(3) as compiler:
  compiler.destination=destination
  result=tilelang.compile(f,target='ascendc',platform='A2',out_idx=[],pass_configs=config)
  if not compiler.commands:raise ValueError('fresh block lowering required')
  source=result.get_kernel_source();(destination/'kernel.cpp').write_text(source)
  shutil.copy2(result.adapter.lib._name,destination/'kernel.so')
  records['block_raw']={'source_sha256':sha256(source.encode()).hexdigest(),'tir_sha256':sha256(f.script().encode()).hexdigest(),'compiler':compiler.commands}
 raw=adapt(source,'BlockRaw')
 records['block_raw']['callable_body_sha256']=sha256(raw[raw.index('{'):].encode()).hexdigest()
 header+='namespace tl::ascend {\n'+raw+'\n}\n'
 header+='#define bf_block(d,a,b,bl,n,ba) BlockRaw((GM_ADDR)d.GetPhyAddr(),(GM_ADDR)a.GetPhyAddr(),(GM_ADDR)b.GetPhyAddr(),(GM_ADDR)S.GetPhyAddr(),bl,n,ba,fftsAddr)\n'
 all='(ba*n*n)'
 ptr=lambda offset:'(GM_ADDR)(d.GetPhyAddr()+'+offset+')'
 shared={'blocks':'bl','n':'n','batch':'ba','fftsAddr':'fftsAddr','shape_hint':'(GM_ADDR)d.GetPhyAddr()'}
 def callback(name,function,macro,formal,mapping,**kwargs):
  nonlocal header
  source,body=add(name,function,**kwargs)
  symbol={'block_pack':'BlockPack','block_product':'BlockProduct','block_decode':'BlockDecode','block_finish':'BlockFinish'}[name]
  parameters=body[body.index('(')+1:body.index(')')].split(',')
  names=[re.search(r'(\w+)\s*$',part).group(1) for part in parameters]
  if not set(names)<=set(mapping):raise ValueError('block ABI changed: '+str(names))
  header+='namespace tl::ascend {\n'+body+'\n}\n'
  header+=f'#define {macro}(d,a,b,{formal}) '+symbol+'('+','.join(mapping[x] for x in names)+')'+('' if kwargs.get('cube') else ',tl::ascend::FrRestorePipe(&pipe)')+'\n'
 callback('block_pack',pack_kernel(instructions),'bf_pack','bl,n,ba',dict(shared,A_handle='(GM_ADDR)fr.input.GetPhyAddr()',Y_handle='(GM_ADDR)fr.output.GetPhyAddr()',AH_handle=ptr('2*'+all),YH_handle=ptr('3*'+all),YN_handle=ptr('8*'+all),S_handle='(GM_ADDR)fr.status.GetPhyAddr()'))
 callback('block_product',product_kernel(),'bf_product','bl,n,ba',dict(shared,A_handle=ptr('2*'+all),Y_handle=ptr('3*'+all),P_handle=ptr('4*'+all)),cube=True)
 callback('block_decode',decode_kernel(instructions),'bf_decode','bl,n,ba',dict(shared,P_handle=ptr('4*'+all),R_handle=ptr('9*'+all)))
 callback('block_finish',finish_kernel(instructions),'bf_finish','bl,n,ba,st',dict(shared,A_handle='(GM_ADDR)fr.input.GetPhyAddr()',Y_handle='(GM_ADDR)fr.output.GetPhyAddr()',W_handle='(GM_ADDR)d.GetPhyAddr()',S_handle='(GM_ADDR)fr.status.GetPhyAddr()',stride='st'))
 header=BEGIN+header+END
 (root/'block_components.h').write_text(header)
 records['header_sha256']=sha256(header.encode()).hexdigest()
 records['instructions_sha256']=sha256(instructions.encode()).hexdigest()
 return header,records
