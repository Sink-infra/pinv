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
 linkage='static __attribute__((noinline))' if symbol=='BlockQuick' else 'inline'
 result=function.replace(DECLARATION,'__aicore__ '+linkage+' void '+symbol+'(',1).replace(metadata[0],'',1)
 if function[function.index('{'):].replace(metadata[0],'',1)!=result[result.index('{'):]:raise ValueError('block numerical body changed')
 return result

def append_block(root,lower,digest):
 import tilelang
 from code_dsl.op_kernel.borrowed_storage import lower_function
 from code_dsl.op_kernel.block_local import quick_kernel
 from code_dsl.op_kernel.block_gemm import block_gemm
 from code_dsl.op_kernel.block_row_gemm import row_gemm
 from code_dsl.op_kernel.block_inverse import raw_kernel
 from code_dsl.op_kernel.block_continuation import pack_kernel,decode_kernel,finish_kernel,accumulate_kernel
 from code_dsl.op_kernel.block_products import product_kernel
 from code_dsl.op_kernel.kernels import instruction_source
 from code_dsl.op_kernel.recovery_backend import composition_backend
 from code_dsl.op_kernel.backend import jit_compiler_options
 op=Path(__file__).parent;records={};parts=[]
 instructions=(op/'block_gemm_pipeline_instructions.h').read_text()+'\n'+(op/'block_instructions.h').read_text()+'\n'+(op/'block_atomic_instructions.h').read_text()+'\n'+(op/'block_chain_instructions.h').read_text()+'\n'+(op/'block_lookahead_instructions.h').read_text()
 header=instructions+'\nusing uint=unsigned int;using uchar=unsigned char;using ushort=unsigned short;\n'
 def add(name,function,**kwargs):
  source=lower(name,function,**kwargs)
  symbol={'block_quick':'BlockQuick','block_gemm':'BlockGemm','block_row_gemm':'BlockRowGemm','block_pack':'BlockPack','block_product':'BlockProduct','block_decode':'BlockDecode','block_finish':'BlockFinish','block_accumulate':'BlockAccumulate'}[name]
  body=adapt(source,symbol)
  storage=None
  if symbol in ('BlockQuick','BlockDecode','BlockPack','BlockFinish'):
   body,storage=lower_function(body,98304)
  records[name]={'source_sha256':sha256(source.encode()).hexdigest(),'callable_body_sha256':sha256(body[body.index('{'):].encode()).hexdigest()}
  if storage is not None:records[name]['storage_policy']=storage
  return source,body
 quick_source,quick=add('block_quick',quick_kernel())
 gemm_source,legacy=add('block_gemm',block_gemm(),cube=True)
 from code_dsl.op_kernel.explicit_cube import lower_pipeline
 pipeline_source,records['block_gemm_pipeline']=lower_pipeline(root,digest)
 gemm=legacy.replace('BlockGemm(','BlockGemmLegacy(',1)+'\n#define bg_legacy(d,a,b,blocks,ba,m,k,n) BlockGemmLegacy((GM_ADDR)d.GetPhyAddr(),(GM_ADDR)a.GetPhyAddr(),(GM_ADDR)b.GetPhyAddr(),blocks,(GM_ADDR)d.GetPhyAddr(),k,n,ba,m,fftsAddr)\n'+adapt(pipeline_source,'BlockGemm')
 row_source,row=add('block_row_gemm',row_gemm(),cube=True)
 header+='namespace tl::ascend {\n'+quick+'\n'+gemm+'\n'+row+'\n}\n'
 header+="""
#define br_quick_strided(d,a,b,ca,ci,so,bl,pitch) BlockQuick((GM_ADDR)(d.GetPhyAddr()+ca),(GM_ADDR)(d.GetPhyAddr()+ci),(GM_ADDR)d.GetPhyAddr(),(GM_ADDR)(d.GetPhyAddr()+so),bl,0,pitch,64,1,0.0f,0.0f,fftsAddr),tl::ascend::FrRestorePipe(&pipe)
#define br_quick(d,a,b,ca,ci,so,bl,job,ba) BlockQuick((GM_ADDR)(d.GetPhyAddr()+ca),(GM_ADDR)(d.GetPhyAddr()+ci),(GM_ADDR)d.GetPhyAddr(),(GM_ADDR)(d.GetPhyAddr()+so),bl,job,64,64,ba,0.0f,0.0f,fftsAddr),tl::ascend::FrRestorePipe(&pipe)
#define br_gemm(d,a,b,oa,ob,oc,bl,ba,m,k,n) BlockGemm((GM_ADDR)(d.GetPhyAddr()+oa),(GM_ADDR)(d.GetPhyAddr()+ob),(GM_ADDR)(d.GetPhyAddr()+oc),bl,(GM_ADDR)d.GetPhyAddr(),k,n,ba,m,fftsAddr)
#define br_row_gemm(d,a,b,oa,ob,oc,bl,ba,m,k,n) BlockRowGemm((GM_ADDR)(d.GetPhyAddr()+oa),(GM_ADDR)(d.GetPhyAddr()+ob),(GM_ADDR)(d.GetPhyAddr()+oc),bl,(GM_ADDR)d.GetPhyAddr(),k,n,ba,m,fftsAddr)
"""
 from code_dsl.op_kernel.block_atomic_components import append_atomic
 atomic_header,atomic_records=append_atomic(root,lower)
 header+=atomic_header
 records['atomic_products']=atomic_records
 from code_dsl.op_kernel.block_chain_components import append_chain
 chain_header,chain_record=append_chain(root,lower)
 header+=chain_header
 records['chain_products']=chain_record
 lookahead_header,lookahead_record=append_chain(root,lower,lookahead=True)
 header+=lookahead_header
 records['lookahead_products']=lookahead_record
 standalone=instruction_source()+'\n'+(op/'parallel_instructions.h').read_text()+'\n'+(op/'recovery_bindings.h').read_text()+'\n'+header
 from code_dsl.op_kernel.block_raw_components import lower_raw
 raw,records['block_raw']=lower_raw(root,digest,standalone)
 header+='namespace tl::ascend {\n'+raw+'\n}\n'
 header+='#define bf_block(d,a,b,bl,n,ba) BlockRaw((GM_ADDR)d.GetPhyAddr(),(GM_ADDR)a.GetPhyAddr(),(GM_ADDR)b.GetPhyAddr(),(GM_ADDR)S.GetPhyAddr(),bl,n,ba,fftsAddr)\n'
 all='(ba*n*n)'
 ptr=lambda offset:'(GM_ADDR)(d.GetPhyAddr()+'+offset+')'
 shared={'blocks':'bl','n':'n','batch':'ba','fftsAddr':'fftsAddr','shape_hint':'(GM_ADDR)d.GetPhyAddr()'}
 def callback(name,function,macro,formal,mapping,**kwargs):
  nonlocal header
  source,body=add(name,function,**kwargs)
  symbol={'block_pack':'BlockPack','block_product':'BlockProduct','block_decode':'BlockDecode','block_finish':'BlockFinish','block_accumulate':'BlockAccumulate'}[name]
  parameters=body[body.index('(')+1:body.index(')')].split(',')
  names=[re.search(r'(\w+)\s*$',part).group(1) for part in parameters]
  if not set(names)<=set(mapping):raise ValueError('block ABI changed: '+str(names))
  header+='namespace tl::ascend {\n'+body+'\n}\n'
  header+=f'#define {macro}(d,a,b,{formal}) '+symbol+'('+','.join(mapping[x] for x in names)+')'+('' if kwargs.get('cube') else ',tl::ascend::FrRestorePipe(&pipe)')+'\n'
 callback('block_pack',pack_kernel(instructions),'bf_pack','bl,n,ba,st,at,rt',dict(shared,stride='st',atol='at',rtol='rt',A_handle='(GM_ADDR)fr.input.GetPhyAddr()',Y_handle='(GM_ADDR)fr.output.GetPhyAddr()',AH_handle=ptr('0'),YH_handle=ptr('3*'+all),YN_handle=ptr('8*'+all),S_handle='(GM_ADDR)fr.status.GetPhyAddr()'))
 callback('block_product',product_kernel(),'bf_product','bl,n,ba',dict(shared,A_handle=ptr('2*'+all),Y_handle=ptr('3*'+all),P_handle=ptr('4*'+all)),cube=True)
 callback('block_decode',decode_kernel(instructions),'bf_decode','bl,n,ba',dict(shared,P_handle=ptr('4*'+all),R_handle=ptr('9*'+all)))
 callback('block_finish',finish_kernel(instructions),'bf_finish','bl,n,ba,st,at,rt',dict(shared,A_handle='(GM_ADDR)fr.input.GetPhyAddr()',Y_handle='(GM_ADDR)fr.output.GetPhyAddr()',W_handle='(GM_ADDR)d.GetPhyAddr()',S_handle='(GM_ADDR)fr.status.GetPhyAddr()',stride='st',atol='at',rtol='rt'))
 callback('block_accumulate',accumulate_kernel(instructions),'bf_accumulate','bl,n,ba',dict(shared,W_handle='(GM_ADDR)d.GetPhyAddr()'))
 header=BEGIN+header+END
 (root/'block_components.h').write_text(header)
 records['header_sha256']=sha256(header.encode()).hexdigest()
 records['instructions_sha256']=sha256(instructions.encode()).hexdigest()
 return header,records
