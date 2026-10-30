"""Compose immutable TileLang product bodies into the block strategy."""
from hashlib import sha256
import re

def append_atomic(root,lower):
 from code_dsl.op_kernel.block_components import adapt
 from code_dsl.op_kernel.block_atomic_products import inplace_panel_kernel,atomic_rows_kernel
 header='';records={}
 for name,symbol,macro,make,left in [
  ('block_inplace_panel','BlockInplacePanel','br_inplace_panel',inplace_panel_kernel,'diag'),
  ('block_atomic_rows','BlockAtomicRows','br_atomic_rows',atomic_rows_kernel,'factors')]:
  source=lower(name,make(),cube=True)
  body=adapt(source,symbol)
  header+='namespace tl::ascend {\n'+body+'\n}\n'
  parameters=body[body.index('(')+1:body.index(')')].split(',')
  names=[re.search(r'(\w+)\s*$',p).group(1) for p in parameters]
  ptr=lambda off:'(GM_ADDR)(d.GetPhyAddr()+'+off+')'
  mapping={'D_handle':ptr('oa'),'F_handle':ptr('oa'),'P_handle':ptr('op'),'blocks':'bl','start_row':'st','shape_hint':'(GM_ADDR)d.GetPhyAddr()','batch':'ba','n':'n','fftsAddr':'fftsAddr'}
  if not set(names)<=mapping.keys():raise ValueError('atomic product ABI changed: '+str(names))
  header+='#define '+macro+'(d,a,b,oa,op,bl,ba,n,st) '+symbol+'('+','.join(mapping[n] for n in names)+')\n'
  records[name]={'source_sha256':sha256(source.encode()).hexdigest(),'callable_body_sha256':sha256(body[body.index('{'):].encode()).hexdigest(),'parameters':names}
 return header,records
