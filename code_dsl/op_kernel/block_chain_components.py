"""Compose unchanged TileLang chained product bodies."""
from pathlib import Path
from hashlib import sha256
import re

def append_chain(root,lower):
 from code_dsl.op_kernel.block_components import adapt
 from code_dsl.op_kernel.block_chain_products import chain_kernel
 source=lower('block_chain_rows',chain_kernel(),cube=True);body=adapt(source,'BlockChainRows')
 names=[re.search(r'(\w+)\s*$',p).group(1) for p in body[body.index('(')+1:body.index(')')].split(',')]
 ptr=lambda x:'(GM_ADDR)(d.GetPhyAddr()+'+x+')'
 mapping={'F_handle':ptr('fo'),'D_handle':ptr('di'),'P_handle':ptr('pi'),'Tmp_handle':ptr('tm'),'Y_handle':ptr('sw'),'blocks':'bl','batch':'ba','n':'n','start_row':'st','shape_hint':'(GM_ADDR)d.GetPhyAddr()','fftsAddr':'fftsAddr'}
 assert set(names)<=set(mapping),names
 header='namespace tl::ascend {\n'+body+'\n}\n#define br_chain(d,a,b,fo,di,pi,tm,sw,bl,ba,n,st) BlockChainRows('+','.join(mapping[x] for x in names)+')\n'
 return header,{'source_sha256':sha256(source.encode()).hexdigest(),'callable_body_sha256':sha256(body[body.index('{'):].encode()).hexdigest(),'parameters':names}
