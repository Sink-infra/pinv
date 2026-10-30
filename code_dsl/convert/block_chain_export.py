"""Export validated per-row Cube products as a separate device header."""
from pathlib import Path
from hashlib import sha256

def split_chain(block,receipt,build,destination,callable_source):
    record=receipt.get('block',{}).get('chain_products')
    if not record:
        return block
    instructions=(Path(__file__).parents[1]/'op_kernel/block_chain_instructions.h').read_text()
    if block.count(instructions)!=1:
        raise ValueError('chain instruction prefix changed')
    block=block.replace(instructions,'',1)
    source=(build/'components/block_chain_rows/kernel.cpp').read_text()
    if sha256(source.encode()).hexdigest()!=record['source_sha256']:
        raise ValueError('chain component source changed')
    function=callable_source(source,'BlockChainRows',noinline=False)
    if sha256(function[function.index('{'):].encode()).hexdigest()!=record['callable_body_sha256']:
        raise ValueError('chain numerical body changed')
    fragment='namespace tl::ascend {\n'+function+'\n}\n'
    if block.count(fragment)!=1:
        raise ValueError('chain component binding changed')
    block=block.replace(fragment,'',1)
    macros=[line for line in block.splitlines() if line.startswith('#define br_chain(')]
    if len(macros)!=1:
        raise ValueError('chain product ABI changed')
    block=block.replace(macros[0]+'\n','',1)
    anchor='namespace tl::ascend {\n__aicore__ inline void BlockRaw('
    if block.count(anchor)!=1:
        raise ValueError('raw component declaration changed')
    block=block.replace(anchor,'#include "pinv_dsl_block_chain.h"\n'+anchor,1)
    header='#pragma once\n#include "pinv_dsl_block_atomic.h"\n'+instructions+'\n'+fragment+macros[0]+'\n'
    (destination/'op_kernel/pinv_dsl_block_chain.h').write_text(header)
    return block
