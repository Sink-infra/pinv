"""Export the validated TileLang atomic products as a separate device header."""
from pathlib import Path
from hashlib import sha256

def split_atomic(block,receipt,build,destination,callable_source):
    products=receipt.get('block',{}).get('atomic_products')
    if not products:
        return block
    instructions=(Path(__file__).parents[1]/'op_kernel/block_atomic_instructions.h').read_text()
    if block.count(instructions)!=1:
        raise ValueError('atomic instruction prefix changed')
    block=block.replace(instructions,'',1)
    header='#pragma once\n#include "pinv_dsl_common.h"\n'+instructions+'\n'
    bindings=[]
    for name,symbol,macro in [
        ('block_inplace_panel','BlockInplacePanel','br_inplace_panel'),
        ('block_atomic_rows','BlockAtomicRows','br_atomic_rows')]:
        source=(build/'components'/name/'kernel.cpp').read_text()
        record=products[name]
        if sha256(source.encode()).hexdigest()!=record['source_sha256']:
            raise ValueError('atomic component source changed: '+name)
        function=callable_source(source,symbol,noinline=False)
        if sha256(function[function.index('{'):].encode()).hexdigest()!=record['callable_body_sha256']:
            raise ValueError('atomic component arithmetic changed: '+name)
        fragment='namespace tl::ascend {\n'+function+'\n}\n'
        if block.count(fragment)!=1:
            raise ValueError('atomic component binding changed: '+name)
        block=block.replace(fragment,'',1)
        header+=fragment
        macros=[line for line in block.splitlines() if line.startswith('#define '+macro+'(')]
        if len(macros)!=1:
            raise ValueError('atomic product ABI changed: '+name)
        bindings+=macros
        block=block.replace(macros[0]+'\n','',1)
    header+='\n'.join(bindings)+'\n'
    anchor='namespace tl::ascend {\n__aicore__ inline void BlockRaw('
    if block.count(anchor)!=1:
        raise ValueError('raw component declaration changed')
    block=block.replace(anchor,'#include "pinv_dsl_block_atomic.h"\n'+anchor,1)
    (destination/'op_kernel/pinv_dsl_block_atomic.h').write_text(header)
    return block
