"""Compose actual TileLang lowering outputs without changing their numerical bodies."""
from pathlib import Path
from hashlib import sha256
import re

DECLARATION='extern "C" __global__ __aicore__ void linalg_pinv_tilelang_kernel('
BOUNDARY='\nvoid linalg_pinv_tilelang_kernel_tiling('


def recovery_device_functions(sources):
    expected={'raw','quant','decode','integer','half','norms','certify','fallback'}
    if set(sources)!=expected:
        raise ValueError('all window recovery components are required')
    raw=sources['raw']
    header=raw.split('#include "tl_templates/ascend/common.h"',1)[0]+'\n'
    for name in ['quant','decode']:
        prefix=sources[name].split('#include "tl_templates/ascend/common.h"',1)[0]
        start=prefix.rfind('namespace tl::ascend {')
        if start<0:
            raise ValueError('missing elementary recovery helpers')
        header+=prefix[start:]+'\n'
    records={}
    for name in ['raw','quant','decode','integer','half','norms','certify','fallback']:
        source=sources[name]
        if source.count(DECLARATION)!=1 or source.count(BOUNDARY)!=1:
            raise ValueError('unexpected component ABI: '+name)
        function=source[source.index(DECLARATION):source.index(BOUNDARY)]
        metadata=re.findall(r'^  KERNEL_TASK_TYPE_DEFAULT\([^\n]+\);\n',function,re.M)
        if len(metadata)!=1:
            raise ValueError('unexpected component metadata: '+name)
        symbol='PvDslRectangular' if name=='fallback' else 'Fused'+name.title()
        linkage='static __attribute__((noinline))' if name=='fallback' else 'inline'
        adapted=function.replace(DECLARATION,'__aicore__ '+linkage+' void '+symbol+'(',1).replace(metadata[0],'',1)
        before=function[function.index('{'):].replace(metadata[0],'',1)
        after=adapted[adapted.index('{'):]
        if before!=after:
            raise ValueError('numerical component body changed: '+name)
        header+='namespace tl::ascend {\n'+adapted+'\n}\n'
        records[name]={'source_sha256':sha256(source.encode()).hexdigest(),
                       'callable_body_sha256':sha256(after.encode()).hexdigest(),
                       'abi_changes':['global entry to device callback','remove launch-only metadata']}
    binding=Path(__file__).with_name('recovery_bindings.h').read_text()
    header+=binding+'\n'
    records['bindings_sha256']=sha256(binding.encode()).hexdigest()
    return header,records
