"""Uniform export of complete runtime-parameter strategies, with one manifest.

Numerical device bodies are copied verbatim from TileLang's actual output.
A partial diagnostic must never be packaged as a complete pseudoinverse.
"""
from hashlib import sha256
import json
from pathlib import Path
from code_dsl.convert.native_source import BASELINE_COMMIT, native_source
from code_dsl.source import dsl_digest

STRATEGIES=('compact','local','distributed','pipeline','peer_pipeline','window','window_cube','rectangular')
CMAKE_FILES=('CMakeLists.txt','op_host/CMakeLists.txt','op_kernel/CMakeLists.txt')
DECLARATION='extern "C" __global__ __aicore__ void linalg_pinv_tilelang_kernel('
BOUNDARY='\nvoid linalg_pinv_tilelang_kernel_tiling()'


def split_source(raw):
    if raw.count(DECLARATION)!=1 or raw.count(BOUNDARY)!=1:
        raise ValueError('unexpected TileLang-Ascend export layout')
    start,end=raw.index(DECLARATION),raw.index(BOUNDARY)
    f=raw[start:end];return raw[:start],f,f[f.index('{'):]


def write_project(destination,compiled,*,jit_acceptance):
    from tests.common.provenance import require_jit_acceptance
    acceptance=require_jit_acceptance(jit_acceptance,dsl_sha256=dsl_digest())
    if set(compiled)!=set(STRATEGIES):
        raise ValueError('all complete strategies are required; probes cannot be exported')
    root=Path(__file__).resolve().parents[2];destination=Path(destination)
    parts={key:split_source(value.get_kernel_source()) for key,value in compiled.items()}
    prelude=max((p[0] for p in parts.values()),key=len)
    for prefix,_,_ in parts.values():
        lines=iter(prelude.splitlines())
        if not all(any(line==expected for line in lines) for expected in prefix.splitlines()):
            raise ValueError('incompatible imported hardware instruction sets')
    instructions=prelude.split('#include "tl_templates/ascend/common.h"',1)[0]
    instructions+='using uint=unsigned int; using uchar=unsigned char; using ushort=unsigned short;\n'
    generated={'op_kernel/pinv_dsl_instructions.h':'#pragma once\n'+instructions}
    records={}
    for key,(_,function,body) in parts.items():
        entry='PvDsl'+''.join(word.title() for word in key.split('_'))
        adapted=function.replace(DECLARATION,'__aicore__ inline void '+entry+'(',1)
        assert adapted[adapted.index('{'):]==body
        path='op_kernel/pinv_dsl_'+key+'.h'
        text='#pragma once\n#include "pinv_dsl_instructions.h"\n'+adapted
        generated[path]=text
        records[key]={'source':path,'entry':entry,'source_sha256':sha256(text.encode()).hexdigest(),
                      'emitted_source_sha256':sha256(compiled[key].get_kernel_source().encode()).hexdigest(),
                      'device_body_sha256':sha256(body.encode()).hexdigest()}
    generated['op_kernel/pinv_dsl_device.h']='#pragma once\n'+''.join(
        '#include "pinv_dsl_'+key+'.h"\n' for key in STRATEGIES)
    # The device entry is an ABI-only adapter owned by the experiment.
    adapter=root/'code_dsl/op_host/device_entry.cpp'
    if not adapter.is_file():raise ValueError('complete CANN ABI adapter is required')
    generated['op_kernel/linalg_pinv.cpp']=adapter.read_text()
    copied=(*CMAKE_FILES,'NUMERICS.md','op_host/linalg_pinv.cpp','op_kernel/pinv_contract.h',
            'op_kernel/linalg_pinv_tiling.h','op_kernel/tiling_key_linalg_pinv.h')
    with native_source() as native:
        files={name:(native/name).read_bytes() for name in copied}
    files.update({name:text.encode() for name,text in generated.items()})
    for name,data in files.items():
        p=destination/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    manifest={'schema_version':1,'baseline_commit':BASELINE_COMMIT,'dsl_sha256':dsl_digest(),'jit_acceptance':acceptance,
              'backend':'tilelang-ascend / AscendC','scope':'complete pinv; all strategies required',
              'specialization':'calculation strategy only; no input-specific variants',
              'runtime_arguments':['A','A_pinv','workspace','tiling','blocks','m','n','batch','atol','rtol','stream'],
              'kernels':records,'source_sha256':{k:sha256(v).hexdigest() for k,v in files.items()},
              'cmake':'three files byte-identical to code/','generation':'verbatim TileLang device bodies; ABI/includes only'}
    (destination/'project_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest
