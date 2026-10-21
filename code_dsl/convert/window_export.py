"""Export all validated Mock DSL paths, retaining other native strategies."""
import argparse,json,re
from pathlib import Path
from hashlib import sha256

from code_dsl.source import dsl_digest
from code_dsl.convert.native_source import native_source
from code_dsl.convert.stage_export import export_stage
from code_dsl.convert.export import CMAKE_FILES
from code_dsl.convert.window_layout import CPP
from tests.common.provenance import require_mock_path_acceptance

DECLARATION='extern "C" __global__ __aicore__ void linalg_pinv_tilelang_kernel('
BOUNDARY='\nvoid linalg_pinv_tilelang_kernel_tiling('


def digest(text):
    return sha256(text.encode()).hexdigest()


def definition(text,name):
    pattern=r'__aicore__ (?:inline|static __attribute__\(\(noinline\)\)) void '+name+r'\('
    match=re.search(pattern,text)
    if match is None:
        raise ValueError('missing device definition: '+name)
    start=match.start();opening=text.index('{',start);depth=1;end=opening+1
    while depth:
        depth+=(text[end]=='{')-(text[end]=='}')
        end+=1
    return text[start:end]


def callable_source(source,symbol,*,noinline=False):
    if source.count(DECLARATION)!=1 or source.count(BOUNDARY)!=1:
        raise ValueError('unexpected lowered device ABI')
    function=source[source.index(DECLARATION):source.index(BOUNDARY)]
    metadata=re.findall(r'^  KERNEL_TASK_TYPE_DEFAULT\([^\n]+\);\n',function,re.M)
    if len(metadata)!=1:
        raise ValueError('unexpected launch metadata')
    linkage='static __attribute__((noinline))' if noinline else 'inline'
    adapted=function.replace(DECLARATION,'__aicore__ '+linkage+' void '+symbol+'(',1).replace(metadata[0],'',1)
    if function[function.index('{'):].replace(metadata[0],'',1)!=adapted[adapted.index('{'):]:
        raise ValueError('numerical device body changed')
    return adapted


def export_window(args,native,current,acceptance):
    export_stage(args,native,current,acceptance)
    destination=args.output
    receipt=json.loads((args.window_build/'build-receipt.json').read_text())
    if receipt.get('producer')!='TileLang JIT' or receipt.get('dsl_sha256')!=current or not receipt.get('complete_continuation'):
        raise ValueError('complete current window JIT provenance is required')
    source=(args.window_build/'kernel.cpp').read_text()
    artifact=receipt['artifacts']['window_cube_module']
    if digest(source)!=artifact['source_sha256'] or sha256((args.window_build/'kernel.so').read_bytes()).hexdigest()!=artifact['library_sha256']:
        raise ValueError('window JIT artifact changed')
    common=(destination/'op_kernel/pinv_dsl_common.h').read_text()
    sources={}
    for name in ['rows','producer','prepared_rows','prepared_producer','transformer','raw','prepared_raw','quant','decode','integer','half','norms','certify','fallback']:
        path=args.window_build/'components'/name/'kernel.cpp'
        text=path.read_text()
        if digest(text)!=receipt['lowering'][name]['source_sha256']:
            raise ValueError('component source changed: '+name)
        sources[name]=text
    shared_fallback=definition(common,'PvDslRectangular')
    lowered_fallback=definition(callable_source(sources['fallback'],'PvDslRectangular',noinline=True),'PvDslRectangular')
    if shared_fallback!=lowered_fallback:
        raise ValueError('shared precise continuation differs')
    runtime=Path(__file__).parents[1]/'op_kernel/recovery_instructions.h'
    if sha256(runtime.read_bytes()).hexdigest()!=receipt['runtime_header_sha256']:
        raise ValueError('recovery runtime changed')
    header='#pragma once\n#include "pinv_dsl_common.h"\n'+runtime.read_text()+'\nusing namespace Catlass;\n'
    for name in ['quant','decode']:
        prefix=sources[name].split('#include "tl_templates/ascend/common.h"',1)[0]
        start=prefix.rfind('namespace tl::ascend {')
        if start<0:raise ValueError('missing elementary helper prefix')
        header+='\n'.join(line.rstrip() for line in prefix[start:].splitlines())+'\n'
    for name in ['rows','producer','prepared_rows','prepared_producer','transformer']:
        symbol='PvDslWindowRows' if name=='rows' else 'PvDslPipeline'+''.join(part.title() for part in name.split('_'))
        function=callable_source(sources[name],symbol)
        body=function[function.index('{'):]
        if digest(body)!=receipt['pipeline'][name]['callable_body_sha256']:
            raise ValueError('pipeline component body changed: '+name)
        header+='namespace tl::ascend {\n'+function+'\n}\n'
    raw_prefix=sources['raw'].split('#include "tl_templates/ascend/common.h"',1)[0]
    for name in ['rows','producer','prepared_rows','prepared_producer','transformer']:
        bindings=[line for line in raw_prefix.splitlines() if line.startswith('#define pv_pipeline_'+name+'(')]
        if len(bindings)!=1:raise ValueError('missing pipeline ABI binding: '+name)
        binding=bindings[0]
        if name=='rows':
            if binding.count('PvDslPipelineRows(')!=1:raise ValueError('unexpected worker callback binding')
            binding=binding.replace('PvDslPipelineRows(','PvDslWindowRows(',1)
            header+='#undef pv_pipeline_rows\n'
        header+=binding+'\n'
    records={}
    for name in ['raw','prepared_raw','quant','decode','integer','half','norms','certify']:
        symbol='FusedPreparedRaw' if name=='prepared_raw' else 'Fused'+name.title()
        function=callable_source(sources[name],symbol,noinline=False)
        body=function[function.index('{'):]
        if digest(body)!=receipt['components'][name]['callable_body_sha256']:
            raise ValueError('component body changed: '+name)
        header+='namespace tl::ascend {\n'+function+'\n}\n'
        records[name]={'callable_body_sha256':digest(body)}
    # This prefix is emitted by the actual final JIT source. It contains only
    # callback declarations, argument bindings and allocator-context restoration.
    bindings_path=Path(__file__).parents[1]/'op_kernel/recovery_bindings.h'
    callback_bindings=bindings_path.read_text()
    if digest(callback_bindings)!=receipt['components']['bindings_sha256']:
        raise ValueError('callback ABI bindings changed')
    if source.count(callback_bindings)!=1:
        raise ValueError('callback bindings are not from the actual JIT source')
    header+=callback_bindings+'\n'
    if 'block' in receipt:
        block=(args.window_build/'block_components.h').read_text()
        if digest(block)!=receipt['block']['header_sha256'] or source.count(block)!=1:
            raise ValueError('block JIT header changed')
        for name,row in receipt['block'].items():
            if not isinstance(row,dict):continue
            component=(args.window_build/'components'/name/'kernel.cpp').read_text()
            if digest(component)!=row['source_sha256']:raise ValueError('block component changed: '+name)
        (destination/'op_kernel/pinv_dsl_block.h').write_text('#pragma once\n'+block)
        header+='#include \"pinv_dsl_block.h\"\n'
    root=callable_source(source,'PvStageWindow')
    header+=root+'\n'
    (destination/'op_kernel/pinv_dsl_window.h').write_text(header)
    entry=destination/'op_kernel/linalg_pinv.cpp'
    text=entry.read_text()
    include='#include "pinv_dsl_peer_pipeline.h"'
    if text.count(include)!=1:raise ValueError('unexpected stage entry')
    text=text.replace(include,include+'\n#include "pinv_dsl_window.h"',1)
    needle='    if constexpr (GROUP && PIPELINE && CUBE && !WINDOW && !WAVE) {'
    addition="""    if constexpr (GROUP && PIPELINE && WINDOW && CUBE && !WAVE) {
        if(pinv_dsl::WindowSupported(data.m,data.n,data.batch,data.stride,data.local,GetBlockNum())) {
            GM_ADDR work=GetUserWorkspace(workspace);
            GM_ADDR status=work+pinv_dsl::WindowWorkFloats(data.batch,data.cols,data.stride,GetBlockNum())*sizeof(float);
            PvStageWindow(A,A_pinv,work,status,GetBlockNum(),data.cols,data.batch,data.stride,data.local,data.atol,data.rtol,0,0);
            AscendC::GlobalTensor<float> flags;
            flags.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(status));
            bool admitted=true;
            for(uint32_t b=0;b<data.batch;++b) {
                AscendC::DataCacheCleanAndInvalid<float,AscendC::CacheLine::SINGLE_CACHE_LINE,AscendC::DcciDst::CACHELINE_OUT>(flags[b*512+16]);
                if(!(flags.GetValue(b*512+16)>0.5f))admitted=false;
            }
            if(admitted)return;
            {
                AscendC::TPipe continuation_barrier;
                AscendC::SyncAll<false>();
                continuation_barrier.Destroy();
            }
        }
    }
"""
    if text.count(needle)!=1:raise ValueError('unexpected dispatch')
    text=text.replace(needle,addition+needle,1)
    old='    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);'
    if text.count(old)!=1:raise ValueError('unexpected task metadata')
    text=text.replace(old,"""#if defined(TILING_KEY_VAR) && (TILING_KEY_VAR & PINV_KEY_WINDOW) && (TILING_KEY_VAR & PINV_KEY_CUBE)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#else
"""+old+'\n#endif',1)
    text=text.replace('#include "pinv_dsl_workspace.h"\n','')
    entry.write_text(text)
    host=destination/'op_host/linalg_pinv.cpp';text=host.read_text()
    needle='    uint32_t RECT=0;\n'
    addition="""    if(GROUP && PIPELINE && WINDOW && CUBE && !WAVE &&
       pinv_dsl::WindowSupported(t->m,t->n,t->batch,t->stride,t->local,context->GetBlockDim())) {
        uint64_t words=pinv_dsl::WindowWorkFloats(t->batch,t->cols,t->stride,context->GetBlockDim())
                       +pinv_dsl::WindowStatusFloats(t->batch);
        ws[0]=std::max<uint64_t>(ws[0],platform.GetLibApiWorkSpaceSize()+words*sizeof(float));
    }
"""
    if text.count(needle)!=1:raise ValueError('unexpected host planning')
    text=text.replace(needle,addition+needle,1)
    text=text.replace('#include "../op_kernel/pinv_dsl_workspace.h"\n','')
    host.write_text(text)
    # Place the shared workspace ABI beside the existing shape contract.
    workspace=destination/'op_kernel/pinv_dsl_workspace.h'
    fragment=workspace.read_text().replace('#pragma once\n','').replace('#include "pinv_contract.h"\n','')
    contract=destination/'op_kernel/pinv_contract.h'
    contract.write_text(contract.read_text()+'\n'+fragment+'\n'+CPP)
    workspace.unlink()
    for name in CMAKE_FILES:
        if (destination/name).read_bytes()!=(native/name).read_bytes():
            raise ValueError('CMake changed')
    manifest=destination/'project_manifest.json';document=json.loads(manifest.read_text())
    document.update(scope='all Mock paths expressed in TileLang; remaining non-Mock strategies retain native code',
                    jit_acceptance=acceptance,dsl_mock_cases=list(range(1,13)),native_mock_cases=[],
                    native_retained='unported non-Mock strategies and rejected window trials; the standalone JIT keeps its DSL Jacobi continuation',
                    window_components=records,window_runtime_sha256=sha256(runtime.read_bytes()).hexdigest())
    document['strategies']['window_cube']={'source':'op_kernel/pinv_dsl_window.h','entry':'PvStageWindow',
                'jit_source_sha256':artifact['source_sha256'],'jit_library_sha256':artifact['library_sha256'],
                'callable_body_sha256':digest(root[root.index('{'):]),
                'shared':'identical Jacobi continuation; window worker retains its own generated epilogue',
                'abi_changes':['device callback declaration','remove launch metadata','relative runtime include',
                               'window worker declaration and callback binding use a private symbol']}
    document['files_sha256']={p.relative_to(destination).as_posix():sha256(p.read_bytes()).hexdigest()
                             for p in sorted(destination.rglob('*')) if p.is_file() and p!=manifest}
    manifest.write_text(json.dumps(document,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['module-build','peer-build','window-build','acceptance','output','framework-common']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--native',type=Path)
    parser.add_argument('--continuation-linkage',choices=['inline','noinline'],default='noinline')
    args=parser.parse_args();current=dsl_digest()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("export requires a fresh output directory")
    acceptance=require_mock_path_acceptance(args.acceptance,dsl_sha256=current)
    with native_source(args.native) as native:
        export_window(args,native,current,acceptance)


if __name__=='__main__':
    main()
