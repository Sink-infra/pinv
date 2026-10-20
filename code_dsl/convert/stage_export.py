"""Export an explicitly partial rollout, retaining native unported strategies.

Use actual JIT compiler output. Solver bodies are verified unchanged except
launch-only metadata. This is not the complete DSL export in export.py.
"""
import argparse,json,re,shutil
from hashlib import sha256
from pathlib import Path
from code_dsl.convert.native_source import native_source
from code_dsl.convert.export import BASELINE_COMMIT,CMAKE_FILES,DECLARATION,BOUNDARY
from code_dsl.source import dsl_digest
from tests.common.provenance import require_stage_acceptance


def digest(data):return sha256(data).hexdigest()


CANN_COMPACT_GRID='        context->SetBlockDim(std::min<uint64_t>(platform.GetCoreNumAiv(),std::max<uint64_t>(batch,std::max<uint32_t>(1,platform.GetCoreNumAiv()/4))));\n'


WORKSPACE=r'''#pragma once
#include "pinv_contract.h"
// Shape/ABI planning only. No input values or numerical decisions.
#if defined(__NPU_ARCH__)
#define PV_STAGE_LAYOUT_FN __aicore__ inline
#else
#define PV_STAGE_LAYOUT_FN inline
#endif
namespace pinv_dsl {
PV_STAGE_LAYOUT_FN uint64_t WorkFloats(uint32_t batch,uint32_t rows,uint32_t cols,uint32_t stride) {
    uint64_t rect=pinv::RectWorkspaceFloats(batch,rows,cols);
    uint64_t peer=batch*(2*uint64_t(cols)*stride+pinv::kMetadataFloats);
    uint64_t words=rect>peer?rect:peer;
    return (words+127)/128*128;
}
}
#undef PV_STAGE_LAYOUT_FN
'''

ENTRY=r'''#include "pinv_native_entry.h"
#include "pinv_dsl_workspace.h"
#include "pinv_dsl_compact.h"
#include "pinv_dsl_local.h"
#include "pinv_dsl_peer_pipeline.h"
template <typename DT_A, bool COMPACT, bool TAIL, bool WAVE, bool LOCAL, bool GROUP, bool PIPELINE, bool WINDOW, bool CUBE, bool RECT>
__global__ __aicore__ void linalg_pinv(GM_ADDR A, GM_ADDR A_pinv, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    REGISTER_TILING_DEFAULT(LinalgPinvCubeTilingData);
    GET_TILING_DATA_WITH_STRUCT(LinalgPinvTilingData, data, tiling);
    if constexpr (COMPACT || LOCAL) {
        if (COMPACT || data.cols<=64) {
            // Complete resident paths keep matrices and admission status in
            // TileLang UB buffers, including rank/truncation continuations.
            GM_ADDR work=nullptr;
            GM_ADDR status=nullptr;
            if constexpr (COMPACT) PvStageCompact(A,A_pinv,work,status,GetBlockNum(),data.m,data.n,data.batch,data.atol,data.rtol,0);
            else PvStageLocal(A,A_pinv,work,status,GetBlockNum(),data.m,data.n,data.batch,data.atol,data.rtol,0);
            return;
        }
    }
    if constexpr (GROUP && PIPELINE && CUBE && !WINDOW && !WAVE) {
        if ASCEND_IS_AIC {return;}
        GM_ADDR work=GetUserWorkspace(workspace);
        GM_ADDR status=work+pinv_dsl::WorkFloats(data.batch,data.rows,data.cols,data.stride)*sizeof(float);
        // CANN initializes the hardware sync base; 0 skips the JIT-only setter.
        PvStagePeerPipeline(A,A_pinv,work,status,GetBlockNum(),data.cols,data.batch,data.stride,0,data.atol,data.rtol,0);
        return;
    }
    PvStageNative<DT_A,COMPACT,TAIL,WAVE,LOCAL,GROUP,PIPELINE,WINDOW,CUBE,RECT>(A,A_pinv,workspace,tiling);
}
'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--module-build',type=Path,required=True);p.add_argument('--peer-build',type=Path,required=True)
    p.add_argument('--acceptance',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--framework-common',type=Path,required=True)
    p.add_argument('--continuation-linkage',choices=['inline','noinline'],default='noinline',
                   help='CANN multi-entry package ABI; numerical continuation bodies are unchanged')
    p.add_argument('--native',type=Path,help='optional pristine native source; default: immutable baseline in Git')
    args=p.parse_args();current=dsl_digest();acceptance=require_stage_acceptance(args.acceptance,dsl_sha256=current)
    with native_source(args.native) as native:
        export_stage(args,native,current,acceptance)


def export_stage(args,native,current,acceptance):
    mr=json.loads((args.module_build/'build-receipt.json').read_text());pr=json.loads((args.peer_build/'build-receipt.json').read_text())
    for receipt in [mr,pr]:
        if receipt['producer']!='TileLang JIT' or receipt['dsl_sha256']!=current:raise ValueError('JIT provenance mismatch')
    roots={'compact':(args.module_build/'compact/kernel.cpp',mr,'PvStageCompact'),
           'local':(args.module_build/'local/kernel.cpp',mr,'PvStageLocal'),
           'peer_pipeline':(args.peer_build/'kernel.cpp',pr,'PvStagePeerPipeline')}
    sources={};records={};functions={}
    for key,(path,receipt,symbol) in roots.items():
        raw=path.read_text();record=receipt['artifacts'][key+'_module']
        if digest(raw.encode())!=record['source_sha256']:raise ValueError('JIT source changed')
        if raw.count(DECLARATION)!=1 or raw.count(BOUNDARY)!=1:raise ValueError('unknown JIT ABI')
        function=raw[raw.index(DECLARATION):raw.index(BOUNDARY)]
        body=function[function.index('{'):]
        metadata=re.findall(r'^  KERNEL_TASK_TYPE_DEFAULT\([^\n]+\);\n',body,re.M)
        if len(metadata)!=1:raise ValueError('unknown task metadata')
        adapted=function.replace(DECLARATION,'__aicore__ inline void '+symbol+'(',1).replace(metadata[0],'',1)
        if adapted[adapted.index('{'):]!=body.replace(metadata[0],'',1):raise ValueError('device body changed')
        functions[key]='#pragma once\n#include "pinv_dsl_common.h"\n'+adapted
        sources[key]=raw
        records[key]={'source':'op_kernel/pinv_dsl_'+key+'.h','entry':symbol,
                      'jit_source_sha256':digest(raw.encode()),'jit_library_sha256':record['library_sha256'],
                      'callable_body_sha256':digest(adapted[adapted.index('{'):].encode()),
                      'abi_changes':['global entry to inline function','remove launch-only metadata'],
                      'runtime_arguments':'shape, batch, strides, tolerance and launch geometry'}
    # All three JIT modules import identical numerical continuation functions.
    # Share their exact definitions, along with the peer primitives, once.
    common=sources['peer_pipeline'].split('#include "tl_templates/ascend/common.h"',1)[0]
    for key in ['compact','local']:
        prefix=sources[key].split('#include "tl_templates/ascend/common.h"',1)[0]
        for name in ['PvDslAccurate','PvDslRectangular','PvDslResidentAccurate','PvDslResidentRectangular']:
            def extract(text):
                declaration=re.search(r'__aicore__ (?:inline|static __attribute__\(\(noinline\)\)) void '+name+r'\(',text)
                if declaration is None:raise ValueError('missing continuation declaration: '+name)
                start=declaration.start();a=text.index('{',start);level=0
                for i in range(a,len(text)):
                    level+=(text[i]=='{')-(text[i]=='}')
                    if level==0:return text[start:i+1]
                raise ValueError('unterminated function')
            if extract(prefix)!=extract(common):raise ValueError('continuations differ')
    # Optional declaration-only control for diagnosing SDK packaging. Both
    # variants retain exactly the numerical bodies produced by TileLang.
    if args.continuation_linkage=='inline':
        for name in ['PvDslAccurate','PvDslRectangular','PvDslResidentAccurate','PvDslResidentRectangular']:
            before='__aicore__ static __attribute__((noinline)) void '+name+'('
            if common.count(before)!=1:raise ValueError('unexpected continuation linkage')
            common=common.replace(before,'__aicore__ inline void '+name+'(',1)
    framework=args.framework_common.read_text();a=framework.index('template <typename T1, typename T2, uint32_t len>\nCATLASS_DEVICE void copy_ub_to_ub')
    b=framework.index('\ntemplate <',a+12);helper=framework[a:b]
    common='#pragma once\n'+common+'\n#define CATLASS_DEVICE __aicore__ inline\nnamespace tl::ascend {\n'+helper+'\n}\n#undef CATLASS_DEVICE\n'
    original=native.resolve();destination=args.output.resolve()
    if original==destination:raise ValueError('export cannot overwrite native sources')
    destination.mkdir(parents=True,exist_ok=True)
    for path in original.rglob('*'):
        if path.is_file():
            target=destination/path.relative_to(original);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    native=(original/'op_kernel/linalg_pinv.cpp').read_text()
    # Match the flat Tensor ABI used by the real JIT lowering. All solver
    # arithmetic is retained; resident/peer keys have no ShapeInfo consumers.
    macro='(TILING_KEY_VAR & PINV_KEY_GROUP)'
    if native.count(macro)!=1:raise ValueError('unexpected native tensor metadata guard')
    native=native.replace(macro,'(TILING_KEY_VAR & (PINV_KEY_GROUP | PINV_KEY_COMPACT | PINV_KEY_LOCAL))',1)
    native=native.replace('__global__ __aicore__ void linalg_pinv(', '__aicore__ inline void PvStageNative(',1)
    native=native.replace('    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);\n','',1)
    # Each generated strategy owns its header; shared primitives and native
    # continuations have one definition. Function bodies remain unchanged.
    entry=ENTRY.replace('#include "pinv_native_entry.h"\n','')
    files={'op_kernel/linalg_pinv.cpp':native+'\n'+entry,
           'op_kernel/pinv_dsl_workspace.h':WORKSPACE,'op_kernel/pinv_dsl_common.h':common}
    files.update({'op_kernel/pinv_dsl_'+key+'.h':text for key,text in functions.items()})
    host=(original/'op_host/linalg_pinv.cpp').read_text()
    host='#include "../op_kernel/pinv_dsl_workspace.h"\n'+host
    needle='    uint32_t RECT=0;\n'
    assert host.count(needle)==1
    host=host.replace(needle,'''    // Partial DSL rollout: numerical strategy selection remains unchanged.
    if(COMPACT || (LOCAL && t->cols<=64)) {
        if(COMPACT) {
'''+CANN_COMPACT_GRID+'''        }
        ws[0]=0;
    } else if(GROUP && PIPELINE && CUBE && !WINDOW && !WAVE) {
        uint64_t dslWords=pinv_dsl::WorkFloats(batch,t->rows,t->cols,t->stride)+batch*32;
        ws[0]=std::max<uint64_t>(ws[0],platform.GetLibApiWorkSpaceSize()+dslWords*sizeof(float));
    }
'''+needle,1)
    files['op_host/linalg_pinv.cpp']=host
    for path,text in files.items():(destination/path).write_text(text)
    for name in ['NUMERICS.md','op_kernel/pinv_native_entry.h','op_kernel/pinv_dsl_generated.h']:
        (destination/name).unlink(missing_ok=True)
    for name in CMAKE_FILES:
        if (destination/name).read_bytes()!=(original/name).read_bytes():raise ValueError('CMake changed')
    license_path=args.framework_common.resolve().parents[3]/'LICENSE'
    manifest={'schema_version':1,'scope':'user-authorized partial DSL rollout; not complete DSL acceptance',
              'baseline_commit':BASELINE_COMMIT,'dsl_sha256':current,'jit_acceptance':acceptance,
              'full_dsl_acceptance':False,'dsl_mock_cases':[1,2,4,5,6,7,8,10,11,12],
              'native_mock_cases':[3,9],'native_retained':'window, window Cube, distributed, local above 64 and rectangular strategies',
              'submission_packaging':'the sole project manifest is project_manifest.json at the export root',
              'strategies':records,'framework_copy_helper_sha256':digest(helper.encode()),
              'third_party_notices':{'TileLang':{'revision':'e909216995c2acc527bdb252039ec93a920c052c',
                  'source':'tilelang/src/tl_templates/ascend/common.h:copy_ub_to_ub',
                  'license':license_path.read_text()}},
              'cmake':'three files byte-identical to baseline','generation':'JIT numerical bodies unchanged; ABI and includes only',
              'workspace_abi':{'compact_local':'zero GM workspace; compact scalar admission and local UB admission',
                               'peer_pipeline':'CANN library prefix and user scratch for SyncAll'},
              'launch_geometry':'compact independent jobs use max(batch,AIV/4) capped by AIV; other strategies retain native host scheduling',
              'resident_layout':{'admission':{'compact':'scalar int32', 'local':'UB int32'}, 'compact_pool_float_offset':0, 'compact_gather_float_offset':1400, 'local_pool_float_offset':8},
              'flat_tensor_metadata':{'K_MAX_SHAPE_DIM':0,'scope':'JIT DSL kernels and CANN resident/peer keys','precondition':'no SetShapeInfo/GetShapeInfo consumers; explicit runtime transfer descriptors retain geometry'},
              'continuation_linkage':{'jit':'noinline','cann':args.continuation_linkage,
                  'scope':'declaration-only SDK multi-entry packaging compatibility; numerical bodies unchanged'},
              'files_sha256':{p.relative_to(destination).as_posix():digest(p.read_bytes()) for p in sorted(destination.rglob('*')) if p.is_file() and p.name!='project_manifest.json'}}
    (destination/'project_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'output':str(destination),'strategies':list(records),'dsl_sha256':current}))


if __name__=='__main__':main()
