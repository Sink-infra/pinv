"""Build the currently implemented shared strategies through TileLang JIT.

This command performs no NPU launches and no independent AscendC build.
The resulting receipt binds later JIT sampling to its source and binary hashes.
Large-matrix continuation/refinement is still under development.
"""
from pathlib import Path
import argparse,json,shutil
from hashlib import sha256


def main(argv=None):
    import tilelang
    from code_dsl.op_kernel.components import COMPONENTS,component_kernel,device_functions,composed_kernel
    from code_dsl.op_kernel.backend import vector_only_backend,jit_compiler_options
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path);parser.add_argument('--optimization-level',type=int,choices=[2,3],default=3);parser.add_argument('--unroll',type=int,choices=[1,2,4,8],default=1);parser.add_argument('--factor-pitch',type=int,choices=[8,16],default=8);args=parser.parse_args(argv)
    root=args.output;root.mkdir(parents=True,exist_ok=False)
    from code_dsl.source import dsl_digest
    source_digest=dsl_digest()
    settings={'tl.disable_safe_memory_legalize':True,'tl.ascend_auto_sync':False,'tl.ascend_memory_planning':True}
    compiled={};records={}
    with vector_only_backend(),jit_compiler_options(args.optimization_level) as compiler:
        for name in COMPONENTS:
            print('building component',name,flush=True)
            destination=root/'components'/name;destination.mkdir(parents=True)
            compiler.destination=destination
            function=component_kernel(name,unroll=args.unroll,factor_pitch=args.factor_pitch).with_attr('pinv.jit.optimization_level',args.optimization_level);(destination/'kernel.tir').write_text(function.script())
            result=tilelang.compile(function,target='ascendc',platform='A2',out_idx=[],pass_configs=settings)
            source=result.get_kernel_source();(destination/'kernel.cpp').write_text(source)
            shutil.copy2(result.adapter.lib._name,destination/'kernel.so')
            compiled[name]=result
            records[name]={'tir_sha256':sha256(function.script().encode()).hexdigest(),
                           'source_sha256':sha256(source.encode()).hexdigest(),
                           'library_sha256':sha256((destination/'kernel.so').read_bytes()).hexdigest()}
            (root/'build-progress.json').write_text(json.dumps(records,indent=2)+'\n')
            print('built component',name,flush=True)
        source,component_records=device_functions(compiled)
        (root/'components.h').write_text(source)
        for name in ['compact','local','rectangular']:
            print('building module',name,flush=True)
            destination=root/name;destination.mkdir()
            compiler.destination=destination
            function=composed_kernel(name,source,unroll=args.unroll,factor_pitch=args.factor_pitch).with_attr('pinv.jit.optimization_level',args.optimization_level);(destination/'kernel.tir').write_text(function.script())
            result=tilelang.compile(function,target='ascendc',platform='A2',out_idx=[],pass_configs=settings)
            raw=result.get_kernel_source();(destination/'kernel.cpp').write_text(raw)
            shutil.copy2(result.adapter.lib._name,destination/'kernel.so')
            records[name+'_module']={'tir_sha256':sha256(function.script().encode()).hexdigest(),
                           'source_sha256':sha256(raw.encode()).hexdigest(),
                           'library_sha256':sha256((destination/'kernel.so').read_bytes()).hexdigest()}
            print('built module',name,flush=True)
    if dsl_digest()!=source_digest:raise RuntimeError('DSL sources changed during compilation; mixed artifacts cannot be accepted')
    (root/'build-receipt.json').write_text(json.dumps({'producer':'TileLang JIT','tilelang_version':tilelang.__version__,'optimization_level':args.optimization_level,'dsl_schedule':{'unroll':args.unroll,'factor_pitch':args.factor_pitch},'jit_compiler_commands':compiler.commands,'dsl_sha256':source_digest,'scope':'implemented compact/local/rectangular strategies; complete pinv acceptance pending','components':component_records,'artifacts':records},indent=2)+'\n')


if __name__=="__main__":
    main()
