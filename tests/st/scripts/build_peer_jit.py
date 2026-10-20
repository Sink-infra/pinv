"""Build the peer pipeline and its numerical continuation through TileLang JIT."""
import argparse,json,shutil
from hashlib import sha256
from pathlib import Path


def main(argv=None):
    import tilelang
    tilelang.disable_cache()
    from code_dsl.op_kernel.parallel_components import rows_kernel,peer_producer_kernel,parallel_device_functions,pipeline_kernel
    from code_dsl.op_kernel.components import COMPONENTS,component_kernel,device_functions
    from code_dsl.op_kernel.backend import vector_only_backend,jit_compiler_options
    from code_dsl.source import dsl_digest

    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--diagnostic',action='store_true');args=p.parse_args(argv)
    root=args.output;root.mkdir(parents=True,exist_ok=False);digest=dsl_digest()
    settings={'tl.disable_safe_memory_legalize':True,'tl.ascend_auto_sync':False,'tl.ascend_memory_planning':True}
    components={};records={};fallback_source=None
    with jit_compiler_options(3) as compiler:
        with vector_only_backend():
            for name,make in [('rows',rows_kernel),('peer',peer_producer_kernel)]:
                print('building component',name,flush=True);destination=root/name;destination.mkdir();compiler.destination=destination
                f=make().with_attr('pinv.jit.optimization_level',3);(destination/'kernel.tir').write_text(f.script())
                result=tilelang.compile(f,target='ascendc',platform='A2',out_idx=[],pass_configs=settings)
                (destination/'kernel.cpp').write_text(result.get_kernel_source());shutil.copy2(result.adapter.lib._name,destination/'kernel.so');components[name]=result
            if not args.diagnostic:
                fallback={}
                for name in COMPONENTS:
                    print('loading DSL continuation component',name,flush=True);destination=root/'fallback'/name;destination.mkdir(parents=True);compiler.destination=destination
                    f=component_kernel(name).with_attr('pinv.jit.optimization_level',3)
                    result=tilelang.compile(f,target='ascendc',platform='A2',out_idx=[],pass_configs=settings);fallback[name]=result
                    (destination/'kernel.tir').write_text(f.script());(destination/'kernel.cpp').write_text(result.get_kernel_source());shutil.copy2(result.adapter.lib._name,destination/'kernel.so')
                fallback_source,records['continuation']=device_functions(fallback,include_instructions=False)
        header,records['pipeline']=parallel_device_functions(components);(root/'components.h').write_text(header+(fallback_source or ''))
        with vector_only_backend(paired=True):
            print('building peer strategy',flush=True);compiler.destination=root
            f=pipeline_kernel(header,paired=True,peer=True,fallback_source=fallback_source).with_attr('pinv.jit.optimization_level',3);(root/'kernel.tir').write_text(f.script())
            result=tilelang.compile(f,target='ascendc',platform='A2',out_idx=[],pass_configs=settings)
            source=result.get_kernel_source();(root/'kernel.cpp').write_text(source);shutil.copy2(result.adapter.lib._name,root/'kernel.so')
    if dsl_digest()!=digest:raise RuntimeError('DSL sources changed during JIT compilation')
    receipt={'producer':'TileLang JIT','dsl_sha256':digest,'strategy':'peer_pipeline','complete_continuation':not args.diagnostic,
             'fast_admission_status_index':31,
             'scope':'peer pipeline with DSL Jacobi continuation' if not args.diagnostic else 'communication diagnostic only',
             'optimization_level':3,'jit_compiler_commands':compiler.commands,'components':records,
             'artifacts':{'peer_pipeline_module':{'source_sha256':sha256(source.encode()).hexdigest(),'library_sha256':sha256((root/'kernel.so').read_bytes()).hexdigest(),'tir_sha256':sha256(f.script().encode()).hexdigest()}}}
    (root/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print('peer JIT strategy built',flush=True)


if __name__=="__main__":
    main()
