"""Build the complete window inverse and precision recovery through TileLang."""
import argparse,json,shutil
from pathlib import Path
from hashlib import sha256


def main(argv=None):
    import tilelang,tvm
    from code_dsl.source import dsl_digest
    from code_dsl.op_kernel.backend import vector_only_backend,vector_only_source,jit_compiler_options
    from code_dsl.op_kernel.parallel_components import rows_kernel,producer_kernel,parallel_device_functions,pipeline_kernel,prepared_rows_kernel,prepared_producer_kernel,transformer_kernel
    from code_dsl.op_kernel.components import component_kernel
    from code_dsl.op_kernel.recovery_quant import quant_kernel
    from code_dsl.op_kernel.recovery_decode import decode_kernel
    from code_dsl.op_kernel.recovery_gemm import integer_gemm_kernel,half_gemm_kernel
    from code_dsl.op_kernel.recovery_certificate import certificate_kernels
    from code_dsl.op_kernel.recovery_components import recovery_device_functions
    from code_dsl.op_kernel.recovery_backend import composition_backend,cube_component_source
    from code_dsl.op_kernel.recovery import window_recovery_kernel

    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    root=args.output;root.mkdir(parents=True,exist_ok=False)
    digest=dsl_digest()
    # Compiler ABI adapters are not part of the upstream cache key.
    tilelang.disable_cache()
    config={'tl.disable_safe_memory_legalize':True,'tl.ascend_auto_sync':False,
            'tl.ascend_memory_planning':True}
    class Source:
        def __init__(self,source):self.source=source
        def get_kernel_source(self):return self.source
    sources={};component_records={}
    def lower(name,function,*,cube=False,paired=False):
        destination=root/'components'/name;destination.mkdir(parents=True)
        function=function.with_attr('pinv.dsl.sha256',digest).with_attr('pinv.jit.optimization_level',3)
        (destination/'kernel.tir').write_text(function.script())
        with vector_only_backend(),tvm.transform.PassContext(opt_level=3,config=config):
            artifact=tilelang.lower(function,target='ascendc',platform='A2')
        raw=artifact.kernel_source
        source=cube_component_source(raw) if cube else vector_only_source(raw,paired=paired)
        (destination/'lowered.cpp').write_text(raw)
        (destination/'kernel.cpp').write_text(source)
        component_records[name]={'tir_sha256':sha256(function.script().encode()).hexdigest(),
                                 'lowered_source_sha256':sha256(raw.encode()).hexdigest(),
                                 'source_sha256':sha256(source.encode()).hexdigest(),
                                 'scope':'source lowering; compiled as part of the final JIT module'}
        return source
    from code_dsl.op_kernel.pipeline_rows import make_pipeline_rows_macro
    from code_dsl.op_kernel.producer import make_producer_macro
    from code_dsl.op_kernel.prepared_rows import make_prepared_rows_macro
    from code_dsl.op_kernel.prepared_producer import make_prepared_producer_macro
    pipeline={}
    for name,factory in [('rows',lambda: rows_kernel(make_pipeline_rows_macro(cache_peak=True))),('producer',lambda: producer_kernel(make_producer_macro(cache_peak=True))),('prepared_rows',lambda: rows_kernel(make_prepared_rows_macro(cache_peak=True))),('prepared_producer',lambda: producer_kernel(make_prepared_producer_macro(cache_peak=True))),('transformer',lambda: transformer_kernel(cache_peak=True))]:
        print('lowering',name,flush=True)
        pipeline[name]=Source(lower(name,factory()))
    header,protocol=parallel_device_functions(pipeline)
    sources['raw']=lower('raw',pipeline_kernel(header,paired=True,peer=False),paired=True)
    sources['prepared_raw']=lower('prepared_raw',pipeline_kernel(header,paired=True,peer=False,prepared=True,prepared_checked=True),paired=True)
    factories={'quant':quant_kernel,'decode':decode_kernel,
               'integer':integer_gemm_kernel,'half':half_gemm_kernel,
               'fallback':lambda:component_kernel('rectangular')}
    functions={name:factory() for name,factory in factories.items()}
    functions.update(certificate_kernels())
    for name,function in functions.items():
        print('lowering',name,flush=True)
        sources[name]=lower(name,function,cube=name in {'integer','half'})
    source,records=recovery_device_functions(sources)
    from code_dsl.op_kernel.block_components import append_block
    block_source,block_records=append_block(root,lower,digest)
    source+='\n'+block_source
    (root/'components.h').write_text(source)
    function=window_recovery_kernel(source).with_attr('pinv.dsl.sha256',digest)
    (root/'kernel.tir').write_text(function.script())
    with composition_backend(),jit_compiler_options(3) as compiler:
        compiler.destination=root
        print('compiling complete window strategy',flush=True)
        result=tilelang.compile(function,target='ascendc',platform='A2',out_idx=[],pass_configs=config)
        if not compiler.commands:
            raise RuntimeError('fresh compiler invocation required')
        emitted=result.get_kernel_source()
        (root/'kernel.cpp').write_text(emitted)
        shutil.copy2(result.adapter.lib._name,root/'kernel.so')
    if dsl_digest()!=digest:
        raise RuntimeError('DSL sources changed during compilation')
    receipt={'producer':'TileLang JIT','dsl_sha256':digest,'strategy':'window_cube',
             'complete_continuation':True,'scope':'window inverse, certified recovery and DSL Jacobi continuation',
             'optimization_level':3,'jit_compiler_commands':compiler.commands,
             'components':records,'block':block_records,'lowering':component_records,'pipeline':protocol,
             'runtime_header_sha256':sha256(Path(__file__).parents[3].joinpath('code_dsl/op_kernel/recovery_instructions.h').read_bytes()).hexdigest(),
             'artifacts':{'window_cube_module':{'source_sha256':sha256(emitted.encode()).hexdigest(),
                 'library_sha256':sha256((root/'kernel.so').read_bytes()).hexdigest(),
                 'tir_sha256':sha256(function.script().encode()).hexdigest()}}}
    (root/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('complete window JIT built',flush=True)


if __name__=='__main__':
    main()
