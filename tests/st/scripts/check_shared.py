"""Check a compiled shared kernel using the original immutable test fixtures.

Launch this module only as a child of NpuSession.run. It performs no numerical
work on the host; verification uses the repository's independent references.
"""
import argparse
import ctypes as C
from hashlib import sha256
import json
from pathlib import Path
import numpy as np
from tests.st.backends.acl import AclRuntime,bind_launch
from code_dsl.op_host.geometry import geometry
from tests.common.cases import load_cases
from tests.st.scripts.verify_result import verify_case_output
from tests.st.scripts.run_tests import comparison_record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library',type=Path,required=True)
    parser.add_argument('--suite',choices=['coverage','correctness'],required=True)
    parser.add_argument('--data',type=Path,default=Path('build/baseline-measurements'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-cols',type=int,default=512)
    parser.add_argument('--min-cols',type=int,default=1)
    parser.add_argument('--names',nargs='*')
    parser.add_argument('--rectangular-only',action='store_true')
    parser.add_argument('--module',action='store_true')
    args=parser.parse_args()
    args.output.mkdir(exist_ok=False,parents=True)
    runtime=AclRuntime()
    launch=bind_launch(C.CDLL(str(args.library.resolve())),module=args.module)
    report={'library_sha256':sha256(args.library.read_bytes()).hexdigest(),'soc':runtime.soc,'checks':[]}
    try:
        for case in load_cases(args.suite):
            if case.expectation!='success' or min(case.a_shape[-2:])>args.max_cols or (args.names and case.name not in args.names):continue
            if min(case.a_shape[-2:])<args.min_cols:continue
            if args.rectangular_only and case.a_shape[-2]==case.a_shape[-1]:continue
            plan=geometry(case.a_shape,atol=case.atol,rtol=case.rtol,hermitian=case.hermitian)
            folder=args.data/('data-'+args.suite)/case.name
            values=np.fromfile(folder/'a.bin',dtype='<f4')
            metadata=json.loads((folder/'case.json').read_text())
            assert sha256(values.tobytes()).hexdigest()==metadata['files']['a']['sha256']
            rows=max(plan.m,plan.n);cols=min(plan.m,plan.n)
            head=max(8,1<<(rows-1).bit_length());basis=max(8,1<<(cols-1).bit_length())
            workspace=2*plan.batch*cols*(head+basis)
            buffers=[runtime.allocate(values.size,values),runtime.allocate(values.size,np.full(values.size,np.nan,dtype=np.float32)),runtime.allocate(workspace)]
            if args.module:buffers.append(runtime.allocate(plan.batch*32))
            checks=[]
            for iteration in range(2):
                launch(buffers,min(plan.batch,40),plan.m,plan.n,plan.batch,plan.atol,plan.rtol,runtime.stream)
                result=runtime.read(buffers[1],plan.output_shape)
                output=args.output/(case.name+f'-{iteration}.bin');result.tofile(output)
                check=verify_case_output(folder,output);checks.append(comparison_record(check))
                if not check.passed:raise AssertionError((case.name,iteration,check))
            report['checks'].append({'case':case.name,'shape':plan.shape,'iterations':checks})
            (args.output/'checks.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
            print(case.name,'Pass',flush=True)
        report['passed']=True
        (args.output/'checks.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    finally:runtime.close()


if __name__=='__main__':main()
