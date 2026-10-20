"""Check runtime-shaped peer JIT and its DSL numerical continuation."""
import argparse,ctypes as C,json
from pathlib import Path
from hashlib import sha256
import numpy as np
from tests.st.backends.acl import AclRuntime
from tests.common.provenance import jit_provenance
from code_dsl.op_host.planner import make_plan
from tests.common.cases import CaseSpec
from tests.common.reference import materialize_case,pinv_reference
from tests.common.observations import compare_mock_output,comparison_record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library',type=Path,required=True);p.add_argument('--jit-receipt',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    provenance=jit_provenance(args.library,args.jit_receipt,'peer_pipeline')
    receipt=json.loads(args.jit_receipt.read_text())
    if receipt.get('complete_continuation') is not True:p.error('complete DSL continuation required')
    admission_index=receipt.get('fast_admission_status_index')
    args.output.mkdir(parents=True,exist_ok=False);report={'provenance':provenance,'checks':[]}
    runtime=AclRuntime()
    try:
        library=C.CDLL(str(args.library.resolve()))
        library.call.argtypes=[C.c_void_p]*4+[C.c_int32]*5+[C.c_float]*2+[C.c_void_p];library.call.restype=None
        for rank in [200,198,0]:
            name=f'peer_n200_rank{rank}'
            case=CaseSpec(name=name,category='correctness',expectation='success',a_shape=(3,200,200),rank=rank,seed=20261020+rank)
            data=materialize_case(case);reference=pinv_reference(case,data)
            a=np.ascontiguousarray(data.a.values,dtype=np.float32);gold=np.ascontiguousarray(reference.values,dtype=np.float64)
            plan=make_plan(a.shape,soc=runtime.soc)
            buffers=[runtime.allocate(a.size,a),runtime.allocate(a.size,np.full(a.size,np.nan,dtype=np.float32)),
                     runtime.allocate(plan.workspace_floats),runtime.allocate(96)]
            for iteration in range(2):
                library.call(*buffers,plan.blocks,200,3,plan.fields['stride'],0,0.0,200*2**-23,runtime.stream);runtime.synchronize()
                output=args.output/f'{name}-{iteration}.bin';runtime.read(buffers[1],gold.shape).tofile(output)
                check=compare_mock_output(output,gold)
                status=runtime.read(buffers[3],(96,))
                admitted=bool(status[admission_index]) if admission_index is not None else None
                report['checks'].append({'case':name,'iteration':iteration,'input_sha256':sha256(a.tobytes()).hexdigest(),
                                         'final_status_slot0':bool(status[0]),'fast_path_accepted':admitted,**comparison_record(check)})
                (args.output/'checks.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
                if not check.passed:raise AssertionError((name,iteration,check))
                if rank==0 and admitted is not False:raise AssertionError('zero input must exercise the DSL continuation')
            print(name,'Pass',flush=True)
        report['passed']=True;(args.output/'checks.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    finally:runtime.close()


if __name__=='__main__':main()
