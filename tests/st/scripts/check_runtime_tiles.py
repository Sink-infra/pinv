"""Numerical JIT checks at shared-tile boundaries, using original references."""
import argparse,ctypes as C,json
from pathlib import Path
from hashlib import sha256
import numpy as np
from tests.st.backends.acl import AclRuntime,bind_launch
from tests.common.provenance import jit_provenance
from tests.common.cases import CaseSpec
from tests.common.reference import materialize_case,pinv_reference
from tests.common.observations import compare_mock_output,comparison_record


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--library',type=Path,required=True);p.add_argument('--jit-receipt',type=Path,required=True);p.add_argument('--strategy',choices=['compact','local'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--sizes',type=int,nargs='+');args=p.parse_args()
 provenance=jit_provenance(args.library,args.jit_receipt,args.strategy)
 sizes=args.sizes or ([15,16] if args.strategy=='compact' else [33,40,41,48,49,56,57,63,64])
 args.output.mkdir(parents=True,exist_ok=False);report={'provenance':provenance,'checks':[],'failures':[]}
 runtime=AclRuntime()
 try:
  launch=bind_launch(C.CDLL(str(args.library.resolve())),module=True)
  cases=[]
  for n in sizes:
   for rank in [n,n-2]:
    cases.append(CaseSpec(name=f'jit_runtime_n{n}_rank{rank}',category='correctness',expectation='success',a_shape=(2,n,n),rank=rank,seed=20261020+n*100+rank))
  n=16 if args.strategy=='compact' else 64
  scales=[]
  for exponent in [-3,0,8,16,64,126]:
   center=np.float32(2.0**exponent)
   scales.extend([np.nextafter(center,np.float32(-np.inf)),center,np.nextafter(center,np.float32(np.inf))])
  center=np.float32(0.1)
  scales.extend([np.nextafter(center,np.float32(-np.inf)),center,np.nextafter(center,np.float32(np.inf))])
  for index,value in enumerate(scales):
   cases.append(CaseSpec(name=f'jit_power_n{n}_edge{index}',category='correctness',expectation='success',a_shape=(2,n,n),input_pattern='diagonal',rank=n,spectrum=(1.0,)*n,scale=float(value)))
  if args.strategy=='compact':
   for label,absolute,relative in [('absolute',2.0,None),('relative',0.0,0.5),('all_truncated',64.0,0.0)]:
    cases.append(CaseSpec(name='jit_compact_cutoff_'+label,category='correctness',expectation='success',
                         a_shape=(2,16,16),input_pattern='diagonal',rank=16,
                         spectrum=(4.0,)*8+(0.1,)*8,atol=absolute,rtol=relative))
  for case in cases:
    n=case.a_shape[-1];name=case.name
    data=materialize_case(case);reference=pinv_reference(case,data)
    a=np.ascontiguousarray(data.a.values,dtype=np.float32);gold=np.ascontiguousarray(reference.values,dtype=np.float64)
    head=max(8,1<<(n-1).bit_length())
    buffers=[runtime.allocate(a.size,a),runtime.allocate(a.size,np.full(a.size,np.nan,dtype=np.float32)),runtime.allocate(4*2*n*head),runtime.allocate(64)]
    for iteration in range(2):
     relative=case.rtol if case.rtol is not None else (0.0 if case.atol>0 else n*2**-23)
     launch(buffers,2,n,n,2,case.atol,relative,runtime.stream)
     path=args.output/(name+f'-{iteration}.bin');runtime.read(buffers[1],gold.shape).tofile(path)
     check=compare_mock_output(path,gold);report['checks'].append({'case':name,'iteration':iteration,'input_sha256':sha256(a.tobytes()).hexdigest(),'scale':case.scale,**comparison_record(check)})
     (args.output/'checks.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
     if not check.passed:report['failures'].append({'case':name,'iteration':iteration,'mismatch_count':check.mismatch_count})
    print(name,'Pass' if all(row['passed'] for row in report['checks'][-2:]) else 'Fail',flush=True)
  report['passed']=not report['failures']
  (args.output/'checks.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
  if not report['passed']:raise RuntimeError('boundary checks failed; see checks.json')
 finally:runtime.close()


if __name__=='__main__':main()
