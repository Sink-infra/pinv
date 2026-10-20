"""Verified communication diagnostic, never a complete-operator acceptance.

Use only with its producer and workers in one launch, through NpuSession.run.
"""
import argparse,ctypes as C,json
from hashlib import sha256
from pathlib import Path
import numpy as np
from tests.st.backends.acl import AclRuntime
from code_dsl.op_host.planner import make_plan
from tests.common.observations import compare_mock_output,comparison_record


def main():
 p=argparse.ArgumentParser();p.add_argument('--library',required=True,type=Path);p.add_argument('--case',required=True,type=int);p.add_argument('--output',required=True,type=Path);p.add_argument('--window',type=int,default=0);p.add_argument('--peer',action='store_true');p.add_argument('--jit-receipt',type=Path);args=p.parse_args()
 args.output.mkdir(parents=True,exist_ok=False)
 root=Path('build/baseline-measurements/data-mock');a=np.load(root/f'case_{args.case:02d}.npy');gold=np.load(root/f'case_{args.case:02d}_reference.npy')
 runtime=AclRuntime()
 try:
  plan=make_plan(a.shape,soc=runtime.soc)
  if plan.geometry.m!=plan.geometry.n:raise ValueError('pipeline probe requires a square matrix')
  n=plan.geometry.n;batch=plan.geometry.batch;stride=plan.fields['stride'];blocks=plan.blocks
  # The single producer diagnostic needs n^2 to fit in UB when window is zero.
  k=(n+7)//8*8;cap=n*k if not args.window else max(n*args.window,(n-args.window)*((n-args.window+7)//8*8))
  if args.peer:
   if args.window:raise ValueError('peer full pipeline does not use a Schur window')
   half=(n+31)//32*16
   cap=max(n*half,n*((n-half+7)//8*8+16))
  words=cap+4096+10*k+2*((k+63)//64*64)+64+(n+15)//16*64
  if words*4>plan.ub_bytes:raise ValueError('producer exceeds physical UB')
  buffers=[runtime.allocate(a.size,a),runtime.allocate(a.size,np.full(a.size,np.nan,dtype=np.float32)),runtime.allocate(plan.workspace_floats),runtime.allocate(batch*32,np.zeros(batch*32,dtype=np.float32))]
  library=C.CDLL(str(args.library.resolve()));library.call.argtypes=[C.c_void_p]*4+[C.c_int32]*5+[C.c_float]*2+[C.c_void_p];library.call.restype=None
  report={'scope':'pipeline communication diagnostic; no performance acceptance','case':args.case,'shape':a.shape,'window':args.window,'peer':args.peer,'blocks':blocks,'soc':runtime.soc,'library_sha256':sha256(args.library.read_bytes()).hexdigest(),'checks':[]}
  if args.jit_receipt:
   from tests.common.provenance import jit_provenance
   report['provenance']=jit_provenance(args.library,args.jit_receipt,'peer_pipeline' if args.peer else 'pipeline')
  for iteration in range(2):
   library.call(*buffers,blocks,n,batch,stride,args.window,plan.geometry.atol,plan.geometry.rtol,runtime.stream);runtime.synchronize()
   flags=runtime.read(buffers[3],(batch*32,));output=args.output/f'output-{iteration}.bin';runtime.read(buffers[1],gold.shape).tofile(output)
   workspace=runtime.read(buffers[2],(plan.workspace_floats,))
   np.save(args.output/f'metadata-{iteration}.npy',workspace[2*batch*n*stride:2*batch*n*stride+batch*8192].reshape(batch,8192),allow_pickle=False)
   check=compare_mock_output(output,gold);report['checks'].append({'iteration':iteration,'flags':flags[:1].tolist(),**comparison_record(check)})
   (args.output/'checks.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
   if not check.passed:raise AssertionError(check)
  report['passed']=True;(args.output/'checks.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
 finally:runtime.close()


if __name__=='__main__':main()
