"""Check the packaged peer continuation on the same inputs as the JIT gate."""
from tests.st.backends.transport import SessionExecutor

import argparse
from hashlib import sha256
import json
from pathlib import Path
import numpy as np
from tests.common.cases import CaseSpec
from tests.common.reference import materialize_case,pinv_reference
from tests.st.scripts import run_tests
from tests.st.scripts.build import verify_build
from npu_runtime import NpuSession


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--jit-checks',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    build=args.build.resolve();jit=json.loads(args.jit_checks.read_text())
    if jit.get('passed') is not True:raise ValueError('passed real JIT checks are required')
    expected={row['case']:row['input_sha256'] for row in jit['checks']}
    report={'stage':'standalone_cann_peer_continuation','build':verify_build(build),
            'jit_checks':str(args.jit_checks),'jit_checks_sha256':sha256(args.jit_checks.read_bytes()).hexdigest(),
            'checks':[],'passed':False}
    with NpuSession(device=0,lock_timeout=1800) as session:
        for rank in [200,198,0]:
            name=f'peer_n200_rank{rank}'
            case=CaseSpec(name=name,category='correctness',expectation='success',
                          a_shape=(3,200,200),rank=rank,seed=20261020+rank)
            data=materialize_case(case);reference=pinv_reference(case,data)
            a=np.ascontiguousarray(data.a.values,dtype=np.float32)
            gold=np.ascontiguousarray(reference.values,dtype=np.float64)
            input_sha=sha256(a.tobytes()).hexdigest()
            if expected[name]!=input_sha:raise ValueError('input differs from real JIT evidence')
            folder=args.output/name;folder.mkdir();input_path=folder/'a.bin';a.tofile(input_path)
            prepared={'name':name,'shape':a.shape,'input':input_path,'atol':0.0,'rtol':-1.0,'gold':gold,'output_init':'nan'}
            row=run_tests.run_case(prepared,build,folder/'run',warmup=0,repeats=2,timeout=1200,executor=SessionExecutor(session),output_init='nan')
            row.update(input_sha256=input_sha,reference_sha256=sha256(gold.tobytes()).hexdigest())
            report['checks'].append(row)
            (args.output/'checks.json').write_text(json.dumps(report,indent=2)+'\n')
            print(name,row['status'],flush=True)
            if row['status']!='Pass':raise RuntimeError('packaged peer continuation failed')
    verify_build(build);report['passed']=True
    (args.output/'checks.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
