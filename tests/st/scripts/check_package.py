"""Verify and time the standalone staged CANN package through NpuSession.

Run with the package project's immutable tests on sys.path before this repo.
The complete original suites are kept; only the device launch transport changes.
"""
from tests.st.backends.transport import SessionExecutor

import argparse,json
from pathlib import Path
from tests.st.scripts.baseline import query_soc
from tests.common.measurements import hardware_snapshot,steady_samples


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--acceptance',type=Path,required=True);p.add_argument('--processes',type=int,default=3)
    p.add_argument('--checks-only',action='store_true');p.add_argument('--profiles-only',action='store_true')
    p.add_argument('--data',type=Path)
    args=p.parse_args()
    from code_dsl.source import dsl_digest
    from tests.common.provenance import require_stage_acceptance
    acceptance=require_stage_acceptance(args.acceptance,dsl_sha256=dsl_digest())
    from tests.st.scripts import run_tests
    from tests.st.scripts.build import verify_build
    from npu_runtime import NpuSession
    args.build=args.build.resolve();args.output=args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=False)
    data=args.data.resolve() if args.data else args.output
    receipt=verify_build(args.build)
    prepared={s:run_tests.prepare_cases(s,[],data/('data-'+s)) for s in ['coverage','correctness','mock']}
    runner=args.build/'tests/linalg_pinv_runner'
    report={'stage':'staged_standalone_cann_package','jit_acceptance':acceptance,'build':receipt,'checks':{},'profiles':[],'passed':False}
    def save():(args.output/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    with NpuSession(device=0,lock_timeout=1800) as npu:
        report['npu_service']={'pid':npu.service_pid,'lock_wait_seconds':npu.lock_wait_seconds}
        report['environment']=hardware_snapshot(query_soc(npu));save()
        if not args.profiles_only:
            for suite,cases in prepared.items():
                out=args.output/('check-'+suite);out.mkdir();rows=[]
                for case in cases:
                    row=run_tests.run_case(case,args.build,out/case['name'],warmup=0,repeats=2,timeout=900,executor=SessionExecutor(npu),output_init=case.get('output_init',run_tests.output_initialization(suite)))
                    rows.append(row);report['checks'][suite]=rows;save();print(suite,case['name'],row['status'],flush=True)
                    if row['status']!='Pass':raise RuntimeError('staged package failed: '+str(row))
        if not args.checks_only:
            for process in range(1,args.processes+1):
                out=args.output/f'process-{process}';out.mkdir();rows=[]
                for case in prepared['mock']:
                    warmup=1000 if case['name'] in ['mock_01','mock_07'] else 80
                    row=run_tests.run_case(case,args.build,out/case['name'],warmup=warmup,repeats=9,timeout=900,profile=True,profiler_backend='task',executor=SessionExecutor(npu),output_init=case.get('output_init','runtime'))
                    row['warmup']=warmup
                    row['steady']=row['status']=='Pass' and steady_samples(row['kernel']['samples_us'])
                    rows.append(row);report['profiles']=(report['profiles'][:process-1]+[rows]);save()
                    print('profile',process,case['name'],row.get('kernel',{}).get('median_us'),row['status'],'steady',row['steady'],flush=True)
                    if row['status']!='Pass':raise RuntimeError('staged package profile failed: '+str(row))
        verify_build(args.build)
        report['passed']=all(row['status']=='Pass' for rows in report['checks'].values() for row in rows) and all(row['steady'] for rows in report['profiles'] for row in rows)
        save()


if __name__=='__main__':main()
