"""Recapture native clocks after verified correctness, preserving noisy runs.

Uses the immutable native runner, verifier and Task Duration parser. Every
requested round completes; an unsteady capture makes the report fail rather
than silently discarding a case or selecting a favorable sample.
"""
from tests.st.backends.transport import SessionExecutor

import argparse,json
from pathlib import Path
from tests.st.scripts.baseline import BASELINE_COMMIT,query_soc,require_native_build
from tests.common.measurements import hardware_snapshot,steady_samples
from tests.st.scripts import run_tests
from tests.st.scripts.build import verify_build


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build',type=Path,required=True)
    p.add_argument('--correctness',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--processes',type=int,default=3)
    p.add_argument('--warmup',type=int,default=80)
    p.add_argument('--cases',type=int,nargs='+',default=list(range(1,13)))
    args=p.parse_args()
    if args.processes<1 or args.warmup<1:p.error('positive process and warmup counts required')
    if len(set(args.cases))!=len(args.cases) or not set(args.cases)<=set(range(1,13)):p.error('unique Mock IDs 1..12 required')
    build=args.build.resolve();receipt=require_native_build(verify_build(build))
    proof=json.loads(args.correctness.read_text())
    if proof.get('baseline_commit')!=BASELINE_COMMIT or proof.get('build')!=receipt:
        p.error('correctness evidence belongs to a different native build')
    if {s:len(rows) for s,rows in proof.get('checks',{}).items()}!={'coverage':18,'correctness':43,'mock':12}:
        p.error('all 73 native correctness checks are required')
    if any(row['status']!='Pass' for rows in proof['checks'].values() for row in rows):
        p.error('native correctness has not passed')
    args.output.mkdir(parents=True,exist_ok=False)
    cases=run_tests.prepare_cases('mock',[f'mock_{i:02d}' for i in args.cases],args.output/'data-mock')
    report={'baseline_commit':BASELINE_COMMIT,'build':receipt,'correctness_evidence':str(args.correctness),
            'checks':proof['checks'],'warmup':args.warmup,'repeats':9,
            'full_mock_selection':set(args.cases)==set(range(1,13)),'profiles':[],'passed':False}
    def save():(args.output/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    from npu_runtime import NpuSession
    runner=build/'tests/linalg_pinv_runner'
    with NpuSession(device=0,lock_timeout=1800) as session:
        report['environment']=hardware_snapshot(query_soc(session))
        report['npu_service']={'pid':session.service_pid,'lock_wait_seconds':session.lock_wait_seconds}
        for process in range(1,args.processes+1):
            folder=args.output/f'process-{process}';folder.mkdir();rows=[];report['profiles'].append(rows)
            for case in cases:
                row=run_tests.run_case(case,build,folder/case['name'],warmup=args.warmup,repeats=9,timeout=600,
                                      profile=True,profiler_backend='task',executor=SessionExecutor(session),output_init=case.get('output_init','runtime'))
                row['steady']=row['status']=='Pass' and steady_samples(row['kernel']['samples_us'])
                rows.append(row);save()
                print(process,case['name'],row['status'],row.get('kernel',{}).get('median_us'),row['steady'],flush=True)
                if row['status']!='Pass':raise RuntimeError('native profile or numerical check failed: '+case['name'])
        verify_build(build)
        report['passed']=all(row['steady'] for rows in report['profiles'] for row in rows);save()


if __name__=='__main__':main()
