"""Capture the immutable native baseline through the existing NPU broker.

Only launch transport is adapted. The repository's original preparation,
runner, first/final numerical checks and task-clock parser remain in use.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from tests.st.backends.transport import SessionExecutor
from tests.st.scripts.runtime import environment
from code_dsl.convert.native_source import BASELINE_COMMIT


def require_native_build(receipt):
    """A generated CANN package must never be relabeled as the native reference."""
    from code_dsl.convert.native_source import native_source
    with native_source() as source:
        expected = {p.relative_to(source).as_posix(): sha256(p.read_bytes()).hexdigest()
                    for p in source.rglob('*') if p.is_file()}
    if receipt.get('layout_files') != expected:
        raise ValueError('native baseline requires the exact pinned Git source; current code/ is a generated package')
    return receipt



def query_soc(session):
    source="import ctypes as C; lib=C.CDLL('libascendcl.so'); assert lib.aclInit(None)==0; assert lib.aclrtSetDevice(0)==0; lib.aclrtGetSocName.restype=C.c_char_p; print(lib.aclrtGetSocName().decode()); assert lib.aclFinalize()==0"
    result=session.run(['python3','-c',source],environment=environment(),timeout=30)
    if result.returncode:raise RuntimeError(result.stderr)
    soc=result.stdout.strip()
    if not soc.startswith('Ascend910'):raise ValueError('unexpected measured SoC: '+soc)
    return soc


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--processes',type=int,default=3)
    parser.add_argument('--finalize-existing',action='store_true')
    args=parser.parse_args()
    if args.processes<1:parser.error('positive process count required')
    from tests.st.scripts import run_tests
    from tests.st.scripts.build import verify_build
    from tests.st.scripts.runtime import ROOT
    from tests.common.measurements import hardware_snapshot,steady_samples
    from npu_runtime import NpuSession
    args.build=args.build.resolve();args.output=args.output.resolve()
    if args.finalize_existing:
        report=json.loads((args.output/'comparison.json').read_text())
        assert report['build']==require_native_build(verify_build(args.build))
        assert {s:len(rows) for s,rows in report['checks'].items()}=={'coverage':18,'correctness':43,'mock':12}
        assert all(r['status']=='Pass' for rows in report['checks'].values() for r in rows)
        assert len(report['profiles'])==args.processes
        assert all(len(rows)==12 and all(r['status']=='Pass' and steady_samples(r['kernel']['samples_us']) for r in rows) for rows in report['profiles'])
        with NpuSession(device=0,lock_timeout=1800) as npu:report['environment']=hardware_snapshot(query_soc(npu))
        report['passed']=True
        (args.output/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        return 0
    args.output.mkdir(parents=True,exist_ok=False)
    receipt=require_native_build(verify_build(args.build))
    prepared={s:run_tests.prepare_cases(s,[],args.output/('data-'+s)) for s in ['coverage','correctness','mock']}
    runner=args.build/'tests/linalg_pinv_runner'
    report={'baseline_commit':BASELINE_COMMIT,'build':receipt,'checks':{},'profiles':[]}
    def save():
        (args.output/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    with NpuSession(device=0,lock_timeout=1800) as npu:
        report['npu_service']={'pid':npu.service_pid,'lock_wait_seconds':npu.lock_wait_seconds}
        report['environment']=hardware_snapshot(query_soc(npu));save()
        for suite,cases in prepared.items():
            out=args.output/('check-'+suite);out.mkdir();rows=[]
            for case in cases:
                row=run_tests.run_case(case,args.build,out/case['name'],warmup=0,repeats=2,timeout=600,executor=SessionExecutor(npu),output_init=case.get('output_init',run_tests.output_initialization(suite)))
                rows.append(row);report['checks'][suite]=rows;save();print(suite,case['name'],row['status'],flush=True)
                if row['status']!='Pass':raise RuntimeError('native baseline failed: '+case['name']+': '+str(row))
        for i in range(args.processes):
            out=args.output/('process-'+str(i+1));out.mkdir();rows=[]
            for case in prepared['mock']:
                row=run_tests.run_case(case,args.build,out/case['name'],warmup=80,repeats=9,timeout=600,profile=True,profiler_backend='task',executor=SessionExecutor(npu),output_init=case.get('output_init','runtime'))
                if row['status']!='Pass':raise RuntimeError('baseline profile failed: '+str(row))
                if not steady_samples(row['kernel']['samples_us']):raise RuntimeError('baseline task clocks are not steady: '+case['name'])
                rows.append(row);print('profile',i+1,case['name'],row['kernel']['median_us'],flush=True)
            report['profiles'].append(rows);save()
        verify_build(args.build)
        report['passed']=True;save()
    return 0


if __name__=='__main__':raise SystemExit(main())
