"""Fresh numerical and three-process JIT evidence for the authorized rollout.

The native reference is fixed, including its original raw measurements. A noisy
or slower repeat is never substituted to improve the reported relative result.
Device children execute through the shared NpuSession service.
"""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--module-build',type=Path,required=True)
    p.add_argument('--peer-build',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    commands=[]
    def record(command,result):
        commands.append({'argv':command,'returncode':result.returncode})
        (args.output/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
        if result.returncode:raise RuntimeError('validation child failed: '+str(command))
    def cpu(module,*options):
        command=[sys.executable,'-m','tests.st.scripts.'+module,*map(str,options)]
        result=subprocess.run(command,check=False)
        record(command,result)
    checks=args.output/'checks'
    from npu_runtime import NpuSession
    with NpuSession(device=0,lock_timeout=1800) as session:
        for strategy,lo,hi in [('compact',1,32),('local',33,64)]:
            library=(args.module_build/strategy/'kernel.so').resolve()
            for suite in ['coverage','correctness']:
                name=strategy+'-'+suite
                command=[sys.executable,'-m','tests.st.scripts.check_shared','--module',
                         '--library',str(library),'--min-cols',str(lo),'--max-cols',str(hi),
                         '--suite',suite,'--output',str((checks/name).resolve())]
                result=session.run(command,cwd=os.getcwd(),environment=os.environ.copy(),timeout=1200)
                (args.output/(name+'.log')).write_text(result.stdout+result.stderr)
                record(command,result);print(name,'Pass',flush=True)
            name=strategy+'-boundaries'
            command=[sys.executable,'-m','tests.st.scripts.check_runtime_tiles',
                     '--library',str(library),'--jit-receipt',str((args.module_build/'build-receipt.json').resolve()),
                     '--strategy',strategy,'--output',str((checks/name).resolve())]
            result=session.run(command,cwd=os.getcwd(),environment=os.environ.copy(),timeout=1200)
            (args.output/(name+'.log')).write_text(result.stdout+result.stderr)
            record(command,result);print(name,'Pass',flush=True)
        name='peer-runtime'
        command=[sys.executable,'-m','tests.st.scripts.check_peer_runtime',
                 '--library',str((args.peer_build/'kernel.so').resolve()),
                 '--jit-receipt',str((args.peer_build/'build-receipt.json').resolve()),
                 '--output',str((checks/name).resolve())]
        result=session.run(command,cwd=os.getcwd(),environment=os.environ.copy(),timeout=1200)
        (args.output/(name+'.log')).write_text(result.stdout+result.stderr)
        record(command,result);print(name,'Pass',flush=True)
    captures=[]
    for name,cases,warmup in [('small',[1,7],1000),('other',[2,4,5,6,8,10,11,12],80)]:
        destination=args.output/name
        cpu('measure_jit','--build',args.module_build,'--peer-build',args.peer_build,
            '--output',destination,'--cases',*cases,'--processes',3,'--warmup',warmup)
        captures.extend(json.loads((destination/'samples.json').read_text())['cases'])
    baseline=json.loads(args.baseline.read_text());rows=[]
    for previous in baseline['cases']:
        row={k:v for k,v in previous.items() if k in
             ['case','native_us','native_processes','native_steady','warmup','repeats']}
        samples=[r for r in captures if r['case']==row['case']]
        assert len(samples)==3
        row.update(jit_us=statistics.median(r['median_us'] for r in samples),
                   jit_processes=3,jit_steady=all(r['steady'] for r in samples),
                   jit_measurements=[r['measurement'] for r in samples])
        row['jit_over_native']=row['jit_us']/row['native_us'];rows.append(row)
    summary={'baseline_commit':baseline['baseline_commit'],'native_reference':{
        'path':str(args.baseline),'sha256':sha256(args.baseline.read_bytes()).hexdigest()},'cases':rows}
    summary_path=args.output/'summary.json';summary_path.write_text(json.dumps(summary,indent=2)+'\n')
    cpu('accept_stage_jit','--module-build',args.module_build,'--peer-build',args.peer_build,
        '--summary',summary_path,'--checks',checks,'--output',args.output/'acceptance.json')


if __name__=='__main__':main()
