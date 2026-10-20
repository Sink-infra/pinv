"""Enforce the partial rollout's per-case five-percent package limit."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import statistics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--package',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    baseline=json.loads(args.baseline.read_text())
    package=json.loads(args.package.read_text())
    profiles=package['profiles']
    if len(profiles)!=3 or any(len(rows)!=12 for rows in profiles):
        raise ValueError('three complete twelve-case package profiles required')
    expected=[f'mock_{i:02d}' for i in range(1,13)]
    if any([r['name'] for r in rows]!=expected for rows in profiles):
        raise ValueError('unexpected package profile order')
    scoped=sorted(r['case'] for r in baseline['cases'])
    if scoped!=[1,2,4,5,6,7,8,10,11,12]:
        raise ValueError('fixed ten-case partial reference required')
    checks=package['checks']
    correct=(package.get('passed') is True and
             set(checks)=={'coverage','correctness','mock'} and
             all(rows and all(r['status']=='Pass' for r in rows) for rows in checks.values()))
    steady=all(r.get('steady') is True and r['status']=='Pass' for rows in profiles for r in rows)
    rows=[]
    for ref in baseline['cases']:
        case=ref['case']
        samples=[process[case-1]['kernel']['median_us'] for process in profiles]
        median=statistics.median(samples)
        rows.append({'case':case,'native_us':ref['native_us'],
                     'package_process_medians_us':samples,'package_median_us':median,
                     'package_over_native':median/ref['native_us']})
    report={'passed':correct and steady and all(r['package_over_native']<=1.05 for r in rows),
            'correctness_passed':correct,'all_profile_processes_steady':steady,
            'metric':'median across three process medians, each with nine steady TaskDuration samples',
            'baseline_commit':baseline['baseline_commit'],'full_dsl_acceptance':False,
            'native_retained_cases':[3,9],
            'input_sha256':{str(p):sha256(p.read_bytes()).hexdigest() for p in [args.baseline,args.package]},
            'maximum_slowdown_percent':100*(max(r['package_over_native'] for r in rows)-1),
            'cases':rows}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['passed','correctness_passed','maximum_slowdown_percent']}))
    if not report['passed']:raise SystemExit('package failed the fixed per-case acceptance limit')


if __name__=='__main__':main()
