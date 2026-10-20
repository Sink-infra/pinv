"""Bind the authorized ten-case rollout to real JIT samples and fresh checks."""
import argparse,json
from pathlib import Path
from hashlib import sha256
from code_dsl.source import dsl_digest
from tests.common.provenance import jit_provenance,require_stage_acceptance


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary',type=Path,required=True);p.add_argument('--checks',type=Path,required=True)
    p.add_argument('--module-build',type=Path,required=True);p.add_argument('--peer-build',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    provenance={}
    for strategy in ['compact','local','peer_pipeline']:
        folder=args.peer_build if strategy=='peer_pipeline' else args.module_build
        lib=folder/'kernel.so' if strategy=='peer_pipeline' else folder/strategy/'kernel.so'
        provenance[strategy]=jit_provenance(lib,folder/'build-receipt.json',strategy)
        if provenance[strategy]['dsl_sha256']!=dsl_digest():raise ValueError('stale JIT build')
    proof=[]
    for name in ['compact-coverage','compact-correctness','compact-boundaries','local-coverage','local-correctness','local-boundaries','peer-runtime']:
        path=args.checks/name/'checks.json';report=json.loads(path.read_text())
        strategy='peer_pipeline' if name=='peer-runtime' else name.split('-')[0]
        measured=report.get('library_sha256',report.get('provenance',{}).get('library_sha256'))
        if report.get('passed') is not True or measured!=provenance[strategy]['library_sha256']:raise ValueError('stale or failed JIT checks')
        proof.append({'path':str(path),'sha256':sha256(path.read_bytes()).hexdigest(),'checks':len(report['checks']),'library_sha256':measured})
    summary=json.loads(args.summary.read_text());rows=[]
    for row in summary['cases']:
        if row['case'] in [3,9]:continue
        strategy='compact' if row['case'] in [1,4,7,10] else 'peer_pipeline' if row['case'] in [6,12] else 'local'
        refs=[]
        for ref in row['jit_measurements']:
            path=Path(ref);report=json.loads(path.read_text())
            if not report.get('passed') or not report.get('steady') or report['provenance']['library_sha256']!=provenance[strategy]['library_sha256']:raise ValueError('JIT sample binary or correctness mismatch')
            refs.append({'path':ref,'sha256':sha256(path.read_bytes()).hexdigest(),
                         'original_dsl_sha256':report['provenance']['dsl_sha256'],
                         'reused_by_identical_binary':report['provenance']['dsl_sha256']!=dsl_digest()})
        row.update(correctness_passed=True,current_provenance=provenance[strategy],measurement_evidence=refs);rows.append(row)
    report={'stage':'tilelang_jit_partial_acceptance','passed':True,'baseline_commit':summary['baseline_commit'],
            'dsl_sha256':dsl_digest(),'max_jit_over_native':1.05,'full_dsl_acceptance':False,
            'native_retained_cases':[3,9],'scope':'user-authorized partial rollout; compact/local/peer strategies only',
            'scoped_correctness_passed':True,'scoped_correctness_evidence':proof,'cases':rows}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    try:
        accepted=require_stage_acceptance(args.output,dsl_sha256=dsl_digest())
    except ValueError as error:
        report['passed']=False
        report['acceptance_failure']=str(error)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
        raise
    print(accepted)


if __name__=='__main__':main()
