"""Accept all Mock DSL paths from shared inputs, complete Tasks and real JIT receipts."""
import argparse,json
from pathlib import Path
from hashlib import sha256
from statistics import median

from code_dsl.source import dsl_digest
from tests.common.paired import compare_backends
from tests.common.provenance import require_mock_path_acceptance
from tests.st.backends.jit import module_identity,verify

CAC='cac1192a095d2485c47b5c881b59eda12c01784f'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mock',type=Path,required=True)
    parser.add_argument('--checks',type=Path,action='append',required=True)
    parser.add_argument('--build-dir',type=Path,required=True)
    parser.add_argument('--gain-reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    identity=verify(args.build_dir);libraries=module_identity(args.build_dir)
    current=dsl_digest()
    jit=json.loads((args.mock/'jit/result.json').read_text())
    ascend=json.loads((args.mock/'ascend/result.json').read_text())
    cac=json.loads(args.gain_reference.read_text())
    if cac['build']['source_sha256']!='b6c4512f221b7b6a9c291099dd1274c8eb8192c847ece7dfaa71d3e111aca232':
        raise ValueError('gain reference is not the exact CAC source snapshot')
    paired=compare_backends(jit['cases'],ascend['cases'])
    gain=compare_backends(jit['cases'],cac['cases'],max_slowdown=1.0)
    if not paired['performance_accepted'] or not paired['correctness_passed']:
        raise ValueError('all Mock paths need correct, steady, three-process paired Task measurements')
    if not gain['correctness_passed']:
        raise ValueError('CAC reference does not match the workloads')
    by_gain={row['name']:row for row in gain['cases']}
    rows=[]
    for row in paired['cases']:
        no=int(row['name'].split('_')[-1])
        evidence=[]
        for measured in [r for r in jit['cases'] if r['name']==row['name']]:
            path=Path(measured['measurement']);document=json.loads(path.read_text())
            provenance=document['provenance']
            expected=libraries[provenance['strategy']]
            if provenance!=expected or provenance['dsl_sha256']!=current:
                raise ValueError('JIT measurement uses a different source or binary')
            if len(document['kernel']['samples_us'])<(5 if max(document['shape'][-2:])<=32 else 3):
                raise ValueError('insufficient Task samples')
            evidence.append({'path':str(path),'sha256':sha256(path.read_bytes()).hexdigest(),
                             'library_sha256':provenance['library_sha256']})
        samples=[v for r in jit['cases'] if r['name']==row['name'] for v in r['kernel']['samples_us']]
        rows.append({**row,'case':no,'current_provenance':libraries[measured['provenance']['strategy']],
                     'measurement_evidence':evidence,'sample_range_us':[min(samples),max(samples)],
                     'cac_us':by_gain[row['name']]['ascend_us'],
                     'jit_over_cac':by_gain[row['name']]['jit_over_ascend'],
                     'cac_steady':by_gain[row['name']]['steady']})
    checks=[]
    for path in args.checks:
        document=json.loads(path.read_text())
        equivalent={}
        if document['build']['dsl_sha256']!=current:
            # Keep historical binary evidence labeled as historical. It can
            # cover unchanged strategies only when their emitted device source
            # and SDK are identical; changed paths need fresh measurements.
            if document['build']['sdk']!=identity['sdk']:
                raise ValueError('historical correctness evidence uses a different SDK')
            for measured in document['cases']:
                if measured['status']!='Pass':continue
                provenance=measured['provenance'];strategy=provenance['strategy']
                expected=libraries[strategy]['source_sha256']
                if provenance['source_sha256']!=expected:
                    raise ValueError('changed strategy needs fresh correctness evidence: '+strategy)
                equivalent[strategy]=expected
        checks.append({'path':str(path),'sha256':sha256(path.read_bytes()).hexdigest(),
                       'suite':document['suite'],'scoped_passed':document.get('scoped_passed'),
                       'measured_dsl_sha256':document['build']['dsl_sha256'],
                       'identical_device_sources':equivalent,
                       'unsupported_cases':document.get('unsupported_cases',[]),
                       'failures':[{'name':r['name'],'status':r['status']} for r in document['cases']
                                   if r['status'] not in ['Pass','Unsupported']]})
    report={'stage':'tilelang_jit_mock_paths_acceptance','passed':True,'dsl_sha256':current,
            'full_dsl_acceptance':False,'native_mock_cases':[],
            'scope':'all twelve Mock paths in DSL; unported non-Mock strategies retain native CANN paths',
            'reference_commit':'9c9791a91ece24ff02ed4b66e9d7e7251ba139b6',
            'gain_reference_commit':CAC,'gain_reference':{'path':str(args.gain_reference),
              'sha256':sha256(args.gain_reference.read_bytes()).hexdigest()},
            'build':identity,'checks':checks,'cases':rows,
            'paired_report':paired,'gain_report':gain}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    require_mock_path_acceptance(args.output,dsl_sha256=current)
    print('all Mock paths accepted with current JIT provenance')


if __name__=='__main__':
    main()
