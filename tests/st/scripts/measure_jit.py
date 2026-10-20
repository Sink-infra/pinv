"""Compatibility report adapter over the unified JIT suite lifecycle.

Existing stage-acceptance readers can consume samples.json; launches, inputs,
checks and device sampling are exclusively owned by run_tests.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import statistics

import numpy as np

from tests.st.scripts.run_tests import main as run_suite


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--peer-build', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--cases', type=int, nargs='+', default=list(range(1, 13)))
    parser.add_argument('--processes', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=80)
    args = parser.parse_args(argv)
    if args.processes < 1 or args.warmup < 1 or len(set(args.cases)) != len(args.cases) or not set(args.cases) <= set(range(1, 13)):
        parser.error('positive process count and unique Mock IDs 1..12 are required')
    build = args.build.resolve()
    if not (build / 'tests/build.json').is_file():
        build = build.parent
    if not (build / 'tests/build.json').is_file():
        parser.error('build through tests.st.scripts.build --backend jit before sampling')
    if args.peer_build and args.peer_build.resolve() != build / 'peer':
        parser.error('resident and peer libraries must belong to one verified JIT build')
    command = ['--backend', 'jit', '--suite', 'mock', '--no-build', '--build-dir', str(build),
               '--output', str(args.output), '--processes', str(args.processes), '--warmup', str(args.warmup),
               '--repeats', '9', '--profile', '--profiler-backend', 'task', '--transport', 'npu-session', '--timeout', '600']
    for case in args.cases:
        command += ['--case', f'mock_{case:02d}']
    code = run_suite(command)
    suite = json.loads((args.output / 'result.json').read_text())
    report = {'stage': 'tilelang_jit_samples', 'scope': 'unified suite measurements; acceptance is separate',
              'dsl_sha256': suite['build']['dsl_sha256'], 'warmup': args.warmup, 'repeats': 9,
              'full_mock_selection': set(args.cases) == set(range(1, 13)), 'cases': [], 'passed': False}
    for row in suite['cases']:
        if row['status'] != 'Pass':
            continue
        case = int(row['name'].rsplit('_', 1)[1])
        if args.data:
            a = np.load(args.data / f'case_{case:02d}.npy', allow_pickle=False)
            gold = np.load(args.data / f'case_{case:02d}_reference.npy', allow_pickle=False)
            if sha256(a.tobytes()).hexdigest() != row['input_sha256'] or sha256(gold.tobytes()).hexdigest() != row['reference_sha256']:
                raise ValueError('legacy data differs from the shared workload catalog')
        folder = Path(row['measurement']).parent
        measurement = {'case': case, 'shape': row['shape'], 'provenance': row['provenance'],
                       'input_sha256': row['input_sha256'], 'reference_sha256': row['reference_sha256'],
                       'library_sha256': row['provenance']['library_sha256'], 'environment': row['environment'],
                       'warmup': args.warmup, 'repeats': 9, 'samples_us': row['kernel']['samples_us'],
                       'median_us': row['kernel']['median_us'], 'steady': row['steady'],
                       'checks': [row['first'], row['final']], 'passed': True,
                       'shared_result': row['measurement'],
                       'shared_result_sha256': sha256(Path(row['measurement']).read_bytes()).hexdigest()}
        path = folder / 'measurement.json'
        path.write_text(json.dumps(measurement, indent=2, allow_nan=False) + '\n')
        report['cases'].append({'case': case, 'process': row['process'], 'median_us': measurement['median_us'],
                                'steady': measurement['steady'], 'measurement': str(path)})
    report['medians_us'] = {str(case): statistics.median(row['median_us'] for row in report['cases'] if row['case'] == case)
                            for case in args.cases if any(row['case'] == case for row in report['cases'])}
    report['passed'] = code == 0 and all(row['steady'] for row in report['cases'])
    (args.output / 'samples.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    if not report['passed']:
        raise RuntimeError('shared JIT suite or steady sampling failed; see result.json')


if __name__ == '__main__':
    main()
