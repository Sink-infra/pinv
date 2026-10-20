"""Enforce the requested order: TileLang JIT first, standalone AscendC later."""
from hashlib import sha256
from pathlib import Path
import json

MOCK_CASES=set(range(1,13))
MAX_JIT_OVER_NATIVE=1.05  # User-approved per-case maximum slowdown.


def jit_provenance(library,receipt,strategy):
    """Bind a measured library to an actual TileLang JIT build receipt."""
    library,receipt=Path(library),Path(receipt)
    document=json.loads(receipt.read_text())
    if document.get('producer')!='TileLang JIT':raise ValueError('a real TileLang JIT build receipt is required')
    expected=document['artifacts'][strategy+'_module']['library_sha256']
    actual=sha256(library.read_bytes()).hexdigest()
    if expected!=actual:raise ValueError('measured library does not match the TileLang JIT receipt')
    return {'stage':'tilelang_jit','strategy':strategy,'build_receipt_sha256':sha256(receipt.read_bytes()).hexdigest(),
            'library_sha256':actual,'dsl_sha256':document.get('dsl_sha256'),
            'source_sha256':document['artifacts'][strategy+'_module']['source_sha256']}


def require_jit_acceptance(path,*,dsl_sha256=None):
    """No standalone-source comparison/export before all JIT cases pass."""
    path=Path(path);report=json.loads(path.read_text())
    if report.get('stage')!='tilelang_jit' or report.get('passed') is not True:
        raise ValueError('TileLang JIT performance must pass before the standalone AscendC stage')
    if dsl_sha256 is not None and report.get('dsl_sha256')!=dsl_sha256:
        raise ValueError('JIT acceptance belongs to a different DSL revision')
    rows=report.get('cases',[])
    if len(rows)!=12 or {r.get('case') for r in rows}!=MOCK_CASES:raise ValueError('all 12 Mock cases are required')
    if report.get('baseline_commit')!='ec5ec7a2b5c9555a46636e08d25d9d05f46c0df2':raise ValueError('baseline changed')
    if report.get('correctness_passed') is not True:raise ValueError('complete correctness checks are required')
    for row in rows:
        if row.get('correctness_passed') is not True or row.get('steady') is not True:raise ValueError('invalid JIT case evidence')
        if row.get('processes',0)<3:raise ValueError('three independent JIT processes per case are required')
        ratio=row.get('jit_over_native')
        if not isinstance(ratio,(int,float)) or not 0<ratio<=MAX_JIT_OVER_NATIVE:raise ValueError('TileLang JIT exceeds the approved 5% slowdown')
        if len(row.get('jit_measurements',[]))<3:raise ValueError('raw JIT measurement references are required')
    return {'path':str(path),'sha256':sha256(path.read_bytes()).hexdigest()}


def require_stage_acceptance(path,*,dsl_sha256):
    """Explicitly authorized partial rollout; never full DSL acceptance."""
    path=Path(path);report=json.loads(path.read_text())
    if report.get('stage')!='tilelang_jit_partial_acceptance' or report.get('passed') is not True:
        raise ValueError('scoped JIT acceptance is required for a staged export')
    if report.get('dsl_sha256')!=dsl_sha256:raise ValueError('DSL revision changed')
    if report.get('baseline_commit')!='ec5ec7a2b5c9555a46636e08d25d9d05f46c0df2':raise ValueError('baseline changed')
    expected=MOCK_CASES-{3,9}
    rows=report.get('cases',[])
    if len(rows)!=len(expected) or {r.get('case') for r in rows}!=expected:raise ValueError('all ten staged Mock cases are required')
    if report.get('scoped_correctness_passed') is not True:raise ValueError('staged strategies need correctness evidence')
    if report.get('full_dsl_acceptance') is not False or report.get('native_retained_cases')!=[3,9]:raise ValueError('partial rollout must disclose native paths')
    for row in rows:
        if not row.get('correctness_passed') or not row.get('jit_steady') or not row.get('native_steady'):raise ValueError('invalid case evidence')
        if min(row.get('jit_processes',0),row.get('native_processes',0))<3:raise ValueError('three processes per path are required')
        if not 0<row.get('jit_over_native',0)<=MAX_JIT_OVER_NATIVE:raise ValueError('JIT exceeds 5%')
        if len(row.get('jit_measurements',[]))<3:raise ValueError('missing raw measurements')
    return {'path':str(path),'sha256':sha256(path.read_bytes()).hexdigest(),'scope':'partial rollout, ten Mock cases; native paths retained'}
