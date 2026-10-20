"""Compare official elapsed times, keeping partial DSL and native cases explicit."""
import argparse,json,statistics
from hashlib import sha256
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--jit',type=Path,required=True);p.add_argument('--package',type=Path,required=True)
    p.add_argument('--commit',required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    baseline=json.loads(args.baseline.read_text());candidate=json.loads(args.candidate.read_text())
    b,c=baseline['record'],candidate['record']
    jit=json.loads(args.jit.read_text());package=json.loads(args.package.read_text())
    if any(r.get('code_tree_identity_status')!='verified' for r in [baseline,candidate]):
        raise ValueError('verified official source identities are required')
    if b.get('status')!='passed' or b.get('passedCases')!=12:raise ValueError('native official reference failed')
    if len(package['profiles'])<3:raise ValueError('three package processes are required')
    if len(b['cases'])!=12 or len(c.get('cases',[]))!=12:raise ValueError('complete official case results are required')
    bm={r['index']:r for r in b['cases']};cm={r['index']:r for r in c['cases']};jm={r['case']:r for r in jit['cases']}
    pm={i+1:statistics.median(rows[i]['kernel']['median_us'] for rows in package['profiles']) for i in range(12)}
    rows=[]
    for case in range(1,13):
        native,current=float(bm[case]['timeMicros']),float(cm[case]['timeMicros'])
        j=jm.get(case);ratio=current/native
        rows.append({'case':case,'implementation':'TileLang generated' if j else 'native retained',
                     'local_native_us':j['native_us'] if j else None,'local_jit_us':j['jit_us'] if j else None,
                     'local_generated_package_us':pm[case],
                     'official_native_us':native,'official_candidate_us':current,
                     'official_over_native':ratio,'official_slowdown_percent':100*(ratio-1),
                     'official_over_local_package_percent':100*(current/pm[case]-1),
                     'official_correctness':cm[case]['status'],
                     'within_5_percent':ratio<=1.05})
    dsl=[r for r in rows if r['case'] in jm]
    report={'stage':'official_partial_dsl_validation','commit':args.commit,'full_dsl_acceptance':False,
            'baseline_commit':jit['baseline_commit'],'baseline_run_id':b['id'],'candidate_run_id':c['id'],
            'baseline_submission_id':baseline['submission_id'],'candidate_submission_id':candidate['submission_id'],
            'official_correctness_passed':c['status']=='passed' and c['passedCases']==12,
            'baseline_score':b['score'],'candidate_score':c['score'],
            'maximum_dsl_slowdown_percent':max(r['official_slowdown_percent'] for r in dsl),
            'all_ten_dsl_cases_within_5_percent':all(r['within_5_percent'] for r in dsl),
            'all_twelve_cases_within_5_percent':all(r['within_5_percent'] for r in rows),
            'local_package_passed':package.get('passed') is True,
            'raw_reports':{str(p):sha256(p.read_bytes()).hexdigest() for p in
                           [args.baseline,args.candidate,args.jit,args.package]},
            'cases':rows}
    report['passed']=report['official_correctness_passed'] and report['all_twelve_cases_within_5_percent']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# 阶段官方验证','',
           f'原基线 `{jit["baseline_commit"][:7]}`：{b["passedCases"]}/12，{b["score"]} 分，Run `{b["id"]}`。',
           f'阶段提交 `{args.commit[:7]}`：{c["passedCases"]}/12，{c["score"]} 分，Run `{c["id"]}`。','',
           '阶段版本的 10 个 Mock 使用 DSL 生成内核；Case 3、9 保留原实现。仅验证当前阶段，不代表所有策略已移植。',
           f'10 个 DSL 用例最大官方减速 **{report["maximum_dsl_slowdown_percent"]:+.2f}%**；全部在 5% 内：**{report["all_ten_dsl_cases_within_5_percent"]}**。','',
           '## 官方相对性能','',
           '| Case | 路径 | 官方原实现 μs | 官方阶段 μs | 减速 | 5% 内 |',
           '|---|---|---:|---:|---:|---|']
    for r in rows:
        lines.append(f'| {r["case"]} | {"DSL" if r["case"] in jm else "保留原生"} | {r["official_native_us"]:.2f} | {r["official_candidate_us"]:.2f} | {r["official_slowdown_percent"]:+.2f}% | {r["within_5_percent"]} |')
    lines+=['','## 本地绝对耗时与官方','',
            '本地为 Sun 的 Ascend910B3 / CANN 9.0.0，三进程各 9 次 Task Duration 的中位数。16 阶热身 1000 次，其余 80 次。官方回执没有公开此次设备 SKU 和计时配置，不能把绝对耗时偏差直接归因于硬件或编译器。','',
            '| Case | 本地 JIT μs | 本地生成 CANN 包 μs | 官方阶段 μs | 官方／本地包 − 1 |',
            '|---|---:|---:|---:|---:|']
    for r in rows:
        j=f'{r["local_jit_us"]:.2f}' if r['local_jit_us'] is not None else '未移植'
        lines.append(f'| {r["case"]} | {j} | {r["local_generated_package_us"]:.2f} | {r["official_candidate_us"]:.2f} | {r["official_over_local_package_percent"]:+.2f}% |')
    lines+=['','5% 的判断使用官方阶段／固定官方原基线逐项比值。得分依赖官方评分参考，验收以绝对耗时和正确性为准。历史失败与不稳态采样保留，不换用较慢基线。','',
            '原始官方回执、本地采样摘要及输入报告 SHA-256 与本报告一并保存；完整输入、输出、CSV 和编译产物另行归档。']
    args.output.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:report[k] for k in ['official_correctness_passed','candidate_score','maximum_dsl_slowdown_percent','all_ten_dsl_cases_within_5_percent']}))


if __name__=='__main__':main()
