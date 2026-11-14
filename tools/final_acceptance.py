"""Audit the completed experiment without running further device work."""
import csv,datetime,json,math,pathlib,statistics
from collections import Counter
from controller import TASK
from freeze_workload import check_freeze
from verify_run import verify_run

def load(p):return json.loads(p.read_text())
def main():
    campaign=load(TASK/'evidence/campaign.json')
    assert len(campaign)==54 and len({r['tag'] for r in campaign})==54
    expected={(b,v,r) for b in ['moonep','balanced','shared'] for v in [.2,1,5,10,15,20] for r in [1,2,3]}
    assert {(r['backend'],r['maxvio'],r['repeat']) for r in campaign}==expected
    freeze=load(TASK/'evidence/workload-freeze.json');check_freeze(freeze)
    router_gate=TASK/'evidence/gate-router-fp32-full'
    assert load(router_gate/'status.json')['qualified']
    router_rows=[row for rank in range(8) for row in load(router_gate/f'rank{rank}.json')]
    assert len(router_rows)==32 and all(r['fp32_gather_bitwise_equal'] and r['ids_equal'] for r in router_rows)
    router_errors={key:max(row[key] for row in router_rows) for key in ['output_relative_l2','dx_relative_l2','router_weight_relative_l2','logit_grad_relative_l2']}
    assert router_errors['output_relative_l2']<=.004 and all(v<=.02 for k,v in router_errors.items() if k!='output_relative_l2')
    (TASK/'evidence/controlled-router-check.json').write_text(json.dumps(dict(passed=True,ranks=8,cases=32,score_shape=[16384,256],topk=8,projection_hidden=32,fp32_gather_bitwise_equal=True,max_errors=router_errors),indent=2))
    identities={};logical_config=None;library_reference={};losses={};max_hbm=0
    for run in campaign:
        out=TASK/'evidence'/run['tag'];qualified=verify_run(out)
        config=dict(model=qualified['model'],optimizer=qualified['optimizer'])
        if logical_config is None:logical_config=config
        assert logical_config==config,run['tag']
        max_hbm=max(max_hbm,run['peak_driver_MiB'])
        rank_rows=[]
        for rank,record in enumerate(qualified['identities']):
            for key in ['model','inputs','routes']:
                index=(rank,key,run['maxvio'] if key=='routes' else None)
                if index in identities:assert identities[index]==record[key],(run['tag'],index)
                else:identities[index]=record[key]
            library_record=load(out/f'libraries-rank{rank}.json')
            assert library_record['sitecustomize'] is None,(run['tag'],'implicit sitecustomize')
            assert library_record['default_dtype']=='torch.float32',run['tag']
            libs=library_record['libraries']
            for name,info in libs.items():assert freeze['binaries'][name]==info['sha256'],(run['tag'],name)
            key=(run['backend'],rank)
            if key in library_reference:assert libs==library_reference[key],key
            else:library_reference[key]=libs
            rows=[json.loads(line) for line in (out/f'steps-rank{rank}.jsonl').read_text().splitlines()]
            rank_rows.append(rows)
        losses[run['tag']]=dict(initial_rank_mean_loss=statistics.mean(x[0]['loss'] for x in rank_rows),
            final_rank_mean_loss=statistics.mean(x[-1]['loss'] for x in rank_rows),
            initial_global_grad_norm=rank_rows[0][0]['grad_norm'], final_global_grad_norm=rank_rows[0][-1]['grad_norm'])
    summary=list(csv.DictReader((TASK/'tables/summary.csv').open()))
    runs=list(csv.DictReader((TASK/'tables/runs.csv').open()))
    steps=list(csv.DictReader((TASK/'tables/steps.csv').open()))
    assert len(summary)==18 and all(int(r['repeats'])==3 for r in summary)
    assert len(runs)==54 and len(steps)==54*25
    assert {r['run'] for r in runs}=={r['tag'] for r in campaign}
    assert all(math.isfinite(float(r[k])) for r in summary for k in ['mean_ms','std_ms','normalized','speedup_vs_moonep','driver_peak_GiB'])
    denominator=load(TASK/'tables/aggregation-status.json')['normalization_baseline_ms']
    for row in summary:assert abs(float(row['mean_ms'])/denominator-float(row['normalized']))<1e-12
    for name in ['absolute-step-time','normalized-step-time','speedup-vs-moonep','whole-card-hbm']:
        for ext in ['png','svg','pdf']:assert (TASK/'figures'/f'{name}.{ext}').stat().st_size>1000
    failures=[];nonformal=[];incomplete=[]
    accepted={r['tag'] for r in campaign}
    for out in sorted((TASK/'evidence').iterdir()):
        if not out.is_dir():continue
        status=out/'status.json'
        if status.exists():
            record=load(status)
            if not record.get('qualified'):failures.append(dict(run=out.name,**record))
            elif out.name not in accepted:nonformal.append(dict(run=out.name,reason='successful gate/diagnostic, not formal timing data'))
        elif (out/'admission.json').exists() or (out/'stdout.log').exists():
            incomplete.append(dict(run=out.name,reason='no completed controller status; excluded'))
    marker_failures=[]
    for p in sorted((TASK/'evidence').glob('*.exit')):
        value=p.read_text().strip()
        if value!='0':marker_failures.append(dict(path=p.relative_to(TASK).as_posix(),value=value))
    excluded=dict(failed_device_runs=failures,nonformal_successes=nonformal,incomplete_runs=incomplete,nonzero_or_unparseable_exit_markers=marker_failures)
    (TASK/'evidence/failure-and-exclusion-index.json').write_text(json.dumps(excluded,indent=2))
    (TASK/'tables/training-loss-audit.json').write_text(json.dumps(losses,indent=2))
    acceptance=dict(passed=True,time=datetime.datetime.now(datetime.timezone.utc).isoformat(),formal_runs=54,
        repeats_per_backend_point=3,measured_global_steps=54*20,measured_rank_steps=54*20*8,
        full_rank_steps=54*25*8,source_files=len(freeze['sources']),loaded_binary_files=len(freeze['binaries']),
        max_driver_hbm_GiB=max_hbm/1024,normalization_baseline_ms=denominator,
        failed_device_runs=len(failures),incomplete_device_runs=len(incomplete),
        scope='common cropped Ascend L2 model synthetic training; not an exact reproduction of the unknown official H20 9-layer harness')
    (TASK/'ACCEPTANCE.json').write_text(json.dumps(acceptance,indent=2))
    lines=['# 失败与排除数据索引','',
        '所有失败、部分输出及成功诊断均保留在项目evidence；只有54次正式合格训练进入曲线。下表是独立运行的控制器状态，构建退出标记另列。详细原因和修复见DECISIONS.jsonl及各目录stdout.log。','',
        '| 运行 | 退出码 | 完成rank | HBM MiB |','|---|---:|---:|---:|']
    for r in failures:lines.append(f"| {r['run']} | {r.get('exit','?')} | {r.get('rank_completions','?')} | {r.get('peak_driver_MiB','?')} |")
    lines+=['','未完成控制器状态的目录：'+('、'.join(r['run'] for r in incomplete) or '无')+'。','',
        f'成功但仅作门禁/诊断的运行共{len(nonformal)}组，详见[evidence/failure-and-exclusion-index.json](../evidence/failure-and-exclusion-index.json)。它们不用于正式性能均值。','',
        '## 构建/解析等退出标记','', '| 记录 | 标记 |','|---|---|']
    for r in marker_failures:lines.append(f"| {r['path']} | {r['value']} |")
    lines+=['','未能解析的早期退出标记不是成功构建证据。最终四套Native32/64产物以r3构建日志、preflight实际能力和后续设备数值门禁为准。原始r1/r2记录保留。','',
        '主要失败包括：早期Native redispatch device参数描述符4字节错位；大模型/中间公共裁剪OOM；不支持的allocator参数；旧ABI或错误CMake目标的能力拒绝；共享/MoonEP额外门禁辅助脚本错误。已定位和修复后使用新tag重新验证；没有将失败运行改为成功。','']
    (TASK/'reports/FAILURE-AND-EXCLUSION-INDEX.md').write_text('\n'.join(lines))
    print(json.dumps(acceptance),flush=True)
if __name__=='__main__':main()
