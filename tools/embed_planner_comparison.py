"""Add a separate Planner chapter, retaining the earlier three-backend chapter."""
import argparse
import base64
from datetime import datetime,timezone
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys

PROJECT=Path(__file__).resolve().parents[1]
BEGIN,END='<!-- PLANNER-COMPARISON-BEGIN -->','<!-- PLANNER-COMPARISON-END -->'
POLICIES=('moonep','native','ultra_ep','echo','eplb_fixed','step_time')
TARGETS=(.2,1,5,10,15,20)

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def uri(data,mime):return 'data:'+mime+';base64,'+base64.b64encode(data).decode('ascii')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--page-root',type=Path,default=PROJECT.parent/'balanced-kernel-explainer-20261106');ap.add_argument('--partial',action='store_true');a=ap.parse_args()
    root=a.page_root.resolve();acceptance=json.loads((PROJECT/'ACCEPTANCE.json').read_text())
    if not acceptance['complete'] and not a.partial:raise RuntimeError('full implementation is not accepted; only an explicitly labelled phase page is permitted')
    report=json.loads((PROJECT/'reports/REPORT-DATA.json').read_text())
    result=json.loads((PROJECT/'analysis/points.json').read_text());points=result['points']
    lookup={(p['planner'],p['maxvio']):p for p in points}
    snapshots=root/'history'/('20261114-before-planner-comparison-'+datetime.now(timezone.utc).strftime('%H%M%S'))
    snapshots.mkdir(parents=True,exist_ok=False)
    for name in ('overview.html','overview-first.template.html','OFFLINE-BUILD.json','FILE-HASHES.json','package_bundle.py','README.md'):
        shutil.copyfile(root/name,snapshots/name)
    old_template=(root/'overview-first.template.html').read_text(encoding='utf-8')
    old_begin,old_end='<!-- EP8-BENCHMARK-BEGIN -->','<!-- EP8-BENCHMARK-END -->'
    previous=old_template[old_template.index(old_begin):old_template.index(old_end)+len(old_end)]
    previous_sha=hashlib.sha256(previous.encode()).hexdigest()
    figures=[];figure_manifest={}
    definitions=[
        ('absolute-step-time','完整训练步时：越低越快','单位ms。每步先取八rank最大完整wall，再求后20步均值；每点三次独立运行，误差棒为重复的最小值到最大值。'),
        ('normalized-step-time','统一归一化：只使用一个分母',('所有点均除以主矩阵MoonEP策略在MaxVio0.2的'+f'{result["common_denominator_ms"]:.3f}ms；不是各策略分别归一化到1。') if result['common_denominator_ms'] else '归一化分母尚无完整三次重复，暂不绘图。'),
        ('speedup-vs-moonep-planner','相对同档MoonEP策略：越高越快','同档MoonEP策略完整步时÷本策略完整步时。这里MoonEP仅指规划策略，执行器始终是普通MegaMoE。'),
        ('whole-card-hbm','整卡HBM采样峰值：每卡，越低越省','包含初始化、预热、框架缓存及通信/计算池；取全部独立重复、八卡及采样时间中的最大值。采样可能漏掉短瞬态。纵轴从80GiB起，共同门槛90GiB。')]
    optional=[
        ('four-round-reductions','第四轮与稳定门槛','前三轮来自分阶段主矩阵，第四轮相邻配对并轮换顺序。只有4/4同向、改善中位数至少1%才达预定门槛；这不自动证明新计划或梯度模型带来收益。'),
        ('heldout-speedup-vs-strongest','两份独立路由银行','seed31415与27182，每点各三次独立运行。分母是同档step_time，分子为主矩阵选出的同一个最强已有策略；未重新拟合，数据不并入主矩阵。'),
        ('isolated-planning-overhead','隔离规划开销诊断','包含当前路由统计、在线策略和公共编译校验；排除调用前对齐及调用后CPU检查。每点40次调用，单个顺序诊断进程，不是三重复整网训练成绩。')]
    definitions += [entry for entry in optional if all((PROJECT/f'figures/{entry[0]}.{ext}').exists() for ext in ('png','svg','pdf'))]
    for stem,title,description in definitions:
        paths=[PROJECT/f'figures/{stem}.{ext}' for ext in ('png','svg','pdf')]
        if not all(path.exists() for path in paths):
            if not a.partial:raise RuntimeError('accepted figure is missing: '+stem)
            continue
        blobs={};fingerprints={}
        for ext,mime in (('png','image/png'),('svg','image/svg+xml'),('pdf','application/pdf')):
            path=PROJECT/f'figures/{stem}.{ext}';blobs[ext]=uri(path.read_bytes(),mime)
            fingerprints[ext]=dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size)
        figure_manifest[stem]=fingerprints
        links=''.join(f'<a href="{blobs[ext]}" download="planner-{stem}.{ext}">下载{ext.upper()}</a>' for ext in blobs)
        figures.append(f'<figure data-pln-chart="{stem}"><h3>{html.escape(title)}</h3><button type="button" class="pln-open" data-pln-open="{stem}" aria-haspopup="dialog" aria-controls="pln-dialog" aria-label="放大：{html.escape(title)}"><img src="{blobs["png"]}" alt="{html.escape(title)}" loading="lazy" width="1728" height="1056"></button><figcaption>{html.escape(description)} 点击图可放大。</figcaption><div class="pln-downloads">{links}</div></figure>')
    table=[]
    for target in TARGETS:
        cells=[f'<th scope="row">{target:g}</th>']
        for policy in POLICIES:
            row=lookup.get((policy,target),{})
            valid=row.get('completed',0)>=3 and not row.get('unresolved_failures',row.get('failed',0))
            pending=('未运行' if not row.get('attempts',0) else f'未完成（{row.get("completed",0)}/3）'+('，有失败待补测' if row.get('unresolved_failures') else ''))
            value=f'{row["mean_ms"]:.2f}<small>三次独立重复</small>' if valid else html.escape(pending)
            cells.append(f'<td data-pln-policy="{policy}" data-pln-vio="{target:g}">{value}</td>')
        table.append('<tr>'+''.join(cells)+'</tr>')
    if not figures:
        figures.append('<div class="pln-note" style="grid-column:1/-1"><strong>正式性能曲线尚未生成。</strong><p>当前节点尚未完成完整主矩阵。旧节点部分测量已封存，不能跨节点混入本表；未完成三次重复的点不画作正式曲线。这里不填官方图读数、预测时间或占位曲线。</p></div>')
    source_names=[('reports/REPORT.md','实验报告与当前状态'),('ACCEPTANCE.json','验收与未完成项'),('REPRODUCE.md','重放说明'),
                  ('reports/ARCHITECTURE.md','接口与生命周期'),('reports/DECISIONS.md','决策和失败记录'),
                  ('patches/native-unified-interface.patch','Native源码补丁')]
    for name,title in [('CLOSEOUT.md','负结果收尾说明'),('CLOSEOUT-AUDIT.json','120次数据身份与收尾审计'),('evidence/USER-CLOSEOUT-RELEASE.json','本任务进程与八卡释放记录')]:
        if (PROJECT/name).exists():source_names.append((name,title))
    source_names.append(('evidence/BASELINE-FREEZE.json','正式基线源码与SO指纹') if (PROJECT/'evidence/BASELINE-FREEZE.json').exists()
                        else ('incoming/WORKSPACE-MIGRATION.json','已迁移源码与SO指纹（正式矩阵尚未冻结）'))
    for name,title in [('reports/STEP-TIME-IMPLEMENTATION.md','新策略实现与模型限制'),('evidence/STEP-TIME-FREEZE.json','新策略源码、插件与四模型指纹'),
            ('configs/step-time-full.json','冻结成本模型'),('analysis/planner-decisions-summary.json','正式执行的采用与回退计数'),
            ('analysis/stability-four-rounds.json','四轮稳定性原始比值'),('analysis/planning-overhead.csv','隔离规划诊断CSV'),
            ('evidence/trajectory-audit-strict-fourth-pause.json','保留的原始轨迹停队检查'),
            ('analysis/fourth-vio5-trajectory-investigation.json','训练轨迹逐步调查'),
            ('evidence/fixed-state-pair-acceptance.json','同权重完整模型正确性诊断')]:
        if (PROJECT/name).exists():source_names.append((name,title))
    data=dict(schema='ordinary-megamoe-planner-comparison-v1',project=str(PROJECT),acceptance=acceptance,
              points=points,common_denominator_ms=result['common_denominator_ms'],figures=figure_manifest,
              prior_three_backend_chapter_sha256=previous_sha,sources={name:dict(path=str(PROJECT/name),sha256=sha(PROJECT/name)) for name,_ in source_names})
    (root/'PLANNER-COMPARISON-DATA.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    replacements=dict(__PLN_STATS__=report['stats_html'],__PLN_CHARTS__='\n'.join(figures),__PLN_CONCLUSION__=report['conclusion_html'],
        __PLN_ROWS__='\n'.join(table),__PLN_DOWNLOADS__=f'<a href="{uri((PROJECT/"analysis/points.csv").read_bytes(),"text/csv;charset=utf-8")}" download="planner-main-points.csv">下载逐点CSV</a><a href="{uri((PROJECT/"analysis/runs.csv").read_bytes(),"text/csv;charset=utf-8")}" download="planner-main-runs.csv">下载逐次运行CSV</a>',
        __PLN_INTERFACE__=report['interface_html'],__PLN_MODEL__=report['model_html'],__PLN_VALIDATION__=report['validation_html'],
        __PLN_COSTS__=report['costs_html'],__PLN_FAILURES__=report['failures_html'],__PLN_PROJECT__=html.escape(str(PROJECT)),
        __PLN_SOURCES__=''.join(f'<a href="{(PROJECT/name).as_uri()}">{html.escape(label)}</a>' for name,label in source_names),
        __PLN_DATA__=json.dumps(data,ensure_ascii=False,separators=(',',':')).replace('</','<\\/'))
    component=(PROJECT/'reports/planner-comparison.template.html').read_text(encoding='utf-8')
    for marker,value in replacements.items():
        if component.count(marker)!=1:raise RuntimeError('template marker missing or duplicated: '+marker)
        component=component.replace(marker,value)
    if acceptance.get('closed_at_user_request'):component=component.replace('09 · Planner对照 · 普通MegaMoE执行器','09 · Planner负结果 · 已按要求收尾')
    elif not acceptance['complete']:component=component.replace('09 · Planner对照 · 普通MegaMoE执行器','09 · Planner阶段结果 · 仍有未完成项')
    wrapped=BEGIN+'\n'+component+'\n'+END
    template=old_template
    if BEGIN in template:template=template[:template.index(BEGIN)]+wrapped+template[template.index(END)+len(END):]
    else:
        marker='\n</main>\n<div id="tooltip"'
        if template.count(marker)!=1:raise RuntimeError('overview insertion anchor changed')
        template=template.replace(marker,'\n'+wrapped+marker,1)
    nav='<a href="#ep8-comparison">08 EP8三路实测</a>'
    if '<a href="#planner-comparison">' not in template:
        if template.count(nav)!=1:raise RuntimeError('navigation anchor changed')
        template=template.replace(nav,nav+'<a href="#planner-comparison">09 Planner对照</a>',1)
    retained=template[template.index(old_begin):template.index(old_end)+len(old_end)]
    if retained!=previous:raise RuntimeError('previous three-backend chapter changed')
    (root/'planner-comparison.component.template.html').write_bytes((PROJECT/'reports/planner-comparison.template.html').read_bytes())
    (root/'planner-comparison.section.html').write_text(component,encoding='utf-8')
    (root/'overview-first.template.html').write_bytes(template.replace('\r\n','\n').replace('\n','\r\n').encode('utf-8'))
    sys.path.insert(0,str(root));from rebuild_offline import render_page
    manifest=json.loads((root/'OFFLINE-BUILD.json').read_text(encoding='utf-8'))
    names=['overview-first.template.html','planner-comparison.component.template.html','planner-comparison.section.html','PLANNER-COMPARISON-DATA.json']
    for name in names:manifest['inputs'][name]=sha(root/name)
    spec=manifest['pages']['overview.html'];encoded=render_page(root,spec,json.loads((root/spec['data']).read_text(encoding='utf-8')))
    spec['sha256']=hashlib.sha256(encoded).hexdigest();(root/'overview.html').write_bytes(encoded)
    (root/'OFFLINE-BUILD.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    hashes=json.loads((root/'FILE-HASHES.json').read_text(encoding='utf-8'))
    for name in names+['overview.html','OFFLINE-BUILD.json']:hashes[name]=dict(sha256=sha(root/name),bytes=(root/name).stat().st_size)
    (root/'FILE-HASHES.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    receipt=dict(passed=True,complete=acceptance['complete'],page=str(root/'overview.html'),sha256=spec['sha256'],
        previous_three_backend_chapter_preserved=True,previous_chapter_sha256=previous_sha,snapshot=str(snapshots),
        figures=len(figure_manifest),downloads_embedded=True,section='planner-comparison',only_actual_completed_points=True)
    (PROJECT/'evidence/PAGE-BUILD.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
    (root/'evidence/PLANNER-COMPARISON-BUILD.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':main()
