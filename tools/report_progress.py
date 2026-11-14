"""Write an evidence-bounded phase report while the authorized campaign runs."""
from datetime import datetime,timezone
import json
from pathlib import Path
from campaign_records import verified_contention

ROOT=Path(__file__).resolve().parents[1]
rows=[]
for name in ('primary-existing','primary-existing-recovery','primary-step-time','primary-step-time-recovery'):
    path=ROOT/f'evidence/campaign-{name}.json'
    if path.exists():rows.extend(json.loads(path.read_text()))
excluded=[row for row in rows if verified_contention(ROOT,row)]
valid=[row for row in rows if row['state']=='pass' and not verified_contention(ROOT,row)]
passed=len(valid);failed=[row for row in rows if row['state']!='pass']
held=any(not path.with_name(path.name.replace('INFRASTRUCTURE-HOLD','RESUMED')).exists()
         for path in (ROOT/'evidence').glob('OWN-QUEUE-INFRASTRUCTURE-HOLD*.json'))
step_file=ROOT/'evidence/step-time-qualification.json'
completed_step=step_file.exists() and json.loads(step_file.read_text()).get('passed') is True
existing_completed=sum(r['planner']!='step_time' for r in valid)
if (ROOT/'evidence/ORIGINAL-HOST-RESUMED-20261114.json').exists():held=False
cal_file=ROOT/'evidence/CALIBRATION-ACCEPTANCE.json'
calibrated=cal_file.exists() and json.loads(cal_file.read_text()).get('passed') is True
driver_file=ROOT/'evidence/step-time-stage-driver.jsonl'
driver=json.loads(driver_file.read_text().splitlines()[-1]) if driver_file.exists() and driver_file.stat().st_size else {}
accepted_file=ROOT/'ACCEPTANCE.json'
accepted=accepted_file.exists() and json.loads(accepted_file.read_text()).get('complete') is True
closed=(ROOT/'evidence/USER-CLOSEOUT-REQUEST.json').exists()
fallback_stage='primary_extended_matrix'
if passed>=108:
    def count(name):
        p=ROOT/f'evidence/campaign-{name}.json'
        return len(json.loads(p.read_text())) if p.exists() else 0
    fallback_stage='fourth_paired' if count('fourth-paired')<12 else 'heldout_banks' if count('heldout-a')<36 or count('heldout-b')<36 else 'final_validation_and_delivery'
stage='waiting_for_exclusive_device' if held else 'final_policy_qualification' if calibrated and not completed_step else 'independent_calibration' if existing_completed==90 and not completed_step else 'primary_existing_matrix' if not completed_step else driver.get('stage',fallback_stage)
status=dict(status='closed_at_user_request' if closed else 'complete' if accepted else 'needs_review' if driver.get('state')=='needs_review' else 'running',stage='closeout' if closed else 'complete' if accepted else stage,
    ordinary_only=True,formal_attempted=len(rows),formal_completed=passed,formal_failed=len(failed),formal_planned_initial=108,
    existing_policies_completed=existing_completed,excluded_contention_attempts=len(excluded),
    receive_capacity=163840,hbm_gate_gib=90,cpu_policy_cases_passed=480,device_final_gates_passed=7,
    full_model_capacity_gates_passed=5,calibration_completed=calibrated,
    step_time_qualified=completed_step,updated_utc=datetime.now(timezone.utc).isoformat())
(ROOT/'STATUS.json').write_text(json.dumps(status,indent=2))
lines=['# 普通MegaMoE Planner对照阶段报告','',f'更新时间：{status["updated_utc"]}。'+('已按用户要求收尾，原完整验证未完成。' if closed else '全部必需测量已验收。' if accepted else '任务仍在执行；本报告不表示完整方案已验收。'),'',
       f'有效主矩阵 **{passed}/108** 次，已有策略部分 **{status["existing_policies_completed"]}/90** 次。原始尝试{len(rows)}次；外部活动窗口覆盖{len(excluded)}次（与失败记录可能重叠），原始退出失败{len(failed)}次。','',
       '已实测通过：CPU480组策略参考；最终七项设备接口/生命周期/32槽/缩放/clip门禁；五种策略在完整两层模型的三次Adam更新与163840共同接收容量门禁。',
       '', '固定条件：普通MegaMoE，L2/EP8/E256/K8/H7168/F2048/S16384/rank，框架shared expert、FP32持久参数/梯度/Adam、BF16专家计算、整层重计算、固定owner、无CPU offload。',
       '', '正式步时口径：每次新进程25步，前5步预热；后20步每步先取八rank最大完整wall，再求本次均值。每点三次独立重复，保留全部慢步和慢轮。',
       '', '| MaxVio | MoonEP策略 | Native | UltraEP策略 | ECHO策略 | EPLB-fixed | step_time |','|---:|---:|---:|---:|---:|---:|---:|']
for vio in (.2,1,5,10,15,20):
    cells=[]
    for policy in ('moonep','native','ultra_ep','echo','eplb_fixed','step_time'):
        selected=[r for r in valid if r['planner']==policy and r['maxvio']==vio]
        cells.append(f'{sum(r["state"]=="pass" for r in selected)}/3'+('，有失败' if any(r['state']!='pass' for r in selected) else ''))
    lines.append('|'+f'{vio:g}|'+'|'.join(cells)+'|')
if valid:lines+=['',f'有效正式运行的整卡HBM采样最高值：{max(r["peak_driver_MiB"] for r in valid)/1024:.4f}GiB；共同门槛90GiB。采样可能漏掉短瞬态，不能当作连续硬件测量的精确峰值。']
if held:lines+=['','当前在作业间持有本任务排队器，等待NPU独占条件恢复。另一容器agy-live-a5的fabric_bench进程已由驱动PID/cgroup记录确认。没有停止对方任务或reset卡。按其容器启动到本任务停队的保守窗口，统一排除所有重叠尝试，不根据快慢选择；将以关联补测恢复计划单元。详见evidence/EXTERNAL-CONTENTION.json。']
pending=[]
if existing_completed<90:pending.append('已有策略全部重复')
if not calibrated:pending.append('独立成本校准')
if not completed_step:pending.append('step_time正确性门禁')
if passed<108:pending.append('step_time主矩阵')
if not accepted:pending+=['第四轮/独立银行及条件消融按各自账本验收','最终交付校验']
lines+=['',(('未执行或未完成阶段按用户要求取消：' if closed else '尚未完成：')+'、'.join(pending)+'。当前不能声称新策略已有效。') if pending else '完整验收见ACCEPTANCE.json；是否有稳定收益以四轮门槛和独立银行为准。',
    '', '主结果、诊断和失败均保存于本项目；原三后端实验未改动。MoonEP/UltraEP/ECHO/EPLB在本节均指策略移植，不是原系统性能复现。',
    '', '数据入口：`evidence/campaign-primary-existing.json`、`analysis/runs.csv`、`analysis/points.csv`。构建及初始冻结：`evidence/BASELINE-FREEZE.json`。接口：`reports/ARCHITECTURE.md`。决策：`reports/DECISIONS.md`。']
(ROOT/'reports/PROGRESS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(status))
