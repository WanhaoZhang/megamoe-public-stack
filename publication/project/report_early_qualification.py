"""Historical reducer, superseded by report_current.py; do not overwrite the current report."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent
DATA = ROOT/'evidence'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def reference(tag):
    directory = DATA/tag
    if not (directory/'status.json').exists():
        return dict(status='pending')
    status = read(directory/'status.json')
    ranks = [read(path) for path in directory.glob('rank*.json')]
    result = dict(status='passed' if status['qualified'] else 'failed', receipt=status)
    if len(ranks) == 8 and all('updates' in rank for rank in ranks):
        result['maximum_relative_l2'] = {name:max(update['comparisons'][layer][index]['relative_l2']
             for rank in ranks for update in rank['updates'] for layer in range(2))
             for index,name in enumerate(('output','dx','routing_weight','gate_up_weight','down_weight'))}
        result['replay_audits'] = [rank['replay_audit'] for rank in ranks]
        result['rank_count'] = len(ranks)
    elif len(ranks)==8 and all(rank.get('passed') for rank in ranks):
        result['rank_count']=8
        result['cases_per_rank']=[rank['cases'] for rank in ranks]
        result['library_sha256']=sorted({rank['sha256'] for rank in ranks})
    return result


def model_run(tag):
    directory = DATA/tag
    if not (directory/'status.json').exists():
        return dict(status='pending')
    receipt = read(directory/'status.json')
    rows = [json.loads(line) for path in directory.glob('steps-rank*.jsonl') for line in path.read_text().splitlines()]
    return dict(status='passed' if receipt['qualified'] else 'failed',receipt=receipt,
        sampled_whole_card_gib=receipt['peak_driver_MiB']/1024,
        memory_gate_pass=receipt['qualified'] and receipt['peak_driver_MiB']<=90*1024,
        completed_rank_steps=len(rows),formal_performance=False)


def compare(tags):
    directories = [DATA/tag for tag in tags]
    if not all((directory/'complete-rank7.json').exists() for directory in directories):
        return dict(status='pending')
    equal = {}
    norm_difference = 0.0
    loss_difference = 0.0
    for identity in ('initial','input-identity','route-identity'):
        equal[identity] = all(read(directory/f'{identity}-rank{rank}.json') ==
                             read(directories[0]/f'{identity}-rank{rank}.json')
                             for directory in directories[1:] for rank in range(8))
    for rank in range(8):
        baseline = [json.loads(line) for line in (directories[0]/f'steps-rank{rank}.jsonl').read_text().splitlines()]
        for directory in directories[1:]:
            current = [json.loads(line) for line in (directory/f'steps-rank{rank}.jsonl').read_text().splitlines()]
            for expected,actual in zip(baseline,current):
                norm_difference = max(norm_difference,abs(expected['grad_norm']-actual['grad_norm'])/max(abs(expected['grad_norm']),1e-30))
                loss_difference = max(loss_difference,abs(expected['loss']-actual['loss']))
    return dict(status='compared',identities_equal=equal,max_norm_trajectory_relative_difference=norm_difference,
                max_loss_absolute_difference=loss_difference,not_independent_full_model_gradient_l2=True)


summary = dict(updated=datetime.now(timezone(timedelta(hours=8))).isoformat(),full_plan_complete=False,
    cpu_contract=9, expert_mb4=reference('mb4-moon-torch'),
    grouped_domains=reference('grouped-domain-pytorch'),
    grouped_rejection_recovery=reference('grouped-domain-rejection'),
    small_windows={str(window):model_run(f'ga4-w{window}-r2') for window in (1,2)},
    primary_windows={str(window):model_run(f'primary-l4-ga4-w{window}') for window in (1,2,4)},
    primary_comparison=compare([f'primary-l4-ga4-w{window}' for window in (1,2,4)]),
    online_small={policy:model_run(tag) for policy,tag in
        [('legacy','frontier-legacy-small'),('easy_home','frontier-easy-small-r2')]},
    same_weight_gradient_comparison=read(DATA/'frontier-gradient-comparison.json'),
    device_scheduler=reference('device-scheduler-reference-v3'),
    online_primary=model_run('frontier-primary-easy-w2-r4'),
    online_capacity_failure=model_run('frontier-primary-easy-w4-r2'),
    online_window4_oom=model_run('frontier-primary-easy-w4-r3'),
    advanced_common_window=2,common_receive_capacity=262144,
    policy_qualification=read(ROOT/'policy-qualification.json') if (ROOT/'policy-qualification.json').exists() else [],
    fenced_diagnosis=read(ROOT/'gradient-mismatch-diagnosis.json') if (ROOT/'gradient-mismatch-diagnosis.json').exists() else [],
    w2_qualification=read(ROOT/'w2-qualification.json') if (ROOT/'w2-qualification.json').exists() else [],
    formal_matrix_completed=0,joint_strategy_implemented=False,afore_cache_implemented=False,
    online_framework_frontier_implemented=True)
(ROOT/'STATUS.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

table = '\n'.join(f"| W{window} | {result['status']} | {result.get('completed_rank_steps',0)} | "
                  f"{result.get('sampled_whole_card_gib',float('nan')):.3f} |"
                  for window,result in summary['primary_windows'].items())
reference_rows = '\n'.join(f"| {label} | {summary[key]['status']} | "
    f"{summary[key].get('maximum_relative_l2',{}).get('output',float('nan')):.6f} | "
    f"{max(summary[key].get('maximum_relative_l2',{}).values(),default=float('nan')):.6f} |"
    for key,label in [('expert_mb4','E256四microbatch保存态'),('grouped_domains','E512双层域分组'),
                      ('grouped_rejection_recovery','分组协议拒绝后恢复')])
gradient_rows='\n'.join(f"| {name} | {record['relative_l2']:.6f} | {record['passed']} |" for name,record
    in summary['same_weight_gradient_comparison']['categories'].items())
report = f'''# 训练版EasyBalance：实现与资格报告

更新时间：{summary['updated']}。

**整网在线就绪调度及AscendC选择器已接入。16K主形状W2完成八卡三次Adam，采样峰值80.78GiB。W4触及显存预算，所有先进调度采用共同W2/GA4条件；小形状固定分组W4发现完整梯度不一致，正在隔离，性能矩阵继续关闭。没有新planner性能收益或论文创新性结论。**

## 代码与固定条件

独立源码：`stack/`，分支`research/temporal-replica`，基线`84a3363`。公开main和旧收尾数据未作为新实验改写。
完整计划见[PLAN.md](PLAN.md)，变更原因和排除依据见[DECISIONS.md](DECISIONS.md)。
原始数据保存在本目录`evidence/`，配置、源码补丁、SO指纹及复现脚本都留在项目内。

新主形状：L4、EP8/E256/Top8、H4096/F1024、每条完整16K序列；每卡microbatch1、GA4，每次更新32条序列/524288个序列token。
V4 attention/mHC和框架shared expert保留；FP32参数/梯度/Adam、BF16专家计算、原整层非重入重计算、无CPU offload。
在线分组运行共同接收容量262144（2×S×K）、共同最大W2，仍为GA4；旧容量163840及W4失败记录另列，不与当前性能条件混算。
clip1和Adam仅在四份microbatch梯度全部完成后各执行一次。恒定LR1e-5、betas(0.9,0.95)、epsilon1e-8、weight decay0.01。

## 已实现

- 路由和计划身份扩展为`(step,microbatch,layer,invocation)`，重计算恢复相同上下文，旧GA1路由种子保持兼容。
- GA4及W1/W2/W4保图入口；每组反序反向，完成后一次clip/Adam，计划按实际microbatch数量退休。
- 同热点、轮换热点、随机热点路由模式；只做统一全局专家ID排列，不能将小切片fixture称作精确MaxVio10。
- FIFO、同层固定分组、EasyBalance-style MaxUtil的CPU参考选择器；W4最小执行集2，推迟两次后强制纳入。
- 双层虚拟专家域、实时owner权重拼接、单次Native调用及输出拆回；保留原始Parameter和梯度归主。
- 显式分组调用身份，复用冻结计划重放。分组ID/域/形状跨rank协议检查及拒绝恢复门禁另列。
- E512固定home分组编译器，使用同一设备端校验器；并未放宽现有E256搜索kernel的静态数组上限。
- 原TimerV2元数据补齐microbatch身份；SO启动前21项SHA校验；非有限更新在Adam前拒绝。
- Attention/router可见性前置计算、整层组checkpoint、跨rank一致波次及零值反向依赖链；前置计算计入完整训练时间。
- 原始FSDP模块/Parameter在构造阶段保留；重放用公开unshard接口恢复layer与HC参数；有效零概率初始化只安装原有梯度缩放hook。
- AscendC FIFO/固定分组/MaxUtil选择器ABI3通过八rank各506项对照/拒绝测试；整数比率、确定性tie-break、到达顺序及两层域约束均与CPU参考一致。

## 数值验证

CPU共9项测试通过，包括1/2/3/4任务组逐行守恒。原始执行回执和CPU-TESTS.json保存在项目内。
NPU以下检查均为八rank、三种负载场景更新，包含空owner、热点改变、池权重污染、checkpoint及反序反向。
参考采用独立PyTorch专家数学计算与可微EP通信；原生池每个原始调用只规划一次。
这些算子门禁使用小形状和SGD检查权重更新，整网三次Adam检查单独列出。

| 检查 | 状态 | 最大输出相对L2 | 所有输出/梯度项最大相对L2 |
|---|---|---:|---:|
{reference_rows}

输出门槛0.004，梯度门槛0.02。完整数据及每类梯度在[STATUS.json](STATUS.json)。

## 新主形状的整网资格测试

| 窗口 | 状态 | 已完成rank×更新数 | 驱动采样整卡峰值GiB |
|---|---|---:|---:|
{table}

每个成功点是8rank×3次Adam更新，检查FP32持久状态、参数更新、路由调用次数、实际库及零遗留计划。
HBM来自npu-smi采样，不是连续瞬时峰值；共同门槛90GiB，allocator预算89.5GiB。
本表是早期普通保图路径、接收容量163840，不能拿这些数字与后续双池容量262144的在线方法直接计算显存优势或加速。
初始参数、输入和路由跨窗口对比，以及范数/损失轨迹差异，见STATUS.json。范数轨迹差异不能替代同权重梯度L2。

**以上每进程只有3次更新，均处于正式协议的前5次预热范围；不用于加速比、最优窗口或论文性能结论。**

## 分组原型的含义与限制

两个层各有32个本地home专家，映射为同一调用的64个home专家，副本槽为0；`(layer,expert)`可逆，owner rank保持不变。
两份FFN输入进入同一次大核，随后按原行位置拆回。其反向梯度回到原参数，未创建新的optimizer owner。
原生token上限仍是16384，因此两个完整16K输入需要在Attention之后分成两轮8K+8K的FFN波次；完整Attention上下文不拆。

普通保图入口仍单独保留，W控制保存态数量；新增`--temporal-policy`入口已经接到Attention/router之后的真实就绪前沿。
CPU EasyBalance原型小形状三次更新中实际出现4组跨层配对（共11次任务集合选择），并非仅保存多份模型图。
当前路由可见性使用额外一次no_grad前缀计算，因此每个逻辑任务的router调用为原始、探测和重放共三次；普通路径为两次。
所有前置计算、HCCL统计、选择、编译和协议检查都在步时内。未优化的额外工作可能抵消收益，需要与无探测的普通Native比较。

### 同权重整网完整梯度

小形状L2/H512/F512/S256、GA4/W4；在clip/Adam之前导出全部FP32 owner梯度，同普通路径比较。
八rank初始逻辑参数、输入和路由SHA一致，参数集合、形状、dtype和有限性逐项检查通过。

| 参数类别 | 相对L2 | ≤0.02 |
|---|---:|---|
{gradient_rows}

详细逐参数数值在[evidence/frontier-gradient-comparison.json](evidence/frontier-gradient-comparison.json)。
这是完整梯度比较；此前不同更新轨迹的范数比较只作回归参考。小切片fixture不宣称精确MaxVio10。
AscendC版本需要完成设备选择对照及后续整网回归，不能直接继承CPU原型的全部验收。

## 失败与排除

- 初始FFN128被原生256对齐约束拒绝；记录保留。
- 初期误选非冻结timeroff库，反向误差/NaN数据全部排除；改用冻结aligned库并增加21-SO门禁。
- 独立Torch参考在冷owner上缺失零梯度依赖，已改为统一可微打包通信及零依赖，重新验收。
- 有空FSDP shard时诊断max归约非法，已修正；两项旧排队测试在准入前取消。
- Native在小工作量上不启副本不代表错误；副本恢复门禁使用MoonEP策略显式激活副本。
- 首次E512分组被E256统一搜索编译器拒绝；采用显式固定home路径并复用公共校验，不扩大搜索数组。
- 其他会话观察到的gradscan任务不纳入本报告；其进程未被本轮停止。
- 首次跨层整网重放因mHC仍分片触发Tensor/DTensor混用，公开unshard恢复后重跑通过，原失败记录保留。
- `frontier-primary-easy-w4`因其他测试占用八卡在准入阶段退出，未启动训练，不算OOM、性能失败或完成点。控制器改为有状态的最长一小时空闲等待。

## 尚未完成与下一步

1. 验证AscendC选择器与CPU参考逐例一致；完成16K在线分组主形状的三次Adam、90GiB、路由/梯度/池生命周期门禁。
2. 完成固定分组、FIFO、EasyBalance及独立复制/Afore-style策略的同运行时对照；原生无前置计算基线必须保留。
3. 用TimerV2定位真实关键路径、权重就绪和梯度交付，而非把各rank独立零点拼接。
4. 仅在简单组合出现可重复缺陷后实现联合策略；规划、编译、通信和校验全部计时。
5. 完成所有正确性门禁后再运行正式108次矩阵、独立银行和条件消融。

完整计划完成状态为false。已发现可验证的两项代价：探测前缀的可见性成本，以及16K原生限制下多波次对完整专家梯度义务的放大。
它们目前是需要打点的机制假设，不能据此宣布新算法有效或具备论文创新性。
'''
(ROOT/'REPORT.md').write_text(report,encoding='utf-8')
print(json.dumps(dict(primary_windows=summary['primary_windows'],full_plan_complete=False),ensure_ascii=False))
