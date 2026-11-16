# raw dW计算排布解耦：实现与匹配筛查

本子实验位于 `D:\megamoe_train\planner-executor-ep8-lab\step-lifetime-study\raw-control-20261116`，独立分支 `research/step-lifetime-raw-control`。只修改普通MegaMoE，框架shared expert保留原计算。旧33点性能结果与causal-v1诊断均不覆盖。

## 实现

ABI2同一个worker内，sidecar7=1保留legacy行为，sidecar7=2独立选择raw输出，保留原专家优先顺序和行前缀处理；单独关闭核内owner Reduce，并将replica dW写入返回张量。raw输出不发布未写入SHMEM partial的ready，空owner可正常清零。没有删除原执行池串行保护。cooperative backward构建在preserve模式下显式拒绝，尚未验收的组合不视为支持。

本轮资格仅覆盖EP8、32home＋32replica、worker前/后向flags0、panel restore2的普通路径。early DW1仍受原形状条件限制，不因新增开关就宣称它已经在全部形状生效。

TimerOFF worker SHA256：`72653abf3c22fb951624dd5dc420c488d2511d1c22e5e605a3621cb4210502bf`。构建记录在implementation。首次构建在部署未就绪时被前置检查拒绝，未开始编译；失败记录保留，部署完成后构建成功。

## 正确性

- 两种schedule×bank/sparse_remote×B0-B3×W1/W2×8rank，共256项独立FFN输出、dx、dp、dW比较全部通过。sparse_remote强制所有owner无本地任务，梯度仅由远端副本交付；每rank32个空home、含24个完全未路由逻辑专家。
- 完整模型L2/H512/F512/S256、GA4/W2、MoonEP正副本规划，两种schedule各完成3次Adam更新，参数实际变化，逐参数梯度finite检查通过；初始参数、输入和整轮路由身份相同。完整step0逐参数FP32梯度最大相对L2为0.00778248，门槛0.02。
- 后续参数数值轨迹未逐元素导出比较；不能将三更新完成与step0梯度资格说成所有参数全程逐位一致。自然训练路由仍未验证。

## TimerOFF匹配筛查

主形状保持L4 EP8/E256/Top8 H4096/F1024，每卡16K完整序列×GA4，W2，recv262144，FP32持久状态/BF16计算，固定owner、原AdamW，无CPU offload。固定Native规划与同层分组、same模式MaxVio10，两个新进程各25次更新，5预热＋20测量；每步取8rank最大完整wall time。

|模式|完整更新中位数|整卡采样HBM峰值|
|---|---:|---:|
|legacy|4.469495s|68.932GiB|
|preserve|4.472070s|69.283GiB|

preserve相对legacy的步时变化（正数表示减少）：-0.058%。每点只有一个独立进程，20个更新不是20次独立重复。本轮只检查执行器修正是否影响性能，没有用同SO重跑最强fixed_home/anchor，也没有证明超过它们，更没有建立新planner收益。

## 下一步边界

当前没有实现新的联合planner。排布开关解耦未带来收益，停止扩展这一候选的重复或大矩阵。先检验反向优先级排序中重复扫描的公共开销；再结合causal诊断中的来源同步等待，判断是否需要“Native进入前完成旧服务”及“同一group反向wave合并后交付”两个小干预。它们也必须超过普通固定分组/公共累积等强对照，才继续新策略。不得仅凭减少原型开销或移动等待位置宣称论文创新。

重放单个冻结case可用replay_case.py，必须给出未使用的新标签。例如在本子项目根运行：`replay_case.py pilot --tag raw-preserve-replay-r2 --schedule preserve`。脚本保留全部正确性前置校验，使用公共准入锁且拒绝覆盖已有证据。开发型run_*_v1登记原始cohort，不覆盖重跑。

正确性回执见raw-gates-v1.json、raw-sparse-gates-v1.json、raw-model-gates-v1.json；正式筛查登记和每步数据见schedule-pilot-v1.json/CSV，图为figures/schedule-pilot-v1.png/svg/pdf。所有原始梯度张量留在对应独立远端项目，未上传报告仓库。
