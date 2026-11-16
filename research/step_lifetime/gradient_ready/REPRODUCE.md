# 本局部的复查入口

长期goal保持暂停，本交付不包含后台继续运行的任务。下面是供以后明确授权重放时使用的脚本。

现有冻结节点可运行 `python replay_case.py --scope call_stream --tag new-unique-tag`，另一方式选bank_write。每次必须新tag；controller拒绝覆盖，使用本任务独立容器/安装目录和全节点准入锁、实际设备占用/双栈端口检查。默认条件固定为主报告中的L4/EP8/GA4/W2、MaxVio10/same、25更新（5预热＋20测量），不自动扩重复或矩阵。

使用项目Python D:/megamoe_train/planner-executor-ep8-lab/.venv/Scripts/python.exe。remote.py含实际SSH别名及本任务目录。源码ZIP按stack/加同层辅助脚本布局解压；stack/research/step_lifetime/gradient_ready是脚本的版本化审阅副本，不要从该嵌套位置直接运行假定根目录布局的驱动。

CPU及设备数学入口在stack/tools/test_gradient_bank_ready.py、test_lifetime_delivery.py、test_gradient_ready_capacity.py和gate_lifetime_math.py。run_cpu_gates.py/deploy_audit_patch.py记录本次分阶段部署历史，含一次性旧SHA前置检查，不能对已经部署的最终树反复套用。要在新节点复查，应先按冻结manifest重建完整环境，再直接运行相应测试；不能仅复制单SO就宣称复现。

本轮没有新编译：两方式均复用priority-cache阶段e2868fd2的TimerOFF ABI2 SO。implementation/worker.so和worker-build.json保留对应二进制与原编译记录；SOURCE-INDEX、每run实际libraries记录和源码快照为审计入口。没有修改旧任务加载的SO。

正式配对的20条rank-max原始步时见measured-updates.csv，进程中位见runs.csv。数值门禁及单轮状态见study-v1.json，身份/流量对照见diagnostics/pair-identity-audit.json。小模型完整梯度PT仍在每个receipt.large_tensors_remote指定的独立远端，当前本地只收集日志、元数据和比较结果，不将远端张量冒称已本地收集。

本次主形状不是MoonEP官方九层/H20实验；它是此前共同裁剪的Ascend研究条件。单进程一轮配对不能证明稳定加速或新planner优越性。
