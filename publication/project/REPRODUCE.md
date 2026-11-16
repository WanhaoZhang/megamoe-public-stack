# 复查与重放

本交付只覆盖独立priority-cache公共执行器干预及同一底座的有界生命周期复查，不是新联合planner。旧工程、旧SO和原33点结果不改动。

## 现有冻结节点上的新进程重放

使用项目Python：D:/megamoe_train/planner-executor-ep8-lab/.venv/Scripts/python.exe。

```
python replay_case.py --case cached_keys --tag priority-replay-cached-unique-01
```

case可选reference、cached_keys、b0、b2、b3、anchor、fixed_home。每次必须使用新tag；远端controller以exist_ok=False拒绝覆盖旧运行。该脚本不改运行库、不部署文件，只调用当前冻结的独立节点目录/容器，复用全节点准入锁、实际设备占用检查及源码/库指纹。所有新进程仍25更新、5预热＋20测量、八rank最大完整更新wall，Timer/profiler/主机诊断均关闭。

remote.py记录实际SSH别名、独立远端和容器。若节点上该冻结环境已清除，不能只复制一个SO就当作重放；应在新的独立目录/容器按依赖版本重建，再重新完整数值验收。不要用remote.deploy覆盖其他正在运行的研究源码，也不要修改旧完成数据。

## 源码与构建

SOURCE-INDEX.json给出Git交付head、原始base、source.zip、source.diff、两份实际SO及构建指纹。source.zip解压到新的独立目录后保留stack/和同层辅助脚本布局；stack/research/step_lifetime/priority_cache是辅助脚本的版本化审阅副本，实际脚本按交付根目录布局执行。

implementation/build-priority-worker.sh与build_worker.py记录真实CANN9.2.0b070、Python3.12.13、SDK/SHMEM、CAP64、restore2、pipeline flags0和TimerOFF。外部大型环境不放进源码ZIP，冻结库清单/实际加载库保存在configs与evidence。reference-worker.so为原已冻结72653a，worker.so为新e2868f；对照构建输入证明只有backward排序头文件变化，不能拿新磁盘源文件冒充旧SO的编译来源。

## 数据与验收边界

summary.json、runs.csv、measured-updates.csv对应三配对和两个控制；common-lifetime-summary.json与CSV对应新增三点生命周期复查。每个evidence/tag保存effective-config、身份、实际库、源码freeze、更新与步骤日志、HBM、退出/独占状态；大梯度PT保留在各receipt.large_tensors_remote指向的同任务远端。计时运行不额外导出全量计划张量，不能声称存在不存在的逐步实际计划快照。

独立小形状输出/梯度及三更新资格、384计划顺序oracle、完整main身份审计是分开的证据。只用实际完成且通过门禁的运行画图；步内20样本不当独立重复；本轮还没有自然路由验证、新策略稳定收益或论文创新性结论。
