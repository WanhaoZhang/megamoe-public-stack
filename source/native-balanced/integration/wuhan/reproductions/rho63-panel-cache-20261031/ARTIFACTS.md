# 原始证据与环境索引

原项目：`D:/megamoe_train/rho63-pipeline-20261031`。

授权实验节点中的同名项目：`/home/z00956592/megaflux/rho63-pipeline-20261031`。`delivery-a1`已镜像最终源码快照、报告及小型分析；实际SO/输入/环境和原始日志仍在原项目，不因本次入仓移动或覆盖。两节点本任务容器已停止，重跑须使用新独立目录/容器及新准入。

| 内容 | 项目相对路径 |
|---|---|
| 完整报告/完成审计/全部产物SHA | `REPORT.md`, `COMPLETION-AUDIT.md`, `ARTIFACT-MANIFEST.json` |
| 历史控制与候选源码 | `source` (f6daec8), `panel-cache-source` (9d036d2) |
| 原始真实输入 | 节点 `inputs/rho{6,3}/rank{0..7}.{pt,json}`；每个PT约499MB，完整SHA见本目录evidence/FINGERPRINTS.json |
| baseline独立普通核参考/OFF/ON三次 | `downloaded-node84/evidence/baseline-node84-rho63-a3`；节点同路径去掉downloaded-node84前缀 |
| 候选生命周期、五输出、NPY与SO | `downloaded-panel-cache-capture-a1`；节点`evidence/panel-cache-node84-a1`和`selected-variants/rho63-panel-cache-a2-{off,on}` |
| 正式16组TimerOFF原始八rank样本 | `downloaded-panel-cache-final-a1/evidence/panel-cache-perf-node84-a1` |
| 原始前后向trace JSON | `analysis/rho{6,3}-{base-a3,panel-cache-a1}-{no-waits,with-waits}/all_8_rank_{forward,backward}_trace.json` |
| 被拒候选全部证据 | `RETIRE-FILL-RESULT.md`, `retire-fill-*`, `candidate-source`, `downloaded-node84/evidence/candidate-node84-a1` |
| 输入迁移/环境/构建记录 | `resources/`, `downloaded-node84/{preparation*,baseline-build-verification.json,rho*-input-receipt.json}`, `downloaded-panel-cache-capture-a1/panel-cache-qualification.json` |
| 实际SDK、运行依赖 | 节点`dependencies/`, `python-runtime/`；CANN只读挂载与镜像ID见preparation及FINGERPRINTS |
| 最终清理 | `own-node84-container-stopped.json`, `own-node73-container-final-audit.json`, 节点`evidence/final-cleanup-node84-a1` |

构建镜像为`sha256:2754601232edf86a4b7dc6e2aca87cf86648e14491e173458ebbf3d5cbd6d2f8`，CANN9.2，Ascend950；实际PyTorch/NPU/SHMEM字节和输入不能仅凭版本字符串代替核验。候选OFF worker SHA为`2d52c337a7e14bf931e687779eb389a4595bb4c197543d3fb436e290b08e7c39`，ON为`e6ca93b98db6ffb0b7480cda9f68bc26ae1bcad4b6162ed508e611d6ac92d650`。

本Git目录只含约1MB内的小型报告/CSV/计时数组/指纹/图和脚本；不含SO、PT、NPY、完整大JSON、依赖安装包、访问凭据、SSH helper或机器密码。没有原始capture时只能另做合成实验，不能将其当成本报告的复现。
