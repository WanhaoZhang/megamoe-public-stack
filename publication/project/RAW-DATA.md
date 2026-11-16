# 原始数据路径索引

本地项目根目录：`D:/megamoe_train/planner-executor-ep8-lab/temporal-replica-study`。
完整回执及身份/数值结果在 `evidence/<tag>/`。每个运行保留 status、stdout、步骤、HBM、源身份和 actual SO 校验。
项目专属远端源快照和大张量保留在 SSH 别名 pod6-1-root 对应节点；不是其他任务的数据。

## 大梯度导出与未完整下载

`metadata-only` 将大 PT 留在远端，JSON 比较由远端 CPU 直接读取这些 PT。它不意味着张量已下载。
`frontier-gradient-easy` 有一次中断的本地 SCP：collection_exit=255，本地文件不宣称完整；远端原始张量仍是数值比较来源。

| tag | 执行退出 | 收集退出 | 本地PT数量 | 大张量专属远端目录 |
|---|---:|---:|---:|---|
| canonical-gradient-fixed_home-w4 | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/canonical-gradient-fixed_home-w4 |
| canonical-gradient-fixed_native-w2 | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/canonical-gradient-fixed_native-w2 |
| canonical-w2-adam3-easy_home | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/canonical-w2-adam3-easy_home |
| canonical-w2-adam3-easy_independent | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/canonical-w2-adam3-easy_independent |
| canonical-w2-adam3-fifo_native | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/canonical-w2-adam3-fifo_native |
| canonical-w2-adam3-fixed_native | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/canonical-w2-adam3-fixed_native |
| canonical-w2-adam3-legacy | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/canonical-w2-adam3-legacy |
| frontier-gradient-easy | 0 | 255 | 8 | 完整性以receipt为准 |
| frontier-gradient-legacy | 0 | 0 | 8 | 完整性以receipt为准 |
| frontier-gradient-v3-easy_home | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/frontier-gradient-v3-easy_home |
| frontier-gradient-v3-easy_independent | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/frontier-gradient-v3-easy_independent |
| frontier-gradient-v3-fifo_native | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/frontier-gradient-v3-fifo_native |
| frontier-gradient-v3-fixed_native | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/frontier-gradient-v3-fixed_native |
| gradient-fenced-fixed_home | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/gradient-fenced-fixed_home |
| gradient-fenced-fixed_native | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/gradient-fenced-fixed_native |
| gradient-fenced-legacy | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/gradient-fenced-legacy |
| gradient-w2-easy_home | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/gradient-w2-easy_home |
| gradient-w2-easy_independent | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/gradient-w2-easy_independent |
| gradient-w2-fifo_native | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/gradient-w2-fifo_native |
| gradient-w2-fixed_native | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/gradient-w2-fixed_native |
| gradient-w2-legacy | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/gradient-w2-legacy |
| lease-gradient-fixed_home-w4 | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/lease-gradient-fixed_home-w4 |
| lease-gradient-legacy-w4 | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/lease-gradient-legacy-w4 |
| no-probe-baseline-fifo_native-adam3 | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/no-probe-baseline-fifo_native-adam3 |
| no-probe-baseline-fixed_native-adam3 | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/no-probe-baseline-fixed_native-adam3 |
| observation-gate-legacy-w1 | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/observation-gate-legacy-w1 |
| observation-gate-offline-no-probe | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/observation-gate-offline-no-probe |
| observation-v1-native-w1 | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/observation-v1-native-w1 |
| observation-v1-native-w2 | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/observation-v1-native-w2 |
| observation-v1-offline-no-probe | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/observation-v1-offline-no-probe |
| observation-v1-online-easy-home | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/observation-v1-online-easy-home |
| ordered-shared-easy_home-w2 | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/ordered-shared-easy_home-w2 |
| ordered-shared-easy_independent-w2 | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/ordered-shared-easy_independent-w2 |
| ordered-shared-fifo_native-w2 | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/ordered-shared-fifo_native-w2 |
| ordered-shared-fixed_home-w4 | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/ordered-shared-fixed_home-w4 |
| ordered-shared-fixed_native-w2 | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/ordered-shared-fixed_native-w2 |
| pilot-clean-timing-random-easy_home-w2 | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/pilot-clean-timing-random-easy_home-w2 |
| pilot-clean-timing-random-legacy-w2 | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/pilot-clean-timing-random-legacy-w2 |
| prefix-diagnostic-backward-global | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/prefix-diagnostic-backward-global |
| prefix-diagnostic-deferred-same-order | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/prefix-diagnostic-deferred-same-order |
| prefix-diagnostic-hold-forward | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/prefix-diagnostic-hold-forward |
| prefix-diagnostic-prefix-device-sync | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/prefix-diagnostic-prefix-device-sync |
| prefix-diagnostic-repeat-control | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/prefix-diagnostic-repeat-control |
| prefix-diagnostic-stage-chain | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/prefix-diagnostic-stage-chain |
| prefix-v2-easy_home-adam3 | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/prefix-v2-easy_home-adam3 |
| prefix-v2-fifo_native-adam3 | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/prefix-v2-fifo_native-adam3 |
| prefix-v2-fixed_native-adam3 | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/prefix-v2-fixed_native-adam3 |
| resident-v3-confirm-fixed_home-r2 | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/resident-v3-confirm-fixed_home-r2 |
| resident-v3-confirm-native_resident-r2 | 1 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/resident-v3-confirm-native_resident-r2 |
| resident-v3-gate-easy_home | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/resident-v3-gate-easy_home |
| resident-v3-gate-easy_independent | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/resident-v3-gate-easy_independent |
| resident-v3-gate-fifo | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/resident-v3-gate-fifo |
| resident-v3-gate-fixed_home | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/resident-v3-gate-fixed_home |
| resident-v3-gate-fixed_native | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/resident-v3-gate-fixed_native |
| resident-v3-gate-native_resident | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/resident-v3-gate-native_resident |
| resident-v3-pilot-easy_home | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/resident-v3-pilot-easy_home |
| resident-v3-pilot-easy_independent | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/resident-v3-pilot-easy_independent |
| resident-v3-pilot-fifo | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/resident-v3-pilot-fifo |
| resident-v3-pilot-fixed_home | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/resident-v3-pilot-fixed_home |
| resident-v3-pilot-fixed_native | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/resident-v3-pilot-fixed_native |
| resident-v3-pilot-native_original | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/resident-v3-pilot-native_original |
| resident-v3-pilot-native_resident | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/resident-v3-pilot-native_resident |
| resident-v3-primary-adam3 | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/resident-v3-primary-adam3 |
| runset-math-v1 | 1 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/runset-math-v1 |
| runset-math-v2 | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/runset-math-v2 |
| screen-v1-vio10-alternating-easy_home-r1 | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/screen-v1-vio10-alternating-easy_home-r1 |
| screen-v1-vio10-alternating-easy_independent-r1 | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/screen-v1-vio10-alternating-easy_independent-r1 |
| screen-v1-vio10-alternating-fifo_native-r1 | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/screen-v1-vio10-alternating-fifo_native-r1 |
| screen-v1-vio10-alternating-fixed_native-r1 | 1 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/screen-v1-vio10-alternating-fixed_native-r1 |
| screen-v1-vio10-alternating-fixed_native-r1-retry1 | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/screen-v1-vio10-alternating-fixed_native-r1-retry1 |
| screen-v1-vio10-alternating-legacy_w1-r1 | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/screen-v1-vio10-alternating-legacy_w1-r1 |
| screen-v1-vio10-alternating-legacy_w2-r1 | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/screen-v1-vio10-alternating-legacy_w2-r1 |
| screen-v1-vio10-random-easy_home-r1 | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/screen-v1-vio10-random-easy_home-r1 |
| screen-v1-vio10-random-easy_independent-r1 | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/screen-v1-vio10-random-easy_independent-r1 |
| screen-v1-vio10-random-fifo_native-r1 | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/screen-v1-vio10-random-fifo_native-r1 |
| screen-v1-vio10-random-fixed_native-r1 | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/screen-v1-vio10-random-fixed_native-r1 |
| screen-v1-vio10-random-legacy_w1-r1 | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/screen-v1-vio10-random-legacy_w1-r1 |
| screen-v1-vio10-random-legacy_w2-r1 | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/screen-v1-vio10-random-legacy_w2-r1 |
| screen-v1-vio10-same-easy_home-r1 | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/screen-v1-vio10-same-easy_home-r1 |
| screen-v1-vio10-same-easy_independent-r1 | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/screen-v1-vio10-same-easy_independent-r1 |
| screen-v1-vio10-same-fifo_native-r1 | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/screen-v1-vio10-same-fifo_native-r1 |
| screen-v1-vio10-same-fixed_native-r1 | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/screen-v1-vio10-same-fixed_native-r1 |
| screen-v1-vio10-same-legacy_w1-r1 | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/screen-v1-vio10-same-legacy_w1-r1 |
| screen-v1-vio10-same-legacy_w2-r1 | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/screen-v1-vio10-same-legacy_w2-r1 |
| temporal-clock-links | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/temporal-clock-links |
| timer-v2-matched-legacy-primary | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/timer-v2-matched-legacy-primary |
| timer-v2-matched-primary | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/timer-v2-matched-primary |
| timer-v2-matched-small | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/timer-v2-matched-small |
| timer-v2-ordered-small | 1 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/timer-v2-ordered-small |
| upstream-maxutil-gate-fixed_home | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/upstream-maxutil-gate-fixed_home |
| upstream-maxutil-gate-upstream_home | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/upstream-maxutil-gate-upstream_home |
| upstream-maxutil-gate-upstream_independent | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/upstream-maxutil-gate-upstream_independent |
| upstream-maxutil-primary-same-adam3 | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/upstream-maxutil-primary-same-adam3 |
| upstream-maxutil-screen-alternating-fixed_home-r1 | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/upstream-maxutil-screen-alternating-fixed_home-r1 |
| upstream-maxutil-screen-alternating-native_resident-r1 | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/upstream-maxutil-screen-alternating-native_resident-r1 |
| upstream-maxutil-screen-alternating-upstream_home-r1 | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/upstream-maxutil-screen-alternating-upstream_home-r1 |
| upstream-maxutil-screen-random-fixed_home-r1 | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/upstream-maxutil-screen-random-fixed_home-r1 |
| upstream-maxutil-screen-random-native_resident-r1 | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/upstream-maxutil-screen-random-native_resident-r1 |
| upstream-maxutil-screen-random-upstream_home-r1 | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/upstream-maxutil-screen-random-upstream_home-r1 |
| upstream-maxutil-screen-same-fixed_home-r1 | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/upstream-maxutil-screen-same-fixed_home-r1 |
| upstream-maxutil-screen-same-native_resident-r1 | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/upstream-maxutil-screen-same-native_resident-r1 |
| upstream-maxutil-screen-same-upstream_home-r1 | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/upstream-maxutil-screen-same-upstream_home-r1 |
| upstream-selector-abi4-gate | 0 | 0 | 0 | /home/z00956592/megaflux/temporal-replica-frontier-20261115/stack/evidence/upstream-selector-abi4-gate |

## 源码与二进制

`evidence/source-snapshots/<tag>.tar.gz` 为运行时冻结源码；`*-source-reference.json` 指向同内容已收集归档，不是缺失源码。
`*-runtime.json` 保存21项冻结运行库，`*-temporal-runtime.json` 保存新AscendC选择器SHA。
`implementation/frozen-build-export.tar.gz` 现已保存真实SO及构建元数据：24个路径别名、22份去重后的SO字节，包含正常worker与独立TimerV2诊断worker；全部SHA由frozen-build-export.json复核，不能将两个worker混用。
`implementation/source-base.tar.gz` 是冻结基线的干净完整导出。先解压它，再覆盖source-changes.tar.gz，可重建本轮源码；SOURCE-MANIFEST.json记录每个覆盖文件SHA。
`implementation/source-changes.tar.gz` 含所有修改及新文件；不应只应用tracked diff而遗漏新接口。原始source.tar含早期overlay，仅作为历史工件保留。
`implementation/replay-tools.tar.gz` 含远端运行、回收、诊断和报告脚本；基础容器镜像与本地依赖不上传报告仓库。
完整机器可读索引：`implementation/raw-data-index.json`。
