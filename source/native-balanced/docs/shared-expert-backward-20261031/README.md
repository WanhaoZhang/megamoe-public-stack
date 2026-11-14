# 共享专家反向融合设计（2026-10-31）

设计基线：[BACKWARD-SHARED-EXPERT-DESIGN.md](BACKWARD-SHARED-EXPERT-DESIGN.md)。设计基线保持原文，当前实现与实测状态以以下两份文档为准。

- [S0 实施说明、接口、构建及回退](IMPLEMENTATION.md)
- [NPU 验证报告、矩阵、指纹及原始数据索引](NPU-VALIDATION.md)
- [正确性测试与复现命令](../../integration/wuhan/SHARED-EXPERT-BACKWARD-TESTS.md)

- [源码版本、哈希和行号证据](SOURCE-EVIDENCE.json)
- [共享临时区与现有路由工作区预算](MEMORY-BUDGET.json)
- [同事的前向与联合设计](../MegaMoe共享专家与反向重派发融合设计-20261031.md)

主文档中的 D:/ 本地路径是原核查环境索引；源码按证据中固定的仓库、SHA和相对路径查阅。
当前已实现同一 native launch 中的 shared S0 串行前缀和原 routed backward，并在参考 shared forward 的小形状矩阵中通过 NPU 正确性验证。S1/P0 重叠仍未实现；生产前向保存契约联调、整网、原 rho 大形状、完整 B/FB 性能及 profiling 尚未验收。功能默认关闭。

本次仅更新kernel仓实验分支。框架子仓和wuhan根gitlink不随文档发布更新。
