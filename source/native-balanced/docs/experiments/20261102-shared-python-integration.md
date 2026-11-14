# 共享专家分支同步Python优化和固定场景

2026-11-02，按用户要求先fetch并ff-only同步四仓 `megamoe-balanced-shared-expert`，再将 `megamoe-balanced` 的Python优化、实验报告、场景工具合入。保留同事及原实验提交历史，不force、不替换运行库。

## 同步前核对

| 仓库 | 本次开始时共享分支远端HEAD | 状态 |
|---|---|---|
| ascend_deepep_hkq_wuhan | `261a08690c269c4c9de18871e98ff3677f7b42d8` | 同事共享前向/V2/V3/keep-recompute/内核X重派发已在分支中，无更新待拉取 |
| FSDPTurbo_wuhan | `bd52ea5e8ad976f7dfc1f3833dfe15abea473ab8` | 同事共享V2框架接线已拉取，无新增源码修改 |
| MindSpeed-LLM_wuhan | `7dc5055abe1c70042714be3bce51d8165733865d` | 同上 |
| wuhan | `b10827b1e9a4bcc7a23d5a2efd6c3f3d5856f81d` | 根gitlink原来仍旧，最终需固定三个最新共享子仓 |

随后fetch算子 `megamoe-balanced`，发现已从刚发布的 `d2c10d8` 更新至
`fac3e973e9a9eab7116a1ae8e64f3d6a3bc7649a`，包含同事
`164376d33849f865ec3aef63a71a577cda672e17`（FSDP2热路径去host fence）。
本次合入该最新头，包含 `b914741`（P1 cache）、`847a1f9`（owner2:1）、
`ea98005`（最终三组报告）及 `9b6c20d`/`d2c10d8` 的场景、说明与构建指纹。

## 合并处理

三处文本冲突逐项保留双方含义：

1. CMake：同时保留P1隔离配置检查，以及同事shared backward source、S9 SwiGLU解析和tile-ready控制；P1/owner编译定义不覆盖shared ABI定义。
2. 构建清单：同时记录shared forward与owner early ratio；保留shared V2/V3 ABI、redispatch诊断延迟、S9有效SwiGLU模式、cmake源码输入等字段。
3. bridge初始化：保留shared Timer/debug属性，同时保留同事 `enable_diagnostics=False` 下不创建capture/audit的控制。

另修正一处跨分支语义不兼容：新热路径默认 `stats=None`，共享forward/backward原本仍直接递增stats。共享forward、dispatch释放字节、backward restore、redispatch次数和backward完成计数统一在 `self.stats is not None` 时写入。诊断关时不报错，诊断开时继续记录；不改变native调用或Timer采集。

同事的routed异步路径保留：发布为两次device copy、保存设备header、backward更新新epoch、调用release_pending释放lease。shared路径仍保留其已有finish/ABI2或ABI3调用；本次不改shared流水。CPU MathPool测试夹具改为返回int32 header，匹配新的真实pool接口。

## 构造和构建

[固定Python场景](../../integration/wuhan/scenarios/python-olmoe-high-skew/README.md) 及
[历史性能](../../integration/wuhan/scenarios/python-olmoe-high-skew/PERFORMANCE.md) 在本分支也可用。
源κ3.0071、目标κ2.4817，world8/两个EP4组/E8/Top2/M8192/H7168/F3072；原始来源与映射标签不改变。

P1 cache默认仍0、owner ratio默认仍1；框架shared默认framework、shared构建开关默认不改变。
明确开启P1/owner候选时使用DXcache1/P1cache1/owner ratio2及原隔离配置，另按同事文档选择shared forward/backward/保存模式。必须新构建匹配binding、worker、planner/validator和ABI2/ABI3库，不能复用balanced-only的旧SO冒充shared组合。

代码合入不自动改变框架是否走V2/V3；框架的shared fused接线仍以框架仓当前实现为准。此routed固定case没有shared权重/宽度，独立复现它不等于验收shared融合。

## 核验范围与历史结果

本次执行7组、共39项不依赖设备的CPU检查：固定场景路由/指纹及CLI保护、原Timer契约、共享benchmark/trace和默认validator等；其中新增真实pool控制流的stub测试覆盖stats开/关×keep/recompute×saved/redispatch八组合，以及设备header clone/epoch回放。stub测试只验证Python调度/委托，不声称真实张量数值或通信正确性。

另检查冲突标记、Python/JSON/shell语法、相对链接、P1/owner原实现及同事shared文件保留情况；25个shared独有native实现文件保持同事版本，P1/owner三个实现头保持balanced版本。**没有新CANN构建、真实Torch autograd/NPU或整网验收**。本地无Torch的测试限制如实保留，MathPool夹具改动未在真实Torch环境重跑。

balanced-only的 [build-reference.json](../../integration/wuhan/scenarios/python-olmoe-high-skew/build-reference.json)
仍是原始历史构建参考。shared分支的CMake、bridge和shared kernel与它不同，**不能继续宣称组合后的全部155编译输入与那份旧SO一致**；本次差异审计见技术项目
`D:/megamoe_train/shared-python-integration-20261102`。历史P1四组7.093%/4.681%和owner三组4.306%/2.838%也不自动成为新shared组合性能；原owner第四组未完成的边界保持。

Native先推本共享分支，再在根仓同分支更新gitlink。两个框架子仓本次仅拉取同事已有提交，不制造新框架提交；不把大PT、SO、trace、模型或依赖环境加入Git。
