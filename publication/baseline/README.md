# MegaMoE Public Stack

新公开训练栈的独立源码仓库。Native、FSDPTurbo、MindSpeed-LLM均为直接入库的完整源码快照，不通过gitlink依赖旧Wuhan框架仓库。

[GitHub](https://github.com/WanhaoZhang/megamoe-public-stack) · [GitCode](https://gitcode.com/zhangwanhao/megamoe-public-stack)

```bash
git clone https://github.com/WanhaoZhang/megamoe-public-stack.git
# 或：git clone https://gitcode.com/zhangwanhao/megamoe-public-stack.git
```

三个主体组件已包含本次实验适配，使用本仓库源码时不要重复应用`patches/`中的历史补丁。导出源码没有保留嵌套Git仓库，组件原始提交、文件内容及执行权限均可通过清单校对。

## 代码布局

|路径|内容|固定版本|
|---|---|---|
|`source/native-balanced`|普通MegaMoE执行器、64槽、统一Planner|`3ec836a8`|
|`source/FSDPTurbo`|公开FSDPTurbo完整源码及已测适配|`08d5caf7`，公开基线`9c443c6`|
|`source/MindSpeed-LLM`|公开MindSpeed-LLM完整源码及已测适配|`fbc1ed7d`，公开基线`dd5ee032`|
|`source/native-balanced/3rdparty`|固定版本的CATLASS、CatCCOS、SHMEM源码|见`COMPONENTS.json`|
|`tools`|整网训练、路由生成、在线Planner、校准及校验脚本|冻结实验入口|
|`configs`|模型与四份冻结成本模型|主形状L2/EP8/E256/Top8/H7168/F2048/S16384|
|`reports`、`figures`、`analysis`|实验说明、派生结果、PNG/SVG/PDF/CSV|仅实际完成数据|

所有导出的组件文件及SHA见`SOURCE-MANIFEST.json`。编译器/驱动/CANN、安装好的运行时库和私有镜像不入Git。
上游组件中可选的googletest、TLA/AscendNPU-IR以及CatCCOS独立开发用的子依赖未递归展开，其固定提交保留在`COMPONENTS.json`；它们与本仓库三个主体训练组件的完整源码分开说明。

## 入口与运行环境

主要入口为`tools/train.py`、`tools/launch.sh`及`configs/`，训练使用普通MegaMoE，模型shared expert仍由框架正常计算。
实际测量使用CANN9.2、Python3.12.13、记录的torch/torch_npu和SHMEM运行时。Host、planner、worker必须按相同64槽能力重建，提交源码不能替代匹配二进制。
冻结脚本保留Linux路径`/home/z00956592/megaflux/planner-executor-ep8-lab`及已记录的依赖布局；可将本仓库目录映射到该路径，准备匹配环境后按`REPRODUCE.md`操作。
源码中有历史三后端资格辅助入口；本轮正式Planner对照统一采用`--backend balanced`，不把MoonEP规划策略冒称完整MoonEP通信后端。

## 已知实验结论

实验已按用户要求收尾：完成108次主矩阵和12次第四轮。step_time主矩阵900次规划全部精确回退Native，完整步时比Native高0.34%–0.85%，没有证明新增搜索收益。
第四轮MaxVio5在第22步的全局梯度范数轨迹差异2.415%仍未解释，同权重25步诊断未完成；后续独立银行与消融已取消。
`ACCEPTANCE.json`保留`complete=false`，`CLOSEOUT-AUDIT.json`仅核对已保留数据及收尾。工程保留停队标记，不自动恢复设备实验。

原始日志、路由二进制、Timer缓冲、SO、SDK和环境镜像保存在原项目`D:/megamoe_train/planner-executor-ep8-lab`。本仓库不发布这些大体积运行产物。

[复现说明](REPRODUCE.md) · [收尾报告](CLOSEOUT.md) · [实验结果](reports/REPORT.md) · [架构](reports/ARCHITECTURE.md) · [许可证](LICENSES.md)
