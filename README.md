# 跨microbatch任务组合与EasyBalance适配

本分支是独立研究源码交付，包含普通MegaMoE算子、FSDPTurbo和MindSpeed-LLM完整源码。研究保持暂停；发布源码不恢复设备队列，也不表示未完成的数值、性能或训练质量资格通过。

- 分支：research/temporal-replica
- 本次打包前的研究提交：44b4e0d2cb46933c22fad1922ff24d603fd5d2ac
- 模型条件：L4/EP8/E256/Top8/H4096/F1024/S16384/GA4/W2，V4 attention/mHC、框架shared专家。
- 精度与训练：FP32持久参数/梯度/Adam状态，BF16专家，固定owner，无CPU offload。
- 结论边界：保留作者选择规则、固定分组和Native控制；首轮筛查未证明动态选择胜过固定分组。

## 源码入口

|目录|职责|
|---|---|
|[source/native-balanced](source/native-balanced)|算子、Planner、Host、AscendC流水及构建脚本|
|[source/FSDPTurbo](source/FSDPTurbo)|专家并行、参数分片与dispatcher接入|
|[source/MindSpeed-LLM](source/MindSpeed-LLM)|模型与FSDP2训练引擎|
|[tools](tools)|该分支实际训练、资格和控制脚本|
|[configs](configs)|模型、运行时与资源配置|
|[publication/project](publication/project)|研究项目说明和外层复现/构建辅助脚本|
|[SOURCE-MANIFEST.json](SOURCE-MANIFEST.json)|当前导出的完整source文件SHA256与执行位|
|[publication/SOURCE-VERIFY.json](publication/SOURCE-VERIFY.json)|源码一致性与发布范围核验|

## 使用与验证边界

三个主体源码直接入库，主源码树没有gitlink或嵌套Git仓库。Native所需CATLASS、CatCCOS、SHMEM源码沿用冻结公开栈；可选第三方开发依赖的固定提交见COMPONENTS.json，许可证按各组件保留。

以本分支为PROJECT/stack，外层脚本可从publication/project复制到PROJECT；算子和框架仍使用stack/source布局。原脚本中固定的Linux目录、SSH别名、CANN和SHMEM路径必须在新的隔离环境中显式映射或调整。单microbatch工具使用PROJECT本身作为源码根，按其README和PLAN检查路径。

源码包含已验证控制与尚未通过NPU资格的探针，应按对应报告和配置逐项选择，不因代码存在而默认开启。重建Host、Planner和worker时必须匹配槽位、ABI、编译配方和实际SO能力。本次仅检查源码/语法/既有CPU控制；没有重建NPU产物或补跑实验。

原始日志、路由张量、梯度PT、SDK、SO和本地环境留原项目及其专用远端；本仓库只保存源码、配置、必要说明和小型派生图表。原main的REPRODUCE/STATUS/SOURCE-MANIFEST作为历史信息保存于publication/baseline。

## 独立分支索引

|方向与控制|分支|链接|
|---|---|---|
|单microbatch前反向与owner服务探索|research/single-mb-fbw-planner|[GitHub](https://github.com/WanhaoZhang/megamoe-public-stack/tree/research/single-mb-fbw-planner) · [GitCode](https://gitcode.com/zhangwanhao/megamoe-public-stack/tree/research/single-mb-fbw-planner)|
|跨microbatch任务组合与EasyBalance适配|research/temporal-replica|[GitHub](https://github.com/WanhaoZhang/megamoe-public-stack/tree/research/temporal-replica) · [GitCode](https://gitcode.com/zhangwanhao/megamoe-public-stack/tree/research/temporal-replica)|
|专家副本与梯度交付生命周期|research/step-lifetime|[GitHub](https://github.com/WanhaoZhang/megamoe-public-stack/tree/research/step-lifetime) · [GitCode](https://gitcode.com/zhangwanhao/megamoe-public-stack/tree/research/step-lifetime)|
|生命周期路径因果诊断|research/step-lifetime-causal|[GitHub](https://github.com/WanhaoZhang/megamoe-public-stack/tree/research/step-lifetime-causal) · [GitCode](https://gitcode.com/zhangwanhao/megamoe-public-stack/tree/research/step-lifetime-causal)|
|原始梯度交付路径控制|research/step-lifetime-raw-control|[GitHub](https://github.com/WanhaoZhang/megamoe-public-stack/tree/research/step-lifetime-raw-control) · [GitCode](https://gitcode.com/zhangwanhao/megamoe-public-stack/tree/research/step-lifetime-raw-control)|
|梯度排序键缓存及共同底座复查|research/step-lifetime-priority-cache|[GitHub](https://github.com/WanhaoZhang/megamoe-public-stack/tree/research/step-lifetime-priority-cache) · [GitCode](https://gitcode.com/zhangwanhao/megamoe-public-stack/tree/research/step-lifetime-priority-cache)|
|主机入口与返回边界诊断|research/step-lifetime-host-boundary|[GitHub](https://github.com/WanhaoZhang/megamoe-public-stack/tree/research/step-lifetime-host-boundary) · [GitCode](https://gitcode.com/zhangwanhao/megamoe-public-stack/tree/research/step-lifetime-host-boundary)|
|梯度bank最后写入就绪依赖控制|research/step-lifetime-grad-ready|[GitHub](https://github.com/WanhaoZhang/megamoe-public-stack/tree/research/step-lifetime-grad-ready) · [GitCode](https://gitcode.com/zhangwanhao/megamoe-public-stack/tree/research/step-lifetime-grad-ready)|
