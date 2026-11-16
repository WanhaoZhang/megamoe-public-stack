# Published research source index

Each linked branch contains the complete operator, FSDPTurbo and MindSpeed-LLM. The two research lines share the public framework baseline but retain separate source histories and diagnostic controls. Main and ordinary/shared collaboration branches are not advanced by this publication.

|Research stage|Branch|
|---|---|
|single|[GitHub](https://github.com/WanhaoZhang/megamoe-public-stack/tree/research/single-mb-fbw-planner) / [GitCode](https://gitcode.com/zhangwanhao/megamoe-public-stack/tree/research/single-mb-fbw-planner)|
|temporal|[GitHub](https://github.com/WanhaoZhang/megamoe-public-stack/tree/research/temporal-replica) / [GitCode](https://gitcode.com/zhangwanhao/megamoe-public-stack/tree/research/temporal-replica)|
|lifetime|[GitHub](https://github.com/WanhaoZhang/megamoe-public-stack/tree/research/step-lifetime) / [GitCode](https://gitcode.com/zhangwanhao/megamoe-public-stack/tree/research/step-lifetime)|
|causal|[GitHub](https://github.com/WanhaoZhang/megamoe-public-stack/tree/research/step-lifetime-causal) / [GitCode](https://gitcode.com/zhangwanhao/megamoe-public-stack/tree/research/step-lifetime-causal)|
|raw|[GitHub](https://github.com/WanhaoZhang/megamoe-public-stack/tree/research/step-lifetime-raw-control) / [GitCode](https://gitcode.com/zhangwanhao/megamoe-public-stack/tree/research/step-lifetime-raw-control)|
|priority|[GitHub](https://github.com/WanhaoZhang/megamoe-public-stack/tree/research/step-lifetime-priority-cache) / [GitCode](https://gitcode.com/zhangwanhao/megamoe-public-stack/tree/research/step-lifetime-priority-cache)|
|host|[GitHub](https://github.com/WanhaoZhang/megamoe-public-stack/tree/research/step-lifetime-host-boundary) / [GitCode](https://gitcode.com/zhangwanhao/megamoe-public-stack/tree/research/step-lifetime-host-boundary)|
|ready|[GitHub](https://github.com/WanhaoZhang/megamoe-public-stack/tree/research/step-lifetime-grad-ready) / [GitCode](https://gitcode.com/zhangwanhao/megamoe-public-stack/tree/research/step-lifetime-grad-ready)|

The single-microbatch component repositories also retain the research/single-mb-fbw-planner branch:

- [Native on GitHub](https://github.com/WanhaoZhang/megaflux/tree/research/single-mb-fbw-planner) / [GitCode](https://gitcode.com/zhangwanhao/ascend_deepep_hkq_wuhan/tree/research/single-mb-fbw-planner)
- [FSDPTurbo on GitHub](https://github.com/WanhaoZhang/FSDPTurbo_wuhan/tree/research/single-mb-fbw-planner) / [GitCode](https://gitcode.com/zhangwanhao/FSDPTurbo_wuhan/tree/research/single-mb-fbw-planner)
- [MindSpeed-LLM on GitHub](https://github.com/WanhaoZhang/MindSpeed-LLM_wuhan/tree/research/single-mb-fbw-planner) / [GitCode](https://gitcode.com/zhangwanhao/MindSpeed-LLM_wuhan/tree/research/single-mb-fbw-planner)

Runtime SOs, SDKs, tensors and original logs stay in their original projects. Optional upstream dependencies are pinned in COMPONENTS.json. Qualification and source publication are separate: no new NPU runs, no newly qualified online planner, and no training-quality acceptance are claimed.
