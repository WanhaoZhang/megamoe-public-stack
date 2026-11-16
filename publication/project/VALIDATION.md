# 验证回执

A组六种方式均完成八rank、三次Adam、输入/路由/初始参数身份检查；五组相对serial_home逐参数FP32梯度比较均通过。主形状门禁峰值65.49GiB。

生命周期CPU契约5项通过；两rankGloo参考覆盖b0-b3、三次Adam和空owner零梯度，结果通过。

独立FFN强制合法复制门禁：八rank×四种交付方式×W1/W2，共64条记录通过。实际存在缓存命中和远端梯度交付；不是仅走本地专家。输出相对L2最大0.00030363924452103674；dx/dp/dW最大0.0027591867838054895。

详见diagnostics/independent-lifetime-math.json及evidence/lifetime-independent-ffn-v3/rank*.json。正式训练计时前，transport要求该证明的核心源码指纹仍与当前运行代码一致。

B完整模型逐参数比较、主形状与性能筛查的最新状态以REPORT.md和lifetime-b2.json为准。非法缓存集体拒绝、匹配prefix的原Timer验证、独立银行和稳定收益尚待完成。

2026-11-16：原worker与ABI1候选默认锚点的相同输入/参数/路由逐参数梯度比较通过，见diagnostics/lifetime-anchor-vs-original-comparison.json。

原AscendTimerV2 prefix资格：v1 Native小形状数值通过但所有copies为空，不能证明副本覆盖。v2采用合法MoonEP计划激励复制，原校验发现重复坐标记录并拒绝，保留失败。定位到两个FFN波次连续反向时，采集缓冲仅在前向清零；在独立诊断副本的每次backward descriptor边界清零本次坐标区、刷新epoch，ready/梯度缓冲不变。v3八rank各72次前向/重放/反向采集，均有非零坐标记录且无重复/越界；三次Adam和对TimerOFF Native的逐参数梯度比较通过。范围仍是L2/H512/F512/S256，MoonEP复制仅作资格激励，未计入性能；主形状及raw B服务关键路径尚未验收。

完成ticket批次的重复identity拒绝在CPU回归中暴露缺口，diagnostics/duplicate-completion-before.json保留失败。修复安排在冻结B矩阵全部结束后，由finish_hardening.py更新，并重新执行CPU契约和独立NPU强制副本门禁；实时状态见diagnostics/hardening.json。原计时核心的v3证书保留，最终代码不能冒用旧源码指纹。

最终回执：A18/18、B15/15首轮进程全部完成。主形状random三次更新中，anchor/B0/B1/B2/B3的copies、source_quota、logical_ids、physical_ids及logical_counts，在八rank各48次原始编译调用逐项哈希一致，实际副本上限8。见diagnostics/lifetime-plan-equality.json。五方式均受同一资源/源文件/设备独占门禁约束。

冻结矩阵结束后增加重复completion批次的原子拒绝，六项CPU契约与before/after回归通过；随后lifetime-independent-ffn-v4重新完成64条有真实缓存/远端梯度的独立NPU对照，结果通过且当前本地核心源码匹配新证书。旧计时点按v3历史证书匹配，不冒用v4。新版不新增性能结论。详见diagnostics/hardening.json与independent-lifetime-math.json。
