# 验证回执

A组六种方式均完成八rank、三次Adam、输入/路由/初始参数身份检查；五组相对serial_home逐参数FP32梯度比较均通过。主形状门禁峰值65.49GiB。

生命周期CPU契约5项通过；两rankGloo参考覆盖b0-b3、三次Adam和空owner零梯度，结果通过。

独立FFN强制合法复制门禁：八rank×四种交付方式×W1/W2，共64条记录通过。实际存在缓存命中和远端梯度交付；不是仅走本地专家。输出相对L2最大0.00030363924452103674；dx/dp/dW最大0.0027591867838054895。

详见diagnostics/independent-lifetime-math.json及evidence/lifetime-independent-ffn-v3/rank*.json。正式训练计时前，transport要求该证明的核心源码指纹仍与当前运行代码一致。

B完整模型逐参数比较、主形状与性能筛查的最新状态以REPORT.md和lifetime-b2.json为准。非法缓存集体拒绝、匹配prefix的原Timer验证、独立银行和稳定收益尚待完成。
