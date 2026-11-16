# Step-lifetime study：实施与实测状态

独立分支：`research/step-lifetime`；源码位于`stack/`。原始数据在本目录的`evidence/`，大梯度PT保留在各receipt标注的本研究远端目录。

固定L4/EP8/E256/Top8/H4096/F1024/S16384，GA4/W2。每个筛查点一次新进程、25更新、5预热＋20测量，取逐更新八rank最大完整wall time。单轮不构成稳定收益。

## 当前阶段

- lifetime-a3: screening_complete；通过执行门禁7项；已完成筛查18项。
- lifetime-b1: stopped_at_gate；通过执行门禁0项；已完成筛查0项。
  失败原因：Numerical execution failed: anchor。保留原始日志，不用失败点作图。
- lifetime-b2: screening_complete；通过执行门禁6项；已完成筛查15项。

## 方式说明

本页所有方式均使用普通MegaMoE执行器；native指原有单调用规划策略，不是另一个模型或共享融合分支。模型shared expert统一由框架计算。

|代号|实际执行|
|---|---|
|serial_home / serial_native|相同就绪前缀下分别执行完整S的FFN；后者允许单调用专家复制|
|fixed_home / fixed_native|同层固定任务分组，分两波半S执行；后者允许单调用专家复制|
|fifo_home|FIFO最小就绪集合，W2时通常单任务，固定home；双任务强对照见fixed_home|
|easy_home|MaxUtil就绪任务选择，固定home|
|anchor|fixed_native原核内dW流水归并；与B使用同一候选SO的默认路径|
|b0|每调用准备权重，公共FP32 dW路径产生后立即提交|
|b1|同参数版本内八条权重驻留；dW立即提交|
|b2|b1权重策略；每GA2窗口提交累积dW|
|b3|b1权重策略；GA4步末优先提交，容量满可提前归并|

## 已完成筛查

|组|路由|方式|平均步时(s)|中位步时(s)|整卡峰值(GiB)|
|---|---|---|---:|---:|---:|
|A|same|serial_home|4.193028|4.135088|66.490|
|A|same|serial_native|4.450400|4.405843|66.159|
|A|same|fixed_home|4.181305|4.123229|66.433|
|A|same|fixed_native|4.436109|4.394731|66.270|
|A|same|fifo_home|4.216082|4.163250|66.061|
|A|same|easy_home|4.214918|4.168749|72.020|
|A|alternating|serial_native|4.450402|4.402991|66.253|
|A|alternating|fixed_home|4.066253|4.025350|66.277|
|A|alternating|fixed_native|4.368961|4.322689|66.604|
|A|alternating|fifo_home|4.201787|4.155034|67.097|
|A|alternating|easy_home|4.069859|4.027920|66.318|
|A|alternating|serial_home|4.176407|4.124008|66.254|
|A|random|fixed_home|4.088051|4.054344|67.310|
|A|random|fixed_native|4.399500|4.355152|66.371|
|A|random|fifo_home|4.213262|4.169941|66.005|
|A|random|easy_home|4.164828|4.116300|71.788|
|A|random|serial_home|4.205285|4.161850|66.260|
|A|random|serial_native|4.426814|4.387839|66.271|
|B-qualification|same|anchor|4.429785|4.408724|67.217|
|B-qualification|same|b0|4.542949|4.496117|69.125|
|B-qualification|same|b1|4.502113|4.468250|68.930|
|B-qualification|same|b2|4.517349|4.477899|66.375|
|B-qualification|same|b3|4.567793|4.507379|66.118|
|B-qualification|alternating|b0|4.443696|4.391762|69.048|
|B-qualification|alternating|b1|4.450603|4.392568|68.874|
|B-qualification|alternating|b2|4.486400|4.426806|66.352|
|B-qualification|alternating|b3|4.486877|4.437982|65.945|
|B-qualification|alternating|anchor|4.363721|4.317423|67.374|
|B-qualification|random|b1|4.462796|4.424525|69.051|
|B-qualification|random|b2|4.477453|4.440684|66.064|
|B-qualification|random|b3|4.507320|4.457764|66.276|
|B-qualification|random|anchor|4.374325|4.336418|67.337|
|B-qualification|random|b0|4.469289|4.429027|68.875|

## 独立执行器验证

门禁标签：lifetime-independent-ffn-v4；通过：True；记录数：64；缓存命中字节：88080384；梯度远端字节：75497472。

当前本地运行代码与最新数值证书匹配：True；接口加固状态：complete。计时快照按自身历史证书核对，未覆盖改写旧数据。

独立诊断lane状态：complete。小形状prefix原Timer副本覆盖资格：True；独立主形状实际copies/quota跨B方式一致性：True。

计划一致性范围：main L4 H4096 F1024 S16384 GA4 W2 random routes, three parameter versions; actual compiled Native plans with positive replicas。它补充三次更新的诊断证据，不等于为全部25更新计时日志补出了逐调用哈希。

Timer资格范围：L2 H512 F512 S256 GA4/W2 prefix path, three Adam updates; MoonEP replica plan as coverage stimulus；与TimerOFF固定Native的逐参数梯度对照通过。主形状关键路径与B公共服务路径时间线仍未完成。

same单轮四格：复制收益（顺序/分组）=-0.257372/-0.254804s；交互差=-0.002569s。负收益表示复制更慢；单轮不构成因果或稳定性结论。

alternating单轮四格：复制收益（顺序/分组）=-0.273995/-0.302708s；交互差=0.028713s。负收益表示复制更慢；单轮不构成因果或稳定性结论。

random单轮四格：复制收益（顺序/分组）=-0.221528/-0.311449s；交互差=0.089921s。负收益表示复制更慢；单轮不构成因果或稳定性结论。

[实测图PNG](figures/A.png) · [SVG](figures/A.svg) · [PDF](figures/A.pdf) · [CSV](results.csv)

[B组资格路径PNG](figures/B-qualification.png) · [SVG](figures/B-qualification.svg) · [PDF](figures/B-qualification.pdf)

## 匹配条件和机制证据

主机为Ascend950DT八卡。本轮使用合法合成token与next-token标签，不是预训练收敛实验。每卡四条完整16K序列，GA4覆盖32条序列、524288个序列token；有效label数按实际mask归一化。
V4 attention/mHC、框架shared expert、FP32参数/梯度/Adam、BF16专家、AdamW LR1e-5/betas(0.9,0.95)/epsilon1e-8/decay0.01/clip1；相同prefix复用、checkpoint切分及FSDP前向驻留。
已完成33个点的八rank初始参数、输入、逻辑路由及重计算调用指纹核对；全部匹配：True。B的实际任务分组/切片顺序也一致。
当前计时日志没有保存每次copies/quota的完整摘要，不能把相同Native输入当作运行期逐项一致的直接证明。这个证据缺口保留，诊断版本另行补齐。
字节和发送次数见[逻辑payload CSV](diagnostics/lifetime-payloads.csv)，为八rank、全部25更新之和，包含预热；不是硬件链路计数。缓存命中包含前向、重放和反向准备，尚未分离跨microbatch收益。
B3不是保证只在步末发送：八条梯度存储耗尽时会确定性提前flush全部pending，必须结合flush次数解释结果。
路由几何诊断[JSON](diagnostics/route-opportunity.json)只给线性计算理想互补空间，不能解释为整步加速上界或实测关键路径。

实际任务组大小与跨层次数见[运行期审计](diagnostics/completed-audit.json)；MaxUtil在same/random确实产生跨层任务组，在alternating采用与固定分组相同的双任务同层组。W2的FIFO最小集合为1，不能把胜过它单独解释为动态任务选择的收益。

[匹配机制差异CSV](diagnostics/matched-contrasts.csv)逐项区分分组、复制、动态选择、驻留和归并时机；单轮触发不等于稳定收益。
最高层筛选仍需比较最强适用A对照与原锚点，不能只超过新增公共路径中较慢的B0。

## 实现与限制

- A：新增serial_home、serial_native、fifo_home；统一prefix/checkpoint条件。serial保留相同就绪前缀，FFN使用独立完整S调用，分组使用两波半S组合。
- B：ABI1候选worker，版本化八条BF16权重缓存、八条FP32梯度累积项、独立owner服务流/HCCL组；复用缓冲与更新前等待交付。
- 缓存命中减少远端权重读取，仍需本地拷入执行槽；没有声称零搬运或自动降低HBM。原始route plan和optimizer owner保留。
- routed原始dW走显式交付，返回None绕过原Parameter hook；向原FP32.grad布局只应用一次EP缩放。GA仍由loss归一化。
- B当前为资格验证路径，CPU读取当前placement的开销全部计入；尚非AscendC联合planner。需NPU正确性与机制时间线通过后才能解释正式性能。
- 两rank Gloo参考检查通过b0-b3、三次Adam及精确梯度求和/EP缩放；初版五项CPU契约通过，加固后六项通过。CPU通过不代替NPU验收。
- 新联合策略、主形状/B服务路径完整时间线、非法缓存集体拒绝、稳定收益、独立银行与自然训练路由验证尚未完成；108矩阵关闭。Timer和copies/quota的小形状诊断状态见上。

## 失败与恢复

- 首次部署未完成时的提前启动：在设备初始化前失败，记录在track-a.json。
- a2冻结检查发现漏复制扩展SO：在训练前拒绝；补齐冻结二进制后改用a3新标签。
- 候选首次构建遇到CRLF脚本；第二次编译发现字段位于关闭的宏内，修正后第三次构建成功。所有构建日志保留。

## 下一步

完成两条门禁与小矩阵；按整步结果和真实搬运/交付记录选择主线。若B资格路径显示损失，先区分服务实现成本与生命周期策略，不据此断言该研究问题无收益。

## 后续判定

状态：complete。两条小矩阵已完成；本轮最好的生命周期候选未相对已跑最强A对照达到3%的单轮触发门槛。未建立稳定收益，停止扩大矩阵；先分析控制、FP32交付和流水资源开销，再决定下一项研究。
