# 下一项干预：对应梯度bank的ready，而非flush时的整个当前流

目标仍是完整更新联合规划；以下是公共梯度路径的单变量因果控制，不预先称新策略。旧33点及优化后B0/B2/B3负结果保留，不启动大矩阵。

## 当前依据与未证明项

同一优化worker的B1/B2/B3已有正式25更新计时；B3通信字节和发送次数减少，却没有赢过强控制。预热后host-only诊断里B1的约200ms入口差距消失，不能再作为稳态机制。B3 legacy的finish封套约163ms，匹配当前preserve路径的B3诊断另补v5。

源码flush调用_delivery_scope，在当时的整个当前流上record ready。原梯度bank可能更早已写完，随后已经排入Attention/FSDP等工作；服务流此时等待的是新record的广义fence。这个源码依赖值得干预，但尚未证明有多少可消除整步时间。

## 独立实现边界

从公共执行器交付68aacae建立新的独立clone/branch和远端目录，使用同一e2868f TimerOFF SO，不改C++、Native计划、FP32通信或owner。加入正交公开开关gradient_ready_scope=call_stream/bank_write，默认保持旧路径。所有B控制可用同一开关，不按planner名称改变执行器。

bank_write方式在每个实际梯度bank最后一次写入后记录ready，按对应pending key/version/slot generation关联。flush仅等待它将读取的本地source bank的最新ready，不无条件等待flush时整个默认流。远端owner接收及FP32加法仍在原service顺序；本地home梯度也沿原service顺序，不同时改变raw输出读取流。

保留现有drain、固定提交顺序和八条目资源，不删除串行保护。所有输入ready/完成Event与inflight ticket共同持有，直到服务完成后才释放；容量不足时的中途flush必须已经有对应最新ready，不能复用旧slot事件。零贡献副本不生成义务，Adam前全部ticket完成，EP缩放仍仅一次。

依赖应为：zero/add bank→本bank ready→service wait→P2P/owner add→service done→lease release。owner.grad初始化及本地home add在同一service序列中先于对应远端add，防止跨流RMW丢更新。Event开销计入正式时间；无CPU参考/离线计划冒充在线成绩。

## 验证与停止条件

先过CPU票据/租约、同版本追加、容量flush、零owner、slot回收、更新切换；NPU小形状W1/W2、B0–B3、空专家/远端贡献，对独立输出/梯度参考；完整三次Adam及step0逐参数门槛不变。确保原call_stream控制数值与实际SO身份仍一致。

再做B3 call_stream/bank_write一配对，保持当前主形状、MaxVio10/same、25更新/5预热/20测量。身份、实际计划/贡献者、源文件/实际SO、HBM和有限性全部检查。主指标仍八rank最大完整更新wall。若只是减少finish主机等待却不减少完整更新时间，保留负结果，不称成功；若有信号，补交换顺序独立重复，并再次与同底座anchor和fixed-home比较。新联合策略投入门槛仍为相对最强适用基线至少3%、三轮同向。

## 另外两个控制，禁止混进本次首次干预

1. raw dW的record_stream(service)可能让大输出缓冲存活到后续P2P完成；B1与B2/B3峰值不同只是线索。另行比较owner局部累积放置/真实最后读取位置，需保持owner.grad RMW顺序。不能在ready试验里同时修改。
2. 现有quota→实际计数/贡献者推导的离线一致性已验证，后续可把不可变计划的元数据在执行前准备好，消除反向的重复CPU读回/对象all-gather。但需核对原始forward登记PlanTicket的时机，逐调用绑定相同计划；host梯度封套含真实核计算等待，不能把其全时长当作可省CPU开销。

上述优化若有效，仍属于公共执行器/依赖修正，向所有适用基线开放。只有后续动态驻留/交付决策在此强公共底座上胜过固定分组、固定布局/归并和简单组合，才讨论新的planner贡献。
