# 实现记录

2026-11-16：新建独立clone，首次clone未在checkout前设置core.longpaths，Windows长路径导致checkout失败、index未写入。未触碰其他工作区。核对origin、HEAD和空index后，仅在刚创建的本任务目录设置longpaths/autocrlf并恢复初始tree；不复用或清理旧实验目录。随后独立分支runtime d974205。

显式--host-boundary-trace默认关闭；仅包装已有函数，入口第一步取monotonic_ns，记录保存在CPU内存，训练循环结束后导出。不增加设备Event、tensor快照、device同步或autograd hooks。追踪本身仍有Python开销，wall time不进入性能结果。GC回调只记录CPU边界。

诊断计划补充：先以B1 legacy、B3 legacy、B1 preserve＋排序键缓存三条小模型路径分别验证完整三更新/step0梯度；全部通过后，用对应三条主形状11更新观察updates0–4和5–10。前两条使用原72653a TimerOFF，第三条使用e2868f优化worker。与旧rich capture的差异不能单独归因于Event记录或特定硬件限制。

生命周期策略、任务顺序、copies/quota、交付时机和资源预算未在此分支修改。实际SO、源文件和数值资格均写入每个独立run目录。诊断状态由diagnosis-v1.json记录；失败不会覆盖或当作完成。

首次v1小模型在torchrun创建TCPStore时因33856端口EADDRINUSE失败，八rank训练均未开始，整卡峰值4788MiB，无外来GPU进程。状态/源码快照/stdout保留。runner随后读不存在的steps文件暴露了其失败处理缺陷；已改为先返回真实非资格状态，再访问训练产物。v2使用独立40100–42099范围并在全节点准入锁内实际bind探测空闲端口；此检查不保证长期保留端口，若再次冲突仍保留失败，不停止任何其他服务。v1不覆盖，所有新tag和状态用v2。

v2仍在torchrun前因40740端口EADDRINUSE失败。只查LISTEN和IPv4 loopback探测不足：实际ss -tanp发现[::1]:40740→[::1]:6443是kube-apiserver(pid226946)的既有ESTAB客户端连接，不是GPU训练。未停止它。v3改用低于该节点ephemeral范围的25100–27099，并分别检查IPv6/IPv4 wildcard bind；仍不把探测当长期保留，失败继续独立记录。v1/v2训练均未开始，不能记为数值失败或诊断完成。

v3成功训练三更新，但门禁发现operator.backward覆盖为0：记录器沿用rich捕获白名单，漏了实际TimerOFF入口backward_replica_redispatch，只包含with_timer版本。MegaMoE.backward封套和其他主机记录存在，未把缺失的operator数据当成通过。补一项包装名称，不改计算/返回值；v4新tag重新完整验收。v3记录保留为覆盖失败，尚未作为数值或因果证据使用。
