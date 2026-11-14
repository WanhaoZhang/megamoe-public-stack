# EP8负载不均三后端比较

该接入用于在同一模型与优化器下比较Ascend MoonEP、Native普通和Native共享融合。两个Native分支都保留同一个模型shared expert；普通分支由框架执行shared，融合分支由Native执行。

## 参数与计算所有权

持久参数、梯度、Adam一阶和二阶状态均为FP32，专家计算镜像为BF16。MoonEP只在框架参数初始化后绑定池化计算镜像，不替换训练Parameter。Expert-FSDP1不再复制一份冗余FSDP BF16权重；梯度钩子在meta物化后的首个专家前向注册，且全局专家梯度只除EP规模一次。

## 生命周期

多个层复用Native或MoonEP通信与镜像池。反向必须恢复当前层的权重和计划，私有返回梯度不能别名下一层会覆盖的池。Native的完成检查必须在梯度裁剪、optimizer.step之前统一执行。重计算上下文捕获原始step，按层恢复对应路由，禁止取当前全局step冒充重放step。

## 构建与能力

EP8/E256需要32个home及32个replica物理槽。Native Host、planner、worker必须同为64槽构建，并验证实际加载二进制；32槽回归使用独立目录。MoonEP使用冻结版本的Buffer API，保留其规划、prefetch、dispatch、combine、梯度归主和必要同步。每进程只加载本后端SHMEM。

## MoonEP配置

异步通信显式设置async_finish=True、npu_async_comm=True，使用固定版本真实异步API和单一通信流的事件依赖。梯度归主保留原生设备同步和HCCL入口rendezvous。shmem_stream_barrier是兼容字段，不能删除原生API内必要同步。trace、VMM探测及额外设备同步均通过类型化配置控制，正式计时关闭。

## 实验资格

先进行独立CPU参考及32/64槽、空专家、热点切换、非法计划恢复、跨层池污染、三次参数更新检查。正式结果必须具有八rank完整步日志、退出记录、初始参数/输入/路由SHA、SO指纹、dtype及重计算次数。每点新进程25步，前5步预热，后20步各取八rank完整设备完成wall time的最大值；三个独立重复轮换后端顺序。

官方九层图未提供完整优化器及训练驱动；共同裁剪后的Ascend结果应标注自身模型形状、词表、attention/mHC配置，不能直接称为官方H20绝对数值复现。原始证据均在实验项目目录。
