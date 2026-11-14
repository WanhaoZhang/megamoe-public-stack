const format = new Intl.NumberFormat('en-US');

function svgNode(x, y, width, height, owner, title, lines) {
    return `<g class="bw-svg-node" data-owner="${owner}">
        <rect x="${x}" y="${y}" width="${width}" height="${height}" rx="5"/>
        <text x="${x + 13}" y="${y + 24}" class="bw-svg-title">${title}</text>
        ${lines.map((line, i) => `<text x="${x + 13}" y="${y + 47 + i * 20}">${line}</text>`).join('')}
    </g>`;
}

function svgMarkers(id, kinds) {
    return `<defs>${kinds.map(kind => `<marker id="${id}-${kind}" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
        <path class="bw-arrow ${kind}" d="M0 0 L8 4 L0 8 Z"/>
    </marker>`).join('')}</defs>`;
}

function svgEdge(id, kind, points) {
    return `<path class="bw-edge ${kind}" d="${points}" marker-end="url(#${id}-${kind})"/>`;
}

function tokenGradients({ focus, shape, route }, source) {
    const stride = shape.home + shape.slots;
    const ranks = shape.groups.find(group => group.includes(focus.rank));
    const [replica, home] = focus.physical.map((physical, index) => {
        const execution = Math.floor(physical / stride);
        const slot = physical % stride;
        const logical = focus.logical[index];
        return {
            physical, logical, slot, rank: ranks[execution],
            owner: ranks[Math.floor(logical / shape.home)],
            rows: format.format(route.executionCounts[execution][slot]),
        };
    });
    const path = (color, points) => svgEdge('bw-route-arrow', color, points);
    return `<figure class="bw-figure">
        <h4>同一条 dY 沿两条路由派发，梯度归三个位置</h4>
        <div class="bw-scroll" tabindex="0" aria-label="反向数据流，可横向滚动">
            <svg viewBox="0 0 1040 362" class="bw-svg" role="img" aria-labelledby="bw-flow-title bw-flow-desc">
                <title id="bw-flow-title">rank${focus.rank} 的 dY[${focus.row}] 沿前向 placement 反向派发</title>
                <desc id="bw-flow-desc">dY 送到 rank${replica.rank} 的 E${replica.logical} 副本与 rank${home.rank} 的 E${home.logical} home。两个实例的输入梯度回到源 rank${focus.rank}，两份权重梯度分别归 rank${replica.owner} 和 rank${home.owner}。</desc>
                ${svgMarkers('bw-route-arrow', ['activation', 'weight'])}
                <text class="bw-svg-column" x="24" y="25">① 源 token 与派发</text>
                <text class="bw-svg-column" x="394" y="25">② 物理实例反向</text>
                <text class="bw-svg-column" x="774" y="25">③ 按语义归还</text>
                <g data-route-slot="0">${path('activation', 'M248 162 H316 V96 H394')}
                ${path('weight', 'M618 72 H774')}${path('activation', 'M618 114 H704 V179 H774')}</g>
                <g data-route-slot="1">${path('activation', 'M248 198 H330 V279 H394')}
                ${path('activation', 'M618 260 H723 V210 H774')}${path('weight', 'M618 302 H774')}</g>
                <text class="bw-svg-note" x="324" y="87">Top2 槽 0</text>
                <text class="bw-svg-note" x="337" y="270">Top2 槽 1</text>
                <text class="bw-svg-note" x="649" y="63">dW1 / dW2</text>
                <text class="bw-svg-note" x="639" y="106">dX / dp</text>
                <text class="bw-svg-note" x="639" y="252">dX / dp</text>
                <text class="bw-svg-note" x="649" y="293">dW1 / dW2</text>
                ${svgNode(24, 131, 224, 96, 'mindspeed', `rank${focus.rank} · dY[${focus.row}]`, [
                    `上游梯度一行 [${shape.H}]`, '原始 dY 扇出两份，p 留给求导',
                ])}
                <g data-route-slot="0">${svgNode(394, 47, 224, 97, 'kernel', `rank${replica.rank} · E${replica.logical} 副本`, [
                    `物理 ID ${replica.physical} · slot ${replica.slot}`, `该实例累计 ${replica.rows} 行的 dW`,
                ])}</g>
                <g data-route-slot="1">${svgNode(394, 234, 224, 97, 'kernel', `rank${home.rank} · E${home.logical} home`, [
                    `物理 ID ${home.physical} · slot ${home.slot}`, `该实例累计 ${home.rows} 行的 dW`,
                ])}</g>
                ${svgNode(774, 41, 242, 87, 'kernel', `rank${replica.owner} · E${replica.logical} 权重梯度`, [
                    'home + rank1 副本 + rank2 副本',
                    '每个 tile 等齐后做 SUM',
                ])}
                ${svgNode(774, 152, 242, 89, 'mindspeed', `rank${focus.rank} · dX[${focus.row}] / dp[${focus.row}]`, [
                    '两条路由 dX 求和 · dp 保持槽序',
                    '冻结 p 与真实 gate 的图断开',
                ])}
                ${svgNode(774, 268, 242, 84, 'kernel', `rank${home.owner} · E${home.logical} 权重梯度`, [
                    '本组 E4 只在 rank2 home 执行',
                    '本地 dW 留在 home',
                ])}
            </svg>
        </div>
        <figcaption>蓝线跟随输入 token，琥珀线跟随专家参数。rank2 的 E1 副本也给 owner 提供 E1 梯度，这一行的 E1 路由只去了 rank1。P5 等回送的 dX 写入退休后，对同一 token 的两槽求和得到 dX[${shape.M}, ${shape.H}]。P6 等各 rank 读完 dX 再复用传输区，按 F 归约 dp，回送到原来的两个槽位 dp[${shape.M}, ${shape.TopK}]。bridge 按原概率和权重 dtype 还原返回值。 ${source('G01', '派发 dY')} · ${source('G15', '还原 dX')} · ${source('G16', '还原 dp')} · ${source('B17', '桥接返回值')}</figcaption>
    </figure>`;
}

function backwardEntry({ shape }, source) {
    const edge = (kind, d) => svgEdge('bw-replay-arrow', kind, d);
    return `<figure class="bw-figure">
        <h4 id="case-backward-replay">反向入口 · 三路输入汇合，再按前向落点回放</h4>
        <p>Trainer 调用 <code>loss.backward()</code> 后，autograd 交来 dY，并触发 FSDP 参数准备与按需 checkpoint 重算。重算走普通前向入口。进入 Native 反向时，bridge 读取当前层的 tape，申请新的 replay lease。 ${source('U01', 'checkpoint')} · ${source('U02', 'FSDP hook')} · ${source('B01', 'autograd 入口')}</p>
        <div class="bw-scroll" tabindex="0" aria-label="反向桥接的三路输入，可横向滚动">
            <svg viewBox="0 0 1088 387" class="bw-svg bw-svg-wide" role="img" aria-labelledby="bw-replay-title bw-replay-desc">
                <title id="bw-replay-title">上游 dY、当前权重和前向 tape 汇合后提交反向 kernel</title>
                <desc id="bw-replay-desc">dY 由 autograd 传入。FSDP 恢复权重后 bridge 刷新 SHMEM home。tape 提供概率、计数、保存态、派发输入和原 placement。反向复用原路由，克隆控制 header 并写入新 epoch。</desc>
                ${svgMarkers('bw-replay-arrow', ['activation', 'weight', 'control'])}
                <text class="bw-svg-column" x="20" y="26">① 数据来自哪里</text>
                <text class="bw-svg-column" x="390" y="26">② fsdp2_bridge.py 准备反向</text>
                <text class="bw-svg-column" x="836" y="26">③ 提交到 NPU stream</text>
                ${edge('activation', 'M305 88 H390')}
                ${edge('weight', 'M305 196 H390')}
                ${edge('control', 'M305 306 H390')}
                ${edge('activation', 'M700 88 H754 V147 H836')}
                ${edge('weight', 'M700 196 H836')}
                ${edge('control', 'M700 306 H754 V245 H836')}
                ${svgNode(20, 47, 285, 82, 'mindspeed', 'autograd · 上游 dY', [
                    `源 token 梯度 [${shape.M}, ${shape.H}]`,
                ])}
                ${svgNode(20, 155, 285, 82, 'fsdp', 'Expert-FSDP · 当前层参数', [
                    'unshard 后的 live home W1 / W2',
                ])}
                ${svgNode(20, 263, 285, 103, 'kernel', 'ctx.saved_tensors · 前向 tape', [
                    'p、counts、BF16 gate/up、dispatch',
                    'logical / physical IDs、copies、header',
                ])}
                ${svgNode(390, 47, 310, 82, 'kernel', '准备 dY', [
                    '连续化，原样送入 kernel',
                ])}
                ${svgNode(390, 155, 310, 82, 'kernel', 'B04 刷新 SHMEM home', [
                    '复制当前 W1/W2，供本次 panel 恢复',
                ])}
                ${svgNode(390, 263, 310, 103, 'kernel', 'B07 回放已保存的 placement', [
                    '沿用前向路由与 copies，p 转 BF16',
                    'clone 控制 header，更新 header[2] 的 epoch',
                ])}
                ${svgNode(836, 123, 232, 146, 'kernel', 'B09 backward_replica', [
                    'C++ 按 descriptor 生成 tiling',
                    '用 counts 与 dispatch 对齐行序',
                    'AIV0 派发 dY，AIV1 恢复权重',
                    '本轮按 panel 消费已就绪权重',
                ])}
            </svg>
        </div>
        <figcaption>每层 tape 持有自己的路由与保存态。反向走 replay policy，复用前向校验过的 placement，轮到这层时再从 FSDP 取 live W1/W2。panel 模式的副本搬运由核内完成。 ${source('B02', '读 tape')} · ${source('B03', '读当前权重')} · ${source('B04', '刷新 home')} · ${source('B07', '更新 epoch')} · ${source('B09', 'C++ 提交')}</figcaption>
    </figure>`;
}

function gradientDependencies(source) {
    const edge = (kind, d) => svgEdge('bw-math-arrow', kind, d);
    return `<figure class="bw-figure">
        <h4 id="case-gradients">一组专家行上，六类中间量怎样形成梯度</h4>
        <div class="bw-scroll" tabindex="0" aria-label="专家反向数学依赖，可横向滚动">
            <svg viewBox="0 0 1200 520" class="bw-svg bw-svg-wide" role="img" aria-labelledby="bw-math-title bw-math-desc">
                <title id="bw-math-title">dHp、Hp、dp、dZ、dX 与权重梯度的数据依赖</title>
                <desc id="bw-math-desc">AIC 用派发 dO 和 W2 得到 dHp。AIV 用 BF16 gate/up 恢复未乘 p 的激活，生成 dp partial、Hp 和 dZ。dW2 读取原 dO 与 Hp，dX 读取 dZ 与 W1，dW1 读取 dZ 与前向派发的 Xpacked。</desc>
                ${svgMarkers('bw-math-arrow', ['activation', 'weight', 'control'])}
                <text class="bw-svg-column" x="20" y="26">① P0 与前向保存态</text>
                <text class="bw-svg-column" x="324" y="26">② P1 · AIC → AIV</text>
                <text class="bw-svg-column" x="650" y="26">③ AIV 写出的三类产物</text>
                <text class="bw-svg-column" x="935" y="26">④ 后续消费</text>
                ${edge('activation', 'M244 91 H324')}
                ${edge('control', 'M244 269 H324')}
                ${edge('activation', 'M442 131 V209')}
                ${edge('activation', 'M562 244 H594 V91 H650')}
                ${edge('weight', 'M562 274 H650')}
                ${edge('weight', 'M562 303 H594 V393 H650')}
                ${edge('activation', 'M864 91 H935')}
                ${edge('weight', 'M864 274 H935')}
                ${edge('activation', 'M864 383 H895 V377 H935')}
                ${edge('weight', 'M864 405 H895 V469 H935')}
                ${svgNode(20, 50, 224, 82, 'kernel', 'dO[m, H] 与 W2[H, F]', [
                    'dO 是派发后的原始 dY',
                ])}
                ${svgNode(20, 219, 224, 102, 'kernel', '保存的 gate/up 与路由 p', [
                    '按物理实例的派发顺序读',
                    'gate/up 来自本层 BF16 保存态',
                ])}
                ${svgNode(324, 50, 238, 82, 'kernel', 'G03 AIC · GMM2 Grad', [
                    'dHp[m, F] = dO × W2',
                ])}
                ${svgNode(324, 209, 238, 129, 'kernel', 'G06 AIV · SwiGLU backward', [
                    '按前向限幅恢复未乘 p 的 h',
                    'h = SwiGLU(gate, up)',
                    'dH = p × dHp，供 SwiGLU 求导',
                    '三种结果按 tile 发布',
                ])}
                ${svgNode(650, 47, 214, 87, 'kernel', 'dp partial', [
                    '按 F 分 tile 累加 dHp ⊙ h',
                ])}
                ${svgNode(650, 229, 214, 88, 'kernel', 'Hp[m, F]', [
                    'Hp = p × h',
                ])}
                ${svgNode(650, 348, 214, 88, 'kernel', 'dZ[m, 2F]', [
                    '用 dH 对 gate/up 求导',
                ])}
                ${svgNode(935, 47, 245, 87, 'mindspeed', 'G16 dp 回原 token 的槽位', [
                    '汇总 F 方向 partial 后回送',
                ])}
                ${svgNode(935, 229, 245, 88, 'kernel', 'G07 dW2[H, F]', [
                    'dOᵀ × Hp，读取原始 dO',
                ])}
                ${svgNode(935, 337, 245, 81, 'kernel', 'G10 dX 路由贡献 [m, H]', [
                    'dZ × W1，使用恢复后的 W1',
                ])}
                ${svgNode(935, 428, 245, 81, 'kernel', 'G08 / G11 dW1[2F, H]', [
                    'dZᵀ × Xpacked，读前向派发 X',
                ])}
            </svg>
        </div>
        <figcaption>m 表示该物理专家实例处理的行数。图中矩阵按实验的 canonical 权重布局表达，蓝线跟随 token 梯度，琥珀线跟随权重梯度，灰虚线标记保存态。AIV 会把 <code>dHp ⊙ h</code> 的 BF16 乘积按列 tile 归成 FP32 partial，再在 P6 归约和回送。固定 p 仍返回 dp，固定路由覆盖使 dp 与原模型 gate 图断开。 ${source('G03', 'dHp')} · ${source('G06', 'SwiGLU 求导')} · ${source('G16', '归约 dp')}</figcaption>
    </figure>`;
}

function kernelSchedule(source) {
    return `<figure class="bw-figure">
        <h4 id="case-owner">设备调度 · 权重 ready、梯度 tile 与 owner 消费</h4>
        <div class="bw-scroll" tabindex="0" aria-label="反向核内依赖表，可横向滚动">
            <table class="bw-table bw-schedule">
                <thead><tr><th>阶段</th><th>AIC · GEMM</th><th>AIV · 恢复 / 后处理</th><th>下游消费的产物</th></tr></thead>
                <tbody>
                    <tr><th>Panel restore + P1</th><td>dO 已派发，W2 对应 panel 已发布当前 epoch 后才读取。计算 <code>dHp = dO × W2</code>。</td><td>AIV0 按专家派发 dO，AIV1 按 panel 搬权重。P1 后处理读取保存的 BF16 gate/up，写 Hp、dp partial 和 dZ。</td><td>Hp 给 dW2，dZ 给 dW1 / dX。 ${source('G02', '恢复')} · ${source('G04', 'acquire')} · ${source('G06', 'SwiGLU 求导')}</td></tr>
                    <tr><th>dW2 → early dW1</th><td><code>dW2 = dYᵀ × Hp</code>。待 dZ/Hp 写入及 dW2 读取 dY 退休，对复制专家提前执行 <code>dW1 = dZᵀ × Xpacked</code>。</td><td>梯度 tile 写完后发布 ready。owner 只消费当前 epoch。</td><td>home 梯度 tile 和远端副本 partial。 ${source('G07', '退休边界')} · ${source('G08', '提前算 dW1')} · ${source('G09', '发布 tile')}</td></tr>
                    <tr><th>dX + 剩余 dW1</th><td><code>dX = dZ × W1</code>，再计算未前移的 dW1。</td><td>沿前向 token/槽映射回送 dX / dp；owner 归并继续处理已就绪 tile。</td><td>源 rank 的 dX/dp 和每个实例的全量梯度 tile。 ${source('G10', 'dX')} · ${source('G11', '补算 dW1')}</td></tr>
                    <tr><th>owner 收口</th><td>所有产出在返回前完成。</td><td>AIV1 在 P1 后开始服务，AIV0 从 P5 加入。各自拥有不同 tile，先 dW2 后 dW1。</td><td>等齐本地与副本 ready 后，BF16 partial 以 FP32 SUM，再按 BF16 写回 home。 ${source('G12', '两路 owner worker')} · ${source('G14', '消费和归并')}</td></tr>
                </tbody>
            </table>
        </div>
        <figcaption>表格按数据依赖阅读。不同核与不同 tile 会交叠，行高和先后间隔均无时间刻度。</figcaption>
    </figure>`;
}

// Original TimerV2 report, layer 7 B, independently normalized per rank.
const timerRanks = [
    [16406864, 10409567, 16166566],
    [16449960, 15911752, null],
    [16470292, 16109590, null],
    [16448783, 14762910, null],
    [16756806, 10731569, 16515835],
    [16795309, 16133277, null],
    [16760652, 16106853, null],
    [16844916, 14854174, null],
];

function ownerTimeline({ focus }) {
    const max = 17000000;
    const percent = cycle => (100 * cycle / max).toFixed(4) + '%';
    return `<figure class="bw-figure bw-time">
        <h4>实际打点：各 rank 的 B 包络与完成时刻</h4>
        <p class="bw-subline">整网 TimerON · step ${focus.step} · layer 7 · 各 rank 以自己的 B 起点为 0</p>
        <div class="bw-time-legend"><span class="envelope">整 B 包络</span><span class="dw1">本地最后 dW1 完成</span><span class="store">owner 最后 store 完成</span></div>
        <div class="bw-time-axis"><span>0</span><span>4M</span><span>8M</span><span>12M</span><span>16M</span></div>
        ${timerRanks.map(([end, dw1, store], rank) => {
            const stats = `包络 ${format.format(end)}、dW1 ${format.format(dw1)}${store ? '、store ' + format.format(store) : ''} cycle`;
            return `<div class="bw-time-row" role="img" aria-label="rank${rank} ${stats}" title="rank${rank} ${stats}">
                <strong>r${rank}</strong><div class="bw-time-track">
                    <span class="envelope" style="--at:${percent(end)}"></span>
                    <i class="dw1" style="--at:${percent(dw1)}"></i>
                    ${store ? `<i class="store" style="--at:${percent(store)}"></i>` : ''}
                </div>
            </div>`;
        }).join('')}
        <figcaption>横轴为百万 cycle。rank0 的本地 dW1 完成于 10.410M，最后一次 owner store 完成于 16.167M，相差 5.757M cycle。该跨度同时覆盖后续贡献就绪与归并。独立统计的 rank0 最终跨 rank 同步最大单 lane 时长为 3,438 cycle。原始数字见整网实验 <code>REPORT.md</code> 的「第 7 层原始 Timer」表。</figcaption>
    </figure>`;
}

function renderBackward(data, source) {
    return backwardEntry(data, source) + gradientDependencies(source) + tokenGradients(data, source)
        + kernelSchedule(source) + ownerTimeline(data);
}

function renderUpdate({ shape }, source) {
    const edge = (kind, d) => svgEdge('bw-update-arrow', kind, d);
    return `<figure class="bw-figure">
        <h4>一份 E1 权重梯度，依次跨两个通信域</h4>
        <div class="bw-scroll" tabindex="0" aria-label="EP 与 Expert-FSDP 梯度通信域，可横向滚动">
            <svg viewBox="0 0 1090 419" class="bw-svg bw-svg-wide" role="img" aria-labelledby="bw-update-title bw-update-desc">
                <title id="bw-update-title">E1 梯度先在 EP 组内归主，再沿 Expert-FSDP 分片同步</title>
                <desc id="bw-update-desc">EP 组 A 的 E1 home 和两个副本梯度汇总到 rank0，组 B 汇总到 rank4。两份 home 梯度由 rank0 和 rank4 的 Expert-FSDP 组归约并切分。shared 参数由普通 FSDP8 处理。</desc>
                ${svgMarkers('bw-update-arrow', ['weight', 'control'])}
                <text class="bw-svg-column" x="20" y="26">① EP 内各物理实例的贡献</text>
                <text class="bw-svg-column" x="312" y="26">② owner 合并后交回</text>
                <text class="bw-svg-column" x="584" y="26">③ FSDP 分片归约</text>
                <text class="bw-svg-column" x="878" y="26">④ 优化器消费</text>
                ${edge('weight', 'M244 93 H312')}
                ${edge('weight', 'M244 223 H312')}
                ${edge('weight', 'M510 93 H549 V135 H584')}
                ${edge('weight', 'M510 223 H549 V182 H584')}
                ${edge('weight', 'M811 135 H846 V93 H878')}
                ${edge('weight', 'M811 182 H846 V223 H878')}
                <path class="bw-divider" d="M20 294 H1070"/>
                ${edge('control', 'M510 362 H584')}
                ${edge('control', 'M811 362 H878')}
                ${svgNode(20, 47, 224, 92, 'kernel', 'EP 组 A · rank0–3', [
                    'rank0 的 E1 home',
                    '+ rank1、rank2 的 E1 副本',
                ])}
                ${svgNode(20, 177, 224, 92, 'kernel', 'EP 组 B · rank4–7', [
                    'rank4 的 E1 home',
                    '+ rank5、rank6 的 E1 副本',
                ])}
                ${svgNode(312, 47, 198, 92, 'kernel', 'rank0 · E1 home dW', [
                    '组 A 的本地 + 副本 SUM',
                    'B16 返回 home 梯度',
                ])}
                ${svgNode(312, 177, 198, 92, 'kernel', 'rank4 · E1 home dW', [
                    '组 B 独立完成 EP 归主',
                    'B16 返回 home 梯度',
                ])}
                ${svgNode(584, 104, 227, 108, 'fsdp', 'Expert-FSDP2 · rank0 / 4', [
                    'U04 FP32 reduce-scatter',
                    '按配置缩放，再各取一份分片',
                ])}
                ${svgNode(878, 47, 192, 92, 'mindspeed', 'rank0 · 参数梯度分片', [
                    '交给对应的优化器参数组',
                ])}
                ${svgNode(878, 177, 192, 92, 'mindspeed', 'rank4 · 参数梯度分片', [
                    '交给对应的优化器参数组',
                ])}
                ${svgNode(20, 321, 490, 82, 'mindspeed', 'shared MLP 与其余普通参数', [
                    '框架 autograd 形成各自的参数梯度',
                ])}
                ${svgNode(584, 321, 227, 82, 'fsdp', '普通 FSDP8', [
                    '在普通参数 mesh 归约与分片',
                ])}
                ${svgNode(878, 321, 192, 82, 'mindspeed', '各 rank 的普通参数分片', [
                    '按优化器组更新',
                ])}
            </svg>
        </div>
        <p class="bw-subline">图只放大 E1。bridge 返回本 rank 的全部 home 梯度，即 <code>dW1[${shape.home}, ${shape.F * 2}, ${shape.H}]</code>、<code>dW2[${shape.home}, ${shape.H}, ${shape.F}]</code>，随后由 autograd 交给对应参数。 ${source('B16', 'home 返回')} · ${source('U04', '参数组 foreach_reduce')} · ${source('U05', 'shared 路径')}</p>
        <ol class="bw-update-steps" aria-label="Trainer 参数更新顺序">
            <li><strong>裁剪 / finite 检查</strong><span>${source('U06', 'Trainer 检查')}</span></li>
            <li><strong>finite 时更新参数</strong><span>${source('U07', 'FSDPTurbo Muon')}</span></li>
            <li><strong>finite 时更新 LR</strong><span>随后统一 ${source('U08', 'zero_grad')}。</span></li>
            <li><strong>下一 step 读取当前权重</strong><span>FSDP 解分片，bridge 刷新 home。</span></li>
        </ol>
        <figcaption>EP 归主解决副本参数贡献的归属，Expert-FSDP 负责两组 home 梯度的同步与分片。两个操作发生在不同通信域。源 token 的 dX 继续沿模型计算图回传，shared 分支的输入梯度在上游累加。</figcaption>
    </figure>`;
}

module.exports = { renderBackward, renderUpdate };
