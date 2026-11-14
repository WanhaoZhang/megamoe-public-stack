const fmt = new Intl.NumberFormat('en-US').format;

function renderStep(data, source) {
    const lane = (kind, id, title, payload) =>
        `<td class="step-${kind}">${source(id, title)}<small>${payload}</small></td>`;
    const ms = (id, title, payload) => lane('ms', id, title, payload);
    const fsdp = (id, title, payload) => lane('fsdp', id, title, payload);
    const native = (id, title, payload) => lane('native', id, title, payload);
    const empty = '<td></td>';
    const row = (phase, ...cells) => `<tr><th scope="row">${phase}</th>${cells.join('')}</tr>`;
    const focus = data.focus;
    return `<p>纵向按一轮训练的依赖顺序读，横向看数据交给了哪个库。表里的箭头表示交接方向，核内并行和 autograd 调度见后两节。</p>
    <figure class="case-diagram">
    <div class="case-scroll" tabindex="0" role="region" aria-label="三仓代码和数据交接泳道">
    <table class="step-lanes"><thead><tr><th scope="col">阶段</th><th scope="col">MindSpeed-LLM</th><th scope="col">FSDPTurbo</th><th scope="col">MegaMoE Native / Kernel</th></tr></thead><tbody>
    ${row('启动 · 一次',
        ms('I13', '训练入口与模型装配 →', '八进程读取模型、数据和并行配置；构建 8 个主干层与 1 个 MTP 层'),
        fsdp('I14', '创建 mesh', '普通 FSDP8；每行 EP4，每列 Expert-FSDP2'), empty)}
    ${row('装配 · 一次',
        ms('I20', 'Trainer 就绪', '模型、dataloader、optimizer、scheduler'),
        fsdp('I10', '切分并接管 experts.forward', 'EP 沿专家维切分；Expert-FSDP 管理 home Parameter；非重入 checkpoint 包装目标层'), empty)}
    ${row('取数',
        ms('S06', 'Trainer 取当前 batch', '每 rank 一个序列，8,192 个 token；input_ids、labels 等模型输入'), empty, empty)}
    ${row('层前向',
        ms('F03', 'embedding → attention → MoE', '每层生成当前 X 与 gate 路由；shared MLP 保持框架路径'),
        fsdp('F18', 'all-gather / unshard', '普通参数和 home 专家使用各自 mesh；当前专家取得两个 home 的 W1/W2'),
        empty)}
    ${row('routed 交接',
        ms('F02', 'X 与 gate 输出 →', 'gate 照常运行，X 来自当前模型层'),
        fsdp('F06', 'balanced dispatcher →', '实验 debug 入口换入冻结 ids/p=.5；每个专家模块持有读取当前 home 权重的闭包'),
        native('F07', '_ExternalExperts / NativeFSDP2Pool', '执行池在进程内按静态配置惰性创建和复用；本次调用刷新权重、规划、提交设备前向并保存 tape'))}
    ${row('输出 / loss',
        ms('S10', '主干与 MTP loss', 'routed Y + shared MLP，后续 decoder、LM head 与 MTP 进入 loss'),
        fsdp('F19', '层输出后的 FSDP hook', 'post-forward 按配置 reshard，并挂反向准备 hook'),
        native('N11', '← 返回 Y 和保存状态', '设备工作沿 stream 依赖推进'))}
    ${row('反向',
        ms('U03', 'loss.backward() →', 'autograd 传播上游 dY；checkpoint 如需重算前向，该次前向再次进入在线 Planner'),
        fsdp('U02', '恢复本层参数 →', 'pre-backward unshard；交回 live home 参数'),
        native('B07', 'tape → 回放 → kernel B', '相同 placement、新 epoch；dX/dp 回源，dW 在 EP 内归主'))}
    ${row('同步 / 更新',
        ms('U07', '← 裁剪、检查并更新分片', 'Muon 配置，更新学习率、清梯度、进入下一步'),
        fsdp('U04', '← reduce-scatter', 'routed home 梯度走 Expert-FSDP，shared 和普通参数走普通 FSDP'),
        native('B16', '交回 home dW', '交给 autograd 对应 Parameter，副本贡献已合入 home'))}
    </tbody></table></div>
    <figcaption>每个带下划线的节点都能跳到对应代码。后续放大的是 step ${focus.step} 的 <code>${focus.layer}</code>。这一层的行为在其他目标层重复，MTP 内也有 decoder。 ${source('S09', 'MTP 入口')}</figcaption></figure>
    ${renderHostScopes(data.hostScopes)}`;
}

function renderHostScopes(scopes) {
    const x = ms => 82 + ms / 2000 * 774;
    const y = layer => 40 + layer * 27;
    const phaseNames = { forward: '前向', recompute: '重算', backward: '反向' };
    const grid = Array.from({length: 9}, (_, layer) => `<text x="12" y="${y(layer)+12}">${layer === 8 ? 'MTP' : '层 '+layer}</text>
        <line x1="82" x2="856" y1="${y(layer)+7}" y2="${y(layer)+7}" stroke="#e1e8ef"/>`).join('');
    const ticks = [0, 500, 1000, 1500, 2000].map(ms => `<line x1="${x(ms)}" x2="${x(ms)}" y1="25" y2="278" stroke="#d5dee8"/>
        <text x="${x(ms)}" y="300" text-anchor="middle">${ms}</text>`).join('');
    const bars = scopes.map(({ layer, phase, startMs, durationMs }) =>
        `<rect class="scope-${phase}" x="${x(startMs)}" y="${y(layer)}" width="${durationMs / 2000 * 774}" height="14" rx="2">
            <title>${layer === 8 ? 'MTP' : '层 '+layer} · ${phaseNames[phase]} · 起点 ${startMs} ms · 时长 ${durationMs} ms</title></rect>`).join('');
    return `<figure class="case-diagram">
        <h4>实测 step 10 · 一步里的九次前向与逐层重算</h4>
        <div class="scope-key"><span class="scope-forward">前向</span><span class="scope-recompute">重算</span><span class="scope-backward">反向</span></div>
        <div class="case-scroll" tabindex="0" role="region" aria-label="rank0 的 27 段 Host Scope">
        <svg class="case-svg" viewBox="0 0 900 313" role="img" aria-label="前向从第0层走向 MTP，反向从 MTP 逐层重算并返回第0层">
            <text x="82" y="18" class="diagram-head">rank0 · 相对首个 routed scope 的时间 / ms</text>${ticks}${grid}${bars}
        </svg></div>
        <figcaption>框架 profiler 的 27 段 routed CPU scope。每个矩形保留实际起点和长度，含采集与 Host 侧工作。空白处还有 attention、loss、FSDP 等其他操作。设备 kernel 周期与 TimerOFF 性能另图展示。</figcaption>
    </figure>`;
}

function svg(id, width, height, title, body) {
    return `<div class="case-scroll" tabindex="0" role="region" aria-label="${title}">
    <svg class="case-svg" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="case-svg-${id}">
    <title id="case-svg-${id}">${title}</title>
    ${body}</svg></div>`;
}

function diagramNode(x, y, width, height, role, title, ...lines) {
    return `<rect class="fwd-node ${role}" x="${x}" y="${y}" width="${width}" height="${height}" rx="4"/>
        <text x="${x+12}" y="${y+23}" class="diagram-head">${title}</text>
        ${lines.map((line, index) => `<text x="${x+12}" y="${y+44+index*18}">${line}</text>`).join('')}`;
}

function diagramArrows(id, edges) {
    return `<defs><marker id="case-arrow-${id}" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L8 4 L0 8z" fill="#51657b"/></marker></defs>
        ${edges.map(([kind, d]) => `<path class="fwd-dependency ${kind}" d="${d}" marker-end="url(#case-arrow-${id})"/>`).join('')}`;
}

function renderPlannerControl(data, source) {
    const { shape } = data;
    const width = shape.home + shape.slots;
    const packed = shape.E + shape.ep * width + shape.ep * shape.slots + 3;
    return `<figure class="case-diagram">
        <h4>Planner 定位置，routed kernel 搬数据</h4>
        ${svg('control', 1060, 321, 'MoonEP、EP 计划校验与 kernel 数据来源', `
            <text x="24" y="29" class="diagram-head">规划与校验 · 处理路由元数据</text>
            ${diagramArrows('control', [
                ['control', 'M192 88 H219'], ['control', 'M383 88 H410'],
                ['control', 'M574 88 H601'], ['control', 'M777 88 H804'],
                ['control', 'M919 132 V196'],
                ['activation', 'M248 211 H804'], ['weight', 'M248 275 H804'],
            ])}
            ${diagramNode(24, 44, 168, 88, 'control', '本 rank 路由', `logical_ids[${shape.M},${shape.TopK}]`, '本次 epoch、容量')}
            ${diagramNode(219, 44, 164, 88, 'control', 'MoonEP 规划', '统计 EP 直方图', '配额 → 物理 ID / 副本')}
            ${diagramNode(410, 44, 164, 88, 'control', '本地校验', '检查物理 ID 和副本', '打包计数及控制字段')}
            ${diagramNode(601, 44, 176, 88, 'control', `EP${shape.ep} HCCL all-gather`, `每 rank 摘要 ${packed} 个 int32`, `拼成 ${shape.ep} 份摘要`)}
            ${diagramNode(804, 44, 231, 88, 'control', '全局校验', '核对全组一致性和容量', '生成受检计划的 header')}
            <text x="930" y="162">header</text><text x="930" y="181">及物理布局</text>
            <text x="24" y="171" class="diagram-head">执行 · 两种实际数据来源</text>
            ${diagramNode(24, 184, 224, 54, 'activation', 'token 来源 rank', '激活 X 与路由权重 p')}
            ${diagramNode(24, 248, 224, 54, 'weight', '专家逻辑 owner', 'SHMEM home 中的 W1/W2')}
            <text x="280" y="203">K02 依 physical ID 与路由地址收集 X / p</text>
            <text x="280" y="267">K03 副本槽按 stripe 取权重，home 槽读取本地权重</text>
            ${diagramNode(804, 196, 231, 106, 'compute', 'NPU routed kernel', '检查 header / epoch / 容量', '按 plan 消费激活和权重')}
        `)}
        <figcaption>HCCL all-gather 交换的是校验摘要。这里每 rank 的 ${packed} 个 int32 包含 ${shape.E} 个逻辑计数、${shape.ep * width} 个物理计数、${shape.ep * shape.slots} 个副本表条目和 3 个控制字段。Y 的回源由 kernel 处理，单行去向见 MegaMoE 前向图。
            ${source('P13','统计路由')} · ${source('P14','分配配额')} · ${source('P16','本地校验')} · ${source('P08','摘要交换')} · ${source('P17','全局校验')} · ${source('N09','核入口检查')}</figcaption>
    </figure>`;
}

function renderAllocation(shape, route, source) {
    const scale = 0.0037;
    const position = counts => counts.reduce((offsets, count, rank) => {
        offsets.push(rank ? offsets[rank - 1] + counts[rank - 1] * scale + 17 : 43);
        return offsets;
    }, []);
    const left = position(route.initialRankRoutes);
    const rightCounts = route.executionCounts.map(slots => slots.reduce((total, n) => total + n, 0));
    const right = position(rightCounts);
    const leftCursor = left.slice();
    const rightCursor = right.slice();
    const flows = [];
    const moves = [];
    route.executionCounts.forEach((slots, dest) => slots.forEach((n, slot) => {
        if (!n) return;
        const expert = slot < shape.home ? dest * shape.home + slot : route.copies[dest][slot - shape.home];
        const owner = Math.floor(expert / shape.home);
        const from = leftCursor[owner];
        const to = rightCursor[dest];
        const h = n * scale;
        leftCursor[owner] += h;
        rightCursor[dest] += h;
        const type = owner === dest ? 'home' : 'replica';
        if (type === 'replica') moves.push(`E${expert} → rank${dest} · ${fmt(n)}`);
        flows.push(`<path class="allocation-${type}" d="M180,${from} C310,${from} 395,${to} 525,${to} v${h} C395,${to+h} 310,${from+h} 180,${from+h} Z"><title>E${expert} · rank${owner} → rank${dest} · ${fmt(n)} 条</title></path>`);
    }));
    const marks = (counts, offsets, x, labelX, anchor) => counts.map((n, rank) => {
        const h = n * scale;
        return `<rect x="${x}" y="${offsets[rank]}" width="10" height="${h}" class="allocation-mark"/>
            <text x="${labelX}" y="${offsets[rank]+h/2+4}" text-anchor="${anchor}">rank${rank} · ${fmt(n)}</text>`;
    }).join('');
    return `<figure class="case-diagram">
        <h4>副本搬走了哪些执行条目</h4>
        ${svg('allocation', 710, 352, '每组路由从逻辑 owner 分配到执行 rank', `<text x="170" y="23" text-anchor="end">逻辑 owner</text><text x="535" y="23">执行 rank</text>
            ${flows.join('')}${marks(route.initialRankRoutes,left,170,161,'end')}${marks(rightCounts,right,525,545,'start')}`)}
        <figcaption>每个 EP 组内的流宽与路由条数成正比。蓝色为 home 执行，橙色为副本执行。rank0 分出的三股是 ${moves.join('、')}。这是执行配额，实际 X 仍从各 token 的来源 rank 发出。 ${source('P14','分配配额')} · ${source('P15','写物理 ID')} · ${source('P17','校验全局计划')}</figcaption>
    </figure>`;
}

function renderDispatch(data, source) {
    const { before, after } = data.dispatchMatrix;
    const matrix = (label, counts) => `<div class="case-dispatch-matrix"><h5>${label}</h5><table class="dispatch-matrix">
        <thead><tr><th scope="col">源 ↘ 目标</th>${counts.map((_, rank) => `<th scope="col">rank${rank}</th>`).join('')}</tr></thead>
        <tbody>${counts.map((row, rank) => `<tr${rank === data.focus.rank ? ' class="dispatch-focus"' : ''}><th scope="row">rank${rank}</th>
        ${row.map(n => `<td style="--intensity:${n / 20000}">${fmt(n)}</td>`).join('')}</tr>`).join('')}</tbody>
    </table></div>`;
    const [r0, r1, r2, r3] = after[data.focus.rank];
    return `<figure class="case-diagram">
        <h4>从真实来源 rank 看这一批 X 往哪发</h4>
        <div class="case-matrices">${matrix('规划前 · 目标为逻辑 owner', before)}${matrix('规划后 · 目标为物理执行 rank', after)}</div>
        <figcaption>按冻结路由与 CPU 重放计数，每行共 16,384 次 TopK 派发。同一色阶用于两表。高亮的源 rank2 规划后分别送出 ${fmt(r0)}、${fmt(r1)}、${fmt(r2)}、${fmt(r3)} 条。第二个 EP4 组复用相同路由，rank 号加 4。 ${source('P15','从逻辑 ID 得到物理 ID')} · ${source('K02','按物理 ID 派发')}</figcaption>
    </figure>`;
}

function renderForwardDependencies(source) {
    return `<figure class="case-diagram">
        <h4>核内依赖 · 哪些数据就绪后才能计算</h4>
        ${svg('dependencies', 1060, 326, 'AIV 派发和搬运、AIC GEMM、SwiGLU 与回源依赖', `
            <text x="24" y="28" class="diagram-head">AIV · 数据准备</text>
            <text x="355" y="28" class="diagram-head">AIC / AIV · 单执行实例</text>
            ${diagramArrows('dependencies', [
                ['activation', 'M259 84 H355'], ['weight', 'M259 172 H322 V103 H355'],
                ['activation', 'M559 87 H630'], ['activation', 'M721 117 V223'],
                ['weight', 'M259 194 H326 V248 H630'],
                ['control', 'M144 117 V138 H299 V279 H630'],
                ['activation', 'M825 256 H863'],
            ])}
            ${diagramNode(24, 52, 235, 65, 'activation', 'K02 · AIV1 派发 X / p', '按物理实例打包并发布就绪')}
            ${diagramNode(24, 150, 235, 65, 'weight', 'K03 · AIV 搬副本 W1 / W2', '搬运和计算可交叠')}
            ${diagramNode(355, 52, 204, 65, 'compute', 'K04 · AIC GMM1', '消费 packed X / W1')}
            ${diagramNode(630, 52, 195, 65, 'compute', 'K05 · AIV SwiGLU', 'gate/up → Hact → Hp')}
            ${diagramNode(630, 223, 195, 68, 'compute', 'K06 · AIC GMM2', '消费 Hp / W2')}
            ${diagramNode(863, 223, 173, 68, 'activation', 'K07 / K08 · 回源', '写回贡献，再合并 TopK')}
            <text x="265" y="77">X ready</text>
            <text x="267" y="166">W1</text>
            <text x="567" y="78">gate/up</text>
            <text x="731" y="176">Hp ready</text>
            <text x="355" y="239">本地 W2 或副本 W2 stripe 就绪</text>
            <text x="355" y="300">虚线 · 本核 AIV1 释放 dispatch UB 后通知 AIC</text>
        `)}
        <figcaption>线表示数据和同步依赖。GMM1 对 X 的等待粒度由 adaptive-auto 分支选择；GMM2 同时受 Hp、W2 与本核 dispatch UB 交接约束。GMM2 的分块回写之后，来源 rank 在同步点后还原 token 并合并两路贡献。图中距离无时间含义。
            ${source('K01','索引与计数')} · ${source('K04','GMM1 入口')} · ${source('K06','GMM2 入口')} · ${source('K08','回源合并')} · ${source('07-kernel-forward','核内时序详图')}</figcaption>
    </figure>`;
}

function renderForward(data, source, table) {
    const { focus, shape } = data;
    const width = 978;
    const paths = focus.logical.map((expert, k) => {
        const phys = focus.physical[k];
        return { k, expert, phys, rank: Math.floor(phys / (shape.home + shape.slots)), slot: phys % (shape.home + shape.slots), owner: Math.floor(expert / shape.home) };
    });
    const labels = paths.map(({k,expert,phys,rank,slot,owner}) => {
        const y = 115 + 140 * k;
        return `<g data-route-slot="${k}" class="learning-original-route"><path class="fwd-data" d="M120 185 C168 185 165 ${y} 218 ${y} H805 C850 ${y} 840 185 880 185"/>
            <path class="fwd-direction" d="M462 ${y-4} l8 4 -8 4 Z M570 ${y-4} l8 4 -8 4 Z M702 ${y-4} l8 4 -8 4 Z"/>
            <path class="fwd-weight" d="M325 ${y-43} v31" marker-end="url(#case-forward-weight)"/>
            <circle class="fwd-port" cx="218" cy="${y}" r="5"/><circle class="fwd-port" cx="805" cy="${y}" r="5"/>
            <text x="214" y="${y-13}">TopK ${k} · ID ${phys}</text>
            <text x="325" y="${y-52}">rank${owner} home W1/W2${owner === rank ? ' · 本地读取' : ' · stripe 拉取'}</text>
            <text x="325" y="${y+26}">rank${rank} · slot ${slot} · E${expert} ${owner === rank ? 'home' : '副本'}</text>
            <text x="490" y="${y-13}">GMM1</text><text x="490" y="${y+26}">Z[6144]</text>
            <text x="601" y="${y-13}">SwiGLU × .5</text><text x="601" y="${y+26}">Hp[3072]</text>
            <text x="734" y="${y-13}">GMM2</text><text x="734" y="${y+26}">Y贡献[7168]</text></g>`;
    }).join('');
    return `<p>在 rank${focus.rank} 的本层输入里取第 ${focus.row} 行。两个 TopK 槽共用一行 X，各有一个 p 和物理目的地。下图只追这一行的依赖，批量执行按实例和 tile 排布。</p>
    <figure class="case-diagram">${svg('forward', width, 327, '一条 token 的 X、p、权重及输出路径', `
        <defs><marker id="case-forward-weight" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L8 4 L0 8z" fill="#7c6450"/></marker></defs>
        <text x="18" y="24" class="diagram-head">源 rank${focus.rank}</text><text x="325" y="24" class="diagram-head">物理实例执行</text><text x="800" y="24" class="diagram-head">回源 rank${focus.rank}</text>
        ${labels}
        <circle class="fwd-port" cx="120" cy="185" r="5"/><circle class="fwd-port" cx="880" cy="185" r="5"/>
        <text x="17" y="173">X[${focus.row}, :]</text><text x="17" y="199">p=[.5,.5]</text>
        <text x="890" y="180">combine</text><text x="890" y="204">routed Y[${focus.row}]</text>
    `)}<figcaption>实线是激活及其结果，虚线是当前层权重。combine 还原 token 位置并对两路结果求和，模型随后加上同一输入的 shared MLP 输出。 ${source('K02','派发')} · ${source('K03','权重 stripe')} · ${source('K05','先乘 p')} · ${source('K08','合并')} · ${source('F16','shared 相加')}</figcaption></figure>
    <p class="case-inline-note">对某个执行实例的 m 行，数学依赖是 <code>X[m,H] · W1ᵀ → [gate,up][m,2F] → Hact[m,F] → Hp[m,F] → Hp · W2ᵀ</code>。本例 <code>L=${shape.limit}</code>，先用 <code>gateC=min(gate,L)</code> 和 <code>upC=clamp(up,−L,L)</code> 计算 <code>SiLU(gateC) ⊙ upC</code> 并舍入为 BF16 的 Hact。乘 p 后再次舍入为 BF16 的 Hp，供 GMM2 读取。BF16 保存态保留限幅前的 gate/up。</p>
    ${renderForwardDependencies(source)}
    <h4>一批张量从 Python 到设备，再回到 autograd</h4>
    ${table(['交接点', '数据合同（每 rank）', '地址、生命周期与代码'], [
        ['模型 → dispatcher', 'X[8192,7168] · BF16；logical_ids / prob[8192,2]', `MoE block 先将 X 展成二维传入 experts。dispatcher 规范化路由并把 ids 转成 int32，设备路径使用 BF16 路由权重。 ${source('F03','专家入口')} · ${source('F06','进 bridge')}`],
        ['Expert-FSDP → SHMEM', 'W1[2,6144,7168]、W2[2,7168,3072] · BF16', `bridge 从 live Parameter 取当前 home 权重，拷入进程级执行池。同一 pool 可复用，各层权重每次发布。 ${source('F08','发布 home')}`],
        ['Planner → C++', 'physical_ids[8192,2]、copies[4,2]、logical_counts[8]、控制 header', `申请 epoch/lease，验证计划后进入 Torch class。C++ 配置 tiling 和输出 buffer，launcher 按 descriptor 将工作提交到当前 NPU stream。 ${source('F09','epoch/lease')} · ${source('N02','Torch class')} · ${source('N06','设备提交')}`],
        ['Kernel → 模型', 'Y[8192,7168] · BF16', `Host 提交后返回 tensor handle，设备按 stream 依赖推进。dispatcher 返回与传入 X 同形的二维 Y，MoE block 恢复 [batch,seq,hidden] 后加 shared 输出。 ${source('N11','返回张量')} · ${source('F16','模型合并')}`],
        ['Kernel / bridge → 本层 tape', 'prob、counts、BF16 state、dispatch、placement、header', `保存 gate/up 和派发输入等反向所需状态。复制 placement 防止其他层复用 scratch 时覆盖，释放本次计划 lease。 ${source('F13','复制计划')} · ${source('F14','释放 lease')} · ${source('F15','保存 tape')}`],
    ])}
    <p class="case-inline-note">图中的 token 落点来自冻结路由的 CPU 重放。八卡物理 ID SHA 与单算子设备记录一致。整网 TimerON 提供同层执行行数；这两条线的宽度只区分数据类型。</p>`;
}

module.exports = { renderStep, renderPlannerControl, renderAllocation, renderDispatch, renderForward };
