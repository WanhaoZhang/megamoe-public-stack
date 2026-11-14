const scenarioPath = '../../integration/wuhan/scenarios/python-olmoe-high-skew/';
const profile = require(scenarioPath + 'source-profile.json');
const fmt = new Intl.NumberFormat('en-US').format;

const callpoint = (id, label) => `<a href="#${id}" data-editor-callpoint="${id}" title="在 VS Code 打开实现行">${label}</a>`;

function plotCounts(counts, rows, selected, label) {
    const left = 42;
    const width = 346;
    const top = 12;
    const bottom = 118;
    const gap = width / counts.length;
    const bars = counts.map((count, expert) => {
        const h = (bottom - top) * count / rows;
        return `<rect class="${expert === selected ? 'peak' : 'regular'}" x="${left + expert * gap + gap * .1}" y="${bottom - h}" width="${gap * .8}" height="${h}">
            <title>E${expert} · ${fmt(count)} 次 · ${(100 * count / rows).toFixed(1)}% token</title></rect>`;
    });
    const labels = counts.map((_, expert) => expert).filter(expert => (
        counts.length <= 8 || expert === selected || expert === counts.length - 1
        || (expert % 8 === 0 && Math.abs(expert - selected) >= 5)
    )).map(expert => `<text x="${left + (expert + .5) * gap}" y="134" text-anchor="middle">E${expert}</text>`);
    const grid = [0, 50, 100].map(percent => {
        const y = bottom - (bottom - top) * percent / 100;
        return `<line x1="${left}" x2="${left + width}" y1="${y}" y2="${y}" />
            <text x="${left - 5}" y="${y + 4}" text-anchor="end">${percent}%</text>`;
    });
    return `<svg class="case-input-hist" role="img" aria-label="${label}。最热专家 E${selected} 命中 ${fmt(counts[selected])} 个 token" viewBox="0 0 396 140">
        <g class="case-input-axis">${grid.join('')}${labels.join('')}</g>${bars.join('')}
    </svg>`;
}

function inputLineage(dataset, shape, sourceSample, route, esc) {
    return `<figure class="case-input-figure" aria-label="同一批 Python 原文经两个处理分支在 MoE 专家入口汇合">
        <div class="case-input-origin">
            <span>同一文本来源</span>
            <strong>StarCoder Python · 官方训练混合数据的首批 ${fmt(dataset.documents)} 篇文档</strong>
            <small>两路共用原始文本；各自用对应的 tokenizer 重新编码。</small>
        </div>
        <div class="case-input-streams">
            <div class="case-input-stream" data-stream="model">
                <h4>训练数据 → 激活 <code>X</code></h4>
                <ol>
                    <li><strong>目标 DeepSeek tokenizer</strong><span>保留原顺序，每篇文档追加一个 EOS。共 ${fmt(dataset.tokens)} 个 token。</span></li>
                    <li><strong>索引数据集</strong><span><code>python_text_document.bin</code> 保存 int32 token ID，<code>.idx</code> 保存索引。 ${callpoint('S03', 'MegatronDataManager')}</span></li>
                    <li><strong>每卡训练 batch</strong><span>一次取 <code>[1, ${shape.M}]</code> 的序列；主干和 MTP 消费真实训练输入。 ${callpoint('S06', 'Trainer 取 batch')}</span></li>
                    <li><strong>MoE 层真实激活</strong><span>decoder 在 attention 和后续归一化后进入 MoE，专家输入展平为 BF16 <code>X[${shape.M}, ${shape.H}]</code>。模型 gate 照常计算。 ${callpoint('S07', 'decoder 调用 MoE')} · ${callpoint('F03', '专家入口')}</span></li>
                </ol>
            </div>
            <div class="case-input-stream" data-stream="route">
                <h4>采样负载 → 专家 <code>ids / p</code></h4>
                <ol>
                    <li><strong>${esc(sourceSample.model)}</strong><span>使用 OLMoE tokenizer 和预训练权重，以训练模式做新的路由采样。取 event ${sourceSample.event}、layer ${sourceSample.layer}（从 0 编号）。</span></li>
                    <li><strong>取专家计数直方图</strong><span>${profile.provenance.contexts_per_event} 段 × ${fmt(profile.provenance.sequence_length)} token，E${sourceSample.experts} / Top${sourceSample.topk}，共 ${fmt(sourceSample.tokens * sourceSample.topk)} 次专家选择。</span></li>
                    <li><strong>CPU 重建固定路由</strong><span>E${shape.E} / Top${shape.TopK}，seed ${route.seed}，生成 4 份 <code>rank0–3-ids.npy</code>。每份 <code>[${shape.M}, ${shape.TopK}]</code>，两组 EP 复用。</span></li>
                    <li><strong>实验入口替换 gate 输出</strong><span>换入冻结 <code>ids</code> 与常量 <code>p=${route.probability}</code>。 ${callpoint('S08', 'fixed_routes')}</span></li>
                </ol>
            </div>
        </div>
        <div class="case-input-join"><strong>两路在 FSDPTurbo dispatcher 汇合</strong>
            <span>真实 <code>X</code> + 冻结 <code>ids / p</code> + Expert-FSDP 解分片后的 <code>W1 / W2</code> 一起交给 MegaMoE。</span>
        </div>
        <figcaption>整网按固定专家选择执行真实的激活计算和参数更新。OLMoE 贡献的是 64 个专家的选中次数，目标路由的逐 token 组合由 CPU 重新构造。</figcaption>
    </figure>`;
}

function distribution(shape, sourceSample, route) {
    const targetRows = shape.ep * shape.M;
    const bucketSize = sourceSample.experts / shape.E;
    const preCap = [...profile.counts].sort((a, b) => b - a).slice(0, bucketSize)
        .reduce((total, count) => total + count, 0) * targetRows * shape.TopK / (sourceSample.tokens * sourceSample.topk);
    const targetPeak = route.logicalCounts.indexOf(Math.max(...route.logicalCounts));
    return `<figure class="case-input-figure case-input-distribution">
        <h4>从 E${sourceSample.experts} 采样到 E${shape.E} 冻结路由</h4>
        <p>柱高是专家命中的 token 占比。两图各按自身的 token 数归一化，纵轴统一为 0–100%。</p>
        <div class="case-input-charts">
            <div>
                <strong>源采样 · E${sourceSample.experts} / Top${sourceSample.topk}</strong>
                ${plotCounts(profile.counts, sourceSample.tokens, sourceSample.hottestExpert, 'OLMoE 64 个专家命中率')}
                <span>E${sourceSample.hottestExpert} 命中 ${fmt(sourceSample.hottestCount)} / ${fmt(sourceSample.tokens)} 个 token。κ = ${sourceSample.kappa.toFixed(3)}。</span>
            </div>
            <div>
                <strong>每 EP 组 · E${shape.E} / Top${shape.TopK}</strong>
                ${plotCounts(route.logicalCounts, targetRows, targetPeak, '固定 8 专家命中率')}
                <span>E${targetPeak} 命中 ${fmt(route.logicalCounts[targetPeak])} / ${fmt(targetRows)} 个 token。κ = ${route.kappa.toFixed(3)}。</span>
            </div>
        </div>
        <ol class="case-input-transform" aria-label="固定路由生成规则">
            <li><strong>排序并归桶</strong><span>按命中次数从高到低排序，每 ${bucketSize} 个源专家积分成 1 桶；按目标总量 ${fmt(targetRows * shape.TopK)} 缩放。</span></li>
            <li><strong>限制单专家配额</strong><span>首桶 ${fmt(preCap)} 超过组内 token 数 ${fmt(targetRows)}。向其余桶按权重回填 ${fmt(preCap - targetRows)}，再按余数补齐整数。</span></li>
            <li><strong>冻结 Top2 组合</strong><span>seed ${route.seed} 置换专家 ID，按配额循环发放，再打散 token 行并切四份；每行两个不同专家。</span></li>
        </ol>
        <figcaption><code>κ = E × Σ count[e]² / (Σ count[e])²</code>；κ 衡量逻辑专家偏斜，下一节 Planner 图看的是执行卡负载。 <a href="${scenarioPath}source-profile.json">源计数与采样证据</a> · <a href="${scenarioPath}generate.py">CPU 生成器源码</a></figcaption>
    </figure>`;
}

function topology(shape) {
    const home = shape.home;
    const row = ranks => ranks.map((rank, epRank) => `<div class="case-rank" data-rank="${rank}">
        <strong>rank${rank}</strong><span>E${epRank * home}/E${epRank * home + 1}</span>
    </div>`).join('');
    return `<figure class="case-input-figure case-input-topology" aria-label="整网并行组拓扑">
        <h4>两条横排跑 routed MoE，四个纵列分片专家权重</h4>
        <div class="case-input-mesh" style="--case-input-ep:${shape.ep}">
            <div class="case-input-mesh-head"><strong>普通 FSDP${shape.world}</strong><span>覆盖全部 rank，管理 attention、shared MLP 等非 routed 参数。</span></div>
            <div class="case-input-ep-title">EP 组 A <span>dispatch / combine / 归主</span></div>
            <div class="case-input-mesh-row" aria-label="EP 组 A">${row(shape.groups[0])}</div>
            <div class="case-input-cross" aria-label="每一列构成一个 Expert-FSDP 组">${shape.groups[0].map(() => '<span aria-hidden="true"></span>').join('')}</div>
            <div class="case-input-mesh-row" aria-label="EP 组 B">${row(shape.groups[1])}</div>
            <div class="case-input-ep-title">EP 组 B <span>同一组路由配方</span></div>
        </div>
        <figcaption><b>横向 EP${shape.ep}</b> 是 planner 放置和 token 派发的范围，副本梯度先归 home。<b>纵向 Expert-FSDP2</b> 让相同 EP 位置分片同一组 home 参数，例如 rank0 ↕ rank4 同管 E0/E1。<b>外框普通 FSDP${shape.world}</b> 用于整网其余参数。 ${callpoint('I14', 'init_parallel_state')}</figcaption>
    </figure>`;
}

function settings(shape) {
    return `<div class="case-input-settings" role="region" aria-label="整网与单算子输入的区别">
        <table>
            <thead><tr><th scope="col">项目</th><th scope="col">整网短跑</th><th scope="col">单算子回放</th></tr></thead>
            <tbody>
                <tr><th scope="row">X、W、dY</th><td>真实 Python batch；随机初始化的 8 个主干层 + 1 个 MTP 层</td><td>CPU 固定 seed <code>42 + rank</code> 构造 BF16 张量</td></tr>
                <tr><th scope="row">执行范围</th><td>FSDP8 + EP4 × Expert-FSDP2；batch 1/rank，梯度累积 1；全层非重入 checkpoint，Muon</td><td>两组独立 EP4；只运行算子前后向</td></tr>
                <tr><th scope="row">共同输入</th><td colspan="2">M=${shape.M}，H=${shape.H}，F=${shape.F}，E=${shape.E}，TopK=${shape.TopK}；BF16 canonical 权重与 BF16 saved state</td></tr>
                <tr><th scope="row">每 rank 容量</th><td colspan="2">${shape.home} 个 home 槽 + ${shape.slots} 个 replica 槽；行容量参数 ${fmt(shape.capacity)}；SwiGLU limit=${shape.limit}</td></tr>
            </tbody>
        </table>
    </div>`;
}

function renderInputs(data, esc) {
    const { dataset, shape, sourceSample, route } = data;
    return inputLineage(dataset, shape, sourceSample, route, esc)
        + distribution(shape, sourceSample, route) + topology(shape) + settings(shape);
}

module.exports = { renderInputs };
