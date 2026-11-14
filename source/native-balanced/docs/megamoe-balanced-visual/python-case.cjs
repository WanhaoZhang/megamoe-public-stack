const data = require('./python-case.json');
const { renderInputs } = require('./case-inputs.cjs');
const { renderStep, renderPlannerControl, renderAllocation, renderDispatch, renderForward } = require('./case-forward.cjs');
const { renderBackward, renderUpdate } = require('./case-backward.cjs');
const { mermaidImage } = require('./map-guide.cjs');
const { renderScenes } = require('./learning-scenes.cjs');

const caseId = '12-python-case';
const fmt = new Intl.NumberFormat('en-US').format;
const sum = values => values.reduce((total, value) => total + value, 0);
const source = (id, label) => {
    const editor = /^[A-Z]\d{2}$/.test(id) ? ` data-editor-callpoint="${id}" title="在 VS Code 打开实现行"` : '';
    return `<a href="#${id}"${editor}>${label}</a>`;
};
const figure = (label, body, caption = '') =>
    `<figure class="case-figure" aria-label="${label}">${body}${caption ? `<figcaption>${caption}</figcaption>` : ''}</figure>`;
const table = (headers, rows) => `<div class="case-table" tabindex="0" role="region" aria-label="${headers[0]}对照表">
    <table><thead><tr>${headers.map(cell => `<th scope="col">${cell}</th>`).join('')}</tr></thead>
    <tbody>${rows.map(row => `<tr><th scope="row">${row[0]}</th>${row.slice(1).map(cell => `<td>${cell}</td>`).join('')}</tr>`).join('')}</tbody></table>
    </div>`;

function loadChart(shape, route) {
    const max = Math.ceil(Math.max(...route.initialRankRoutes) / 10000) * 10000;
    const bar = (value, type) =>
        `<span class="case-bar ${type}" style="--width:${100 * value / max}%" title="${fmt(value)} 条路由"></span>`;
    const bars = route.executionCounts.map((counts, rank) => {
        const initial = route.initialRankRoutes[rank];
        const home = sum(counts.slice(0, shape.home));
        const replica = sum(counts.slice(shape.home));
        return `<div class="case-load-row">
            <strong>${rank} / ${rank + shape.ep}</strong>
            <div><div class="case-bar-line" aria-label="规划前 ${fmt(initial)} 条路由"><span>前</span><div>${bar(initial, 'before')}</div><b>${fmt(initial)}</b></div>
            <div class="case-bar-line" aria-label="规划后 ${fmt(home)} 条 home 路由和 ${fmt(replica)} 条副本路由"><span>后</span><div>${bar(home, 'home')}${replica ? bar(replica, 'replica') : ''}</div><b>${fmt(home + replica)}</b></div></div>
        </div>`;
    });
    const slots = route.executionCounts.map((counts, rank) => [
        `rank${rank} / ${rank + shape.ep}`,
        counts.slice(0, shape.home).map((value, slot) => `E${rank * shape.home + slot} · ${fmt(value)}`).join(' / '),
        route.copies[rank].map((expert, slot) => expert < 0 ? '' : `E${expert} · ${fmt(counts[shape.home + slot])}`).filter(Boolean).join(' / ') || '—',
        fmt(sum(counts)),
    ]);
    return figure('MoonEP 规划前后的每卡执行量', `<div class="case-load">
        <div class="case-chart-key"><span class="before">规划前</span><span class="home">规划后 · home</span><span class="replica">规划后 · 副本</span></div>
        ${bars.join('')}
        <div class="case-axis"><span>0</span><span>${fmt(max / 2)}</span><span>${fmt(max)} 条路由</span></div>
    </div>`, '每组共有 65,536 条路由。最重卡 / 平均值从 2.736 降到 1，逻辑专家计数和 κ 保持一致。')
        + table(['全局 rank（两组）', 'home 执行条目', '活动副本执行条目', '每卡合计'], slots);
}

function timingPlot(title, measurements, maximum) {
    return `<figure class="case-diagram case-timing">
        <h4>${title}</h4>
        <div class="timing-key">● p50　◇ p90　·　横轴单位 ms</div>
        ${measurements.map(([name, timing, scope]) => `<div class="timing-row">
            <div><strong>${name}</strong><small>${scope}</small></div>
            <div><div class="timing-track" style="--p50:${timing.p50 / maximum * 100}%;--p90:${timing.p90 / maximum * 100}%">
                <span class="timing-median"></span><span class="timing-p90"></span>
            </div><div class="timing-axis"><span>0</span><span>${maximum / 2}</span><span>${maximum}</span></div></div>
            <div class="timing-values"><strong>${timing.p50.toFixed(3)} ms</strong><span>P90 ${timing.p90.toFixed(3)} ms</span></div>
        </div>`).join('')}
    </figure>`;
}

function caseMermaid(id, description, calls, esc) {
    return `<figure class="case-diagram">
        <div class="map-canvas" tabindex="0" role="region" aria-label="${esc(description)}">
            ${mermaidImage(id, description, esc)}
        </div>
        <figcaption>${calls.map(([code, title]) => source(code, title)).join(' · ')}
            · <a href="maps/${id}.mmd">Mermaid 源码</a></figcaption>
    </figure>`;
}

function renderPythonCase(esc) {
    const { focus, shape, route, sourceCommits, wholeTiming, operatorTiming, evidence } = data;
    const chapters = [
        ['conditions', '实验数据与配置'],
        ['step', '整体训练流程'],
        ['scenes', '交互阅读场景'],
        ['forward', '前向', [
            ['forward-entry', '框架交接'],
            ['planner', 'Planner 与校验'],
            ['kernel', 'MegaMoE 派发与计算'],
            ['tape', '保存反向状态'],
        ]],
        ['backward', '反向与梯度', [
            ['backward-replay', '恢复与回放'],
            ['gradients', 'dX、dp、dW 的去向'],
            ['owner', 'EP 归主与打点'],
        ]],
        ['update', 'FSDP 同步与更新'],
        ['evidence', '计时与证据'],
    ];
    const heading = key => {
        const index = chapters.findIndex(([id]) => id === key);
        return `<h3 id="case-${key}"><span>0${index + 1}</span>${chapters[index][1]}</h3>`;
    };
    return `<article class="panel case-page" id="${caseId}" data-chapter="${caseId}">
    <header class="case-header">
        <p class="case-kicker">${data.date} · ${data.branch} · 八卡实测</p>
        <h2>八卡 Python 实验 · 一步训练的数据流</h2>
        <p>从真实文本与冻结负载读起，沿整批张量追到参数更新。再放大 step ${focus.step}，跟踪 <strong>rank${focus.rank} 的 <code>X[${focus.row}]</code></strong> 穿过三个库。</p>
    </header>
    <div class="case-layout">
    <aside class="case-sidebar">
        <strong class="case-sidebar-title">实验目录</strong>
        <button type="button" class="case-menu-toggle" aria-expanded="false" aria-controls="case-toc">目录 · <span>实验数据与配置</span><span aria-hidden="true">▾</span></button>
        <div class="case-toc" id="case-toc" role="navigation" aria-label="实验实例目录">
            ${chapters.map(([id, title, children], index) => `<div class="case-toc-group">
                <a href="#case-${id}" data-title="${title}"><span aria-hidden="true">0${index+1}</span>${title}</a>
                ${children ? `<div class="case-toc-sub">${children.map(([subId, subTitle]) => `<a href="#case-${subId}" data-title="${subTitle}">${subTitle}</a>`).join('')}</div>` : ''}
            </div>`).join('')}
        </div>
    </aside>
    <div class="case-content">
    <section>
        ${heading('conditions')}
        ${renderInputs(data, esc)}
    </section>

    <section>
        ${heading('step')}
        <p>先看三个库如何衔接，再用实测 Host scope 对照各层的前向、重算和反向。 ${source('training-workflow', '打开三库训练泳道')}</p>
        ${renderStep(data, source)}
    </section>

    <section>
        ${heading('scenes')}
        ${renderScenes(data, esc)}
    </section>

    <section>
        ${heading('forward')}
        <h4 id="case-forward-entry">模型与参数如何交给 Native</h4>
        ${caseMermaid('case-forward-entry', '模型、Expert-FSDP、dispatcher 与 bridge 的调用顺序', [
            ['F18', '当前参数就绪'], ['S08', '实验路由覆盖'], ['F05', '共享执行池'], ['F08', '刷新 home'],
        ], esc)}
        <h4 id="case-planner">Planner 与校验</h4>
        ${renderPlannerControl(data, source)}
        <p>组内 rank0 持有 E0/E1，规划前负载 44,819 条。MoonEP 根据每卡缺口分配行配额，给 rank1/2 各放一个 E1 副本，给 rank3 放一个 E0 副本。<strong>物理 ID = 组内执行 rank × 4 + 本地 slot</strong>。前两个 slot 是 home，后两个是副本槽。 ${source('03-planner', '看规划详图')}</p>
        ${loadChart(shape, route)}
        ${renderAllocation(shape, route, source)}
        ${table(['逻辑专家', ...route.logicalCounts.map((_, expert) => 'E' + expert)], [
            ['每 EP 组路由条数', ...route.logicalCounts.map(fmt)],
        ])}
        <h4 id="case-kernel">MegaMoE 派发、专家计算与合并</h4>
        ${renderDispatch(data, source) + renderForward(data, source, table)}
        <h4 id="case-tape">保存反向状态</h4>
        <p>前向提交后，bridge 在同一 stream 冻结 placement 和 header，再释放计划租约。checkpoint 如需重算，会再次走前向规划；实际算子反向复用 tape 里的物理路由，并给本次回放写入新 epoch。</p>
        ${caseMermaid('case-tape', '本层 tape 的来源、checkpoint 重算与 Native 反向回放', [
            ['F13', '冻结计划'], ['F14', '释放租约'], ['F15', '保存 tape'], ['U01', '按需重算'], ['B07', 'placement 回放'],
        ], esc)}
    </section>

    <section>
        ${heading('backward')}
        ${renderBackward(data, source)}
    </section>

    <section>
        ${heading('update')}
        ${renderUpdate(data, source)}
    </section>

    <section>
        ${heading('evidence')}
        <p>计时汇总用 p50 与 p90 端点。单算子 B 和 FB 共用横轴，整网 step 单列。</p>
        ${timingPlot('单算子 TimerOFF', [
            ['完整 B', operatorTiming.backward, '从反向提交到恢复、计算、归主及完成。'],
            ['直接 FB', operatorTiming.fb, '前向含 home 刷新，紧接完整反向。'],
        ], 30)}
        ${timingPlot('整网完整 step · OFF', [
            ['完整 step', wholeTiming, `step ${wholeTiming.steps}，${wholeTiming.samples} 个样本，取 batch 至更新与清梯度完成。`],
        ], 3000)}
        <p>两类计时均逐样本取八 rank 最大 wall。单算子预热 ${operatorTiming.fb.warmup} 次、测量 ${operatorTiming.fb.samples} 次。图中的箭头表示依赖，整网 step 10 的 TimerON 记录用于核对阶段和行数。</p>
        <ul>
            <li><strong>整网实测</strong>　八 rank 的前向和重算执行计数都与 placement 表一致。</li>
            <li><strong>冻结输入 + CPU 重放</strong>　从四份逻辑路由重放 MoonEP，八卡物理路由 SHA 与单算子设备记录一致。示例 token 的具体落点来自该重放。</li>
            <li><strong>源码</strong>　通过图中的编号查看桥接、核内和更新调用点。当前网页源码快照与下方实测版本分别记录。</li>
        </ul>
        ${table(['整网实测仓库', '固定提交'], Object.entries(sourceCommits).map(([name, commit]) => [esc(name), `<code>${commit}</code>`]))}
        <p><strong>整网构建</strong>　配对 OFF/ON；early dW1=1、P1/DX cache=1、owner ratio=2、panel restore=2、operand-ready=1、flags=0、owner=16、compact=0，GMM1 adaptive-auto / SwiGLU auto。</p>
        <p><a href="python-case.json">实验清单（含每 rank 路由 SHA）</a>。下列路径相对于原实验工作区，清单内记录了每份证据的 SHA256。</p>
        ${table(['证据', '项目内位置'], [
            ['采样与构造', evidence.sample.path], ['冻结路由', evidence.routes.path],
            ['整网构建', evidence.build.path], ['整网计时', evidence.wholeTiming.path],
            ['step 10 框架 Host scope', evidence.hostScopes.path], ['step 10 Timer 导出索引', evidence.wholeTimer.path],
            ['单算子验收与计时', evidence.operatorTiming.path],
        ].map(([label, value]) => [label, `<code>${esc(value)}</code>`]))}
    </section>
    <p class="case-return">${source('case-conditions', '↑ 回到实验数据')} · ${source('00-lifecycle', '回到生命周期总览')}</p>
    </div>
    </div>
    </article>`;
}

module.exports = { caseId, renderPythonCase };
