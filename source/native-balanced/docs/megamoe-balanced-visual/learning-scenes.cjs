const evidence = require('./source-evidence.json');
const config = require('./learning-scenes.json');

function sceneData(data) {
    const { focus, shape, route } = data;
    const ranks = shape.groups.find(group => group.includes(focus.rank));
    const stride = shape.home + shape.slots;
    const routes = focus.logical.map((expert, slot) => {
        const physical = focus.physical[slot];
        const localRank = Math.floor(physical / stride);
        const localSlot = physical % stride;
        return { slot, expert, physical, rank: ranks[localRank], localSlot,
            owner: ranks[Math.floor(expert / shape.home)], replica: localSlot >= shape.home,
            rows: route.executionCounts[localRank][localSlot], probability: route.probability,
            shared: route.copies.some(copies => copies.includes(expert)) };
    });
    const endpoint = key => {
        const file = evidence.files[key];
        const current = file.path.startsWith('ascend_deepep_hkq/');
        return { path: file.path, sha256: file.sha256, kind: current ? '当前算子仓源码' : '随文档固定的框架 / 实验快照' };
    };
    return { ...config, focus, shape, routes, date: data.date,
        sourceHeads: evidence.heads, experimentCommits: data.sourceCommits,
        references: Object.fromEntries(evidence.rows.map(row => [row.id, {
            detail: row.detail, from: { ...endpoint(row.from), line: row.fromLine },
            to: { ...endpoint(row.to), line: row.toLine },
        }])) };
}

function renderScenes(data, esc) {
    const model = sceneData(data);
    const select = (id, label) => `<button type="button" data-learning-callpoint="${id}">${label}</button>`;
    const routeCards = model.routes.map(route => `<button type="button" class="learning-route-card" data-route-slot="${route.slot}" data-learning-slot="${route.slot}">
        <strong>TopK ${route.slot} · E${route.expert}</strong><span>物理 ID ${route.physical} → rank${route.rank} / slot${route.localSlot}</span>
        <span>${route.replica ? '副本' : 'home'} · owner rank${route.owner} · p=${route.probability}</span></button>`).join('');
    const payload = JSON.stringify(model).replaceAll('<', '\\u003c');
    return `<div class="learning-workbench" id="learning-workbench">
    <p class="learning-intro">同一个样本，换三个问题阅读。选择 TopK 槽会同时高亮本节与下方原有前后向图。样本来自 ${esc(model.date)} 冻结路由重放，代码引用区分本仓实现与随文档保存的跨仓快照。</p>
    <div class="learning-context" aria-live="polite"><strong>step ${model.focus.step} · rank${model.focus.rank} · X[${model.focus.row}] / dY[${model.focus.row}]</strong><span id="learning-context-route"></span></div>
    <div class="learning-slot-controls" role="group" aria-label="选择当前 token 的 TopK 槽">${routeCards}</div>
    <div class="learning-tabs" role="tablist" aria-label="三个阅读场景">${model.scenes.map((scene, index) => `<button type="button" role="tab" id="learning-tab-${scene.id}" aria-controls="learning-scene-${scene.id}" aria-selected="${index === 0}" data-learning-scene="${scene.id}">${scene.title}</button>`).join('')}</div>
    <div class="learning-layout"><div class="learning-content">
    <section id="learning-scene-token" class="learning-scene" role="tabpanel" aria-labelledby="learning-tab-token">
      <h4>同一条输入，两条独立路由</h4>
      <p>两条路由读取的 X 相同；反向读取的 dY 也相同，各自拥有不同的派发地址。p 在专家 SwiGLU 路径使用；反向派发的是原始 dY。</p>
      <div class="learning-token-paths">${model.routes.map(route => `<article class="learning-token-path" data-route-slot="${route.slot}">
        <h5>TopK ${route.slot} · E${route.expert} ${route.replica ? '副本' : 'home'}</h5>
        <div class="learning-path"><span>源 rank${model.focus.rank}<br>X[${model.focus.row}]</span><b>→</b><span>rank${route.rank} / slot${route.localSlot}<br>GMM1 → SwiGLU × ${route.probability} → GMM2</span><b>→</b><span>源 rank${model.focus.rank}<br>Y 路由贡献</span></div>
        <div class="learning-path"><span>源 rank${model.focus.rank}<br>dY[${model.focus.row}]</span><b>→</b><span>相同 physical ID ${route.physical}<br>dHp → dZ → dX / dp / dW</span><b>→</b><span>dX / dp 回 rank${model.focus.rank}<br>dW 归 owner rank${route.owner}</span></div>
        <p>该执行实例共 ${route.rows.toLocaleString('en-US')} 行；dW 是整段专家行的贡献，不只是这一 token。</p>
        <div class="learning-links">${select('K02', '看 X / p 派发')}${select('G01', '看原始 dY 派发')}${select('G10', '看 dX 回源')}${select('G14', '看 dW 归主')}</div>
      </article>`).join('')}</div>
      <p class="learning-equation">最终 dX[${model.focus.row}] = ${model.routes.map(route => `TopK${route.slot} 的 dX`).join(' + ')}；dp 保持 [${model.focus.row}, TopK槽]，两路不会相加成一个 dp。</p>
      <div class="learning-links"><a href="#case-kernel">去原有前向详图 ↓</a><a href="#case-gradients">去原有反向详图 ↓</a></div>
    </section>
    <section id="learning-scene-tape" class="learning-scene" role="tabpanel" aria-labelledby="learning-tab-tape" hidden>
      <h4>前向留下什么，反向由谁读取</h4>
      <p>tape 属于这一层；执行池权重属于进程级复用空间，反向从本层 live Parameter 重新发布 W1/W2。</p>
      <div class="learning-tape">
        <div class="learning-tape-head"><strong>前向生产 →</strong><strong>本层保存 →</strong><strong>反向消费</strong></div>
        ${model.tape.map(item => `<article class="learning-tape-row">
          <button type="button" class="learning-producer" data-learning-callpoint="${item.id === 'placement' ? 'F13' : 'F15'}"><small>前向生产</small><strong>${item.id === 'placement' ? '冻结路由落点' : 'save_for_backward'}</strong><span>${item.id === 'placement' ? 'clone placement / header' : '写入本层 autograd tape'}</span></button>
          <div class="learning-saved"><small>本层保存</small><strong>${item.title}</strong><span>${item.saved}</span></div>
          <button type="button" class="learning-consumer" data-learning-callpoint="${item.callpoint}"><small>反向消费</small><strong>${item.callpoint} · 消费入口</strong><span>${item.used}</span></button>
        </article>`).join('')}
      </div>
      <div class="learning-links">${select('F13', 'placement clone')}${select('F15', 'save_for_backward')}${select('B03', '当前层 live 权重')}${select('B04', '重新发布 home')}</div>
      <h4>同一地址，分段持有不同数据</h4>
      <div class="learning-buffer-tabs" role="group" aria-label="选择复用缓冲">${model.buffers.map((buffer, index) => `<button type="button" data-learning-buffer="${buffer.id}" aria-pressed="${index === 0}">${buffer.title}</button>`).join('')}</div>
      <div id="learning-buffer-view"></div>
    </section>
    <section id="learning-scene-pipeline" class="learning-scene" role="tabpanel" aria-labelledby="learning-tab-pipeline" hidden>
      <h4>GM 主线：多个 tile、专家退休与 UB 复用</h4>
      <p>每列是依赖检查点，不表示耗时。此处展开已有 kernelSchedule 的 BF16 saved / panel / early dW1 实验主线。</p>
      <div class="learning-pipeline-controls"><button type="button" id="learning-previous">← 上一检查点</button><output id="learning-step-label"></output><button type="button" id="learning-next">下一检查点 →</button></div>
      <div class="learning-pipeline-scroll" tabindex="0" aria-label="多 tile 依赖检查点"><div id="learning-pipeline-grid"></div></div>
      <div class="learning-checkpoint" aria-live="polite"><h5 id="learning-checkpoint-title"></h5><p id="learning-checkpoint-text"></p><button type="button" id="learning-checkpoint-source">查看本检查点源码</button></div>
      <div class="learning-links"><a href="#case-owner">对照既有调度表与实测打点 ↓</a></div>
    </section>
    </div><aside class="learning-inspector" aria-label="当前选择和常驻源码详情">
      <div class="learning-inspector-context" id="learning-inspector-context"></div>
      <h4>对应源码</h4><div id="learning-source-detail" aria-live="polite"></div>
    </aside></div>
    <script id="learning-scenes-data" type="application/json">${payload}</script>
    </div>`;
}

module.exports = { renderScenes, sceneData };
