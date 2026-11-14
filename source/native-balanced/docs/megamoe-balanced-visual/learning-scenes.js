(() => {
    'use strict';
    const data = JSON.parse(document.getElementById('learning-scenes-data').textContent);
    const root = document.getElementById('learning-workbench');
    const byId = id => document.getElementById(id);
    const esc = value => String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
    const state = { scene: data.scenes[0].id, slot: 0, buffer: data.buffers[0].id, checkpoint: 0, callpoint: data.scenes[0].callpoint };
    function selectSource(id) {
        const reference = data.references[id];
        if (!reference) return;
        state.callpoint = id;
        document.querySelectorAll('.learning-source-selected').forEach(node => node.classList.remove('learning-source-selected'));
        root.querySelectorAll('[data-learning-callpoint]').forEach(node => node.classList.toggle('learning-source-selected', node.dataset.learningCallpoint === id));
        document.querySelectorAll('svg [data-callpoint]').forEach(node => node.classList.toggle('learning-callpoint-selected', node.dataset.callpoint === id));
        const detail = document.getElementById(id);
        const excerpts = detail ? [...detail.querySelectorAll('pre')].map(pre => pre.textContent) : [];
        const endpoint = reference.to;
        byId('learning-source-detail').innerHTML = `<strong>${esc(id)}</strong><p class="learning-source-summary">${esc(reference.detail)}</p>
            <span class="learning-source-kind">${esc(endpoint.kind)} · 采集指纹 ${endpoint.sha256.slice(0, 12)}</span><p class="learning-source-path">${esc(endpoint.path)}:${endpoint.line}</p>
            <div class="learning-source-actions"><button type="button" data-learning-open-source="to">打开实现</button><button type="button" data-learning-open-source="from">打开调用处</button></div>
            <pre>${esc(excerpts.at(-1) || '完整实现通过上方按钮打开。')}</pre>`;
    }
    function chooseSlot(slot) {
        if (!Number.isInteger(slot) || !data.routes[slot]) return;
        state.slot = slot;
        const route = data.routes[slot];
        const text = `TopK ${slot} · E${route.expert} → rank${route.rank} / slot${route.localSlot} · owner rank${route.owner}`;
        byId('learning-context-route').textContent = text;
        byId('learning-inspector-context').innerHTML = `<strong>当前 token：rank${data.focus.rank} / 行 ${data.focus.row}</strong>${esc(text)}<br>p=${route.probability} · 实例 ${route.rows.toLocaleString('en-US')} 行`;
        document.querySelectorAll('[data-route-slot]').forEach(node => {
            const selected = Number(node.dataset.routeSlot) === slot;
            node.classList.toggle('is-route-selected', selected);
            node.classList.toggle('is-route-muted', !selected);
            if (node.hasAttribute('data-learning-slot')) node.setAttribute('aria-pressed', String(selected));
        });
        const earlyCells = root.querySelectorAll('.learning-pipeline-grid [data-learning-stage="6"]');
        if (earlyCells[0]) earlyCells[0].textContent = route.shared ? '当前专家 early dW1' : '当前专家未共享：跳过前移';
        if (earlyCells[2]) earlyCells[2].textContent = route.shared ? '消费本共享专家的梯度 tile' : 'owner 可服务其他共享专家';
        if (state.scene === 'pipeline') chooseCheckpoint(state.checkpoint);
    }
    function chooseScene(id) {
        const scene = data.scenes.find(scene => scene.id === id);
        if (!scene) return;
        state.scene = id;
        root.querySelectorAll('[data-learning-scene]').forEach(button => button.setAttribute('aria-selected', String(button.dataset.learningScene === id)));
        root.querySelectorAll('.learning-scene').forEach(panel => { panel.hidden = panel.id !== `learning-scene-${id}`; });
        if (id === 'pipeline') chooseCheckpoint(state.checkpoint);
        else if (id === 'tape') chooseBuffer(state.buffer);
        else selectSource(scene.callpoint);
    }
    function chooseBuffer(id) {
        const buffer = data.buffers.find(buffer => buffer.id === id);
        if (!buffer) return;
        state.buffer = id;
        root.querySelectorAll('[data-learning-buffer]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.learningBuffer === id)));
        const rowBytes = buffer.dimension ? data.shape[buffer.dimension] * 2 : null;
        const values = { hp: ['dHp', 'Hp'], do: ['dO', 'dX'], return: ['路由 dX', '路由 dp'] }[buffer.id];
        const boundary = buffer.stages[2];
        byId('learning-buffer-view').innerHTML = `<h5>${esc(buffer.title)}</h5><p>${esc(buffer.shape)}${rowBytes ? ` · 每专家行 ${rowBytes.toLocaleString('en-US')} 字节（BF16）` : ''}；m 为该物理实例行数。</p>
            <div class="learning-lifetime-band" aria-label="同一地址的先后用途"><div class="learning-alias-before"><small>原数据仍活跃</small><strong>${values[0]}</strong><span>${esc(buffer.stages[1].title)}</span></div><button type="button" class="learning-alias-boundary" data-learning-callpoint="${boundary.callpoint}"><small>复用边界 →</small><strong>${esc(boundary.title)}</strong><span>${esc(boundary.text)}</span></button><div class="learning-alias-after"><small>同一地址的新用途</small><strong>${values[1]}</strong><span>${esc(buffer.stages[4].title)}</span></div></div>
            <details class="learning-lifetime-detail"><summary>展开 5 个阶段及对应源码</summary><ol class="learning-buffer-stages">${buffer.stages.map(stage => `<li><strong>${esc(stage.title)}</strong><p>${esc(stage.text)}</p><button type="button" data-learning-callpoint="${stage.callpoint}">${stage.callpoint} 对应源码</button></li>`).join('')}</ol></details>`;
        selectSource(buffer.callpoint);
    }
    function pipelineGrid() {
        const tiles = Array.from({ length: data.tileWindow }, (_, i) => `<span class="learning-tile">tile i${i ? '+' + i : ''}</span>`).join('');
        const cells = [
            ['AIC', '', `${tiles}<br>写独立 GM`, 'PIPE_ALL → dHpReady', '', '等 dZ / Hp', 'dW2 → doReleased', '共享专家 early dW1', `${tiles}<br>覆盖 dO 为 dX`, 'PIPE_ALL → dXReady', '其余 dW1 / 返回'],
            ['AIV0', '派发 dO', '后续专家仍可派发', '等专家 dHpReady', `${tiles}<br>交替所属 tile`, 'MTE3 写完 → dzReady', '', '', '等待专家 dXReady', 'UB: 读 → 发 → 退休 → 下一 tile', 'P5 dX → 同步 → P6 dp'],
            ['AIV1', '按 panel 恢复 W', '后续权重可交叠', '等专家 dHpReady', '消费自己的交替 tile', 'MTE3 写完 → dzReady', '按 worker 调度服务 owner', '消费同 epoch 梯度 tile', '与 AIC / AIV0 交叠', 'owner 独占 tile 服务', 'owner 收口'],
            ['活跃 GM', 'dO', 'dHp', 'dHp 已完整写入', 'dHp → Hp；另写 dZ / dp partial', 'Hp / dZ', '仍为 dO，等读者退休', '原 dO 已可复用', 'dO → dX', 'dX 回源 → ptrD', 'ptrD: dX → dp'],
        ];
        const count = data.checkpoints.length;
        byId('learning-pipeline-grid').innerHTML = `<div class="learning-pipeline-grid" style="grid-template-columns:100px repeat(${count},minmax(88px,1fr))"><strong class="learning-lane">依赖检查点</strong>${data.checkpoints.map((step, i) => `<button type="button" class="learning-checkpoint-button" data-learning-checkpoint="${i}" title="${esc(step.title)}">${i + 1}<br>${esc(step.title)}</button>`).join('')}${cells.map(row => `<strong class="learning-lane">${row[0]}</strong>${row.slice(1).map((text, i) => `<div data-learning-stage="${i}" class="${text.includes('等') ? 'learning-wait' : text.includes('退休') || text.includes('ready') ? 'learning-retire' : 'learning-work'}">${text || '·'}</div>`).join('')}`).join('')}</div>`;
    }
    function chooseCheckpoint(index) {
        if (!Number.isInteger(index) || !data.checkpoints[index]) return;
        state.checkpoint = index;
        const step = data.checkpoints[index];
        byId('learning-step-label').textContent = `${index + 1} / ${data.checkpoints.length}`;
        byId('learning-checkpoint-title').textContent = step.title;
        const route = data.routes[state.slot];
        byId('learning-checkpoint-text').textContent = step.text + (index === 6 ?
            (route.shared ? ` 当前 E${route.expert} 有副本，适用这一条件分支。` : ` 当前 E${route.expert} 没有副本，本实例跳过前移，余下 dW1 由后续阶段完成。`) : '');
        byId('learning-previous').disabled = index === 0;
        byId('learning-next').disabled = index === data.checkpoints.length - 1;
        root.querySelectorAll('[data-learning-stage], [data-learning-checkpoint]').forEach(node => {
            const stage = Number(node.dataset.learningStage ?? node.dataset.learningCheckpoint);
            node.classList.toggle('is-current', stage === index);
            node.classList.toggle('is-future', stage > index);
            if (node.hasAttribute('data-learning-checkpoint')) node.setAttribute('aria-pressed', String(stage === index));
        });
        selectSource(step.callpoint);
    }
    root.addEventListener('click', event => {
        const slot = event.target.closest('[data-learning-slot]');
        const scene = event.target.closest('[data-learning-scene]');
        const buffer = event.target.closest('[data-learning-buffer]');
        const checkpoint = event.target.closest('[data-learning-checkpoint]');
        const callpoint = event.target.closest('[data-learning-callpoint]');
        const source = event.target.closest('[data-learning-open-source]');
        if (slot) chooseSlot(Number(slot.dataset.learningSlot));
        else if (scene) chooseScene(scene.dataset.learningScene);
        else if (buffer) chooseBuffer(buffer.dataset.learningBuffer);
        else if (checkpoint) chooseCheckpoint(Number(checkpoint.dataset.learningCheckpoint));
        else if (callpoint) selectSource(callpoint.dataset.learningCallpoint);
        else if (source) window.megaMoeGuide.openSource(state.callpoint, source.dataset.learningOpenSource, false);
    });
    byId('learning-previous').onclick = () => chooseCheckpoint(state.checkpoint - 1);
    byId('learning-next').onclick = () => chooseCheckpoint(state.checkpoint + 1);
    byId('learning-checkpoint-source').onclick = () => selectSource(data.checkpoints[state.checkpoint].callpoint);
    window.addEventListener('megamoe-callpoint', event => selectSource(event.detail.id));
    document.querySelectorAll('svg [data-route-slot]').forEach(node => {
        node.setAttribute('role', 'button'); node.setAttribute('tabindex', '0');
        node.setAttribute('aria-label', '选择 TopK ' + node.dataset.routeSlot + ' 路由');
        node.addEventListener('click', () => chooseSlot(Number(node.dataset.routeSlot)));
        node.addEventListener('keydown', event => { if (['Enter', ' '].includes(event.key)) { event.preventDefault(); chooseSlot(Number(node.dataset.routeSlot)); } });
    });
    pipelineGrid(); chooseBuffer(state.buffer); chooseCheckpoint(state.checkpoint); chooseSlot(state.slot); chooseScene(state.scene);
    window.__learningScenes = { data, state, chooseSlot, chooseScene, chooseBuffer, chooseCheckpoint };
})();
