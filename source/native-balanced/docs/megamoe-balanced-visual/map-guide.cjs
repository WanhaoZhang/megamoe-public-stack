const fs = require('node:fs');
const path = require('node:path');

const guideId = 'system-guide';
const maps = [
    {
        id: 'system-architecture', title: '系统架构', kind: '分组 Flowchart', syntax: 'flowchart',
        purpose: '先看三个仓各自负责什么。边上的张量和状态说明模型、FSDP、Native Host 和 NPU 之间怎样交接。',
        deep: ['06-model-step', '整网前向时序'],
        calls: [['F03', '绑定后的专家入口'], ['F06', 'dispatcher 交给 bridge'], ['P05', '规划结果校验'], ['N04', 'Host 提交'], ['F16', '模型合并 shared 输出']],
    },
    {
        id: 'run-lifecycle', title: '训练进程生命周期', kind: 'State Diagram', syntax: 'stateDiagram-v2',
        purpose: '以整次运行为对象，从 torchrun 装配到多步训练和退出。采集轮在训练结束后经过单独的导出状态。',
        deep: ['00-lifecycle', '启动到结束的调用时序'],
        calls: [['I02', '启动 rank'], ['I14', '建立并行组'], ['S06', '取 batch'], ['L04', '退出训练循环'], ['L05', '采集导出']],
    },
    {
        id: 'training-workflow', title: '一次训练 step 的跨库泳道', kind: 'Swimlane Beta', syntax: 'swimlane-beta',
        purpose: '沿箭头读每个库处理了哪些数据。重点看 X / ids / p 进入算子，routed Y 回模型，home dW 交给 Expert-FSDP。',
        deep: ['11-model-backward', '反向到更新的调用时序'],
        calls: [['S08', '固定路由覆盖'], ['F08', '发布 home 权重'], ['F15', '保存 tape'], ['B07', 'placement 回放'], ['U04', 'Expert-FSDP 梯度同步']],
    },
    {
        id: 'parameter-lifecycle', title: '专家权重的生命周期', kind: 'State Diagram', syntax: 'stateDiagram-v2',
        purpose: '追踪一个 routed 专家的逻辑参数。FSDP 管 Parameter 与梯度分片，Native pool 持有供执行使用的拷贝，反向从本层 tape 取回 placement。',
        deep: ['09-update', '梯度同步与参数更新时序'],
        calls: [['F18', '前向权重解分片'], ['F19', '前向后参数处理'], ['U02', '反向权重准备'], ['B16', '交回 home 梯度'], ['U07', '更新参数分片']],
    },
];

function mermaidImage(id, description, esc) {
    const asset = `maps/${id}.svg`;
    const svg = fs.readFileSync(path.join(__dirname, asset), 'utf8');
    const viewBox = svg.match(/<svg[^>]*\bviewBox="([^"]+)"/)?.[1];
    const [, , width, height] = viewBox?.split(/\s+/).map(Number) || [];
    if (!(width > 0 && height > 0)) throw new Error('Missing diagram dimensions: ' + asset);
    return `<img src="data:image/svg+xml;base64,${Buffer.from(svg).toString('base64')}" data-map-source="${asset}" width="${Math.ceil(width)}" height="${Math.ceil(height)}" style="--map-min:${Math.min(650, Math.ceil(width))}px" alt="${esc(description)}">`;
}

function renderGuide(esc, callpoints) {
    const ids = new Set(callpoints.map(row => row.id));
    return `<section class="panel map-guide" id="${guideId}" data-chapter="${guideId}">
        <h2>系统导览</h2>
        <p class="lead">先找模块归属，再区分一轮运行和一轮训练 step，最后跟进参数的变化。图描述源码结构和数据依赖，实验实测见 <a href="#12-python-case">实验实例</a>。</p>
        <div class="drilldowns" aria-label="系统导览目录">
            ${maps.map(map => `<a href="#${map.id}">${esc(map.title)}</a>`).join('')}
        </div>
        ${maps.map((map, index) => `<figure class="map-figure">
            <h3 id="${map.id}"><small>0${index + 1} · ${esc(map.kind)}</small>${esc(map.title)}</h3>
            <p>${esc(map.purpose)}</p>
            <div class="map-canvas" role="region" tabindex="0" aria-label="${esc(map.title)}，窄窗口可横向滚动">
                ${mermaidImage(map.id, map.purpose, esc)}
            </div>
            <figcaption>
                <a href="#${map.deep[0]}">${esc(map.deep[1])}</a>
                <a href="maps/${map.id}.svg">独立 SVG</a>
                <a href="maps/${map.id}.mmd">Mermaid 源码</a>
            </figcaption>
            <p class="map-callpoints">源码 ${map.calls.map(([id, label]) => {
                if (!ids.has(id)) throw new Error('Unknown guide callpoint: ' + id);
                return `<a href="#${id}" data-editor-callpoint="${id}">${id} ${esc(label)}</a>`;
            }).join(' · ')}</p>
        </figure>`).join('')}
    </section>`;
}

module.exports = { guideId, maps, renderGuide, mermaidImage };
