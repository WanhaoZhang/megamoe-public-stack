const fs = require('node:fs');
const path = require('node:path');
const meta = require('./source-evidence.json');
const snippets = require('./source-excerpts.json');
const { caseId, renderPythonCase } = require('./python-case.cjs');
const { guideId, renderGuide } = require('./map-guide.cjs');
const OUT = __dirname;
const M = 'MindSpeed-LLM/';
const F = 'FSDPTurbo/';
const rows = meta.rows;
const files = Object.fromEntries(Object.entries(meta.files).map(([key, info]) => [key, info.path]));
const diagramSource = id => fs.readFileSync(path.join(OUT,'diagrams',id+'.mmd'),'utf8');
const esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
const diagrams = [
{id:'00-lifecycle',prefix:'L',title:'训练生命周期',
 subtitle:'一次启动包住多次训练 step。总览按职责分列，详图展开到源码文件。实线表示调用或提交，虚线表示返回或框架调度。',
 source:diagramSource('00-lifecycle')},
{id:'01-init',prefix:'I',title:'初始化与绑定',
 subtitle:'以仓库 run_training_case.sh 的方法 C 为入口。Python 固定路由与采集使用独立实验包装器，参数见顶部说明。',
 source:diagramSource('01-init')},
{id:'06-model-step',prefix:'S',title:'整网前向：从 batch 经 MegaMoE kernel 到 loss',
 subtitle:'从 Trainer 取 batch 开始，沿主干 decoder 追踪一次 routed MoE。MTP 的 decoder 复用这条专家路径。',
 source:diagramSource('06-model-step')},
{id:'02-forward',prefix:'F',title:'框架前向：从 experts.forward 到 Native bridge',
 subtitle:'沿方法 C / MoonEP planner 展开。设备消费与快照复制按 stream 排序，release_pending 释放调用租约。',
 source:diagramSource('02-forward')},
{id:'03-planner',prefix:'P',title:'MoonEP Planner：从逻辑路由到物理计划',
 subtitle:'EP4、未强制 GM planner 的方法 C 路径。HCCL all-gather 是 validator 的调用；图中每一列仍是源码文件。',
 source:diagramSource('03-planner')},
{id:'04-native',prefix:'N',title:'Python/C++ 提交：从 Torch 类到 AIC / AIV kernel',
 subtitle:'worker 构建目标复制为 early SO；宏与 include 是编译关系，图中以注释标注。kernel 内部阶段可交错，纵向不是实测耗时。',
 source:diagramSource('04-native')},
{id:'07-kernel-forward',prefix:'K',title:'前向 kernel：派发、权重搬运、计算与合并',
 subtitle:'接 Python/C++ 提交图的 N10。按 BF16 saved 主线展开，AIC/AIV 按 ready 交叠。',
 source:diagramSource('07-kernel-forward')},
{id:'11-model-backward',title:'整网反向与更新：从 loss.backward 到下一步',
 subtitle:'从整网 loss 跟到分片参数更新。图按依赖组织，重算和各分支的实际调度由 autograd 决定。',
 source:diagramSource('11-model-backward')},
{id:'05-backward',prefix:'B',title:'反向桥接与 placement 回放',
 subtitle:'主路径选择 panel，staged 分支在 Host 侧显式恢复。虚线标出 autograd 与 FSDP 的间接调度。',
 source:diagramSource('05-backward')},
{id:'08-kernel-backward',prefix:'G',title:'反向 kernel 与 owner 归并',
 subtitle:'接反向桥接图的 B13。BF16 saved、panel2、earlyDW1=1、owner ratio2 为 11 月 2 日实际实验配置。',
 source:diagramSource('08-kernel-backward')},
{id:'09-update',prefix:'U',title:'FSDP 梯度同步与参数更新',
 subtitle:'将自动微分调度、框架 FSDP hook、算子副本 SUM 和优化器更新分开。',
 source:diagramSource('09-update')},
{id:'10-saved-mode',prefix:'H',title:'补充阅读：保存格式与两层重算',
 subtitle:'将 11 月 2 日实际运行的 BF16 saved 主线，与当前 checkout 的新增可选模式并列标记。',
 source:diagramSource('10-saved-mode')}
];

const chapters = [
 {id:'00-lifecycle', label:'生命周期总览'},
 {id:'01-init', label:'初始化与绑定'},
 {id:'06-model-step', label:'整网前向', details:[
  ['02-forward','框架前向'], ['03-planner','MoonEP Planner'],
  ['04-native','Python/C++ 提交'], ['07-kernel-forward','前向 kernel'],
 ]},
 {id:'11-model-backward', label:'整网反向与更新', details:[
  ['05-backward','反向桥接与 placement 回放'], ['08-kernel-backward','反向 kernel 与 owner 归并'],
  ['09-update','FSDP 梯度同步与参数更新'],
 ], extra:['10-saved-mode','保存格式与两层重算']},
];
const chapterOf = new Map();
for (const chapter of chapters) {
 chapterOf.set(chapter.id, chapter);
 for (const [id] of chapter.details || []) chapterOf.set(id, chapter);
 if (chapter.extra) chapterOf.set(chapter.extra[0], chapter);
}
const diagramRows = d => d.prefix ? rows.filter(r=>r.id.startsWith(d.prefix)) : [];
const link = ([id,label], current) => `<a href="#${id}"${current===id?' aria-current="page"':''}>${esc(label)}</a>`;

function chapterNavigation(d,chapter) {
 const back = d.id!==chapter.id ? `<p class="chapter-back"><a href="#${chapter.id}">← 返回${esc(chapter.label)}</a></p>` : '';
 const steps = {
  '00-lifecycle': chapters.slice(1).map(c=>[c.id,c.label]),
  '01-init': [['06-model-step','下一步：整网前向']],
 };
 const links = (chapter.details || steps[d.id] || []).map(item=>link(item,d.id)).join('');
 const label = chapter.details ? '本章详图' : '沿训练顺序阅读';
 const extra = chapter.extra ? `<p class="chapter-extra">补充阅读 · ${link(chapter.extra,d.id)}</p>` : '';
 const example=d.id==='00-lifecycle' ? `<p class="chapter-extra">${link([guideId,'系统导览'],d.id)} · ${link([caseId,'八卡 Python 实验实例'],d.id)}</p>` : '';
 return back + `<div class="drilldowns" aria-label="${label}"><strong>${label}</strong>${links}</div>` + extra + example;
}
function location(k,line){return `${decodeURIComponent(sourceUrl(k))}:${line}`;}
function sourceUrl(k){return meta.files[k].href;}
function sourceLink(k,line){
 const label=path.basename(files[k])+':'+line;
 return `<a class="source-link" href="${esc(sourceUrl(k))}" data-source-line="${line}" title="在 VS Code 打开 ${esc(label)}"><code>${esc(label)}</code></a>`;
}
function excerpt(k,line){
 const key=k+':'+line;
 if(!Object.hasOwn(snippets,key))throw new Error('Missing source excerpt: '+key);
 return snippets[key];
}
function table(data){return `<table><thead><tr><th>编号</th><th>调用点 → 进入位置</th><th>数据与职责</th></tr></thead><tbody>${data.map(r=>`<tr><td><a href="#${r.id}">${r.id}</a></td><td><details id="${r.id}"><summary>${sourceLink(r.from,r.fromLine)} → ${sourceLink(r.to,r.toLine)}</summary><p class="filepath">${esc(location(r.from,r.fromLine))}</p><pre>${esc(excerpt(r.from,r.fromLine))}</pre>${r.from!==r.to||r.fromLine!==r.toLine?`<p class="filepath">${esc(location(r.to,r.toLine))}</p><pre>${esc(excerpt(r.to,r.toLine))}</pre>`:''}</details></td><td>${esc(r.detail)}</td></tr>`).join('')}</tbody></table>`;}

const components = {
 mindspeed: {label:'MindSpeed-LLM', header:'#dbeafe', ink:'#275589'},
 fsdp: {label:'FSDPTurbo / FSDP', header:'#d8efe5', ink:'#22634d'},
 kernel: {label:'MegaMoE 算子 / Kernel', header:'#ffe8d3', ink:'#90521d'},
};
const componentForFile = new Map(Object.entries(files).map(([key, file])=>[
 path.basename(file),
 file.startsWith(F) ? 'fsdp' : (file.startsWith(M) || ['runner','stage','run','entry','fixed'].includes(key)) ? 'mindspeed' : 'kernel',
]));
const componentForActor = new Map([
 ...componentForFile,
 ...Object.entries(components).map(([key, value])=>[value.label,key]),
]);
const attr=(tag,name)=>tag.match(new RegExp(' '+name+'="([^"]*)"'))?.[1];
const setAttr=(tag,name,value)=>{
 const clean=tag.replace(new RegExp(' '+name+'="[^"]*"'),'');
 return clean.replace(/(\/?>)$/,' '+name+'="'+esc(value)+'"$1');
};

function colorSequenceParticipants(raw, diagram){
 let svg=raw.replace(/<style data-role="component-colors">[\s\S]*?<\/style>/g,'');
 const owners=new Map([...diagram.source.matchAll(/^\s+participant (\S+) as (.+)$/gm)].map(([,id,label])=>{
  const owner=componentForActor.get(label)||componentForActor.get(path.basename(label));
  if(!owner)throw new Error('Missing component: '+diagram.id+' / '+label);
  return [id,owner];
 }));
 const headers=[...svg.matchAll(/<rect\b[^>]*\bclass="[^"]*\bactor-top\b[^"]*"[^>]*>/g)];
 if(headers.length!==owners.size)throw new Error('Missing lane header: '+diagram.id);
 svg=svg.replace(/<(?:rect|line)\b[^>]*\bclass="[^"]*\b(?:actor-top|actor-bottom|actor-line)\b[^"]*"[^>]*>/g,tag=>{
  const owner=owners.get(attr(tag,'name'));
  if(!owner)throw new Error('Unclassified lane: '+diagram.id+' '+tag);
  return setAttr(tag,'data-component',owner);
 });
 const styles=Object.entries(components).map(([key,c])=>
  `rect.actor[data-component="${key}"]{fill:${c.header}!important;stroke:${c.ink}!important}line.actor-line[data-component="${key}"]{stroke:${c.ink}!important;stroke-width:1.1px!important;opacity:.75}`
 ).join('');
 return svg.replace('</style>',`</style><style data-role="component-colors">${styles}</style>`);
}

function packageHtml(){
 const sections=diagrams.map((d,i)=>{
  const svgFile=path.join(OUT,'diagrams',d.id+'.svg');
  let svg=colorSequenceParticipants(fs.readFileSync(svgFile,'utf8'),d);
  fs.writeFileSync(svgFile,svg,'utf8');
  svg=svg.replace(/<\?xml[^>]+>/,'');
  const svgIds=[...new Set([...svg.matchAll(/\sid="([^"]+)"/g)].map(m=>m[1]))];
  const scopedIds=new Map(svgIds.map(id=>[id,'s'+d.id+'-'+id]));
  for(const [id,next] of scopedIds){svg=svg.replaceAll('id="'+id+'"','id="'+next+'"').replaceAll('url(#'+id+')','url(#'+next+')').replaceAll('href="#'+id+'"','href="#'+next+'"');}
  // Rewrite ID selectors in rule headers, preserving declaration colors and quoted values.
  svg=svg.replace(/<style\b[^>]*>[\s\S]*?<\/style>/g,style=>style.replace(/(^|[{}])([^{}]*)(?=\{)/g,(_,boundary,selector)=>
   boundary+selector.replace(/"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|#([\w-]+)/g,(token,id)=>scopedIds.has(id)?'#'+scopedIds.get(id):token)));
  svg=svg.replace(/(<svg[^>]*>)/,'$1<title>'+esc(d.title)+'</title><desc>'+esc(d.subtitle)+'</desc>');
  const chapter=chapterOf.get(d.id);
  const navigation=chapterNavigation(d,chapter);
  const handoff=d.id==='06-model-step'
   ? '<p class="handoff">本层 tape 保存 counts、state、dispatch 和 placement。<a href="#11-model-backward">下一步：整网反向与更新</a></p>'
   : d.id==='11-model-backward'
    ? '<p class="handoff">更新完成后进入下一步，FSDP 和执行池读取当前参数。<a href="#06-model-step">回到整网前向</a></p>'
    : '';
  const data=diagramRows(d);
  const hint='本地打开网页时，点击图中编号在 VS Code 跳到实现行。表格中的文件名分别打开调用处和实现处，左侧编号展开内嵌源码。';
  return `<section class="panel ${i===0?'active':''}" id="${d.id}" data-chapter="${chapter.id}"><h2>${esc(d.title)}</h2><p class="muted">${esc(d.subtitle)}</p>${navigation}<div class="tools"><button class="fit">适合窗口</button><button class="actual">原始尺寸</button><button class="minus">−</button><span class="zoom-label">100%</span><button class="plus">＋</button><button class="expand" aria-expanded="false">展开整张图</button><a href="diagrams/${d.id}.svg">独立 SVG</a><a href="diagrams/${d.id}.mmd">Mermaid 源码</a></div><div class="viewport" tabindex="0" aria-label="Mermaid 源码时序图，可横向与纵向滚动"><div class="canvas">${svg}</div></div><p class="hint">${hint}</p>${handoff}${data.length?table(data):''}</section>`;
 }).join('');
 const html=`<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MegaMoE Balanced · 框架到算子源码时序图</title><style>
 *{box-sizing:border-box}body{margin:0;background:#f6f8fa;color:#15283f;font:15px/1.7 "Microsoft YaHei","Segoe UI",sans-serif}main{max-width:1760px;margin:auto;padding:30px 34px 70px}h1{font-size:30px;margin:0 0 8px;letter-spacing:.02em}h2{font-size:22px;margin:0 0 8px}.lead{max-width:1250px;margin:0 0 18px}.muted,.hint{color:#51657b}.scope{border-left:4px solid #527a9d;background:#eaf0f6;padding:12px 18px;margin:18px 0}.scope p{margin:2px 0}.tags{display:flex;gap:10px;flex-wrap:wrap}.tags span{background:#fff;border:1px solid #c7d2df;padding:3px 11px;border-radius:4px}nav{display:flex;gap:7px;flex-wrap:wrap;position:sticky;top:0;background:#f6f8fa;padding:13px 0;z-index:4;border-bottom:1px solid #c7d2df;margin-bottom:24px}button,nav a{cursor:pointer;background:#fff;border:1px solid #8da0b5;border-radius:4px;padding:7px 13px;color:#183650;font:inherit;text-decoration:none}nav button.selected{background:#224d70;color:#fff;border-color:#224d70}.panel{display:none}.panel.active{display:block}.tools{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:15px 0}.tools button{padding:4px 12px}.tools a{margin-left:8px}.zoom-label{min-width:52px;text-align:center;font-variant-numeric:tabular-nums}.viewport{overflow:auto;border:1px solid #b9c8d8;background:#fff;max-height:78vh;min-height:320px}.canvas{width:max-content}.canvas svg{display:block;max-width:none!important}table{width:100%;border-collapse:collapse;background:#fff;margin:16px 0 30px;table-layout:fixed}th,td{text-align:left;vertical-align:top;padding:12px 14px;border:1px solid #d5dee8}th{background:#e9eff5}th:first-child{width:72px}th:nth-child(2){width:44%}td{overflow-wrap:anywhere}summary{cursor:pointer;color:#214e78;font-weight:600}code,pre{font-family:Consolas,"Microsoft YaHei",monospace;font-size:13px}pre{background:#f3f6f9;padding:12px;overflow:auto;white-space:pre;line-height:1.65;border-left:3px solid #9db0c3}.filepath{font-size:12px;margin:9px 0 3px}a{color:#215785}a:hover{text-decoration:underline}.technical{border-top:1px solid #b9c8d8;padding-top:22px;margin-top:35px}.notes li{margin-bottom:10px}.heads{font-size:12px;word-break:break-all}.footer{margin-top:28px;font-size:13px;color:#53667c}details:target{background:#fff7df}body.print .panel{display:block}@media print{nav,.tools{display:none}main{padding:0}.panel{display:block;page-break-before:always}.viewport{max-height:none;overflow:visible}.canvas svg{width:100%!important;height:auto!important}.canvas{width:100%}details pre{white-space:pre-wrap}}
 </style></head><body><main><h1>MegaMoE Balanced：从框架到算子的源码时序图</h1><p class="lead">横轴是实际文件名，纵向是调用顺序。实线表示调用或提交，虚线表示返回或框架间接调度；宏展开、继承与编译包含关系另作注释。图上编号与源码表一一对应。</p><div class="tags"><span>方法 C · MoonEP planner</span><span>EP4 × Expert-FSDP2</span><span>E8 / Top2 / M8192 / H7168 / F3072</span><span>panel 恢复 · BF16</span></div><div class="scope"><p><strong>当前源码静态核对 · 2026-10-30。</strong>历史运行样例 fast-rho6-C-f-a1：40/40 步、run.exit=0。当前代码已有后续更新；图描述当前源码路径，旧 SO 名不能当成当前二进制验收。</p><p>单rank / 单MoE层路径；普通层、MTP、重算会重复进入。纵向不是时间比例图，核内阶段可能交错。代码行号与 SHA 记录在 <a href="source-evidence.json">source-evidence.json</a>。</p></div><nav>${diagrams.map((d,i)=>`<button data-tab="${d.id}" class="${i===0?'selected':''}">${['① 初始化绑定','② 框架前向','③ Planner','④ Native / kernel','⑤ 反向归主'][i]}</button>`).join('')}<a href="#supplement" id="extra-link">继承 / SO / 梯度边界</a></nav>${sections}<section class="technical" id="supplement"><h2>读图时必须分清的四个边界</h2><ol class="notes"><li><strong>框架入口：</strong>模型仍调用 self.experts；expert_parallel.py 通过 MethodType 改变专家模块 forward。Attention、gate、共享专家、loss 与 optimizer 仍在模型/训练框架中。</li><li><strong>autograd 与继承：</strong>整网使用 fsdp2_bridge._ExternalExperts。Worker → Early → Full → Stream → Pipeline → Device → Mixed → Train 是Python继承链；真正前向来自Full，create_plan / finish来自Device。图中仅画实际覆盖实现，省略纯继承跳转。</li><li><strong>运行库与源码：</strong>构建目标worker_bf16会复制为early SO。WORKER_NATIVE / LD_LIBRARY_PATH决定隔离变体；launcher根据descriptor选择导出符号。宏include复用full.cpp，不是额外一次运行时函数调用。</li><li><strong>权重与梯度：</strong>optimizer持有框架Parameter；pool是跨层共享执行存储。前反向都刷新本层home。算子副本partial先归主，然后FSDP完成自己的梯度同步/缩放，二者职责不同。</li></ol>${table(rows.filter(r=>r.id.startsWith('X')))}<h2>形状与保存状态</h2><table><tr><th>数据</th><th>本例形状 / 生命周期</th><th>用途</th></tr><tr><td>x / y</td><td>[8192,7168]</td><td>本rank token激活；bridge返回后恢复原模型形状</td></tr><tr><td>logical ids / prob</td><td>[8192,2]；ids=int32；执行器prob转BF16</td><td>逻辑专家选择与combine权重；fix_router开启时覆盖模型gate结果</td></tr><tr><td>home W1 / W2</td><td>[2,6144,7168] / [2,7168,3072]</td><td>当前层unshard后的EP本地权重；每次publish刷新共享home</td></tr><tr><td>copies / physical ids</td><td>[EP,slots]=[4,2] / [8192,2]</td><td>planner决定的副本表和物理路由；反向回放前向快照</td></tr><tr><td>counts / state / dispatch</td><td>按native layout保存；state与dispatch为opaque byte tensor</td><td>反向使用的执行计数、数值保存状态与派发输入；不等同完整pool权重快照</td></tr><tr><td>placement / control</td><td>四个计划tensor + 8-word控制status</td><td>通过save_for_backward可见于UAM/checkpoint；反向使用新epoch与原placement</td></tr></table><h2>固定的源码版本</h2><div class="heads">${Object.entries(meta.heads).map(([k,v])=>`<p><strong>${k}</strong> ${v}</p>`).join('')}</div><p><a href="CALLPOINTS.md">完整调用点分析 Markdown</a> · <a href="source-evidence.json">源码证据与SHA</a></p></section><p class="footer">协作入口见 README。Mermaid 源图位于 diagrams/，内嵌调用点来自已核对的源码快照。</p></main><script>
 (()=>{
 const editorBridge=typeof acquireVsCodeApi==='function'?acquireVsCodeApi():null;
 const panels=[...document.querySelectorAll('.panel')];
 const diagramPanels=panels.filter(panel=>panel.querySelector('.viewport'));
 function setZoom(panel,scale){const svg=panel.querySelector('svg');const vb=svg.viewBox.baseVal;scale=Math.max(.15,Math.min(2.5,scale));svg.style.width=(vb.width*scale)+'px';svg.style.height=(vb.height*scale)+'px';panel.dataset.zoom=scale;panel.querySelector('.zoom-label').textContent=Math.round(scale*100)+'%';}
 function fit(panel){const svg=panel.querySelector('svg');setZoom(panel,Math.min(1,(panel.querySelector('.viewport').clientWidth-4)/svg.viewBox.baseVal.width));}
 const topTabs=[...document.querySelectorAll('[data-tab]')];
 function activate(panel){
  panels.forEach(p=>p.classList.toggle('active',p===panel));
  topTabs.forEach(tab=>{
   const current=tab.dataset.tab===panel.dataset.chapter;
   tab.classList.toggle('selected',current);
   if(current)tab.setAttribute('aria-current','page');else tab.removeAttribute('aria-current');
  });
  if(diagramPanels.includes(panel))fit(panel);
 }
 topTabs.forEach(tab=>tab.addEventListener('click',()=>{
  const hash='#'+tab.dataset.tab;
  if(location.hash===hash)jumpToSource();else location.hash=hash;
 }));
 diagramPanels.forEach(p=>{p.querySelector('.fit').onclick=()=>fit(p);p.querySelector('.actual').onclick=()=>setZoom(p,1);p.querySelector('.plus').onclick=()=>setZoom(p,+(p.dataset.zoom||1)*1.2);p.querySelector('.minus').onclick=()=>setZoom(p,+(p.dataset.zoom||1)/1.2);});
 diagramPanels.forEach(p=>{
  const expand=p.querySelector('.expand');
  expand.onclick=()=>{
   const expanded=p.querySelector('.viewport').classList.toggle('expanded');
   expand.setAttribute('aria-expanded',expanded);
   expand.textContent=expanded?'收起整张图':'展开整张图';
  };
 });
 function jumpToSource(){
  const target=document.getElementById(location.hash.slice(1));
  if(!target){
   if(!location.hash)activate(panels[0]);
   return;
  }
  const isDetail=target.tagName==='DETAILS';
  if(!isDetail&&!target.classList.contains('panel')&&!target.matches('.case-page :is(h3,h4)[id], .map-guide h3[id]'))return;
  if(isDetail)target.open=true;
  const panel=target.closest('.panel');
  if(panel&&!panel.classList.contains('active'))activate(panel);
  target.scrollIntoView({block:isDetail?'center':'start'});
 }
 const localPage=location.protocol==='file:';
 if(localPage){
  document.querySelectorAll('a.source-link').forEach(link=>{
   const file=new URL(link.getAttribute('href'),location.href);
   const absolutePath=(file.host?'//'+file.host:'')+file.pathname;
   link.href='vscode://file'+absolutePath+':'+link.dataset.sourceLine+':1';
  });
 }
 function openSource(id,side='to',navigate=true){
  const detail=document.getElementById(id);
  if(!detail||!['from','to'].includes(side))return;
  if(navigate){if(location.hash!=='#'+id)history.pushState(null,'','#'+id);jumpToSource();}
  window.dispatchEvent(new CustomEvent('megamoe-callpoint',{detail:{id,side}}));
  if(editorBridge)editorBridge.postMessage({type:'openSource',callpointId:id,side});
  else if(localPage){const links=detail.querySelectorAll('summary .source-link');(side==='from'?links[0]:links[links.length-1]).click();}
 }
 function openCallpoint(id){openSource(id,'to',true);}
 window.megaMoeGuide={openSource};
 if(editorBridge){
  document.querySelectorAll('a.source-link').forEach(link=>link.addEventListener('click',event=>{
   event.preventDefault();const detail=link.closest('details');const links=[...detail.querySelectorAll('summary .source-link')];
   openSource(detail.id,links.indexOf(link)===0?'from':'to',false);
  }));
  document.querySelectorAll('a[href="source-evidence.json"]').forEach(link=>link.addEventListener('click',event=>{event.preventDefault();editorBridge.postMessage({type:'openMapping'});}));
 }
 document.querySelectorAll('[data-editor-callpoint]').forEach(link=>{
  link.addEventListener('click',event=>{
   event.preventDefault();
   openCallpoint(link.dataset.editorCallpoint);
  });
 });
 document.querySelectorAll('svg text').forEach(label=>{
  const id=label.textContent.match(/^([A-Z][0-9]{2})(?![A-Z0-9])/)?.[1];
  if(!id||!document.getElementById(id))return;
  label.dataset.callpoint=id;
  label.classList.add('messageText');
  label.setAttribute('tabindex','0');
  label.setAttribute('role','link');
  label.setAttribute('aria-label',id+(localPage?' 在 VS Code 打开实现行':' 查看源码调用点'));
  const open=()=>openCallpoint(id);
  label.addEventListener('click',open);
  label.addEventListener('keydown',event=>{if(event.key==='Enter')open();});
 });
 window.addEventListener('load',()=>{fit(panels[0]);jumpToSource();});
 window.addEventListener('hashchange',jumpToSource);
 const globalNav=document.querySelector('main > nav');
 const casePanel=document.querySelector('.case-page');
 const caseSidebar=casePanel.querySelector('.case-sidebar');
 const caseMenu=caseSidebar.querySelector('.case-menu-toggle');
 const caseLinks=[...caseSidebar.querySelectorAll('.case-toc a')];
 const caseTargets=caseLinks.map(link=>document.getElementById(link.hash.slice(1)));
 let selectedCaseLink=null;
 function updateCaseLocation(){
  if(!casePanel.classList.contains('active'))return;
  const edge=Math.max(globalNav.getBoundingClientRect().bottom,caseSidebar.getBoundingClientRect().top)
   +(caseMenu.offsetHeight ? caseSidebar.offsetHeight : 0)+24;
  let active=caseLinks[0];
  for(let i=0;i<caseTargets.length;i++){
   if(caseTargets[i].getBoundingClientRect().top>edge)break;
   active=caseLinks[i];
  }
  if(selectedCaseLink===active)return;
  selectedCaseLink?.removeAttribute('aria-current');
  selectedCaseLink=active;
  active.setAttribute('aria-current','location');
  caseMenu.querySelector('span').textContent=active.dataset.title;
 }
 function showCaseMenu(open){
  caseSidebar.classList.toggle('is-open',open);
  caseMenu.setAttribute('aria-expanded',String(open));
 }
 caseMenu.addEventListener('click',()=>showCaseMenu(!caseSidebar.classList.contains('is-open')));
 caseLinks.forEach(link=>link.addEventListener('click',()=>showCaseMenu(false)));
 caseSidebar.addEventListener('keydown',event=>{
  if(event.key==='Escape'&&caseSidebar.classList.contains('is-open')){showCaseMenu(false);caseMenu.focus();}
 });
 const syncNavHeight=()=>{
  document.documentElement.style.setProperty('--main-nav-height',globalNav.offsetHeight+'px');
  updateCaseLocation();
 };
 syncNavHeight();
 new ResizeObserver(syncNavHeight).observe(globalNav);
 let pendingCaseLocation=false;
 function scheduleCaseLocation(){
  if(pendingCaseLocation||!casePanel.classList.contains('active'))return;
  pendingCaseLocation=true;
  requestAnimationFrame(()=>{pendingCaseLocation=false;updateCaseLocation();});
 }
 window.addEventListener('scroll',scheduleCaseLocation,{passive:true});
 window.addEventListener('hashchange',scheduleCaseLocation);
 window.addEventListener('load',scheduleCaseLocation);
 })();
 </script></body></html>`;
 const caseColors = Object.entries(components).map(([key, value]) =>
  `--case-${key}-ink:${value.ink};--case-${key}-bg:${value.header}`
 ).join(';');
 const caseStyles = `.case-page{${caseColors}}` + ['python-case.css','case-inputs.css','case-forward.css','case-backward.css','map-guide.css','learning-scenes.css']
  .map(file=>fs.readFileSync(path.join(OUT,file),'utf8')).join('\n');
 const page = html
  .replace(/当前源码静态核对 · 2026-10-30。<\/strong>[^<]+/, '当前源码静态核对 · 2026-11-03。</strong> ' + esc(meta.scope))
  .replace('create_plan / finish来自Device', 'create_plan / release_pending来自Device')
  .replace('横轴是实际文件名，纵向是调用顺序。', '从总览进入初始化、整网前向和整网反向。Mermaid 参与者按三组件分组，详图的参与者对应源码文件，从上往下读调用顺序。开放箭头表示异步提交。')
  .replace('</style>', '.viewport.expanded{max-height:none}.panel,.technical{scroll-margin-top:135px}.drilldowns,.component-key{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin:10px 0}.drilldowns strong,.chapter-extra{color:#51657b;font-size:13px}.drilldowns a{padding:4px 10px;border:1px solid #b9c8d8;background:white;border-radius:4px}.drilldowns a[aria-current]{border-color:#224d70;background:#eaf0f6;font-weight:600}.chapter-back,.chapter-extra{margin:8px 0}.component-key{flex-basis:100%;margin:2px 0 0}.component-key span{padding:2px 10px;border:1px solid var(--ink);border-radius:4px;color:var(--ink);background:var(--fill);font-size:13px}.handoff{margin:16px 0;padding:10px 14px;border-left:3px solid #527a9d;background:#eaf0f6}.messageText[data-callpoint]{cursor:pointer}.messageText[data-callpoint]:hover,.messageText[data-callpoint]:focus{fill:#155e75!important;text-decoration:underline}</style>')
  .replace('</style>', caseStyles + '</style>')
  .replace('<section class="technical" id="supplement">', renderPythonCase(esc).replace(/[ \t]+$/gm,'') + '<section class="technical" id="supplement">')
  .replace('<section class="technical" id="supplement">', renderGuide(esc, rows) + '<section class="technical" id="supplement">')
  .replace(/<nav>.*?<\/nav>/, `<nav aria-label="系统导览、训练阶段与实验实例"><button data-tab="${guideId}">系统导览</button>`+chapters.map((c,i)=>`<button data-tab="${c.id}"${i===0?' class="selected" aria-current="page"':''}>${esc(c.label)}</button>`).join('')+`<button data-tab="${caseId}">实验实例</button>`+'<div class="component-key" aria-label="组件配色">'+Object.values(components).map(c=>`<span style="--ink:${c.ink};--fill:${c.header}">${esc(c.label)}</span>`).join('')+'</div></nav>')
  .replace('<section class="technical" id="supplement"><h2>读图时必须分清的四个边界</h2>', '<section class="technical" id="supplement"><h2>参考：参数、执行池、每层 tape 与运行库</h2>')
  .replace('</main>',()=>'<section class="technical"><h2>交互场景：同址生命周期出处</h2>'+table(rows.filter(row=>row.id.startsWith('V')))+'</section></main>')
  .replace('</body>',()=>'<script>'+fs.readFileSync(path.join(OUT,'learning-scenes.js'),'utf8')+'</script></body>');
 fs.writeFileSync(path.join(OUT,'index.html'),page,'utf8');
 let md='# MegaMoE Balanced 框架到算子：调用点分析\n\n'+meta.scope+'\n\n入口：[系统导览](index.html#system-guide) · [源码时序图](index.html)。生命周期总览按职责分列，详图按文件分列。纵向为调用顺序。\n\n';
 for(let i=0;i<diagrams.length;i++){
  const d=diagrams[i];
  md+=`## ${d.title}\n\n${d.subtitle}\n\n![${d.title}](diagrams/${d.id}.svg)\n\n`;
  const data=diagramRows(d);
  if(data.length) {
   md+='| 编号 | 调用点 | 进入位置 | 参数、返回与职责 |\n|---|---|---|---|\n';
   for(const r of data)md+=`| ${r.id} | [${path.basename(files[r.from])}:${r.fromLine}](${sourceUrl(r.from)}) | [${path.basename(files[r.to])}:${r.toLine}](${sourceUrl(r.to)}) | ${r.detail.replaceAll('|',' / ')} |\n`;
  } else {
   const ids=[...new Set(d.source.match(/\b[A-Z]\d{2}\b/g))];
   md+='复用详图调用点：'+ids.map(id=>`[${id}](index.html#${id})`).join(' · ')+'\n';
  }
  md+='\n';
 }
 md+='## 补充：继承、构建与运行库\n\n';for(const r of rows.filter(r=>r.id.startsWith('X')))md+=`- ${r.id}：${location(r.from,r.fromLine)}。${r.detail}\n`;
 md+='\n## 版本\n\n'+Object.entries(meta.heads).map(([k,v])=>`- ${k}: ${v}`).join('\n')+'\n\n每个引用文件的SHA256与解析行号见 source-evidence.json。未执行NPU训练/性能测量。\n';
 fs.writeFileSync(path.join(OUT,'CALLPOINTS.md'),md,'utf8');
 console.log(JSON.stringify({html:path.join(OUT,'index.html'),diagrams:diagrams.length,callpoints:rows.length,files:Object.keys(files).length}));
}
packageHtml();
