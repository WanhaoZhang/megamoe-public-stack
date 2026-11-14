const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { pathToFileURL, fileURLToPath } = require('node:url');
const { chromium } = require('playwright');
const puppeteer = require('puppeteer');
const { caseId } = require('./python-case.cjs');
const { guideId, maps } = require('./map-guide.cjs');
const experiment = require('./python-case.json');
const { sceneData } = require('./learning-scenes.cjs');
const theme = require('./mermaid-config.json').themeVariables;
const rgb = hex => 'rgb(' + hex.slice(1).match(/../g).map(value => parseInt(value, 16)).join(', ') + ')';

const htmlPath = path.join(__dirname, 'index.html');
const evidence = JSON.parse(fs.readFileSync(path.join(__dirname, 'source-evidence.json')));
const hashes = Object.entries(evidence.files);
const diagramIds = fs.readdirSync(path.join(__dirname, 'diagrams'))
    .filter(name => name.endsWith('.mmd')).map(name => path.basename(name, '.mmd'));
const callpoints = new Set(evidence.rows.map(row => row.id));
assert.equal(callpoints.size, evidence.rows.length);
const snippets = require('./source-excerpts.json');
const filePage = pathToFileURL(htmlPath);
const repository = path.resolve(__dirname, '../..');
const sourceFiles = Object.fromEntries(hashes.map(([key, file]) => {
    assert(!path.isAbsolute(file.href) && !file.href.includes(':'), key + ' relative link');
    const url = new URL(file.href, filePage);
    const filename = fileURLToPath(url);
    const relative = path.relative(repository, filename);
    assert(!path.isAbsolute(relative) && !relative.startsWith('..' + path.sep), key + ' inside repository');
    return [key, { url, lines: fs.readFileSync(filename, 'utf8').split(/\r?\n/) }];
}));
for (const row of evidence.rows) {
    assert(row.fromLine > 0 && row.fromLine <= evidence.files[row.from].lines, row.id);
    assert(row.toLine > 0 && row.toLine <= evidence.files[row.to].lines, row.id);
    assert(Object.hasOwn(snippets, row.from + ':' + row.fromLine), row.id);
    assert(Object.hasOwn(snippets, row.to + ':' + row.toLine), row.id);
    for (const [key, line] of [[row.from, row.fromLine], [row.to, row.toLine]]) {
        const excerpt = sourceFiles[key].lines.slice(Math.max(0,line-3),line+7)
            .map((text,index)=>(String(Math.max(1,line-2)+index).padStart(5)+'  '+text).trimEnd()).join('\n');
        assert.equal(excerpt, snippets[key + ':' + line], row.id + ' matches local source lines');
    }
}

function editorHref(key, line) {
    const url = sourceFiles[key].url;
    return 'vscode://file' + (url.host ? '//' + url.host : '') + url.pathname + ':' + line + ':1';
}

async function verifyEditorLinks(page) {
    const actual = await page.locator('details[id]').evaluateAll(details => details.map(detail => [
        detail.id, [...detail.querySelectorAll('summary .source-link')].map(link => link.href),
    ]));
    const expected = new Map(evidence.rows.map(row => [row.id, [
        editorHref(row.from, row.fromLine), editorHref(row.to, row.toLine),
    ]]));
    assert.equal(actual.length, expected.size);
    for (const [id, links] of actual) assert.deepEqual(links, expected.get(id), id);
    return expected;
}

function verifyPlacement() {
    const { shape, route, focus, fingerprints } = experiment;
    const sum = values => values.reduce((total, value) => total + value, 0);
    const assignments = shape.ep * shape.M * shape.TopK;
    assert.equal(assignments, 65536);
    assert.equal(sum(route.logicalCounts), assignments);
    assert.equal(sum(route.initialRankRoutes), assignments);
    const allocated = Array(shape.E).fill(0);
    route.executionCounts.forEach((slots, rank) => {
        assert.equal(sum(slots), shape.M * shape.TopK);
        assert.equal(route.initialRankRoutes[rank],
            sum(route.logicalCounts.slice(rank * shape.home, (rank + 1) * shape.home)));
        slots.forEach((count, slot) => {
            const expert = slot < shape.home ? rank * shape.home + slot : route.copies[rank][slot - shape.home];
            if (expert === -1) assert.equal(count, 0);
            else allocated[expert] += count;
        });
    });
    assert.deepEqual(allocated, route.logicalCounts);
    const { before, after } = experiment.dispatchMatrix;
    for (const [matrix, expected] of [[before, route.initialRankRoutes], [after, route.executionCounts.map(sum)]]) {
        assert.equal(matrix.length, shape.ep);
        matrix.forEach(row => assert.equal(sum(row), shape.M * shape.TopK));
        assert.deepEqual(expected, expected.map((_, col) => sum(matrix.map(row => row[col]))));
    }
    const scopes = experiment.hostScopes;
    assert.equal(scopes.length, 27);
    for (let layer = 0; layer < 9; layer++) {
        const layerScopes = scopes.filter(scope => scope.layer === layer);
        assert.deepEqual(layerScopes.map(scope => scope.phase), ['forward', 'recompute', 'backward']);
        assert(layerScopes.every(scope => scope.durationMs > 0));
    }
    assert.deepEqual(focus, {
        step: 10, layer: 'model.model.layers.7.mlp.experts', rank: 2, row: 2, logical: [1, 4], physical: [6, 8],
    });
    const stride = shape.home + shape.slots;
    const paths = focus.physical.map((id, lane) => {
        const rank = Math.floor(id / stride);
        const slot = id % stride;
        const expert = slot < shape.home ? rank * shape.home + slot : route.copies[rank][slot - shape.home];
        assert.equal(expert, focus.logical[lane]);
        return { rank, slot, owner: Math.floor(expert / shape.home) };
    });
    assert.deepEqual(paths, [{ rank: 1, slot: 2, owner: 0 }, { rank: 2, slot: 0, owner: 2 }]);
    assert.equal(fingerprints.length, shape.world);
    fingerprints.forEach(({ rank, logical, physicalReplay, physicalDevice }, index) => {
        assert.equal(rank, index);
        assert.match(logical, /^[a-f0-9]{64}$/);
        assert.match(physicalReplay, /^[a-f0-9]{64}$/);
        assert.equal(physicalReplay, physicalDevice);
    });
    return { assignments, paths, verifiedRankFingerprints: fingerprints.length };
}

async function verifyExperiment(page, overview, editorLinks) {
    const example = page.locator('[id="' + caseId + '"]');
    const exampleTab = page.locator('[data-tab="' + caseId + '"]');
    await overview.locator('a[href="#' + caseId + '"]').click();
    await example.waitFor({ state: 'visible' });
    assert.equal(await exampleTab.getAttribute('aria-current'), 'page');
    assert.equal(await example.locator('.case-rank').count(), 8);
    assert.equal(await example.locator('.case-load-row').count(), 4);
    const fills = await example.locator('.step-ms, .step-fsdp, .step-native').evaluateAll(nodes =>
        nodes.map(node => getComputedStyle(node).backgroundColor));
    assert.equal(new Set(fills).size, 3, 'case component colors');
    assert.equal(await example.locator('svg rect.scope-forward, svg rect.scope-recompute, svg rect.scope-backward').count(), 27);
    assert.equal(await example.locator('table.dispatch-matrix td').count(), 32);
    const clipped = await example.locator('svg.case-svg text, svg.bw-svg text').evaluateAll(labels => labels.filter(label => {
        const bounds = label.getBBox();
        const view = label.ownerSVGElement.viewBox.baseVal;
        return bounds.x < 0 || bounds.y < 0 || bounds.x + bounds.width > view.width || bounds.y + bounds.height > view.height;
    }).map(label => label.textContent));
    assert.deepEqual(clipped, [], 'case SVG text fits within viewBox');
    const toc = example.locator('.case-toc a');
    const sections = ['conditions', 'step', 'scenes', 'forward', 'forward-entry', 'planner', 'kernel', 'tape',
        'backward', 'backward-replay', 'gradients', 'owner', 'update', 'evidence'];
    assert.deepEqual(await toc.evaluateAll(nodes => nodes.map(node => node.hash)),
        sections.map(section => '#case-' + section));
    for (const section of sections) {
        const hash = '#case-' + section;
        await example.locator('.case-toc a[href="' + hash + '"]').click();
        await page.waitForFunction(hash => location.hash === hash, hash);
        await page.waitForFunction(hash =>
            document.querySelector('.case-toc [aria-current="location"]')?.hash === hash, hash);
        const title = await page.locator(hash).boundingBox();
        assert(title && title.y >= 0 && title.y < 400, section + ' in view');
    }
    const sidebar = example.locator('.case-sidebar');
    assert.equal(await sidebar.evaluate(node => getComputedStyle(node).position), 'sticky');
    const sideTop = (await sidebar.boundingBox()).y;
    await page.evaluate(() => window.scrollBy(0, 160));
    assert(Math.abs((await sidebar.boundingBox()).y - sideTop) < 2, 'case sidebar remains pinned during scroll');
    for (const id of ['case-forward-entry', 'case-tape']) {
        const img = example.locator('img[data-map-source="maps/' + id + '.svg"]');
        assert(await img.evaluate(image => image.complete && image.naturalWidth > 0), id + ' Mermaid image loads offline');
    }
    assert.deepEqual(await example.locator('.timing-values strong').allTextContents(),
        ['16.168 ms', '25.889 ms', '2812.821 ms']);
    const learningScenes = await verifyLearningScenes(page, example, editorLinks);
    await example.getByRole('link', { name: '更新 epoch', exact: true }).click();
    await page.waitForFunction(() => document.getElementById('B07').open);
    assert.equal(await page.evaluate(() => window.lastEditorLink), editorLinks.get('B07')[1]);
    await page.goBack();
    await example.waitFor({ state: 'visible' });
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto(pathToFileURL(htmlPath).href + '#case-update');
    await page.reload();
    await example.waitFor({ state: 'visible' });
    assert(await page.evaluate(() => document.documentElement.scrollWidth === window.innerWidth),
        'narrow case has no page-level overflow');
    const narrowTitle = await page.locator('#case-update').boundingBox();
    const nav = await page.locator('nav').boundingBox();
    const menu = example.locator('.case-menu-toggle');
    const menuBox = await menu.boundingBox();
    assert(narrowTitle.y >= menuBox.y + menuBox.height && menuBox.y >= nav.y + nav.height - 1
        && narrowTitle.y < 400, 'deep link clears both sticky controls at narrow width');
    await menu.click();
    await example.locator('.case-toc a[href="#case-planner"]').click();
    await page.waitForFunction(() => location.hash === '#case-planner');
    assert.equal(await menu.getAttribute('aria-expanded'), 'false');
    await page.setViewportSize({ width: 1600, height: 1000 });
    return { sections, stickySidebar: true, narrowWidth: 390, narrowMenu: true, sourceJump: 'B07',
        hostScopes: 27, dispatchMatrixCells: 32, learningScenes, ...verifyPlacement() };
}

async function verifyLearningScenes(page, example, editorLinks) {
    const data = sceneData(experiment);
    const assertCalls = values => values.forEach(value => assert(callpoints.has(value.callpoint), value.callpoint));
    assert.equal(data.routes.length, experiment.shape.TopK);
    assertCalls(data.scenes); assertCalls(data.tape); assertCalls(data.checkpoints); assertCalls(data.buffers);
    data.buffers.forEach(buffer => assertCalls(buffer.stages));
    for (const route of data.routes) {
        assert.equal(route.physical, experiment.focus.physical[route.slot]);
        assert.equal(route.expert, experiment.focus.logical[route.slot]);
        assert.equal(route.probability, experiment.route.probability);
        assert.equal(route.shared, experiment.route.copies.some(copies => copies.includes(route.expert)));
    }
    await example.locator('.case-toc a[href="#case-scenes"]').click();
    await page.waitForFunction(() => location.hash === '#case-scenes');
    const workbench = example.locator('#learning-workbench');
    assert.equal(await workbench.locator('[role="tab"]').count(), data.scenes.length);
    for (const route of data.routes) {
        await workbench.locator('[data-learning-slot="' + route.slot + '"]').click();
        assert.equal(await page.evaluate(() => window.__learningScenes.state.slot), route.slot);
        assert((await workbench.locator('#learning-context-route').innerText()).includes('E' + route.expert));
        assert.equal(await example.locator('svg[aria-labelledby="case-svg-forward"] [data-route-slot="' + route.slot + '"].is-route-selected').count(), 1);
        assert.equal(await example.locator('svg[aria-labelledby="bw-flow-title bw-flow-desc"] [data-route-slot="' + route.slot + '"].is-route-selected').count(), 2);
    }
    await workbench.locator('[data-learning-callpoint="G10"]').first().click();
    const before = await page.evaluate(() => location.hash);
    await workbench.locator('[data-learning-open-source="to"]').click();
    assert.equal(await page.evaluate(() => window.lastEditorLink), editorLinks.get('G10')[1]);
    assert.equal(await page.evaluate(() => location.hash), before, 'source opening preserves scene context');
    fs.mkdirSync(path.join(__dirname, '.artifacts'), { recursive: true });
    await workbench.screenshot({ path: path.join(__dirname, '.artifacts/learning-token.png') });
    await workbench.locator('[data-learning-scene="tape"]').click();
    assert.equal(await workbench.locator('.learning-tape-row').count(), data.tape.length);
    for (const buffer of data.buffers) {
        await workbench.locator('[data-learning-buffer="' + buffer.id + '"]').click();
        assert.equal(await workbench.locator('.learning-buffer-stages li').count(), buffer.stages.length);
        assert.equal(await workbench.locator('.learning-lifetime-band > *').count(), 3);
        assert.equal(await page.evaluate(() => window.__learningScenes.state.callpoint), buffer.callpoint);
    }
    await workbench.screenshot({ path: path.join(__dirname, '.artifacts/learning-tape.png') });
    await workbench.locator('[data-learning-scene="pipeline"]').click();
    for (let i = 0; i < data.checkpoints.length; ++i) {
        await workbench.locator('[data-learning-checkpoint="' + i + '"]').click();
        assert.equal(await workbench.locator('#learning-checkpoint-title').innerText(), data.checkpoints[i].title);
        assert.equal(await page.evaluate(() => window.__learningScenes.state.callpoint), data.checkpoints[i].callpoint);
        if (i === 6) assert((await workbench.locator('#learning-checkpoint-text').innerText()).includes('跳过前移'));
        if (i === 3) assert(await workbench.locator('.learning-tape [data-learning-callpoint="G06"].learning-source-selected').count() > 0,
            'P1 consumer selection also highlights the saved gate/up bridge');
    }
    await workbench.locator('#learning-previous').click();
    assert.equal(await page.evaluate(() => window.__learningScenes.state.checkpoint), data.checkpoints.length - 2);
    await workbench.locator('#learning-next').click();
    assert.equal(await workbench.locator('#learning-next').isDisabled(), true);
    assert.equal(await page.evaluate(() => window.__learningScenes.state.slot), data.routes.length - 1, 'route choice persists across scenes');
    await workbench.screenshot({ path: path.join(__dirname, '.artifacts/learning-pipeline.png') });
    return { scenes: data.scenes.map(scene => scene.id), routes: data.routes, buffers: data.buffers.map(buffer => buffer.id), checkpoints: data.checkpoints.length };
}

async function verifyNativeBridge(browser) {
    const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
    await page.addInitScript(() => {
        window.bridgeCalls = 0; window.bridgeMessages = [];
        window.acquireVsCodeApi = () => { ++window.bridgeCalls; return { postMessage: message => window.bridgeMessages.push(message) }; };
    });
    await page.goto(pathToFileURL(htmlPath).href + '#case-scenes');
    await page.waitForFunction(() => window.__learningScenes);
    const desktopHeading = await page.locator('#case-scenes').boundingBox();
    const desktopNav = await page.locator('main > nav').boundingBox();
    assert(desktopHeading.y >= desktopNav.y + desktopNav.height && desktopHeading.y < 400, 'desktop scene entry clears sticky navigation');
    fs.mkdirSync(path.join(__dirname, '.artifacts'), { recursive: true });
    await page.screenshot({ path: path.join(__dirname, '.artifacts/learning-desktop-entry.png') });
    assert.equal(await page.evaluate(() => window.bridgeCalls), 1);
    await page.locator('#learning-workbench [data-learning-open-source="to"]').click();
    assert.deepEqual(await page.evaluate(() => window.bridgeMessages.at(-1)), { type: 'openSource', callpointId: 'K02', side: 'to' });
    assert.equal(await page.evaluate(() => location.hash), '#case-scenes');
    await page.locator('#learning-workbench [data-learning-open-source="from"]').click();
    assert.deepEqual(await page.evaluate(() => window.bridgeMessages.at(-1)), { type: 'openSource', callpointId: 'K02', side: 'from' });
    await page.evaluate(() => window.megaMoeGuide.openSource('V02', 'to', false));
    assert.deepEqual(await page.evaluate(() => window.bridgeMessages.at(-1)), { type: 'openSource', callpointId: 'V02', side: 'to' });
    assert.equal(await page.locator('#learning-source-detail strong').innerText(), 'V02');
    assert.equal(await page.evaluate(() => Object.hasOwn(window, 'editorBridge')), false);
    assert.equal(await page.locator('img:not([src^="data:"])').count(), 0, 'all maps work with the native data-only image CSP');
    await page.screenshot({ path: path.join(__dirname, '.artifacts/learning-native-bridge.png') });
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto(pathToFileURL(htmlPath).href + '#case-scenes');
    await page.waitForFunction(() => window.__learningScenes);
    assert(await page.evaluate(() => document.documentElement.scrollWidth === innerWidth), 'narrow scenes have no page overflow');
    const heading = await page.locator('#case-scenes').boundingBox();
    const menu = await page.locator('.case-menu-toggle').boundingBox();
    const nav = await page.locator('main > nav').boundingBox();
    assert(heading.y >= menu.y + menu.height && menu.y >= nav.y + nav.height - 1 && heading.y < 400,
        'mobile scene entry clears navigation and sticky experiment menu');
    await page.screenshot({ path: path.join(__dirname, '.artifacts/learning-mobile.png') });
    await page.close();
    return { acquireCalls: 1, messages: ['to', 'from', 'V02'], inlinedMaps: true, mobileWidth: 390 };
}

async function verify() {
    const browser = await chromium.launch({
        executablePath: process.env.CHROME_PATH || puppeteer.executablePath(),
        headless: true,
    });
    const checks = [];
    const errors = [];
    let caseChecks;
    let guideChecks;
    let nativeBridge;
    const mainIds = ['00-lifecycle', '01-init', '06-model-step', '11-model-backward'];
    const detailsByChapter = {
        '06-model-step': ['02-forward', '03-planner', '04-native', '07-kernel-forward'],
        '11-model-backward': ['05-backward', '08-kernel-backward', '09-update'],
    };
    try {
        const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
        page.on('pageerror', error => errors.push(error.message));
        await page.addInitScript(() => {
            document.addEventListener('click', event => {
                const link = event.target.closest('a.source-link');
                if (link?.protocol !== 'vscode:') return;
                event.preventDefault();
                window.lastEditorLink = link.href;
            });
        });
        await page.route(/^https?:/, route => {
            errors.push('Unexpected network request: ' + route.request().url());
            return route.abort();
        });
        await page.goto(pathToFileURL(htmlPath).href);
        const overview = page.locator('[id="00-lifecycle"]');
        assert(await overview.isVisible(), 'lifecycle overview is the default tab');
        const overviewColors = await overview.locator('rect.actor-top').evaluateAll(
            nodes => nodes.map(node => getComputedStyle(node).fill),
        );
        assert.equal(overviewColors.length, 3);
        assert.equal(new Set(overviewColors).size, 3, 'three components have distinct visible colors');
        const top = await page.locator('nav [data-tab]').evaluateAll(nodes => nodes.map(node => node.dataset.tab));
        assert.deepEqual(top, [guideId, ...mainIds, caseId], 'guide, four training stages and experiment');
        const panelAt = id => page.locator('section[id="' + id + '"]');
        const awaitPanel = id => page.waitForFunction(id => document.getElementById(id)?.classList.contains('active'), id);
        const openStage = async id => {
            await page.locator('nav [data-tab="' + id + '"]').click();
            await awaitPanel(id);
        };
        const drilldowns = await overview.locator('.drilldowns a').evaluateAll(
            nodes => nodes.map(node => node.getAttribute('href')),
        );
        assert.deepEqual(drilldowns, mainIds.slice(1).map(id => '#'+id));
        const editorLinks = await verifyEditorLinks(page);
        await openStage(guideId);
        const guide = panelAt(guideId);
        assert.deepEqual(await guide.locator('h3').evaluateAll(nodes => nodes.map(node => node.id)), maps.map(map => map.id));
        for (const map of maps) {
            const source = fs.readFileSync(path.join(__dirname, 'maps', map.id + '.mmd'), 'utf8');
            assert(source.split(/\r?\n/).some(line => line === map.syntax || line.startsWith(map.syntax + ' ')), map.id);
            const img = guide.locator('img[data-map-source="maps/' + map.id + '.svg"]');
            assert(await img.evaluate(image => image.complete && image.naturalWidth > 0 && image.naturalHeight > 0
                && image.width > 80 && image.height > 80),
                map.id + ' rendered SVG loads offline');
        }
        await guide.locator('[data-editor-callpoint="P05"]').click();
        await page.waitForFunction(() => document.getElementById('P05').open);
        assert.equal(await page.evaluate(() => window.lastEditorLink), editorLinks.get('P05')[1]);
        await page.setViewportSize({ width: 390, height: 844 });
        await page.goto(pathToFileURL(htmlPath).href + '#parameter-lifecycle');
        await page.reload();
        await awaitPanel(guideId);
        assert(await page.evaluate(() => document.documentElement.scrollWidth === window.innerWidth), 'narrow guide has no page overflow');
        await page.setViewportSize({ width: 1600, height: 1000 });
        guideChecks = maps.map(map => ({ id: map.id, syntax: map.syntax }));
        await openStage('00-lifecycle');
        caseChecks = await verifyExperiment(page, overview, editorLinks);
        await openStage('00-lifecycle');
        await overview.locator('.drilldowns a[href="#01-init"]').click();
        await awaitPanel('01-init');
        await panelAt('01-init').locator('.drilldowns a[href="#06-model-step"]').click();
        await awaitPanel('06-model-step');
        await panelAt('06-model-step').locator('.handoff a[href="#11-model-backward"]').click();
        await awaitPanel('11-model-backward');
        await panelAt('11-model-backward').locator('.handoff a[href="#06-model-step"]').click();
        await awaitPanel('06-model-step');
        for (const [chapter, ids] of Object.entries(detailsByChapter)) {
            await openStage(chapter);
            const parent = panelAt(chapter);
            const targets = await parent.locator('.drilldowns a').evaluateAll(
                nodes => nodes.map(node => node.getAttribute('href')),
            );
            assert.deepEqual(targets, ids.map(id => '#'+id));
            for (const id of ids) {
                await parent.locator('.drilldowns a[href="#'+id+'"]').click();
                await awaitPanel(id);
                assert.equal(await page.locator('nav [aria-current="page"]').getAttribute('data-tab'), chapter);
                await panelAt(id).locator('.chapter-back a').click();
                await awaitPanel(chapter);
            }
        }
        await panelAt('11-model-backward').locator('.chapter-extra a[href="#10-saved-mode"]').click();
        await awaitPanel('10-saved-mode');
        await panelAt('10-saved-mode').locator('.chapter-back a').click();
        await awaitPanel('11-model-backward');
        await openStage('00-lifecycle');
        await overview.locator('.expand').click();
        await overview.locator('svg').screenshot({
            path: path.join(__dirname, 'diagrams', '00-lifecycle.png'),
            style: 'nav { position: static; }',
        });
        await overview.locator('.expand').click();
        const htmlIds = await page.locator('[id]').evaluateAll(nodes => nodes.map(node => node.id));
        assert.equal(new Set(htmlIds).size, htmlIds.length, 'duplicate HTML / SVG ids');
        const links = await page.locator('a[href]').evaluateAll(nodes =>
            nodes.filter(node => !node.matches('.source-link')).map(node => node.href));
        for (const link of new Set(links)) {
            const url = new URL(link);
            if (url.hash) {
                assert(htmlIds.includes(decodeURIComponent(url.hash.slice(1))), link);
            } else {
                assert(url.protocol === 'file:' && fs.existsSync(fileURLToPath(url)), link);
            }
        }
        assert.equal(await page.locator('.panel').count(), diagramIds.length + 2);
        assert.equal(await page.locator('details[id]').count(), callpoints.size);
        for (const id of diagramIds) {
            const src = fs.readFileSync(path.join(__dirname, 'diagrams', id + '.mmd'), 'utf8');
            assert.match(src, /^sequenceDiagram\s*$/m, id + ' native Mermaid sequence source');
            const groupColors = [...src.matchAll(/^\s*box (rgb\([^)]+\))/gm)].map(([, color]) => color);
            assert(groupColors.length > 0, id + ' native participant groups');
            const labels = [...new Set(src.match(/\b[A-Z]\d{2}\b/g) || [])];
            assert(labels.every(label => callpoints.has(label)), id + ' source callpoint');
            await page.goto(pathToFileURL(htmlPath).href + '#' + id);
            await awaitPanel(id);
            await page.mouse.move(0, 0);
            const panel = panelAt(id);
            assert(await panel.isVisible());
            const report = await panel.evaluate((node, groupColors) => {
                const svg = node.querySelector('svg');
                const vb = svg.viewBox.baseVal;
                const view = node.querySelector('.viewport');
                const outside = [...svg.querySelectorAll('text')].filter(t => {
                    const b = t.getBBox();
                    return b.x < vb.x - 3 || b.x + b.width > vb.x + vb.width + 3
                        || b.y < vb.y - 3 || b.y + b.height > vb.y + vb.height + 3;
                }).map(t => t.textContent);
                const lifelines = [...svg.querySelectorAll('line.actor-line')];
                const groups = [...svg.querySelectorAll('rect.rect[fill]')].filter(rect =>
                    groupColors.includes(rect.getAttribute('fill')));
                const laneMarks = svg.querySelectorAll('line.actor-line, rect.actor-top');
                const messages = [...svg.querySelectorAll('.messageLine0, .messageLine1')];
                const lastMessage = Math.max(...messages.map(m => {
                    const box = m.getBBox();
                    return box.y + box.height;
                }));
                return {
                    width: vb.width, height: vb.height, outside,
                    actors: [...svg.querySelectorAll('rect.actor-top')].map(rect => rect.getAttribute('name')),
                    links: [...svg.querySelectorAll('[data-callpoint]')].map(t => t.dataset.callpoint),
                    fitted: view.scrollWidth <= view.clientWidth + 2,
                    lifelines: lifelines.length,
                    classified: lifelines.every(line => !!line.dataset.component),
                    fullLifelines: lifelines.every(line => Number(line.getAttribute('y2')) >= lastMessage),
                    nativeGroups: groups.length,
                    groupsBehindMarks: groups.every(group => [...laneMarks].every(mark =>
                        !!(group.compareDocumentPosition(mark) & Node.DOCUMENT_POSITION_FOLLOWING))),
                    bottomParticipants: svg.querySelectorAll('rect.actor-bottom').length,
                    themePaints: Object.fromEntries([
                        ['labelBoxes', '.labelBox', 'fill'],
                        ['notes', '.note', 'fill'],
                        ['messages', '.messageText:not(.noteText):not(.learning-callpoint-selected)', 'fill'],
                        ['noteText', '.noteText:not(.learning-callpoint-selected)', 'fill'],
                        ['arrows', '.messageLine0, .messageLine1', 'stroke'],
                    ].map(([name, selector, property]) => [name,
                        [...new Set([...svg.querySelectorAll(selector)].map(node => getComputedStyle(node)[property]))],
                    ])),
                };
            }, groupColors);
            assert(report.width > 0 && report.height > 0 && report.actors.length > 0, id);
            assert.deepEqual(report.outside, [], id + ' clipped text');
            assert(report.fitted, id + ' fit width');
            assert(labels.every(label => report.links.includes(label)), id + ' rendered labels');
            const actorCount = (src.match(/^\s+participant /gm) || []).length;
            assert.equal(report.actors.length, actorCount, id + ' participant columns');
            assert.equal(report.lifelines, actorCount, id + ' lifeline count');
            assert.equal(report.nativeGroups, groupColors.length, id + ' native Mermaid groups');
            assert(report.groupsBehindMarks, id + ' native groups behind participants');
            assert.equal(report.bottomParticipants, actorCount, id + ' bottom participants');
            assert(report.classified && report.fullLifelines, id + ' full colored lanes through last arrow');
            for (const [name, expected] of Object.entries({
                labelBoxes: theme.actorBkg, notes: theme.noteBkgColor,
                messages: theme.signalTextColor, noteText: theme.noteTextColor, arrows: theme.signalColor,
            })) {
                for (const actual of report.themePaints[name]) assert.equal(actual, rgb(expected), id + ' themed ' + name);
            }
            if (id === '06-model-step' || id === '11-model-backward') {
                assert.equal(await panel.locator('rect.actor-top[data-component="kernel"]').count(),
                    id === '06-model-step' ? 3 : 2, id + ' kernel component columns');
                const chain = id === '06-model-step' ? ['F03','F06','N04','N10','K06','N11','F16']
                    : ['B01','B07','B09','B13','B14','B16','U04','U07'];
                assert(chain.every((label,i)=> report.links.indexOf(label)>=0 &&
                    (!i || report.links.indexOf(label)>report.links.indexOf(chain[i-1]))), id + ' ordered kernel handoff');
            }
            await panel.locator('.actual').click();
            const view = panel.locator('.viewport');
            const scroll = await view.evaluate(el => {
                el.scrollTo(el.scrollWidth, el.scrollHeight);
                return {
                    x: el.scrollLeft, y: el.scrollTop,
                    endX: el.scrollWidth - el.clientWidth, endY: el.scrollHeight - el.clientHeight,
                };
            });
            assert(Math.abs(scroll.x - scroll.endX) <= 2 && Math.abs(scroll.y - scroll.endY) <= 2, id + ' full scroll');
            await panel.locator('.fit').click();
            await panel.locator('.expand').click();
            assert(await view.evaluate(el => el.clientHeight >= el.scrollHeight - 1), id + ' expanded height');
            await panel.locator('.expand').click();
            await panel.locator('.plus').click();
            await panel.locator('.minus').click();
            await panel.locator('.fit').click();
            const first = panel.locator('svg [data-callpoint]').first();
            const label = await first.getAttribute('data-callpoint');
            await first.click();
            await page.waitForFunction(value => document.getElementById(value)?.open, label);
            assert.equal(await page.evaluate(() => window.lastEditorLink), editorLinks.get(label)[1]);
            checks.push({ id, ...report, actorCount, callpoints: labels.length, scroll });
        }
        await page.goto(pathToFileURL(htmlPath).href + '#H01');
        await page.waitForFunction(() => document.getElementById('H01').open);
        await page.reload();
        await awaitPanel('10-saved-mode');
        assert(await page.locator('#H01').evaluate(node => node.open));
        await openStage('01-init');
        const initViewport = page.locator('[id="01-init"] .viewport');
        await initViewport.evaluate(el => el.scrollTo(0, el.scrollHeight));
        await initViewport.screenshot({
            path: path.join(__dirname, 'diagrams', '01-init-tail.png'),
            style: 'nav { position: static; }',
        });
        await openStage('11-model-backward');
        await page.evaluate(() => window.scrollTo(0, 0));
        await page.screenshot({ path: path.join(__dirname, 'viewer-preview.png') });
        nativeBridge = await verifyNativeBridge(browser);
        assert.deepEqual(errors, []);
    } finally {
        await browser.close();
    }
    const result = {
        ok: true, errors, sourceFiles: hashes.length, callpoints: callpoints.size,
        checks, linkCoverage: 'packaged assets, embedded source excerpts and SVG callpoint links',
        lifecycle: { defaultTab: true, distinctColors: 3, topStages: mainIds, verifiedDetailReturns: 8 },
        experiment: caseChecks,
        systemGuide: guideChecks,
        nativeBridge,
        editorLinks: { referencedFiles: hashes.length, sourceEndpoints: callpoints.size * 2, protocol: 'vscode:' },
    };
    fs.writeFileSync(path.join(__dirname, 'verification.json'), JSON.stringify(result, null, 2));
    console.log(JSON.stringify({
        ok: true, diagrams: checks.length, sourceFiles: hashes.length, callpoints: callpoints.size,
        actorColumns: checks.map(check => check.actorCount),
        experiment: caseChecks,
        systemGuide: guideChecks,
    }));
}

verify().catch(error => {
    console.error(error);
    process.exitCode = 1;
});
