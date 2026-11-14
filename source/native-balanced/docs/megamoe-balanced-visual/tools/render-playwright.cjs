// Render the same pinned Mermaid sources through the existing Playwright browser.
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const { chromium } = require('playwright');

async function main() {
    const root = path.resolve(__dirname, '..');
    const config = JSON.parse(fs.readFileSync(path.join(root, 'mermaid-config.json'), 'utf8'));
    const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROME_PATH || undefined });
    const records = [];
    try {
        const page = await browser.newPage({ viewport: { width: 1800, height: 1200 } });
        await page.setContent('<!doctype html><html><head><meta charset="utf-8"></head><body><div id="container"></div></body></html>');
        await page.addScriptTag({ path: require.resolve('mermaid/dist/mermaid.min.js') });
        for (const directory of ['diagrams', 'maps']) {
            const folder = path.join(root, directory);
            for (const name of fs.readdirSync(folder).filter(name => name.endsWith('.mmd')).sort()) {
                const input = path.join(folder, name);
                const definition = fs.readFileSync(input, 'utf8');
                const id = 'visual-' + path.basename(name, '.mmd').replace(/[^a-zA-Z0-9_-]/g, '-');
                const svg = await page.evaluate(async ({ definition, config, id }) => {
                    const container = document.getElementById('container');
                    container.innerHTML = '';
                    window.mermaid.initialize({ startOnLoad: false, ...config, deterministicIds: true, deterministicIDSeed: id });
                    const result = await window.mermaid.render(id, definition, container);
                    container.innerHTML = result.svg;
                    const element = container.querySelector('svg');
                    element.style.backgroundColor = 'transparent';
                    return new XMLSerializer().serializeToString(element);
                }, { definition, config, id });
                fs.writeFileSync(input.replace(/\.mmd$/, '.svg'), svg + '\n');
                records.push({ source: `${directory}/${name}`, sha256: crypto.createHash('sha256').update(definition).digest('hex') });
                console.log(`Rendered ${directory}/${name}`);
            }
        }
        fs.mkdirSync(path.join(root, '.artifacts'), { recursive: true });
        fs.writeFileSync(path.join(root, '.artifacts', 'render-playwright.json'), JSON.stringify({
            ok: true, renderer: 'Playwright + pinned Mermaid', browser: browser.version(), diagrams: records,
        }, null, 2) + '\n');
    } finally {
        await browser.close();
    }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
