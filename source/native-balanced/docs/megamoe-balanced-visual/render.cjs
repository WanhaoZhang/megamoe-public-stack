const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { execFileSync } = require('node:child_process');

if (process.env.MERMAID_ENGINE === 'playwright') {
    execFileSync(process.execPath, [path.join(__dirname, 'tools/render-playwright.cjs')], { stdio: 'inherit' });
    process.exit(0);
}

const cli = path.join(path.dirname(require.resolve('@mermaid-js/mermaid-cli')), 'cli.js');
const config = require('./puppeteer-config.json');
if (process.env.CHROME_PATH) config.executablePath = process.env.CHROME_PATH;
const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'megamoe-swimlanes-'));
try {
    const puppeteerConfig = path.join(temp, 'puppeteer.json');
    fs.writeFileSync(puppeteerConfig, JSON.stringify(config));
    for (const directory of ['diagrams', 'maps']) {
        const root = path.join(__dirname, directory);
        const sources = fs.readdirSync(root).filter(name => name.endsWith('.mmd')).sort();
        for (const name of sources) {
            execFileSync(process.execPath, [
                cli, '-i', path.join(root, name),
                '-o', path.join(root, name.replace(/\.mmd$/, '.svg')),
                '-c', path.join(__dirname, 'mermaid-config.json'),
                '-p', puppeteerConfig, '-b', 'transparent',
            ], { stdio: 'inherit' });
        }
    }
} finally {
    fs.unlinkSync(path.join(temp, 'puppeteer.json'));
    fs.rmdirSync(temp);
}
